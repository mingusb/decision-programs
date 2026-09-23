#include "ghb/instrumentation.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef GHB_WITH_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace ghb::instrumentation {
namespace {

constexpr std::array<const char*, static_cast<std::size_t>(Stage::count)> stage_names{
    "quantize", "upload", "initialize", "gradients", "histogram", "histogram_subtract",
    "split_search", "route", "prediction", "evaluate", "download", "checkpoint", "tree_build"};
using Clock = std::chrono::steady_clock;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void reject_capture(cudaStream_t stream) {
  cudaStreamCaptureStatus capture{};
  check_cuda(cudaStreamIsCapturing(stream, &capture), "query instrumentation stream capture");
  if (capture != cudaStreamCaptureStatusNone)
    throw std::logic_error("instrumentation cannot record during stream capture");
}

std::uint64_t next_generation() {
  // Unique across recorder lifetimes as well as reset. An allocator reusing an
  // Impl address must not make a ticket from a destroyed recorder valid again.
  static std::atomic<std::uint64_t> next{1};
  auto value = next.load(std::memory_order_relaxed);
  for (;;) {
    if (value == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("instrumentation generation exhausted");
    if (next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed)) return value;
  }
}

struct Nvtx {
#ifdef GHB_WITH_NVTX
  nvtxDomainHandle_t domain{};
  std::array<nvtxStringHandle_t, stage_names.size()> names{};
  bool enabled{};
  explicit Nvtx(bool requested) : enabled(requested) {
    if (enabled) {
      domain = nvtxDomainCreateA("ghb");
      for (std::size_t index = 0; index < names.size(); ++index)
        names[index] = nvtxDomainRegisterStringA(domain, stage_names[index]);
    }
  }
  ~Nvtx() { if (enabled) nvtxDomainDestroy(domain); }
  std::uint64_t begin(Stage stage, std::size_t id) const noexcept {
    if (!enabled) return 0;
    nvtxEventAttributes_t attributes{};
    attributes.version = NVTX_VERSION;
    attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attributes.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    attributes.message.registered = names[static_cast<std::size_t>(stage)];
    attributes.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
    attributes.payload.ullValue = static_cast<std::uint64_t>(id);
    return nvtxDomainRangeStartEx(domain, &attributes);
  }
  void end(std::uint64_t range) const noexcept {
    if (enabled) nvtxDomainRangeEnd(domain, range);
  }
#else
  explicit Nvtx(bool) {}
  std::uint64_t begin(Stage, std::size_t) const noexcept { return 0; }
  void end(std::uint64_t) const noexcept {}
#endif
  Nvtx(const Nvtx&) = delete;
  Nvtx& operator=(const Nvtx&) = delete;
};

enum class State { unused, open, closed };
struct Slot {
  Sample sample{};
  cudaStream_t stream{};
  cudaEvent_t start{}, end{};
  std::uint64_t nvtx_range{};
  State state{State::unused};
  Slot() = default;
  Slot(const Slot&) = delete;
  Slot& operator=(const Slot&) = delete;
  ~Slot() {
    // CUDA retains any resources still needed by pending work. Destruction is
    // deliberately not an implicit collection or device synchronization.
    if (end) cudaEventDestroy(end);
    if (start) cudaEventDestroy(start);
  }
};

}  // namespace

std::string_view name(Stage stage) {
  const auto index = static_cast<std::size_t>(stage);
  return index < stage_names.size() ? stage_names[index] : "unknown";
}

struct Recorder::Impl {
  const Clock::time_point origin{Clock::now()};
  const std::thread::id owner_thread{std::this_thread::get_id()};
  Nvtx nvtx;
  std::vector<Slot> slots;
  std::size_t used{}, next_id{};
  std::uint64_t generation{next_generation()};

  explicit Impl(std::size_t capacity, bool enable_nvtx)
      : nvtx(enable_nvtx), slots(capacity) {
    for (auto& slot : slots) {
      check_cuda(cudaEventCreateWithFlags(&slot.start, cudaEventDefault), "create instrumentation start event");
      check_cuda(cudaEventCreateWithFlags(&slot.end, cudaEventDefault), "create instrumentation end event");
    }
  }
  ~Impl() {
    for (std::size_t index = 0; index < used; ++index)
      if (slots[index].state == State::open) nvtx.end(slots[index].nvtx_range);
  }
  void check_thread() const {
    if (std::this_thread::get_id() != owner_thread)
      throw std::logic_error("instrumentation recorder belongs to another host thread");
  }
  std::uint64_t now() const noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - origin).count());
  }
  void reject_open() const {
    for (std::size_t index = 0; index < used; ++index)
      if (slots[index].state == State::open)
        throw std::logic_error("instrumentation contains an unended scope");
  }
  bool events_ready() const {
    for (std::size_t index = 0; index < used; ++index) {
      const auto& slot = slots[index];
      if (slot.sample.timing == Timing::gpu) {
        const auto status = cudaEventQuery(slot.end);
        if (status == cudaErrorNotReady) {
          // Pending work is an expected query result, not a CUDA launch error
          // to leak into the caller's next cudaGetLastError(). Surface any
          // different error rather than silently clearing it.
          const auto last = cudaGetLastError();
          if (last != cudaErrorNotReady) check_cuda(last, "query instrumentation pending event");
          return false;
        }
        check_cuda(status, "query instrumentation end event");
      }
    }
    return true;
  }
};

Recorder::Recorder(std::size_t capacity, bool nvtx) : impl_(std::make_unique<Impl>(capacity, nvtx)) {}
Recorder::~Recorder() = default;

Ticket Recorder::begin(Stage stage, const Context& context, Timing timing, cudaStream_t stream) {
  auto& impl = *impl_;
  impl.check_thread();
  if (static_cast<std::size_t>(stage) >= stage_names.size())
    throw std::invalid_argument("invalid instrumentation stage");
  if (timing != Timing::host && timing != Timing::gpu)
    throw std::invalid_argument("invalid instrumentation timing mode");
  if (impl.used == impl.slots.size())
    throw std::length_error("instrumentation capacity exhausted");
  if (impl.next_id == std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("instrumentation sample IDs exhausted");
  reject_capture(stream);
  auto& slot = impl.slots[impl.used];
  const auto start = impl.now();
  if (timing == Timing::gpu)
    check_cuda(cudaEventRecord(slot.start, stream), "record instrumentation start event");
  // No slot state is committed until every fallible validation/CUDA call has
  // succeeded. NVTX names are registered at construction, outside this path.
  slot.sample = Sample{impl.next_id, stage, timing, context, start, 0, std::nullopt};
  slot.stream = stream;
  slot.nvtx_range = impl.nvtx.begin(stage, impl.next_id);
  slot.state = State::open;
  const Ticket ticket{impl.used, impl.generation, impl_.get()};
  ++impl.used;
  ++impl.next_id;
  return ticket;
}

void Recorder::end(Ticket ticket) {
  auto& impl = *impl_;
  impl.check_thread();
  if (ticket.owner != impl_.get() || ticket.generation != impl.generation || ticket.index >= impl.used)
    throw std::invalid_argument("stale or foreign instrumentation ticket");
  auto& slot = impl.slots[ticket.index];
  if (slot.state != State::open)
    throw std::logic_error("instrumentation ticket was already ended");
  reject_capture(slot.stream);
  if (slot.sample.timing == Timing::gpu)
    check_cuda(cudaEventRecord(slot.end, slot.stream), "record instrumentation end event");
  impl.nvtx.end(slot.nvtx_range);
  slot.sample.host_end_ns = impl.now();
  slot.state = State::closed;
}

bool Recorder::ready() const {
  const auto& impl = *impl_;
  impl.check_thread();
  for (std::size_t index = 0; index < impl.used; ++index)
    if (impl.slots[index].state == State::open) return false;
  return impl.events_ready();
}

std::optional<std::vector<Sample>> Recorder::collect(bool wait) {
  auto& impl = *impl_;
  impl.check_thread();
  impl.reject_open();
  if (wait) {
    for (std::size_t index = 0; index < impl.used; ++index) {
      const auto& slot = impl.slots[index];
      if (slot.sample.timing == Timing::gpu)
        check_cuda(cudaEventSynchronize(slot.end), "wait for instrumentation end event");
    }
  } else if (!impl.events_ready()) {
    return std::nullopt;
  }
  std::vector<Sample> samples;
  samples.reserve(impl.used);
  for (std::size_t index = 0; index < impl.used; ++index) {
    const auto& slot = impl.slots[index];
    Sample sample = slot.sample;
    if (sample.timing == Timing::gpu) {
      float milliseconds{};
      check_cuda(cudaEventElapsedTime(&milliseconds, slot.start, slot.end), "read instrumentation elapsed time");
      sample.gpu_ms = static_cast<double>(milliseconds);
    }
    samples.push_back(sample);
  }
  return samples;
}

void Recorder::reset() {
  auto& impl = *impl_;
  impl.check_thread();
  impl.reject_open();
  if (!impl.events_ready())
    throw std::logic_error("cannot reset instrumentation with pending GPU work");
  const auto generation = next_generation();
  for (std::size_t index = 0; index < impl.used; ++index) impl.slots[index].state = State::unused;
  impl.used = 0;
  impl.generation = generation;
}

std::size_t Recorder::size() const noexcept { return impl_->used; }
std::size_t Recorder::capacity() const noexcept { return impl_->slots.size(); }
bool Recorder::nvtx_available() noexcept {
#ifdef GHB_WITH_NVTX
  return true;
#else
  return false;
#endif
}

}  // namespace ghb::instrumentation
