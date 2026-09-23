#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace ghb::instrumentation {

enum class Stage : std::uint32_t {
  quantize, upload, initialize, gradients, histogram, histogram_subtract,
  split_search, route, prediction, evaluate, download, checkpoint, tree_build, count
};
enum class Timing : std::uint32_t { host, gpu };
std::string_view name(Stage stage);

struct Context {
  std::int32_t round{-1}, depth{-1}, output{-1}, repetition{-1};
  std::uint32_t active_nodes{}, features{}, bins{}, stream_id{};
  std::uint64_t rows{}, logical_read_bytes{}, logical_write_bytes{}, scratch_bytes{}, operations{1};
};
struct Ticket {
  std::size_t index{};
  std::uint64_t generation{};
  const void* owner{};
};
struct Sample {
  std::size_t id{};
  Stage stage{};
  Timing timing{};
  Context context{};
  // Relative to this recorder's construction. Host spans include instrumentation
  // submission overhead. GPU intervals can overlap; do not sum across streams.
  std::uint64_t host_start_ns{}, host_end_ns{};
  std::optional<double> gpu_ms;
};

// One host thread owns a recorder. Construct outside measured work: every event
// and record slot is allocated here. Successful begin/end make no recorder-owned
// allocations or explicit synchronization calls; CUDA/NVTX internals may add cost.
// GPU stages surround ordinary launches or graph replay on the supplied stream.
// Recording during stream capture is rejected; replayed events would otherwise
// overwrite timestamps. Use a distinct slot for every graph replay occurrence.
class Recorder {
 public:
  explicit Recorder(std::size_t capacity, bool nvtx = true);
  ~Recorder();
  Recorder(const Recorder&) = delete;
  Recorder& operator=(const Recorder&) = delete;
  Recorder(Recorder&&) = delete;
  Recorder& operator=(Recorder&&) = delete;

  Ticket begin(Stage stage, const Context& context = {},
               Timing timing = Timing::gpu, cudaStream_t stream = nullptr);
  void end(Ticket ticket);
  bool ready() const;
  // Nonblocking collection returns nullopt if any completed scope is pending.
  // Open scopes are a logic error. wait=true waits only for recorded end events.
  std::optional<std::vector<Sample>> collect(bool wait = false);
  void reset(); // rejects open scopes or GPU work that has not finished
  std::size_t size() const noexcept;
  std::size_t capacity() const noexcept;
  static bool nvtx_available() noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Select this type at compile time in uninstrumented builds. It has no state,
// event creation, clock reads, synchronization, NVTX calls or capacity checks.
struct NullRecorder {
  constexpr explicit NullRecorder(std::size_t = 0, bool = false) noexcept {}
  constexpr Ticket begin(Stage, const Context& = {}, Timing = Timing::gpu,
                         cudaStream_t = nullptr) const noexcept { return {}; }
  constexpr void end(Ticket) const noexcept {}
  constexpr bool ready() const noexcept { return true; }
  std::optional<std::vector<Sample>> collect(bool = false) const { return std::vector<Sample>{}; }
  constexpr void reset() const noexcept {}
  constexpr std::size_t size() const noexcept { return 0; }
  constexpr std::size_t capacity() const noexcept { return 0; }
  static constexpr bool nvtx_available() noexcept { return false; }
};

} // namespace ghb::instrumentation
