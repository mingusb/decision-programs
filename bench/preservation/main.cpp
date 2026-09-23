#include "backend.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
CUcontext current_context() {
  CUcontext context{};
  if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context)
    throw std::runtime_error("no current CUDA context");
  return context;
}
void check_context(CUcontext expected) {
  if (current_context() != expected) throw std::runtime_error("CUDA context changed");
}
struct Buffer {
  void* data{};
  explicit Buffer(std::size_t bytes) { check(cudaMalloc(&data, bytes)); }
  ~Buffer() { cudaFree(data); }
  Buffer(const Buffer&) = delete;
};
struct PinnedBuffer {
  void* data{};
  explicit PinnedBuffer(std::size_t bytes) { check(cudaMallocHost(&data, bytes)); }
  ~PinnedBuffer() { cudaFreeHost(data); }
  PinnedBuffer(const PinnedBuffer&) = delete;
};
struct Stream {
  cudaStream_t value{};
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
};
struct Event {
  cudaEvent_t value{};
  Event() { check(cudaEventCreate(&value)); }
  ~Event() { cudaEventDestroy(value); }
};
struct Graph {
  Event start, end;
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  ~Graph() {
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
  }
  template<class Function>
  void capture(cudaStream_t stream, int batch, Function operation) {
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    try {
      check(cudaEventRecordWithFlags(start.value, stream, cudaEventRecordExternal));
      for (int i = 0; i < batch; ++i) operation();
      check(cudaEventRecordWithFlags(end.value, stream, cudaEventRecordExternal));
    } catch (...) {
      cudaGraph_t discarded{};
      cudaStreamEndCapture(stream, &discarded);
      if (discarded) cudaGraphDestroy(discarded);
      throw;
    }
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiateWithFlags(&executable, graph, 0));
  }
};
struct Slot {
  const BackendApi* api;
  void* state{};
  std::unique_ptr<Graph> graph;
  Slot(const BackendApi* backend, const PairedConfig& config) : api(backend) {
    check(api->create(&config, &state));
    if (!state) throw std::runtime_error("backend returned null state");
  }
  ~Slot() { graph.reset(); api->destroy(state); }
  void operation(const void* input, void* output, cudaStream_t stream) const {
    check(api->launch(state, input, output, stream));
  }
  void launch(const void* input, void* output, cudaStream_t stream) const {
    if (graph) check(cudaGraphLaunch(graph->executable, stream));
    else operation(input, output, stream);
  }
};
struct Options {
  std::string case_name = "single", comparison = "old-new", sync_mode = "legacy";
  std::uint64_t data_seed = 2026092201, order_seed = 2026092251;
  int quartets = 32, batch = 32, warmup_ms = 200;
};
std::uint64_t number(const std::string& value) {
  if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
    throw std::runtime_error("invalid nonnegative integer: " + value);
  std::size_t consumed{};
  auto result = std::stoull(value, &consumed);
  if (consumed != value.size()) throw std::runtime_error("invalid integer");
  return result;
}
Options parse(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--help") {
      std::cout << "histogram_preservation --case single|stream4096 "
                   "--comparison old-new|new-old|old-old|new-new "
                   "[--seed N --order-seed N --quartets EVEN --batch N --warmup-ms N] "
                   "[--sync-mode legacy|position|quartet]\n";
      std::exit(0);
    }
    if (++i == argc) throw std::runtime_error("missing option value");
    std::string value = argv[i];
    if (key == "--case") options.case_name = value;
    else if (key == "--comparison") options.comparison = value;
    else if (key == "--sync-mode") options.sync_mode = value;
    else if (key == "--seed") options.data_seed = number(value);
    else if (key == "--order-seed") options.order_seed = number(value);
    else if (key == "--quartets" || key == "--batch" || key == "--warmup-ms") {
      auto n = number(value);
      if (n > 60000) throw std::runtime_error("option exceeds bounded diagnostic limit");
      if (key == "--quartets") options.quartets = static_cast<int>(n);
      else if (key == "--batch") options.batch = static_cast<int>(n);
      else options.warmup_ms = static_cast<int>(n);
    } else throw std::runtime_error("unknown option: " + key);
  }
  if (options.case_name != "single" && options.case_name != "stream4096")
    throw std::runtime_error("unsupported case");
  if (options.comparison != "old-new" && options.comparison != "new-old" &&
      options.comparison != "old-old" && options.comparison != "new-new")
    throw std::runtime_error("unsupported comparison");
  if (options.sync_mode != "legacy" && options.sync_mode != "position" && options.sync_mode != "quartet")
    throw std::runtime_error("unsupported synchronization mode");
  if (options.quartets < 2 || options.quartets % 2 || options.batch < 1)
    throw std::runtime_error("quartets must be positive/even; batch must be positive");
  return options;
}
std::string address(std::uintptr_t value) {
  std::ostringstream result;
  result << "0x" << std::hex << value;
  return result.str();
}
std::string uuid_string(const cudaUUID_t& uuid) {
  std::ostringstream result;
  result << "GPU-" << std::hex << std::setfill('0');
  for (int i = 0; i < 16; ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) result << '-';
    result << std::setw(2) << static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[i]));
  }
  return result.str();
}
double microseconds(Clock::time_point a, Clock::time_point b, int batch) {
  return std::chrono::duration<double, std::micro>(b - a).count() / batch;
}
struct Measurement {
  int quartet, position;
  std::string pattern;
  char slot;
  const char* backend;
  double event_us, submit_us, total_host_us;
  double quartet_host_us{};
  int timing_pair{-1};
};

int run(const Options& options) {
  const auto* old_api = paired_old_api();
  const auto* new_api = paired_new_api();
  if (!old_api || !new_api || old_api->launch == new_api->launch ||
      std::string(old_api->name) != "old" || std::string(new_api->name) != "new")
    throw std::runtime_error("backend identity or binding differs");
  const bool single = options.case_name == "single";
  const PairedConfig config{1u << 20, single ? 256u : 4096u,
                            single ? 6 : 10, single ? 192 : 48, single, single};
  const auto separator = options.comparison.find('-');
  const std::string a_name = options.comparison.substr(0, separator);
  const std::string b_name = options.comparison.substr(separator + 1);
  check(cudaSetDevice(0));
  Stream stream;
  const CUcontext context = current_context();
  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, 0));
  int driver{}, runtime{};
  check(cudaDriverGetVersion(&driver));
  check(cudaRuntimeGetVersion(&runtime));
  Slot a(a_name == "old" ? old_api : new_api, config);
  Slot b(b_name == "old" ? old_api : new_api, config);
  Slot* slots[] = {&a, &b};
  std::vector<unsigned> input(config.n), expected(config.bins);
  std::mt19937_64 random(options.data_seed);
  for (auto& key : input) {
    const auto candidate = static_cast<unsigned>(random() % config.bins);
    key = single ? config.bins - 1 : candidate;
    ++expected[key];
  }
  Buffer device_input(input.size() * sizeof(unsigned));
  constexpr std::size_t guard = 64;
  constexpr unsigned poison = 0xa5a5a5a5u;
  const std::size_t output_words = config.bins + guard * 2;
  Buffer device_output(output_words * sizeof(unsigned));
  auto* output = static_cast<unsigned*>(device_output.data) + guard;
  std::vector<unsigned> actual(output_words);
  check(cudaMemcpyAsync(device_input.data, input.data(), input.size() * sizeof(unsigned),
                        cudaMemcpyHostToDevice, stream.value));
  check(cudaMemsetAsync(device_output.data, 0xa5, output_words * sizeof(unsigned), stream.value));
  check(cudaStreamSynchronize(stream.value));
  auto verify_words = [&](const unsigned* words) {
    for (std::size_t i = 0; i < output_words; ++i) {
      const auto want = i >= guard && i < guard + config.bins ? expected[i - guard] : poison;
      if (words[i] != want) throw std::runtime_error("output or canary mismatch at word " + std::to_string(i));
    }
    check_context(context);
  };
  auto verify = [&] {
    check(cudaMemcpyAsync(actual.data(), device_output.data, output_words * sizeof(unsigned),
                          cudaMemcpyDeviceToHost, stream.value));
    check(cudaStreamSynchronize(stream.value));
    verify_words(actual.data());
  };
  for (auto* slot : slots) {
    check(cudaMemsetAsync(output, 0xa5, config.bins * sizeof(unsigned), stream.value));
    for (int i = 0; i < 2; ++i) {
      slot->operation(device_input.data, output, stream.value);
      verify();
    }
    if (config.graph && options.sync_mode == "legacy") {
      slot->graph = std::make_unique<Graph>();
      slot->graph->capture(stream.value, options.batch,
                          [&] { slot->operation(device_input.data, output, stream.value); });
      check(cudaMemsetAsync(output, 0xa5, config.bins * sizeof(unsigned), stream.value));
      for (int i = 0; i < 2; ++i) {
        slot->launch(device_input.data, output, stream.value);
        verify();
      }
    }
  }
  // Matched modes own two distinct occurrences per slot. Each of the four
  // timing pairs identifies only one measured position per quartet; replay of
  // one slot cannot overwrite timestamps from its earlier occurrence.
  std::array<std::unique_ptr<Graph>, 4> occurrences;
  std::unique_ptr<PinnedBuffer> snapshots;
  if (options.sync_mode != "legacy") {
    snapshots = std::make_unique<PinnedBuffer>(4 * output_words * sizeof(unsigned));
    for (int occurrence = 0; occurrence < 4; ++occurrence) {
      occurrences[occurrence] = std::make_unique<Graph>();
      auto* slot = slots[occurrence / 2];
      if (config.graph) {
        occurrences[occurrence]->capture(stream.value, options.batch,
            [&] { slot->operation(device_input.data, output, stream.value); });
        check(cudaMemsetAsync(output, 0xa5, config.bins * sizeof(unsigned), stream.value));
        for (int i = 0; i < 2; ++i) {
          check(cudaGraphLaunch(occurrences[occurrence]->executable, stream.value));
          verify();
        }
      }
    }
  }
  // Identical backend warmup alternates slots; no label gets a private warmup epoch.
  if (options.warmup_ms) {
    auto begin = Clock::now();
    do {
      for (int index = 0; index < 2; ++index) {
        if (options.sync_mode != "legacy" && config.graph)
          check(cudaGraphLaunch(occurrences[index * 2]->executable, stream.value));
        else slots[index]->launch(device_input.data, output, stream.value);
        check(cudaStreamSynchronize(stream.value));
      }
    } while (Clock::now() - begin < std::chrono::milliseconds(options.warmup_ms));
  }
  std::vector<std::string> patterns(options.quartets, "BAAB");
  std::fill_n(patterns.begin(), options.quartets / 2, "ABBA");
  std::uint64_t order_state = options.order_seed;
  for (std::size_t i = patterns.size() - 1; i > 0; --i) {
    order_state = order_state * 6364136223846793005ULL + 1442695040888963407ULL;
    std::swap(patterns[i], patterns[order_state % (i + 1)]);
  }
  Event start, end;
  std::vector<Measurement> measurements;
  measurements.reserve(options.quartets * 4);
  if (options.sync_mode == "legacy") {
    for (int q = 0; q < options.quartets; ++q) {
      for (int position = 0; position < 4; ++position) {
        const char label = patterns[q][position];
        auto* slot = slots[label == 'A' ? 0 : 1];
        std::string range = options.case_name + ":" + options.comparison + ":q" + std::to_string(q)
                          + ":p" + std::to_string(position) + ":" + label + ":" + slot->api->name;
        nvtxRangePushA(range.c_str());
        // As in the original warm protocol, one untimed launch precedes timing.
        // Its pending work can contribute to host enqueue-through-sync duration.
        slot->launch(device_input.data, output, stream.value);
        const auto host_start = Clock::now();
        if (slot->graph) {
          slot->launch(device_input.data, output, stream.value);
        } else {
          check(cudaEventRecord(start.value, stream.value));
          for (int repetition = 0; repetition < options.batch; ++repetition)
            slot->operation(device_input.data, output, stream.value);
          check(cudaEventRecord(end.value, stream.value));
        }
        const auto submitted = Clock::now();
        if (slot->graph) check(cudaStreamSynchronize(stream.value));
        else check(cudaEventSynchronize(end.value));
        const auto finished = Clock::now();
        float milliseconds{};
        check(cudaEventElapsedTime(&milliseconds,
                                   slot->graph ? slot->graph->start.value : start.value,
                                   slot->graph ? slot->graph->end.value : end.value));
        nvtxRangePop();
        const double elapsed = milliseconds * 1000.0 / options.batch;
        const double submit = microseconds(host_start, submitted, options.batch);
        const double total = microseconds(host_start, finished, options.batch);
        if (!(elapsed > 0) || !(submit > 0) || total < submit || !std::isfinite(elapsed))
          throw std::runtime_error("invalid measured duration");
        // Check every measured backend, after events; a shared buffer must not mask a failing slot.
        verify();
        measurements.push_back({q, position, patterns[q], label, slot->api->name, elapsed, submit, total});
      }
    }
  } else {
    const bool position_sync = options.sync_mode == "position";
    for (int q = 0; q < options.quartets; ++q) {
      std::array<Measurement, 4> pending;
      std::array<int, 2> seen{};
      const auto quartet_start = Clock::now();
      Clock::time_point quartet_finished{};
      for (int position = 0; position < 4; ++position) {
        const char label = patterns[q][position];
        const int slot_index = label == 'A' ? 0 : 1;
        const int occurrence = slot_index * 2 + seen[slot_index]++;
        auto* slot = slots[slot_index];
        auto& timing = *occurrences[occurrence];
        auto* snapshot = static_cast<unsigned*>(snapshots->data) + position * output_words;
        const std::string range = options.case_name + ":" + options.comparison + ":" + options.sync_mode
            + ":q" + std::to_string(q) + ":p" + std::to_string(position) + ":" + label;
        nvtxRangePushA(range.c_str());
        // Match the legacy warm protocol: one untimed graph batch or one
        // untimed stream operation precedes each timed position.
        if (config.graph) check(cudaGraphLaunch(timing.executable, stream.value));
        else slot->operation(device_input.data, output, stream.value);
        const auto host_start = Clock::now();
        if (config.graph) check(cudaGraphLaunch(timing.executable, stream.value));
        else {
          check(cudaEventRecord(timing.start.value, stream.value));
          for (int repetition = 0; repetition < options.batch; ++repetition)
            slot->operation(device_input.data, output, stream.value);
          check(cudaEventRecord(timing.end.value, stream.value));
        }
        const auto submitted = Clock::now();
        // A distinct pinned snapshot is enqueued after each timing end, before
        // any subsequent operation can overwrite the common device output.
        check(cudaMemcpyAsync(snapshot, device_output.data, output_words * sizeof(unsigned),
                              cudaMemcpyDeviceToHost, stream.value));
        double position_host = 0;
        if (position_sync) {
          check(cudaStreamSynchronize(stream.value));
          const auto finished = Clock::now();
          if (position == 3) quartet_finished = finished;
          position_host = microseconds(host_start, finished, options.batch);
          verify_words(snapshot);
        }
        nvtxRangePop();
        pending[position] = {q, position, patterns[q], label, slot->api->name, 0,
                             microseconds(host_start, submitted, options.batch), position_host, 0, occurrence};
        // Neither event reads nor synchronization happen here in quartet mode.
        if (position_sync) {
          float milliseconds{};
          check(cudaEventElapsedTime(&milliseconds, timing.start.value, timing.end.value));
          pending[position].event_us = milliseconds * 1000.0 / options.batch;
        }
      }
      if (!position_sync) {
        check(cudaStreamSynchronize(stream.value));
        quartet_finished = Clock::now();
        for (int position = 0; position < 4; ++position) {
          auto& timing = *occurrences[pending[position].timing_pair];
          float milliseconds{};
          check(cudaEventElapsedTime(&milliseconds, timing.start.value, timing.end.value));
          pending[position].event_us = milliseconds * 1000.0 / options.batch;
          verify_words(static_cast<unsigned*>(snapshots->data) + position * output_words);
        }
      }
      const double quartet_host = microseconds(quartet_start, quartet_finished, 1);
      for (auto& m : pending) {
        m.quartet_host_us = quartet_host;
        if (!(m.event_us > 0) || !(m.submit_us > 0) || !(quartet_host > 0) ||
            !std::isfinite(m.event_us) || (position_sync && m.total_host_us < m.submit_us))
          throw std::runtime_error("invalid matched-mode duration");
        measurements.push_back(m);
      }
    }
  }
  std::cerr << "Verified both archived backends; " << measurements.size()
            << " measured positions passed complete output and guard checks in one context/stream.\n"
            << "Same-process diagnostic; graph event nodes exclude host submission. "
            << "Host timings are separate diagnostics, never added to GPU event times.\n";
  std::cout << "case,comparison,slot_a,slot_b,data_seed,order_seed,quartet,pattern,position,slot,backend,"
               "n,bins,tuning,blocks,launch,clear,batch,warmup_ms,event_us,submit_us,total_host_us,"
               "gpu,gpu_uuid,driver,runtime,old_launch_address,new_launch_address,context_address,stream_address";
  if (options.sync_mode != "legacy")
    std::cout << ",sync_mode,snapshot_storage,timing_pair,quartet_host_us,position_host_scope";
  std::cout << '\n';
  std::cout << std::fixed << std::setprecision(9);
  for (const auto& m : measurements) {
    std::cout << options.case_name << ',' << options.comparison << ',' << a_name << ',' << b_name << ','
              << options.data_seed << ',' << options.order_seed << ',' << m.quartet << ',' << m.pattern << ','
              << m.position << ',' << m.slot << ',' << m.backend << ',' << config.n << ',' << config.bins << ','
              << config.tuning << ',' << config.blocks << ',' << (config.graph ? "graph" : "stream") << ','
              << (config.kernel_clear ? "kernel" : "runtime") << ',' << options.batch << ',' << options.warmup_ms << ','
              << m.event_us << ',' << m.submit_us << ',';
    if (options.sync_mode != "quartet") std::cout << m.total_host_us;
    std::cout << ',' << std::quoted(properties.name) << ','
              << uuid_string(properties.uuid) << ',' << driver << ',' << runtime << ','
              << address(reinterpret_cast<std::uintptr_t>(old_api->launch)) << ','
              << address(reinterpret_cast<std::uintptr_t>(new_api->launch)) << ','
              << address(reinterpret_cast<std::uintptr_t>(context)) << ','
              << address(reinterpret_cast<std::uintptr_t>(stream.value));
    if (options.sync_mode != "legacy")
      std::cout << ',' << options.sync_mode << ",pinned_four_snapshots," << m.timing_pair << ','
                << m.quartet_host_us << ','
                << (options.sync_mode == "position" ? "timed_enqueue_through_snapshot_completion_per_operation" : "unobserved");
    std::cout << '\n';
  }
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  try { return run(parse(argc, argv)); }
  catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
