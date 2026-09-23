#include "common.hpp"

#include <array>
#include <climits>
#include <iostream>
#include <limits>

namespace {

constexpr std::size_t guard_bytes = 256;
std::size_t cases = 0, executions = 0, structural_checks = 0;

gh::Config make_config(int tuning, unsigned bins, std::size_t size, unsigned window_bins = 524288) {
  gh::Config config;
  config.algorithm = gh::Algorithm::global_window;
  config.input_type = gh::InputType::u32;
  config.counter_type = gh::CounterType::u64;
  config.local_counter = gh::LocalCounter::u32;
  config.size = size;
  config.bins = bins;
  config.window_bins = window_bins;
  config.tuning = tuning;
  config.blocks = 3;
  return config;
}

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

void structural_cases() {
  const auto accept = [](const gh::Config& config) {
    require(gh::supported(config), "valid global-window shape rejected");
    // Scalar explicit preparation and workspace queries invoke no CUDA API.
    require(gh::prepare(config) == cudaSuccess, "valid explicit preparation failed");
    std::size_t bytes = 123;
    require(gh::workspace_bytes(config, bytes) == cudaSuccess, "workspace query failed");
    const std::size_t expected = config.size
        ? static_cast<std::size_t>(std::min(config.bins, config.window_bins)) * sizeof(unsigned) : 0;
    require(bytes == expected, "incorrect global-window scratch size");
    ++structural_checks;
  };
  const auto reject = [](const gh::Config& config) {
    std::size_t bytes = 123;
    require(!gh::supported(config), "invalid global-window shape accepted");
    require(gh::workspace_bytes(config, bytes) == cudaErrorInvalidValue && bytes == 0,
            "invalid workspace query accepted or size not reset");
    require(gh::prepare(config) == cudaErrorInvalidValue, "invalid preparation accepted");
    ++structural_checks;
  };
  for (int tuning = 0; tuning < 6; ++tuning) {
    for (unsigned bins : {1u, 17u, 524287u, 524288u, 524289u, 1048576u,
                          static_cast<unsigned>(INT_MAX - 1)})
      for (unsigned window : {1u, 17u, 262144u, 524288u, 1048576u,
                              static_cast<unsigned>(INT_MAX - 1)})
        for (std::size_t size : {std::size_t{0}, std::size_t{33},
                                 static_cast<std::size_t>(UINT_MAX)})
          accept(make_config(tuning, bins, size, window));
  }
  const auto config = make_config(0, 1048576, 33);
  auto bad = config; bad.size = static_cast<std::size_t>(UINT_MAX) + 1; reject(bad);
  bad = config; bad.size = std::numeric_limits<std::size_t>::max(); reject(bad);
  for (auto bins : {0u, static_cast<unsigned>(INT_MAX), UINT_MAX}) {
    bad = config; bad.bins = bins; reject(bad);
    bad = config; bad.window_bins = bins; reject(bad);
  }
  for (int tuning : {-1, 6, 7, 14, 15, static_cast<int>(gh::tuning_count)}) {
    bad = config; bad.tuning = tuning; reject(bad);
  }
  for (int blocks : {0, -1}) { bad = config; bad.blocks = blocks; reject(bad); }
  bad = config; bad.input_type = gh::InputType::u8; reject(bad);
  bad = config; bad.counter_type = gh::CounterType::u32; reject(bad);
  bad = config; bad.local_counter = gh::LocalCounter::native; reject(bad);
  bad = config; bad.output_clear = static_cast<gh::OutputClear>(99); reject(bad);
  bad = config; bad.launch = static_cast<gh::LaunchMode>(99); reject(bad);
  bad = config; bad.cache = static_cast<gh::CacheMode>(99); reject(bad);

  // Invalid arguments must fail before accessing even these host-only sentinel
  // pointers. No GPU call is needed to verify short/null scratch rejection.
  alignas(16) std::array<std::uint64_t, 8> sentinel{};
  void* pointer = sentinel.data();
  std::size_t required = 0;
  require(gh::workspace_bytes(config, required) == cudaSuccess, "workspace query failed");
  for (auto bytes : {std::size_t{0}, required - 1}) {
    require(gh::histogram(config, pointer, pointer, pointer, bytes) == cudaErrorInvalidValue,
            "short scratch accepted");
    ++structural_checks;
  }
  require(gh::histogram(config, pointer, pointer, nullptr, required) == cudaErrorInvalidValue,
          "null scratch accepted");
  require(gh::histogram(config, nullptr, pointer, pointer, required) == cudaErrorInvalidValue,
          "null nonempty input accepted");
  require(gh::histogram(config, pointer, nullptr, pointer, required) == cudaErrorInvalidValue,
          "null output accepted");
  structural_checks += 3;
}

struct GuardedBuffer {
  DeviceBuffer storage;
  std::size_t bytes, prefix;
  explicit GuardedBuffer(std::size_t bytes, std::size_t offset = 0)
      : storage(bytes + 2 * guard_bytes + offset), bytes(bytes), prefix(guard_bytes + offset) {}
  void* data() const { return static_cast<unsigned char*>(storage.data) + prefix; }
  void poison(cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(storage.data, 0xa5, storage.bytes, stream));
  }
  void verify_guards() const {
    std::vector<unsigned char> actual(prefix);
    CUDA_CHECK(cudaMemcpy(actual.data(), storage.data, prefix, cudaMemcpyDeviceToHost));
    require(std::all_of(actual.begin(), actual.end(), [](auto byte) { return byte == 0xa5; }),
            "global-window leading canary overwritten");
    actual.resize(guard_bytes);
    CUDA_CHECK(cudaMemcpy(actual.data(), static_cast<unsigned char*>(storage.data) + prefix + bytes,
                          guard_bytes, cudaMemcpyDeviceToHost));
    require(std::all_of(actual.begin(), actual.end(), [](auto byte) { return byte == 0xa5; }),
            "global-window trailing canary overwritten");
  }
};

struct Graph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  ~Graph() {
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
  }
};

void make_input(const gh::Config& config, int phase, std::vector<unsigned>& keys,
                std::vector<std::uint64_t>& expected) {
  std::fill(expected.begin(), expected.end(), 0);
  const unsigned boundary = std::min(config.bins, config.window_bins);
  const std::array<unsigned, 7> boundaries{
      0, boundary - 1, std::min(boundary, config.bins - 1),
      std::min(boundary + 1, config.bins - 1), config.bins - 1,
      (config.bins - 1) / 2, std::min(config.window_bins, config.bins - 1)};
  std::mt19937_64 random(731 + phase);
  for (std::size_t index = 0; index < keys.size(); ++index) {
    const unsigned key = phase == 1 ? config.bins - 1 : phase == 2 ? 0
        : index % 2 == 0 ? boundaries[random() % boundaries.size()]
                        : static_cast<unsigned>(random() % config.bins);
    keys[index] = key;
    ++expected[key];
  }
  if (phase == 3) std::sort(keys.begin(), keys.end());
}

void run_case(gh::Config config, std::size_t offset, bool graph_mode,
              gh::OutputClear clear = gh::OutputClear::runtime) try {
  config.launch = graph_mode ? gh::LaunchMode::graph : gh::LaunchMode::stream;
  config.output_clear = clear;
  CUDA_CHECK(gh::prepare(config));
  std::size_t bytes = 0;
  CUDA_CHECK(gh::workspace_bytes(config, bytes));
  Stream stream;
  GuardedBuffer input(config.size * sizeof(unsigned), offset);
  GuardedBuffer output(static_cast<std::size_t>(config.bins) * sizeof(std::uint64_t));
  GuardedBuffer workspace(bytes);
  require(reinterpret_cast<std::uintptr_t>(input.data()) % 16 == offset,
          "input alignment case not obtained");
  const void* samples = config.size ? input.data() : nullptr;
  void* scratch = bytes ? workspace.data() : nullptr;
  input.poison(stream.value);
  output.poison(stream.value);
  workspace.poison(stream.value);
  CUDA_CHECK(cudaStreamSynchronize(stream.value));

  Graph graph;
  if (graph_mode) {
    CUDA_CHECK(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
    const auto status = gh::histogram(config, samples, output.data(), scratch, bytes, stream.value);
    const auto end_status = cudaStreamEndCapture(stream.value, &graph.graph);
    CUDA_CHECK(status);
    CUDA_CHECK(end_status);
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph.executable, graph.graph, 0));
  }
  std::vector<unsigned> keys(config.size), copied(config.size);
  std::vector<std::uint64_t> expected(config.bins);
  for (int phase = 0; phase < 4; ++phase) {
    make_input(config, phase, keys, expected);
    if (config.size)
      CUDA_CHECK(cudaMemcpyAsync(input.data(), keys.data(), keys.size() * sizeof(unsigned),
                                cudaMemcpyHostToDevice, stream.value));
    if (graph_mode) {
      CUDA_CHECK(cudaGraphLaunch(graph.executable, stream.value));
    } else {
      config.output_clear = phase % 2 ? gh::OutputClear::kernel : gh::OutputClear::runtime;
      CUDA_CHECK(gh::histogram(config, samples, output.data(), scratch, bytes, stream.value));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
    verify_output(config, output.data(), expected);
    if (config.size) {
      CUDA_CHECK(cudaMemcpy(copied.data(), input.data(), keys.size() * sizeof(unsigned),
                            cudaMemcpyDeviceToHost));
      require(copied == keys, "input was modified");
    }
    input.verify_guards(); output.verify_guards(); workspace.verify_guards();
    ++executions;
  }
  ++cases;
} catch (const std::exception& error) {
  throw std::runtime_error(std::string(error.what()) + " tuning=" + std::to_string(config.tuning) +
      " bins=" + std::to_string(config.bins) + " n=" + std::to_string(config.size) +
      " window=" + std::to_string(config.window_bins) + " offset=" + std::to_string(offset) +
      " graph=" + std::to_string(graph_mode) + " clear=" + gh::name(config.output_clear));
}

void gpu_cases(bool sanitizer) {
  for (int tuning = 0; tuning < 6; ++tuning) {
    const auto policy = gh::tuning_catalog[tuning];
    const std::size_t tile = static_cast<std::size_t>(policy.threads) * policy.items;
    const auto sizes = sanitizer ? std::vector<std::size_t>{33, tile + 1, 6 * tile + 33}
        : std::vector<std::size_t>{0, 1, 31, 32, 33, tile - 1, tile, tile + 1, 6 * tile + 33};
    for (auto size : sizes)
      for (auto offset : {std::size_t{0}, sizeof(unsigned)})
        run_case(make_config(tuning, 53, size, 17), offset, false);
    for (auto clear : {gh::OutputClear::runtime, gh::OutputClear::kernel})
      for (auto offset : {std::size_t{0}, sizeof(unsigned)})
        run_case(make_config(tuning, 53, 3 * tile + 33, 17), offset, true, clear);
  }
  // Actual cache-sized ranges include one-pass controls, exact boundaries,
  // final partial windows, and windows larger than the entire histogram.
  for (unsigned bins : {524287u, 524288u, 524289u, 786432u, 1048576u, 1048577u})
    for (unsigned window : {262144u, 524288u, 1048576u})
      run_case(make_config(0, bins, 1537, window), sizeof(unsigned), false);
  for (auto clear : {gh::OutputClear::runtime, gh::OutputClear::kernel}) {
    run_case(make_config(0, 1048577, 1537), sizeof(unsigned), true, clear);
    run_case(make_config(0, 524289, 0), 0, true, clear);
  }
  for (unsigned window : {1u, 17u, 524288u})
    run_case(make_config(0, 1, 513, window), sizeof(unsigned), false);
}

}  // namespace

int main(int argc, char** argv) try {
  bool sanitizer = false, structural_only = false;
  for (int argument = 1; argument < argc; ++argument) {
    const std::string value = argv[argument];
    if (value == "--sanitizer") sanitizer = true;
    else if (value == "--structural-only") structural_only = true;
    else throw std::runtime_error("usage: histogram_global_window_tests [--sanitizer] [--structural-only]");
  }
  structural_cases();
  if (!structural_only) gpu_cases(sanitizer);
  std::cout << "PASS: " << structural_checks << " structural checks, " << cases
            << " cases, " << executions << " verified global-window executions.\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "FAIL: " << error.what() << '\n';
  return 1;
}
