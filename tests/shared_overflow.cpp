#include "common.hpp"

#include <array>
#include <iostream>
#include <limits>

namespace {

constexpr unsigned prefix_bins = 24576;
constexpr std::size_t guard_bytes = 256;
std::size_t executions = 0;
std::size_t cases = 0;
std::size_t structural_checks = 0;

struct GuardedOutput {
  DeviceBuffer storage;
  std::size_t bytes;
  explicit GuardedOutput(unsigned bins)
      : storage(static_cast<std::size_t>(bins) * sizeof(std::uint64_t) + 2 * guard_bytes),
        bytes(static_cast<std::size_t>(bins) * sizeof(std::uint64_t)) {}
  void* data() const { return static_cast<unsigned char*>(storage.data) + guard_bytes; }
  void poison(cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(storage.data, 0xa5, storage.bytes, stream));
  }
  void verify_guards() const {
    std::array<unsigned char, guard_bytes> actual{};
    for (auto offset : {std::size_t{0}, guard_bytes + bytes}) {
      CUDA_CHECK(cudaMemcpy(actual.data(), static_cast<unsigned char*>(storage.data) + offset,
                            actual.size(), cudaMemcpyDeviceToHost));
      if (!std::all_of(actual.begin(), actual.end(), [](unsigned char value) { return value == 0xa5; }))
        throw std::runtime_error("shared-overflow output guard overwritten");
    }
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

gh::Config make_config(int tuning, unsigned bins, std::size_t size, int blocks = 3) {
  gh::Config config;
  config.algorithm = gh::Algorithm::shared_overflow;
  config.input_type = gh::InputType::u32;
  config.counter_type = gh::CounterType::u64;
  config.local_counter = gh::LocalCounter::u32;
  config.tuning = tuning;
  config.bins = bins;
  config.size = size;
  config.blocks = blocks;
  return config;
}

void structural_cases() {
  constexpr std::size_t limit = std::numeric_limits<unsigned>::max();
  for (int tuning : {14, 15}) {
    auto config = make_config(tuning, prefix_bins + 1, 0);
    const auto expect_supported = [](const gh::Config& candidate) {
      if (!gh::supported(candidate)) throw std::runtime_error("valid shared-overflow shape rejected");
      std::size_t bytes = 123;
      CUDA_CHECK(gh::workspace_bytes(candidate, bytes));
      if (bytes != 0) throw std::runtime_error("shared-overflow unexpectedly requires scratch");
      ++structural_checks;
    };
    const auto expect_rejected = [](const gh::Config& candidate) {
      std::size_t bytes = 123;
      if (gh::supported(candidate) ||
          gh::workspace_bytes(candidate, bytes) != cudaErrorInvalidValue || bytes != 0)
        throw std::runtime_error("invalid shared-overflow shape accepted or workspace not reset");
      ++structural_checks;
    };
    expect_supported(config);
    for (int blocks : {1, 2, 7, 192}) {
      config.blocks = blocks;
      const auto policy = gh::tuning_catalog[tuning];
      const std::size_t tile = static_cast<std::size_t>(policy.threads) * policy.items;
      // At this exact extent CTA0 owns UINT_MAX samples; one additional sample
      // overflows that CTA even when the total input divided by blocks still fits.
      // These are structural queries only: no large allocation or GPU launch.
      const std::size_t boundary = (limit / tile) * static_cast<std::size_t>(blocks) * tile + limit % tile;
      for (auto size : {boundary - 1, boundary}) {
        config.size = size;
        expect_supported(config);
      }
      config.size = boundary + 1;
      expect_rejected(config);
    }
    config = make_config(tuning, prefix_bins + 1, 33);
    for (auto bins : {0u, prefix_bins - 1, prefix_bins}) {
      auto bad = config; bad.bins = bins; expect_rejected(bad);
    }
    for (int invalid : {-1, 0, 13, static_cast<int>(gh::tuning_count)}) {
      auto bad = config; bad.tuning = invalid; expect_rejected(bad);
    }
    for (int blocks : {0, -1}) {
      auto bad = config; bad.blocks = blocks; expect_rejected(bad);
    }
    auto bad = config; bad.input_type = gh::InputType::u8; expect_rejected(bad);
    bad = config; bad.counter_type = gh::CounterType::u32; expect_rejected(bad);
    bad = config; bad.local_counter = gh::LocalCounter::native; expect_rejected(bad);
  }
}

// First exercise both sides of the shared/global boundary, including absent
// output bins. Subsequent replays move every sample first to the final global
// bin, then to the final shared bin; stale results must disappear each time.
void make_input(std::vector<unsigned>& keys, std::vector<std::uint64_t>& expected,
                unsigned bins, int phase) {
  std::fill(expected.begin(), expected.end(), 0);
  const std::array<unsigned, 6> boundary_keys{
      0, prefix_bins - 2, prefix_bins - 1, prefix_bins,
      std::min(prefix_bins + 1, bins - 1), bins - 1};
  std::mt19937_64 random(987123);
  for (auto& key : keys) {
    key = phase == 0 ? boundary_keys[random() % boundary_keys.size()]
                    : phase == 1 ? bins - 1 : prefix_bins - 1;
    ++expected[key];
  }
}

void run_case(gh::Config config, std::size_t input_offset, bool graph_mode,
              gh::OutputClear graph_clear = gh::OutputClear::runtime) try {
  if (!gh::supported(config)) throw std::runtime_error("expected GPU test shape unsupported");
  config.launch = graph_mode ? gh::LaunchMode::graph : gh::LaunchMode::stream;
  config.output_clear = graph_clear;
  CUDA_CHECK(gh::prepare(config));
  std::size_t bytes = 123;
  CUDA_CHECK(gh::workspace_bytes(config, bytes));
  if (bytes != 0) throw std::runtime_error("shared-overflow workspace is not zero");

  Stream stream;
  DeviceBuffer input(config.size * sizeof(unsigned) + input_offset);
  auto* input_data = static_cast<unsigned char*>(input.data) + input_offset;
  const void* samples = config.size ? input_data : nullptr;
  if (reinterpret_cast<std::uintptr_t>(input_data) % 16 != input_offset)
    throw std::runtime_error("test did not obtain the requested input alignment");
  GuardedOutput output(config.bins);
  std::vector<unsigned> keys(config.size);
  std::vector<std::uint64_t> expected(config.bins);
  output.poison(stream.value);
  CUDA_CHECK(cudaStreamSynchronize(stream.value));

  Graph graph;
  if (graph_mode) {
    CUDA_CHECK(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
    const auto status = gh::histogram(config, samples, output.data(), nullptr, 0, stream.value);
    const auto end_status = cudaStreamEndCapture(stream.value, &graph.graph);
    CUDA_CHECK(status);
    CUDA_CHECK(end_status);
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph.executable, graph.graph, 0));
  }
  for (int phase = 0; phase < 3; ++phase) {
    make_input(keys, expected, config.bins, phase);
    if (config.size)
      CUDA_CHECK(cudaMemcpyAsync(input_data, keys.data(), keys.size() * sizeof(unsigned),
                                cudaMemcpyHostToDevice, stream.value));
    if (graph_mode) {
      CUDA_CHECK(cudaGraphLaunch(graph.executable, stream.value));
    } else {
      // Both stream clear modes see poisoned or previously nonzero output.
      config.output_clear = phase == 1 ? gh::OutputClear::kernel : gh::OutputClear::runtime;
      CUDA_CHECK(gh::histogram(config, samples, output.data(), nullptr, 0, stream.value));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
    verify_output(config, output.data(), expected);
    output.verify_guards();
    ++executions;
  }
  ++cases;
} catch (const std::exception& error) {
  throw std::runtime_error(std::string(error.what()) + " tuning=" + std::to_string(config.tuning) +
      " bins=" + std::to_string(config.bins) + " n=" + std::to_string(config.size) +
      " blocks=" + std::to_string(config.blocks) + " input_offset=" + std::to_string(input_offset) +
      " graph=" + std::to_string(graph_mode) + " clear=" + gh::name(config.output_clear));
}

void gpu_cases(bool sanitizer) {
  for (int tuning : {14, 15}) {
    const auto policy = gh::tuning_catalog[tuning];
    const std::size_t tile = static_cast<std::size_t>(policy.threads) * policy.items;
    const auto sizes = sanitizer ? std::vector<std::size_t>{33, tile, 3 * tile + 33}
        : std::vector<std::size_t>{0, 1, 31, 32, 33, tile - 1, tile, tile + 1, 6 * tile + 33};
    for (unsigned bins : {24577u, 32768u, 65536u})
      for (auto size : sizes)
        for (std::size_t offset : {std::size_t{0}, sizeof(unsigned)})
          run_case(make_config(tuning, bins, size), offset, false);

    // Nondivisible million-bin output checks sparse coverage and the final
    // clearing block; the input itself remains small even under sanitizers.
    for (std::size_t offset : {std::size_t{0}, sizeof(unsigned)}) {
      run_case(make_config(tuning, 1048577, tile + 33), offset, false);
      for (auto clear : {gh::OutputClear::runtime, gh::OutputClear::kernel})
        run_case(make_config(tuning, 32768, 3 * tile + 33), offset, true, clear);
    }
    // Empty graph input explicitly passes null input and null scratch. Replays
    // must still clear all bins, regardless of the configured clearing policy.
    for (auto clear : {gh::OutputClear::runtime, gh::OutputClear::kernel})
      run_case(make_config(tuning, 24577, 0), 0, true, clear);
  }
}

}  // namespace

int main(int argc, char** argv) try {
  bool sanitizer = false;
  for (int argument = 1; argument < argc; ++argument) {
    if (std::string(argv[argument]) == "--sanitizer") sanitizer = true;
    else throw std::runtime_error("usage: histogram_shared_overflow_tests [--sanitizer]");
  }
  structural_cases();
  int device = 0, capacity = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(&capacity, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  if (capacity < 96 * 1024) {
    std::cout << "SKIP: shared-overflow requires 96 KiB opt-in shared memory.\n";
    return 77;
  }
  gpu_cases(sanitizer);
  std::cout << "PASS: " << structural_checks << " structural checks, " << cases
            << " cases, " << executions << " verified shared-overflow executions.\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "FAIL: " << error.what() << '\n';
  return 1;
}
