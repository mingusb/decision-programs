#include "common.hpp"
#include <array>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>

namespace {
constexpr std::size_t guard = 256;
constexpr gh::Algorithm shared_algorithms[] = {
  gh::Algorithm::shared_atomic, gh::Algorithm::shared_rle,
  gh::Algorithm::shared_warp, gh::Algorithm::shared_partial};
constexpr gh::Algorithm global_algorithms[] = {
  gh::Algorithm::global_atomic, gh::Algorithm::warp_aggregated};
struct GuardedBuffer {
  DeviceBuffer storage;
  std::size_t size;
  explicit GuardedBuffer(std::size_t bytes) : storage(bytes + 2 * guard), size(bytes) {}
  void* data() const { return static_cast<unsigned char*>(storage.data) + guard; }
  void poison(cudaStream_t stream) { CUDA_CHECK(cudaMemsetAsync(storage.data, 0xa5, storage.bytes, stream)); }
  void verify() const {
    std::array<unsigned char, guard> bytes{};
    for (std::size_t offset : {std::size_t{0}, guard + size}) {
      CUDA_CHECK(cudaMemcpy(bytes.data(), static_cast<unsigned char*>(storage.data) + offset, guard, cudaMemcpyDeviceToHost));
      if (!std::all_of(bytes.begin(), bytes.end(), [](auto x) { return x == 0xa5; }))
        throw std::runtime_error("output/workspace canary overwritten");
    }
  }
};
std::size_t executions = 0;
void run_case(gh::Config config, const char* distribution, const char* order,
              std::optional<std::size_t> input_offset = std::nullopt) try {
  if (!gh::supported(config)) return;
  CUDA_CHECK(gh::prepare(config));
  std::size_t bytes = 0;
  CUDA_CHECK(gh::workspace_bytes(config, bytes));
  Dataset dataset(config.size, config.bins, config.input_type, distribution, order, 987123);
  // Every production kernel needs only natural input alignment.
  const auto element_bytes = gh::input_bytes(config.input_type);
  const auto offset = input_offset.value_or(element_bytes);
  if (offset % element_bytes) throw std::runtime_error("test input offset violates natural alignment");
  DeviceBuffer input(config.size * element_bytes + offset);
  auto* input_data = static_cast<unsigned char*>(input.data) + offset;
  GuardedBuffer output(config.bins * gh::counter_bytes(config.counter_type));
  GuardedBuffer workspace(bytes);
  Stream stream;
  if (config.size) CUDA_CHECK(cudaMemcpyAsync(input_data, dataset.data(config.input_type), config.size * element_bytes,
                                            cudaMemcpyHostToDevice, stream.value));
  output.poison(stream.value);
  workspace.poison(stream.value);
  CUDA_CHECK(gh::histogram(config, input_data, output.data(), workspace.data(), bytes, stream.value));
  CUDA_CHECK(cudaStreamSynchronize(stream.value));
  verify_output(config, output.data(), dataset.expected);
  output.verify(); workspace.verify();
  // A second call must overwrite, rather than accumulate the previous result.
  // Exercise both initialization policies on the same data and existing output.
  config.output_clear = gh::OutputClear::kernel;
  CUDA_CHECK(gh::histogram(config, input_data, output.data(), workspace.data(), bytes, stream.value));
  CUDA_CHECK(cudaStreamSynchronize(stream.value));
  verify_output(config, output.data(), dataset.expected);
  output.verify(); workspace.verify();
  executions += 2;
} catch (const std::exception& error) {
  throw std::runtime_error(std::string(error.what()) + " input=" + gh::name(config.input_type) +
                           " counter=" + gh::name(config.counter_type) + " distribution=" + distribution +
                           " order=" + order + " blocks=" + std::to_string(config.blocks) +
                           " local=" + gh::name(config.local_counter) +
                           " input_offset=" + (input_offset ? std::to_string(*input_offset) : "default"));
}
void automatic_cases() {
  gh::Config pending;
  pending.input_type = gh::InputType::u8;
  pending.size = 4096;
  pending.launch = gh::LaunchMode::graph;
  if (pending.algorithm != gh::Algorithm::automatic || !gh::supported(pending))
    throw std::runtime_error("default configuration is not a supported automatic request");
  std::size_t bytes = 123;
  DeviceBuffer input(pending.size), output(pending.bins * sizeof(unsigned));
  // Unresolved defaults cannot size a different backend's workspace or launch it.
  const auto& unresolved = pending;
  if (gh::prepare(unresolved) != cudaErrorInvalidValue ||
      gh::workspace_bytes(pending, bytes) != cudaErrorInvalidValue || bytes != 0 ||
      gh::histogram(pending, input.data, output.data, nullptr, 0) != cudaErrorInvalidValue)
    throw std::runtime_error("unprepared automatic configuration accepted");
  CUDA_CHECK(gh::prepare(pending));
  if (pending.algorithm == gh::Algorithm::automatic)
    throw std::runtime_error("preparation did not freeze a concrete default");
  CUDA_CHECK(gh::workspace_bytes(pending, bytes));
  run_case(pending, "uniform", "shuffled");
  run_case(pending, "single", "shuffled");

  gh::Config stream;
  stream.size = 1 << 20;
  stream.bins = 4096;
  run_case(stream, "uniform", "shuffled");
  stream.launch = gh::LaunchMode::graph;
  stream.bins = 256;
  run_case(stream, "hot99", "sorted");

  // Unmeasured sizes and empty input must remain usable through the fallback.
  gh::Config fallback;
  fallback.size = 4099;
  fallback.bins = 257;
  fallback.counter_type = gh::CounterType::u64;
  run_case(fallback, "two", "shuffled");
  fallback.size = 0;
  run_case(fallback, "uniform", "shuffled");
}

void invalid_configs() {
  DeviceBuffer input(sizeof(unsigned));
  GuardedBuffer output(256 * sizeof(std::uint64_t));
  Stream stream;
  output.poison(stream.value);
  const auto expect_invalid = [](cudaError_t status, const char* what) {
    if (status != cudaErrorInvalidValue)
      throw std::runtime_error(std::string(what) + " did not return cudaErrorInvalidValue");
  };
  const auto reject_config = [&](const gh::Config& candidate) {
    if (gh::supported(candidate)) throw std::runtime_error("invalid configuration accepted");
    std::size_t bytes = 123;
    expect_invalid(gh::prepare(candidate), "invalid preparation");
    expect_invalid(gh::workspace_bytes(candidate, bytes), "invalid workspace query");
    expect_invalid(gh::histogram(candidate, input.data, output.data(), nullptr, 0, stream.value),
                   "invalid histogram configuration");
  };
  gh::Config config;
  for (auto tuning : {-1, static_cast<int>(gh::tuning_count)}) {
    config = {}; config.tuning = tuning; reject_config(config);
  }
  for (auto blocks : {0, -1}) { config = {}; config.blocks = blocks; reject_config(config); }
  for (auto bins : {0u, static_cast<unsigned>(std::numeric_limits<int>::max()),
                    std::numeric_limits<unsigned>::max()}) {
    config = {}; config.bins = bins; reject_config(config);
  }
  config = {}; config.algorithm = static_cast<gh::Algorithm>(-1); reject_config(config);
  config = {}; config.input_type = static_cast<gh::InputType>(-1); reject_config(config);
  config = {}; config.counter_type = static_cast<gh::CounterType>(-1); reject_config(config);
  config = {}; config.local_counter = static_cast<gh::LocalCounter>(-1); reject_config(config);
  config = {}; config.output_clear = static_cast<gh::OutputClear>(-1); reject_config(config);
  config = {}; config.launch = static_cast<gh::LaunchMode>(-1); reject_config(config);
  config = {}; config.cache = static_cast<gh::CacheMode>(-1); reject_config(config);
  config = {};
  config.size = std::size_t{std::numeric_limits<unsigned>::max()} + 1;
  reject_config(config);
  config.counter_type = gh::CounterType::u64;
  if (!gh::supported(config)) throw std::runtime_error("64-bit count domain rejected");
  config.size = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(unsigned) + 1;
  reject_config(config);
  config = {}; config.input_type = gh::InputType::u8; config.bins = 257; reject_config(config);
  config = {}; config.algorithm = gh::Algorithm::bitplane; config.bins = 257; reject_config(config);
  for (auto algorithm : algorithms) {
    config = {}; config.algorithm = algorithm; config.local_counter = gh::LocalCounter::u32;
    reject_config(config);  // Explicit local-u32 policy requires u64 public counts.
    if (algorithm == gh::Algorithm::bitplane) {
      config.counter_type = gh::CounterType::u64;
      reject_config(config);
    }
  }
  for (auto algorithm : {gh::Algorithm::shared_atomic, gh::Algorithm::shared_rle,
                         gh::Algorithm::shared_warp, gh::Algorithm::shared_partial}) {
    for (auto counter : {gh::CounterType::u32, gh::CounterType::u64})
      for (int tuning = 0; tuning < static_cast<int>(gh::tuning_count); ++tuning) {
        config = {}; config.algorithm = algorithm; config.counter_type = counter; config.tuning = tuning;
        config.bins = static_cast<unsigned>(gh::tuning_catalog[tuning].shared_limit /
            (gh::counter_bytes(counter) * gh::tuning_catalog[tuning].replicas)) + 1;
        reject_config(config);
      }
  }
  for (int tuning = 0; tuning < static_cast<int>(gh::tuning_count); ++tuning) {
    const auto policy = gh::tuning_catalog[tuning];
    if (policy.load != gh::LoadPolicy::scalar) {
      for (auto algorithm : {gh::Algorithm::global_atomic, gh::Algorithm::warp_aggregated}) {
        config = {}; config.algorithm = algorithm; config.tuning = tuning;
        config.input_type = gh::InputType::u8;
        reject_config(config);
        config.counter_type = gh::CounterType::u64;
        config.local_counter = gh::LocalCounter::u32;
        reject_config(config);
      }
    }
    if (policy.shared_limit > 48 * 1024) {
      config = {}; config.algorithm = gh::Algorithm::bitplane; config.tuning = tuning;
      reject_config(config);  // Opt-in dynamic capacity is a shared-family policy.
    }
    if (policy.threads == 1024) {
      config = {}; config.algorithm = gh::Algorithm::bitplane; config.tuning = tuning;
      config.counter_type = gh::CounterType::u64; config.bins = 129;
      reject_config(config);  // Rounded capacity 256 needs 64 KiB of static shared memory.
      config.bins = 128;
      if (!gh::supported(config)) throw std::runtime_error("valid bitplane static-shared boundary rejected");
    }
  }

  config = {}; config.algorithm = gh::Algorithm::global_atomic; config.size = 1;
  expect_invalid(gh::histogram(config, nullptr, output.data(), nullptr, 0, stream.value), "null input");
  expect_invalid(gh::histogram(config, input.data, nullptr, nullptr, 0, stream.value), "null output");
  for (auto algorithm : {gh::Algorithm::shared_partial}) {
    config.algorithm = algorithm;
    std::size_t bytes = 0;
    CUDA_CHECK(gh::workspace_bytes(config, bytes));
    if (!bytes) throw std::runtime_error("workspace backend reported zero scratch bytes");
    GuardedBuffer workspace(bytes);
    expect_invalid(gh::histogram(config, input.data, output.data(), nullptr, bytes, stream.value), "null workspace");
    expect_invalid(gh::histogram(config, input.data, output.data(), workspace.data(), bytes - 1, stream.value),
                   "insufficient workspace");
  }
  for (auto algorithm : global_algorithms) {
    config = {}; config.algorithm = algorithm; config.size = 1;
    config.counter_type = gh::CounterType::u64; config.local_counter = gh::LocalCounter::u32;
    std::size_t bytes = 0;
    CUDA_CHECK(gh::workspace_bytes(config, bytes));
    if (bytes != config.bins * sizeof(unsigned))
      throw std::runtime_error("narrowed global workspace is not one u32 counter per bin");
    GuardedBuffer workspace(bytes);
    for (auto clear : {gh::OutputClear::runtime, gh::OutputClear::kernel}) {
      config.output_clear = clear;
      expect_invalid(gh::histogram(config, input.data, output.data(), nullptr, bytes, stream.value),
                     "null narrowed global workspace");
      expect_invalid(gh::histogram(config, input.data, output.data(), workspace.data(), bytes - 1, stream.value),
                     "insufficient narrowed global workspace");
    }
    config.size = 0;
    bytes = 123;
    CUDA_CHECK(gh::workspace_bytes(config, bytes));
    if (bytes) throw std::runtime_error("empty narrowed global histogram requires scratch");
    for (auto clear : {gh::OutputClear::runtime, gh::OutputClear::kernel}) {
      config.output_clear = clear;
      output.poison(stream.value);
      CUDA_CHECK(gh::histogram(config, nullptr, output.data(), nullptr, 0, stream.value));
      CUDA_CHECK(cudaStreamSynchronize(stream.value));
      verify_output(config, output.data(), std::vector<std::uint64_t>(config.bins, 0));
    }
  }
  // Empty input may be null, and requires no scratch even for workspace backends.
  config = {};
  config.size = 0;
  for (auto algorithm : algorithms) {
    config.algorithm = algorithm;
    config.input_type = gh::InputType::u32;
    // Some explicit backends have a restricted domain even for empty input;
    // shared_overflow's large-bin empty case has its own focused test suite.
    if (!gh::supported(config)) continue;
    CUDA_CHECK(gh::prepare(config));
    CUDA_CHECK(gh::histogram(config, nullptr, output.data(), nullptr, 0, stream.value));
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
    verify_output(config, output.data(), std::vector<std::uint64_t>(config.bins, 0));
  }
  output.verify();
}

void local_counter_boundaries() {
  constexpr std::size_t limit = std::numeric_limits<unsigned>::max();
  // Unlike private shared counters, all CTAs update this one global scratch
  // histogram. More blocks can never make a total count above UINT_MAX safe.
  for (auto algorithm : global_algorithms) for (int tuning = 0; tuning < 6; ++tuning)
    for (int blocks : {1, 7, 192, std::numeric_limits<int>::max()}) {
      gh::Config config;
      config.algorithm = algorithm; config.counter_type = gh::CounterType::u64;
      config.local_counter = gh::LocalCounter::u32;
      config.bins = 257; config.tuning = tuning; config.blocks = blocks;
      for (auto size : {limit - 1, limit}) {
        config.size = size;
        if (!gh::supported(config)) throw std::runtime_error("safe narrowed global total-count boundary rejected");
        std::size_t bytes = 0;
        CUDA_CHECK(gh::workspace_bytes(config, bytes));
        if (bytes != config.bins * sizeof(unsigned))
          throw std::runtime_error("narrowed global workspace changed at the total-count boundary");
      }
      config.size = limit + 1;
      std::size_t bytes = 123;
      if (gh::supported(config) || gh::workspace_bytes(config, bytes) != cudaErrorInvalidValue)
        throw std::runtime_error("overflowing narrowed global total count accepted");
      config.local_counter = gh::LocalCounter::native;
      if (!gh::supported(config)) throw std::runtime_error("native global u64 total-count boundary rejected");
    }
  for (auto algorithm : shared_algorithms) for (int tuning = 0; tuning < static_cast<int>(gh::tuning_count); ++tuning)
    for (int blocks : {1, 2, 7, 192, std::numeric_limits<int>::max()}) {
      gh::Config config;
      config.algorithm = algorithm; config.input_type = gh::InputType::u8;
      config.counter_type = gh::CounterType::u64; config.local_counter = gh::LocalCounter::u32;
      config.bins = 1; config.tuning = tuning; config.blocks = blocks;
      const auto policy = gh::tuning_catalog[tuning];
      const std::size_t tile = static_cast<std::size_t>(policy.threads) * policy.items;
      // Exact boundary: full grid rounds followed by tile-1 elements in CTA0.
      // Adding one element overflows that CTA although ceil(N/blocks) can fit.
      config.size = (limit / tile) * static_cast<std::size_t>(blocks) * tile + limit % tile;
      if (!gh::supported(config)) throw std::runtime_error("safe local-u32 boundary rejected");
      ++config.size;
      if (gh::supported(config)) throw std::runtime_error("overflowing local-u32 CTA accepted");
      std::size_t bytes = 0;
      if (gh::workspace_bytes(config, bytes) != cudaErrorInvalidValue)
        throw std::runtime_error("overflowing local-u32 workspace query accepted");
      config.local_counter = gh::LocalCounter::native;
      if (!gh::supported(config)) throw std::runtime_error("native u64 CTA boundary rejected");
    }
  for (int tuning = 0; tuning < static_cast<int>(gh::tuning_count); ++tuning) {
    gh::Config config;
    config.algorithm = gh::Algorithm::shared_partial; config.counter_type = gh::CounterType::u64;
    config.tuning = tuning; config.size = 4099;
    std::size_t native_bytes = 0, narrow_bytes = 0;
    CUDA_CHECK(gh::workspace_bytes(config, native_bytes));
    config.local_counter = gh::LocalCounter::u32;
    CUDA_CHECK(gh::workspace_bytes(config, narrow_bytes));
    if (native_bytes != 2 * narrow_bytes)
      throw std::runtime_error("local-u32 scratch did not halve partial counter storage");
    config.bins = static_cast<unsigned>(gh::tuning_catalog[tuning].shared_limit /
                                       (sizeof(unsigned) * gh::tuning_catalog[tuning].replicas));
    if (!gh::supported(config)) throw std::runtime_error("local-u32 shared-capacity boundary rejected");
    config.local_counter = gh::LocalCounter::native;
    if (gh::supported(config)) throw std::runtime_error("shared-capacity check used incorrect local width");
  }
}

void local_counter_cases(bool sanitizer) {
  const std::vector<std::size_t> sizes = sanitizer ? std::vector<std::size_t>{33, 4099}
                                                   : std::vector<std::size_t>{0, 33, 4099, 65539};
  const char* orders[] = {"shuffled", "sorted", "roundrobin"};
  for (int tuning = 0; tuning < static_cast<int>(gh::tuning_count); ++tuning)
    for (auto input : {gh::InputType::u8, gh::InputType::u32}) for (auto n : sizes)
      for (auto algorithm : shared_algorithms) for (auto local : {gh::LocalCounter::native, gh::LocalCounter::u32}) {
        gh::Config config;
        config.algorithm = algorithm; config.input_type = input; config.counter_type = gh::CounterType::u64;
        config.local_counter = local; config.size = n; config.bins = input == gh::InputType::u8 ? 33 : 257;
        config.tuning = tuning; config.blocks = tuning % 2 ? 7 : 1;
        if (!gh::supported(config)) throw std::runtime_error("expected local-counter test configuration unsupported");
        run_case(config, tuning % 2 ? "hot99" : "uniform", orders[tuning % 3]);
      }
}

void global_narrow_cases(bool sanitizer) {
  for (auto algorithm : global_algorithms) for (int tuning = 0; tuning < 6; ++tuning)
    for (auto input : {gh::InputType::u8, gh::InputType::u32}) {
      const auto policy = gh::tuning_catalog[tuning];
      const std::size_t tile = static_cast<std::size_t>(policy.threads) * policy.items;
      const auto sizes = sanitizer ? std::vector<std::size_t>{33, tile + 33}
          : std::vector<std::size_t>{0, 1, 31, 32, 33, tile - 1, tile, tile + 1, 3 * tile + 17};
      for (std::size_t index = 0; index < sizes.size(); ++index) {
        gh::Config config;
        config.algorithm = algorithm; config.input_type = input; config.counter_type = gh::CounterType::u64;
        config.local_counter = gh::LocalCounter::u32; config.tuning = tuning;
        config.blocks = tuning % 2 ? 7 : 1; config.size = sizes[index];
        config.bins = input == gh::InputType::u8 ? 33 : 257;
        if (!gh::supported(config)) throw std::runtime_error("expected narrowed global configuration unsupported");
        // Scalar loads still promise natural alignment for both public input types.
        const auto offset = index % 2 ? gh::input_bytes(input) : std::size_t{0};
        run_case(config, index % 3 == 0 ? "single" : index % 3 == 1 ? "uniform" : "hot99",
                 index % 3 == 2 ? "sorted" : "shuffled", offset);
      }
    }
  // Sparse occupancy of a large output catches a widening kernel that neglects
  // absent bins or assumes the bin extent is divisible by its block size.
  for (auto algorithm : global_algorithms) for (auto bins : {24577u, 65536u, 1048577u}) {
    if (sanitizer && bins != 24577) continue;
    gh::Config config;
    config.algorithm = algorithm; config.counter_type = gh::CounterType::u64;
    config.local_counter = gh::LocalCounter::u32; config.tuning = 4;
    config.size = 4099; config.bins = bins; config.blocks = 7;
    if (!gh::supported(config)) throw std::runtime_error("large-bin narrowed global configuration unsupported");
    run_case(config, "uniform", "shuffled");
    run_case(config, "single", "shuffled");
  }
}

void global_narrow_graph_cases() {
  for (auto algorithm : global_algorithms) for (auto input_type : {gh::InputType::u8, gh::InputType::u32})
    for (auto clear : {gh::OutputClear::runtime, gh::OutputClear::kernel}) {
      gh::Config config;
      config.algorithm = algorithm; config.input_type = input_type; config.counter_type = gh::CounterType::u64;
      config.local_counter = gh::LocalCounter::u32; config.output_clear = clear;
      config.launch = gh::LaunchMode::graph; config.size = 4099;
      config.bins = input_type == gh::InputType::u8 ? 33 : 24577; config.blocks = 7;
      config.tuning = algorithm == gh::Algorithm::global_atomic ? 0 : 4;
      if (!gh::supported(config)) throw std::runtime_error("expected narrowed global graph configuration unsupported");
      CUDA_CHECK(gh::prepare(config));
      std::size_t bytes = 0;
      CUDA_CHECK(gh::workspace_bytes(config, bytes));
      Dataset first(config.size, config.bins, input_type, "uniform", "shuffled", 987123);
      Dataset second(config.size, config.bins, input_type, "single", "shuffled", 987124);
      const auto element_bytes = gh::input_bytes(input_type);
      DeviceBuffer input(config.size * element_bytes + element_bytes);
      auto* input_data = static_cast<unsigned char*>(input.data) + element_bytes;
      GuardedBuffer output(config.bins * sizeof(std::uint64_t)), workspace(bytes);
      Stream stream;
      CUDA_CHECK(cudaMemcpyAsync(input_data, first.data(input_type), config.size * element_bytes,
                                cudaMemcpyHostToDevice, stream.value));
      output.poison(stream.value); workspace.poison(stream.value);
      CUDA_CHECK(cudaStreamSynchronize(stream.value));
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t executable = nullptr;
      CUDA_CHECK(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
      CUDA_CHECK(gh::histogram(config, input_data, output.data(), workspace.data(), bytes, stream.value));
      CUDA_CHECK(cudaStreamEndCapture(stream.value, &graph));
      CUDA_CHECK(cudaGraphInstantiateWithFlags(&executable, graph, 0));
      for (int replay = 0; replay < 3; ++replay) {
        // Change the input once so old nonzero bins must be overwritten with zero.
        if (replay == 1)
          CUDA_CHECK(cudaMemcpyAsync(input_data, second.data(input_type), config.size * element_bytes,
                                    cudaMemcpyHostToDevice, stream.value));
        CUDA_CHECK(cudaGraphLaunch(executable, stream.value));
        CUDA_CHECK(cudaStreamSynchronize(stream.value));
        verify_output(config, output.data(), replay == 0 ? first.expected : second.expected);
        output.verify(); workspace.verify();
        ++executions;
      }
      CUDA_CHECK(cudaGraphExecDestroy(executable));
      CUDA_CHECK(cudaGraphDestroy(graph));
    }
}

void loaded_policy_cases(bool sanitizer) {
  constexpr gh::Algorithm loaded_algorithms[] = {
      gh::Algorithm::shared_atomic, gh::Algorithm::shared_rle, gh::Algorithm::shared_warp,
      gh::Algorithm::shared_partial, gh::Algorithm::bitplane};
  for (int tuning = 0; tuning < static_cast<int>(gh::tuning_count); ++tuning) {
    const auto policy = gh::tuning_catalog[tuning];
    if (policy.load == gh::LoadPolicy::scalar) continue;
    const std::size_t tile = static_cast<std::size_t>(policy.threads) * policy.items;
    // The compact sanitizer case still executes both the complete-tile path and
    // a final partial warp in the same block. Normal cases also cross grid rounds.
    const auto sizes = sanitizer ? std::vector<std::size_t>{tile + 33}
                                 : std::vector<std::size_t>{tile - 1, tile, tile + 1, 5 * tile + 33};
    for (auto input : {gh::InputType::u8, gh::InputType::u32})
      for (auto counter : {gh::CounterType::u32, gh::CounterType::u64})
        for (auto local : {gh::LocalCounter::native, gh::LocalCounter::u32}) {
          if (local == gh::LocalCounter::u32 && counter != gh::CounterType::u64) continue;
          for (auto algorithm : loaded_algorithms) {
            if (algorithm == gh::Algorithm::bitplane && local != gh::LocalCounter::native) continue;
            if (algorithm == gh::Algorithm::bitplane && policy.shared_limit > 48 * 1024) continue;
            gh::Config config;
            config.algorithm = algorithm; config.input_type = input; config.counter_type = counter;
            config.local_counter = local; config.tuning = tuning; config.blocks = sanitizer ? 1 : 2;
            config.bins = algorithm == gh::Algorithm::bitplane ? 129 : input == gh::InputType::u8 ? 129 : 257;
            if (algorithm == gh::Algorithm::bitplane && policy.threads == 1024 &&
                counter == gh::CounterType::u64) config.bins = 128;
            for (std::size_t index = 0; index < sizes.size(); ++index) {
              config.size = sizes[index];
              if (!gh::supported(config)) throw std::runtime_error("expected loaded-policy configuration unsupported");
              const char* distribution = !sanitizer && index == 2 ? "hot99" : "uniform";
              const char* order = !sanitizer && index == 1 ? "roundrobin"
                                      : !sanitizer && index == 2 ? "sorted" : "shuffled";
              // cudaMalloc alignment selects the packed-load path. Offsetting
              // by one element retains natural alignment but forces its fallback.
              for (std::size_t offset : {std::size_t{0}, gh::input_bytes(input)}) {
                // All families get sanitizer coverage on aligned data. RLE and
                // bitplane also exercise every policy/type's unaligned path.
                if (sanitizer && offset != 0 && algorithm != gh::Algorithm::shared_rle &&
                    algorithm != gh::Algorithm::bitplane) continue;
                run_case(config, distribution, order, offset);
              }
            }
          }
        }
  }
}

void optin_shared_cases(bool sanitizer) {
  int device = 0, shared_limit = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  constexpr unsigned bins = 16384;
  if (shared_limit < 96 * 1024) {
    std::cout << "SKIP: device does not support the 96 KiB opt-in policy capacity.\n";
    return;
  }
  for (int tuning = 0; tuning < static_cast<int>(gh::tuning_count); ++tuning) {
    const auto policy = gh::tuning_catalog[tuning];
    if (policy.shared_limit <= 48 * 1024 || policy.replicas != 1) continue;
    const std::size_t tile = static_cast<std::size_t>(policy.threads) * policy.items;
    for (auto algorithm : shared_algorithms) for (auto counter : {gh::CounterType::u32, gh::CounterType::u64}) {
      gh::Config config;
      config.algorithm = algorithm; config.input_type = gh::InputType::u32;
      config.counter_type = counter;
      config.local_counter = counter == gh::CounterType::u64 ? gh::LocalCounter::u32 : gh::LocalCounter::native;
      config.tuning = tuning; config.bins = bins; config.blocks = 2;
      config.size = (sanitizer ? 1 : 5) * tile + 33;
      if (!gh::supported(config)) throw std::runtime_error("expected opt-in shared-memory configuration unsupported");
      if (counter == gh::CounterType::u64) {
        auto native = config; native.local_counter = gh::LocalCounter::native;
        if (gh::supported(native)) throw std::runtime_error("128 KiB native histogram accepted by 96 KiB policy");
      }
      run_case(config, "uniform", "shuffled", 0);
    }
  }
}

bool large_count_case() {
  constexpr std::size_t size = std::size_t{std::numeric_limits<unsigned>::max()} + 33;
  constexpr std::size_t reserve = 256 * 1024 * 1024;
  std::size_t free_bytes = 0, total_bytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  if (free_bytes < size + reserve) {
    std::cout << "SKIP: --large-count needs about 4GiB plus 256MiB free device memory; available=" << free_bytes << ".\n";
    return false;
  }
  void* allocation = nullptr;
  const auto status = cudaMalloc(&allocation, size);
  if (status == cudaErrorMemoryAllocation) {
    (void)cudaGetLastError();
    std::cout << "SKIP: --large-count device allocation failed due to insufficient memory.\n";
    return false;
  }
  CUDA_CHECK(status);
  std::unique_ptr<void, decltype(&cudaFree)> input(allocation, cudaFree);
  Stream stream;
  CUDA_CHECK(cudaMemsetAsync(input.get(), 0, size, stream.value));
  CUDA_CHECK(cudaStreamSynchronize(stream.value));
  gh::Config config;
  config.input_type = gh::InputType::u8; config.counter_type = gh::CounterType::u64;
  config.size = size; config.bins = 256; config.blocks = 192;
  std::vector<std::uint64_t> expected(config.bins, 0);
  expected[0] = size;
  GuardedBuffer output(config.bins * sizeof(std::uint64_t));
  const auto run_large = [&](gh::Algorithm algorithm, gh::LocalCounter local, int tuning) {
      config.algorithm = algorithm; config.local_counter = local; config.tuning = tuning;
      if (!gh::supported(config)) throw std::runtime_error("large-count configuration unexpectedly unsupported");
      CUDA_CHECK(gh::prepare(config));
      std::size_t bytes = 0;
      CUDA_CHECK(gh::workspace_bytes(config, bytes));
      GuardedBuffer workspace(bytes);
      output.poison(stream.value); workspace.poison(stream.value);
      for (int repeat = 0; repeat < 2; ++repeat) {
        CUDA_CHECK(gh::histogram(config, input.get(), output.data(), workspace.data(), bytes, stream.value));
        CUDA_CHECK(cudaStreamSynchronize(stream.value));
        verify_output(config, output.data(), expected);
        output.verify(); workspace.verify();
        ++executions;
      }
      std::cout << "Verified large count " << size << " with " << gh::name(algorithm)
                << " local=" << gh::name(local) << " tuning=" << tuning << ".\n";
  };
  for (auto algorithm : {gh::Algorithm::shared_rle, gh::Algorithm::shared_partial})
    for (auto local : {gh::LocalCounter::native, gh::LocalCounter::u32}) {
      if (algorithm == gh::Algorithm::shared_partial && local == gh::LocalCounter::native) continue;
      run_large(algorithm, local, 2);
    }
  bool vector_case = false;
  for (int tuning = 0; tuning < static_cast<int>(gh::tuning_count); ++tuning)
    if (gh::tuning_catalog[tuning].load == gh::LoadPolicy::vector4) {
      run_large(gh::Algorithm::shared_rle, gh::LocalCounter::u32, tuning);
      vector_case = true;
      break;
    }
  if (!vector_case) throw std::runtime_error("large-count vector-policy test was not exercised");
  return true;
}
}
int main(int argc, char** argv) try {
  const bool sanitizer = argc == 2 && std::string(argv[1]) == "--sanitizer";
  const bool large_count = argc == 2 && std::string(argv[1]) == "--large-count";
  if (argc > 1 && !sanitizer && !large_count)
    throw std::runtime_error("usage: histogram_correctness [--sanitizer|--large-count]");
  invalid_configs();
  local_counter_boundaries();
  if (large_count) {
    if (large_count_case()) std::cout << "PASS: " << executions << " histogram executions with exact counts above UINT32_MAX.\n";
    return 0;
  }
  automatic_cases();
  gh::Config config;
  const std::vector<std::size_t> sizes = sanitizer ? std::vector<std::size_t>{0, 33, 4099}
      : std::vector<std::size_t>{0, 1, 2, 31, 32, 33, 127, 255, 256, 257, 1023, 4099};
  const std::vector<unsigned> bins = sanitizer ? std::vector<unsigned>{1, 3, 5, 9, 33, 256, 257, 4096}
      : std::vector<unsigned>{1, 2, 3, 5, 9, 31, 32, 33, 63, 127, 255, 256, 257, 511, 4096};
  const char* distributions[] = {"uniform", "single", "two", "hot90", "hot99"};
  const char* orders[] = {"shuffled", "sorted", "roundrobin"};
  std::size_t case_index = 0;
  for (auto n : sizes) for (auto b : bins) {
    for (auto input_type : {gh::InputType::u8, gh::InputType::u32}) {
      if (input_type == gh::InputType::u8 && b > 256) continue;
      for (auto counter_type : {gh::CounterType::u32, gh::CounterType::u64}) {
        config.size = n; config.bins = b; config.input_type = input_type; config.counter_type = counter_type;
        config.blocks = (case_index % 3 == 0) ? 1 : (case_index % 3 == 1 ? 7 : 192);
        config.tuning = static_cast<int>(case_index % 6);
        for (auto algorithm : algorithms) {
          config.algorithm = algorithm;
          run_case(config, distributions[case_index % 5], orders[case_index % 3]);
        }
        ++case_index;
      }
    }
  }
  if (!sanitizer) {
    // Keep the original scalar-policy controls; loaded policies are covered separately below.
    for (auto b : {31u, 256u, 257u, 4096u}) for (int tuning = 0; tuning < 6; ++tuning)
      for (auto input : {gh::InputType::u8, gh::InputType::u32})
      for (auto count : {gh::CounterType::u32, gh::CounterType::u64}) {
        if (input == gh::InputType::u8 && b > 256) continue;
        config = {}; config.size = 131071; config.bins = b; config.blocks = 1;
        config.input_type = input; config.counter_type = count; config.tuning = tuning;
        for (auto algorithm : algorithms) {
          config.algorithm = algorithm;
          run_case(config, "single", "shuffled");
          run_case(config, "uniform", "sorted");
        }
      }
  }
  local_counter_cases(sanitizer);
  global_narrow_cases(sanitizer);
  global_narrow_graph_cases();
  loaded_policy_cases(sanitizer);
  optin_shared_cases(sanitizer);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "PASS: " << executions << " histogram executions; independent CPU counts, sum, output/scratch canaries, tails, repeat calls, and nondefault streams.\n";
  return 0;
} catch (const std::exception& e) {
  std::cerr << "FAIL: " << e.what() << '\n';
  return 1;
}
