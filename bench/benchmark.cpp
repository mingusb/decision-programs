#include "common.hpp"
#include "profiling.hpp"
#include "cache_flush.hpp"
#include "references.hpp"
#include <cub/version.cuh>
#include <nvtx3/nvToolsExt.h>
#include <chrono>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

struct Options {
  gh::Config config;
  std::string distribution = "uniform", order = "shuffled", algorithm = "auto";
  std::string variants;
  std::string cache = "warm", launch = "stream", clear = "auto";
  std::uint64_t seed = 12345;
  int samples = 9, batch = 10, warmup_ms = 0;
  bool sweep = false, explicit_blocks = false, explicit_policy = false;
};
gh::LocalCounter parse_local_counter(const std::string& value) {
  if (value == "native") return gh::LocalCounter::native;
  if (value == "u32") return gh::LocalCounter::u32;
  throw std::runtime_error("local counter must be native or u32");
}
gh::OutputClear parse_output_clear(const std::string& value) {
  if (value == "runtime") return gh::OutputClear::runtime;
  if (value == "kernel") return gh::OutputClear::kernel;
  throw std::runtime_error("clear policy must be runtime or kernel");
}
struct BenchmarkAlgorithm {
  gh::Algorithm algorithm;
  ghbench::Reference reference = ghbench::Reference::none;
};
// Reference identity stays outside gh::Algorithm. The custom field is only a
// placeholder for references; no production selector, prepare, or launch sees it.
inline constexpr BenchmarkAlgorithm benchmark_algorithms[] = {
  {gh::Algorithm::global_atomic, ghbench::Reference::cub},
  {gh::Algorithm::global_atomic}, {gh::Algorithm::warp_aggregated},
  {gh::Algorithm::shared_atomic}, {gh::Algorithm::shared_rle}, {gh::Algorithm::shared_warp},
  {gh::Algorithm::shared_partial}, {gh::Algorithm::bitplane}, {gh::Algorithm::shared_overflow},
  {gh::Algorithm::global_atomic, ghbench::Reference::nvidia_sample256}};
bool fixed_baseline(ghbench::Reference reference) {
  return reference != ghbench::Reference::none;
}
const char* algorithm_name(BenchmarkAlgorithm choice) {
  return fixed_baseline(choice.reference) ? ghbench::name(choice.reference) : gh::name(choice.algorithm);
}
bool supported(BenchmarkAlgorithm choice, const gh::Config& config) {
  return fixed_baseline(choice.reference) ? ghbench::supported(choice.reference, config) : gh::supported(config);
}
cudaError_t workspace_bytes(BenchmarkAlgorithm choice, const gh::Config& config, std::size_t& bytes) {
  return fixed_baseline(choice.reference) ? ghbench::workspace_bytes(choice.reference, config, bytes)
                                           : gh::workspace_bytes(config, bytes);
}
bool shared_algorithm(gh::Algorithm algorithm) {
  return algorithm == gh::Algorithm::shared_atomic || algorithm == gh::Algorithm::shared_rle ||
         algorithm == gh::Algorithm::shared_warp || algorithm == gh::Algorithm::shared_partial ||
         algorithm == gh::Algorithm::shared_overflow;
}
gh::LoadPolicy effective_load(gh::Algorithm algorithm, const gh::Tuning& tuning) {
  if (algorithm == gh::Algorithm::bitplane && tuning.load != gh::LoadPolicy::scalar)
    return gh::LoadPolicy::full_tile;
  return tuning.load;
}
Options parse(int argc, char** argv) {
  Options o; o.config.size = 1 << 20; o.config.bins = 4096;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--sweep") { o.sweep = true; continue; }
    if (key == "--help") {
      std::cout << "histogram_bench [--n N] [--bins B] [--input u8|u32] [--counter u32|u64]\n"
                   "  [--distribution uniform|single|two|hot90|hot99|hot99@BIN] [--order shuffled|sorted|roundrobin]\n"
                   "  [--seed N] [--samples N] [--batch N] [--algorithm auto|NAME|all]\n"
                   "  [--tuning INDEX] [--blocks N] [--local-counter native|u32] [--sweep]\n"
                   "  [--window-bins N] (explicit global_window only; default 524288)\n"
                   "  [--variants algorithm:tuning:blocks[:native|u32[:runtime|kernel]],...]\n"
                   "  [--cache warm|cold] [--launch stream|graph]\n"
                   "  [--clear auto|runtime|kernel] (auto: runtime for stream, kernel for graph)\n"
                   "  [--warmup-ms N] (optional GPU work before measurement; default 0)\n"
                   "CSV on stdout. Defaults: measured automatic policy, warm cache, stream launches.\n"
                   "Auto uses saved A5000 workload settings with an in-house fallback.\n"
                   "CUB and the NVIDIA sample are isolated benchmark references only.\n"
                   "global_window is an explicit u32/u64/local-u32 experiment, excluded from all/sweep.\n"
                   "--sweep searches all algorithms; manual policy flags require NAME or all.\n"
                   "Tuning indices: 0.." << gh::tuning_count - 1 << ". Sweep includes supported u32 locals for shared/global/warp u64 output.\n"
                   "Whole device operation timed; allocations, graph setup, and cache eviction excluded.\n"
                   "Warm: time a batch and divide by batch size. Cold: evict L2 before every operation,\n"
                   "time each with its own event pair, then average the batch. Graph mode captures\n"
                   "timing events inside the batch graph; cold graphs evict before each operation.\n";
      std::cout << "clear_policy records the requested custom output-clear strategy; CUB, the NVIDIA\n"
                   "sample, shared_partial, and empty inputs retain their own initialization paths.\n";
      std::exit(0);
    }
    if (i + 1 == argc) throw std::runtime_error("missing value for " + key);
    const std::string value = argv[++i];
    if (key == "--n") o.config.size = std::stoull(value);
    else if (key == "--bins") {
      auto b = std::stoull(value);
      if (b >= 2147483647ULL || b == 0) throw std::runtime_error("bins must be in [1, INT_MAX-1]");
      o.config.bins = static_cast<unsigned>(b);
    } else if (key == "--input") {
      if (value != "u8" && value != "u32") throw std::runtime_error("invalid input type");
      o.config.input_type = value == "u8" ? gh::InputType::u8 : gh::InputType::u32;
    } else if (key == "--counter") {
      if (value != "u32" && value != "u64") throw std::runtime_error("invalid counter type");
      o.config.counter_type = value == "u32" ? gh::CounterType::u32 : gh::CounterType::u64;
    } else if (key == "--distribution") o.distribution = value;
    else if (key == "--local-counter") { o.config.local_counter = parse_local_counter(value); o.explicit_policy = true; }
    else if (key == "--window-bins") {
      std::size_t consumed = 0;
      const auto bins = std::stoull(value, &consumed);
      if (consumed != value.size() || bins == 0 || bins >= 2147483647ULL)
        throw std::runtime_error("window-bins must be in [1, INT_MAX-1]");
      o.config.window_bins = static_cast<unsigned>(bins);
      o.explicit_policy = true;
    }
    else if (key == "--cache") o.cache = value;
    else if (key == "--launch") o.launch = value;
    else if (key == "--clear") o.clear = value;
    else if (key == "--order") o.order = value;
    else if (key == "--algorithm") o.algorithm = value;
    else if (key == "--variants") o.variants = value;
    else if (key == "--seed") o.seed = std::stoull(value);
    else if (key == "--samples") o.samples = std::stoi(value);
    else if (key == "--batch") o.batch = std::stoi(value);
    else if (key == "--warmup-ms") {
      std::size_t consumed = 0;
      o.warmup_ms = std::stoi(value, &consumed);
      if (consumed != value.size()) throw std::runtime_error("invalid warmup-ms");
    }
    else if (key == "--blocks") { o.config.blocks = std::stoi(value); o.explicit_blocks = true; o.explicit_policy = true; }
    else if (key == "--tuning") { o.config.tuning = std::stoi(value); o.explicit_policy = true; }
    else throw std::runtime_error("unknown option " + key);
  }
  if (o.samples < 3 || o.samples > 1001 || o.batch < 1 || o.batch > 10000)
    throw std::runtime_error("samples must be 3..1001; batch 1..10000");
  if (o.warmup_ms < 0 || o.warmup_ms > 60000)
    throw std::runtime_error("warmup-ms must be 0..60000");
  if (o.cache != "warm" && o.cache != "cold") throw std::runtime_error("cache must be warm or cold");
  if (o.launch != "stream" && o.launch != "graph") throw std::runtime_error("launch must be stream or graph");
  o.config.launch = o.launch == "graph" ? gh::LaunchMode::graph : gh::LaunchMode::stream;
  o.config.cache = o.cache == "cold" ? gh::CacheMode::cold : gh::CacheMode::warm;
  // Resolve once before constructing candidates; histogram() needs no capture-state query.
  o.config.output_clear = o.clear == "auto"
      ? (o.launch == "graph" ? gh::OutputClear::kernel : gh::OutputClear::runtime)
      : parse_output_clear(o.clear);
  if (o.config.tuning < 0 || o.config.tuning >= static_cast<int>(gh::tuning_count) || o.config.blocks <= 0)
    throw std::runtime_error("invalid tuning/grid");
  if (o.algorithm == "auto" && o.sweep) o.algorithm = "all";
  if (o.algorithm == "auto" && o.explicit_policy && o.variants.empty())
    throw std::runtime_error("manual tuning/grid/local-counter flags require --algorithm NAME or all");
  if (o.algorithm == "global_window" && o.sweep)
    throw std::runtime_error("global_window requires explicit tuning; it is excluded from --sweep");
  if (o.algorithm != "auto" && o.algorithm != "all" && o.algorithm != "global_window" && !std::any_of(std::begin(benchmark_algorithms), std::end(benchmark_algorithms), [&](auto a) { return o.algorithm == algorithm_name(a); }))
    throw std::runtime_error("unknown algorithm " + o.algorithm);
  return o;
}
struct TimingPair { Event start, end; };
struct CapturedGraph {
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  std::deque<TimingPair> timings;
  CapturedGraph() = default;
  CapturedGraph(const CapturedGraph&) = delete;
  CapturedGraph& operator=(const CapturedGraph&) = delete;
  ~CapturedGraph() {
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
  }
  template<class Eviction, class Operation>
  void capture(cudaStream_t stream, int repetitions, bool cold,
               Eviction evict, Operation operation) {
    timings.resize(cold ? repetitions : 1);
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    try {
      if (cold) {
        for (auto& pair : timings) {
          evict();
          CUDA_CHECK(cudaEventRecordWithFlags(pair.start.value, stream, cudaEventRecordExternal));
          operation();
          CUDA_CHECK(cudaEventRecordWithFlags(pair.end.value, stream, cudaEventRecordExternal));
        }
      } else {
        CUDA_CHECK(cudaEventRecordWithFlags(timings.front().start.value, stream, cudaEventRecordExternal));
        for (int i = 0; i < repetitions; ++i) operation();
        CUDA_CHECK(cudaEventRecordWithFlags(timings.front().end.value, stream, cudaEventRecordExternal));
      }
    } catch (...) {
      // End even an invalidated capture, releasing the stream and any graph.
      cudaGraph_t discarded{};
      cudaStreamEndCapture(stream, &discarded);
      if (discarded) cudaGraphDestroy(discarded);
      throw;
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&executable, graph, 0));
  }
  void launch(cudaStream_t stream) const { CUDA_CHECK(cudaGraphLaunch(executable, stream)); }
  double elapsed_milliseconds(cudaStream_t stream) const {
    // A completed launch guarantees every captured event was recorded, including
    // the most recent launch when an untimed warmup used these same events.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    double milliseconds = 0;
    for (const auto& pair : timings) {
      float elapsed = 0;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed, pair.start.value, pair.end.value));
      milliseconds += elapsed;
    }
    return milliseconds;
  }
};
struct Candidate {
  gh::Config config;
  std::size_t scratch;
  std::vector<double> times;
  std::unique_ptr<CapturedGraph> graph;
  ghbench::Reference reference = ghbench::Reference::none;
};
const char* algorithm_name(const Candidate& candidate) {
  return algorithm_name(BenchmarkAlgorithm{candidate.config.algorithm, candidate.reference});
}
struct NvtxRange {
  explicit NvtxRange(const std::string& label) { nvtxRangePushA(label.c_str()); }
  ~NvtxRange() { nvtxRangePop(); }
};
double percentile(std::vector<double> values, double q) {
  std::sort(values.begin(), values.end());
  const auto i = static_cast<std::size_t>(std::ceil(q * values.size())) - 1;
  return values[std::min(i, values.size() - 1)];
}
std::string csv_quote(std::string value) {
  std::string result = "\"";
  for (char c : value) { if (c == '"') result += '"'; result += c; }
  return result + '"';
}
int main(int argc, char** argv) try {
  auto options = parse(argc, argv);
  cudaDeviceProp gpu{};
  CUDA_CHECK(cudaGetDeviceProperties(&gpu, 0));
  int driver = 0, runtime = 0;
  CUDA_CHECK(cudaDriverGetVersion(&driver)); CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  if (!options.explicit_blocks) options.config.blocks = gpu.multiProcessorCount * 4;
  std::vector<Candidate> candidates;
  if (!options.variants.empty()) {
    if (options.sweep) throw std::runtime_error("--variants and --sweep are mutually exclusive");
    if (options.variants.back() == ',') throw std::runtime_error("invalid --variants entry");
    std::istringstream entries(options.variants);
    std::string entry;
    while (std::getline(entries, entry, ',')) {
      const auto separators = std::count(entry.begin(), entry.end(), ':');
      if (entry.empty() || entry.front() == ':' || entry.back() == ':' || entry.find("::") != std::string::npos ||
          (separators != 2 && separators != 3 && separators != 4)) throw std::runtime_error("invalid --variants entry");
      std::replace(entry.begin(), entry.end(), ':', ' ');
      std::istringstream fields(entry);
      std::string name, local, clear, extra;
      auto config = options.config;
      config.local_counter = gh::LocalCounter::native;
      if (!(fields >> name >> config.tuning >> config.blocks))
        throw std::runtime_error("invalid --variants entry");
      if (separators >= 3) {
        if (!(fields >> local)) throw std::runtime_error("invalid --variants entry");
        config.local_counter = parse_local_counter(local);
      }
      if (separators == 4) {
        if (!(fields >> clear)) throw std::runtime_error("invalid --variants entry");
        config.output_clear = parse_output_clear(clear);
      }
      if (fields >> extra) throw std::runtime_error("invalid --variants entry");
      bool found = false;
      BenchmarkAlgorithm choice{config.algorithm};
      for (auto implementation : benchmark_algorithms) if (name == algorithm_name(implementation)) {
        choice = implementation;
        config.algorithm = implementation.algorithm;
        found = true;
      }
      if (name == "global_window") {
        config.algorithm = gh::Algorithm::global_window;
        choice = {config.algorithm};
        found = true;
      }
      if (!found || !supported(choice, config)) throw std::runtime_error("unsupported --variants entry");
      std::size_t scratch = 0;
      CUDA_CHECK(workspace_bytes(choice, config, scratch));
      candidates.push_back({config, scratch, {}, {}, choice.reference});
    }
  } else if (options.algorithm == "global_window") {
    auto config = options.config;
    config.algorithm = gh::Algorithm::global_window;
    if (!gh::supported(config)) throw std::runtime_error("unsupported global_window configuration");
    std::size_t scratch = 0;
    CUDA_CHECK(gh::workspace_bytes(config, scratch));
    candidates.push_back({config, scratch, {}, {}});
  } else if (options.algorithm == "auto") {
    auto config = gh::default_config(options.config, gpu);
    if (options.clear != "auto") config.output_clear = options.config.output_clear;
    if (!gh::supported(config)) throw std::runtime_error("unsupported automatic configuration");
    std::size_t scratch = 0;
    CUDA_CHECK(gh::workspace_bytes(config, scratch));
    candidates.push_back({config, scratch, {}, {}});
  } else for (auto implementation : benchmark_algorithms) {
    const auto algorithm = implementation.algorithm;
    if (options.algorithm != "all" && options.algorithm != algorithm_name(implementation)) continue;
    const bool tune = options.sweep && !fixed_baseline(implementation.reference);
    const auto grids = tune && !options.explicit_blocks
        ? std::vector<int>{gpu.multiProcessorCount, gpu.multiProcessorCount * 2, gpu.multiProcessorCount * 4, gpu.multiProcessorCount * 8}
        : std::vector<int>{options.config.blocks};
    std::vector<gh::LocalCounter> local_counters{options.config.local_counter};
    if (options.sweep && fixed_baseline(implementation.reference))
      local_counters = {gh::LocalCounter::native};
    else if (tune && options.config.counter_type == gh::CounterType::u64 &&
        (shared_algorithm(algorithm) || algorithm == gh::Algorithm::global_atomic ||
         algorithm == gh::Algorithm::warp_aggregated))
      local_counters = {gh::LocalCounter::native, gh::LocalCounter::u32};
    for (int tuning = tune ? 0 : options.config.tuning;
         tuning < (tune ? static_cast<int>(gh::tuning_count) : options.config.tuning + 1); ++tuning) {
      // Replication does not affect register/global methods; suppress equivalent policies.
      if (tune && !shared_algorithm(algorithm)) {
        bool duplicate = false;
        for (int earlier = 0; earlier < tuning; ++earlier)
          if (gh::tuning_catalog[earlier].threads == gh::tuning_catalog[tuning].threads &&
              gh::tuning_catalog[earlier].items == gh::tuning_catalog[tuning].items &&
              effective_load(algorithm, gh::tuning_catalog[earlier]) ==
                  effective_load(algorithm, gh::tuning_catalog[tuning])) duplicate = true;
        if (duplicate) continue;
      }
      for (int blocks : grids) {
        for (auto local_counter : local_counters) {
          auto config = options.config; config.algorithm = algorithm; config.tuning = tuning; config.blocks = blocks;
          config.local_counter = local_counter;
          if (!supported(implementation, config)) continue;
          std::size_t scratch = 0;
          CUDA_CHECK(workspace_bytes(implementation, config, scratch));
          candidates.push_back({config, scratch, {}, {}, implementation.reference});
        }
      }
    }
  }
  if (candidates.empty()) throw std::runtime_error("no supported configurations for this workload");
  Dataset dataset(options.config.size, options.config.bins, options.config.input_type,
                  options.distribution, options.order, options.seed);
  DeviceBuffer input(options.config.size * gh::input_bytes(options.config.input_type));
  DeviceBuffer output(options.config.bins * gh::counter_bytes(options.config.counter_type));
  const auto max_scratch = std::max_element(candidates.begin(), candidates.end(), [](auto& a, auto& b) { return a.scratch < b.scratch; })->scratch;
  DeviceBuffer workspace(max_scratch);
  Stream stream;
  const std::size_t eviction_bytes = options.cache == "cold"
      ? std::max<std::size_t>(64ULL << 20, std::size_t(std::max(gpu.l2CacheSize, 0)) * 8) : 0;
  std::unique_ptr<DeviceBuffer> eviction;
  if (eviction_bytes) {
    eviction = std::make_unique<DeviceBuffer>(eviction_bytes);
    CUDA_CHECK(initialize_eviction_buffer(eviction->data, eviction->bytes,
                                         gpu.multiProcessorCount, stream.value));
  }
  if (options.config.size) CUDA_CHECK(cudaMemcpyAsync(input.data, dataset.data(options.config.input_type), input.bytes, cudaMemcpyHostToDevice, stream.value));
  CUDA_CHECK(cudaStreamSynchronize(stream.value));
  auto operation = [&](const Candidate& candidate) {
    if (fixed_baseline(candidate.reference)) {
      CUDA_CHECK(ghbench::histogram(candidate.reference, candidate.config, input.data, output.data,
                                  workspace.data, candidate.scratch, stream.value));
    } else {
      CUDA_CHECK(gh::histogram(candidate.config, input.data, output.data,
                               workspace.data, candidate.scratch, stream.value));
    }
  };
  auto verify = [&](const Candidate& candidate) {
    try {
      verify_output(candidate.config, output.data, dataset.expected);
    } catch (const std::exception& error) {
      throw std::runtime_error(std::string(algorithm_name(candidate)) + ": " + error.what());
    }
  };
  auto evict = [&] {
    CUDA_CHECK(evict_l2(eviction->data, eviction->bytes, gpu.multiProcessorCount, stream.value));
  };
  // Query/allocation, CPU-reference checks, warmup, capture and graph instantiation
  // all precede measurement. Repeated execution must overwrite existing output.
  for (auto& candidate : candidates) {
    if (!fixed_baseline(candidate.reference)) CUDA_CHECK(gh::prepare(candidate.config));
    CUDA_CHECK(cudaMemsetAsync(output.data, 0xa5, output.bytes, stream.value));
    CUDA_CHECK(cudaMemsetAsync(workspace.data, 0xa5, workspace.bytes, stream.value));
    for (int validation = 0; validation < 2; ++validation) {
      operation(candidate);
      CUDA_CHECK(cudaStreamSynchronize(stream.value));
      verify(candidate);
    }
    if (options.launch == "graph") {
      candidate.graph = std::make_unique<CapturedGraph>();
      candidate.graph->capture(stream.value, options.batch, bool(eviction), evict,
                               [&] { operation(candidate); });
      // Capture records operations without executing them. Actually launch each
      // graph twice, checking both poisoned-output initialization and overwrite.
      CUDA_CHECK(cudaMemsetAsync(output.data, 0xa5, output.bytes, stream.value));
      CUDA_CHECK(cudaMemsetAsync(workspace.data, 0xa5, workspace.bytes, stream.value));
      for (int validation = 0; validation < 2; ++validation) {
        candidate.graph->launch(stream.value);
        CUDA_CHECK(cudaStreamSynchronize(stream.value));
        verify(candidate);
      }
    }
  }
  if (options.warmup_ms) {
    const auto start = std::chrono::steady_clock::now();
    const auto duration = std::chrono::milliseconds(options.warmup_ms);
    std::size_t next = 0;
    do {
      auto& candidate = candidates[next];
      if (candidate.graph) candidate.graph->launch(stream.value);
      else {
        if (eviction) evict();
        operation(candidate);
      }
      CUDA_CHECK(cudaStreamSynchronize(stream.value));
      next = (next + 1) % candidates.size();
    } while (std::chrono::steady_clock::now() - start < duration);
  }
  std::cerr << "Validated " << candidates.size() << " configurations on " << gpu.name
            << "; timing " << options.samples << " randomized rounds x " << options.batch
            << " complete operations; cache=" << options.cache << ", launch=" << options.launch
            << ", default_clear=" << gh::name(options.config.output_clear)
            << ", timing_protocol=3, warmup_ms=" << options.warmup_ms << ".\n";
  std::cerr << "clear_policy records the requested custom initialization strategy; CUB, NVIDIA sample, "
               "shared_partial, and empty inputs keep their own initialization.\n";
  if (eviction)
    std::cerr << "Cold method: " << eviction_bytes << "-byte .cg-load/.wb-store L2 eviction before EVERY operation; "
                 "eviction is outside independent event pairs; each sample is the mean of batch event durations.\n";
  else
    std::cerr << "Warm method: one untimed warmup then event timing around "
              << (options.launch == "graph" ? "one graph containing the full batch" : "the full stream batch")
              << "; each sample divides elapsed time by batch size.\n";
  if (options.launch == "graph")
    std::cerr << "Graph timing: event record nodes are inside the captured batch graph; "
                 "host submission is outside the measured event intervals.\n";
  std::vector<std::size_t> sequence(candidates.size());
  std::iota(sequence.begin(), sequence.end(), 0);
  std::mt19937_64 random(0xdecafbadULL ^ options.seed);
  Event start, end;
  // Every cold operation has its own pair: re-recording one pair before reading
  // elapsed times would retain only the last repetition's timestamps.
  std::deque<TimingPair> cold_events(eviction && options.launch == "stream" ? options.batch : 0);
  gh::profiling::CaptureRange capture("count");
  for (int round = 0; round < options.samples; ++round) {
    std::shuffle(sequence.begin(), sequence.end(), random);
    for (auto index : sequence) {
      auto& candidate = candidates[index];
      std::string label = std::string(algorithm_name(candidate)) + ":t" + std::to_string(candidate.config.tuning)
          + ":grid" + std::to_string(candidate.config.blocks) + ":local=" + gh::name(candidate.config.local_counter)
          + ":clear=" + gh::name(candidate.config.output_clear)
          + ":" + options.cache + ":" + options.launch;
      NvtxRange range(label);
      auto launch = [&] {
        if (candidate.graph) candidate.graph->launch(stream.value);
        else operation(candidate);
      };
      // For cold mode the following eviction removes this warmup's data state.
      // Graph warmup uses the same complete batch graph, including per-operation
      // eviction in cold mode. Its event timestamps are overwritten by measurement.
      launch();
      double milliseconds = 0;
      if (candidate.graph) {
        launch();
        milliseconds = candidate.graph->elapsed_milliseconds(stream.value);
      } else if (eviction) {
        for (auto& pair : cold_events) {
          CUDA_CHECK(evict_l2(eviction->data, eviction->bytes, gpu.multiProcessorCount, stream.value));
          CUDA_CHECK(cudaEventRecord(pair.start.value, stream.value));
          launch();
          CUDA_CHECK(cudaEventRecord(pair.end.value, stream.value));
        }
        CUDA_CHECK(cudaEventSynchronize(cold_events.back().end.value));
        for (const auto& pair : cold_events) {
          float elapsed = 0;
          CUDA_CHECK(cudaEventElapsedTime(&elapsed, pair.start.value, pair.end.value));
          milliseconds += elapsed;
        }
      } else {
        CUDA_CHECK(cudaEventRecord(start.value, stream.value));
        for (int repetition = 0; repetition < options.batch; ++repetition) launch();
        CUDA_CHECK(cudaEventRecord(end.value, stream.value));
        CUDA_CHECK(cudaEventSynchronize(end.value));
        float elapsed = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start.value, end.value));
        milliseconds = elapsed;
      }
      const double microseconds = milliseconds * 1000.0 / options.batch;
      if (!(microseconds > 0) || !std::isfinite(microseconds)) throw std::runtime_error("invalid measured duration");
      candidate.times.push_back(microseconds);
    }
  }
  capture.finish();
  const bool has_window = std::any_of(candidates.begin(), candidates.end(), [](const auto& candidate) {
    return candidate.config.algorithm == gh::Algorithm::global_window;
  });
  std::cout << "gpu,sm,driver_api,runtime,cub_version,n,bins,input,counter,distribution,order,seed,algorithm,tuning,threads,items,replicas,blocks,scratch_bytes,samples,batch,median_us,min_us,p95_us,max_us,input_gb_s,sample_us,cache,launch,eviction_bytes,local_counter,timing_protocol,warmup_ms,load_policy,shared_limit,clear_policy";
  if (has_window) std::cout << ",window_bins";
  std::cout << '\n';
  std::cout << std::fixed << std::setprecision(6);
  for (const auto& candidate : candidates) {
    const auto& c = candidate.config;
    const auto policy = fixed_baseline(candidate.reference) ? gh::Tuning{0, 0, 0} : gh::tuning_catalog[c.tuning];
    const auto median = percentile(candidate.times, .5);
    std::cout << csv_quote(gpu.name) << ',' << gpu.major * 10 + gpu.minor << ',' << driver << ',' << runtime << ',' << CUB_VERSION << ','
              << c.size << ',' << c.bins << ',' << gh::name(c.input_type) << ',' << gh::name(c.counter_type) << ','
              << options.distribution << ',' << options.order << ',' << options.seed << ',' << algorithm_name(candidate) << ','
              << c.tuning << ',' << policy.threads << ',' << policy.items << ',' << policy.replicas << ',' << c.blocks << ','
              << candidate.scratch << ',' << options.samples << ',' << options.batch << ',' << median << ','
              << *std::min_element(candidate.times.begin(), candidate.times.end()) << ',' << percentile(candidate.times, .95) << ','
              << *std::max_element(candidate.times.begin(), candidate.times.end()) << ','
              << double(c.size) * gh::input_bytes(c.input_type) / (median * 1000.0) << ',';
    for (std::size_t i = 0; i < candidate.times.size(); ++i) {
      if (i) std::cout << ';';
      std::cout << candidate.times[i];
    }
    std::cout << ',' << options.cache << ',' << options.launch << ',' << eviction_bytes << ','
              << gh::name(c.local_counter) << ",3," << options.warmup_ms << ','
              << (fixed_baseline(candidate.reference) ? "reference" : gh::name(effective_load(c.algorithm, policy))) << ','
              << (fixed_baseline(candidate.reference) ? 0 : policy.shared_limit) << ','
              << gh::name(c.output_clear);
    if (has_window)
      std::cout << ',' << (c.algorithm == gh::Algorithm::global_window ? c.window_bins : 0);
    std::cout << '\n';
  }
  return 0;
} catch (const std::exception& error) {
  std::cerr << "ERROR: " << error.what() << '\n';
  return 1;
}
