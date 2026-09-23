#include "common.hpp"
#include "cache_flush.hpp"
#include <nvtx3/nvToolsExt.h>
#include <iostream>

namespace {
struct ProbeGraph {
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  ProbeGraph() = default;
  ProbeGraph(const ProbeGraph&) = delete;
  ProbeGraph& operator=(const ProbeGraph&) = delete;
  ~ProbeGraph() {
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
  }
  template<class Operation>
  void capture(cudaStream_t stream, Operation operation) {
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    try {
      operation();
    } catch (...) {
      cudaGraph_t discarded{};
      cudaStreamEndCapture(stream, &discarded);
      if (discarded) cudaGraphDestroy(discarded);
      throw;
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&executable, graph, 0));
  }
  void launch(cudaStream_t stream) const {
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
  }
};
struct NvtxRange {
  explicit NvtxRange(const std::string& label) { nvtxRangePushA(label.c_str()); }
  ~NvtxRange() { nvtxRangePop(); }
};
}

int main(int argc, char** argv) try {
  std::string order = "evict-first";
  int runs = 3;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help") {
      std::cout << "histogram_cache_probe [--order evict-first|evict-between] [--runs N]\n"
                   "Fixed workload: shared:2:192:native:kernel, N262144, u32 input/output, B4096.\n"
                   "One graph contains E,H,H or H,E,H, where E is the same L2 eviction\n"
                   "and H is the same complete overwrite histogram in both cases.\n"
                   "Every graph launch is synchronized and checked against the CPU reference.\n"
                   "Profile the whole graph with cache-control all; compare read-hit/miss deltas\n"
                   "between the two orders. NVTX push/pop range: cache_probe:<order>.\n";
      return 0;
    }
    if (i + 1 == argc) throw std::runtime_error("missing value for " + key);
    const std::string value = argv[++i];
    if (key == "--order") order = value;
    else if (key == "--runs") {
      std::size_t consumed = 0;
      runs = std::stoi(value, &consumed);
      if (consumed != value.size()) throw std::runtime_error("invalid runs");
    } else throw std::runtime_error("unknown option " + key);
  }
  if (order != "evict-first" && order != "evict-between")
    throw std::runtime_error("order must be evict-first or evict-between");
  if (runs < 1 || runs > 1000) throw std::runtime_error("runs must be 1..1000");

  gh::Config config;
  config.size = 262144;
  config.bins = 4096;
  config.input_type = gh::InputType::u32;
  config.counter_type = gh::CounterType::u32;
  config.algorithm = gh::Algorithm::shared_atomic;
  config.tuning = 2;
  config.blocks = 192;
  config.local_counter = gh::LocalCounter::native;
  config.output_clear = gh::OutputClear::kernel;
  if (!gh::supported(config)) throw std::runtime_error("fixed probe configuration unsupported");
  CUDA_CHECK(gh::prepare(config));
  std::size_t scratch = 0;
  CUDA_CHECK(gh::workspace_bytes(config, scratch));
  cudaDeviceProp gpu{};
  CUDA_CHECK(cudaGetDeviceProperties(&gpu, 0));
  const std::size_t eviction_bytes = std::max<std::size_t>(
      64ULL << 20, std::size_t(std::max(gpu.l2CacheSize, 0)) * 8);
  Dataset dataset(config.size, config.bins, config.input_type, "uniform", "shuffled", 12345);
  DeviceBuffer input(config.size * gh::input_bytes(config.input_type));
  DeviceBuffer output(config.bins * gh::counter_bytes(config.counter_type));
  DeviceBuffer workspace(scratch);
  DeviceBuffer eviction(eviction_bytes);
  Stream stream;
  CUDA_CHECK(initialize_eviction_buffer(eviction.data, eviction.bytes,
                                       gpu.multiProcessorCount, stream.value));
  CUDA_CHECK(cudaMemcpyAsync(input.data, dataset.data(config.input_type), input.bytes,
                            cudaMemcpyHostToDevice, stream.value));
  CUDA_CHECK(cudaMemsetAsync(output.data, 0xa5, output.bytes, stream.value));
  CUDA_CHECK(cudaStreamSynchronize(stream.value));
  auto histogram = [&] {
    CUDA_CHECK(gh::histogram(config, input.data, output.data, workspace.data, scratch, stream.value));
  };
  auto evict = [&] {
    CUDA_CHECK(evict_l2(eviction.data, eviction.bytes, gpu.multiProcessorCount, stream.value));
  };
  ProbeGraph graph;
  graph.capture(stream.value, [&] {
    if (order == "evict-first") evict();
    histogram();
    if (order == "evict-between") evict();
    histogram();
  });

  std::cout << "order,run,n,bins,input,counter,algorithm,tuning,blocks,eviction_bytes,correct\n";
  for (int run = 0; run < runs; ++run) {
    {
      NvtxRange range("cache_probe:" + order);
      graph.launch(stream.value);
      CUDA_CHECK(cudaStreamSynchronize(stream.value));
    }
    verify_output(config, output.data, dataset.expected);
    std::cout << order << ',' << run + 1 << ',' << config.size << ',' << config.bins
              << ",u32,u32,shared,2,192," << eviction_bytes << ",PASS\n";
  }
  std::cerr << "Validated " << runs << " launches of "
            << (order == "evict-first" ? "E,H,H" : "H,E,H")
            << "; input_bytes=" << input.bytes << ", L2_bytes=" << gpu.l2CacheSize
            << ", eviction_bytes=" << eviction_bytes << ".\n"
               "Whole-graph counters include the identical eviction traffic in both orders.\n"
               "Effective eviction predicts fewer read hits and more misses for H,E,H;\n"
               "the target difference is one input read (32768 32-byte sectors).\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "ERROR: " << error.what() << '\n';
  return 1;
}
