#include "gh/count.cuh"
#include "check.cuh"
#include <climits>

namespace gh::test {
namespace {
using namespace gh::count;
constexpr unsigned max_input = 8193, max_bins = 32769, workspace_bytes = 32768;
__device__ __align__(16) std::byte input_storage[(max_input + 8) * 4];
__device__ __align__(16) std::byte output_storage[(max_bins + 8) * 8];
__device__ __align__(16) std::byte scratch_storage[workspace_bytes + 16];
__device__ Status result;
__device__ unsigned graph_epoch;
constexpr unsigned variants = 16 * 9 * 2 * 2 * 2;
constexpr unsigned ordinary_cases = variants * 2;
constexpr unsigned extra_cases = 23;

struct Case { Config config; unsigned shift{}, pattern{}, seed{}; u32 error{};
              u64 scratch{workspace_bytes}; Hardware hardware; };
__device__ Case make_case(unsigned id) {
  Case value;
  auto& c = value.config;
  c.blocks = 3;
  c.output_clear = id & 1 ? OutputClear::runtime : OutputClear::kernel;
  if (id < ordinary_cases) {
    const unsigned mode = id / variants;
    unsigned key = id % variants;
    c.policy = key % 16; key /= 16;
    const unsigned algorithm = key % 9; key /= 9;
    c.algorithm = Algorithm(algorithm < 7 ? algorithm : algorithm + 1);
    c.input_type = InputType(key % 2); key /= 2;
    c.counter_type = CounterType(key % 2); key /= 2;
    c.local_counter = LocalCounter(key);
    c.bins = c.algorithm == Algorithm::shared_overflow ? 32769 : c.algorithm == Algorithm::global_window ? 521 : 33;
    c.window_bins = 257;
    c.size = mode ? 257 : max_input;
    value.shift = mode; value.pattern = mode; value.seed = 19 + mode;
    if (c.algorithm == Algorithm::shared_overflow) value.pattern = 3;
  } else {
    const unsigned key = id - ordinary_cases;
    constexpr unsigned bins[] = {1,8,31,32,33,128,255,256};
    c.input_type = InputType::u32; c.counter_type = CounterType::u64;
    c.algorithm = key < 16 ? Algorithm::bitplane : Algorithm::shared_partial;
    c.policy = key < 16 ? (key & 1 ? 7 : 0) : 2;
    c.bins = key < 16 ? bins[key / 2] : 33;
    c.size = key < 16 ? (key & 1 ? 33 : 0) : 257;
    value.pattern = key % 3; value.seed = 11;
    if (key == 17) { c.local_counter = LocalCounter::u32; value.scratch = u64(c.blocks) * c.bins * 4 - 1; value.error = capacity; }
    if (key == 18) { c.bins = 0; value.error = shape; }
    if (key == 19) { c.size = 0; }
    if (key == 20) { c.policy = 16; value.error = shape; }
    if (key == 21) { value.scratch = u64(c.blocks) * c.bins * 8; }
    if (key == 22) {
      c.algorithm = Algorithm::bitplane; c.policy = 0;
      value.hardware.shared_bytes = 256; value.error = unsupported;
    }
  }
  return value;
}
__device__ unsigned sample(unsigned index, unsigned bins, unsigned pattern, unsigned seed) {
  if (pattern == 0) return (index + seed) % bins;
  if (pattern == 1) return (index / 7 + seed) % bins;
  if (pattern == 3) {
    const unsigned part = (index + seed) % 3;
    return part == 0 ? 0 : part == 1 ? 24575 : bins - 1;
  }
  return unsigned(mix(u64(index) + seed) % bins);
}
__device__ u64 expected(Config c, unsigned bin, unsigned pattern, unsigned seed) {
  if (pattern == 3) {
    if (bin != 0 && bin != 24575 && bin != c.bins - 1) return 0;
    const unsigned part = bin == 0 ? 0 : bin == 24575 ? 1 : 2;
    const unsigned offset = (part + 3 - seed % 3) % 3;
    return c.size / 3 + (offset < c.size % 3);
  }
  if (pattern == 2) {
    u64 count = 0;
    for (u64 i = 0; i < c.size; ++i) count += sample(unsigned(i),c.bins,pattern,seed) == bin;
    return count;
  }
  const u64 offset = (u64(bin) + c.bins - seed % c.bins) % c.bins;
  const u64 groups = pattern == 0 ? c.size : c.size / 7;
  const u64 count = groups / c.bins + (offset < groups % c.bins);
  return pattern == 0 ? count : 7 * count + (offset == groups % c.bins ? c.size % 7 : 0);
}
__device__ std::byte* input_ptr(Case c) {
  return input_storage + 16 + c.shift * (c.config.input_type == InputType::u8 ? 1 : 4);
}
__global__ void initialize(Case c) {
  const unsigned lane = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned i = lane; i < sizeof(output_storage); i += blockDim.x * gridDim.x)
    output_storage[i] = std::byte{0xa5};
  for (unsigned i = lane; i < sizeof(scratch_storage); i += blockDim.x * gridDim.x)
    scratch_storage[i] = std::byte{0x6b};
  for (u64 i = lane; i < c.config.size; i += blockDim.x * gridDim.x) {
    const auto key = sample(unsigned(i),max(c.config.bins,1u),c.pattern,c.seed);
    if (c.config.input_type == InputType::u8) reinterpret_cast<unsigned char*>(input_ptr(c))[i] = key;
    else reinterpret_cast<unsigned*>(input_ptr(c))[i] = key;
  }
}
__global__ void next_case(unsigned id);
__global__ void check_case(Case c, unsigned next, bool graph) {
  const unsigned lane = blockIdx.x * blockDim.x + threadIdx.x;
  GH_CHECK(result.done == 1);
  GH_CHECK(result.errors == c.error);
  const u64 bytes = u64(c.config.bins) * (c.config.counter_type == CounterType::u32 ? 4 : 8);
  for (unsigned i = lane; i < sizeof(output_storage); i += blockDim.x * gridDim.x) {
    if (c.error || i < 16 || i >= 16 + bytes) GH_CHECK(output_storage[i] == std::byte{0xa5});
  }
  const u64 scratch_used = c.error ? 0 : required_bytes(resolve(c.config));
  for (u64 i = lane; i < sizeof(scratch_storage); i += blockDim.x * gridDim.x)
    if (i >= scratch_used) GH_CHECK(scratch_storage[i] == std::byte{0x6b});
  if (!c.error) {
    for (unsigned bin = lane; bin < c.config.bins; bin += blockDim.x * gridDim.x) {
      const u64 actual = c.config.counter_type == CounterType::u32
          ? reinterpret_cast<unsigned*>(output_storage + 16)[bin]
          : reinterpret_cast<u64*>(output_storage + 16)[bin];
      GH_CHECK(actual == expected(c.config,bin,c.pattern,c.seed));
    }
    for (u64 i = lane; i < c.config.size; i += blockDim.x * gridDim.x) {
      const unsigned actual = c.config.input_type == InputType::u8
          ? reinterpret_cast<unsigned char*>(input_ptr(c))[i] : reinterpret_cast<unsigned*>(input_ptr(c))[i];
      GH_CHECK(actual == sample(unsigned(i),c.config.bins,c.pattern,c.seed));
    }
  }
  if (!lane && !graph) {
    if (next < ordinary_cases + extra_cases) {
      next_case<<<1,1,0,cudaStreamTailLaunch>>>(next); submitted(cudaGetLastError());
    }
    else printf("count GPU policy/type/capacity checks completed\n");
  }
}
__device__ void launch_case(Case c, unsigned next, bool graph) {
  result = {};
  initialize<<<32,256>>>(c);
  submitted(cudaGetLastError());
  const u64 width = c.config.input_type == InputType::u8 ? 1 : 4;
  const auto launched = gh::count::count(c.config,{input_ptr(c),c.config.size * width},
      {output_storage + 16,sizeof(output_storage) - 16}, {scratch_storage,c.scratch}, &result,c.hardware);
  if (launched != cudaSuccess)
    printf("count case=%u algorithm=%u policy=%u input=%u output=%u local=%u size=%llu CUDA=%d\n",
      next - 1, unsigned(c.config.algorithm), c.config.policy, unsigned(c.config.input_type),
      unsigned(c.config.counter_type), unsigned(c.config.local_counter),
      static_cast<unsigned long long>(c.config.size), int(launched));
  submitted(launched);
  check_case<<<32,256,0,cudaStreamTailLaunch>>>(c,next,graph);
  submitted(cudaGetLastError());
}
__global__ void next_case(unsigned id) {
  for (; id < ordinary_cases; ++id) if (supported(make_case(id).config)) break;
  if (id < ordinary_cases + extra_cases) launch_case(make_case(id),id + 1,false);
}
__device__ void metadata_checks() {
  Config c;
  c.algorithm = Algorithm::shared_atomic; c.counter_type = CounterType::u64;
  c.local_counter = LocalCounter::u32; c.bins = 1; c.blocks = 1; c.policy = 0;
  c.size = UINT_MAX; GH_CHECK(supported(c));
  c.size = u64(UINT_MAX) + 1; GH_CHECK(!supported(c));
  c.blocks = 2; GH_CHECK(supported(c));
  c.algorithm = Algorithm::global_atomic; GH_CHECK(!supported(c));
  c.algorithm = Algorithm::warp_aggregated; GH_CHECK(!supported(c));
  c.algorithm = Algorithm::global_window; GH_CHECK(!supported(c));
  c.size = UINT_MAX; GH_CHECK(supported(c));
  c.local_counter = LocalCounter::native; c.counter_type = CounterType::u32;
  c.algorithm = Algorithm::global_atomic;
  c.size = u64(UINT_MAX) + 1; GH_CHECK(!supported(c));
  c.counter_type = CounterType::u64; c.algorithm = Algorithm::global_atomic;
  c.size = u64(PTRDIFF_MAX) / 4 + 1; GH_CHECK(!supported(c));
  c.size = 17; c.policy = 6; GH_CHECK(!supported(c));
  c.algorithm = Algorithm::bitplane; c.local_counter = LocalCounter::u32; GH_CHECK(!supported(c));
  c.local_counter = LocalCounter::native; c.policy = 11; c.bins = 256; GH_CHECK(!supported(c));
  c.bins = 128; GH_CHECK(supported(c));
  c.algorithm = Algorithm::shared_atomic; c.policy = 14; c.local_counter = LocalCounter::u32;
  c.bins = 24576; GH_CHECK(supported(c)); c.bins = 24577; GH_CHECK(!supported(c));
  c.algorithm = Algorithm::shared_overflow; GH_CHECK(supported(c));
  c.bins = 24576; GH_CHECK(!supported(c));
  c = {}; c.algorithm = Algorithm::shared_partial; c.counter_type = CounterType::u64;
  c.size = 1; c.blocks = INT_MAX; c.bins = 6000; GH_CHECK(supported(c));
  GH_CHECK(required_bytes(c) == u64(INT_MAX) * 6000 * 8);
  c.size = 0; GH_CHECK(required_bytes(c) == 0);

  struct Default { u64 size; unsigned bins; InputType input; CounterType counter;
                   LaunchMode launch; CacheMode cache; unsigned policy,blocks; LocalCounter local; };
  constexpr Default defaults[] = {
    {1048576,4096,InputType::u32,CounterType::u32,LaunchMode::stream,CacheMode::warm,10,48,LocalCounter::native},
    {4096,256,InputType::u8,CounterType::u32,LaunchMode::graph,CacheMode::warm,1,96,LocalCounter::native},
    {1048576,256,InputType::u8,CounterType::u32,LaunchMode::graph,CacheMode::warm,11,48,LocalCounter::native},
    {16777216,256,InputType::u8,CounterType::u32,LaunchMode::graph,CacheMode::warm,10,96,LocalCounter::native},
    {1048576,8,InputType::u32,CounterType::u32,LaunchMode::graph,CacheMode::warm,6,192,LocalCounter::native},
    {1048576,256,InputType::u32,CounterType::u32,LaunchMode::graph,CacheMode::warm,11,48,LocalCounter::native},
    {16777216,4096,InputType::u32,CounterType::u32,LaunchMode::graph,CacheMode::warm,11,48,LocalCounter::native},
    {16777216,4096,InputType::u32,CounterType::u64,LaunchMode::graph,CacheMode::warm,10,48,LocalCounter::u32},
    {16777216,8192,InputType::u32,CounterType::u64,LaunchMode::graph,CacheMode::warm,15,48,LocalCounter::u32},
    {16777216,16384,InputType::u32,CounterType::u64,LaunchMode::graph,CacheMode::warm,15,48,LocalCounter::u32},
    {1048576,4096,InputType::u32,CounterType::u64,LaunchMode::graph,CacheMode::cold,10,48,LocalCounter::u32}};
  for (const auto expected : defaults) {
    Config request;
    request.size = expected.size; request.bins = expected.bins;
    request.input_type = expected.input; request.counter_type = expected.counter;
    request.launch = expected.launch; request.cache = expected.cache;
    const auto chosen = resolve(request);
    GH_CHECK(chosen.algorithm == Algorithm::shared_atomic && chosen.policy == expected.policy &&
        chosen.blocks == expected.blocks && chosen.local_counter == expected.local);
    request.size += 1;
    GH_CHECK(resolve(request).policy != expected.policy);
    request.size -= 1;
    Hardware other; other.a5000_laptop = false;
    GH_CHECK(resolve(request,other).policy != expected.policy);
    GH_CHECK(chosen.output_clear == (request.launch == LaunchMode::graph ? OutputClear::kernel : OutputClear::runtime));
  }
  Config explicit_config; explicit_config.algorithm = Algorithm::global_atomic;
  explicit_config.policy = 0; explicit_config.blocks = 7;
  GH_CHECK(resolve(explicit_config).blocks == 7 && resolve(explicit_config).policy == 0);
}
} // namespace

__global__ void run() {
  metadata_checks();
  next_case<<<1,1>>>(0);
  submitted(cudaGetLastError());
}
// Only this CUDA bootstrap is captured. Input changes and checks are GPU work;
// no assertion below assumes the old direct-leaf graph timing topology.
__global__ void graph_case() {
  const unsigned epoch = graph_epoch++;
  Case c;
  c.config.algorithm = Algorithm::shared_atomic; c.config.policy = 7;
  c.config.size = 4097; c.config.bins = 33; c.config.blocks = 3;
  c.config.launch = LaunchMode::graph; c.config.output_clear = OutputClear::kernel;
  c.seed = epoch * 7 + 3; c.pattern = epoch % 2;
  launch_case(c,0,true);
}
__global__ void graph_replays_checked() { GH_CHECK(graph_epoch == 2); }
} // namespace gh::test

// CUDA-only graph bootstrap. The driver calls this after the ordinary suite.
// A host graph with CDP nodes is distinct from a device-launchable graph.
cudaError_t count_graph_checks() {
  cudaStream_t stream{}; cudaGraph_t graph{}; cudaGraphExec_t executable{};
  auto error = cudaStreamCreate(&stream);
  if (error != cudaSuccess) return error;
  error = cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal);
  if (error == cudaSuccess) {
    gh::test::graph_case<<<1,1,0,stream>>>();
    error = cudaStreamEndCapture(stream,&graph);
  }
  if (error == cudaSuccess) error = cudaGraphInstantiate(&executable,graph,0);
  if (error == cudaSuccess) error = cudaGraphLaunch(executable,stream);
  if (error == cudaSuccess) error = cudaGraphLaunch(executable,stream);
  if (error == cudaSuccess) {
    gh::test::graph_replays_checked<<<1,1,0,stream>>>();
    error = cudaStreamSynchronize(stream);
  }
  // Cleanup drains outstanding graph work even when a later submission fails.
  const auto drained = cudaStreamSynchronize(stream);
  const auto released = executable ? cudaGraphExecDestroy(executable) : cudaSuccess;
  const auto destroyed = graph ? cudaGraphDestroy(graph) : cudaSuccess;
  const auto closed = cudaStreamDestroy(stream);
  return error != cudaSuccess ? error : drained != cudaSuccess ? drained :
      released != cudaSuccess ? released : destroyed != cudaSuccess ? destroyed : closed;
}
