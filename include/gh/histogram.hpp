#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace gh {

enum class InputType { u8, u32 };
enum class CounterType { u32, u64 };
enum class LocalCounter { native, u32 };
enum class LoadPolicy { scalar, full_tile, vector4 };
enum class OutputClear { runtime, kernel };
enum class LaunchMode { stream, graph };
enum class CacheMode { warm, cold };
enum class Algorithm { global_atomic, warp_aggregated, shared_atomic,
                       shared_rle, shared_warp, shared_partial, bitplane, automatic,
                       shared_overflow, global_window };

// A deliberately bounded catalog of compile-time policies. Grid size remains runtime.
struct Tuning {
  int threads;
  int items;
  int replicas;
  LoadPolicy load = LoadPolicy::scalar;
  std::size_t shared_limit = 48 * 1024;
};
inline constexpr Tuning tuning_catalog[] = {
    {128, 4, 1}, {256, 4, 1}, {256, 8, 1},
    {256, 16, 1}, {128, 8, 4}, {256, 8, 4},
    {256, 8, 1, LoadPolicy::full_tile}, {256, 8, 1, LoadPolicy::vector4},
    {256, 16, 1, LoadPolicy::full_tile}, {256, 16, 1, LoadPolicy::vector4},
    {512, 8, 1, LoadPolicy::vector4}, {1024, 8, 1, LoadPolicy::vector4},
    {256, 8, 8, LoadPolicy::vector4}, {512, 8, 16, LoadPolicy::vector4},
    {256, 8, 1, LoadPolicy::vector4, 96 * 1024},
    {512, 8, 1, LoadPolicy::vector4, 96 * 1024}};
inline constexpr std::size_t tuning_count = sizeof(tuning_catalog) / sizeof(tuning_catalog[0]);

struct Config {
  Algorithm algorithm = Algorithm::automatic;
  InputType input_type = InputType::u32;
  CounterType counter_type = CounterType::u32;
  LocalCounter local_counter = LocalCounter::native;
  std::size_t size = 0;
  unsigned bins = 256;
  int blocks = 192;
  int tuning = 2;
  OutputClear output_clear = OutputClear::runtime;
  LaunchMode launch = LaunchMode::stream;
  CacheMode cache = CacheMode::warm;
  // Explicit global_window experiment only: maximum u32 counters reused per pass.
  unsigned window_bins = 524288;
};

inline constexpr const char* name(Algorithm a) {
  switch (a) {
    case Algorithm::global_atomic: return "global";
    case Algorithm::warp_aggregated: return "warp";
    case Algorithm::shared_atomic: return "shared";
    case Algorithm::shared_rle: return "shared_rle";
    case Algorithm::shared_warp: return "shared_warp";
    case Algorithm::shared_partial: return "shared_partial";
    case Algorithm::bitplane: return "bitplane";
    case Algorithm::automatic: return "auto";
    case Algorithm::shared_overflow: return "shared_overflow";
    case Algorithm::global_window: return "global_window";
  }
  return "unknown";
}
inline constexpr const char* name(InputType t) { return t == InputType::u8 ? "u8" : "u32"; }
inline constexpr const char* name(CounterType t) { return t == CounterType::u32 ? "u32" : "u64"; }
inline constexpr const char* name(LocalCounter t) { return t == LocalCounter::native ? "native" : "u32"; }
inline constexpr const char* name(OutputClear clear) {
  switch (clear) {
    case OutputClear::runtime: return "runtime";
    case OutputClear::kernel: return "kernel";
  }
  return "unknown";
}
inline constexpr const char* name(LaunchMode launch) {
  switch (launch) {
    case LaunchMode::stream: return "stream";
    case LaunchMode::graph: return "graph";
  }
  return "unknown";
}
inline constexpr const char* name(CacheMode cache) {
  switch (cache) {
    case CacheMode::warm: return "warm";
    case CacheMode::cold: return "cold";
  }
  return "unknown";
}
inline constexpr const char* name(LoadPolicy load) {
  switch (load) {
    case LoadPolicy::scalar: return "scalar";
    case LoadPolicy::full_tile: return "full_tile";
    case LoadPolicy::vector4: return "vector4";
  }
  return "unknown";
}
inline constexpr std::size_t input_bytes(InputType t) { return t == InputType::u8 ? 1 : 4; }
inline constexpr std::size_t counter_bytes(CounterType t) { return t == CounterType::u32 ? 4 : 8; }

// Contract: device-resident direct bin IDs in [0,bins), dense output is overwritten.
// Input/output/workspace must not overlap. Invalid IDs are a caller precondition.
// u32 counters require size <= UINT32_MAX. Empty input is supported and clears output.
// LocalCounter::u32 requires u64 output. Shared algorithms prove that the maximum
// samples assigned to one CTA fit a u32 counter. Global/warp algorithms instead
// require total size <= UINT32_MAX: every block updates the same bins*4-byte
// scratch histogram, which is cleared, counted, then widened to overwrite output.
// shared_overflow supports u32 input/u64 output/u32 locals with policies 14/15
// and bins >24576. Each CTA counts the first 24576 bins in shared memory and
// updates remaining output bins directly with u64 atomics; no scratch is needed.
// global_window is an explicit experiment for u32 input/u64 output/u32 locals,
// scalar policies 0..5, size <= UINT32_MAX, and window_bins in [1, INT_MAX-1].
// Each pass rescans the input for one contiguous bin window, then overwrites
// that output range. Scratch is min(bins, window_bins)*4 bytes for nonempty input.
// Ordinary stream ordering permits safe reuse and graph replay with changed input.
// Input and output pointers require only natural alignment of their element types.
// Shared vector4 policies pack four adjacent samples per thread on aligned full
// tiles; scalar fallback handles other inputs and tails. This changes per-thread
// sample order (and hence RLE opportunities), but not CTA ownership or counts.
// OutputClear selects initialization for nonempty custom atomic/bit-plane paths.
// The runtime default uses cudaMemsetAsync. Set output_clear to kernel before
// graph capture to record the explicit clearing kernel. The choice is explicit;
// histogram() does not query capture state. Empty inputs and shared_partial
// retain their own initialization paths.
// Stream-ordered, allocation-free execution; workspace is queried/provisioned separately.
// These prototypes target sm_86. Policies with shared_limit >48 KiB are restricted
// to shared algorithms and require prepare() on the current device before use.
// A default Config is unresolved (algorithm == automatic). Call the mutable
// prepare() overload before workspace queries or execution. Resolution uses the
// declared launch/cache mode, measured choices, and resource-safe heuristics for
// other shapes; it neither inspects input values nor detects cache residency.
// Changing the workload after resolution requires resetting algorithm to
// automatic and preparing again. Explicit algorithms retain their given policy.
bool supported(const Config& config);
// Pure selector: no CUDA queries, allocations, input inspection, or launches.
// Returns explicit configurations unchanged. All choices use this library's kernels.
Config default_config(const Config& config, const cudaDeviceProp& device);
// Resolve automatic once on the current device, then apply resource setup.
// This may query the device, so call it outside capture and timed execution.
// Configuration is assigned only after successful resolution and preparation.
cudaError_t prepare(Config& config);
// Explicit setup, outside timed calls and graph capture. supported() is structural;
// prepare() additionally checks the declared opt-in capacity for large policies and
// configures their exact kernel. Returns cudaErrorNotSupported on insufficient
// device capacity. Repeat after changing devices or resetting the CUDA context.
// Policies requiring <=48 KiB need no setup and return success after validation.
// This const overload rejects automatic; it never resolves a policy implicitly.
cudaError_t prepare(const Config& config);
cudaError_t workspace_bytes(const Config& config, std::size_t& bytes);
cudaError_t histogram(const Config& config, const void* input, void* output,
                      void* workspace, std::size_t bytes, cudaStream_t stream = nullptr);

namespace detail {
// Internal bit-plane backend: common entry point owns validation and output clearing.
cudaError_t launch_bitplane(const Config&, const void*, void*, cudaStream_t);
// Internal narrow global backend: owns scratch initialization and output overwrite.
cudaError_t launch_global_narrow(const Config&, const void*, void*, void*, cudaStream_t);
cudaError_t launch_global_window(const Config&, const void*, void*, void*, cudaStream_t);
cudaError_t prepare_shared_overflow(const Config&);
cudaError_t launch_shared_overflow(const Config&, const void*, void*, cudaStream_t);
}
}  // namespace gh
