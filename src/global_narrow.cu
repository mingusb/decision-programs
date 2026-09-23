#include <gh/histogram.hpp>

#include <cuda_runtime.h>

#include <algorithm>

namespace gh::detail {
namespace {

using Counter64 = unsigned long long;
static_assert(sizeof(Counter64) == sizeof(std::uint64_t));

__global__ void clear_narrow_counts(unsigned* counts, unsigned bins) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t bin = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       bin < bins; bin += stride)
    counts[bin] = 0;
}

__global__ void widen_counts(const unsigned* counts, Counter64* output, unsigned bins) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t bin = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       bin < bins; bin += stride)
    output[bin] = static_cast<Counter64>(counts[bin]);
}

// Match the existing global backend: every lane ballots, and exactly the lanes
// named by the resulting mask participate in match_any, including partial tails.
__device__ __forceinline__ void warp_add_narrow(unsigned* counts, unsigned key, bool valid) {
  const unsigned live = __ballot_sync(0xffffffffu, valid);
  if (valid) {
    const unsigned peers = __match_any_sync(live, key);
    if ((threadIdx.x & 31u) == static_cast<unsigned>(__ffs(peers) - 1))
      atomicAdd(counts + key, static_cast<unsigned>(__popc(peers)));
  }
}

template <typename Input, int Threads, int Items, bool Warp>
__global__ __launch_bounds__(Threads)
void global_narrow_histogram(const Input* input, std::size_t size, unsigned* counts) {
  constexpr std::size_t tile_size = static_cast<std::size_t>(Threads) * Items;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * tile_size;
  std::size_t base = static_cast<std::size_t>(blockIdx.x) * tile_size;
  while (base < size) {
    unsigned keys[Items];
    #pragma unroll
    for (int item = 0; item < Items; ++item) {
      const std::size_t offset = static_cast<std::size_t>(item) * Threads + threadIdx.x;
      keys[item] = offset < size - base ? static_cast<unsigned>(input[base + offset]) : 0;
    }
    #pragma unroll
    for (int item = 0; item < Items; ++item) {
      const std::size_t offset = static_cast<std::size_t>(item) * Threads + threadIdx.x;
      const bool valid = offset < size - base;
      if constexpr (Warp)
        warp_add_narrow(counts, keys[item], valid);
      else if (valid)
        atomicAdd(counts + keys[item], 1u);
    }
    if (stride >= size - base) break;
    base += stride;
  }
}

template <typename Input, int Threads, int Items>
cudaError_t launch_policy(const Config& config, const Input* input, unsigned* counts,
                          cudaStream_t stream) {
  if (config.algorithm == Algorithm::global_atomic)
    global_narrow_histogram<Input, Threads, Items, false>
        <<<config.blocks, Threads, 0, stream>>>(input, config.size, counts);
  else if (config.algorithm == Algorithm::warp_aggregated)
    global_narrow_histogram<Input, Threads, Items, true>
        <<<config.blocks, Threads, 0, stream>>>(input, config.size, counts);
  else
    return cudaErrorInvalidValue;
  return cudaGetLastError();
}

template <typename Input, std::size_t Index = 0>
cudaError_t launch_tuning(const Config& config, const void* input, unsigned* counts,
                          cudaStream_t stream) {
  static_assert(Index < 6);
  constexpr auto policy = tuning_catalog[Index];
  static_assert(policy.load == LoadPolicy::scalar);
  if (config.tuning == static_cast<int>(Index))
    return launch_policy<Input, policy.threads, policy.items>(
        config, static_cast<const Input*>(input), counts, stream);
  if constexpr (Index + 1 < 6)
    return launch_tuning<Input, Index + 1>(config, input, counts, stream);
  return cudaErrorInvalidValue;
}

}  // namespace

// The public dispatcher validates scalar global/warp policies, u64 output,
// local-u32 selection, and size <= UINT_MAX before reaching this backend.
// This bound applies to the entire histogram, not to an individual CTA.
// Nonempty calls supply bins*sizeof(unsigned) bytes of nonoverlapping workspace.
cudaError_t launch_global_narrow(const Config& config, const void* input, void* output,
                                void* workspace, cudaStream_t stream) {
  auto* counts = static_cast<unsigned*>(workspace);
  constexpr unsigned threads = 256;
  const unsigned blocks = std::min((config.bins + threads - 1) / threads, 256u);
  cudaError_t status;
  if (config.output_clear == OutputClear::kernel) {
    clear_narrow_counts<<<blocks, threads, 0, stream>>>(counts, config.bins);
    status = cudaGetLastError();
  } else {
    status = cudaMemsetAsync(counts, 0, static_cast<std::size_t>(config.bins) * sizeof(unsigned), stream);
  }
  if (status != cudaSuccess) return status;

  if (config.input_type == InputType::u8)
    status = launch_tuning<unsigned char>(config, input, counts, stream);
  else
    status = launch_tuning<unsigned>(config, input, counts, stream);
  if (status != cudaSuccess) return status;

  // Stream ordering completes all atomics before widening. Every output bin is
  // overwritten, so output initialization and a second atomic reduction are unnecessary.
  widen_counts<<<blocks, threads, 0, stream>>>(counts, static_cast<Counter64*>(output), config.bins);
  return cudaGetLastError();
}

}  // namespace gh::detail
