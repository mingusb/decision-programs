#include <gh/histogram.hpp>

#include <cuda_runtime.h>

#include <algorithm>

namespace gh::detail {
namespace {

using Counter64 = unsigned long long;
static_assert(sizeof(Counter64) == sizeof(std::uint64_t));

__global__ void clear_window_counts(unsigned* counts, unsigned bins) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t bin = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       bin < bins; bin += stride)
    counts[bin] = 0;
}

__global__ void widen_window_counts(const unsigned* counts, Counter64* output, unsigned bins) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t bin = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       bin < bins; bin += stride)
    output[bin] = static_cast<Counter64>(counts[bin]);
}

template <int Threads, int Items>
__global__ __launch_bounds__(Threads)
void global_window_histogram(const unsigned* input, std::size_t size, unsigned* counts,
                             unsigned first_bin, unsigned bins) {
  constexpr std::size_t tile_size = static_cast<std::size_t>(Threads) * Items;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * tile_size;
  std::size_t base = static_cast<std::size_t>(blockIdx.x) * tile_size;
  while (base < size) {
    unsigned keys[Items];
    #pragma unroll
    for (int item = 0; item < Items; ++item) {
      const std::size_t offset = static_cast<std::size_t>(item) * Threads + threadIdx.x;
      keys[item] = offset < size - base ? input[base + offset] : 0;
    }
    #pragma unroll
    for (int item = 0; item < Items; ++item) {
      const std::size_t offset = static_cast<std::size_t>(item) * Threads + threadIdx.x;
      // Unsigned subtraction rejects keys below first_bin by wraparound:
      // both valid IDs and window extents are below INT_MAX. No invalid lane
      // loads input or updates a counter, including partial warp/tile tails.
      const unsigned local_bin = keys[item] - first_bin;
      if (offset < size - base && local_bin < bins)
        atomicAdd(counts + local_bin, 1u);
    }
    if (stride >= size - base) break;
    base += stride;
  }
}

template <std::size_t Index = 0>
cudaError_t launch_tuning(const Config& config, const void* input, unsigned* counts,
                          unsigned first_bin, unsigned bins, cudaStream_t stream) {
  static_assert(Index < 6);
  constexpr auto policy = tuning_catalog[Index];
  static_assert(policy.load == LoadPolicy::scalar);
  if (config.tuning == static_cast<int>(Index)) {
    global_window_histogram<policy.threads, policy.items>
        <<<config.blocks, policy.threads, 0, stream>>>(
            static_cast<const unsigned*>(input), config.size, counts, first_bin, bins);
    return cudaGetLastError();
  }
  if constexpr (Index + 1 < 6)
    return launch_tuning<Index + 1>(config, input, counts, first_bin, bins, stream);
  return cudaErrorInvalidValue;
}

}  // namespace

// Validation guarantees nonempty u32 input, u64 output, u32 locals, size <=
// UINT32_MAX, and min(bins, window_bins)*4 bytes of nonoverlapping scratch.
// Filtering does not relax the counter bound: every sample may select one bin.
cudaError_t launch_global_window(const Config& config, const void* input, void* output,
                                void* workspace, cudaStream_t stream) {
  auto* counts = static_cast<unsigned*>(workspace);
  auto* result = static_cast<Counter64*>(output);
  constexpr unsigned threads = 256;
  for (unsigned first_bin = 0; first_bin < config.bins;) {
    const unsigned bins = std::min(config.window_bins, config.bins - first_bin);
    const unsigned blocks = std::min((bins + threads - 1) / threads, 256u);
    cudaError_t status;
    if (config.output_clear == OutputClear::kernel) {
      clear_window_counts<<<blocks, threads, 0, stream>>>(counts, bins);
      status = cudaGetLastError();
    } else {
      status = cudaMemsetAsync(counts, 0, static_cast<std::size_t>(bins) * sizeof(unsigned), stream);
    }
    if (status != cudaSuccess) return status;
    status = launch_tuning(config, input, counts, first_bin, bins, stream);
    if (status != cudaSuccess) return status;
    // The output range is overwritten, including absent bins. All scratch
    // reads finish on this stream before the next pass clears and reuses it.
    widen_window_counts<<<blocks, threads, 0, stream>>>(counts, result + first_bin, bins);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    // bins <= config.bins-first_bin, so even the final partial pass is safe.
    first_bin += bins;
  }
  return cudaSuccess;
}

}  // namespace gh::detail
