#include <gh/histogram.hpp>

#include <cuda_runtime.h>

namespace gh::detail {
namespace {

using Counter64 = unsigned long long;
static_assert(sizeof(Counter64) == sizeof(std::uint64_t));
static_assert(sizeof(unsigned) == sizeof(std::uint32_t));

constexpr unsigned prefix_bins = 24576;
constexpr std::size_t shared_bytes = static_cast<std::size_t>(prefix_bins) * sizeof(unsigned);
static_assert(shared_bytes == 96 * 1024);

// This backend has its own kernel instantiations, so opting in here does not
// change the function attributes of any existing shared-histogram kernel.
template <int Threads, bool FullTile, bool Vector4>
__device__ __forceinline__ void accumulate_overflow_tile(
    const unsigned* input, std::size_t base, std::size_t remaining,
    unsigned* prefix, Counter64* output) {
  static_assert(!Vector4 || FullTile);
  constexpr int items = 8;
  unsigned keys[items];
  if constexpr (Vector4) {
    #pragma unroll
    for (int group = 0; group < items / 4; ++group) {
      const std::size_t vector_index = static_cast<std::size_t>(group) * Threads + threadIdx.x;
      const uint4 packed = reinterpret_cast<const uint4*>(input + base)[vector_index];
      keys[group * 4] = packed.x;
      keys[group * 4 + 1] = packed.y;
      keys[group * 4 + 2] = packed.z;
      keys[group * 4 + 3] = packed.w;
    }
  } else {
    #pragma unroll
    for (int item = 0; item < items; ++item) {
      const std::size_t offset = static_cast<std::size_t>(item) * Threads + threadIdx.x;
      if constexpr (FullTile)
        keys[item] = input[base + offset];
      else
        keys[item] = offset < remaining ? input[base + offset] : 0;
    }
  }
  #pragma unroll
  for (int item = 0; item < items; ++item) {
    const bool valid = FullTile || static_cast<std::size_t>(item) * Threads + threadIdx.x < remaining;
    if (valid) {
      const unsigned key = keys[item];
      if (key < prefix_bins)
        atomicAdd(prefix + key, 1u);
      else
        atomicAdd(output + key, Counter64{1});
    }
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads)
void shared_overflow_histogram(const unsigned* input, std::size_t size, Counter64* output) {
  static_assert(Threads == 256 || Threads == 512);
  extern __shared__ __align__(16) unsigned char storage[];
  auto* prefix = reinterpret_cast<unsigned*>(storage);
  for (unsigned bin = threadIdx.x; bin < prefix_bins; bin += Threads)
    prefix[bin] = 0;
  __syncthreads();

  // Identical CTA tile ownership to shared_histogram_loaded: eight items per
  // thread, with each CTA traversing every gridDim.x-th tile. A full aligned
  // tile uses vector loads; natural four-byte alignment alone remains valid.
  constexpr std::size_t tile_size = static_cast<std::size_t>(Threads) * 8;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * tile_size;
  std::size_t base = static_cast<std::size_t>(blockIdx.x) * tile_size;
  const bool vector_aligned = reinterpret_cast<std::uintptr_t>(input) % sizeof(uint4) == 0;
  while (base < size) {
    const std::size_t remaining = size - base;
    if (remaining >= tile_size) {
      if (vector_aligned)
        accumulate_overflow_tile<Threads, true, true>(input, base, remaining, prefix, output);
      else
        accumulate_overflow_tile<Threads, true, false>(input, base, remaining, prefix, output);
    } else {
      accumulate_overflow_tile<Threads, false, false>(input, base, remaining, prefix, output);
    }
    // The subtraction form also prevents an overflowing final base increment.
    if (stride >= remaining) break;
    base += stride;
  }
  __syncthreads();

  // Prefix and overflow bins are disjoint. Each CTA publishes every nonzero
  // prefix count exactly once; global atomics combine all CTA contributions.
  for (unsigned bin = threadIdx.x; bin < prefix_bins; bin += Threads) {
    const unsigned count = prefix[bin];
    if (count != 0) atomicAdd(output + bin, static_cast<Counter64>(count));
  }
}

bool shape_supported(const Config& config) {
  return config.input_type == InputType::u32 &&
         config.counter_type == CounterType::u64 &&
         config.local_counter == LocalCounter::u32 &&
         config.bins > prefix_bins && config.blocks > 0 &&
         (config.tuning == 14 || config.tuning == 15);
}

static_assert(tuning_catalog[14].threads == 256 && tuning_catalog[15].threads == 512);
static_assert(tuning_catalog[14].items == 8 && tuning_catalog[15].items == 8);
static_assert(tuning_catalog[14].replicas == 1 && tuning_catalog[15].replicas == 1);
static_assert(tuning_catalog[14].load == LoadPolicy::vector4 &&
              tuning_catalog[15].load == LoadPolicy::vector4);
static_assert(tuning_catalog[14].shared_limit == shared_bytes &&
              tuning_catalog[15].shared_limit == shared_bytes);

}  // namespace

// The common host dispatcher additionally proves the maximum samples owned by
// one CTA fit UINT_MAX, validates pointers, and checks device opt-in shared-memory
// capacity before calling this setup. Keys in [0, bins) remain a caller precondition.
// Setup is outside graph capture and timed execution; repeat on device changes
// or context reset, just as for existing opt-in shared kernels.
cudaError_t prepare_shared_overflow(const Config& config) {
  if (!shape_supported(config)) return cudaErrorInvalidValue;
  if (config.tuning == 14)
    return cudaFuncSetAttribute(shared_overflow_histogram<256>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(shared_bytes));
  return cudaFuncSetAttribute(shared_overflow_histogram<512>,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              static_cast<int>(shared_bytes));
}

// Output must already be zeroed in this stream. No workspace is needed, and
// launch performs no allocation, device query, or function-attribute mutation.
cudaError_t launch_shared_overflow(const Config& config, const void* input, void* output,
                                   cudaStream_t stream) {
  if (!shape_supported(config)) return cudaErrorInvalidValue;
  if (config.size == 0) return cudaSuccess;
  if (config.tuning == 14)
    shared_overflow_histogram<256><<<config.blocks, 256, shared_bytes, stream>>>(
        static_cast<const unsigned*>(input), config.size, static_cast<Counter64*>(output));
  else
    shared_overflow_histogram<512><<<config.blocks, 512, shared_bytes, stream>>>(
        static_cast<const unsigned*>(input), config.size, static_cast<Counter64*>(output));
  return cudaGetLastError();
}

}  // namespace gh::detail
