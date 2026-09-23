#include "gh/histogram.hpp"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace gh::detail {
namespace {

template <unsigned Value>
inline constexpr unsigned log2_v = [] {
  unsigned result = 0;
  for (unsigned value = Value; value > 1; value >>= 1) ++result;
  return result;
}();

template <class Counter>
__device__ __forceinline__ void add_nonzero(Counter* output, Counter value) {
  if (value != 0) atomicAdd(output, value);
}

// All threads execute the same tile loop. Within a tile, branches surrounding
// warp collectives are uniform for the entire warp, including the final tile.
// A lane owns bins lane + 32*q. Its counters persist over all assigned tiles.
template <class Input, class Counter, unsigned Capacity, int Threads, int Items>
__global__ void bitplane_histogram(const Input* __restrict__ input,
                                  std::size_t size,
                                  Counter* __restrict__ output,
                                  unsigned bins) {
  static_assert(Threads % 32 == 0);
  static_assert(Capacity >= 1 && Capacity <= 256);
  static_assert((Capacity & (Capacity - 1)) == 0);
  constexpr unsigned planes = log2_v<Capacity>;
  constexpr unsigned low_planes = planes < 5 ? planes : 5;
  constexpr unsigned counters = Capacity <= 32 ? 1 : Capacity / 32;
  constexpr unsigned warps = Threads / 32;
  constexpr unsigned full_warp = 0xffffffffu;
  constexpr std::size_t tile_size = std::size_t{Threads} * Items;

  const unsigned lane = threadIdx.x & 31u;
  const unsigned warp = threadIdx.x / 32u;
  Counter count[counters] = {};
  __shared__ Counter partial[warps][Capacity];

  const std::size_t stride = std::size_t{gridDim.x} * tile_size;
  for (std::size_t tile = std::size_t{blockIdx.x} * tile_size;
       tile < size; tile += stride) {
#pragma unroll
    for (int item = 0; item < Items; ++item) {
      const std::size_t warp_base = tile + static_cast<std::size_t>(item) * Threads + warp * 32u;
      if (warp_base >= size) continue;  // Uniform across all 32 lanes.
      const bool valid = lane < size - warp_base;
      const unsigned bucket = valid ? static_cast<unsigned>(input[warp_base + lane]) : 0u;
      // Starting with valid_mask makes every complement below safe for tails.
      unsigned base = size - warp_base >= 32
                          ? full_warp
                          : __ballot_sync(full_warp, valid);
#pragma unroll
      for (unsigned bit = 0; bit < low_planes; ++bit) {
        const unsigned plane = __ballot_sync(full_warp, (bucket >> bit) & 1u);
        base &= ((lane >> bit) & 1u) ? plane : ~plane;
      }

      unsigned masks[counters];
      masks[0] = base;
      // After processing high bit h, masks[q] identifies the lanes whose
      // high bits equal q. Appending at j+offset preserves the binary bin order;
      // placing children at 2*j instead would permute bins for Capacity >=128.
#pragma unroll
      for (unsigned bit = 5; bit < planes; ++bit) {
        const unsigned plane = __ballot_sync(full_warp, (bucket >> bit) & 1u);
        const unsigned offset = 1u << (bit - 5);
#pragma unroll
        for (unsigned j = 0; j < offset; ++j) {
          const unsigned previous = masks[j];
          masks[j] = previous & ~plane;
          masks[j + offset] = previous & plane;
        }
      }
#pragma unroll
      for (unsigned q = 0; q < counters; ++q) {
        count[q] += static_cast<Counter>(__popc(masks[q]));
      }
    }
  }

#pragma unroll
  for (unsigned q = 0; q < counters; ++q) {
    const unsigned bin = lane + 32u * q;
    if (bin < Capacity) partial[warp][bin] = count[q];
  }
  __syncthreads();

  // Exactly one block thread sums each output bin across all warps. Bins
  // between bins and Capacity have no public output and are never published.
  for (unsigned bin = threadIdx.x; bin < bins; bin += Threads) {
    Counter total = 0;
#pragma unroll
    for (unsigned w = 0; w < warps; ++w) total += partial[w][bin];
    add_nonzero(output + bin, total);
  }
}

template <class Input, class Counter, unsigned Capacity, int Threads, int Items>
cudaError_t launch_policy(const Config& config, const void* input, void* output,
                          cudaStream_t stream) {
  bitplane_histogram<Input, Counter, Capacity, Threads, Items>
      <<<config.blocks, Threads, 0, stream>>>(static_cast<const Input*>(input), config.size,
                                            static_cast<Counter*>(output), config.bins);
  return cudaGetLastError();
}

// Separate from the scalar control kernel above so policy0..5 retain their
// original instruction stream. All 32 lanes must call this helper together.
template <class Counter, unsigned Capacity>
__device__ __forceinline__ void accumulate_bitplanes(
    unsigned bucket, unsigned lane, unsigned valid_mask,
    Counter (&count)[Capacity <= 32 ? 1 : Capacity / 32]) {
  constexpr unsigned planes = log2_v<Capacity>;
  constexpr unsigned low_planes = planes < 5 ? planes : 5;
  constexpr unsigned counters = Capacity <= 32 ? 1 : Capacity / 32;
  constexpr unsigned full_warp = 0xffffffffu;
  unsigned base = valid_mask;
#pragma unroll
  for (unsigned bit = 0; bit < low_planes; ++bit) {
    const unsigned plane = __ballot_sync(full_warp, (bucket >> bit) & 1u);
    base &= ((lane >> bit) & 1u) ? plane : ~plane;
  }
  unsigned masks[counters];
  masks[0] = base;
#pragma unroll
  for (unsigned bit = 5; bit < planes; ++bit) {
    const unsigned plane = __ballot_sync(full_warp, (bucket >> bit) & 1u);
    const unsigned offset = 1u << (bit - 5);
#pragma unroll
    for (unsigned j = 0; j < offset; ++j) {
      const unsigned previous = masks[j];
      masks[j] = previous & ~plane;
      masks[j + offset] = previous & plane;
    }
  }
#pragma unroll
  for (unsigned q = 0; q < counters; ++q)
    count[q] += static_cast<Counter>(__popc(masks[q]));
}

template <class Input, class Counter, unsigned Capacity, int Threads, int Items>
__global__ void bitplane_histogram_full_tile(const Input* __restrict__ input,
                                            std::size_t size,
                                            Counter* __restrict__ output,
                                            unsigned bins) {
  static_assert(Threads > 0 && Threads <= 1024 && Threads % 32 == 0);
  static_assert(Items > 0);
  static_assert(Capacity >= 1 && Capacity <= 256 && (Capacity & (Capacity - 1)) == 0);
  constexpr unsigned counters = Capacity <= 32 ? 1 : Capacity / 32;
  constexpr unsigned warps = Threads / 32;
  constexpr unsigned full_warp = 0xffffffffu;
  constexpr std::size_t tile_size = std::size_t{Threads} * Items;
  static_assert(std::size_t{warps} * Capacity * sizeof(Counter) <= 48 * 1024);

  const unsigned lane = threadIdx.x & 31u;
  const unsigned warp = threadIdx.x / 32u;
  Counter count[counters] = {};
  __shared__ Counter partial[warps][Capacity];
  const std::size_t stride = std::size_t{gridDim.x} * tile_size;
  const std::size_t full_end = size - size % tile_size;
  std::size_t tile = std::size_t{blockIdx.x} * tile_size;

  // This block owns only complete tiles in the hot loop. Every load is valid:
  // no warp entry check, tail predicate, or validity ballot is needed here.
  for (; tile < full_end; tile += stride) {
#pragma unroll
    for (int item = 0; item < Items; ++item) {
      const std::size_t index = tile + static_cast<std::size_t>(item) * Threads + threadIdx.x;
      const unsigned bucket = static_cast<unsigned>(input[index]);
      accumulate_bitplanes<Counter, Capacity>(bucket, lane, full_warp, count);
    }
  }

  // Exactly one block can own the remaining incomplete tile. Branches around
  // the collectives are warp-uniform, and invalid lanes are masked from counts.
  if (tile < size) {
#pragma unroll
    for (int item = 0; item < Items; ++item) {
      const std::size_t warp_base = tile + static_cast<std::size_t>(item) * Threads + warp * 32u;
      if (warp_base >= size) continue;
      const bool valid = lane < size - warp_base;
      const unsigned bucket = valid ? static_cast<unsigned>(input[warp_base + lane]) : 0u;
      const unsigned valid_mask = size - warp_base >= 32
                                      ? full_warp
                                      : __ballot_sync(full_warp, valid);
      accumulate_bitplanes<Counter, Capacity>(bucket, lane, valid_mask, count);
    }
  }

#pragma unroll
  for (unsigned q = 0; q < counters; ++q) {
    const unsigned bin = lane + 32u * q;
    if (bin < Capacity) partial[warp][bin] = count[q];
  }
  __syncthreads();
  for (unsigned bin = threadIdx.x; bin < bins; bin += Threads) {
    Counter total = 0;
#pragma unroll
    for (unsigned w = 0; w < warps; ++w) total += partial[w][bin];
    add_nonzero(output + bin, total);
  }
}

template <class Input, class Counter, unsigned Capacity, int Threads, int Items>
cudaError_t launch_full_tile_policy(const Config& config, const void* input, void* output,
                                    cudaStream_t stream) {
  bitplane_histogram_full_tile<Input, Counter, Capacity, Threads, Items>
      <<<config.blocks, Threads, 0, stream>>>(static_cast<const Input*>(input), config.size,
                                            static_cast<Counter*>(output), config.bins);
  return cudaGetLastError();
}

template <class Input, class Counter, unsigned Capacity, std::size_t Index = 0>
cudaError_t select_policy(const Config& config, const void* input, void* output,
                         cudaStream_t stream) {
  // Replication applies to the atomic backends. Here each warp already owns
  // private register counters, so replica-only policy changes are equivalent.
  if (config.tuning == static_cast<int>(Index)) {
    constexpr auto policy = tuning_catalog[Index];
    constexpr std::size_t shared_bytes = std::size_t{policy.threads / 32} * Capacity * sizeof(Counter);
    // Gate instantiation as well as launch: unsupported static shared arrays
    // must not be emitted merely because a policy exists in the shared catalog.
    if constexpr (policy.threads > 1024 || shared_bytes > 48 * 1024) {
      return cudaErrorInvalidValue;
    } else if constexpr (policy.load == LoadPolicy::scalar) {
      return launch_policy<Input, Counter, Capacity, policy.threads, policy.items>(config, input, output, stream);
    } else {
      // For bitplanes, vector4 is an alias for full_tile. There are no packed
      // loads here; benchmark metadata/deduplication must use that effective policy.
      return launch_full_tile_policy<Input, Counter, Capacity, policy.threads, policy.items>(
          config, input, output, stream);
    }
  }
  if constexpr (Index + 1 < tuning_count)
    return select_policy<Input, Counter, Capacity, Index + 1>(config, input, output, stream);
  return cudaErrorInvalidValue;
}

template <class Input, class Counter>
cudaError_t select_capacity(const Config& config, const void* input, void* output,
                           cudaStream_t stream) {
  if (config.bins == 0 || config.bins > 256) return cudaErrorInvalidValue;
  if (config.bins <= 1) return select_policy<Input, Counter, 1>(config, input, output, stream);
  if (config.bins <= 2) return select_policy<Input, Counter, 2>(config, input, output, stream);
  if (config.bins <= 4) return select_policy<Input, Counter, 4>(config, input, output, stream);
  if (config.bins <= 8) return select_policy<Input, Counter, 8>(config, input, output, stream);
  if (config.bins <= 16) return select_policy<Input, Counter, 16>(config, input, output, stream);
  if (config.bins <= 32) return select_policy<Input, Counter, 32>(config, input, output, stream);
  if (config.bins <= 64) return select_policy<Input, Counter, 64>(config, input, output, stream);
  if (config.bins <= 128) return select_policy<Input, Counter, 128>(config, input, output, stream);
  return select_policy<Input, Counter, 256>(config, input, output, stream);
}

}  // namespace

cudaError_t launch_bitplane(const Config& config, const void* input, void* output,
                           cudaStream_t stream) {
  if (config.size == 0) return cudaSuccess;  // Common entry point clears output.
  if (config.counter_type == CounterType::u32) {
    if (config.input_type == InputType::u8)
      return select_capacity<std::uint8_t, unsigned>(config, input, output, stream);
    return select_capacity<std::uint32_t, unsigned>(config, input, output, stream);
  }
  if (config.input_type == InputType::u8)
    return select_capacity<std::uint8_t, unsigned long long>(config, input, output, stream);
  return select_capacity<std::uint32_t, unsigned long long>(config, input, output, stream);
}

}  // namespace gh::detail
