#include "frozen_count.cuh"

// Isolated benchmark reference: these five definitions are byte-identical to
// the frozen source. Provenance hashes are recorded beside benchmark receipts.
namespace gh::bench::frozen_count {
namespace {
constexpr std::size_t shared_memory_limit = 48 * 1024;
using Counter64 = unsigned long long;
static_assert(sizeof(Counter64) == sizeof(std::uint64_t));
enum class Update { atomic, warp, rle };
enum class LoadPolicy { scalar, full_tile, vector4 };

template <typename Counter>
__device__ __forceinline__ void add(Counter* address, Counter value) {
  atomicAdd(address, value);
}

template <typename Counter>
__global__ void clear_histogram_output(Counter* output, unsigned bins) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t bin = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       bin < bins; bin += stride)
    output[bin] = Counter{0};
}

template <typename Counter>
__device__ __forceinline__ void warp_add(Counter* counts, unsigned key, bool valid) {
  const unsigned live = __ballot_sync(0xffffffffu, valid);
  if (valid) {
    const unsigned peers = __match_any_sync(live, key);
    if ((threadIdx.x & 31u) == static_cast<unsigned>(__ffs(peers) - 1))
      add(counts + key, static_cast<Counter>(__popc(peers)));
  }
}

template <typename Input, typename Local, int Threads, int Items, Update Method,
          bool FullTile, bool Vector4>
__device__ __forceinline__ void accumulate_shared_tile(
    const Input* input, std::size_t base, std::size_t remaining,
    Local* private_counts, unsigned& previous, Local& run) {
  static_assert(!Vector4 || FullTile);
  static_assert(!Vector4 || Items % 4 == 0);
  unsigned keys[Items];
  if constexpr (Vector4) {
    #pragma unroll
    for (int group = 0; group < Items / 4; ++group) {
      const std::size_t vector_index = static_cast<std::size_t>(group) * Threads + threadIdx.x;
      if constexpr (sizeof(Input) == 1) {
        const unsigned packed = reinterpret_cast<const unsigned*>(input + base)[vector_index];
        #pragma unroll
        for (int item = 0; item < 4; ++item)
          keys[group * 4 + item] = (packed >> (8 * item)) & 0xffu;
      } else {
        static_assert(sizeof(Input) == sizeof(unsigned));
        const uint4 packed = reinterpret_cast<const uint4*>(input + base)[vector_index];
        keys[group * 4] = packed.x;
        keys[group * 4 + 1] = packed.y;
        keys[group * 4 + 2] = packed.z;
        keys[group * 4 + 3] = packed.w;
      }
    }
  } else {
    #pragma unroll
    for (int item = 0; item < Items; ++item) {
      const std::size_t offset = static_cast<std::size_t>(item) * Threads + threadIdx.x;
      if constexpr (FullTile)
        keys[item] = static_cast<unsigned>(input[base + offset]);
      else
        keys[item] = offset < remaining ? static_cast<unsigned>(input[base + offset]) : 0;
    }
  }
  #pragma unroll
  for (int item = 0; item < Items; ++item) {
    const bool valid = FullTile || static_cast<std::size_t>(item) * Threads + threadIdx.x < remaining;
    if constexpr (Method == Update::warp) {
      if constexpr (FullTile) {
        const unsigned peers = __match_any_sync(0xffffffffu, keys[item]);
        if ((threadIdx.x & 31u) == static_cast<unsigned>(__ffs(peers) - 1))
          add(private_counts + keys[item], static_cast<Local>(__popc(peers)));
      } else {
        warp_add(private_counts, keys[item], valid);
      }
    } else if constexpr (Method == Update::rle) {
      if (valid) {
        const unsigned key = keys[item];
        if (run != 0 && key != previous) {
          add(private_counts + previous, run);
          run = 0;
        }
        previous = key;
        ++run;
      }
    } else {
      if (valid) add(private_counts + keys[item], Local{1});
    }
  }
}

template <typename Input, typename Counter, typename Local, int Threads, int Items, int Replicas,
          Update Method, bool Partial, LoadPolicy Load, std::size_t SharedLimit>
__global__ __launch_bounds__(Threads)
void shared_histogram_loaded(const Input* input, std::size_t size, unsigned bins,
                             Counter* output, Local* partials) {
  static_assert(Load != LoadPolicy::scalar);
  // The capacity dimension isolates opt-in function attributes from <=48 KiB
  // policies with otherwise identical parameters.
  static_assert(SharedLimit == shared_memory_limit || SharedLimit == 96 * 1024);
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  auto* counts = reinterpret_cast<Local*>(shared_bytes);
  for (unsigned i = threadIdx.x; i < bins * Replicas; i += Threads) counts[i] = 0;
  __syncthreads();
  Local* private_counts = counts + ((threadIdx.x / 32) % Replicas) * bins;
  constexpr std::size_t tile_size = static_cast<std::size_t>(Threads) * Items;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * tile_size;
  std::size_t base = static_cast<std::size_t>(blockIdx.x) * tile_size;
  const bool vector_aligned = reinterpret_cast<std::uintptr_t>(input) % (4 * sizeof(Input)) == 0;
  unsigned previous = 0;
  Local run = 0;
  while (base < size) {
    const std::size_t remaining = size - base;
    if (remaining >= tile_size) {
      if constexpr (Load == LoadPolicy::vector4) {
        if (vector_aligned)
          accumulate_shared_tile<Input, Local, Threads, Items, Method, true, true>(
              input, base, remaining, private_counts, previous, run);
        else
          accumulate_shared_tile<Input, Local, Threads, Items, Method, true, false>(
              input, base, remaining, private_counts, previous, run);
      } else {
        accumulate_shared_tile<Input, Local, Threads, Items, Method, true, false>(
            input, base, remaining, private_counts, previous, run);
      }
    } else {
      accumulate_shared_tile<Input, Local, Threads, Items, Method, false, false>(
          input, base, remaining, private_counts, previous, run);
    }
    if (stride >= remaining) break;
    base += stride;
  }
  if constexpr (Method == Update::rle) {
    if (run != 0) add(private_counts + previous, run);
  }
  __syncthreads();
  for (unsigned bin = threadIdx.x; bin < bins; bin += Threads) {
    Local count = 0;
    #pragma unroll
    for (int replica = 0; replica < Replicas; ++replica)
      count += counts[static_cast<unsigned>(replica) * bins + bin];
    if constexpr (Partial)
      partials[static_cast<std::size_t>(blockIdx.x) * bins + bin] = count;
    else if (count != 0)
      add(output + bin, static_cast<Counter>(count));
  }
}
}

__device__ cudaError_t count(u64 size, Array<const std::byte> input,
    Array<std::byte> output, Workspace workspace, Status* status) {
  if (!status) return cudaErrorInvalidValue;
  if (size != (16ULL << 20) && size != (64ULL << 20)) { fail(status,shape); return finish(status); }
  constexpr u32 bins = 16384;
  if (!contains(input,size*sizeof(unsigned)) || !contains(output,u64(bins)*sizeof(Counter64)) ||
      reinterpret_cast<std::uintptr_t>(input.data) % alignof(unsigned) ||
      reinterpret_cast<std::uintptr_t>(output.data) % alignof(Counter64)) {
    fail(status,extent); return finish(status);
  }
  const Arena empty{workspace};
  if (!empty.fits(status)) return finish(status);
  auto* result = reinterpret_cast<Counter64*>(output.data);
  clear_histogram_output<<<64,256>>>(result,bins);
  auto error = cudaGetLastError();
  if (error == cudaSuccess) {
    shared_histogram_loaded<unsigned,Counter64,unsigned,512,8,1,Update::atomic,false,
        LoadPolicy::vector4,96*1024><<<48,512,bins*sizeof(unsigned)>>>(
          reinterpret_cast<const unsigned*>(input.data),size,bins,result,nullptr);
    error = cudaGetLastError();
  }
  if (error != cudaSuccess) { fail(status,runtime); finish(status); return error; }
  return finish(status);
}
cudaError_t initialize_runtime() {
  return cudaFuncSetAttribute(shared_histogram_loaded<unsigned,Counter64,unsigned,512,8,1,
    Update::atomic,false,LoadPolicy::vector4,96*1024>,cudaFuncAttributeMaxDynamicSharedMemorySize,96*1024);
}
}
