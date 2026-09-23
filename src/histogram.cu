#include <gh/histogram.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>

namespace gh {
namespace {

constexpr std::size_t shared_memory_limit = 48 * 1024;
using Counter64 = unsigned long long;
static_assert(sizeof(Counter64) == sizeof(std::uint64_t));

enum class Update { atomic, warp, rle };

std::size_t local_bytes(const Config& config) {
  return config.local_counter == LocalCounter::u32 ? sizeof(unsigned) : counter_bytes(config.counter_type);
}

std::size_t partial_bytes(const Config& config) {
  return static_cast<std::size_t>(config.blocks) * config.bins *
         local_bytes(config);
}

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

cudaError_t clear_custom_output(const Config& config, void* output, cudaStream_t stream) {
  constexpr unsigned threads = 256;
  const unsigned blocks = std::min((config.bins + threads - 1) / threads, 256u);
  // Keep initialization as a kernel node for the nonempty custom counting path.
  // Same-stream ordering completes every store before any histogram update.
  if (config.counter_type == CounterType::u32)
    clear_histogram_output<<<blocks, threads, 0, stream>>>(static_cast<unsigned*>(output), config.bins);
  else
    clear_histogram_output<<<blocks, threads, 0, stream>>>(static_cast<Counter64*>(output), config.bins);
  return cudaGetLastError();
}

// Every lane participates in the ballot. Only lanes named in its result enter
// match_any, including when the last tile ends in the middle of a warp.
template <typename Counter>
__device__ __forceinline__ void warp_add(Counter* counts, unsigned key, bool valid) {
  const unsigned live = __ballot_sync(0xffffffffu, valid);
  if (valid) {
    const unsigned peers = __match_any_sync(live, key);
    if ((threadIdx.x & 31u) == static_cast<unsigned>(__ffs(peers) - 1))
      add(counts + key, static_cast<Counter>(__popc(peers)));
  }
}

template <typename Input, typename Counter, int Threads, int Items, Update Method>
__global__ __launch_bounds__(Threads)
void global_histogram(const Input* input, std::size_t size, Counter* output) {
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
      if constexpr (Method == Update::warp) {
        warp_add(output, keys[item], valid);
      } else {
        if (valid) add(output + keys[item], Counter{1});
      }
    }
    if (stride >= size - base) break;
    base += stride;
  }
}

template <typename Input, typename Counter, typename Local, int Threads, int Items, int Replicas,
          Update Method, bool Partial>
__global__ __launch_bounds__(Threads)
void shared_histogram(const Input* input, std::size_t size, unsigned bins,
                      Counter* output, Local* partials) {
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  auto* counts = reinterpret_cast<Local*>(shared_bytes);
  for (unsigned i = threadIdx.x; i < bins * Replicas; i += Threads) counts[i] = 0;
  __syncthreads();

  // All lanes in a warp share a replica, so warp matching can combine their
  // increments. Replicas reduce interference between different warps.
  Local* private_counts = counts + ((threadIdx.x / 32) % Replicas) * bins;
  constexpr std::size_t tile_size = static_cast<std::size_t>(Threads) * Items;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * tile_size;
  std::size_t base = static_cast<std::size_t>(blockIdx.x) * tile_size;
  unsigned previous = 0;
  Local run = 0;
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
      if constexpr (Method == Update::warp) {
        warp_add(private_counts, keys[item], valid);
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
    if (stride >= size - base) break;
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
    if constexpr (Partial) {
      // Every block writes every bin, including zeros. Scratch needs no reset.
      partials[static_cast<std::size_t>(blockIdx.x) * bins + bin] = count;
    } else {
      if (count != 0) add(output + bin, static_cast<Counter>(count));
    }
  }
}

// Separate from the original scalar kernel so policies 0--5 remain a stable
// ablation. Full tiles have no per-item bounds checks or live-mask ballots.
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

template <typename Input, typename Counter, typename Local, int Threads, int Items, int Replicas,
          Update Method, bool Partial, LoadPolicy Load, std::size_t SharedLimit>
void launch_shared(const Config& config, const Input* input, Counter* output, Local* partials,
                   std::size_t shared_bytes, cudaStream_t stream) {
  if constexpr (Load == LoadPolicy::scalar)
    shared_histogram<Input, Counter, Local, Threads, Items, Replicas, Method, Partial>
        <<<config.blocks, Threads, shared_bytes, stream>>>(input, config.size, config.bins, output, partials);
  else
    shared_histogram_loaded<Input, Counter, Local, Threads, Items, Replicas, Method, Partial, Load,
                            SharedLimit>
        <<<config.blocks, Threads, shared_bytes, stream>>>(input, config.size, config.bins, output, partials);
}

template <typename Counter, typename Local>
__global__ void reduce_partials(const Local* partials, unsigned bins, int blocks,
                               Counter* output) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t bin = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       bin < bins; bin += stride) {
    Counter count = 0;
    for (int block = 0; block < blocks; ++block)
      count += partials[static_cast<std::size_t>(block) * bins + bin];
    output[bin] = count;
  }
}

template <typename Input, typename Counter, typename Local, int Threads, int Items, int Replicas,
          LoadPolicy Load, std::size_t SharedLimit>
cudaError_t launch_policy(const Config& config, const void* input, void* output,
                          void* workspace, cudaStream_t stream) {
  const auto* samples = static_cast<const Input*>(input);
  auto* counts = static_cast<Counter*>(output);
  const std::size_t shared_bytes = static_cast<std::size_t>(config.bins) * Replicas * sizeof(Local);
  switch (config.algorithm) {
    case Algorithm::global_atomic:
      if constexpr (Load == LoadPolicy::scalar)
        global_histogram<Input, Counter, Threads, Items, Update::atomic>
            <<<config.blocks, Threads, 0, stream>>>(samples, config.size, counts);
      else
        return cudaErrorInvalidValue;
      break;
    case Algorithm::warp_aggregated:
      if constexpr (Load == LoadPolicy::scalar)
        global_histogram<Input, Counter, Threads, Items, Update::warp>
            <<<config.blocks, Threads, 0, stream>>>(samples, config.size, counts);
      else
        return cudaErrorInvalidValue;
      break;
    case Algorithm::shared_atomic:
      launch_shared<Input, Counter, Local, Threads, Items, Replicas, Update::atomic, false, Load, SharedLimit>(
          config, samples, counts, nullptr, shared_bytes, stream);
      break;
    case Algorithm::shared_rle:
      launch_shared<Input, Counter, Local, Threads, Items, Replicas, Update::rle, false, Load, SharedLimit>(
          config, samples, counts, nullptr, shared_bytes, stream);
      break;
    case Algorithm::shared_warp:
      launch_shared<Input, Counter, Local, Threads, Items, Replicas, Update::warp, false, Load, SharedLimit>(
          config, samples, counts, nullptr, shared_bytes, stream);
      break;
    case Algorithm::shared_partial: {
      auto* partials = static_cast<Local*>(workspace);
      launch_shared<Input, Counter, Local, Threads, Items, Replicas, Update::atomic, true, Load, SharedLimit>(
          config, samples, counts, partials, shared_bytes, stream);
      const cudaError_t status = cudaGetLastError();
      if (status != cudaSuccess) return status;
      const unsigned merge_blocks = std::min((config.bins + 255u) / 256u,
                                             static_cast<unsigned>(config.blocks));
      reduce_partials<<<merge_blocks, 256, 0, stream>>>(partials, config.bins, config.blocks, counts);
      break;
    }
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

template <typename Input, typename Counter, typename Local, std::size_t Index = 0>
cudaError_t launch_tuning(const Config& config, const void* input, void* output,
                         void* workspace, cudaStream_t stream) {
  if (config.tuning == static_cast<int>(Index)) {
    constexpr auto tuning = tuning_catalog[Index];
    return launch_policy<Input, Counter, Local, tuning.threads, tuning.items, tuning.replicas,
                         tuning.load, tuning.shared_limit>(
        config, input, output, workspace, stream);
  }
  if constexpr (Index + 1 < sizeof(tuning_catalog) / sizeof(tuning_catalog[0]))
    return launch_tuning<Input, Counter, Local, Index + 1>(config, input, output, workspace, stream);
  return cudaErrorInvalidValue;
}

template <typename Input, typename Counter, typename Local, int Threads, int Items, int Replicas,
          Update Method, bool Partial, LoadPolicy Load, std::size_t SharedLimit>
cudaError_t prepare_shared() {
  return cudaFuncSetAttribute(
      shared_histogram_loaded<Input, Counter, Local, Threads, Items, Replicas, Method, Partial,
                              Load, SharedLimit>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(SharedLimit));
}

template <typename Input, typename Counter, typename Local, std::size_t Index = 0>
cudaError_t prepare_tuning(const Config& config) {
  if (config.tuning == static_cast<int>(Index)) {
    constexpr auto tuning = tuning_catalog[Index];
    if constexpr (tuning.shared_limit <= shared_memory_limit) {
      return cudaSuccess;
    } else {
      static_assert(tuning.load != LoadPolicy::scalar);
      switch (config.algorithm) {
        case Algorithm::shared_atomic:
          return prepare_shared<Input, Counter, Local, tuning.threads, tuning.items, tuning.replicas,
                                Update::atomic, false, tuning.load, tuning.shared_limit>();
        case Algorithm::shared_rle:
          return prepare_shared<Input, Counter, Local, tuning.threads, tuning.items, tuning.replicas,
                                Update::rle, false, tuning.load, tuning.shared_limit>();
        case Algorithm::shared_warp:
          return prepare_shared<Input, Counter, Local, tuning.threads, tuning.items, tuning.replicas,
                                Update::warp, false, tuning.load, tuning.shared_limit>();
        case Algorithm::shared_partial:
          return prepare_shared<Input, Counter, Local, tuning.threads, tuning.items, tuning.replicas,
                                Update::atomic, true, tuning.load, tuning.shared_limit>();
        default:
          return cudaErrorInvalidValue;
      }
    }
  }
  if constexpr (Index + 1 < tuning_count)
    return prepare_tuning<Input, Counter, Local, Index + 1>(config);
  return cudaErrorInvalidValue;
}

template <typename Function>
cudaError_t dispatch_types(const Config& config, Function function) {
  if (config.input_type == InputType::u8) {
    if (config.counter_type == CounterType::u32)
      return function.template operator()<unsigned char, unsigned>();
    return function.template operator()<unsigned char, Counter64>();
  }
  if (config.counter_type == CounterType::u32)
    return function.template operator()<unsigned, unsigned>();
  return function.template operator()<unsigned, Counter64>();
}

}  // namespace

cudaError_t prepare(Config& config) {
  if (config.algorithm != Algorithm::automatic)
    return prepare(static_cast<const Config&>(config));
  if (!supported(config)) return cudaErrorInvalidValue;
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) return status;
  Config resolved = default_config(config, properties);
  status = prepare(static_cast<const Config&>(resolved));
  if (status == cudaSuccess) config = resolved;
  return status;
}

cudaError_t prepare(const Config& config) {
  if (config.algorithm == Algorithm::automatic || !supported(config)) return cudaErrorInvalidValue;
  const auto policy = tuning_catalog[config.tuning];
  if (policy.shared_limit <= shared_memory_limit) return cudaSuccess;
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  int capacity = 0;
  status = cudaDeviceGetAttribute(&capacity, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (status != cudaSuccess) return status;
  if (capacity < 0 || static_cast<std::size_t>(capacity) < policy.shared_limit)
    return cudaErrorNotSupported;
  if (config.algorithm == Algorithm::shared_overflow)
    return detail::prepare_shared_overflow(config);
  return dispatch_types(config, [&]<typename Input, typename Counter>() {
    if (config.local_counter == LocalCounter::u32)
      return prepare_tuning<Input, Counter, unsigned>(config);
    return prepare_tuning<Input, Counter, Counter>(config);
  });
}

cudaError_t workspace_bytes(const Config& config, std::size_t& bytes) {
  bytes = 0;
  if (config.algorithm == Algorithm::automatic || !supported(config)) return cudaErrorInvalidValue;
  if (config.size == 0) return cudaSuccess;
  if (config.algorithm == Algorithm::global_window) {
    bytes = static_cast<std::size_t>(std::min(config.bins, config.window_bins)) * sizeof(unsigned);
    return cudaSuccess;
  }
  if (config.local_counter == LocalCounter::u32 &&
      (config.algorithm == Algorithm::global_atomic || config.algorithm == Algorithm::warp_aggregated)) {
    bytes = static_cast<std::size_t>(config.bins) * sizeof(unsigned);
    return cudaSuccess;
  }
  if (config.algorithm == Algorithm::shared_partial) {
    bytes = partial_bytes(config);
    return cudaSuccess;
  }
  return cudaSuccess;
}

cudaError_t histogram(const Config& config, const void* input, void* output,
                      void* workspace, std::size_t bytes, cudaStream_t stream) {
  if (config.algorithm == Algorithm::automatic || !supported(config) ||
      output == nullptr || (config.size != 0 && input == nullptr))
    return cudaErrorInvalidValue;
  if (config.size == 0)
    return cudaMemsetAsync(output, 0, static_cast<std::size_t>(config.bins) *
                          counter_bytes(config.counter_type), stream);
  if (config.algorithm == Algorithm::global_window) {
    const std::size_t required =
        static_cast<std::size_t>(std::min(config.bins, config.window_bins)) * sizeof(unsigned);
    if (workspace == nullptr || bytes < required) return cudaErrorInvalidValue;
    return detail::launch_global_window(config, input, output, workspace, stream);
  }
  if (config.local_counter == LocalCounter::u32 &&
      (config.algorithm == Algorithm::global_atomic || config.algorithm == Algorithm::warp_aggregated)) {
    if (workspace == nullptr || bytes < static_cast<std::size_t>(config.bins) * sizeof(unsigned))
      return cudaErrorInvalidValue;
    return detail::launch_global_narrow(config, input, output, workspace, stream);
  }
  if (config.algorithm == Algorithm::shared_partial) {
    if (workspace == nullptr || bytes < partial_bytes(config)) return cudaErrorInvalidValue;
  } else {
    const cudaError_t status = config.output_clear == OutputClear::kernel
        ? clear_custom_output(config, output, stream)
        : cudaMemsetAsync(output, 0, static_cast<std::size_t>(config.bins) *
                          counter_bytes(config.counter_type), stream);
    if (status != cudaSuccess) return status;
  }
  if (config.algorithm == Algorithm::shared_overflow)
    return detail::launch_shared_overflow(config, input, output, stream);
  if (config.algorithm == Algorithm::bitplane)
    return detail::launch_bitplane(config, input, output, stream);
  return dispatch_types(config, [&]<typename Input, typename Counter>() {
    if (config.local_counter == LocalCounter::u32)
      return launch_tuning<Input, Counter, unsigned>(config, input, output, workspace, stream);
    return launch_tuning<Input, Counter, Counter>(config, input, output, workspace, stream);
  });
}

}  // namespace gh
