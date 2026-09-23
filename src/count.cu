#include "gh/count.cuh"
#include <climits>

namespace gh::count {
namespace {
using Wide = unsigned long long;
static_assert(sizeof(Wide) == sizeof(u64));
__device__ bool shared(Algorithm a) {
  return a == Algorithm::shared_atomic || a == Algorithm::shared_rle ||
      a == Algorithm::shared_warp || a == Algorithm::shared_partial || a == Algorithm::shared_overflow;
}
__device__ unsigned input_width(Config c) { return c.input_type == InputType::u8 ? 1 : 4; }
__device__ unsigned output_width(Config c) { return c.counter_type == CounterType::u32 ? 4 : 8; }
__device__ unsigned local_width(Config c) { return c.local_counter == LocalCounter::u32 ? 4 : output_width(c); }
__device__ u64 block_bound(Config c) {
  const auto p = tuning(c.policy);
  const u64 tile = u64(p.threads) * p.items, stride = tile * c.blocks;
  return (c.size / stride) * tile + min(c.size % stride, tile);
}
__device__ u32 shared_size(Config c) {
  if (c.algorithm == Algorithm::shared_overflow) return 96 * 1024;
  if (c.algorithm == Algorithm::bitplane) {
    unsigned capacity = 1;
    while (capacity < c.bins) capacity *= 2;
    return (tuning(c.policy).threads / 32) * capacity * output_width(c);
  }
  return shared(c.algorithm) ? c.bins * tuning(c.policy).replicas * local_width(c) : 0;
}
__device__ bool fits(Config c, Hardware h) {
  if (!supported(c)) return false;
  const auto p = tuning(c.policy);
  return p.threads <= h.max_threads && c.blocks <= h.max_grid &&
      (!(shared(c.algorithm) || c.algorithm == Algorithm::bitplane) || (p.shared_limit > 48 * 1024
          ? h.shared_optin_bytes >= p.shared_limit : shared_size(c) <= h.shared_bytes));
}
__device__ unsigned bounded_blocks(Config c, Hardware h) {
  const auto p = tuning(c.policy);
  return unsigned(max(u64{1}, min(ceil_div(c.size, u64(p.threads) * p.items),
      min(u64(max(h.sms, 1u)) * 4, u64(h.max_grid)))));
}

template<class T> __device__ void increment(T* address, T value) { atomicAdd(address, value); }
template<class T> __global__ void zero(T* values, unsigned count) {
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += u64(gridDim.x) * blockDim.x)
    values[i] = 0;
}
template<class T> __device__ cudaError_t clear(T* values, unsigned count, OutputClear method) {
  if (method == OutputClear::runtime) return cudaMemsetAsync(values, 0, u64(count) * sizeof(T));
  zero<<<min(unsigned(ceil_div(count, 256)), 256u), 256>>>(values, count);
  return cudaGetLastError();
}
template<class Out, class Local> __global__ void combine(const Local* scratch, Out* output,
                                                       unsigned bins, unsigned rows) {
  for (u64 bin = u64(blockIdx.x) * blockDim.x + threadIdx.x; bin < bins; bin += u64(gridDim.x) * blockDim.x) {
    Out sum = 0;
    for (unsigned row = 0; row < rows; ++row) sum += scratch[u64(row) * bins + bin];
    output[bin] = sum;
  }
}
template<class T> __device__ void matched_increment(T* values, unsigned key, bool present) {
  const unsigned live = __ballot_sync(0xffffffffu, present);
  if (present) {
    const unsigned matches = __match_any_sync(live, key);
    if ((threadIdx.x & 31) == unsigned(__ffs(matches) - 1)) increment(values + key, T(__popc(matches)));
  }
}

template<unsigned Policy, class Input, class Output, class Local>
__global__ __launch_bounds__(tuning(Policy).threads)
void histogram(Config c, const Input* source, Output* output, Local* scratch,
               unsigned window_begin = 0, unsigned window_count = 0) {
  constexpr auto p = tuning(Policy);
  constexpr u64 tile = u64(p.threads) * p.items;
  extern __shared__ __align__(16) unsigned char storage[];
  auto* local = reinterpret_cast<Local*>(storage);
  const bool in_shared = c.algorithm == Algorithm::shared_atomic || c.algorithm == Algorithm::shared_rle ||
      c.algorithm == Algorithm::shared_warp || c.algorithm == Algorithm::shared_partial ||
      c.algorithm == Algorithm::shared_overflow;
  const bool overflow = c.algorithm == Algorithm::shared_overflow;
  const unsigned local_bins = overflow ? 24576 : c.bins;
  if (in_shared) {
    for (unsigned i = threadIdx.x; i < local_bins * p.replicas; i += p.threads) local[i] = 0;
    __syncthreads();
  }
  Local* destination = in_shared ? local + ((threadIdx.x / 32) % p.replicas) * local_bins : scratch;
  const bool warp = c.algorithm == Algorithm::warp_aggregated || c.algorithm == Algorithm::shared_warp;
  const bool rle = c.algorithm == Algorithm::shared_rle;
  const bool window = c.algorithm == Algorithm::global_window;
  const u64 stride = u64(c.blocks) * tile;
  unsigned previous = 0;
  Local run = 0;
  for (u64 start = u64(blockIdx.x) * tile; start < c.size;) {
    const u64 remaining = c.size - start;
    unsigned keys[p.items];
    const bool packed = p.load == LoadPolicy::vector4 && remaining >= tile &&
        reinterpret_cast<std::uintptr_t>(source) % (4 * sizeof(Input)) == 0;
    if constexpr (p.load == LoadPolicy::vector4) {
      if (packed) {
        #pragma unroll
        for (unsigned group = 0; group < p.items / 4; ++group) {
          const u64 position = u64(group) * p.threads + threadIdx.x;
          if constexpr (sizeof(Input) == 1) {
            const unsigned word = reinterpret_cast<const unsigned*>(source + start)[position];
            #pragma unroll
            for (unsigned part = 0; part < 4; ++part) keys[4 * group + part] = (word >> (8 * part)) & 255;
          } else {
            const uint4 word = reinterpret_cast<const uint4*>(source + start)[position];
            keys[4 * group] = word.x; keys[4 * group + 1] = word.y;
            keys[4 * group + 2] = word.z; keys[4 * group + 3] = word.w;
          }
        }
      }
    }
    if (!packed) {
      #pragma unroll
      for (unsigned item = 0; item < p.items; ++item) {
        const u64 offset = u64(item) * p.threads + threadIdx.x;
        keys[item] = p.load != LoadPolicy::scalar && remaining >= tile
            ? source[start + offset] : offset < remaining ? source[start + offset] : 0;
      }
    }
    #pragma unroll
    for (unsigned item = 0; item < p.items; ++item) {
      const bool valid = packed || u64(item) * p.threads + threadIdx.x < remaining;
      const unsigned key = keys[item];
      if (warp) matched_increment(destination, key, valid);
      else if (rle) {
        if (valid) {
          if (run && key != previous) { increment(destination + previous, run); run = 0; }
          previous = key; ++run;
        }
      } else if (valid) {
        if (overflow && key >= local_bins) increment(output + key, Output{1});
        else if (window) {
          const unsigned relative = key - window_begin;
          if (relative < window_count) increment(destination + relative, Local{1});
        } else increment(destination + key, Local{1});
      }
    }
    if (stride >= remaining) break;
    start += stride;
  }
  if (rle && run) increment(destination + previous, run);
  if (in_shared) {
    __syncthreads();
    for (unsigned bin = threadIdx.x; bin < local_bins; bin += p.threads) {
      Local total = 0;
      #pragma unroll
      for (unsigned replica = 0; replica < p.replicas; ++replica) total += local[replica * local_bins + bin];
      if (c.algorithm == Algorithm::shared_partial) scratch[u64(blockIdx.x) * c.bins + bin] = total;
      else if (total) increment(output + bin, Output(total));
    }
  }
}

template<unsigned Policy, class Input, class Output>
__global__ __launch_bounds__(tuning(Policy).threads)
void bitplanes(Config c, const Input* source, Output* output, unsigned capacity) {
  constexpr auto p = tuning(Policy);
  constexpr u64 tile = u64(p.threads) * p.items;
  constexpr unsigned warps = p.threads / 32;
  extern __shared__ __align__(16) unsigned char storage[];
  auto* partial = reinterpret_cast<Output*>(storage);
  const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  Output totals[8]{};
  const unsigned groups = max(capacity / 32, 1u);
  const unsigned bits = unsigned(__ffs(int(capacity)) - 1);
  const u64 stride = u64(c.blocks) * tile;
  for (u64 start = u64(blockIdx.x) * tile; start < c.size;) {
    const bool full = p.load != LoadPolicy::scalar && c.size - start >= tile;
    #pragma unroll
    for (unsigned item = 0; item < p.items; ++item) {
      const u64 index = start + u64(item) * p.threads + threadIdx.x;
      const bool valid = full || index < c.size;
      const unsigned key = valid ? source[index] : 0;
      const unsigned live = full ? 0xffffffffu : __ballot_sync(0xffffffffu, valid);
      unsigned planes[8];
      for (unsigned bit = 0; bit < bits; ++bit) planes[bit] = __ballot_sync(0xffffffffu, (key >> bit) & 1);
      for (unsigned group = 0; group < groups; ++group) {
        const unsigned bin = lane + 32 * group;
        unsigned matches = live;
        for (unsigned bit = 0; bit < bits; ++bit) matches &= ((bin >> bit) & 1) ? planes[bit] : ~planes[bit];
        if (bin < capacity) totals[group] += Output(__popc(matches));
      }
    }
    if (stride >= c.size - start) break;
    start += stride;
  }
  for (unsigned group = 0; group < groups; ++group) {
    const unsigned bin = lane + 32 * group;
    if (bin < capacity) partial[warp * capacity + bin] = totals[group];
  }
  __syncthreads();
  for (unsigned bin = threadIdx.x; bin < c.bins; bin += p.threads) {
    Output total = 0;
    #pragma unroll
    for (unsigned w = 0; w < warps; ++w) total += partial[w * capacity + bin];
    if (total) increment(output + bin, total);
  }
}

template<unsigned Policy, class Input, class Output, class Local>
__device__ cudaError_t execute(Config c, const Input* input, Output* output, Local* scratch) {
  constexpr auto p = tuning(Policy);
  if (c.size == 0) return clear(output, c.bins, OutputClear::runtime);
  const bool partial = c.algorithm == Algorithm::shared_partial;
  const bool narrow_global = !shared(c.algorithm) && c.local_counter == LocalCounter::u32;
  if (c.algorithm == Algorithm::global_window) {
    for (unsigned begin = 0; begin < c.bins;) {
      const unsigned width = min(c.window_bins, c.bins - begin);
      auto error = clear(scratch, width, c.output_clear);
      if (error != cudaSuccess) return error;
      histogram<Policy><<<c.blocks,p.threads>>>(c, input, output, scratch, begin, width);
      error = cudaGetLastError();
      if (error != cudaSuccess) return error;
      combine<<<min(unsigned(ceil_div(width,256)),256u),256>>>(scratch,output + begin,width,1);
      error = cudaGetLastError();
      if (error != cudaSuccess) return error;
      begin += width;
    }
    return cudaSuccess;
  }
  if (!partial) {
    const auto error = narrow_global ? clear(scratch,c.bins,c.output_clear) : clear(output,c.bins,c.output_clear);
    if (error != cudaSuccess) return error;
  }
  if (c.algorithm == Algorithm::bitplane) {
    if constexpr (p.shared_limit <= 48 * 1024) {
      const unsigned bytes = shared_size(c);
      const unsigned capacity = bytes / (p.threads / 32) / sizeof(Output);
      bitplanes<Policy><<<c.blocks,p.threads,bytes>>>(c,input,output,capacity);
    }
  } else {
    Local* counters = shared(c.algorithm) || narrow_global ? scratch : reinterpret_cast<Local*>(output);
    histogram<Policy><<<c.blocks,p.threads,shared_size(c)>>>(c,input,output,counters);
  }
  const auto error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  if (partial || narrow_global) {
    combine<<<min(unsigned(ceil_div(c.bins,256)),256u),256>>>(scratch,output,c.bins,partial ? c.blocks : 1);
    return cudaGetLastError();
  }
  return cudaSuccess;
}
template<class Input, class Output, class Local>
__device__ cudaError_t dispatch(Config c, const void* input, void* output, void* scratch) {
  #define GH_COUNT_CASE(P) case P: return execute<P>(c,static_cast<const Input*>(input),static_cast<Output*>(output),static_cast<Local*>(scratch))
  switch (c.policy) {
    GH_COUNT_CASE(0); GH_COUNT_CASE(1); GH_COUNT_CASE(2); GH_COUNT_CASE(3);
    GH_COUNT_CASE(4); GH_COUNT_CASE(5); GH_COUNT_CASE(6); GH_COUNT_CASE(7);
    GH_COUNT_CASE(8); GH_COUNT_CASE(9); GH_COUNT_CASE(10); GH_COUNT_CASE(11);
    GH_COUNT_CASE(12); GH_COUNT_CASE(13); GH_COUNT_CASE(14); GH_COUNT_CASE(15);
  }
  #undef GH_COUNT_CASE
  return cudaErrorInvalidValue;
}
template<class Input> __device__ cudaError_t dispatch_types(Config c, const void* input, void* output, void* scratch) {
  if (c.counter_type == CounterType::u32) return dispatch<Input,unsigned,unsigned>(c,input,output,scratch);
  return c.local_counter == LocalCounter::u32
      ? dispatch<Input,Wide,unsigned>(c,input,output,scratch)
      : dispatch<Input,Wide,Wide>(c,input,output,scratch);
}
} // namespace

__device__ bool supported(Config c) {
  if (unsigned(c.input_type) > 1 || unsigned(c.counter_type) > 1 || unsigned(c.local_counter) > 1 ||
      unsigned(c.output_clear) > 1 || unsigned(c.launch) > 1 || unsigned(c.cache) > 1 ||
      unsigned(c.algorithm) > unsigned(Algorithm::global_window) || !c.bins || c.bins >= INT_MAX ||
      (c.input_type == InputType::u8 && c.bins > 256) ||
      (c.counter_type == CounterType::u32 && c.size > UINT_MAX) ||
      c.size > u64(PTRDIFF_MAX) / input_width(c) || !c.blocks || c.blocks > INT_MAX || c.policy >= tuning_count)
    return false;
  const auto p = tuning(c.algorithm == Algorithm::automatic ? 2 : c.policy);
  if ((p.shared_limit > 48 * 1024 && !shared(c.algorithm)) ||
      (p.load != LoadPolicy::scalar && !shared(c.algorithm) && c.algorithm != Algorithm::bitplane)) return false;
  if (c.algorithm != Algorithm::automatic && c.local_counter == LocalCounter::u32) {
    if (c.counter_type != CounterType::u64) return false;
    if (shared(c.algorithm)) { if (block_bound(c) > UINT_MAX) return false; }
    else if (c.algorithm == Algorithm::global_atomic || c.algorithm == Algorithm::warp_aggregated ||
             c.algorithm == Algorithm::global_window) { if (c.size > UINT_MAX) return false; }
    else return false;
  }
  switch (c.algorithm) {
    case Algorithm::shared_atomic: case Algorithm::shared_rle:
    case Algorithm::shared_warp: case Algorithm::shared_partial:
      return u64(c.bins) * p.replicas * local_width(c) <= p.shared_limit &&
          (c.algorithm != Algorithm::shared_partial || u64(c.blocks) <= u64(PTRDIFF_MAX) / c.bins / local_width(c));
    case Algorithm::shared_overflow:
      return c.input_type == InputType::u32 && c.counter_type == CounterType::u64 &&
          c.local_counter == LocalCounter::u32 && c.bins > 24576 && c.policy >= 14;
    case Algorithm::global_window:
      return c.input_type == InputType::u32 && c.counter_type == CounterType::u64 &&
          c.local_counter == LocalCounter::u32 && c.policy < 6 && c.window_bins && c.window_bins < INT_MAX;
    case Algorithm::bitplane: {
      if (c.bins > 256) return false;
      unsigned capacity = 1;
      while (capacity < c.bins) capacity *= 2;
      return u64(p.threads / 32) * capacity * output_width(c) <= 48 * 1024;
    }
    default: return true;
  }
}

__device__ Config resolve(Config c, Hardware h) {
  if (c.algorithm != Algorithm::automatic) return c;
  Config fallback = c;
  fallback.algorithm = Algorithm::global_atomic;
  fallback.policy = h.max_threads < 256 ? 0 : 2;
  fallback.blocks = bounded_blocks(fallback,h);
  fallback.local_counter = LocalCounter::native;
  fallback.output_clear = c.launch == LaunchMode::graph ? OutputClear::kernel : OutputClear::runtime;
  if (c.size && supported(fallback)) {
    bool chosen = false;
    for (unsigned local = 0; local < 2 && !chosen; ++local) {
      if (local == 0 && c.counter_type != CounterType::u64) continue;
      for (unsigned choice = 0; choice < 2; ++choice) {
        Config trial = fallback;
        trial.algorithm = Algorithm::shared_atomic;
        trial.policy = choice ? 14 : (fallback.policy == 0 ? 0 : 7);
        trial.blocks = bounded_blocks(trial,h);
        trial.local_counter = local == 0 ? LocalCounter::u32 : LocalCounter::native;
        if (fits(trial,h)) { fallback = trial; chosen = true; break; }
      }
    }
  }
  if (!h.a5000_laptop || h.major != 8 || h.minor != 6 || h.sms != 48) return fallback;
  struct Key { u64 size; unsigned bins; InputType input; CounterType counter;
               LaunchMode launch; CacheMode cache; unsigned policy, blocks; LocalCounter local; };
  constexpr Key keys[] = {
    {1u<<20,4096,InputType::u32,CounterType::u32,LaunchMode::stream,CacheMode::warm,10,48,LocalCounter::native},
    {4096,256,InputType::u8,CounterType::u32,LaunchMode::graph,CacheMode::warm,1,96,LocalCounter::native},
    {1u<<20,256,InputType::u8,CounterType::u32,LaunchMode::graph,CacheMode::warm,11,48,LocalCounter::native},
    {1u<<24,256,InputType::u8,CounterType::u32,LaunchMode::graph,CacheMode::warm,10,96,LocalCounter::native},
    {1u<<20,8,InputType::u32,CounterType::u32,LaunchMode::graph,CacheMode::warm,6,192,LocalCounter::native},
    {1u<<20,256,InputType::u32,CounterType::u32,LaunchMode::graph,CacheMode::warm,11,48,LocalCounter::native},
    {1u<<24,4096,InputType::u32,CounterType::u32,LaunchMode::graph,CacheMode::warm,11,48,LocalCounter::native},
    {1u<<24,4096,InputType::u32,CounterType::u64,LaunchMode::graph,CacheMode::warm,10,48,LocalCounter::u32},
    {1u<<24,8192,InputType::u32,CounterType::u64,LaunchMode::graph,CacheMode::warm,15,48,LocalCounter::u32},
    {1u<<24,16384,InputType::u32,CounterType::u64,LaunchMode::graph,CacheMode::warm,15,48,LocalCounter::u32},
    {1u<<20,4096,InputType::u32,CounterType::u64,LaunchMode::graph,CacheMode::cold,10,48,LocalCounter::u32}};
  for (const auto key : keys) {
    if (c.size != key.size || c.bins != key.bins || c.input_type != key.input ||
        c.counter_type != key.counter || c.launch != key.launch || c.cache != key.cache) continue;
    Config selected = fallback;
    selected.algorithm = Algorithm::shared_atomic; selected.policy = key.policy;
    selected.blocks = key.blocks; selected.local_counter = key.local;
    return fits(selected,h) ? selected : fallback;
  }
  return fallback;
}

__device__ u64 required_bytes(Config c) {
  if (!c.size) return 0;
  if (c.algorithm == Algorithm::shared_partial) return u64(c.blocks) * c.bins * local_width(c);
  if (c.algorithm == Algorithm::global_window) return u64(min(c.bins,c.window_bins)) * 4;
  return !shared(c.algorithm) && c.local_counter == LocalCounter::u32 ? u64(c.bins) * 4 : 0;
}

__device__ cudaError_t count(Config request, Array<const std::byte> input,
    Array<std::byte> output, Workspace workspace, Status* status, Hardware hardware) {
  if (!status) return cudaErrorInvalidValue;
  if (!supported(request)) { fail(status,shape); return finish(status); }
  const Config c = resolve(request,hardware);
  if (!fits(c,hardware)) { fail(status,unsupported); return finish(status); }
  const u64 input_bytes = c.size * input_width(c), output_bytes = u64(c.bins) * output_width(c);
  if (!contains(input,input_bytes) || !contains(output,output_bytes) ||
      (c.size && reinterpret_cast<std::uintptr_t>(input.data) % input_width(c)) ||
      reinterpret_cast<std::uintptr_t>(output.data) % output_width(c)) {
    fail(status,extent); return finish(status);
  }
  Arena arena{workspace};
  void* scratch = arena.take<std::byte>(required_bytes(c));
  if (!arena.fits(status)) return finish(status);
  const auto error = c.input_type == InputType::u8
      ? dispatch_types<unsigned char>(c,input.data,output.data,scratch)
      : dispatch_types<unsigned>(c,input.data,output.data,scratch);
  if (error != cudaSuccess) { fail(status,runtime); finish(status); return error; }
  return finish(status);
}

namespace {
template<unsigned Policy, class Input, class Output, class Local>
cudaError_t optin() {
  return cudaFuncSetAttribute(histogram<Policy,Input,Output,Local>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,96 * 1024);
}
template<unsigned Policy> cudaError_t optin_types() {
  #define GH_OPTIN(I,O,L) do { const auto e = optin<Policy,I,O,L>(); if (e != cudaSuccess) return e; } while(false)
  GH_OPTIN(unsigned char,unsigned,unsigned); GH_OPTIN(unsigned,unsigned,unsigned);
  GH_OPTIN(unsigned char,Wide,unsigned); GH_OPTIN(unsigned,Wide,unsigned);
  GH_OPTIN(unsigned char,Wide,Wide); GH_OPTIN(unsigned,Wide,Wide);
  #undef GH_OPTIN
  return cudaSuccess;
}
}
cudaError_t initialize_runtime() {
  const auto error = optin_types<14>();
  return error == cudaSuccess ? optin_types<15>() : error;
}
} // namespace gh::count
