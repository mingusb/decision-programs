// GH_SOURCE_CATEGORY: production
#include "gpu_histogram.hpp"

// GH_SOURCE_CATEGORY: production
// These unchanged data leaf bodies compile once for both fitting and direct
// sanitizer checks. Buffers, shapes, integer schedules and synchronization keep
// their existing contracts. This object contains no coordinator or child launch;
// the failed full-object initcheck receipt motivates the separate link boundary.
// Verify linked symbols and repeat fitting, leaf checks and sanitizer activity.
#if GH_DATA_LEAF_IMPLEMENTATION
namespace gh::data_impl {
struct Prefix { u32 before, total; };
// Fixed 256-thread integer scan; every lane participates, including zero tails.
__device__ Prefix block_prefix(u32 value, u32* warp_totals) {
  const u32 lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  u32 sum = value;
  for (u32 step = 1; step < 32; step *= 2) {
    const u32 other = __shfl_up_sync(missing, sum, step);
    if (lane >= step) sum += other;
  }
  if (lane == 31) warp_totals[warp] = sum;
  __syncthreads();
  if (!warp) {
    u32 all = lane < 8 ? warp_totals[lane] : 0;
    for (u32 step = 1; step < 32; step *= 2) {
      const u32 other = __shfl_up_sync(missing, all, step);
      if (lane >= step) all += other;
    }
    if (lane < 8) warp_totals[lane] = all;
  }
  __syncthreads();
  const Prefix result{sum - value + (warp ? warp_totals[warp - 1] : 0), warp_totals[7]};
  __syncthreads();
  return result;
}

__global__ void scan_tiles(u32* values, u32 length, u32 chunks, u32* totals) {
  __shared__ u32 warps[8];
  const u32 feature = blockIdx.x / chunks, chunk = blockIdx.x % chunks;
  const u64 index = u64(chunk) * threads + threadIdx.x, base = u64(feature) * length;
  const auto prefix = block_prefix(index < length ? values[base + index] : 0, warps);
  if (index < length) values[base + index] = prefix.before;
  if (!threadIdx.x && totals) totals[u64(feature) * chunks + chunk] = prefix.total;
}
__global__ void scan_offsets(u32* values, const u32* totals, u32 length, u32 chunks, u64 cells) {
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < cells; i += u64(gridDim.x) * threads)
    values[i] += totals[(i / length) * chunks + (i % length) / threads];
}
template<u32 Bits>
__global__ void radix_counts(const u32* keys, u32* counts, u32 rows, u32 blocks, u32 shift) {
  constexpr u32 radix = 1u << Bits;
  __shared__ u32 local[radix];
  const u32 lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const u32 feature = blockIdx.x / blocks, block = blockIdx.x % blocks;
  for (u32 d = threadIdx.x; d < radix; d += threads) local[d] = 0;
  __syncthreads();
  for (u32 item = 0; item < 4; ++item) {
    const u64 row = u64(block) * tile_keys + warp * 128 + item * 32 + lane;
    const bool live = row < rows;
    const u32 digit = ((live ? keys[u64(feature) * rows + row] : 0) >> shift) & (radix - 1);
    const u32 active = __ballot_sync(missing, live);
    const u32 peers = __match_any_sync(missing, digit) & active;
    if (live && lane == u32(__ffs(peers) - 1)) atomicAdd(local + digit, u32(__popc(peers)));
  }
  __syncthreads();
  for (u32 d = threadIdx.x; d < radix; d += threads)
    counts[(u64(feature) * radix + d) * blocks + block] = local[d];
}

template<u32 Bits>
__global__ void radix_move(const u32* source, u32* destination, const u32* prefixes,
                           u32 rows, u32 blocks, u32 shift) {
  constexpr u32 radix = 1u << Bits;
  __shared__ u32 warp_counts[8 * radix], bases[radix], ordered[tile_keys], totals[8];
  const u32 lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const u32 feature = blockIdx.x / blocks, block = blockIdx.x % blocks;
  const u64 start = u64(block) * tile_keys;
  u32 keys[4], ranks[4];
  for (u32 i = threadIdx.x; i < 8 * radix; i += threads) warp_counts[i] = 0;
  __syncthreads();
  for (u32 item = 0; item < 4; ++item) {
    const u64 row = start + warp * 128 + item * 32 + lane;
    const bool live = row < rows;
    keys[item] = live ? source[u64(feature) * rows + row] : 0;
    const u32 digit = (keys[item] >> shift) & (radix - 1);
    const u32 active = __ballot_sync(missing, live);
    const u32 peers = __match_any_sync(missing, digit) & active;
    ranks[item] = warp_counts[warp * radix + digit] + __popc(peers & ((1u << lane) - 1));
    __syncwarp(); // All readers precede the elected writer for this digit.
    if (live && lane == u32(__ffs(peers) - 1)) warp_counts[warp * radix + digit] += __popc(peers);
    __syncwarp();
  }
  __syncthreads();
  u32 count = 0;
  if (threadIdx.x < radix)
    for (u32 w = 0; w < 8; ++w) count += warp_counts[w * radix + threadIdx.x];
  const auto prefix = block_prefix(count, totals);
  if (threadIdx.x < radix) bases[threadIdx.x] = prefix.before;
  __syncthreads();
  for (u32 item = 0; item < 4; ++item) {
    const u32 digit = (keys[item] >> shift) & (radix - 1);
    u32 rank = ranks[item];
    for (u32 w = 0; w < warp; ++w) rank += warp_counts[w * radix + digit];
    if (start + warp * 128 + item * 32 + lane < rows) ordered[bases[digit] + rank] = keys[item];
  }
  __syncthreads();
  for (u32 i = threadIdx.x; i < min(u64(tile_keys), u64(rows) - start); i += threads) {
    const u32 key = ordered[i], digit = (key >> shift) & (radix - 1);
    const u32 target = prefixes[(u64(feature) * radix + digit) * blocks + block] + i - bases[digit];
    destination[u64(feature) * rows + target] = key;
  }
}

__device__ bool first_key(const u32* keys, u64 row, u32 rows) {
  return row < rows && keys[row] != missing && (!row || keys[row] != keys[row - 1]);
}
template<bool Compact>
__global__ void unique_keys(const u32* keys, u32* prefixes, u32* values, u32 rows, u32 blocks) {
  __shared__ u32 totals[8];
  const u32 feature = blockIdx.x / blocks, block = blockIdx.x % blocks;
  const u64 row = u64(block) * threads + threadIdx.x;
  const auto* column = keys + u64(feature) * rows;
  const bool first = first_key(column, row, rows);
  const auto prefix = block_prefix(first, totals);
  const u64 base = u64(feature) * (blocks + 1);
  if constexpr (Compact) {
    if (first) values[u64(feature) * rows + prefixes[base + block] + prefix.before] = column[row];
  } else if (!threadIdx.x) {
    prefixes[base + block] = prefix.total;
    if (!block) prefixes[base + blocks] = 0;
  }
}

template __global__ void radix_counts<4>(const u32*,u32*,u32,u32,u32);
template __global__ void radix_counts<8>(const u32*,u32*,u32,u32,u32);
template __global__ void radix_move<4>(const u32*,u32*,const u32*,u32,u32,u32);
template __global__ void radix_move<8>(const u32*,u32*,const u32*,u32,u32,u32);
template __global__ void unique_keys<false>(const u32*,u32*,u32*,u32,u32);
template __global__ void unique_keys<true>(const u32*,u32*,u32*,u32,u32);
}
#endif // GH_DATA_LEAF_IMPLEMENTATION

#if GH_IMPLEMENTATION

// GH_SOURCE_CATEGORY: production
// core
namespace gh {
__global__ void complete(Status* status) { if (!threadIdx.x && !blockIdx.x) status->done = 1; }
__device__ cudaError_t finish(Status* status) {
  const auto prior = cudaGetLastError();
  if (prior != cudaSuccess) fail(status, runtime);
  complete<<<1, 1, 0, cudaStreamTailLaunch>>>(status);
  const auto next = cudaGetLastError();
  if (next != cudaSuccess) fail(status, runtime);
  return prior != cudaSuccess ? prior : next;
}
}

// GH_SOURCE_CATEGORY: production
// count
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

// GH_SOURCE_CATEGORY: production
// data
namespace gh {
namespace data_impl {
__device__ u32 grid(u64 n) { return u32(min(ceil_div(n, threads), u64(65535))); }
__device__ bool finite(float x) { return (__float_as_uint(x) & 0x7fffffffu) < 0x7f800000u; }
__device__ bool matrix(Dataset d, bool nonempty) {
  const u64 cells = u64(d.rows) * d.columns;
  return (!nonempty || d.rows) && d.columns && d.columns <= INT32_MAX &&
    mul_fits(cells, sizeof(float)) && contains(d.values, cells);
}
__device__ cudaError_t observed(Status* s) {
  const auto e = cudaGetLastError();
  if (e != cudaSuccess) fail(s, runtime);
  return e;
}
__device__ cudaError_t close(Status* s, cudaError_t e = cudaSuccess) {
  const auto last = finish(s);
  return e == cudaSuccess ? last : e;
}

struct Scratch {
  Arena arena;
  u32 *a, *b, *histogram, *unique, *scan;
  u32 blocks, unique_blocks;
};
__device__ Scratch scratch(Workspace w, u32 rows, u32 features, u32 radix) {
  Arena a{w};
  const u32 blocks = u32(ceil_div(rows, tile_keys));
  const u32 unique_blocks = u32(ceil_div(rows, threads));
  auto* first = a.take<u32>(u64(rows) * features);
  auto* second = a.take<u32>(u64(rows) * features);
  auto* histogram = a.take<u32>(u64(blocks) * radix * features);
  auto* unique = a.take<u32>(u64(unique_blocks + 1) * features);
  u64 scan_words = 0;
  for (u32 n = blocks * radix; n > threads;) {
    n = u32(ceil_div(n, threads));
    scan_words += u64(n) * features;
  }
  auto* scan = a.take<u32>(scan_words);
  return {a, first, second, histogram, unique, scan, blocks, unique_blocks};
}

__global__ void make_keys(Dataset d, u32 first, u32 count, u32* keys, Status* s) {
  __shared__ u32 tile[32][33];
  for (u64 row_base = u64(blockIdx.x) * 32; row_base < d.rows; row_base += u64(gridDim.x) * 32) {
    for (u32 y = threadIdx.y; y < 32; y += 8) {
      const u64 row = row_base + y;
      u32 key = missing;
      if (row < d.rows && threadIdx.x < count) {
        u32 bits = __float_as_uint(d.values.data[row * d.columns + first + threadIdx.x]);
        const u32 magnitude = bits & 0x7fffffffu;
        if (magnitude < 0x7f800000u) {
          if (!magnitude) bits = 0;
          key = bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
        } else if (magnitude == 0x7f800000u) fail(s, input);
      }
      tile[y][threadIdx.x] = key;
    }
    __syncthreads();
    const u64 row = row_base + threadIdx.x;
    for (u32 f = threadIdx.y; f < count; f += 8)
      if (row < d.rows) keys[u64(f) * d.rows + row] = tile[threadIdx.x][f];
    __syncthreads();
  }
}

__device__ cudaError_t scan(u32* values, u32 length, u32 features, u32* workspace, Status* s) {
  u32* levels[5]{values};
  u32 lengths[5]{length};
  u32 depth = 0;
  while (true) {
    const u32 chunks = u32(ceil_div(lengths[depth], threads));
    u32* next = chunks > 1 ? workspace : nullptr;
    scan_tiles<<<features * chunks, threads>>>(levels[depth], lengths[depth], chunks, next);
    if (const auto e = observed(s); e != cudaSuccess) return e;
    if (chunks == 1) break;
    levels[++depth] = next;
    lengths[depth] = chunks;
    workspace += u64(chunks) * features;
  }
  while (depth) {
    --depth;
    const u64 cells = u64(lengths[depth]) * features;
    scan_offsets<<<grid(cells), threads>>>(levels[depth], levels[depth + 1],
      lengths[depth], lengths[depth + 1], cells);
    if (const auto e = observed(s); e != cudaSuccess) return e;
  }
  return cudaSuccess;
}

__global__ void plan_metadata(Schema* schema, Array<const FeatureType> types, const u32* unique,
                              u32 blocks, u32 first, u32 count, u32 max_bins, Status* s) {
  if (s->errors) return;
  u64 metadata = schema->metadata_count, bins = schema->total_bins;
  for (u32 f = 0; f < count; ++f) {
    const auto type = types.size ? types.data[first + f] : FeatureType::numeric;
    const u32 distinct = unique[u64(f) * (blocks + 1) + blocks];
    if (type == FeatureType::categorical && distinct >= max_bins) { fail(s, capacity); return; }
    const u32 intervals = min(distinct, max_bins - 1);
    const u32 n = type == FeatureType::categorical ? distinct : intervals ? intervals - 1 : 0;
    metadata += n;
    bins += n + (type == FeatureType::numeric ? 2 : 1);
  }
  if (bins > UINT32_MAX) { fail(s, extent); return; }
  if (!contains(schema->metadata, metadata)) { fail(s, capacity); return; }
  for (u32 f = 0; f < count; ++f) {
    const auto type = types.size ? types.data[first + f] : FeatureType::numeric;
    const u32 distinct = unique[u64(f) * (blocks + 1) + blocks];
    const u32 intervals = min(distinct, max_bins - 1);
    const u32 n = type == FeatureType::categorical ? distinct : intervals ? intervals - 1 : 0;
    schema->features.data[first + f] = {schema->metadata_count, n, type};
    schema->metadata_count += n;
    const u32 feature_bins = n + (type == FeatureType::numeric ? 2 : 1);
    schema->total_bins += feature_bins;
    schema->offsets.data[first + f + 1] = schema->total_bins;
    schema->max_feature_bins = max(schema->max_feature_bins, feature_bins);
  }
}
__global__ void select_metadata(const u32* distinct, const u32* prefixes, u32 rows, u32 blocks,
                                Schema* schema, u32 first, u32 count, u32 max_bins, Status* s) {
  if (s->errors) return;
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < u64(count) * (max_bins - 1);
       i += u64(gridDim.x) * threads) {
    const u32 f = u32(i / (max_bins - 1)), k = u32(i % (max_bins - 1));
    const auto feature = schema->features.data[first + f];
    if (k >= feature.count) continue;
    const u32 total = prefixes[u64(f) * (blocks + 1) + blocks];
    const u32 rank = feature.type == FeatureType::categorical ? k : u32(u64(k + 1) * total / (feature.count + 1) - 1);
    const u32 key = distinct[u64(f) * rows + rank];
    schema->metadata.data[feature.begin + k] = __uint_as_float(key & 0x80000000u ? key ^ 0x80000000u : ~key);
  }
}

__global__ void validate_schema(const Schema* schema, u32 columns, Status* s) {
  const auto v = *schema;
  if (v.columns != columns || !contains(v.features, columns) ||
      !contains(v.offsets, u64(columns) + 1) || !contains(v.metadata, v.metadata_count) ||
      !v.max_feature_bins || v.max_feature_bins > 65536) { fail(s, shape); return; }
  for (u64 f = u64(blockIdx.x) * threads + threadIdx.x; f < columns; f += u64(gridDim.x) * threads) {
    const auto feature = v.features.data[f];
    const bool number = feature.type == FeatureType::numeric;
    if ((!number && feature.type != FeatureType::categorical) ||
        feature.count > (number ? 65534u : 65535u) || feature.begin > v.metadata_count ||
        feature.count > v.metadata_count - feature.begin) { fail(s, model); continue; }
    const u32 bins = feature.count + (number ? 2 : 1);
    if (v.offsets.data[f] > UINT32_MAX - bins || v.offsets.data[f + 1] != v.offsets.data[f] + bins ||
        bins > v.max_feature_bins || (!f && (v.offsets.data[0] || feature.begin)) ||
        (f && (v.features.data[f - 1].begin > feature.begin ||
          v.features.data[f - 1].count != feature.begin - v.features.data[f - 1].begin)) ||
        (f + 1 == columns && (v.offsets.data[f + 1] != v.total_bins || feature.begin + feature.count != v.metadata_count)))
      fail(s, model);
    for (u32 k = 0; k < feature.count; ++k) {
      const float x = v.metadata.data[feature.begin + k];
      if (!finite(x) || (k && !(v.metadata.data[feature.begin + k - 1] < x))) fail(s, model);
    }
  }
}
__global__ void encode_rows(Dataset d, const Schema* schema, std::uint16_t* bins, Status* s) {
  __shared__ float tile[32][33];
  __shared__ bool active;
  if (!threadIdx.x && !threadIdx.y) active = !s->errors;
  __syncthreads();
  if (!active) return;
  const u64 row_tiles = ceil_div(d.rows, 32), tasks = row_tiles * ceil_div(d.columns, 32);
  for (u64 task = blockIdx.x; task < tasks; task += gridDim.x) {
    const u64 row_base = (task % row_tiles) * 32;
    const u32 first = u32(task / row_tiles) * 32;
    for (u32 y = threadIdx.y; y < 32; y += 8)
      if (row_base + y < d.rows && first + threadIdx.x < d.columns)
        tile[y][threadIdx.x] = d.values.data[(row_base + y) * d.columns + first + threadIdx.x];
    __syncthreads();
    const u64 row = row_base + threadIdx.x;
    for (u32 f = threadIdx.y; f < 32 && first + f < d.columns; f += 8) {
      if (row >= d.rows) continue;
      const auto feature = schema->features.data[first + f];
      const float x = tile[threadIdx.x][f];
      const u32 magnitude = __float_as_uint(x) & 0x7fffffffu;
      u32 bin = 0;
      if (magnitude < 0x7f800000u) {
        u32 lo = 0, hi = feature.count;
        while (lo < hi) {
          const u32 mid = lo + (hi - lo) / 2;
          if (schema->metadata.data[feature.begin + mid] < x) lo = mid + 1; else hi = mid;
        }
        if (feature.type == FeatureType::numeric ||
            (lo < feature.count && schema->metadata.data[feature.begin + lo] == x)) bin = lo + 1;
      } else if (magnitude == 0x7f800000u) fail(s, input);
      bins[u64(first + f) * d.rows + row] = std::uint16_t(bin);
    }
    __syncthreads();
  }
}
__device__ cudaError_t submit_encoding(Dataset d, const Schema* schema, std::uint16_t* bins, Status* s) {
  if (!d.rows) return cudaSuccess;
  const u64 tiles = ceil_div(d.rows, 32) * ceil_div(d.columns, 32);
  encode_rows<<<u32(min(tiles, u64(65535))), dim3(32, 8)>>>(d, schema, bins, s);
  return observed(s);
}

template<u32 Bits>
__device__ cudaError_t submit_fit(Dataset d, Array<const FeatureType> types, u32 max_bins,
    Schema* schema, std::uint16_t* bins, Scratch w, u32 capacity, Status* s) {
  for (u32 first = 0; first < d.columns;) {
    const u32 count = min(capacity, d.columns - first);
    make_keys<<<u32(min(ceil_div(d.rows, 32), u64(65535))), dim3(32, 8)>>>(d, first, count, w.a, s);
    if (const auto e = observed(s); e != cudaSuccess) return e;
    u32* source = w.a;
    u32* target = w.b;
    for (u32 shift = 0; shift < 32; shift += Bits) {
      radix_counts<Bits><<<count * w.blocks, threads>>>(source, w.histogram, d.rows, w.blocks, shift);
      if (const auto e = observed(s); e != cudaSuccess) return e;
      if (const auto e = scan(w.histogram, w.blocks * (1u << Bits), count, w.scan, s); e != cudaSuccess) return e;
      radix_move<Bits><<<count * w.blocks, threads>>>(source, target, w.histogram, d.rows, w.blocks, shift);
      if (const auto e = observed(s); e != cudaSuccess) return e;
      auto* previous = source; source = target; target = previous;
    }
    unique_keys<false><<<count * w.unique_blocks, threads>>>(source, w.unique, target, d.rows, w.unique_blocks);
    if (const auto e = observed(s); e != cudaSuccess) return e;
    if (const auto e = scan(w.unique, w.unique_blocks + 1, count, w.scan, s); e != cudaSuccess) return e;
    unique_keys<true><<<count * w.unique_blocks, threads>>>(source, w.unique, target, d.rows, w.unique_blocks);
    if (const auto e = observed(s); e != cudaSuccess) return e;
    plan_metadata<<<1, 1>>>(schema, types, w.unique, w.unique_blocks, first, count, max_bins, s);
    if (const auto e = observed(s); e != cudaSuccess) return e;
    select_metadata<<<grid(u64(count) * (max_bins - 1)), threads>>>(target, w.unique, d.rows,
      w.unique_blocks, schema, first, count, max_bins, s);
    if (const auto e = observed(s); e != cudaSuccess) return e;
    first += count;
  }
  return submit_encoding(d, schema, bins, s);
}

__device__ u32 get_word(const std::byte* bytes) {
  u32 result = 0;
  for (u32 i = 0; i < 4; ++i) result |= u32(bytes[i]) << (8 * i);
  return result;
}
__device__ void put_word(std::byte* bytes, u32 value) {
  for (u32 i = 0; i < 4; ++i) bytes[i] = std::byte(value >> (8 * i));
}
__device__ bool record_shape(u32 rows, u32 columns, u32 outputs, Objective objective, u32 classes,
                              u64& cells, u64& bytes) {
  if (!rows || !columns || !outputs || u32(objective) > 2 ||
      (objective == Objective::multiclass_softmax && (outputs != 1 || classes < 2))) return false;
  cells = u64(rows) * (u64(columns) + outputs);
  if (!mul_fits(rows, u64(columns) + outputs) || !mul_fits(cells, 4) || !add_fits(cells * 4, 32)) return false;
  bytes = cells * 4 + 32;
  return true;
}
__device__ bool target_valid(float x, Objective objective, u32 classes) {
  return finite(x) && (objective != Objective::binary_logistic || x == 0 || x == 1) &&
    (objective != Objective::multiclass_softmax || (x >= 0 && double(x) < classes && floorf(x) == x));
}
template<bool Decode>
__global__ void dataset_payload(const std::byte* input_bytes, std::byte* output_bytes,
    DatasetRecord record, float* values, float* targets, u64 cells, Status* s) {
  const u64 feature_cells = u64(record.data.rows) * record.data.columns;
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < cells; i += u64(gridDim.x) * threads) {
    u32 bits;
    if constexpr (Decode) bits = get_word(input_bytes + 32 + i * 4);
    else bits = __float_as_uint(i < feature_cells ? record.data.values.data[i] : record.data.targets.data[i - feature_cells]);
    const float value = __uint_as_float(bits);
    if (i < feature_cells ? (bits & 0x7fffffffu) == 0x7f800000u : !target_valid(value, record.objective, record.classes)) fail(s, input);
    if constexpr (Decode) {
      if (i < feature_cells) values[i] = value; else targets[i - feature_cells] = value;
    } else put_word(output_bytes + 32 + i * 4, bits);
  }
}
constexpr u64 magic = 0x3130305344424847ULL;
}

__device__ cudaError_t fit_schema(Dataset d, Array<const FeatureType> types, u32 max_bins,
    Schema* schema, Array<std::uint16_t> bins, Workspace workspace, Status* s, RadixPolicy policy) { using namespace data_impl;
  if (!s) return cudaErrorInvalidValue;
  if (!contains(Array<Schema>{schema, 1}, 1) || !matrix(d, true) || max_bins < 2 || max_bins > 65536 ||
      (types.size && !contains(types, d.columns)) || !contains(bins, u64(d.rows) * d.columns) ||
      !contains(schema->features, d.columns) || !contains(schema->offsets, u64(d.columns) + 1)) {
    fail(s, shape); return finish(s);
  }
  if (policy != RadixPolicy::radix8 && policy != RadixPolicy::radix4) { fail(s, unsupported); return finish(s); }
  for (u32 f = 0; types.size && f < d.columns; ++f)
    if (types.data[f] != FeatureType::numeric && types.data[f] != FeatureType::categorical) {
      fail(s, input); return finish(s);
    }
  u32 tile = min(d.columns, 32u);
  auto w = scratch(workspace, d.rows, tile, 1u << u32(policy));
  while (tile > 1 && (!w.arena.valid || w.arena.used > workspace.bytes))
    w = scratch(workspace, d.rows, --tile, 1u << u32(policy));
  if (!w.arena.fits(s)) return finish(s);
  schema->columns = d.columns; schema->metadata_count = 0;
  schema->total_bins = schema->max_feature_bins = 0; schema->offsets.data[0] = 0;
  const auto result = policy == RadixPolicy::radix8
    ? submit_fit<8>(d, types, max_bins, schema, bins.data, w, tile, s)
    : submit_fit<4>(d, types, max_bins, schema, bins.data, w, tile, s);
  return close(s, result);
}
__device__ cudaError_t encode(Dataset d, const Schema* schema, Array<std::uint16_t> bins, Status* s) { using namespace data_impl;
  if (!s) return cudaErrorInvalidValue;
  if (!contains(Array<const Schema>{schema, 1}, 1) || !matrix(d, false) || !contains(bins, u64(d.rows) * d.columns)) { fail(s, shape); return finish(s); }
  validate_schema<<<grid(d.columns), threads>>>(schema, d.columns, s);
  auto result = observed(s);
  if (result == cudaSuccess) result = submit_encoding(d, schema, bins.data, s);
  return close(s, result);
}
__device__ cudaError_t decode_dataset(Array<const std::byte> bytes, Array<float> values,
    Array<float> targets, DatasetRecord* record, Status* s) { using namespace data_impl;
  if (!s) return cudaErrorInvalidValue;
  if (!contains(Array<DatasetRecord>{record, 1}, 1) || !contains(bytes, 32)) { fail(s, shape); return finish(s); }
  for (u32 i = 0; i < 8; ++i)
    if (bytes.data[i] != std::byte(magic >> (i * 8))) { fail(s, input); return finish(s); }
  const u32 version = get_word(bytes.data + 8), rows = get_word(bytes.data + 12);
  const u32 columns = get_word(bytes.data + 16), outputs = get_word(bytes.data + 20);
  const auto objective = Objective(get_word(bytes.data + 24));
  const u32 classes = get_word(bytes.data + 28);
  u64 cells{}, required{};
  if (version != 1 || !record_shape(rows, columns, outputs, objective, classes, cells, required) || bytes.size != required) {
    fail(s, input); return finish(s);
  }
  if (!contains(values, u64(rows) * columns) || !contains(targets, u64(rows) * outputs)) { fail(s, capacity); return finish(s); }
  *record = {{{values.data, u64(rows) * columns}, {targets.data, u64(rows) * outputs}, {}, rows, columns, outputs}, objective, classes};
  dataset_payload<true><<<grid(cells), threads>>>(bytes.data, nullptr, *record, values.data, targets.data, cells, s);
  return close(s, observed(s));
}
__device__ cudaError_t encode_dataset(const DatasetRecord* record, Array<std::byte> bytes, Status* s) { using namespace data_impl;
  if (!s) return cudaErrorInvalidValue;
  if (!contains(Array<const DatasetRecord>{record, 1}, 1)) { fail(s, shape); return finish(s); }
  const auto value = *record;
  const auto d = value.data;
  u64 cells{}, required{};
  if (!record_shape(d.rows, d.columns, d.outputs, value.objective, value.classes, cells, required) ||
      !contains(d.values, u64(d.rows) * d.columns) || !contains(d.targets, u64(d.rows) * d.outputs) || d.weights.size) {
    fail(s, shape); return finish(s);
  }
  s->required_bytes = required;
  if (!contains(bytes, required)) { fail(s, capacity); return finish(s); }
  for (u32 i = 0; i < 8; ++i) bytes.data[i] = std::byte(magic >> (i * 8));
  put_word(bytes.data + 8, 1); put_word(bytes.data + 12, d.rows); put_word(bytes.data + 16, d.columns);
  put_word(bytes.data + 20, d.outputs); put_word(bytes.data + 24, u32(value.objective)); put_word(bytes.data + 28, value.classes);
  dataset_payload<false><<<grid(cells), threads>>>(nullptr, bytes.data, value, nullptr, nullptr, cells, s);
  return close(s, observed(s));
}
}

// GH_SOURCE_CATEGORY: production
// model
namespace gh {
namespace model_impl {
constexpr u64 budget = 1ULL << 30;
constexpr u32 threads = 256;
template<class T> __device__ bool enough(Array<T> a, u64 n) {
  return mul_fits(n, sizeof(T)) && contains(a, n);
}
__device__ unsigned grid(u64 n) { return unsigned(n ? (ceil_div(n, threads) < 65535 ? ceil_div(n, threads) : 65535) : 1); }
__device__ unsigned tree_grid(u64 n) { return unsigned(n && n < 65535 ? n : n ? 65535 : 1); }
__device__ void launch_status(Status* s) { if (cudaGetLastError() != cudaSuccess) fail(s, runtime); }
__device__ bool finite(double x) { return (cuda::std::bit_cast<u64>(x) & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL; }
__device__ u32 feature_bins(Feature f) { return f.count + (f.type == FeatureType::numeric ? 2 : 1); }
__device__ bool header(const Model* p, Status* s) {
  if (!p || reinterpret_cast<std::uintptr_t>(p) % alignof(Model)) { fail(s, shape); return false; }
  const Model& m = *p;
  if (u32(m.objective) > 2 || !m.outputs || m.outputs > INT32_MAX ||
      (m.objective == Objective::multiclass_softmax && m.outputs < 2) ||
      !m.schema.columns || m.schema.columns > INT32_MAX ||
      !m.schema.max_feature_bins || m.schema.max_feature_bins > 65536) {
    fail(s, model); return false;
  }
  if (!enough(m.base, m.outputs) || !enough(m.nodes, m.node_count) ||
      !enough(m.trees, m.tree_count) || !enough(m.output_offsets, u64(m.outputs) + 1) ||
      !enough(m.schema.features, m.schema.columns) ||
      !enough(m.schema.offsets, u64(m.schema.columns) + 1) ||
      !enough(m.schema.metadata, m.schema.metadata_count)) { fail(s, capacity); return false; }
  return true;
}

// Nonnegative binary floating accumulator with a 64-bit significand. Integer
// guard/sticky rounding reproduces the archived binary80 addition precision.
struct Bound { u64 mantissa{}; int exponent{}; };
__device__ Bound positive(u64 bits) {
  const u64 fraction = bits & 0xfffffffffffffULL;
  const int exponent = int(bits >> 52);
  if (exponent) return {((1ULL << 52) | fraction) << 11, exponent - 1086};
  if (!fraction) return {};
  const int shift = __clzll(fraction);
  return {fraction << shift, -1074 - shift};
}
__device__ Bound add_bound(Bound a, Bound b) {
  if (!a.mantissa) return b;
  if (!b.mantissa) return a;
  if (a.exponent < b.exponent) { const auto t = a; a = b; b = t; }
  const unsigned d = unsigned(a.exponent - b.exponent);
  const u64 integer = d < 64 ? b.mantissa >> d : 0;
  const u64 sum = a.mantissa + integer;
  const bool carry = sum < a.mantissa;
  const bool remainder = d >= 64 ? b.mantissa != 0 : d && (b.mantissa & ((1ULL << d) - 1));
  bool guard = false, sticky = false;
  if (carry) {
    guard = sum & 1; sticky = remainder;
    a.mantissa = (1ULL << 63) | (sum >> 1); ++a.exponent;
  } else {
    a.mantissa = sum;
    if (d && d <= 64) {
      guard = (b.mantissa >> (d - 1)) & 1;
      sticky = d > 1 && (b.mantissa & ((1ULL << (d - 1)) - 1));
    }
  }
  if (guard && (sticky || (a.mantissa & 1)) && !++a.mantissa) {
    a.mantissa = 1ULL << 63; ++a.exponent;
  }
  return a;
}

struct ValidationMemory { u32* parent; u32* next; u64* maximum; u32* control; };
__device__ ValidationMemory validation_memory(Arena& a, const Model& m) {
  return {a.take<u32>(m.node_count), a.take<u32>(m.node_count),
          a.take<u64>(m.tree_count), a.take<u32>(2)};
}
__global__ void initialize_validation(Model m, ValidationMemory w) {
  const u64 first = u64(blockIdx.x) * blockDim.x + threadIdx.x, stride = u64(gridDim.x) * blockDim.x;
  for (u64 i = first; i < m.node_count; i += stride) w.parent[i] = 0;
  for (u64 i = first; i < m.tree_count; i += stride) w.maximum[i] = 0;
  if (first < 2) w.control[first] = 0;
}
__global__ void validate_schema(Model m, ValidationMemory w, Status* s) {
  const u64 first = u64(blockIdx.x) * blockDim.x + threadIdx.x, stride = u64(gridDim.x) * blockDim.x;
  for (u64 i = first; i < m.outputs; i += stride) {
    if (!finite(m.base.data[i]) || m.output_offsets.data[i] > m.output_offsets.data[i + 1] ||
        m.output_offsets.data[i + 1] > m.tree_count) fail(s, model);
  }
  if (!first && (m.output_offsets.data[0] || m.output_offsets.data[m.outputs] != m.tree_count ||
      m.schema.offsets.data[0] || m.schema.offsets.data[m.schema.columns] != m.schema.total_bins)) fail(s, model);
  for (u64 i = first; i < m.schema.columns; i += stride) {
    const auto f = m.schema.features.data[i];
    const u32 extra = f.type == FeatureType::numeric ? 2 : 1;
    if (u32(f.type) > 1 || f.count > 65536 - extra || f.begin > m.schema.metadata_count ||
        f.count > m.schema.metadata_count - f.begin) { fail(s, model); continue; }
    if ((!i && f.begin) || (i && (!add_fits(m.schema.features.data[i - 1].begin,
        m.schema.features.data[i - 1].count) || f.begin != m.schema.features.data[i - 1].begin +
        m.schema.features.data[i - 1].count)) || (i + 1 == m.schema.columns &&
        f.begin + f.count != m.schema.metadata_count) ||
        u64(m.schema.offsets.data[i]) + f.count + extra != m.schema.offsets.data[i + 1]) fail(s, model);
    atomicMax(w.control + 1, f.count + extra);
    for (u32 j = 0; j < f.count; ++j) {
      const float value = m.schema.metadata.data[f.begin + j];
      if (!finite(value) || (j && !(m.schema.metadata.data[f.begin + j - 1] < value))) fail(s, model);
    }
  }
}
__global__ void claim_segments(Model m, ValidationMemory w, Status* s) {
  if (s->errors) return;
  for (u64 t = blockIdx.x; t < m.tree_count; t += gridDim.x) {
    const Tree tree = m.trees.data[t];
    if (!tree.count || tree.count > INT32_MAX || tree.output >= m.outputs ||
        tree.begin > m.node_count || tree.count > m.node_count - tree.begin ||
        t < m.output_offsets.data[tree.output] || t >= m.output_offsets.data[tree.output + 1]) {
      if (!threadIdx.x) fail(s, model); continue;
    }
    if (!threadIdx.x) atomicMax(w.control, tree.count);
    for (u64 j = threadIdx.x; j < tree.count; j += blockDim.x)
      if (atomicCAS(w.parent + tree.begin + j, 0, 1)) fail(s, model);
  }
}
__global__ void check_coverage(Model m, ValidationMemory w, Status* s) {
  if (s->errors) return;
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < m.node_count; i += u64(gridDim.x) * blockDim.x) {
    if (w.parent[i] != 1) fail(s, model);
    w.parent[i] = UINT32_MAX;
  }
}
__global__ void claim_parents(Model m, ValidationMemory w, Status* s) {
  if (s->errors) return;
  for (u64 t = blockIdx.x; t < m.tree_count; t += gridDim.x) {
    const Tree tree = m.trees.data[t];
    for (u64 j = threadIdx.x; j < tree.count; j += blockDim.x) {
      const Node n = m.nodes.data[tree.begin + j];
      if (!j && atomicCAS(w.parent + tree.begin, UINT32_MAX, 0) != UINT32_MAX) fail(s, model);
      if (!finite(n.value) || n.missing_left > 1) { fail(s, model); continue; }
      if (n.feature == -1) {
        if (n.left != -1 || n.right != -1 || n.threshold) fail(s, model);
        atomicMax(reinterpret_cast<unsigned long long*>(w.maximum + t),
                  static_cast<unsigned long long>(cuda::std::bit_cast<u64>(n.value) & 0x7fffffffffffffffULL));
      } else {
        if (n.feature < 0 || u32(n.feature) >= m.schema.columns || n.left < 0 || n.right < 0 ||
            u32(n.left) >= tree.count || u32(n.right) >= tree.count ||
            n.threshold >= feature_bins(m.schema.features.data[n.feature])) { fail(s, model); continue; }
        if (atomicCAS(w.parent + tree.begin + u32(n.left), UINT32_MAX, u32(j)) != UINT32_MAX ||
            atomicCAS(w.parent + tree.begin + u32(n.right), UINT32_MAX, u32(j)) != UINT32_MAX) fail(s, model);
      }
    }
  }
}
__global__ void jump_parents(Model m, const u32* parent, u32* next, Status* s) {
  if (s->errors) return;
  for (u64 t = blockIdx.x; t < m.tree_count; t += gridDim.x) {
    const auto tree = m.trees.data[t];
    for (u64 j = threadIdx.x; j < tree.count; j += blockDim.x) {
      const u32 p = parent[tree.begin + j];
      if (p >= tree.count) { fail(s, model); continue; }
      next[tree.begin + j] = parent[tree.begin + p];
    }
  }
}
__global__ void check_graph_bounds(Model m, ValidationMemory w, const u32* parent, Status* s) {
  if (s->errors) return;
  const u64 first = u64(blockIdx.x) * blockDim.x + threadIdx.x, stride = u64(gridDim.x) * blockDim.x;
  for (u64 i = first; i < m.node_count; i += stride) if (parent[i]) fail(s, model);
  if (!first && w.control[1] != m.schema.max_feature_bins) fail(s, model);
  for (u64 output = first; output < m.outputs; output += stride) {
    Bound bound = positive(cuda::std::bit_cast<u64>(m.base.data[output]) & 0x7fffffffffffffffULL);
    for (u64 t = m.output_offsets.data[output]; t < m.output_offsets.data[output + 1]; ++t) {
      bound = add_bound(bound, positive(w.maximum[t]));
      if (bound.exponent > 960 || (bound.exponent == 960 && bound.mantissa > 0xfffffffffffff800ULL)) {
        fail(s, numeric); break;
      }
    }
  }
}
__global__ void finish_validation(Model m, ValidationMemory w, Status* s) {
  if (s->errors) return;
  u32* parent = w.parent;
  u32* next = w.next;
  for (u64 span = 1; span < w.control[0]; span *= 2) {
    jump_parents<<<tree_grid(m.tree_count), threads>>>(m, parent, next, s);
    launch_status(s);
    const auto temporary = parent; parent = next; next = temporary;
  }
  check_graph_bounds<<<grid(m.node_count > m.outputs ? m.node_count : m.outputs), threads>>>(m, w, parent, s);
  launch_status(s);
}

__global__ void check_bins(Model m, const std::uint16_t* bins, u32 rows, Status* s) {
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < u64(rows) * m.schema.columns;
       i += u64(gridDim.x) * blockDim.x)
    if (u32(bins[i]) >= feature_bins(m.schema.features.data[i / rows])) fail(s, input);
}
__global__ void margins(Model m, const std::uint16_t* bins, u32 rows, double* out, Status* s) {
  if (s->errors) return;
  for (u64 task = u64(blockIdx.x) * blockDim.x + threadIdx.x; task < u64(rows) * m.outputs;
       task += u64(gridDim.x) * blockDim.x) {
    const u32 output = u32(task / rows), row = u32(task % rows);
    double value = m.base.data[output];
    for (u64 t = m.output_offsets.data[output]; t < m.output_offsets.data[output + 1]; ++t) {
      const Tree tree = m.trees.data[t];
      u32 node = 0;
      for (u32 step = 0; step < tree.count; ++step) {
        const auto n = m.nodes.data[tree.begin + node];
        if (n.feature == -1) { value = __dadd_rn(value, n.value); break; }
        const auto bin = bins[u64(n.feature) * rows + row];
        const bool left = !bin ? n.missing_left != 0 :
            m.schema.features.data[n.feature].type == FeatureType::numeric ? bin <= n.threshold : bin == n.threshold;
        node = u32(left ? n.left : n.right);
      }
    }
    out[u64(row) * m.outputs + output] = value;
  }
}
__global__ void sigmoid(double* out, u64 size, Status* s) {
  if (s->errors) return;
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < size; i += u64(gridDim.x) * blockDim.x) {
    const double x = out[i], e = exp(x >= 0 ? -x : x);
    out[i] = x >= 0 ? 1.0 / (1.0 + e) : e / (1.0 + e);
  }
}
__global__ void softmax(double* out, u32 rows, u32 outputs, Status* s) {
  if (s->errors) return;
  const u32 lane = threadIdx.x % 32;
  for (u64 row = u64(blockIdx.x) * 8 + threadIdx.x / 32; row < rows; row += u64(gridDim.x) * 8) {
    const u64 begin = row * outputs;
    double maximum = -INFINITY;
    for (u64 column = lane; column < outputs; column += 32) maximum = fmax(maximum, out[begin + column]);
    for (unsigned offset = 16; offset; offset /= 2) maximum = fmax(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    double denominator = 0;
    for (u64 column = lane; column < outputs; column += 32)
      denominator = __dadd_rn(denominator, exp(out[begin + column] - maximum));
    for (unsigned offset = 16; offset; offset /= 2) denominator = __dadd_rn(denominator, __shfl_down_sync(0xffffffff, denominator, offset));
    denominator = __shfl_sync(0xffffffff, denominator, 0);
    for (u64 column = lane; column < outputs; column += 32)
      out[begin + column] = exp(out[begin + column] - maximum) / denominator;
  }
}
__global__ void final_values(const double* out, u64 size, Status* s) {
  if (s->errors) return;
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < size; i += u64(gridDim.x) * blockDim.x)
    if (!finite(out[i])) fail(s, numeric);
}

struct Reader {
  Array<const std::byte> bytes;
  u64 position{};
  bool valid{true};
  __device__ u64 get(unsigned width) {
    if (position > bytes.size || width > bytes.size - position) { valid = false; return 0; }
    u64 value = 0;
    for (unsigned j = 0; j < width; ++j) value |= u64(bytes.data[position++]) << (8 * j);
    return value;
  }
  __device__ void skip(u64 n) {
    if (position > bytes.size || n > bytes.size - position) valid = false;
    else position += n;
  }
};
__device__ u64 load(Array<const std::byte> b, u64 at, unsigned width) { Reader r{b, at}; return r.get(width); }
__device__ void store(Array<std::byte> b, u64 at, u64 value, unsigned width) {
  for (unsigned j = 0; j < width; ++j) b.data[at + j] = std::byte((value >> (8 * j)) & 255);
}
struct Frame { u32 objective{}, outputs{}, columns{}; u64 trees{}, metadata{}, nodes{}; };
__device__ bool account(u64& total, u64 count, u64 width) {
  if (!mul_fits(count, width) || count * width > budget - total) return false;
  total += count * width; return true;
}
__device__ bool frame(Array<const std::byte> bytes, Frame& f) {
  if (!contains(bytes, bytes.size) || bytes.size < 32 || bytes.size > budget) return false;
  Reader r{bytes};
  if (r.get(8) != 0x4c45444f4d424847ULL || r.get(4) != 1) return false;
  f.objective = u32(r.get(4)); f.outputs = u32(r.get(4)); f.columns = u32(r.get(4)); f.trees = r.get(8);
  if (f.objective > 2 || !f.outputs || f.outputs > INT32_MAX || !f.columns || f.columns > INT32_MAX ||
      (f.objective == 2 && f.outputs < 2)) return false;
  u64 allocation = 0;
  if (!account(allocation, f.outputs, 8) || !account(allocation, f.columns, 56) || !account(allocation, f.trees, 32)) return false;
  r.skip(u64(f.outputs) * 8);
  for (u32 i = 0; i < f.columns && r.valid; ++i) {
    const u32 type = u32(r.get(4)), cuts = u32(r.get(4)), categories = u32(r.get(4));
    if (type > 1 || cuts > 65534 || categories > 65535 || (type ? cuts : categories)) return false;
    const u32 count = cuts + categories;
    if (!account(allocation, count, 4)) return false;
    f.metadata += count; r.skip(u64(count) * 4);
  }
  for (u64 t = 0; t < f.trees && r.valid; ++t) {
    const u32 output = u32(r.get(4)), count = u32(r.get(4));
    if (output >= f.outputs || !count || count > INT32_MAX || !account(allocation, count, 32)) return false;
    f.nodes += count; r.skip(u64(count) * 28);
  }
  return r.valid && r.position == bytes.size;
}
struct WireTree { u64 position, begin; u32 count, output; };
struct CodecMemory { u64* features; WireTree* trees; u64* cursor; };
__device__ CodecMemory codec_memory(Arena& a, u32 columns, u64 trees, u32 outputs) {
  return {a.take<u64>(columns), a.take<WireTree>(trees), a.take<u64>(outputs)};
}
template<bool Decode> __global__ void codec_base(Model m, Array<std::byte> bytes) {
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < m.outputs; i += u64(gridDim.x) * blockDim.x) {
    if constexpr (Decode) m.base.data[i] = cuda::std::bit_cast<double>(load({bytes.data, bytes.size}, 32 + 8 * i, 8));
    else store(bytes, 32 + 8 * i, cuda::std::bit_cast<u64>(m.base.data[i]), 8);
  }
}
template<bool Decode> __global__ void codec_features(Model m, Array<std::byte> bytes, const u64* positions) {
  for (u64 f = blockIdx.x; f < m.schema.columns; f += gridDim.x) {
    const Feature feature = m.schema.features.data[f];
    const u64 at = positions[f];
    if constexpr (!Decode) if (!threadIdx.x) {
      store(bytes, at, u32(feature.type), 4);
      store(bytes, at + 4, feature.type == FeatureType::numeric ? feature.count : 0, 4);
      store(bytes, at + 8, feature.type == FeatureType::categorical ? feature.count : 0, 4);
    }
    for (u64 j = threadIdx.x; j < feature.count; j += blockDim.x) {
      if constexpr (Decode) m.schema.metadata.data[feature.begin + j] = cuda::std::bit_cast<float>(u32(load({bytes.data, bytes.size}, at + 12 + 4 * j, 4)));
      else store(bytes, at + 12 + 4 * j, cuda::std::bit_cast<u32>(m.schema.metadata.data[feature.begin + j]), 4);
    }
  }
}
template<bool Decode> __global__ void codec_nodes(Model m, Array<std::byte> bytes, const WireTree* wire) {
  for (u64 t = blockIdx.x; t < m.tree_count; t += gridDim.x) {
    const auto tree = wire[t];
    if constexpr (!Decode) if (!threadIdx.x) { store(bytes, tree.position, tree.output, 4); store(bytes, tree.position + 4, tree.count, 4); }
    for (u64 j = threadIdx.x; j < tree.count; j += blockDim.x) {
      const u64 at = tree.position + 8 + j * 28;
      if constexpr (Decode) {
        const Array<const std::byte> b{bytes.data, bytes.size};
        m.nodes.data[tree.begin + j] = {cuda::std::bit_cast<std::int32_t>(u32(load(b, at, 4))),
          cuda::std::bit_cast<std::int32_t>(u32(load(b, at + 4, 4))), cuda::std::bit_cast<std::int32_t>(u32(load(b, at + 8, 4))),
          u32(load(b, at + 12, 4)), u32(load(b, at + 16, 4)), cuda::std::bit_cast<double>(load(b, at + 20, 8))};
      } else {
        const auto n = m.nodes.data[tree.begin + j];
        store(bytes, at, cuda::std::bit_cast<u32>(n.feature), 4); store(bytes, at + 4, cuda::std::bit_cast<u32>(n.left), 4);
        store(bytes, at + 8, cuda::std::bit_cast<u32>(n.right), 4); store(bytes, at + 12, n.threshold, 4);
        store(bytes, at + 16, n.missing_left, 4); store(bytes, at + 20, cuda::std::bit_cast<u64>(n.value), 8);
      }
    }
  }
}
template<bool Decode> __device__ void codec_payload(Model m, Array<std::byte> b, CodecMemory w, Status* s) {
  codec_base<Decode><<<grid(m.outputs), threads>>>(m, b); launch_status(s);
  codec_features<Decode><<<tree_grid(m.schema.columns), threads>>>(m, b, w.features); launch_status(s);
  if (m.tree_count) { codec_nodes<Decode><<<tree_grid(m.tree_count), threads>>>(m, b, w.trees); launch_status(s); }
}
}

__device__ cudaError_t validate_model(const Model* p, Workspace workspace, Status* s) { using namespace model_impl;
  if (!s) return cudaErrorInvalidValue;
  if (!header(p, s)) return finish(s);
  const Model m = *p;
  Arena arena{workspace}; const auto w = validation_memory(arena, m);
  if (!arena.fits(s)) return finish(s);
  initialize_validation<<<grid(m.node_count > m.tree_count ? m.node_count : m.tree_count), threads>>>(m, w); launch_status(s);
  validate_schema<<<grid(m.outputs > m.schema.columns ? m.outputs : m.schema.columns), threads>>>(m, w, s); launch_status(s);
  claim_segments<<<tree_grid(m.tree_count), threads>>>(m, w, s); launch_status(s);
  check_coverage<<<grid(m.node_count), threads>>>(m, w, s); launch_status(s);
  claim_parents<<<tree_grid(m.tree_count), threads>>>(m, w, s); launch_status(s);
  finish_validation<<<1, 1, 0, cudaStreamTailLaunch>>>(m, w, s); launch_status(s);
  return finish(s);
}
__device__ cudaError_t predict(const Model* p, Array<const std::uint16_t> bins, u32 rows,
                             Array<double> out, bool raw, Status* s) { using namespace model_impl;
  if (!s) return cudaErrorInvalidValue;
  if (!header(p, s)) return finish(s);
  const Model m = *p;
  if (!rows) return finish(s);
  const u64 size = u64(rows) * m.outputs;
  if (!enough(bins, u64(rows) * m.schema.columns) || !enough(out, size)) { fail(s, capacity); return finish(s); }
  check_bins<<<grid(u64(rows) * m.schema.columns), threads>>>(m, bins.data, rows, s); launch_status(s);
  margins<<<grid(size), threads>>>(m, bins.data, rows, out.data, s); launch_status(s);
  if (!raw && m.objective == Objective::binary_logistic) { sigmoid<<<grid(size), threads>>>(out.data, size, s); launch_status(s); }
  if (!raw && m.objective == Objective::multiclass_softmax) { softmax<<<grid(u64(rows) * 32), threads>>>(out.data, rows, m.outputs, s); launch_status(s); }
  final_values<<<grid(size), threads>>>(out.data, size, s); launch_status(s);
  return finish(s);
}
__device__ cudaError_t decode_model(Array<const std::byte> bytes, Model* p, Workspace workspace, Status* s) { using namespace model_impl;
  if (!s) return cudaErrorInvalidValue;
  Frame f;
  if (!p || reinterpret_cast<std::uintptr_t>(p) % alignof(Model)) { fail(s, shape); return finish(s); }
  if (!frame(bytes, f)) { fail(s, model); return finish(s); }
  if (!enough(p->base, f.outputs) || !enough(p->schema.features, f.columns) ||
      !enough(p->schema.metadata, f.metadata) || !enough(p->schema.offsets, u64(f.columns) + 1) ||
      !enough(p->trees, f.trees) || !enough(p->nodes, f.nodes) || !enough(p->output_offsets, u64(f.outputs) + 1)) {
    fail(s, capacity); return finish(s);
  }
  Arena arena{workspace}; const auto w = codec_memory(arena, f.columns, f.trees, f.outputs);
  Model m = *p; m.outputs = f.outputs; m.objective = Objective(f.objective); m.tree_count = f.trees; m.node_count = f.nodes;
  m.schema.columns = f.columns; m.schema.metadata_count = f.metadata;
  Arena validation{workspace}; (void)validation_memory(validation, m);
  const u64 required = arena.used > validation.used ? arena.used : validation.used;
  if (!arena.valid || !validation.valid) { fail(s, extent); return finish(s); }
  Arena combined{workspace}; combined.used = required;
  if (!combined.fits(s)) return finish(s);
  Reader r{bytes, 32 + u64(f.outputs) * 8};
  u64 metadata_begin = 0, node_begin = 0, total_bins = 0;
  m.schema.max_feature_bins = 0;
  for (u32 i = 0; i < f.columns; ++i) {
    w.features[i] = r.position;
    const auto type = FeatureType(u32(r.get(4))); const u32 cuts = u32(r.get(4)), categories = u32(r.get(4));
    const u32 count = cuts + categories, nbins = count + (type == FeatureType::numeric ? 2 : 1);
    m.schema.features.data[i] = {metadata_begin, count, type};
    m.schema.offsets.data[i] = u32(total_bins);
    metadata_begin += count; total_bins += nbins;
    if (nbins > m.schema.max_feature_bins) m.schema.max_feature_bins = nbins;
    r.skip(u64(count) * 4);
  }
  if (total_bins > UINT32_MAX) { fail(s, model); return finish(s); }
  m.schema.total_bins = u32(total_bins); m.schema.offsets.data[f.columns] = u32(total_bins);
  for (u32 o = 0; o < f.outputs; ++o) { m.output_offsets.data[o] = 0; w.cursor[o] = 0; }
  m.output_offsets.data[f.outputs] = 0;
  for (u64 t = 0; t < f.trees; ++t) {
    const u64 at = r.position; const u32 output = u32(r.get(4)), count = u32(r.get(4));
    w.trees[t] = {at, node_begin, count, output}; node_begin += count;
    ++m.output_offsets.data[output + 1]; r.skip(u64(count) * 28);
  }
  for (u32 o = 0; o < f.outputs; ++o) m.output_offsets.data[o + 1] += m.output_offsets.data[o];
  for (u64 t = 0; t < f.trees; ++t) {
    const auto tree = w.trees[t];
    m.trees.data[m.output_offsets.data[tree.output] + w.cursor[tree.output]++] = {tree.begin, tree.count, tree.output};
  }
  *p = m;
  codec_payload<true>(m, {const_cast<std::byte*>(bytes.data), bytes.size}, w, s);
  // Same-thread NULL-stream ordering completes payload readers before validation
  // reuses their framing arena. Header/descriptor values were written above.
  const auto result = validate_model(p, workspace, s);
  s->required_bytes = required;
  return result;
}
__device__ cudaError_t encode_model(const Model* p, Array<std::byte> bytes, Array<u64> written,
                                  Workspace workspace, Status* s) { using namespace model_impl;
  if (!s) return cudaErrorInvalidValue;
  if (!header(p, s)) return finish(s);
  const Model m = *p;
  Arena arena{workspace}; const auto w = codec_memory(arena, m.schema.columns, m.tree_count, 0);
  if (!arena.fits(s)) return finish(s);
  u64 position = 32;
  if (!account(position, m.outputs, 8)) { fail(s, extent); return finish(s); }
  for (u32 i = 0; i < m.schema.columns; ++i) {
    w.features[i] = position;
    if (!account(position, 1, 12) || !account(position, m.schema.features.data[i].count, 4)) { fail(s, extent); return finish(s); }
  }
  for (u64 t = 0; t < m.tree_count; ++t) {
    const Tree tree = m.trees.data[t]; w.trees[t] = {position, tree.begin, tree.count, tree.output};
    if (!account(position, 1, 8) || !account(position, tree.count, 28)) { fail(s, extent); return finish(s); }
  }
  if (!contains(bytes, position) || !contains(written, 1)) { fail(s, capacity); return finish(s); }
  store(bytes, 0, 0x4c45444f4d424847ULL, 8); store(bytes, 8, 1, 4); store(bytes, 12, u32(m.objective), 4);
  store(bytes, 16, m.outputs, 4); store(bytes, 20, m.schema.columns, 4); store(bytes, 24, m.tree_count, 8);
  written.data[0] = position;
  codec_payload<false>(m, bytes, w, s);
  return finish(s);
}
}

// GH_SOURCE_CATEGORY: production
// prediction
namespace gh {
namespace prediction_impl {
template<bool Decode>
__global__ void payload(const std::byte* source, std::byte* target,
                        const double* values, double* output, u64 count, Status* status) {
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += u64(gridDim.x) * blockDim.x) {
    u64 bits = 0;
    if constexpr (Decode) {
      #pragma unroll
      for (u32 k = 0; k < 8; ++k) bits |= u64(source[8*i+k]) << (8*k);
    } else bits = cuda::std::bit_cast<u64>(values[i]);
    if ((bits & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL) { fail(status,numeric); continue; }
    if constexpr (Decode) output[i] = cuda::std::bit_cast<double>(bits);
    else {
      #pragma unroll
      for (u32 k = 0; k < 8; ++k) target[8*i+k] = std::byte(bits >> (8*k));
    }
  }
}
__device__ bool dimensions(u32 rows, u32 outputs, u64& count, Status* status) {
  if (!outputs) { fail(status,gh::shape); return false; }
  count = u64(rows)*outputs;
  if (!mul_fits(count,8)) { fail(status,extent); return false; }
  status->required_bytes = count*8;
  return true;
}
__device__ cudaError_t complete_payload(Status* status) {
  const auto launched = cudaGetLastError();
  if (launched != cudaSuccess) fail(status,runtime);
  const auto completed = finish(status);
  return launched != cudaSuccess ? launched : completed;
}
}
__device__ cudaError_t decode_predictions(Array<const std::byte> bytes, u32 rows,
    u32 outputs, Array<double> values, Status* status) { using namespace prediction_impl;
  if (!status) return cudaErrorInvalidValue;
  u64 count{};
  if (!dimensions(rows,outputs,count,status)) return finish(status);
  if (bytes.size != status->required_bytes) { fail(status,input); return finish(status); }
  if (!contains(bytes,status->required_bytes) || !contains(values,count)) { fail(status,capacity); return finish(status); }
  if (count) payload<true><<<u32(min(ceil_div(count,256),u64(65535))),256>>>(bytes.data,nullptr,nullptr,values.data,count,status);
  return complete_payload(status);
}
__device__ cudaError_t encode_predictions(Array<const double> values, u32 rows,
    u32 outputs, Array<std::byte> bytes, Status* status) { using namespace prediction_impl;
  if (!status) return cudaErrorInvalidValue;
  u64 count{};
  if (!dimensions(rows,outputs,count,status)) return finish(status);
  if (!contains(values,count) || !contains(bytes,status->required_bytes)) { fail(status,capacity); return finish(status); }
  if (count) payload<false><<<u32(min(ceil_div(count,256),u64(65535))),256>>>(nullptr,bytes.data,values.data,nullptr,count,status);
  return complete_payload(status);
}
}

// GH_SOURCE_CATEGORY: production
// csv
namespace gh {
namespace csv_impl {
constexpr u32 threads = 256;
struct Text { char data[25]{}; u32 size{}; };
__device__ u32 grid(u64 n) { return u32(min(u64{65535}, ceil_div(n, threads))); }
__device__ bool launched(Status* s) {
  const auto error = cudaGetLastError();
  if (error != cudaSuccess) fail(s, runtime);
  return error == cudaSuccess;
}
__device__ void append(Text& out, char c) { out.data[out.size++] = c; }
__device__ void integer(Text& out, u32 value, u32 minimum = 1) {
  char digits[10]; u32 n = 0;
  do { digits[n++] = char('0' + value % 10); value /= 10; } while (value || n < minimum);
  while (n) append(out, digits[--n]);
}
__device__ u64 header_size(u32 outputs, CsvKind kind) {
  const bool numbered = outputs > 1 || kind == CsvKind::multiclass;
  u64 bytes = 7 + (kind == CsvKind::targets ? 7 : 0) +
    u64(outputs) * (kind == CsvKind::targets ? 7 : kind == CsvKind::multiclass ? 2 : 11);
  if (numbered) {
    bytes += outputs; // At least one digit per column.
    for (u64 power = 10; power < outputs; power *= 10) bytes += outputs - power;
    if (kind != CsvKind::multiclass) bytes += outputs;
  }
  return bytes;
}
__device__ Text decimal(double value, Status* status) {
  Text out;
  const u64 bits = cuda::std::bit_cast<u64>(value);
  const u32 exponent = u32((bits >> 52) & 2047);
  if (exponent == 2047) { fail(status, input); return out; }
  if (bits >> 63) append(out, '-');
  const u64 mantissa = (bits & 0xfffffffffffffULL) | (exponent ? 1ULL << 52 : 0);
  if (!mantissa) { append(out, '0'); return out; }
  u32 limbs[86]{}; u32 used = 0;
  for (u64 x = mantissa; x; x /= 1000000000) limbs[used++] = u32(x % 1000000000);
  const int binary_exponent = exponent ? int(exponent) - 1075 : -1074;
  u32 remaining = u32(abs(binary_exponent));
  while (remaining) {
    const u32 step = min(remaining, binary_exponent < 0 ? 12u : 29u);
    u32 factor = 1;
    for (u32 i = 0; i < step; ++i) factor *= binary_exponent < 0 ? 5u : 2u;
    u64 carry = 0;
    for (u32 i = 0; i < used; ++i) {
      const u64 product = u64(limbs[i]) * factor + carry;
      limbs[i] = u32(product % 1000000000); carry = product / 1000000000;
    }
    if (carry) limbs[used++] = u32(carry);
    remaining -= step;
  }
  u32 leading = 0;
  for (u32 x = limbs[used - 1]; x; x /= 10) ++leading;
  const u32 digits = (used - 1) * 9 + leading;
  auto digit = [&](u32 index) {
    const u32 position = digits - 1 - index, limb = position / 9;
    u32 divisor = 1;
    for (u32 i = 0; i < position % 9; ++i) divisor *= 10;
    return limbs[limb] / divisor % 10;
  };
  char significant[17]; u32 count = min(digits, 17u);
  for (u32 i = 0; i < count; ++i) significant[i] = char('0' + digit(i));
  int power = int(digits) - 1 - (binary_exponent < 0 ? -binary_exponent : 0);
  if (digits > 17) {
    const u32 guard = digit(17); bool sticky = false;
    for (u32 i = 18; i < digits && !sticky; ++i) sticky = digit(i) != 0;
    if (guard > 5 || (guard == 5 && (sticky || ((significant[16] - '0') & 1)))) {
      int i = 16;
      while (i >= 0 && significant[i] == '9') significant[i--] = '0';
      if (i >= 0) ++significant[i];
      else { significant[0] = '1'; ++power; }
    }
  }
  while (count > 1 && significant[count - 1] == '0') --count;
  if (power < -4 || power >= 17) {
    append(out, significant[0]);
    if (count > 1) append(out, '.');
    for (u32 i = 1; i < count; ++i) append(out, significant[i]);
    append(out, 'e'); append(out, power < 0 ? '-' : '+'); integer(out, u32(abs(power)), 2);
  } else if (power < 0) {
    append(out, '0'); append(out, '.');
    for (int i = -1; i > power; --i) append(out, '0');
    for (u32 i = 0; i < count; ++i) append(out, significant[i]);
  } else {
    for (u32 i = 0; i <= u32(power); ++i) append(out, i < count ? significant[i] : '0');
    if (count > u32(power) + 1) append(out, '.');
    for (u32 i = u32(power) + 1; i < count; ++i) append(out, significant[i]);
  }
  return out;
}
__global__ void format(Array<const double> values, Array<const double> weights,
    u32 rows, u32 outputs, CsvKind kind, Text* text, u64* lengths, Status* status) {
  const u64 columns = u64(outputs) + 1 + (kind == CsvKind::targets);
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < u64(rows) * columns; i += u64(gridDim.x) * threads) {
    const u32 row = u32(i / columns);
    const u64 column = i % columns;
    Text out;
    if (!column) integer(out, row);
    else if (column <= outputs) out = decimal(values.data[u64(row) * outputs + column - 1], status);
    else {
      const double w = weights.size ? weights.data[row] : 1;
      if (w < 0) fail(status, input);
      out = decimal(w, status);
    }
    append(out, column + 1 == columns ? '\n' : ',');
    text[i] = out; lengths[i] = out.size;
  }
}
__global__ void prefix(u64* lengths, u64 count, u64* totals) {
  __shared__ u64 sums[threads];
  for (u64 block = blockIdx.x; block < ceil_div(count, threads); block += gridDim.x) {
    const u64 i = block * threads + threadIdx.x;
    sums[threadIdx.x] = i < count ? lengths[i] : 0;
    __syncthreads();
    for (u32 stride = 1; stride < threads; stride *= 2) {
      const u64 value = threadIdx.x >= stride ? sums[threadIdx.x - stride] : 0;
      __syncthreads();
      sums[threadIdx.x] += value;
      __syncthreads();
    }
    if (i < count) lengths[i] = sums[threadIdx.x];
    if (!threadIdx.x && totals) totals[block] = sums[threads - 1];
    __syncthreads();
  }
}
__global__ void add_offsets(u64* lengths, u64 count, const u64* totals) {
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < count; i += u64(gridDim.x) * threads)
    if (i >= threads) lengths[i] += totals[i / threads - 1];
}
__device__ void put(Array<std::byte> out, u64& position, char c) {
  if (position < out.size) out.data[position] = std::byte(c);
  ++position;
}
__global__ void header(u32 outputs, CsvKind kind, Array<std::byte> out,
    Array<u64> written, u64* header_bytes, const u64* offsets, u64 count, Status* s) {
  if (s->errors) return;
  u64 p = 0;
  for (const char* x = "row_id"; *x; ++x) put(out, p, *x);
  for (u64 column = 0; column < outputs; ++column) {
    put(out, p, ',');
    const char* name = kind == CsvKind::targets ? "target" : kind == CsvKind::multiclass ? "p" : "prediction";
    for (; *name; ++name) put(out, p, *name);
    if (outputs > 1 || kind == CsvKind::multiclass) {
      if (kind != CsvKind::multiclass) put(out, p, '_');
      Text number; integer(number, u32(column));
      for (u32 i = 0; i < number.size; ++i) put(out, p, number.data[i]);
    }
  }
  if (kind == CsvKind::targets) for (const char* x = ",weight"; *x; ++x) put(out, p, *x);
  put(out, p, '\n'); *header_bytes = p;
  const u64 payload = count ? offsets[count - 1] : 0;
  if (!add_fits(p, payload)) { fail(s, extent); return; }
  written.data[0] = s->required_bytes = p + payload;
  if (!contains(out, p + payload)) fail(s, capacity);
}
__global__ void compact(const Text* text, const u64* offsets, u64 count,
    const u64* header_bytes, Array<std::byte> out, Status* s) {
  if (s->errors) return;
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < count; i += u64(gridDim.x) * threads) {
    const u64 p = *header_bytes + (i ? offsets[i - 1] : 0);
    for (u32 k = 0; k < text[i].size; ++k) out.data[p + k] = std::byte(text[i].data[k]);
  }
}
}
__device__ cudaError_t encode_csv(Array<const double> values, u32 rows, u32 outputs,
    CsvKind kind, Array<const double> weights, Array<std::byte> bytes,
    Array<u64> written, Workspace workspace, Status* status) { using namespace csv_impl;
  if (!status) return cudaErrorInvalidValue;
  if (!outputs || u32(kind) > u32(CsvKind::multiclass) || !contains(values, u64(rows) * outputs) ||
      !contains(written, 1) || !contains(bytes, bytes.size) ||
      ((weights.data || weights.size) && (kind != CsvKind::targets || !contains(weights, rows)))) {
    fail(status, shape); return finish(status);
  }
  const u64 columns = u64(outputs) + 1 + (kind == CsvKind::targets);
  if (bytes.size < header_size(outputs, kind)) {
    status->required_bytes = header_size(outputs, kind); fail(status, capacity); return finish(status);
  }
  if (!mul_fits(rows, columns) || !mul_fits(u64(rows) * columns, 25)) { fail(status, extent); return finish(status); }
  const u64 count = u64(rows) * columns;
  Arena arena{workspace};
  auto* text = arena.take<Text>(count); auto* header_bytes = arena.take<u64>(1);
  u64* levels[9]; u64 counts[9]{count}; u32 depth = 0;
  levels[0] = arena.take<u64>(count);
  while (counts[depth] > threads) { counts[depth + 1] = ceil_div(counts[depth], threads); ++depth; levels[depth] = arena.take<u64>(counts[depth]); }
  if (!arena.fits(status)) return finish(status);
  if (count) {
    format<<<grid(count), threads>>>(values, weights, rows, outputs, kind, text, levels[0], status);
    if (!launched(status)) return finish(status);
    for (u32 i = 0; i <= depth; ++i) {
      prefix<<<grid(counts[i]), threads>>>(levels[i], counts[i], i < depth ? levels[i + 1] : nullptr);
      if (!launched(status)) return finish(status);
    }
    for (u32 i = depth; i; --i) {
      add_offsets<<<grid(counts[i - 1]), threads>>>(levels[i - 1], counts[i - 1], levels[i]);
      if (!launched(status)) return finish(status);
    }
  }
  header<<<1, 1>>>(outputs, kind, bytes, written, header_bytes, levels[0], count, status);
  if (!launched(status)) return finish(status);
  if (count) compact<<<grid(count), threads>>>(text, levels[0], count, header_bytes, bytes, status);
  return finish(status);
}
}

// GH_SOURCE_CATEGORY: production
// metrics
namespace gh {
namespace metrics_impl {
using detail::finite_metric;
using detail::numpy_sum;
using detail::positive_sum;
using detail::scaled_mean;
constexpr u32 threads = 256;
constexpr double clip = 1e-15;
__device__ u32 grid(u64 n) { return u32(min(ceil_div(n, threads), u64(65535))); }
__device__ bool observed(Status* s) {
  const auto e = cudaGetLastError();
  if (e != cudaSuccess) fail(s, runtime);
  return e == cudaSuccess;
}
template<class T> __device__ bool pointer(T* p) { return p && reinterpret_cast<std::uintptr_t>(p) % alignof(T) == 0; }
__device__ bool independent(MetricTask t) { return t == MetricTask::binary || t == MetricTask::multilabel; }
__device__ double weight(MetricInput d, u32 row) { return d.weights.size ? d.weights.data[row] : 1.0; }
__device__ double square(double x) { return __dmul_rn(x, x); }
__device__ double f1(u64 tp, u64 fp, u64 fn) { return tp ? __ddiv_rn(double(2 * tp), double(2 * tp + fp + fn)) : 0; }

struct Pair { u32 positives{}, groups{}; };
struct Group { u32 end{}, positives{}; };
struct RankMemory { u32 *a, *b; Pair *prefix, *scratch; Group* groups; u32* counts; };
__device__ u64 scan_size(u32 length, u32 groups) {
  u64 size = 0;
  while (length > threads) { length = u32(ceil_div(length, threads)); size += u64(length) * groups; }
  return size;
}
__device__ RankMemory ranking_memory(Arena& arena, u32 length, u32 groups) {
  const u64 n = u64(length) * groups;
  return {arena.take<u32>(n), arena.take<u32>(n), arena.take<Pair>(n),
    arena.take<Pair>(max(scan_size(length, groups), scan_size(u32(n), 1))),
    arena.take<Group>(n), arena.take<u32>(groups)};
}
__device__ bool before(const double* p, u32 a, u32 b) {
  return p[a] > p[b] || (p[a] == p[b] && a < b);
}
__global__ void rank_indices(u32* out, u32 length, u32 groups, u32 outputs, bool pooled) {
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < u64(length) * groups; i += u64(gridDim.x) * threads)
    out[i] = pooled ? u32(i) : u32((i % length) * outputs + i / length);
}
// Each thread merges eight adjacent output positions after a merge-path search.
__global__ void merge_indices(const double* p, const u32* in, u32* out, u32 length,
                              u32 groups, u64 width, Status* s) {
  if (s->errors) return;
  const u64 runs = ceil_div(length, width * 2), chunks = ceil_div(width * 2, 8);
  const u64 jobs_per_group = runs * chunks;
  for (u64 job = u64(blockIdx.x) * threads + threadIdx.x; job < jobs_per_group * groups; job += u64(gridDim.x) * threads) {
    const u64 group = job / jobs_per_group, local = job % jobs_per_group;
    const u64 first = (local / chunks) * width * 2, diagonal = (local % chunks) * 8;
    const u32 a = u32(min(width, u64(length) - first));
    const u32 b = u32(min(width, u64(length) - first - a));
    if (diagonal >= u64(a) + b) continue;
    const u64 base = group * length + first;
    u32 lo = u32(diagonal > b ? diagonal - b : 0), hi = u32(min(diagonal, u64(a)));
    while (lo < hi) {
      const u32 middle = lo + (hi - lo) / 2, right = u32(diagonal) - middle;
      if (middle < a && right && before(p, in[base + middle], in[base + a + right - 1])) lo = middle + 1;
      else hi = middle;
    }
    u32 left = lo, right = u32(diagonal) - lo;
    for (u32 k = 0; k < 8 && diagonal + k < u64(a) + b; ++k) {
      const bool take_left = left < a && (right == b || before(p, in[base + left], in[base + a + right]));
      out[base + diagonal + k] = take_left ? in[base + left++] : in[base + a + right++];
    }
  }
}
__device__ Pair plus(Pair a, Pair b) { return {a.positives + b.positives, a.groups + b.groups}; }
__device__ Pair inclusive(Pair x) {
  for (u32 step = 1; step < 32; step *= 2) {
    const Pair previous{__shfl_up_sync(0xffffffffu, x.positives, step), __shfl_up_sync(0xffffffffu, x.groups, step)};
    if (threadIdx.x % 32 >= step) x = plus(previous, x);
  }
  return x;
}
__global__ void rank_flags(const double* p, const double* y, const u32* order, Pair* flags,
                           u32 length, u32 groups, Status* s) {
  if (s->errors) return;
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < u64(length) * groups; i += u64(gridDim.x) * threads)
    flags[i] = {u32(y[order[i]] == 1), u32(i % length + 1 == length || p[order[i]] != p[order[i + 1]])};
}
__global__ void scan_pairs(Pair* data, u32 length, u32 blocks, Pair* sums, Status* s) {
  if (s->errors) return;
  __shared__ Pair totals[8];
  const u32 group = blockIdx.x / blocks, block = blockIdx.x % blocks, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const u64 at = u64(block) * threads + threadIdx.x, base = u64(group) * length;
  const Pair original = at < length ? data[base + at] : Pair{};
  Pair value = inclusive(original);
  if (lane == 31) totals[warp] = value;
  __syncthreads();
  if (!warp) {
    const Pair all = inclusive(lane < 8 ? totals[lane] : Pair{});
    if (lane < 8) totals[lane] = all;
  }
  __syncthreads();
  if (warp) value = plus(totals[warp - 1], value);
  if (at < length) data[base + at] = {value.positives - original.positives, value.groups - original.groups};
  if (!threadIdx.x && sums) sums[u64(group) * blocks + block] = totals[7];
}
__global__ void add_pair_offsets(Pair* values, const Pair* totals, u32 length, u32 blocks, u64 n, Status* s) {
  if (s->errors) return;
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < n; i += u64(gridDim.x) * threads)
    values[i] = plus(values[i], totals[(i / length) * blocks + (i % length) / threads]);
}
__global__ void compact_groups(const double* p, const double* y, const u32* order, const Pair* prefix,
                               Group* out, u32* counts, u32 length, u32 groups, Status* s) {
  if (s->errors) return;
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < u64(length) * groups; i += u64(gridDim.x) * threads) {
    const u32 local = u32(i % length);
    if (local + 1 == length || p[order[i]] != p[order[i + 1]]) {
      out[(i / length) * length + prefix[i].groups] = {local + 1, prefix[i].positives + u32(y[order[i]] == 1)};
      if (local + 1 == length) counts[i / length] = prefix[i].groups + 1;
    }
  }
}
__device__ u32* rank(const double* p, const double* y, u32 length, u32 groups,
                    u32 outputs, bool pooled, RankMemory w, Status* s) {
  const u64 n = u64(length) * groups;
  rank_indices<<<grid(n), threads>>>(w.a, length, groups, outputs, pooled);
  if (!observed(s)) return nullptr;
  u32* source = w.a; u32* destination = w.b;
  for (u64 width = 1; width < length; width *= 2) {
    const u64 jobs = ceil_div(length, width * 2) * ceil_div(width * 2, 8) * groups;
    merge_indices<<<grid(jobs), threads>>>(p, source, destination, length, groups, width, s);
    if (!observed(s)) return nullptr;
    auto* temporary = source; source = destination; destination = temporary;
  }
  rank_flags<<<grid(n), threads>>>(p, y, source, w.prefix, length, groups, s);
  if (!observed(s)) return nullptr;
  Pair* levels[5]{w.prefix}; u32 lengths[5]{length}, depth = 0;
  auto* scratch = w.scratch;
  while (true) {
    const u32 blocks = u32(ceil_div(lengths[depth], threads));
    auto* next = blocks > 1 ? scratch : nullptr;
    scan_pairs<<<groups * blocks, threads>>>(levels[depth], lengths[depth], blocks, next, s);
    if (!observed(s)) return nullptr;
    if (blocks == 1) break;
    levels[++depth] = next; lengths[depth] = blocks; scratch += u64(blocks) * groups;
  }
  while (depth) {
    --depth;
    add_pair_offsets<<<grid(u64(lengths[depth]) * groups), threads>>>(levels[depth], levels[depth + 1],
      lengths[depth], lengths[depth + 1], u64(lengths[depth]) * groups, s);
    if (!observed(s)) return nullptr;
  }
  compact_groups<<<grid(n), threads>>>(p, y, source, w.prefix, w.groups, w.counts, length, groups, s);
  return observed(s) ? source : nullptr;
}

struct Row { double norm{}, brier{}; u32 chosen{}, exact{}, top[3]{}; };
struct Column {
  double rmse{}, mae{}, r2{}, loss{}, accuracy{}, brier{}, auc{}, ap{};
  u64 tp{}, fp{}, fn{}, positives{};
  u32 auc_available{};
};
__global__ void validate_inputs(MetricInput d, Status* s) {
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < u64(d.rows) * d.outputs; i += u64(gridDim.x) * threads) {
    const double p = d.predictions.data[i];
    if (!finite_metric(p) || (d.task != MetricTask::regression && (p < 0 || p > 1))) fail(s, input);
    if (d.task != MetricTask::multiclass || i < d.rows) {
      const double y = d.targets.data[i];
      if (!finite_metric(y) || (independent(d.task) && y != 0 && y != 1) ||
          (d.task == MetricTask::multiclass && (y < 0 || y >= d.outputs || floor(y) != y))) fail(s, input);
    }
    if (i < d.rows && d.weights.size && (!finite_metric(d.weights.data[i]) || d.weights.data[i] < 0)) fail(s, input);
  }
}
__global__ void weight_total(MetricInput d, double* total, Status* s) {
  if (s->errors) return;
  *total = d.weights.size ? positive_sum(d.rows, [=](u32 i) { return weight(d, i); }, s) : double(d.rows);
  if (!(*total > 0)) fail(s, input);
}
__global__ void row_statistics(MetricInput d, Row* out, Status* s) {
  if (s->errors) return;
  for (u64 row = u64(blockIdx.x) * threads + threadIdx.x; row < d.rows; row += u64(gridDim.x) * threads) {
    Row result{}; result.norm = 1; result.exact = 1;
    const u64 begin = row * d.outputs;
    u32 top[5]{}; u32 used = 0;
    for (u32 k = 0; k < d.outputs; ++k) {
      if (d.task == MetricTask::multiclass && d.predictions.data[begin + k] > d.predictions.data[begin + result.chosen]) result.chosen = k;
      if (independent(d.task)) {
        result.exact &= (d.predictions.data[begin + k] >= .5) == (d.targets.data[begin + k] == 1);
        if (d.profile == MetricProfile::real_data && d.task == MetricTask::multilabel) {
          u32 at = 0;
          while (at < used && d.predictions.data[begin + top[at]] >= d.predictions.data[begin + k]) ++at;
          if (at < 5) {
            for (u32 j = min(used, 4u); j > at; --j) top[j] = top[j - 1];
            top[at] = k; used = min(used + 1, 5u);
          }
        }
      }
    }
    if (d.task == MetricTask::multiclass) {
      const auto probability = [=](u32 k) { return d.predictions.data[begin + k]; };
      result.norm = d.profile == MetricProfile::synthetic ? positive_sum(d.outputs, probability, s) : numpy_sum(d.outputs, probability);
      if (!(result.norm > 0) || fabs(result.norm - 1) > d.probability_tolerance) fail(s, input);
      if (d.profile == MetricProfile::real_data) result.norm = 1;
      result.exact = result.chosen == u32(d.targets.data[row]);
      if (d.profile == MetricProfile::real_data) result.brier = numpy_sum(d.outputs, [=](u32 k) {
          return square(__dsub_rn(d.predictions.data[begin + k], double(k == u32(d.targets.data[row]))));
        });
    }
    if (used) {
      u32 hits = 0;
      for (u32 k = 0; k < used; ++k) {
        hits += d.targets.data[begin + top[k]] == 1;
        if (k == 0) result.top[0] = hits;
        if (k == 2) result.top[1] = hits;
        if (k == 4) result.top[2] = hits;
      }
    }
    out[row] = result;
  }
}
__device__ double correct_probability(MetricInput d, const Row* rows, u32 row, u32 output) {
  if (d.task == MetricTask::multiclass)
    return __ddiv_rn(d.predictions.data[u64(row) * d.outputs + u32(d.targets.data[row])], rows[row].norm);
  const u64 i = u64(row) * d.outputs + output;
  return d.targets.data[i] == 1 ? d.predictions.data[i] : __dsub_rn(1.0, d.predictions.data[i]);
}
__device__ double synthetic_loss(double p) { return -log(fmin(1 - clip, fmax(clip, p))); }
__global__ void column_statistics(MetricInput d, const Row* rows, const double* total, Column* out, Status* s) {
  if (s->errors) return;
  for (u64 output = u64(blockIdx.x) * threads + threadIdx.x; output < d.outputs; output += u64(gridDim.x) * threads) {
    Column result{};
    const auto w = [=](u32 row) { return weight(d, row); };
    if (d.task == MetricTask::regression) {
      const auto error = [=](u32 row) { return fabs(__dsub_rn(d.targets.data[u64(row) * d.outputs + output], d.predictions.data[u64(row) * d.outputs + output])); };
      if (d.profile == MetricProfile::synthetic) {
        result.rmse = scaled_mean<2>(d.rows, error, w, *total, s);
        result.mae = scaled_mean<1>(d.rows, error, w, *total, s);
      } else {
        const double mean = __ddiv_rn(detail::column_sum(d.rows, d.outputs, [=](u32 row) {
          return d.targets.data[u64(row) * d.outputs + output]; }), d.rows);
        const double numerator = detail::column_sum(d.rows, d.outputs, [=](u32 row) { return square(error(row)); });
        const double denominator = detail::column_sum(d.rows, d.outputs, [=](u32 row) {
          return square(__dsub_rn(d.targets.data[u64(row) * d.outputs + output], mean)); });
        result.r2 = !numerator ? 1 : !denominator ? 0 : __dsub_rn(1.0, __ddiv_rn(numerator, denominator));
      }
    } else {
      if (d.profile == MetricProfile::real_data) for (u32 row = 0; row < d.rows; ++row) {
        const bool positive = d.task == MetricTask::multiclass ? d.targets.data[row] == output : d.targets.data[u64(row) * d.outputs + output] == 1;
        const bool predicted = d.task == MetricTask::multiclass ? rows[row].chosen == output : d.predictions.data[u64(row) * d.outputs + output] >= .5;
        result.positives += positive; result.tp += positive && predicted;
        result.fp += !positive && predicted; result.fn += positive && !predicted;
      }
      if (d.profile == MetricProfile::synthetic && independent(d.task)) {
        result.loss = scaled_mean<1>(d.rows, [=](u32 row) { return synthetic_loss(correct_probability(d, rows, row, u32(output))); }, w, *total, s);
        result.brier = scaled_mean<1>(d.rows, [=](u32 row) {
          return square(__dsub_rn(d.predictions.data[u64(row) * d.outputs + output], d.targets.data[u64(row) * d.outputs + output])); }, w, *total, s);
        result.accuracy = __ddiv_rn(positive_sum(d.rows, [=](u32 row) {
          const u64 i = u64(row) * d.outputs + output;
          return (d.predictions.data[i] >= .5) == (d.targets.data[i] == 1) ? weight(d, row) : 0.0;
        }, s), *total);
      }
    }
    out[output] = result;
  }
}

__device__ double weighted_auc(MetricInput d, const u32* order, const Group* groups, u32 count, Status* s, bool& available) {
  const double positive = positive_sum(d.rows, [=](u32 row) {
    const u32 i = order[row]; return d.targets.data[i] == 1 ? weight(d, i / d.outputs) : 0.0;
  }, s);
  const double negative = positive_sum(d.rows, [=](u32 row) {
    const u32 i = order[row]; return d.targets.data[i] == 0 ? weight(d, i / d.outputs) : 0.0;
  }, s);
  available = positive > 0 && negative > 0;
  if (!available) return 0;
  detail::PositiveSum concordance;
  double before_negative = 0, correction = 0;
  for (u32 reverse = count; reverse; --reverse) {
    const u32 g = reverse - 1, begin = g ? groups[g - 1].end : 0, end = groups[g].end;
    const double p = positive_sum(end - begin, [=](u32 k) {
      const u32 i = order[begin + k];
      return d.targets.data[i] == 1 ? __ddiv_rn(weight(d, i / d.outputs), positive) : 0.0;
    }, s);
    const double n = positive_sum(end - begin, [=](u32 k) {
      const u32 i = order[begin + k];
      return d.targets.data[i] == 0 ? __ddiv_rn(weight(d, i / d.outputs), negative) : 0.0;
    }, s);
    concordance.add(__dmul_rn(p, __dadd_rn(before_negative, __dmul_rn(.5, n))), s);
    const double increment = __dsub_rn(n, correction), updated = __dadd_rn(before_negative, increment);
    correction = __dsub_rn(__dsub_rn(updated, before_negative), increment); before_negative = updated;
  }
  return fmin(1.0, fmax(0.0, concordance.value(s)));
}
__global__ void ranking_statistics(MetricInput d, const u32* order, RankMemory w,
                                   Column* out, bool pooled, Status* s) {
  if (s->errors) return;
  const u32 columns = pooled ? 1 : d.outputs, length = pooled ? d.rows * d.outputs : d.rows;
  for (u64 output = u64(blockIdx.x) * threads + threadIdx.x; output < columns; output += u64(gridDim.x) * threads) {
    const auto* groups = w.groups + output * length;
    const u32 count = w.counts[output], positives = groups[count - 1].positives, negatives = length - positives;
    const u32 slot = pooled ? d.outputs : u32(output);
    out[slot].positives = positives; out[slot].auc_available = positives && negatives;
    if (d.profile == MetricProfile::synthetic) {
      bool available;
      out[slot].auc = weighted_auc(d, order + output * length, groups, count, s, available);
      out[slot].auc_available = available;
      continue;
    }
    out[slot].ap = positives ? fmax(0.0, -numpy_sum(count, [=](u32 k) {
      const u32 g = count - 1 - k;
      const double current = __ddiv_rn(double(groups[g].positives), positives);
      const double next = g ? __ddiv_rn(double(groups[g - 1].positives), positives) : 0;
      return __dmul_rn(__dsub_rn(next, current), __ddiv_rn(double(groups[g].positives), groups[g].end));
    })) : 0;
    if (positives && negatives) {
      // Prefix storage has no readers after compaction; reuse it for ROC areas.
      auto* areas = reinterpret_cast<double*>(w.prefix) + output * length;
      double previous_fpr = 0, previous_tpr = 0;
      u32 used = 0;
      for (u32 g = 0; g < count; ++g) {
        const auto point = groups[g];
        bool keep = !g || g + 1 == count;
        if (!keep) {
          const auto a = groups[g - 1], b = groups[g + 1];
          keep = std::int64_t(b.positives) - 2 * std::int64_t(point.positives) + a.positives ||
            (std::int64_t(b.end) - b.positives) - 2 * (std::int64_t(point.end) - point.positives) + (std::int64_t(a.end) - a.positives);
        }
        if (!keep) continue;
        const double fpr = __ddiv_rn(double(point.end - point.positives), negatives), tpr = __ddiv_rn(double(point.positives), positives);
        areas[used++] = __ddiv_rn(__dmul_rn(__dsub_rn(fpr, previous_fpr), __dadd_rn(tpr, previous_tpr)), 2.0);
        previous_fpr = fpr; previous_tpr = tpr;
      }
      out[slot].auc = numpy_sum(used, [=](u32 i) { return areas[i]; });
    }
  }
}

struct Writer {
  MetricReport* report; Status* status;
  __device__ void add(MetricName name, double value, bool available = true, u32 output = aggregate_output) {
    if (available && !finite_metric(value)) fail(status, numeric);
    report->metrics.data[report->count++] = {name, output, value, u32(available)};
  }
};
__global__ void aggregate_statistics(MetricInput d, const Row* rows, const Column* columns,
    const double* total, double* compact, MetricReport* report, Status* s) {
  if (s->errors) return;
  Writer write{report, s};
  const u32 n = d.rows * d.outputs;
  const auto unit = [](u32) { return 1.0; };
  if (d.profile == MetricProfile::synthetic) {
    if (d.task == MetricTask::regression) {
      write.add(MetricName::rmse, scaled_mean<2>(d.outputs, [=](u32 k) { return columns[k].rmse; }, unit, d.outputs, s));
      write.add(MetricName::mae, scaled_mean<1>(d.outputs, [=](u32 k) { return columns[k].mae; }, unit, d.outputs, s));
      for (u32 k = 0; k < d.outputs; ++k) {
        write.add(MetricName::rmse, columns[k].rmse, true, k); write.add(MetricName::mae, columns[k].mae, true, k);
      }
    } else if (d.task == MetricTask::multiclass) {
      write.add(MetricName::logloss, scaled_mean<1>(d.rows, [=](u32 row) { return synthetic_loss(correct_probability(d, rows, row, 0)); },
        [=](u32 row) { return weight(d, row); }, *total, s));
      write.add(MetricName::accuracy, __ddiv_rn(positive_sum(d.rows, [=](u32 row) { return rows[row].exact ? weight(d, row) : 0.0; }, s), *total));
    } else {
      bool auc_available = true;
      for (u32 k = 0; k < d.outputs; ++k) auc_available &= columns[k].auc_available != 0;
      write.add(MetricName::logloss, scaled_mean<1>(d.outputs, [=](u32 k) { return columns[k].loss; }, unit, d.outputs, s));
      write.add(MetricName::accuracy, scaled_mean<1>(d.outputs, [=](u32 k) { return columns[k].accuracy; }, unit, d.outputs, s));
      write.add(MetricName::brier, scaled_mean<1>(d.outputs, [=](u32 k) { return columns[k].brier; }, unit, d.outputs, s));
      write.add(MetricName::auc, auc_available ? scaled_mean<1>(d.outputs, [=](u32 k) { return columns[k].auc; }, unit, d.outputs, s) : 0, auc_available);
      if (d.task == MetricTask::multilabel) for (u32 k = 0; k < d.outputs; ++k) {
        write.add(MetricName::logloss, columns[k].loss, true, k); write.add(MetricName::accuracy, columns[k].accuracy, true, k);
        write.add(MetricName::brier, columns[k].brier, true, k); write.add(MetricName::auc, columns[k].auc, columns[k].auc_available, k);
      }
    }
    return;
  }
  if (d.task == MetricTask::regression) {
    const double mse = __ddiv_rn(numpy_sum(n, [=](u32 i) { return square(__dsub_rn(d.predictions.data[i], d.targets.data[i])); }), n);
    write.add(MetricName::mse, mse); write.add(MetricName::rmse, sqrt(mse));
    write.add(MetricName::mae, __ddiv_rn(numpy_sum(n, [=](u32 i) { return fabs(__dsub_rn(d.predictions.data[i], d.targets.data[i])); }), n));
    write.add(MetricName::r2, __ddiv_rn(numpy_sum(d.outputs, [=](u32 k) { return columns[k].r2; }), d.outputs), d.rows > 1);
    return;
  }
  if (d.task == MetricTask::multiclass) {
    write.add(MetricName::logloss, -__ddiv_rn(numpy_sum(d.rows, [=](u32 row) {
      return log(fmin(1.0, fmax(clip, d.predictions.data[u64(row) * d.outputs + u32(d.targets.data[row])]))); }), d.rows));
    u64 correct = 0; for (u32 row = 0; row < d.rows; ++row) correct += rows[row].exact;
    write.add(MetricName::accuracy, __ddiv_rn(double(correct), d.rows));
    write.add(MetricName::macro_f1, __ddiv_rn(numpy_sum(d.outputs, [=](u32 k) { return f1(columns[k].tp, columns[k].fp, columns[k].fn); }), d.outputs));
    write.add(MetricName::brier, __ddiv_rn(numpy_sum(d.rows, [=](u32 row) { return rows[row].brier; }), d.rows));
    return;
  }
  write.add(MetricName::logloss, -__ddiv_rn(numpy_sum(n, [=](u32 i) {
    const double p = fmin(1 - clip, fmax(clip, d.predictions.data[i])), y = d.targets.data[i];
    return __dadd_rn(__dmul_rn(y, log(p)), __dmul_rn(__dsub_rn(1.0, y), log1p(-p)));
  }), n));
  write.add(MetricName::brier, __ddiv_rn(numpy_sum(n, [=](u32 i) { return square(__dsub_rn(d.predictions.data[i], d.targets.data[i])); }), n));
  u64 tp = 0, fp = 0, fn = 0, exact = 0;
  for (u32 k = 0; k < d.outputs; ++k) { tp += columns[k].tp; fp += columns[k].fp; fn += columns[k].fn; }
  for (u32 row = 0; row < d.rows; ++row) exact += rows[row].exact;
  write.add(MetricName::hamming_loss, __ddiv_rn(double(fp + fn), n));
  write.add(MetricName::exact_match, __ddiv_rn(double(exact), d.rows));
  if (d.outputs == 1) {
    write.add(MetricName::accuracy, __ddiv_rn(double(n - fp - fn), n)); write.add(MetricName::f1, f1(tp, fp, fn));
    write.add(MetricName::auc, columns[0].auc, columns[0].auc_available); write.add(MetricName::average_precision, columns[0].ap);
  } else {
    write.add(MetricName::micro_f1, f1(tp, fp, fn));
    write.add(MetricName::macro_f1, __ddiv_rn(numpy_sum(d.outputs, [=](u32 k) { return f1(columns[k].tp, columns[k].fp, columns[k].fn); }), d.outputs));
    write.add(MetricName::micro_ap, columns[d.outputs].ap); write.add(MetricName::micro_auc, columns[d.outputs].auc, columns[d.outputs].auc_available);
    u32 present = 0, varying = 0;
    for (u32 k = 0; k < d.outputs; ++k) if (columns[k].positives) compact[present++] = columns[k].ap;
    write.add(MetricName::macro_ap, present ? __ddiv_rn(numpy_sum(present, [=](u32 k) { return compact[k]; }), present) : 0, present);
    for (u32 k = 0; k < d.outputs; ++k) if (columns[k].auc_available) compact[varying++] = columns[k].auc;
    write.add(MetricName::macro_auc, varying ? __ddiv_rn(numpy_sum(varying, [=](u32 k) { return compact[k]; }), varying) : 0, varying);
    report->ap_outputs = present; report->auc_outputs = varying;
    const MetricName names[3]{MetricName::precision_at_1, MetricName::precision_at_3, MetricName::precision_at_5};
    for (u32 j = 0; j < 3; ++j) {
      const u32 k = 2 * j + 1;
      if (k > d.outputs) continue;
      u64 hits = 0; for (u32 row = 0; row < d.rows; ++row) hits += rows[row].top[j];
      write.add(names[j], __ddiv_rn(double(hits), double(u64(d.rows) * k)));
    }
  }
}

__device__ u64 metric_count(MetricInput d) {
  if (d.profile == MetricProfile::synthetic) {
    if (d.task == MetricTask::regression) return 2 + 2ULL * d.outputs;
    if (d.task == MetricTask::multiclass) return 2;
    return d.task == MetricTask::binary ? 4 : 4 + 4ULL * d.outputs;
  }
  if (!independent(d.task)) return 4;
  return d.outputs == 1 ? 8 : 11 + u64(d.outputs >= 3) + u64(d.outputs >= 5);
}
__device__ bool valid_shape(MetricInput d) {
  const u64 n = u64(d.rows) * d.outputs;
  return d.rows && d.outputs && n <= UINT32_MAX && u32(d.task) <= 3 && u32(d.profile) <= 1 &&
    (!independent(d.task) || u64(d.outputs) * ceil_div(d.rows, threads) <= INT32_MAX) &&
    (d.task != MetricTask::binary || d.outputs == 1) && (d.task != MetricTask::multiclass || d.outputs >= 2) &&
    finite_metric(d.probability_tolerance) && d.probability_tolerance >= 0 && d.probability_tolerance < 1 &&
    contains(d.predictions, n) && contains(d.targets, d.task == MetricTask::multiclass ? d.rows : n) &&
    (!d.weights.size || (d.profile == MetricProfile::synthetic && contains(d.weights, d.rows)));
}

__device__ OperatingPoint point(u64 positives, u64 negatives, u64 tp, u64 fp, double threshold) {
  return {threshold, __ddiv_rn(double(tp), double(positives)), __ddiv_rn(double(fp), double(negatives)),
    tp + fp ? __ddiv_rn(double(tp), double(tp + fp)) : 0, f1(tp, fp, positives - tp),
    tp, fp, positives - tp, negatives - fp, positives, negatives, u32(20 * fp <= negatives)};
}
__global__ void signal_result(SignalInput d, ThresholdMode mode, double threshold, const u32* order,
                              RankMemory w, SignalReport* report, Status* s) {
  if (s->errors) return;
  const u32 count = w.counts[0], positives = w.groups[count - 1].positives, negatives = d.size - positives;
  if (!positives || !negatives) { fail(s, input); return; }
  u32 fixed_tp = 0, fixed_fp = 0, selected_tp = 0, selected_fp = 0;
  double selected = mode == ThresholdMode::validation ? nextafter(d.predictions.data[order[0]], double(INFINITY)) : threshold;
  for (u32 g = 0; g < count; ++g) {
    const auto v = w.groups[g]; const u32 fp = v.end - v.positives;
    const double score = d.predictions.data[order[v.end - 1]];
    if (score >= .5) { fixed_tp = v.positives; fixed_fp = fp; }
    if (mode == ThresholdMode::validation) {
      if (20ULL * fp <= negatives && v.positives > selected_tp) { selected_tp = v.positives; selected_fp = fp; selected = score; }
    } else if (score >= threshold) { selected_tp = v.positives; selected_fp = fp; }
  }
  *report = {point(positives, negatives, fixed_tp, fixed_fp, .5), point(positives, negatives, selected_tp, selected_fp, selected), mode};
}
__device__ bool minimize(MetricName name) {
  return name == MetricName::mse || name == MetricName::rmse || name == MetricName::mae ||
    name == MetricName::logloss || name == MetricName::brier || name == MetricName::hamming_loss;
}
__global__ void metric_gate(const MetricReport* reference, const MetricReport* candidate,
    MetricVerdict* verdicts, MetricGate* result, Status* s) {
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < reference->count; i += u64(gridDim.x) * threads) {
    const auto a = reference->metrics.data[i], b = candidate->metrics.data[i];
    if (a.name != b.name || a.output != b.output || u32(a.name) > u32(MetricName::precision_at_5) ||
        a.available > 1 || b.available > 1 || a.available != b.available ||
        (a.available && (!finite_metric(a.value) || !finite_metric(b.value)))) {
      verdicts[i] = MetricVerdict::invalid; atomicAdd(&result->invalid, 1); fail(s, input);
    } else if (!a.available) { verdicts[i] = MetricVerdict::not_applicable; atomicAdd(&result->unavailable, 1); }
    else {
      const bool worse = minimize(a.name) ? b.value > a.value : b.value < a.value;
      verdicts[i] = worse ? MetricVerdict::regression : MetricVerdict::pass;
      atomicAdd(&result->checked, 1); if (worse) atomicAdd(&result->regressions, 1);
    }
  }
}
}

__device__ cudaError_t evaluate_metrics(MetricInput d, MetricReport* report, Workspace workspace, Status* s) { using namespace metrics_impl;
  if (!s) return cudaErrorInvalidValue;
  if (!pointer(report) || !valid_shape(d)) { fail(s, shape); return finish(s); }
  const u64 required = metric_count(d);
  if (required > UINT32_MAX || !contains(report->metrics, required)) { fail(s, capacity); return finish(s); }
  Arena arena{workspace};
  RankMemory rankings{};
  if (independent(d.task)) rankings = ranking_memory(arena, d.rows, d.outputs);
  auto* rows = arena.take<Row>(d.task == MetricTask::regression ? 0 : d.rows);
  auto* columns = arena.take<Column>(u64(d.outputs) + 1);
  auto* total = arena.take<double>(1);
  auto* compact = arena.take<double>(d.outputs);
  if (!arena.fits(s)) return finish(s);
  report->count = report->ap_outputs = report->auc_outputs = 0;
  report->outputs = d.outputs; report->task = d.task; report->profile = d.profile;
  validate_inputs<<<grid(u64(d.rows) * d.outputs), threads>>>(d, s); observed(s);
  weight_total<<<1, 1>>>(d, total, s); observed(s);
  if (d.task != MetricTask::regression) { row_statistics<<<grid(d.rows), threads>>>(d, rows, s); observed(s); }
  if (d.task != MetricTask::multiclass || d.profile == MetricProfile::real_data) {
    column_statistics<<<grid(d.outputs), threads>>>(d, rows, total, columns, s); observed(s);
  }
  if (independent(d.task)) {
    if (const auto* sorted = rank(d.predictions.data, d.targets.data, d.rows, d.outputs, d.outputs, false, rankings, s)) {
      ranking_statistics<<<grid(d.outputs), threads>>>(d, sorted, rankings, columns, false, s); observed(s);
    }
    if (d.profile == MetricProfile::real_data && d.outputs > 1) {
      if (const auto* sorted = rank(d.predictions.data, d.targets.data, d.rows * d.outputs, 1, d.outputs, true, rankings, s)) {
        ranking_statistics<<<1, 1>>>(d, sorted, rankings, columns, true, s); observed(s);
      }
    }
  }
  aggregate_statistics<<<1, 1>>>(d, rows, columns, total, compact, report, s); observed(s);
  return finish(s);
}
__device__ cudaError_t compare_metrics(const MetricReport* reference, const MetricReport* candidate,
    Array<MetricVerdict> verdicts, MetricGate* result, Status* s) { using namespace metrics_impl;
  if (!s) return cudaErrorInvalidValue;
  if (!pointer(reference) || !pointer(candidate) || !pointer(result)) { fail(s, shape); return finish(s); }
  if (!reference->count || reference->count != candidate->count || reference->outputs != candidate->outputs ||
      reference->task != candidate->task || reference->profile != candidate->profile ||
      reference->ap_outputs != candidate->ap_outputs || reference->auc_outputs != candidate->auc_outputs ||
      !contains(reference->metrics, reference->count) || !contains(candidate->metrics, candidate->count) || !contains(verdicts, reference->count)) {
    fail(s, shape); return finish(s);
  }
  *result = {};
  metric_gate<<<grid(reference->count), threads>>>(reference, candidate, verdicts.data, result, s); observed(s);
  return finish(s);
}
__device__ cudaError_t signal_metrics(SignalInput d, ThresholdMode mode, double threshold,
    SignalReport* report, Workspace workspace, Status* s) { using namespace metrics_impl;
  if (!s) return cudaErrorInvalidValue;
  const MetricInput input{d.labels, d.predictions, {}, d.size, 1, MetricTask::binary, MetricProfile::real_data};
  if (!pointer(report) || !valid_shape(input) || u32(mode) > 1 || (mode == ThresholdMode::frozen && !finite_metric(threshold))) {
    fail(s, shape); return finish(s);
  }
  Arena arena{workspace}; const auto w = ranking_memory(arena, d.size, 1);
  if (!arena.fits(s)) return finish(s);
  validate_inputs<<<grid(d.size), threads>>>(input, s); observed(s);
  if (const auto* sorted = rank(d.predictions.data, d.labels.data, d.size, 1, 1, true, w, s)) {
    signal_result<<<1, 1>>>(d, mode, threshold, sorted, w, report, s); observed(s);
  }
  return finish(s);
}
}

// GH_SOURCE_CATEGORY: production
// reference
namespace gh {
namespace reference_impl {
constexpr u32 limbs = 36;
struct Integer {
  u32 word[limbs]{};
  __device__ int bits() const {
    for (int i = limbs - 1; i >= 0; --i) if (word[i]) return i * 32 + 32 - __clz(word[i]);
    return 0;
  }
  __device__ bool times_five() {
    u64 carry = 0;
    for (u32 i = 0; i < limbs; ++i) { const u64 v = u64(word[i]) * 5 + carry; word[i] = u32(v); carry = v >> 32; }
    return !carry;
  }
  __device__ bool shift(u32 n) {
    if (n > limbs * 32 || bits() + n > limbs * 32) return false;
    const u32 whole = n / 32, part = n % 32;
    for (int i = limbs - 1; i >= 0; --i) {
      u32 v = u32(i) >= whole ? word[i - whole] << part : 0;
      if (part && u32(i) > whole) v |= word[i - whole - 1] >> (32 - part);
      word[i] = v;
    }
    return true;
  }
  __device__ int compare(const Integer& b) const {
    for (int i = limbs - 1; i >= 0; --i) if (word[i] != b.word[i]) return word[i] > b.word[i] ? 1 : -1;
    return 0;
  }
  __device__ void subtract(const Integer& b) {
    u64 borrow = 0;
    for (u32 i = 0; i < limbs; ++i) {
      const u64 a = word[i], sub = u64(b.word[i]) + borrow;
      word[i] = u32(a - sub); borrow = a < sub;
    }
  }
};
__device__ bool digit(char c) { return c >= '0' && c <= '9'; }
__device__ u64 bits(double x) { return __double_as_longlong(x); }
__device__ bool finite(double x) { return (bits(x) & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL; }
__device__ bool decimal_parts(Array<const char> text, u64& mantissa, int& exponent, bool& negative) {
  if (!text.size || text.size > 768 || !contains(text, text.size)) return false;
  u64 p = 0; negative = text.data[p] == '-'; p += negative;
  if (p == text.size || !digit(text.data[p])) return false;
  u32 significant = 0, fraction = 0;
  const auto append = [&](char c) {
    if (mantissa || c != '0') { if (++significant > 19) return false; mantissa = mantissa * 10 + u32(c - '0'); }
    return true;
  };
  if (text.data[p] == '0') ++p;
  else while (p < text.size && digit(text.data[p])) if (!append(text.data[p++])) return false;
  if (p < text.size && text.data[p] == '.') {
    ++p; const u64 first = p;
    while (p < text.size && digit(text.data[p])) { ++fraction; if (!append(text.data[p++])) return false; }
    if (p == first) return false;
  }
  int explicit_exponent = 0;
  if (p < text.size && (text.data[p] == 'e' || text.data[p] == 'E')) {
    ++p; bool minus = false;
    if (p < text.size && (text.data[p] == '+' || text.data[p] == '-')) minus = text.data[p++] == '-';
    const u64 first = p;
    while (p < text.size && digit(text.data[p])) {
      explicit_exponent = explicit_exponent * 10 + text.data[p++] - '0';
      if (explicit_exponent > 4096) return false;
    }
    if (p == first) return false;
    if (minus) explicit_exponent = -explicit_exponent;
  }
  exponent = explicit_exponent - int(fraction);
  return p == text.size && exponent >= -342 && exponent <= 308;
}
struct Text { char data[80]{}; u32 size{}; };
__device__ bool equal(const Text& a, const char* b) {
  u32 i = 0; while (i < a.size && b[i] && a.data[i] == b[i]) ++i;
  return i == a.size && !b[i];
}
struct Json {
  Array<const std::byte> bytes;
  u64 at{};
  __device__ char peek() const { return at < bytes.size ? char(bytes.data[at]) : '\0'; }
  __device__ void spaces() { while (peek() == ' ' || peek() == '\n' || peek() == '\r' || peek() == '\t') ++at; }
  __device__ bool take(char c) { spaces(); if (peek() != c || at == bytes.size) return false; ++at; return true; }
  __device__ bool literal(const char* text) {
    while (*text) if (at == bytes.size || peek() != *text++) return false; else ++at;
    return true;
  }
  __device__ bool string(Text& text) {
    text.size = 0;
    if (!take('"')) return false;
    while (at < bytes.size && peek() != '"') {
      u32 c = u32(static_cast<unsigned char>(peek())); ++at;
      if (c < 32 || c > 127) return false;
      if (c == '\\') {
        if (at == bytes.size) return false;
        c = u32(static_cast<unsigned char>(peek())); ++at;
        if (c == 'u') {
          c = 0;
          for (u32 i = 0; i < 4; ++i) {
            const char h = peek(); const int v = digit(h) ? h - '0' : h >= 'a' && h <= 'f' ? h - 'a' + 10 : h >= 'A' && h <= 'F' ? h - 'A' + 10 : -1;
            if (at == bytes.size || v < 0) return false;
            ++at; c = c * 16 + u32(v);
          }
          if (c > 127) return false;
        } else if (c == 'b') c = '\b'; else if (c == 'f') c = '\f';
        else if (c == 'n') c = '\n'; else if (c == 'r') c = '\r'; else if (c == 't') c = '\t';
        else if (c != '"' && c != '\\' && c != '/') return false;
      }
      if (text.size == sizeof(text.data)) return false;
      text.data[text.size++] = char(c);
    }
    return take('"');
  }
  __device__ bool number(double& value, bool& available) {
    spaces(); available = peek() != 'n';
    if (!available) { value = 0; return literal("null"); }
    const u64 first = at;
    while (digit(peek()) || peek() == '-' || peek() == '+' || peek() == '.' || peek() == 'e' || peek() == 'E') ++at;
    return parse_decimal({reinterpret_cast<const char*>(bytes.data + first), at - first}, &value) && finite(value);
  }
};
__device__ int metric_key(const Text& key) {
  if (equal(key, "mse")) return int(MetricName::mse);
  if (equal(key, "rmse")) return int(MetricName::rmse);
  if (equal(key, "mae")) return int(MetricName::mae);
  if (equal(key, "r2")) return int(MetricName::r2);
  if (equal(key, "log_loss")) return int(MetricName::logloss);
  if (equal(key, "accuracy")) return int(MetricName::accuracy);
  if (equal(key, "brier")) return int(MetricName::brier);
  if (equal(key, "roc_auc")) return int(MetricName::auc);
  if (equal(key, "f1")) return int(MetricName::f1);
  if (equal(key, "average_precision")) return int(MetricName::average_precision);
  if (equal(key, "hamming_loss")) return int(MetricName::hamming_loss);
  if (equal(key, "exact_match_accuracy")) return int(MetricName::exact_match);
  if (equal(key, "micro_f1")) return int(MetricName::micro_f1);
  if (equal(key, "macro_f1")) return int(MetricName::macro_f1);
  if (equal(key, "micro_ap")) return int(MetricName::micro_ap);
  if (equal(key, "macro_ap")) return int(MetricName::macro_ap);
  if (equal(key, "macro_auc")) return int(MetricName::macro_auc);
  if (equal(key, "precision_at_1")) return int(MetricName::precision_at_1);
  if (equal(key, "precision_at_3")) return int(MetricName::precision_at_3);
  if (equal(key, "precision_at_5")) return int(MetricName::precision_at_5);
  return -1;
}
__device__ u32 flag(MetricName name) { return 1u << u32(name); }
struct Set {
  double values[21]{}, selection_value{}, clip{};
  u32 present{}, available{}, extras{}, ap_outputs{}, auc_outputs{};
  int selection{-1};
};
__device__ bool unique(u32& set, u32 bit) { if (set & bit) return false; set |= bit; return true; }
__device__ bool metrics(Json& json, Set& set) {
  if (!json.take('{')) return false;
  do {
    Text key; if (!json.string(key) || !json.take(':')) return false;
    const int name = metric_key(key);
    if (name >= 0) {
      if (!unique(set.present, 1u << name)) return false;
      bool available; if (!json.number(set.values[name], available)) return false;
      if (available) set.available |= 1u << name;
    } else if (equal(key, "selection_metric")) {
      Text value; if (!unique(set.extras, 1) || !json.string(value)) return false;
      set.selection = metric_key(value);
    } else {
      const u32 bit = equal(key, "selection_value") ? 2 : equal(key, "log_clip_epsilon") ? 4 :
        equal(key, "macro_ap_labels") ? 8 : equal(key, "macro_auc_labels") ? 16 : 0;
      double value; bool available;
      if (!bit || !unique(set.extras, bit) || !json.number(value, available) || !available) return false;
      if (bit == 2) set.selection_value = value;
      else if (bit == 4) set.clip = value;
      else {
        if (value < 0 || value > UINT32_MAX || double(u32(value)) != value) return false;
        if (bit == 8) set.ap_outputs = u32(value); else set.auc_outputs = u32(value);
      }
    }
    if (json.take('}')) return true;
  } while (json.take(','));
  return false;
}
__device__ bool hash(const Text& text) {
  if (text.size != 64) return false;
  for (u32 i = 0; i < 64; ++i) if (!digit(text.data[i]) && !(text.data[i] >= 'a' && text.data[i] <= 'f')) return false;
  return true;
}
__device__ bool decode(Json& json, Set& values, Set& baseline) {
  u32 seen = 0;
  if (!json.take('{')) return false;
  do {
    Text key; if (!json.string(key) || !json.take(':')) return false;
    const u32 bit = equal(key, "reference") ? 1 : equal(key, "fixture_sha256") ? 2 :
      equal(key, "training_fixture_sha256") ? 4 : equal(key, "predictions_sha256") ? 8 :
      equal(key, "metrics") ? 16 : equal(key, "training_mean_baseline") ? 32 : 0;
    if (!bit || !unique(seen, bit)) return false;
    if (bit >= 16) { if (!metrics(json, bit == 16 ? values : baseline)) return false; }
    else {
      Text value; if (!json.string(value) || (bit == 1 ? !equal(value, "CPU float64 common metric implementation") : !hash(value))) return false;
    }
    if (json.take('}')) { json.spaces(); return seen == 63 && json.at == json.bytes.size; }
  } while (json.take(','));
  return false;
}
__device__ u32 required(MetricTask task, u32 outputs) {
  if (task == MetricTask::regression) return flag(MetricName::mse) | flag(MetricName::rmse) | flag(MetricName::mae) | flag(MetricName::r2);
  u32 mask = flag(MetricName::logloss) | flag(MetricName::brier);
  if (task == MetricTask::multiclass) return mask | flag(MetricName::accuracy) | flag(MetricName::macro_f1);
  mask |= flag(MetricName::hamming_loss) | flag(MetricName::exact_match);
  if (outputs == 1) return mask | flag(MetricName::accuracy) | flag(MetricName::f1) | flag(MetricName::auc) | flag(MetricName::average_precision);
  mask |= flag(MetricName::micro_f1) | flag(MetricName::macro_f1) | flag(MetricName::micro_ap) |
    flag(MetricName::macro_ap) | flag(MetricName::macro_auc) | flag(MetricName::precision_at_1);
  if (outputs >= 3) mask |= flag(MetricName::precision_at_3);
  if (outputs >= 5) mask |= flag(MetricName::precision_at_5);
  return mask;
}
__device__ bool schema(const Set& set, const MetricReport& report, u32 mask) {
  const bool regression = report.task == MetricTask::regression;
  const bool multi = report.task == MetricTask::multilabel && report.outputs > 1;
  const u32 extras = regression ? 3 : multi ? 31 : 7;
  const auto selected = regression ? MetricName::mse : MetricName::logloss;
  return set.present == mask && set.extras == extras && set.selection == int(selected) &&
    (set.available & flag(selected)) && bits(set.selection_value) == bits(set.values[u32(selected)]) &&
    (regression || bits(set.clip) == bits(1e-15)) && set.ap_outputs <= report.outputs && set.auc_outputs <= set.ap_outputs &&
    (!multi || (bool(set.available & flag(MetricName::macro_ap)) == bool(set.ap_outputs) &&
      bool(set.available & flag(MetricName::macro_auc)) == bool(set.auc_outputs)));
}
}

__device__ bool parse_decimal(Array<const char> text, double* output) { using namespace reference_impl;
  if (!output) return false;
  u64 mantissa = 0; int decimal = 0; bool negative = false;
  if (!decimal_parts(text, mantissa, decimal, negative)) return false;
  const u64 sign = u64(negative) << 63;
  if (!mantissa) { *output = __longlong_as_double(sign); return true; }
  Integer numerator, denominator; numerator.word[0] = u32(mantissa); numerator.word[1] = u32(mantissa >> 32); denominator.word[0] = 1;
  for (int i = 0; i < (decimal < 0 ? -decimal : decimal); ++i)
    if (!(decimal < 0 ? denominator.times_five() : numerator.times_five())) return false;
  int k = numerator.bits() - denominator.bits();
  Integer trial = k >= 0 ? denominator : numerator;
  if (!trial.shift(k >= 0 ? u32(k) : u32(-k))) return false;
  if (k >= 0 ? numerator.compare(trial) < 0 : trial.compare(denominator) < 0) --k;
  int exponent = k + decimal;
  if (exponent > 1023) { *output = __longlong_as_double(sign | 0x7ff0000000000000ULL); return true; }
  if (exponent < -1075) { *output = __longlong_as_double(sign); return true; }
  const int quantum = exponent >= -1022 ? exponent - 52 : -1074, shift = decimal - quantum;
  if (!(shift >= 0 ? numerator.shift(u32(shift)) : denominator.shift(u32(-shift)))) return false;
  u64 significand = 0;
  const int top = numerator.bits() - denominator.bits();
  if (top > 53) return false;
  for (int b = top; b >= 0; --b) {
    trial = denominator; if (!trial.shift(u32(b))) return false;
    if (numerator.compare(trial) >= 0) { numerator.subtract(trial); significand |= 1ULL << b; }
  }
  if (!numerator.shift(1)) return false;
  const int halfway = numerator.compare(denominator);
  significand += halfway > 0 || (halfway == 0 && (significand & 1));
  u64 result;
  if (exponent < -1022) result = significand;
  else {
    if (significand == 1ULL << 53) { significand >>= 1; ++exponent; }
    result = exponent > 1023 ? 0x7ff0000000000000ULL : (u64(exponent + 1023) << 52) | (significand & 0xfffffffffffffULL);
  }
  *output = __longlong_as_double(sign | result);
  return true;
}
__device__ cudaError_t decode_metric_reference(Array<const std::byte> bytes, const MetricReport* computed,
    MetricReport* reference, MetricReport* aligned, Status* status) { using namespace reference_impl;
  if (!status) return cudaErrorInvalidValue;
  if (!contains(bytes, bytes.size) || !bytes.size || bytes.size > (1u << 20) ||
      !contains(Array<const MetricReport>{computed, 1}, 1) || !contains(Array<MetricReport>{reference, 1}, 1) ||
      !contains(Array<MetricReport>{aligned, 1}, 1)) { fail(status, shape); return finish(status); }
  const auto report = *computed;
  if (report.profile != MetricProfile::real_data || u32(report.task) > 3 || !report.outputs ||
      (report.task == MetricTask::binary && report.outputs != 1) || (report.task == MetricTask::multiclass && report.outputs < 2) ||
      !report.count || report.count > 21 || !contains(report.metrics, report.count)) { fail(status, shape); return finish(status); }
  Set values, baseline; Json json{bytes};
  const u32 mask = required(report.task, report.outputs);
  if (!decode(json, values, baseline) || !schema(values, report, mask) || !schema(baseline, report, mask)) {
    fail(status, input); return finish(status);
  }
  const u32 count = __popc(mask);
  if (!contains(reference->metrics, count) || !contains(aligned->metrics, count)) { fail(status, capacity); return finish(status); }
  u32 seen = 0;
  for (u32 i = 0; i < report.count; ++i) {
    const auto m = report.metrics.data[i];
    if (u32(m.name) > 20 || m.output != aggregate_output || m.available > 1 ||
        !unique(seen, flag(m.name)) || (m.available && !finite(m.value))) { fail(status, input); return finish(status); }
  }
  const u32 extra = report.task == MetricTask::multilabel && report.outputs > 1 ? flag(MetricName::micro_auc) : 0;
  if (seen != (mask | extra)) { fail(status, input); return finish(status); }
  reference->count = aligned->count = 0;
  reference->outputs = aligned->outputs = report.outputs;
  reference->task = aligned->task = report.task;
  reference->profile = aligned->profile = report.profile;
  reference->ap_outputs = values.ap_outputs; reference->auc_outputs = values.auc_outputs;
  aligned->ap_outputs = report.ap_outputs; aligned->auc_outputs = report.auc_outputs;
  for (u32 i = 0; i < report.count; ++i) {
    const auto m = report.metrics.data[i];
    if (!(mask & flag(m.name))) continue;
    reference->metrics.data[reference->count++] = {m.name, aggregate_output, values.values[u32(m.name)], u32(bool(values.available & flag(m.name)))};
    aligned->metrics.data[aligned->count++] = m;
  }
  return finish(status);
}
}

// GH_SOURCE_CATEGORY: production
// observe
namespace gh::observe {
namespace {
__device__ u64 tick() { u64 value; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value)); return value; }
__global__ void append(Trace trace, Stage stage, u32 iteration, bool end, Status* status) {
  if (!trace.count || !contains(trace.records,trace.records.size)) { fail(status,extent); return; }
  const u32 slot = *trace.count;
  if (slot >= trace.records.size || slot == UINT32_MAX) { fail(status,capacity); return; }
  trace.records.data[slot] = {tick(),stage,iteration,end};
  *trace.count = slot + 1;
}
__global__ void stamp(Array<Sample> raw, u32 ordinal, bool end, Status* status) {
  if (ordinal >= samples || !contains(raw,u64(ordinal)+1)) { fail(status,capacity); return; }
  if (end) raw.data[ordinal].end = tick(); else raw.data[ordinal].begin = tick();
}
__global__ void paired(Array<const Sample> raw, Summary* output, Status* status) {
  if (!output || !contains(raw,samples)) { fail(status,capacity); return; }
  u64 minimum = UINT64_MAX;
  for (u32 i = 0; i < samples; ++i) {
    const auto value = raw.data[i];
    if (value.end <= value.begin || (i && value.begin < raw.data[i-1].end)) { fail(status,numeric); return; }
    minimum = min(minimum,value.end-value.begin);
  }
  double logs[pairs], mean = 0;
  for (u32 pair = 0; pair < pairs; ++pair) {
    const u32 first = 2 * (warmups + pair);
    const u32 a = first + schedule(first).variant, b = first + 1 - schedule(first).variant;
    const double x = double(raw.data[a].end - raw.data[a].begin);
    const double y = double(raw.data[b].end - raw.data[b].begin);
    logs[pair] = log(x/y);
    mean += logs[pair] / pairs;
  }
  double variance = 0;
  for (const double value : logs) variance += (value-mean)*(value-mean) / (pairs-1);
  const double sd = sqrt(variance), margin = 2.1447866879169273 * sd / sqrt(double(pairs));
  *output = {exp(mean),exp(mean-margin),exp(mean+margin),sd,minimum};
}
__device__ cudaError_t checked(Status* status) {
  const auto result = cudaGetLastError();
  if (result != cudaSuccess) fail(status,runtime);
  return result;
}
}
__device__ cudaError_t enqueue(Trace trace, Stage stage, u32 iteration, bool end, Status* status) {
  if (!status) return cudaErrorInvalidValue;
  append<<<1,1>>>(trace,stage,iteration,end,status);
  return checked(status);
}
__device__ cudaError_t boundary(Array<Sample> raw, u32 ordinal, bool end, Status* status) {
  if (!status) return cudaErrorInvalidValue;
  stamp<<<1,1,0,end ? cudaStreamTailLaunch : nullptr>>>(raw,ordinal,end,status);
  return checked(status);
}
__device__ cudaError_t summarize(Array<const Sample> raw, Summary* output, Status* status) {
  if (!status) return cudaErrorInvalidValue;
  paired<<<1,1>>>(raw,output,status);
  const auto launched = checked(status), completed = finish(status);
  return launched != cudaSuccess ? launched : completed;
}
}

// GH_SOURCE_CATEGORY: production
// learn
namespace gh {
namespace learn_impl {
using namespace detail;
constexpr unsigned threads = 256;
// A failed submission ends this chain. Callers propagate failure without a
// second completion; already queued children drain before the completion tail.
__device__ cudaError_t checked_launch(Status* status, cudaError_t error) {
  if (error != cudaSuccess) { fail(status, runtime); finish(status); }
  return error;
}
__device__ bool launch_ok(Status* status) {
  return checked_launch(status, cudaGetLastError()) == cudaSuccess;
}
struct Calibration { u32 ordinal{}; u64 begin{}, ticks[2][5]{}; };
__device__ u64 timestamp() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__global__ void begin_sample(Calibration* sample) { sample->begin = timestamp(); }
__device__ unsigned grid(u64 n, unsigned block = threads) {
  return static_cast<unsigned>(min(u64{65535}, max(u64{1}, ceil_div(n, block))));
}
template<unsigned O> struct State {
  Dataset data;
  Schema schema;
  const std::uint16_t* bins{};
  TrainConfig config;
  Training* output{};
  Status* status{};
  double *derivative[O]{}, *initial{}, *weight{}, *loss_partial{};
  Stats<O> *hist{}, *roots{};
  unsigned long long* counts{};
  Choice *candidates{}, *winners{}, *root_winners{};
  Node* nodes{};
  std::int32_t *assignments{}, *frontier[2]{};
  u32 *active{}, *used{}, *next{}, *old_used{}, *prefix{}, *block_prefix{};
  u64* export_begin{};
  Histogram* root_policy{};
  Calibration* calibration{};
  u32 outputs{}, derivative_capacity{}, tile_capacity{}, batch_capacity{};
  u32 frontier_capacity{}, node_capacity{}, init_chunks{}, loss_blocks{}, scan_blocks{};
  u32 round{}, tile_begin{}, tile_live{}, tree_begin{}, batch_live{}, depth{}, side{};
};
template<unsigned O> __device__ bool mark(State<O>* s, observe::Stage stage, bool end) {
  if constexpr (observe::enabled) if (s->output->trace.count) {
    return checked_launch(s->status, observe::mark(s->output->trace, stage, s->round, end, s->status)) == cudaSuccess;
  }
  return true;
}
template<class T> __global__ void fill(T* values, u64 n, T value) {
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += u64(gridDim.x) * blockDim.x) values[i] = value;
}
template<unsigned O> __device__ double weight(const State<O>& s, u64 row) {
  return s.data.weights.data ? s.data.weights.data[row] : 1.0;
}
template<unsigned O> __device__ bool target_ok(const State<O>& s, double y) {
  if (!isfinite(y)) return false;
  if (s.config.objective == Objective::binary_logistic) return y == 0 || y == 1;
  if (s.config.objective == Objective::multiclass_softmax) return y >= 0 && y < s.outputs && floor(y) == y;
  return true;
}

// Kind: weights, narrow independent outputs, 32-output tile, coupled classes.
template<unsigned O, unsigned Kind> __global__ void initial_partial(State<O>* state) {
  const auto& s = *state;
  __shared__ double scratch[8][32];
  const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const u64 tasks = Kind == 0 || Kind == 3 ? s.init_chunks :
    u64(Kind == 2 ? ceil_div(s.outputs, 32) : s.outputs) * s.init_chunks;
  for (u64 task = blockIdx.x; task < tasks; task += gridDim.x) {
    const unsigned chunk = task % s.init_chunks;
    const u64 output = Kind == 2 ? (task / s.init_chunks) * 32 + lane : task / s.init_chunks;
    if constexpr (Kind == 3) {
      for (u64 k = threadIdx.x; k < s.outputs; k += threads) s.initial[u64(chunk) * s.outputs + k] = 0;
      __syncthreads();
    }
    double sum = 0;
    const u64 first = Kind == 2 ? u64(chunk) * 8 + warp : u64(chunk) * threads + threadIdx.x;
    const u64 stride = u64(s.init_chunks) * (Kind == 2 ? 8 : threads);
    if (Kind != 2 || output < s.outputs) {
      for (u64 row = first; row < s.data.rows; row += stride) {
        const double w = weight(s, row);
        const bool valid_w = isfinite(w) && w >= 0;
        if constexpr (Kind == 0 || Kind == 3) if (!valid_w) fail(s.status, input);
        if constexpr (Kind == 0) { if (valid_w) sum += w; }
        else {
          const double y = s.data.targets.data[Kind == 3 ? row : row * s.outputs + output];
          const bool valid_y = target_ok(s, y);
          if (!valid_y) fail(s.status, input);
          if constexpr (Kind == 3) {
            if (valid_w) sum += w;
            if (valid_w && valid_y && w > 0) atomicAdd(s.initial + u64(chunk) * s.outputs + static_cast<u32>(y), w);
          } else if (valid_w && valid_y) sum += w * y;
        }
      }
    }
    if constexpr (Kind == 2) {
      scratch[warp][lane] = sum; __syncthreads();
      for (unsigned d = 4; d; d >>= 1) {
        if (warp < d) scratch[warp][lane] += scratch[warp + d][lane];
        __syncthreads();
      }
      if (!warp && output < s.outputs) s.initial[u64(chunk) * s.outputs + output] = scratch[0][lane];
    } else {
      sum = block_sum(sum, scratch[0]);
      if (!threadIdx.x) {
        if constexpr (Kind == 0 || Kind == 3) s.initial[u64(s.init_chunks) * s.outputs + chunk] = sum;
        else s.initial[u64(chunk) * s.outputs + output] = sum;
      }
    }
    __syncthreads();
  }
}
template<unsigned O> __global__ void finish_weight(State<O>* state) {
  const auto& s = *state;
  __shared__ double scratch[8];
  double sum = 0;
  for (unsigned k = threadIdx.x; k < s.init_chunks; k += threads) sum += s.initial[u64(s.init_chunks) * s.outputs + k];
  sum = block_sum(sum, scratch);
  if (!threadIdx.x) { *s.weight = sum; if (!(sum > 0) || !isfinite(sum)) fail(s.status, input); }
}
template<unsigned O> __global__ void finish_base(State<O>* state) {
  const auto& s = *state;
  for (u64 k = u64(blockIdx.x) * threads + threadIdx.x; k < s.outputs; k += u64(gridDim.x) * threads) {
    double sum = 0;
    for (unsigned chunk = 0; chunk < s.init_chunks; ++chunk) sum += s.initial[u64(chunk) * s.outputs + k];
    double value = 0;
    if (*s.weight > 0 && isfinite(*s.weight)) {
      const double mean = sum / *s.weight;
      if (s.config.objective == Objective::squared_error) value = mean;
      else if (s.config.objective == Objective::binary_logistic) {
        const double p = fmin(1 - 1e-12, fmax(1e-12, mean)); value = log(p) - log1p(-p);
      } else value = log(fmax(mean, 1e-12));
    }
    if (!isfinite(value)) fail(s.status, numeric);
    s.output->model->base.data[k] = value;
  }
}
template<unsigned O> __global__ void initialize_margins(State<O>* state) {
  const auto& s = *state;
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < u64(s.data.rows) * s.outputs; i += u64(gridDim.x) * threads)
    s.output->margins.data[i] = s.output->model->base.data[i % s.outputs];
}
template<unsigned O, bool Multiclass> __global__ void derivatives(State<O>* state) {
  const auto& s = *state;
  const auto* margins = s.output->margins.data;
  if constexpr (Multiclass) {
    const unsigned lane = threadIdx.x & 31;
    for (u64 row = u64(blockIdx.x) * 8 + threadIdx.x / 32; row < s.data.rows; row += u64(gridDim.x) * 8) {
      const double w = weight(s, row);
      double maximum = -CUDART_INF, sum = 0;
      if (w > 0) {
        for (u64 k = lane; k < s.outputs; k += 32) maximum = fmax(maximum, margins[row * s.outputs + k]);
        maximum = shuffle<2>(warp_max(maximum), 0);
        for (u64 k = lane; k < s.outputs; k += 32) sum += exp(margins[row * s.outputs + k] - maximum);
        sum = shuffle<2>(warp_sum(sum), 0);
      }
      for (u64 k = lane; k < s.outputs; k += 32) {
        const u64 i = row * s.outputs + k;
        const double p = w > 0 ? exp(margins[i] - maximum) / sum : 0;
        s.derivative[0][i] = w > 0 ? w * (p - (k == static_cast<u32>(s.data.targets.data[row]) ? 1.0 : 0.0)) : 0;
        s.derivative[1][i] = w > 0 ? w * fmax(2.0 * p * (1.0 - p), 1e-16) : 0;
      }
    }
  } else {
    for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < u64(s.data.rows) * s.tile_live; i += u64(gridDim.x) * threads) {
      const u64 row = i / s.tile_live, from = row * s.outputs + s.tile_begin + i % s.tile_live;
      const double w = weight(s, row), y = s.data.targets.data[from], margin = margins[from];
      Stats<O> value;
      if (w > 0) {
        if constexpr (O > 2) {
          const double u = exp(-fabs(margin)), inverse = 1.0 / (1.0 + u), small = u * inverse;
          const double p = margin >= 0 ? inverse : small, q = margin >= 0 ? small : inverse;
          const double curvature = u * inverse * inverse;
          value.d[0] = w * (y == 1.0 ? -q : p); value.d[1] = w * curvature;
          value.d[2] = value.d[1] * (1.0 - 2.0 * p);
          if constexpr (O == 4) value.d[3] = value.d[1] * (1.0 - 6.0 * curvature);
        } else if (s.config.objective == Objective::squared_error) {
          value.d[0] = w * (margin - y); value.d[1] = w;
        } else {
          const double p = sigmoid(margin);
          value.d[0] = w * (p - y); value.d[1] = w * fmax(p * (1.0 - p), 1e-16);
        }
      }
#pragma unroll
      for (unsigned k = 0; k < O; ++k) s.derivative[k][i] = value.d[k];
    }
  }
}
template<unsigned O, bool Shared> __global__ void root_counts(State<O>* state) {
  const auto& s = *state;
  extern __shared__ unsigned long long local[];
  if constexpr (!Shared) {
    for (u64 row = u64(blockIdx.x) * threads + threadIdx.x; row < s.data.rows; row += u64(gridDim.x) * threads)
      for (unsigned f = 0; f < s.schema.columns; ++f)
        atomicAdd(s.counts + s.schema.offsets.data[f] + s.bins[u64(f) * s.data.rows + row], 1ULL);
  } else {
    const unsigned chunks = min(256u, static_cast<u32>(ceil_div(s.data.rows, 4096)));
    for (u64 task = blockIdx.x; task < u64(s.schema.columns) * chunks; task += gridDim.x) {
      const unsigned f = task / chunks, chunk = task % chunks;
      const unsigned offset = s.schema.offsets.data[f], bins = s.schema.offsets.data[f + 1] - offset;
      for (unsigned b = threadIdx.x; b < bins; b += threads) local[b] = 0;
      __syncthreads();
      for (u64 row = u64(chunk) * threads + threadIdx.x; row < s.data.rows; row += u64(chunks) * threads)
        atomicAdd(local + s.bins[u64(f) * s.data.rows + row], 1ULL);
      __syncthreads();
      for (unsigned b = threadIdx.x; b < bins; b += threads) if (local[b]) atomicAdd(s.counts + offset + b, local[b]);
      __syncthreads();
    }
  }
}
template<unsigned O> struct HistogramJob {
  Stats<O>* output;
  u32 capacity, live, derivative_begin, derivative_stride, groups;
};
template<unsigned O, bool Root, bool Cached> __global__ void clear_hist(State<O>* state, HistogramJob<O> job) {
  const auto& s = *state;
  for (unsigned output = blockIdx.y; output < job.live; output += gridDim.y) {
    const u32 nodes = Root ? 1 : s.active[output];
    for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < u64(nodes) * s.schema.total_bins; i += u64(gridDim.x) * threads) {
      Stats<O> value;
      if constexpr (Cached) value.count = s.counts[i];
      job.output[u64(output) * job.capacity * s.schema.total_bins + i] = value;
    }
  }
}
template<unsigned O, unsigned Width, bool Root, bool Cached> __global__ void global_hist(State<O>* state, HistogramJob<O> job) {
  const auto& s = *state;
  const unsigned lane = threadIdx.x & (Width - 1);
  const unsigned peers = (0xffffffffu >> (32 - Width)) << ((threadIdx.x & 31) & ~(Width - 1));
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < u64(s.data.rows) * job.groups * Width; i += u64(gridDim.x) * threads) {
    const u64 row = (i / Width) / job.groups;
    const unsigned output = ((i / Width) % job.groups) * Width + lane;
    int node = 0;
    bool live = output < job.live;
    if constexpr (!Root) {
      node = live ? s.assignments[u64(output) * s.data.rows + row] : -1;
      live = live && node >= 0 && static_cast<u32>(node) < s.active[output];
    }
    if constexpr (Width == 1) { if (!live) continue; }
    else if (!__any_sync(peers, live)) continue;
    Stats<O> value;
    if (live) {
      const u64 from = row * job.derivative_stride + job.derivative_begin + output;
#pragma unroll
      for (unsigned k = 0; k < O; ++k) value.d[k] = s.derivative[k][from];
    }
    for (unsigned f = 0; f < s.schema.columns; ++f) {
      unsigned bin = !lane ? s.bins[u64(f) * s.data.rows + row] : 0;
      if constexpr (Width > 1) bin = __shfl_sync(peers, bin, 0, Width);
      if (live) {
        auto* target = job.output + (u64(output) * job.capacity + node) * s.schema.total_bins + s.schema.offsets.data[f] + bin;
#pragma unroll
        for (unsigned k = 0; k < O; ++k) if (value.d[k] != 0) atomicAdd(target->d + k, value.d[k]);
        if constexpr (!Cached) atomicAdd(&target->count, 1ULL);
      }
    }
  }
}
template<unsigned O> __global__ void shared_hist(State<O>* state, HistogramJob<O> job, unsigned chunks) {
  const auto& s = *state;
  extern __shared__ unsigned long long words[];
  auto* local = reinterpret_cast<Stats<O>*>(words);
  for (unsigned output = blockIdx.y; output < job.live; output += gridDim.y) {
    const unsigned nodes = s.active[output];
    for (u64 task = blockIdx.x; task < u64(s.schema.columns) * chunks; task += gridDim.x) {
      const unsigned f = task / chunks, chunk = task % chunks;
      const unsigned offset = s.schema.offsets.data[f], bins = s.schema.offsets.data[f + 1] - offset;
      for (u64 i = threadIdx.x; i < u64(nodes) * bins; i += threads) local[i] = {};
      __syncthreads();
      for (u64 row = u64(chunk) * threads + threadIdx.x; row < s.data.rows; row += u64(chunks) * threads) {
        const int node = s.assignments[u64(output) * s.data.rows + row];
        if (node < 0 || static_cast<u32>(node) >= nodes) continue;
        const u64 from = row * job.derivative_stride + job.derivative_begin + output;
        auto* target = local + u64(node) * bins + s.bins[u64(f) * s.data.rows + row];
#pragma unroll
        for (unsigned k = 0; k < O; ++k) { const double x = s.derivative[k][from]; if (x != 0) atomicAdd(target->d + k, x); }
        atomicAdd(&target->count, 1ULL);
      }
      __syncthreads();
      for (u64 i = threadIdx.x; i < u64(nodes) * bins; i += threads) {
        const auto value = local[i];
        if (!value.count) continue;
        auto* target = job.output + (u64(output) * job.capacity + i / bins) * s.schema.total_bins + offset + i % bins;
#pragma unroll
        for (unsigned k = 0; k < O; ++k) if (value.d[k] != 0) atomicAdd(target->d + k, value.d[k]);
        atomicAdd(&target->count, value.count);
      }
      __syncthreads();
    }
  }
}

template<unsigned O> __device__ Choice consider(Choice best, Stats<O> total,
    Stats<O> left_present, Stats<O> missing, unsigned feature, unsigned threshold,
    const TrainConfig& config, double parent_benefit) {
  for (unsigned direction = 0; direction != 2; ++direction) {
    const auto left = direction ? add(left_present, missing) : left_present;
    const auto right = subtract(total, left);
    if (left.count < config.min_leaf_rows || right.count < config.min_leaf_rows ||
        left.d[1] < config.min_child_hessian || right.d[1] < config.min_child_hessian) continue;
    const auto l = leaf(left, config), r = leaf(right, config);
    const double gain = l.benefit + r.benefit - parent_benefit;
    if (isfinite(gain) && gain > config.min_gain)
      best = better(best, {static_cast<int>(feature), threshold, direction, gain, best.value, l.value, r.value});
  }
  return best;
}
__device__ Choice shuffle_choice(Choice c, unsigned delta) {
  c.feature = shuffle<0>(c.feature, delta);
  c.threshold = shuffle<0>(c.threshold, delta);
  c.missing_left = shuffle<0>(c.missing_left, delta);
  c.gain = shuffle<0>(c.gain, delta);
  c.value = shuffle<0>(c.value, delta);
  c.left = shuffle<0>(c.left, delta);
  c.right = shuffle<0>(c.right, delta);
  return c;
}
template<unsigned Block> __device__ Choice best_choice(Choice value, Choice* scratch) {
  if constexpr (Block == 32) {
    for (unsigned d = 16; d; d >>= 1) {
      const auto other = shuffle_choice(value, d);
      if (threadIdx.x < d) value = better(value, other);
    }
    return value;
  } else {
    scratch[threadIdx.x] = value; __syncthreads();
    for (unsigned d = Block / 2; d; d >>= 1) {
      if (threadIdx.x < d) scratch[threadIdx.x] = better(scratch[threadIdx.x], scratch[threadIdx.x + d]);
      __syncthreads();
    }
    value = scratch[0]; __syncthreads(); return value;
  }
}
template<unsigned O> struct SplitJob {
  const Stats<O>* hist;
  Choice* winners;
  u32 capacity, live, winner_stride;
  bool root;
};
template<unsigned O, unsigned Block> __global__ void candidates(State<O>* state, SplitJob<O> job) {
  const auto& s = *state;
  __shared__ Stats<O> sums[8];
  __shared__ Choice choices[Block];
  for (u64 task = blockIdx.x; task < u64(job.live) * job.capacity * s.schema.columns; task += gridDim.x) {
    const u64 node = task / s.schema.columns;
    if (!job.root && node % job.capacity >= s.active[node / job.capacity]) continue;
    const unsigned f = task % s.schema.columns, begin = s.schema.offsets.data[f];
    const unsigned bins = s.schema.offsets.data[f + 1] - begin;
    const auto* cells = job.hist + node * s.schema.total_bins + begin;
    Stats<O> sum;
    for (unsigned b = threadIdx.x; b < bins; b += Block) sum = add(sum, cells[b]);
    const auto total = block_sum<O, Block>(sum, sums), missing = cells[0];
    const auto parent = leaf(total, s.config);
    Choice best; best.value = parent.value;
    if (s.config.max_depth && total.count >= 2ULL * s.config.min_leaf_rows) {
      if (!threadIdx.x) best = consider(best, total, Stats<O>{}, missing, f, 0, s.config, parent.benefit);
      if (s.schema.features.data[f].type == FeatureType::categorical) {
        for (unsigned b = threadIdx.x + 1; b < bins; b += Block)
          best = consider(best, total, cells[b], missing, f, b, s.config, parent.benefit);
      } else {
        Stats<O> carry;
        for (unsigned begin_bin = 1; begin_bin < bins; begin_bin += Block) {
          const unsigned b = begin_bin + threadIdx.x;
          Stats<O> tile_total;
          const auto prefix = add(carry, block_scan<O, Block>(b < bins ? cells[b] : Stats<O>{}, sums, tile_total));
          carry = add(carry, tile_total);
          if (b < bins) best = consider(best, total, prefix, missing, f, b, s.config, parent.benefit);
        }
      }
    }
    best = best_choice<Block>(best, choices);
    if (!threadIdx.x) s.candidates[task] = best;
    if constexpr (Block > 32) __syncthreads();
  }
}
template<unsigned O, unsigned Block> __global__ void winners(State<O>* state, SplitJob<O> job) {
  const auto& s = *state;
  __shared__ Choice scratch[Block];
  for (u64 node = blockIdx.x; node < u64(job.live) * job.capacity; node += gridDim.x) {
    if (!job.root && node % job.capacity >= s.active[node / job.capacity]) continue;
    Choice best; best.value = s.candidates[node * s.schema.columns].value;
    for (u64 f = threadIdx.x; f < s.schema.columns; f += Block) best = better(best, s.candidates[node * s.schema.columns + f]);
    best = best_choice<Block>(best, scratch);
    if (!threadIdx.x) job.winners[(node / job.capacity) * job.winner_stride + node % job.capacity] = best;
    if constexpr (Block > 32) __syncthreads();
  }
}
template<unsigned O> __device__ bool split(State<O>* s, SplitJob<O> job) {
  if (!mark(s, observe::Stage::split, false)) return false;
  const bool narrow = s->schema.max_feature_bins <= 32 &&
    (s->schema.columns <= 32 || s->config.splits == SplitPolicy::warp_wide);
  if (s->config.splits != SplitPolicy::block256 && narrow) {
    candidates<O, 32><<<grid(u64(job.live) * job.capacity * s->schema.columns, 1), 32>>>(s, job);
    if (!launch_ok(s->status)) return false;
    if (s->schema.columns <= 32) winners<O, 32><<<grid(u64(job.live) * job.capacity, 1), 32>>>(s, job);
    else winners<O, 256><<<grid(u64(job.live) * job.capacity, 1), 256>>>(s, job);
  } else {
    candidates<O, 256><<<grid(u64(job.live) * job.capacity * s->schema.columns, 1), 256>>>(s, job);
    if (!launch_ok(s->status)) return false;
    winners<O, 256><<<grid(u64(job.live) * job.capacity, 1), 256>>>(s, job);
  }
  return launch_ok(s->status) && mark(s, observe::Stage::split, true);
}
template<unsigned O> __global__ void begin_trees(State<O>* state) {
  const auto& s = *state;
  for (unsigned output = blockIdx.y; output < s.batch_live; output += gridDim.y) {
    for (u64 row = u64(blockIdx.x) * threads + threadIdx.x; row < s.data.rows; row += u64(gridDim.x) * threads)
      s.assignments[u64(output) * s.data.rows + row] = 0;
    if (!threadIdx.x && !blockIdx.x) {
      s.active[output] = s.used[output] = 1;
      s.next[output] = 0;
      s.frontier[0][u64(output) * s.frontier_capacity] = 0;
      s.nodes[u64(output) * s.node_capacity] = {};
      if (s.config.batched_root_splits)
        s.winners[u64(output) * s.frontier_capacity] = s.root_winners[s.tree_begin + output];
    }
  }
}
__device__ unsigned exclusive_scan(unsigned* values, unsigned width) {
  __syncthreads();
  for (unsigned step = 1; step < width; step <<= 1) {
    for (unsigned i = (threadIdx.x + 1) * step * 2 - 1; i < width; i += blockDim.x * step * 2) values[i] += values[i - step];
    __syncthreads();
  }
  const unsigned total = values[width - 1]; __syncthreads();
  if (!threadIdx.x) values[width - 1] = 0;
  __syncthreads();
  for (unsigned step = width / 2; step; step >>= 1) {
    for (unsigned i = (threadIdx.x + 1) * step * 2 - 1; i < width; i += blockDim.x * step * 2) {
      const unsigned left = values[i - step]; values[i - step] = values[i]; values[i] += left;
    }
    __syncthreads();
  }
  return total;
}
template<unsigned O> __device__ unsigned split_flag(const State<O>& s, Choice c) {
  if (!isfinite(c.gain) || !isfinite(c.value) || !isfinite(c.left) || !isfinite(c.right) ||
      !isfinite(s.config.learning_rate * c.value) || !isfinite(s.config.learning_rate * c.left) ||
      !isfinite(s.config.learning_rate * c.right)) { fail(s.status, numeric); return 0; }
  if (c.feature < -1 || c.missing_left > 1 || (c.feature >= 0 &&
      (static_cast<u32>(c.feature) >= s.schema.columns || c.threshold >=
        s.schema.offsets.data[c.feature + 1] - s.schema.offsets.data[c.feature]))) {
    fail(s.status, model); return 0;
  }
  return c.feature >= 0;
}
template<unsigned O> __device__ void reserve_children(const State<O>& s, unsigned output, unsigned splits) {
  if (atomicAdd(&s.status->errors, 0u)) { s.used[output] = s.next[output] = 0; return; }
  const bool expand = s.depth + 1 < s.config.max_depth;
  s.old_used[output] = s.used[output];
  if (u64(s.used[output]) + u64(splits) * 2 > s.node_capacity ||
      (expand && u64(splits) * 2 > s.frontier_capacity)) {
    fail(s.status, capacity); s.used[output] = s.next[output] = 0;
  } else { s.used[output] += splits * 2; s.next[output] = expand ? splits * 2 : 0; }
}
template<unsigned O> __device__ void write_node(const State<O>& s, unsigned output, unsigned index, unsigned ordinal) {
  const u64 slot = u64(output) * s.frontier_capacity + index;
  const Choice c = s.winners[slot];
  Node* nodes = s.nodes + u64(output) * s.node_capacity;
  const int current = s.frontier[s.side][slot];
  if (c.feature < 0) { nodes[current] = {-1, -1, -1, 0, 0, s.config.learning_rate * c.value}; return; }
  const unsigned left = s.old_used[output] + 2 * ordinal, right = left + 1;
  nodes[current] = {c.feature, static_cast<int>(left), static_cast<int>(right), c.threshold, c.missing_left, 0};
  nodes[left] = {-1, -1, -1, 0, 0, s.config.learning_rate * c.left};
  nodes[right] = {-1, -1, -1, 0, 0, s.config.learning_rate * c.right};
  if (s.next[output]) {
    const u64 next = u64(output) * s.frontier_capacity + 2 * ordinal;
    s.frontier[!s.side][next] = left; s.frontier[!s.side][next + 1] = right;
  }
}
template<unsigned O, bool Small> __global__ void scan_splits(State<O>* state, unsigned width) {
  const auto& s = *state;
  __shared__ unsigned scan[1024];
  for (unsigned output = blockIdx.y; output < s.batch_live; output += gridDim.y) {
    const u64 base = u64(output) * s.frontier_capacity, first = u64(blockIdx.x) * 1024;
    for (unsigned i = threadIdx.x; i < width; i += blockDim.x) {
      const u64 index = first + i;
      scan[i] = index < s.active[output] ? split_flag(s, s.winners[base + index]) : 0;
    }
    const unsigned total = exclusive_scan(scan, width);
    for (unsigned i = threadIdx.x; i < width && first + i < s.frontier_capacity; i += blockDim.x)
      s.prefix[base + first + i] = scan[i];
    if (!threadIdx.x) {
      s.block_prefix[u64(output) * s.scan_blocks + blockIdx.x] = Small ? 0 : total;
      if constexpr (Small) reserve_children(s, output, total);
    }
    __syncthreads();
    if constexpr (Small) if (s.used[output])
      for (unsigned i = threadIdx.x; i < s.active[output]; i += blockDim.x) write_node(s, output, i, scan[i]);
    __syncthreads();
  }
}
template<unsigned O> __global__ void prefix_blocks(State<O>* state) {
  const auto& s = *state;
  for (unsigned output = threadIdx.x; output < s.batch_live; output += blockDim.x) {
    unsigned total = 0;
    for (unsigned b = 0; b < s.scan_blocks; ++b) {
      auto& value = s.block_prefix[u64(output) * s.scan_blocks + b];
      const unsigned old = value; value = total; total += old;
    }
    reserve_children(s, output, total);
  }
}
template<unsigned O> __global__ void materialize(State<O>* state) {
  const auto& s = *state;
  for (unsigned output = blockIdx.y; output < s.batch_live; output += gridDim.y) {
    if (!s.used[output]) continue;
    for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < s.active[output]; i += u64(gridDim.x) * threads)
      write_node(s, output, static_cast<u32>(i), s.prefix[u64(output) * s.frontier_capacity + i] +
        s.block_prefix[u64(output) * s.scan_blocks + i / 1024]);
  }
}
template<unsigned O> __global__ void route(State<O>* state) {
  const auto& s = *state;
  for (unsigned output = blockIdx.y; output < s.batch_live; output += gridDim.y) {
    if (!s.used[output]) continue;
    const u64 base = u64(output) * s.frontier_capacity;
    const Node* nodes = s.nodes + u64(output) * s.node_capacity;
    for (u64 row = u64(blockIdx.x) * threads + threadIdx.x; row < s.data.rows; row += u64(gridDim.x) * threads) {
      auto& assignment = s.assignments[u64(output) * s.data.rows + row];
      if (assignment < 0 || static_cast<u32>(assignment) >= s.active[output]) continue;
      const u64 slot = base + assignment;
      const Node n = nodes[s.frontier[s.side][slot]];
      double value = n.value;
      if (n.feature >= 0) {
        const unsigned bin = s.bins[u64(n.feature) * s.data.rows + row];
        const bool left = goes_left(bin, s.schema.features.data[n.feature].type, n.threshold, n.missing_left);
        if (s.next[output]) {
          const unsigned ordinal = s.prefix[slot] + s.block_prefix[u64(output) * s.scan_blocks + static_cast<u32>(assignment) / 1024];
          assignment = static_cast<int>(2 * ordinal + !left); continue;
        }
        value = nodes[left ? n.left : n.right].value;
      }
      double& margin = s.output->margins.data[row * s.outputs + s.tile_begin + s.tree_begin + output];
      margin = __dadd_rn(margin, value); assignment = -1;
    }
  }
}
template<unsigned O> __global__ void export_nodes(State<O>* state) {
  const auto& s = *state;
  for (unsigned output = blockIdx.y; output < s.batch_live; output += gridDim.y)
    for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < s.used[output]; i += u64(gridDim.x) * threads)
      s.output->model->nodes.data[s.export_begin[output] + i] = s.nodes[u64(output) * s.node_capacity + i];
}

template<unsigned O, bool Multiclass> __global__ void loss_terms(State<O>* state) {
  const auto& s = *state;
  __shared__ double scratch[8];
  const auto* margins = s.output->margins.data;
  double sum = 0;
  if constexpr (Multiclass) {
    const unsigned lane = threadIdx.x & 31;
    for (u64 row = u64(blockIdx.x) * 8 + threadIdx.x / 32; row < s.data.rows; row += u64(gridDim.x) * 8) {
      const double w = weight(s, row);
      if (w == 0) continue;
      double maximum = -CUDART_INF, denominator = 0;
      for (u64 k = lane; k < s.outputs; k += 32) maximum = fmax(maximum, margins[row * s.outputs + k]);
      maximum = shuffle<2>(warp_max(maximum), 0);
      for (u64 k = lane; k < s.outputs; k += 32) denominator += exp(margins[row * s.outputs + k] - maximum);
      denominator = shuffle<2>(warp_sum(denominator), 0);
      if (!lane) sum += w * ((maximum - margins[row * s.outputs + static_cast<u32>(s.data.targets.data[row])]) + log(denominator));
    }
  } else {
    for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < u64(s.data.rows) * s.outputs; i += u64(gridDim.x) * threads) {
      const double w = weight(s, i / s.outputs);
      if (w == 0) continue;
      const double margin = margins[i], y = s.data.targets.data[i];
      if (s.config.objective == Objective::squared_error) { const double e = margin - y; sum += (0.5 * w * e) * e; }
      else sum += w * (margin >= 0 ? (1.0 - y) * margin + log1p(exp(-margin)) : -y * margin + log1p(exp(margin)));
    }
  }
  sum = block_sum(sum, scratch);
  if (!threadIdx.x) s.loss_partial[blockIdx.x] = sum;
}
template<unsigned O> __global__ void finish_loss(State<O>* state, unsigned round) {
  const auto& s = *state;
  __shared__ double scratch[256];
  double sum = 0;
  for (unsigned i = threadIdx.x; i < s.loss_blocks; i += threads) sum += s.loss_partial[i];
  scratch[threadIdx.x] = sum; __syncthreads();
  for (unsigned d = 128; d; d >>= 1) {
    if (threadIdx.x < d) scratch[threadIdx.x] += scratch[threadIdx.x + d];
    __syncthreads();
  }
  if (!threadIdx.x) {
    const double divisor = *s.weight * (s.config.objective == Objective::multiclass_softmax ? 1 : s.outputs);
    const double value = scratch[0] / divisor;
    s.output->loss.data[round] = value;
    if (!isfinite(value)) fail(s.status, numeric);
  }
}
template<unsigned O> __device__ bool loss(State<O>* s, unsigned round) {
  if (!mark(s, observe::Stage::loss, false)) return false;
  if (s->config.objective == Objective::multiclass_softmax) loss_terms<O, true><<<s->loss_blocks, threads>>>(s);
  else loss_terms<O, false><<<s->loss_blocks, threads>>>(s);
  if (!launch_ok(s->status)) return false;
  finish_loss<<<1, threads>>>(s, round);
  return launch_ok(s->status) && mark(s, observe::Stage::loss, true);
}
template<unsigned O, bool Root, bool Cached> __device__ bool launch_global(State<O>* s, HistogramJob<O> job) {
  const unsigned mapping_width = Root && s->config.tree_build == TreeBuild::output_batch ? s->tile_capacity : job.live;
  auto launch = [&]<unsigned Width>() {
    job.groups = static_cast<unsigned>(ceil_div(mapping_width, Width));
    global_hist<O, Width, Root, Cached><<<grid(u64(s->data.rows) * job.groups * Width), threads>>>(s, job);
  };
  if (mapping_width == 1) launch.template operator()<1>();
  else if (mapping_width <= 2) launch.template operator()<2>();
  else if (mapping_width <= 4) launch.template operator()<4>();
  else if (mapping_width <= 8) launch.template operator()<8>();
  else if (mapping_width <= 16) launch.template operator()<16>();
  else launch.template operator()<32>();
  return launch_ok(s->status);
}
template<unsigned O, bool Root, bool Cached> __device__ bool histogram(State<O>* s, HistogramJob<O> job, Histogram policy) {
  if (!mark(s, observe::Stage::histogram, false)) return false;
  clear_hist<O, Root, Cached><<<dim3(grid(u64(job.capacity) * s->schema.total_bins), min(job.live, 65535u)), threads>>>(s, job);
  if (!launch_ok(s->status)) return false;
  if constexpr (!Root && O == 2) {
    if (policy == Histogram::shared) {
      const unsigned chunks = min(256u, static_cast<u32>(ceil_div(s->data.rows, 4096)));
      const u64 bytes = u64(s->depth ? job.capacity : 1) * s->schema.max_feature_bins * sizeof(Stats<O>);
      shared_hist<<<dim3(grid(u64(s->schema.columns) * chunks, 1), min(job.live, 65535u)), threads, bytes>>>(s, job, chunks);
      return launch_ok(s->status) && mark(s, observe::Stage::histogram, true);
    }
  }
  return launch_global<O, Root, Cached>(s, job) && mark(s, observe::Stage::histogram, true);
}
template<unsigned O> __device__ HistogramJob<O> current_histogram(State<O>* s) {
  const bool multi = s->config.objective == Objective::multiclass_softmax;
  return {s->hist, s->frontier_capacity, s->batch_live,
    (multi ? s->tile_begin : 0) + s->tree_begin, multi ? s->outputs : s->tile_live, 0};
}
enum class Phase { initialized, round, tile, trees, level, calibrated, routed, exported, round_done };
template<unsigned O> __global__ void advance(State<O>* s, Phase phase);
template<unsigned O> __device__ cudaError_t next(State<O>* s, Phase phase) {
  const auto result = checked_launch(s->status, cudaGetLastError());
  if (result != cudaSuccess) return result;
  advance<<<1, 1, 0, cudaStreamTailLaunch>>>(s, phase);
  return checked_launch(s->status, cudaGetLastError());
}
template<unsigned O> __device__ void materialize_level(State<O>* s) {
  if (!mark(s, observe::Stage::route, false)) return;
  const unsigned fc = s->frontier_capacity;
  if (fc <= 1024) {
    unsigned width = 1; while (width < fc) width <<= 1;
    scan_splits<O, true><<<dim3(1, min(s->batch_live, 65535u)), min(256u, width)>>>(s, width);
  } else {
    scan_splits<O, false><<<dim3(s->scan_blocks, min(s->batch_live, 65535u)), threads>>>(s, 1024);
    if (!launch_ok(s->status)) return;
    prefix_blocks<<<1, threads>>>(s);
    if (!launch_ok(s->status)) return;
    materialize<<<dim3(grid(fc), min(s->batch_live, 65535u)), threads>>>(s);
  }
  if (!launch_ok(s->status)) return;
  route<<<dim3(grid(s->data.rows), min(s->batch_live, 65535u)), threads>>>(s);
  if (!launch_ok(s->status) || !mark(s, observe::Stage::route, true)) return;
  next(s, Phase::routed);
}
__device__ unsigned sample_variant(unsigned ordinal) {
  const unsigned pair = ordinal / 2;
  return ordinal % 2 ^ (pair < 2 ? 0 : (pair - 2) % 2);
}
template<unsigned O> __device__ void calibrate(State<O>* s) {
  const auto variant = sample_variant(s->calibration->ordinal);
  begin_sample<<<1, 1>>>(s->calibration);
  if (!launch_ok(s->status) || !histogram<O, false, false>(s, current_histogram(s), variant ? Histogram::shared : Histogram::global)) return;
  next(s, Phase::calibrated);
}
__device__ u64 median5(const u64* input) {
  u64 values[5];
  for (unsigned i = 0; i < 5; ++i) {
    unsigned j = i;
    while (j && values[j - 1] > input[i]) { values[j] = values[j - 1]; --j; }
    values[j] = input[i];
  }
  return values[2];
}
template<unsigned O> __global__ void advance(State<O>* s, Phase phase) {
  if (threadIdx.x || blockIdx.x) return;
  if (s->status->errors) { finish(s->status); return; }
  switch (phase) {
    case Phase::initialized:
      initialize_margins<<<grid(u64(s->data.rows) * s->outputs), threads>>>(s);
      if (!launch_ok(s->status) || !mark(s, observe::Stage::base, true)) return;
      if (s->counts) {
        fill<<<grid(s->schema.total_bins), threads>>>(s->counts, s->schema.total_bins, 0ULL);
        if (!launch_ok(s->status)) return;
        if (s->config.root_counts == RootCounts::shared)
          root_counts<O, true><<<grid(u64(s->schema.columns) * min(256u, static_cast<u32>(ceil_div(s->data.rows, 4096))), 1), threads,
            u64(s->schema.max_feature_bins) * sizeof(unsigned long long)>>>(s);
        else root_counts<O, false><<<grid(s->data.rows), threads>>>(s);
        if (!launch_ok(s->status)) return;
      }
      if (loss(s, 0)) next(s, Phase::round);
      return;
    case Phase::round:
      if (s->round == s->config.rounds) { finish(s->status); return; }
      s->tile_begin = 0;
      if (s->config.objective == Objective::multiclass_softmax)
      {
        if (!mark(s, observe::Stage::gradient, false)) return;
        derivatives<O, true><<<grid(s->data.rows, 8), threads>>>(s);
        if (!launch_ok(s->status) || !mark(s, observe::Stage::gradient, true)) return;
      }
      next(s, Phase::tile); return;
    case Phase::tile: {
      if (s->tile_begin == s->outputs) {
        if (loss(s, s->round + 1)) next(s, Phase::round_done);
        return;
      }
      s->tile_live = min(s->tile_capacity, s->outputs - s->tile_begin); s->tree_begin = 0;
      const bool multi = s->config.objective == Objective::multiclass_softmax;
      if (!multi) {
        if (!mark(s, observe::Stage::gradient, false)) return;
        derivatives<O, false><<<grid(u64(s->data.rows) * s->tile_live), threads>>>(s);
        if (!launch_ok(s->status) || !mark(s, observe::Stage::gradient, true)) return;
      }
      if (s->config.batched_roots) {
        HistogramJob<O> job{s->roots, 1, s->tile_live, multi ? s->tile_begin : 0, multi ? s->outputs : s->tile_live, 0};
        if (s->counts) { if (!histogram<O, true, true>(s, job, Histogram::global)) return; }
        else if (!histogram<O, true, false>(s, job, Histogram::global)) return;
        if (s->config.batched_root_splits && !split(s, {s->roots, s->root_winners, 1, s->tile_live, 1, true})) return;
      }
      next(s, Phase::trees); return;
    }
    case Phase::trees:
      if (s->tree_begin == s->tile_live) { s->tile_begin += s->tile_live; next(s, Phase::tile); return; }
      s->batch_live = min(s->batch_capacity, s->tile_live - s->tree_begin); s->depth = s->side = 0;
      begin_trees<<<dim3(grid(s->data.rows), min(s->batch_live, 65535u)), threads>>>(s);
      if (!launch_ok(s->status)) return;
      next(s, Phase::level); return;
    case Phase::level: {
      if constexpr (O == 2) if (!s->depth && s->root_policy) {
        const unsigned output = s->tile_begin + s->tree_begin;
        if (s->root_policy[output] == Histogram::automatic) {
          if (u64(s->schema.max_feature_bins) * sizeof(Stats<O>) <= 49152) {
            *s->calibration = {}; calibrate(s); return;
          }
          s->root_policy[output] = Histogram::global;
        }
      }
      if (s->depth == 0 && s->config.batched_root_splits) { materialize_level(s); return; }
      if (s->depth == 0 && s->config.batched_roots) {
        if (!split(s, {s->roots + u64(s->tree_begin) * s->schema.total_bins, s->winners, 1, s->batch_live, s->frontier_capacity, true})) return;
      } else {
        auto job = current_histogram(s);
        Histogram policy = s->config.histogram == Histogram::shared ? Histogram::shared : Histogram::global;
        if (!s->depth && s->root_policy) policy = s->root_policy[s->tile_begin + s->tree_begin];
        if (!histogram<O, false, false>(s, job, policy) ||
            !split(s, {s->hist, s->winners, s->frontier_capacity, s->batch_live, s->frontier_capacity, false})) return;
      }
      materialize_level(s); return;
    }
    case Phase::calibrated:
      if constexpr (O == 2) {
        auto& sample = *s->calibration;
        const unsigned ordinal = sample.ordinal, pair = ordinal / 2, variant = sample_variant(ordinal);
        if (pair >= 2) sample.ticks[variant][pair - 2] = timestamp() - sample.begin;
        if (++sample.ordinal < 14) { calibrate(s); return; }
        const unsigned output = s->tile_begin + s->tree_begin;
        const auto policy = median5(sample.ticks[1]) < median5(sample.ticks[0]) ? Histogram::shared : Histogram::global;
        s->root_policy[output] = policy;
        if (s->output->tuning.data) {
          auto& record = s->output->tuning.data[output];
          record.output = output; record.selected = policy; record.measured = true;
          for (unsigned i = 0; i < 5; ++i) { record.global_ticks[i] = sample.ticks[0][i]; record.shared_ticks[i] = sample.ticks[1][i]; }
        }
        next(s, Phase::level); return;
      } else { fail(s->status, unsupported); finish(s->status); return; }
    case Phase::routed: {
      bool active = false;
      for (unsigned k = 0; k < s->batch_live; ++k) { s->active[k] = s->next[k]; active |= s->active[k] != 0; }
      ++s->depth; s->side ^= 1;
      if (active) { next(s, Phase::level); return; }
      auto* m = s->output->model;
      u64 nodes = m->node_count;
      for (unsigned k = 0; k < s->batch_live; ++k) {
        if (!s->used[k] || !add_fits(nodes, s->used[k]) || nodes + s->used[k] > m->nodes.size) { fail(s->status, capacity); finish(s->status); return; }
        s->export_begin[k] = nodes; nodes += s->used[k];
      }
      for (unsigned k = 0; k < s->batch_live; ++k) {
        const unsigned output = s->tile_begin + s->tree_begin + k;
        m->trees.data[u64(output) * s->config.rounds + s->round] = {s->export_begin[k], s->used[k], output};
      }
      m->node_count = nodes;
      if (!mark(s, observe::Stage::export_model, false)) return;
      export_nodes<<<dim3(grid(s->node_capacity), min(s->batch_live, 65535u)), threads>>>(s);
      if (!launch_ok(s->status) || !mark(s, observe::Stage::export_model, true)) return;
      next(s, Phase::exported); return;
    }
    case Phase::exported:
      s->tree_begin += s->batch_live; next(s, Phase::trees); return;
    case Phase::round_done:
      ++s->round; next(s, Phase::round); return;
  }
}

__device__ bool valid_config(Dataset data, const Schema& schema, const TrainConfig& c) {
  const unsigned outputs = c.objective == Objective::multiclass_softmax ? c.classes : data.outputs;
  const bool objective = c.objective == Objective::squared_error || c.objective == Objective::binary_logistic ||
    (c.objective == Objective::multiclass_softmax && c.classes >= 2 && data.outputs == 1);
  if (!objective || !data.rows || !data.columns || data.columns > INT32_MAX || !outputs || outputs > INT32_MAX ||
      !c.min_leaf_rows || !c.output_tile || c.max_depth > 30 || c.order < 2 || c.order > 4 ||
      !c.max_histogram_bytes || !c.max_device_bytes ||
      !isfinite(c.learning_rate) || !(c.learning_rate > 0) || !isfinite(c.l2) || c.l2 < 0 ||
      !isfinite(c.min_child_hessian) || c.min_child_hessian < 0 || !isfinite(c.min_gain) || c.min_gain < 0 ||
      !isfinite(c.max_leaf_value) || c.max_leaf_value < 0) return false;
  if (static_cast<u32>(c.histogram) > static_cast<u32>(Histogram::automatic) ||
      static_cast<u32>(c.splits) > static_cast<u32>(SplitPolicy::warp_wide) ||
      static_cast<u32>(c.tree_build) > static_cast<u32>(TreeBuild::output_batch) ||
      static_cast<u32>(c.root_counts) > static_cast<u32>(RootCounts::shared)) return false;
  if (c.order > 2 && (c.objective != Objective::binary_logistic || c.tree_build != TreeBuild::output_batch ||
      c.histogram == Histogram::shared || !(c.max_leaf_value > 0) || c.splits == SplitPolicy::warp_wide)) return false;
  if ((!c.batched_roots && (c.root_counts != RootCounts::per_output || c.batched_root_splits)) ||
      (c.tree_build == TreeBuild::output_batch && (!c.batched_roots || !c.batched_root_splits))) return false;
  return schema.columns == data.columns && contains(schema.features, data.columns) &&
    contains(schema.offsets, u64(data.columns) + 1) && contains(schema.metadata, schema.metadata_count) &&
    schema.total_bins >= data.columns && schema.max_feature_bins && schema.max_feature_bins <= 65536 &&
    schema.max_feature_bins <= schema.total_bins && contains(data.targets, u64(data.rows) * data.outputs) &&
    (!data.weights.size || contains(data.weights, data.rows)) && (!data.weights.data || data.weights.size >= data.rows);
}
template<unsigned O> __device__ cudaError_t submit_training(Dataset data, const Schema& schema,
    const std::uint16_t* bins, TrainConfig config, Training* output, Workspace workspace, Status* status) {
  State<O> plan;
  plan.data = data; plan.schema = schema; plan.bins = bins; plan.config = config; plan.output = output; plan.status = status;
  plan.outputs = config.objective == Objective::multiclass_softmax ? config.classes : data.outputs;
  plan.tile_capacity = min(plan.outputs, config.output_tile);
  plan.batch_capacity = config.tree_build == TreeBuild::output_batch ? plan.tile_capacity : 1;
  plan.derivative_capacity = config.objective == Objective::multiclass_softmax ? plan.outputs : plan.tile_capacity;
  plan.init_chunks = min(256u, static_cast<u32>(ceil_div(data.rows, 1024)));
  plan.loss_blocks = min(4096u, static_cast<u32>(ceil_div(data.rows, 256)));
  const u64 leaves = max(u64{1}, u64(data.rows) / config.min_leaf_rows);
  const u64 node_bound = min((u64{1} << (config.max_depth + 1)) - 1, leaves * 2 - 1);
  const u64 level_bound = min(u64{1} << (config.max_depth ? config.max_depth - 1 : 0), leaves);
  if (node_bound > INT32_MAX) { fail(status, extent); return finish(status); }
  const u64 root_cells = config.rounds && config.batched_roots ? u64(plan.tile_capacity) * schema.total_bins : 0;
  const u64 count_cells = config.rounds && config.root_counts != RootCounts::per_output ? schema.total_bins : 0;
  Arena histogram_size{{nullptr, UINT64_MAX}};
  histogram_size.take<Stats<O>>(root_cells); histogram_size.take<unsigned long long>(count_cells);
  if (!histogram_size.valid || (config.rounds && histogram_size.used >= config.max_histogram_bytes)) { fail(status, capacity); return finish(status); }
  const u64 per_level_cells = u64(plan.batch_capacity) * schema.total_bins;
  if (!mul_fits(per_level_cells, sizeof(Stats<O>))) { fail(status, extent); return finish(status); }
  plan.frontier_capacity = config.rounds ? static_cast<u32>(min(level_bound,
    (config.max_histogram_bytes - histogram_size.used) / (per_level_cells * sizeof(Stats<O>)))) : 0;
  plan.node_capacity = config.rounds ? static_cast<u32>(node_bound) : 0;
  plan.scan_blocks = static_cast<u32>(ceil_div(plan.frontier_capacity, 1024));
  if (config.rounds && (!plan.frontier_capacity || (config.histogram == Histogram::shared &&
      u64(plan.frontier_capacity) * schema.max_feature_bins > 49152 / sizeof(Stats<O>)) ||
      (config.root_counts == RootCounts::shared && schema.max_feature_bins > 49152 / sizeof(unsigned long long)))) {
    fail(status, capacity); return finish(status);
  }
  const u64 hist_cells = config.rounds ? per_level_cells * plan.frontier_capacity : 0;
  histogram_size.take<Stats<O>>(hist_cells);
  const u64 slots = config.rounds ? u64(plan.batch_capacity) * plan.frontier_capacity : 0;
  const u64 root_slots = config.rounds && config.batched_root_splits ? plan.tile_capacity : 0;
  Arena arena{workspace};
  auto* state = arena.take<State<O>>(1);
  const u64 derivative_cells = config.rounds ? u64(data.rows) * plan.derivative_capacity : 0;
  for (unsigned k = 0; k < O; ++k) plan.derivative[k] = arena.take<double>(derivative_cells);
  plan.initial = arena.take<double>(u64(plan.init_chunks) * (u64(plan.outputs) + 1));
  plan.weight = arena.take<double>(1); plan.loss_partial = arena.take<double>(plan.loss_blocks);
  plan.roots = arena.take<Stats<O>>(root_cells); plan.counts = arena.take<unsigned long long>(count_cells);
  plan.hist = arena.take<Stats<O>>(hist_cells);
  plan.candidates = arena.take<Choice>(max(slots, root_slots) * schema.columns);
  plan.winners = arena.take<Choice>(slots); plan.root_winners = arena.take<Choice>(root_slots);
  plan.nodes = arena.take<Node>(config.rounds ? u64(plan.batch_capacity) * plan.node_capacity : 0);
  plan.assignments = arena.take<std::int32_t>(config.rounds ? u64(plan.batch_capacity) * data.rows : 0);
  for (unsigned side = 0; side < 2; ++side) plan.frontier[side] = arena.take<std::int32_t>(slots);
  const unsigned batch = config.rounds ? plan.batch_capacity : 0;
  plan.active = arena.take<u32>(batch); plan.used = arena.take<u32>(batch); plan.next = arena.take<u32>(batch);
  plan.old_used = arena.take<u32>(batch); plan.prefix = arena.take<u32>(slots);
  plan.block_prefix = arena.take<u32>(u64(batch) * plan.scan_blocks); plan.export_begin = arena.take<u64>(batch);
  const bool autotune = O == 2 && config.rounds && !config.batched_roots && config.histogram == Histogram::automatic;
  plan.root_policy = arena.take<Histogram>(autotune ? plan.outputs : 0);
  plan.calibration = arena.take<Calibration>(autotune ? 1 : 0);
  if (!arena.fits(status)) return finish(status);
  auto* model = output->model;
  const u64 trees = u64(config.rounds) * plan.outputs, predictions = u64(data.rows) * plan.outputs;
  if (!contains(Array<Model>{model, 1}, 1) || !contains(model->base, plan.outputs) || !contains(model->trees, trees) ||
      !contains(model->output_offsets, u64(plan.outputs) + 1) || !contains(model->nodes, trees ? 1 : 0) ||
      !contains(output->margins, predictions) || !contains(output->loss, u64(config.rounds) + 1)) {
    fail(status, capacity); return finish(status);
  }
  if ((output->tuning.data || output->tuning.size) && !contains(output->tuning, plan.outputs)) { fail(status, capacity); return finish(status); }
  Arena resident{{nullptr, UINT64_MAX}};
  resident.take<std::byte>(arena.used);
  resident.take<float>(data.values.size); resident.take<float>(data.targets.size); resident.take<float>(data.weights.size);
  resident.take<std::uint16_t>(u64(data.rows) * data.columns);
  resident.take<Feature>(schema.columns); resident.take<float>(schema.metadata_count); resident.take<u32>(u64(schema.columns) + 1);
  resident.take<Node>(model->nodes.size); resident.take<Tree>(trees); resident.take<double>(plan.outputs);
  resident.take<u64>(u64(plan.outputs) + 1); resident.take<double>(predictions); resident.take<double>(u64(config.rounds) + 1);
  resident.take<TuningRecord>(output->tuning.size);
  resident.take<observe::Stamp>(output->trace.records.size);
  if (!resident.valid || resident.used > config.max_device_bytes) { fail(status, capacity); return finish(status); }
  *state = plan;
  output->workspace_bytes = arena.used; output->histogram_bytes = histogram_size.used;
  output->derivative_bytes = derivative_cells * O * sizeof(double);
  output->tree_state_bytes = u64(batch) * (u64(plan.node_capacity) * sizeof(Node) + u64(data.rows) * sizeof(int) +
    u64(plan.frontier_capacity) * (2 * sizeof(int) + sizeof(u32)) + u64(plan.scan_blocks) * sizeof(u32) + 4 * sizeof(u32) + sizeof(u64));
  output->frontier_capacity = plan.frontier_capacity; output->tree_capacity = plan.node_capacity; output->output_capacity = batch;
  model->schema = schema; model->outputs = plan.outputs; model->objective = config.objective;
  model->node_count = 0; model->tree_count = trees;
  for (u64 k = 0; k <= plan.outputs; ++k) model->output_offsets.data[k] = k * config.rounds;
  if (plan.root_policy) {
    fill<<<grid(plan.outputs), threads>>>(plan.root_policy, plan.outputs, Histogram::automatic);
    if (auto error = checked_launch(status, cudaGetLastError()); error != cudaSuccess) return error;
  }
  if (output->tuning.data) {
    fill<<<grid(plan.outputs), threads>>>(output->tuning.data, plan.outputs, TuningRecord{});
    if (auto error = checked_launch(status, cudaGetLastError()); error != cudaSuccess) return error;
  }
  if (!mark(state, observe::Stage::validate, false)) return cudaErrorLaunchFailure;
  if (config.objective == Objective::multiclass_softmax) initial_partial<O, 3><<<plan.init_chunks, threads>>>(state);
  else {
    initial_partial<O, 0><<<plan.init_chunks, threads>>>(state);
    if (auto error = checked_launch(status, cudaGetLastError()); error != cudaSuccess) return error;
    if (plan.outputs < 8) initial_partial<O, 1><<<grid(u64(plan.init_chunks) * plan.outputs, 1), threads>>>(state);
    else initial_partial<O, 2><<<grid(u64(plan.init_chunks) * ceil_div(plan.outputs, 32), 1), threads>>>(state);
  }
  if (auto error = checked_launch(status, cudaGetLastError()); error != cudaSuccess) return error;
  finish_weight<<<1, threads>>>(state);
  if (auto error = checked_launch(status, cudaGetLastError()); error != cudaSuccess) return error;
  if (!mark(state, observe::Stage::validate, true) || !mark(state, observe::Stage::base, false)) return cudaErrorLaunchFailure;
  finish_base<<<grid(plan.outputs), threads>>>(state);
  if (auto error = checked_launch(status, cudaGetLastError()); error != cudaSuccess) return error;
  return next(state, Phase::initialized);
}
}

__device__ cudaError_t train(Dataset data, const Schema* schema, Array<const std::uint16_t> bins,
    TrainConfig config, Training* output, Workspace workspace, Status* status) { using namespace learn_impl;
  if (!status) return cudaErrorInvalidValue;
  if (!contains(Array<const Schema>{schema, 1}, 1) || !contains(Array<Training>{output, 1}, 1) ||
      !contains(bins, u64(data.rows) * data.columns) || !valid_config(data, *schema, config)) {
    fail(status, shape); return finish(status);
  }
  switch (config.order) {
    case 2: return submit_training<2>(data, *schema, bins.data, config, output, workspace, status);
    case 3: return submit_training<3>(data, *schema, bins.data, config, output, workspace, status);
    default: return submit_training<4>(data, *schema, bins.data, config, output, workspace, status);
  }
}
}


#endif // GH_IMPLEMENTATION

// GH_SOURCE_CATEGORY: tests
// core checks
#if GH_MODE == 1
namespace gh::test::core_suite {
__global__ void run();
namespace {
__device__ __align__(16) std::byte storage[128];
__device__ Status status;
__device__ Status planning_status;
__device__ u32 observed;
__global__ void produce(u32 value) { observed = value; }
__global__ void verify() {
  succeeded(status);
  GH_CHECK(observed == 29);
  printf("PASS core: checked extents, arena bounds, ordered child/tail completion\n");
}
}
__global__ void run() {
  GH_CHECK(add_fits(UINT64_MAX, 0));
  GH_CHECK(!add_fits(UINT64_MAX, 1));
  GH_CHECK(mul_fits(UINT64_MAX, 1));
  GH_CHECK(!mul_fits(UINT64_MAX, 2));
  GH_CHECK(ceil_div(UINT64_MAX, 256) == (UINT64_MAX / 256) + 1);
  GH_CHECK(contains(Array<double>{nullptr, 0}, 0));
  GH_CHECK(!contains(Array<double>{nullptr, 1}, 1));
  auto& local = planning_status;
  local = {};
  Arena layout{{storage, 128}};
  GH_CHECK(layout.take<double>(0) == nullptr && layout.used == 0);
  GH_CHECK(layout.take<u32>(1) == reinterpret_cast<u32*>(storage));
  GH_CHECK(layout.take<double>(2) == reinterpret_cast<double*>(storage + 8));
  GH_CHECK(layout.fits(&local) && local.required_bytes == 24);
  GH_CHECK(!layout.take<double>(UINT64_MAX));
  GH_CHECK(!layout.fits(&local) && (local.errors & extent));
  local = {};
  Arena small{{storage, 7}};
  GH_CHECK(!small.take<double>(1));
  GH_CHECK(!small.fits(&local) && (local.errors & capacity));
  local = {};
  Arena wrapped{{reinterpret_cast<std::byte*>(UINT64_MAX - 15), 32}};
  GH_CHECK(!wrapped.take<double>(4));
  GH_CHECK(!wrapped.fits(&local) && (local.errors & extent));
  status = {};
  produce<<<1, 1>>>(11);
  produce<<<1, 1>>>(29);
  submitted(finish(&status));
  verify<<<1, 1, 0, cudaStreamTailLaunch>>>();
}
}
#endif

// GH_SOURCE_CATEGORY: tests
// count checks
#if GH_MODE == 2
namespace gh::test::count_suite {
__global__ void run();
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
    gh::test::count_suite::graph_case<<<1,1,0,stream>>>();
    error = cudaStreamEndCapture(stream,&graph);
  }
  if (error == cudaSuccess) error = cudaGraphInstantiate(&executable,graph,0);
  if (error == cudaSuccess) error = cudaGraphLaunch(executable,stream);
  if (error == cudaSuccess) error = cudaGraphLaunch(executable,stream);
  if (error == cudaSuccess) {
    gh::test::count_suite::graph_replays_checked<<<1,1,0,stream>>>();
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
#endif

// GH_SOURCE_CATEGORY: tests
// data checks
#if GH_MODE == 3
namespace gh::test::data_suite {
__global__ void run();
namespace {
constexpr u32 cells_capacity = 300000, metadata_capacity = 70000, fit_cases = 26;
__device__ float values[cells_capacity], metadata[metadata_capacity + 2];
__device__ Feature descriptors[65];
__device__ FeatureType types[65];
__device__ u32 offsets[66];
__device__ std::uint16_t bins[cells_capacity + 2];
__device__ __align__(16) std::byte scratch_bytes[4 << 20];
__device__ Schema schema;
__device__ Status status;

struct Case { u32 rows, columns, max_bins; };
__device__ Case shape_for(u32 test) {
  switch (test) {
    case 0: return {1, 1, 32};
    case 1: return {31, 3, 32};
    case 2: return {32, 31, 32};
    case 3: return {33, 32, 32};
    case 4: return {257, 33, 32};
    case 5: return {1025, 65, 32};
    case 6: return {4097, 65, 32};
    case 7: return {65537, 3, 32};
    case 8: return {257, 33, 32};
    case 9: return {1025, 3, 2};
    case 10: return {65535, 1, 65536};
    case 11: return {65536, 1, 65536};
    case 14: return {1025, 65, 32};
    default: return {33, 3, 32};
  }
}
__device__ FeatureType feature_type(u32 test, u32 f) {
  if (test == 10 || test == 11) return FeatureType::categorical;
  if (test == 0 || test == 9) return FeatureType::numeric;
  return f % 3 == 1 ? FeatureType::categorical : FeatureType::numeric;
}
__device__ float fixture_value(u32 test, u32 row, u32 feature, bool query = false) {
  if (test == 10 || test == 11) return query && row % 17 == 0 ? -1.0f : float(row);
  if (test == 8 || row % 29 == 0) return __uint_as_float(0x7fc00001u + row % 127);
  if ((test == 12 || test == 25) && row == 32 && feature == 2)
    return __uint_as_float(test == 12 ? 0x7f800000u : 0xff800000u);
  if (query && row % 17 == 0) return 9999.0f;
  if (row % 19 == 0) return __uint_as_float(row % 2 ? 0x80000000u : 0u);
  const u32 width = feature_type(test, feature) == FeatureType::categorical ? 11 : 73;
  return float(int((row * 17 + feature * 13) % width) - int(width / 2));
}
__global__ void fixture(u32 test, bool query) {
  const auto c = shape_for(test);
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < u64(c.rows) * c.columns; i += u64(gridDim.x) * blockDim.x)
    values[i] = fixture_value(test, u32(i / c.columns), u32(i % c.columns), query);
}
__global__ void fit_start(u32 id);
__global__ void encode_start(u32 id);
__global__ void codec_start(u32 id);

// Independent insertion-sort oracle on a bounded distinct domain. Large
// category fixtures have an analytic ordered sequence instead of quadratic work.
__global__ void check_feature(u32 id, bool query) {
  const u32 test = id % fit_cases, f = blockIdx.x;
  const auto c = shape_for(test);
  const auto type = feature_type(test, f);
  float distinct[80];
  u32 count = 0;
  if (test == 10) count = c.rows;
  else for (u32 row = 0; row < c.rows; ++row) {
    float x = fixture_value(test, row, f);
    if (!isfinite(x)) continue;
    if (x == 0) x = 0.0f;
    u32 at = 0;
    while (at < count && distinct[at] < x) ++at;
    if (at < count && distinct[at] == x) continue;
    GH_CHECK(count < 80);
    for (u32 j = count; j > at; --j) distinct[j] = distinct[j - 1];
    distinct[at] = x; ++count;
  }
  const u32 intervals = min(count, c.max_bins - 1);
  const u32 expected = type == FeatureType::categorical ? count : intervals ? intervals - 1 : 0;
  const auto feature = schema.features.data[f];
  GH_CHECK(feature.type == type && feature.count == expected);
  GH_CHECK(offsets[f + 1] - offsets[f] == expected + (type == FeatureType::numeric ? 2 : 1));
  for (u32 k = 0; k < expected; ++k) {
    const u32 rank = type == FeatureType::categorical ? k : u32(u64(k + 1) * count / intervals - 1);
    const float value = test == 10 ? float(rank) : distinct[rank];
    GH_CHECK(__float_as_uint(metadata[1 + feature.begin + k]) == __float_as_uint(value));
  }
  for (u32 row = 0; row < c.rows; ++row) {
    const float x = fixture_value(test, row, f, query);
    GH_CHECK(__float_as_uint(values[u64(row) * c.columns + f]) == __float_as_uint(x));
    u32 expected_bin = 0;
    if (isfinite(x)) {
      if (test == 10) expected_bin = x >= 0 ? row + 1 : 0;
      else if (type == FeatureType::categorical) {
        for (u32 k = 0; k < count; ++k) if (distinct[k] == x) expected_bin = k + 1;
      } else {
        expected_bin = 1;
        for (u32 k = 1; k < intervals; ++k)
          expected_bin += distinct[u32(u64(k) * count / intervals - 1)] < x;
      }
    }
    GH_CHECK(bins[1 + u64(f) * c.rows + row] == expected_bin);
  }
}
__device__ u32 expected_failure(u32 test) {
  switch (test) {
    case 11: case 15: case 18: case 24: return capacity;
    case 12: case 13: case 25: return input;
    case 16: case 17: case 19: case 20: case 22: case 23: return shape;
    case 21: return unsupported;
    default: return 0;
  }
}
__device__ void guards() {
  GH_CHECK(bins[0] == 0xdead && bins[cells_capacity + 1] == 0xbeef);
  GH_CHECK(__float_as_uint(metadata[0]) == 0x7fa12345u);
  GH_CHECK(__float_as_uint(metadata[metadata_capacity + 1]) == 0x7fa54321u);
}
__global__ void fit_check(u32 id, bool query) {
  const u32 test = id % fit_cases;
  GH_CHECK(status.done == 1);
  guards();
  if (const u32 error = expected_failure(test); error) {
    GH_CHECK(status.errors & error);
    if (test == 15) GH_CHECK(status.required_bytes > 16);
    fit_start<<<1, 1, 0, cudaStreamTailLaunch>>>(id + 1);
    return;
  }
  succeeded(status);
  const auto c = shape_for(test);
  GH_CHECK(schema.columns == c.columns && schema.total_bins == offsets[c.columns]);
  GH_CHECK(schema.metadata_count == descriptors[c.columns - 1].begin + descriptors[c.columns - 1].count);
  GH_CHECK(offsets[0] == 0);
  if (test == 14) GH_CHECK(status.required_bytes <= 65536);
  check_feature<<<c.columns, 1>>>(id, query);
  if (query) fit_start<<<1, 1, 0, cudaStreamTailLaunch>>>(id + 1);
  else encode_start<<<1, 1, 0, cudaStreamTailLaunch>>>(id);
}
__global__ void encode_start(u32 id) {
  const u32 test = id % fit_cases;
  const auto c = shape_for(test);
  status = {};
  fixture<<<64, 256>>>(test, true);
  const Dataset d{{values, u64(c.rows) * c.columns}, {}, {}, c.rows, c.columns, 1};
  submitted(encode(d, &schema, {bins + 1, cells_capacity}, &status));
  fit_check<<<1, 1, 0, cudaStreamTailLaunch>>>(id, true);
}
__global__ void fit_start(u32 id) {
  if (id == fit_cases * 2) { codec_start<<<1, 1, 0, cudaStreamTailLaunch>>>(0); return; }
  const u32 test = id % fit_cases;
  auto c = shape_for(test);
  status = {};
  schema = {{descriptors, test == 16 ? 0u : 65u}, {metadata + 1, test == 18 ? 0u : metadata_capacity},
    {offsets, test == 17 ? 0u : 66u}};
  for (u32 f = 0; f < c.columns; ++f) types[f] = test == 13 ? FeatureType(99) : feature_type(test, f);
  bins[0] = 0xdead; bins[cells_capacity + 1] = 0xbeef;
  metadata[0] = __uint_as_float(0x7fa12345u); metadata[metadata_capacity + 1] = __uint_as_float(0x7fa54321u);
  fixture<<<64, 256>>>(test, false);
  const Dataset d{{test == 22 ? nullptr : values, u64(c.rows) * c.columns}, {}, {},
    test == 23 ? 0u : c.rows, c.columns, 1};
  const auto policy = test == 21 ? RadixPolicy(3) : id < fit_cases ? RadixPolicy::radix4 : RadixPolicy::radix8;
  const Workspace w{scratch_bytes + (test == 24), test == 15 ? 16u : test == 14 ? 65536u : sizeof(scratch_bytes) - 16};
  submitted(fit_schema(d, {types, test == 0 ? 0u : c.columns}, test == 20 ? 1u : c.max_bins,
    &schema, {bins + 1, test == 19 ? 0u : cells_capacity}, w, &status, policy));
  fit_check<<<1, 1, 0, cudaStreamTailLaunch>>>(id, false);
}

__device__ std::byte encoded[2049], repeated[2049];
__device__ float codec_values[21], codec_targets[14], decoded_values[23], decoded_targets[16];
__device__ DatasetRecord record, decoded;
__device__ u32 word(const std::byte* p) {
  return u32(p[0]) | u32(p[1]) << 8 | u32(p[2]) << 16 | u32(p[3]) << 24;
}
__device__ void set_word(std::byte* p, u32 x) {
  p[0] = std::byte(x); p[1] = std::byte(x >> 8); p[2] = std::byte(x >> 16); p[3] = std::byte(x >> 24);
}
__global__ void schema_fail_start(u32 id);
__global__ void codec_roundtrip(u32 id, u64 size) {
  succeeded(status);
  for (u64 i = 0; i < size; ++i) GH_CHECK(encoded[i + 1] == repeated[i + 1]);
  codec_start<<<1, 1, 0, cudaStreamTailLaunch>>>(id + 1);
}
__global__ void codec_decoded(u32 id, u64 size) {
  GH_CHECK(status.done == 1);
  if (id >= 5) {
    GH_CHECK(status.errors != 0);
    codec_start<<<1, 1, 0, cudaStreamTailLaunch>>>(id + 1);
    return;
  }
  succeeded(status);
  GH_CHECK(decoded.data.rows == 7 && decoded.data.columns == 3 && decoded.data.outputs == record.data.outputs);
  GH_CHECK(decoded.objective == record.objective && decoded.classes == record.classes && decoded.data.weights.size == 0);
  for (u32 i = 0; i < 21; ++i) GH_CHECK(__float_as_uint(decoded_values[i + 1]) == __float_as_uint(codec_values[i]));
  for (u32 i = 0; i < 7 * record.data.outputs; ++i) GH_CHECK(__float_as_uint(decoded_targets[i + 1]) == __float_as_uint(codec_targets[i]));
  GH_CHECK(decoded_values[0] == -999 && decoded_values[22] == -999);
  GH_CHECK(decoded_targets[0] == -999 && decoded_targets[15] == -999);
  status = {};
  submitted(encode_dataset(&decoded, {repeated + 1, sizeof(repeated) - 1}, &status));
  codec_roundtrip<<<1, 1, 0, cudaStreamTailLaunch>>>(id, size);
}
__global__ void codec_encoded(u32 id) {
  GH_CHECK(status.done == 1);
  if (id == 3 || id == 4) {
    GH_CHECK(status.errors & (id == 3 ? shape : capacity));
    codec_start<<<1, 1, 0, cudaStreamTailLaunch>>>(id + 1);
    return;
  }
  succeeded(status);
  u64 size = status.required_bytes;
  auto* bytes = encoded + 1;
  GH_CHECK(word(bytes) == 0x44424847u && word(bytes + 4) == 0x31303053u);
  GH_CHECK(word(bytes + 8) == 1 && word(bytes + 12) == 7 && word(bytes + 16) == 3);
  GH_CHECK(word(bytes + 20) == record.data.outputs && word(bytes + 24) == u32(record.objective));
  GH_CHECK(word(bytes + 28) == record.classes);
  for (u32 i = 0; i < 21; ++i) GH_CHECK(word(bytes + 32 + i * 4) == __float_as_uint(codec_values[i]));
  for (u32 i = 0; i < 7 * record.data.outputs; ++i) GH_CHECK(word(bytes + 32 + (21 + i) * 4) == __float_as_uint(codec_targets[i]));
  if (id == 5) bytes[0] = std::byte(0);
  if (id == 6) set_word(bytes + 8, 2);
  if (id == 7) --size;
  if (id == 8) ++size;
  if (id == 9) set_word(bytes + 12, 0);
  if (id == 10) set_word(bytes + 24, 7);
  if (id == 11) set_word(bytes + 32, 0x7f800000u);
  if (id == 12) set_word(bytes + 32 + 21 * 4, 0x7fc00001u);
  if (id == 13) set_word(bytes + 24, u32(Objective::binary_logistic));
  if (id == 15) { set_word(bytes + 12, UINT32_MAX); set_word(bytes + 16, UINT32_MAX); }
  status = {};
  submitted(decode_dataset({bytes, size}, {decoded_values + 1, id == 14 ? 0u : 21u},
    {decoded_targets + 1, 14}, &decoded, &status));
  codec_decoded<<<1, 1, 0, cudaStreamTailLaunch>>>(id, size);
}
__global__ void codec_start(u32 id) {
  if (id == 16) { schema_fail_start<<<1, 1, 0, cudaStreamTailLaunch>>>(0); return; }
  status = {};
  const auto objective = id < 3 ? Objective(id) : Objective::squared_error;
  const u32 outputs = objective == Objective::multiclass_softmax ? 1 : 2;
  for (u32 i = 0; i < 21; ++i) codec_values[i] = i == 0 ? __uint_as_float(0xffc00017u) : i == 1 ? -0.0f : float(int(i) - 10);
  for (u32 i = 0; i < 7 * outputs; ++i)
    codec_targets[i] = objective == Objective::binary_logistic ? float(i % 2) : objective == Objective::multiclass_softmax ? float(i % 3) : float(int(i) - 5);
  decoded_values[0] = decoded_values[22] = -999; decoded_targets[0] = decoded_targets[15] = -999;
  record = {{{codec_values, 21}, {codec_targets, u64(7) * outputs},
    {codec_targets, id == 3 ? 7u : 0u}, 7, 3, outputs}, objective, objective == Objective::multiclass_softmax ? 3u : 0u};
  submitted(encode_dataset(&record, {encoded + 1, id == 4 ? 8u : sizeof(encoded) - 1}, &status));
  codec_encoded<<<1, 1, 0, cudaStreamTailLaunch>>>(id);
}

__global__ void schema_fail_checked(u32 id) {
  GH_CHECK(status.done == 1);
  if (!id) succeeded(status); else GH_CHECK(status.errors != 0);
  GH_CHECK(bins[0] == 0xdead && bins[1] == 0x5555);
  schema_fail_start<<<1, 1, 0, cudaStreamTailLaunch>>>(id + 1);
}
__global__ void schema_fail_start(u32 id) {
  if (id == 8) { printf("GPU data checks passed: 52 fits, fitted/query encoding, 16 codec cases, 8 schema cases\n"); return; }
  status = {};
  descriptors[0] = {0, 2, FeatureType::numeric}; metadata[1] = -1; metadata[2] = 1;
  offsets[0] = 0; offsets[1] = 4;
  schema = {{descriptors, 1}, {metadata + 1, 2}, {offsets, 2}, 1, 4, 4, 2};
  bins[0] = 0xdead; bins[1] = 0x5555;
  if (id == 1) descriptors[0].begin = UINT64_MAX;
  if (id == 2) metadata[2] = -2;
  if (id == 3) metadata[2] = __uint_as_float(0x7f800000u);
  if (id == 4) offsets[1] = 9;
  if (id == 5) schema.features.size = 0;
  if (id == 6) schema.columns = 2;
  if (id == 7) schema.metadata_count = 3;
  submitted(encode({{}, {}, {}, 0, 1, 1}, &schema, {bins + 1, 0}, &status));
  schema_fail_checked<<<1, 1, 0, cudaStreamTailLaunch>>>(id);
}
}
__global__ void run() { fit_start<<<1, 1>>>(0); }
}
#endif

// GH_SOURCE_CATEGORY: tests
// model checks
#if GH_MODE == 4
namespace gh::test::model_suite {
__global__ void run();
namespace {
constexpr u32 rows = 65, model_cases = 34, wire_cases = 16;
constexpr double guard = -987654.25;
__device__ Model resident;
__device__ Feature features[4];
__device__ float metadata[8];
__device__ u32 offsets[5];
__device__ Node nodes[18];
__device__ Tree trees[10];
__device__ double base[35], output[rows * 33 + 2];
__device__ u64 output_offsets[36], written;
__device__ std::uint16_t bins[rows * 2 + 2];
__device__ __align__(16) std::byte scratch[16 + 4096 + 16];
__device__ std::byte input_bytes[1026], encoded[1026];
__device__ Status status;
__device__ const unsigned char literal[]{
  0x47,0x48,0x42,0x4d,0x4f,0x44,0x45,0x4c, 1,0,0,0, 0,0,0,0,
  1,0,0,0, 1,0,0,0, 1,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0x80, 0,0,0,0, 0,0,0,0, 0,0,0,0,
  0,0,0,0, 1,0,0,0,
  255,255,255,255, 255,255,255,255, 255,255,255,255,
  0,0,0,0, 1,0,0,0, 0,0,0,0,0,0,0xf8,0x3f};
static_assert(sizeof(literal) == 88);
__device__ Workspace workspace() { return {scratch + 16, 4096}; }
__device__ u64 bits(double value) { return cuda::std::bit_cast<u64>(value); }
__device__ double number(u64 value) { return cuda::std::bit_cast<double>(value); }

__device__ void initialize() {
  status = {};
  for (auto& x : scratch) x = std::byte{0x5a};
  for (auto& x : output) x = guard;
  for (auto& x : base) x = guard;
  for (auto& x : nodes) x = {-1,-1,-1,0,0,guard};
  for (auto& x : trees) x = {123,456,789};
  for (auto& x : metadata) x = -12345;
  for (auto& x : bins) x = 65535;
  features[1] = {0,3,FeatureType::numeric}; features[2] = {3,3,FeatureType::categorical};
  metadata[1] = -3; metadata[2] = 0; metadata[3] = 2;
  metadata[4] = 2; metadata[5] = 5; metadata[6] = 8;
  offsets[1] = 0; offsets[2] = 5; offsets[3] = 9;
  base[1] = .25; base[2] = -.5; base[3] = -0.; base[4] = .25;
  // Physical node order differs from grouped descriptor order.
  nodes[1] = {0,1,2,2,1,0}; nodes[2].value = 1; nodes[3].value = -2;
  nodes[4].value = 0x1p53;
  nodes[5] = {1,1,2,2,0,0}; nodes[6].value = 2; nodes[7].value = -1;
  nodes[8].value = 1; nodes[9].value = .125; nodes[10].value = -0x1p53;
  trees[1] = {4,3,0}; trees[2] = {0,3,1}; trees[3] = {8,1,1};
  trees[4] = {3,1,3}; trees[5] = {7,1,3}; trees[6] = {9,1,3};
  output_offsets[1] = 0; output_offsets[2] = 1; output_offsets[3] = 3;
  output_offsets[4] = 3; output_offsets[5] = 6;
  for (u32 r = 0; r < rows; ++r) { bins[1 + r] = r % 5; bins[1 + rows + r] = r % 4; }
  resident = {{{features + 1,2},{metadata + 1,6},{offsets + 1,3},2,9,5,6},
              {nodes + 1,16},{trees + 1,8},{base + 1,33},{output_offsets + 1,34},10,6,4,Objective::squared_error};
}
__device__ void one_output(u32 count) {
  resident.outputs = 1; resident.node_count = count; resident.tree_count = count;
  output_offsets[1] = 0; output_offsets[2] = count;
  for (u32 i = 0; i < count; ++i) { trees[i + 1] = {i,1,0}; nodes[i + 1] = {-1,-1,-1,0,0,0}; }
}
__device__ void guards() {
  GH_CHECK(base[0] == guard && base[34] == guard && nodes[0].value == guard && nodes[17].value == guard);
  GH_CHECK(trees[0].begin == 123 && trees[9].begin == 123);
  GH_CHECK(metadata[0] == -12345 && metadata[7] == -12345);
  GH_CHECK(bins[0] == 65535 && bins[rows * 2 + 1] == 65535);
  for (u32 i = 0; i < 16; ++i) GH_CHECK(scratch[i] == std::byte{0x5a} && scratch[4112 + i] == std::byte{0x5a});
}
__device__ bool valid_case(u32 c) { return c == 0 || c == 21 || c == 22 || c == 24 || (c >= 28 && c <= 32); }
__global__ void model_case(u32 c);
__global__ void wire_case(u32 c);
__global__ void roundtrip_start();
__device__ void next_model(u32 c) {
  if (c + 1 < model_cases) model_case<<<1,1,0,cudaStreamTailLaunch>>>(c + 1);
  else wire_case<<<1,1,0,cudaStreamTailLaunch>>>(0);
  submitted(cudaGetLastError());
}
__global__ void prediction_checked(u32 c) {
  GH_CHECK(status.done == 1);
  guards();
  if (c >= 30) {
    if (c == 31) succeeded(status);
    else GH_CHECK(status.errors & (c == 30 ? input : capacity));
    for (auto x : output) GH_CHECK(x == guard);
  } else {
    succeeded(status);
    for (u32 r = 0; r < rows; ++r) for (u32 o = 0; o < resident.outputs; ++o) {
      const double expected = c == 28 ? (o == 0 ? .5 : o == 1 ? 1. : 0.) :
          c == 29 ? __ddiv_rn(1.,33.) : c != 0 ? number(0x7fefffffffffffffULL) :
          o == 0 ? (r % 4 == 2 ? 2.25 : -.75) : o == 1 ? (r % 5 <= 2 ? .625 : -2.375) : o == 2 ? -0. : 0.;
      GH_CHECK(bits(output[1 + u64(r) * resident.outputs + o]) == bits(expected));
    }
    GH_CHECK(output[0] == guard && output[1 + u64(rows) * resident.outputs] == guard);
  }
  next_model(c);
}
__global__ void validation_checked(u32 c) {
  GH_CHECK(status.done == 1);
  guards();
  if (!valid_case(c)) {
    GH_CHECK(status.errors & (c == 20 || c == 23 || c == 25 ? numeric : c == 26 || c == 27 || c == 33 ? capacity : model));
    if (c == 27) GH_CHECK(status.required_bytes == 136);
    next_model(c); return;
  }
  succeeded(status); status = {};
  if (c == 30) bins[4] = 65535;
  submitted(predict(&resident,{bins + 1,rows * 2},c == 31 ? 0 : rows,
                    {output + 1,c == 32 ? 0u : u64(rows) * resident.outputs},c < 28,&status));
  prediction_checked<<<1,1,0,cudaStreamTailLaunch>>>(c); submitted(cudaGetLastError());
}
__global__ void model_case(u32 c) {
  initialize();
  switch (c) {
    case 1: trees[2].begin = 4; break;
    case 2: resident.node_count = 11; break;
    case 3: nodes[1].right = 1; break;
    case 4: nodes[1].left = 0; break;
    case 5:
      one_output(1); resident.node_count = 3; trees[1].count = 3;
      nodes[2] = {0,1,2,1,0,0}; nodes[3] = {-1,-1,-1,0,0,0}; break;
    case 6: one_output(1); resident.node_count = 2; trees[1].count = 2; break;
    case 7: nodes[2].threshold = 1; break;
    case 8: nodes[2].missing_left = 2; break;
    case 9: nodes[1].feature = 2; break;
    case 10: nodes[1].threshold = 5; break;
    case 11: metadata[2] = -3; break;
    case 12: features[1].type = FeatureType(99); break;
    case 13: base[1] = number(0x7ff0000000000000ULL); break;
    case 14: nodes[2].value = number(0x7ff0000000000001ULL); break;
    case 15: trees[1].output = 3; break;
    case 16: output_offsets[5] = 5; break;
    case 17: offsets[3] = 8; break;
    case 18: features[2].begin = 4; break;
    case 19: resident.objective = Objective(99); break;
    case 20: base[1] = nodes[6].value = number(0x7fefffffffffffffULL); break;
    case 21: case 22: case 23: case 24: case 25:
      one_output(c == 24 ? 2 : 1); base[1] = number(0x7fefffffffffffffULL);
      nodes[1].value = c == 21 ? 1. : c == 23 ? 0x1.0000000000001p959 : c == 25 ? -base[1] : 0x1p959;
      if (c == 24) nodes[2].value = 0x1p959;
      break;
    case 26: resident.nodes.size = 9; break;
    case 28: case 29:
      resident.outputs = c == 28 ? 3 : 33; resident.tree_count = resident.node_count = 0;
      resident.objective = c == 28 ? Objective::binary_logistic : Objective::multiclass_softmax;
      for (u32 o = 0; o <= resident.outputs; ++o) output_offsets[o + 1] = 0;
      for (u32 o = 0; o < resident.outputs; ++o) base[o + 1] = c == 28 ? (o == 0 ? -0. : o == 1 ? 1000. : -1000.) : 0.;
      break;
  }
  submitted(validate_model(&resident,c == 27 ? Workspace{scratch + 16,135} :
                                     c == 33 ? Workspace{scratch + 17,4095} : workspace(),&status));
  validation_checked<<<1,1,0,cudaStreamTailLaunch>>>(c); submitted(cudaGetLastError());
}

__device__ unsigned char wire_byte(u32 c, u32 i) {
  if (c == 1 && i == 8) return 2;
  if (c == 2 && i == 12) return 99;
  if (c == 3 && i == 40) return 2;
  if (c == 4 && i == 48) return 1;
  if (c == 5 && i == 56) return 0;
  if (c == 6 && i == 72) return 1;
  if (c == 7 && i == 76) return 2;
  if (c == 8 && i == 86) return 0xf0;
  if (c == 8 && i == 87) return 0x7f;
  if (c == 13 && i >= 60 && i < 72) return 0;
  if (c == 14 && i == 38) return 0xf0;
  if (c == 14 && i == 39) return 0x7f;
  if (c == 15 && i == 16) return 0;
  return i < 88 ? literal[i] : 0;
}
__device__ void next_wire(u32 c) {
  if (c + 1 < wire_cases) wire_case<<<1,1,0,cudaStreamTailLaunch>>>(c + 1);
  else roundtrip_start<<<1,1,0,cudaStreamTailLaunch>>>();
  submitted(cudaGetLastError());
}
__global__ void encode_rejection(u32 c);
__global__ void encode_rejection_checked(u32 c) {
  GH_CHECK(status.done == 1 && (status.errors & capacity));
  GH_CHECK(written == 0xface);
  for (auto b : encoded) GH_CHECK(b == std::byte{0xa5});
  guards();
  if (c < 2) { encode_rejection<<<1,1,0,cudaStreamTailLaunch>>>(c + 1); submitted(cudaGetLastError()); }
  else next_wire(0);
}
__global__ void encode_rejection(u32 c) {
  status = {}; written = 0xface;
  for (auto& b : encoded) b = std::byte{0xa5};
  submitted(encode_model(&resident,{encoded + 1,c == 0 ? 87u : 1024u},{&written,c == 1 ? 0u : 1u},
                         c == 2 ? Workspace{scratch + 16,0} : workspace(),&status));
  encode_rejection_checked<<<1,1,0,cudaStreamTailLaunch>>>(c); submitted(cudaGetLastError());
}
__global__ void literal_encoded() {
  succeeded(status); GH_CHECK(written == 88);
  for (u32 i = 0; i < 88; ++i) GH_CHECK(encoded[i + 1] == std::byte(literal[i]));
  GH_CHECK(encoded[0] == std::byte{0xa5} && encoded[89] == std::byte{0xa5});
  encode_rejection<<<1,1,0,cudaStreamTailLaunch>>>(0); submitted(cudaGetLastError());
}
__global__ void wire_checked(u32 c) {
  GH_CHECK(status.done == 1); guards();
  for (u32 i = 0; i < 89; ++i) GH_CHECK(input_bytes[i + 1] == std::byte(wire_byte(c,i)));
  GH_CHECK(input_bytes[0] == std::byte{0xa5} && input_bytes[90] == std::byte{0xa5});
  if (c) { GH_CHECK(status.errors & (c == 11 || c == 12 ? capacity : model)); next_wire(c); return; }
  succeeded(status);
  GH_CHECK(resident.outputs == 1 && resident.tree_count == 1 && resident.node_count == 1);
  GH_CHECK(bits(base[1]) == 0x8000000000000000ULL && nodes[1].feature == -1 && nodes[1].missing_left == 1 && nodes[1].value == 1.5);
  GH_CHECK(resident.schema.total_bins == 2 && features[1].count == 0 && output_offsets[2] == 1);
  status = {};
  for (auto& b : encoded) b = std::byte{0xa5};
  submitted(encode_model(&resident,{encoded + 1,1024},{&written,1},workspace(),&status));
  literal_encoded<<<1,1,0,cudaStreamTailLaunch>>>(); submitted(cudaGetLastError());
}
__global__ void wire_case(u32 c) {
  initialize();
  for (auto& b : input_bytes) b = std::byte{0xa5};
  for (u32 i = 0; i < 89; ++i) input_bytes[i + 1] = std::byte(wire_byte(c,i));
  if (c == 11) resident.nodes.size = 0;
  submitted(decode_model({input_bytes + 1,c == 9 ? 87u : c == 10 ? 89u : 88u},&resident,
                          c == 12 ? Workspace{scratch + 16,0} : workspace(),&status));
  wire_checked<<<1,1,0,cudaStreamTailLaunch>>>(c); submitted(cudaGetLastError());
}
__global__ void roundtrip_prediction() {
  succeeded(status); guards();
  for (u32 r = 0; r < rows; ++r) for (u32 o = 0; o < 4; ++o) {
    const double expected = o == 0 ? (r % 4 == 2 ? 2.25 : -.75) : o == 1 ? (r % 5 <= 2 ? .625 : -2.375) : o == 2 ? -0. : 0.;
    GH_CHECK(bits(output[1 + r * 4 + o]) == bits(expected));
  }
  GH_CHECK(output[0] == guard && output[rows * 4 + 1] == guard);
  printf("model checks passed: %u model/prediction cases, %u wire cases, literal and permuted-forest roundtrip\n",model_cases,wire_cases);
}
__global__ void roundtrip_decoded() {
  succeeded(status); GH_CHECK(resident.node_count == 10 && resident.tree_count == 6);
  GH_CHECK(trees[1].begin == 0 && trees[1].output == 0 && trees[2].begin == 3 && trees[2].output == 1);
  GH_CHECK(nodes[1].feature == 1 && nodes[4].feature == 0 && bits(base[3]) == 0x8000000000000000ULL);
  status = {};
  submitted(predict(&resident,{bins + 1,rows * 2},rows,{output + 1,rows * 4},true,&status));
  roundtrip_prediction<<<1,1,0,cudaStreamTailLaunch>>>(); submitted(cudaGetLastError());
}
__global__ void roundtrip_encoded() {
  succeeded(status); GH_CHECK(written == 440);
  GH_CHECK(encoded[0] == std::byte{0xa5} && encoded[441] == std::byte{0xa5});
  initialize();
  submitted(decode_model({encoded + 1,written},&resident,workspace(),&status));
  roundtrip_decoded<<<1,1,0,cudaStreamTailLaunch>>>(); submitted(cudaGetLastError());
}
__global__ void roundtrip_validated() {
  succeeded(status); status = {};
  for (auto& b : encoded) b = std::byte{0xa5};
  submitted(encode_model(&resident,{encoded + 1,1024},{&written,1},workspace(),&status));
  roundtrip_encoded<<<1,1,0,cudaStreamTailLaunch>>>(); submitted(cudaGetLastError());
}
__global__ void roundtrip_start() {
  initialize(); submitted(validate_model(&resident,workspace(),&status));
  roundtrip_validated<<<1,1,0,cudaStreamTailLaunch>>>(); submitted(cudaGetLastError());
}
}
__global__ void run() { model_case<<<1,1,0,cudaStreamTailLaunch>>>(0); submitted(cudaGetLastError()); }
}
#endif

// GH_SOURCE_CATEGORY: tests
// learn checks
#if GH_MODE == 5
namespace gh::test::learn_suite {
__global__ void run();
namespace {
constexpr u32 good_cases = 35, bad_cases = 41, max_rows = 65, max_outputs = 33;
constexpr u32 large_rows = 2048, large_columns = 11;
constexpr u32 max_columns = 33, node_capacity = 4096, tree_capacity = 128;
constexpr u64 workspace_bytes = 4ULL << 20;
constexpr double guard = -987654.25;
__device__ float values[max_rows * max_columns], targets[max_rows * max_outputs], weights[max_rows];
__device__ std::uint16_t bins[max_rows * max_columns];
__device__ Feature features[max_columns];
__device__ float metadata[8192];
__device__ u32 offsets[max_columns + 1];
__device__ Node nodes[node_capacity + 2];
__device__ Tree trees[tree_capacity + 2];
__device__ double base[max_outputs + 2], margins[max_rows * max_outputs + 2], losses[6];
__device__ u64 output_offsets[max_outputs + 3], input_hash;
__device__ __align__(16) std::byte scratch[workspace_bytes + 32];
__device__ Schema schema;
__device__ Model model;
__device__ Dataset data;
__device__ Training training;
__device__ TuningRecord tuning[max_outputs];
__device__ TrainConfig config;
__device__ Status status;
__device__ observe::Stamp trace_records[2048];
__device__ u32 trace_count;
__device__ float large_values[large_rows*large_columns], large_targets[large_rows];
__device__ std::uint16_t large_bins[large_rows*large_columns];
__device__ double large_margins[large_rows+2];

__device__ u64 bits(double x) { return cuda::std::bit_cast<u64>(x); }
__device__ Workspace workspace() { return {scratch + 16,workspace_bytes}; }
__device__ bool close(double a, double b) { return isfinite(a) && isfinite(b) && fabs(a-b) <= 2e-12 * (1 + fabs(b)); }
__device__ void math_checks() {
  TrainConfig c; c.l2 = 0; c.max_leaf_value = 4;
  detail::Stats<3> cubic{{1,1,1},1};
  auto value = detail::leaf(cubic,c);
  GH_CHECK(value.value == -2 && close(value.benefit,4./3));
  cubic.d[2] = 0; value = detail::leaf(cubic,c);
  GH_CHECK(value.value == -1 && value.benefit == .5);
  cubic.d[2] = 2 - 0x1p-39; value = detail::leaf(cubic,c);
  GH_CHECK(value.value == -1); // denominator 2^-40 is below 1e-12.
  cubic.d[2] = 2 - 0x1p-38; value = detail::leaf(cubic,c);
  GH_CHECK(value.value == -4); // denominator 2^-39 allows the clipped proposal.
  cubic.d[2] = 3; value = detail::leaf(cubic,c); GH_CHECK(value.value == -1);
  cubic = {{0,1,0},1}; value = detail::leaf(cubic,c);
  GH_CHECK(bits(value.value) == bits(-0.) && value.benefit == 0);
  cubic = {{CUDART_INF,1,0},1}; value = detail::leaf(cubic,c);
  GH_CHECK(value.value == 0 && value.benefit == 0);
  cubic = {{1,CUDART_INF,0},1}; value = detail::leaf(cubic,c);
  GH_CHECK(value.value == 0 && value.benefit == 0);
  cubic = {{1,1,CUDART_NAN},1}; value = detail::leaf(cubic,c);
  GH_CHECK(value.value == 0 && value.benefit == 0);
  c.max_leaf_value = 1;
  cubic = {{0x1.fffffffffffffp1023,0x1p-1022,0},1}; value = detail::leaf(cubic,c);
  GH_CHECK(value.value == -1 && value.benefit == 0x1.fffffffffffffp1023);
  cubic = {{0x1.fffffffffffffp1023,1,2},1}; value = detail::leaf(cubic,c);
  GH_CHECK(value.value == -1 && value.benefit == 0x1.fffffffffffffp1023);
  detail::Stats<4> quartic{{1,1,0,-3},1}; c.max_leaf_value = 4;
  value = detail::leaf(quartic,c); GH_CHECK(value.value == -2 && value.benefit == 2);
  c.max_leaf_value = .5; value = detail::leaf(quartic,c);
  GH_CHECK(value.value == -.5 && value.benefit == .3828125);
  c.max_leaf_value = 1; quartic = {{0x1p600,1,0,1},1}; value = detail::leaf(quartic,c);
  GH_CHECK(value.value == -1 && value.benefit == 0x1p600);
  quartic = {{1,0,0,0},1}; value = detail::leaf(quartic,c);
  GH_CHECK(value.value == 0 && value.benefit == 0);
  const detail::Choice a{2,3,1,4,0,0,0}, feature{1,9,1,4,0,0,0}, threshold{2,2,1,4,0,0,0}, missing{2,3,0,4,0,0,0};
  GH_CHECK(detail::better(a,feature).feature == 1);
  GH_CHECK(detail::better(a,threshold).threshold == 2);
  GH_CHECK(detail::better(a,missing).missing_left == 0);
}
__device__ void unbatched() {
  config.batched_roots = config.batched_root_splits = false;
  config.root_counts = RootCounts::per_output;
}
__device__ u64 fingerprint() {
  u64 h = 0;
  for (u32 i = 0; i < max_rows * max_columns; ++i) h ^= mix(u64(cuda::std::bit_cast<u32>(values[i])) + (u64(i) << 32)) ^ mix(bins[i] + u64(i) * 65537);
  for (u32 i = 0; i < max_rows * max_outputs; ++i) h ^= mix(u64(cuda::std::bit_cast<u32>(targets[i])) + u64(i) * 123456789);
  for (u32 i = 0; i < max_rows; ++i) h ^= mix(u64(cuda::std::bit_cast<u32>(weights[i])) + u64(i) * 987654321);
  for (u32 f = 0; f < schema.columns; ++f)
    h ^= mix(features[f].begin + (u64(features[f].count) << 32)) ^ mix(u64(features[f].type) + offsets[f]);
  for (u64 i = 0; i < schema.metadata_count; ++i) h ^= mix(cuda::std::bit_cast<u32>(metadata[i]) + i * 31337);
  h ^= mix(offsets[schema.columns]);
  return h;
}
__device__ void initialize(u32 c) {
  config = {}; config.rounds = 1; config.max_depth = 1; config.min_leaf_rows = 1;
  config.learning_rate = 1; config.l2 = 0; config.histogram = Histogram::global;
  u32 rows = 8, columns = 2, outputs = 1;
  switch (c) {
    case 1: unbatched(); config.splits = SplitPolicy::block256; break;
    case 2: config.batched_root_splits = false; config.root_counts = RootCounts::per_output; break;
    case 3: config.root_counts = RootCounts::shared; break;
    case 4: unbatched(); config.histogram = Histogram::shared; config.splits = SplitPolicy::block256; break;
    case 5: outputs = 3; config.output_tile = 2; config.tree_build = TreeBuild::output_batch; break;
    case 6: outputs = 33; config.tree_build = TreeBuild::output_batch; break;
    case 7: outputs = 7; config.output_tile = 4; break;
    case 8: config.max_depth = 2; columns = 1; break;
    case 9: config.max_depth = 3; config.rounds = 2; columns = 1; unbatched(); config.histogram = Histogram::shared; break;
    case 10: config.max_depth = 0; break;
    case 11: config.min_leaf_rows = 5; break;
    case 12: config.min_child_hessian = 5; break;
    case 13: config.min_gain = 100; break;
    case 14: config.l2 = 4; break;
    case 15: config.min_gain = 16; break;
    case 16: config.max_leaf_value = .5; config.learning_rate = .25; break;
    case 17: config.rounds = 0; outputs = 3; break;
    case 18: config.min_leaf_rows = 4; break;
    case 19: config.objective = Objective::binary_logistic; break;
    case 20: case 21: case 23:
      config.objective = Objective::binary_logistic; config.order = c == 20 ? 3 : 4;
      config.max_leaf_value = .5; config.tree_build = TreeBuild::output_batch;
      outputs = c == 23 ? 3 : 1; config.rounds = c == 23 ? 2 : 1; break;
    case 22: config.objective = Objective::binary_logistic; outputs = 3; config.tree_build = TreeBuild::output_batch; break;
    case 24: config.objective = Objective::multiclass_softmax; config.classes = 3; rows = 12;
      config.rounds = 2; config.tree_build = TreeBuild::output_batch; config.output_tile = 2; break;
    case 25: config.objective = Objective::multiclass_softmax; config.classes = 33; rows = 33;
      config.max_depth = 0; config.tree_build = TreeBuild::output_batch; break;
    case 26: columns = 1; break;
    case 27: columns = 33; rows = 65; outputs = 3; config.splits = SplitPolicy::warp_wide; break;
    case 28: columns = 1; outputs = 3; config.max_depth = 2; config.rounds = 3; config.learning_rate = 2; break;
    case 29: config.histogram = Histogram::automatic; outputs = 3; config.rounds = 2; unbatched(); break;
    case 30: columns = 33; rows = 65; outputs = 3; config.splits = SplitPolicy::warp_wide; break;
    case 31: rows = 1; config.max_depth = 2; break;
    case 32: config.rounds = 0; config.max_histogram_bytes = 1; break;
    case 33: config.rounds = 0; config.max_depth = 7; config.histogram = Histogram::shared;
      rows = 65; columns = 1; break;
    case 34: config.rounds = 0; config.root_counts = RootCounts::shared; columns = 1; break;
  }
  for (auto& x : values) x = -123;
  for (auto& x : targets) x = -123;
  for (auto& x : weights) x = 1;
  for (auto& x : bins) x = 65535;
  u32 total = 0, meta = 0, maximum = 0;
  for (u32 f = 0; f < columns; ++f) {
    const bool categorical = c == 26 || f == 1;
    const u32 count = c == 34 ? 8190 : ((c == 27 || c == 33) && f == 0) ? 40 : categorical ? 3 : 2;
    features[f] = {meta,count,categorical ? FeatureType::categorical : FeatureType::numeric};
    for (u32 k = 0; k < count; ++k) metadata[meta + k] = float(k);
    meta += count; offsets[f] = total;
    const u32 n = count + (categorical ? 1 : 2); total += n; maximum = max(maximum,n);
    for (u32 r = 0; r < rows; ++r) {
      u32 b = r % 4;
      if (c == 8 || c == 9 || c == 28) b = r < 2 ? (r == 0 ? 0 : 1) : r < 4 ? 2 : 3;
      if (c == 27 && f == 0) b = r % 2 ? 41 : 1;
      bins[u64(f) * rows + r] = b; values[u64(r) * columns + f] = b ? float(b - 1) : CUDART_NAN_F;
    }
  }
  offsets[columns] = total;
  for (u32 r = 0; r < rows; ++r) {
    weights[r] = c == 18 && r >= 4 ? 0 : 1;
    for (u32 o = 0; o < outputs; ++o) {
      float y = r % 4 < 2 ? -2 : 2;
      if (c == 8 || c == 9 || c == 28) y = r < 2 ? -4 : r < 4 ? 0 : 2;
      if (c == 28) y += 0x1p23f;
      if (config.objective == Objective::binary_logistic) y = float((r % 4 >= 2) != bool(o & 1));
      else if (config.objective == Objective::multiclass_softmax) y = float(r % config.classes);
      else if (o & 1) y = -y;
      targets[u64(r) * outputs + o] = y;
    }
  }
  schema = {{features,columns},{metadata,meta},{offsets,u64(columns)+1},columns,total,maximum,meta};
  data = {{values,u64(rows)*columns},{targets,u64(rows)*outputs},{weights,rows},rows,columns,outputs};
  if (c == 31) data.weights = {};
  for (auto& x : nodes) x = {-1,-1,-1,0,0,guard};
  for (auto& x : trees) x = {123,456,789};
  for (auto& x : base) x = guard;
  for (auto& x : margins) x = guard;
  for (auto& x : losses) x = guard;
  for (auto& x : output_offsets) x = UINT64_MAX;
  for (u32 i = 0; i < 16; ++i) scratch[i] = scratch[workspace_bytes + 16 + i] = std::byte{0x5a};
  model = {schema,{nodes+1,node_capacity},{trees+1,tree_capacity},{base+1,max_outputs},
    {output_offsets+1,max_outputs+1},0,0,1,Objective::squared_error};
  training = {}; training.model = &model;
  training.margins = {margins+1,max_rows*max_outputs}; training.loss = {losses+1,4};
  training.tuning = {tuning,max_outputs};
  trace_count = 0;
  if constexpr (observe::enabled) training.trace = {{trace_records, 2048}, &trace_count};
  status = {};
}

// Direct row partitions intentionally share no histogram/prefix/leaf helper.
struct Sum { double residual{}, hessian{}; u32 count{}; };
__device__ double step(Sum s) {
  const double denominator = s.hessian + config.l2;
  double value = denominator > 0 ? (s.residual == 0 ? -0. : s.residual / denominator) : 0;
  if (config.max_leaf_value > 0) value = fmax(-config.max_leaf_value,fmin(config.max_leaf_value,value));
  return value;
}
__device__ double benefit(Sum s) {
  const double value = step(s);
  return value * s.residual - .5 * (s.hessian + config.l2) * value * value;
}
__device__ bool left(u32 bin, FeatureType type, u32 threshold, u32 missing) {
  return bin == 0 ? bool(missing) : type == FeatureType::categorical ? bin == threshold : bin <= threshold;
}
__device__ void exhaustive_split(u32 output) {
  Sum total;
  for (u32 r = 0; r < data.rows; ++r) {
    total.residual += weights[r] * (targets[u64(r)*data.outputs+output] - model.base.data[output]);
    total.hessian += weights[r]; ++total.count;
  }
  int feature = -1; u32 threshold = 0, missing = 0;
  double best = config.min_gain, lv = 0, rv = 0;
  if (config.max_depth && total.count >= 2 * config.min_leaf_rows)
    for (u32 f = 0; f < data.columns; ++f) for (u32 t = 0; t < offsets[f+1]-offsets[f]; ++t)
      for (u32 m = 0; m < 2; ++m) {
        Sum l, r;
        for (u32 row = 0; row < data.rows; ++row) {
          Sum& s = left(bins[u64(f)*data.rows+row],features[f].type,t,m) ? l : r;
          s.residual += weights[row] * (targets[u64(row)*data.outputs+output] - model.base.data[output]);
          s.hessian += weights[row]; ++s.count;
        }
        if (l.count < config.min_leaf_rows || r.count < config.min_leaf_rows ||
            l.hessian < config.min_child_hessian || r.hessian < config.min_child_hessian) continue;
        const double gain = benefit(l) + benefit(r) - benefit(total);
        if (gain > best) { best = gain; feature = f; threshold = t; missing = m; lv = step(l); rv = step(r); }
      }
  const Tree tree = model.trees.data[output];
  const Node root = model.nodes.data[tree.begin];
  GH_CHECK(root.feature == feature);
  if (feature < 0) { GH_CHECK(tree.count == 1); GH_CHECK(bits(root.value) == bits(config.learning_rate*step(total))); }
  else {
    GH_CHECK(tree.count == 3 && root.threshold == threshold && root.missing_left == missing);
    GH_CHECK(bits(model.nodes.data[tree.begin+root.left].value) == bits(config.learning_rate*lv));
    GH_CHECK(bits(model.nodes.data[tree.begin+root.right].value) == bits(config.learning_rate*rv));
  }
}
__device__ double traversal(u32 row, u32 output, u32 rounds) {
  double result = model.base.data[output];
  for (u32 round = 0; round < rounds; ++round) {
    const Tree tree = model.trees.data[u64(output)*config.rounds+round];
    GH_CHECK(tree.output == output && tree.count && tree.begin + tree.count <= model.node_count);
    u32 index = 0, visited = 0;
    while (true) {
      GH_CHECK(index < tree.count && ++visited <= tree.count);
      const Node node = model.nodes.data[tree.begin+index];
      if (node.feature < 0) { result = __dadd_rn(result,node.value); break; }
      GH_CHECK(u32(node.feature) < data.columns);
      const u32 b = bins[u64(node.feature)*data.rows+row];
      index = left(b,features[node.feature].type,node.threshold,node.missing_left) ? node.left : node.right;
    }
  }
  return result;
}
__device__ void check_base_loss() {
  double mass = 0;
  for (u32 r = 0; r < data.rows; ++r) mass += weights[r];
  for (u32 o = 0; o < model.outputs; ++o) {
    double sum = 0;
    for (u32 r = 0; r < data.rows; ++r) sum += weights[r] *
      (config.objective == Objective::multiclass_softmax ? double(targets[r] == o) : targets[u64(r)*data.outputs+o]);
    double expected = sum / mass;
    if (config.objective == Objective::binary_logistic) { expected = fmin(1-1e-12,fmax(1e-12,expected)); expected = log(expected)-log1p(-expected); }
    if (config.objective == Objective::multiclass_softmax) expected = log(fmax(1e-12,expected));
    GH_CHECK(close(model.base.data[o],expected));
  }
  for (u32 round = 0; round <= config.rounds; ++round) {
    double expected = 0;
    for (u32 r = 0; r < data.rows; ++r) {
      if (weights[r] == 0) continue;
      if (config.objective == Objective::multiclass_softmax) {
        double maximum = -CUDART_INF, denominator = 0;
        for (u32 o = 0; o < model.outputs; ++o) maximum = fmax(maximum,traversal(r,o,round));
        for (u32 o = 0; o < model.outputs; ++o) denominator += exp(traversal(r,o,round)-maximum);
        expected += weights[r] * (maximum - traversal(r,u32(targets[r]),round) + log(denominator));
      } else for (u32 o = 0; o < model.outputs; ++o) {
        const double margin = traversal(r,o,round), y = targets[u64(r)*data.outputs+o];
        expected += weights[r] * (config.objective == Objective::squared_error ? .5*(margin-y)*(margin-y) :
          fmax(margin,0.) - y*margin + log1p(exp(-fabs(margin))));
      }
    }
    expected /= mass * (config.objective == Objective::multiclass_softmax ? 1 : model.outputs);
    GH_CHECK(close(training.loss.data[round],expected));
  }
}
__device__ void guards() {
  GH_CHECK(nodes[0].value == guard && nodes[node_capacity+1].value == guard);
  GH_CHECK(trees[0].begin == 123 && trees[tree_capacity+1].begin == 123);
  GH_CHECK(base[0] == guard && base[max_outputs+1] == guard);
  GH_CHECK(margins[0] == guard && margins[max_rows*max_outputs+1] == guard);
  GH_CHECK(losses[0] == guard && losses[5] == guard);
  GH_CHECK(output_offsets[0] == UINT64_MAX && output_offsets[max_outputs+2] == UINT64_MAX);
  for (u32 i = 0; i < 16; ++i) GH_CHECK(scratch[i] == std::byte{0x5a} && scratch[workspace_bytes+16+i] == std::byte{0x5a});
  GH_CHECK(fingerprint() == input_hash);
}
__global__ void start(u32 c);
__global__ void large_start();
__device__ void next(u32 c) {
  if (c+1 < good_cases+bad_cases) { start<<<1,1,0,cudaStreamTailLaunch>>>(c+1); submitted(cudaGetLastError()); }
  else { large_start<<<1,256,0,cudaStreamTailLaunch>>>(); submitted(cudaGetLastError()); }
}
__global__ void validated(u32 c) { succeeded(status); guards(); next(c); }
__device__ u64 middle(const u64* observations) {
  for (u32 i = 0; i < 5; ++i) {
    u32 lower = 0, higher = 0;
    for (u32 j = 0; j < 5; ++j) { lower += observations[j] < observations[i]; higher += observations[j] > observations[i]; }
    if (lower <= 2 && higher <= 2) return observations[i];
  }
  GH_CHECK(false); return 0;
}
__global__ void trained(u32 c) {
  GH_CHECK(status.done == 1); guards();
  if (c >= good_cases) {
    GH_CHECK(status.errors && !(status.errors & runtime));
    if (c-good_cases == 40) {
      GH_CHECK(status.errors & numeric);
      for (u32 r = 0; r < data.rows; ++r) GH_CHECK(training.margins.data[r] == 0);
    }
    next(c); return;
  }
  if (status.errors) printf("learn fixture %u returned errors=%u\n",c,status.errors);
  succeeded(status);
  if constexpr (observe::enabled) {
    GH_CHECK(trace_count >= 6 && trace_count <= 2048 && !(trace_count % 2));
    for (u32 i = 0; i < trace_count; i += 2) {
      GH_CHECK(trace_records[i].stage == trace_records[i + 1].stage);
      GH_CHECK(!trace_records[i].end && trace_records[i + 1].end);
      GH_CHECK(trace_records[i + 1].ticks > trace_records[i].ticks);
      if (i) GH_CHECK(trace_records[i].ticks >= trace_records[i - 1].ticks);
    }
  } else GH_CHECK(trace_count == 0);
  GH_CHECK(model.outputs == (config.objective == Objective::multiclass_softmax ? config.classes : data.outputs));
  GH_CHECK(model.objective == config.objective && model.tree_count == u64(model.outputs)*config.rounds);
  GH_CHECK(model.node_count <= node_capacity && training.workspace_bytes <= workspace_bytes);
  GH_CHECK(training.histogram_bytes <= config.max_histogram_bytes);
  if (!config.rounds) GH_CHECK(!training.histogram_bytes && !training.derivative_bytes && !training.tree_state_bytes &&
      !training.frontier_capacity && !training.tree_capacity && !training.output_capacity);
  for (u32 o = 0; o < model.outputs; ++o) {
    GH_CHECK(tuning[o].measured == (c == 29));
    if (c == 29) {
      GH_CHECK(tuning[o].output == o && (tuning[o].selected == Histogram::global || tuning[o].selected == Histogram::shared));
      for (u32 i = 0; i < 5; ++i) GH_CHECK(tuning[o].global_ticks[i] && tuning[o].shared_ticks[i]);
      GH_CHECK(tuning[o].selected == (middle(tuning[o].shared_ticks) < middle(tuning[o].global_ticks) ? Histogram::shared : Histogram::global));
    }
  }
  for (u32 o = 0; o <= model.outputs; ++o) GH_CHECK(model.output_offsets.data[o] == u64(o)*config.rounds);
  for (u32 r = 0; r < data.rows; ++r) for (u32 o = 0; o < model.outputs; ++o)
    GH_CHECK(bits(training.margins.data[u64(r)*model.outputs+o]) == bits(traversal(r,o,config.rounds)));
  if (config.rounds == 1 && config.max_depth <= 1 && config.objective == Objective::squared_error && data.rows == 8)
    for (u32 o = 0; o < model.outputs; ++o) exhaustive_split(o);
  if (c == 8) for (u32 r = 0; r < data.rows; ++r) GH_CHECK(training.margins.data[r] == targets[r]);
  if (c == 20 || c == 21) for (u32 r = 0; r < data.rows; ++r) GH_CHECK(training.margins.data[r] == (targets[r] ? .5 : -.5));
  check_base_loss();
  status = {}; submitted(validate_model(&model,workspace(),&status));
  validated<<<1,1,0,cudaStreamTailLaunch>>>(c); submitted(cudaGetLastError());
}
__global__ void start(u32 c) {
  initialize(c < good_cases ? c : 0);
  Workspace work = workspace();
  Array<const std::uint16_t> input_bins{bins,u64(data.rows)*data.columns};
  if (c >= good_cases) switch (c-good_cases) {
    case 0: data.rows = 0; break;
    case 1: data.outputs = 0; break;
    case 2: --input_bins.size; break;
    case 3: --data.targets.size; break;
    case 4: --data.weights.size; break;
    case 5: training.margins.size = data.rows-1; break;
    case 6: training.loss.size = config.rounds; break;
    case 7: model.base.size = 0; break;
    case 8: model.trees.size = 0; break;
    case 9: model.nodes.size = 0; break;
    case 10: work.bytes = 1; break;
    case 11: config.max_histogram_bytes = 1; break;
    case 12: config.max_device_bytes = 1; break;
    case 13: targets[0] = cuda::std::bit_cast<float>(0x7fc00000u); break;
    case 14: weights[0] = CUDART_INF_F; break;
    case 15: weights[0] = -1; break;
    case 16: for (u32 r = 0; r < data.rows; ++r) weights[r] = 0; break;
    case 17: config.objective = Objective::binary_logistic; break;
    case 18: config.objective = Objective::multiclass_softmax; config.classes = 3; targets[0] = 3; break;
    case 19: config.order = 1; break;
    case 20: config.order = 3; break;
    case 21: config.histogram = Histogram(99); break;
    case 22: config.splits = SplitPolicy(99); break;
    case 23: config.tree_build = TreeBuild(99); break;
    case 24: config.root_counts = RootCounts(99); break;
    case 25: config.l2 = -1; break;
    case 26: config.learning_rate = 0; break;
    case 27: config.min_child_hessian = CUDART_NAN; break;
    case 28: config.max_depth = 31; break;
    case 29: config.min_leaf_rows = 0; break;
    case 30: config.output_tile = 0; break;
    case 31: config.objective = Objective(99); break;
    case 32: config.tree_build = TreeBuild::output_batch; config.batched_root_splits = false; break;
    case 33: config.batched_roots = false; break;
    case 34: model.output_offsets.size = 1; break;
    case 35: config.objective = Objective::binary_logistic; config.order = 3; config.max_leaf_value = .5; break;
    case 36: config.objective = Objective::binary_logistic; config.order = 4; config.tree_build = TreeBuild::output_batch; break;
    case 37: config.objective = Objective::binary_logistic; config.order = 3; config.max_leaf_value = .5;
      config.tree_build = TreeBuild::output_batch; config.histogram = Histogram::shared; break;
    case 38: config.objective = Objective::binary_logistic; config.order = 4; config.max_leaf_value = .5;
      config.tree_build = TreeBuild::output_batch; config.splits = SplitPolicy::warp_wide; break;
    case 39: targets[0] = CUDART_NAN_F; weights[0] = 0; break;
    case 40:
      config.learning_rate = 0x1p1023; config.max_depth = 2;
      for (u32 r = 0; r < data.rows; ++r) {
        const u32 group = r / 4, bit = r & 1;
        targets[r] = 3.f * (float(group)+float(bit)-1.f);
        bins[r] = 1+group; bins[data.rows+r] = 1+bit;
        values[u64(r)*2] = float(group); values[u64(r)*2+1] = float(bit);
      }
      break;
  }
  input_hash = fingerprint();
  submitted(train(data,&schema,input_bins,config,&training,work,&status));
  trained<<<1,1,0,cudaStreamTailLaunch>>>(c); submitted(cudaGetLastError());
}
__global__ void large_validated() {
  succeeded(status);
  GH_CHECK(nodes[0].value == guard && nodes[node_capacity+1].value == guard);
  for (u32 i = 0; i < 16; ++i) GH_CHECK(scratch[i] == std::byte{0x5a} && scratch[workspace_bytes+16+i] == std::byte{0x5a});
  printf("learn checks passed: higher-order landmarks, %u small success fixtures, %u rejection fixtures, one 2048-live-node frontier; calibration records checked\n",good_cases,bad_cases);
}
__global__ void large_validate() {
  status = {}; submitted(validate_model(&model,workspace(),&status));
  large_validated<<<1,1,0,cudaStreamTailLaunch>>>(); submitted(cudaGetLastError());
}
__global__ void large_checked() {
  succeeded(status);
  GH_CHECK(model.base.data[0] == 1023.5 && model.tree_count == 1 && model.node_count == 4095);
  GH_CHECK(training.frontier_capacity == 2048 && training.tree_capacity == 4095);
  GH_CHECK(model.trees.data[0].begin == 0 && model.trees.data[0].count == 4095 && model.trees.data[0].output == 0);
  GH_CHECK(training.loss.data[0] == (double(large_rows)*large_rows-1)/24 && training.loss.data[1] == 0);
  GH_CHECK(large_margins[0] == guard && large_margins[large_rows+1] == guard);
  for (u32 i = blockIdx.x*blockDim.x+threadIdx.x; i < 4095; i += gridDim.x*blockDim.x) {
    const Node n = model.nodes.data[i];
    if (i < 2047) {
      u32 depth = 0; for (u32 x = i+1; x > 1; x >>= 1) ++depth;
      GH_CHECK(n.feature == int(depth) && n.threshold == 1 && n.missing_left == 0);
      GH_CHECK(n.left == int(2*i+1) && n.right == int(2*i+2));
    } else GH_CHECK(n.feature == -1 && n.value == double(i-2047)-1023.5);
  }
  for (u32 r = blockIdx.x*blockDim.x+threadIdx.x; r < large_rows; r += gridDim.x*blockDim.x) {
    GH_CHECK(training.margins.data[r] == r && large_targets[r] == r);
    for (u32 f = 0; f < large_columns; ++f) {
      const u32 bit = (r >> (10-f)) & 1;
      GH_CHECK(large_bins[u64(f)*large_rows+r] == 1+bit && large_values[u64(r)*large_columns+f] == bit);
    }
  }
  if (!blockIdx.x && !threadIdx.x) { large_validate<<<1,1,0,cudaStreamTailLaunch>>>(); submitted(cudaGetLastError()); }
}
__global__ void large_start() {
  if (!threadIdx.x) {
    initialize(0); config.max_depth = 12;
    for (u32 f = 0; f < large_columns; ++f) {
      features[f] = {f,1,FeatureType::numeric}; metadata[f] = .5f; offsets[f] = 3*f;
    }
    offsets[large_columns] = 3*large_columns;
    schema = {{features,large_columns},{metadata,large_columns},{offsets,large_columns+1},large_columns,3*large_columns,3,large_columns};
    data = {{large_values,large_rows*large_columns},{large_targets,large_rows},{},large_rows,large_columns,1};
    training.margins = {large_margins+1,large_rows}; large_margins[0] = large_margins[large_rows+1] = guard;
  }
  __syncthreads();
  for (u32 r = threadIdx.x; r < large_rows; r += blockDim.x) {
    large_targets[r] = float(r);
    for (u32 f = 0; f < large_columns; ++f) {
      const u32 bit = (r >> (10-f)) & 1;
      large_values[u64(r)*large_columns+f] = float(bit); large_bins[u64(f)*large_rows+r] = 1+bit;
    }
  }
  __syncthreads();
  if (!threadIdx.x) {
    submitted(train(data,&schema,{large_bins,large_rows*large_columns},config,&training,workspace(),&status));
    large_checked<<<16,256,0,cudaStreamTailLaunch>>>(); submitted(cudaGetLastError());
  }
}
}
__global__ void run() { math_checks(); start<<<1,1>>>(0); submitted(cudaGetLastError()); }
}
#endif

// GH_SOURCE_CATEGORY: tests
// observe checks
#if GH_MODE == 6
namespace gh::test::observe_suite {
__global__ void run();
namespace {
using namespace gh::observe;
__device__ Stamp stages[2];
__device__ u32 stage_count;
__device__ Status trace_status, full_status, boundary_status, empty_status, sample_status, summary_status[6];
__device__ Sample synthetic[6][samples], measured[samples];
__device__ Sample empty;
__device__ Summary results[6], timing;
__device__ u64 work[64], clock_begin[64], clock_end[64];
__device__ u32 clock_sm[64], disabled_value;
__global__ void work_and_clocks(u32 ordinal) {
  u64 a,b; u32 sm;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(a));
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  u64 value = ordinal + blockIdx.x;
  for (u32 i = 0; i < 256; ++i) value = mix(value);
  work[blockIdx.x] = value;
  __nanosleep(10000);
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(b));
  clock_begin[blockIdx.x] = a; clock_end[blockIdx.x] = b; clock_sm[blockIdx.x] = sm;
}
__global__ void fixture() {
  for (u32 k = 0; k < 6; ++k) {
    u64 cursor = 100;
    for (u32 i = 0; i < samples; ++i) {
      const auto slot = schedule(i);
      u64 duration = slot.variant ? 100 : 200;
      if (k == 1) duration *= 7;
      if (k == 2) duration = slot.variant ? 200 : 100;
      if (k == 3 && !slot.warmup) {
        const bool high = slot.pair % 2 == 0;
        duration = slot.pair == 14 ? 100 : slot.variant ? 100 : high ? 200 : 50;
      }
      synthetic[k][i] = {cursor,cursor + duration};
      cursor += duration + 5;
    }
    if (k == 4) synthetic[k][8].end = synthetic[k][8].begin;
    if (k == 5) synthetic[k][8].begin = synthetic[k][7].end - 1;
  }
}
__global__ void run_sample(u32 ordinal);
__global__ void empty_check() {
  succeeded(empty_status);
  GH_CHECK(empty.end > empty.begin);
  printf("GH_RAW_EMPTY globaltimer-cdp-tail-v1 begin=%llu end=%llu\n",
    static_cast<unsigned long long>(empty.begin),static_cast<unsigned long long>(empty.end));
  run_sample<<<1,1,0,cudaStreamTailLaunch>>>(0);
  submitted(cudaGetLastError());
}
__global__ void empty_sample() {
  submitted(boundary({&empty,1},0,false,&empty_status));
  submitted(finish(&empty_status));
  submitted(boundary({&empty,1},0,true,&empty_status));
  empty_check<<<1,1,0,cudaStreamTailLaunch>>>();
  submitted(cudaGetLastError());
}
__global__ void final_check() {
  succeeded(sample_status);
  GH_CHECK(timing.ratio > 0 && timing.lower <= timing.ratio && timing.upper >= timing.ratio);
  GH_CHECK(timing.minimum_ticks > 0);
  for (u32 i = 0; i < samples; ++i) {
    GH_CHECK(measured[i].end > measured[i].begin);
    if (i) GH_CHECK(measured[i].begin >= measured[i-1].end);
    printf("GH_RAW_SAMPLE globaltimer-cdp-tail-v1 ordinal=%u variant=%u warmup=%u begin=%llu end=%llu\n",
      i,schedule(i).variant,unsigned(schedule(i).warmup),
      static_cast<unsigned long long>(measured[i].begin),static_cast<unsigned long long>(measured[i].end));
  }
  printf("GH_GPU_ACTIVITY observe samples=36 checks=pass; timing qualification, no ranking\n");
}
__global__ void checked_sample(u32 ordinal) {
  succeeded(trace_status);
  bool other_sm = false;
  for (u32 i = 0; i < 64; ++i) {
    GH_CHECK(clock_end[i] > clock_begin[i]);
    GH_CHECK(clock_begin[i] >= measured[ordinal].begin && clock_end[i] <= measured[ordinal].end);
    other_sm |= clock_sm[i] != clock_sm[0];
    u64 expected = ordinal + i;
    for (u32 j = 0; j < 256; ++j) expected = mix(expected);
    GH_CHECK(work[i] == expected);
  }
  GH_CHECK(other_sm);
  if (ordinal + 1 < samples) run_sample<<<1,1,0,cudaStreamTailLaunch>>>(ordinal+1);
  else {
    submitted(summarize({measured,samples},&timing,&sample_status));
    final_check<<<1,1,0,cudaStreamTailLaunch>>>();
  }
  submitted(cudaGetLastError());
}
__global__ void run_sample(u32 ordinal) {
  trace_status = {};
  submitted(boundary({measured,samples},ordinal,false,&sample_status));
  work_and_clocks<<<64,1>>>(ordinal);
  submitted(cudaGetLastError());
  submitted(finish(&trace_status));
  submitted(boundary({measured,samples},ordinal,true,&sample_status));
  checked_sample<<<1,1,0,cudaStreamTailLaunch>>>(ordinal);
  submitted(cudaGetLastError());
}
__global__ void summary_check() {
  for (u32 i = 0; i < 4; ++i) succeeded(summary_status[i]);
  for (u32 i = 4; i < 6; ++i) GH_CHECK(summary_status[i].done && summary_status[i].errors == numeric);
  GH_CHECK(fabs(results[0].ratio - 2) < 1e-12);
  GH_CHECK(fabs(results[0].upper - results[0].lower) < 1e-12);
  GH_CHECK(fabs(results[1].ratio - results[0].ratio) < 1e-12);
  GH_CHECK(results[1].minimum_ticks == 7 * results[0].minimum_ticks);
  GH_CHECK(fabs(results[2].ratio - 0.5) < 1e-12);
  GH_CHECK(fabs(results[3].ratio - 1) < 1e-12);
  GH_CHECK(results[3].lower < 1 && results[3].upper > 1);
  GH_CHECK(fabs(results[3].log_stddev - log(2.0)) < 1e-12);
  empty_sample<<<1,1,0,cudaStreamTailLaunch>>>();
  submitted(cudaGetLastError());
}
__global__ void trace_check() {
  succeeded(trace_status);
  GH_CHECK(full_status.done && full_status.errors == capacity);
  GH_CHECK(boundary_status.done && boundary_status.errors == capacity);
  GH_CHECK(disabled_value == 73 && stage_count == 2);
  GH_CHECK(stages[0].stage == Stage::gradient && stages[1].stage == Stage::gradient);
  GH_CHECK(!stages[0].end && stages[1].end && stages[0].iteration == 9 && stages[1].iteration == 9);
  GH_CHECK(stages[1].ticks > stages[0].ticks);
  fixture<<<1,1>>>();
  for (u32 k = 0; k < 6; ++k) submitted(summarize({synthetic[k],samples},results+k,summary_status+k));
  summary_check<<<1,1,0,cudaStreamTailLaunch>>>();
  submitted(cudaGetLastError());
}
}
// Deliberately public symbol for offline PTX/SASS erasure inspection; exercised.
__global__ void disabled_probe() {
  mark<false>({},Stage::gradient,0,false,nullptr);
  disabled_value = 73;
}
__global__ void run() {
  u32 seen[2]{};
  for (u32 i = 0; i < samples; ++i) {
    const auto slot = schedule(i);
    GH_CHECK(slot.variant < 2 && slot.pair < (slot.warmup ? warmups : pairs));
    if (!slot.warmup) ++seen[slot.variant];
    if (i % 2) GH_CHECK(slot.variant != schedule(i-1).variant);
    if (i >= 2) GH_CHECK(slot.variant != schedule(i-2).variant);
  }
  GH_CHECK(seen[0] == 15 && seen[1] == 15);
  disabled_probe<<<1,1>>>();
  submitted(mark<true>({{stages,2},&stage_count},Stage::gradient,9,false,&trace_status));
  work_and_clocks<<<64,1>>>(0);
  submitted(mark<true>({{stages,2},&stage_count},Stage::gradient,9,true,&trace_status));
  submitted(mark<true>({{stages,2},&stage_count},Stage::loss,9,true,&full_status));
  submitted(boundary({},0,false,&boundary_status));
  submitted(finish(&boundary_status));
  submitted(finish(&trace_status)); submitted(finish(&full_status));
  trace_check<<<1,1,0,cudaStreamTailLaunch>>>();
  submitted(cudaGetLastError());
}
}
#endif

// GH_SOURCE_CATEGORY: tests
// metrics checks
#if GH_MODE == 7
namespace gh::test::metrics_suite {
__global__ void run();
namespace {
constexpr u32 capacity_cells = 70000, cases = 35;
__device__ double target[capacity_cells], prediction[capacity_cells], weights[capacity_cells];
__device__ Metric entries[70], candidate_entries[70];
__device__ MetricReport report, candidate;
__device__ MetricVerdict verdict[70];
__device__ MetricGate gate;
__device__ Status status;
__device__ Status numerical_status[1024];
__device__ SignalReport signal;
__device__ __align__(16) std::byte scratch[5 << 20];

// Independent small CPython-style partial expansion. Tests only, bounded to
// 128 operands; production exact summation uses integer limbs instead.
__device__ double expansion(const double* values, u32 size) {
  double partial[128]; u32 count = 0;
  for (u32 k = 0; k < size; ++k) {
    double x = values[k]; u32 used = 0;
    for (u32 j = 0; j < count; ++j) {
      double y = partial[j];
      if (fabs(x) < fabs(y)) { const double t = x; x = y; y = t; }
      const double hi = __dadd_rn(x, y), lo = __dsub_rn(y, __dsub_rn(hi, x));
      if (lo) partial[used++] = lo;
      x = hi;
    }
    partial[used++] = x; count = used;
  }
  double hi = 0, lo = 0;
  if (count) {
    hi = partial[--count];
    while (count) {
      const double x = hi, y = partial[--count];
      hi = __dadd_rn(x, y); lo = __dsub_rn(y, __dsub_rn(hi, x));
      if (lo) break;
    }
    if (count && ((lo < 0 && partial[count - 1] < 0) || (lo > 0 && partial[count - 1] > 0))) {
      const double y = __dmul_rn(lo, 2.0), x = __dadd_rn(hi, y);
      if (y == __dsub_rn(x, hi)) hi = x;
    }
  }
  return hi;
}
__global__ void numerical_checks() {
  const u32 test = blockIdx.x * blockDim.x + threadIdx.x;
  double values[128]; const u32 n = test % 128 + 1;
  for (u32 i = 0; i < n; ++i) {
    const u64 bits = mix(u64(test) * 128 + i);
    const u32 exponent = test % 3 == 0 ? 0 : test % 3 == 1 ? 1023 + i % 3 : u32(bits % 2000);
    values[i] = cuda::std::bit_cast<double>((u64(exponent) << 52) | (bits & 0xfffffffffffffULL));
  }
  Status& local = numerical_status[test]; local = {};
  const double actual = detail::positive_sum(n, [&](u32 i) { return values[i]; }, &local);
  GH_CHECK(local.errors == 0);
  GH_CHECK(cuda::std::bit_cast<u64>(actual) == cuda::std::bit_cast<u64>(expansion(values, n)));
  if (test) return;
  const auto one = [](u32) { return 1.0; };
  const double maximum = cuda::std::bit_cast<double>(0x7fefffffffffffffULL);
  GH_CHECK(detail::scaled_mean<2>(2, [=](u32) { return maximum; }, one, 2, &local) == maximum);
  GH_CHECK(detail::scaled_mean<1>(2, [=](u32) { return maximum; }, one, 2, &local) == maximum);
  const double tiny = cuda::std::bit_cast<double>(1ULL);
  GH_CHECK(detail::scaled_mean<2>(2, [=](u32) { return tiny; }, one, 2, &local) == tiny);
  GH_CHECK(detail::scaled_mean<1>(2, [=](u32 i) { return i ? 2.0 : INFINITY; },
    [](u32 i) { return i ? 1.0 : 0.0; }, 1, &local) == 2);
  // Weighted scale exponents exceed binary64 while the final mean is finite.
  GH_CHECK(detail::scaled_mean<2>(2, [=](u32) { return 0x1p1000; },
    [](u32) { return 0x1p1000; }, 0x1p1001, &local) == 0x1p1000);
  GH_CHECK(local.errors == 0);
  detail::PositiveSum overflow; overflow.add(maximum, &local); overflow.add(maximum, &local);
  overflow.value(&local); GH_CHECK(local.errors == numeric);
  local = {}; detail::PositiveSum invalid; invalid.add(-1, &local); GH_CHECK(local.errors == numeric);
  local = {};
  GH_CHECK(detail::positive_sum(3, [](u32 i) { return i ? 0x1p-53 : 1.0; }, &local) == 0x1.0000000000001p0);
  GH_CHECK(detail::positive_sum(2, [](u32 i) { return i ? 0x1p-53 : 1.0; }, &local) == 1.0);
  GH_CHECK(cuda::std::bit_cast<u64>(detail::positive_sum(2, [](u32) { return -0.0; }, &local)) == 0);
  // NumPy's eight-lane leaf differs deliberately from a sequential fold.
  GH_CHECK(detail::numpy_sum(9, [](u32 i) { return i == 0 ? 0x1p53 : i == 8 ? -0x1p53 : 1.0; }) == 6.0);
  GH_CHECK(detail::column_sum(9, 2, [](u32 i) { return i == 0 ? 0x1p53 : i == 8 ? -0x1p53 : 1.0; }) == 0.0);
  for (u32 n : {0u, 1u, 7u, 8u, 127u, 128u, 129u, 257u, 1025u})
    GH_CHECK(detail::numpy_sum(n, [](u32) { return 1.0; }) == n);
}

__device__ MetricInput input_for(u32 id) {
  MetricInput d{{target, capacity_cells}, {prediction, capacity_cells}, {}, 4, 1,
    MetricTask::binary, MetricProfile::real_data};
  if (id == 0 || id == 1) { d.task = MetricTask::regression; d.outputs = 2; }
  if (id == 0 || id == 3 || id == 5 || id == 8 || id == 10 || id == 12 || id == 13 || id == 14 || id == 16 || id == 17)
    d.profile = MetricProfile::synthetic;
  if (id == 0 || id == 3 || id == 12 || id == 13 || id == 14 || id == 15 || id == 16 || id == 17) d.weights = {weights, 4};
  if (id == 4 || id == 5 || id == 6 || id == 23 || id == 24 || id == 25) { d.task = MetricTask::multilabel; d.outputs = 5; }
  if (id == 7 || id == 8 || id == 9 || id == 10 || id == 18) { d.task = MetricTask::multiclass; d.outputs = 2; }
  if (id == 11) { d.task = MetricTask::regression; d.rows = 1; }
  if (id >= 26 && id <= 28) d.rows = id == 26 ? 257 : id == 27 ? 1025 : 65537;
  if (id >= 29 && id <= 32) {
    d.rows = 17;
    if (id >= 31) { d.outputs = 3; d.task = MetricTask::multilabel; }
    if (id == 30 || id == 32) { d.profile = MetricProfile::synthetic; d.weights = {weights,17}; }
  }
  if (id >= 33) d.task = MetricTask::regression;
  if (id == 19) d.rows = 0;
  if (id == 20) d.predictions.size = 0;
  return d;
}
__global__ void fixture(u32 id) {
  const auto d = input_for(id);
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < u64(d.rows) * d.outputs; i += u64(gridDim.x) * blockDim.x) {
    const u32 row = u32(i / d.outputs), col = u32(i % d.outputs);
    double y = row == 1 || row == 2, p = row == 0 ? .125 : row == 3 ? .875 : .5;
    if (d.task == MetricTask::regression) { y = double(row * 2 + col); p = y + (id == 0 ? 2 : 1); }
    if (d.task == MetricTask::multilabel) { y = col == 0 ? 0 : col == 1 ? 1 : double(row == 1 || row == 2); p = .5; }
    if (d.task == MetricTask::multiclass) { y = row % 2; p = .5; }
    if (id == 6) y = 0;
    if (id == 9 || id == 10) p += col == row % 2 ? 0x1p-24 : 0;
    if (id == 18) p = .6;
    if (id == 23) y = 1;
    if (id == 24) { y = col == 0; p = .5; }
    if (id == 25) { y = (row + col) % 2; p = y; }
    if (id >= 26 && id <= 28) { y = row % 2; p = y; }
    if (id >= 29 && id <= 32) { y = double((row * 7 + col * 3) % 5 < 2); p = double((row * 3 + col * 2) % 5) / 4; }
    if (id >= 33) { y = 2; p = id == 33 ? 2 : 3; }
    if (id == 16 && i == 0) p = INFINITY;
    if (id == 17 && i == 0) y = .25;
    prediction[i] = p;
    if (d.task != MetricTask::multiclass) target[i] = y;
    if (i < d.rows) {
      if (d.task == MetricTask::multiclass) target[i] = i % 2;
      weights[i] = id == 3 ? (i == 1 || i == 3 ? 2.0 : 1.0) : 1.0;
      if (id == 30 || id == 32) weights[i] = i % 4 ? ldexp(1.0, int(i % 9) - 4) : 0.0;
      if (id == 12) weights[i] = 0;
      if (id == 13 && i == 0) weights[i] = -1;
      if (id == 14 && i == 0) weights[i] = INFINITY;
    }
  }
}
__device__ Metric get(MetricName name, u32 output = aggregate_output) {
  for (u32 i = 0; i < report.count; ++i) if (entries[i].name == name && entries[i].output == output) return entries[i];
  GH_CHECK(false); return {};
}
__device__ void exact(MetricName name, double value, u32 output = aggregate_output) {
  const auto metric = get(name, output); GH_CHECK(metric.available == 1); GH_CHECK(metric.value == value);
}
struct RankAnswer { double ap, auc; };
// Exhaust every known score threshold; no production sorting/scan is reused.
// At most 51 cells and five groups keep reference reductions below eight terms.
__device__ RankAnswer rank_oracle(MetricInput d, u32 output, bool pooled) {
  const u32 size = pooled ? d.rows * d.outputs : d.rows;
  u32 tp[5], fp[5], count = 0, positives = 0;
  for (u32 k = 0; k < size; ++k) positives += target[pooled ? k : k * d.outputs + output] == 1;
  const u32 negatives = size - positives;
  for (int level = 4; level >= 0; --level) {
    u32 p = 0, n = 0, tied = 0;
    for (u32 k = 0; k < size; ++k) {
      const u32 i = pooled ? k : k * d.outputs + output;
      tied += prediction[i] == double(level) / 4;
      if (prediction[i] >= double(level) / 4) { p += target[i] == 1; n += target[i] == 0; }
    }
    if (tied) { tp[count] = p; fp[count++] = n; }
  }
  double ap = 0, auc = 0, previous_fpr = 0, previous_tpr = 0;
  for (u32 k = count; k; --k) {
    const u32 j = k - 1;
    const double recall = __ddiv_rn(double(tp[j]), positives);
    const double next = j ? __ddiv_rn(double(tp[j - 1]), positives) : 0;
    ap = __dadd_rn(ap, __dmul_rn(__dsub_rn(next, recall), __ddiv_rn(double(tp[j]), tp[j] + fp[j])));
  }
  for (u32 k = 0; k < count; ++k) {
    if (k && k + 1 < count && int(tp[k + 1]) - 2 * int(tp[k]) + int(tp[k - 1]) == 0 &&
        int(fp[k + 1]) - 2 * int(fp[k]) + int(fp[k - 1]) == 0) continue;
    const double fpr = __ddiv_rn(double(fp[k]), negatives), tpr = __ddiv_rn(double(tp[k]), positives);
    auc = __dadd_rn(auc, __ddiv_rn(__dmul_rn(__dsub_rn(fpr, previous_fpr), __dadd_rn(tpr, previous_tpr)), 2.0));
    previous_fpr = fpr; previous_tpr = tpr;
  }
  if (d.profile == MetricProfile::synthetic) {
    double mass[64], terms[5];
    for (u32 k = 0; k < size; ++k) mass[k] = target[k * d.outputs + output] == 1 ? weights[k] : 0;
    const double positive = expansion(mass, size);
    for (u32 k = 0; k < size; ++k) mass[k] = target[k * d.outputs + output] == 0 ? weights[k] : 0;
    const double negative = expansion(mass, size);
    double before = 0, correction = 0;
    for (u32 level = 0; level < 5; ++level) {
      for (u32 k = 0; k < size; ++k) mass[k] = prediction[k * d.outputs + output] == double(level) / 4 && target[k * d.outputs + output] == 1 ? __ddiv_rn(weights[k], positive) : 0;
      const double p = expansion(mass, size);
      for (u32 k = 0; k < size; ++k) mass[k] = prediction[k * d.outputs + output] == double(level) / 4 && target[k * d.outputs + output] == 0 ? __ddiv_rn(weights[k], negative) : 0;
      const double n = expansion(mass, size);
      terms[level] = __dmul_rn(p, __dadd_rn(before, __dmul_rn(.5, n)));
      const double change = __dsub_rn(n, correction), updated = __dadd_rn(before, change);
      correction = __dsub_rn(__dsub_rn(updated, before), change); before = updated;
    }
    auc = expansion(terms, 5);
  }
  return {-ap, auc};
}
__global__ void start(u32 id);
__global__ void gate_start(u32 id);
__global__ void check(u32 id) {
  GH_CHECK(status.done == 1);
  const auto d = input_for(id);
  if (id >= 12 && id <= 22) {
    const u32 error = id <= 14 || id == 16 || id == 17 || id == 18 ? input : id >= 21 ? capacity : shape;
    GH_CHECK(status.errors & error);
    if (id == 21) GH_CHECK(status.required_bytes > 1);
  } else {
    if (status.errors) printf("metric case %u errors=%u required_bytes=%llu\n", id, status.errors, static_cast<unsigned long long>(status.required_bytes));
    succeeded(status);
    for (u32 i = 0; i < report.count; ++i) if (entries[i].available) GH_CHECK(detail::finite_metric(entries[i].value));
    if (id == 0) { exact(MetricName::rmse, 2); exact(MetricName::mae, 2); for (u32 k = 0; k < 2; ++k) { exact(MetricName::rmse, 2, k); exact(MetricName::mae, 2, k); } }
    if (id == 1) { exact(MetricName::mse, 1); exact(MetricName::rmse, 1); exact(MetricName::mae, 1); exact(MetricName::r2, __dsub_rn(1.0, .2)); }
    if (id == 2) { exact(MetricName::auc, .5); exact(MetricName::average_precision, __ddiv_rn(2.0, 3.0)); exact(MetricName::f1, .8); exact(MetricName::accuracy, .75); exact(MetricName::brier, .3203125); }
    if (id == 3) { exact(MetricName::auc, __ddiv_rn(1.0, 3.0)); exact(MetricName::accuracy, __ddiv_rn(2.0, 3.0)); exact(MetricName::brier, .3828125); }
    if (id == 4) {
      exact(MetricName::micro_ap, .5); exact(MetricName::micro_auc, .5); exact(MetricName::macro_auc, .5);
      exact(MetricName::macro_ap, .625); GH_CHECK(report.ap_outputs == 4 && report.auc_outputs == 3);
      exact(MetricName::precision_at_1, 0); exact(MetricName::precision_at_3, .5); exact(MetricName::precision_at_5, .5);
      exact(MetricName::hamming_loss, .5); exact(MetricName::exact_match, 0); exact(MetricName::brier, .25);
    }
    if (id == 5) { GH_CHECK(!get(MetricName::auc).available); exact(MetricName::auc, .5, 2); exact(MetricName::brier, .25); exact(MetricName::accuracy, .5); }
    if (id == 6) { GH_CHECK(!get(MetricName::macro_ap).available && !get(MetricName::macro_auc).available && !get(MetricName::micro_auc).available); exact(MetricName::micro_ap, 0); exact(MetricName::macro_f1, 0); }
    if (id == 7 || id == 8) { exact(MetricName::accuracy, .5); exact(MetricName::logloss, -log(.5)); if (id == 7) { exact(MetricName::brier, .5); exact(MetricName::macro_f1, __ddiv_rn(1.0, 3.0)); } }
    if (id == 9 || id == 10) { exact(MetricName::accuracy, 1); const double p = .5 + 0x1p-24; exact(MetricName::logloss, -log(id == 10 ? __ddiv_rn(p, 1 + 0x1p-24) : p)); }
    if (id == 11) { GH_CHECK(!get(MetricName::r2).available); exact(MetricName::rmse, 1); }
    if (id == 23) { exact(MetricName::macro_ap, 1); exact(MetricName::micro_ap, 1); GH_CHECK(!get(MetricName::macro_auc).available); }
    if (id == 24) { exact(MetricName::precision_at_1, 1); exact(MetricName::precision_at_3, __ddiv_rn(1.0, 3.0)); exact(MetricName::precision_at_5, .2); }
    if (id >= 25 && id <= 28) { exact(id == 25 ? MetricName::micro_auc : MetricName::auc, 1); exact(id == 25 ? MetricName::micro_ap : MetricName::average_precision, 1); exact(MetricName::brier, 0); exact(MetricName::hamming_loss, 0); exact(MetricName::exact_match, 1); }
    if (id == 29 || id == 30) { const auto reference = rank_oracle(d, 0, false); exact(MetricName::auc, reference.auc); if (id == 29) exact(MetricName::average_precision, reference.ap); }
    if (id == 31) {
      const auto pooled = rank_oracle(d, 0, true); exact(MetricName::micro_ap, pooled.ap); exact(MetricName::micro_auc, pooled.auc);
      double ap = 0, auc = 0;
      for (u32 k = 0; k < 3; ++k) { const auto reference = rank_oracle(d, k, false); ap = __dadd_rn(ap, reference.ap); auc = __dadd_rn(auc, reference.auc); }
      exact(MetricName::macro_ap, __ddiv_rn(ap, 3.0)); exact(MetricName::macro_auc, __ddiv_rn(auc, 3.0));
    }
    if (id == 32) for (u32 k = 0; k < 3; ++k) exact(MetricName::auc, rank_oracle(d,k,false).auc, k);
    if (id >= 33) { exact(MetricName::r2, id == 33 ? 1 : 0); exact(MetricName::mse, id == 33 ? 0 : 1); }
  }
  if (id + 1 == cases) gate_start<<<1, 1, 0, cudaStreamTailLaunch>>>(0);
  else start<<<1, 1, 0, cudaStreamTailLaunch>>>(id + 1);
  submitted(cudaGetLastError());
}
__global__ void start(u32 id) {
  status = {}; report = {{entries, id == 22 ? 0u : 70u}};
  fixture<<<64, 256>>>(id); submitted(cudaGetLastError());
  submitted(evaluate_metrics(input_for(id), &report, {scratch, id == 21 ? 1u : sizeof(scratch)}, &status));
  check<<<1, 1, 0, cudaStreamTailLaunch>>>(id); submitted(cudaGetLastError());
}

__global__ void signal_start(u32 id);
__global__ void gate_check(u32 id) {
  GH_CHECK(status.done);
  if (id == 2) GH_CHECK(status.errors == shape);
  else if (id == 1) { GH_CHECK(status.errors == input); GH_CHECK(gate.invalid == 2); GH_CHECK(verdict[0] == MetricVerdict::invalid && verdict[1] == MetricVerdict::invalid); }
  else {
    succeeded(status); GH_CHECK(gate.checked == 4 && gate.regressions == 2 && gate.unavailable == 1 && !gate.invalid);
    GH_CHECK(verdict[0] == MetricVerdict::pass && verdict[1] == MetricVerdict::regression && verdict[2] == MetricVerdict::regression && verdict[3] == MetricVerdict::not_applicable && verdict[4] == MetricVerdict::pass);
  }
  if (id == 2) signal_start<<<1, 1, 0, cudaStreamTailLaunch>>>(0);
  else gate_start<<<1, 1, 0, cudaStreamTailLaunch>>>(id + 1);
  submitted(cudaGetLastError());
}
__global__ void gate_start(u32 id) {
  status = {}; report = {{entries,70},5,1}; candidate = {{candidate_entries,70},5,1};
  entries[0] = {MetricName::mse,aggregate_output,-0.0,1}; candidate_entries[0] = {MetricName::mse,aggregate_output,0.0,1};
  entries[1] = {MetricName::r2,aggregate_output,0x1p1023,1}; candidate_entries[1] = {MetricName::r2,aggregate_output,-0x1p1023,1};
  entries[2] = {MetricName::mae,aggregate_output,1,1}; candidate_entries[2] = {MetricName::mae,aggregate_output,nextafter(1.0,2.0),1};
  entries[3] = {MetricName::auc,aggregate_output,0,0}; candidate_entries[3] = entries[3];
  entries[4] = {MetricName::accuracy,aggregate_output,1,1}; candidate_entries[4] = entries[4];
  if (id == 1) { candidate_entries[0].available = 0; candidate_entries[1].value = INFINITY; }
  if (id == 2) candidate.ap_outputs = 1;
  submitted(compare_metrics(&report,&candidate,{verdict,70},&gate,&status));
  gate_check<<<1,1,0,cudaStreamTailLaunch>>>(id); submitted(cudaGetLastError());
}
__global__ void signal_check(u32 id) {
  GH_CHECK(status.done);
  if (id == 3) GH_CHECK(status.errors == input);
  else {
    succeeded(status);
    if (id <= 1) {
      GH_CHECK(signal.fixed.true_positive == 2 && signal.fixed.false_positive == 3);
      GH_CHECK(signal.selected.threshold == .6 && signal.selected.true_positive == 2 && signal.selected.false_positive == 2);
      GH_CHECK(signal.selected.meets_five_percent_fpr && !signal.fixed.meets_five_percent_fpr);
      GH_CHECK(signal.selected.recall == 1 && signal.selected.false_positive_rate == .05 && signal.selected.precision == .5);
    } else { GH_CHECK(signal.selected.threshold == nextafter(.9, double(INFINITY))); GH_CHECK(signal.selected.true_positive == 0 && signal.selected.false_positive == 0); }
  }
  if (id < 3) signal_start<<<1,1,0,cudaStreamTailLaunch>>>(id+1);
  else printf("GH_GPU_ACTIVITY metrics cases=35 gates=3 signals=4 positive-sum-oracles=1024 checks=pass\n");
  submitted(cudaGetLastError());
}
__global__ void signal_start(u32 id) {
  status = {};
  for (u32 i = 0; i < 42; ++i) {
    target[i] = id == 3 ? 0 : double(i < 2);
    prediction[i] = i == 0 ? .8 : i == 1 || i == 3 ? .6 : i == 2 ? .7 : i == 4 ? .55 : .1;
    if (id == 2) prediction[i] = i < 2 ? .1 : .9;
  }
  submitted(signal_metrics({{target,42},{prediction,42},42}, id == 1 ? ThresholdMode::frozen : ThresholdMode::validation, .6, &signal,{scratch,sizeof(scratch)},&status));
  signal_check<<<1,1,0,cudaStreamTailLaunch>>>(id); submitted(cudaGetLastError());
}
}
__global__ void run() {
  numerical_checks<<<32,32>>>(); submitted(cudaGetLastError());
  start<<<1,1,0,cudaStreamTailLaunch>>>(0); submitted(cudaGetLastError());
}
}
#endif

// GH_SOURCE_CATEGORY: tests
// prediction checks
#if GH_MODE == 8
namespace gh::test::prediction_suite {
__global__ void run();
namespace {
constexpr double guard = -987654.25;
constexpr u64 scratch_bytes = 1 << 20;
__device__ const unsigned char legacy_predictions[64] = {
  0,0,0,0,0,0,0,0,       0,0,0,0,0,0,0,0x80,
  0,0,0,0,0,0,0xf0,0x3f, 0,0,0,0,0,0,0,0xc0,
  0,0,0,0,0,0,0x10,0,    1,0,0,0,0,0,0,0,
  0xff,0xff,0xff,0xff,0xff,0xff,0xef,0x7f,
  0xff,0xff,0xff,0xff,0xff,0xff,0xef,0xff};
__device__ const u64 expected_bits[8] = {0,0x8000000000000000ULL,0x3ff0000000000000ULL,
  0xc000000000000000ULL,0x0010000000000000ULL,1,0x7fefffffffffffffULL,0xffefffffffffffffULL};
// Literal GHBDS001 little-endian examples. Unused row padding is excluded from
// each exact file extent; no production encoder generates these input fixtures.
__device__ const unsigned char legacy_datasets[4][80] = {
  {'G','H','B','D','S','0','0','1', 1,0,0,0, 4,0,0,0, 1,0,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0,
   0,0,0x80,0xbf, 0,0,0x80,0xbf, 0,0,0x80,0x3f, 0,0,0x80,0x3f,
   0,0,0,0xc0, 0,0,0,0xc0, 0,0,0,0x40, 0,0,0,0x40},
  {'G','H','B','D','S','0','0','1', 1,0,0,0, 4,0,0,0, 1,0,0,0, 2,0,0,0, 0,0,0,0, 0,0,0,0,
   0,0,0x80,0xbf, 0,0,0x80,0xbf, 0,0,0x80,0x3f, 0,0,0x80,0x3f,
   0,0,0,0xc0, 0,0,0x80,0x40, 0,0,0,0xc0, 0,0,0x80,0x40,
   0,0,0,0x40, 0,0,0x80,0xc0, 0,0,0,0x40, 0,0,0x80,0xc0},
  {'G','H','B','D','S','0','0','1', 1,0,0,0, 4,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0, 2,0,0,0,
   0,0,0x80,0xbf, 0,0,0x80,0xbf, 0,0,0x80,0x3f, 0,0,0x80,0x3f,
   0,0,0,0, 0,0,0,0, 0,0,0x80,0x3f, 0,0,0x80,0x3f},
  {'G','H','B','D','S','0','0','1', 1,0,0,0, 6,0,0,0, 1,0,0,0, 1,0,0,0, 2,0,0,0, 3,0,0,0,
   0,0,0x80,0xbf, 0,0,0x80,0xbf, 0,0,0x80,0xbf,
   0,0,0x80,0x3f, 0,0,0x80,0x3f, 0,0,0x80,0x3f,
   0,0,0,0, 0,0,0x80,0x3f, 0,0,0,0x40,
   0,0,0,0, 0,0,0x80,0x3f, 0,0,0,0x40}};
__device__ std::byte wire[82];
__device__ double values[10];
__device__ Status status;
__device__ u64 invalid_bits(u32 k) {
  return k == 10 ? 0x7ff0000000000000ULL : k == 11 ? 0xfff0000000000000ULL :
    k == 12 ? 0x7ff8000000000123ULL : 0x7ff0000000000001ULL;
}
__device__ u32 error_for(u32 k) {
  return k < 2 ? 0 : k == 2 ? shape : k == 3 ? extent : k < 6 ? input : k < 10 ? capacity : numeric;
}
__global__ void codec(u32 id);
__global__ void pipeline(u32 id, u32 phase);
__global__ void codec_check(u32 id) {
  const bool decode = id < 14;
  const u32 k = id % 14;
  const u32 expected_error = !decode && (k == 4 || k == 5) ? capacity : error_for(k);
  GH_CHECK(status.done && status.errors == expected_error);
  if (k != 2 && k != 3) GH_CHECK(status.required_bytes == (k == 1 ? 0 : 64));
  if (decode) {
    for (u32 i = 0; i < 10; ++i) {
      const bool written = i > 0 && i <= 8 && (k == 0 || (k >= 10 && i > 1));
      GH_CHECK(cuda::std::bit_cast<u64>(values[i]) == (written ? expected_bits[i-1] : cuda::std::bit_cast<u64>(guard)));
    }
    for (u32 i = 0; i < 82; ++i) {
      const auto expected = !i || i >= 65 ? std::byte{0xa5} : k >= 10 && i < 9
        ? std::byte(invalid_bits(k) >> (8*(i-1))) : std::byte(legacy_predictions[i-1]);
      GH_CHECK(wire[i] == expected);
    }
  } else {
    for (u32 i = 0; i < 82; ++i) {
      const bool written = i >= 1 && i < 65 && (k == 0 || (k >= 10 && i >= 9));
      GH_CHECK(wire[i] == (written ? std::byte(legacy_predictions[i-1]) : std::byte{0xa5}));
    }
    for (u32 i = 0; i < 8; ++i)
      GH_CHECK(cuda::std::bit_cast<u64>(values[i+1]) == (k >= 10 && !i ? invalid_bits(k) : expected_bits[i]));
    GH_CHECK(values[0] == guard && values[9] == guard);
  }
  if (id + 1 < 28) codec<<<1,1,0,cudaStreamTailLaunch>>>(id+1);
  else pipeline<<<1,1,0,cudaStreamTailLaunch>>>(0,0);
  submitted(cudaGetLastError());
}
__global__ void codec(u32 id) {
  const bool decode = id < 14;
  const u32 k = id % 14;
  for (auto& x : wire) x = std::byte{0xa5};
  for (auto& x : values) x = guard;
  for (u32 i = 0; i < 8; ++i) {
    if (!decode) values[i+1] = cuda::std::bit_cast<double>(expected_bits[i]);
    else for (u32 b = 0; b < 8; ++b) wire[1+8*i+b] = std::byte(legacy_predictions[8*i+b]);
  }
  if (k >= 10) {
    if (decode) for (u32 b = 0; b < 8; ++b) wire[1+b] = std::byte(invalid_bits(k) >> (8*b));
    else values[1] = cuda::std::bit_cast<double>(invalid_bits(k));
  }
  u32 rows = k == 1 ? 0 : k == 3 ? UINT32_MAX : 2;
  u32 outputs = k == 2 ? 0 : k == 3 ? UINT32_MAX : 4;
  Array<std::byte> bytes{wire+1,k == 1 ? 0u : k == 4 ? 63u : k == 5 && decode ? 65u : 64u};
  Array<double> data{values+1,k == 6 ? 7u : 8u};
  if (!decode && k == 5) data.size = 7;
  if (k == 7) data.data = reinterpret_cast<double*>(reinterpret_cast<std::byte*>(values)+1);
  if (k == 8) bytes.data = nullptr;
  if (k == 9) bytes.data = reinterpret_cast<std::byte*>(UINT64_MAX-3);
  status = {};
  if (decode) submitted(decode_predictions({bytes.data,bytes.size},rows,outputs,data,&status));
  else submitted(encode_predictions({data.data,data.size},rows,outputs,bytes,&status));
  codec_check<<<1,1,0,cudaStreamTailLaunch>>>(id);
  submitted(cudaGetLastError());
}

struct ModelStorage {
  Feature features[3]; float metadata[10]; u32 offsets[4];
  Node nodes[10]; Tree trees[5]; double base[5]; u64 output_offsets[6];
};
__device__ ModelStorage original_storage{}, restored_storage{};
__device__ Model original, restored;
__device__ Schema fitted;
__device__ DatasetRecord record;
__device__ Training training;
__device__ float x[8], y[10];
__device__ std::uint16_t bins[8];
__device__ double margins[20], losses[4], before[20], after[20], raw[20], decoded[20];
__device__ std::byte dataset_bytes[82], model_bytes[1026], prediction_bytes[146];
__device__ u64 model_size;
__device__ __align__(16) std::byte scratch[scratch_bytes+32];
__device__ Workspace workspace() { return {scratch+16,scratch_bytes}; }
__device__ Model initialize_model(ModelStorage& storage) {
  for (auto& f : storage.features) f = {777,777,FeatureType::categorical};
  for (auto& v : storage.metadata) v = float(guard);
  for (auto& v : storage.offsets) v = 777;
  for (auto& n : storage.nodes) n = {-1,-1,-1,0,0,guard};
  for (auto& t : storage.trees) t = {777,777,777};
  for (auto& v : storage.base) v = guard;
  for (auto& v : storage.output_offsets) v = 777;
  return {{{storage.features+1,1},{storage.metadata+1,8},{storage.offsets+1,2}},
    {storage.nodes+1,8},{storage.trees+1,3},{storage.base+1,3},{storage.output_offsets+1,4}};
}
__device__ void model_guards(const ModelStorage& s) {
  GH_CHECK(s.features[0].begin == 777 && s.features[2].begin == 777);
  GH_CHECK(s.metadata[0] == float(guard) && s.metadata[9] == float(guard));
  GH_CHECK(s.offsets[0] == 777 && s.offsets[3] == 777);
  GH_CHECK(s.nodes[0].value == guard && s.nodes[9].value == guard);
  GH_CHECK(s.trees[0].begin == 777 && s.trees[4].begin == 777);
  GH_CHECK(s.base[0] == guard && s.base[4] == guard);
  GH_CHECK(s.output_offsets[0] == 777 && s.output_offsets[5] == 777);
}
__device__ double analytical(u32 id, u32 row, u32 output, bool transformed) {
  if (id == 3) return transformed ? 1./3. : -1.0986122886681098;
  const double sign = row < 2 ? -1. : 1.;
  if (id == 2 && transformed) return row < 2 ? .11920292202211755 : .8807970779778823;
  return sign * (id == 1 && output ? -4. : 2.);
}
__device__ void check_prediction(u32 id, const double* prediction, bool transformed) {
  for (u32 r = 0; r < record.data.rows; ++r) for (u32 o = 0; o < original.outputs; ++o) {
    const double actual = prediction[u64(r)*original.outputs+o], expected = analytical(id,r,o,transformed);
    GH_CHECK(isfinite(actual) && fabs(actual-expected) <= 4e-15);
  }
}
__global__ void pipeline(u32 id, u32 phase) {
  if (phase) succeeded(status);
  const u64 prior_required = status.required_bytes;
  status = {};
  const u64 cells = phase ? u64(record.data.rows)*original.outputs : 0;
  switch (phase) {
    case 0:
      original = initialize_model(original_storage); restored = initialize_model(restored_storage);
      fitted = original.schema;
      for (auto& v : x) v = float(guard);
      for (auto& v : y) v = float(guard);
      for (auto& v : bins) v = 65535;
      for (auto& v : margins) v = guard;
      for (auto& v : losses) v = guard;
      for (auto& v : before) v = guard;
      for (auto& v : after) v = guard;
      for (auto& v : raw) v = guard;
      for (auto& v : decoded) v = guard;
      for (auto& v : dataset_bytes) v = std::byte{0xa5};
      for (auto& v : model_bytes) v = std::byte{0xa5};
      for (auto& v : prediction_bytes) v = std::byte{0xa5};
      for (u32 i = 0; i < 16; ++i) scratch[i] = scratch[scratch_bytes+16+i] = std::byte{0xa5};
      for (u32 i = 0; i < (id == 1 || id == 3 ? 80 : 64); ++i) dataset_bytes[i+1] = std::byte(legacy_datasets[id][i]);
      submitted(decode_dataset({dataset_bytes+1,id == 1 || id == 3 ? 80u : 64u},
        {x+1,6},{y+1,8},&record,&status));
      break;
    case 1:
      GH_CHECK(record.data.rows == (id == 3 ? 6 : 4) && record.data.columns == 1);
      GH_CHECK(record.data.outputs == (id == 1 ? 2 : 1));
      submitted(fit_schema(record.data,{},4,&fitted,{bins+1,6},workspace(),&status));
      break;
    case 2: {
      GH_CHECK(fitted.metadata_count == 1 && fitted.metadata.data[0] == -1.f);
      GH_CHECK(fitted.total_bins == 3 && fitted.features.data[0].type == FeatureType::numeric);
      for (u32 r = 0; r < record.data.rows; ++r) GH_CHECK(bins[r+1] == (r < record.data.rows/2 ? 1 : 2));
      TrainConfig config;
      config.objective = record.objective; config.classes = record.classes;
      config.rounds = id == 3 ? 0 : 1; config.max_depth = 1; config.min_leaf_rows = 1;
      config.l2 = 0; config.learning_rate = 1; config.histogram = Histogram::global;
      config.tree_build = TreeBuild::output_batch; config.output_tile = 2;
      training = {}; training.model = &original; training.margins = {margins+1,18}; training.loss = {losses+1,2};
      submitted(train(record.data,&fitted,{bins+1,6},config,&training,workspace(),&status));
      break;
    }
    case 3:
      GH_CHECK(original.outputs == (id == 3 ? 3 : id == 1 ? 2 : 1));
      check_prediction(id,margins+1,false);
      submitted(validate_model(&original,workspace(),&status));
      break;
    case 4: submitted(predict(&original,{bins+1,6},record.data.rows,{before+1,18},false,&status)); break;
    case 5:
      check_prediction(id,before+1,true);
      submitted(encode_model(&original,{model_bytes+1,1024},{&model_size,1},workspace(),&status));
      break;
    case 6:
      GH_CHECK(model_size == (id == 1 ? 248 : id == 3 ? 72 : 148));
      submitted(decode_model({model_bytes+1,model_size},&restored,workspace(),&status)); break;
    case 7: submitted(predict(&restored,{bins+1,6},record.data.rows,{raw+1,18},true,&status)); break;
    case 8:
      check_prediction(id,raw+1,false);
      for (u64 i = 0; i < cells; ++i) GH_CHECK(cuda::std::bit_cast<u64>(raw[i+1]) == cuda::std::bit_cast<u64>(margins[i+1]));
      submitted(predict(&restored,{bins+1,6},record.data.rows,{after+1,18},false,&status));
      break;
    case 9:
      check_prediction(id,after+1,true);
      for (u64 i = 0; i < cells; ++i) GH_CHECK(cuda::std::bit_cast<u64>(after[i+1]) == cuda::std::bit_cast<u64>(before[i+1]));
      submitted(encode_predictions({after+1,18},record.data.rows,restored.outputs,{prediction_bytes+1,144},&status));
      break;
    case 10:
      GH_CHECK(prior_required == cells*8);
      submitted(decode_predictions({prediction_bytes+1,cells*8},record.data.rows,restored.outputs,{decoded+1,18},&status));
      break;
    default:
      GH_CHECK(prior_required == cells*8);
      for (u64 i = 0; i < cells; ++i) GH_CHECK(cuda::std::bit_cast<u64>(decoded[i+1]) == cuda::std::bit_cast<u64>(before[i+1]));
      for (u32 i = 0; i < 146; ++i) if (!i || i > cells*8) GH_CHECK(prediction_bytes[i] == std::byte{0xa5});
      for (u32 i = 0; i < 1026; ++i) if (!i || i > model_size) GH_CHECK(model_bytes[i] == std::byte{0xa5});
      for (u32 i = 0; i < 82; ++i) GH_CHECK(dataset_bytes[i] ==
        (i && i <= (id == 1 || id == 3 ? 80 : 64) ? std::byte(legacy_datasets[id][i-1]) : std::byte{0xa5}));
      GH_CHECK(x[0] == float(guard) && x[7] == float(guard) && y[0] == float(guard) && y[9] == float(guard));
      GH_CHECK(bins[0] == 65535 && bins[7] == 65535 && losses[0] == guard && losses[3] == guard);
      for (u32 i = record.data.rows+1; i < 8; ++i) GH_CHECK(x[i] == float(guard) && bins[i] == 65535);
      for (u32 i = record.data.rows*record.data.outputs+1; i < 10; ++i) GH_CHECK(y[i] == float(guard));
      for (u32 r = 0; r < record.data.rows; ++r) {
        GH_CHECK(x[r+1] == (r < record.data.rows/2 ? -1.f : 1.f));
        for (u32 o = 0; o < record.data.outputs; ++o) {
          const float target = id == 3 ? float(r%3) : id == 2 ? float(r>=2) : float(analytical(id,r,o,false));
          GH_CHECK(y[1+r*record.data.outputs+o] == target);
        }
      }
      for (u32 i = 0; i < 20; ++i) if (!i || i > cells) {
        GH_CHECK(margins[i] == guard && before[i] == guard && after[i] == guard);
        GH_CHECK(raw[i] == guard && decoded[i] == guard);
      }
      for (u32 i = 0; i < 16; ++i) GH_CHECK(scratch[i] == std::byte{0xa5} && scratch[scratch_bytes+16+i] == std::byte{0xa5});
      model_guards(original_storage); model_guards(restored_storage);
      if (id < 3) pipeline<<<1,1,0,cudaStreamTailLaunch>>>(id+1,0);
      else printf("PASS prediction: literal binary64 codec, malformed extents, finite bits and four resident dataset-to-prediction pipelines\n");
      submitted(cudaGetLastError()); return;
  }
  pipeline<<<1,1,0,cudaStreamTailLaunch>>>(id,phase+1);
  submitted(cudaGetLastError());
}
}
__global__ void run() { codec<<<1,1>>>(0); submitted(cudaGetLastError()); }
}
#endif

// GH_SOURCE_CATEGORY: tests
// csv checks
#if GH_MODE == 9
namespace gh::test::csv_suite {
__global__ void run();
namespace {
struct Example { double value; char text[32]; };
__device__ const Example examples[]{
  {0., "0"}, {-0., "-0"}, {1., "1"}, {-1., "-1"},
  {.1, "0.10000000000000001"}, {.0001, "0.0001"}, {.00001, "1.0000000000000001e-05"},
  {1e16, "10000000000000000"}, {1e17, "1e+17"},
  {0x1p-1074, "4.9406564584124654e-324"}, {0x1p-1022, "2.2250738585072014e-308"},
  {0x1.fffffffffffffp1023, "1.7976931348623157e+308"},
  {0x1.0000000000001p0, "1.0000000000000002"},
  {9007199254740991., "9007199254740991"},
  {0x1.0000000000001p-4, "0.062500000000000014"},
  {0x1p-25, "2.9802322387695312e-08"}, {0x1.8p-24, "8.9406967163085938e-08"},
  {-0x1p-1074, "-4.9406564584124654e-324"},
  {-1e-4, "-0.0001"}, {1e15, "1000000000000000"}
};
constexpr u32 example_count = sizeof(examples) / sizeof(Example), cases = 16;
constexpr u64 scratch_bytes = 8ULL << 20, output_bytes = 2ULL << 20;
__device__ double values[65537], weights[2];
__device__ __align__(16) std::byte scratch[scratch_bytes + 32];
__device__ std::byte output[output_bytes + 2];
__device__ u64 written;
__device__ Status status;
__device__ void byte(u64& position, char c) {
  GH_CHECK(position < written);
  GH_CHECK(output[++position] == std::byte(c));
}
__device__ void literal(u64& position, const char* text) { while (*text) byte(position, *text++); }
__device__ void row_id(u64& position, u32 row) {
  u32 power = 1;
  while (power <= row / 10) power *= 10;
  do { byte(position, char('0' + row / power % 10)); power /= 10; } while (power);
}
__global__ void next(u32);
__global__ void verify(u32 c) {
  if (c < 7) {
    if (status.errors) printf("csv success case %u errors %u required %llu\n", c, status.errors, (unsigned long long)status.required_bytes);
    succeeded(status);
    u64 position = 0;
    if (c == 0) {
      literal(position, "row_id,prediction\n");
      for (u32 row = 0; row < example_count; ++row) {
        row_id(position, row); byte(position, ','); literal(position, examples[row].text); byte(position, '\n');
      }
    } else if (c == 1) literal(position, "row_id,target_0,target_1,weight\n0,0.25,0.5,0.5\n1,0.25,1,1\n");
    else if (c == 2) literal(position, "row_id,prediction_0,prediction_1\n0,0.25,0.5\n1,0.25,1\n");
    else if (c == 3) literal(position, "row_id,p0,p1,p2\n0,0.25,0.5,0.25\n");
    else if (c == 4) literal(position, "row_id,target_0,target_1,weight\n");
    else if (c == 5) {
      literal(position, "row_id,prediction\n");
      for (u32 row = 0; row < 65537; ++row) {
        row_id(position, row); byte(position, ','); byte(position, char('0' + row % 10)); byte(position, '\n');
      }
    } else literal(position, "row_id,prediction\n0,0.25\n");
    GH_CHECK(position == written && status.required_bytes == written);
    GH_CHECK(output[written + 1] == std::byte{0x6d});
  } else {
    GH_CHECK(status.done && status.errors);
    GH_CHECK(written == UINT64_MAX || c == 11);
  }
  GH_CHECK(output[0] == std::byte{0x6d} && output[output_bytes + 1] == std::byte{0x6d});
  for (u32 i = 0; i < 16; ++i) GH_CHECK(scratch[i] == std::byte{0x5a} && scratch[scratch_bytes + 16 + i] == std::byte{0x5a});
  if (c + 1 < cases) { next<<<1, 1, 0, cudaStreamTailLaunch>>>(c + 1); submitted(cudaGetLastError()); }
  else printf("PASS csv: exact precision-17 literals, ties, subnormals, headers, hierarchical compaction and rejections\n");
}
__global__ void next(u32 c) {
  u32 rows = 1, outputs = 1; CsvKind kind = CsvKind::predictions;
  for (u64 i = 0; i < output_bytes + 2; ++i) output[i] = std::byte{0x6d};
  for (u32 i = 0; i < 16; ++i) scratch[i] = scratch[scratch_bytes + 16 + i] = std::byte{0x5a};
  values[0] = .25; values[1] = .5; values[2] = .25; values[3] = 1;
  weights[0] = .5; weights[1] = 1;
  Array<const double> weight;
  if (c == 0) { rows = example_count; for (u32 i = 0; i < rows; ++i) values[i] = examples[i].value; }
  if (c == 1 || c == 2) { rows = outputs = 2; if (c == 1) { kind = CsvKind::targets; weight = {weights, 2}; } }
  if (c == 3) { outputs = 3; kind = CsvKind::multiclass; }
  if (c == 4) { rows = 0; outputs = 2; kind = CsvKind::targets; }
  if (c == 5) { rows = 65537; for (u32 i = 0; i < rows; ++i) values[i] = i % 10; }
  Array<const double> input{values, u64(rows) * outputs};
  Array<std::byte> bytes{output + 1, output_bytes};
  Workspace work{scratch + 16, scratch_bytes};
  if (c == 6) bytes.size = 25;
  if (c == 7) { kind = CsvKind::targets; weights[0] = -1; weight = {weights, 1}; }
  if (c == 8) values[0] = CUDART_NAN;
  if (c == 9) weight = {weights, 1};
  if (c == 10) work.bytes = 1;
  if (c == 11) bytes.size = 24;
  if (c == 12) bytes = {reinterpret_cast<std::byte*>(UINT64_MAX - 7), 32};
  if (c == 13) { rows = outputs = UINT32_MAX; input.size = UINT64_MAX; }
  if (c == 14) outputs = 0;
  if (c == 15) { rows = 0; outputs = UINT32_MAX; input = {}; }
  status = {}; written = UINT64_MAX;
  submitted(encode_csv(input, rows, outputs, kind, weight, bytes, {&written, 1}, work, &status));
  verify<<<1, 1, 0, cudaStreamTailLaunch>>>(c); submitted(cudaGetLastError());
}
}
__global__ void run() { next<<<1, 1>>>(0); submitted(cudaGetLastError()); }
}
#endif

// GH_SOURCE_CATEGORY: tests
// reference checks
#if GH_MODE == 10
namespace gh::test::reference_suite {
__global__ void run();
namespace {
struct Number { const char* text; u64 bits; bool accepted{true}; };
__device__ const Number numbers[]{
  {"0", 0}, {"-0", 0x8000000000000000ULL}, {"-0.0e+308", 0x8000000000000000ULL},
  {"0.1", 0x3fb999999999999aULL}, {"0.10000000000000001", 0x3fb999999999999aULL},
  {"0.10000000000000002", 0x3fb999999999999bULL}, {"1.0000000000000001", 0x3ff0000000000000ULL},
  {"1.0000000000000002", 0x3ff0000000000001ULL}, {"5e-324", 1}, {"-5e-324", 0x8000000000000001ULL},
  {"4.9406564584124654e-324", 1}, {"2.4703282292062327e-324", 0}, {"2.4703282292062328e-324", 1},
  {"-2.4703282292062327e-324", 0x8000000000000000ULL}, {"1e-342", 0},
  {"2.2250738585072011e-308", 0x000fffffffffffffULL}, {"2.2250738585072014e-308", 0x0010000000000000ULL},
  {"1.7976931348623157e308", 0x7fefffffffffffffULL}, {"1.7976931348623158e308", 0x7fefffffffffffffULL},
  {"1.7976931348623159e308", 0x7ff0000000000000ULL}, {"-1.7976931348623159e308", 0xfff0000000000000ULL},
  {"9007199254740993", 0x4340000000000000ULL}, {"9007199254740995", 0x4340000000000002ULL},
  {"9999999999999999999", 0x43e158e460913d00ULL},
  {"-9007199254740993", 0xc340000000000000ULL}, {"1.25E+2", 0x405f400000000000ULL},
  {"1e309", 0, false}, {"1e-343", 0, false}, {"12345678901234567890", 0, false},
  {"01", 0, false}, {"+1", 0, false}, {".1", 0, false}, {"1.", 0, false},
  {"1e", 0, false}, {"1e+", 0, false}, {"NaN", 0, false}, {"Infinity", 0, false},
  {" 1", 0, false}, {"1 ", 0, false}, {"--1", 0, false}, {"1e9999999999", 0, false}, {"", 0, false}
};
__device__ u64 length(const char* text) { u64 n = 0; while (text[n]) ++n; return n; }
__global__ void number_checks() {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < sizeof(numbers) / sizeof(numbers[0])) {
    const auto value = numbers[i]; double out = 17;
    GH_CHECK(parse_decimal({value.text, length(value.text)}, &out) == value.accepted);
    GH_CHECK(u64(__double_as_longlong(out)) == (value.accepted ? value.bits : 0x4031000000000000ULL));
  }
  // Integer lattices have known spacing: two at 2^53 and 2048 at 2^63.
  // The latter also exercises the supported 19-digit mantissa boundary.
  if (i < 2048) {
    for (u32 scale = 0; scale < 2; ++scale) {
      char reversed[32], text[32]; u32 n = 0; u64 value = (1ULL << (scale ? 63 : 53)) + i;
      do { reversed[n++] = char('0' + value % 10); value /= 10; } while (value);
      for (u32 j = 0; j < n; ++j) text[j] = reversed[n - j - 1];
      const u64 expected = scale ? 0x43e0000000000000ULL + u64(i > 1024) :
        0x4340000000000000ULL + i / 2 + ((i & 1) && ((i / 2) & 1));
      double out{}; GH_CHECK(parse_decimal({text, n}, &out));
      GH_CHECK(u64(__double_as_longlong(out)) == expected);
    }
  }
}
__device__ char json[8192];
__device__ u32 json_size;
__device__ Metric computed_values[16], reference_values[16], aligned_values[16];
__device__ MetricReport computed, reference, aligned;
__device__ MetricVerdict verdicts[16];
__device__ MetricGate gate;
__device__ Status status;
__device__ const char* regression = "\"mse\":0.5,\"rmse\":0.5,\"mae\":0.5,\"r2\":0.5,\"selection_metric\":\"mse\",\"selection_value\":0.5";
__device__ const char* binary = "\"log_loss\":0.5,\"brier\":0.5,\"hamming_loss\":0.5,\"exact_match_accuracy\":0.5,\"accuracy\":0.5,\"f1\":0.5,\"roc_auc\":0.5,\"average_precision\":0.5,\"selection_metric\":\"log_loss\",\"selection_value\":0.5,\"log_clip_epsilon\":1e-15";
__device__ const char* multiclass = "\"log_loss\":0.5,\"brier\":0.5,\"accuracy\":0.5,\"macro_f1\":0.5,\"selection_metric\":\"log_loss\",\"selection_value\":0.5,\"log_clip_epsilon\":1e-15";
__device__ const char* multilabel = "\"log_loss\":0.5,\"brier\":0.5,\"hamming_loss\":0.5,\"exact_match_accuracy\":0.5,\"micro_f1\":0.5,\"macro_f1\":0.5,\"micro_ap\":0.5,\"macro_ap\":0.5,\"macro_auc\":0.5,\"precision_at_1\":0.5,\"precision_at_3\":0.5,\"precision_at_5\":0.5,\"macro_ap_labels\":4,\"macro_auc_labels\":3,\"selection_metric\":\"log_loss\",\"selection_value\":0.5,\"log_clip_epsilon\":1e-15";
__device__ void append(const char* text) { while (*text) { GH_CHECK(json_size < sizeof(json)); json[json_size++] = *text++; } }
__device__ void replace(const char* old, const char* value) {
  const u32 n = u32(length(old)), m = u32(length(value));
  for (u32 i = 0; i + n <= json_size; ++i) {
    u32 j = 0; while (j < n && json[i + j] == old[j]) ++j;
    if (j != n) continue;
    GH_CHECK(json_size + m - n < sizeof(json));
    if (m > n) for (u32 k = json_size; k > i + n; --k) json[k + m - n - 1] = json[k - 1];
    else for (u32 k = i + n; k < json_size; ++k) json[k + m - n] = json[k];
    for (u32 k = 0; k < m; ++k) json[i + k] = value[k];
    json_size += m - n; return;
  }
  GH_CHECK(false);
}
__device__ void add(MetricName name, bool available = true) {
  computed_values[computed.count++] = {name, aggregate_output, available ? .5 : 0.0, u32(available)};
}
__global__ void setup(u32 id);
__device__ void next(u32 id) {
  if (id == 27) printf("reference checks passed: 42 decimal landmarks, 4096 integer rounding cases, 27 JSON cases\n");
  else { setup<<<1, 1, 0, cudaStreamTailLaunch>>>(id); submitted(cudaGetLastError()); }
}
__global__ void checked_gate(u32 id) {
  GH_CHECK(status.done == 1);
  GH_CHECK(status.errors == (id == 25 ? u32(shape) : u32(input)));
  next(id + 1);
}
__global__ void checked(u32 id) {
  GH_CHECK(status.done == 1);
  if (id >= 8 && id <= 24) GH_CHECK(status.errors == (id == 15 ? u32(capacity) : u32(input)));
  else {
    succeeded(status);
    GH_CHECK(reference.count == (computed.task == MetricTask::multilabel ? 12 : computed.count));
    GH_CHECK(reference.count == aligned.count && reference.outputs == computed.outputs);
    for (u32 i = 0; i < reference.count; ++i) {
      const auto a = reference_values[i], b = aligned_values[i];
      GH_CHECK(a.name == b.name && a.output == aggregate_output && a.name != MetricName::micro_auc);
      GH_CHECK(a.available == b.available || id == 26);
      GH_CHECK(u64(__double_as_longlong(a.value)) == (a.available ? 0x3fe0000000000000ULL : 0));
    }
    if (id == 3) GH_CHECK(reference.ap_outputs == 4 && reference.auc_outputs == 3);
    if (id == 5) GH_CHECK(reference.ap_outputs == 0 && reference.auc_outputs == 0);
    if (id >= 25) {
      status = {}; submitted(compare_metrics(&reference, &aligned, {verdicts, 16}, &gate, &status));
      checked_gate<<<1, 1, 0, cudaStreamTailLaunch>>>(id); submitted(cudaGetLastError()); return;
    }
  }
  next(id + 1);
}
__global__ void setup(u32 id) {
  status = {}; json_size = 0; computed = {}; reference = {}; aligned = {};
  computed.metrics = {computed_values, 16}; reference.metrics = {reference_values, id == 15 ? 0u : 16u}; aligned.metrics = {aligned_values, 16};
  const bool multi = id == 3 || id == 5 || id == 13 || id == 25;
  const bool bin = id == 1 || id == 7 || id == 12 || id == 26;
  computed.task = multi ? MetricTask::multilabel : bin ? MetricTask::binary : id == 2 ? MetricTask::multiclass : MetricTask::regression;
  computed.outputs = multi ? 5 : id == 2 ? 3 : 1; computed.profile = MetricProfile::real_data;
  const char* body = multi ? multilabel : bin ? binary : id == 2 ? multiclass : regression;
  if (computed.task == MetricTask::regression) {
    add(MetricName::mse); add(MetricName::rmse); add(MetricName::mae); add(MetricName::r2, id != 4);
  } else {
    add(MetricName::logloss); add(MetricName::brier);
    if (id == 2) { add(MetricName::accuracy); add(MetricName::macro_f1); }
    else {
      add(MetricName::hamming_loss); add(MetricName::exact_match);
      if (bin) { add(MetricName::accuracy); add(MetricName::f1); add(MetricName::auc, id != 7 && id != 26); add(MetricName::average_precision); }
      else {
        add(MetricName::micro_f1); add(MetricName::macro_f1); add(MetricName::micro_ap); add(MetricName::micro_auc);
        add(MetricName::macro_ap, id != 5); add(MetricName::macro_auc, id != 5);
        add(MetricName::precision_at_1); add(MetricName::precision_at_3); add(MetricName::precision_at_5);
        computed.ap_outputs = id == 5 ? 0 : 4; computed.auc_outputs = id == 5 ? 0 : id == 25 ? 2 : 3;
      }
    }
  }
  append("{\"reference\":\"CPU float64 common metric implementation\",\"fixture_sha256\":\"");
  for (u32 i = 0; i < 64; ++i) append("0");
  append("\",\"training_fixture_sha256\":\""); for (u32 i = 0; i < 64; ++i) append("1");
  append("\",\"predictions_sha256\":\""); for (u32 i = 0; i < 64; ++i) append("a");
  append("\",\"metrics\":{"); append(body);
  if (id == 8) append(",\"mse\":0.5");
  if (id == 9) append(",\"bogus\":0.5");
  if (id == 24) append(",");
  append("},\"training_mean_baseline\":{"); append(body); append("}");
  if (id == 17) { append(",\"metrics\":{"); append(body); append("}"); }
  append("}");
  if (id == 4) replace("\"r2\":0.5", "\"r2\":null");
  if (id == 5) {
    replace("\"macro_ap\":0.5", "\"macro_ap\":null"); replace("\"macro_auc\":0.5", "\"macro_auc\":null");
    replace("\"macro_ap_labels\":4", "\"macro_ap_labels\":0"); replace("\"macro_auc_labels\":3", "\"macro_auc_labels\":0");
  }
  if (id == 6) replace("\"mse\":", "\"\\u006dse\":");
  if (id == 7) replace("\"roc_auc\":0.5", "\"roc_auc\":null");
  if (id == 10) replace("\"mae\":0.5,", "");
  if (id == 11) replace("\"selection_value\":0.5", "\"selection_value\":0.25");
  if (id == 12) replace("1e-15", "1e-14");
  if (id == 13) replace("\"macro_ap_labels\":4", "\"macro_ap_labels\":6");
  if (id == 14) --computed.count;
  if (id == 16) append(" true");
  if (id == 18) replace("0000", "z000");
  if (id == 19) replace("\"mse\":", "\"mse\\u0000\":");
  if (id == 20) replace("\"selection_value\":0.5", "\"selection_value\":null");
  if (id == 21) replace("\"mse\":0.5", "\"mse\":1.8e308");
  if (id == 22) replace("CPU float64", "GPU float64");
  if (id == 23) replace("\"mse\":0.5", "\"mse\":01");
  submitted(decode_metric_reference({reinterpret_cast<const std::byte*>(json), json_size}, &computed, &reference, &aligned, &status));
  checked<<<1, 1, 0, cudaStreamTailLaunch>>>(id); submitted(cudaGetLastError());
}
}
__global__ void run() {
  GH_CHECK(!parse_decimal({}, nullptr));
  GH_CHECK(decode_metric_reference({}, nullptr, nullptr, nullptr, nullptr) == cudaErrorInvalidValue);
  number_checks<<<8, 256>>>(); submitted(cudaGetLastError());
  next(0);
}
}
#endif

// GH_SOURCE_CATEGORY: tooling
// Frozen benchmark reference
#if GH_MODE == 11
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
#endif

// GH_SOURCE_CATEGORY: tests
// Complete-operation benchmark
#if GH_MODE == 11
namespace gh::test::benchmark_suite {
__global__ void run();
namespace {
constexpr u32 cases = 8, count_bins = 16384, rows = 8192, columns = 8, outputs = 32, rounds = 3;
constexpr u64 max_count = 64ULL << 20, cells = u64(rows)*outputs, feature_cells = u64(rows)*columns;
constexpr u64 scratch_bytes = 16ULL << 20, wire_capacity = 64ULL << 10, node_capacity = 1440;
constexpr u64 count_guard = 0xfedcba9876543210ULL;
constexpr double guard = -987654.25;
__device__ __align__(16) u32 ids[max_count+8];
__device__ u64 counts[count_bins+2];
__device__ float values[feature_cells+2], targets[cells+2];
__device__ std::uint16_t bins[feature_cells+2];
__device__ Feature features[columns+2];
__device__ float metadata[columns*3+2];
__device__ u32 offsets[columns+3];
__device__ Node nodes[node_capacity+2];
__device__ Tree trees[outputs*rounds+2];
__device__ double base[outputs+2], margins[cells+2], prediction[cells+2], losses[rounds+3];
__device__ u64 output_offsets[outputs+3], wire_size;
__device__ std::byte wire[wire_capacity+2];
__device__ __align__(16) std::byte scratch[scratch_bytes+32];
__device__ Schema schema;
__device__ Model forest;
__device__ Training training;
__device__ Status operation[5], timing_status[cases];
__device__ observe::Sample raw[cases][observe::samples];
__device__ observe::Summary summaries[cases];
__device__ u64 fit_bytes, validation_bytes, export_bytes;

__device__ bool pipeline_case(u32 id) { return id == 4 || id == 5; }
__device__ u64 count_size(u32 id) { return id == 3 || id == 7 ? max_count : 16ULL << 20; }
__device__ float feature_value(u32 row, u32 feature) { return (row >> feature) & 1 ? 1.f : -1.f; }
__device__ float target_value(u32 row, u32 output) { return feature_value(row,output%columns)*float(1+output%3); }
__device__ Workspace workspace() { return {scratch+16,scratch_bytes}; }
__device__ Dataset dataset() { return {{values+1,feature_cells},{targets+1,cells},{},rows,columns,outputs}; }
__device__ count::Config count_config(u32 id, u32 variant) {
  count::Config result;
  result.algorithm = count::Algorithm::shared_atomic;
  result.input_type = count::InputType::u32; result.counter_type = count::CounterType::u64;
  result.local_counter = count::LocalCounter::u32;
  result.size = count_size(id); result.bins = count_bins;
  const bool alternative = (id == 2 || id == 3) && variant;
  result.policy = alternative ? 14 : 15; result.blocks = alternative ? 192 : 48;
  if (id >= 6) result.output_clear = count::OutputClear::kernel;
  return result;
}
__device__ TrainConfig train_config(u32 id, u32 variant) {
  TrainConfig result;
  result.rounds = rounds; result.max_depth = 3; result.min_leaf_rows = 1;
  result.learning_rate = 1; result.l2 = 0; result.histogram = Histogram::global;
  const bool batched = id == 5 && variant;
  result.tree_build = batched ? TreeBuild::output_batch : TreeBuild::per_output;
  result.root_counts = batched ? RootCounts::global : RootCounts::per_output;
  result.batched_roots = batched; result.batched_root_splits = batched;
  return result;
}
__global__ void fixtures() {
  const u64 first = u64(blockIdx.x)*blockDim.x+threadIdx.x, stride = u64(gridDim.x)*blockDim.x;
  for (u64 i = first; i < max_count+8; i += stride)
    ids[i] = i >= 4 && i < max_count+4 ? u32((13*(i-4)+7)%count_bins) : 0xfedcba98u;
  for (u64 i = first; i < feature_cells+2; i += stride)
    values[i] = i && i <= feature_cells ? feature_value(u32((i-1)/columns),u32((i-1)%columns)) : float(guard);
  for (u64 i = first; i < cells+2; i += stride)
    targets[i] = i && i <= cells ? target_value(u32((i-1)/outputs),u32((i-1)%outputs)) : float(guard);
}
__global__ void validate_inputs() {
  const u64 first = u64(blockIdx.x)*blockDim.x+threadIdx.x, stride = u64(gridDim.x)*blockDim.x;
  for (u64 i = first; i < max_count; i += stride) GH_CHECK(ids[i+4] < count_bins);
  for (u64 i = first; i < feature_cells; i += stride)
    GH_CHECK(values[i+1] == feature_value(u32(i/columns),u32(i%columns)));
  for (u64 i = first; i < cells; i += stride)
    GH_CHECK(targets[i+1] == target_value(u32(i/outputs),u32(i%outputs)));
}
// Poison reusable destinations before the start marker, preserving resident inputs.
__global__ void reset(u32 id) {
  const u64 first = u64(blockIdx.x)*blockDim.x+threadIdx.x, stride = u64(gridDim.x)*blockDim.x;
  if (!first) {
    for (auto& s : operation) s = {};
    schema = {{features+1,columns},{metadata+1,columns*3},{offsets+1,columns+1}};
    forest = {schema,{nodes+1,node_capacity},{trees+1,outputs*rounds},
      {base+1,outputs},{output_offsets+1,outputs+1}};
    training = {}; training.model = &forest; training.margins = {margins+1,cells};
    training.loss = {losses+1,rounds+1}; wire_size = 0;
  }
  for (u64 i = first; i < 16; i += stride) scratch[i] = scratch[scratch_bytes+16+i] = std::byte{0xa5};
  if (!pipeline_case(id)) {
    for (u64 i = first; i < count_bins+2; i += stride) counts[i] = count_guard;
    return;
  }
  for (u64 i = first; i < cells+2; i += stride) margins[i] = prediction[i] = guard;
  for (u64 i = first; i < feature_cells+2; i += stride) bins[i] = 65535;
  for (u64 i = first; i < columns+2; i += stride) features[i] = {777,777,FeatureType::categorical};
  for (u64 i = first; i < columns*3+2; i += stride) metadata[i] = float(guard);
  for (u64 i = first; i < columns+3; i += stride) offsets[i] = 777;
  for (u64 i = first; i < node_capacity+2; i += stride) nodes[i] = {-1,-1,-1,0,0,guard};
  for (u64 i = first; i < outputs*rounds+2; i += stride) trees[i] = {777,777,777};
  for (u64 i = first; i < outputs+2; i += stride) base[i] = guard;
  for (u64 i = first; i < outputs+3; i += stride) output_offsets[i] = 777;
  for (u64 i = first; i < rounds+3; i += stride) losses[i] = guard;
  for (u64 i = first; i < wire_capacity+2; i += stride) wire[i] = std::byte{0xa5};
}

__global__ void pipeline(u32 id, u32 variant, u32 stage) {
  if (stage) succeeded(operation[stage-1]);
  switch (stage) {
    case 0: submitted(fit_schema(dataset(),{},4,&schema,{bins+1,feature_cells},workspace(),operation)); break;
    case 1:
      fit_bytes = operation[0].required_bytes;
      submitted(train(dataset(),&schema,{bins+1,feature_cells},train_config(id,variant),&training,workspace(),operation+1)); break;
    case 2: submitted(validate_model(&forest,workspace(),operation+2)); break;
    case 3:
      validation_bytes = operation[2].required_bytes;
      submitted(predict(&forest,{bins+1,feature_cells},rows,{prediction+1,cells},false,operation+3)); break;
    default:
      submitted(encode_model(&forest,{wire+1,wire_capacity},{&wire_size,1},workspace(),operation+4)); return;
  }
  pipeline<<<1,1,0,cudaStreamTailLaunch>>>(id,variant,stage+1);
  submitted(cudaGetLastError());
}
// The entire API call executes behind the start marker, including GPU planning.
__global__ void execute(u32 id, u32 variant) {
  if (!id) submitted(finish(operation));
  else if (!pipeline_case(id)) {
    const auto config = count_config(id,variant);
    const Array<const std::byte> input{reinterpret_cast<const std::byte*>(ids+4),config.size*sizeof(u32)};
    const Array<std::byte> output{reinterpret_cast<std::byte*>(counts+1),count_bins*sizeof(u64)};
    submitted(id >= 6 && !variant
      ? bench::frozen_count::count(config.size,input,output,workspace(),operation)
      : count::count(config,input,output,workspace(),operation));
  } else {
    pipeline<<<1,1>>>(id,variant,0); submitted(cudaGetLastError());
  }
}

// Independent byte reader: no production codec/parser is the export oracle.
__device__ u64 read_wire(u64 at, u32 width) {
  GH_CHECK(at <= wire_size && width <= wire_size-at);
  u64 result = 0;
  for (u32 b = 0; b < width; ++b) result |= u64(wire[1+at+b]) << (8*b);
  return result;
}
__device__ void check_export() {
  GH_CHECK(wire_size == 5664 && read_wire(0,8) == 0x4c45444f4d424847ULL);
  GH_CHECK(read_wire(8,4) == 1 && read_wire(12,4) == 0);
  GH_CHECK(read_wire(16,4) == outputs && read_wire(20,4) == columns && read_wire(24,8) == outputs*rounds);
  for (u32 o = 0; o < outputs; ++o) GH_CHECK(read_wire(32+8*o,8) == 0);
  u64 position = 32+8*outputs;
  for (u32 f = 0; f < columns; ++f, position += 16) {
    GH_CHECK(read_wire(position,4) == 0 && read_wire(position+4,4) == 1 && read_wire(position+8,4) == 0);
    GH_CHECK(read_wire(position+12,4) == 0xbf800000u);
  }
  for (u32 t = 0; t < outputs*rounds; ++t) {
    const auto tree = forest.trees.data[t];
    GH_CHECK(read_wire(position,4) == tree.output && read_wire(position+4,4) == tree.count);
    position += 8;
    for (u32 j = 0; j < tree.count; ++j, position += 28) {
      const auto n = forest.nodes.data[tree.begin+j];
      GH_CHECK(read_wire(position,4) == cuda::std::bit_cast<u32>(n.feature));
      GH_CHECK(read_wire(position+4,4) == cuda::std::bit_cast<u32>(n.left));
      GH_CHECK(read_wire(position+8,4) == cuda::std::bit_cast<u32>(n.right));
      GH_CHECK(read_wire(position+12,4) == n.threshold && read_wire(position+16,4) == n.missing_left);
      GH_CHECK(read_wire(position+20,8) == cuda::std::bit_cast<u64>(n.value));
    }
  }
  GH_CHECK(position == wire_size);
}
__global__ void check_payload(u32 id) {
  const u64 first = u64(blockIdx.x)*blockDim.x+threadIdx.x, stride = u64(gridDim.x)*blockDim.x;
  if (!first) for (u32 i = 0; i < 16; ++i)
    GH_CHECK(scratch[i] == std::byte{0xa5} && scratch[scratch_bytes+16+i] == std::byte{0xa5});
  if (!id) return;
  if (!pipeline_case(id)) {
    for (u64 i = first; i < count_size(id); i += stride) GH_CHECK(ids[i+4] == (13*i+7)%count_bins);
    for (u64 i = first; i < count_bins; i += stride) GH_CHECK(counts[i+1] == count_size(id)/count_bins);
    if (!first) {
      GH_CHECK(counts[0] == count_guard && counts[count_bins+1] == count_guard);
      for (u32 i = 0; i < 4; ++i) GH_CHECK(ids[i] == 0xfedcba98u && ids[max_count+4+i] == 0xfedcba98u);
    }
    return;
  }
  for (u64 i = first; i < cells; i += stride) {
    const double expected = target_value(u32(i/outputs),u32(i%outputs));
    GH_CHECK(targets[i+1] == expected && margins[i+1] == expected && prediction[i+1] == expected);
  }
  for (u64 i = first; i < feature_cells; i += stride) {
    GH_CHECK(values[i+1] == feature_value(u32(i/columns),u32(i%columns)));
    GH_CHECK(bins[i+1] == (((i%rows) >> (i/rows)) & 1 ? 2 : 1));
  }
  for (u64 i = first; i < wire_capacity+2; i += stride)
    if (!i || i > wire_size) GH_CHECK(wire[i] == std::byte{0xa5});
  if (first) return;
  GH_CHECK(schema.columns == columns && schema.metadata_count == columns && schema.total_bins == columns*3 && schema.max_feature_bins == 3);
  for (u32 f = 0; f < columns; ++f) {
    GH_CHECK(features[f+1].type == FeatureType::numeric && features[f+1].begin == f && features[f+1].count == 1);
    GH_CHECK(metadata[f+1] == -1 && offsets[f+1] == f*3);
  }
  GH_CHECK(offsets[columns+1] == columns*3);
  GH_CHECK(forest.outputs == outputs && forest.tree_count == outputs*rounds && forest.node_count == outputs*5);
  for (u32 o = 0; o < outputs; ++o) {
    GH_CHECK(base[o+1] == 0 && output_offsets[o+1] == o*rounds);
    for (u32 r = 0; r < rounds; ++r) {
      const auto t = forest.trees.data[o*rounds+r];
      GH_CHECK(t.output == o && t.count == (r ? 1 : 3));
    }
  }
  GH_CHECK(output_offsets[outputs+1] == outputs*rounds);
  GH_CHECK(losses[1] == 145./64.);
  for (u32 r = 1; r <= rounds; ++r) GH_CHECK(losses[r+1] == 0);
  GH_CHECK(values[0] == guard && values[feature_cells+1] == guard && targets[0] == guard && targets[cells+1] == guard);
  GH_CHECK(margins[0] == guard && margins[cells+1] == guard && prediction[0] == guard && prediction[cells+1] == guard);
  GH_CHECK(bins[0] == 65535 && bins[feature_cells+1] == 65535);
  GH_CHECK(features[0].begin == 777 && features[columns+1].begin == 777);
  GH_CHECK(metadata[0] == guard && metadata[columns*3+1] == guard);
  GH_CHECK(offsets[0] == 777 && offsets[columns+2] == 777);
  GH_CHECK(nodes[0].value == guard && nodes[node_capacity+1].value == guard);
  GH_CHECK(trees[0].begin == 777 && trees[outputs*rounds+1].begin == 777);
  GH_CHECK(base[0] == guard && base[outputs+1] == guard);
  GH_CHECK(output_offsets[0] == 777 && output_offsets[outputs+2] == 777);
  GH_CHECK(losses[0] == guard && losses[rounds+2] == guard);
  check_export();
}

__global__ void sample(u32 id, u32 ordinal);
__global__ void summary_check(u32 id) {
  succeeded(timing_status[id]);
  const auto s = summaries[id];
  GH_CHECK(isfinite(s.ratio) && s.lower > 0 && s.lower <= s.ratio && s.upper >= s.ratio);
  printf("GH_BENCH_SUMMARY case=%u protocol=globaltimer-cdp-tail-v1 pairs=15 ratio_A_over_B=%.17g lower95=%.17g upper95=%.17g log_sd=%.17g minimum_ticks=%llu\n",
    id,s.ratio,s.lower,s.upper,s.log_stddev,static_cast<unsigned long long>(s.minimum_ticks));
  if (id+1 < cases) sample<<<1,1,0,cudaStreamTailLaunch>>>(id+1,0);
  else printf("GH_GPU_ACTIVITY benchmark cases=8 raw_samples=288 checks=pass instrumentation=%u\n",unsigned(observe::enabled));
  submitted(cudaGetLastError());
}
__global__ void after_checks(u32 id, u32 ordinal) {
  const auto slot = observe::schedule(ordinal);
  printf("GH_BENCH_CHECK case=%u ordinal=%u checked=1\n",id,ordinal);
  if (pipeline_case(id) && ordinal < 2) {
    export_bytes = operation[4].required_bytes;
    printf("GH_BENCH_MEMORY case=%u variant=%u arena=%llu fit=%llu train=%llu hist=%llu derivatives=%llu tree_state=%llu validation=%llu export=%llu wire=%llu\n",
      id,slot.variant,static_cast<unsigned long long>(scratch_bytes),static_cast<unsigned long long>(fit_bytes),
      static_cast<unsigned long long>(training.workspace_bytes),static_cast<unsigned long long>(training.histogram_bytes),
      static_cast<unsigned long long>(training.derivative_bytes),static_cast<unsigned long long>(training.tree_state_bytes),
      static_cast<unsigned long long>(validation_bytes),static_cast<unsigned long long>(export_bytes),static_cast<unsigned long long>(wire_size));
  }
  if (ordinal+1 < observe::samples) sample<<<1,1,0,cudaStreamTailLaunch>>>(id,ordinal+1);
  else {
    submitted(observe::summarize({raw[id],observe::samples},summaries+id,timing_status+id));
    summary_check<<<1,1,0,cudaStreamTailLaunch>>>(id);
  }
  submitted(cudaGetLastError());
}
__global__ void checked(u32 id, u32 ordinal) {
  const auto slot = observe::schedule(ordinal);
  const auto sample_value = raw[id][ordinal];
  // Preserve this raw observation even when the following correctness gate fails.
  printf("GH_BENCH_RAW case=%u ordinal=%u pair=%u variant=%u warmup=%u begin=%llu end=%llu\n",
    id,ordinal,slot.pair,slot.variant,unsigned(slot.warmup),static_cast<unsigned long long>(sample_value.begin),
    static_cast<unsigned long long>(sample_value.end));
  for (u32 stage = 0; stage < (pipeline_case(id) ? 5u : 1u); ++stage) succeeded(operation[stage]);
  GH_CHECK(timing_status[id].errors == 0 && raw[id][ordinal].end > raw[id][ordinal].begin);
  if (ordinal) GH_CHECK(raw[id][ordinal].begin >= raw[id][ordinal-1].end);
  check_payload<<<512,256>>>(id);
  after_checks<<<1,1,0,cudaStreamTailLaunch>>>(id,ordinal);
  submitted(cudaGetLastError());
}
__global__ void sample(u32 id, u32 ordinal) {
  if (!ordinal) {
    printf("GH_BENCH_CASE case=%u scope=%s instrumentation=%u warmups_per_variant=3 measured_pairs=15\n",
      id,id == 0 ? "empty-CDP-completion" : !pipeline_case(id) ? "clear-count-completion-CDP-stream" : "fit-train-validate-predict-modelbytes-CDP-stream",unsigned(observe::enabled));
    for (u32 v = 0; v < 2; ++v) {
      if (id && !pipeline_case(id)) {
        const auto c = count_config(id,v);
        GH_CHECK(count::supported(c) && count::required_bytes(c) == 0);
        printf("GH_BENCH_CONFIG case=%u variant=%u backend=%s family=shared_atomic input=u32 output=u64 local=u32 elements=%llu bins=%u policy=%u blocks=%u clear=%s scratch=0 cache=resident-no-flush selection=explicit\n",
          id,v,id >= 6 && !v ? "frozen" : "fresh",static_cast<unsigned long long>(c.size),c.bins,c.policy,c.blocks,
          c.output_clear == count::OutputClear::kernel ? "kernel" : "runtime");
      } else if (pipeline_case(id)) {
        const auto c = train_config(id,v);
        printf("GH_BENCH_CONFIG case=%u variant=%u rows=8192 features=8 outputs=32 rounds=3 depth=3 histogram=global split=warp32 tree_build=%s batched_roots=%u root_counts=%s\n",
          id,v,c.tree_build == TreeBuild::output_batch ? "output_batch" : "per_output",unsigned(c.batched_roots),c.root_counts == RootCounts::global ? "global" : "per_output");
      }
    }
  }
  reset<<<128,256>>>(id);
  submitted(observe::boundary({raw[id],observe::samples},ordinal,false,timing_status+id));
  execute<<<1,1>>>(id,observe::schedule(ordinal).variant);
  submitted(cudaGetLastError());
  submitted(observe::boundary({raw[id],observe::samples},ordinal,true,timing_status+id));
  checked<<<1,1,0,cudaStreamTailLaunch>>>(id,ordinal);
  submitted(cudaGetLastError());
}
__global__ void inputs_ready() {
  validate_inputs<<<1024,256>>>();
  sample<<<1,1,0,cudaStreamTailLaunch>>>(0,0);
  submitted(cudaGetLastError());
}
}
__global__ void run() {
  printf("GH_BENCH_PROTOCOL globaltimer-cdp-tail-v1 raw-ticks no-empty-subtraction no-host-timing ranking_allowed=%u\n",unsigned(!observe::enabled));
  fixtures<<<1024,256>>>();
  inputs_ready<<<1,1,0,cudaStreamTailLaunch>>>();
  submitted(cudaGetLastError());
}
}
#endif

// GH_SOURCE_CATEGORY: tests
// Frozen quality import
#if GH_MODE == 12
namespace gh::quality {
constexpr u32 threads = 256, metric_capacity = 16;
enum class Phase : u32 { dataset, model, prepared, encoded, predicted, compared,
                         reference_metrics, legacy_import, legacy_gate, candidate_metrics, gate };
struct State {
  Array<const std::byte> model_bytes, prediction_bytes, quality_bytes;
  Workspace storage, scratch;
  u64 permanent{}, cells{};
  DatasetRecord dataset;
  Model model;
  Array<float> values, targets;
  Array<std::uint16_t> bins;
  Array<double> reference, candidate, labels;
  MetricInput input;
  Metric metrics[2][metric_capacity];
  MetricReport reports[2];
  MetricVerdict verdicts[metric_capacity];
  MetricGate gate;
  Metric legacy_metrics[2][metric_capacity];
  MetricReport legacy_reports[2];
  MetricVerdict legacy_verdicts[metric_capacity];
  MetricGate legacy_gate;
  bool legacy_pass{};
  Status status;
  unsigned long long bit_differences{}, value_differences{}, decisions{}, nonfinite{};
  unsigned long long first_difference{UINT64_MAX}, maximum_absolute_bits{};
};
__device__ State state{};
__global__ void advance(Phase);

__device__ bool require(bool condition, const char* what) {
  if (!condition) { printf("quality failure: %s\n", what); assert(condition); }
  return condition;
}
__device__ bool submitted(cudaError_t error) {
  if (error != cudaSuccess) printf("quality CUDA submission failure: %u\n", u32(error));
  return require(error == cudaSuccess, "CUDA submission");
}
__device__ bool completed(Phase phase) {
  const auto s = state.status;
  if (!s.done || s.errors)
    printf("quality phase=%u done=%u errors=%u required_bytes=%llu\n", u32(phase), s.done,
      s.errors, static_cast<unsigned long long>(s.required_bytes));
  return require(s.done && !s.errors, "phase completion");
}
__device__ void then(cudaError_t error, Phase phase) {
  if (!submitted(error)) return;
  advance<<<1, 1, 0, cudaStreamTailLaunch>>>(phase);
  submitted(cudaGetLastError());
}
__device__ u32 grid(u64 n) { return u32(min(ceil_div(n, threads), u64(65535))); }
__device__ unsigned long long bits(double x) { return __double_as_longlong(x); }
__device__ bool finite(double x) { return (bits(x) & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL; }
__device__ bool suffix(Arena arena) {
  if (!arena.valid || !add_fits(arena.used, 15)) return require(false, "arena extent");
  arena.used = (arena.used + 15) & ~u64(15);
  if (!arena.fits(&state.status)) return require(false, "512 MiB arena capacity");
  state.permanent = arena.used;
  state.scratch = {state.storage.data + arena.used, state.storage.bytes - arena.used};
  return true;
}
__global__ void convert() {
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < state.cells; i += u64(gridDim.x) * threads) {
    u64 word = 0;
    for (u32 b = 0; b < 8; ++b) word |= u64(state.prediction_bytes.data[i * 8 + b]) << (8 * b);
    state.reference.data[i] = __longlong_as_double(word);
  }
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < state.labels.size; i += u64(gridDim.x) * threads)
    state.labels.data[i] = double(state.dataset.data.targets.data[i]);
}
__global__ void compare_predictions() {
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < state.cells; i += u64(gridDim.x) * threads) {
    const double a = state.reference.data[i], b = state.candidate.data[i];
    if (!finite(a)) atomicAdd(&state.nonfinite, 1ULL);
    if (bits(a) != bits(b)) {
      atomicAdd(&state.bit_differences, 1ULL);
      atomicMin(&state.first_difference, static_cast<unsigned long long>(i));
    }
    if (a != b) atomicAdd(&state.value_differences, 1ULL);
    if (finite(a) && finite(b)) atomicMax(&state.maximum_absolute_bits, bits(fabs(__dsub_rn(a, b))));
    if (state.model.objective == Objective::binary_logistic && ((a >= .5) != (b >= .5)))
      atomicAdd(&state.decisions, 1ULL);
  }
  if (state.model.objective == Objective::multiclass_softmax) {
    for (u64 r = u64(blockIdx.x) * threads + threadIdx.x; r < state.dataset.data.rows; r += u64(gridDim.x) * threads) {
      u32 a = 0, b = 0;
      for (u32 o = 1; o < state.model.outputs; ++o) {
        if (state.reference.data[r * state.model.outputs + o] > state.reference.data[r * state.model.outputs + a]) a = o;
        if (state.candidate.data[r * state.model.outputs + o] > state.candidate.data[r * state.model.outputs + b]) b = o;
      }
      if (a != b) atomicAdd(&state.decisions, 1ULL);
    }
  }
}
__device__ const char* metric_name(MetricName name) {
  switch (name) {
    case MetricName::mse: return "mse";
    case MetricName::rmse: return "rmse";
    case MetricName::mae: return "mae";
    case MetricName::r2: return "r2";
    case MetricName::logloss: return "logloss";
    case MetricName::accuracy: return "accuracy";
    case MetricName::brier: return "brier";
    case MetricName::auc: return "auc";
    case MetricName::f1: return "f1";
    case MetricName::average_precision: return "average_precision";
    case MetricName::hamming_loss: return "hamming_loss";
    case MetricName::exact_match: return "exact_match";
    case MetricName::micro_f1: return "micro_f1";
    case MetricName::macro_f1: return "macro_f1";
    case MetricName::micro_ap: return "micro_ap";
    case MetricName::macro_ap: return "macro_ap";
    case MetricName::micro_auc: return "micro_auc";
    case MetricName::macro_auc: return "macro_auc";
    case MetricName::precision_at_1: return "precision_at_1";
    case MetricName::precision_at_3: return "precision_at_3";
    case MetricName::precision_at_5: return "precision_at_5";
  }
  return "invalid";
}
__device__ void prediction_report() {
  printf("prediction cells=%llu bit_differences=%llu value_differences=%llu decision_differences=%llu nonfinite_reference=%llu max_abs=%.17g\n",
    static_cast<unsigned long long>(state.cells), state.bit_differences, state.value_differences,
    state.decisions, state.nonfinite, __longlong_as_double(state.maximum_absolute_bits));
  if (state.bit_differences) {
    const u64 i = state.first_difference;
    const double a = state.reference.data[i], b = state.candidate.data[i];
    printf("first_difference index=%llu row=%llu output=%u old=%.17g new=%.17g old_bits=%016llx new_bits=%016llx\n",
      static_cast<unsigned long long>(i), static_cast<unsigned long long>(i / state.model.outputs),
      u32(i % state.model.outputs), a, b, bits(a), bits(b));
  }
}
__device__ void metric_report() {
  for (u32 i = 0; i < state.reports[0].count; ++i) {
    const auto a = state.metrics[0][i], b = state.metrics[1][i];
    printf("metric name=%s output=%u new_name=%s new_output=%u old_available=%u new_available=%u old=%.17g new=%.17g old_bits=%016llx new_bits=%016llx verdict=%u\n",
      metric_name(a.name), a.output, metric_name(b.name), b.output, a.available, b.available,
      a.value, b.value, bits(a.value), bits(b.value), u32(state.verdicts[i]));
  }
  printf("same_engine_quality checked=%u regressions=%u unavailable=%u invalid=%u old_ap_outputs=%u new_ap_outputs=%u old_auc_outputs=%u new_auc_outputs=%u\n",
    state.gate.checked, state.gate.regressions, state.gate.unavailable, state.gate.invalid,
    state.reports[0].ap_outputs, state.reports[1].ap_outputs, state.reports[0].auc_outputs, state.reports[1].auc_outputs);
  if (!state.quality_bytes.size) printf("legacy_json_metric_arithmetic=NOT_CHECKED\n");
  printf("metric_gate done=%u errors=%u\n", state.status.done, state.status.errors);
  const bool pass = state.status.done && !state.status.errors && !state.bit_differences &&
    !state.nonfinite && !state.gate.regressions && !state.gate.invalid;
  printf("frozen_prediction_and_same_engine_quality=%s\n", pass ? "PASS" : "FAIL");
  require(pass && (!state.quality_bytes.size || state.legacy_pass), "strict conformance gates");
}
__device__ void legacy_report() {
  u32 differences = 0;
  for (u32 i = 0; i < state.legacy_reports[0].count; ++i) {
    const auto a = state.legacy_metrics[0][i], b = state.legacy_metrics[1][i];
    const bool differs = a.available != b.available || (a.available && bits(a.value) != bits(b.value));
    differences += differs;
    printf("legacy_metric name=%s available=%u computed_available=%u legacy=%.17g computed=%.17g legacy_bits=%016llx computed_bits=%016llx bit_difference=%u verdict=%u\n",
      metric_name(a.name), a.available, b.available, a.value, b.value, bits(a.value), bits(b.value), u32(differs), u32(state.legacy_verdicts[i]));
  }
  const auto a = state.legacy_reports[0], b = state.legacy_reports[1];
  const bool metadata = a.ap_outputs == b.ap_outputs && a.auc_outputs == b.auc_outputs;
  const bool valid = state.status.done && !state.status.errors && !state.legacy_gate.invalid;
  const bool exact = valid && metadata && !differences;
  const bool quality = valid && !state.legacy_gate.regressions;
  printf("legacy_eligibility ap=%u computed_ap=%u auc=%u computed_auc=%u\n", a.ap_outputs, b.ap_outputs, a.auc_outputs, b.auc_outputs);
  printf("legacy_json_metric_arithmetic=%s differences=%u\n", exact ? "PASS" : "FAIL", differences);
  printf("legacy_zero_allowance_quality=%s checked=%u regressions=%u unavailable=%u invalid=%u errors=%u\n",
    quality ? "PASS" : "FAIL", state.legacy_gate.checked, state.legacy_gate.regressions,
    state.legacy_gate.unavailable, state.legacy_gate.invalid, state.status.errors);
  state.legacy_pass = exact && quality;
}
__global__ void advance(Phase phase) {
  if (phase == Phase::gate) { metric_report(); return; }
  if (phase == Phase::legacy_gate) {
    legacy_report(); state.status = {};
    state.input.predictions = {state.candidate.data, state.candidate.size};
    then(evaluate_metrics(state.input, &state.reports[1], state.scratch, &state.status), Phase::candidate_metrics);
    return;
  }
  if (!completed(phase)) return;
  state.status = {};
  switch (phase) {
    case Phase::dataset:
      then(decode_model(state.model_bytes, &state.model, state.scratch, &state.status), Phase::model);
      return;
    case Phase::model: {
      const auto d = state.dataset.data; const auto m = state.model;
      const bool multiclass = m.objective == Objective::multiclass_softmax;
      if (!require(d.columns == m.schema.columns && state.dataset.objective == m.objective &&
          (multiclass ? d.outputs == 1 && state.dataset.classes == m.outputs : d.outputs == m.outputs), "model/dataset compatibility")) return;
      state.cells = u64(d.rows) * m.outputs;
      if (!require(state.cells && mul_fits(state.cells, 8) && state.prediction_bytes.size == state.cells * 8,
          "prediction wire extent")) return;
      Arena arena{state.storage, state.permanent};
      state.bins = {arena.take<std::uint16_t>(u64(d.rows) * d.columns), u64(d.rows) * d.columns};
      state.reference = {arena.take<double>(state.cells), state.cells};
      state.candidate = {arena.take<double>(state.cells), state.cells};
      state.labels = {arena.take<double>(d.targets.size), d.targets.size};
      if (!suffix(arena)) return;
      state.status = {};
      const auto task = m.objective == Objective::squared_error ? MetricTask::regression :
        multiclass ? MetricTask::multiclass : m.outputs == 1 ? MetricTask::binary : MetricTask::multilabel;
      state.input = {{state.labels.data, state.labels.size}, {}, {}, d.rows, m.outputs, task, MetricProfile::real_data};
      for (u32 i = 0; i < 2; ++i) state.reports[i].metrics = {state.metrics[i], metric_capacity};
      printf("quality objective=%u task=%u rows=%u features=%u outputs=%u trees=%llu nodes=%llu permanent_bytes=%llu scratch_bytes=%llu\n",
        u32(m.objective), u32(task), d.rows, d.columns, m.outputs, static_cast<unsigned long long>(m.tree_count),
        static_cast<unsigned long long>(m.node_count), static_cast<unsigned long long>(state.permanent),
        static_cast<unsigned long long>(state.scratch.bytes));
      convert<<<grid(state.cells), threads>>>();
      if (submitted(cudaGetLastError())) then(finish(&state.status), Phase::prepared);
      return;
    }
    case Phase::prepared:
      then(encode(state.dataset.data, &state.model.schema, state.bins, &state.status), Phase::encoded);
      return;
    case Phase::encoded:
      then(predict(&state.model, {state.bins.data, state.bins.size}, state.dataset.data.rows,
        state.candidate, false, &state.status), Phase::predicted);
      return;
    case Phase::predicted:
      compare_predictions<<<grid(state.cells), threads>>>();
      if (submitted(cudaGetLastError())) then(finish(&state.status), Phase::compared);
      return;
    case Phase::compared:
      prediction_report();
      state.input.predictions = {state.reference.data, state.reference.size};
      then(evaluate_metrics(state.input, &state.reports[0], state.scratch, &state.status), Phase::reference_metrics);
      return;
    case Phase::reference_metrics:
      if (state.quality_bytes.size) {
        for (u32 i = 0; i < 2; ++i) state.legacy_reports[i].metrics = {state.legacy_metrics[i], metric_capacity};
        then(decode_metric_reference(state.quality_bytes, &state.reports[0], &state.legacy_reports[0],
          &state.legacy_reports[1], &state.status), Phase::legacy_import);
        return;
      }
      state.input.predictions = {state.candidate.data, state.candidate.size};
      then(evaluate_metrics(state.input, &state.reports[1], state.scratch, &state.status), Phase::candidate_metrics);
      return;
    case Phase::legacy_import:
      then(compare_metrics(&state.legacy_reports[0], &state.legacy_reports[1], {state.legacy_verdicts, metric_capacity},
        &state.legacy_gate, &state.status), Phase::legacy_gate);
      return;
    case Phase::candidate_metrics:
      then(compare_metrics(&state.reports[0], &state.reports[1], {state.verdicts, metric_capacity}, &state.gate,
        &state.status), Phase::gate);
      return;
    case Phase::gate:
    case Phase::legacy_gate:
      return;
  }
}
__global__ void run(Array<const std::byte> data, Array<const std::byte> model,
                    Array<const std::byte> predictions, Array<const std::byte> quality, Workspace storage) {
  state.model_bytes = model; state.prediction_bytes = predictions;
  state.quality_bytes = quality;
  state.storage = storage; state.status = {};
  Arena arena{storage};
  state.values = {arena.take<float>(data.size / 4), data.size / 4};
  state.targets = {arena.take<float>(data.size / 4), data.size / 4};
  auto& m = state.model;
  m.schema.features = {arena.take<Feature>(model.size / 12), model.size / 12};
  m.schema.metadata = {arena.take<float>(model.size / 4), model.size / 4};
  m.schema.offsets = {arena.take<u32>(model.size / 12 + 1), model.size / 12 + 1};
  m.nodes = {arena.take<Node>(model.size / 28), model.size / 28};
  m.trees = {arena.take<Tree>(model.size / 8), model.size / 8};
  m.base = {arena.take<double>(model.size / 8), model.size / 8};
  m.output_offsets = {arena.take<u64>(model.size / 8 + 1), model.size / 8 + 1};
  if (!suffix(arena)) return;
  state.status = {};
  then(decode_dataset(data, state.values, state.targets, &state.dataset, &state.status), Phase::dataset);
}
}

extern "C" cudaError_t gh_quality_launch(const void* data, std::size_t data_n,
    const void* model, std::size_t model_n, const void* predictions, std::size_t predictions_n,
    const void* quality, std::size_t quality_n, void* arena, std::size_t arena_n) {
  gh::quality::run<<<1,1>>>({static_cast<const std::byte*>(data),data_n},
    {static_cast<const std::byte*>(model),model_n}, {static_cast<const std::byte*>(predictions),predictions_n},
    {static_cast<const std::byte*>(quality),quality_n}, {static_cast<std::byte*>(arena),arena_n});
  return cudaGetLastError();
}
#endif

// GH_SOURCE_CATEGORY: tests
// Direct production leaf checks
#if GH_MODE == 13
namespace gh::data_leaf {
using namespace gh::data_impl;
constexpr u32 rows = 1025, features = 2, blocks = 2, unique_blocks = 5;
constexpr u32 cells = rows * features, histogram_cells = 512 * features;
constexpr u32 canary = 0xded1ca7e;
struct Buffers {
  u32 keys[2][cells + 2];
  u32 histogram[histogram_cells + 2];
  u32 scan_totals[features * 2 + 2];
  u32 unique[features * (unique_blocks + 1) + 2];
};
__device__ Buffers buffers;

__device__ u32 domain(u32 feature) { return feature ? 241 : 251; }
__device__ u32 key(u32 feature, u32 value) {
  return feature ? (value << 24) | (((value * 13) % 241) << 8) | ((value * 17) % 16)
                 : value * 0x01010101u;
}
__device__ u32 original(u32 feature, u32 row) {
  if (row % 97 == 0) return UINT32_MAX;
  const u32 value = feature ? (row * 37 + 11) % 241 : (row * 73 + 19) % 251;
  return key(feature, value);
}
__global__ void fixture() {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cells + 2) {
    buffers.keys[0][i] = i && i <= cells ? original((i - 1) / rows, (i - 1) % rows) : canary;
    buffers.keys[1][i] = canary;
  }
  if (i < histogram_cells + 2) buffers.histogram[i] = canary;
  if (i < features * 2 + 2) buffers.scan_totals[i] = canary;
  if (i < features * (unique_blocks + 1) + 2) buffers.unique[i] = canary;
}
__device__ void guards() {
  for (u32 side = 0; side < 2; ++side)
    GH_CHECK(buffers.keys[side][0] == canary && buffers.keys[side][cells + 1] == canary);
  GH_CHECK(buffers.histogram[0] == canary && buffers.histogram[histogram_cells + 1] == canary);
  GH_CHECK(buffers.scan_totals[0] == canary && buffers.scan_totals[features * 2 + 1] == canary);
  GH_CHECK(buffers.unique[0] == canary && buffers.unique[features * (unique_blocks + 1) + 1] == canary);
}
template<u32 Bits, u32 Shift, bool Prefix>
__global__ void check_histogram(const u32* source) {
  constexpr u32 radix = 1u << Bits, length = radix * blocks;
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (!i) guards();
  if (i >= features * length) return;
  const u32 feature = i / length, digit = (i % length) / blocks, block = i % blocks;
  u32 expected = 0;
  for (u32 row = 0; row < rows; ++row) {
    const u32 current = (source[feature * rows + row] >> Shift) & (radix - 1);
    if constexpr (Prefix) expected += current < digit || (current == digit && row < block * 1024);
    else expected += row / 1024 == block && current == digit;
  }
  GH_CHECK(buffers.histogram[i + 1] == expected);
  if constexpr (Bits == 4) {
    for (u32 j = i + features * length + 1; j < histogram_cells + 1; j += features * length)
      GH_CHECK(buffers.histogram[j] == canary);
  }
}
template<u32 Processed> __device__ u32 low(u32 value) {
  if constexpr (Processed == 32) return value;
  else return value & ((1u << Processed) - 1);
}
template<u32 Processed> __device__ u32 rank(u32 feature, u32 row) {
  const u32 value = low<Processed>(original(feature, row));
  u32 before = 0;
  for (u32 other = 0; other < rows; ++other) {
    const u32 current = low<Processed>(original(feature, other));
    before += current < value || (current == value && other < row);
  }
  return before;
}
template<u32 Bits, u32 Shift> __global__ void check_pass() {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (!i) guards();
  if (i >= cells) return;
  constexpr u32 source = (Shift / Bits) % 2, destination = 1 - source;
  const u32 feature = i / rows, row = i % rows, expected = original(feature, row);
  GH_CHECK(buffers.keys[destination][1 + feature * rows + rank<Shift + Bits>(feature, row)] == expected);
  if constexpr (Shift == 0) GH_CHECK(buffers.keys[source][i + 1] == expected);
  else GH_CHECK(buffers.keys[source][1 + feature * rows + rank<Shift>(feature, row)] == expected);
}
__global__ void prepare_unique() {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cells) buffers.keys[1][i + 1] = canary;
}
// Count analytic distinct-key starts preceding a sorted-position boundary.
__device__ u32 unique_before(u32 feature, u32 end) {
  u32 distinct = 0;
  for (u32 value = 0; value < domain(feature); ++value) {
    const u32 candidate = key(feature, value);
    u32 first = 0, present = 0;
    for (u32 row = 0; row < rows; ++row) {
      const u32 actual = original(feature, row);
      first += actual < candidate; present += actual == candidate;
    }
    GH_CHECK(present);
    distinct += first < end;
  }
  return distinct;
}
template<bool Prefix> __global__ void check_unique_counts() {
  const u32 i = threadIdx.x;
  if (!i) guards();
  if (i >= features * (unique_blocks + 1)) return;
  const u32 feature = i / (unique_blocks + 1), block = i % (unique_blocks + 1);
  const u32 begin = min(block * 256, rows);
  const u32 expected = Prefix ? unique_before(feature, begin) : block == unique_blocks ? 0 :
    unique_before(feature, min(begin + 256, rows)) - unique_before(feature, begin);
  GH_CHECK(buffers.unique[i + 1] == expected);
}
template<u32 Bits> __global__ void check_compact() {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (!i) guards();
  if (i < cells) {
    const u32 feature = i / rows, value = i % rows;
    GH_CHECK(buffers.keys[1][i + 1] == (value < domain(feature) ? key(feature, value) : canary));
  }
  if (!i) printf("GH_GPU_ACTIVITY data_leaf radix=%u rows=1025 features=2 passes=%u stable_counts_prefix_unique=pass\n", Bits, 32 / Bits);
}

// All expressions below are fixed runtime submission/constant address marshalling.
// No host code computes fixtures, observed extents, or expected outcomes.
#define GH_DATA_LEAF(...) do { __VA_ARGS__; const auto error = cudaGetLastError(); \
  if (error != cudaSuccess) return error; } while (false)
template<u32 Bits, u32 Shift = 0> cudaError_t digits(Buffers* p) {
  constexpr u32 length = (1u << Bits) * blocks;
  const auto* source = p->keys[(Shift / Bits) % 2] + 1;
  auto* destination = p->keys[1 - (Shift / Bits) % 2] + 1;
  GH_DATA_LEAF(radix_counts<Bits><<<4, 256>>>(source, p->histogram + 1, rows, blocks, Shift));
  GH_DATA_LEAF(check_histogram<Bits, Shift, false><<<4, 256>>>(source));
  if constexpr (Bits == 4) {
    GH_DATA_LEAF(scan_tiles<<<2, 256>>>(p->histogram + 1, length, 1, nullptr));
  } else {
    GH_DATA_LEAF(scan_tiles<<<4, 256>>>(p->histogram + 1, length, 2, p->scan_totals + 1));
    GH_DATA_LEAF(scan_tiles<<<2, 256>>>(p->scan_totals + 1, 2, 1, nullptr));
    GH_DATA_LEAF(scan_offsets<<<4, 256>>>(p->histogram + 1, p->scan_totals + 1, length, 2, 1024));
  }
  GH_DATA_LEAF(check_histogram<Bits, Shift, true><<<4, 256>>>(source));
  GH_DATA_LEAF(radix_move<Bits><<<4, 256>>>(source, destination, p->histogram + 1, rows, blocks, Shift));
  GH_DATA_LEAF(check_pass<Bits, Shift><<<9, 256>>>());
  if constexpr (Shift + Bits < 32) return digits<Bits, Shift + Bits>(p);
  return cudaSuccess;
}
template<u32 Bits> cudaError_t run(Buffers* p) {
  GH_DATA_LEAF(fixture<<<9, 256>>>());
  const auto error = digits<Bits>(p);
  if (error != cudaSuccess) return error;
  GH_DATA_LEAF(prepare_unique<<<9, 256>>>());
  GH_DATA_LEAF(unique_keys<false><<<10, 256>>>(p->keys[0] + 1, p->unique + 1, nullptr, rows, unique_blocks));
  GH_DATA_LEAF(check_unique_counts<false><<<1, 32>>>());
  GH_DATA_LEAF(scan_tiles<<<2, 256>>>(p->unique + 1, 6, 1, nullptr));
  GH_DATA_LEAF(check_unique_counts<true><<<1, 32>>>());
  GH_DATA_LEAF(unique_keys<true><<<10, 256>>>(p->keys[0] + 1, p->unique + 1, p->keys[1] + 1, rows, unique_blocks));
  GH_DATA_LEAF(check_compact<Bits><<<9, 256>>>());
  return cudaSuccess;
}
#undef GH_DATA_LEAF
}
#endif

// GH_SOURCE_CATEGORY: tooling
// CUDA bootstrap wrappers
#if !GH_IMPLEMENTATION && GH_MODE > 0 && GH_MODE != 12
extern "C" cudaError_t gh_initialize(int) {
#if GH_MODE == 2 || GH_MODE == 11
  const auto error = gh::count::initialize_runtime();
  if (error != cudaSuccess) return error;
#endif
#if GH_MODE == 11
  return gh::bench::frozen_count::initialize_runtime();
#else
  return cudaSuccess;
#endif
}
extern "C" cudaError_t gh_launch(int) {
#if GH_MODE == 1
  gh::test::core_suite::run<<<1,1>>>();
#endif
#if GH_MODE == 2
  gh::test::count_suite::run<<<1,1>>>();
#endif
#if GH_MODE == 3
  gh::test::data_suite::run<<<1,1>>>();
#endif
#if GH_MODE == 4
  gh::test::model_suite::run<<<1,1>>>();
#endif
#if GH_MODE == 5
  gh::test::learn_suite::run<<<1,1>>>();
#endif
#if GH_MODE == 6
  gh::test::observe_suite::run<<<1,1>>>();
#endif
#if GH_MODE == 7
  gh::test::metrics_suite::run<<<1,1>>>();
#endif
#if GH_MODE == 8
  gh::test::prediction_suite::run<<<1,1>>>();
#endif
#if GH_MODE == 9
  gh::test::csv_suite::run<<<1,1>>>();
#endif
#if GH_MODE == 10
  gh::test::reference_suite::run<<<1,1>>>();
#endif
#if GH_MODE == 11
  gh::test::benchmark_suite::run<<<1,1>>>();
#endif
#if GH_MODE == 13
  gh::data_leaf::Buffers* storage{};
  auto error = cudaGetSymbolAddress(reinterpret_cast<void**>(&storage),gh::data_leaf::buffers);
  if (error == cudaSuccess) error = gh::data_leaf::run<4>(storage);
  if (error == cudaSuccess) error = gh::data_leaf::run<8>(storage);
  return error;
#else
  return cudaGetLastError();
#endif
}
extern "C" cudaError_t gh_after(int) {
#if GH_MODE == 2
  return count_graph_checks();
#else
  return cudaSuccess;
#endif
}
#endif
