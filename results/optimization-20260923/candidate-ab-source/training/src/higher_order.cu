#include "ghb/higher_order.cuh"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ghb::gpu {
namespace {
using Index = unsigned long long;
constexpr unsigned threads = 256, mask = 0xffffffffu;
constexpr auto limit = std::numeric_limits<std::size_t>::max();
unsigned grid_for(Index work) { return unsigned(std::min<Index>((work + threads - 1) / threads, 65535)); }
bool valid_data(DataView d) {
  return d.rows && d.columns && d.columns <= unsigned(INT32_MAX) && d.bins && d.offsets && d.types &&
      d.total_bins >= d.columns && d.max_feature_bins && d.max_feature_bins <= 65536 &&
      d.max_feature_bins <= d.total_bins && Index(d.rows) * d.columns <= limit / sizeof(std::uint16_t);
}
template<unsigned O> bool valid_derivatives(const double* g, const double* h, const double* t, const double* q) {
  return g && h && t && (O == 3 || q);
}
__device__ bool select(const OutputBatch* selector, unsigned capacity, unsigned& stride,
                       unsigned& first, unsigned& live) {
  const auto s = *selector;
  live = min(capacity, s.output_count);
  if (!live || !s.derivative_stride || s.derivative_stride > stride ||
      s.derivative_begin >= s.derivative_stride || live > s.derivative_stride - s.derivative_begin) return false;
  stride = s.derivative_stride; first = s.derivative_begin;
  return true;
}

template<unsigned O>
__global__ void derivatives(const double* prediction, const float* target, const float* weights,
                            double* g, double* h, double* t, double* q,
                            unsigned rows, unsigned outputs, unsigned begin, unsigned count) {
  for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < Index(rows) * count;
       i += Index(gridDim.x) * blockDim.x) {
    const Index row = i / count, source = row * outputs + begin + i % count;
    const auto value = higher_logistic<O>(prediction[source], target[source], weights ? weights[row] : 1.0);
    g[i] = value.gradient; h[i] = value.hessian; t[i] = value.extra[0];
    if constexpr (O == 4) q[i] = value.extra[1];
  }
}

template<unsigned O, bool Root>
__global__ void clear_histogram(DataView d, unsigned derivative_capacity, unsigned batch_capacity,
    unsigned capacity, Index output_stride, const unsigned* active, const OutputBatch* selector,
    const unsigned long long* counts, HigherStats<O>* histogram) {
  unsigned first{}, live{};
  if (!select(selector, batch_capacity, derivative_capacity, first, live)) return;
  for (Index output = blockIdx.y; output < live; output += gridDim.y) {
    const unsigned nodes = Root ? 1U : min(capacity, active[output]);
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < Index(nodes) * d.total_bins;
         i += Index(gridDim.x) * blockDim.x) {
      HigherStats<O> value{};
      if constexpr (Root) if (counts) value.count = counts[i];
      histogram[output * output_stride + i] = value;
    }
  }
}

template<unsigned O, unsigned Width, bool Root, bool Cached>
__global__ void accumulate(DataView d, const int* assignments, const double* g, const double* h,
    const double* t, const double* q, unsigned stride, unsigned batch_capacity, unsigned capacity,
    Index output_stride, const unsigned* active, const OutputBatch* selector, HigherStats<O>* histogram) {
  unsigned first{}, live{};
  if (!select(selector, batch_capacity, stride, first, live)) return;
  const unsigned groups = 1 + (live - 1) / Width;
  const Index work = Index(d.rows) * groups * Width;
  const unsigned lane = threadIdx.x & (Width - 1);
  const unsigned peers = (mask >> (32 - Width)) << ((threadIdx.x & 31u) & ~(Width - 1u));
  for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < work; i += Index(gridDim.x) * blockDim.x) {
    const Index group = i / Width, row = group / groups;
    const unsigned output = unsigned(group % groups) * Width + lane;
    int node = 0;
    bool valid = output < live;
    if constexpr (!Root) {
      node = -1;
      const unsigned nodes = valid ? min(capacity, active[output]) : 0;
      if (nodes) node = assignments[Index(output) * d.rows + row];
      valid = node >= 0 && unsigned(node) < nodes;
    }
    if constexpr (Width == 1) { if (!valid) continue; }
    else if (!__any_sync(peers, valid)) continue;
    HigherStats<O> value{};
    if (valid) {
      const Index derivative = row * stride + first + output;
      value.gradient = g[derivative]; value.hessian = h[derivative]; value.extra[0] = t[derivative];
      if constexpr (O == 4) value.extra[1] = q[derivative];
    }
    for (unsigned feature = 0; feature < d.columns; ++feature) {
      unsigned bin{};
      if (!lane) bin = d.bins[Index(feature) * d.rows + row];
      if constexpr (Width != 1) bin = __shfl_sync(peers, bin, 0, Width);
      if (valid) {
        auto* destination = histogram + Index(output) * output_stride + Index(node) * d.total_bins + d.offsets[feature] + bin;
        if (value.gradient != 0) atomicAdd(&destination->gradient, value.gradient);
        if (value.hessian != 0) atomicAdd(&destination->hessian, value.hessian);
#pragma unroll
        for (unsigned k = 0; k < O - 2; ++k)
          if (value.extra[k] != 0) atomicAdd(&destination->extra[k], value.extra[k]);
        if constexpr (!Cached) atomicAdd(&destination->count, 1ULL);
      }
    }
  }
}
template<unsigned O, bool Root, bool Cached>
void launch_accumulate(DataView d, const int* assignments, const double* g, const double* h,
    const double* t, const double* q, unsigned stride, unsigned batch, unsigned capacity,
    Index output_stride, const unsigned* active, const OutputBatch* selector, HigherStats<O>* histogram,
    cudaStream_t stream) {
  auto launch = [&]<unsigned W>() {
    const unsigned groups = 1 + (batch - 1) / W;
    accumulate<O, W, Root, Cached><<<grid_for(Index(d.rows) * groups * W), threads, 0, stream>>>(
        d, assignments, g, h, t, q, stride, batch, capacity, output_stride, active, selector, histogram);
  };
  if (batch == 1) launch.template operator()<1>();
  else if (batch <= 2) launch.template operator()<2>();
  else if (batch <= 4) launch.template operator()<4>();
  else if (batch <= 8) launch.template operator()<8>();
  else if (batch <= 16) launch.template operator()<16>();
  else launch.template operator()<32>();
}

template<unsigned O> __device__ HigherStats<O> plus(HigherStats<O> a, HigherStats<O> b) {
  a.gradient += b.gradient; a.hessian += b.hessian; a.count += b.count;
#pragma unroll
  for (unsigned k = 0; k < O - 2; ++k) a.extra[k] += b.extra[k];
  return a;
}
template<unsigned O> __device__ HigherStats<O> minus(HigherStats<O> a, HigherStats<O> b) {
  a.gradient -= b.gradient; a.hessian = fmax(0.0, a.hessian - b.hessian); a.count -= b.count;
#pragma unroll
  for (unsigned k = 0; k < O - 2; ++k) a.extra[k] -= b.extra[k];
  return a;
}
template<unsigned Mode, class T> __device__ T shuffle(T x, unsigned delta) {
  if constexpr (Mode == 0) return __shfl_down_sync(mask, x, delta);
  else if constexpr (Mode == 1) return __shfl_up_sync(mask, x, delta);
  else return __shfl_sync(mask, x, delta);
}
template<unsigned Mode, unsigned O> __device__ HigherStats<O> shuffle(HigherStats<O> x, unsigned delta) {
  x.gradient = shuffle<Mode>(x.gradient, delta); x.hessian = shuffle<Mode>(x.hessian, delta);
  x.count = shuffle<Mode>(x.count, delta);
#pragma unroll
  for (unsigned k = 0; k < O - 2; ++k) x.extra[k] = shuffle<Mode>(x.extra[k], delta);
  return x;
}
template<unsigned O> __device__ HigherStats<O> warp_sum(HigherStats<O> x) {
  for (unsigned offset = 16; offset; offset >>= 1) x = plus(x, shuffle<0>(x, offset));
  return x;
}
template<unsigned O> __device__ HigherStats<O> warp_scan(HigherStats<O> x) {
  for (unsigned offset = 1; offset < 32; offset <<= 1) {
    const auto other = shuffle<1>(x, offset);
    if ((threadIdx.x & 31) >= offset) x = plus(x, other);
  }
  return x;
}
template<unsigned O, unsigned Block> __device__ HigherStats<O> block_sum(HigherStats<O> x, HigherStats<O>* scratch) {
  x = warp_sum(x);
  if constexpr (Block == 32) {
    return shuffle<2>(x, 0);
  } else {
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    if (!lane) scratch[warp] = x;
    __syncthreads();
    if (!warp) {
      x = warp_sum(lane < Block / 32 ? scratch[lane] : HigherStats<O>{});
      if (!lane) scratch[0] = x;
    }
    __syncthreads(); x = scratch[0]; __syncthreads(); return x;
  }
}
template<unsigned O, unsigned Block> __device__ HigherStats<O> block_scan(HigherStats<O> x, HigherStats<O>* scratch,
                                                                        HigherStats<O>& total) {
  const unsigned lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  x = warp_scan(x);
  if constexpr (Block == 32) { total = shuffle<2>(x, 31); return x; }
  else {
    if (lane == 31) scratch[warp] = x;
    __syncthreads();
    if (!warp) {
      const auto prefix = warp_scan(lane < Block / 32 ? scratch[lane] : HigherStats<O>{});
      if (lane < Block / 32) scratch[lane] = prefix;
    }
    __syncthreads();
    if (warp) x = plus(scratch[warp - 1], x);
    total = scratch[Block / 32 - 1]; __syncthreads(); return x;
  }
}
__device__ Split better(Split a, Split b) {
  if (b.feature < 0) return a;
  if (a.feature < 0 || b.gain > a.gain) return b;
  if (b.gain < a.gain) return a;
  if (b.feature != a.feature) return b.feature < a.feature ? b : a;
  if (b.threshold != a.threshold) return b.threshold < a.threshold ? b : a;
  return b.missing_left < a.missing_left ? b : a;
}
template<unsigned Block> __device__ Split best_split(Split x, Split* scratch) {
  if constexpr (Block == 32) {
    for (unsigned offset = 16; offset; offset >>= 1) {
      Split other;
      other.feature = shuffle<0>(x.feature, offset); other.threshold = shuffle<0>(x.threshold, offset);
      other.missing_left = shuffle<0>(x.missing_left, offset); other.gain = shuffle<0>(x.gain, offset);
      other.value = shuffle<0>(x.value, offset); other.left_value = shuffle<0>(x.left_value, offset);
      other.right_value = shuffle<0>(x.right_value, offset);
      if (threadIdx.x < offset) x = better(x, other);
    }
    return x;
  } else {
    scratch[threadIdx.x] = x; __syncthreads();
    for (unsigned stride = Block / 2; stride; stride >>= 1) {
      if (threadIdx.x < stride) scratch[threadIdx.x] = better(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
      __syncthreads();
    }
    x = scratch[0]; __syncthreads(); return x;
  }
}
template<unsigned O>
__device__ Split consider(Split best, HigherStats<O> total, HigherStats<O> left_present,
    HigherStats<O> missing, double parent_benefit, unsigned feature, unsigned threshold, SplitConfig config) {
  for (unsigned missing_left = 0; missing_left < 2; ++missing_left) {
    const auto left = missing_left ? plus(left_present, missing) : left_present;
    const auto right = minus(total, left);
    if (left.count < config.min_leaf_rows || right.count < config.min_leaf_rows ||
        left.hessian < config.min_child_hessian || right.hessian < config.min_child_hessian) continue;
    const auto l = higher_leaf<O>(left, config), r = higher_leaf<O>(right, config);
    const double gain = l.benefit + r.benefit - parent_benefit;
    if (!(gain > config.min_gain) || !isfinite(gain)) continue;
    Split next;
    next.feature = int(feature); next.threshold = threshold; next.missing_left = missing_left;
    next.value = best.value; next.gain = gain; next.left_value = l.value; next.right_value = r.value;
    best = better(best, next);
  }
  return best;
}
template<unsigned O, unsigned Block>
__global__ void split_candidates(DataView d, const HigherStats<O>* histogram, unsigned outputs,
    unsigned capacity, const unsigned* active, const unsigned* batch_count, SplitConfig config,
    bool force_leaf, Split* candidates) {
  const Index nodes = Index(capacity) * (batch_count ? min(outputs, *batch_count) : outputs);
  __shared__ HigherStats<O> sums[Block / 32];
  __shared__ Split choices[Block];
  for (Index task = blockIdx.x; task < nodes * d.columns; task += gridDim.x) {
    const Index node = task / d.columns;
    if (node % capacity >= min(capacity, active[node / capacity])) continue;
    const unsigned feature = unsigned(task % d.columns), offset = d.offsets[feature];
    const unsigned bins = d.offsets[feature + 1] - offset;
    const auto* cells = histogram + node * d.total_bins + offset;
    HigherStats<O> local{};
    for (unsigned bin = threadIdx.x; bin < bins; bin += Block) local = plus(local, cells[bin]);
    const auto total = block_sum<O, Block>(local, sums), missing = cells[0];
    const auto parent = higher_leaf<O>(total, config);
    Split best; best.value = parent.value;
    if (!force_leaf && total.count >= 2ULL * config.min_leaf_rows) {
      if (!threadIdx.x) best = consider(best, total, HigherStats<O>{}, missing, parent.benefit, feature, 0, config);
      if (d.types[feature] == FeatureType::categorical) {
        for (unsigned bin = 1 + threadIdx.x; bin < bins; bin += Block)
          best = consider(best, total, cells[bin], missing, parent.benefit, feature, bin, config);
      } else {
        HigherStats<O> preceding{};
        for (unsigned start = 1; start < bins; start += Block) {
          const unsigned bin = start + threadIdx.x;
          HigherStats<O> tile_total;
          const auto prefix = plus(preceding, block_scan<O, Block>(bin < bins ? cells[bin] : HigherStats<O>{}, sums, tile_total));
          preceding = plus(preceding, tile_total);
          if (bin < bins) best = consider(best, total, prefix, missing, parent.benefit, feature, bin, config);
        }
      }
    }
    best = best_split<Block>(best, choices);
    if (!threadIdx.x) candidates[task] = best;
    if constexpr (Block != 32) __syncthreads();
  }
}
template<unsigned Block>
__global__ void split_winners(const Split* candidates, unsigned outputs, unsigned capacity,
    const unsigned* active, const unsigned* batch_count, unsigned columns, Split* winners) {
  const Index nodes = Index(capacity) * (batch_count ? min(outputs, *batch_count) : outputs);
  __shared__ Split choices[Block];
  for (Index node = blockIdx.x; node < nodes; node += gridDim.x) {
    if (node % capacity >= min(capacity, active[node / capacity])) continue;
    Split best; best.value = candidates[node * columns].value;
    for (unsigned feature = threadIdx.x; feature < columns; feature += Block)
      best = better(best, candidates[node * columns + feature]);
    best = best_split<Block>(best, choices);
    if (!threadIdx.x) winners[node] = best;
    if constexpr (Block != 32) __syncthreads();
  }
}
} // namespace

template<unsigned O>
cudaError_t higher_gradients_tile(const double* prediction, const float* target, const float* weights,
    double* g, double* h, double* t, double* q, unsigned rows, unsigned outputs, unsigned begin,
    unsigned count, cudaStream_t stream) {
  if (!prediction || !target || !valid_derivatives<O>(g, h, t, q) || !rows || !outputs || !count ||
      begin >= outputs || count > outputs - begin || Index(rows) * outputs > limit / sizeof(double)) return cudaErrorInvalidValue;
  derivatives<O><<<grid_for(Index(rows) * count), threads, 0, stream>>>(prediction, target, weights, g, h, t, q, rows, outputs, begin, count);
  return cudaGetLastError();
}
template<unsigned O>
cudaError_t higher_root_histogram_batch(DataView d, const double* g, const double* h, const double* t, const double* q,
    unsigned derivative_capacity, unsigned batch_capacity, std::size_t output_stride, const OutputBatch* selector,
    HigherStats<O>* output, cudaStream_t stream, const unsigned long long* counts) {
  if (!valid_data(d) || !valid_derivatives<O>(g, h, t, q) || !selector || !output || !derivative_capacity ||
      !batch_capacity || batch_capacity > derivative_capacity || output_stride < d.total_bins ||
      output_stride > limit / sizeof(HigherStats<O>) / batch_capacity ||
      Index(d.rows) * derivative_capacity > limit / sizeof(double)) return cudaErrorInvalidValue;
  clear_histogram<O, true><<<dim3(grid_for(d.total_bins), std::min(batch_capacity, 65535U)), threads, 0, stream>>>(
      d, derivative_capacity, batch_capacity, 1, output_stride, nullptr, selector, counts, output);
  auto error = cudaGetLastError(); if (error != cudaSuccess) return error;
  if (counts) launch_accumulate<O, true, true>(d, nullptr, g, h, t, q, derivative_capacity, batch_capacity, 1, output_stride, nullptr, selector, output, stream);
  else launch_accumulate<O, true, false>(d, nullptr, g, h, t, q, derivative_capacity, batch_capacity, 1, output_stride, nullptr, selector, output, stream);
  return cudaGetLastError();
}
template<unsigned O>
cudaError_t higher_deeper_histogram(DataView d, const int* assignments, const double* g, const double* h,
    const double* t, const double* q, unsigned derivative_capacity, unsigned batch_capacity, unsigned capacity,
    const unsigned* active, const OutputBatch* selector, HigherStats<O>* output, cudaStream_t stream) {
  if (!valid_data(d) || !valid_derivatives<O>(g, h, t, q) || !assignments || !active || !selector || !output ||
      !derivative_capacity || !batch_capacity || batch_capacity > derivative_capacity || !capacity || capacity > unsigned(INT32_MAX) ||
      Index(d.rows) * derivative_capacity > limit / sizeof(double) || Index(d.rows) * batch_capacity > limit / sizeof(int) ||
      Index(capacity) * d.total_bins > limit / sizeof(HigherStats<O>) / batch_capacity) return cudaErrorInvalidValue;
  const Index stride = Index(capacity) * d.total_bins;
  clear_histogram<O, false><<<dim3(grid_for(stride), std::min(batch_capacity, 65535U)), threads, 0, stream>>>(
      d, derivative_capacity, batch_capacity, capacity, stride, active, selector, nullptr, output);
  auto error = cudaGetLastError(); if (error != cudaSuccess) return error;
  launch_accumulate<O, false, false>(d, assignments, g, h, t, q, derivative_capacity, batch_capacity, capacity, stride, active, selector, output, stream);
  return cudaGetLastError();
}
template<unsigned O>
cudaError_t higher_find_splits_batched_active(DataView d, const HigherStats<O>* histogram, unsigned outputs,
    unsigned capacity, const unsigned* active, SplitConfig config, bool force_leaf, Split* candidates,
    Split* winners, SplitPolicy policy, cudaStream_t stream, const unsigned* batch_count) {
  if (!valid_data(d) || !histogram || !outputs || !capacity || capacity > unsigned(INT32_MAX) || !active || !candidates || !winners ||
      Index(capacity) * d.total_bins > limit / sizeof(HigherStats<O>) / outputs ||
      Index(capacity) * d.columns > limit / sizeof(Split) / outputs || !config.min_leaf_rows ||
      !std::isfinite(config.l2) || config.l2 < 0 || !std::isfinite(config.min_child_hessian) || config.min_child_hessian < 0 ||
      !std::isfinite(config.min_gain) || config.min_gain < 0 || !std::isfinite(config.max_leaf_value) || config.max_leaf_value <= 0 ||
      (policy != SplitPolicy::block256 && policy != SplitPolicy::warp32)) return cudaErrorInvalidValue;
  const Index nodes = Index(outputs) * capacity;
  auto launch = [&]<unsigned Block>() {
    split_candidates<O, Block><<<unsigned(std::min<Index>(nodes * d.columns, 65535)), Block, 0, stream>>>(
        d, histogram, outputs, capacity, active, batch_count, config, force_leaf, candidates);
    auto error = cudaGetLastError(); if (error != cudaSuccess) return error;
    split_winners<Block><<<unsigned(std::min<Index>(nodes, 65535)), Block, 0, stream>>>(candidates, outputs, capacity, active, batch_count, d.columns, winners);
    return cudaGetLastError();
  };
  if (policy == SplitPolicy::warp32 && d.max_feature_bins <= 32 && d.columns <= 32) return launch.template operator()<32>();
  return launch.template operator()<256>();
}

#define GHB_INSTANTIATE(O) \
  template cudaError_t higher_gradients_tile<O>(const double*, const float*, const float*, double*, double*, double*, double*, unsigned, unsigned, unsigned, unsigned, cudaStream_t); \
  template cudaError_t higher_root_histogram_batch<O>(DataView, const double*, const double*, const double*, const double*, unsigned, unsigned, std::size_t, const OutputBatch*, HigherStats<O>*, cudaStream_t, const unsigned long long*); \
  template cudaError_t higher_deeper_histogram<O>(DataView, const int*, const double*, const double*, const double*, const double*, unsigned, unsigned, unsigned, const unsigned*, const OutputBatch*, HigherStats<O>*, cudaStream_t); \
  template cudaError_t higher_find_splits_batched_active<O>(DataView, const HigherStats<O>*, unsigned, unsigned, const unsigned*, SplitConfig, bool, Split*, Split*, SplitPolicy, cudaStream_t, const unsigned*);
GHB_INSTANTIATE(3)
GHB_INSTANTIATE(4)
#undef GHB_INSTANTIATE
} // namespace ghb::gpu
