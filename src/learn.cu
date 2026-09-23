#include "gh/detail/learning.cuh"
#include <math_constants.h>

namespace gh {
namespace {
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
    TrainConfig config, Training* output, Workspace workspace, Status* status) {
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
