#include "ghb/kernels.cuh"
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace ghb::gpu {
namespace {

constexpr unsigned threads = 256;
constexpr unsigned warps = threads / 32;
constexpr unsigned shared_limit = 48 * 1024;
constexpr unsigned max_grid = 65535;
using Index = unsigned long long;

unsigned grid_for(Index size) {
  return static_cast<unsigned>(std::min<Index>((size + threads - 1) / threads, max_grid));
}
bool objective_valid(Objective objective, unsigned outputs) {
  return outputs && (objective == Objective::squared_error || objective == Objective::binary_logistic ||
      (objective == Objective::multiclass_softmax && outputs >= 2));
}
bool prediction_size_valid(unsigned rows, unsigned outputs) {
  return rows && outputs && Index(rows) * outputs <= std::numeric_limits<std::size_t>::max() / sizeof(double);
}
bool data_valid(DataView data) {
  return data.rows && data.columns && data.columns <= unsigned(INT32_MAX) && data.bins && data.offsets && data.types &&
      data.total_bins >= data.columns && data.max_feature_bins && data.max_feature_bins <= 65536 &&
      data.max_feature_bins <= data.total_bins;
}
bool histogram_size_valid(DataView data, unsigned nodes) {
  return nodes && nodes <= unsigned(INT32_MAX) &&
      Index(nodes) * data.total_bins <= std::numeric_limits<std::size_t>::max() / sizeof(Stats);
}

__device__ Stats plus(Stats a, Stats b) {
  return {a.gradient + b.gradient, a.hessian + b.hessian, a.count + b.count};
}
__device__ Stats minus(Stats a, Stats b) {
  // H is nonnegative; independent floating-point reduction orders can leave a
  // tiny negative residual when subtracting a prefix containing the whole node.
  return {a.gradient - b.gradient, fmax(0.0, a.hessian - b.hessian), a.count - b.count};
}
__device__ Stats warp_sum(Stats value) {
  for (unsigned offset = 16; offset; offset >>= 1) {
    Stats other{__shfl_down_sync(0xffffffff, value.gradient, offset),
                __shfl_down_sync(0xffffffff, value.hessian, offset),
                __shfl_down_sync(0xffffffff, value.count, offset)};
    value = plus(value, other);
  }
  return value;
}
__device__ Stats block_sum(Stats value, Stats* buffer) {
  const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  value = warp_sum(value);
  if (!lane) buffer[warp] = value;
  __syncthreads();
  if (!warp) {
    value = lane < warps ? buffer[lane] : Stats{};
    value = warp_sum(value);
    if (!lane) buffer[0] = value;
  }
  __syncthreads();
  value = buffer[0];
  __syncthreads();
  return value;
}
__device__ Stats warp_scan(Stats value) {
  const unsigned lane = threadIdx.x % 32;
  for (unsigned offset = 1; offset < 32; offset <<= 1) {
    Stats other{__shfl_up_sync(0xffffffff, value.gradient, offset),
                __shfl_up_sync(0xffffffff, value.hessian, offset),
                __shfl_up_sync(0xffffffff, value.count, offset)};
    if (lane >= offset) value = plus(value, other);
  }
  return value;
}
__device__ Stats block_scan(Stats value, Stats* buffer, Stats& total) {
  const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  value = warp_scan(value);
  if (lane == 31) buffer[warp] = value;
  __syncthreads();
  if (!warp) {
    Stats prefixes = warp_scan(lane < warps ? buffer[lane] : Stats{});
    if (lane < warps) buffer[lane] = prefixes;
  }
  __syncthreads();
  if (warp) value = plus(buffer[warp - 1], value);
  total = buffer[warps - 1];
  __syncthreads();
  return value;
}
__device__ double warp_sum(double value) {
  for (unsigned offset = 16; offset; offset >>= 1)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return value;
}
__device__ double warp_max(double value) {
  for (unsigned offset = 16; offset; offset >>= 1)
    value = fmax(value, __shfl_down_sync(0xffffffff, value, offset));
  return value;
}
__device__ double block_sum(double value, double* buffer) {
  const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  value = warp_sum(value);
  if (!lane) buffer[warp] = value;
  __syncthreads();
  if (!warp) value = warp_sum(lane < warps ? buffer[lane] : 0.0);
  return value;
}
__device__ double sigmoid(double value) {
  if (value >= 0) return 1.0 / (1.0 + exp(-value));
  const double exponential = exp(value);
  return exponential / (1.0 + exponential);
}
__device__ double binary_loss(double margin, double target) {
  return margin >= 0 ? (1.0 - target) * margin + log1p(exp(-margin))
                     : -target * margin + log1p(exp(margin));
}

__global__ void initialize_kernel(double* prediction, const double* base, Index size, unsigned outputs) {
  for (Index index = Index(blockIdx.x) * blockDim.x + threadIdx.x; index < size;
       index += Index(gridDim.x) * blockDim.x)
    prediction[index] = base[index % outputs];
}

__global__ void independent_gradients(Objective objective, const double* prediction, const float* target,
    const float* weights, double* gradient, double* hessian, Index size, unsigned outputs,
    unsigned begin, unsigned count) {
  for (Index index = Index(blockIdx.x) * blockDim.x + threadIdx.x; index < size;
       index += Index(gridDim.x) * blockDim.x) {
    const Index row = index / count, source = row * outputs + begin + index % count;
    const double weight = weights ? weights[row] : 1.0;
    double g = 0.0, h = 0.0;
    if (weight > 0) {
      if (objective == Objective::squared_error) {
        g = weight * (prediction[source] - target[source]); h = weight;
      } else {
        const double p = sigmoid(prediction[source]);
        g = weight * (p - target[source]);
        h = weight * fmax(p * (1.0 - p), 1e-16);
      }
    }
    gradient[index] = g; hessian[index] = h;
  }
}

// One warp per row gives coalesced class loads for large output spaces. All
// requested outputs use the same complete pre-round margin vector.
__global__ void softmax_gradients(const double* prediction, const float* target, const float* weights,
    double* gradient, double* hessian, unsigned rows, unsigned outputs, unsigned begin, unsigned count) {
  const unsigned lane = threadIdx.x % 32;
  for (Index row = Index(blockIdx.x) * warps + threadIdx.x / 32; row < rows;
       row += Index(gridDim.x) * warps) {
    const double weight = weights ? weights[row] : 1.0;
    if (weight == 0) {
      for (Index local = lane; local < count; local += 32)
        gradient[row * count + local] = hessian[row * count + local] = 0.0;
      continue;
    }
    double maximum = -CUDART_INF;
    for (Index output = lane; output < outputs; output += 32)
      maximum = fmax(maximum, prediction[row * outputs + output]);
    maximum = __shfl_sync(0xffffffff, warp_max(maximum), 0);
    double denominator = 0;
    for (Index output = lane; output < outputs; output += 32)
      denominator += exp(prediction[row * outputs + output] - maximum);
    denominator = __shfl_sync(0xffffffff, warp_sum(denominator), 0);
    const unsigned label = static_cast<unsigned>(target[row]);
    for (Index local = lane; local < count; local += 32) {
      const unsigned output = begin + static_cast<unsigned>(local);
      const double p = exp(prediction[row * outputs + output] - maximum) / denominator;
      gradient[row * count + local] = weight * (p - (output == label ? 1.0 : 0.0));
      // Diagonal upper-bound convention, not the full softmax Hessian.
      hessian[row * count + local] = weight * fmax(2.0 * p * (1.0 - p), 1e-16);
    }
  }
}

__global__ void global_histogram(DataView data, const std::int32_t* assignments, const double* gradient,
    const double* hessian, unsigned outputs, unsigned output, unsigned nodes, Stats* histogram) {
  for (Index row = Index(blockIdx.x) * blockDim.x + threadIdx.x; row < data.rows;
       row += Index(gridDim.x) * blockDim.x) {
    const int node = assignments[row];
    if (node < 0 || unsigned(node) >= nodes) continue;
    const double g = gradient[row * outputs + output], h = hessian[row * outputs + output];
    for (unsigned feature = 0; feature < data.columns; ++feature) {
      const unsigned bin = data.bins[Index(feature) * data.rows + row];
      Stats* destination = histogram + Index(node) * data.total_bins + data.offsets[feature] + bin;
      if (g != 0) atomicAdd(&destination->gradient, g);
      if (h != 0) atomicAdd(&destination->hessian, h);
      atomicAdd(&destination->count, 1ULL);
    }
  }
}

// A CTA owns one feature and one interleaved row tile. Every active node's
// feature histogram fits in shared memory, so assignments are read once per
// feature/tile rather than rescanning the tile independently for each node.
__global__ void shared_histogram(DataView data, const std::int32_t* assignments, const double* gradient,
    const double* hessian, unsigned outputs, unsigned output, unsigned nodes, unsigned chunks, Stats* histogram) {
  extern __shared__ unsigned long long shared_words[];
  auto* local = reinterpret_cast<Stats*>(shared_words);
  const Index tasks = Index(data.columns) * chunks;
  for (Index task = blockIdx.x; task < tasks; task += gridDim.x) {
    const unsigned feature = static_cast<unsigned>(task / chunks), chunk = task % chunks;
    const unsigned offset = data.offsets[feature], bins = data.offsets[feature + 1] - offset;
    const unsigned cells = nodes * bins;
    for (unsigned cell = threadIdx.x; cell < cells; cell += blockDim.x) local[cell] = Stats{};
    __syncthreads();
    for (Index row = Index(chunk) * blockDim.x + threadIdx.x; row < data.rows;
         row += Index(chunks) * blockDim.x) {
      const int node = assignments[row];
      if (node < 0 || unsigned(node) >= nodes) continue;
      const unsigned bin = data.bins[Index(feature) * data.rows + row];
      Stats* destination = local + unsigned(node) * bins + bin;
      const double g = gradient[row * outputs + output], h = hessian[row * outputs + output];
      if (g != 0) atomicAdd(&destination->gradient, g);
      if (h != 0) atomicAdd(&destination->hessian, h);
      atomicAdd(&destination->count, 1ULL);
    }
    __syncthreads();
    for (unsigned cell = threadIdx.x; cell < cells; cell += blockDim.x) {
      const Stats value = local[cell];
      if (!value.count) continue;
      Stats* destination = histogram + Index(cell / bins) * data.total_bins + offset + cell % bins;
      if (value.gradient != 0) atomicAdd(&destination->gradient, value.gradient);
      if (value.hessian != 0) atomicAdd(&destination->hessian, value.hessian);
      atomicAdd(&destination->count, value.count);
    }
    __syncthreads();
  }
}

__device__ double leaf_value(Stats stats, SplitConfig config) {
  const double denominator = stats.hessian + config.l2;
  if (!(denominator > 0)) return 0.0;
  double value = -stats.gradient / denominator;
  if (config.max_leaf_value > 0) value = fmin(config.max_leaf_value, fmax(-config.max_leaf_value, value));
  return value;
}
__device__ double benefit(Stats stats, double value, SplitConfig config) {
  return -value * (stats.gradient + 0.5 * (stats.hessian + config.l2) * value);
}
__device__ Split better(Split a, Split b) {
  if (b.feature < 0) return a;
  if (a.feature < 0 || b.gain > a.gain) return b;
  if (b.gain < a.gain) return a;
  if (b.feature != a.feature) return b.feature < a.feature ? b : a;
  if (b.threshold != a.threshold) return b.threshold < a.threshold ? b : a;
  return b.missing_left < a.missing_left ? b : a;
}
__device__ Split consider(Split best, Stats total, Stats nonmissing_left, Stats missing,
    unsigned feature, unsigned threshold, SplitConfig config) {
  for (unsigned missing_left = 0; missing_left < 2; ++missing_left) {
    const Stats left = missing_left ? plus(nonmissing_left, missing) : nonmissing_left;
    const Stats right = minus(total, left);
    if (left.count < config.min_leaf_rows || right.count < config.min_leaf_rows ||
        left.hessian < config.min_child_hessian || right.hessian < config.min_child_hessian) continue;
    const double l = leaf_value(left, config), r = leaf_value(right, config);
    const double gain = benefit(left, l, config) + benefit(right, r, config) - benefit(total, best.value, config);
    if (!(gain > config.min_gain) || !isfinite(gain)) continue;
    Split split;
    split.feature = static_cast<int>(feature); split.threshold = threshold; split.missing_left = missing_left;
    split.gain = gain; split.value = best.value; split.left_value = l; split.right_value = r;
    best = better(best, split);
  }
  return best;
}
__device__ Split block_best(Split value, Split* buffer) {
  buffer[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = threads / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride) buffer[threadIdx.x] = better(buffer[threadIdx.x], buffer[threadIdx.x + stride]);
    __syncthreads();
  }
  value = buffer[0];
  __syncthreads();
  return value;
}

__global__ void split_candidates(DataView data, const Stats* histogram, unsigned nodes,
    SplitConfig config, bool force_leaf, Split* candidates) {
  __shared__ Stats sums[warps];
  __shared__ Split choices[threads];
  for (Index task = blockIdx.x; task < Index(nodes) * data.columns; task += gridDim.x) {
    const unsigned node = task / data.columns, feature = task % data.columns;
    const unsigned offset = data.offsets[feature], bins = data.offsets[feature + 1] - offset;
    const Stats* feature_histogram = histogram + Index(node) * data.total_bins + offset;
    Stats local{};
    for (unsigned bin = threadIdx.x; bin < bins; bin += blockDim.x) local = plus(local, feature_histogram[bin]);
    const Stats total = block_sum(local, sums), missing = feature_histogram[0];
    Split best; best.value = leaf_value(total, config);
    if (!force_leaf && total.count >= 2ULL * config.min_leaf_rows) {
      // threshold zero represents a missing-only left child for either type.
      if (!threadIdx.x) best = consider(best, total, {}, missing, feature, 0, config);
      if (data.types[feature] == FeatureType::categorical) {
        for (unsigned bin = 1 + threadIdx.x; bin < bins; bin += blockDim.x)
          best = consider(best, total, feature_histogram[bin], missing, feature, bin, config);
      } else {
        Stats preceding{};
        for (unsigned start = 1; start < bins; start += blockDim.x) {
          const unsigned bin = start + threadIdx.x;
          Stats tile_total;
          const Stats prefix = plus(preceding, block_scan(bin < bins ? feature_histogram[bin] : Stats{}, sums, tile_total));
          preceding = plus(preceding, tile_total);
          if (bin < bins) best = consider(best, total, prefix, missing, feature, bin, config);
        }
      }
    }
    best = block_best(best, choices);
    if (!threadIdx.x) candidates[task] = best;
    __syncthreads();
  }
}
__global__ void split_winners(const Split* candidates, unsigned nodes, unsigned columns, Split* winners) {
  __shared__ Split choices[threads];
  for (Index node = blockIdx.x; node < nodes; node += gridDim.x) {
    Split best; best.value = candidates[node * columns].value;
    for (unsigned feature = threadIdx.x; feature < columns; feature += blockDim.x)
      best = better(best, candidates[node * columns + feature]);
    best = block_best(best, choices);
    if (!threadIdx.x) winners[node] = best;
    __syncthreads();
  }
}
__device__ bool goes_left(unsigned bin, FeatureType type, unsigned threshold, unsigned missing_left) {
  if (!bin) return missing_left != 0;
  return type == FeatureType::categorical ? bin == threshold : bin <= threshold;
}
__global__ void route_kernel(DataView data, std::int32_t* assignments, const Split* winners,
    const std::int32_t* left_map, const std::int32_t* right_map) {
  for (Index row = Index(blockIdx.x) * blockDim.x + threadIdx.x; row < data.rows;
       row += Index(gridDim.x) * blockDim.x) {
    const int node = assignments[row];
    if (node < 0) continue;
    const Split split = winners[node];
    if (split.feature < 0) { assignments[row] = -1; continue; }
    const unsigned bin = data.bins[Index(split.feature) * data.rows + row];
    assignments[row] = goes_left(bin, data.types[split.feature], split.threshold, split.missing_left)
        ? left_map[node] : right_map[node];
  }
}
__global__ void tree_kernel(DataView data, const Node* nodes, unsigned node_count,
    unsigned output, unsigned outputs, double* prediction) {
  for (Index row = Index(blockIdx.x) * blockDim.x + threadIdx.x; row < data.rows;
       row += Index(gridDim.x) * blockDim.x) {
    int current = 0;
    // Host model validation owns graph validity; bounds also prevent malformed
    // device tree data from producing an out-of-range read or an infinite loop.
    for (unsigned step = 0; step < node_count && current >= 0 && unsigned(current) < node_count; ++step) {
      const Node node = nodes[current];
      if (node.feature < 0) { prediction[row * outputs + output] += node.value; break; }
      if (unsigned(node.feature) >= data.columns) break;
      const unsigned bin = data.bins[Index(node.feature) * data.rows + row];
      current = goes_left(bin, data.types[node.feature], node.threshold, node.missing_left) ? node.left : node.right;
    }
  }
}
__global__ void sigmoid_kernel(double* prediction, Index size) {
  for (Index index = Index(blockIdx.x) * blockDim.x + threadIdx.x; index < size;
       index += Index(gridDim.x) * blockDim.x) prediction[index] = sigmoid(prediction[index]);
}
__global__ void softmax_kernel(double* prediction, unsigned rows, unsigned outputs) {
  const unsigned lane = threadIdx.x % 32;
  for (Index row = Index(blockIdx.x) * warps + threadIdx.x / 32; row < rows;
       row += Index(gridDim.x) * warps) {
    double maximum = -CUDART_INF;
    for (Index output = lane; output < outputs; output += 32) maximum = fmax(maximum, prediction[row * outputs + output]);
    maximum = __shfl_sync(0xffffffff, warp_max(maximum), 0);
    double denominator = 0;
    for (Index output = lane; output < outputs; output += 32) denominator += exp(prediction[row * outputs + output] - maximum);
    denominator = __shfl_sync(0xffffffff, warp_sum(denominator), 0);
    for (Index output = lane; output < outputs; output += 32)
      prediction[row * outputs + output] = exp(prediction[row * outputs + output] - maximum) / denominator;
  }
}

__global__ void loss_kernel(Objective objective, const double* prediction, const float* targets,
    const float* weights, unsigned rows, unsigned outputs, double* partial) {
  __shared__ double sums[warps];
  const unsigned lane = threadIdx.x % 32;
  double value = 0;
  for (Index row = Index(blockIdx.x) * warps + threadIdx.x / 32; row < rows;
       row += Index(gridDim.x) * warps) {
    const double weight = weights ? weights[row] : 1.0;
    if (weight == 0) continue;
    if (objective == Objective::multiclass_softmax) {
      double maximum = -CUDART_INF;
      for (Index output = lane; output < outputs; output += 32) maximum = fmax(maximum, prediction[row * outputs + output]);
      maximum = __shfl_sync(0xffffffff, warp_max(maximum), 0);
      double denominator = 0;
      for (Index output = lane; output < outputs; output += 32) denominator += exp(prediction[row * outputs + output] - maximum);
      denominator = __shfl_sync(0xffffffff, warp_sum(denominator), 0);
      if (!lane) value += weight * ((maximum - prediction[row * outputs + static_cast<unsigned>(targets[row])]) + log(denominator));
    } else {
      for (Index output = lane; output < outputs; output += 32) {
        const Index index = row * outputs + output;
        if (objective == Objective::squared_error) {
          const double error = prediction[index] - targets[index];
          value += (0.5 * weight * error) * error;
        } else value += weight * binary_loss(prediction[index], targets[index]);
      }
    }
  }
  value = block_sum(value, sums);
  if (!threadIdx.x) partial[blockIdx.x] = value;
}

} // namespace

cudaError_t initialize_predictions(double* predictions, const double* base, unsigned rows,
    unsigned outputs, cudaStream_t stream) {
  if (!predictions || !base || !prediction_size_valid(rows, outputs)) return cudaErrorInvalidValue;
  const Index size = Index(rows) * outputs;
  initialize_kernel<<<grid_for(size), threads, 0, stream>>>(predictions, base, size, outputs);
  return cudaGetLastError();
}
cudaError_t gradients_tile(Objective objective, const double* predictions, const float* targets,
    const float* weights, double* gradient, double* hessian, unsigned rows, unsigned outputs,
    unsigned output_begin, unsigned output_count, cudaStream_t stream) {
  if (!objective_valid(objective, outputs) || !prediction_size_valid(rows, outputs) || !predictions || !targets || !gradient || !hessian ||
      !output_count || output_begin >= outputs || output_count > outputs - output_begin)
    return cudaErrorInvalidValue;
  if (objective == Objective::multiclass_softmax) {
    const unsigned grid = static_cast<unsigned>(std::min<Index>((Index(rows) + warps - 1) / warps, max_grid));
    softmax_gradients<<<grid, threads, 0, stream>>>(predictions, targets, weights, gradient, hessian,
        rows, outputs, output_begin, output_count);
  } else {
    const Index size = Index(rows) * output_count;
    independent_gradients<<<grid_for(size), threads, 0, stream>>>(objective, predictions, targets, weights,
        gradient, hessian, size, outputs, output_begin, output_count);
  }
  return cudaGetLastError();
}
cudaError_t gradients(Objective objective, const double* predictions, const float* targets, const float* weights,
    double* gradient, double* hessian, unsigned rows, unsigned outputs, cudaStream_t stream) {
  return gradients_tile(objective, predictions, targets, weights, gradient, hessian, rows, outputs, 0, outputs, stream);
}
bool shared_supported(DataView data, unsigned nodes) {
  return data_valid(data) && histogram_size_valid(data, nodes) &&
      Index(nodes) * data.max_feature_bins <= shared_limit / sizeof(Stats);
}
cudaError_t histogram(DataView data, const std::int32_t* assignments, const double* gradient, const double* hessian,
    unsigned outputs, unsigned output, unsigned nodes, Stats* histograms, HistogramPolicy policy, cudaStream_t stream) {
  if (!data_valid(data) || !histogram_size_valid(data, nodes) || !assignments || !gradient || !hessian ||
      !histograms || !prediction_size_valid(data.rows, outputs) || output >= outputs ||
      (policy != HistogramPolicy::global && policy != HistogramPolicy::shared) ||
      (policy == HistogramPolicy::shared && !shared_supported(data, nodes))) return cudaErrorInvalidValue;
  const auto cleared = cudaMemsetAsync(histograms, 0, std::size_t(nodes) * data.total_bins * sizeof(Stats), stream);
  if (cleared != cudaSuccess) return cleared;
  if (policy == HistogramPolicy::global) {
    global_histogram<<<grid_for(data.rows), threads, 0, stream>>>(data, assignments, gradient, hessian,
        outputs, output, nodes, histograms);
  } else {
    const unsigned chunks = static_cast<unsigned>(std::min<Index>((Index(data.rows) + 4095) / 4096, 256));
    const unsigned grid = static_cast<unsigned>(std::min<Index>(Index(data.columns) * chunks, max_grid));
    const auto bytes = std::size_t(nodes) * data.max_feature_bins * sizeof(Stats);
    shared_histogram<<<grid, threads, bytes, stream>>>(data, assignments, gradient, hessian, outputs, output,
        nodes, chunks, histograms);
  }
  return cudaGetLastError();
}
cudaError_t find_splits(DataView data, const Stats* histograms, unsigned nodes, SplitConfig config,
    bool force_leaf, Split* candidates, Split* winners, cudaStream_t stream) {
  if (!data_valid(data) || !histogram_size_valid(data, nodes) || !histograms || !candidates || !winners ||
      Index(nodes) * data.columns > std::numeric_limits<std::size_t>::max() / sizeof(Split) ||
      !config.min_leaf_rows || !std::isfinite(config.l2) || config.l2 < 0 ||
      !std::isfinite(config.min_child_hessian) || config.min_child_hessian < 0 ||
      !std::isfinite(config.min_gain) || config.min_gain < 0 ||
      !std::isfinite(config.max_leaf_value) || config.max_leaf_value < 0) return cudaErrorInvalidValue;
  const unsigned grid = static_cast<unsigned>(std::min<Index>(Index(nodes) * data.columns, max_grid));
  split_candidates<<<grid, threads, 0, stream>>>(data, histograms, nodes, config, force_leaf, candidates);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  split_winners<<<std::min(nodes, max_grid), threads, 0, stream>>>(candidates, nodes, data.columns, winners);
  return cudaGetLastError();
}
cudaError_t route(DataView data, std::int32_t* assignments, const Split* winners,
    const std::int32_t* left_map, const std::int32_t* right_map, cudaStream_t stream) {
  if (!data_valid(data) || !assignments || !winners || !left_map || !right_map) return cudaErrorInvalidValue;
  route_kernel<<<grid_for(data.rows), threads, 0, stream>>>(data, assignments, winners, left_map, right_map);
  return cudaGetLastError();
}
cudaError_t add_tree(DataView data, const Node* nodes, unsigned node_count, unsigned output,
    unsigned outputs, double* predictions, cudaStream_t stream) {
  if (!data_valid(data) || !nodes || !node_count || node_count > unsigned(INT32_MAX) || !predictions ||
      !prediction_size_valid(data.rows, outputs) || output >= outputs) return cudaErrorInvalidValue;
  tree_kernel<<<grid_for(data.rows), threads, 0, stream>>>(data, nodes, node_count, output, outputs, predictions);
  return cudaGetLastError();
}
cudaError_t transform(Objective objective, double* predictions, unsigned rows, unsigned outputs, cudaStream_t stream) {
  if (!objective_valid(objective, outputs) || !predictions || !prediction_size_valid(rows, outputs)) return cudaErrorInvalidValue;
  if (objective == Objective::squared_error) return cudaSuccess;
  if (objective == Objective::binary_logistic) {
    const Index size = Index(rows) * outputs;
    sigmoid_kernel<<<grid_for(size), threads, 0, stream>>>(predictions, size);
  } else {
    const unsigned grid = static_cast<unsigned>(std::min<Index>((Index(rows) + warps - 1) / warps, max_grid));
    softmax_kernel<<<grid, threads, 0, stream>>>(predictions, rows, outputs);
  }
  return cudaGetLastError();
}
cudaError_t loss(Objective objective, const double* predictions, const float* targets, const float* weights,
    unsigned rows, unsigned outputs, double* partial_loss, unsigned blocks, cudaStream_t stream) {
  if (!objective_valid(objective, outputs) || !predictions || !targets || !prediction_size_valid(rows, outputs) || !partial_loss ||
      !blocks || blocks > max_grid) return cudaErrorInvalidValue;
  loss_kernel<<<blocks, threads, 0, stream>>>(objective, predictions, targets, weights, rows, outputs, partial_loss);
  return cudaGetLastError();
} 

} // namespace ghb::gpu
