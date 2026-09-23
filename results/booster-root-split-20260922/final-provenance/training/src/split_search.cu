#include "ghb/split_search.cuh"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ghb::gpu {
namespace {
using Index = unsigned long long;
constexpr unsigned full_warp = 0xffffffffu;

__device__ Stats add(Stats a, Stats b) {
  return {a.gradient + b.gradient, a.hessian + b.hessian, a.count + b.count};
}
__device__ Stats subtract(Stats a, Stats b) {
  return {a.gradient - b.gradient, fmax(0.0, a.hessian - b.hessian), a.count - b.count};
}
__device__ Stats down(Stats value, unsigned offset) {
  return {__shfl_down_sync(full_warp, value.gradient, offset),
          __shfl_down_sync(full_warp, value.hessian, offset),
          __shfl_down_sync(full_warp, value.count, offset)};
}
__device__ Stats sum(Stats value) {
  for (unsigned offset = 16; offset; offset >>= 1) value = add(value, down(value, offset));
  return value;
}
__device__ Stats scan(Stats value) {
  for (unsigned offset = 1; offset < 32; offset <<= 1) {
    Stats other{__shfl_up_sync(full_warp, value.gradient, offset),
                __shfl_up_sync(full_warp, value.hessian, offset),
                __shfl_up_sync(full_warp, value.count, offset)};
    if (threadIdx.x >= offset) value = add(value, other);
  }
  return value;
}
__device__ Stats broadcast(Stats value) {
  return {__shfl_sync(full_warp, value.gradient, 0),
          __shfl_sync(full_warp, value.hessian, 0),
          __shfl_sync(full_warp, value.count, 0)};
}
__device__ double leaf(Stats stats, SplitConfig config) {
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
    const Stats left = missing_left ? add(nonmissing_left, missing) : nonmissing_left;
    const Stats right = subtract(total, left);
    if (left.count < config.min_leaf_rows || right.count < config.min_leaf_rows ||
        left.hessian < config.min_child_hessian || right.hessian < config.min_child_hessian) continue;
    const double l = leaf(left, config), r = leaf(right, config);
    const double gain = benefit(left, l, config) + benefit(right, r, config) - benefit(total, best.value, config);
    if (!(gain > config.min_gain) || !isfinite(gain)) continue;
    Split split;
    split.feature = static_cast<int>(feature); split.threshold = threshold; split.missing_left = missing_left;
    split.gain = gain; split.value = best.value; split.left_value = l; split.right_value = r;
    best = better(best, split);
  }
  return best;
}
__device__ Split warp_best(Split value) {
  for (unsigned offset = 16; offset; offset >>= 1) {
    Split other;
    other.feature = __shfl_down_sync(full_warp, value.feature, offset);
    other.threshold = __shfl_down_sync(full_warp, value.threshold, offset);
    other.missing_left = __shfl_down_sync(full_warp, value.missing_left, offset);
    other.gain = __shfl_down_sync(full_warp, value.gain, offset);
    other.value = __shfl_down_sync(full_warp, value.value, offset);
    other.left_value = __shfl_down_sync(full_warp, value.left_value, offset);
    other.right_value = __shfl_down_sync(full_warp, value.right_value, offset);
    if (threadIdx.x < offset) value = better(value, other);
  }
  return value;
}

__global__ void warp_candidates(DataView data, const Stats* histograms, unsigned capacity,
                                const unsigned* active, SplitConfig config,
                                bool force_leaf, Split* candidates) {
  const unsigned nodes = min(capacity, *active);
  for (Index task = blockIdx.x; task < Index(nodes) * data.columns; task += gridDim.x) {
    const unsigned node = task / data.columns, feature = task % data.columns;
    const unsigned offset = data.offsets[feature], bins = data.offsets[feature + 1] - offset;
    const Stats* histogram = histograms + Index(node) * data.total_bins + offset;
    Stats local{};
    if (threadIdx.x < bins) local = add(local, histogram[threadIdx.x]);
    local = sum(local);
    // Match both levels of the original 256-thread reduction. Its seven
    // unused warps contribute positive zero; keeping this level also retains
    // signed-zero behavior instead of assuming x + 0 is bitwise dispensable.
    local = sum(threadIdx.x == 0 ? local : Stats{});
    const Stats total = broadcast(local), missing = histogram[0];
    Split best; best.value = leaf(total, config);
    if (!force_leaf && total.count >= 2ULL * config.min_leaf_rows) {
      if (!threadIdx.x) best = consider(best, total, {}, missing, feature, 0, config);
      const unsigned bin = 1 + threadIdx.x;
      if (data.types[feature] == FeatureType::categorical) {
        if (bin < bins) best = consider(best, total, histogram[bin], missing, feature, bin, config);
      } else {
        // The original useful lanes are entirely in warp zero and in the
        // first 256-bin scan tile. Retain its explicit zero carry addition.
        const Stats prefix = add(Stats{}, scan(bin < bins ? histogram[bin] : Stats{}));
        if (bin < bins) best = consider(best, total, prefix, missing, feature, bin, config);
      }
    }
    best = warp_best(best);
    if (!threadIdx.x) candidates[task] = best;
  }
}
__global__ void warp_winners(const Split* candidates, unsigned capacity, const unsigned* active,
                             unsigned columns, Split* winners) {
  const unsigned nodes = min(capacity, *active);
  for (Index node = blockIdx.x; node < nodes; node += gridDim.x) {
    Split best; best.value = candidates[node * columns].value;
    if (threadIdx.x < columns) best = better(best, candidates[node * columns + threadIdx.x]);
    best = warp_best(best);
    if (!threadIdx.x) winners[node] = best;
  }
}
} // namespace

cudaError_t find_splits_warp_active(DataView data, const Stats* histograms, unsigned capacity,
                                    const unsigned* active, SplitConfig config, bool force_leaf,
                                    Split* candidates, Split* winners, cudaStream_t stream) {
  if (data.max_feature_bins > 32 || data.columns > 32)
    return find_splits_active(data, histograms, capacity, active, config, force_leaf, candidates, winners, stream);
  if (!data.rows || !data.columns || !data.bins || !data.offsets || !data.types ||
      data.total_bins < data.columns || !data.max_feature_bins || data.max_feature_bins > data.total_bins ||
      !capacity || capacity > unsigned(INT32_MAX) || !active || !histograms || !candidates || !winners ||
      Index(capacity) * data.total_bins > std::numeric_limits<std::size_t>::max() / sizeof(Stats) ||
      Index(capacity) * data.columns > std::numeric_limits<std::size_t>::max() / sizeof(Split) ||
      !config.min_leaf_rows || !std::isfinite(config.l2) || config.l2 < 0 ||
      !std::isfinite(config.min_child_hessian) || config.min_child_hessian < 0 ||
      !std::isfinite(config.min_gain) || config.min_gain < 0 ||
      !std::isfinite(config.max_leaf_value) || config.max_leaf_value < 0) return cudaErrorInvalidValue;
  const auto grid = unsigned(std::min<Index>(Index(capacity) * data.columns, 65535));
  warp_candidates<<<grid, 32, 0, stream>>>(data, histograms, capacity, active, config, force_leaf, candidates);
  const auto error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  warp_winners<<<std::min(capacity, 65535u), 32, 0, stream>>>(candidates, capacity, active, data.columns, winners);
  return cudaGetLastError();
}
} // namespace ghb::gpu
