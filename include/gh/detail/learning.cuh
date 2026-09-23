#pragma once
#include "gh/learn.cuh"
#include <cmath>

namespace gh::detail {
template<unsigned Order> struct Stats {
  double d[Order]{};
  unsigned long long count{};
};
static_assert(sizeof(Stats<2>) == 24 && sizeof(Stats<3>) == 32 && sizeof(Stats<4>) == 40);
struct Choice {
  std::int32_t feature{-1};
  u32 threshold{}, missing_left{};
  double gain{}, value{}, left{}, right{};
};
struct Leaf { double value{}, benefit{}; };
__device__ inline double sigmoid(double x) {
  if (x >= 0) return 1.0 / (1.0 + exp(-x));
  const double e = exp(x); return e / (1.0 + e);
}
template<unsigned O> __device__ Stats<O> add(Stats<O> a, const Stats<O>& b) {
#pragma unroll
  for (unsigned k = 0; k < O; ++k) a.d[k] += b.d[k];
  a.count += b.count; return a;
}
template<unsigned O> __device__ Stats<O> subtract(Stats<O> a, const Stats<O>& b) {
#pragma unroll
  for (unsigned k = 0; k < O; ++k) a.d[k] -= b.d[k];
  a.d[1] = fmax(0.0, a.d[1]); a.count -= b.count; return a;
}
template<unsigned Mode, class T> __device__ T shuffle(T x, unsigned delta) {
  if constexpr (Mode == 0) return __shfl_down_sync(0xffffffff, x, delta);
  if constexpr (Mode == 1) return __shfl_up_sync(0xffffffff, x, delta);
  if constexpr (Mode == 2) return __shfl_sync(0xffffffff, x, delta);
}
template<unsigned Mode, unsigned O> __device__ Stats<O> shuffle(Stats<O> x, unsigned delta) {
#pragma unroll
  for (unsigned k = 0; k < O; ++k) x.d[k] = shuffle<Mode>(x.d[k], delta);
  x.count = shuffle<Mode>(x.count, delta); return x;
}
template<unsigned O> __device__ Stats<O> warp_sum(Stats<O> x) {
  for (unsigned d = 16; d; d >>= 1) x = add(x, shuffle<0>(x, d));
  return x;
}
template<unsigned O> __device__ Stats<O> warp_scan(Stats<O> x) {
  for (unsigned d = 1; d < 32; d <<= 1) {
    auto y = shuffle<1>(x, d);
    if ((threadIdx.x & 31) >= d) x = add(x, y);
  }
  return x;
}
__device__ inline double warp_sum(double x) {
  for (unsigned d = 16; d; d >>= 1) x += shuffle<0>(x, d);
  return x;
}
__device__ inline double warp_max(double x) {
  for (unsigned d = 16; d; d >>= 1) x = fmax(x, shuffle<0>(x, d));
  return x;
}
__device__ inline double block_sum(double x, double* scratch) {
  x = warp_sum(x);
  if (!(threadIdx.x & 31)) scratch[threadIdx.x >> 5] = x;
  __syncthreads();
  if (threadIdx.x < 32) x = warp_sum(threadIdx.x < 8 ? scratch[threadIdx.x] : 0.0);
  return x;
}
template<unsigned O, unsigned Block> __device__ Stats<O> block_sum(Stats<O> x, Stats<O>* scratch) {
  x = warp_sum(x);
  if constexpr (Block == 32) {
    // The order-2 narrow split emulates the original two-level block schedule.
    if constexpr (O == 2) x = warp_sum(threadIdx.x == 0 ? x : Stats<O>{});
    return shuffle<2>(x, 0);
  } else {
    if (!(threadIdx.x & 31)) scratch[threadIdx.x >> 5] = x;
    __syncthreads();
    if (threadIdx.x < 32) {
      x = warp_sum(threadIdx.x < Block / 32 ? scratch[threadIdx.x] : Stats<O>{});
      if (!threadIdx.x) scratch[0] = x;
    }
    __syncthreads(); x = scratch[0]; __syncthreads(); return x;
  }
}
template<unsigned O, unsigned Block> __device__ Stats<O> block_scan(Stats<O> x, Stats<O>* scratch, Stats<O>& total) {
  x = warp_scan(x);
  if constexpr (Block == 32) { total = shuffle<2>(x, 31); return x; }
  const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 31) scratch[warp] = x;
  __syncthreads();
  if (!warp) {
    auto prefix = warp_scan(lane < Block / 32 ? scratch[lane] : Stats<O>{});
    if (lane < Block / 32) scratch[lane] = prefix;
  }
  __syncthreads();
  if (warp) x = add(scratch[warp - 1], x);
  total = scratch[Block / 32 - 1]; __syncthreads(); return x;
}
template<unsigned O> __device__ double benefit(const Stats<O>& s, double v, double l2) {
  if constexpr (O == 2) return -v * (s.d[0] + 0.5 * (s.d[1] + l2) * v);
  double tail = s.d[2] / 6.0;
  if constexpr (O == 4) tail += v * (s.d[3] / 24.0);
  return -v * (s.d[0] + v * (0.5 * (s.d[1] + l2) + v * tail));
}
template<unsigned O> __device__ Leaf leaf(const Stats<O>& s, const TrainConfig& c) {
  const double a = s.d[1] + c.l2;
  if (!(a > 0)) return {};
  double v = -s.d[0] / a;
  if (c.max_leaf_value > 0) v = fmin(c.max_leaf_value, fmax(-c.max_leaf_value, v));
  Leaf result{v, benefit(s, v, c.l2)};
  if constexpr (O > 2) {
    if (!isfinite(s.d[0]) || !isfinite(a) || !(c.max_leaf_value > 0) || !isfinite(c.max_leaf_value)) return {};
    if (!isfinite(result.benefit) || result.benefit < 0) result = {};
    const double x = s.d[0] / a, ratio = x * (s.d[2] / a);
    if (!isfinite(x) || !isfinite(ratio)) return result;
    double numerator = 1, denominator = 1 - 0.5 * ratio;
    if constexpr (O == 4) {
      const double fourth = x * x * (s.d[3] / a);
      if (!isfinite(fourth)) return result;
      numerator = 1 - 0.5 * ratio; denominator = 1 - ratio + fourth / 6;
    }
    if (!isfinite(numerator) || !isfinite(denominator) || !(denominator > 1e-12)) return result;
    double proposed = -x * numerator / denominator;
    if (!isfinite(proposed) || !((s.d[0] > 0 && proposed < 0) || (s.d[0] < 0 && proposed > 0))) return result;
    proposed = fmin(c.max_leaf_value, fmax(-c.max_leaf_value, proposed));
    const double score = benefit(s, proposed, c.l2);
    if (isfinite(score) && score > result.benefit) result = {proposed, score};
  }
  return result;
}
__device__ inline Choice better(Choice a, Choice b) {
  if (b.feature < 0) return a;
  if (a.feature < 0 || b.gain > a.gain) return b;
  if (b.gain < a.gain) return a;
  if (b.feature != a.feature) return b.feature < a.feature ? b : a;
  if (b.threshold != a.threshold) return b.threshold < a.threshold ? b : a;
  return b.missing_left < a.missing_left ? b : a;
}
__device__ inline bool goes_left(u32 bin, FeatureType type, u32 threshold, u32 missing_left) {
  return !bin ? missing_left != 0 : type == FeatureType::categorical ? bin == threshold : bin <= threshold;
}
}
