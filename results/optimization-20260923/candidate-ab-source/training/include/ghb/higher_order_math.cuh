#pragma once

#include "ghb/kernels.cuh"

#include <cmath>

namespace ghb::gpu {

// Separate from Stats: the measured order-2 layout and arithmetic stay intact.
template<unsigned Order> struct HigherStats {
  static_assert(Order == 3 || Order == 4);
  double gradient{}, hessian{}, extra[Order - 2]{};
  unsigned long long count{};
};

struct LeafEstimate { double value{}, benefit{}; };

// True weighted logistic loss derivatives. Count belongs to the histogram,
// not the loss; it is left zero here. Targets/weights are validated by training.
template<unsigned Order>
__host__ __device__ inline HigherStats<Order> higher_logistic(
    double margin, double target, double weight) {
  HigherStats<Order> result;
  if (!(weight > 0.0)) return result;
  const double u = ::exp(-::fabs(margin));
  const double inverse = 1.0 / (1.0 + u);
  const double small = u * inverse;
  const double p = margin >= 0.0 ? inverse : small;
  const double q = margin >= 0.0 ? small : inverse;
  const double curvature = u * inverse * inverse;
  result.gradient = weight * (target == 1.0 ? -q : p);
  result.hessian = weight * curvature;
  result.extra[0] = result.hessian * (1.0 - 2.0 * p);
  if constexpr (Order == 4)
    result.extra[1] = result.hessian * (1.0 - 6.0 * curvature);
  return result;
}

// Benefit at the actual proposed, unshrunk leaf value, including lambda.
// This is a local Taylor-model score, not an actual-loss decrease certificate.
template<unsigned Order>
__host__ __device__ inline double higher_benefit(
    const HigherStats<Order>& stats, double value, double l2) {
  double higher = stats.extra[0] / 6.0;
  if constexpr (Order == 4) higher += value * (stats.extra[1] / 24.0);
  return -value * (stats.gradient + value * (0.5 * (stats.hessian + l2) + value * higher));
}

template<unsigned Order>
__host__ __device__ inline LeafEstimate higher_leaf(
    const HigherStats<Order>& stats, const SplitConfig& config) {
  LeafEstimate best;
  const double curvature = stats.hessian + config.l2;
  const double radius = config.max_leaf_value;
  if (!std::isfinite(stats.gradient) || !std::isfinite(curvature) || !(curvature > 0.0) ||
      !std::isfinite(radius) || !(radius > 0.0)) return best;

  const double x = stats.gradient / curvature;
  const double newton = ::fmin(radius, ::fmax(-radius, -x));
  const double newton_benefit = higher_benefit(stats, newton, config.l2);
  if (std::isfinite(newton_benefit) && newton_benefit >= 0.0)
    best = {newton, newton_benefit};

  // Reject overflow in normalized ratios instead of forming A^2 or A^3.
  if (!std::isfinite(x)) return best;
  const double ratio = x * (stats.extra[0] / curvature);
  if (!std::isfinite(ratio)) return best;
  double numerator = 1.0;
  double denominator = 1.0 - 0.5 * ratio;
  if constexpr (Order == 4) {
    const double fourth = x * x * (stats.extra[1] / curvature);
    if (!std::isfinite(fourth)) return best;
    numerator = 1.0 - 0.5 * ratio;
    denominator = 1.0 - ratio + fourth / 6.0;
  }
  if (!std::isfinite(numerator) || !std::isfinite(denominator) || !(denominator > 1e-12)) return best;
  double proposed = -x * numerator / denominator;
  // Compare signs directly: multiplying can overflow or underflow to -0.
  if (!std::isfinite(proposed) ||
      !((stats.gradient > 0.0 && proposed < 0.0) ||
        (stats.gradient < 0.0 && proposed > 0.0))) return best;
  proposed = ::fmin(radius, ::fmax(-radius, proposed));
  const double benefit = higher_benefit(stats, proposed, config.l2);
  // Strict improvement retains Newton when the two candidates have equal score.
  if (std::isfinite(benefit) && benefit > best.benefit)
    best = {proposed, benefit};
  return best;
}

static_assert(sizeof(HigherStats<3>) == 32);
static_assert(sizeof(HigherStats<4>) == 40);

} // namespace ghb::gpu
