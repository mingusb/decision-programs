#include "ghb/higher_order_math.cuh"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

// CPU-only validation: no CUDA calls and no production training reference.
// Loss derivatives use independent long-double Taylor arithmetic. Leaf steps
// use reciprocal-series coefficients instead of the production rational form.
namespace {
using ghb::gpu::HigherStats;
using ghb::gpu::LeafEstimate;
using ghb::gpu::SplitConfig;
std::size_t checks{};

void require(bool condition, const char* message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}
void close(double actual, long double expected, const char* message,
           long double relative = 128 * std::numeric_limits<double>::epsilon(),
           long double absolute = 0) {
  ++checks;
  const long double bound = relative * std::abs(expected) + absolute +
                           8 * static_cast<long double>(std::numeric_limits<double>::denorm_min());
  if (!std::isfinite(actual) || !std::isfinite(expected) || std::abs(actual - expected) > bound) {
    std::ostringstream text;
    text << std::setprecision(22) << message << ": actual=" << actual << " expected=" << expected
         << " difference=" << std::abs(actual - expected) << " bound=" << bound;
    throw std::runtime_error(text.str());
  }
}
SplitConfig configuration(double l2 = 0.0, double radius = 1.0) {
  return {1, l2, 0, 0, radius};
}

struct Jet {
  std::array<long double, 5> c{}; // c[k] = derivative[k] / k!
};
Jet scale(Jet a, long double value) {
  for (auto& x : a.c) x *= value;
  return a;
}
Jet add(Jet a, const Jet& b) {
  for (unsigned k = 0; k < a.c.size(); ++k) a.c[k] += b.c[k];
  return a;
}
Jet inverse(const Jet& a) {
  Jet result;
  result.c[0] = 1 / a.c[0];
  for (unsigned k = 1; k < result.c.size(); ++k) {
    for (unsigned j = 1; j <= k; ++j) result.c[k] -= a.c[j] * result.c[k - j];
    result.c[k] /= a.c[0];
  }
  return result;
}
Jet exponential(const Jet& a) {
  Jet result;
  result.c[0] = std::exp(a.c[0]);
  for (unsigned k = 1; k < result.c.size(); ++k) {
    for (unsigned j = 1; j <= k; ++j) result.c[k] += j * a.c[j] * result.c[k - j];
    result.c[k] /= k;
  }
  return result;
}
Jet logarithm_one_plus(const Jet& a) {
  Jet shifted = a;
  shifted.c[0] += 1;
  const Jet reciprocal = inverse(shifted);
  Jet result;
  result.c[0] = std::log1p(a.c[0]);
  // d log(1+a) / dt = a'/(1+a), followed by integration.
  for (unsigned k = 1; k < result.c.size(); ++k) {
    for (unsigned j = 1; j <= k; ++j) result.c[k] += j * a.c[j] * reciprocal.c[k - j];
    result.c[k] /= k;
  }
  return result;
}
Jet logistic_loss(long double margin, unsigned target, long double weight) {
  Jet z;
  z.c[0] = margin;
  z.c[1] = 1;
  const Jet loss = margin >= 0
      ? add(scale(z, 1 - target), logarithm_one_plus(exponential(scale(z, -1))))
      : add(scale(z, -static_cast<long double>(target)), logarithm_one_plus(exponential(z)));
  return scale(loss, weight);
}

template<unsigned Order> void derivatives() {
  constexpr std::array margins{-1000.0, -745.0, -700.0, -100.0, -40.0, -2.0,
                               -1.3169578969248166, -.125, -1e-12,
                               0.0, 1e-12, .125, 1.3169578969248166, 2.0, 40.0,
                               100.0, 700.0, 745.0, 1000.0};
  constexpr std::array weights{0.0, .125, 1.0, 16.0};
  for (double margin : margins) for (unsigned target : {0u, 1u}) for (double weight : weights) {
    const auto actual = ghb::gpu::higher_logistic<Order>(margin, target, weight);
    const Jet expected = logistic_loss(margin, target, weight);
    require(actual.count == 0, "derivative helper leaves histogram count zero");
    close(actual.gradient, expected.c[1], "logistic first derivative");
    close(actual.hessian, 2 * expected.c[2], "logistic second derivative");
    // T and Q cross zero. Bound primitive-rounding cancellation by H rather
    // than requiring a relative error bound against an arbitrarily tiny result.
    const long double cancellation = 16 * std::numeric_limits<double>::epsilon() * std::abs(expected.c[2]);
    close(actual.extra[0], 6 * expected.c[3], "logistic third derivative", 3e-14L, cancellation);
    if constexpr (Order == 4)
      close(actual.extra[1], 24 * expected.c[4], "logistic fourth derivative", 3e-14L, cancellation);
    const auto reflected = ghb::gpu::higher_logistic<Order>(-margin, 1 - target, weight);
    close(actual.gradient, -static_cast<long double>(reflected.gradient), "gradient reflection");
    close(actual.hessian, reflected.hessian, "Hessian reflection");
    // 1-2*p has cancellation near zero: use a bound scaled by H for this
    // symmetry check, as in the independent Taylor check above.
    require(std::abs(actual.extra[0] + reflected.extra[0]) <=
            4 * std::numeric_limits<double>::epsilon() * actual.hessian +
            8 * std::numeric_limits<double>::denorm_min(), "third derivative reflection");
    if constexpr (Order == 4)
      close(actual.extra[1], reflected.extra[1], "fourth derivative reflection", 3e-14L, cancellation);
  }
  const auto saturated = ghb::gpu::higher_logistic<Order>(40.0, 1.0, 1.0);
  require(saturated.gradient < 0 && saturated.hessian > 0 && saturated.hessian < 1e-16,
          "stable saturated residual and unfloored Hessian");
  const auto zero = ghb::gpu::higher_logistic<Order>(std::numeric_limits<double>::quiet_NaN(), 1, 0);
  require(zero.gradient == 0 && zero.hessian == 0 && zero.extra[0] == 0,
          "zero weight avoids nonfinite loss arithmetic");
  if constexpr (Order == 4) {
    const auto midpoint = ghb::gpu::higher_logistic<4>(0, 0, 1);
    require(midpoint.hessian == .25 && midpoint.extra[0] == 0 && midpoint.extra[1] == -.125,
            "negative fourth derivative is retained");
  }
}

template<unsigned Order>
long double reference_benefit(const HigherStats<Order>& s, long double x, long double l2) {
  const long double x2 = x * x, x3 = x2 * x;
  long double loss = s.gradient * x + (s.hessian + l2) * x2 / 2 + s.extra[0] * x3 / 6;
  if constexpr (Order == 4) loss += s.extra[1] * x3 * x / 24;
  return -loss;
}

template<unsigned Order>
long double reciprocal_step(const HigherStats<Order>& s, long double l2) {
  Jet derivative;
  derivative.c[0] = s.gradient;
  derivative.c[1] = s.hessian + l2;
  derivative.c[2] = static_cast<long double>(s.extra[0]) / 2;
  if constexpr (Order == 4) derivative.c[3] = static_cast<long double>(s.extra[1]) / 6;
  const Jet reciprocal = inverse(derivative);
  if constexpr (Order == 3) return reciprocal.c[1] / reciprocal.c[2];
  else return reciprocal.c[2] / reciprocal.c[3];
}

struct ReferenceLeaf { long double value{}, benefit{}; };
template<unsigned Order>
ReferenceLeaf reference_leaf(const HigherStats<Order>& s, const SplitConfig& config) {
  const long double a = static_cast<long double>(s.hessian) + config.l2;
  const long double radius = config.max_leaf_value;
  if (!std::isfinite(s.gradient) || !std::isfinite(a) || !(a > 0) ||
      !std::isfinite(radius) || !(radius > 0)) return {};
  ReferenceLeaf best;
  const long double n = std::clamp(-s.gradient / a, -radius, radius);
  const long double b = reference_benefit(s, n, config.l2);
  if (std::isfinite(b) && b >= 0) best = {n, b};
  if (s.gradient == 0) return best;
  // The expanded mathematical denominators provide an independent guard
  // calculation for these bounded reference fixtures.
  long double denominator = (2*a*a - s.gradient*s.extra[0]) / (2*a*a);
  if constexpr (Order == 4)
    denominator = (6*a*a*a - 6*s.gradient*a*s.extra[0] +
                   static_cast<long double>(s.gradient)*s.gradient*s.extra[1]) / (6*a*a*a);
  if (!std::isfinite(denominator) || !(denominator > 1e-12L)) return best;
  long double candidate = reciprocal_step(s, config.l2);
  if (!std::isfinite(candidate) || !(candidate * s.gradient < 0)) return best;
  candidate = std::clamp(candidate, -radius, radius);
  const long double score = reference_benefit(s, candidate, config.l2);
  if (std::isfinite(score) && score > best.benefit) best = {candidate, score};
  return best;
}

template<unsigned Order>
void compare_leaf(const HigherStats<Order>& stats, const SplitConfig& config) {
  const auto actual = ghb::gpu::higher_leaf(stats, config);
  const auto expected = reference_leaf(stats, config);
  close(actual.value, expected.value, "leaf against reciprocal-series reference", 2e-13L);
  close(actual.benefit, expected.benefit, "benefit against expanded Taylor reference", 2e-13L);
  require(std::isfinite(actual.value) && std::isfinite(actual.benefit) && actual.benefit >= 0 &&
          std::abs(actual.value) <= config.max_leaf_value, "bounded finite nonnegative leaf benefit");
  require(actual.value == 0 || (actual.value < 0) != (stats.gradient < 0), "leaf is a descent direction");
}

template<unsigned Order> void proposals() {
  HigherStats<Order> sample;
  sample.gradient = .75; sample.hessian = 1; sample.extra[0] = .5; sample.count = 10;
  if constexpr (Order == 4) sample.extra[1] = 1;
  const auto config = configuration(.5, 3);
  compare_leaf(sample, config);
  const auto chosen = ghb::gpu::higher_leaf(sample, config);
  require(chosen.value != -.5, "fixture exercises higher-order proposal, not only fallback");
  close(chosen.value, reciprocal_step(sample, config.l2), "correct Householder reciprocal ratio");

  std::uint64_t random = 0x734ea892ULL;
  auto next = [&] { random ^= random << 13; random ^= random >> 7; random ^= random << 17; return random; };
  for (unsigned i = 0; i < 2048; ++i) {
    HigherStats<Order> stats;
    stats.gradient = (double(int(next() % 127) - 63) + .5) / 32;
    stats.hessian = double(1 + next() % 127) / 32;
    stats.extra[0] = double(int(next() % 129) - 64) / 16;
    if constexpr (Order == 4) stats.extra[1] = double(int(next() % 129) - 64) / 16;
    stats.count = 1 + next() % 127;
    compare_leaf(stats, configuration(double(next() % 17) / 8, double(1 + next() % 8) / 4));
  }

  // Genuine logistic aggregates, independently summed before leaf evaluation.
  for (unsigned seed = 0; seed < 32; ++seed) {
    HigherStats<Order> stats;
    for (unsigned row = 0; row < 19; ++row) {
      const Jet loss = logistic_loss((int((seed * 19 + row * 7) % 97) - 48) / 16.0L,
                                    (row + seed) % 2, 1 + row % 3);
      stats.gradient += static_cast<double>(loss.c[1]);
      stats.hessian += static_cast<double>(2 * loss.c[2]);
      stats.extra[0] += static_cast<double>(6 * loss.c[3]);
      if constexpr (Order == 4) stats.extra[1] += static_cast<double>(24 * loss.c[4]);
      ++stats.count;
    }
    compare_leaf(stats, configuration(.5, 1));
  }
}

template<unsigned Order> void safeguards() {
  const auto config = configuration(0, 1);
  HigherStats<Order> stats;
  stats.gradient = 1; stats.hessian = 1; stats.count = 1;
  for (double gradient : {-2.0, -0.0, 0.0, .125, 2.0}) {
    stats.gradient = gradient;
    const auto chosen = ghb::gpu::higher_leaf(stats, config);
    const double n = std::fmin(1.0, std::fmax(-1.0, -gradient));
    require(std::bit_cast<std::uint64_t>(chosen.value) == std::bit_cast<std::uint64_t>(n),
            "zero higher derivatives retain clipped Newton including tie/sign");
  }
  stats.gradient = 1;
  stats.hessian = 0;
  require(ghb::gpu::higher_leaf(stats, config).value == 0, "zero curvature returns zero");
  stats.hessian = -1;
  require(ghb::gpu::higher_leaf(stats, config).value == 0, "negative regularized curvature returns zero");
  stats.hessian = std::numeric_limits<double>::denorm_min();
  require(ghb::gpu::higher_leaf(stats, config).value == -1, "overflowed raw Newton clips to finite radius");
  stats.hessian = 1;
  stats.extra[0] = Order == 3 ? 2 : 1;
  compare_leaf(stats, config);
  require(ghb::gpu::higher_leaf(stats, config).value == -1, "singular denominator keeps Newton");
  stats.extra[0] = Order == 3 ? 4 : 2;
  compare_leaf(stats, config);
  require(ghb::gpu::higher_leaf(stats, config).value == -1, "negative denominator keeps Newton");
  stats.extra[0] = (Order == 3 ? 2 : 1) * (1.0 - 5e-13);
  require(ghb::gpu::higher_leaf(stats, configuration(0, 4)).value == -1,
          "positive denominator below guard threshold keeps Newton");
  stats.extra[0] = (Order == 3 ? 2 : 1) * (1.0 - 2e-12);
  require(ghb::gpu::higher_leaf(stats, configuration(0, 4)).value == -4,
          "denominator above guard threshold admits bounded improving proposal");
  stats.extra[0] = std::numeric_limits<double>::quiet_NaN();
  require(ghb::gpu::higher_leaf(stats, config).value == 0, "nonfinite Taylor scores use zero");
  stats.extra[0] = 0;
  stats.gradient = std::numeric_limits<double>::infinity();
  require(ghb::gpu::higher_leaf(stats, config).value == 0, "nonfinite gradient returns zero");
  stats.gradient = 1;
  require(ghb::gpu::higher_leaf(stats, configuration(0, 0)).value == 0, "invalid radius returns zero");
  require(ghb::gpu::higher_leaf(stats, configuration(0, std::numeric_limits<double>::infinity())).value == 0,
          "infinite radius returns zero");
  if constexpr (Order == 4) {
    stats.extra[0] = 3; stats.extra[1] = 18;
    require(ghb::gpu::higher_leaf(stats, config).value == -1, "ascent Householder proposal rejected");
    stats.extra[0] = 4; stats.extra[1] = 30;
    const auto none = ghb::gpu::higher_leaf(stats, config);
    require(none.value == 0 && none.benefit == 0, "zero beats negative-score Newton and rejected proposal");
  }
}

template<unsigned Order> void split_scores() {
  HigherStats<Order> left, right, total;
  left.gradient = -2; left.hessian = 1; left.extra[0] = .5; left.count = 4;
  right.gradient = 1; right.hessian = 2; right.extra[0] = -.25; right.count = 6;
  if constexpr (Order == 4) { left.extra[1] = -.25; right.extra[1] = .75; }
  total.gradient = left.gradient + right.gradient;
  total.hessian = left.hessian + right.hessian;
  total.count = left.count + right.count;
  for (unsigned i = 0; i < Order - 2; ++i) total.extra[i] = left.extra[i] + right.extra[i];
  const auto config = configuration(.5, 1);
  const auto l = ghb::gpu::higher_leaf(left, config), r = ghb::gpu::higher_leaf(right, config),
             p = ghb::gpu::higher_leaf(total, config);
  const double gain = l.benefit + r.benefit - p.benefit;
  const long double expected = reference_benefit(left, l.value, config.l2) +
                               reference_benefit(right, r.value, config.l2) -
                               reference_benefit(total, p.value, config.l2);
  close(gain, expected, "children-minus-parent uses consistent Taylor order");
  const double quadratic_only = -l.value*(left.gradient + .5*(left.hessian+config.l2)*l.value)
                                -r.value*(right.gradient + .5*(right.hessian+config.l2)*r.value)
                                +p.value*(total.gradient + .5*(total.hessian+config.l2)*p.value);
  require(std::abs(gain - quadratic_only) > .01, "fixture detects order-2 score substitution");
}
} // namespace

int main() try {
  static_assert(std::is_same_v<std::conditional_t<true, ghb::gpu::Stats, HigherStats<2>>, ghb::gpu::Stats>);
  static_assert(sizeof(ghb::gpu::Stats) == 24);
  derivatives<3>(); derivatives<4>();
  proposals<3>(); proposals<4>();
  safeguards<3>(); safeguards<4>();
  split_scores<3>(); split_scores<4>();
  std::cout << "higher-order independent CPU math checks passed: " << checks << '\n';
  return 0;
} catch (const std::exception& error) {
  std::cerr << "higher-order CPU math: " << error.what() << '\n';
  return 1;
}
