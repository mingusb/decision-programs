#include "ghb/detail/finite_validation.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

// CPU-only semantic and plan checks. These do not establish CUDA execution,
// asynchronous lifetime safety, launch count, or validation performance.
namespace {
std::size_t checks{};
void require(bool condition, const char* message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}
using namespace ghb::detail;

void binary64_classes() {
  static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
  static_assert(std::is_empty_v<NonfiniteBits>);
  static_assert(NonfiniteBits{}(0x7ff0000000000000ULL) == 1);
  static_assert(NonfiniteBits{}(0xffefffffffffffffULL) == 0);
  // All 2,048 exponent classes and both signs, including low/high payload
  // bits, quiet/signaling NaNs, and mantissas straddling the quiet bit.
  constexpr std::array<std::uint64_t, 7> payloads{
      0, 1, 0x0007ffffffffffffULL, 0x0008000000000000ULL,
      0x000fffffffffffffULL, 0x0005555555555555ULL, 0x000aaaaaaaaaaaaaULL};
  for (const auto sign : {std::uint64_t{0}, std::uint64_t{1} << 63}) {
    for (std::uint64_t exponent = 0; exponent < 2048; ++exponent) {
      for (const auto payload : payloads) {
        const auto bits = sign | (exponent << 52) | payload;
        require(NonfiniteBits{}(bits) == (exponent == 2047 ? 1U : 0U),
                "binary64 exponent class or payload classification differs");
      }
    }
  }
  // Explicit semantic landmarks make the class sweep's boundary intent
  // independent of whether the platform preserves signaling NaNs in FP loads.
  struct Example { std::uint64_t bits; std::uint32_t bad; };
  constexpr std::array examples{
      Example{0x0000000000000000ULL, 0}, Example{0x8000000000000000ULL, 0},
      Example{0x0000000000000001ULL, 0}, Example{0x8000000000000001ULL, 0},
      Example{0x000fffffffffffffULL, 0}, Example{0x800fffffffffffffULL, 0},
      Example{0x0010000000000000ULL, 0}, Example{0x8010000000000000ULL, 0},
      Example{0x7fefffffffffffffULL, 0}, Example{0xffefffffffffffffULL, 0},
      Example{0x7ff0000000000000ULL, 1}, Example{0xfff0000000000000ULL, 1},
      Example{0x7ff0000000000001ULL, 1}, Example{0xfff0000000000001ULL, 1},
      Example{0x7ff8000000000000ULL, 1}, Example{0xfff8000000000000ULL, 1},
      Example{0x7fffffffffffffffULL, 1}, Example{0xffffffffffffffffULL, 1}};
  for (const auto example : examples)
    require(NonfiniteBits{}(example.bits) == example.bad, "binary64 semantic landmark differs");
}

void launch_plans() {
  static_assert(finite_validation_threads == 256 && finite_validation_status_bytes == 4);
  static_assert(finite_validation_identity == 0);
  constexpr auto empty = make_finite_validation_plan(0);
  static_assert(empty && empty->elements == 0 && empty->blocks == 0);
  static_assert(std::is_const_v<decltype(FiniteValidationPlan::elements)> &&
                std::is_const_v<decltype(FiniteValidationPlan::blocks)>);
  constexpr std::array<std::uint64_t, 18> extents{
      0, 1, 2, 31, 32, 33, 63, 64, 65, 255, 256, 257, 511, 512, 513, 65535, 65536, 65537};
  for (const auto elements : extents) {
    const auto plan = make_finite_validation_plan(elements);
    require(bool(plan), "small valid extent rejected");
    require(plan->elements == elements, "plan changes the input extent");
    if (!elements) {
      require(plan->blocks == 0, "empty plan requests a kernel grid");
    } else {
      const auto capacity = std::uint64_t(plan->blocks) * finite_validation_threads;
      require(plan->blocks > 0 && plan->blocks <= finite_validation_max_blocks,
              "nonempty plan exceeds legal grid bounds");
      require(capacity >= elements && capacity - elements < finite_validation_threads,
              "launch does not cover the input with only one partial block");
    }
  }
  // Count-only capacity boundaries never allocate their apparent payloads.
  constexpr auto byte_bound = std::numeric_limits<std::size_t>::max() / sizeof(double);
  constexpr auto grid_bound = std::uint64_t(finite_validation_max_blocks) * finite_validation_threads;
  constexpr auto valid_bound = grid_bound < byte_bound ? grid_bound : byte_bound;
  const auto boundary = make_finite_validation_plan(valid_bound);
  require(bool(boundary) && boundary->elements == valid_bound,
          "last representable launch extent rejected");
  const auto too_many_bytes = make_finite_validation_plan(std::uint64_t(byte_bound) + 1);
  require(!too_many_bytes && too_many_bytes.error() == FiniteValidationError::extent_overflow,
          "byte overflow must reject before grid overflow");
  const auto maximum = make_finite_validation_plan(std::numeric_limits<std::uint64_t>::max());
  require(!maximum && maximum.error() == FiniteValidationError::extent_overflow,
          "maximum extent wraps or changes error precedence");
  if constexpr (grid_bound < byte_bound) {
    const auto too_many_blocks = make_finite_validation_plan(grid_bound + 1);
    require(!too_many_blocks && too_many_blocks.error() == FiniteValidationError::grid_overflow,
            "first oversized grid accepted or misclassified");
    require(boundary->blocks == finite_validation_max_blocks,
            "exact maximum grid has the wrong block count");
  }
}
} // namespace

int main() {
  try {
    binary64_classes();
    launch_plans();
    std::cout << "finite validation CPU checks passed: " << checks << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "finite validation CPU test failed: " << error.what() << '\n';
    return 1;
  }
}
