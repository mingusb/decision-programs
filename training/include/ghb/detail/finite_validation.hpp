#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>

#if defined(__CUDACC__)
#define GHB_FINITE_HD __host__ __device__
#else
#define GHB_FINITE_HD
#endif

namespace ghb::detail {

static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559 &&
              std::numeric_limits<double>::digits == 53);
static_assert(sizeof(std::uint32_t) == 4 && sizeof(std::size_t) <= sizeof(std::uint64_t));

// Pure binary64 classification, including every NaN payload without FP arithmetic.
struct NonfiniteBits {
  GHB_FINITE_HD static constexpr std::uint32_t operator()(std::uint64_t bits) noexcept {
    return (bits & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000);
  }
};

inline constexpr std::uint32_t finite_validation_identity = 0;
inline constexpr std::uint32_t finite_validation_threads = 256;
// The full-cover one-dimensional launch targets the SM86 grid-x limit.
inline constexpr std::uint32_t finite_validation_max_blocks = 0x7fffffff;
inline constexpr std::size_t finite_validation_status_bytes = sizeof(std::uint32_t);

enum class FiniteValidationError { extent_overflow, grid_overflow };

struct FiniteValidationPlan {
  const std::uint64_t elements;
  const std::uint32_t blocks;
};

// Pure checked launch arithmetic; this neither queries nor promises GPU capacity.
constexpr std::expected<FiniteValidationPlan, FiniteValidationError>
make_finite_validation_plan(std::uint64_t elements) noexcept {
  if (elements > std::numeric_limits<std::size_t>::max() / sizeof(double))
    return std::unexpected(FiniteValidationError::extent_overflow);
  const auto blocks = elements / finite_validation_threads +
                      (elements % finite_validation_threads != 0);
  if (blocks > finite_validation_max_blocks)
    return std::unexpected(FiniteValidationError::grid_overflow);
  return FiniteValidationPlan{elements, std::uint32_t(blocks)};
}

} // namespace ghb::detail

#undef GHB_FINITE_HD
