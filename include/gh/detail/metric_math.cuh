#pragma once
#include "gh/core.cuh"
#include <cuda/std/bit>
#include <cmath>

namespace gh::detail {
__device__ inline bool finite_metric(double x) {
  return (cuda::std::bit_cast<u64>(x) & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}
// Actual fsum operands in the synthetic protocol are finite and nonnegative.
// Three 26-bit pieces per input; <=UINT32_MAX inputs cannot overflow a limb.
struct PositiveSum {
  u64 limbs[84]{};
  __device__ void add(double x, Status* s) {
    if (!finite_metric(x) || x < 0) { fail(s, numeric); return; }
    const u64 bits = cuda::std::bit_cast<u64>(x);
    const u32 exponent = u32((bits >> 52) & 2047);
    const u64 significand = (bits & 0xfffffffffffffULL) | (exponent ? 1ULL << 52 : 0);
    const u32 shift = exponent ? exponent - 1 : 0, first = shift / 26, offset = shift % 26;
    for (u32 k = 0; k < 3; ++k) {
      const int right = int(k * 26) - int(offset);
      limbs[first + k] += (right >= 0 ? significand >> right : significand << -right) & 0x3ffffffULL;
    }
  }
  __device__ double value(Status* s) {
    u64 carry = 0;
    for (u32 i = 0; i < 84; ++i) {
      const u64 v = limbs[i] + carry; limbs[i] = v & 0x3ffffffULL; carry = v >> 26;
    }
    int last = 83;
    while (last >= 0 && !limbs[last]) --last;
    if (last < 0) return 0;
    u32 highest = u32(last * 26 + 63 - __clzll(limbs[last]));
    const u32 shift = highest > 52 ? highest - 52 : 0, first = shift / 26, offset = shift % 26;
    u64 significant = ((limbs[first] >> offset) | (limbs[first + 1] << (26 - offset)) |
      (limbs[first + 2] << (52 - offset))) & 0x1fffffffffffffULL;
    if (shift) {
      const u32 bit = shift - 1, word = bit / 26, within = bit % 26;
      const bool guard = (limbs[word] >> within) & 1;
      bool sticky = (limbs[word] & ((1ULL << within) - 1)) != 0;
      for (u32 i = 0; i < word; ++i) sticky |= limbs[i] != 0;
      if (guard && (sticky || (significant & 1))) {
        if (++significant == (1ULL << 53)) { significant >>= 1; ++highest; }
      }
    }
    if (highest > 2097 || carry) { fail(s, numeric); return 0; }
    const u64 bits = highest < 52 ? significant : (u64(highest - 51) << 52) | (significant & 0xfffffffffffffULL);
    return cuda::std::bit_cast<double>(bits);
  }
};
template<class Get> __device__ double positive_sum(u32 count, Get get, Status* s) {
  PositiveSum sum;
  for (u32 i = 0; i < count; ++i) sum.add(get(i), s);
  return sum.value(s);
}

// NumPy's 128-value leaves and eight ordered accumulators. The explicit stack
// is bounded by the u32 count; this is an arithmetic schedule, not a generic sum.
template<class Get> __device__ double numpy_sum(u32 count, Get get) {
  u32 starts[32], lengths[32];
  double left_values[32];
  bool right[32];
  u32 begin = 0, length = count, depth = 0;
  double result;
  while (true) {
    while (length > 128) {
      const u32 left = (length / 2) & ~7u;
      starts[depth] = begin + left; lengths[depth] = length - left; right[depth++] = false;
      length = left;
    }
    if (length < 8) {
      result = -0.0;
      for (u32 i = 0; i < length; ++i) result = __dadd_rn(result, get(begin + i));
    } else {
      double parts[8];
      for (u32 k = 0; k < 8; ++k) parts[k] = get(begin + k);
      u32 i = 8;
      for (; i < (length & ~7u); i += 8)
        for (u32 k = 0; k < 8; ++k) parts[k] = __dadd_rn(parts[k], get(begin + i + k));
      result = __dadd_rn(__dadd_rn(__dadd_rn(parts[0], parts[1]), __dadd_rn(parts[2], parts[3])),
        __dadd_rn(__dadd_rn(parts[4], parts[5]), __dadd_rn(parts[6], parts[7])));
      for (; i < length; ++i) result = __dadd_rn(result, get(begin + i));
    }
    while (depth && right[depth - 1]) { --depth; result = __dadd_rn(left_values[depth], result); }
    if (!depth) return __dadd_rn(0.0, result); // NumPy's reduction identity.
    right[depth - 1] = true; left_values[depth - 1] = result;
    begin = starts[depth - 1]; length = lengths[depth - 1];
  }
}
template<class Get> __device__ double column_sum(u32 count, u32 columns, Get get) {
  if (columns == 1) return numpy_sum(count, get);
  double sum = 0;
  for (u32 i = 0; i < count; ++i) sum = __dadd_rn(sum, get(i));
  return sum;
}

template<u32 Power, class ErrorAt, class WeightAt>
__device__ double scaled_mean(u32 count, ErrorAt error_at, WeightAt weight_at, double total, Status* s) {
  int denominator_exponent;
  const double denominator = frexp(total, &denominator_exponent);
  int maximum_exponent = INT32_MIN;
  double maximum_error = 0;
  for (u32 i = 0; i < count; ++i) {
    const double weight = weight_at(i);
    if (weight <= 0) continue;
    const double error = error_at(i);
    if (!finite_metric(error) || error < 0) { fail(s, numeric); return 0; }
    maximum_error = fmax(maximum_error, error);
    if (!error) continue;
    int exponent, weight_exponent;
    frexp(error, &exponent); frexp(weight, &weight_exponent);
    maximum_exponent = max(maximum_exponent, int(Power) * exponent + weight_exponent - denominator_exponent);
  }
  if (maximum_exponent == INT32_MIN) return 0;
  PositiveSum sum;
  for (u32 i = 0; i < count; ++i) {
    const double weight = weight_at(i);
    if (weight <= 0) continue;
    const double error = error_at(i);
    if (!error) continue;
    int exponent, weight_exponent;
    const double magnitude = frexp(error, &exponent), wm = frexp(weight, &weight_exponent);
    const double powered = Power == 2 ? __dmul_rn(magnitude, magnitude) : magnitude;
    const double term = __ddiv_rn(__dmul_rn(powered, wm), denominator);
    sum.add(ldexp(term, int(Power) * exponent + weight_exponent - denominator_exponent - maximum_exponent), s);
  }
  double result;
  if constexpr (Power == 2) {
    int exponent = maximum_exponent / 2, remainder = maximum_exponent % 2;
    if (remainder < 0) { --exponent; remainder += 2; }
    result = ldexp(sqrt(ldexp(sum.value(s), remainder)), exponent);
  } else result = ldexp(sum.value(s), maximum_exponent);
  return fmin(result, maximum_error);
}
}
