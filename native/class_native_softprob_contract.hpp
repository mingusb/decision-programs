#pragma once
// Numerical sufficient condition for the existing qualified native transform.
// This constant does not authenticate a source, library, loaded kernel or GPU.
// NativeClassSeparation.lean proves the conditional separation with the SAME
// exp relative error 2^-16 and division relative error 2^-20 as the older guard.
// For RN64 endpoint subtraction inside [-10,10], subtracting 2^-48 still leaves
// a true gap above representable 3*2^-16. RN32 monotonicity then bounds the
// shifted rival before the reviewed exponential/division contracts are used.
namespace native_softprob_gap {
inline constexpr double computed_gap_minimum = 0x1p-14;
inline constexpr double representable_shift_gap = 3.0 * 0x1p-16;
// Exact integer forms of the two numerical obligations, also checked by Lean.
static_assert((1ull << 34) - 1 > (3ull << 32));
static_assert(65537ull * 1048577ull * 65536ull <
              65539ull * 65535ull * 1048575ull);
} // namespace native_softprob_gap
