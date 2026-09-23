#include "ghb/finite_validation.cuh"
#include "ghb/detail/finite_validation.hpp"

#include <cuda/std/bit>
#include <cuda_runtime.h>
#include <type_traits>

namespace ghb::gpu {
namespace {

static_assert(std::is_same_v<std::uint32_t, unsigned int>);
static_assert(detail::finite_validation_threads % 32 == 0);

// Effectful lowering of OR(map(NonfiniteBits, values)). The exact OR identity is
// zero. Full warps vote even on tails; only elected lanes of bad warps write.
__global__ void finite_status_or(const double* values, std::uint64_t elements,
                                  std::uint32_t* status) {
  const auto index = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto contribution = index < elements
      ? detail::NonfiniteBits{}(cuda::std::bit_cast<std::uint64_t>(values[index]))
      : detail::finite_validation_identity;
  const auto warp_bad = __any_sync(0xffffffffu, contribution != 0);
  if ((threadIdx.x & 31u) == 0 && warp_bad) atomicOr(status, 1u);
}

} // namespace

cudaError_t validate_finite(const double* values, std::uint64_t elements,
                            std::uint32_t* status, cudaStream_t stream) {
  const auto plan = detail::make_finite_validation_plan(elements);
  if (!plan || !status || (elements && !values)) return cudaErrorInvalidValue;
  const auto cleared = cudaMemsetAsync(status, detail::finite_validation_identity,
                                       detail::finite_validation_status_bytes, stream);
  if (cleared != cudaSuccess || !plan->elements) return cleared;
  finite_status_or<<<plan->blocks, detail::finite_validation_threads, 0, stream>>>(
      values, plan->elements, status);
  return cudaGetLastError();
}

} // namespace ghb::gpu
