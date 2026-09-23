#include "cache_flush.hpp"
#include <cuda_runtime.h>
#include <cstdint>

namespace {
template<bool Initialize>
__global__ void eviction_kernel(std::uint32_t* buffer, std::size_t words) {
  const std::size_t stride = std::size_t(blockDim.x) * gridDim.x;
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < words; i += stride) {
    std::uint32_t value;
    if constexpr (Initialize) {
      value = static_cast<std::uint32_t>(i) ^ 0x9e3779b9u;
    } else {
      // .cg bypasses L1 and allocates in L2. Volatile asm and the dependent
      // write prevent either memory access from being optimized away.
      asm volatile("ld.global.cg.u32 %0, [%1];"
                   : "=r"(value) : "l"(buffer + i) : "memory");
      value += 0x9e3779b9u;
    }
    asm volatile("st.global.wb.u32 [%0], %1;"
                 :: "l"(buffer + i), "r"(value) : "memory");
  }
}

template<bool Initialize>
cudaError_t launch(void* buffer, std::size_t bytes, int multiprocessors,
                   cudaStream_t stream) {
  if (!buffer || bytes == 0 || bytes % sizeof(std::uint32_t) || multiprocessors <= 0)
    return cudaErrorInvalidValue;
  eviction_kernel<Initialize><<<multiprocessors * 8, 256, 0, stream>>>(
      static_cast<std::uint32_t*>(buffer), bytes / sizeof(std::uint32_t));
  return cudaGetLastError();
}
}  // namespace

cudaError_t initialize_eviction_buffer(void* buffer, std::size_t bytes,
                                      int multiprocessors, cudaStream_t stream) {
  return launch<true>(buffer, bytes, multiprocessors, stream);
}

cudaError_t evict_l2(void* buffer, std::size_t bytes,
                     int multiprocessors, cudaStream_t stream) {
  return launch<false>(buffer, bytes, multiprocessors, stream);
}
