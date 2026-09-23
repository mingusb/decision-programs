#include <cuda/std/numeric>
#include <cuda/std/limits>
static_assert(cuda::std::saturating_add<unsigned>(~0u,1u)==~0u);
static_assert(cuda::std::saturating_sub<unsigned>(0u,1u)==0u);
static_assert(cuda::std::saturating_mul<unsigned>(~0u,2u)==~0u);
static_assert(cuda::std::saturating_div<int>(cuda::std::numeric_limits<int>::min(),-1)==cuda::std::numeric_limits<int>::max());
static_assert(cuda::std::saturating_cast<unsigned char>(1000u)==255u);
extern "C" __global__ void probe_kernel(const unsigned* in,unsigned* out){
 const unsigned x=*in;
 *out=cuda::std::saturating_add(x,1u)^cuda::std::saturating_sub(x,1u)^cuda::std::saturating_mul(x,2u)^cuda::std::saturating_div(x,2u)^cuda::std::saturating_cast<unsigned>(static_cast<unsigned long long>(x));
}
