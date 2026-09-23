#include <cuda/std/numeric>
__device__ int probe(int x){return cuda::std::saturating_add(x,1)+cuda::std::saturating_sub(x,1)+cuda::std::saturating_mul(x,2)+cuda::std::saturating_div(x,2)+cuda::std::saturating_cast<int>(static_cast<long long>(x));}
extern "C" __global__ void probe_kernel(const int* in,int* out){*out=probe(*in);}
