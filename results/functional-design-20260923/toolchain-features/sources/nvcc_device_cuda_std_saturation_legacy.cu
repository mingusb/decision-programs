#include <cuda/std/numeric>
__device__ int probe(int x){return cuda::std::add_sat(x,1)+cuda::std::sub_sat(x,1)+cuda::std::mul_sat(x,2)+cuda::std::div_sat(x,2)+cuda::std::saturate_cast<int>(static_cast<long long>(x));}
extern "C" __global__ void probe_kernel(const int* in,int* out){*out=probe(*in);}
