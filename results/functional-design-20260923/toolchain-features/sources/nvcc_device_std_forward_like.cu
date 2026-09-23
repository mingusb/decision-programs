#include <utility>
__device__ int probe(int x){return std::forward_like<const int&>(x);}
extern "C" __global__ void probe_kernel(const int* in,int* out){*out=probe(*in);}
