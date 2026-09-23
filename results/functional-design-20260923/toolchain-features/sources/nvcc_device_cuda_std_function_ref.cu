#include <cuda/std/functional>
__device__ int probe(int x){auto fn = [] __device__ (int y){return y+1;}; cuda::std::function_ref<int(int)> ref(fn); return ref(x);}
extern "C" __global__ void probe_kernel(const int* in,int* out){*out=probe(*in);}
