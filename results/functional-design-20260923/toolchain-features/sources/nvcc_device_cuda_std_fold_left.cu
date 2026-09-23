#include <cuda/std/algorithm>
__device__ int probe(int x){const int a[]{x,2,3}; return cuda::std::ranges::fold_left(a,0,[] __device__ (int acc,int y){return acc+y;});}
extern "C" __global__ void probe_kernel(const int* in,int* out){*out=probe(*in);}
