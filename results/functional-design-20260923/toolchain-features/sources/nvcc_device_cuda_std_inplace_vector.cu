#include <cuda/std/inplace_vector>
__device__ int probe(int x){cuda::std::inplace_vector<int,4> v; v.push_back(x); v.push_back(2); return v[0]+v[1];}
extern "C" __global__ void probe_kernel(const int* in,int* out){*out=probe(*in);}
