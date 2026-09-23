#include <cuda/std/expected>
__device__ int probe(int x){const cuda::std::expected<int,int> v(x); const auto r=v.transform([] (int y){return y+1;}).and_then([] (int y){return cuda::std::expected<int,int>(y+2);}); return *r;}
extern "C" __global__ void probe_kernel(const int* in,int* out){*out=probe(*in);}
