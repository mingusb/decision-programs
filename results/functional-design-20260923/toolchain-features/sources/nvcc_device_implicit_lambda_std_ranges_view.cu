#include <ranges>
__device__ int probe(int x){const int a[]{x,2,3}; auto v=a | std::views::transform([] (int y){return y+1;}); return v[0]+v[1]+v[2];}
extern "C" __global__ void probe_kernel(const int* in,int* out){*out=probe(*in);}
