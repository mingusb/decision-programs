#include <functional>
__device__ int probe(int x){const auto fn=std::bind_back([] __device__ (int a,int b){return a+b;},2); return fn(x);}
extern "C" __global__ void probe_kernel(const int* in,int* out){*out=probe(*in);}
