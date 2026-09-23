__device__ constexpr int f(int x){if consteval {return x+1;} else {return x+2;}}
static_assert(f(1)==2);
__device__ int probe(int x){return f(x);}
extern "C" __global__ void probe_kernel(const int* in,int* out){*out=probe(*in);}
