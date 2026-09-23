template<class... T> __device__ constexpr auto sum(T... xs){return (0+...+xs);}
__device__ int probe(int x){return sum(x,2,3);}
extern "C" __global__ void probe_kernel(const int* in,int* out){*out=probe(*in);}
