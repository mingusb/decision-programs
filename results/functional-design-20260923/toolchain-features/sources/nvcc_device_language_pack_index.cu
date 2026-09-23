template<class... Ts> __device__ constexpr auto first(Ts... xs){return xs...[0];}
__device__ int probe(int x){return first(x,2,3);}
extern "C" __global__ void probe_kernel(const int* in,int* out){*out=probe(*in);}
