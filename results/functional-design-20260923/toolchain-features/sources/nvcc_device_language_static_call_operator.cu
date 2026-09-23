struct F { __device__ static constexpr int operator()(int x){return x+1;} };
__device__ int probe(int x){return F{}(x);}
extern "C" __global__ void probe_kernel(const int* in,int* out){*out=probe(*in);}
