struct F { __device__ constexpr int operator()(this auto self,int x){return x<=0?0:1+self(x-1);} };
__device__ int probe(int x){return F{}(x);}
extern "C" __global__ void probe_kernel(const int* in,int* out){*out=probe(*in);}
