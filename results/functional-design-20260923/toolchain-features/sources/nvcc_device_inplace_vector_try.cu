#include <cuda/std/inplace_vector>
extern "C" __global__ void probe_kernel(const int* in,int* out){
 cuda::std::inplace_vector<int,1> v;
 const auto inserted=v.try_push_back(*in);
 const auto full=v.try_push_back(1);
 *out=inserted && !full ? v[0] : 0;
}
