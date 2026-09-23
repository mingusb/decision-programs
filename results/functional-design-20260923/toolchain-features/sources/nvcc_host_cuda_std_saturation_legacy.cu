#include <cuda/std/numeric>
 int probe(int x){return cuda::std::add_sat(x,1)+cuda::std::sub_sat(x,1)+cuda::std::mul_sat(x,2)+cuda::std::div_sat(x,2)+cuda::std::saturate_cast<int>(static_cast<long long>(x));}
