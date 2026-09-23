#include <cuda/std/numeric>
 int probe(int x){return cuda::std::saturating_add(x,1)+cuda::std::saturating_sub(x,1)+cuda::std::saturating_mul(x,2)+cuda::std::saturating_div(x,2)+cuda::std::saturating_cast<int>(static_cast<long long>(x));}
