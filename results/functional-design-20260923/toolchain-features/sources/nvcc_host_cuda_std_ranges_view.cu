#include <cuda/std/ranges>
 int probe(int x){const int a[]{x,2,3}; auto v=a | cuda::std::views::transform([]  (int y){return y+1;}); return v[0]+v[1]+v[2];}
