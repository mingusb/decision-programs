#include <cuda/std/expected>
 int probe(int x){const cuda::std::expected<int,int> v(x); const auto r=v.transform([]  (int y){return y+1;}).and_then([]  (int y){return cuda::std::expected<int,int>(y+2);}); return *r;}
