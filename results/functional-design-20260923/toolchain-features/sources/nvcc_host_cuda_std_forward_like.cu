#include <cuda/std/utility>
 int probe(int x){return cuda::std::forward_like<const int&>(x);}
