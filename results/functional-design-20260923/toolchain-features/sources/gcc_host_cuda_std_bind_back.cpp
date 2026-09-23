#include <cuda/std/functional>
 int probe(int x){const auto fn=cuda::std::bind_back([]  (int a,int b){return a+b;},2); return fn(x);}
