#include <cuda/std/functional>
 int probe(int x){auto fn = []  (int y){return y+1;}; cuda::std::function_ref<int(int)> ref(fn); return ref(x);}
