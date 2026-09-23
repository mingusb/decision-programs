#include <functional>
 int probe(int x){const auto fn=std::bind_back([]  (int a,int b){return a+b;},2); return fn(x);}
