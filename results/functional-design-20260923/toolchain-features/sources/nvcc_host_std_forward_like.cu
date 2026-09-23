#include <utility>
 int probe(int x){return std::forward_like<const int&>(x);}
