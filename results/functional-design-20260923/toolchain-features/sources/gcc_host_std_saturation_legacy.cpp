#include <numeric>
 int probe(int x){return std::add_sat(x,1)+std::sub_sat(x,1)+std::mul_sat(x,2)+std::div_sat(x,2)+std::saturate_cast<int>(static_cast<long long>(x));}
