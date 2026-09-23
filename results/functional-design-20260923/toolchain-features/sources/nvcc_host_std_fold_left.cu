#include <algorithm>
 int probe(int x){const int a[]{x,2,3}; return std::ranges::fold_left(a,0,[]  (int acc,int y){return acc+y;});}
