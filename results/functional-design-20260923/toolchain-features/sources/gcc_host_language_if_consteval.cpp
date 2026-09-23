 constexpr int f(int x){if consteval {return x+1;} else {return x+2;}}
static_assert(f(1)==2);
 int probe(int x){return f(x);}
