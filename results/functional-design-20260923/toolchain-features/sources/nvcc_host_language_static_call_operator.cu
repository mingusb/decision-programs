struct F {  static constexpr int operator()(int x){return x+1;} };
 int probe(int x){return F{}(x);}
