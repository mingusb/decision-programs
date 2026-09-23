struct F {  constexpr int operator()(this auto self,int x){return x<=0?0:1+self(x-1);} };
 int probe(int x){return F{}(x);}
