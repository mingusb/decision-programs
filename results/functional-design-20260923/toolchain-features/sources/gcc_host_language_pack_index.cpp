template<class... Ts>  constexpr auto first(Ts... xs){return xs...[0];}
 int probe(int x){return first(x,2,3);}
