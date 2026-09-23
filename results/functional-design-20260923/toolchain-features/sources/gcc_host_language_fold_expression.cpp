template<class... T>  constexpr auto sum(T... xs){return (0+...+xs);}
 int probe(int x){return sum(x,2,3);}
