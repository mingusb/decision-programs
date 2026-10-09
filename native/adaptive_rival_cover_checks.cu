// CUDA checks for class-specific complete covers. Numeric gate qualification
// remains a test-fixture premise; this file grants no native RuntimeGate.
#include "class_conversion/adaptive_cover_proof.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace a=class_conversion_adaptive;
using U=std::uint32_t;
struct Report {U checks=0,failures=0,first_failure=0;};
__device__ void expect(Report& r,bool ok,U code){++r.checks;if(!ok){++r.failures;if(!r.first_failure)r.first_failure=code;}}
struct Fixture {
  static constexpr U N=128,F=65,K=7,T=16;
  int feature[N],left[N],right[N],roots[T],channels[T],feature_group[F],feature_numeric[F],residual[T];
  float cut[N],value[N],bias[K],range_lower[K],range_upper[K];
  unsigned char missing[N];
  U minimum[N],maximum[N],lower[F],upper[F],missing_allowed[F],feature_bit[F];
  U group_word_offsets[2],group_feature_offsets[2],group_widths[1],group_features[F];
  U words[K],positions[K],scratch[512];
  a::u64 initial_masks[2],allowed[2];
  a::State state[1];
  a::EngineView e;
  a::domain::RegionView region;
  U nodes=0,trees=0,features=0,classes=0,group_width=0;
  __device__ void init(U f=1,bool nan=true,U group=0,U k=2) {
    nodes=trees=0;features=f;classes=k;group_width=group;e={};state[0]={};
    for(U i=0;i<F;++i){feature_group[i]=group?0:-1;feature_numeric[i]=group?-1:int(i);feature_bit[i]=i;group_features[i]=i;lower[i]=a::domain::finite_min_key;upper[i]=a::domain::finite_max_key;missing_allowed[i]=nan;}
    for(U i=0;i<K;++i){bias[i]=0;words[i]=positions[i]=0;range_lower[i]=range_upper[i]=0;}
    for(U i=0;i<T;++i)residual[i]=-1;
    group_word_offsets[0]=group_feature_offsets[0]=0;
    group_word_offsets[1]=(group+63)/64;group_feature_offsets[1]=group;group_widths[0]=group;
    initial_masks[0]=allowed[0]=group>=64?UINT64_MAX:(group?(a::u64(1)<<group)-1:0);
    initial_masks[1]=allowed[1]=group>64?(a::u64(1)<<(group-64))-1:0;
    region={lower,upper,missing_allowed,allowed};
  }
  __device__ U leaf(float x){const U i=nodes++;feature[i]=-1;left[i]=right[i]=-1;cut[i]=0;value[i]=x;missing[i]=0;return i;}
  __device__ U stump(U f,float threshold,float low,float high,bool nan_left=true){
    const U i=nodes++;feature[i]=int(f);cut[i]=threshold;value[i]=0;missing[i]=nan_left;
    left[i]=int(leaf(low));right[i]=int(leaf(high));return i;
  }
  __device__ U mismatch(U a,U b,float threshold){
    const U i=nodes++;feature[i]=int(a);cut[i]=threshold;value[i]=0;missing[i]=1;
    left[i]=int(stump(b,threshold,0,1));right[i]=int(stump(b,threshold,1,0));return i;
  }
  __device__ U parity_four(){
    const U start=nodes;nodes+=31;
    for(U i=0;i<31;++i){const U node=start+i;cut[node]=0;missing[node]=1;
      if(i<15){feature[node]=int(31-__clz(i+1));left[node]=int(start+2*i+1);right[node]=int(start+2*i+2);value[node]=0;}
      else {feature[node]=-1;left[node]=right[node]=-1;value[node]=float(__popc(i-15)%2);}
    }return start;
  }
  __device__ void tree(U root,U c=0){roots[trees]=residual[trees]=int(root);channels[trees++]=int(c);}
  __device__ void finish(float first_bias=1.f){
    bias[0]=first_bias;
    for(U c=0;c<classes;++c)words[c]=__float_as_uint(bias[c]);
    for(U i=nodes;i-->0;){
      if(left[i]<0)minimum[i]=maximum[i]=__float_as_uint(value[i]);
      else {minimum[i]=__float_as_uint(fminf(__uint_as_float(minimum[left[i]]),__uint_as_float(minimum[right[i]])));maximum[i]=__float_as_uint(fmaxf(__uint_as_float(maximum[left[i]]),__uint_as_float(maximum[right[i]])));}
    }
    e.source={feature,left,right,roots,channels,cut,value,bias,missing,features,classes,nodes,trees,0};
    e.domain={features,group_width?1u:0u,(group_width+63)/64,missing_allowed[0],feature_group,feature_bit,group_word_offsets,group_feature_offsets,group_widths,group_features,initial_masks,group_width?0u:features,feature_numeric};
    e.arena.states=state;e.arena.state_capacity=1;e.arena.words=words;e.arena.positions=positions;e.arena.residual=residual;
    e.minimum=minimum;e.maximum=maximum;e.range_lower=range_lower;e.range_upper=range_upper;
    e.qualified_gap=true;e.joint_pair_visit_budget=UINT32_MAX;
  }
};
__device__ float evaluate_leaf(const Fixture& f,U root,const float* row){
  while(f.left[root]>=0){const auto x=row[f.feature[root]];const bool l=isnan(x)?bool(f.missing[root]):x<f.cut[root];root=U(l?f.left[root]:f.right[root]);}return f.value[root];
}
__device__ bool numeric_inside(const Fixture& f,U feature,float x){
  if(isnan(x))return f.missing_allowed[feature];
  const U key=a::domain::sortable_word(__float_as_uint(x));return f.lower[feature]<=key&&key<=f.upper[feature];
}

void ck(cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
void finish(){ck(cudaGetLastError());ck(cudaDeviceSynchronize());}


namespace cp=a::cover;
// Correctness fixtures share one device body instead of repeatedly specializing
// the complete proof portfolio. The timing kernel below calls production code.
__device__ __noinline__ cp::Result fixture_portfolio_label(a::EngineView e,U id,
    a::domain::RegionView original,U predicate,a::domain::RegionView private_region,
    U* private_stack,U stack_capacity,U global_visit_budget) {
  return cp::portfolio_label(e,id,original,predicate,private_region,private_stack,
                             stack_capacity,global_visit_budget);
}
__device__ __noinline__ cp::Result fixture_rival_cover(a::EngineView e,U id,
    a::domain::RegionView original,a::domain::RegionView private_region,U* stack,
    U capacity,U budget,U winner,U rival) {
  return cp::detail::rival_cover(e,id,original,private_region,stack,capacity,budget,winner,rival);
}
// Effort correctness checks also share one compiled device body.
__device__ __noinline__ a::effort::Result fixture_effort_label(a::EngineView e,U id,
    a::domain::RegionView region,U* stack,U capacity,U budget,U winner=a::none,U rival=a::none) {
  return a::effort::interval_label(e,id,region,stack,capacity,budget,winner,rival);
}
struct RivalScratch {
  U lower[Fixture::F],upper[Fixture::F],missing[Fixture::F],stack[256];
  a::u64 allowed[2];
  float range_lower[Fixture::K],range_upper[Fixture::K];
  __device__ a::domain::RegionView region(){return {lower,upper,missing,allowed};}
};
struct Snapshot {
  U words[Fixture::K],positions[Fixture::K],lower[Fixture::F],upper[Fixture::F],missing[Fixture::F];
  int residual[Fixture::T];a::u64 allowed[2];a::State state;
  __device__ void save(const Fixture& f){
    for(U c=0;c<f.classes;++c){words[c]=f.words[c];positions[c]=f.positions[c];}
    for(U t=0;t<f.trees;++t)residual[t]=f.residual[t];
    for(U n=0;n<f.e.domain.numeric_features;++n){lower[n]=f.lower[n];upper[n]=f.upper[n];missing[n]=f.missing_allowed[n];}
    for(U w=0;w<f.e.domain.mask_words;++w)allowed[w]=f.allowed[w];
    state=f.state[0];
  }
  __device__ bool unchanged(const Fixture& f)const{
    for(U c=0;c<f.classes;++c)if(words[c]!=f.words[c]||positions[c]!=f.positions[c])return false;
    for(U t=0;t<f.trees;++t)if(residual[t]!=f.residual[t])return false;
    for(U n=0;n<f.e.domain.numeric_features;++n)if(lower[n]!=f.lower[n]||upper[n]!=f.upper[n]||missing[n]!=f.missing_allowed[n])return false;
    for(U w=0;w<f.e.domain.mask_words;++w)if(allowed[w]!=f.allowed[w])return false;
    const auto&s=f.state[0];
    return state.phase==s.phase&&state.predicate==s.predicate&&state.left==s.left&&state.right==s.right&&state.node==s.node&&state.reserved==s.reserved&&state.hash==s.hash;
  }
};
// Scores: [2a,2b,2+a+b]. The true winner is always class 2. A common
// one-predicate cover cannot establish this from independent tree ranges,
// but the a-cover proves class 2 beats class 0 and the b-cover beats class 1.
__device__ void setup_rivals(Fixture& f,bool grouped=false,bool opposite_missing=false){
  f.init(grouped?65:2,true,grouped?65:0,3);f.bias[2]=2;
  const U b=grouped?64:1;const float cut=grouped?.5f:0.f;
  f.tree(f.stump(0,cut,0,1,true),2);f.tree(f.stump(0,cut,0,2,!opposite_missing),0);
  f.tree(f.stump(b,cut,0,1,true),2);f.tree(f.stump(b,cut,0,2,true),1);
  f.finish(0);f.e.joint_pair_visit_budget=0;f.e.rival_cover_enabled=true;
}
__device__ int class_on_row(const Fixture& f,const float* row){
  float sums[Fixture::K];for(U c=0;c<f.classes;++c)sums[c]=f.bias[c];
  for(U t=0;t<f.trees;++t){const U c=U(f.channels[t]);sums[c]=__fadd_rn(sums[c],evaluate_leaf(f,U(f.roots[t]),row));}
  U best=0;for(U c=1;c<f.classes;++c)if(sums[c]>sums[best])best=c;return int(best);
}
__device__ void accepted_matches_all_signatures(const Fixture& f,const cp::Result& result,Report& report,U code){
  if(!result.success)return;
  const float atoms[5]={-1.f,-0.f,0.f,1.f,__uint_as_float(0x7fc00001u)};
  float row[Fixture::F];for(U c=0;c<f.features;++c)row[c]=0;
  U count=f.group_width;
  if(!f.group_width){if(f.features>4){expect(report,false,code);return;}count=1;for(U ftr=0;ftr<f.features;++ftr)count*=5;}
  for(U i=0;i<count;++i){
    if(f.group_width){for(U j=0;j<f.features;++j)row[j]=float(i==j);if(!(f.allowed[i/64]&(a::u64(1)<<(i%64))))continue;}
    else {U index=i;bool inside=true;for(U ftr=0;ftr<f.features;++ftr){row[ftr]=atoms[index%5];index/=5;inside&=numeric_inside(f,ftr,row[ftr]);}if(!inside)continue;}
    expect(report,class_on_row(f,row)==result.label,code);
  }
}
// A frustrated three-factor cycle. The rival score is 0 or 2, whereas
// independent/adjacent-pair bounds retain 3 after any one-axis cover. Two
// adaptive proof levels suffice; no input observation is used for acceptance.
__device__ void setup_triangle(Fixture&f,bool grouped=false,float winner=2.5f){
  f.init(grouped?65:3,true,grouped?65:0,2);
  const U x=0,y=1,z=grouped?64:2;const float cut=grouped?.5f:0.f;
  f.tree(f.mismatch(x,y,cut),1);f.tree(f.mismatch(y,z,cut),1);f.tree(f.mismatch(z,x,cut),1);
  f.finish(winner);f.e.rival_cover_enabled=true;
}
__device__ void deeper_cover_checks(Report&r,Fixture&f,RivalScratch&scratch,Snapshot&saved){
  setup_triangle(f);saved.save(f);auto e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  const auto ordinary=a::conditioned_subtree_extrema(e,f.region,U(f.roots[0]),scratch.stack,256,1024);
  const auto metadata=a::conditioned_subtree_extrema(e,f.region,U(f.roots[0]),scratch.stack,256,1024,false,true);
  expect(r,ordinary.first_unforced==a::none&&metadata.first_unforced==U(f.roots[0]),100);
  expect(r,ordinary.minimum==metadata.minimum&&ordinary.maximum==metadata.maximum&&ordinary.visited==metadata.visited,101);
  const auto direct=fixture_effort_label(e,0,f.region,scratch.stack,256,4096,0,1);
  expect(r,direct.label<0&&direct.selected_predicate==U(f.roots[0]),102);
  for(U i=0;i<f.trees;++i){
    const auto common=cp::interval_label(e,0,f.region,U(f.roots[i]),scratch.region(),scratch.stack,256,4096);
    expect(r,!common.success,103);
  }
  auto result=fixture_portfolio_label(e,0,f.region,U(f.roots[0]),scratch.region(),scratch.stack,256,4096);
  expect(r,result.success&&result.label==0&&result.visited<=4096&&saved.unchanged(f),104);
  accepted_matches_all_signatures(f,result,r,105);
  // Every prefix of the work allowance and every small stack must refuse safely
  // or prove all branches. They may never turn a partial cover into acceptance.
  for(U budget=0;budget<=384;++budget){
    const auto q=fixture_portfolio_label(e,0,f.region,U(f.roots[0]),scratch.region(),scratch.stack,256,budget);
    expect(r,q.visited<=budget&&saved.unchanged(f),106);accepted_matches_all_signatures(f,q,r,107);
  }
  for(U capacity=0;capacity<=24;++capacity){
    const auto q=fixture_portfolio_label(e,0,f.region,U(f.roots[0]),scratch.region(),scratch.stack,capacity,4096);
    expect(r,q.visited<=4096&&saved.unchanged(f),108);accepted_matches_all_signatures(f,q,r,109);
  }
  // Equal four-axis parity factors in different channels require four proof
  // levels with marginal bounds. This guards against a hidden depth-two limit.
  f.init(4,true,0,2);f.tree(f.parity_four(),0);f.tree(f.parity_four(),1);f.finish(.5f);f.e.rival_cover_enabled=true;
  saved.save(f);e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  result=fixture_portfolio_label(e,0,f.region,U(f.roots[0]),scratch.region(),scratch.stack,256,4096);
  expect(r,result.success&&result.label==0&&result.certified_cases==16&&result.feasible_cases>=31&&saved.unchanged(f),127);
  accepted_matches_all_signatures(f,result,r,128);
  // Statically defeated later classes do not divide away the hard rival's work.
  setup_triangle(f);e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  const auto two_classes=fixture_portfolio_label(e,0,f.region,U(f.roots[0]),scratch.region(),scratch.stack,256,512);
  f.classes=Fixture::K;f.e.source.classes=Fixture::K;
  for(U c=2;c<Fixture::K;++c){f.bias[c]=-5;f.words[c]=__float_as_uint(-5.f);}
  saved.save(f);e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  const auto seven_classes=fixture_portfolio_label(e,0,f.region,U(f.roots[0]),scratch.region(),scratch.stack,256,512);
  expect(r,two_classes.success&&seven_classes.success&&two_classes.visited==seven_classes.visited&&saved.unchanged(f),129);
  // A zero-width factor cannot yield a useful conditioned split proposal.
  f.init(1,false,0,2);f.bias[1]=1;f.tree(f.stump(0,0,1,1));f.finish(0);
  e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  const auto constant=fixture_effort_label(e,0,f.region,scratch.stack,256,4096,0,1);
  expect(r,constant.label<0&&constant.selected_predicate==a::none,130);
  // The endpoint predicts class 0 but another feasible input predicts class 1.
  // In particular, pending right branches cannot be discarded at unwind.
  setup_triangle(f,false,1.5f);saved.save(f);e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  result=fixture_portfolio_label(e,0,f.region,U(f.roots[0]),scratch.region(),scratch.stack,256,4096);
  expect(r,!result.success&&result.visited<=4096&&saved.unchanged(f),110);
  const float mixed[3]={1,-1,-1};expect(r,class_on_row(f,mixed)==1,111);
  // Wide exactly-one masks and the NaN-only numeric domain use the same domain
  // partition operations as the constructor, with no scalar-mask shortcut.
  setup_triangle(f,true);saved.save(f);e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  result=fixture_portfolio_label(e,0,f.region,U(f.roots[0]),scratch.region(),scratch.stack,256,4096);
  expect(r,result.success&&result.label==0&&saved.unchanged(f),112);accepted_matches_all_signatures(f,result,r,113);
  f.allowed[0]=0;f.allowed[1]=1;saved.save(f);
  result=fixture_portfolio_label(e,0,f.region,U(f.roots[0]),scratch.region(),scratch.stack,256,4096);
  expect(r,result.success&&result.label==0&&saved.unchanged(f),114);accepted_matches_all_signatures(f,result,r,115);
  setup_triangle(f);for(U n=0;n<3;++n){f.lower[n]=a::domain::finite_min_key;f.upper[n]=a::domain::finite_min_key-1;f.missing_allowed[n]=1;}
  saved.save(f);e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  result=fixture_portfolio_label(e,0,f.region,U(f.roots[0]),scratch.region(),scratch.stack,256,4096);
  expect(r,result.success&&result.label==0&&saved.unchanged(f),116);accepted_matches_all_signatures(f,result,r,117);
  // Candidate evaluation preserves RN32 source order and the saved consumed
  // prefix: reassociating these operands would propose the other class.
  f.init(1,false,0,2);f.bias[1]=.5f;f.tree(f.leaf(100000000.f));f.tree(f.leaf(1.f));f.tree(f.leaf(-100000000.f));f.finish(0);
  e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;saved.save(f);U used=0,candidate=a::none;
  bool proposed=cp::detail::propose_class(e,0,f.region,64,used,candidate);
  expect(r,proposed&&candidate==1&&used==3&&e.range_lower[0]==0&&saved.unchanged(f),118);
  f.residual[0]=-1;f.words[0]=__float_as_uint(100000000.f);saved.save(f);
  proposed=cp::detail::propose_class(e,0,f.region,64,used,candidate);
  expect(r,proposed&&candidate==1&&used==2&&e.range_lower[0]==0&&saved.unchanged(f),119);
  // Missing proposal-only input data, malformed channels and nonfinite scores
  // cannot manufacture a candidate or modify the admitted state.
  e.source.value=nullptr;proposed=cp::detail::propose_class(e,0,f.region,64,used,candidate);
  expect(r,!proposed&&candidate==a::none&&saved.unchanged(f),120);e.source.value=f.value;
  f.channels[1]=int(f.classes);proposed=cp::detail::propose_class(e,0,f.region,64,used,candidate);
  expect(r,!proposed&&candidate==a::none&&saved.unchanged(f),121);f.channels[1]=0;
  f.words[0]=0x7f800000u;saved.save(f);proposed=cp::detail::propose_class(e,0,f.region,64,used,candidate);
  expect(r,!proposed&&candidate==a::none&&saved.unchanged(f),122);
  setup_triangle(f);e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  f.feature_numeric[0]=int(Fixture::F);saved.save(f);
  proposed=cp::detail::propose_class(e,0,f.region,64,used,candidate);
  expect(r,!proposed&&candidate==a::none&&saved.unchanged(f),123);
  setup_triangle(f,true);e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  f.allowed[0]=f.allowed[1]=0;saved.save(f);proposed=cp::detail::propose_class(e,0,f.region,64,used,candidate);
  expect(r,!proposed&&candidate==a::none&&saved.unchanged(f),124);
  f.init(1,false,0,2);f.tree(f.leaf(3.e38f));f.finish(3.e38f);saved.save(f);
  e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  proposed=cp::detail::propose_class(e,0,f.region,64,used,candidate);
  expect(r,!proposed&&candidate==a::none&&used==1&&saved.unchanged(f),125);
  setup_triangle(f);f.right[f.roots[0]]=int(f.nodes);saved.save(f);
  e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  proposed=cp::detail::propose_class(e,0,f.region,64,used,candidate);
  expect(r,!proposed&&candidate==a::none&&saved.unchanged(f),126);
}
// Interleaved source channels exercise the same ordered folds with a seven-
// class model; two large irrelevant factors are retained at their static bounds.
__device__ void setup_selective(Fixture& f,bool unresolved_other=false){
  f.init(4,false,0,7);for(U c=2;c<7;++c)f.bias[c]=-5.f;
  f.tree(f.parity_four(),2);f.tree(f.stump(0,0,1,2),0);f.tree(f.stump(0,0,0,3),1);
  f.tree(f.parity_four(),6);f.tree(f.stump(0,0,1,2),0);f.tree(f.stump(0,0,0,3),1);
  if(unresolved_other){f.bias[2]=0;for(U i=0;i<31;++i)if(f.left[i]<0)f.value[i]*=3.f;}
  f.finish(0);f.upper[0]=a::domain::sortable_word(__float_as_uint(0.f))-1;
}
__device__ __noinline__ void effort_encloses_signatures(const Fixture& f,const a::EngineView& e,
    const a::effort::Result& result,Report& r,U code){
  float row[Fixture::F],sums[Fixture::K];for(U i=0;i<f.features;++i)row[i]=0;
  // These fixtures use four finite axes and the single threshold zero.
  for(U bits=0;bits<16;++bits){
    bool inside=true;for(U i=0;i<4;++i){row[i]=(bits&(1u<<i))?1.f:-1.f;inside&=numeric_inside(f,i,row[i]);}
    if(!inside)continue;
    for(U c=0;c<f.classes;++c)sums[c]=__uint_as_float(f.words[c]);
    for(U t=0;t<f.trees;++t)if(f.residual[t]>=0){const U c=U(f.channels[t]);sums[c]=__fadd_rn(sums[c],evaluate_leaf(f,U(f.residual[t]),row));}
    U best=0;for(U c=0;c<f.classes;++c){expect(r,e.range_lower[c]<=sums[c]&&sums[c]<=e.range_upper[c],code);if(sums[c]>sums[best])best=c;}
    if(result.label>=0)expect(r,result.label==int(best),code);
  }
}
__device__ __noinline__ void selective_effort_checks(Report&r,Fixture&f,RivalScratch&scratch,Snapshot&saved){
  setup_selective(f);saved.save(f);auto e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  expect(r,a::qualified_interval_label(e,0)<0,200);
  float static_lo[Fixture::K],static_hi[Fixture::K];
  for(U c=0;c<f.classes;++c){static_lo[c]=e.range_lower[c];static_hi[c]=e.range_upper[c];}
  const auto all=fixture_effort_label(e,0,f.region,scratch.stack,256,4096);
  const float all_lo0=e.range_lower[0],all_hi0=e.range_upper[0],all_lo1=e.range_lower[1],all_hi1=e.range_upper[1];
  const auto selected=fixture_effort_label(e,0,f.region,scratch.stack,256,4096,0,1);
  expect(r,all.label==0&&selected.label==0&&selected.visited<all.visited&&selected.refined_roots==4&&saved.unchanged(f),201);
  expect(r,e.range_lower[0]==all_lo0&&e.range_upper[0]==all_hi0&&e.range_lower[1]==all_lo1&&e.range_upper[1]==all_hi1,202);
  for(U c=2;c<f.classes;++c)expect(r,__float_as_uint(e.range_lower[c])==__float_as_uint(static_lo[c])&&__float_as_uint(e.range_upper[c])==__float_as_uint(static_hi[c]),203);
  effort_encloses_signatures(f,e,selected,r,204);
  for(U budget=0;budget<=96;++budget){
    const auto q=fixture_effort_label(e,0,f.region,scratch.stack,256,budget,0,1);
    expect(r,q.visited<=budget&&saved.unchanged(f),205);effort_encloses_signatures(f,e,q,r,206);
  }
  // Another class keeps the global label unknown, but the requested pair is
  // already settled. Its proof need not spend an adjacent-pair allowance.
  setup_selective(f,true);saved.save(f);e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  const auto pair=fixture_effort_label(e,0,f.region,scratch.stack,256,4096,0,1);
  expect(r,pair.label<0&&!pair.additional_prune&&pair.pair_attempts==0&&cp::detail::beats(e,0,1)&&saved.unchanged(f),207);
  effort_encloses_signatures(f,e,pair,r,208);
  // Both optional whole-class families retain the full metadata/traversal path.
  e.relational_bounds_enabled=true;
  const auto rel=fixture_effort_label(e,0,f.region,scratch.stack,256,4096,0,1);
  expect(r,rel.refined_roots==f.trees&&rel.refined_roots>pair.refined_roots&&rel.visited<=4096,209);
  effort_encloses_signatures(f,e,rel,r,210);e.relational_bounds_enabled=false;
  U axes[Fixture::T],minimum[Fixture::T],maximum[Fixture::T],order[Fixture::T],cuts[128];double first[Fixture::T];
  e.unary_bounds_enabled=true;e.unary_axes=axes;e.unary_minimum=minimum;e.unary_maximum=maximum;e.unary_order=order;
  e.unary_cuts=cuts;e.unary_cut_capacity=128;e.unary_first=first;e.unary_first_capacity=Fixture::T;
  const auto unary=fixture_effort_label(e,0,f.region,scratch.stack,256,4096,0,1);
  expect(r,unary.refined_roots==f.trees&&unary.refined_roots>pair.refined_roots&&unary.visited<=4096,211);
  effort_encloses_signatures(f,e,unary,r,212);
  // A non-target static endpoint outside the native range disables selective
  // work. Full conditioning can still recover a qualified interval afterward.
  setup_selective(f);f.tree(f.stump(0,0,0,20),3);f.finish(0);saved.save(f);
  e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  a::qualified_interval_label(e,0);expect(r,e.range_upper[3]>10.f,213);
  const auto full=fixture_effort_label(e,0,f.region,scratch.stack,256,4096,0,1);
  expect(r,full.refined_roots==f.trees&&full.label==0&&e.range_upper[3]<=10.f&&saved.unchanged(f),214);
  effort_encloses_signatures(f,e,full,r,215);
  // Unrecoverable range/nonfinite premises never grant a native certificate.
  f.words[3]=__float_as_uint(20.f);saved.save(f);
  const auto outside=fixture_effort_label(e,0,f.region,scratch.stack,256,4096,0,1);
  expect(r,outside.label<0&&!cp::detail::beats(e,0,1)&&saved.unchanged(f),216);
  f.words[3]=0x7fc00001u;saved.save(f);
  const auto nan=fixture_effort_label(e,0,f.region,scratch.stack,256,4096,0,1);
  expect(r,nan.label<0&&!cp::detail::beats(e,0,1)&&saved.unchanged(f),217);
  e.qualified_gap=false;const auto absent=fixture_effort_label(e,0,f.region,scratch.stack,256,4096,0,1);
  expect(r,absent.label<0&&!absent.attempted&&!absent.visited&&saved.unchanged(f),218);
  // Two certified opposite branch labels already prove the original box mixed.
  // No proposed winner or deeper uniform-class search can succeed there.
  f.init(1,false,0,2);f.bias[1]=1;f.tree(f.stump(0,0,2,0),0);f.finish(0);f.e.rival_cover_enabled=true;
  saved.save(f);e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  const auto mixed_common=cp::interval_label(e,0,f.region,U(f.roots[0]),scratch.region(),scratch.stack,256,4096);
  const auto mixed_portfolio=fixture_portfolio_label(e,0,f.region,U(f.roots[0]),scratch.region(),scratch.stack,256,4096);
  expect(r,!mixed_common.success&&mixed_common.rejection==cp::Rejection::differing_labels,219);
  expect(r,!mixed_portfolio.success&&mixed_portfolio.rejection==cp::Rejection::differing_labels&&mixed_portfolio.visited==mixed_common.visited&&!mixed_portfolio.rival_covers&&saved.unchanged(f),220);
}
__device__ __noinline__ void cover_cost_shortcut_checks(Report&r,Fixture&f,RivalScratch&scratch,Snapshot&saved){
  setup_triangle(f);saved.save(f);auto e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  const auto first_failure=cp::interval_label(e,0,f.region,U(f.roots[0]),scratch.region(),scratch.stack,256,4096);
  expect(r,!first_failure.success&&first_failure.rejection==cp::Rejection::uncertified_case&&first_failure.feasible_cases==1&&first_failure.certified_cases==0,230);
  expect(r,first_failure.branch_labels[0]<0&&first_failure.branch_labels[1]<0&&saved.unchanged(f),231);
  // Bit 63 is a cached static certificate at 64 classes; larger class counts
  // exercise the generic fallback with an actual unresolved rival at 64/65.
  constexpr U C=66;float bias[C],lo[C],hi[C],scores[C];U words[C],positions[C];
  for(U classes=64;classes<=66;++classes){
    setup_triangle(f);const U rival=classes==64?1:classes-1;
    for(U t=0;t<f.trees;++t)f.channels[t]=int(rival);
    for(U c=0;c<classes;++c){bias[c]=c==0?2.5f:(c==rival?0.f:-5.f);words[c]=__float_as_uint(bias[c]);positions[c]=0;}
    e=f.e;e.source.classes=classes;e.source.bias=bias;e.arena.words=words;e.arena.positions=positions;e.range_lower=lo;e.range_upper=hi;
    saved.save(f);const auto q=fixture_portfolio_label(e,0,f.region,U(f.roots[0]),scratch.region(),scratch.stack,256,4096);
    expect(r,q.success&&q.label==0&&q.rivals_certified==classes-1&&q.rival_covers==1&&q.visited<=4096&&saved.unchanged(f),232);
    for(U c=0;c<classes;++c)expect(r,words[c]==__float_as_uint(bias[c])&&positions[c]==0,233);
    float row[3];for(U bits=0;bits<8;++bits){
      for(U i=0;i<3;++i)row[i]=(bits&(1u<<i))?1.f:-1.f;
      for(U c=0;c<classes;++c)scores[c]=bias[c];
      for(U t=0;t<f.trees;++t){const U c=U(f.channels[t]);scores[c]=__fadd_rn(scores[c],evaluate_leaf(f,U(f.roots[t]),row));}
      U best=0;for(U c=1;c<classes;++c)if(scores[c]>scores[best])best=c;expect(r,int(best)==q.label,234);
    }
  }
}
__device__ __noinline__ void negative_target_checks(Report&r,Fixture&f,RivalScratch&scratch,Snapshot&saved){
  // A tie has one native argmax class in this fixture, but no positive target
  // gap. Refusing the attempted gap is not a claim that the region is mixed.
  f.init(1,false,0,2);f.tree(f.leaf(1),0);f.tree(f.leaf(1),1);f.finish(0);saved.save(f);
  auto e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  auto q=fixture_rival_cover(e,0,f.region,scratch.region(),scratch.stack,256,4096,0,1);
  const float row[1]={0};
  expect(r,!q.success&&q.rejection==cp::Rejection::target_gap_refuted&&q.feasible_cases==1&&q.certified_cases==0,240);
  expect(r,class_on_row(f,row)==0&&saved.unchanged(f),241);
  // A qualified competing whole-class certificate has its own diagnostic.
  f.init(1,false,0,2);f.tree(f.leaf(0),0);f.tree(f.leaf(1),1);f.finish(0);saved.save(f);
  e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  q=fixture_rival_cover(e,0,f.region,scratch.region(),scratch.stack,256,4096,0,1);
  expect(r,!q.success&&q.rejection==cp::Rejection::candidate_refuted&&class_on_row(f,row)==1&&saved.unchanged(f),242);
  // A third channel prevents a whole-class decision, yet the finite target
  // enclosure alone refutes winner 0 against rival 1 without any split.
  f.init(1,false,0,3);f.tree(f.leaf(0),0);f.tree(f.stump(0,0,0,2),2);f.tree(f.leaf(1),1);f.finish(0);saved.save(f);
  e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  expect(r,a::qualified_interval_label(e,0)<0,243);
  q=fixture_rival_cover(e,0,f.region,scratch.region(),scratch.stack,256,4096,0,1);
  expect(r,!q.success&&q.rejection==cp::Rejection::target_gap_refuted&&q.feasible_cases==1&&q.certified_cases==0&&saved.unchanged(f),244);
  // Nonfinite endpoints do not constitute an opposing interval certificate.
  f.words[0]=0x7fc00001u;saved.save(f);
  q=fixture_rival_cover(e,0,f.region,scratch.region(),scratch.stack,256,4096,0,1);
  expect(r,!q.success&&q.rejection==cp::Rejection::uncertified_case&&saved.unchanged(f),245);
}
struct RivalReport {Report checked;U common_visits=0,portfolio_visits=0,covers=0,certified=0,naive_partition_cases=4,separate_cases=4;};
__global__ void rival_checks(RivalReport* output){
  if(blockIdx.x||threadIdx.x)return;RivalReport observed;auto&r=observed.checked;Fixture f;RivalScratch scratch;Snapshot saved;
  setup_rivals(f);saved.save(f);
  auto e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  expect(r,a::qualified_interval_label(e,0)==-1,1);
  auto common=cp::interval_label(e,0,f.region,U(f.roots[0]),scratch.region(),scratch.stack,256,512);
  auto result=fixture_portfolio_label(e,0,f.region,U(f.roots[0]),scratch.region(),scratch.stack,256,512);
  expect(r,!common.success&&common.rejection==cp::Rejection::uncertified_case,2);
  expect(r,result.success&&result.label==2&&result.rival_covers==2&&result.rivals_certified==2,3);
  expect(r,result.visited<=512&&saved.unchanged(f),4);accepted_matches_all_signatures(f,result,r,5);
  observed.common_visits=common.visited;observed.portfolio_visits=result.visited;observed.covers=result.rival_covers;observed.certified=result.rivals_certified;
  e.rival_cover_enabled=false;auto disabled=fixture_portfolio_label(e,0,f.region,U(f.roots[0]),scratch.region(),scratch.stack,256,512);
  expect(r,!disabled.success&&!disabled.rival_covers,6);e.rival_cover_enabled=true;
  // Every bounded attempt may refuse, but may never manufacture a wrong label.
  for(U budget=0;budget<=128;++budget){
    const auto q=fixture_portfolio_label(e,0,f.region,U(f.roots[0]),scratch.region(),scratch.stack,256,budget);
    expect(r,q.visited<=budget&&saved.unchanged(f),10);
    accepted_matches_all_signatures(f,q,r,11);
    if(!budget)expect(r,!q.success&&q.rejection==cp::Rejection::budget_disabled,12);
  }
  for(U capacity=0;capacity<=4;++capacity){
    const auto q=fixture_portfolio_label(e,0,f.region,U(f.roots[0]),scratch.region(),scratch.stack,capacity,512);
    expect(r,q.visited<=512&&saved.unchanged(f),13);accepted_matches_all_signatures(f,q,r,14);
    if(!capacity)expect(r,!q.success,15);
  }
  auto alias=fixture_portfolio_label(e,0,f.region,U(f.roots[0]),f.region,scratch.stack,256,512);
  expect(r,!alias.success&&alias.rejection==cp::Rejection::invalid_scratch&&saved.unchanged(f),16);
  e.qualified_gap=false;auto gated=fixture_portfolio_label(e,0,f.region,U(f.roots[0]),scratch.region(),scratch.stack,256,512);
  expect(r,!gated.success&&!gated.attempted&&gated.rejection==cp::Rejection::gate_unavailable,17);
  // NaN-only first feature retains exactly the branch dictated by missing routing.
  setup_rivals(f);f.lower[0]=a::domain::finite_min_key;f.upper[0]=a::domain::finite_min_key-1;f.missing_allowed[0]=1;saved.save(f);
  e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  result=fixture_portfolio_label(e,0,f.region,U(f.roots[0]),scratch.region(),scratch.stack,256,512);
  expect(r,result.success&&result.label==2&&saved.unchanged(f),20);accepted_matches_all_signatures(f,result,r,21);
  // Inconsistent missing routes create a real tie won by lower class ID.
  setup_rivals(f,false,true);saved.save(f);e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  result=fixture_portfolio_label(e,0,f.region,U(f.roots[0]),scratch.region(),scratch.stack,256,512);
  expect(r,!result.success&&saved.unchanged(f),22);
  float counterexample[2]={__uint_as_float(0x7fc00001u),-1.f};expect(r,class_on_row(f,counterexample)==0,23);
  // A wide one-hot group includes bit 64 and has no numeric axes.
  setup_rivals(f,true);saved.save(f);e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  result=fixture_portfolio_label(e,0,f.region,U(f.roots[0]),scratch.region(),scratch.stack,256,512);
  expect(r,result.success&&result.label==2&&saved.unchanged(f),24);accepted_matches_all_signatures(f,result,r,25);
  // A tied class is not prunable merely because another rival was certified.
  setup_rivals(f);f.bias[2]=1;f.words[2]=__float_as_uint(1.f);saved.save(f);e=f.e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  result=fixture_portfolio_label(e,0,f.region,U(f.roots[0]),scratch.region(),scratch.stack,256,512);
  expect(r,!result.success&&saved.unchanged(f),26);
  deeper_cover_checks(r,f,scratch,saved);
  selective_effort_checks(r,f,scratch,saved);
  cover_cost_shortcut_checks(r,f,scratch,saved);
  negative_target_checks(r,f,scratch,saved);
  *output=observed;
}
__global__ void rival_setup_benchmark(Fixture* f){if(!blockIdx.x&&!threadIdx.x)setup_rivals(*f);}
__global__ void rival_benchmark(const Fixture* f,U* output,U count,bool portfolio){
  const U job=blockIdx.x*blockDim.x+threadIdx.x;if(job>=count)return;
  RivalScratch scratch;auto e=f->e;e.range_lower=scratch.range_lower;e.range_upper=scratch.range_upper;
  const auto r=portfolio?cp::portfolio_label(e,0,f->region,U(f->roots[0]),scratch.region(),scratch.stack,256,512):
    cp::interval_label(e,0,f->region,U(f->roots[0]),scratch.region(),scratch.stack,256,512);
  output[job]=r.visited+U(r.label+1)*1000;
}
float rival_timed(const Fixture* f,U* out,bool portfolio){
  constexpr U jobs=1024,repetitions=20;cudaEvent_t begin,end;ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));
  rival_benchmark<<<jobs/128,128>>>(f,out,jobs,portfolio);finish();ck(cudaEventRecord(begin));
  for(U i=0;i<repetitions;++i)rival_benchmark<<<jobs/128,128>>>(f,out,jobs,portfolio);
  ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));float ms=0;ck(cudaEventElapsedTime(&ms,begin,end));ck(cudaEventDestroy(begin));ck(cudaEventDestroy(end));return ms;
}
int main(){try{
  RivalReport* device=nullptr;ck(cudaMalloc(reinterpret_cast<void**>(&device),sizeof(RivalReport)));rival_checks<<<1,1>>>(device);finish();
  RivalReport r;ck(cudaMemcpy(&r,device,sizeof(r),cudaMemcpyDeviceToHost));ck(cudaFree(device));
  if(r.checked.failures){std::cerr<<"rival cover checks failed: "<<r.checked.failures<<" first="<<r.checked.first_failure<<"\n";return 1;}
  Fixture* f=nullptr;U* out=nullptr;ck(cudaMalloc(reinterpret_cast<void**>(&f),sizeof(Fixture)));ck(cudaMalloc(reinterpret_cast<void**>(&out),1024*sizeof(U)));
  rival_setup_benchmark<<<1,1>>>(f);finish();const float common_ms=rival_timed(f,out,false),portfolio_ms=rival_timed(f,out,true);
  ck(cudaFree(out));ck(cudaFree(f));
  std::cout<<"{\"complete\":true,\"CUDA_executed\":true,\"checks\":"<<r.checked.checks<<",\"failures\":0,\"common_cover_visits\":"<<r.common_visits<<",\"rival_portfolio_visits\":"<<r.portfolio_visits<<",\"rival_covers\":"<<r.covers<<",\"rivals_certified\":"<<r.certified<<",\"common_cover_proves_root\":false,\"rival_portfolio_proves_root\":true,\"timing_jobs_per_launch\":1024,\"timing_launches\":20,\"common_cover_ms\":"<<common_ms<<",\"rival_portfolio_ms\":"<<portfolio_ms<<",\"timing_scope\":\"synthetic proof kernel cost, not end-to-end conversion speedup\",\"native_RuntimeGate_qualified\":false}\n";
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
