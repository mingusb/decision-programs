// Mechanics checks. Synthetic qualified-gap/source correspondence
// premises are explicit fixtures, not creation of authentic native authority.
#include "class_conversion/adaptive_cover_proof.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>
namespace a=class_conversion_adaptive;namespace cp=a::cover;using U=a::u32;
struct Report{U checks=0,failures=0,first_failure=0;};
__device__ void expect(Report&r,bool x,U code){++r.checks;if(!x){++r.failures;if(!r.first_failure)r.first_failure=code;}}
struct Fixture {
  static constexpr U N=128,F=65,K=7,T=16;
  int feature[N],left[N],right[N],roots[T],channels[T],feature_group[F],feature_numeric[F],residual[T];
  float cut[N],value[N],bias[K],range_lower[K],range_upper[K];
  unsigned char missing[N];
  U minimum[N],maximum[N],lower[F],upper[F],missing_allowed[F],feature_bit[F];
  U group_word_offsets[2],group_feature_offsets[2],group_widths[1],group_features[F];
  U words[K],positions[K],scratch[512];
  a::u64 initial_masks[2],allowed[2],wallowed[2]; U wlower[F],wupper[F],wmissing[F];
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
  __device__ void witness_copy(){for(U i=0;i<F;++i){wlower[i]=lower[i];wupper[i]=upper[i];wmissing[i]=missing_allowed[i];}for(U i=0;i<2;++i)wallowed[i]=allowed[i];e.arena.witness_lower=wlower;e.arena.witness_upper=wupper;e.arena.witness_missing=wmissing;e.arena.witness_allowed=wallowed;}
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
    witness_copy();
  }
};

struct Scratch {U lower[65],upper[65],missing[65],stack[256];a::u64 allowed[2];
 __device__ a::domain::RegionView region(){return {lower,upper,missing,allowed};}};
__device__ void point_checks(Report&r,Fixture&f,Scratch&s){
 f.init(2,false);f.lower[0]=a::domain::sortable_word(__float_as_uint(-2.f));f.upper[0]=a::domain::sortable_word(__float_as_uint(3.f));f.finish(.5f);
 float x=99;expect(r,cp::detail::point_coordinate(f.e,f.region,0,x,false)&&x==-2,100);expect(r,cp::detail::point_coordinate(f.e,f.region,0,x,true)&&x==3,101);
 // A finite-empty permitted missing coordinate chooses the NaN atom in both modes.
 f.e.domain.allow_nan=true;f.lower[1]=a::domain::finite_min_key;f.upper[1]=a::domain::finite_min_key-1;f.missing_allowed[1]=1;
 expect(r,cp::detail::point_coordinate(f.e,f.region,1,x,false)&&isnan(x),102);expect(r,cp::detail::point_coordinate(f.e,f.region,1,x,true)&&isnan(x),103);
 // Mixed finite/missing coordinates intentionally choose a finite endpoint.
 f.missing_allowed[0]=1;expect(r,cp::detail::point_coordinate(f.e,f.region,0,x,false)&&x==-2,104);expect(r,cp::detail::point_coordinate(f.e,f.region,0,x,true)&&x==3,105);
 f.missing_allowed[1]=0;expect(r,!cp::detail::point_coordinate(f.e,f.region,1,x,true),106);
 // Whole-group choices remain consistent across word boundaries and holes.
 f.init(65,false,65);f.allowed[0]=(a::u64(1)<<2)|(a::u64(1)<<63);f.allowed[1]=1;f.finish(0);
 for(U i=0;i<65;++i){expect(r,cp::detail::point_coordinate(f.e,f.region,i,x,false)&&x==float(i==2),110);expect(r,cp::detail::point_coordinate(f.e,f.region,i,x,true)&&x==float(i==64),111);}
 // Original source slot order, consumed omission and repeated operands.
 f.init(1,false);f.tree(f.leaf(16777216.f));f.tree(f.leaf(1.f));f.tree(f.leaf(-16777216.f));f.finish(0);
 U visits=0,winner=a::none;expect(r,cp::detail::propose_class(f.e,0,f.region,3,visits,winner,false)&&visits==3&&f.range_lower[0]==0,120);
 expect(r,!cp::detail::propose_class(f.e,0,f.region,2,visits,winner,true)&&visits==2,121);
 f.residual[0]=-1;expect(r,cp::detail::propose_class(f.e,0,f.region,3,visits,winner)&&visits==2&&f.range_lower[0]==-16777215.f,122);
 f.init(1,false);const U node=f.leaf(1.f);f.tree(node);f.tree(node);f.finish(-.5f);expect(r,cp::detail::propose_class(f.e,0,f.region,2,visits,winner)&&visits==2&&f.range_lower[0]==1.5f,123);
 f.residual[0]=-2;expect(r,!cp::detail::propose_class(f.e,0,f.region,2,visits,winner),126);f.residual[0]=int(node);
 f.words[0]=0x7fc00000;expect(r,!cp::detail::propose_class(f.e,0,f.region,2,visits,winner)&&visits==0,124);
 f.words[0]=__float_as_uint(0.f);f.value[node]=__uint_as_float(0x7f800000);expect(r,!cp::detail::propose_class(f.e,0,f.region,2,visits,winner),125);
}
__device__ void intersection_checks(Report&r,Fixture&f,Scratch&s){
 f.init(2,true);f.finish(0);f.wlower[0]=a::domain::sortable_word(__float_as_uint(-1.f));f.wupper[0]=a::domain::sortable_word(__float_as_uint(1.f));f.lower[0]=a::domain::sortable_word(__float_as_uint(0.f));f.upper[0]=a::domain::sortable_word(__float_as_uint(2.f));f.wmissing[0]=0;
 expect(r,cp::detail::witness_intersection(f.e,0,f.region,s.region(),s.stack,256),200);expect(r,s.lower[0]==f.lower[0]&&s.upper[0]==f.wupper[0]&&s.missing[0]==0,201);
 f.wupper[0]=a::domain::sortable_word(__float_as_uint(-.5f));expect(r,!cp::detail::witness_intersection(f.e,0,f.region,s.region(),s.stack,256),202);
 // Finite ranges disjoint but shared NaN is a nonempty intersection.
 f.wmissing[0]=1;expect(r,cp::detail::witness_intersection(f.e,0,f.region,s.region(),s.stack,256)&&s.lower[0]>s.upper[0]&&s.missing[0],203);
 f.e.arena.witness_lower=nullptr;expect(r,!cp::detail::witness_intersection(f.e,0,f.region,s.region(),s.stack,256),204);
 // Valid witness inputs may not alias the writable intersection destination.
 f.init(2,false);f.finish(0);for(U i=0;i<2;++i){s.lower[i]=f.lower[i];s.upper[i]=f.upper[i];s.missing[i]=f.missing_allowed[i];}
 f.e.arena.witness_lower=s.lower;f.e.arena.witness_upper=s.upper;f.e.arena.witness_missing=s.missing;
 expect(r,!cp::detail::witness_intersection(f.e,0,f.region,s.region(),s.stack,256),207);
 for(U i=0;i<2;++i)expect(r,s.lower[i]==f.lower[i]&&s.upper[i]==f.upper[i]&&s.missing[i]==f.missing_allowed[i],208);
 f.init(65,false,65);f.finish(0);f.allowed[0]=(a::u64(1)<<63)|4;f.allowed[1]=1;f.wallowed[0]=(a::u64(1)<<63)|2;f.wallowed[1]=1;
 expect(r,cp::detail::witness_intersection(f.e,0,f.region,s.region(),s.stack,256)&&s.allowed[0]==(a::u64(1)<<63)&&s.allowed[1]==1,205);
 f.wallowed[0]=2;f.wallowed[1]=0;expect(r,!cp::detail::witness_intersection(f.e,0,f.region,s.region(),s.stack,256),206);
}

__device__ __noinline__ cp::Result portfolio(a::EngineView e,a::domain::RegionView R,U predicate,Scratch&s,U budget){
 return cp::portfolio_label(e,0,R,predicate,s.region(),s.stack,256,budget);
}
__device__ void setup_sum(Fixture&f,float winner=.5f){
 f.init(2,false);for(U i=0;i<2;++i){f.lower[i]=a::domain::sortable_word(__float_as_uint(-1.f));f.upper[i]=a::domain::sortable_word(__float_as_uint(1.f));}
 f.tree(f.stump(0,0,0,1),1);f.tree(f.stump(1,0,0,1),1);f.finish(winner);f.e.rival_cover_enabled=true;
}
__device__ void singleton_checks(Report&r,Fixture&f){
 f.init(1,false);f.finish(0);f.range_lower[0]=float(native_softprob_gap::computed_gap_minimum);f.range_lower[1]=0;f.range_upper[0]=f.range_upper[1]=9;
 expect(r,cp::detail::qualified_point_label(f.e)==0&&f.range_upper[0]==f.range_lower[0]&&f.range_upper[1]==0,250);
 f.range_lower[0]=__uint_as_float(__float_as_uint(f.range_lower[0])-1);expect(r,cp::detail::qualified_point_label(f.e)<0,251);
 f.range_lower[0]=f.range_lower[1]=-0.f;expect(r,cp::detail::qualified_point_label(f.e)<0,252);
 f.range_lower[0]=11;expect(r,cp::detail::qualified_point_label(f.e)<0,253);f.range_lower[0]=__uint_as_float(0x7f800000);expect(r,cp::detail::qualified_point_label(f.e)<0,254);
 f.range_lower[0]=__uint_as_float(0x7fc00000);expect(r,cp::detail::qualified_point_label(f.e)<0,255);f.range_lower[0]=1;f.e.qualified_gap=false;expect(r,cp::detail::qualified_point_label(f.e)<0,256);
}
__device__ void screen_checks(Report&r,Fixture&f,Scratch&s){
 cp::PointScreen stats{};setup_sum(f);f.e.point_screen=&stats;f.e.two_point_screen_enabled=true;
 const auto first_root=U(f.roots[0]);auto q=portfolio(f.e,f.region,first_root,s,4096);
 expect(r,!q.success&&q.label<0&&q.rejection==cp::Rejection::qualified_points_differ,300);
 expect(r,stats.attempted&&stats.intersection_ready&&stats.first_complete&&stats.first_qualified&&stats.second_complete&&stats.second_qualified&&stats.mixed&&!stats.inconclusive,301);
 expect(r,stats.first_visits==4&&stats.second_visits==4&&q.visited<=4096,302);
 // Each bounded attempt must remain conservative and leave original inputs intact.
 for(U budget=0;budget<96;++budget){stats={};q=portfolio(f.e,f.region,first_root,s,budget);expect(r,!q.success&&q.label<0&&q.visited<=budget,303);expect(r,stats.first_visits+stats.second_visits<=q.visited,304);expect(r,!stats.mixed||(stats.first_complete&&stats.first_qualified&&stats.second_complete&&stats.second_qualified&&stats.intersection_ready),305);expect(r,f.words[0]==__float_as_uint(.5f)&&f.words[1]==0&&f.positions[0]==0&&f.residual[0]==f.roots[0]&&f.residual[1]==f.roots[1],306);}
 // First tie suppresses second evaluation; second tie is not a mixed certificate.
 setup_sum(f,0);f.e.point_screen=&stats;f.e.two_point_screen_enabled=true;stats={};q=portfolio(f.e,f.region,U(f.roots[0]),s,4096);expect(r,stats.first_complete&&!stats.first_qualified&&!stats.second_complete&&!stats.mixed,310);
 setup_sum(f,2);f.e.point_screen=&stats;f.e.two_point_screen_enabled=true;stats={};q=portfolio(f.e,f.region,U(f.roots[0]),s,4096);expect(r,stats.first_qualified&&stats.second_complete&&!stats.second_qualified&&!stats.mixed,311);
 // First/last endpoints agree in a nonuniform XOR; no false uniformity result.
 f.init(2,false);f.tree(f.mismatch(0,1,0),1);f.finish(.5f);f.e.rival_cover_enabled=true;f.e.point_screen=&stats;f.e.two_point_screen_enabled=true;stats={};q=portfolio(f.e,f.region,U(f.roots[0]),s,4096);expect(r,stats.first_qualified&&stats.second_qualified&&!stats.mixed&&stats.inconclusive&&!q.success,312);
 // No W or an empty intersection falls back without negative point authority.
 setup_sum(f);f.e.point_screen=&stats;f.e.two_point_screen_enabled=true;f.e.arena.witness_lower=nullptr;stats={};q=portfolio(f.e,f.region,U(f.roots[0]),s,4096);expect(r,stats.attempted&&!stats.intersection_ready&&!stats.mixed&&stats.first_complete&&!stats.second_complete&&!q.success,313);
 setup_sum(f);f.e.point_screen=&stats;f.e.two_point_screen_enabled=true;f.wlower[0]=a::domain::sortable_word(__float_as_uint(2.f));f.wupper[0]=a::domain::sortable_word(__float_as_uint(3.f));stats={};q=portfolio(f.e,f.region,U(f.roots[0]),s,4096);expect(r,!stats.intersection_ready&&!stats.mixed&&!q.success,314);
 // R endpoints differ, but W excludes the competing-label portion. The screen
 // must use W intersect R rather than falsely reporting those exterior points.
 setup_sum(f);for(U i=0;i<2;++i){f.wlower[i]=a::domain::sortable_word(__float_as_uint(-1.f));f.wupper[i]=a::domain::sortable_word(__float_as_uint(-.5f));}f.e.point_screen=&stats;f.e.two_point_screen_enabled=true;stats={};q=portfolio(f.e,f.region,U(f.roots[0]),s,4096);expect(r,stats.intersection_ready&&stats.first_qualified&&stats.second_qualified&&!stats.mixed&&!q.success,320);
 // A nonfinite first point or later point can never become screen evidence.
 setup_sum(f);f.value[f.left[f.roots[0]]]=__uint_as_float(0x7f800000);f.finish(.5f);f.e.rival_cover_enabled=true;f.e.point_screen=&stats;f.e.two_point_screen_enabled=true;stats={};q=portfolio(f.e,f.region,U(f.roots[0]),s,4096);expect(r,!stats.first_complete&&!stats.mixed&&!q.success,315);
 setup_sum(f);f.value[f.right[f.roots[0]]]=__uint_as_float(0x7f800000);f.finish(.5f);f.e.rival_cover_enabled=true;f.e.point_screen=&stats;f.e.two_point_screen_enabled=true;stats={};q=portfolio(f.e,f.region,U(f.roots[0]),s,4096);expect(r,stats.first_qualified&&!stats.second_complete&&!stats.mixed&&!q.success,316);
 // Missing-only guards are constant and remain handled by the incumbent cover.
 setup_sum(f);f.e.domain.allow_nan=true;for(U i=0;i<2;++i){f.lower[i]=a::domain::finite_min_key;f.upper[i]=a::domain::finite_min_key-1;f.missing_allowed[i]=1;}f.witness_copy();f.e.point_screen=&stats;f.e.two_point_screen_enabled=true;stats={};q=portfolio(f.e,f.region,U(f.roots[0]),s,4096);expect(r,q.success&&q.label==0&&!stats.attempted&&!stats.mixed,317);
 // Both representatives are the same finite point, even though NaN alternatives
 // make the full guard mixed: first qualified point must not trigger duplicate work.
 setup_sum(f);f.e.domain.allow_nan=true;for(U i=0;i<2;++i){f.upper[i]=f.lower[i];f.missing_allowed[i]=1;f.missing[f.roots[i]]=0;}f.witness_copy();f.e.point_screen=&stats;f.e.two_point_screen_enabled=true;stats={};q=portfolio(f.e,f.region,U(f.roots[0]),s,4096);expect(r,stats.attempted&&stats.first_qualified&&!stats.second_complete&&stats.second_visits==0&&!stats.mixed&&!q.success,319);
 // Disabling the option preserves the normal proof path and zero screen work.
 setup_sum(f);f.e.point_screen=&stats;f.e.two_point_screen_enabled=false;stats={};q=portfolio(f.e,f.region,U(f.roots[0]),s,4096);expect(r,!q.success&&!stats.attempted&&!stats.mixed,318);
}
__global__ void checks(Report*out){if(blockIdx.x||threadIdx.x)return;Report r;Fixture f;Scratch s;point_checks(r,f,s);intersection_checks(r,f,s);singleton_checks(r,f);screen_checks(r,f,s);*out=r;}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int main(int argc,char**argv){if(argc==2&&std::string(argv[1])=="--help"){std::cout<<"Two-point screen CUDA mechanics fixture; synthetic proof premises only. Help initializes no CUDA.\n";return 0;}
 try{if(argc!=1)throw std::runtime_error("no arguments expected");Report*p;ck(cudaMalloc(&p,sizeof(Report)));checks<<<1,1>>>(p);ck(cudaGetLastError());ck(cudaDeviceSynchronize());Report r;ck(cudaMemcpy(&r,p,sizeof(r),cudaMemcpyDeviceToHost));ck(cudaFree(p));std::cout<<"{\"checks\":"<<r.checks<<",\"failures\":"<<r.failures<<",\"first_failure\":"<<r.first_failure<<"}\n";return r.failures?1:0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
