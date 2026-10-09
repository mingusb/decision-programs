// Synthetic CUDA enclosure/coverage checks. These tests exercise a numerical
// gate flag in a fixture; they do not qualify a native XGBoost RuntimeGate.
#include "class_conversion/adaptive_effort.cuh"
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
  static constexpr U N=128,F=65,K=7,T=32;
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
// All synthetic numeric thresholds are -1,0,1. The seven witnesses therefore
// exhaust their finite comparison signatures, signed zeros, and missing atom.
__device__ void exhaustive_pair(Fixture& f,Report& report,U code,float incoming_lower=1.f,float incoming_upper=1.f){
  const auto bound=a::joint::pair_enclosure(f.e,f.region,U(f.roots[0]),U(f.roots[1]),incoming_lower,incoming_upper,f.scratch,512,100000);
  expect(report,bound.complete,code);
  const float atoms[7]={-2.f,-1.f,-0.f,0.f,1.f,2.f,__uint_as_float(0x7fc00001u)};
  float row[Fixture::F];for(U i=0;i<f.features;++i)row[i]=0;
  float minimum=0,maximum=0;bool any=false;
  const U count=f.group_width?f.group_width:(f.features==1?7:49);
  for(U i=0;i<count;++i){
    if(f.group_width){for(U j=0;j<f.features;++j)row[j]=float(j==i);if(!(f.allowed[i/64]&(a::u64(1)<<(i%64))))continue;}
    else {row[0]=atoms[i%7];if(f.features>1)row[1]=atoms[i/7];if(!numeric_inside(f,0,row[0])||(f.features>1&&!numeric_inside(f,1,row[1])))continue;}
    const float first=evaluate_leaf(f,U(f.roots[0]),row),second=evaluate_leaf(f,U(f.roots[1]),row);
    const float lo=__fadd_rn(__fadd_rn(incoming_lower,first),second),hi=__fadd_rn(__fadd_rn(incoming_upper,first),second);
    expect(report,bound.lower<=lo&&hi<=bound.upper,code+1);
    if(!any){minimum=lo;maximum=hi;any=true;}else{minimum=fminf(minimum,lo);maximum=fmaxf(maximum,hi);}
  }
  expect(report,any&&bound.lower==minimum&&bound.upper==maximum,code+2);
}
// Frozen exhaustive interval rule: independent differential oracle for the
// linear selector. All numerical comparisons execute on the GPU. This tests
// interval selection, not native-source qualification.
__device__ int exhaustive_label_reference(a::EngineView e){
  if(!e.qualified_gap)return -1;
  for(U c=0;c<e.source.classes;++c){const float lo=e.range_lower[c],hi=e.range_upper[c];
    if(!isfinite(lo)||!isfinite(hi)||lo>hi||lo < -10.f||hi > 10.f)return -1;}
  for(U w=0;w<e.source.classes;++w){bool wins=true;
    for(U c=0;c<e.source.classes;++c)if(c!=w&&
      __dsub_rn(double(e.range_lower[w]),double(e.range_upper[c])) < native_softprob_gap::computed_gap_minimum)wins=false;
    if(wins)return int(w);}
  return -1;
}
__device__ void selector_checks(Report& r){
  a::EngineView e{};e.qualified_gap=true;
  expect(r,a::qualified_range_label(e)==-1,3000); // Empty: no array read.
  e.source.classes=70;e.qualified_gap=false;
  expect(r,a::qualified_range_label(e)==-1,3001); // Disabled: no array read.
  float lo[70],hi[70];e.range_lower=lo;e.range_upper=hi;e.qualified_gap=true;
  const U sizes[]={1,2,7,10,70};U seed=0x6d2b79f5u;
  for(U k:sizes){e.source.classes=k;e.source.native_margin_classes=1;
    for(U trial=0;trial<128;++trial){
      for(U c=0;c<k;++c){seed=1664525u*seed+1013904223u;const float x=float(int(seed%20481u)-10240)/1024.f;
        seed=1664525u*seed+1013904223u;const float y=float(int(seed%20481u)-10240)/1024.f;
        lo[c]=fminf(x,y);hi[c]=fmaxf(x,y);}
      expect(r,a::qualified_range_label(e)==exhaustive_label_reference(e),3010+k);
    }
    for(U w=0;w<k;++w){
      for(U c=0;c<k;++c){lo[c]=-2.f;hi[c]=-1.f;}lo[w]=0;hi[w]=10;
      expect(r,a::qualified_range_label(e)==int(w),3100+k); // Exclude own upper.
    }
    for(U c=0;c<k;++c){lo[c]=-2;hi[c]=0;}
    expect(r,a::qualified_range_label(e)==(k==1?0:-1),3200+k); // Tied lowers.
    for(U c=0;c<k;++c)lo[c]=hi[c]=__uint_as_float(c&1?0u:0x80000000u);
    expect(r,a::qualified_range_label(e)==(k==1?0:-1),3210+k);
    // Every class, including the structural suffix, must still be validated.
    for(U c=0;c<k;++c)for(U mode=0;mode<6;++mode){
      for(U j=0;j<k;++j){lo[j]=-2;hi[j]=-1;}lo[0]=1;hi[0]=2;
      if(mode==0)lo[c]=__uint_as_float(0x7fc00001u);
      if(mode==1)hi[c]=INFINITY;
      if(mode==2)lo[c]=-INFINITY;
      if(mode==3){lo[c]=1;hi[c]=0;}
      if(mode==4)lo[c]=nextafterf(-10.f,-INFINITY);
      if(mode==5)hi[c]=nextafterf(10.f,INFINITY);
      expect(r,a::qualified_range_label(e)==-1,3300+k);
    }
  }
  e.source.classes=2;const float gap=float(native_softprob_gap::computed_gap_minimum);
  for(U w=0;w<2;++w)for(U side=0;side<3;++side){
    lo[0]=hi[0]=lo[1]=hi[1]=0;
    lo[w]=hi[w]=side==0?nextafterf(gap,-INFINITY):side==1?gap:nextafterf(gap,INFINITY);
    expect(r,a::qualified_range_label(e)==(side?int(w):-1),3400+w*3+side);
  }
}

__global__ void checks(Report* output){
  if(blockIdx.x||threadIdx.x)return;Report r;selector_checks(r);Fixture f;
  f.init();f.tree(f.stump(0,0,-2,2));f.tree(f.stump(0,0,2,-2));f.finish();
  exhaustive_pair(f,r,10);exhaustive_pair(f,r,20,-2,3);
  auto p=a::joint::pair_enclosure(f.e,f.region,0,3,1,1,f.scratch,12,100);
  expect(r,p.complete&&p.lower==1&&p.upper==1&&p.leaves==2&&p.rejected_sides==2,30);
  for(U budget=0;budget<5;++budget){auto q=a::joint::pair_enclosure(f.e,f.region,0,3,1,1,f.scratch,512,budget);expect(r,!q.complete&&q.lower==-3&&q.upper==5&&q.visited<=budget,31);}
  for(U capacity=0;capacity<12;++capacity){auto q=a::joint::pair_enclosure(f.e,f.region,0,3,1,1,f.scratch,capacity,100);expect(r,!q.complete&&q.lower==-3&&q.upper==5,32);}
  auto effect=a::effort::interval_label(f.e,0,f.region,f.scratch,512,100);
  expect(r,effect.static_label==-1&&effect.baseline_label==-1&&effect.label==0&&effect.pair_additional_prune&&effect.pair_completed==1&&effect.pair_tightened==1&&effect.visited<=100&&effect.pair_visits<=effect.pair_visit_budget,33);
  expect(r,effect.baseline_visit_budget==6&&effect.pair_visit_budget==94&&effect.visited==11,37);
  f.e.joint_pair_visit_budget=1;auto capped=a::effort::interval_label(f.e,0,f.region,f.scratch,512,100);
  expect(r,capped.pair_visit_budget==1&&capped.baseline_visit_budget==99&&capped.pair_visits<=1&&capped.label==-1,38);
  f.e.joint_pair_visit_budget=0;auto old=a::effort::interval_label(f.e,0,f.region,f.scratch,512,100);
  expect(r,old.label==-1&&old.pair_attempts==0&&old.baseline_visit_budget==100,34);
  f.e.joint_pair_visit_budget=UINT32_MAX;
  for(U budget=1;budget<25;++budget){auto q=a::effort::interval_label(f.e,0,f.region,f.scratch,512,budget);expect(r,q.visited<=budget&&q.pair_visits<=q.pair_visit_budget&&q.baseline_visit_budget+q.pair_visit_budget==budget,35);if(q.pair_completed==0)expect(r,q.label==q.baseline_label&&f.range_lower[0]==-3&&f.range_upper[0]==5,36);}
  // A failed joint attempt must retain a CONDITIONED incumbent that is already
  // tighter than the static pair fallback, even when that incumbent cannot prune.
  f.init(2);f.tree(f.stump(0,0,-2,2));f.tree(f.stump(1,0,-2,2));f.finish(-3);
  f.lower[0]=a::domain::zero_key;f.missing_allowed[0]=0;f.e.joint_pair_visit_budget=1;
  effect=a::effort::interval_label(f.e,0,f.region,f.scratch,512,100);
  expect(r,effect.label==-1&&effect.baseline_label==-1&&effect.pair_fallbacks==1&&f.range_lower[0]==-3&&f.range_upper[0]==1,39);
  f.init();f.tree(f.stump(0,0,-2,2,true));f.tree(f.stump(0,0,2,-2,false));f.finish();
  exhaustive_pair(f,r,40);effect=a::effort::interval_label(f.e,0,f.region,f.scratch,512,100);expect(r,effect.label==-1&&f.range_lower[0]==-3&&f.range_upper[0]==1,43);
  f.lower[0]=a::domain::finite_min_key;f.upper[0]=a::domain::finite_min_key-1;exhaustive_pair(f,r,44);
  f.init(2,false,2);f.tree(f.stump(0,.5f,-2,2));f.tree(f.stump(1,.5f,-2,2));f.finish();exhaustive_pair(f,r,50);
  f.init(65,false,65);f.tree(f.stump(0,.5f,-2,2));f.tree(f.stump(64,.5f,-2,2));f.finish();exhaustive_pair(f,r,60);
  f.allowed[0]=1;f.allowed[1]=1;exhaustive_pair(f,r,63);
  f.init();
  // The nominal 100-valued leaf is impossible: x<0 and x>=1.
  f.nodes=1;f.feature[0]=0;f.cut[0]=0;f.missing[0]=1;f.value[0]=0;
  f.left[0]=int(f.stump(0,1,-2,100));f.right[0]=int(f.leaf(2));f.tree(0);f.tree(f.stump(0,0,2,-2));f.finish();exhaustive_pair(f,r,70);
  f.init();f.tree(f.leaf(1));f.tree(f.leaf(-16777216.f));f.finish();p=a::joint::pair_enclosure(f.e,f.region,0,1,16777216.f,16777216.f,f.scratch,512,10);
  expect(r,p.complete&&p.lower==0&&p.upper==0&&__fadd_rn(16777216.f,__fadd_rn(1.f,-16777216.f))==1.f,80);
  f.init();f.tree(f.leaf(__uint_as_float(1)));f.tree(f.leaf(-__uint_as_float(1)));f.finish();p=a::joint::pair_enclosure(f.e,f.region,0,1,0,0,f.scratch,512,10);expect(r,p.complete&&p.lower==0&&p.upper==0,81);
  // Infinite cuts partition finite inputs deterministically, while the NaN
  // atom still follows each independent source default direction.
  f.init();f.tree(f.stump(0,__uint_as_float(0x7f800000u),-2,2,false));f.tree(f.stump(0,__uint_as_float(0xff800000u),2,-2,true));f.finish();
  p=a::joint::pair_enclosure(f.e,f.region,0,3,1,1,f.scratch,512,100);
  expect(r,p.complete&&p.lower==-3&&p.upper==5&&p.leaves==2,82);
  f.missing_allowed[0]=0;f.e.domain.allow_nan=0;
  p=a::joint::pair_enclosure(f.e,f.region,0,3,1,1,f.scratch,512,100);
  expect(r,p.complete&&p.lower==-3&&p.upper==-3&&p.leaves==1,83);
  // Different thresholds, features and missing routes; exhaustive signatures.
  for(U variant=0;variant<36;++variant){f.init(2);const U root=f.nodes++;f.feature[root]=int(variant%2);f.cut[root]=float(int(variant%3)-1);f.value[root]=0;f.missing[root]=variant&1;
    f.left[root]=int(f.stump((variant+1)%2,0,-3,1,variant&2));f.right[root]=int(f.stump(variant%2,1,2,-1,variant&4));f.tree(root);
    f.tree(f.stump((variant/2)%2,float(int((variant/3)%3)-1),-2,3,variant&8));f.finish();exhaustive_pair(f,r,100+variant*3);
  }
  // Same-channel adjacency must skip interleaved other-channel trees and retain
  // the original per-channel order, including an unpaired trailing tree.
  f.init(1,true,0,7);f.tree(f.stump(0,0,-2,2),0);f.tree(f.leaf(0),1);f.tree(f.stump(0,0,2,-2),0);f.tree(f.leaf(.25f),0);f.finish();
  effect=a::effort::interval_label(f.e,0,f.region,f.scratch,512,100);expect(r,effect.label==0&&effect.pair_additional_prune&&f.range_lower[0]==1.25f&&f.range_upper[0]==1.25f,220);
  f.e.qualified_gap=false;effect=a::effort::interval_label(f.e,0,f.region,f.scratch,512,100);expect(r,effect.label==-1&&!effect.attempted&&effect.visited==0,221);
  *output=r;
}

__global__ void setup_benchmark(Fixture* f,bool correlated){if(blockIdx.x||threadIdx.x)return;f->init(2);f->tree(f->stump(0,0,-2,2));f->tree(f->stump(correlated?0:1,0,2,-2));f->finish();}
__global__ void benchmark(const Fixture* f,U* output,U count,bool paired){
  const U job=blockIdx.x*blockDim.x+threadIdx.x;if(job>=count)return;
  auto e=f->e;float lower[2],upper[2];U scratch[128];e.range_lower=lower;e.range_upper=upper;e.joint_pair_visit_budget=paired?UINT32_MAX:0;
  const auto result=a::effort::interval_label(e,0,f->region,scratch,128,128);
  output[job]=result.visited+U(result.label+1)*1000+result.pair_tightened*100000;
}
void ck(cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
void finish(){ck(cudaGetLastError());ck(cudaDeviceSynchronize());}
float timed(const Fixture* f,U* out,bool paired){
  constexpr U jobs=4096,repetitions=40;cudaEvent_t begin,end;ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));
  benchmark<<<jobs/128,128>>>(f,out,jobs,paired);finish();ck(cudaEventRecord(begin));
  for(U i=0;i<repetitions;++i)benchmark<<<jobs/128,128>>>(f,out,jobs,paired);
  ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));float ms=0;ck(cudaEventElapsedTime(&ms,begin,end));ck(cudaEventDestroy(begin));ck(cudaEventDestroy(end));return ms;
}
int main(){try{Report* device=nullptr;Fixture* fixture=nullptr;U* outputs=nullptr;
  ck(cudaMalloc(reinterpret_cast<void**>(&device),sizeof(Report)));checks<<<1,1>>>(device);finish();Report r;ck(cudaMemcpy(&r,device,sizeof(r),cudaMemcpyDeviceToHost));ck(cudaFree(device));
  if(r.failures){std::cerr<<"joint enclosure checks failed: "<<r.failures<<" first="<<r.first_failure<<"\n";return 1;}
  ck(cudaMalloc(reinterpret_cast<void**>(&fixture),sizeof(Fixture)));ck(cudaMalloc(reinterpret_cast<void**>(&outputs),4096*sizeof(U)));
  setup_benchmark<<<1,1>>>(fixture,true);finish();const float correlated_control=timed(fixture,outputs,false),correlated_joint=timed(fixture,outputs,true);
  setup_benchmark<<<1,1>>>(fixture,false);finish();const float independent_control=timed(fixture,outputs,false),independent_joint=timed(fixture,outputs,true);
  ck(cudaFree(outputs));ck(cudaFree(fixture));
  std::cout<<"{\"complete\":true,\"CUDA_executed\":true,\"checks\":"<<r.checks<<",\"failures\":0,\"native_RuntimeGate_qualified\":false,\"timing_jobs_per_launch\":4096,\"timing_launches\":40,\"correlated_control_ms\":"<<correlated_control<<",\"correlated_joint_ms\":"<<correlated_joint<<",\"independent_control_ms\":"<<independent_control<<",\"independent_joint_ms\":"<<independent_joint<<",\"timing_scope\":\"synthetic enclosure kernel cost; excludes construction work avoided\"}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
