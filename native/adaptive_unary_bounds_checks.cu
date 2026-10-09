// Synthetic certificates do not qualify an external native XGBoost runtime.
#define main unary_fixture_unused_main
#include "adaptive_joint_bounds_checks.cu"
#undef main
#include "class_conversion/adaptive_unary_bounds.cuh"
#include "class_conversion/adaptive_cover_proof.cuh"

struct UnaryFixture : Fixture {
  U axes[T],lows[T],highs[T],cuts[N],order[T];
  double first[T];
  __device__ void bind_unary() {
    e.unary_axes=axes;e.unary_minimum=lows;e.unary_maximum=highs;
    e.unary_cuts=cuts;e.unary_cut_capacity=N;e.unary_order=order;
    e.unary_first=first;e.unary_first_capacity=T;
  }
  __device__ a::unary::Scratch certificates() {return {axes,lows,highs,cuts,trees,N,order,T,first,T};}
  __device__ void certify(U budget=100000,U capacity=512) {
    bind_unary();
    for(U c=0;c<classes;++c)range_lower[c]=range_upper[c]=__uint_as_float(words[c]);
    for(U t=0;t<trees;++t) {
      axes[t]=a::none;lows[t]=highs[t]=0;
      if(residual[t]<0)continue;
      const auto cert=a::conditioned_subtree_extrema(e,region,U(residual[t]),scratch,capacity,budget,true);
      axes[t]=cert.axis;lows[t]=cert.minimum;highs[t]=cert.maximum;
      const U channel=U(channels[t]);
      range_lower[channel]=__fadd_rn(range_lower[channel],__uint_as_float(lows[t]));
      range_upper[channel]=__fadd_rn(range_upper[channel],__uint_as_float(highs[t]));
    }
  }
};
__device__ void split_contributions(UnaryFixture& f) {
  f.init(4);
  for(U i=0;i<4;++i) { const U axis=(3*i+2)%4;
    f.tree(f.stump(axis,0,0,.5f),0);f.tree(f.stump(axis,0,0,.5f),0);
    f.tree(f.stump(axis,0,0,1),1);
  }
  f.finish(.25f);f.bind_unary();
}
__device__ a::unary::Result unary_run(UnaryFixture& f,Report& report,U code,
    U budget=100000,U stack_words=512,U cut_capacity=UnaryFixture::N) {
  f.certify();auto cert=f.certificates();cert.cut_capacity=cut_capacity;
  U lo[Fixture::K],hi[Fixture::K],words[Fixture::K];
  for(U c=0;c<f.classes;++c){lo[c]=__float_as_uint(f.range_lower[c]);hi[c]=__float_as_uint(f.range_upper[c]);words[c]=f.words[c];}
  const auto result=a::unary::interval_label(f.e,0,f.region,f.scratch,stack_words,cert,budget);
  expect(report,result.visited<=budget,code);
  for(U c=0;c<f.classes;++c)expect(report,lo[c]==__float_as_uint(f.range_lower[c])&&hi[c]==__float_as_uint(f.range_upper[c])&&words[c]==f.words[c],code+1);
  return result;
}
// Exhaust every comparison signature for thresholds -1,0,1, plus both zeros
// and missing. Category groups enumerate their allowed bits across all words.
__device__ void check_unary_label(UnaryFixture& f,Report& report,U code,int label,float exact_gap=-1) {
  const float atoms[7]={-2.f,-1.f,-0.f,0.f,1.f,2.f,__uint_as_float(0x7fc00001u)};
  U count=f.group_width?f.group_width:1;
  if(!f.group_width)for(U j=0;j<f.features;++j)count*=7;
  float row[Fixture::F],scores[Fixture::K];bool any=false;
  for(U i=0;i<count;++i) {
    bool inside=true;
    if(f.group_width) {
      for(U j=0;j<f.features;++j)row[j]=float(j==i);
      inside=(f.allowed[i/64]&(a::u64(1)<<(i%64)))!=0;
    } else {
      U index=i;for(U j=0;j<f.features;++j){row[j]=atoms[index%7];index/=7;inside&=numeric_inside(f,j,row[j]);}
    }
    if(!inside)continue;any=true;
    for(U c=0;c<f.classes;++c)scores[c]=__uint_as_float(f.words[c]);
    for(U t=0;t<f.trees;++t)if(f.residual[t]>=0) {
      const U c=U(f.channels[t]);scores[c]=__fadd_rn(scores[c],evaluate_leaf(f,U(f.residual[t]),row));
    }
    if(exact_gap>=0)expect(report,__dsub_rn(double(scores[0]),double(scores[1]))==double(exact_gap),code+2);
    if(label>=0)for(U c=0;c<f.classes;++c)if(c!=U(label))
      expect(report,__dsub_rn(double(scores[label]),double(scores[c]))>=native_softprob_gap::computed_gap_minimum,code);
  }
  expect(report,any,code+1);
}
__global__ void unary_checks(Report* output) {
  if(blockIdx.x||threadIdx.x)return;Report report;UnaryFixture f;
  split_contributions(f);f.e.relational_bounds_enabled=true;
  auto control=a::effort::interval_label(f.e,0,f.region,f.scratch,512,100000);
  expect(report,control.label<0&&control.baseline_label<0,1);
  U cover_lo[Fixture::F],cover_hi[Fixture::F],cover_missing[Fixture::F];a::u64 cover_allowed[2];
  f.e.rival_cover_enabled=true;
  const auto cover=a::cover::portfolio_label(f.e,0,f.region,U(f.roots[0]),
      {cover_lo,cover_hi,cover_missing,cover_allowed},f.scratch,512,100000);
  expect(report,!cover.success,2);
  auto result=unary_run(f,report,10);
  expect(report,result.success&&result.label==0&&result.groups==4&&result.completed==4&&result.visited==108,12);
  check_unary_label(f,report,13,result.label,.25f);
  result=unary_run(f,report,188,6);
  expect(report,!result.optimistic_rejected&&result.label<0&&result.visited==6&&
      isinf(f.first[1])&&isinf(f.first[2])&&isinf(f.first[3]),190);
  auto short_probe=f.certificates();short_probe.first_capacity=1;
  f.first[0]=42;f.first[1]=43;
  result=a::unary::interval_label(f.e,0,f.region,f.scratch,512,short_probe,100000);
  expect(report,result.label==0&&!result.optimistic_rejected&&result.visited==108&&
      f.first[0]==42&&f.first[1]==43,202);
  auto short_order=f.certificates();short_order.order_capacity=f.trees-1;
  result=a::unary::interval_label(f.e,0,f.region,f.scratch,512,short_order,100000);
  expect(report,!result.attempted&&result.label<0&&result.visited==0,160);
  short_order=f.certificates();short_order.order=nullptr;
  result=a::unary::interval_label(f.e,0,f.region,f.scratch,512,short_order,100000);
  expect(report,!result.attempted&&result.label<0&&result.visited==0,161);
  for(U budget=0;budget<80;++budget) {
    result=unary_run(f,report,20,budget);
    expect(report,result.label<0||result.label==0,22);
    if(!budget)expect(report,!result.attempted&&result.visited==0,23);
    if(result.completed<4)expect(report,result.label<0,24);
  }
  result=unary_run(f,report,25,100000,0);expect(report,result.label<0,27);
  result=unary_run(f,report,28,100000,1);expect(report,result.label<0,30);
  result=unary_run(f,report,31,100000,512,0);expect(report,result.label<0&&result.fallbacks>0,33);
  f.certify(1);for(U t=0;t<f.trees;++t)expect(report,f.axes[t]==a::none,34);
  result=a::unary::interval_label(f.e,0,f.region,f.scratch,512,f.certificates(),100000);expect(report,result.label<0,35);
  // Every traversal shares the one dynamic allocation, even with both advanced
  // margin methods enabled. Exact source evaluation validates accepted labels.
  split_contributions(f);f.e.unary_bounds_enabled=f.e.relational_bounds_enabled=true;
  control=a::effort::interval_label(f.e,0,f.region,f.scratch,512,0);
  expect(report,!control.attempted&&!control.unary_attempted&&control.visited==0,40);
  for(U budget=1;budget<=400;++budget) {
    control=a::effort::interval_label(f.e,0,f.region,f.scratch,512,budget);
    expect(report,control.visited<=budget&&control.baseline_visit_budget+control.pair_visit_budget+control.unary_visit_budget+control.relational_visit_budget==budget,41);
    expect(report,control.label<0||(control.label==0&&control.unary_additional_prune),42);
  }
  expect(report,control.label==0&&control.unary_completed==4,43);check_unary_label(f,report,44,control.label);
  // A valid optimistic cap can reject before any group enumeration.
  // The local first-atom value is above the independent floor in this fixture.
  f.init();f.tree(f.stump(0,0,1,0),0);f.tree(f.stump(0,0,0,1),0);
  f.tree(f.stump(0,0,2,0),1);f.finish(.25f);
  result=unary_run(f,report,191);
  expect(report,result.optimistic_rejected&&result.label<0&&result.visited==6&&result.groups==0,193);
  auto unavailable=f.certificates();unavailable.first=nullptr;
  result=a::unary::interval_label(f.e,0,f.region,f.scratch,512,unavailable,100000);
  expect(report,!result.optimistic_rejected&&result.label<0&&result.visited==27&&result.completed==1,194);
  unavailable=f.certificates();unavailable.first_capacity=0;
  result=a::unary::interval_label(f.e,0,f.region,f.scratch,512,unavailable,100000);
  expect(report,!result.optimistic_rejected&&result.label<0&&result.visited==27,195);
  for(U budget=1;budget<6;++budget) {
    result=unary_run(f,report,196,budget);
    expect(report,!result.optimistic_rejected&&result.label<0&&result.visited==budget,198);
  }
  // Axis1 is irrelevant to rival1 but relevant to rival2. Poisoned entries must
  // neither shift group ordinals nor survive the per-rival reset.
  f.init(2,true,0,3);f.tree(f.stump(0,0,0,.5f),0);f.tree(f.stump(0,0,0,.5f),0);
  f.tree(f.stump(0,0,0,1),1);f.tree(f.stump(1,0,.5f,-1),2);f.finish(.25f);f.certify();
  for(U i=0;i<Fixture::T;++i)f.first[i]=-1000;
  result=a::unary::interval_label(f.e,0,f.region,f.scratch,512,f.certificates(),100000);
  expect(report,result.label<0&&result.optimistic_rejected&&f.first[0]==0&&f.first[1]==-.5,203);
  unavailable=f.certificates();unavailable.first=nullptr;
  const auto uncached=a::unary::interval_label(f.e,0,f.region,f.scratch,512,unavailable,100000);
  expect(report,uncached.label==result.label&&!uncached.optimistic_rejected&&uncached.visited==result.visited,204);
  // Category probes use allowed representatives.
  f.init(65,false,65);f.tree(f.stump(64,.5f,0,1),0);f.finish(0);
  f.allowed[0]=1;f.allowed[1]=1;
  result=unary_run(f,report,199);
  expect(report,result.optimistic_rejected&&result.label<0&&result.visited==2,201);
  // A local no-improvement probe also supplies a whole-rival rejection here.
  f.init();f.tree(f.stump(0,0,0,1),0);f.finish(0);
  result=unary_run(f,report,170,100000,512,0);
  expect(report,result.label<0&&result.optimistic_rejected&&result.groups==0&&result.visited==2,172);
  for(U budget=0;budget<=2;++budget) {
    result=unary_run(f,report,173,budget);
    expect(report,result.visited==budget&&result.completed==0&&result.optimistic_rejected==(budget==2),175);
    if(budget==1)expect(report,result.fallbacks==1,176);
  }
  result=unary_run(f,report,177,100000,0);
  expect(report,result.visited==0&&result.completed==0&&result.fallbacks==1,179);
  // Exercise missing-only and bit64 representatives directly: the ordinary
  // classifier would mark these fully forced factors constant (axis=none).
  const U one_member[1]={0};double group_lower=-2;a::unary::Result group_work;
  f.init();f.tree(f.stump(0,0,-2,3,false),0);f.finish(0);
  f.upper[0]=f.lower[0]-1;f.certify();
  bool complete=a::unary::detail::group_floor(f.e,f.region,f.residual,f.certificates(),
      one_member,1,0,0,1,f.scratch,512,group_lower,group_work,100000);
  expect(report,complete&&group_lower==3&&group_work.visited==4,180);
  f.init(65,false,65);f.tree(f.stump(64,.5f,0,1),0);f.finish(0);
  f.allowed[0]=0;f.allowed[1]=1;f.certify();group_lower=0;group_work={};
  complete=a::unary::detail::group_floor(f.e,f.region,f.residual,f.certificates(),
      one_member,1,0,0,1,f.scratch,512,group_lower,group_work,100000);
  expect(report,complete&&group_lower==1&&group_work.visited==4,181);
  // A unary singleton is not disposable: its low leaf is unreachable because
  // x<0 implies x<1. The first atom is inconclusive and the full scan tightens.
  f.init(2);const U outer=f.nodes++;f.feature[outer]=0;f.cut[outer]=0;
  f.value[outer]=0;f.missing[outer]=1;f.left[outer]=int(f.stump(0,1,1,-4));
  f.right[outer]=int(f.leaf(1));f.tree(outer,0);f.tree(f.stump(1,0,0,-4),1);f.finish(0);
  result=unary_run(f,report,182);
  expect(report,result.label==0&&result.completed==2&&result.visited==17,184);
  check_unary_label(f,report,185,result.label);
  // Matching finite paths are insufficient when source missing routes differ.
  f.init();f.tree(f.stump(0,0,0,.5f),0);f.tree(f.stump(0,0,0,.5f),0);f.tree(f.stump(0,0,0,1,false),1);f.finish(.25f);
  result=unary_run(f,report,50);expect(report,result.label<0&&!result.optimistic_rejected&&result.completed==1&&result.visited==27,52);
  f.missing_allowed[0]=0;f.e.domain.allow_nan=0;
  result=unary_run(f,report,53);expect(report,result.label==0&&result.visited==21,55);check_unary_label(f,report,56,result.label);
  // A missing-only coordinate remains a valid atom; signed zeros share the
  // finite comparison route while the original leaf/addition words are kept.
  f.missing_allowed[0]=1;f.e.domain.allow_nan=1;f.missing[U(f.roots[2])]=1;
  f.lower[0]=a::domain::finite_min_key;f.upper[0]=a::domain::finite_min_key-1;
  result=unary_run(f,report,60);expect(report,result.label==0,62);check_unary_label(f,report,63,result.label);
  f.init();f.tree(f.stump(0,-0.f,-0.f,.5f),0);f.tree(f.stump(0,+0.f,+0.f,.5f),0);f.tree(f.stump(0,0,-0.f,1),1);f.finish(.25f);
  result=unary_run(f,report,64);expect(report,result.label==0,66);check_unary_label(f,report,67,result.label);
  // All categorical members are one semantic axis, including bit64.
  f.init(65,false,65);
  for(U feature=0;feature<=64;feature+=64){f.tree(f.stump(feature,.5f,0,.5f),0);f.tree(f.stump(feature,.5f,0,.5f),0);f.tree(f.stump(feature,.5f,0,1),1);}
  f.finish(.25f);result=unary_run(f,report,70);expect(report,result.label==0&&result.completed==1,72);check_unary_label(f,report,73,result.label);
  f.allowed[0]=1;f.allowed[1]=1;result=unary_run(f,report,74);expect(report,result.label==0,76);check_unary_label(f,report,77,result.label);
  // A buried second feature is removed only after its region predicate is
  // forced. This is a deeper tree, not a stump-only shape recognition rule.
  f.init(2);f.nodes=1;f.feature[0]=0;f.cut[0]=0;f.value[0]=0;f.missing[0]=1;
  f.left[0]=int(f.stump(1,0,0,2));f.right[0]=int(f.leaf(1));f.tree(0,1);
  f.tree(f.stump(0,0,0,.5f),0);f.tree(f.stump(0,0,0,.5f),0);f.finish(.25f);f.certify();
  expect(report,f.axes[0]==a::none,80);result=unary_run(f,report,81);expect(report,result.label<0,83);
  f.upper[1]=a::domain::predecessor_key(a::domain::zero_key);f.missing_allowed[1]=0;
  result=unary_run(f,report,84);expect(report,f.axes[0]==0&&result.label==0,86);check_unary_label(f,report,87,result.label);
  // Distinct cuts create an interior atom; endpoints alone would miss its
  // possible counterexample. The second winner factor is a depth-two tree.
  f.init();f.tree(f.stump(0,-1,2,-1),0);
  const U second=f.nodes++;f.feature[second]=0;f.cut[second]=-1;f.value[second]=0;f.missing[second]=1;
  f.left[second]=int(f.leaf(-1));f.right[second]=int(f.stump(0,1,2,-1));f.tree(second,0);
  f.tree(f.stump(0,1,1,-2),1);f.finish(1);
  result=unary_run(f,report,140);expect(report,result.label==0,142);check_unary_label(f,report,143,result.label,1);
  result=unary_run(f,report,146,100000,512,1);expect(report,result.label<0&&result.fallbacks>0,148);
  f.value[U(f.left[U(f.right[second])])]=0;f.finish(1);
  result=unary_run(f,report,150);expect(report,result.label<0&&!result.optimistic_rejected,152);check_unary_label(f,report,153,result.label);
  // Consumed roots contribute only through the saved prefix. An ungrouped
  // trailing constant retains its own source addition and signed floor.
  f.init();f.tree(f.leaf(100),0);f.tree(f.stump(0,0,0,.5f),1);f.tree(f.stump(0,0,0,.5f),1);f.tree(f.stump(0,0,0,1),0);f.tree(f.leaf(.125f),0);
  f.bias[1]=.5f;f.finish(0);f.residual[0]=-1;
  result=unary_run(f,report,90);expect(report,result.label==1,92);check_unary_label(f,report,93,result.label);
  // Real-number reassociation would add one; the original FP32 fold ties.
  f.init();f.tree(f.leaf(1),0);f.tree(f.leaf(-16777216.f),0);f.tree(f.leaf(0),1);f.finish(16777216.f);
  result=unary_run(f,report,100);expect(report,result.label<0,102);check_unary_label(f,report,103,result.label);
  // Unequal subnormal terms cancel in the exact signed sum; final acceptance
  // must still account for each original channel's rounding error.
  f.init();const float tiny=__uint_as_float(1);
  f.tree(f.stump(0,0,-tiny,tiny),0);f.tree(f.stump(0,0,-tiny,tiny),0);f.tree(f.stump(0,0,-2*tiny,2*tiny),1);f.finish(.25f);
  result=unary_run(f,report,110);expect(report,result.label==0,112);check_unary_label(f,report,113,result.label);
  // Perturbed margins around the native threshold exercise directed bound
  // arithmetic; every accepted label is checked against the ordered RN32 fold.
  for(U variant=0;variant<48;++variant) {
    f.init();const float term=__uint_as_float(0x3f000000u+variant*997u);
    f.tree(f.stump(0,0,-term,term),0);f.tree(f.stump(0,0,-term,term),0);f.tree(f.stump(0,0,-2*term,2*term),1);
    f.finish(float(native_softprob_gap::computed_gap_minimum*(variant%3==0?0.5:(variant%3==1?1.0:2.0))));
    result=unary_run(f,report,120);check_unary_label(f,report,122,result.label);
    if(variant%3==0)expect(report,result.label<0,124);
    if(variant%3==2)expect(report,result.label==0,125);
  }
  // Every class shares shuffled axes; each rival receives exactly four groups.
  f.init(4,true,0,Fixture::K);
  for(U i=0;i<4;++i) {const U axis=(3*i+2)%4;
    f.tree(f.stump(axis,0,0,.5f),0);f.tree(f.stump(axis,0,0,.5f),0);
    for(U c=1;c<f.classes;++c)f.tree(f.stump(axis,0,0,1),c);
  }
  f.finish(.25f);result=unary_run(f,report,162);
  expect(report,result.label==0&&result.groups==4*(f.classes-1)&&result.completed==result.groups,164);
  check_unary_label(f,report,165,result.label,.25f);
  // Sorting only unary IDs must produce the identical full permutation:
  // shuffled unary axes first, then interleaved none IDs in source order.
  f.init(4,true,0,3);
  f.tree(f.stump(3,0,0,1),2);f.tree(f.leaf(0),0);
  f.tree(f.stump(1,0,0,1),0);f.tree(f.stump(3,0,0,1),1);
  f.tree(f.stump(0,0,0,1),2);f.tree(f.leaf(-1),2);
  f.tree(f.stump(1,0,0,1),1);f.tree(f.leaf(0),1);
  f.finish();f.residual[4]=-1;f.certify();U members=0;
  expect(report,a::unary::detail::order_members(f.e,f.residual,f.certificates(),members)&&members==7,168);
  const U expected_order[7]={2,6,0,3,1,5,7};
  for(U i=0;i<7;++i)expect(report,f.order[i]==expected_order[i],169);
  auto tail_short=f.certificates();tail_short.order_capacity=5;
  f.order[5]=f.order[6]=0xa5a5a5a5u;
  expect(report,!a::unary::detail::order_members(f.e,f.residual,tail_short,members)&&members==5,205);
  for(U i=0;i<5;++i)expect(report,f.order[i]==expected_order[i],206);
  expect(report,f.order[5]==0xa5a5a5a5u&&f.order[6]==0xa5a5a5a5u,207);
  // With zero unary groups the none tail still needs its original capacity.
  f.init();for(U i=0;i<4;++i)f.tree(f.leaf(float(i)),i%2);
  f.finish();f.residual[1]=-1;f.certify();
  expect(report,a::unary::detail::order_members(f.e,f.residual,f.certificates(),members)&&members==3,208);
  expect(report,f.order[0]==0&&f.order[1]==2&&f.order[2]==3,209);
  tail_short=f.certificates();tail_short.order_capacity=2;f.order[2]=0xa5a5a5a5u;
  expect(report,!a::unary::detail::order_members(f.e,f.residual,tail_short,members)&&members==2&&f.order[2]==0xa5a5a5a5u,210);
  tail_short=f.certificates();tail_short.order=nullptr;
  expect(report,!a::unary::detail::order_members(f.e,f.residual,tail_short,members)&&members==0,211);
  for(U i=0;i<f.trees;++i)f.residual[i]=-1;
  expect(report,a::unary::detail::order_members(f.e,f.residual,tail_short,members)&&members==0,212);
  split_contributions(f);f.certify();f.e.qualified_gap=false;
  result=a::unary::interval_label(f.e,0,f.region,f.scratch,512,f.certificates(),100000);expect(report,result.label<0&&!result.attempted,130);
  f.e.qualified_gap=true;f.range_upper[1]=11;
  result=a::unary::interval_label(f.e,0,f.region,f.scratch,512,f.certificates(),100000);expect(report,result.label<0,131);
  *output=report;
}
int main(){try {
  Report* device=nullptr;ck(cudaMalloc(reinterpret_cast<void**>(&device),sizeof(Report)));
  unary_checks<<<1,1>>>(device);finish();Report report;ck(cudaMemcpy(&report,device,sizeof(report),cudaMemcpyDeviceToHost));ck(cudaFree(device));
  if(report.failures){std::cerr<<"unary bounds checks failed: "<<report.failures<<" first="<<report.first_failure<<'\n';return 1;}
  std::cout<<"{\"complete\":true,\"CUDA_executed\":true,\"checks\":"<<report.checks<<",\"failures\":0,\"native_RuntimeGate_qualified\":false,\"four_axis_split_contribution_closed\":true}\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
