// Synthetic CUDA checks of correlated score margins; the fixture gate does
// not qualify any external native XGBoost library.
#define main joint_checks_unused_main
#include "adaptive_joint_bounds_checks.cu"
#undef main
#include "class_conversion/adaptive_relational_bounds.cuh"

__device__ a::relational::Result relational_run(Fixture& f,Report& report,U code,U budget=100000,U capacity=512) {
  a::qualified_interval_label(f.e,0);
  U lo[Fixture::K],hi[Fixture::K],prefix[Fixture::K];
  for(U c=0;c<f.classes;++c){lo[c]=__float_as_uint(f.range_lower[c]);hi[c]=__float_as_uint(f.range_upper[c]);prefix[c]=f.words[c];}
  const auto result=a::relational::interval_label(f.e,0,f.region,f.scratch,capacity,budget);
  expect(report,result.visited<=budget&&result.pair_visits==result.visited,code);
  for(U c=0;c<f.classes;++c)expect(report,lo[c]==__float_as_uint(f.range_lower[c])&&hi[c]==__float_as_uint(f.range_upper[c])&&prefix[c]==f.words[c],code+1);
  return result;
}
// Thresholds -1,0,1 are exhausted by these comparison signatures. Signed zero
// and the missing atom are separate witnesses; categories enumerate every bit.
__device__ void exhaustive_relational(Fixture& f,Report& report,U code,const a::relational::Result& bound) {
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
    if(bound.label>=0)for(U c=0;c<f.classes;++c)if(c!=U(bound.label))
      expect(report,__dsub_rn(double(scores[bound.label]),double(scores[c]))>=native_softprob_gap::computed_gap_minimum,code);
  }
  expect(report,any,code+1);
}
__device__ void shared_stumps(Fixture& f) {
  f.init(4);
  for(U p=0;p<4;++p){f.tree(f.stump(p,0,0,2),0);f.tree(f.stump(p,0,0,2),1);}
  f.finish(1);
}
__global__ void relational_checks(Report* output) {
  if(blockIdx.x||threadIdx.x)return;Report report;Fixture f;
  shared_stumps(f);
  auto old=a::effort::interval_label(f.e,0,f.region,f.scratch,512,10000);
  expect(report,old.label==-1&&old.baseline_label==-1,1);
  auto result=relational_run(f,report,10);
  expect(report,result.success&&result.label==0&&result.pairs_complete==4&&result.rivals_checked==1,12);
  exhaustive_relational(f,report,13,result);
  for(U budget=0;budget<32;++budget) {
    result=relational_run(f,report,20,budget);
    if(!budget)expect(report,!result.attempted&&result.label<0,22);
    if(result.pairs_complete<4)expect(report,result.label<0,23);
    exhaustive_relational(f,report,24,result);
  }
  f.e.relational_bounds_enabled=true;
  auto integrated=a::effort::interval_label(f.e,0,f.region,f.scratch,512,0);
  expect(report,!integrated.attempted&&!integrated.relational_attempted&&integrated.visited==0&&integrated.label<0,26);
  for(U budget=1;budget<=200;++budget) {
    integrated=a::effort::interval_label(f.e,0,f.region,f.scratch,512,budget);
    expect(report,integrated.visited<=budget&&integrated.baseline_visit_budget+integrated.pair_visit_budget+integrated.relational_visit_budget==budget,27);
    expect(report,integrated.label<0||(integrated.label==0&&integrated.relational_additional_prune),28);
  }
  expect(report,integrated.label==0&&integrated.relational_completed==4,29);
  result.label=integrated.label;exhaustive_relational(f,report,300,result);
  // A successful conditioned baseline remains the answer and needs no later
  // relational attempt; reserved and ordinary allocations still sum correctly.
  f.init();f.tree(f.stump(0,0,-2,2),0);f.finish(1);f.e.relational_bounds_enabled=true;
  f.lower[0]=a::domain::zero_key;f.missing_allowed[0]=0;
  integrated=a::effort::interval_label(f.e,0,f.region,f.scratch,512,200);
  expect(report,integrated.label==0&&integrated.baseline_label==0&&!integrated.relational_attempted&&integrated.visited<=200&&integrated.baseline_visit_budget+integrated.pair_visit_budget+integrated.relational_visit_budget==200,302);
  f.finish(5);f.e.relational_bounds_enabled=true;
  integrated=a::effort::interval_label(f.e,0,f.region,f.scratch,512,200);
  expect(report,integrated.static_label==0&&integrated.label==0&&!integrated.attempted&&!integrated.relational_attempted&&integrated.visited==0,303);
  shared_stumps(f);
  for(U capacity=0;capacity<12;++capacity) {
    result=relational_run(f,report,30,1000,capacity);
    expect(report,result.label<0&&result.pair_fallbacks>0,32);
  }
  // Other channel interleaving and unequal residual lengths retain every term.
  f.init(1,true,0,3);f.bias[2]=-2;
  f.tree(f.stump(0,0,0,2),0);f.tree(f.leaf(-1),2);f.tree(f.stump(0,0,0,2),1);f.tree(f.leaf(.25f),0);f.finish(1);
  result=relational_run(f,report,40);expect(report,result.label==0&&result.rivals_checked==2,42);exhaustive_relational(f,report,43,result);
  f.init();f.tree(f.stump(0,0,0,2),0);f.tree(f.stump(0,0,0,2),1);f.tree(f.leaf(.25f),1);f.finish(1);
  result=relational_run(f,report,44);expect(report,result.label==0,46);exhaustive_relational(f,report,47,result);
  // Already consumed roots must neither be paired nor added again; the exact
  // prefix includes their effects. The chosen winner need not be channel zero.
  f.init();f.tree(f.leaf(100),1);f.tree(f.stump(0,0,0,2),0);f.tree(f.stump(0,0,0,2),1);f.bias[1]=1;f.finish(0);f.residual[0]=-1;
  result=relational_run(f,report,50);expect(report,result.label==1&&result.pairs_complete==1,52);exhaustive_relational(f,report,53,result);
  // Opposite missing defaults defeat a correlation valid on finite inputs.
  f.init();f.tree(f.stump(0,0,0,2,true),0);f.tree(f.stump(0,0,0,2,false),1);f.finish(1);
  result=relational_run(f,report,60);expect(report,result.label<0,62);
  f.missing_allowed[0]=0;f.e.domain.allow_nan=0;
  result=relational_run(f,report,63);expect(report,result.label==0,65);exhaustive_relational(f,report,66,result);
  f.missing_allowed[0]=1;f.e.domain.allow_nan=1;f.lower[0]=a::domain::finite_min_key;f.upper[0]=a::domain::finite_min_key-1;
  f.range_lower[0]=f.range_upper[0]=1;f.range_lower[1]=f.range_upper[1]=2;
  result=a::relational::interval_label(f.e,0,f.region,f.scratch,512,1000);expect(report,result.label==1,69);exhaustive_relational(f,report,70,result);
  // Exactly-one groups, including a bit in the second mask word.
  for(U width=2;width<=65;width+=63) {
    f.init(width,false,width);f.tree(f.stump(0,.5f,0,2),0);f.tree(f.stump(0,.5f,0,2),1);
    f.tree(f.stump(width-1,.5f,0,2),0);f.tree(f.stump(width-1,.5f,0,2),1);f.finish(1);
    result=relational_run(f,report,80);expect(report,result.label==0,82);exhaustive_relational(f,report,83,result);
    f.allowed[0]=1;if(width>64)f.allowed[1]=1;
    result=relational_run(f,report,84);expect(report,result.label==0,86);exhaustive_relational(f,report,87,result);
  }
  // The 100-valued leaf violates its own ancestor predicate and must not lower
  // a complete feasible pair floor. Incumbent ranges are conditioned first.
  f.init();f.nodes=1;f.feature[0]=0;f.cut[0]=0;f.missing[0]=1;f.value[0]=0;
  f.left[0]=int(f.stump(0,1,0,100));f.right[0]=int(f.leaf(2));f.tree(0,1);f.tree(f.stump(0,0,0,2),0);f.finish(1);
  a::qualified_interval_label(f.e,0);f.range_upper[1]=2;
  result=a::relational::interval_label(f.e,0,f.region,f.scratch,512,10000);
  expect(report,result.label==0,90);exhaustive_relational(f,report,91,result);
  // A lower rival leaf under another input is a real counterexample, not an
  // impossible ancestor path; no certificate may discard it.
  f.init(2);f.tree(f.stump(0,0,0,2),0);f.tree(f.stump(1,0,0,2),1);f.finish(1);
  result=relational_run(f,report,93);expect(report,result.label<0,95);exhaustive_relational(f,report,96,result);
  // Different thresholds, features, and missing defaults exercise the shared
  // pair walk while checking every accepted margin against the ordered fold.
  for(U variant=0;variant<36;++variant) {
    f.init(2);const U root=f.nodes++;f.feature[root]=int(variant%2);f.cut[root]=float(int(variant%3)-1);f.value[root]=0;f.missing[root]=variant&1;
    f.left[root]=int(f.stump((variant+1)%2,0,-3,1,variant&2));f.right[root]=int(f.stump(variant%2,1,2,-1,variant&4));f.tree(root,0);
    f.tree(f.stump((variant/2)%2,float(int((variant/3)%3)-1),-2,3,variant&8),1);f.finish(1);
    result=relational_run(f,report,100+variant*4);exhaustive_relational(f,report,102+variant*4,result);
  }
  // Reassociation would make channel zero appear to win by one. Its actual
  // ordered RN32 fold ties, so the original-order error envelope must reject.
  f.init();f.tree(f.leaf(1),0);f.tree(f.leaf(-16777216.f),0);f.tree(f.leaf(0),1);f.finish(16777216.f);
  result=relational_run(f,report,250);expect(report,result.label<0,252);exhaustive_relational(f,report,253,result);
  // Subnormal residuals and cancellation still have a finite conservative ULP.
  f.init();f.tree(f.stump(0,0,__uint_as_float(1),-__uint_as_float(1)),0);f.tree(f.stump(0,0,__uint_as_float(1),-__uint_as_float(1)),1);f.finish(1);
  result=relational_run(f,report,260);expect(report,result.label==0,262);exhaustive_relational(f,report,263,result);
  auto rounding=a::relational::detail::channel_roundoff(f.e,f.words,f.residual,1);
  expect(report,rounding.valid&&rounding.error==0.5*double(__uint_as_float(1)),265);
  // RN-even ties at normal powers of two attain the half-spacing bound,
  // including a subnormal half-spacing operand at the smallest tested binade.
  const int exponents[5]={-125,-20,0,3,100};
  for(U i=0;i<5;++i)for(U sign=0;sign<2;++sign) {
    const int exponent=exponents[i],half_exponent=exponent-24;
    const U power_bits=U(exponent+127)<<23;
    const U half_bits=half_exponent==-149?1u:U(half_exponent+127)<<23;
    const float power=__uint_as_float(power_bits|(sign<<31));
    const float leaf=__uint_as_float(half_bits|(sign<<31));
    f.init();f.tree(f.leaf(leaf),0);f.finish(power);
    rounding=a::relational::detail::channel_roundoff(f.e,f.words,f.residual,0);
    const float actual=__fadd_rn(power,leaf);
    const double exact=__dadd_rn(double(power),double(leaf));
    const double error=fabs(__dsub_rn(double(actual),exact));
    expect(report,actual==power&&rounding.valid&&rounding.error==double(__uint_as_float(half_bits))&&error==rounding.error,304);
  }
  // At a binade boundary the predecessor spacing is smaller. Its midpoint
  // still rounds to the even power, with error below the outward half spacing.
  for(U sign=0;sign<2;++sign) {
    const float power=sign?-1.f:1.f,leaf=sign?0x1p-25f:-0x1p-25f;
    f.init();f.tree(f.leaf(leaf),0);f.finish(power);
    rounding=a::relational::detail::channel_roundoff(f.e,f.words,f.residual,0);
    const float actual=__fadd_rn(power,leaf);
    const double exact=__dadd_rn(double(power),double(leaf));
    expect(report,actual==power&&rounding.valid&&rounding.error==0x1p-24&&
        fabs(__dsub_rn(double(actual),exact))==0x1p-25,305);
  }
  // Signed zeros and the least subnormal retain a positive 2^-150 allowance
  // in FP64. No exact-addition special case is used by this revision.
  for(U signs=0;signs<4;++signs) {
    const U prefix_bits=(signs&1)<<31,leaf_sign=(signs>>1)<<31;
    f.init();f.tree(f.leaf(__uint_as_float(leaf_sign)),0);f.finish(__uint_as_float(prefix_bits));
    rounding=a::relational::detail::channel_roundoff(f.e,f.words,f.residual,0);
    expect(report,rounding.valid&&rounding.error==0x1p-150&&
        __float_as_uint(__fadd_rn(__uint_as_float(prefix_bits),__uint_as_float(leaf_sign)))==
          (prefix_bits&leaf_sign),306);
    f.value[0]=__uint_as_float(leaf_sign|1u);f.finish(__uint_as_float(prefix_bits));
    rounding=a::relational::detail::channel_roundoff(f.e,f.words,f.residual,0);
    expect(report,rounding.valid&&rounding.error==0x1p-150&&
        __float_as_uint(__fadd_rn(__uint_as_float(prefix_bits),f.value[0]))==(leaf_sign|1u),307);
  }
  // A strict native-gap witness that the former full-spacing penalty misses.
  // The source scores still use their unchanged original RN32 additions.
  f.init();f.tree(f.stump(0,0,0,1),0);f.tree(f.stump(0,0,0,1),1);
  const float near_gap=__uint_as_float(0x38806000u); // 2^-14 + 3*2^-24
  f.finish(near_gap);
  expect(report,a::qualified_interval_label(f.e,0)<0&&f.range_lower[0]<f.range_upper[1],308);
  const auto winner_error=a::relational::detail::channel_roundoff(f.e,f.words,f.residual,0);
  const auto rival_error=a::relational::detail::channel_roundoff(f.e,f.words,f.residual,1);
  expect(report,winner_error.valid&&rival_error.valid&&winner_error.error==0x1p-24&&rival_error.error==0x1p-24,309);
  expect(report,__dsub_rd(double(near_gap),0x1p-22)<native_softprob_gap::computed_gap_minimum&&
      __dsub_rd(double(near_gap),0x1p-23)==native_softprob_gap::computed_gap_minimum+0x1p-24&&
      __float_as_uint(__fadd_rn(1.f,near_gap))==0x3f800202u,310);
  result=relational_run(f,report,311);expect(report,result.success&&result.label==0,313);
  exhaustive_relational(f,report,314,result);
  // A rounded endpoint at the finite maximum has no finite outward neighbor;
  // an intermediate overflow is rejected even if supplied final ranges fit.
  f.init();f.tree(f.leaf(0),0);f.finish(__uint_as_float(0x7f7fffffu));
  rounding=a::relational::detail::channel_roundoff(f.e,f.words,f.residual,0);expect(report,!rounding.valid,270);
  f.value[0]=__uint_as_float(0x7f7fffffu);f.finish(__uint_as_float(0x7f7fffffu));
  rounding=a::relational::detail::channel_roundoff(f.e,f.words,f.residual,0);expect(report,!rounding.valid,271);
  // Compare an original RN32 fold with directed FP64 enclosures of its exact
  // real sum. Values span signs, zero, subnormals, cancellation and exponents;
  // finite error bounds must contain the complete fold's accumulated error.
  U random=0x38194271u,valid_folds=0;
  for(U variant=0;variant<160;++variant) {
    f.init(1,false);
    for(U t=0;t<12;++t) {
      random=random*1664525u+1013904223u;
      const U exponent=(variant+t)%9==0?0u:((random>>24)%254u);
      const U bits=(random&0x807fffffu)|(exponent<<23);
      const float value=(variant%5==0&&t%2)?-f.value[t-1]:__uint_as_float(bits);
      f.tree(f.leaf(value),0);
    }
    f.finish(variant%3?1.f:-0.f);
    rounding=a::relational::detail::channel_roundoff(f.e,f.words,f.residual,0);
    if(rounding.valid) {
      ++valid_folds;
      float actual=__uint_as_float(f.words[0]);double exact_lo=actual,exact_hi=actual;
      for(U t=0;t<f.trees;++t){const float value=f.value[f.roots[t]];actual=__fadd_rn(actual,value);exact_lo=__dadd_rd(exact_lo,double(value));exact_hi=__dadd_ru(exact_hi,double(value));}
      expect(report,isfinite(actual)&&__dsub_ru(double(actual),exact_lo)<=rounding.error&&__dsub_ru(exact_hi,double(actual))<=rounding.error,275);
    }
  }
  expect(report,valid_folds>=100,276);
  // A tiny positive exact margin is below native qualification's required gap.
  f.init();f.tree(f.stump(0,0,0,2),0);f.tree(f.stump(0,0,0,2),1);f.finish(float(native_softprob_gap::computed_gap_minimum*0.5));
  result=relational_run(f,report,280);expect(report,result.label<0,282);
  // Exact threshold neighbors distinguish the new guard from a weakened test
  // oracle. Eligibility is still required even at or above the numerical gap.
  const float gap=float(native_softprob_gap::computed_gap_minimum);
  f.init();f.finish(nextafterf(gap,0.f));
  expect(report,a::qualified_interval_label(f.e,0)<0,315);
  f.finish(gap);expect(report,a::qualified_interval_label(f.e,0)==0,316);
  f.finish(nextafterf(gap,INFINITY));expect(report,a::qualified_interval_label(f.e,0)==0,317);
  f.e.qualified_gap=false;expect(report,a::qualified_interval_label(f.e,0)<0,318);
  // Every channel must satisfy the authentic range gate before candidate choice.
  shared_stumps(f);a::qualified_interval_label(f.e,0);f.e.qualified_gap=false;
  result=a::relational::interval_label(f.e,0,f.region,f.scratch,512,1000);expect(report,!result.attempted&&result.label<0,290);f.e.qualified_gap=true;
  const float invalid[4]={11.f,__uint_as_float(0x7f800000u),__uint_as_float(0x7fc00001u),-11.f};
  for(U i=0;i<4;++i){a::qualified_interval_label(f.e,0);f.range_upper[1]=invalid[i];result=a::relational::interval_label(f.e,0,f.region,f.scratch,512,1000);expect(report,!result.attempted&&result.label<0,291);}
  *output=report;
}

int main(){try {
  Report* device=nullptr;ck(cudaMalloc(reinterpret_cast<void**>(&device),sizeof(Report)));
  relational_checks<<<1,1>>>(device);finish();Report report;ck(cudaMemcpy(&report,device,sizeof(report),cudaMemcpyDeviceToHost));ck(cudaFree(device));
  if(report.failures){std::cerr<<"relational bounds checks failed: "<<report.failures<<" first="<<report.first_failure<<'\n';return 1;}
  std::cout<<"{\"complete\":true,\"CUDA_executed\":true,\"checks\":"<<report.checks<<",\"failures\":0,\"native_RuntimeGate_qualified\":false,\"shared_stumps_relational_closed\":true,\"shared_stumps_independent_and_same_channel_open\":true}\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
