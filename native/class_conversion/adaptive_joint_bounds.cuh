#pragma once
#include "adaptive_engine.cuh"

namespace class_conversion_adaptive::joint {
// A bounded depth-first cover of two residual trees. A frame records a source
// predicate and the active side, not a materialized Cartesian leaf product.
// Scratch is exclusively owned by this worker and measured in u32 words.
struct Frame { u32 first=0,second=0,phase=0,predicate=none; };
static_assert(sizeof(Frame)==4*sizeof(u32)&&alignof(Frame)==alignof(u32));
enum class Fallback : u32 {none=0,disabled,scratch,budget,invalid,nonfinite};
struct WalkResult {
  u32 visited=0,leaves=0,rejected_sides=0;
  bool complete=false;
  Fallback fallback=Fallback::none;
};
struct Result : WalkResult {
  float lower=0,upper=0;
  bool tightened=false;
};

// All earlier accepted path restrictions are feasible. A new predicate can
// only alter its own numeric coordinate or its entire exactly-one group.
// Replaying that projection avoids a full region copy in every stack frame.
__device__ inline bool path_feasible(EngineView e,domain::RegionView original,
    const Frame* stack,u32 depth,u32 feature) {
  const auto group=e.domain.feature_group[feature];
  if(group<0) {
    const u32 slot=u32(e.domain.feature_numeric[feature]);
    u32 lo=original.lower[slot],hi=original.upper[slot],missing=original.missing[slot];
    for(u32 i=0;i<depth;++i) {
      const u32 p=stack[i].predicate;
      if(u32(e.source.feature[p])!=feature)continue;
      const bool right=stack[i].phase==2;
      if(domain::finite_nonempty(lo,hi)) {
        const u32 key=domain::sortable_word(__float_as_uint(e.source.cut[p]));
        if(right) {lo=max(lo,key);if(lo==domain::negative_zero_hole)lo=domain::zero_key;}
        else {hi=min(hi,domain::predecessor_key(key));if(hi==domain::negative_zero_hole)--hi;}
        if(!domain::finite_nonempty(lo,hi)){lo=domain::finite_min_key;hi=domain::finite_min_key-1;}
      }
      missing=missing&&(bool(e.source.missing[p])!=right);
    }
    return domain::finite_nonempty(lo,hi)||missing;
  }
  bool any=false;
  for(u32 w=e.domain.group_word_offsets[group];w<e.domain.group_word_offsets[group+1];++w) {
    u64 allowed=original.allowed[w];
    for(u32 i=0;i<depth&&allowed;++i) {
      const u32 p=stack[i].predicate,f=u32(e.source.feature[p]);
      if(e.domain.feature_group[f]!=group)continue;
      const bool right=stack[i].phase==2;
      const u32 bit=e.domain.feature_bit[f],word=e.domain.group_word_offsets[group]+bit/64;
      const u64 own=w==word?u64(1)<<(bit%64):0;
      const float cut=e.source.cut[p];
      const bool keep_one=(1.f<cut)==!right,keep_zero=(0.f<cut)==!right;
      allowed&=(keep_one?own:0)|(keep_zero?~own:0);
    }
    any|=allowed!=0;
  }
  return any;
}

// One shared path-cover traversal serves ordered sums and cross-class bounds.
// A consumer can omit a frontier only when it supplies a conservative bound for
// ALL leaves below it. The existing exhaustive-cover floor theorem applies to
// those omitted frontiers as well as the visited leaves.
template<class Consumer>
__device__ inline WalkResult walk_pairs(EngineView e,domain::RegionView original,
    u32 first,u32 second,u32* scratch,u32 scratch_words,u32 visit_budget,
    Consumer& consumer) {
  WalkResult out;
  if(first>=e.source.nodes||second>=e.source.nodes||
     !e.source.left||!e.source.right||!e.source.feature||!e.source.cut||!e.source.missing||
     !e.source.value||e.source.features!=e.domain.features||
     !domain::region_valid(e.domain,original)) {out.fallback=Fallback::invalid;return out;}
  if(!visit_budget){out.fallback=Fallback::disabled;return out;}
  const u32 capacity=scratch_words/4;
  if(!scratch||!capacity){out.fallback=Fallback::scratch;return out;}
  auto* stack=reinterpret_cast<Frame*>(scratch);
  u32 depth=1;stack[0]={first,second,0,none};
  while(depth) {
    auto& current=stack[depth-1];
    if(!current.phase) {
      if(out.visited==visit_budget){out.fallback=Fallback::budget;return out;}
      ++out.visited;
      if(consumer.skip(current.first,current.second)){--depth;continue;}
      const bool first_leaf=e.source.left[current.first]<0,second_leaf=e.source.left[current.second]<0;
      if(first_leaf&&second_leaf) {
        const float a=e.source.value[current.first],b=e.source.value[current.second];
        if(!isfinite(a)||!isfinite(b)||!consumer.add(a,b)){out.fallback=Fallback::nonfinite;return out;}
        ++out.leaves;--depth;continue;
      }
      current.predicate=first_leaf?current.second:current.first;
      const u32 p=current.predicate;
      if(e.source.feature[p]<0||u32(e.source.feature[p])>=e.source.features||
         e.source.right[p]<0||u32(e.source.left[p])>=e.source.nodes||
         u32(e.source.right[p])>=e.source.nodes||e.source.missing[p]>1||
         domain::nan_word(__float_as_uint(e.source.cut[p]))) {out.fallback=Fallback::invalid;return out;}
    }
    if(current.phase==2){--depth;continue;}
    const bool right=current.phase++==1;
    const u32 p=current.predicate;
    if(!path_feasible(e,original,stack,depth,u32(e.source.feature[p]))) {++out.rejected_sides;continue;}
    if(depth==capacity){out.fallback=Fallback::scratch;return out;}
    const u32 child=u32(right?e.source.right[p]:e.source.left[p]);
    const bool split_first=e.source.left[current.first]>=0;
    stack[depth++]={split_first?child:current.first,split_first?current.second:child,0,none};
  }
  if(!out.leaves){out.fallback=Fallback::invalid;return out;}
  out.complete=true;return out;
}
struct OrderedPair {
  float incoming_lower,incoming_upper,lower=0,upper=0;
  bool collected=false;
  __device__ bool skip(u32,u32) const {return false;}
  __device__ bool add(float a,float b) {
    const float lo=__fadd_rn(__fadd_rn(incoming_lower,a),b);
    const float hi=__fadd_rn(__fadd_rn(incoming_upper,a),b);
    if(!isfinite(lo)||!isfinite(hi)||lo>hi)return false;
    if(!collected){lower=lo;upper=hi;collected=true;}
    else {lower=fminf(lower,lo);upper=fmaxf(upper,hi);}
    return true;
  }
};

// Endpoints are incoming channel accumulator bounds. Every feasible leaf pair
// applies TWO ordered RN32 additions, including their intermediate rounding.
// On ANY incomplete traversal return the unchanged independent pair enclosure;
// a minimum/maximum over a partial leaf cover is never admitted.
__device__ inline Result pair_enclosure(EngineView e,domain::RegionView original,
    u32 first,u32 second,float incoming_lower,float incoming_upper,
    u32* scratch,u32 scratch_words,u32 visit_budget) {
  Result out;
  if(first>=e.source.nodes||second>=e.source.nodes||!e.minimum||!e.maximum||
     !isfinite(incoming_lower)||!isfinite(incoming_upper)||incoming_lower>incoming_upper){
    out.fallback=Fallback::invalid;return out;
  }
  out.lower=__fadd_rn(__fadd_rn(incoming_lower,__uint_as_float(e.minimum[first])),
      __uint_as_float(e.minimum[second]));
  out.upper=__fadd_rn(__fadd_rn(incoming_upper,__uint_as_float(e.maximum[first])),
      __uint_as_float(e.maximum[second]));
  if(!isfinite(out.lower)||!isfinite(out.upper)||out.lower>out.upper){out.fallback=Fallback::nonfinite;return out;}
  OrderedPair consumer{incoming_lower,incoming_upper};
  static_cast<WalkResult&>(out)=walk_pairs(e,original,first,second,scratch,scratch_words,visit_budget,consumer);
  if(!out.complete)return out;
  if(consumer.lower<out.lower||consumer.upper>out.upper||consumer.lower>consumer.upper){
    out.complete=false;out.fallback=Fallback::invalid;return out;
  }
  out.tightened=consumer.lower>out.lower||consumer.upper<out.upper;
  out.lower=consumer.lower;out.upper=consumer.upper;return out;
}
} // namespace class_conversion_adaptive::joint
