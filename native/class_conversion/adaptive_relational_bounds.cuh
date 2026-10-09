#pragma once
#include "adaptive_joint_bounds.cuh"

namespace class_conversion_adaptive::relational {
struct Result {
  int label=-1;
  u32 visited=0,pair_visits=0,pairs_attempted=0,pairs_complete=0,pair_fallbacks=0,rivals_checked=0;
  bool attempted=false,success=false;
};
namespace detail {
struct Roundoff { double error=0; bool valid=false; };
// Bound the error of every ORIGINAL RN32 addition, in channel order. The
// endpoints enclose its rounded output. Half the outward spacing at their
// largest magnitude bounds RN-even error on either side, including binade
// boundaries. Halve in FP64 so zero/subnormal spacing retains 2^-150.
// An infinite successor still cannot supply a finite bound.
__device__ inline Roundoff channel_roundoff(EngineView e,const u32* words,
    const std::int32_t* roots,u32 channel) {
  Roundoff out;
  float lo=__uint_as_float(words[channel]),hi=lo;
  if(!isfinite(lo))return out;
  for(u32 t=0;t<e.source.trees;++t)if(roots[t]>=0&&u32(e.source.channels[t])==channel) {
    const u32 root=u32(roots[t]);
    const float a=__uint_as_float(e.minimum[root]),b=__uint_as_float(e.maximum[root]);
    if(!isfinite(a)||!isfinite(b)||a>b)return out;
    lo=__fadd_rn(lo,a);hi=__fadd_rn(hi,b);
    if(!isfinite(lo)||!isfinite(hi)||lo>hi)return out;
    const float magnitude=fmaxf(fabsf(lo),fabsf(hi));
    const float next=nextafterf(magnitude,__uint_as_float(0x7f800000u));
    if(!isfinite(next))return out;
    const double radius=__dmul_ru(0.5,__dsub_ru(double(next),double(magnitude)));
    out.error=__dadd_ru(out.error,radius);
  }
  out.valid=isfinite(out.error);return out;
}
struct DifferenceFloor {
  EngineView e;
  double lower=0;
  bool collected=false;
  __device__ bool skip(u32 first,u32 second) const {
    if(!collected)return false;
    const float a=__uint_as_float(e.minimum[first]),b=__uint_as_float(e.maximum[second]);
    return isfinite(a)&&isfinite(b)&&__dsub_rd(double(a),double(b))>=lower;
  }
  __device__ bool add(float a,float b) {
    const double value=__dsub_rd(double(a),double(b));
    if(!isfinite(value))return false;
    if(!collected||value<lower)lower=value;
    collected=true;return true;
  }
};
__device__ inline u32 next_root(EngineView e,const std::int32_t* roots,u32 at,u32 channel) {
  while(at<e.source.trees&&(roots[at]<0||u32(e.source.channels[at])!=channel))++at;
  return at;
}
} // namespace detail

// Consume the incumbent's qualified ranges without changing them. Pair the kth
// active winner/rival trees over the SAME feasible input, but retain every
// channel's original RN32 sequence through its separate roundoff envelope.
// Only the exact-real bookkeeping sum is regrouped. Any incomplete pair uses
// its full static floor; a partial cover's observed minimum is never admitted.
__device__ inline Result interval_label(EngineView e,u32 id,domain::RegionView R,
    u32* scratch,u32 scratch_words,u32 visit_budget) {
  Result out;
  if(!e.qualified_gap||!visit_budget||e.source.classes<2||!e.range_lower||!e.range_upper||
     !e.arena.words||id>=e.arena.state_capacity||
     (e.source.trees&&(!e.arena.residual||!e.source.channels||!e.minimum||!e.maximum))||
     !domain::region_valid(e.domain,R))return out;
  u32 winner=0;
  for(u32 c=0;c<e.source.classes;++c) {
    const float lo=e.range_lower[c],hi=e.range_upper[c];
    if(!isfinite(lo)||!isfinite(hi)||lo>hi||lo<-10.f||hi>10.f)return out;
    if(lo>e.range_lower[winner])winner=c;
  }
  const u32* words=e.arena.words+u64(id)*e.source.classes;
  const auto* roots=e.source.trees?e.arena.residual+u64(id)*e.source.trees:nullptr;
  for(u32 t=0;t<e.source.trees;++t)
    if(e.source.channels[t]<0||u32(e.source.channels[t])>=e.source.classes||
       (roots[t]>=0&&u32(roots[t])>=e.source.nodes))return out;
  const auto winner_roundoff=detail::channel_roundoff(e,words,roots,winner);
  if(!winner_roundoff.valid)return out;
  out.attempted=true;
  u32 remaining=visit_budget;
  for(u32 rival=0;rival<e.source.classes;++rival)if(rival!=winner) {
    ++out.rivals_checked;
    // This rival may already be excluded by a tighter incumbent enclosure.
    if(__dsub_rd(double(e.range_lower[winner]),double(e.range_upper[rival]))>=native_softprob_gap::computed_gap_minimum)continue;
    const auto rival_roundoff=detail::channel_roundoff(e,words,roots,rival);
    if(!rival_roundoff.valid)return out;
    double floor=__dsub_rd(double(__uint_as_float(words[winner])),double(__uint_as_float(words[rival])));
    u32 w=detail::next_root(e,roots,0,winner),r=detail::next_root(e,roots,0,rival);
    while(w<e.source.trees||r<e.source.trees) {
      double term;
      if(w<e.source.trees&&r<e.source.trees) {
        const u32 first=u32(roots[w]),second=u32(roots[r]);
        const double fallback=__dsub_rd(double(__uint_as_float(e.minimum[first])),double(__uint_as_float(e.maximum[second])));
        term=fallback;
        if(remaining) {
          ++out.pairs_attempted;
          detail::DifferenceFloor consumer{e};
          const auto walk=joint::walk_pairs(e,R,first,second,scratch,scratch_words,remaining,consumer);
          remaining-=walk.visited;out.visited+=walk.visited;out.pair_visits+=walk.visited;
          if(walk.complete&&consumer.collected&&consumer.lower>=fallback) {
            term=consumer.lower;++out.pairs_complete;
          } else ++out.pair_fallbacks;
        } else ++out.pair_fallbacks;
        w=detail::next_root(e,roots,w+1,winner);r=detail::next_root(e,roots,r+1,rival);
      } else if(w<e.source.trees) {
        term=double(__uint_as_float(e.minimum[roots[w]]));
        w=detail::next_root(e,roots,w+1,winner);
      } else {
        term=-double(__uint_as_float(e.maximum[roots[r]]));
        r=detail::next_root(e,roots,r+1,rival);
      }
      floor=__dadd_rd(floor,term);
    }
    floor=__dsub_rd(__dsub_rd(floor,winner_roundoff.error),rival_roundoff.error);
    if(!isfinite(floor)||floor<native_softprob_gap::computed_gap_minimum)return out;
  }
  out.label=int(winner);out.success=true;return out;
}
} // namespace class_conversion_adaptive::relational
