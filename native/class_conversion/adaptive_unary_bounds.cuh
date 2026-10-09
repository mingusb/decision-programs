#pragma once
#include "adaptive_relational_bounds.cuh"

namespace class_conversion_adaptive::unary {
// Per-tree certificates belong to this exact region. Cuts, order, cached probes
// and the private stack are writable; none may alias each other or certificates/source.
struct Scratch {
  u32 *axes=nullptr,*minimum=nullptr,*maximum=nullptr,*cuts=nullptr;
  u32 trees=0,cut_capacity=0;
  u32* order=nullptr;u32 order_capacity=0;
  double* first=nullptr;u32 first_capacity=0;
};
struct Result {
  int label=-1;
  u32 visited=0,groups=0,completed=0,fallbacks=0; // completed includes proved no-improvement
  bool attempted=false,success=false,optimistic_rejected=false;
};
namespace detail {
__device__ inline u32 feature_axis(EngineView e,u32 feature) {
  const auto group=e.domain.feature_group[feature],numeric=e.domain.feature_numeric[feature];
  return group>=0?e.domain.numeric_features+u32(group):(numeric>=0?u32(numeric):none);
}
__device__ inline u32 tree_axis(EngineView e,Scratch s,u32 tree) {
  if(!s.axes||tree>=s.trees)return none;
  const u32 axis=s.axes[tree];
  return u64(axis)<u64(e.domain.numeric_features)+e.domain.groups?axis:none;
}
// A single active-ID permutation serves every rival. The source index breaks
// axis ties, preserving the existing signed accumulation order within a group.
__device__ inline bool before(EngineView e,Scratch s,u32 a,u32 b) {
  const u32 x=tree_axis(e,s,a),y=tree_axis(e,s,b);
  return x<y||(x==y&&a<b);
}
__device__ inline void sift(EngineView e,Scratch s,u32 root,u32 count) {
  while(root<count/2) {
    u32 child=2*root+1;
    if(child+1<count&&before(e,s,s.order[child],s.order[child+1]))++child;
    if(!before(e,s,s.order[root],s.order[child]))break;
    const u32 item=s.order[root];s.order[root]=s.order[child];s.order[child]=item;
    root=child;
  }
}
__device__ inline bool order_members(EngineView e,const std::int32_t* roots,Scratch s,u32& count) {
  count=0;
  for(u32 t=0;t<e.source.trees;++t)if(roots[t]>=0&&tree_axis(e,s,t)!=none) {
    if(!s.order||count==s.order_capacity)return false;
    s.order[count++]=t;
  }
  for(u32 start=count/2;start;--start)sift(e,s,start-1,count);
  for(u32 end=count;end>1;) {
    --end;const u32 item=s.order[0];s.order[0]=s.order[end];s.order[end]=item;
    sift(e,s,0,end);
  }
  // none sorts last; these IDs are already ascending in the original source.
  for(u32 t=0;t<e.source.trees;++t)if(roots[t]>=0&&tree_axis(e,s,t)==none) {
    if(!s.order||count==s.order_capacity)return false;
    s.order[count++]=t;
  }
  return true;
}
__device__ inline bool relevant(EngineView e,const std::int32_t* roots,u32 tree,u32 winner,u32 rival) {
  return roots[tree]>=0&&(u32(e.source.channels[tree])==winner||u32(e.source.channels[tree])==rival);
}
__device__ inline double independent(EngineView e,const std::int32_t* roots,Scratch s,u32 tree,u32 winner) {
  const u32 root=u32(roots[tree]);
  float lo=__uint_as_float(e.minimum[root]),hi=__uint_as_float(e.maximum[root]);
  if(s.minimum&&s.maximum&&tree<s.trees) {
    const float a=__uint_as_float(s.minimum[tree]),b=__uint_as_float(s.maximum[tree]);
    if(isfinite(a)&&isfinite(b)&&lo<=a&&a<=b&&b<=hi){lo=a;hi=b;}
  }
  return u32(e.source.channels[tree])==winner?double(lo):-double(hi);
}
__device__ inline bool predicate_valid(EngineView e,u32 node) {
  return e.source.feature[node]>=0&&u32(e.source.feature[node])<e.source.features&&
    e.source.right[node]>=0&&u32(e.source.left[node])<e.source.nodes&&
    u32(e.source.right[node])<e.source.nodes&&e.source.missing[node]<=1&&
    !domain::nan_word(__float_as_uint(e.source.cut[node]));
}
__device__ inline bool visit(Result& out,u32 budget) {
  if(out.visited==budget)return false;
  ++out.visited;return true;
}
// Collect the COMPLETE union of numeric thresholds after original-region
// forced branches. The lower endpoint is evaluated separately and uses no cut
// slot. Duplicate cuts, +/-0, and cuts outside the finite interval add no atom.
__device__ inline bool collect(EngineView e,domain::RegionView R,u32 root,u32 axis,
    u32* stack,u32 capacity,Scratch s,u32& cuts,Result& out,u32 budget) {
  if(!stack||!capacity)return false;
  u32 pending=1;stack[0]=root;
  while(pending) {
    if(!visit(out,budget))return false;
    const u32 node=stack[--pending];
    if(node>=e.source.nodes)return false;
    if(e.source.left[node]<0){if(!isfinite(e.source.value[node]))return false;continue;}
    if(!predicate_valid(e,node))return false;
    const u32 f=u32(e.source.feature[node]);bool right=false;
    if(domain::forced_side(e.domain,R,f,__float_as_uint(e.source.cut[node]),bool(e.source.missing[node]),right)) {
      stack[pending++]=u32(right?e.source.right[node]:e.source.left[node]);continue;
    }
    if(feature_axis(e,f)!=axis)return false;
    if(axis<e.domain.numeric_features&&domain::finite_nonempty(R.lower[axis],R.upper[axis])) {
      const u32 key=domain::sortable_word(__float_as_uint(e.source.cut[node]));
      if(R.lower[axis]<key&&key<=R.upper[axis]) {
        bool duplicate=false;for(u32 i=0;i<cuts;++i)duplicate|=s.cuts[i]==key;
        if(!duplicate){if(!s.cuts||cuts==s.cut_capacity)return false;s.cuts[cuts++]=key;}
      }
    }
    if(capacity-pending<2)return false;
    stack[pending++]=u32(e.source.left[node]);stack[pending++]=u32(e.source.right[node]);
  }
  return true;
}
// One representative fixes the entire semantic axis. Other predicates must be
// forced by R, as certified by the complete original-region classification.
__device__ inline bool evaluate(EngineView e,domain::RegionView R,u32 root,u32 axis,
    u32 atom,bool missing,float& value,Result& out,u32 budget) {
  u32 node=root;
  while(true) {
    if(!visit(out,budget)||node>=e.source.nodes)return false;
    if(e.source.left[node]<0){value=e.source.value[node];return isfinite(value);}
    if(!predicate_valid(e,node))return false;
    const u32 f=u32(e.source.feature[node]);bool right=false;
    if(feature_axis(e,f)==axis) {
      if(axis<e.domain.numeric_features)
        right=missing?!bool(e.source.missing[node]):
          !(__uint_as_float(domain::word_from_key(atom))<e.source.cut[node]);
      else right=!(float(e.domain.feature_bit[f]==atom)<e.source.cut[node]);
    } else if(!domain::forced_side(e.domain,R,f,__float_as_uint(e.source.cut[node]),bool(e.source.missing[node]),right))return false;
    node=u32(right?e.source.right[node]:e.source.left[node]);
  }
}
__device__ inline bool atom_sum(EngineView e,domain::RegionView R,const std::int32_t* roots,
    const u32* members,u32 count,u32 axis,u32 winner,u32 rival,u32 atom,bool missing,
    double& sum,Result& out,u32 budget) {
  sum=0;
  for(u32 i=0;i<count;++i) {
    const u32 t=members[i];if(!relevant(e,roots,t,winner,rival))continue;
    float value;
    if(!evaluate(e,R,u32(roots[t]),axis,atom,missing,value,out,budget))return false;
    sum=__dadd_rd(sum,u32(e.source.channels[t])==winner?double(value):-double(value));
  }
  return isfinite(sum);
}
// The representative is shared by the cheap rejection probe and the full scan.
__device__ inline bool first_atom(EngineView e,domain::RegionView R,u32 axis,u32& atom,bool& missing) {
  atom=0;missing=false;
  if(axis<e.domain.numeric_features) {
    if(domain::finite_nonempty(R.lower[axis],R.upper[axis])){atom=R.lower[axis];return true;}
    missing=R.missing[axis];return missing;
  }
  const u32 group=axis-e.domain.numeric_features,start=e.domain.group_word_offsets[group];
  for(u32 word=start;word<e.domain.group_word_offsets[group+1];++word)if(R.allowed[word]) {
    atom=(word-start)*64+u32(__ffsll(static_cast<long long>(R.allowed[word]))-1);return true;
  }
  return false;
}
// lower starts as the independent floor for ALL group members. A first atom
// at or below it proves that the complete NUMERICAL scan cannot improve it;
// this is a completed bound, not an exhaustive scan or an exact-minimum claim.
// Otherwise commit only after complete cut/atom coverage. An interrupted scan
// retains the entire fallback; its first value is reused, never recomputed.
// A finite cached_first must be this group/rival representative's computed fold.
__device__ inline bool group_floor(EngineView e,domain::RegionView R,const std::int32_t* roots,
    Scratch s,const u32* members,u32 count,u32 axis,u32 winner,u32 rival,u32* stack,u32 capacity,
    double& lower,Result& out,u32 budget,double cached_first=INFINITY) {
  if(!stack||!capacity)return false;
  u32 first;bool first_missing;
  if(!first_atom(e,R,axis,first,first_missing))return false;
  double minimum=cached_first;
  if(!isfinite(minimum)&&!atom_sum(e,R,roots,members,count,axis,winner,rival,first,first_missing,minimum,out,budget))return false;
  if(minimum<=lower)return true;
  u32 cuts=0;
  for(u32 i=0;i<count;++i) {
    const u32 t=members[i];if(!relevant(e,roots,t,winner,rival))continue;
    if(!collect(e,R,u32(roots[t]),axis,stack,capacity,s,cuts,out,budget))return false;
  }
  auto absorb=[&](u32 atom,bool missing) {
    double sum;
    if(!atom_sum(e,R,roots,members,count,axis,winner,rival,atom,missing,sum,out,budget))return false;
    if(sum<minimum)minimum=sum;
    return true;
  };
  if(axis<e.domain.numeric_features) {
    if(domain::finite_nonempty(R.lower[axis],R.upper[axis])) {
      for(u32 i=0;i<cuts;++i)if(!absorb(s.cuts[i],false))return false;
    }
    if(R.missing[axis]&&!first_missing&&!absorb(0,true))return false;
  } else {
    const u32 group=axis-e.domain.numeric_features,start=e.domain.group_word_offsets[group];
    for(u32 word=start;word<e.domain.group_word_offsets[group+1];++word) {
      u64 bits=R.allowed[word];
      while(bits) {
        const u32 bit=u32(__ffsll(static_cast<long long>(bits))-1);
        const u32 atom=(word-start)*64+bit;
        if(atom!=first&&!absorb(atom,false))return false;
        bits&=bits-1;
      }
    }
  }
  if(!isfinite(minimum))return false;
  lower=minimum>lower?minimum:lower;return true;
}
} // namespace detail

// This bound groups the exact signed leaf contributions only. Prefix words and
// every original RN32 operation stay unchanged; the separate original-channel
// error envelopes account for rounding, including active constant residuals.
__device__ inline Result interval_label(EngineView e,u32 id,domain::RegionView R,
    u32* stack,u32 stack_capacity,Scratch s,u32 visit_budget) {
  Result out;
  if(!e.qualified_gap||!visit_budget||e.source.classes<2||!e.range_lower||!e.range_upper||
     !e.arena.words||id>=e.arena.state_capacity||
     (e.source.trees&&(!e.arena.residual||!e.source.channels||!e.minimum||!e.maximum||
       !e.source.feature||!e.source.left||!e.source.right||!e.source.cut||!e.source.missing||!e.source.value))||
     e.source.features!=e.domain.features||!domain::region_valid(e.domain,R))return out;
  u32 winner=0;
  for(u32 c=0;c<e.source.classes;++c) {
    const float lo=e.range_lower[c],hi=e.range_upper[c];
    if(!isfinite(lo)||!isfinite(hi)||lo>hi||lo<-10.f||hi>10.f)return out;
    if(lo>e.range_lower[winner])winner=c;
  }
  const u32* words=e.arena.words+u64(id)*e.source.classes;
  const auto* roots=e.source.trees?e.arena.residual+u64(id)*e.source.trees:nullptr;
  for(u32 t=0;t<e.source.trees;++t) {
    if(e.source.channels[t]<0||u32(e.source.channels[t])>=e.source.classes)return out;
    if(roots[t]>=0) {
      const u32 root=u32(roots[t]);if(root>=e.source.nodes)return out;
      const float lo=__uint_as_float(e.minimum[root]),hi=__uint_as_float(e.maximum[root]);
      if(!isfinite(lo)||!isfinite(hi)||lo>hi)return out;
    }
  }
  const auto winner_error=relational::detail::channel_roundoff(e,words,roots,winner);
  if(!winner_error.valid)return out;
  u32 members=0;
  if(!detail::order_members(e,roots,s,members))return out;
  u32 group_count=0,previous=none;
  for(u32 i=0;i<members;++i) {
    const u32 axis=detail::tree_axis(e,s,s.order[i]);
    if(axis!=none&&axis!=previous)++group_count;
    previous=axis;
  }
  const bool probe_groups=group_count&&s.first&&s.first_capacity>=group_count&&stack&&stack_capacity;
  out.attempted=true;
  for(u32 rival=0;rival<e.source.classes;++rival)if(rival!=winner) {
    if(__dsub_rd(double(e.range_lower[winner]),double(e.range_upper[rival]))>=native_softprob_gap::computed_gap_minimum)continue;
    const auto rival_error=relational::detail::channel_roundoff(e,words,roots,rival);
    if(!rival_error.valid)return out;
    const double prefix=__dsub_rd(double(__uint_as_float(words[winner])),double(__uint_as_float(words[rival])));
    if(probe_groups) {
      // These caps bound the COMPUTED routine, not the true score. Every cache
      // entry is cleared per rival so an interrupted pass cannot reuse stale work.
      for(u32 i=0;i<group_count;++i)s.first[i]=INFINITY;
      double cap=prefix;bool valid=true;u32 ordinal=0;
      for(u32 begin=0;begin<members;) {
        const u32 axis=detail::tree_axis(e,s,s.order[begin]);
        u32 end=begin+1;
        while(end<members&&detail::tree_axis(e,s,s.order[end])==axis)++end;
        double lower=0;bool included=false;
        for(u32 i=begin;i<end;++i) {
          const u32 t=s.order[i];if(!detail::relevant(e,roots,t,winner,rival))continue;
          const double term=detail::independent(e,roots,s,t,winner);
          if(axis==none)cap=__dadd_rd(cap,term);
          else lower=__dadd_rd(lower,term);
          included=true;
        }
        if(axis!=none) {
          if(included) {
            u32 atom;bool missing;double value;
            if(!detail::first_atom(e,R,axis,atom,missing)||
               !detail::atom_sum(e,R,roots,s.order+begin,end-begin,axis,winner,rival,
                                 atom,missing,value,out,visit_budget)){valid=false;break;}
            s.first[ordinal]=value;
            cap=__dadd_rd(cap,value>lower?value:lower);
          }
          ++ordinal;
        }
        begin=end;
      }
      cap=__dsub_rd(__dsub_rd(cap,winner_error.error),rival_error.error);
      if(valid&&isfinite(cap)&&cap<native_softprob_gap::computed_gap_minimum){out.optimistic_rejected=true;return out;}
    }
    double floor=prefix;u32 ordinal=0;
    for(u32 begin=0;begin<members;) {
      const u32 axis=detail::tree_axis(e,s,s.order[begin]);
      u32 end=begin+1;
      while(end<members&&detail::tree_axis(e,s,s.order[end])==axis)++end;
      double lower=0;bool included=false;
      for(u32 i=begin;i<end;++i) {
        const u32 t=s.order[i];if(!detail::relevant(e,roots,t,winner,rival))continue;
        const double term=detail::independent(e,roots,s,t,winner);
        if(axis==none)floor=__dadd_rd(floor,term);
        else lower=__dadd_rd(lower,term);
        included=true;
      }
      if(axis!=none&&included) {
        ++out.groups;
        if(detail::group_floor(e,R,roots,s,s.order+begin,end-begin,axis,winner,rival,
                              stack,stack_capacity,lower,out,visit_budget,probe_groups?s.first[ordinal]:INFINITY))++out.completed;
        else ++out.fallbacks;
        floor=__dadd_rd(floor,lower);
      }
      if(axis!=none)++ordinal;
      begin=end;
    }
    floor=__dsub_rd(__dsub_rd(floor,winner_error.error),rival_error.error);
    if(!isfinite(floor)||floor<native_softprob_gap::computed_gap_minimum)return out;
  }
  out.label=int(winner);out.success=true;return out;
}
} // namespace class_conversion_adaptive::unary
