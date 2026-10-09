#pragma once
#include "adaptive_effort.cuh"

namespace class_conversion_adaptive::cover {
enum class Rejection : u32 {
  none=0, gate_unavailable, budget_disabled, invalid_state, invalid_predicate,
  invalid_region, invalid_scratch, uncertified_case, differing_labels,
  no_feasible_case, invalid_source, visit_budget_violation,
  candidate_refuted, target_gap_refuted, qualified_points_differ
};
struct Result {
  std::int32_t label=-1, branch_labels[2]={-1,-1};
  Rejection rejection=Rejection::none;
  u32 attempted=0, success=0, feasible_cases=0, certified_cases=0;
  u32 visited=0, visit_budget=0;
  u32 rival_covers=0, rivals_certified=0;
  u64 refined_roots=0, tightened_roots=0, rejected_refinements=0, fallback_frontiers=0;
};
static_assert(sizeof(Result)==80);
// Observation only; never part of state keys, native authority or Result80 ABI.
struct PointScreen {
  u32 attempted=0,intersection_ready=0,first_complete=0,first_qualified=0;
  u32 second_complete=0,second_qualified=0,mixed=0,inconclusive=0;
  u64 first_visits=0,second_visits=0;
};

namespace detail {
struct Span { const void* pointer; u64 bytes; };
__device__ inline bool overlap(Span a,Span b) {
  if(!a.pointer||!b.pointer||!a.bytes||!b.bytes)return false;
  const auto x=reinterpret_cast<std::uintptr_t>(a.pointer);
  const auto y=reinterpret_cast<std::uintptr_t>(b.pointer);
  return x<=y ? u64(y-x)<a.bytes : u64(x-y)<b.bytes;
}
__device__ inline bool private_scratch(EngineView e,domain::RegionView original,
    domain::RegionView scratch,u32* stack,u32 capacity) {
  const u64 numeric=u64(e.domain.numeric_features)*sizeof(u32);
  const u64 masks=u64(e.domain.mask_words)*sizeof(u64);
  if((numeric&&(!scratch.lower||!scratch.upper||!scratch.missing))||
     (masks&&!scratch.allowed)||!e.range_lower||!e.range_upper||
     (capacity&&!stack))return false;
  const Span output[]={{scratch.lower,numeric},{scratch.upper,numeric},
    {scratch.missing,numeric},{scratch.allowed,masks},
    {stack,u64(capacity)*sizeof(u32)},
    {e.range_lower,u64(e.source.classes)*sizeof(float)},
    {e.range_upper,u64(e.source.classes)*sizeof(float)}};
  const Span parent[]={{original.lower,numeric},{original.upper,numeric},
    {original.missing,numeric},{original.allowed,masks},
    {e.arena.words,u64(e.arena.state_capacity)*e.source.classes*sizeof(u32)},
    {e.arena.positions,u64(e.arena.state_capacity)*e.source.classes*sizeof(u32)},
    {e.arena.residual,u64(e.arena.state_capacity)*e.source.trees*sizeof(std::int32_t)}};
  for(u32 i=0;i<7;++i){
    for(u32 j=0;j<i;++j)if(overlap(output[i],output[j]))return false;
    for(const auto& p:parent)if(overlap(output[i],p))return false;
  }
  return true;
}
// Same endpoint and strict-margin preconditions as qualified_range_label.
// All channels must stay inside the qualified native softprob range.
__device__ inline bool beats(EngineView e,u32 winner,u32 rival) {
  if(!e.qualified_gap||winner>=e.source.classes||rival>=e.source.classes||winner==rival)return false;
  for(u32 c=0;c<e.source.classes;++c) {
    const auto lo=e.range_lower[c],hi=e.range_upper[c];
    if(!isfinite(lo)||!isfinite(hi)||lo>hi||lo < -10.f||hi > 10.f)return false;
  }
  return __dsub_rn(double(e.range_lower[winner]),double(e.range_upper[rival]))>=native_softprob_gap::computed_gap_minimum;
}
} // namespace detail

// Caller supplies an immutable validated source/state and exclusive scratch.
// Scratch, ranges and stack must be disjoint from ALL source/arena storage;
// direct parent/payload overlap and scratch overlap are additionally refused.
// Each successful region_restrict partitions the ORIGINAL region using the
// exact source predicate, including missing routing and one-hot constraints.
// Prefix words, positions and residual roots are never normalized or changed.
// The unchanged effort rule folds every residual in original RN32 source order.
// Exhausted visits/stack use existing conservative static enclosures. Every
// feasible side must certify the same label; no incomplete side is omitted.
// This result does not create native RuntimeGate authority.
__device__ inline Result interval_label(EngineView e,u32 id,
    domain::RegionView original,u32 predicate,domain::RegionView private_region,
    u32* private_stack,u32 stack_capacity,u32 global_visit_budget,
    int pair_winner=-1,int pair_rival=-1) {
  Result out;out.visit_budget=global_visit_budget;
  if(!e.qualified_gap){out.rejection=Rejection::gate_unavailable;return out;}
  if(!global_visit_budget){out.rejection=Rejection::budget_disabled;return out;}
  if(id>=e.arena.state_capacity||!e.arena.states||!e.arena.words||
     e.arena.states[id].phase==free_phase){out.rejection=Rejection::invalid_state;return out;}
  if(!e.source.classes||!e.source.features||e.source.features!=e.domain.features||
     !e.source.feature||!e.source.left||!e.source.right||!e.source.cut||
     !e.source.missing||(e.source.trees&&(!e.arena.residual||!e.source.channels||
       !e.minimum||!e.maximum))){out.rejection=Rejection::invalid_source;return out;}
  if(predicate>=e.source.nodes||e.source.left[predicate]<0||
     e.source.right[predicate]<0||u32(e.source.left[predicate])>=e.source.nodes||
     u32(e.source.right[predicate])>=e.source.nodes||e.source.feature[predicate]<0||
     u32(e.source.feature[predicate])>=e.source.features||e.source.missing[predicate]>1||
     domain::nan_word(__float_as_uint(e.source.cut[predicate]))){
    out.rejection=Rejection::invalid_predicate;return out;
  }
  if((e.domain.numeric_features&&(!original.lower||!original.upper||!original.missing))||
     (e.domain.mask_words&&!original.allowed)||!domain::region_valid(e.domain,original)){
    out.rejection=Rejection::invalid_region;return out;
  }
  if(!detail::private_scratch(e,original,private_region,private_stack,stack_capacity)){
    out.rejection=Rejection::invalid_scratch;return out;
  }
  for(u32 t=0;t<e.source.trees;++t){
    const auto root=e.arena.residual[u64(id)*e.source.trees+t];
    if((root>=0&&u32(root)>=e.source.nodes)||e.source.channels[t]<0||
       u32(e.source.channels[t])>=e.source.classes){
      out.rejection=Rejection::invalid_source;return out;
    }
  }
  const u32 feature=u32(e.source.feature[predicate]);
  const u32 cut=__float_as_uint(e.source.cut[predicate]);
  const bool default_left=bool(e.source.missing[predicate]);
  out.attempted=1;u32 remaining=global_visit_budget;
  for(u32 side=0;side<2;++side){
    if(!domain::region_split_feasible(e.domain,original,feature,cut,default_left,bool(side)))continue;
    ++out.feasible_cases;copy_region(e,private_region,original);
    if(!domain::region_restrict(e.domain,private_region,feature,cut,default_left,bool(side))){
      out.rejection=Rejection::invalid_region;return out;
    }
    const auto result=effort::interval_label(e,id,private_region,private_stack,stack_capacity,remaining);
    if(result.visited>remaining){out.rejection=Rejection::visit_budget_violation;return out;}
    remaining-=result.visited;out.visited+=result.visited;
    out.refined_roots+=result.refined_roots;out.tightened_roots+=result.tightened_roots;
    out.rejected_refinements+=result.rejected_refinements;
    out.fallback_frontiers+=result.fallback_frontiers;
    const int label=pair_winner<0 ? result.label :
      (pair_rival>=0&&detail::beats(e,u32(pair_winner),u32(pair_rival))?pair_winner:-1);
    out.branch_labels[side]=label;
    if(label>=0){++out.certified_cases;}
    else {
      // This particular cover cannot succeed after one uncertified side.
      // Count only evaluated cases; preserve an ordinary failure so another
      // proof strategy can use the unspent allowance on the original region.
      out.rejection=Rejection::uncertified_case;return out;
    }
  }
  if(!out.feasible_cases){out.rejection=Rejection::no_feasible_case;return out;}
  if(out.certified_cases!=out.feasible_cases){out.rejection=Rejection::uncertified_case;return out;}
  int common=-1;
  for(u32 side=0;side<2;++side)if(out.branch_labels[side]>=0){
    if(common>=0&&common!=out.branch_labels[side]){
      out.rejection=Rejection::differing_labels;return out;
    }
    common=out.branch_labels[side];
  }
  out.label=common;out.success=1;return out;
}
namespace detail {
__device__ inline bool valid_split(EngineView e,u32 node) {
  return node<e.source.nodes&&e.source.left[node]>=0&&e.source.right[node]>=0&&
    u32(e.source.left[node])<e.source.nodes&&u32(e.source.right[node])<e.source.nodes&&
    e.source.feature[node]>=0&&u32(e.source.feature[node])<e.source.features&&
    e.source.missing[node]<=1&&!domain::nan_word(__float_as_uint(e.source.cut[node]));
}
// Modes match region_witness(0/1): lower/upper finite endpoint and first/last
// whole-group bit, without allocating a feature-sized row. Missing-only stays NaN.
__device__ inline bool point_coordinate(EngineView e,domain::RegionView R,u32 feature,float& value,bool upper=false) {
  if(feature>=e.domain.features||!e.domain.feature_group||!e.domain.feature_numeric)return false;
  const auto group=e.domain.feature_group[feature];
  if(group>=0){
    if(u32(group)>=e.domain.groups||!e.domain.feature_bit||!e.domain.group_word_offsets||
       !e.domain.group_widths||!R.allowed)return false;
    const u32 start=e.domain.group_word_offsets[group],end=e.domain.group_word_offsets[group+1];
    const u32 width=e.domain.group_widths[group],bit=e.domain.feature_bit[feature];
    if(start>=end||end>e.domain.mask_words||!width||bit>=width)return false;
    if(upper){for(u32 word=end;word>start;){--word;if(R.allowed[word]){
      const u64 selected=u64(word-start)*64+63-u32(__clzll(R.allowed[word]));
      if(selected>=width)return false;value=float(selected==bit);return true;
    }}return false;}
    for(u32 word=start;word<end;++word)if(R.allowed[word]){
      const u64 selected=u64(word-start)*64+u32(__ffsll(static_cast<long long>(R.allowed[word]))-1);
      if(selected>=width)return false;value=float(selected==bit);return true;
    }
    return false;
  }
  const auto numeric=e.domain.feature_numeric[feature];
  if(numeric<0||u32(numeric)>=e.domain.numeric_features||!R.lower||!R.upper||!R.missing)return false;
  const u32 n=u32(numeric);
  if(domain::finite_nonempty(R.lower[n],R.upper[n])){
    value=__uint_as_float(domain::word_from_key(upper?R.upper[n]:R.lower[n]));return isfinite(value);
  }
  if(!R.missing[n])return false;value=__uint_as_float(0x7fc00001u);return true;
}
// Called only after interval_label validates source/state/scratch structure;
// additional point-only value and feature-map requirements are checked below.
// This is proposal selection only. A point never certifies any input region.
// The original saved prefix and all active residuals use their original RN32
// source order, including repeated leaves and consumed-root omission.
__device__ inline bool propose_class(EngineView e,u32 id,domain::RegionView R,
    u32 budget,u32& visited,u32& winner,bool upper=false) {
  visited=0;winner=none;
  if(!e.source.value||!e.source.classes||!e.range_lower||!e.arena.words||
     (e.source.trees&&(!e.source.channels||!e.arena.residual)))return false;
  const auto*roots=e.source.trees?e.arena.residual+u64(id)*e.source.trees:nullptr;
  const auto*words=e.arena.words+u64(id)*e.source.classes;
  for(u32 c=0;c<e.source.classes;++c){e.range_lower[c]=__uint_as_float(words[c]);if(!isfinite(e.range_lower[c]))return false;}
  for(u32 t=0;t<e.source.trees;++t){
    const auto channel=e.source.channels[t];if(channel<0||u32(channel)>=e.source.classes)return false;
    if(roots[t]<-1)return false;if(roots[t]<0)continue;u32 node=u32(roots[t]);
    for(;;){
      if(visited==budget)return false;++visited;
      if(node>=e.source.nodes)return false;
      if(e.source.left[node]<0){
        if(e.source.right[node]>=0||!isfinite(e.source.value[node]))return false;
        const float sum=__fadd_rn(e.range_lower[channel],e.source.value[node]);
        if(!isfinite(sum))return false;e.range_lower[channel]=sum;break;
      }
      if(!valid_split(e,node))return false;float coordinate;
      if(!point_coordinate(e,R,u32(e.source.feature[node]),coordinate,upper))return false;
      const bool left=isnan(coordinate)?bool(e.source.missing[node]):coordinate<e.source.cut[node];
      node=u32(left?e.source.left[node]:e.source.right[node]);
    }
  }
  winner=0;for(u32 c=1;c<e.source.classes;++c)if(e.range_lower[c]>e.range_lower[winner])winner=c;
  return true;
}
// Build I=W intersect R before either point is treated as a native witness.
// Check owning pointers BEFORE witness_region performs its state-offset arithmetic.
// Both inputs stay immutable; invalid/empty I confers no negative authority.
__device__ inline bool witness_intersection(EngineView e,u32 id,domain::RegionView R,
    domain::RegionView scratch,u32* stack,u32 capacity,bool* distinct=nullptr) {
  if(distinct)*distinct=false;
  const u32 N=e.domain.numeric_features,W=e.domain.mask_words;
  if(id>=e.arena.state_capacity||(N&&(!e.arena.witness_lower||!e.arena.witness_upper||
      !e.arena.witness_missing))||(W&&!e.arena.witness_allowed))return false;
  const auto witness=witness_region(e,id);
  if(!domain::region_valid(e.domain,R)||!domain::region_valid(e.domain,witness)||
     !private_scratch(e,R,scratch,stack,capacity)||
     !private_scratch(e,witness,scratch,stack,capacity))return false;
  bool different=false;u64 allowed_choices=0;
  for(u32 n=0;n<N;++n){
    scratch.lower[n]=max(R.lower[n],witness.lower[n]);
    scratch.upper[n]=min(R.upper[n],witness.upper[n]);
    scratch.missing[n]=R.missing[n]&witness.missing[n];
    different|=scratch.lower[n]<scratch.upper[n];
  }
  for(u32 w=0;w<W;++w){scratch.allowed[w]=R.allowed[w]&witness.allowed[w];
    allowed_choices+=u32(__popcll(scratch.allowed[w]));}
  const bool valid=domain::region_valid(e.domain,scratch);
  // A valid group has >=1 bit; total choices>groups iff some group has >=2.
  // Finite+NaN endpoints still both choose finite, so NaN alone adds no difference.
  if(valid&&distinct)*distinct=different||allowed_choices>e.domain.groups;
  return valid;
}
__device__ inline int qualified_point_label(EngineView e) {
  // propose_class completed ALL original-order RN32 channels in range_lower.
  // Stale common-cover upper endpoints cannot enter this singleton certificate.
  for(u32 c=0;c<e.source.classes;++c)e.range_upper[c]=e.range_lower[c];
  return qualified_range_label(e);
}
// An iterative proof cover, not a graph constructor. The stack tail holds only
// (predicate,side) path frames; effort owns the disjoint remaining prefix.
// Reconstructing each cell from the immutable original keeps numeric missing
// values and arbitrary-width exactly-one masks under the existing domain rules.
// Source/pair visits retain their existing counter meaning. Region-copy and
// path-restriction work is bounded by the finite path stack and proof visits;
// its measured execution time remains part of the controller's observed cost.
__device__ inline Result rival_cover(EngineView e,u32 id,domain::RegionView original,
    domain::RegionView private_region,u32* stack,u32 capacity,u32 budget,u32 winner,u32 rival) {
  Result out;out.visit_budget=budget;
  if(winner>=e.source.classes||rival>=e.source.classes||winner==rival){out.rejection=Rejection::invalid_source;return out;}
  if(!stack||!capacity){out.rejection=Rejection::invalid_scratch;return out;}
  if(!budget){out.rejection=Rejection::budget_disabled;return out;}
  out.attempted=1;u32 remaining=budget,depth=0;copy_region(e,private_region,original);
  for(;;){
    const u32 work_capacity=capacity-2*depth;
    const auto proof=effort::interval_label(e,id,private_region,stack,work_capacity,remaining,winner,rival);
    if(proof.visited>remaining){out.rejection=Rejection::visit_budget_violation;return out;}
    remaining-=proof.visited;out.visited+=proof.visited;++out.feasible_cases;
    out.refined_roots+=proof.refined_roots;out.tightened_roots+=proof.tightened_roots;
    out.rejected_refinements+=proof.rejected_refinements;out.fallback_frontiers+=proof.fallback_frontiers;
    // Preserve a stronger whole-class proof from an optional proof family,
    // even if the marginal interval buffers by themselves still overlap.
    if(proof.label>=0&&u32(proof.label)!=winner){out.rejection=Rejection::candidate_refuted;return out;}
    if(proof.label==int(winner)||beats(e,winner,rival)){
      ++out.certified_cases;bool pending=false;
      // Complete left before right. Pop a frame only after its right subtree
      // has also completed; no unresolved branch can disappear at unwind.
      while(depth){const u32 frame=capacity-2*depth;
        if(stack[frame+1]==0){stack[frame+1]=1;pending=true;break;}--depth;
      }
      if(!pending){out.label=int(winner);out.success=1;return out;}
    }else{
      // Sound opposing enclosures rule out this positive-gap proof on the
      // nonempty cell. This is not a mixed-native-class certificate: a tie
      // can still have one native class. Whole-class proofs were handled above.
      const float wl=e.range_lower[winner],wu=e.range_upper[winner];
      const float rl=e.range_lower[rival],ru=e.range_upper[rival];
      if(isfinite(wl)&&isfinite(wu)&&isfinite(rl)&&isfinite(ru)&&
         wl<=wu&&rl<=ru&&wu<=rl){
        out.rejection=Rejection::target_gap_refuted;return out;
      }
      const u32 selected=proof.selected_predicate;
      if(!remaining||work_capacity<3||selected==none){out.rejection=Rejection::uncertified_case;return out;}
      if(!valid_split(e,selected)){out.rejection=Rejection::invalid_predicate;return out;}
      const u32 feature=u32(e.source.feature[selected]),cut=__float_as_uint(e.source.cut[selected]);
      const bool missing=bool(e.source.missing[selected]);
      if(!domain::region_split_feasible(e.domain,private_region,feature,cut,missing,false)||
         !domain::region_split_feasible(e.domain,private_region,feature,cut,missing,true)){
        out.rejection=Rejection::uncertified_case;return out;
      }
      ++depth;const u32 frame=capacity-2*depth;stack[frame]=selected;stack[frame+1]=0;
    }
    copy_region(e,private_region,original);
    for(u32 level=1;level<=depth;++level){const u32 frame=capacity-2*level,predicate=stack[frame];
      if(!domain::region_restrict(e.domain,private_region,u32(e.source.feature[predicate]),
          __float_as_uint(e.source.cut[predicate]),bool(e.source.missing[predicate]),bool(stack[frame+1]))){
        out.rejection=Rejection::invalid_region;return out;
      }
    }
  }
}
} // namespace detail

// Each rival may use a distinct exhaustive proof partition of the same immutable
// region. Their partitions are neither multiplied nor published in the graph.
// Repeated finite-cover composition and RegionEnvelope.rival_specific_covers
// retain the unchanged native gate. Work/stack capacity bound available depth.
__device__ inline Result portfolio_label(EngineView e,u32 id,
    domain::RegionView original,u32 predicate,domain::RegionView private_region,
    u32* private_stack,u32 stack_capacity,u32 global_visit_budget) {
  const bool rivals=e.rival_cover_enabled&&e.source.classes>1;
  if(e.point_screen)*e.point_screen=PointScreen{};
  // Retain the full incumbent cover budget; only unused work feeds rivals.
  const u32 common_budget=global_visit_budget;
  auto out=interval_label(e,id,original,predicate,private_region,private_stack,stack_capacity,common_budget);
  out.visit_budget=global_visit_budget;
  // Differing fully certified, feasible sides are a proof that this region
  // is mixed. No deeper cover can certify a common class, so retain that
  // negative result without spending proposal or rival-search work.
  if(!rivals||out.success||!out.attempted||
     out.rejection!=Rejection::uncertified_case)return out;
  if(out.visited>global_visit_budget){out.rejection=Rejection::visit_budget_violation;return out;}
  u32 remaining=global_visit_budget-out.visited;
  const bool screen=e.two_point_screen_enabled&&e.point_screen;
  bool intersection=false,distinct=false;
  if(screen){e.point_screen->attempted=1;e.point_screen->inconclusive=1;
    intersection=detail::witness_intersection(e,id,original,private_region,private_stack,stack_capacity,&distinct);
    e.point_screen->intersection_ready=intersection;
  }
  const auto proposal_region=intersection?private_region:original;
  u32 winner=none,proposal_visits=0;
  const bool proposed=detail::propose_class(e,id,proposal_region,remaining,proposal_visits,winner);
  if(screen){e.point_screen->first_visits=proposal_visits;e.point_screen->first_complete=proposed;}
  if(proposal_visits>remaining){out.rejection=Rejection::visit_budget_violation;return out;}
  remaining-=proposal_visits;out.visited+=proposal_visits;
  if(!proposed){out.rejection=Rejection::uncertified_case;return out;}
  // The first raw proposal is retained in winner; the second never changes it.
  if(intersection){
    const int first=detail::qualified_point_label(e);e.point_screen->first_qualified=first>=0;
    if(first>=0&&distinct){
      u32 second_visits=0,unused=none;
      const bool completed=detail::propose_class(e,id,private_region,remaining,second_visits,unused,true);
      e.point_screen->second_visits=second_visits;e.point_screen->second_complete=completed;
      if(second_visits>remaining){out.rejection=Rejection::visit_budget_violation;return out;}
      remaining-=second_visits;out.visited+=second_visits;
      if(completed){const int second=detail::qualified_point_label(e);e.point_screen->second_qualified=second>=0;
        if(second>=0&&second!=first){
          e.point_screen->mixed=1;e.point_screen->inconclusive=0;
          out.rejection=Rejection::qualified_points_differ;return out;
        }
      }
    }
  }
  // Reserve work only for rivals that actually need a cover. Static-certified
  // rivals, including those later in class order, must not dilute the budget.
  qualified_interval_label(e,id);u32 unresolved=0;
  // Local certificates remain valid while nested calls reuse range scratch.
  // One hardware word caches the common case without allocation; larger class
  // counts retain the same general recomputation path and have no class limit.
  const bool cache_static=e.source.classes<=64;u64 static_rivals=0;
  for(u32 rival=0;rival<e.source.classes;++rival)if(rival!=winner){
    const bool proven=detail::beats(e,winner,rival);
    if(proven){if(cache_static)static_rivals|=u64(1)<<rival;}
    else ++unresolved;
  }
  for(u32 rival=0;rival<e.source.classes;++rival)if(rival!=winner) {
    bool proven;
    if(cache_static)proven=bool((static_rivals>>rival)&1u);
    else {qualified_interval_label(e,id);proven=detail::beats(e,winner,rival);}
    if(proven){++out.rivals_certified;continue;}
    if(!remaining){out.rejection=Rejection::uncertified_case;return out;}
    const u32 budget=remaining/unresolved ? remaining/unresolved : remaining;
    const auto pair=detail::rival_cover(e,id,original,private_region,private_stack,
                                      stack_capacity,budget,winner,rival);
    ++out.rival_covers;
    if(pair.visited>remaining){out.rejection=Rejection::visit_budget_violation;return out;}
    remaining-=pair.visited;out.visited+=pair.visited;
    out.feasible_cases+=pair.feasible_cases;out.certified_cases+=pair.certified_cases;
    out.refined_roots+=pair.refined_roots;out.tightened_roots+=pair.tightened_roots;
    out.rejected_refinements+=pair.rejected_refinements;out.fallback_frontiers+=pair.fallback_frontiers;
    if(!pair.success){out.rejection=pair.rejection;return out;}
    ++out.rivals_certified;--unresolved;
  }
  out.label=int(winner);out.success=1;out.rejection=Rejection::none;return out;
}
} // namespace class_conversion_adaptive::cover
