#pragma once
#include "adaptive_engine.cuh"
#include "adaptive_joint_bounds.cuh"
#include "adaptive_relational_bounds.cuh"
#include "adaptive_unary_bounds.cuh"

namespace class_conversion_adaptive::effort {
// Caller owns the private stack, range buffers and runtime work policy. This
// result is observation data; it grants no native RuntimeGate authority.
struct Result {
  int label=-1,static_label=-1,baseline_label=-1;
  bool attempted=false,additional_prune=false;
  u32 visited=0,refined_roots=0,tightened_roots=0,rejected_refinements=0;
  u64 fallback_frontiers=0;
  u32 baseline_visit_budget=0,pair_visit_budget=0;
  u32 pair_attempts=0,pair_completed=0,pair_tightened=0,pair_fallbacks=0,pair_visits=0;
  bool pair_additional_prune=false;
  u32 relational_visit_budget=0,relational_visits=0,relational_pairs=0,relational_completed=0,relational_fallbacks=0;
  bool relational_attempted=false,relational_additional_prune=false;
  u32 unary_visit_budget=0,unary_visits=0,unary_groups=0,unary_completed=0,unary_fallbacks=0;
  bool unary_attempted=false,unary_additional_prune=false,unary_optimistic_rejected=false;
  u32 selected_predicate=none; // optional widest-factor proposal, not a proof result
};

// The same admitted residual state, original tree order and RN32 additions are
// retained. R never changes during a walk. Unvisited frontiers use their static
// enclosure, so exhausted work remains conservative. Both decisions call the
// engine's unchanged qualified_range_label rule under its authentic gate flag.
__device__ Result interval_label(EngineView e,u32 id,domain::RegionView R,
    u32* private_stack,u32 stack_capacity,u32 global_visit_budget,
    u32 selector_winner=none,u32 selector_rival=none) {
  Result out;
  if(!e.qualified_gap)return out;
  out.static_label=out.baseline_label=out.label=qualified_interval_label(e,id);
  if(out.label>=0||!global_visit_budget||!e.source.trees)return out;
  out.attempted=true;
  const u32* words=e.arena.words+u64(id)*e.source.classes;
  const auto* roots=e.arena.residual+u64(id)*e.source.trees;
  const bool select=selector_winner<e.source.classes&&selector_rival<e.source.classes&&selector_winner!=selector_rival;
  const bool classify=e.unary_bounds_enabled&&e.unary_axes&&e.unary_minimum&&e.unary_maximum&&e.unary_order;
  // A rival comparison needs tighter scores only for its two channels. The
  // entry pass already encloses every other original ordered score. If those
  // static endpoints meet the native range premise, retain them unchanged:
  // their trees need neither conditional walks nor a repeated rounded fold.
  // Optional whole-class families keep their existing metadata and work path.
  bool target_only=select&&!classify&&!e.relational_bounds_enabled;
  if(target_only)for(u32 c=0;c<e.source.classes;++c)
    if(c!=selector_winner&&c!=selector_rival){
      const float lo=e.range_lower[c],hi=e.range_upper[c];
      if(!isfinite(lo)||!isfinite(hi)||lo>hi||lo < -10.f||hi > 10.f){target_only=false;break;}
    }
  for(u32 c=0;c<e.source.classes;++c)
    if(!target_only||c==selector_winner||c==selector_rival)
      e.range_lower[c]=e.range_upper[c]=__uint_as_float(words[c]);
  // Joint search shares the existing effort controller's total visit budget.
  // The ordinary per-tree pass owns at least half and supplies an incumbent
  // enclosure. The optional joint pass may only tighten that saved enclosure.
  out.unary_visit_budget=classify?global_visit_budget/2:0;
  out.relational_visit_budget=e.relational_bounds_enabled&&e.source.classes>1?
      (global_visit_budget-out.unary_visit_budget)/3:0;
  const u32 ordinary_budget=global_visit_budget-out.relational_visit_budget-out.unary_visit_budget;
  out.pair_visit_budget=private_stack&&stack_capacity>=4&&e.source.trees>=2?
      min(e.joint_pair_visit_budget,ordinary_budget/2):0;
  out.baseline_visit_budget=ordinary_budget-out.pair_visit_budget;
  u32 remaining=out.baseline_visit_budget;
  double widest=-1.;
  const u32 capacity=private_stack?stack_capacity:0;
  for(u32 t=0;t<e.source.trees;++t)if(roots[t]>=0){
    const u32 root=u32(roots[t]),channel=u32(e.source.channels[t]);
    if(target_only&&channel!=selector_winner&&channel!=selector_rival)continue;
    u32 lo=e.minimum[root],hi=e.maximum[root];
    if(classify)e.unary_axes[t]=none;
    if(remaining){
      const bool proposal_factor=select&&(channel==selector_winner||channel==selector_rival);
      const auto refined=conditioned_subtree_extrema(e,R,root,private_stack,capacity,remaining,classify,proposal_factor);
      remaining-=refined.visited;out.visited+=refined.visited;
      out.fallback_frontiers+=refined.fallbacks;++out.refined_roots;
      const float a=__uint_as_float(refined.minimum),b=__uint_as_float(refined.maximum);
      const float original_a=__uint_as_float(lo),original_b=__uint_as_float(hi);
      if(isfinite(a)&&isfinite(b)&&a<=b&&a>=original_a&&b<=original_b){
        if(a>original_a||b<original_b)++out.tightened_roots;
        lo=refined.minimum;hi=refined.maximum;
        if(classify)e.unary_axes[t]=refined.axis;
        // Source order and a strict improvement retain the first source ID on
        // width ties. Even incomplete metadata only chooses a cover split;
        // it cannot authorize a label or strengthen an enclosure.
        if(proposal_factor&&refined.first_unforced!=none){
          const double width=double(b)-double(a);
          if(width>0.&&width>widest){widest=width;out.selected_predicate=refined.first_unforced;}
        }
      }else{++out.rejected_refinements;++out.fallback_frontiers;}
    }else{++out.fallback_frontiers;}
    if(classify){e.unary_minimum[t]=lo;e.unary_maximum[t]=hi;}
    e.range_lower[channel]=__fadd_rn(e.range_lower[channel],__uint_as_float(lo));
    e.range_upper[channel]=__fadd_rn(e.range_upper[channel],__uint_as_float(hi));
  }
  out.baseline_label=out.label=qualified_range_label(e);
  // A rival caller needs this comparison, not a complete class certificate.
  // The retained non-target intervals already satisfy the range premise. If
  // the conditioned target intervals suffice, do not spend work on pairs just
  // because another class still overlaps. Leave an unknown class label unknown.
  if(target_only&&out.label<0){
    const float wl=e.range_lower[selector_winner],wu=e.range_upper[selector_winner];
    const float rl=e.range_lower[selector_rival],ru=e.range_upper[selector_rival];
    if(isfinite(wl)&&isfinite(wu)&&isfinite(rl)&&isfinite(ru)&&wl<=wu&&rl<=ru&&
       wl>=-10.f&&wu<=10.f&&rl>=-10.f&&ru<=10.f&&
       __dsub_rn(double(wl),double(ru))>=native_softprob_gap::computed_gap_minimum)
      return out;
  }
  if(out.label<0&&out.pair_visit_budget) {
    // Return allowance the incumbent did not spend. The reported allocations
    // still sum to the caller's total, and the explicit joint ceiling remains
    // respected. This does not debit the conditioned incumbent after the fact.
    const u32 returned=min(remaining,e.joint_pair_visit_budget-out.pair_visit_budget);
    out.baseline_visit_budget-=returned;out.pair_visit_budget+=returned;
    u32 pair_remaining=out.pair_visit_budget;
    // Adjacent residual trees WITHIN a channel may have other channels between
    // them globally. Those other accumulators are independent; each channel's
    // exact leaf-addition sequence is unchanged. No leaf sums are reassociated.
    for(u32 c=0;c<e.source.classes;++c) {
      if(target_only&&c!=selector_winner&&c!=selector_rival)continue;
      float lo=__uint_as_float(words[c]),hi=lo;
      for(u32 t=0;t<e.source.trees;) {
        while(t<e.source.trees&&(roots[t]<0||u32(e.source.channels[t])!=c))++t;
        if(t==e.source.trees)break;
        u32 next=t+1;
        while(next<e.source.trees&&(roots[next]<0||u32(e.source.channels[next])!=c))++next;
        if(next==e.source.trees) {
          lo=__fadd_rn(lo,__uint_as_float(e.minimum[roots[t]]));
          hi=__fadd_rn(hi,__uint_as_float(e.maximum[roots[t]]));
          break;
        }
        const u32 first=u32(roots[t]),second=u32(roots[next]);
        if(pair_remaining) {
          ++out.pair_attempts;
          const auto paired=joint::pair_enclosure(e,R,first,second,lo,hi,
              private_stack,stack_capacity,pair_remaining);
          pair_remaining-=paired.visited;out.pair_visits+=paired.visited;out.visited+=paired.visited;
          if(paired.complete) {
            ++out.pair_completed;out.pair_tightened+=paired.tightened;
            lo=paired.lower;hi=paired.upper;
          } else {
            ++out.pair_fallbacks;
            lo=__fadd_rn(__fadd_rn(lo,__uint_as_float(e.minimum[first])),__uint_as_float(e.minimum[second]));
            hi=__fadd_rn(__fadd_rn(hi,__uint_as_float(e.maximum[first])),__uint_as_float(e.maximum[second]));
          }
        } else {
          lo=__fadd_rn(__fadd_rn(lo,__uint_as_float(e.minimum[first])),__uint_as_float(e.minimum[second]));
          hi=__fadd_rn(__fadd_rn(hi,__uint_as_float(e.maximum[first])),__uint_as_float(e.maximum[second]));
        }
        t=next+1;
      }
      // Both intervals independently enclose the same ordered channel score.
      // An exhausted pair contributes its unchanged static enclosure, and the
      // saved conditioned per-tree incumbent can never become weaker.
      if(isfinite(lo)&&isfinite(hi)&&lo<=hi) {
        const float lower=fmaxf(e.range_lower[c],lo),upper=fminf(e.range_upper[c],hi);
        if(lower<=upper){e.range_lower[c]=lower;e.range_upper[c]=upper;}
        else ++out.rejected_refinements;
      }
    }
    out.label=qualified_range_label(e);
    out.pair_additional_prune=out.baseline_label<0&&out.label>=0;
  }
  if(out.label<0&&classify) {
    out.baseline_visit_budget=out.visited-out.pair_visits;
    out.pair_visit_budget=out.pair_visits;
    out.unary_visit_budget=global_visit_budget-out.visited-out.relational_visit_budget;
    unary::Scratch scratch{e.unary_axes,e.unary_minimum,e.unary_maximum,e.unary_cuts,
                          e.source.trees,e.unary_cut_capacity,e.unary_order,e.source.trees,e.unary_first,e.unary_first_capacity};
    const auto grouped=unary::interval_label(e,id,R,private_stack,stack_capacity,scratch,out.unary_visit_budget);
    out.unary_attempted=grouped.attempted;
    out.unary_optimistic_rejected=grouped.optimistic_rejected;
    out.unary_visits=grouped.visited;out.visited+=grouped.visited;
    out.unary_groups=grouped.groups;out.unary_completed=grouped.completed;out.unary_fallbacks=grouped.fallbacks;
    if(grouped.success){out.label=grouped.label;out.unary_additional_prune=true;}
  }
  if(out.label<0&&e.relational_bounds_enabled) {
    // Return all unused ordinary allowance to the relational proof; actual
    // charged visits still never exceed this call's shared dynamic budget.
    out.baseline_visit_budget=out.visited-out.pair_visits-out.unary_visits;
    out.pair_visit_budget=out.pair_visits;
    out.unary_visit_budget=out.unary_visits;
    out.relational_visit_budget=global_visit_budget-out.visited;
    const auto comparison=relational::interval_label(e,id,R,private_stack,stack_capacity,out.relational_visit_budget);
    out.relational_attempted=comparison.attempted;
    out.relational_visits=comparison.visited;out.visited+=comparison.visited;
    out.relational_pairs=comparison.pairs_attempted;out.relational_completed=comparison.pairs_complete;
    out.relational_fallbacks=comparison.pair_fallbacks;
    if(comparison.success){out.label=comparison.label;out.relational_additional_prune=true;}
  }
  out.additional_prune=out.label>=0;
  return out;
}
} // namespace class_conversion_adaptive::effort
