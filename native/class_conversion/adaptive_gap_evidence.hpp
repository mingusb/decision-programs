#pragma once
#include <cstdint>

// Observation-only lifecycle evidence for the current source/gate generation.
// CompletedQualified concerns saved-prefix + original ordered residual scores
// on the ENTIRE projected guard, and the output program on that same guard.
// It is not a full-original-forest certificate on widened input coordinates.
// Every positive leaf event must already possess the current finite score
// window/all-rival positive true-gap certificate. In particular the rounded
// comparison threshold is not itself the exact-real gap (the native contract
// includes its conservative arithmetic slack). These helpers prove no numeric
// fact and grant no RuntimeGate authority or cache admission.
#if defined(__CUDACC__)
#define DP_GAP_HD __host__ __device__
#else
#define DP_GAP_HD
#endif
namespace class_conversion_adaptive::gap_evidence {

enum class Evidence : std::uint8_t {
  Unknown, PendingAllQualified, PendingBlocked, CompletedQualified
};
static_assert(sizeof(Evidence)==1);
struct QualifiedPremises {
  bool full_projected_guard;  // Saved prefix + residual scores, not original forest on widened guard.
  bool exact_source_output_contract;
  bool live_native_gate;
};
DP_GAP_HD constexpr bool admitted(QualifiedPremises p) {
  return p.full_projected_guard && p.exact_source_output_contract && p.live_native_gate;
}
DP_GAP_HD constexpr void fresh_or_recycled(Evidence& value) { value=Evidence::Unknown; }
DP_GAP_HD constexpr bool start_new_split(Evidence& value,bool phase_zero,bool first_expansion_publication) {
  if(!phase_zero||!first_expansion_publication||value!=Evidence::Unknown)return false;
  value=Evidence::PendingAllQualified;return true;
}
// Called before resolve_edge discards child-state identity. Existing scheduler
// must already have validated phase2, unresolved edge and child completion.
DP_GAP_HD constexpr bool observe_edge(Evidence& value,Evidence child,bool valid_edge_delivery) {
  if(!valid_edge_delivery||value==Evidence::CompletedQualified)return false;
  if(value==Evidence::PendingAllQualified&&child!=Evidence::CompletedQualified)
    value=Evidence::PendingBlocked;
  return true;  // Restored Unknown deliberately stays Unknown.
}
DP_GAP_HD constexpr bool finish_split(Evidence& value,bool phase_two,std::uint32_t pending,
                            bool node_publication_succeeded) {
  if(!phase_two||pending||!node_publication_succeeded)return false;
  if(value==Evidence::PendingAllQualified)value=Evidence::CompletedQualified;
  else if(value!=Evidence::CompletedQualified)value=Evidence::Unknown;
  return true;
}
DP_GAP_HD constexpr bool complete_qualified_rule(Evidence& value,QualifiedPremises premise,
                                      bool node_publication_succeeded) {
  if(!admitted(premise)||!node_publication_succeeded)return false;
  value=Evidence::CompletedQualified;return true;
}
DP_GAP_HD constexpr bool complete_native_label(Evidence& value,bool node_publication_succeeded) {
  if(!node_publication_succeeded)return false;
  value=Evidence::Unknown;return true; // Including native probability ties.
}
DP_GAP_HD constexpr Evidence exact_completed_hit(Evidence donor,bool complete_phase,
                                      bool exact_payload,QualifiedPremises premise) {
  return complete_phase&&exact_payload&&admitted(premise)&&donor==Evidence::CompletedQualified
    ? Evidence::CompletedQualified : Evidence::Unknown;
}
} // namespace class_conversion_adaptive::gap_evidence
#undef DP_GAP_HD
