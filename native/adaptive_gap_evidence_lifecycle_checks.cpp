// Host checks for evidence transitions. No model arithmetic, CUDA, queue or
// converter is implemented here. The caller already proves source/guard
// correspondence and live native-contract premises and owns publication/phase
// invariants. This code checks only lossless, fail-closed evidence bookkeeping.
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "class_conversion/adaptive_gap_evidence.hpp"
namespace gap_evidence = class_conversion_adaptive::gap_evidence;

int main(){try {
 using namespace gap_evidence;using E=Evidence;
 std::uint32_t checks=0,scenarios=0;
 auto check=[&](bool condition,const char*message){++checks;if(!condition)throw std::runtime_error(message);};
 constexpr QualifiedPremises valid{true,true,true};
 const auto qualified=[](){auto value=E::Unknown;complete_qualified_rule(value,valid,true);return value;};
 // Both complete children, either arrival order, through node-capacity refusal.
 for(bool reverse:{false,true}){++scenarios;auto value=E::Unknown;
  check(start_new_split(value,true,true),"new expansion refused");
  const auto left=qualified(),right=qualified();
  check(observe_edge(value,reverse?right:left,true),"first qualified edge refused");
  check(!finish_split(value,true,1,true)&&value==E::PendingAllQualified,"one edge prematurely certified");
  check(observe_edge(value,reverse?left:right,true),"second qualified edge refused");
  check(!finish_split(value,true,0,false)&&value==E::PendingAllQualified,"failed node publication consumed evidence");
  check(finish_split(value,true,0,true)&&value==E::CompletedQualified,"completed expansion lost full evidence");
  check(!finish_split(value,false,0,true)&&value==E::CompletedQualified,"phase3 retry mutated evidence");
 }
 // Unknown on either edge blocks the conjunction; later valid edges cannot repair it.
 for(bool unknown_first:{false,true}){++scenarios;auto value=E::Unknown;start_new_split(value,true,true);
  observe_edge(value,unknown_first?E::Unknown:qualified(),true);
  observe_edge(value,unknown_first?qualified():E::Unknown,true);
  check(value==E::PendingBlocked,"unknown edge did not block");
  check(!start_new_split(value,true,true)&&value==E::PendingBlocked,"retry reset blocked expansion");
  check(finish_split(value,true,0,true)&&value==E::Unknown,"blocked expansion upgraded");
  check(finish_split(value,true,0,true)&&value==E::Unknown,"repeat publication upgraded unknown");
 }
 {++scenarios;auto value=E::Unknown;start_new_split(value,true,true);const auto before=value;
  check(!observe_edge(value,E::Unknown,false)&&value==before,"rejected duplicate edge changed evidence");
  observe_edge(value,qualified(),true);observe_edge(value,qualified(),true);
  check(finish_split(value,true,0,true)&&value==E::CompletedQualified,"same qualified source feeding two sides failed");
 }
 // Node identity is intentionally absent from transition functions: equal edges
 // can have different source-context evidence, and still must both be discharged.
 {++scenarios;struct Context{std::uint32_t node;E evidence;};const Context left{19,qualified()},right{19,E::Unknown};
  auto value=E::Unknown;start_new_split(value,true,true);observe_edge(value,left.evidence,true);observe_edge(value,right.evidence,true);
  check(left.node==right.node&&finish_split(value,true,0,true)&&value==E::Unknown,"equal node bypass laundered unknown context");
 }
 {++scenarios;auto value=E::Unknown;
  check(!complete_qualified_rule(value,valid,false)&&value==E::Unknown,"failed leaf publication upgraded");
  check(complete_qualified_rule(value,valid,true)&&value==E::CompletedQualified,"qualified publication missing");
  check(complete_qualified_rule(value,valid,true)&&value==E::CompletedQualified,"repeated publication not idempotent");
  fresh_or_recycled(value);check(value==E::Unknown,"recycled slot retained certificate");
  check(!complete_qualified_rule(value,{false,true,true},true)&&value==E::Unknown,"witness-only proof accepted");
  check(!complete_qualified_rule(value,{true,false,true},true)&&value==E::Unknown,"different source/output contract accepted");
  check(!complete_qualified_rule(value,{true,true,false},true)&&value==E::Unknown,"absent native gate accepted");
 }
 {++scenarios;auto value=qualified();check(!complete_native_label(value,false)&&value==E::CompletedQualified,"unpublished native label mutated context");
  fresh_or_recycled(value);check(complete_native_label(value,true)&&value==E::Unknown,"native tie upgraded evidence");
  check(exact_completed_hit(value,true,true,valid)==E::Unknown,"native completed hit laundered evidence");
  check(exact_completed_hit(qualified(),true,true,valid)==E::CompletedQualified,"exact qualified hit lost evidence");
  check(exact_completed_hit(qualified(),false,true,valid)==E::Unknown,"pending donor supplied completed evidence");
  check(exact_completed_hit(qualified(),true,false,valid)==E::Unknown,"unequal payload inherited evidence");
  check(exact_completed_hit(qualified(),true,true,{true,false,true})==E::Unknown,"different contract inherited evidence");
 }
 // Old checkpoint may contain phase2 with one already-resolved edge whose
 // context is gone. Starting Unknown must remain conservative after the other.
 {++scenarios;auto restored=E::Unknown;
  check(!start_new_split(restored,false,true)&&restored==E::Unknown,"restored phase2 was initialized optimistically");
  observe_edge(restored,qualified(),true);
  check(finish_split(restored,true,0,true)&&restored==E::Unknown,"restore lost-edge provenance invented");
  check(exact_completed_hit(restored,true,true,valid)==E::Unknown,"restored unknown admitted");
 }
 // Sidecar copy staging, not a queue/scheduler: failure preserves old owner;
 // successful resize retains live evidence and leaves fresh tail unknown.
 {++scenarios;std::vector<E> owner{qualified(),E::PendingBlocked,E::Unknown};
  {auto staged=owner;staged.resize(8,E::Unknown);check(owner.size()==3&&owner[1]==E::PendingBlocked,"staging changed live owner");}
  check(owner.size()==3&&owner[0]==E::CompletedQualified,"abandoned staged growth changed live owner");
  auto staged=owner;staged.resize(8,E::Unknown);owner.swap(staged);
  check(owner.size()==8&&owner[0]==E::CompletedQualified&&owner[1]==E::PendingBlocked&&owner[7]==E::Unknown,"growth copy/tail incorrect");
  const auto delivered=owner[0];fresh_or_recycled(owner[0]);auto waiting=E::Unknown;start_new_split(waiting,true,true);
  observe_edge(waiting,delivered,true);observe_edge(waiting,qualified(),true);
  check(finish_split(waiting,true,0,true)&&waiting==E::CompletedQualified&&owner[0]==E::Unknown,"eviction after evidence delivery broke ownership");
 }
 std::cout<<"{\"scenarios\":"<<scenarios<<",\"checks\":"<<checks
  <<",\"failures\":0,\"evidence_bytes\":"<<sizeof(E)
  <<",\"model_evaluations\":0,\"gpu_execution\":false,\"scope\":\"event bookkeeping under explicit caller source/guard/native premises; not a new certificate\"}\n";
 return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
