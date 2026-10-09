// Actual scheduler-hook checks using opaque synthetic state/node metadata.
// No tree prediction, source numerical evaluation, or native qualification is
// performed. Positive proof events are explicit bookkeeping fixture premises.
#include "class_conversion/adaptive_parallel_schedule.cuh"
#include <iostream>
#include <cstring>

namespace a=class_conversion_adaptive;
namespace f=a::frontier;
namespace d=f::detail;
using U=a::u32;
using E=a::gap_evidence::Evidence;
struct Report {U checks=0,failures=0,first_failure=0;};
struct Predicate {int feature=0;float cut=0;unsigned char missing=0;};
__device__ void expect(Report* r,bool ok,U code){
  ++r->checks;if(!ok){++r->failures;if(!r->first_failure)r->first_failure=code;}
}
// Populate a valid metadata hash inventory; this does not evaluate a model.
__device__ void index_states(a::EngineView e){
  for(U i=0;i<e.arena.state_buckets;++i)e.arena.table[i]=a::none;
  for(U id=0;id<e.status->states;++id){
    auto& s=e.arena.states[id];if(s.phase==a::free_phase)continue;
    s.hash=a::state_hash(e,a::region(e,id),e.arena.words+2*id,e.arena.positions+2*id,nullptr);
    U at=a::hash_bucket(s.hash,e.arena.state_buckets);
    while(e.arena.table[at]!=a::none)at=(at+1)&(e.arena.state_buckets-1);
    e.arena.table[at]=id;
  }
}
__global__ void reset_fixture(a::EngineView e,d::View v,Predicate* predicate){
  if(blockIdx.x||threadIdx.x)return;
  *predicate=Predicate{};*e.status=a::Status{};*v.control=d::Control{};
  e.status->states=e.status->state_creations=1;
  for(U i=0;i<e.arena.state_capacity;++i){
    e.arena.states[i]=a::State{};e.arena.states[i].predicate=0;
    e.arena.words[2*i]=i;e.arena.words[2*i+1]=0;
    e.arena.positions[2*i]=e.arena.positions[2*i+1]=0;
    e.arena.lower[i]=e.arena.witness_lower[i]=a::domain::finite_min_key;
    e.arena.upper[i]=e.arena.witness_upper[i]=a::domain::finite_max_key;
    e.arena.missing[i]=e.arena.witness_missing[i]=0;
    v.queue.ready[i]=v.queue.finish[i]=v.queue.completed[i]=0;
    v.queue.flags[i]=v.queue.pending[i]=0;
    v.queue.wait_head[i]=v.queue.edge_next[2*i]=v.queue.edge_next[2*i+1]=a::none;
    if(v.gap_evidence)v.gap_evidence[i]=E::Unknown;
  }
  for(U i=0;i<e.arena.state_buckets;++i)e.arena.terminal_table[i]=a::none;
  for(U i=0;i<e.arena.node_buckets;++i)e.arena.node_table[i]=a::none;
  v.jobs[0]=0;v.kinds[0]=1;v.prune_labels[0]=0;
  v.native_ids[0]=0;v.native_labels[0]=0;
  index_states(e);
}
// mode: bit0 queued deliveries, bit1 unknown second context, bit2 same node,
// bit3 restored unknown expansion. Every edge uses the real attach/resolve path.
__global__ void prepare_edges(a::EngineView e,d::View v,U mode,Report* r){
  if(blockIdx.x||threadIdx.x)return;
  e.status->states=e.status->state_creations=3;
  U left=a::intern_node(e,{-1,0,0,0});
  U right=(mode&4)?left:a::intern_node(e,{-1,1,0,0});
  auto& root=e.arena.states[0];root.phase=2;root.predicate=0;
  v.queue.pending[0]=2;v.queue.flags[0]=d::ready_once;
  v.gap_evidence[0]=(mode&8)?E::Unknown:E::PendingAllQualified;
  for(U id=1;id<3;++id){
    e.arena.states[id].phase=(mode&1)?0:3;e.arena.states[id].node=id==1?left:right;
    v.gap_evidence[id]=(id==2&&(mode&2))?E::Unknown:E::CompletedQualified;
  }
  d::attach_edge(e,v,0,0,1,false);d::attach_edge(e,v,0,1,2,false);
  expect(r,!e.status->error,100+mode);
  expect(r,v.queue.pending[0]==((mode&1)?2u:0u),120+mode);
  if(mode&1){
    expect(r,root.left==1&&root.right==2,140+mode);
    for(U id=1;id<3;++id){e.arena.states[id].phase=3;d::enqueue_finish(e,v,id);}
  }
  index_states(e);
}
__global__ void check_edges(a::EngineView e,d::View v,U mode,Report* r){
  if(blockIdx.x||threadIdx.x)return;
  expect(r,!e.status->error&&e.status->complete&&e.arena.states[0].phase==3,200+mode);
  const E wanted=(mode&(2|8))?E::Unknown:E::CompletedQualified;
  expect(r,v.gap_evidence[0]==wanted&&v.control->resolved_edges==2,220+mode);
  if(mode&4)expect(r,e.arena.states[0].node==e.arena.states[1].node,240+mode);
}
__global__ void check_retry(a::EngineView e,d::View v,bool leaf,Report* r){
  if(blockIdx.x||threadIdx.x)return;
  expect(r,!e.status->error&&e.status->request==3&&e.arena.states[0].phase==(leaf?0u:2u),300+leaf);
  expect(r,v.gap_evidence[0]==(leaf?E::Unknown:E::PendingAllQualified),310+leaf);
  expect(r,leaf?v.control->cursor==0:v.control->finish_head==0,320+leaf);
}
__global__ void prepare_publication(a::EngineView e,d::View v,U kind){
  if(blockIdx.x||threadIdx.x)return;
  v.control->stage=d::draft_commit;v.control->jobs=1;
  v.kinds[0]=kind;v.queue.flags[0]=d::ready_once;
}
__global__ void check_publication(a::EngineView e,d::View v,bool eligible,Report* r){
  if(blockIdx.x||threadIdx.x)return;
  expect(r,!e.status->error&&e.arena.states[0].phase==3&&e.status->class_pruned_states==1,350+eligible);
  expect(r,v.gap_evidence[0]==(eligible?E::CompletedQualified:E::Unknown),360+eligible);
}
__global__ void prepare_native(a::EngineView e,d::View v,bool cached){
  if(blockIdx.x||threadIdx.x)return;
  v.gap_evidence[0]=E::CompletedQualified; // Deliberately stale poison must clear.
  e.arena.states[0].predicate=a::none;
  if(cached){
    e.status->states=e.status->state_creations=2;
    e.arena.states[1].phase=3;e.arena.states[1].predicate=a::none;
    e.arena.states[1].node=a::intern_node(e,{-1,0,0,0});
    e.arena.words[2]=e.arena.words[0];e.arena.words[3]=e.arena.words[1];
    v.gap_evidence[1]=E::CompletedQualified;
    a::terminal_lookup(e,e.arena.words+2,true,1);
    v.control->stage=d::draft_commit;v.control->jobs=1;v.kinds[0]=0;
  }else{
    e.arena.states[0].phase=1;v.control->stage=d::native_commit;v.control->native_count=1;
  }
}
__global__ void check_native(a::EngineView e,d::View v,bool cached,Report* r){
  if(blockIdx.x||threadIdx.x)return;
  expect(r,!e.status->error&&e.arena.states[0].phase==3&&v.gap_evidence[0]==E::Unknown,400+cached);
  expect(r,cached?e.status->terminal_hits==1:e.status->native_terminals==1,410+cached);
}
// These are already-prepared opaque draft payloads, not model-generated scores.
// mode0 fresh high-water slots; mode1 recycled slot; mode2 exact completed hit.
__global__ void prepare_admission(a::EngineView e,d::View v,U mode){
  if(blockIdx.x||threadIdx.x)return;
  v.control->stage=d::draft_commit;v.control->jobs=1;v.control->expanding=1;
  v.kinds[0]=2;v.queue.flags[0]=d::ready_once;
  v.gap_evidence[1]=v.gap_evidence[2]=E::CompletedQualified;
  if(mode==1){e.status->states=e.status->state_creations=2;e.status->free_head=1;e.status->free_count=1;
    e.arena.states[1].phase=a::free_phase;e.arena.states[1].reserved=a::none;
  }else if(mode==2){e.status->states=e.status->state_creations=2;
    e.arena.states[1].phase=3;e.arena.states[1].predicate=a::none;
    e.arena.states[1].node=a::intern_node(e,{-1,0,0,0});e.arena.words[2]=10;
  }
  for(U side=0;side<2;++side){auto child=d::private_view(e,v,side);*child.status=a::Status{};
    *child.draft=a::State{};child.draft->predicate=a::none;
    child.draft_words[0]=(mode==2)?10:10+side;child.draft_words[1]=0;
    child.draft_positions[0]=child.draft_positions[1]=0;
    a::copy_region(child,child.draft_region,a::region(e,0));
    a::copy_region(child,child.draft_witness,a::witness_region(e,0));
    child.draft->hash=a::state_hash(child,child.draft_region,child.draft_words,child.draft_positions,nullptr);
  }
  index_states(e);
}
__global__ void check_admission(a::EngineView e,d::View v,U mode,Report* r){
  if(blockIdx.x||threadIdx.x)return;
  expect(r,!e.status->error&&e.arena.states[0].phase==2,450+mode);
  expect(r,v.gap_evidence[0]==E::PendingAllQualified,460+mode);
  if(mode==2){
    expect(r,e.status->states==2&&e.status->state_creations==2&&v.gap_evidence[1]==E::CompletedQualified,470+mode);
    expect(r,v.queue.pending[0]==0&&v.control->completed_edges==2,480+mode);
  }else{
    expect(r,e.status->states==3&&v.gap_evidence[1]==E::Unknown&&v.gap_evidence[2]==E::Unknown,470+mode);
    expect(r,e.status->state_reuses==(mode==1?1u:0u)&&v.queue.pending[0]==2,480+mode);
  }
}
__global__ void prepare_reclaim(a::EngineView e,d::View v){
  if(blockIdx.x||threadIdx.x)return;
  e.status->states=e.status->state_creations=2;
  e.arena.states[1].phase=3;e.arena.states[1].node=a::intern_node(e,{-1,0,0,0});
  v.gap_evidence[1]=E::CompletedQualified;v.queue.flags[1]=d::cached_once;
  v.control->cache_tail=1;v.queue.completed[0]=1;index_states(e);
}
__global__ void check_reclaim(a::EngineView e,d::View v,Report* r){
  if(blockIdx.x||threadIdx.x)return;
  expect(r,!e.status->error&&e.status->state_evictions==1&&e.arena.states[1].phase==a::free_phase,500);
  expect(r,v.gap_evidence[1]==E::Unknown&&v.queue.flags[1]==0&&v.queue.wait_head[1]==a::none,501);
}
__global__ void prepare_growth(a::EngineView e,d::View v){
  if(blockIdx.x||threadIdx.x)return;
  e.status->states=e.status->state_creations=3;
  v.gap_evidence[0]=E::PendingAllQualified;v.gap_evidence[1]=E::PendingBlocked;v.gap_evidence[2]=E::CompletedQualified;
  e.arena.states[1].phase=2;e.arena.states[2].phase=3;
  e.arena.states[2].node=a::intern_node(e,{-1,0,0,0});index_states(e);
}
struct Fixture {
  a::Budget budget{64ull*1024*1024};
  std::unique_ptr<a::StateStorage> states;
  std::unique_ptr<a::NodeStorage> nodes;
  std::unique_ptr<d::QueueStorage> queue;
  a::Buffer<d::Control> control;
  a::Buffer<a::Status> status;
  a::Buffer<E> evidence;
  a::Buffer<Predicate> predicate;
  a::Buffer<Report> report;
  d::DraftStorage drafts;
  a::Buffer<U> jobs,kinds,prune_labels,native_ids,native_labels;
  a::EngineView e{};
  explicit Fixture(U capacity=8):states(std::make_unique<a::StateStorage>(budget,capacity,1,2,0,0)),
    nodes(std::make_unique<a::NodeStorage>(budget,8)),queue(std::make_unique<d::QueueStorage>(budget,capacity)),
    control(budget,1),status(budget,1),evidence(budget,capacity),predicate(budget,1),report(budget,1),
    drafts(budget,2,1,2,0,0,1,0),jobs(budget,1),kinds(budget,1),prune_labels(budget,1),native_ids(budget,1),native_labels(budget,1){
    e.source.feature=&predicate.data->feature;e.source.cut=&predicate.data->cut;e.source.missing=&predicate.data->missing;
    e.source.features=1;e.source.classes=2;e.source.nodes=1;e.domain.features=1;e.domain.numeric_features=1;
    e.status=status.data;e.support_words=1;e.qualified_gap=true;a::bind(e,*states,*nodes);report.zero();
  }
  d::View view(bool eligible=true){d::View v{};v.queue=queue->view();v.draft=drafts.view();v.control=control.data;
    v.jobs=jobs.data;v.kinds=kinds.data;v.prune_labels=prune_labels.data;v.native_ids=native_ids.data;v.native_labels=native_labels.data;
    v.batch=1;v.gap_evidence=evidence.data;v.gap_context_supported=eligible;return v;
  }
  void reset(){reset_fixture<<<1,1>>>(e,view(),predicate.data);a::synchronize();}
  Report checked(){return report.download(1)[0];}
};

int main(int argc,char**argv){try{
  if(argc==2&&!std::strcmp(argv[1],"--help")){
    std::cout<<"Checks actual scheduler evidence hooks using synthetic metadata premises; no model evaluation or native qualification. Running without --help uses CUDA.\n";return 0;
  }
  a::require(argc==1,"unknown gap evidence check argument");
  Fixture q;U scenarios=0;
  for(U mode:{0u,1u,2u,3u,4u,5u,6u,7u,8u,9u}){
    ++scenarios;q.reset();prepare_edges<<<1,1>>>(q.e,q.view(),mode,q.report.data);a::synchronize();
    d::drain_completions<<<1,1>>>(q.e,q.view());a::synchronize();
    check_edges<<<1,1>>>(q.e,q.view(),mode,q.report.data);a::synchronize();
  }
  // Existing leaves fill the temporary arena; parent publication must retry
  // without losing either resolved edge or its accumulated context evidence.
  ++scenarios;q.reset();prepare_edges<<<1,1>>>(q.e,q.view(),0,q.report.data);a::synchronize();
  auto limited=q.e;limited.arena.node_capacity=2;
  d::drain_completions<<<1,1>>>(limited,q.view());a::synchronize();
  check_retry<<<1,1>>>(q.e,q.view(),false,q.report.data);a::synchronize();
  d::drain_completions<<<1,1>>>(q.e,q.view());a::synchronize();
  check_edges<<<1,1>>>(q.e,q.view(),0,q.report.data);a::synchronize();
  for(bool eligible:{true,false}){
    ++scenarios;q.reset();prepare_publication<<<1,1>>>(q.e,q.view(),1);a::synchronize();
    limited=q.e;limited.arena.node_capacity=0;
    d::commit_drafts<<<1,32>>>(limited,q.view(eligible),0);a::synchronize();
    check_retry<<<1,1>>>(q.e,q.view(),true,q.report.data);a::synchronize();
    d::commit_drafts<<<1,32>>>(q.e,q.view(eligible),0);a::synchronize();
    check_publication<<<1,1>>>(q.e,q.view(),eligible,q.report.data);a::synchronize();
  }
  for(bool cached:{false,true}){
    ++scenarios;q.reset();prepare_native<<<1,1>>>(q.e,q.view(),cached);a::synchronize();
    if(cached)d::commit_drafts<<<1,32>>>(q.e,q.view(),0);else d::commit_native<<<1,1>>>(q.e,q.view());
    a::synchronize();check_native<<<1,1>>>(q.e,q.view(),cached,q.report.data);a::synchronize();
  }
  for(U mode=0;mode<3;++mode){
    ++scenarios;q.reset();prepare_admission<<<1,1>>>(q.e,q.view(),mode);a::synchronize();
    d::commit_drafts<<<1,32>>>(q.e,q.view(),0);a::synchronize();
    check_admission<<<1,1>>>(q.e,q.view(),mode,q.report.data);a::synchronize();
    if(mode==2){d::drain_completions<<<1,1>>>(q.e,q.view());a::synchronize();
      // One completed context can satisfy both exact incoming payloads.
      const auto h=q.status.download(1)[0];const auto ev=q.evidence.download(1)[0];
      a::require(!h.error&&h.complete&&ev==E::CompletedQualified,"exact-hit expansion evidence");}
  }
  ++scenarios;q.reset();prepare_reclaim<<<1,1>>>(q.e,q.view());a::synchronize();
  d::reclaim_completed<<<1,1>>>(q.e,q.view(),0,0);a::synchronize();
  check_reclaim<<<1,1>>>(q.e,q.view(),q.report.data);a::synchronize();
  const auto report=q.checked();a::require(!report.failures,("scheduler evidence assertion "+std::to_string(report.first_failure)+"; failures="+std::to_string(report.failures)+"; checks="+std::to_string(report.checks)).c_str());
  // Real coupled owner transaction, with a deliberate last-owner allocation
  // refusal. Any smaller retry is also refused so the original owner persists.
  Fixture growing(4);growing.reset();prepare_growth<<<1,1>>>(growing.e,growing.view());a::synchronize();
  const auto h=growing.status.download(1)[0];const auto before=growing.evidence.download(4);
  auto* old_states=growing.states.get();auto* old_queue=growing.queue.get();
  auto* old_control=growing.control.data;auto* old_evidence=growing.evidence.data;const auto bytes=growing.budget.used;
  U injected=0;bool refused=false;
  try{d::grow_state_queue_transaction(growing.e,growing.states,*growing.nodes,growing.queue,growing.control,growing.budget,h,8,8,{},
      [&](const char* owner){if(!std::strcmp(owner,"gap_evidence")){++injected;throw a::AllocationRefusal(cudaSuccess,0,"test last owner refusal");}},0,&growing.evidence);
  }catch(const a::AllocationRefusal&){refused=true;}
  a::require(refused&&injected&&growing.states.get()==old_states&&growing.queue.get()==old_queue&&
    growing.control.data==old_control&&growing.evidence.data==old_evidence&&growing.budget.used==bytes&&
    growing.evidence.download(4)==before,"gap evidence coupled allocation rollback");++scenarios;
  const auto outcome=d::grow_state_queue_transaction(growing.e,growing.states,*growing.nodes,growing.queue,growing.control,growing.budget,h,8,8,{}, {},0,&growing.evidence);
  const auto after=growing.evidence.download(8);
  a::require(outcome.complete&&growing.states->capacity==8&&growing.queue->capacity==8&&growing.evidence.size==8&&
    growing.evidence.budget==&growing.budget&&growing.e.arena.states==growing.states->states.data,
    "gap evidence coupled growth publication");
  for(U i=0;i<8;++i)a::require(after[i]==(i<4?before[i]:E::Unknown),"gap evidence copy or fresh growth tail");++scenarios;
  std::cout<<"{\"scenarios\":"<<scenarios<<",\"device_checks\":"<<report.checks
    <<",\"device_failures\":"<<report.failures<<",\"model_evaluations\":0,\"native_RuntimeGate_qualified\":false"
      ",\"synthetic_metadata_premises\":true,\"CUDA_executed\":true,\"growth_rollback\":true,\"growth_copy_tail\":true}\n";
  return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}}
