// Completed synthetic frontier conversions, not native RuntimeGate evidence.
// Reuse the device fixtures and exact ordered source evaluator from the focused
// joint checks; this executable does not run their kernel microbenchmark.
#define main preserved_joint_checks_main
#include "adaptive_joint_bounds_checks.cu"
#undef main
#include "class_conversion/adaptive_parallel_schedule.cuh"
#include <chrono>
#include <array>
#include <algorithm>

namespace af=a::frontier;
using Json=nlohmann::json;
__global__ void conversion_fixture(Fixture* f,U scenario){
  if(blockIdx.x||threadIdx.x)return;
  if(scenario==3){
    // Different rivals need different covers: scores [2a,2b,2+a+b].
    f->init(2,true,0,3);f->bias[2]=2;
    f->tree(f->stump(0,0,0,1),2);f->tree(f->stump(0,0,0,2),0);
    f->tree(f->stump(1,0,0,1),2);f->tree(f->stump(1,0,0,2),1);f->finish(0);
  }else if(scenario==1){
    // No within-pair correlation: every combination is actually feasible.
    f->init(4);for(U i=0;i<4;++i)f->tree(f->stump(i,0,-1,1));f->finish(.25f);
  }else{
    f->init(3);for(U i=0;i<3;++i){f->tree(f->stump(i,0,-1,1));f->tree(f->stump(i,0,1,-1));}
    // A tied winner cannot satisfy the strict qualified-gap pruning rule.
    f->finish(scenario==2?0.f:1.f);
  }
}
__device__ U source_class(const Fixture& f,const float* row,float* scores){
  for(U c=0;c<f.classes;++c)scores[c]=f.bias[c];
  for(U t=0;t<f.trees;++t){const U c=U(f.channels[t]);scores[c]=__fadd_rn(scores[c],evaluate_leaf(f,U(f.roots[t]),row));}
  U best=0;for(U c=1;c<f.classes;++c)if(scores[c]>scores[best])best=c;return best;
}
__global__ void synthetic_native(const Fixture* f,const float* rows,a::u64 count,bool margins,float* output){
  const a::u64 i=a::u64(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
  float scores[Fixture::K];const U label=source_class(*f,rows+i*f->features,scores);
  if(margins)for(U c=0;c<f->classes;++c)output[i*f->classes+c]=scores[c];
  else output[i]=float(label);
}
// All fixture cuts are zero; five atoms exhaust their branch signatures and
// explicitly include both signed zeros plus the independently routed NaN atom.
__global__ void verify_converted(const Fixture* f,const a::Node* nodes,U node_count,U root,U count,Report* report){
  const U i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
  const float atoms[5]={-1.f,-0.f,0.f,1.f,__uint_as_float(0x7fc00001u)};
  float row[Fixture::F],scores[Fixture::K];U code=i;
  for(U feature=0;feature<f->features;++feature){row[feature]=atoms[code%5];code/=5;}
  const U expected=source_class(*f,row,scores);U current=root,steps=0;bool valid=true;
  while(current<node_count&&nodes[current].feature!=-1&&steps++<=node_count){
    const auto n=nodes[current];const bool missing_left=n.feature<=-2;
    const U feature=U(missing_left?-std::int64_t(n.feature)-2:n.feature);
    if(feature>=f->features){valid=false;break;}
    const float x=row[feature];const bool left=isnan(x)?missing_left:x<__uint_as_float(n.payload);
    current=left?n.left:n.right;
  }
  valid=valid&&current<node_count&&nodes[current].feature==-1&&nodes[current].payload==expected;
  atomicAdd(&report->checks,1u);if(!valid){atomicAdd(&report->failures,1u);atomicCAS(&report->first_failure,0u,i+1);}
}

Json convert_case(U scenario,bool joint,bool rival,U budget_multiplier,bool dynamic_defaults=false){
  a::Budget memory{64ull*1024*1024};a::Buffer<Fixture> fixture(memory,1);
  conversion_fixture<<<1,1>>>(fixture.data,scenario);finish();
  a::EngineView e;ck(cudaMemcpy(&e,&fixture.data->e,sizeof(e),cudaMemcpyDeviceToHost));
  const U F=e.source.features,K=e.source.classes,T=e.source.trees,N=e.domain.numeric_features,W=e.domain.mask_words;
  constexpr U capacity=256,batch=32;
  auto states=std::make_unique<a::StateStorage>(memory,capacity,N,K,T,W);
  auto nodes=std::make_unique<a::NodeStorage>(memory,capacity);a::bind(e,*states,*nodes);
  a::Buffer<a::State> draft(memory,1);a::Buffer<a::Status> status(memory,1);
  a::Buffer<U> words(memory,K),positions(memory,K),blocked(memory,K),lower(memory,N),upper(memory,N),missing(memory,N);
  a::Buffer<U> witness_lower(memory,N),witness_upper(memory,N),witness_missing(memory,N),audit_stack(memory,3*e.source.nodes);
  a::Buffer<int> residual(memory,T);a::Buffer<a::u64> allowed(memory,W),witness_allowed(memory,W),support(memory,e.source.nodes),active_support(memory,1);
  a::Buffer<float> range_lower(memory,K),range_upper(memory,K),native_output(memory,batch*K);
  a::Buffer<Report> checked(memory,1);checked.zero();
  e.draft=draft.data;e.draft_words=words.data;e.draft_positions=positions.data;e.blocked=blocked.data;e.draft_residual=residual.data;
  e.draft_region={lower.data,upper.data,missing.data,allowed.data};
  e.draft_witness={witness_lower.data,witness_upper.data,witness_missing.data,witness_allowed.data};
  e.support=support.data;e.active_support=active_support.data;e.support_words=1;
  e.range_lower=range_lower.data;e.range_upper=range_upper.data;e.status=status.data;
  e.refinement_stack_capacity=128;
  e.refinement_maximum_visits=e.source.nodes*(dynamic_defaults?(joint?2u:1u):budget_multiplier);
  e.joint_pair_visit_budget=joint?UINT32_MAX:0;e.rival_cover_enabled=rival;
  // This synthetic fixture assumes the existing numeric gate premise. It does
  // not qualify, bypass or modify any production RuntimeGate decision.
  e.qualified_gap=true;
  a::subtree_extrema<<<1,1>>>(e.source,const_cast<U*>(e.minimum),const_cast<U*>(e.maximum),audit_stack.data,support.data,1);finish();
  af::Limits limits;limits.batch_capacity=batch;limits.maximum_batch_capacity=batch;
  limits.max_states=capacity;limits.max_nodes=capacity;limits.admission_threads=32;limits.draft_threads=32;
  limits.refinement_visit_budget=e.refinement_maximum_visits;
  limits.cover_visit_budget=2*e.refinement_maximum_visits;
  if(dynamic_defaults){limits.dynamic_refinement=true;limits.dynamic_cover=true;}
  const af::NativePredict native=[&](const float* rows,a::u64 count,bool margins)->const float*{
    a::require(count<=batch,"synthetic callback capacity");
    synthetic_native<<<U((count+127)/128),128>>>(fixture.data,rows,count,margins,native_output.data);
    ck(cudaGetLastError());return native_output.data;
  };
  const auto start=std::chrono::steady_clock::now();
  a::initialize<<<1,1>>>(e);finish();
  const auto result=af::run(e,states,nodes,memory,limits,native,true,{});finish();
  const double milliseconds=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  a::require(result.status.complete&&!result.status.error&&!result.stopped,"synthetic conversion must complete");
  U signatures=1;for(U i=0;i<F;++i)signatures*=5;
  verify_converted<<<(signatures+127)/128,128>>>(fixture.data,e.arena.nodes,U(result.status.nodes),result.status.root,signatures,checked.data);finish();
  const auto verified=checked.download(1)[0];a::require(verified.checks==signatures&&!verified.failures,"synthetic converted graph mismatch");
  return Json{{"elapsed_ms",milliseconds},{"complete",true},{"states_high_water",result.status.states},
    {"state_creations",result.status.state_creations},{"nodes",result.status.nodes},{"expansions",result.status.expansions},
    {"normalization_steps",result.status.normalization_steps},{"prefix_additions",result.status.prefix_additions},
    {"class_pruned_states",result.status.class_pruned_states},{"native_terminals",result.status.native_terminals},
    {"native_margin_rows",result.native_margin_rows},{"native_public_rows",result.native_public_rows},
    {"draft_batches",result.draft_batches},{"refinement_visits",result.refinement_visits},{"pair_visits",result.refinement_pair_visits},
    {"pair_attempts",result.refinement_pair_attempts},{"pair_completed",result.refinement_pair_completed},
    {"pair_additional_prunes",result.refinement_pair_additional_prunes},{"cover_visits",result.cover_visits},
    {"rival_attempts",result.cover_rival_attempts},{"rival_prunes",result.cover_rival_prunes},{"verified_signatures",verified.checks},
    {"refinement_visit_budget",result.refinement_visit_budget},{"cover_visit_budget",result.cover_visit_budget}};
}

int main(){try{
  constexpr U repeats=5;const std::array<const char*,4> scenarios={"three_cancelling_pairs","independent_no_gain","tied_no_gain","different_rival_covers"};
  Json results=Json::array();
  for(U policy=0;policy<4;++policy)for(U scenario=0;scenario<scenarios.size();++scenario){
    std::array<std::vector<Json>,4> samples;
    const U multiplier=policy==1?2u:policy==2?4u:1u;const bool dynamic=policy==3;
    for(U mode=0;mode<4;++mode)(void)convert_case(scenario,mode&1,mode&2,multiplier,dynamic);
    // Rotate order across repetitions to reduce systematic clock/cache bias.
    for(U repeat=0;repeat<repeats;++repeat)for(U offset=0;offset<4;++offset){const U mode=(repeat+offset)%4;samples[mode].push_back(convert_case(scenario,mode&1,mode&2,multiplier,dynamic));}
    for(U mode=0;mode<4;++mode){std::vector<double> times;for(const auto& r:samples[mode])times.push_back(r.at("elapsed_ms"));auto sorted=times;std::sort(sorted.begin(),sorted.end());
      auto result=samples[mode].front();result.erase("elapsed_ms");
      // Fixed-budget work must be deterministic. Dynamic effort responds to
      // timing observations, so retain its per-run work if a policy diverges.
      bool deterministic=true;Json work_samples=Json::array();
      for(auto r:samples[mode]){r.erase("elapsed_ms");deterministic&=r==result;work_samples.push_back(r);}
      a::require(dynamic||deterministic,"synthetic fixed-budget counters changed across repetitions");
      result["work_deterministic"]=deterministic;if(!deterministic)result["work_samples"]=work_samples;
      result["scenario"]=scenarios[scenario];result["joint"]=bool(mode&1);result["rival"]=bool(mode&2);
      result["policy"]=policy==0?"matched_original_visits":policy==1?"matched_twice_original_visits":policy==2?"matched_four_times_original_visits":"dynamic_default_refinement_and_cover";
      result["elapsed_ms_samples"]=times;result["median_elapsed_ms"]=sorted[repeats/2];results.push_back(result);
    }
  }
  std::cout<<Json{{"complete",true},{"CUDA_executed",true},{"native_RuntimeGate_qualified",false},
    {"synthetic_numeric_gate_assumed",true},{"repetitions",repeats},{"mismatches",0},
    {"timing_scope","completed frontier conversion including initialization and scheduler allocations; excludes source transport/setup and signature verification"},
    {"results",results}}.dump(2)<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
