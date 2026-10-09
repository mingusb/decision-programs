#include "class_conversion/adaptive_identity_signature.cuh"
#include <chrono>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
using namespace class_conversion_adaptive::identity;
namespace {
void need(bool b,const char*message){if(!b)throw std::runtime_error(message);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
u32 evaluate(const std::vector<Node>&nodes,u32 root,u32 assignment){
  while(nodes[root].guard!=terminal_guard){const auto n=nodes[root];root=(assignment&(1u<<n.guard))?n.high:n.low;}return nodes[root].code;
}
bool same_outputs(const std::vector<Node>&nodes,u32 r,Pair p){
  for(u32 mask=0;mask<(1u<<r);++mask)if(evaluate(nodes,p.a,mask)!=evaluate(nodes,p.b,mask))return false;return true;
}
// Independent exact FREE-POLYNOMIAL oracle, bounded tiny fixtures only.
// Expands coefficients deliberately, unlike the production-candidate jets.
using Poly=std::map<std::string,__int128_t>;
std::vector<Poly>polynomials(const std::vector<Node>&nodes){
  std::vector<Poly>out(nodes.size());
  for(u32 i=0;i<nodes.size();++i){const auto n=nodes[i];
    if(n.guard==terminal_guard){if(n.code)out[i][""]=n.code;continue;}
    out[i]=out[n.low];const std::string g(1,char('a'+n.guard));
    for(const auto&[w,c]:out[n.high])out[i][g+w]+=c;
    for(const auto&[w,c]:out[n.low])out[i][g+w]-=c;
    for(auto it=out[i].begin();it!=out[i].end();)if(it->second==0)it=out[i].erase(it);else ++it;
  }return out;
}
std::vector<u32>structural(const std::vector<Node>&nodes){
  std::map<std::tuple<u32,u32,u32,u32>,u32>table;std::vector<u32>ids(nodes.size());u32 next=0;
  for(u32 i=0;i<nodes.size();++i){const auto n=nodes[i];
    if(n.guard!=terminal_guard&&ids[n.low]==ids[n.high]){ids[i]=ids[n.low];continue;}
    const auto key=n.guard==terminal_guard?std::tuple{n.guard,0u,0u,n.code}:std::tuple{n.guard,ids[n.low],ids[n.high],0u};
    auto[pos,inserted]=table.emplace(key,next);if(inserted)++next;ids[i]=pos->second;
  }return ids;
}
std::vector<Pair>all_pairs(u32 n){std::vector<Pair>out;for(u32 a=0;a<n;++a)for(u32 b=a;b<n;++b)out.push_back({a,b});return out;}
unsigned check_fragment(const std::vector<Node>&nodes,u32 guards){
  const auto pairs=all_pairs(u32(nodes.size()));const auto expected=polynomials(nodes);const auto result=certify(nodes,guards,pairs);
  need(result.plan.status==Status::ready,"valid fragment refused");unsigned checks=0;
  for(u32 i=0;i<pairs.size();++i){const auto p=pairs[i];const bool free_equal=expected[p.a]==expected[p.b];
    need((result.verdicts[i]==Verdict::independently_verified_boolean_equivalence)==free_equal,"exact modular jets disagree with expanded free-polynomial oracle");
    if(result.verdicts[i]==Verdict::independently_verified_boolean_equivalence)need(same_outputs(nodes,guards,p),"false semantic acceptance");++checks;
  }return checks;
}
__global__ void exhaustive_kernel(const Node*nodes,const Pair*pairs,u32 count,u32 assignments,u32*out){
  u32 i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;const Pair p=pairs[i];out[i]=1;
  for(u32 a=0;a<assignments;++a)if(detail::evaluate_code(nodes,p.a,a)!=detail::evaluate_code(nodes,p.b,a)){out[i]=0;return;}
}
// Deliberately forge an equal signature buffer for unequal terminal outputs.
// The independent acceptance check must reject even a broken jet implementation.
void check_independent_rejection(){
  const std::vector<Node>nodes={Node::terminal(0),Node::terminal(1)};
  const Pair pair{0,1};const u32 D=3,P=1;
  detail::DeviceAllocation<Node>dn;detail::DeviceAllocation<Pair>dq;
  detail::DeviceAllocation<u32>ds;detail::DeviceAllocation<Verdict>dv;
  ck(dn.allocate(2));ck(dq.allocate(1));ck(ds.allocate(3*D*P));ck(dv.allocate(1));
  ck(cudaMemcpy(dn.data,nodes.data(),nodes.size()*sizeof(Node),cudaMemcpyHostToDevice));
  ck(cudaMemcpy(dq.data,&pair,sizeof(Pair),cudaMemcpyHostToDevice));
  ck(cudaMemset(ds.data,0,3*D*P*sizeof(u32)));
  detail::compare_kernel<<<1,1>>>(dn.data,dq.data,1,D,P,ds.data,2,dv.data);
  ck(cudaGetLastError());ck(cudaDeviceSynchronize());Verdict result;
  ck(cudaMemcpy(&result,dv.data,sizeof(result),cudaMemcpyDeviceToHost));
  need(result==Verdict::verification_rejected,"forged equal jets bypass independent output check");
}
template<class F>float elapsed(F launch,unsigned iterations=20){
  cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));
  launch();ck(cudaDeviceSynchronize());ck(cudaEventRecord(a));for(unsigned i=0;i<iterations;++i)launch();
  ck(cudaEventRecord(b));ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));cudaEventDestroy(a);cudaEventDestroy(b);return ms/iterations;
}
void benchmark(const std::vector<Node>&nodes,u32 guards){
  auto pairs=all_pairs(u32(nodes.size()));auto plan=make_plan(nodes,guards);need(plan.status==Status::ready,"benchmark plan refused");
  detail::DeviceAllocation<Node>dn;detail::DeviceAllocation<Pair>dq;detail::DeviceAllocation<u32>dp,ds,plain;
  detail::DeviceAllocation<Verdict>dv;ck(dn.allocate(nodes.size()));ck(dq.allocate(pairs.size()));ck(dp.allocate(plan.primes.size()));
  ck(ds.allocate(plan.scratch_bytes/4));ck(plain.allocate(pairs.size()));ck(dv.allocate(pairs.size()));
  ck(cudaMemcpy(dn.data,nodes.data(),nodes.size()*sizeof(Node),cudaMemcpyHostToDevice));
  ck(cudaMemcpy(dq.data,pairs.data(),pairs.size()*sizeof(Pair),cudaMemcpyHostToDevice));
  ck(cudaMemcpy(dp.data,plan.primes.data(),plan.primes.size()*4,cudaMemcpyHostToDevice));
  const auto P=u32(plan.primes.size()),D=plan.dimension;
  const float jets=elapsed([&]{detail::jets_kernel<<<(P+127)/128,128>>>(dn.data,plan.nodes,D,dp.data,P,ds.data);});
  const float certified=elapsed([&]{detail::jets_kernel<<<(P+127)/128,128>>>(dn.data,plan.nodes,D,dp.data,P,ds.data);
    detail::compare_kernel<<<(pairs.size()+127)/128,128>>>(dn.data,dq.data,u32(pairs.size()),D,P,ds.data,1u<<guards,dv.data);});
  const float exhaustive=elapsed([&]{exhaustive_kernel<<<(pairs.size()+127)/128,128>>>(dn.data,dq.data,u32(pairs.size()),1u<<guards,plain.data);});
  constexpr unsigned repeats=1000;std::size_t consume=0;auto start=std::chrono::steady_clock::now();
  for(unsigned i=0;i<repeats;++i){auto ids=structural(nodes);for(auto p:pairs)consume+=ids[p.a]==ids[p.b];}
  const double cpu_structural=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repeats;
  start=std::chrono::steady_clock::now();for(unsigned i=0;i<repeats;++i)for(auto p:pairs)consume+=same_outputs(nodes,guards,p);
  const double cpu_exhaustive=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repeats;
  auto ids=structural(nodes);auto free=polynomials(nodes);u32 structural_hits=0,identity_hits=0,truth_hits=0;
  for(auto p:pairs){structural_hits+=ids[p.a]==ids[p.b];identity_hits+=free[p.a]==free[p.b];truth_hits+=same_outputs(nodes,guards,p);}
  std::cout<<"BENCH nodes="<<nodes.size()<<" guards="<<guards<<" pairs="<<pairs.size()<<" dimension="<<D<<" primes="<<P
    <<" bound_bits="<<plan.numerator_bits<<" scratch_bytes="<<plan.scratch_bytes<<" modular_cells="<<plan.jet_operations
    <<" gpu_jets_ms="<<jets<<" gpu_certified_ms="<<certified<<" gpu_exhaustive_ms="<<exhaustive
    <<" cpu_structural_us="<<cpu_structural<<" cpu_exhaustive_us="<<cpu_exhaustive<<" structural_hits="<<structural_hits
    <<" identity_hits="<<identity_hits<<" truth_hits="<<truth_hits<<" checksum="<<consume<<'\n';
}
} // namespace
int main(){try{
  ck(cudaSetDevice(0));unsigned checks=0;
  check_independent_rejection();++checks;
  // Repeated guard exposes a genuine distributive identity beyond hash-consing:
  // g*(2+g)+(1-g)*g = 3*g, with final codes 0 and 3 on feasible assignments.
  const std::vector<Node>factor={Node::terminal(0),Node::terminal(1),Node::terminal(2),Node::terminal(3),
    Node::branch(0,0,1),Node::branch(0,2,3),Node::branch(0,4,5),Node::branch(0,0,3)};
  auto baseline=structural(factor);need(baseline[6]!=baseline[7],"fixture already structurally shared");
  checks+=check_fragment(factor,1);auto accepted=certify(factor,1,{{6,7}});
  need(accepted.verdicts[0]==Verdict::independently_verified_boolean_equivalence,"distributive identity not detected");
  // Independent guard-order equivalence is intentionally not a free identity.
  const std::vector<Node>commute={Node::terminal(0),Node::terminal(1),Node::branch(0,0,1),Node::branch(1,0,1),
    Node::branch(0,0,3),Node::branch(1,0,2)};
  need(same_outputs(commute,2,{4,5}),"commutativity fixture differs");
  need(certify(commute,2,{{4,5}}).verdicts[0]==Verdict::different_signature,"free-word order unexpectedly collapsed");
  checks+=check_fragment(commute,2);
  // Exact FP32 word semantics: +0/-0 and different NaN payloads remain distinct.
  const std::vector<Node>words={Node::terminal(0),Node::terminal(0x80000000u),Node::terminal(0x7fc00001u),Node::terminal(0x7fc00002u)};
  checks+=check_fragment(words,1);
  // A deliberate collision at the FIRST prime is not accepted.
  const u32 prime=0x7fffffffu;const std::vector<Node>collision={Node::terminal(0),Node::terminal(prime)};
  auto collision_plan=make_plan(collision,1);need(collision_plan.primes.size()>1,"single-prime coverage accepted");
  need(collision[0].code%prime==collision[1].code%prime,"collision fixture failed");
  need(certify(collision,1,{{0,1}}).verdicts[0]==Verdict::different_signature,"single-prime collision accepted");
  need(!detail::product_exceeds_bound({prime},collision_plan.numerator_bits,collision_plan.dimension,1),"short CRT product accepted");
  need(!detail::product_exceeds_bound({prime,prime},0,3,1),"duplicate CRT primes accepted");checks+=5;
  // Malformed topology and resource exhaustion fail before any GPU acceptance.
  need(make_plan({Node::branch(0,0,0)},1).status==Status::invalid_fragment,"cycle accepted");
  need(make_plan({Node::terminal(0),Node::branch(1,0,0)},1).status==Status::invalid_fragment,"guard out of range accepted");
  Limits tiny;tiny.max_primes=1;need(certify(factor,1,{{6,7}},tiny).plan.status==Status::resource_limit,"prime budget ignored");
  tiny={};tiny.max_scratch_bytes=1;need(certify(factor,1,{{6,7}},tiny).plan.status==Status::resource_limit,"scratch budget ignored");
  need(certify(factor,1,{{6,99}}).plan.status==Status::invalid_fragment,"root out of range accepted");
  Limits research;research.verify_outputs=false;need(certify(factor,1,{{6,7}},research).verdicts[0]==Verdict::exact_jets_equal,"research result grants authority");checks+=6;
  // All 256 assignments are checked at the default verification guard limit.
  const std::vector<Node>eight={Node::terminal(0),Node::terminal(1),Node::branch(7,0,1),Node::branch(7,0,1)};
  checks+=check_fragment(eight,8);
  need(certify(eight,9,{{2,3}}).plan.status==Status::resource_limit,"guard budget ignored");++checks;
  // All pairs versus independently expanded exact coefficients and truth tables.
  std::mt19937 rng(20261008);std::vector<Node>last;
  for(unsigned trial=0;trial<20;++trial){std::vector<Node>nodes={Node::terminal(0),Node::terminal(1),Node::terminal(3),Node::terminal(UINT32_MAX)};
    for(u32 i=4;i<14;++i)nodes.push_back(Node::branch(rng()%3,rng()%i,rng()%i));
    checks+=check_fragment(nodes,3);last=nodes;
  }
  benchmark(factor,1);benchmark(last,3);
  std::cout<<"PASS focused_checks="<<checks<<" exact_acceptance=CRT_plus_exhaustive_guard_validation production_gate=not_integrated\n";
  return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
