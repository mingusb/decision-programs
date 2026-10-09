#pragma once
// Bounded F116 jet-signature prototype for ALREADY COMPACT decision DAGs.
// This header does not install or grant adaptive RuntimeGate authority.
// Terminal codes represent final class IDs or exact FP32 bit patterns, never
// unfinished floating-point score algebra. Guard i is an abstract Boolean;
// correlated guards are soundly overapproximated, but scalar equivalences
// relying on commutativity or Boolean idempotence can be missed.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace class_conversion_adaptive::identity {
using u32=std::uint32_t;
using u64=std::uint64_t;
constexpr u32 terminal_guard=UINT32_MAX;
struct Node {
  u32 guard=terminal_guard, low=0, high=0, code=0;
  static Node terminal(u32 word) { return {terminal_guard,0,0,word}; }
  static Node branch(u32 variable,u32 lo,u32 hi) { return {variable,lo,hi,0}; }
};
struct Pair { u32 a=0,b=0; };
enum class Status : u32 { ready, invalid_fragment, resource_limit, cuda_failure };
enum class Verdict : u32 { unavailable, different_signature, exact_jets_equal,
                           independently_verified_boolean_equivalence, verification_rejected };
struct Limits {
  u32 max_nodes=32,max_guards=8,max_dimension=512,max_primes=128;
  u64 max_scratch_bytes=16ULL*1024*1024;
  // Independently verifies final acceptance on ALL abstract guard assignments.
  // This bounded fallback avoids relying on an unchecked hitting-theorem port.
  // Set false only to obtain exact_jets_equal research results, never authority.
  bool verify_outputs=true;
  u32 max_verification_guards=8;
};
struct Plan {
  Status status=Status::invalid_fragment;
  u32 nodes=0,guards=0,dimension=0,height=0,numerator_bits=0;
  u64 scratch_bytes=0,jet_operations=0,verification_assignments=0;
  std::vector<u32> primes;
};
struct Report {
  Plan plan;
  std::vector<Verdict> verdicts;
  cudaError_t cuda_error=cudaSuccess;
};
namespace detail {
inline u32 ceil_log2(u32 x) {
  if(x<=1)return 0;
  u32 b=0; --x; while(x){++b;x>>=1;}return b;
}
__host__ __device__ inline u32 mul(u32 a,u32 b,u32 p) {
  return u32((u64(a)*b)%p);
}
__host__ __device__ inline u32 add(u32 a,u32 b,u32 p) {
  const u32 s=a+b;return s>=p?s-p:s;
}
__host__ __device__ inline u32 sub(u32 a,u32 b,u32 p) {
  return a>=b?a-b:p-(b-a);
}
inline u32 power(u32 a,u32 e,u32 p) {
  u32 value=1;while(e){if(e&1)value=mul(value,a,p);a=mul(a,a,p);e>>=1;}return value;
}
inline bool prime(u32 p) {
  if(p<2)return false;
  for(u32 a:{2u,3u,5u,7u,11u}){if(p==a)return true;if(p%a==0)return false;}
  u32 d=p-1,s=0;while(!(d&1)){d>>=1;++s;}
  // These bases are deterministic for every 32-bit unsigned integer.
  for(u32 a:{2u,3u,5u,7u,11u}){
    u32 x=power(a,d,p);if(x==1||x==p-1)continue;
    bool witness=true;for(u32 j=1;j<s;++j){x=mul(x,x,p);if(x==p-1){witness=false;break;}}
    if(witness)return false;
  }return true;
}
inline bool product_exceeds_bound(const std::vector<u32>&primes,u32 bits,u32 D,u32 r) {
  if(u64(primes.size())*30<=bits)return false;
  for(std::size_t i=0;i<primes.size();++i){const u32 p=primes[i];
    if(p<=(1u<<30)||p>=0x80000000u||p<=D||p<=r||!prime(p))return false;
    for(std::size_t j=0;j<i;++j)if(primes[j]==p)return false;
  }return true;
}
} // namespace detail

inline Plan make_plan(const std::vector<Node>&nodes,u32 guards,const Limits&limits={}) {
  Plan out;
  // Hard caps keep all resource arithmetic bounded, even for custom Limits.
  if(nodes.empty()||nodes.size()>4096||!guards||guards>256)return out;
  out.nodes=u32(nodes.size());out.guards=guards;
  if(nodes.size()>limits.max_nodes||guards>limits.max_guards){out.status=Status::resource_limit;return out;}
  std::vector<u32>heights(nodes.size());u32 maximum_code=0;
  for(u32 i=0;i<out.nodes;++i){const auto&n=nodes[i];
    if(n.guard==terminal_guard){maximum_code=std::max(maximum_code,n.code);continue;}
    if(n.guard>=guards||n.low>=i||n.high>=i)return out;
    heights[i]=1+std::max(heights[n.low],heights[n.high]);
    out.height=std::max(out.height,heights[i]);
  }
  // All continuations AND THEIR DIFFERENCES inhabit the SAME m-vertex ABP:
  // original nodes plus a common sink, with arbitrary signed source vectors.
  // Do not use a formula-tree size or expand shared descendants.
  const u64 m=u64(out.nodes)+1;
  const u64 D=(u64(guards)-1)*m*(m-1)/2+m;
  if(D>limits.max_dimension||D>UINT32_MAX){out.status=Status::resource_limit;return out;}
  out.dimension=u32(D);
  // For every jet coefficient j, S_j=j!*(r!)^j clears denominators.
  // Nested integration indices are strictly increasing, so their product
  // divides j!; guard-base exponents sum to j, so their product divides (r!)^j.
  // A height-H continuation has free-polynomial coefficient l1 norm <= C*3^H.
  // A word's jth jet coefficient has absolute value <= 2^j (compositions of j).
  // Therefore |S_j*(v_a-v_b)_j| <= 2*C*3^H*2^j*S_j.
  // Integer ceil-log bounds below deliberately overestimate this for ALL j<D.
  u64 factorial_bits=0,guard_factorial_bits=0;
  for(u32 j=1;j<out.dimension;++j)factorial_bits+=detail::ceil_log2(j);
  for(u32 j=1;j<=guards;++j)guard_factorial_bits+=detail::ceil_log2(j);
  const u64 j=D-1;
  const u64 bits=1+detail::ceil_log2(std::max(1u,maximum_code))+2*u64(out.height)+j+
    factorial_bits+j*guard_factorial_bits;
  if(bits>UINT32_MAX){out.status=Status::resource_limit;return out;}
  out.numerator_bits=u32(bits);
  const u64 count=bits/30+1; // strict 30*count > bits
  if(count>limits.max_primes||count>1024){out.status=Status::resource_limit;return out;}
  out.scratch_bytes=(u64(out.nodes)+1)*D*count*sizeof(u32); // inverse table + jets
  out.jet_operations=u64(out.nodes)*D*count;
  if(out.scratch_bytes>limits.max_scratch_bytes){out.status=Status::resource_limit;return out;}
  if(limits.verify_outputs){
    if(guards>limits.max_verification_guards||guards>16){out.status=Status::resource_limit;return out;}
    out.verification_assignments=u64(1)<<guards;
  }
  for(u32 candidate=0x7fffffffu;out.primes.size()<count;candidate-=2){
    if(candidate<=(1u<<30)){out.status=Status::resource_limit;return out;}
    if(detail::prime(candidate))out.primes.push_back(candidate);
  }
  if(!detail::product_exceeds_bound(out.primes,out.numerator_bits,out.dimension,guards))return out;
  out.status=Status::ready;return out;
}

namespace detail {
// Layout [row][coefficient][prime] coalesces lanes across prime moduli.
// Row 0 is an inverse table; row node+1 is its jet. All indices are validated
// on the host from the immutable Node copy used for this launch.
struct LaneAccessor {
  u32*scratch;u32 D,P,lane;
  __device__ u32&operator()(u32 row,u32 j)const{return scratch[(u64(row)*D+j)*P+lane];}
};
__global__ void jets_kernel(const Node*nodes,u32 count,u32 D,const u32*primes,u32 P,u32*scratch) {
  const u32 lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=P)return;
  const u32 p=primes[lane];
  const LaneAccessor at{scratch,D,P,lane};
  at(0,0)=0;at(0,1)=1;
  for(u32 j=2;j<D;++j)at(0,j)=p-mul(p/j,at(0,p%j),p);
  for(u32 index=0;index<count;++index){const Node n=nodes[index];const u32 row=index+1;
    if(n.guard==terminal_guard){at(row,0)=n.code%p;for(u32 j=1;j<D;++j)at(row,j)=0;continue;}
    at(row,0)=at(n.low+1,0);const u32 inv_guard=at(0,n.guard+1);u32 previous=0;
    for(u32 j=1;j<D;++j){
      const u32 u=sub(at(n.high+1,j-1),at(n.low+1,j-1),p);
      const u32 a=mul(sub(u,previous,p),inv_guard,p);
      at(row,j)=add(at(n.low+1,j),mul(a,at(0,j),p),p);previous=a;
    }
  }
}
__device__ inline u32 evaluate_code(const Node*nodes,u32 root,u32 assignment) {
  while(nodes[root].guard!=terminal_guard){const Node n=nodes[root];root=(assignment&(1u<<n.guard))?n.high:n.low;}
  return nodes[root].code;
}
__global__ void compare_kernel(const Node*nodes,const Pair*pairs,u32 count,u32 D,u32 P,
    const u32*scratch,u32 assignments,Verdict*out) {
  const u32 index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=count)return;
  const Pair pair=pairs[index];
  for(u32 j=0;j<D;++j)for(u32 lane=0;lane<P;++lane){
    if(scratch[(u64(pair.a+1)*D+j)*P+lane]!=scratch[(u64(pair.b+1)*D+j)*P+lane]){
      out[index]=Verdict::different_signature;return;
    }
  }
  // CRT gives EXACT jets here: the integer numerator is divisible by the
  // pairwise-coprime product, yet its absolute value is strictly smaller.
  out[index]=Verdict::exact_jets_equal;
  if(!assignments)return;
  for(u32 mask=0;mask<assignments;++mask){
    if(evaluate_code(nodes,pair.a,mask)!=evaluate_code(nodes,pair.b,mask)){
      out[index]=Verdict::verification_rejected;return;
    }
  }
  out[index]=Verdict::independently_verified_boolean_equivalence;
}
template<class T>struct DeviceAllocation {
  T*data=nullptr;
  DeviceAllocation()=default;DeviceAllocation(const DeviceAllocation&)=delete;
  DeviceAllocation&operator=(const DeviceAllocation&)=delete;
  ~DeviceAllocation(){if(data)cudaFree(data);}
  cudaError_t allocate(u64 count){return cudaMalloc(reinterpret_cast<void**>(&data),count*sizeof(T));}
};
} // namespace detail

// Standalone prototype wrapper: resource planning only is on CPU; modular
// jets, comparisons, and the independent bounded output check execute on GPU.
// `exact_jets_equal` alone is NOT exposed as established runtime authority.
// `independently_verified_boolean_equivalence` additionally checked every abstract guard assignment
// using original DAG traversal. This final bounded safety check is explicit;
// removing it requires a separately checked F116 ABP/denominator/jet bridge.
inline Report certify(const std::vector<Node>&nodes,u32 guards,const std::vector<Pair>&pairs,
                      const Limits&limits={}) {
  Report out;out.plan=make_plan(nodes,guards,limits);
  out.verdicts.assign(pairs.size(),Verdict::unavailable);
  if(out.plan.status!=Status::ready)return out;
  if(pairs.empty()||pairs.size()>65536){out.plan.status=Status::invalid_fragment;return out;}
  for(const auto&pair:pairs)if(pair.a>=nodes.size()||pair.b>=nodes.size()){
    out.plan.status=Status::invalid_fragment;return out;
  }
  detail::DeviceAllocation<Node>dn;detail::DeviceAllocation<Pair>dq;
  detail::DeviceAllocation<u32>dp,ds;detail::DeviceAllocation<Verdict>dv;
  auto checked=[&](cudaError_t error){if(error==cudaSuccess)return true;
    out.cuda_error=error;out.plan.status=Status::cuda_failure;
    std::fill(out.verdicts.begin(),out.verdicts.end(),Verdict::unavailable);return false;};
  if(!checked(dn.allocate(nodes.size()))||!checked(dq.allocate(pairs.size()))||
     !checked(dp.allocate(out.plan.primes.size()))||!checked(ds.allocate(out.plan.scratch_bytes/4))||
     !checked(dv.allocate(pairs.size())))return out;
  if(!checked(cudaMemcpy(dn.data,nodes.data(),nodes.size()*sizeof(Node),cudaMemcpyHostToDevice))||
     !checked(cudaMemcpy(dq.data,pairs.data(),pairs.size()*sizeof(Pair),cudaMemcpyHostToDevice))||
     !checked(cudaMemcpy(dp.data,out.plan.primes.data(),out.plan.primes.size()*4,cudaMemcpyHostToDevice)))return out;
  const u32 P=u32(out.plan.primes.size()),D=out.plan.dimension;
  detail::jets_kernel<<<(P+127)/128,128>>>(dn.data,out.plan.nodes,D,dp.data,P,ds.data);
  if(!checked(cudaGetLastError()))return out;
  detail::compare_kernel<<<(pairs.size()+127)/128,128>>>(dn.data,dq.data,u32(pairs.size()),D,P,ds.data,
    u32(out.plan.verification_assignments),dv.data);
  if(!checked(cudaGetLastError())||!checked(cudaDeviceSynchronize())||
     !checked(cudaMemcpy(out.verdicts.data(),dv.data,pairs.size()*sizeof(Verdict),cudaMemcpyDeviceToHost)))return out;
  return out;
}
} // namespace class_conversion_adaptive::identity
