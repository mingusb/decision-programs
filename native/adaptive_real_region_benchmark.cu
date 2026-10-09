// Real-source restricted-domain comparisons using the maintained adaptive engine.
// No training, no second converter, and no native authority from stored receipts.
#include "class_conversion/class_io.hpp"
#include "class_study_native_gate.hpp"
#include "class_conversion/adaptive_parallel_schedule.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/system/cuda/execution_policy.h>
#include <array>
#include <bit>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>

using J=nlohmann::json;
using U=std::uint64_t;
namespace fs=std::filesystem;
namespace a=class_conversion_adaptive;
namespace d=class_conversion_adaptive_domain;
namespace af=a::frontier;
using u32=std::uint32_t;
using u64=std::uint64_t;
void need(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
void cu(cudaError_t error){a::check(error,"real-region benchmark CUDA");}
void gpu_sync(){a::synchronize();}
#include "class_model_native_adapter.cuh"

namespace {
using Clock=std::chrono::steady_clock;
double milliseconds(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
struct Options {
 fs::path model,training,library,helper,qualification,snapshot,output,region_json;
 std::vector<u64> sample_indices;
 u32 count=8,cells=2,repetitions=3,inspect_states=0,initial_states=1024;
 u64 byte_budget=2ull<<30,verification_limit=1ull<<20,witness_lookups=0,prefix_reuse_pairs=0;
 std::string policy="both",method="relational";
 bool prefix_gap_evidence=false;
};
u64 number(const std::string& text){std::size_t n=0;auto v=std::stoull(text,&n);need(n==text.size()&&!text.empty()&&text[0]!='-',"invalid unsigned CLI value");return v;}
Options parse(int argc,char**argv){
 Options o;bool count_given=false,cells_given=false;
 for(int i=1;i<argc;i+=2){need(i+1<argc,"each option requires a value");const std::string k=argv[i],v=argv[i+1];
  if(k=="--model")o.model=v;else if(k=="--training")o.training=v;else if(k=="--library")o.library=v;
  else if(k=="--gate-helper")o.helper=v;else if(k=="--gate-qualification")o.qualification=v;else if(k=="--gate-snapshot")o.snapshot=v;
  else if(k=="--output")o.output=v;else if(k=="--policy")o.policy=v;else if(k=="--method")o.method=v;
  else if(k=="--region-json")o.region_json=v;
  else if(k=="--sample-indices"){
   need(o.sample_indices.empty(),"sample indices may only be specified once");std::size_t first=0;
   do{const auto end=v.find(',',first);const auto index=number(v.substr(first,end==std::string::npos?end:end-first));
    need(std::find(o.sample_indices.begin(),o.sample_indices.end(),index)==o.sample_indices.end(),"duplicate sample index");o.sample_indices.push_back(index);
    if(end==std::string::npos)break;first=end+1;
   }while(true);
  }
  else if(k=="--count"){auto n=number(v);need(n>0&&n<=65536,"count outside 1..65536");o.count=u32(n);count_given=true;}
  else if(k=="--cells"){auto n=number(v);need(n>0&&n<=64,"cells outside 1..64");o.cells=u32(n);cells_given=true;}
  else if(k=="--repetitions"){auto n=number(v);need(n>0&&n<=1000,"repetitions outside 1..1000");o.repetitions=u32(n);}
  else if(k=="--initial-states"){auto n=number(v);need(n>0&&n<=0x3fffffffull,"initial states outside 1..1073741823");o.initial_states=u32(n);}
  else if(k=="--inspect-states"){auto n=number(v);need(n<=65536,"inspection capacity outside 0..65536");o.inspect_states=u32(n);}
  else if(k=="--prefix-reuse-pairs")o.prefix_reuse_pairs=number(v);
  else if(k=="--prefix-gap-evidence"){const auto n=number(v);need(n<=1,"prefix gap evidence must be 0 or 1");o.prefix_gap_evidence=n!=0;}
  else if(k=="--witness-lookups")o.witness_lookups=number(v);
  else if(k=="--gpu-bytes")o.byte_budget=number(v);
  else if(k=="--verification-limit")o.verification_limit=number(v);
  else throw std::runtime_error("unknown option: "+k);
 }
 for(const auto&p:{o.model,o.training,o.library,o.helper,o.qualification,o.snapshot})need(!p.empty()&&p.is_absolute(),"all source and gate paths must be absolute");
 need(o.output.empty()||o.output.is_absolute(),"output path must be absolute");
 need(o.region_json.empty()==o.sample_indices.empty(),"--region-json and --sample-indices must be supplied together");
 if(!o.region_json.empty()){
  need(o.region_json.is_absolute(),"region JSON path must be absolute");need(!count_given&&!cells_given,"frozen proposals use --sample-indices instead of --count/--cells");
  need(o.sample_indices.size()<=65536,"too many sample indices");o.count=u32(o.sample_indices.size());
 }
 need(o.policy=="fixed"||o.policy=="dynamic"||o.policy=="both","policy must be fixed, dynamic or both");
 need(o.method=="relational"||o.method=="unary"||o.method=="two_point","method must be relational, unary or two_point");
 need(o.byte_budget>0&&o.verification_limit>0,"GPU bytes and verification limit must be positive");
 need(!o.prefix_gap_evidence||o.prefix_reuse_pairs>0,"--prefix-gap-evidence requires --prefix-reuse-pairs greater than zero");
 return o;
}
// Stored geometry is an untrusted proposal; all runtime authority still comes
// from the unchanged source and the same-process native gate.
struct FrozenBox {u32 lo[10]{},hi[10]{};u64 allowed=0,sample_index=0,task_id=0;};
struct FrozenInput {std::vector<FrozenBox>boxes;std::vector<u32>root_max;std::string sha;};
u64 json_unsigned(const J&v,const std::string&name){
 need(v.is_number_unsigned()||(v.is_number_integer()&&v.get<std::int64_t>()>=0),name+" must be a nonnegative integer");return v.get<u64>();
}
FrozenInput read_frozen(const Options&o,const std::string&source_identity){
 FrozenInput out;if(o.region_json.empty())return out;
 const auto bytes=dpnative::read_text(o.region_json);const auto input=J::parse(bytes);out.sha=dpnative::sha256(bytes);
 need(input.at("format")=="paired-pending-box-sample-1","unsupported frozen proposal format");
 need(input.at("source_sha256")==source_identity,"frozen proposal source identity differs");
 const auto&root=input.at("root_box");const auto&root_lo=root.at("lo");const auto&root_hi=root.at("hi");
 need(root_lo.is_array()&&root_hi.is_array()&&root_lo.size()==10&&root_hi.size()==10,"frozen root dimensions differ");
 need(json_unsigned(root.at("allowed"),"root mask")==((u64(1)<<44)-1),"frozen root must describe the existing declared domain");
 for(u32 f=0;f<10;++f){need(json_unsigned(root_lo[f],"root lower rank")==0,"frozen root lower rank must be zero");
  const auto hi=json_unsigned(root_hi[f],"root upper rank");need(hi<UINT32_MAX,"frozen root rank too large");out.root_max.push_back(u32(hi));}
 const auto&tasks=input.at("tasks");need(tasks.is_array(),"frozen tasks must be an array");
 for(auto index:o.sample_indices){need(index<tasks.size(),"sample index outside frozen tasks");const auto&task=tasks[index];const auto&box=task.at("box");
  const auto&lo=box.at("lo");const auto&hi=box.at("hi");need(lo.is_array()&&hi.is_array()&&lo.size()==10&&hi.size()==10,"frozen box dimensions differ");
  FrozenBox b{};b.sample_index=index;b.task_id=json_unsigned(task.at("id"),"task id");b.allowed=json_unsigned(box.at("allowed"),"box mask");
  need((b.allowed>>44)==0&&(b.allowed&15)&&((b.allowed>>4)&((u64(1)<<40)-1)),"frozen one-hot masks are invalid");
  for(u32 f=0;f<10;++f){const auto low=json_unsigned(lo[f],"lower rank"),high=json_unsigned(hi[f],"upper rank");
   need(low<=high&&high<=out.root_max[f],"frozen ranks outside root");b.lo[f]=u32(low);b.hi[f]=u32(high);}
  out.boxes.push_back(b);
 }
 return out;
}
struct Native {
 native_class_reference::Api api;u32 classes;
 Native(const fs::path&library,const std::string& bytes,u32 F,u32 K):api(library,F),classes(K){
  api.reset(false);api.check(api.symbol<int(*)(void*,const void*,U)>("XGBoosterLoadModelFromBuffer")(api.model,bytes.data(),bytes.size()),"load benchmark source from RAM");
  api.set("device","cuda:0");api.set("nthread","4");gpu_sync();
 }
 const float*predict(const float*x,u64 rows,bool margins){
  gpu_sync();const U*shape=nullptr;U dimensions=0;const float*out=nullptr;
  const auto input=api.array(x,rows);
  auto options=J{{"type",margins?1:0},{"training",false},{"iteration_begin",0},{"iteration_end",0},{"strict_shape",true},{"cache_id",0}}.dump();
  options.pop_back();options+=",\"missing\":NaN}";
  api.check(api.symbol<int(*)(void*,const char*,const char*,void*,const U**,U*,const float**)>("XGBoosterPredictFromCudaArray")(api.model,input.c_str(),options.c_str(),nullptr,&shape,&dimensions,&out),"real source native CUDA prediction");
  need(shape&&dimensions==2&&shape[0]==rows&&shape[1]==classes&&out,"native softprob/margin output shape differs");
  cudaPointerAttributes attr{};cu(cudaPointerGetAttributes(&attr,out));need(attr.type==cudaMemoryTypeDevice&&attr.device==0,"native output must remain on CUDA0");gpu_sync();return out;
 }
};
struct SourceStorage {
 a::Buffer<std::int32_t> feature,left,right,roots,channels;
 a::Buffer<float> cut,value,bias;a::Buffer<std::uint8_t> missing;
 a::Buffer<u32> minimum,maximum,audit,walk;
 a::Buffer<u64> support;
 a::SourceView view;u32 support_words;
 SourceStorage(a::Budget&b,const class_conversion_native::SourceData&s):
 feature(b,s.feature.size()),left(b,s.left.size()),right(b,s.right.size()),roots(b,s.roots.size()),channels(b,s.channels.size()),
 cut(b,s.cut.size()),value(b,s.value.size()),bias(b,s.bias.size()),missing(b,s.missing_left.size()),
 minimum(b,s.feature.size()),maximum(b,s.feature.size()),audit(b,3*s.feature.size()),walk(b,2),
 support(b,s.feature.size()*((u64(s.features)+63)/64)),support_words((u32(s.features)+63)/64){
 feature.upload(s.feature);left.upload(s.left);right.upload(s.right);roots.upload(s.roots);channels.upload(s.channels);
 cut.upload(s.cut);value.upload(s.value);bias.upload(s.bias);missing.upload(s.missing_left);
 view={feature.data,left.data,right.data,roots.data,channels.data,cut.data,value.data,bias.data,missing.data,
  u32(s.features),u32(s.outputs),u32(s.feature.size()),u32(s.roots.size()),u32(s.native_margin_outputs)};
 a::subtree_extrema<<<1,1>>>(view,minimum.data,maximum.data,audit.data,support.data,support_words,walk.data);gpu_sync();
 }
};
struct DomainStorage {
 a::Buffer<std::int32_t>group,numeric;a::Buffer<u32>bit,word_offsets,feature_offsets,widths,features;a::Buffer<u64>masks;d::DomainView view;
 DomainStorage(a::Budget&b,const d::HostMetadata&h):group(b,h.feature_group.size()),numeric(b,h.feature_numeric.size()),
 bit(b,h.feature_bit.size()),word_offsets(b,h.group_word_offsets.size()),feature_offsets(b,h.group_feature_offsets.size()),
 widths(b,h.group_widths.size()),features(b,h.group_features.size()),masks(b,h.initial_masks.size()){
 group.upload(h.feature_group);numeric.upload(h.feature_numeric);bit.upload(h.feature_bit);word_offsets.upload(h.group_word_offsets);
 feature_offsets.upload(h.group_feature_offsets);widths.upload(h.group_widths);features.upload(h.group_features);masks.upload(h.initial_masks);
 view={h.features,h.groups,h.mask_words,h.allow_nan,group.data,bit.data,word_offsets.data,feature_offsets.data,widths.data,features.data,masks.data,h.numeric_features,numeric.data};
 }
};
struct RegionStorage {
 a::Buffer<u32>lower,upper,missing; a::Buffer<u64>allowed;
 RegionStorage(a::Budget&b,d::DomainView d):lower(b,d.numeric_features),upper(b,d.numeric_features),missing(b,d.numeric_features),allowed(b,d.mask_words){}
 d::RegionView view(){return {lower.data,upper.data,missing.data,allowed.data};}
};
__global__ void threshold_keys(a::SourceView s,d::DomainView domain,u64*keys){
 const u32 i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=s.nodes)return;
 keys[i]=UINT64_MAX;if(s.left[i]<0)return;
 const u32 f=u32(s.feature[i]);if(domain.feature_numeric[f]<0)return;
 const float cut=s.cut[i];if(!isfinite(cut))return;const u32 key=d::sortable_word(__float_as_uint(cut));
 if(key>d::finite_min_key&&key<=d::finite_max_key)keys[i]=(u64(f)<<32)|key;
}
__device__ u32 lower_key(const u64*keys,u32 count,u64 value){u32 lo=0,hi=count;while(lo<hi){u32 m=lo+(hi-lo)/2;if(keys[m]<value)lo=m+1;else hi=m;}return lo;}
struct RegionInfo {u64 row=0,signatures=1;u32 bad=0;};
// Select adjacent complete source-threshold atoms, not integer-only points.
__global__ void make_region(a::SourceView s,d::DomainView domain,const u64*keys,u32 key_count,
 const float*training,u64 rows,u32 region_id,u32 region_count,u32 width,d::RegionView r,
 u32*starts,u32*counts,RegionInfo*out){
 if(blockIdx.x||threadIdx.x)return;*out={};out->row=u64(region_id)*rows/region_count;
 if(!d::initial_domain(domain,r)){out->bad=1;return;}
 const float*row=training+out->row*s.features;
 for(u32 f=0;f<s.features;++f){const auto slot=domain.feature_numeric[f];if(slot<0)continue;
  if(!isfinite(row[f])){out->bad=2;return;}
  const u32 begin=lower_key(keys,key_count,u64(f)<<32),end=lower_key(keys,key_count,u64(f+1)<<32);
  const u32 cuts=end-begin;const u32 rank=lower_key(keys+begin,cuts,(u64(f)<<32)|u64(d::sortable_word(__float_as_uint(row[f])))+1);
  const u32 cells=min(width,cuts+1);u32 start=rank>(cells-1)/2?rank-(cells-1)/2:0;
  if(start+cells>cuts+1)start=cuts+1-cells;
  starts[slot]=start;counts[slot]=cells;
  r.lower[slot]=start?u32(keys[begin+start-1]):d::finite_min_key;
  r.upper[slot]=start+cells<=cuts?d::predecessor_key(u32(keys[begin+start+cells-1])):d::finite_max_key;r.missing[slot]=0;
  if(out->signatures>UINT64_MAX/cells){out->bad=3;return;}out->signatures*=cells;
 }
 for(u32 g=0;g<domain.groups;++g){u32 chosen=UINT32_MAX;
  for(u32 j=domain.group_feature_offsets[g];j<domain.group_feature_offsets[g+1];++j){const float v=row[domain.group_features[j]];
   if(v==1.f){if(chosen!=UINT32_MAX){out->bad=4;return;}chosen=j-domain.group_feature_offsets[g];}else if(v!=0.f){out->bad=4;return;}}
  if(chosen==UINT32_MAX){out->bad=4;return;}
  for(u32 w=domain.group_word_offsets[g];w<domain.group_word_offsets[g+1];++w)r.allowed[w]=0;
  r.allowed[domain.group_word_offsets[g]+chosen/64]=u64(1)<<(chosen%64);
 }
 if(!d::region_valid(domain,r))out->bad=5;
}
__global__ void make_frozen_region(a::SourceView s,d::DomainView domain,const u64*keys,u32 key_count,
 const FrozenBox*proposals,const u32*root_max,u32 id,d::RegionView r,u32*starts,u32*counts,RegionInfo*out){
 if(blockIdx.x||threadIdx.x)return;*out={};out->row=UINT64_MAX;
 if(!d::initial_domain(domain,r)){out->bad=1;return;}const auto box=proposals[id];
 if(s.features!=54||domain.numeric_features!=10||domain.groups!=2||domain.mask_words!=2){out->bad=2;return;}
 for(u32 f=0;f<10;++f){const auto slot=domain.feature_numeric[f];if(slot<0){out->bad=2;return;}
  const u32 begin=lower_key(keys,key_count,u64(f)<<32),end=lower_key(keys,key_count,u64(f+1)<<32),cuts=end-begin;
  if(cuts!=root_max[f]||box.lo[f]>box.hi[f]||box.hi[f]>cuts){out->bad=2;return;}
  const u32 cells=box.hi[f]-box.lo[f]+1;starts[slot]=box.lo[f];counts[slot]=cells;
  r.lower[slot]=box.lo[f]?u32(keys[begin+box.lo[f]-1]):d::finite_min_key;
  r.upper[slot]=box.hi[f]<cuts?d::predecessor_key(u32(keys[begin+box.hi[f]])):d::finite_max_key;r.missing[slot]=0;
  if(out->signatures>UINT64_MAX/cells){out->bad=3;return;}out->signatures*=cells;
 }
 r.allowed[0]=box.allowed&15;r.allowed[1]=(box.allowed>>4)&((u64(1)<<40)-1);
 if((box.allowed>>44)||!r.allowed[0]||!r.allowed[1]){out->bad=4;return;}
 for(u32 g=0;g<2;++g){const u32 categories=__popcll(r.allowed[g]);if(out->signatures>UINT64_MAX/categories){out->bad=3;return;}out->signatures*=categories;}
 if(!d::region_valid(domain,r))out->bad=5;
}
__device__ void signature_row(a::SourceView s,d::DomainView domain,const u64*keys,u32 key_count,
 d::RegionView r,const u32*starts,const u32*counts,u64 index,float*row){
 d::region_witness(domain,r,row,0);
 for(u32 f=0;f<s.features;++f){const auto n=domain.feature_numeric[f];if(n<0)continue;
  const u32 cell=starts[n]+u32(index%counts[n]);index/=counts[n];
  const u32 begin=lower_key(keys,key_count,u64(f)<<32);const u32 key=cell?u32(keys[begin+cell-1]):d::finite_min_key;
  row[f]=__uint_as_float(d::word_from_key(key));
 }
 // Enumerate every permitted categorical signature, including multi-bit boxes.
 for(u32 g=0;g<domain.groups;++g){u32 cardinality=0;
  for(u32 w=domain.group_word_offsets[g];w<domain.group_word_offsets[g+1];++w)cardinality+=__popcll(r.allowed[w]);
  const u32 chosen=u32(index%cardinality);index/=cardinality;u32 ordinal=0;
  for(u32 j=domain.group_feature_offsets[g];j<domain.group_feature_offsets[g+1];++j){const u32 bit=j-domain.group_feature_offsets[g];
   const bool permitted=(r.allowed[domain.group_word_offsets[g]+bit/64]>>(bit%64))&1;
   row[domain.group_features[j]]=permitted&&ordinal==chosen?1.f:0.f;if(permitted)++ordinal;
  }
 }
}
__global__ void signature_rows(a::SourceView s,d::DomainView domain,const u64*keys,u32 key_count,
 d::RegionView r,const u32*starts,const u32*counts,u64 offset,u32 count,float*rows){
 const u32 i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)signature_row(s,domain,keys,key_count,r,starts,counts,offset+i,rows+u64(i)*s.features);
}
__global__ void decode_expected(const float*p,u32 count,u32 K,u32*out,u32*bad){
 const u32 i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;u32 best=0;
 for(u32 c=0;c<K;++c){float v=p[u64(i)*K+c];if(!isfinite(v)||v<0.f||v>1.f)atomicOr(bad,1u);if(v>p[u64(i)*K+best])best=c;}out[i]=best;
}
__global__ void compare_graph(const a::Node*nodes,u32 node_count,u32 root,const float*rows,u32 F,
 const u32*expected,u32 count,u32*bad){
 const u32 i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;u32 at=root,steps=0;
 while(at<node_count&&nodes[at].feature!=-1&&steps++<node_count){const auto n=nodes[at];const bool ml=n.feature<=-2;
  const u32 f=u32(ml?-std::int64_t(n.feature)-2:n.feature);if(f>=F){atomicOr(bad,2u);return;}
  const float x=rows[u64(i)*F+f];at=(isnan(x)?ml:x<__uint_as_float(n.payload))?n.left:n.right;
 }
 if(at>=node_count||nodes[at].feature!=-1||nodes[at].payload!=expected[i])atomicOr(bad,4u);
}
__global__ void reachable_nodes(const a::Node*nodes,u32 count,u32 root,u32*marks,u32*out){
 if(blockIdx.x||threadIdx.x)return;*out=0;if(root>=count)return;marks[root]=1;
 for(u32 i=count;i>0;){--i;if(!marks[i])continue;++*out;const auto n=nodes[i];if(n.feature!=-1){if(n.left<i)marks[n.left]=1;if(n.right<i)marks[n.right]=1;}}
}

// Post-conversion diagnostic only: query labels already obtained from the
// unchanged native source on the root signature grid. No certificate admission.
enum class WitnessReason:u32 {none,missing,invalid_region,outside_root,unaligned,invalid_grid,invalid_label,lookup_limit,membership};
struct WitnessResult {
 u64 local_signatures,lookups,first_index,second_index;
 u32 expanded,classification,reason,first_class,second_class;
}; // classification: 0 unknown, 1 two native labels, 2 exhaustive saved-context constant.
__device__ WitnessResult empty_witness_result(){return {0,0,UINT64_MAX,UINT64_MAX,0,0,0,a::none,a::none};}
__device__ u32 witness_grid(a::SourceView s,d::DomainView domain,const u64*keys,u32 key_count,
 d::RegionView root,const u32*starts,const u32*counts,d::RegionView witness,u64 signatures,
 u64*strides,u32*lows,u32*radices,u64&local){
 const u32 N=domain.numeric_features;
 for(u32 n=0;n<N;++n)if(root.missing[n]||witness.missing[n])return u32(WitnessReason::missing);
 if(!d::region_valid(domain,root)||!d::region_valid(domain,witness))return u32(WitnessReason::invalid_region);
 u64 total=1;local=1;
 for(u32 f=0;f<s.features;++f){const auto slot=domain.feature_numeric[f];if(slot<0)continue;
  const u32 n=u32(slot);if(n>=N)return u32(WitnessReason::invalid_grid);
  if(witness.lower[n]<root.lower[n]||witness.upper[n]>root.upper[n])return u32(WitnessReason::outside_root);
  const u32 begin=lower_key(keys,key_count,u64(f)<<32),end=lower_key(keys,key_count,u64(f+1)<<32),cuts=end-begin;
  if(!counts[n]||starts[n]>cuts||counts[n]>u64(cuts)+1-starts[n])return u32(WitnessReason::invalid_grid);
  const u32 root_high=starts[n]+counts[n]-1;
  const u32 root_low_key=starts[n]?u32(keys[begin+starts[n]-1]):d::finite_min_key;
  const u32 root_high_key=root_high<cuts?d::predecessor_key(u32(keys[begin+root_high])):d::finite_max_key;
  if(root.lower[n]!=root_low_key||root.upper[n]!=root_high_key)return u32(WitnessReason::unaligned);
  const u32 lo=lower_key(keys+begin,cuts,(u64(f)<<32)|(u64(witness.lower[n])+1));
  const u32 hi=lower_key(keys+begin,cuts,(u64(f)<<32)|(u64(witness.upper[n])+1));
  const u32 low_key=lo?u32(keys[begin+lo-1]):d::finite_min_key;
  const u32 high_key=hi<cuts?d::predecessor_key(u32(keys[begin+hi])):d::finite_max_key;
  if(witness.lower[n]!=low_key||witness.upper[n]!=high_key)return u32(WitnessReason::unaligned);
  if(lo<starts[n]||hi>root_high||lo>hi)return u32(WitnessReason::outside_root);
  lows[n]=lo;radices[n]=hi-lo+1;strides[n]=total;
  if(total>UINT64_MAX/counts[n]||local>UINT64_MAX/radices[n])return u32(WitnessReason::invalid_grid);
  total*=counts[n];local*=radices[n];
 }
 for(u32 g=0;g<domain.groups;++g){u32 full=0,part=0;
  for(u32 w=domain.group_word_offsets[g];w<domain.group_word_offsets[g+1];++w){
   if(witness.allowed[w]&~root.allowed[w])return u32(WitnessReason::outside_root);
   full+=__popcll(root.allowed[w]);part+=__popcll(witness.allowed[w]);}
  if(!full||!part||total>UINT64_MAX/full||local>UINT64_MAX/part)return u32(WitnessReason::invalid_grid);
  strides[N+g]=total;radices[N+g]=part;total*=full;local*=part;
 }
 return total==signatures?0:u32(WitnessReason::invalid_grid);
}
__device__ bool witness_root_index(a::SourceView s,d::DomainView domain,d::RegionView root,
 const u32*starts,d::RegionView witness,const u64*strides,const u32*lows,const u32*radices,
 u64 local,u64 signatures,u64&global){
 global=0;const u32 N=domain.numeric_features;
 for(u32 f=0;f<s.features;++f){const auto n=domain.feature_numeric[f];if(n<0)continue;
  const u32 digit=u32(local%radices[n]);local/=radices[n];global+=(u64(lows[n])+digit-starts[n])*strides[n];}
 for(u32 g=0;g<domain.groups;++g){u32 remaining=u32(local%radices[N+g]);local/=radices[N+g];u32 ordinal=0;bool found=false;
  for(u32 w=domain.group_word_offsets[g];w<domain.group_word_offsets[g+1];++w){u64 bits=witness.allowed[w];const u32 available=__popcll(bits);
   if(remaining>=available){remaining-=available;ordinal+=__popcll(root.allowed[w]);continue;}
   while(remaining){bits&=bits-1;--remaining;}const u32 bit=u32(__ffsll(static_cast<long long>(bits))-1);
   const u64 below=bit?((u64(1)<<bit)-1):0;ordinal+=__popcll(root.allowed[w]&below);
   global+=u64(ordinal)*strides[N+g];found=true;break;}
  if(!found)return false;
 }
 return local==0&&global<signatures;
}
__device__ bool witness_row_member(a::SourceView s,d::DomainView domain,d::RegionView r,const float*row){
 for(u32 f=0;f<s.features;++f){const auto n=domain.feature_numeric[f];if(n<0)continue;
  if(!isfinite(row[f]))return false;const u32 key=d::sortable_word(__float_as_uint(row[f]));
  if(key<r.lower[n]||key>r.upper[n])return false;}
 for(u32 g=0;g<domain.groups;++g){u32 selected=0;
  for(u32 j=domain.group_feature_offsets[g];j<domain.group_feature_offsets[g+1];++j){const float value=row[domain.group_features[j]];
   if(value==0.f)continue;if(value!=1.f)return false;const u32 bit=j-domain.group_feature_offsets[g];
   if(!(r.allowed[domain.group_word_offsets[g]+bit/64]&(u64(1)<<(bit%64))))return false;++selected;}
  if(selected!=1)return false;}
 return true;
}
// All 32 lanes call together. The first lookup is counted once; tiles stop as
// soon as they contain a differing label. No row outside this local box is read.
__device__ __noinline__ void scan_witness_grid(a::SourceView source,d::DomainView domain,const u64*keys,u32 key_count,
 d::RegionView root,const u32*starts,const u32*counts,d::RegionView witness,const u64*strides,
 const u32*lows,const u32*radices,const u32*expected,u64 signatures,u64 maximum,
 WitnessResult&out,float*rows){
 const u32 lane=threadIdx.x;const u64 limit=min(maximum,out.local_signatures);
 if(lane==0){u64 index=0;
  if(!limit||!witness_root_index(source,domain,root,starts,witness,strides,lows,radices,0,signatures,index))out.reason=u32(WitnessReason::invalid_grid);
  else{out.first_index=index;out.first_class=expected[index];out.lookups=1;if(out.first_class>=source.classes)out.reason=u32(WitnessReason::invalid_label);}}
 __syncthreads();if(out.reason)return;
 for(u64 base=1;base<limit;){const bool active=u64(lane)<limit-base;const u64 local=active?base+lane:limit;u64 index=0;u32 label=out.first_class;
  bool invalid=false,loaded=false;if(local<limit){invalid=!witness_root_index(source,domain,root,starts,witness,strides,lows,radices,local,signatures,index);
   if(!invalid){label=expected[index];loaded=true;invalid=label>=source.classes;}}
  const u32 reads=__ballot_sync(0xffffffffu,loaded);
  const u32 bad=__ballot_sync(0xffffffffu,invalid),different=__ballot_sync(0xffffffffu,local<limit&&!invalid&&label!=out.first_class);
  const u32 chosen=different?u32(__ffs(different)-1):0;
  const u64 differing_index=__shfl_sync(0xffffffffu,static_cast<unsigned long long>(index),chosen);
  const u32 differing_class=__shfl_sync(0xffffffffu,label,chosen);
  if(lane==0){out.lookups+=__popc(reads);if(bad)out.reason=u32(WitnessReason::invalid_label);
   else if(different){out.classification=1;out.second_index=differing_index;out.second_class=differing_class;}}
  __syncthreads();if(bad||different||limit-base<=32)break;base+=32;
 }
 if(lane==0){
  if(!out.reason&&!out.classification){if(out.lookups==out.local_signatures)out.classification=2;else out.reason=u32(WitnessReason::lookup_limit);}
  if(out.classification){signature_row(source,domain,keys,key_count,root,starts,counts,out.first_index,rows);
   bool member=witness_row_member(source,domain,witness,rows);
   if(out.classification==1){signature_row(source,domain,keys,key_count,root,starts,counts,out.second_index,rows+source.features);member=member&&witness_row_member(source,domain,witness,rows+source.features);}
   if(!member){out.classification=0;out.reason=u32(WitnessReason::membership);}}
 }
 __syncthreads();
}
__global__ void inspect_native_witnesses(a::EngineView e,d::RegionView root,const u32*starts,const u32*counts,
 const u64*keys,u32 key_count,const u32*expected,u64 signatures,u64 maximum,WitnessResult*results,float*rows){
 const u32 slot=blockIdx.x;if(slot>=e.status->states)return;
 const auto state=e.arena.states[slot];if(state.phase!=3||state.left==a::none||state.right==a::none)return;
 const u32 N=e.domain.numeric_features,A=N+e.domain.groups;
 extern __shared__ u64 workspace[];auto*strides=workspace;auto*lows=reinterpret_cast<u32*>(strides+A);auto*radices=lows+N;
 __shared__ WitnessResult result;
 const auto witness=a::witness_region(e,slot);
 if(threadIdx.x==0){result=empty_witness_result();result.expanded=1;result.reason=witness_grid(e.source,e.domain,keys,key_count,root,starts,counts,witness,signatures,strides,lows,radices,result.local_signatures);if(result.reason)result.local_signatures=0;}
 __syncthreads();if(!result.reason)scan_witness_grid(e.source,e.domain,keys,key_count,root,starts,counts,witness,strides,lows,radices,expected,signatures,maximum,result,rows+u64(slot)*2*e.source.features);
 if(threadIdx.x==0)results[slot]=result;
}
__global__ void capture_constant_contexts(a::EngineView e,const u32*ids,u32 count,u32*records,u64*masks){
 const u32 i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;const u32 id=ids[i],N=e.domain.numeric_features,K=e.source.classes,T=e.source.trees,W=e.domain.mask_words;
 u32*out=records+u64(i)*(2*K+T+6*N);for(u32 c=0;c<K;++c)*out++=e.arena.words[u64(id)*K+c];
 for(u32 c=0;c<K;++c)*out++=e.arena.positions[u64(id)*K+c];for(u32 t=0;t<T;++t)*out++=u32(e.arena.residual[u64(id)*T+t]);
 const auto projected=a::region(e,id),witness=a::witness_region(e,id);
 for(u32 which=0;which<2;++which){const auto r=which?witness:projected;
  for(u32 n=0;n<N;++n)*out++=r.lower[n];for(u32 n=0;n<N;++n)*out++=r.upper[n];for(u32 n=0;n<N;++n)*out++=r.missing[n];
  for(u32 w=0;w<W;++w)masks[(u64(i)*2+which)*W+w]=r.allowed[w];}
}
J inspect_witness_labels(a::Budget&memory,a::EngineView e,const af::Snapshot&result,d::RegionView root,
 const u32*starts,const u32*counts,const u64*keys,u32 key_count,const u32*expected,u64 signatures,u64 maximum){
 const auto begin=Clock::now();const u32 slots=u32(result.status.states),F=e.source.features,N=e.domain.numeric_features,K=e.source.classes,T=e.source.trees,W=e.domain.mask_words;
 a::Buffer<WitnessResult>summaries(memory,slots);summaries.zero();a::Buffer<float>rows(memory,u64(slots)*2*F);
 const u64 shared=u64(N+e.domain.groups)*sizeof(u64)+u64(2*N+e.domain.groups)*sizeof(u32);
 inspect_native_witnesses<<<slots,32,shared>>>(e,root,starts,counts,keys,key_count,expected,signatures,maximum,summaries.data,rows.data);gpu_sync();
 const auto summaries_cpu=summaries.download(slots);const auto rows_cpu=rows.download(u64(slots)*2*F);
 std::vector<u32>witness_lower(u64(slots)*N),witness_upper(u64(slots)*N),witness_missing(u64(slots)*N);std::vector<u64>witness_allowed(u64(slots)*W);
 if(N){cu(cudaMemcpy(witness_lower.data(),e.arena.witness_lower,witness_lower.size()*sizeof(u32),cudaMemcpyDeviceToHost));cu(cudaMemcpy(witness_upper.data(),e.arena.witness_upper,witness_upper.size()*sizeof(u32),cudaMemcpyDeviceToHost));cu(cudaMemcpy(witness_missing.data(),e.arena.witness_missing,witness_missing.size()*sizeof(u32),cudaMemcpyDeviceToHost));}
 if(W)cu(cudaMemcpy(witness_allowed.data(),e.arena.witness_allowed,witness_allowed.size()*sizeof(u64),cudaMemcpyDeviceToHost));
 const auto witness_geometry=[&](u32 slot){return J{{"lower_fp32_order_keys",std::vector<u32>(witness_lower.begin()+u64(slot)*N,witness_lower.begin()+u64(slot+1)*N)},
  {"upper_fp32_order_keys",std::vector<u32>(witness_upper.begin()+u64(slot)*N,witness_upper.begin()+u64(slot+1)*N)},{"allow_nan",std::vector<u32>(witness_missing.begin()+u64(slot)*N,witness_missing.begin()+u64(slot+1)*N)},
  {"one_hot_masks",std::vector<u64>(witness_allowed.begin()+u64(slot)*W,witness_allowed.begin()+u64(slot+1)*W)}};};
 const char*reasons[]={"none","missing_not_enumerated","invalid_region","outside_verified_root","not_source_atom_aligned","invalid_grid","invalid_native_label_or_index","lookup_limit","witness_membership_failed"};
 u64 expanded=0,mixed=0,constant=0,unknown=0,lookups=0,local=0;J counts_json=J::object(),mixed_samples=J::array(),unknown_samples=J::array(),constant_samples=J::array();std::vector<u32>constant_ids;
 for(u32 slot=0;slot<slots;++slot){const auto&r=summaries_cpu[slot];if(!r.expanded)continue;++expanded;
  need(local<=UINT64_MAX-r.local_signatures&&lookups<=UINT64_MAX-r.lookups,"witness diagnostic count overflow");local+=r.local_signatures;lookups+=r.lookups;
  if(r.classification==1){++mixed;mixed_samples.push_back({{"state_slot",slot},{"local_signatures",r.local_signatures},{"lookups",r.lookups},{"native_classes",{r.first_class,r.second_class}},{"root_signature_indices",{r.first_index,r.second_index}},{"witness_region",witness_geometry(slot)},
    {"rows",{std::vector<float>(rows_cpu.begin()+u64(slot)*2*F,rows_cpu.begin()+(u64(slot)*2+1)*F),std::vector<float>(rows_cpu.begin()+(u64(slot)*2+1)*F,rows_cpu.begin()+(u64(slot)*2+2)*F)}},{"membership_checked",true}});}
  else if(r.classification==2){++constant;constant_ids.push_back(slot);}
  else{++unknown;const auto name=r.reason<std::size(reasons)?reasons[r.reason]:"invalid_reason";counts_json[name]=counts_json.value(name,u64(0))+1;unknown_samples.push_back({{"state_slot",slot},{"reason",name},{"local_signatures",r.local_signatures},{"lookups",r.lookups}});}}
 if(!constant_ids.empty()){
  const u64 stride=2ull*K+T+6ull*N;a::Buffer<u32>ids(memory,constant_ids.size()),records(memory,constant_ids.size()*stride);a::Buffer<u64>masks(memory,constant_ids.size()*2*W);ids.upload(constant_ids);
  capture_constant_contexts<<<(constant_ids.size()+127)/128,128>>>(e,ids.data,u32(constant_ids.size()),records.data,masks.data);gpu_sync();
  const auto data=records.download(records.size);const auto allowed=masks.download(masks.size);
  for(u32 i=0;i<constant_ids.size();++i){const u32 slot=constant_ids[i];const auto&r=summaries_cpu[slot];u64 at=u64(i)*stride;const auto take=[&](u32 n){std::vector<u32>v(data.begin()+at,data.begin()+at+n);at+=n;return v;};
   J item={{"state_slot",slot},{"native_class",r.first_class},{"local_signatures",r.local_signatures},{"lookups",r.lookups},{"root_signature_index",r.first_index},{"prefix_fp32_words",take(K)},{"source_positions",take(K)}};
   const auto raw=take(T);std::vector<std::int32_t>residual;for(auto x:raw)residual.push_back(std::bit_cast<std::int32_t>(x));item["residual_roots"]=residual;
   for(u32 which=0;which<2;++which){J region={{"lower_fp32_order_keys",take(N)},{"upper_fp32_order_keys",take(N)},{"allow_nan",take(N)},{"one_hot_masks",std::vector<u64>(allowed.begin()+(u64(i)*2+which)*W,allowed.begin()+(u64(i)*2+which+1)*W)}};item[which?"witness_region":"projected_residual_region"]=std::move(region);}
   constant_samples.push_back(std::move(item));}
 }
 return {{"enabled",true},{"outside_conversion_timing",true},{"elapsed_ms",milliseconds(begin)},{"scope","native classes on each retained expanded state's first stored witness context, not the union of shared incoming contexts or projected residual region"},
  {"source","existing exhaustive native root-signature expected labels"},{"per_state_lookup_limit",maximum},{"retained_expanded_states",expanded},{"mixed_contexts",mixed},{"exhaustive_constant_contexts",constant},{"unknown_contexts",unknown},
  {"actual_lookups",lookups},{"local_signature_total",local},{"local_signature_total_scope","sum for mapped contexts; may count the same root signature in multiple contexts"},{"state_evictions",result.status.state_evictions},
  {"unknown_reasons",std::move(counts_json)},{"mixed_witnesses",std::move(mixed_samples)},{"constant_contexts",std::move(constant_samples)},{"unknown_context_details",std::move(unknown_samples)}};
}


// Small synthetic mapping tests exercise the diagnostic, not converter proofs.
__global__ void witness_mapping_checks_kernel(d::DomainView domain,d::RegionView root,d::RegionView witness,u32*out){
 __shared__ u64 keys[5],strides[3];__shared__ u32 starts[2],counts[2],lows[2],radices[3],expected[60],checks,failures;
 __shared__ float rows[136];__shared__ WitnessResult result;
 a::SourceView source{};source.features=68;source.classes=2;
 if(threadIdx.x==0){
  checks=failures=0;auto check=[&](bool ok){++checks;if(!ok)++failures;};
  keys[0]=d::sortable_word(__float_as_uint(-1.f));keys[1]=d::sortable_word(__float_as_uint(0.f));keys[2]=d::sortable_word(__float_as_uint(2.f));
  keys[3]=(u64(1)<<32)|d::sortable_word(__float_as_uint(5.f));keys[4]=(u64(1)<<32)|d::sortable_word(__float_as_uint(7.f));
  check(d::initial_domain(domain,root)&&d::initial_domain(domain,witness));starts[0]=starts[1]=0;counts[0]=4;counts[1]=3;
  root.allowed[0]=(u64(1)<<0)|(u64(1)<<2)|(u64(1)<<63);root.allowed[1]=3;
  witness.lower[0]=u32(keys[0]);witness.upper[0]=d::predecessor_key(u32(keys[2]));
  witness.lower[1]=u32(keys[3]);witness.upper[1]=d::predecessor_key(u32(keys[4]));witness.allowed[0]=u64(1)<<2;witness.allowed[1]=1;
  u64 total=0;check(witness_grid(source,domain,keys,5,root,starts,counts,root,60,strides,lows,radices,total)==0&&total==60);
  for(u64 i=0;i<60;++i){u64 global=UINT64_MAX;check(witness_root_index(source,domain,root,starts,root,strides,lows,radices,i,60,global)&&global==i);}
  check(witness_grid(source,domain,keys,5,root,starts,counts,witness,60,strides,lows,radices,total)==0&&total==4);
  const u64 wanted[]={17,18,41,42};for(u32 i=0;i<4;++i){u64 global=0;check(witness_root_index(source,domain,root,starts,witness,strides,lows,radices,i,60,global)&&global==wanted[i]);
   signature_row(source,domain,keys,5,root,starts,counts,global,rows);check(witness_row_member(source,domain,witness,rows));check(rows[i<2?4:66]==1.f);}
  check(d::sortable_word(__float_as_uint(-0.f))==d::sortable_word(__float_as_uint(0.f)));
  check(d::predecessor_key(d::zero_key)!=d::negative_zero_hole);
  witness.missing[0]=1;check(witness_grid(source,domain,keys,5,root,starts,counts,witness,60,strides,lows,radices,total)==u32(WitnessReason::missing));witness.missing[0]=0;
  ++witness.lower[0];check(witness_grid(source,domain,keys,5,root,starts,counts,witness,60,strides,lows,radices,total)==u32(WitnessReason::unaligned));--witness.lower[0];
  witness.allowed[0]|=u64(1)<<1;check(witness_grid(source,domain,keys,5,root,starts,counts,witness,60,strides,lows,radices,total)==u32(WitnessReason::outside_root));witness.allowed[0]&=~(u64(1)<<1);
  witness.allowed[1]|=u64(1)<<2;check(witness_grid(source,domain,keys,5,root,starts,counts,witness,60,strides,lows,radices,total)==u32(WitnessReason::invalid_region));witness.allowed[1]&=~(u64(1)<<2);
  root.lower[0]=u32(keys[0]);starts[0]=1;counts[0]=3;witness.lower[0]=d::finite_min_key;
  check(witness_grid(source,domain,keys,5,root,starts,counts,witness,45,strides,lows,radices,total)==u32(WitnessReason::outside_root));
  root.lower[0]=d::finite_min_key;starts[0]=0;counts[0]=4;witness.lower[0]=u32(keys[0]);
  check(witness_grid(source,domain,keys,5,root,starts,counts,witness,61,strides,lows,radices,total)==u32(WitnessReason::invalid_grid));
  counts[0]=0;check(witness_grid(source,domain,keys,5,root,starts,counts,witness,60,strides,lows,radices,total)==u32(WitnessReason::invalid_grid));counts[0]=4;
  witness.lower[0]=d::negative_zero_hole;check(witness_grid(source,domain,keys,5,root,starts,counts,witness,60,strides,lows,radices,total)==u32(WitnessReason::invalid_region));witness.lower[0]=u32(keys[0]);
 }
 __syncthreads();
 for(u32 trial=0;trial<5;++trial){if(threadIdx.x==0){for(u32 i=0;i<60;++i)expected[i]=0;
   if(trial==0||trial==2)expected[18]=1;if(trial==3)expected[17]=2;if(trial==4)expected[41]=2;
   result=empty_witness_result();result.reason=witness_grid(source,domain,keys,5,root,starts,counts,witness,60,strides,lows,radices,result.local_signatures);}
  __syncthreads();scan_witness_grid(source,domain,keys,5,root,starts,counts,witness,strides,lows,radices,expected,60,trial==2?1:4,result,rows);
  if(threadIdx.x==0){++checks;bool good=false;
   if(trial==0)good=result.classification==1&&result.reason==0&&result.lookups==4&&result.first_index==17&&result.second_index==18&&result.first_class==0&&result.second_class==1;
   if(trial==1)good=result.classification==2&&result.reason==0&&result.lookups==4&&result.first_class==0;
   if(trial==2)good=result.classification==0&&result.reason==u32(WitnessReason::lookup_limit)&&result.lookups==1;
   if(trial==3)good=result.classification==0&&result.reason==u32(WitnessReason::invalid_label)&&result.lookups==1;
   if(trial==4)good=result.classification==0&&result.reason==u32(WitnessReason::invalid_label)&&result.lookups==4;
   if(!good)++failures;}
  __syncthreads();}
 if(threadIdx.x==0){out[0]=checks;out[1]=failures;}
}
J check_witness_mapping(){
 cu(cudaSetDevice(0));a::Budget memory{64ull<<20};std::vector<std::vector<u32>>groups(1);for(u32 f=2;f<68;++f)groups[0].push_back(f);
 DomainStorage domain(memory,d::prepare_domain_metadata(68,groups,false));RegionStorage root(memory,domain.view),witness(memory,domain.view);a::Buffer<u32>results(memory,2);results.zero();
 witness_mapping_checks_kernel<<<1,32>>>(domain.view,root.view(),witness.view(),results.data);gpu_sync();const auto r=results.download(2);
 need(r[1]==0,"witness mapping checks failed: "+std::to_string(r[1]));return {{"checks",r[0]},{"failures",r[1]},{"scope","synthetic diagnostic-index and membership checks only"}};
}


// Optional metadata-only opportunity analysis. No source arithmetic, model
// prediction, proof acceptance, or live state mutation occurs in this grouping.
struct PrefixLayout {
 u32 N,K,T,W,score_channels;
 bool gap_evidence=false;
 u64 roots()const{return 2;}u64 positions()const{return 2ull+T;}
 u64 prefixes()const{return 2ull+T+K;}u64 projected()const{return 2ull+T+2ull*K;}
 u64 guard_words()const{return 3ull*N+2ull*W;}u64 witness()const{return projected()+guard_words();}
 u64 evidence()const{return witness()+guard_words();}
 u64 stride()const{return evidence()+u64(gap_evidence);}
};
__global__ void gather_prefix_metadata(a::EngineView e,u32*records,u64*stats,const a::gap_evidence::Evidence* evidence=nullptr){
 const u32 slot=blockIdx.x;if(slot>=e.status->states)return;
 const auto state=e.arena.states[slot];if(state.phase!=3||state.left==a::none||state.right==a::none)return;
 if(state.node>=e.status->nodes){if(!threadIdx.x)atomicAdd(reinterpret_cast<unsigned long long*>(stats+1),1ull);return;}
 __shared__ u64 row;
 if(!threadIdx.x)row=atomicAdd(reinterpret_cast<unsigned long long*>(stats),1ull);__syncthreads();
 const u32 K=e.source.classes,T=e.source.trees,N=e.domain.numeric_features,W=e.domain.mask_words;
 const u64 prefix=2ull+T+K,projected=prefix+K,guard=3ull*N+2ull*W,proof_at=projected+2*guard,stride=proof_at+u64(evidence!=nullptr);
 const auto R=a::region(e,slot),witness=a::witness_region(e,slot);
 for(u64 j=threadIdx.x;j<stride;j+=blockDim.x){u32 value=0;
  if(j==proof_at){value=u32(evidence[slot]);
   if(value!=0&&value!=u32(a::gap_evidence::Evidence::CompletedQualified))atomicAdd(reinterpret_cast<unsigned long long*>(stats+1),1ull);}
  else if(j==0)value=slot;else if(j==1)value=state.node;
  else if(j<2ull+T)value=u32(e.arena.residual[u64(slot)*T+j-2]);
  else if(j<prefix)value=e.arena.positions[u64(slot)*K+j-(2ull+T)];
  else if(j<projected)value=e.arena.words[u64(slot)*K+j-prefix];
  else{const u64 offset=j-projected;const auto r=offset<guard?R:witness;const u64 at=offset%guard;
   if(at<N)value=r.lower[at];else if(at<2ull*N)value=r.upper[at-N];else if(at<3ull*N)value=r.missing[at-2ull*N];
   else{const u64 part=at-3ull*N;value=u32(r.allowed[part/2]>>(32*(part%2)));}}
  records[row*stride+j]=value;}
}
J prefix_guard_json(const std::vector<u32>&data,u64 at,const PrefixLayout&d){
 const auto take=[&](u32 n){std::vector<u32>v(data.begin()+at,data.begin()+at+n);at+=n;return v;};
 J out={{"lower_fp32_order_keys",take(d.N)},{"upper_fp32_order_keys",take(d.N)},{"allow_nan",take(d.N)}};
 std::vector<u64>masks;for(u32 w=0;w<d.W;++w){masks.push_back(u64(data[at])|(u64(data[at+1])<<32));at+=2;}out["one_hot_masks"]=masks;return out;
}
J summarize_prefix_metadata(const std::vector<u32>&data,const PrefixLayout&d,u64 pair_limit,bool force_fingerprint_collision=false){
 need(d.K&&d.score_channels&&d.score_channels<=d.K&&data.size()%d.stride()==0,"invalid prefix metadata layout");
 const u64 count=data.size()/d.stride();need(count<=UINT32_MAX,"prefix metadata count exceeds index representation");
 const auto at=[&](u32 row,u64 col)->u32{return data[u64(row)*d.stride()+col];};
 const auto qualified=[&](u32 row){return d.gap_evidence&&at(row,d.evidence())==u32(a::gap_evidence::Evidence::CompletedQualified);};
 u64 qualified_contexts=0;
 for(u32 row=0;row<count;++row)if(d.gap_evidence){const auto tag=at(row,d.evidence());
  need(tag==0||tag==u32(a::gap_evidence::Evidence::CompletedQualified),"completed context has invalid gap evidence state");qualified_contexts+=qualified(row);}
 const auto compare_range=[&](u32 left,u32 right,u64 begin,u64 words){for(u64 j=0;j<words;++j){const auto a=at(left,begin+j),b=at(right,begin+j);if(a<b)return -1;if(a>b)return 1;}return 0;};
 const auto compare_key=[&](u32 left,u32 right){int c=compare_range(left,right,1,1);if(c)return c;
  c=compare_range(left,right,d.roots(),u64(d.T)+d.K);if(c)return c;
  c=compare_range(left,right,d.projected(),d.guard_words());if(c)return c;
  // Structural guard words are immutable key data, never interval candidates.
  return compare_range(left,right,d.prefixes()+d.score_channels,d.K-d.score_channels);};
 const auto finite=[&](u32 row){for(u32 c=0;c<d.K;++c)if((at(row,d.prefixes()+c)&0x7fffffffu)>=0x7f800000u)return false;return true;};
 const auto compare_prefix=[&](u32 left,u32 right,bool canonical){for(u32 c=0;c<d.K;++c){u32 a=at(left,d.prefixes()+c),b=at(right,d.prefixes()+c);
   if(canonical){if(!(a&0x7fffffffu))a=0;if(!(b&0x7fffffffu))b=0;}if(a<b)return -1;if(a>b)return 1;}return 0;};
 std::vector<u64>hashes(count);std::vector<u32>order(count),nodes;std::iota(order.begin(),order.end(),0);nodes.reserve(count);
 for(u32 row=0;row<count;++row){u64 h=1469598103934665603ull;const auto absorb=[&](u64 begin,u64 words){for(u64 j=0;j<words;++j)h=(h^at(row,begin+j))*1099511628211ull;};
  absorb(1,1);absorb(d.roots(),u64(d.T)+d.K);absorb(d.projected(),d.guard_words());absorb(d.prefixes()+d.score_channels,d.K-d.score_channels);
  hashes[row]=force_fingerprint_collision?0:h;nodes.push_back(at(row,1));}
 std::sort(order.begin(),order.end(),[&](u32 a,u32 b){if(hashes[a]!=hashes[b])return hashes[a]<hashes[b];const int c=compare_key(a,b);return c?c<0:at(a,0)<at(b,0);});
 std::sort(nodes.begin(),nodes.end());nodes.erase(std::unique(nodes.begin(),nodes.end()),nodes.end());
 struct Candidate {u64 group_size,distinct;u32 first,second,group_order;};std::vector<Candidate>candidates;
 u64 groups=0,distinct_bitwise=0,distinct_numeric=0,finite_contexts=0,bitwise_groups=0,numeric_groups=0,bitwise_contexts=0,numeric_contexts=0,zero_only_groups=0,excess_numeric=0,collision_boundaries=0;
 for(u64 begin=0;begin<count;){u64 end=begin+1;while(end<count&&hashes[order[begin]]==hashes[order[end]]&&compare_key(order[begin],order[end])==0)++end;
  if(begin&&hashes[order[begin-1]]==hashes[order[begin]])++collision_boundaries;
  const u64 group_size=end-begin;std::vector<u32>prefixes(order.begin()+begin,order.begin()+end),numbers;
  std::sort(prefixes.begin(),prefixes.end(),[&](u32 a,u32 b){const auto c=compare_prefix(a,b,false);return c?c<0:at(a,0)<at(b,0);});
  u64 words=0;for(u64 i=0;i<prefixes.size();++i){if(!i||compare_prefix(prefixes[i-1],prefixes[i],false))++words;if(finite(prefixes[i]))numbers.push_back(prefixes[i]);}
  std::sort(numbers.begin(),numbers.end(),[&](u32 a,u32 b){const auto c=compare_prefix(a,b,true);if(c)return c<0;const auto bits=compare_prefix(a,b,false);return bits?bits<0:at(a,0)<at(b,0);});
  u64 values=0;u32 second=a::none;for(u64 i=0;i<numbers.size();++i)if(!i||compare_prefix(numbers[i-1],numbers[i],true)){++values;if(values==2)second=numbers[i];}
  distinct_bitwise+=words;distinct_numeric+=values;finite_contexts+=numbers.size();
  if(words>1){++bitwise_groups;bitwise_contexts+=group_size;}
  if(values>1){++numeric_groups;numeric_contexts+=numbers.size();excess_numeric+=values-1;candidates.push_back({group_size,values,numbers.front(),second,u32(groups)});}
  if(words>1&&values==1&&numbers.size()==group_size)++zero_only_groups;
  ++groups;begin=end;
 }
 std::sort(candidates.begin(),candidates.end(),[](const auto&a,const auto&b){return a.distinct!=b.distinct?a.distinct>b.distinct:a.group_order<b.group_order;});
 const u64 reported=std::min<u64>(pair_limit,candidates.size());J pairs=J::array();
 for(u64 i=0;i<reported;++i){const auto&c=candidates[i];const u32 first=c.first,second=c.second;std::vector<std::int32_t>roots;for(u32 t=0;t<d.T;++t)roots.push_back(std::bit_cast<std::int32_t>(at(first,d.roots()+t)));
  std::vector<u32>positions;for(u32 k=0;k<d.K;++k)positions.push_back(at(first,d.positions()+k));J contexts=J::array();
  for(auto row:{first,second}){std::vector<u32>prefix;for(u32 k=0;k<d.K;++k)prefix.push_back(at(row,d.prefixes()+k));contexts.push_back({{"state_slot",at(row,0)},{"prefix_fp32_words",prefix},{"witness_region",prefix_guard_json(data,u64(row)*d.stride()+d.witness(),d)}});}
  pairs.push_back({{"final_node",at(first,1)},{"group_contexts",c.group_size},{"distinct_finite_numeric_prefixes",c.distinct},{"residual_roots",roots},{"source_positions",positions},
   {"projected_residual_region",prefix_guard_json(data,u64(first)*d.stride()+d.projected(),d)},{"same_witness_guard",compare_range(first,second,d.witness(),d.guard_words())==0},{"contexts",std::move(contexts)}});
 }
 // A separate geometry-only census for the one-coordinate interpolation
 // premise. Each channel gets a sort, never a quadratic scan of context pairs.
 // Structural suffix words remain opaque exact key data; only genuine score
 // coordinates need to be finite here. The older all-word summary is unchanged.
 const auto finite_scores=[&](u32 row){for(u32 c=0;c<d.score_channels;++c)
   if((at(row,d.prefixes()+c)&0x7fffffffu)>=0x7f800000u)return false;return true;};
 const auto numeric_key=[](u32 word){if(!(word&0x7fffffffu))word=0;
   return word&0x80000000u ? ~word : word^0x80000000u;};
 struct IntervalCandidate {u64 group_size,finite_size,distinct,group_order;u32 channel,lower,upper;u64 qualified_size=0,enclosed_size=0;};
 std::vector<IntervalCandidate>intervals,qualified_intervals;std::vector<std::uint8_t>eligible(count,0),qualified_enclosed(count,0);
 u64 qualified_memberships=0,enclosed_memberships=0;
 std::vector<u64>channel_hashes(count);std::vector<u32>channel_order(count);
 u64 channel_groups=0,eligible_memberships=0,finite_score_contexts=0,zero_width_groups=0,channel_collisions=0;
 for(u32 row=0;row<count;++row)finite_score_contexts+=finite_scores(row);
 J per_channel=J::array();
 for(u32 channel=0;channel<d.score_channels;++channel){
  const auto compare_other=[&](u32 left,u32 right){const int base=compare_key(left,right);if(base)return base;
   for(u32 c=0;c<d.score_channels;++c)if(c!=channel){const u32 a=at(left,d.prefixes()+c),b=at(right,d.prefixes()+c);if(a<b)return -1;if(a>b)return 1;}return 0;};
  std::iota(channel_order.begin(),channel_order.end(),0);
  for(u32 row=0;row<count;++row){u64 h=hashes[row];for(u32 c=0;c<d.score_channels;++c)if(c!=channel)h=(h^at(row,d.prefixes()+c))*1099511628211ull;
   channel_hashes[row]=force_fingerprint_collision?0:h;}
  std::sort(channel_order.begin(),channel_order.end(),[&](u32 a,u32 b){if(channel_hashes[a]!=channel_hashes[b])return channel_hashes[a]<channel_hashes[b];
   const int c=compare_other(a,b);return c?c<0:at(a,0)<at(b,0);});
  u64 local_groups=0,local_eligible=0,local_members=0,local_zero=0,local_collisions=0;
  u64 local_qualified_groups=0,local_qualified_members=0,local_enclosed=0;
  for(u64 begin=0;begin<count;){u64 end=begin+1;while(end<count&&channel_hashes[channel_order[begin]]==channel_hashes[channel_order[end]]&&compare_other(channel_order[begin],channel_order[end])==0)++end;
   if(begin&&channel_hashes[channel_order[begin-1]]==channel_hashes[channel_order[begin]])++local_collisions;
   std::vector<u32>numbers;for(u64 i=begin;i<end;++i)if(finite_scores(channel_order[i]))numbers.push_back(channel_order[i]);
   std::sort(numbers.begin(),numbers.end(),[&](u32 a,u32 b){const u32 av=numeric_key(at(a,d.prefixes()+channel)),bv=numeric_key(at(b,d.prefixes()+channel));
    if(av!=bv)return av<bv;return at(a,0)<at(b,0);});
   u64 distinct=0;bool positive_zero=false,negative_zero=false;
   for(u64 i=0;i<numbers.size();++i){const auto word=at(numbers[i],d.prefixes()+channel);
    if(!i||numeric_key(word)!=numeric_key(at(numbers[i-1],d.prefixes()+channel)))++distinct;
    positive_zero|=word==0;negative_zero|=word==0x80000000u;}
   if(distinct>1){++local_eligible;local_members+=numbers.size();for(auto row:numbers)eligible[row]=1;
    intervals.push_back({end-begin,numbers.size(),distinct,channel_groups+local_groups,channel,numbers.front(),numbers.back()});}
   if(distinct==1&&positive_zero&&negative_zero)++local_zero;
   // Restrict endpoint selection to explicit full projected-guard gap/window
   // evidence. Unqualified outer points must never widen this interval.
   if(d.gap_evidence){std::vector<u32>certified;for(auto row:numbers)if(qualified(row))certified.push_back(row);
    u64 certified_values=0;for(u64 i=0;i<certified.size();++i)
     if(!i||numeric_key(at(certified[i],d.prefixes()+channel))!=numeric_key(at(certified[i-1],d.prefixes()+channel)))++certified_values;
    if(certified_values>1){const auto low=numeric_key(at(certified.front(),d.prefixes()+channel)),high=numeric_key(at(certified.back(),d.prefixes()+channel));
     u64 enclosed=0;for(auto row:numbers){const auto key=numeric_key(at(row,d.prefixes()+channel));if(key>=low&&key<=high){++enclosed;qualified_enclosed[row]=1;}}
     ++local_qualified_groups;local_qualified_members+=certified.size();local_enclosed+=enclosed;
     qualified_intervals.push_back({end-begin,numbers.size(),certified_values,channel_groups+local_groups,channel,certified.front(),certified.back(),certified.size(),enclosed});
    }
   }
   ++local_groups;begin=end;
  }
  channel_groups+=local_groups;eligible_memberships+=local_members;zero_width_groups+=local_zero;channel_collisions+=local_collisions;
  qualified_memberships+=local_qualified_members;enclosed_memberships+=local_enclosed;
  per_channel.push_back({{"channel",channel},{"channel_key_groups",local_groups},{"eligible_geometry_groups",local_eligible},
   {"finite_context_memberships",local_members},{"finite_signed_zero_only_groups",local_zero},{"fingerprint_collision_group_boundaries",local_collisions},
   {"qualified_endpoint_groups",local_qualified_groups},{"qualified_endpoint_context_memberships",local_qualified_members},{"contexts_enclosed_by_qualified_endpoints",local_enclosed}});
 }
 const auto rank_interval=[](const auto&a,const auto&b){if(a.distinct!=b.distinct)return a.distinct>b.distinct;
  if(a.channel!=b.channel)return a.channel<b.channel;return a.group_order<b.group_order;};
 std::sort(intervals.begin(),intervals.end(),rank_interval);std::sort(qualified_intervals.begin(),qualified_intervals.end(),rank_interval);
 const auto interval_json=[&](const IntervalCandidate&c){std::vector<std::int32_t>roots;std::vector<u32>positions;
  for(u32 t=0;t<d.T;++t)roots.push_back(std::bit_cast<std::int32_t>(at(c.lower,d.roots()+t)));
  for(u32 k=0;k<d.K;++k)positions.push_back(at(c.lower,d.positions()+k));J contexts=J::array();
  for(auto row:{c.lower,c.upper}){std::vector<u32>prefix;for(u32 k=0;k<d.K;++k)prefix.push_back(at(row,d.prefixes()+k));
   contexts.push_back({{"state_slot",at(row,0)},{"prefix_fp32_words",prefix},{"full_guard_gap_window_evidence",qualified(row)},{"witness_region",prefix_guard_json(data,u64(row)*d.stride()+d.witness(),d)}});}
  return J({{"changing_score_channel",c.channel},{"minimum_prefix_fp32_word",at(c.lower,d.prefixes()+c.channel)},
   {"maximum_prefix_fp32_word",at(c.upper,d.prefixes()+c.channel)},{"final_node",at(c.lower,1)},{"channel_key_group_contexts",c.group_size},
   {"finite_contexts",c.finite_size},{"distinct_finite_numeric_values",c.distinct},{"residual_roots",roots},{"source_positions",positions},
   {"projected_residual_region",prefix_guard_json(data,u64(c.lower)*d.stride()+d.projected(),d)},
   {"same_witness_guard",compare_range(c.lower,c.upper,d.witness(),d.guard_words())==0},{"contexts_lower_then_upper",std::move(contexts)}});
 };
 const u64 interval_reported=std::min<u64>(pair_limit,intervals.size()),qualified_reported=std::min<u64>(pair_limit,qualified_intervals.size());
 J interval_examples=J::array(),qualified_examples=J::array();
 for(u64 i=0;i<interval_reported;++i)interval_examples.push_back(interval_json(intervals[i]));
 for(u64 i=0;i<qualified_reported;++i){auto item=interval_json(qualified_intervals[i]);
  item["qualified_endpoint_contexts"]=qualified_intervals[i].qualified_size;item["enclosed_retained_contexts"]=qualified_intervals[i].enclosed_size;
  qualified_examples.push_back(std::move(item));}
 J qualification={{"enabled",d.gap_evidence},{"full_guard_gap_window_contexts",qualified_contexts},{"unknown_contexts",count-qualified_contexts},
  {"scope","saved prefix plus ordered residual continuation over the entire projected guard; not the original forest over a widened feature region"},
  {"unknown_is_failure",false},{"qualified_endpoint_groups",qualified_intervals.size()},{"qualified_endpoint_context_memberships",qualified_memberships},
  {"enclosed_retained_context_memberships",enclosed_memberships},{"distinct_enclosed_retained_contexts",std::count(qualified_enclosed.begin(),qualified_enclosed.end(),std::uint8_t(1))},
  {"representative_pair_limit",pair_limit},{"representative_pairs_available",qualified_intervals.size()},{"representative_pairs_reported",qualified_reported},
  {"representative_pairs_omitted",qualified_intervals.size()-qualified_reported},{"counts_truncated",false},{"examples_truncated",qualified_reported<qualified_intervals.size()},
  {"cache_admission",false},{"avoided_construction_measured",false},{"certifies_prefix_box",false},
  {"interpretation","same-program one-coordinate endpoint evidence; counts include endpoints and do not establish donors were available before construction"},
  {"representative_pairs",std::move(qualified_examples)}};
 J one_coordinate={{"metadata_only",true},{"provenance_eligibility",d.gap_evidence?"reported_separately":"unknown"},{"cache_admission",false},{"certifies_prefix_interval",false},
  {"qualified_endpoint_geometry",std::move(qualification)},
  {"premise_missing",d.gap_evidence?"unknown entries have no retained full-guard certificate; runtime cache admission and GPU integration validation are separate":"optional full-program positive-gap and score-window provenance collection is disabled"},
  {"key","existing exact base key plus every genuine score prefix word except one changing channel; structural suffix remains exact"},
  {"finite_scope","all genuine score coordinates finite; structural suffix is opaque exact key data"},
  {"channel_key_groups_examined",channel_groups},{"eligible_geometry_groups",intervals.size()},{"eligible_finite_context_memberships",eligible_memberships},
  {"distinct_eligible_contexts",std::count(eligible.begin(),eligible.end(),std::uint8_t(1))},{"finite_genuine_score_contexts",finite_score_contexts},
  {"nonfinite_genuine_score_contexts",count-finite_score_contexts},{"finite_signed_zero_only_channel_key_groups",zero_width_groups},
  {"membership_scope","memberships sum over eligible changing-channel groups; distinct contexts counts each retained context once"},
  {"fingerprint_collision_group_boundaries",channel_collisions},{"fingerprints_are_equality",false},{"per_channel",std::move(per_channel)},
  {"representative_pair_definition","one numeric minimum/maximum finite endpoint pair per eligible geometry group; rank by distinct value count, channel, then fingerprint/full-key order"},
  {"representative_pair_limit",pair_limit},{"representative_pairs_available",intervals.size()},{"representative_pairs_reported",interval_reported},
  {"representative_pairs_omitted",intervals.size()-interval_reported},{"counts_truncated",false},{"examples_truncated",interval_reported<intervals.size()},
  {"pair_matrix_materialized",false},{"representative_pairs",std::move(interval_examples)}};
 return {{"one_coordinate_interpolation_geometry",std::move(one_coordinate)},{"enabled",true},{"metadata_only",true},{"model_evaluations_on_cpu",0},{"cache_admission",false},{"certifies_prefix_box",false},
  {"key","same final node, exact ordered residual roots, source positions, projected residual guard and structural-guard prefix words; genuine score prefix words excluded"},
  {"scope","one completed arena under the report's unchanged source, domain, output interpretation and native gate; counts cover retained expanded contexts only"},
  {"witness_scope","first stored original feature context, reported separately from the projected guard; no certified union of incoming contexts"},
  {"retained_expanded_contexts",count},{"distinct_final_nodes",nodes.size()},{"groups",groups},{"groups_with_bitwise_prefix_variation",bitwise_groups},{"contexts_in_bitwise_varied_groups",bitwise_contexts},
  {"groups_with_finite_numeric_prefix_variation",numeric_groups},{"finite_contexts_in_numeric_varied_groups",numeric_contexts},{"signed_zero_only_varied_groups",zero_only_groups},
  {"distinct_bitwise_prefix_vectors_across_groups",distinct_bitwise},{"exact_bitwise_repeat_contexts",count-distinct_bitwise},{"finite_prefix_contexts",finite_contexts},{"nonfinite_prefix_contexts",count-finite_contexts},
  {"distinct_finite_numeric_prefix_vectors_across_groups",distinct_numeric},{"finite_numeric_repeat_contexts",finite_contexts-distinct_numeric},{"excess_distinct_finite_numeric_prefix_vectors",excess_numeric},
  {"score_prefix_channels",d.score_channels},{"structural_guard_prefix_channels_in_key",d.K-d.score_channels},{"fingerprint_collision_group_boundaries",collision_boundaries},{"fingerprints_are_equality",false},
  {"representative_pair_definition","one first-distinct finite canonical-word-lexicographic prefix pair per eligible group; groups ranked by distinct finite prefix count, then fingerprint/full-key order"},
  {"representative_pair_limit",pair_limit},{"representative_pairs_available",candidates.size()},{"representative_pairs_reported",reported},{"representative_pairs_omitted",candidates.size()-reported},{"counts_truncated",false},{"representative_pairs",std::move(pairs)}};
}
// Optional syntactic output-label support, computed once for the immutable
// graph on the GPU. Source interning guarantees lower node IDs for successors.
// A single warp follows that topological order; words within a large class mask
// are parallel. This metadata census is outside all comparative timings.
__global__ void program_label_support(const a::Node*nodes,u32 count,u32 features,u32 classes,u64*masks,u32*sizes,u32*bad){
 if(blockIdx.x||blockDim.x!=32)return;const u32 lane=threadIdx.x,words=u32((u64(classes)+63)/64);
 for(u32 id=0;id<count;++id){const auto n=nodes[id];
  const auto feature=n.feature<=-2?-std::int64_t(n.feature)-2:std::int64_t(n.feature);
  const bool invalid=n.feature==-1?n.payload>=classes:(n.left>=id||n.right>=id||feature<0||u64(feature)>=features);
  if(invalid){if(!lane)*bad=id+1;return;}
  u32 local=0;
  for(u32 w=lane;w<words;w+=32){const u64 mask=n.feature==-1?(w==n.payload/64?u64(1)<<(n.payload%64):0):
    masks[u64(n.left)*words+w]|masks[u64(n.right)*words+w];
   masks[u64(id)*words+w]=mask;local+=__popcll(mask);}
  for(u32 delta=16;delta;delta/=2)local+=__shfl_down_sync(0xffffffffu,local,delta);
  if(!lane)sizes[id]=local;
  __syncwarp(); // All current masks are visible before a later node reads them.
 }
}
J summarize_program_support(const std::vector<u32>&data,const PrefixLayout&layout,const std::vector<u32>&sizes){
 need(layout.gap_evidence&&data.size()%layout.stride()==0,"label support requires context evidence");
 const u64 count=data.size()/layout.stride();std::vector<u64>qualified_histogram(u64(layout.score_channels)+1,0);
 std::vector<std::uint8_t>qualified_node(sizes.size(),0);u64 qualified=0,one_unused=0,two_unused=0;
 for(u64 row=0;row<count;++row){const auto offset=row*layout.stride();const u32 node=data[offset+1];
  need(node<sizes.size()&&sizes[node]>0&&sizes[node]<=layout.score_channels,"invalid program output support metadata");
  const auto tag=data[offset+layout.evidence()];need(tag==0||tag==u32(a::gap_evidence::Evidence::CompletedQualified),"invalid completed program support evidence");
  if(tag!=u32(a::gap_evidence::Evidence::CompletedQualified))continue;
  const u32 used=sizes[node];++qualified;++qualified_histogram[used];qualified_node[node]=1;
  one_unused+=used<layout.score_channels;two_unused+=layout.score_channels-used>=2;
 }
 u64 unique=0,unique_one=0,unique_two=0;
 for(u64 node=0;node<sizes.size();++node)if(qualified_node[node]){++unique;unique_one+=sizes[node]<layout.score_channels;unique_two+=layout.score_channels-sizes[node]>=2;}
 return {{"enabled",true},{"qualified_expanded_contexts",qualified},{"unknown_expanded_contexts",count-qualified},
  {"qualified_contexts_by_used_class_count",qualified_histogram},{"qualified_contexts_with_unused_classes",one_unused},
  {"qualified_contexts_with_at_least_two_unused_classes",two_unused},{"distinct_qualified_program_nodes",unique},
  {"distinct_qualified_programs_with_unused_classes",unique_one},{"distinct_qualified_programs_with_at_least_two_unused_classes",unique_two},
  {"syntactic_support_superset",true},{"scope","reachable output labels may include infeasible branches; evidence belongs to a context, not a node"},
  {"box_lower_bounds_certified",false},{"cache_admission",false},{"actual_target_hits_measured",false}};
}
J inspect_program_support(a::Budget&memory,a::EngineView e,const af::Snapshot&result,const PrefixLayout&layout,const std::vector<u32>&data){
 if(layout.score_channels!=e.source.classes||a::output_class_count(e)!=layout.score_channels)
  return {{"enabled",false},{"reason","public classes do not have the full genuine-score interpretation"}};
 const u32 nodes=u32(result.status.nodes),words=u32((u64(layout.score_channels)+63)/64);
 a::Buffer<u64>masks(memory,a::multiply(nodes,words));a::Buffer<u32>sizes(memory,nodes),bad(memory,1);bad.zero();
 if(nodes)program_label_support<<<1,32>>>(e.arena.nodes,nodes,e.source.features,layout.score_channels,masks.data,sizes.data,bad.data);
 gpu_sync();need(!bad.download(1)[0],"invalid output graph in label support census");
 const auto counts=sizes.download(nodes);auto summary=summarize_program_support(data,layout,counts);
 summary["gpu_mask_bytes"]=masks.size*sizeof(u64);summary["gpu_count_bytes"]=sizes.size*sizeof(u32)+sizeof(u32);
 summary["host_count_readback_bytes"]=counts.size()*sizeof(u32);summary["numeric_model_evaluations"]=0;
 return summary;
}
J check_program_support(){
 a::Budget memory{1ull<<20};
 const std::vector<a::Node>nodes={{-1,0,0,0},{-1,65,0,0},{0,0,0,1},{-1,129,0,0},{0,0,2,3},{0,0,2,2}};
 a::Buffer<a::Node>graph(memory,nodes.size());graph.upload(nodes);
 a::Buffer<u64>masks(memory,nodes.size()*3);a::Buffer<u32>sizes(memory,nodes.size()),bad(memory,1);bad.zero();
 program_label_support<<<1,32>>>(graph.data,u32(nodes.size()),1,130,masks.data,sizes.data,bad.data);gpu_sync();
 const auto counts=sizes.download(nodes.size());const auto actual=masks.download(nodes.size()*3);
 need(!bad.download(1)[0]&&counts==std::vector<u32>{1,1,2,1,3,2},"cross-word program support count differs");
 need(actual[4*3]==1&&actual[4*3+1]==2&&actual[4*3+2]==2,"cross-word program label masks differ");
 auto invalid=nodes;invalid[4].right=4;graph.upload(invalid);bad.zero();
 program_label_support<<<1,32>>>(graph.data,u32(nodes.size()),1,130,masks.data,sizes.data,bad.data);gpu_sync();
 need(bad.download(1)[0]==5,"program support accepted cycle/forward reference");
 invalid=nodes;invalid[1].payload=130;graph.upload(invalid);bad.zero();
 program_label_support<<<1,32>>>(graph.data,u32(nodes.size()),1,130,masks.data,sizes.data,bad.data);gpu_sync();
 need(bad.download(1)[0]==2,"program support accepted out-of-range output label");
 return {{"checks",4},{"failures",0},{"gpu_execution",true},{"model_evaluations",0},{"scope","GPU graph metadata union, duplicate label sharing and malformed graph rejection"}};
}
J inspect_prefix_opportunities(a::Budget&memory,a::EngineView e,const af::Snapshot&result,u64 pair_limit,const a::Buffer<a::gap_evidence::Evidence>*evidence=nullptr){
 const auto start=Clock::now();const u32 slots=u32(result.status.states);const PrefixLayout layout{e.domain.numeric_features,e.source.classes,e.source.trees,e.domain.mask_words,e.source.native_margin_classes?e.source.native_margin_classes:e.source.classes,evidence!=nullptr};
 need(!evidence||(evidence->size==e.arena.state_capacity&&evidence->data),"gap evidence owner does not match state capacity");
 a::Buffer<u32>packed(memory,u64(slots)*layout.stride());a::Buffer<u64>stats(memory,2);stats.zero();
 if(slots)gather_prefix_metadata<<<slots,32>>>(e,packed.data,stats.data,evidence?evidence->data:nullptr);gpu_sync();const auto counts=stats.download(2);need(!counts[1]&&counts[0]<=slots,"invalid prefix metadata state/node reference");
 const auto data=packed.download(counts[0]*layout.stride());auto out=summarize_prefix_metadata(data,layout,pair_limit);
 if(evidence)out["program_label_support"]=inspect_program_support(memory,e,result,layout,data);
 out["outside_conversion_timing"]=true;out["elapsed_ms"]=milliseconds(start);out["state_slot_highwater"]=slots;out["state_evictions"]=result.status.state_evictions;out["state_reuses"]=result.status.state_reuses;
 out["state_hits"]=result.status.state_hits;out["node_hits"]=result.status.node_hits;out["prefix_record_bytes"]=layout.stride()*sizeof(u32);out["packed_readback_bytes"]=data.size()*sizeof(u32);out["gpu_scratch_bytes"]=packed.size*sizeof(u32)+stats.size*sizeof(u64);
 out["gap_evidence_bytes"]=evidence?evidence->size*sizeof(a::gap_evidence::Evidence):0;out["gap_evidence_persisted"]=false;
 out["source_score_storage_channels"]=e.source.classes;out["public_output_classes"]=a::output_class_count(e);return out;
}
J check_prefix_grouping(){
 u64 checks=0;const auto check=[&](bool ok){++checks;need(ok,"CPU prefix grouping check failed at "+std::to_string(checks));};
 const PrefixLayout layout{1,2,2,1,2};std::vector<u32>data;
 const auto add=[&](u32 slot,u32 node,u32 first,u32 second,u32 upper=20,u32 root=10){const u64 start=data.size();data.resize(start+layout.stride());data[start]=slot;data[start+1]=node;
  data[start+layout.roots()]=root;data[start+layout.roots()+1]=11;data[start+layout.positions()]=1;data[start+layout.positions()+1]=1;
  data[start+layout.prefixes()]=first;data[start+layout.prefixes()+1]=second;
  for(auto offset:{layout.projected(),layout.witness()}){data[start+offset]=5;data[start+offset+1]=upper;data[start+offset+2]=0;data[start+offset+3]=3;data[start+offset+4]=0;}};
 const u32 one=0x3f800000u,two=0x40000000u;add(1,50,0,one);add(2,50,0x80000000u,one);add(3,50,one,one);add(4,50,one,one);
 add(5,51,0,one);add(6,50,0,one,21);add(7,50,0,one,20,12);add(8,52,0x7fc00001u,one);add(9,52,0,one);
 add(10,53,0,one);add(11,53,0x80000000u,one);add(12,54,one,one);add(13,54,two,one);
 const auto summary=summarize_prefix_metadata(data,layout,1,true);
 check(summary.at("retained_expanded_contexts")==13);check(summary.at("groups")==7);check(summary.at("distinct_final_nodes")==5);
 check(summary.at("groups_with_bitwise_prefix_variation")==4);check(summary.at("groups_with_finite_numeric_prefix_variation")==2);
 check(summary.at("signed_zero_only_varied_groups")==1);check(summary.at("nonfinite_prefix_contexts")==1);
 check(summary.at("exact_bitwise_repeat_contexts")==1);check(summary.at("representative_pairs_available")==2);
 check(summary.at("representative_pairs_reported")==1&&summary.at("representative_pairs_omitted")==1);check(summary.at("fingerprint_collision_group_boundaries")==6);
 const auto uncapped=summarize_prefix_metadata(data,layout,100);check(uncapped.at("groups")==summary.at("groups"));check(uncapped.at("representative_pairs_reported")==2);
 const auto none=summarize_prefix_metadata(data,layout,0);check(none.at("representative_pairs_reported")==0&&none.at("groups")==7);
 auto structural=layout;structural.score_channels=1;std::vector<u32>guarded;guarded.insert(guarded.end(),data.begin()+11*layout.stride(),data.begin()+13*layout.stride());
 guarded[structural.prefixes()+1]=0;guarded[structural.stride()+structural.prefixes()+1]=0x80000000u;
 const auto guards=summarize_prefix_metadata(guarded,structural,100,true);check(guards.at("groups")==2&&guards.at("representative_pairs_available")==0);
 const auto empty=summarize_prefix_metadata({},layout,1);check(empty.at("groups")==0&&empty.at("representative_pairs_reported")==0);
 const auto&one_geometry=summary.at("one_coordinate_interpolation_geometry");
 check(one_geometry.at("eligible_geometry_groups")==2&&one_geometry.at("distinct_eligible_contexts")==6);
 check(one_geometry.at("eligible_finite_context_memberships")==6&&one_geometry.at("nonfinite_genuine_score_contexts")==1);
 check(one_geometry.at("representative_pairs_reported")==1&&one_geometry.at("representative_pairs_omitted")==1&&one_geometry.at("examples_truncated")==true);
 check(one_geometry.at("provenance_eligibility")=="unknown"&&one_geometry.at("cache_admission")==false&&one_geometry.at("certifies_prefix_interval")==false);
 check(one_geometry.at("eligible_geometry_groups")==uncapped.at("one_coordinate_interpolation_geometry").at("eligible_geometry_groups"));
 check(none.at("one_coordinate_interpolation_geometry").at("eligible_geometry_groups")==2&&none.at("one_coordinate_interpolation_geometry").at("representative_pairs_reported")==0);
 const auto records_for=[&](std::initializer_list<std::array<u32,2>>prefixes){std::vector<u32>v;u32 slot=0;
  for(const auto&words:prefixes){const auto offset=v.size();v.insert(v.end(),data.begin(),data.begin()+layout.stride());v[offset]=++slot;
   v[offset+layout.prefixes()]=words[0];v[offset+layout.prefixes()+1]=words[1];}return v;};
 const auto geometry=[&](const std::vector<u32>&v,u64 cap,bool collide=false){return summarize_prefix_metadata(v,layout,cap,collide).at("one_coordinate_interpolation_geometry");};
 const auto two_changed=geometry(records_for({{one,one},{two,two}}),10,true);check(two_changed.at("eligible_geometry_groups")==0);
 const auto one_changed=geometry(records_for({{one,one},{two,one}}),10,true);
 check(one_changed.at("eligible_geometry_groups")==1&&one_changed.at("distinct_eligible_contexts")==2);
 check(one_changed.at("representative_pairs")[0].at("changing_score_channel")==0&&one_changed.at("representative_pairs")[0].at("minimum_prefix_fp32_word")==one&&one_changed.at("representative_pairs")[0].at("maximum_prefix_fp32_word")==two);
 const auto second_changed=geometry(records_for({{one,one},{one,two}}),10);check(second_changed.at("representative_pairs")[0].at("changing_score_channel")==1);
 const auto signed_endpoints=geometry(records_for({{0xbf800000u,one},{two,one},{0xc0000000u,one},{0,one},{0x80000000u,one},{two,one}}),10,true);
 check(signed_endpoints.at("eligible_geometry_groups")==1&&signed_endpoints.at("eligible_finite_context_memberships")==6);
 check(signed_endpoints.at("representative_pairs")[0].at("minimum_prefix_fp32_word")==0xc0000000u&&signed_endpoints.at("representative_pairs")[0].at("maximum_prefix_fp32_word")==two);
 check(signed_endpoints.at("representative_pairs")[0].at("distinct_finite_numeric_values")==4);
 const auto zero_width=geometry(records_for({{0,one},{0x80000000u,one}}),10,true);
 check(zero_width.at("eligible_geometry_groups")==0&&zero_width.at("finite_signed_zero_only_channel_key_groups")==1);
 const auto other_zero_words=geometry(records_for({{one,0},{two,0x80000000u}}),10);check(other_zero_words.at("eligible_geometry_groups")==0);
 const auto nonfinite=geometry(records_for({{0x7fc00001u,one},{one,one},{two,one},{0x7f800000u,one}}),10,true);
 check(nonfinite.at("eligible_geometry_groups")==1&&nonfinite.at("eligible_finite_context_memberships")==2&&nonfinite.at("nonfinite_genuine_score_contexts")==2);
 const auto nonfinite_fixed=geometry(records_for({{one,0x7fc00001u},{two,0x7fc00001u}}),10);check(nonfinite_fixed.at("eligible_geometry_groups")==0);
 const auto square_records=records_for({{one,one},{two,one},{one,two},{two,two}});
 const auto square=geometry(square_records,2,true),square_uncapped=geometry(square_records,100),square_none=geometry(square_records,0);
 check(square.at("eligible_geometry_groups")==4&&square.at("eligible_finite_context_memberships")==8&&square.at("distinct_eligible_contexts")==4);
 check(square.at("representative_pairs_reported")==2&&square.at("representative_pairs_omitted")==2&&square.at("counts_truncated")==false);
 check(square_uncapped.at("representative_pairs_reported")==4&&square_uncapped.at("eligible_geometry_groups")==square.at("eligible_geometry_groups"));
 check(square_none.at("eligible_geometry_groups")==4&&square_none.at("representative_pairs_available")==4&&square_none.at("representative_pairs_reported")==0);
 check(square.at("fingerprint_collision_group_boundaries")==2);
 const auto suffix_changed=summarize_prefix_metadata(records_for({{one,0},{two,0x80000000u}}),structural,10,true).at("one_coordinate_interpolation_geometry");
 check(suffix_changed.at("eligible_geometry_groups")==0);
 const auto opaque_suffix=summarize_prefix_metadata(records_for({{one,0x7fc00001u},{two,0x7fc00001u}}),structural,10,true).at("one_coordinate_interpolation_geometry");
 check(opaque_suffix.at("eligible_geometry_groups")==1&&opaque_suffix.at("nonfinite_genuine_score_contexts")==0);
 const PrefixLayout three{1,3,2,1,3};std::vector<u32>three_records(2*three.stride());
 for(u32 row=0;row<2;++row){const u64 offset=u64(row)*three.stride();three_records[offset]=row+1;three_records[offset+1]=50;
  for(u32 c=0;c<3;++c)three_records[offset+three.prefixes()+c]=one;}
 three_records[three.stride()+three.prefixes()+2]=two;
 const auto third=summarize_prefix_metadata(three_records,three,10,true).at("one_coordinate_interpolation_geometry");
 check(third.at("eligible_geometry_groups")==1&&third.at("representative_pairs")[0].at("changing_score_channel")==2);
 check(empty.at("one_coordinate_interpolation_geometry").at("eligible_geometry_groups")==0&&empty.at("one_coordinate_interpolation_geometry").at("distinct_eligible_contexts")==0);
 // These are synthetic metadata flags, not certificates for a model. They
 // exercise selection/counting only; actual producer and CUDA hook checks are
 // independent prerequisites before interpreting real endpoint counts.
 auto tracked=layout;tracked.gap_evidence=true;
 const auto tagged=[&](const std::vector<u32>&v,std::initializer_list<u32>flags){
  need(v.size()==flags.size()*layout.stride(),"CPU evidence fixture extent");std::vector<u32>out;u64 row=0;
  for(auto flag:flags){out.insert(out.end(),v.begin()+row*layout.stride(),v.begin()+(row+1)*layout.stride());out.push_back(flag);++row;}return out;};
 const auto qualification=[&](const std::vector<u32>&v,u64 cap=10,bool collide=true){
  return summarize_prefix_metadata(v,tracked,cap,collide).at("one_coordinate_interpolation_geometry").at("qualified_endpoint_geometry");};
 const u32 complete=u32(a::gap_evidence::Evidence::CompletedQualified);
 const auto endpoints=records_for({{one,one},{two,one}});
 const auto yes=qualification(tagged(endpoints,{complete,complete}));
 check(yes.at("enabled")==true&&yes.at("full_guard_gap_window_contexts")==2&&yes.at("unknown_contexts")==0);
 check(yes.at("qualified_endpoint_groups")==1&&yes.at("distinct_enclosed_retained_contexts")==2);
 check(yes.at("cache_admission")==false&&yes.at("avoided_construction_measured")==false&&yes.at("certifies_prefix_box")==false);
 check(yes.at("representative_pairs")[0].at("contexts_lower_then_upper")[0].at("full_guard_gap_window_evidence")==true);
 const auto missing_endpoint=qualification(tagged(endpoints,{complete,0}));
 check(missing_endpoint.at("qualified_endpoint_groups")==0&&missing_endpoint.at("unknown_contexts")==1);
 check(qualification(tagged(endpoints,{0,0})).at("full_guard_gap_window_contexts")==0);
 const auto inside=records_for({{0,one},{one,one},{two,one},{0x40400000u,one},{0x40800000u,one}});
 const auto narrowed=qualification(tagged(inside,{0,complete,0,complete,0}));
 check(narrowed.at("qualified_endpoint_groups")==1&&narrowed.at("distinct_enclosed_retained_contexts")==3);
 check(narrowed.at("qualified_endpoint_context_memberships")==2&&narrowed.at("enclosed_retained_context_memberships")==3);
 check(narrowed.at("representative_pairs")[0].at("minimum_prefix_fp32_word")==one&&narrowed.at("representative_pairs")[0].at("maximum_prefix_fp32_word")==0x40400000u);
 check(narrowed.at("representative_pairs")[0].at("enclosed_retained_contexts")==3);
 check(qualification(tagged(records_for({{one,one},{one,one}}),{complete,complete})).at("qualified_endpoint_groups")==0);
 check(qualification(tagged(records_for({{0,one},{0x80000000u,one}}),{complete,complete})).at("qualified_endpoint_groups")==0);
 check(qualification(tagged(records_for({{one,0},{two,0x80000000u}}),{complete,complete})).at("qualified_endpoint_groups")==0);
 auto mismatch=endpoints;mismatch[layout.stride()+1]=51;
 check(qualification(tagged(mismatch,{complete,complete})).at("qualified_endpoint_groups")==0);
 mismatch=endpoints;mismatch[layout.stride()+layout.roots()]=12;
 check(qualification(tagged(mismatch,{complete,complete})).at("qualified_endpoint_groups")==0);
 mismatch=endpoints;mismatch[layout.stride()+layout.positions()]=3;
 check(qualification(tagged(mismatch,{complete,complete})).at("qualified_endpoint_groups")==0);
 mismatch=endpoints;mismatch[layout.stride()+layout.projected()+1]=21;
 check(qualification(tagged(mismatch,{complete,complete})).at("qualified_endpoint_groups")==0);
 mismatch=endpoints;mismatch[layout.stride()+layout.witness()+1]=21;
 const auto distinct_witness=qualification(tagged(mismatch,{complete,complete}));
 check(distinct_witness.at("qualified_endpoint_groups")==1&&distinct_witness.at("representative_pairs")[0].at("same_witness_guard")==false);
 const auto bad_score=qualification(tagged(records_for({{one,one},{0x7f800000u,one}}),{complete,complete}));
 check(bad_score.at("qualified_endpoint_groups")==0);
 bool rejected_pending=false;try{qualification(tagged(endpoints,{u32(a::gap_evidence::Evidence::PendingAllQualified),complete}));}catch(const std::runtime_error&){rejected_pending=true;}
 check(rejected_pending);
 bool rejected_unknown_enum=false;try{qualification(tagged(endpoints,{4,complete}));}catch(const std::runtime_error&){rejected_unknown_enum=true;}
 check(rejected_unknown_enum);
 const auto tracked_square=tagged(square_records,{complete,complete,complete,complete});
 const auto full_square=qualification(tracked_square,100),cap_square=qualification(tracked_square,1),zero_square=qualification(tracked_square,0);
 check(full_square.at("qualified_endpoint_groups")==4&&full_square.at("enclosed_retained_context_memberships")==8&&full_square.at("distinct_enclosed_retained_contexts")==4);
 check(cap_square.at("qualified_endpoint_groups")==4&&cap_square.at("representative_pairs_reported")==1&&cap_square.at("representative_pairs_omitted")==3&&cap_square.at("counts_truncated")==false);
 check(zero_square.at("qualified_endpoint_groups")==4&&zero_square.at("representative_pairs_reported")==0);
 check(qualification(tracked_square,100,false).at("qualified_endpoint_groups")==full_square.at("qualified_endpoint_groups"));
 const auto no_records=qualification({},0);
 check(no_records.at("full_guard_gap_window_contexts")==0&&no_records.at("qualified_endpoint_groups")==0&&no_records.at("distinct_enclosed_retained_contexts")==0);
 check(summary.at("one_coordinate_interpolation_geometry").at("qualified_endpoint_geometry").at("enabled")==false);
 // Support-count summaries are opaque metadata too; graph support itself is
 // computed and separately checked on CUDA, never by a host tree evaluator.
 const PrefixLayout support_layout{1,4,0,0,4,true};
 const auto support_rows=[&](std::initializer_list<std::array<u32,2>>rows){std::vector<u32>out;u32 slot=0;
  for(auto row:rows){const auto start=out.size();out.resize(start+support_layout.stride());out[start]=slot++;out[start+1]=row[0];out[start+support_layout.evidence()]=row[1];}return out;};
 const std::vector<u32>support_sizes={1,2,4};
 const auto supports=summarize_program_support(support_rows({{0,complete},{1,complete},{1,0},{2,complete}}),support_layout,support_sizes);
 check(supports.at("qualified_expanded_contexts")==3&&supports.at("unknown_expanded_contexts")==1);
 check(supports.at("qualified_contexts_with_unused_classes")==2&&supports.at("qualified_contexts_with_at_least_two_unused_classes")==2);
 check(supports.at("distinct_qualified_program_nodes")==3&&supports.at("distinct_qualified_programs_with_at_least_two_unused_classes")==2);
 check(supports.at("qualified_contexts_by_used_class_count")==std::vector<u64>{0,1,1,0,1});
 check(supports.at("syntactic_support_superset")==true&&supports.at("cache_admission")==false&&supports.at("box_lower_bounds_certified")==false);
 const auto repeated_support=summarize_program_support(support_rows({{1,complete},{1,complete},{0,0}}),support_layout,support_sizes);
 check(repeated_support.at("qualified_expanded_contexts")==2&&repeated_support.at("distinct_qualified_program_nodes")==1);
 check(repeated_support.at("qualified_contexts_by_used_class_count")==std::vector<u64>{0,0,2,0,0});
 const auto no_support=summarize_program_support(support_rows({{0,0},{1,0}}),support_layout,support_sizes);
 check(no_support.at("qualified_contexts_with_unused_classes")==0&&no_support.at("distinct_qualified_program_nodes")==0);
 check(summarize_program_support({},support_layout,{}).at("qualified_expanded_contexts")==0);
 bool bad_node=false;try{summarize_program_support(support_rows({{3,complete}}),support_layout,support_sizes);}catch(const std::runtime_error&){bad_node=true;}check(bad_node);
 bool bad_size=false;try{summarize_program_support(support_rows({{0,complete}}),support_layout,{0});}catch(const std::runtime_error&){bad_size=true;}check(bad_size);
 bool too_large=false;try{summarize_program_support(support_rows({{0,complete}}),support_layout,{5});}catch(const std::runtime_error&){too_large=true;}check(too_large);
 bool pending_support=false;try{summarize_program_support(support_rows({{0,1}}),support_layout,support_sizes);}catch(const std::runtime_error&){pending_support=true;}check(pending_support);
 auto untracked_support=support_layout;untracked_support.gap_evidence=false;
 bool disabled_support=false;try{summarize_program_support({},untracked_support,{});}catch(const std::runtime_error&){disabled_support=true;}check(disabled_support);
 return {{"checks",checks},{"failures",0},{"gpu_execution",false},{"model_evaluations",0},
  {"scope","CPU-only exact metadata grouping and one-coordinate geometry; forced collisions, finite numeric endpoints, signed zero, opaque suffix, per-channel membership, synthetic provenance selection, output-support metadata counts and example caps"}};
}

struct Training {u64 rows;u32 features,classes;std::vector<float>values;};
Training read_training(const fs::path&path){
 std::ifstream in(path,std::ios::binary);need(bool(in),"cannot read training fixture");char magic[8];in.read(magic,8);need(std::string(magic,8)=="GHDTSET1","training fixture format differs");
 std::array<u64,3>shape{};in.read(reinterpret_cast<char*>(shape.data()),24);need(bool(in)&&shape[0]&&shape[1]==54&&shape[2]==7,"training fixture shape differs");
 need(shape[0]<=SIZE_MAX/(shape[1]*sizeof(float)),"training fixture extent overflow");Training t{shape[0],u32(shape[1]),u32(shape[2]),std::vector<float>(shape[0]*shape[1])};
 in.read(reinterpret_cast<char*>(t.values.data()),std::streamsize(t.values.size()*sizeof(float)));need(bool(in),"training fixture values truncated");return t; // Labels are never read.
}
// Diagnostic readback happens only after the timed conversion and native
// comparison. No scheduler hook, per-state disk write or proof authority is
// added. Slots can be recycled; this observes retained completed states only.
// A state with resolved outgoing edges was actually expanded. If its final
// graph is a leaf, that construction eventually proved a constant result.
__global__ void inspect_constant_expansions(a::EngineView e,u32 capacity,
 u32*records,u64*masks,u64*stats){
 if(blockIdx.x||threadIdx.x)return;
 const u32 N=e.domain.numeric_features,K=e.source.classes,T=e.source.trees,W=e.domain.mask_words;
 const u64 stride=8ull+2ull*K+T+6ull*N;
 for(u64 slot=0;slot<e.status->states;++slot){
  const auto state=e.arena.states[slot];
  if(state.phase!=3||state.left==a::none||state.right==a::none)continue;
  ++stats[0];
  if(state.node>=e.status->nodes){++stats[3];continue;}
  const auto node=e.arena.nodes[state.node];if(node.feature!=-1)continue;
  ++stats[1];if(stats[2]>=capacity)continue;
  const u64 row=stats[2]++;u32*out=records+row*stride;
  *out++=u32(slot);*out++=state.predicate;*out++=state.node;
  *out++=node.payload;*out++=state.left;*out++=state.right;
  *out++=u32(state.hash);*out++=u32(state.hash>>32);
  for(u32 c=0;c<K;++c)*out++=e.arena.words[slot*K+c];
  for(u32 c=0;c<K;++c)*out++=e.arena.positions[slot*K+c];
  for(u32 t=0;t<T;++t)*out++=u32(e.arena.residual[slot*T+t]);
  const auto projected=a::region(e,u32(slot)),witness=a::witness_region(e,u32(slot));
  for(u32 n=0;n<N;++n)*out++=projected.lower[n];
  for(u32 n=0;n<N;++n)*out++=projected.upper[n];
  for(u32 n=0;n<N;++n)*out++=projected.missing[n];
  for(u32 n=0;n<N;++n)*out++=witness.lower[n];
  for(u32 n=0;n<N;++n)*out++=witness.upper[n];
  for(u32 n=0;n<N;++n)*out++=witness.missing[n];
  for(u32 w=0;w<W;++w){masks[row*2*W+w]=projected.allowed[w];masks[row*2*W+W+w]=witness.allowed[w];}
 }
}
// Re-evaluate the retained state in its original residual region. This calls
// the maintained proof APIs only; the source, prefixes and arena stay read-only.
struct StateProofReplay {
 a::effort::Result direct{};
 a::cover::Result cover{};
 u32 state=0,expected_class=0,multiplier=0,refinement_budget=0,cover_budget=0;
 u32 expected_pairs=0,error=0;
};
__device__ u32 scaled_replay_budget(u32 budget,u32 multiplier){
 const u64 scaled=u64(budget)*multiplier;return scaled>UINT32_MAX?UINT32_MAX:u32(scaled);
}
__global__ void replay_state_proofs(a::EngineView original,af::detail::DraftView scratch,
 const u32*records,u64 stride,u32 count,u32 refinement_budget,u32 cover_budget,
 StateProofReplay*out,u32*range_words){
 const u32 lane=threadIdx.x;af::detail::View view{};view.draft=scratch;
 auto e=af::detail::private_view(original,view,lane);
 auto*stack=scratch.refinement_stack+u64(lane)*e.refinement_stack_capacity;
 const u32 K=e.source.classes,T=e.source.trees;
 for(u32 row=lane;row<count;row+=blockDim.x){
  const u32 id=records[u64(row)*stride],expected=records[u64(row)*stride+3];
  for(u32 pass=0;pass<3;++pass){
   const u64 at=u64(row)*3+pass;StateProofReplay r{};
   r.state=id;r.expected_class=expected;r.multiplier=pass==0?1u:(pass==1?4u:16u);
   r.refinement_budget=scaled_replay_budget(refinement_budget,r.multiplier);
   r.cover_budget=scaled_replay_budget(cover_budget,r.multiplier);
   *e.status=a::Status{};
   if(id>=original.status->states||id>=e.arena.state_capacity||expected>=K||
      e.arena.states[id].phase!=3||e.arena.states[id].predicate==a::none){r.error=1;out[at]=r;continue;}
   const auto R=a::region(original,id);
   for(u32 c=0;c<K;++c){u32 active=0;for(u32 t=0;t<T;++t)
    if(e.arena.residual[u64(id)*T+t]>=0&&u32(e.source.channels[t])==c)++active;
    r.expected_pairs+=active/2;
   }
   r.direct=a::effort::interval_label(e,id,R,stack,e.refinement_stack_capacity,r.refinement_budget);
   // Cover scratch ranges describe its last subcase, not the whole state.
   // Save the direct enclosure before invoking the cover portfolio.
   for(u32 c=0;c<K;++c){range_words[at*2*K+c]=__float_as_uint(e.range_lower[c]);
    range_words[at*2*K+K+c]=__float_as_uint(e.range_upper[c]);}
   if(r.direct.label<0)r.cover=a::cover::portfolio_label(e,id,R,e.arena.states[id].predicate,
     e.draft_region,stack,e.refinement_stack_capacity,r.cover_budget);
   if((r.direct.label>=0&&u32(r.direct.label)!=expected)||
      (r.cover.label>=0&&u32(r.cover.label)!=expected))r.error=2;
   out[at]=r;
  }
 }
}
J replay_json(const StateProofReplay&r,const std::vector<u32>&ranges,u64 at,u32 K){
 const auto&p=r.direct;const auto&c=r.cover;
 std::vector<u32>lower_words(ranges.begin()+at*2*K,ranges.begin()+at*2*K+K);
 std::vector<u32>upper_words(ranges.begin()+at*2*K+K,ranges.begin()+at*2*K+2*K);
 std::vector<float>lower,upper;for(auto word:lower_words)lower.push_back(std::bit_cast<float>(word));
 for(auto word:upper_words)upper.push_back(std::bit_cast<float>(word));
 const bool pair_complete=p.pair_attempts==r.expected_pairs&&p.pair_completed==r.expected_pairs&&p.pair_fallbacks==0;
 return {{"multiplier",r.multiplier},{"refinement_visit_budget",r.refinement_budget},{"cover_visit_budget",r.cover_budget},
  {"expected_collapsed_class",r.expected_class},{"success",p.label>=0||c.label>=0},
  {"direct",{{"label",p.label},{"static_label",p.static_label},{"conditioned_label",p.baseline_label},
    {"attempted",p.attempted},{"visited",p.visited},{"refined_roots",p.refined_roots},
    {"tightened_roots",p.tightened_roots},{"rejected_refinements",p.rejected_refinements},
    {"fallback_frontiers",p.fallback_frontiers},{"conditioned_visit_allocation",p.baseline_visit_budget},
    {"pair_visit_allocation",p.pair_visit_budget},{"pair_visits",p.pair_visits},
    {"pair_attempts",p.pair_attempts},{"pair_completed",p.pair_completed},
    {"pair_fallbacks",p.pair_fallbacks},{"expected_adjacent_pairs",r.expected_pairs},
    {"all_adjacent_pairs_completed",pair_complete},
    {"lower_fp32_words",std::move(lower_words)},{"upper_fp32_words",std::move(upper_words)},
    {"lower",std::move(lower)},{"upper",std::move(upper)}}},
  {"cover",{{"attempted",c.attempted},{"label",c.label},{"success",c.success},
    {"rejection_code",u32(c.rejection)},{"branch_labels",{c.branch_labels[0],c.branch_labels[1]}},
    {"visited",c.visited},{"feasible_cases",c.feasible_cases},{"certified_cases",c.certified_cases},
    {"rival_covers",c.rival_covers},{"rivals_certified",c.rivals_certified},
    {"refined_roots",c.refined_roots},{"tightened_roots",c.tightened_roots},
    {"rejected_refinements",c.rejected_refinements},{"fallback_frontiers",c.fallback_frontiers}}}};
}
J inspect_states(a::Budget&memory,a::EngineView e,const af::Snapshot&result,u32 capacity,
 bool replay_fixed,u32 refinement_budget,u32 cover_budget){
 const u32 N=e.domain.numeric_features,K=e.source.classes,T=e.source.trees,W=e.domain.mask_words;
 const u64 stride=8ull+2ull*K+T+6ull*N;
 const u32 requested=capacity;capacity=u32(std::min<u64>(capacity,result.status.states));
 a::Buffer<u32>records(memory,u64(capacity)*stride);
 a::Buffer<u64>masks(memory,u64(capacity)*2*W),stats(memory,4);stats.zero();
 inspect_constant_expansions<<<1,1>>>(e,capacity,records.data,masks.data,stats.data);gpu_sync();
 const auto counters=stats.download(4);need(counters[3]==0,"invalid completed graph reference in state inspection");
 const auto words=records.download(counters[2]*stride);const auto allowed=masks.download(counters[2]*2*W);
 std::vector<StateProofReplay>replayed;std::vector<u32>replay_ranges;
 if(replay_fixed&&counters[2]){
  need(!e.unary_bounds_enabled&&!e.relational_bounds_enabled,"state replay requires unchanged baseline proof policy");
  constexpr u32 workers=32;
  af::detail::DraftStorage private_scratch(memory,workers,N,K,T,W,e.support_words,e.refinement_stack_capacity);
  a::Buffer<StateProofReplay>proofs(memory,counters[2]*3);
  a::Buffer<u32>ranges(memory,counters[2]*3*2*K);
  replay_state_proofs<<<1,workers>>>(e,private_scratch.view(),records.data,stride,u32(counters[2]),
    refinement_budget,cover_budget,proofs.data,ranges.data);gpu_sync();
  replayed=proofs.download(counters[2]*3);replay_ranges=ranges.download(counters[2]*3*2*K);
  for(const auto&r:replayed)need(r.error==0,"same-state proof replay rejected a state or disagreed with its completed class: "+std::to_string(r.error));
 }
 J samples=J::array();
 for(u64 row=0;row<counters[2];++row){u64 at=row*stride;
  J sample={{"state_slot",words[at]},{"selected_source_predicate",words[at+1]},
   {"final_graph_node",words[at+2]},{"class",words[at+3]},
   {"resolved_left_node",words[at+4]},{"resolved_right_node",words[at+5]},
   {"state_hash_decimal",std::to_string(u64(words[at+6])|(u64(words[at+7])<<32))}};at+=8;
  const auto take=[&](u32 n){std::vector<u32>v(words.begin()+at,words.begin()+at+n);at+=n;return v;};
  sample["prefix_fp32_words"]=take(K);sample["source_positions"]=take(K);
  const auto raw_roots=take(T);std::vector<std::int32_t>roots;roots.reserve(T);
  for(auto word:raw_roots)roots.push_back(std::bit_cast<std::int32_t>(word));sample["residual_roots"]=roots;
  for(u32 which=0;which<2;++which){J region;
   region["lower_fp32_order_keys"]=take(N);region["upper_fp32_order_keys"]=take(N);region["allow_nan"]=take(N);
   region["one_hot_masks"]=std::vector<u64>(allowed.begin()+row*2*W+which*W,allowed.begin()+row*2*W+(which+1)*W);
   sample[which?"witness_region":"projected_residual_region"]=std::move(region);
  }
  if(replay_fixed){J passes=J::array();for(u32 pass=0;pass<3;++pass)
    passes.push_back(replay_json(replayed[row*3+pass],replay_ranges,row*3+pass,K));
   sample["fixed_budget_proof_replays"]=std::move(passes);}
  samples.push_back(std::move(sample));
 }
 return {{"scope","retained expanded states whose completed graph is one class leaf"},
  {"selection","ascending retained state slot; not a random or full-history sample"},
  {"outside_conversion_timing",true},{"requested_capacity",requested},{"allocated_capacity",capacity},
  {"state_slot_highwater",result.status.states},
  {"witness_scope","first admitted path context for a shared state; not the union of all incoming contexts"},
  {"retained_expanded_states",counters[0]},{"retained_constant_expanded_states",counters[1]},
  {"sample_count",counters[2]},{"state_evictions",result.status.state_evictions},
  {"state_reuses",result.status.state_reuses},{"free_state_slots",result.status.free_count},
  {"margin_certificate_implied_by_leaf_collapse",false},
  {"same_state_proof_replay",{{"enabled",replay_fixed},{"outside_conversion_timing",true},
    {"scope","same retained arena state, prefix, residual roots, selected predicate and projected residual region"},
    {"multipliers",{1,4,16}},{"budget_overflow_policy","saturate at UINT32_MAX"},
    {"original_fixed_refinement_budget",replay_fixed?J(refinement_budget):J(nullptr)},
    {"original_fixed_cover_budget",replay_fixed?J(cover_budget):J(nullptr)},
    {"dynamic_replay_omitted_reason","historical per-state adaptive budgets were not retained"},
    {"success_labels_match_completed_class",true},
    {"cover_fallback_scope","conditioned-frontier fallbacks only; nested pair fallback and skipped-pair counts are not exposed by the cover API"},
    {"cover_branch_labels_scope","common-cover branches; rival portfolio aggregate counts are separate"},
    {"direct_ranges_scope","saved immediately after direct effort, before cover scratch is overwritten"},
    {"cover_rejection_codes",{{"none",0},{"gate_unavailable",1},{"budget_disabled",2},{"invalid_state",3},
      {"invalid_predicate",4},{"invalid_region",5},{"invalid_scratch",6},{"uncertified_case",7},
      {"differing_labels",8},{"no_feasible_case",9},{"invalid_source",10},{"visit_budget_violation",11},
       {"candidate_refuted",12},{"target_gap_refuted",13}}}}},
  {"full_source_analysis_domain","witness_region; projected_residual_region requires the saved prefix and residual roots"},
  {"samples",std::move(samples)}};
}
J one_conversion(a::Budget& memory,SourceStorage& source,DomainStorage& domain,RegionStorage& box,
 const u32* starts,const u32*counts,const u64*keys,u32 key_count,u64 signatures,const u32*expected,
 Native&native,const std::function<void()>&check_gate,const std::string&method,bool enabled,bool dynamic,u64 byte_limit,u32 initial_states,u32 inspection_capacity=0,u64 witness_lookup_limit=0,u64 prefix_pair_limit=0,bool prefix_gap_evidence=false){
 const bool relational=enabled&&method=="relational",unary=enabled&&method=="unary";
 const u32 F=source.view.features,K=source.view.classes,T=source.view.trees,N=domain.view.numeric_features,W=domain.view.mask_words;
 auto states=std::make_unique<a::StateStorage>(memory,initial_states,N,K,T,W);
 auto nodes=std::make_unique<a::NodeStorage>(memory,1024);a::EngineView e{};
 e.source=source.view;e.domain=domain.view;e.output_classes=K;a::bind(e,*states,*nodes);
 a::Buffer<a::State>draft(memory,1);a::Buffer<a::Status>status(memory,1);status.zero();
 a::Buffer<u32>words(memory,K),positions(memory,K),blocked(memory,K);a::Buffer<std::int32_t>residual(memory,T);
 RegionStorage scratch(memory,domain.view),witness(memory,domain.view);a::Buffer<u64>active_support(memory,source.support_words);
 a::Buffer<float>range_lower(memory,K),range_upper(memory,K);
 e.draft=draft.data;e.draft_words=words.data;e.draft_positions=positions.data;e.blocked=blocked.data;e.draft_residual=residual.data;
 e.draft_region=scratch.view();e.draft_witness=witness.view();e.support=source.support.data;e.active_support=active_support.data;e.support_words=source.support_words;
 e.minimum=source.minimum.data;e.maximum=source.maximum.data;e.range_lower=range_lower.data;e.range_upper=range_upper.data;e.status=status.data;
 const auto shape=source.walk.download(2);e.refinement_stack_capacity=std::max(shape[0],4*(2*shape[0]+1));
 e.refinement_maximum_visits=shape[1]*(dynamic?(2u+u32(relational)+u32(unary)):3u);e.joint_pair_visit_budget=UINT32_MAX;e.rival_cover_enabled=true;
 if(method=="two_point")e.two_point_screen_enabled=enabled;
 const bool points=e.two_point_screen_enabled;
 e.relational_bounds_enabled=relational;e.unary_bounds_enabled=unary;e.qualified_gap=true; // Authentic gate is checked before every run and at scheduler boundaries.
 af::Limits limits;limits.batch_capacity=32;limits.maximum_batch_capacity=32;limits.max_states=0x3fffffffu;limits.max_nodes=0x3fffffffu;
 limits.admission_threads=32;limits.draft_threads=32;limits.refinement_visit_budget=e.refinement_maximum_visits;
 limits.cover_visit_budget=e.refinement_maximum_visits*2*(K+1);limits.dynamic_refinement=dynamic;limits.dynamic_cover=dynamic;
 limits.checkpoint_on_completion=false;
 auto callback=[&](const float*x,u64 rows,bool margin){return native.predict(x,rows,margin);};
 check_gate();memory.peak=memory.used;const auto start=Clock::now();
 a::initialize_region<<<1,1>>>(e,box.view());gpu_sync();
 a::Buffer<a::gap_evidence::Evidence>gap_evidence;
 const auto result=af::run(e,states,nodes,memory,limits,callback,false,{},check_gate,nullptr,nullptr,prefix_gap_evidence?&gap_evidence:nullptr);gpu_sync();
 const double elapsed=milliseconds(start);const u64 peak=memory.peak;
 need(result.status.complete&&!result.status.error&&!result.stopped,"restricted-domain conversion did not complete");
 a::Buffer<u32>marks(memory,result.status.nodes),reachable(memory,1);marks.zero();
 reachable_nodes<<<1,1>>>(e.arena.nodes,u32(result.status.nodes),result.status.root,marks.data,reachable.data);gpu_sync();const u32 stored=reachable.download(1)[0];
 constexpr u32 tile=4096;a::Buffer<float>rows(memory,u64(tile)*F);a::Buffer<u32>bad(memory,1);bad.zero();
 for(u64 offset=0;offset<signatures;offset+=tile){const u32 count=u32(std::min<u64>(tile,signatures-offset));
  signature_rows<<<(count+127)/128,128>>>(source.view,domain.view,keys,key_count,box.view(),starts,counts,offset,count,rows.data);
  compare_graph<<<(count+127)/128,128>>>(e.arena.nodes,u32(result.status.nodes),result.status.root,rows.data,F,expected+offset,count,bad.data);gpu_sync();
 }
 need(bad.download(1)[0]==0,"converted graph differs from native source on a complete restricted signature domain");
 J inspection;if(inspection_capacity){check_gate();inspection=inspect_states(memory,e,result,inspection_capacity,
   !dynamic,limits.refinement_visit_budget,limits.cover_visit_budget);}
 J witnesses;if(witness_lookup_limit){check_gate();witnesses=inspect_witness_labels(memory,e,result,box.view(),starts,counts,keys,key_count,expected,signatures,witness_lookup_limit);}
 J prefix_reuse;if(prefix_pair_limit){check_gate();prefix_reuse=inspect_prefix_opportunities(memory,e,result,prefix_pair_limit,prefix_gap_evidence?&gap_evidence:nullptr);}
 return {{"prefix_reuse_opportunity",std::move(prefix_reuse)},{"native_witness_inspection",std::move(witnesses)},{"state_inspection",std::move(inspection)},{"method",method},{"enabled",enabled},{"relational",relational},{"unary",unary},{"two_point",points},{"policy",dynamic?"dynamic_proof_effort":"matched_fixed_proof_effort"},{"complete",true},
  {"elapsed_ms",elapsed},{"states",result.status.state_creations},{"expansions",result.status.expansions},
  {"arena_nodes",result.status.nodes},{"reachable_graph_nodes",stored},{"canonical_graph_bytes",64ull+16ull*stored},
  {"peak_owned_gpu_bytes",peak},{"gpu_byte_limit",byte_limit},{"initial_state_capacity",initial_states},{"native_margin_rows",result.native_margin_rows},{"native_public_rows",result.native_public_rows},
  {"class_pruned_states",result.status.class_pruned_states},{"refinement_visits",result.refinement_visits},{"pair_visits",result.refinement_pair_visits},
  {"pair_additional_prunes",result.refinement_pair_additional_prunes},{"cover_visits",result.cover_visits},{"rival_prunes",result.cover_rival_prunes},
  {"relational_attempts",result.relational_attempts},{"relational_visits",result.relational_visits},{"relational_pairs",result.relational_pairs},
  {"relational_completed",result.relational_completed},{"relational_fallbacks",result.relational_fallbacks},{"relational_prunes",result.relational_prunes},
  {"unary_attempts",result.unary_attempts},{"unary_visits",result.unary_visits},{"unary_groups",result.unary_groups},
  {"unary_completed",result.unary_completed},{"unary_fallbacks",result.unary_fallbacks},{"unary_prunes",result.unary_prunes},
  {"unary_optimistic_rejections",result.unary_optimistic_rejections},
   {"point_screen_attempts",result.point_screen_attempts},
   {"point_screen_intersections",result.point_screen_intersections},
   {"point_screen_first_complete",result.point_screen_first_complete},
   {"point_screen_first_qualified",result.point_screen_first_qualified},
   {"point_screen_second_complete",result.point_screen_second_complete},
   {"point_screen_second_qualified",result.point_screen_second_qualified},
   {"point_screen_mixed",result.point_screen_mixed},
   {"point_screen_inconclusive",result.point_screen_inconclusive},
   {"point_screen_first_visits",result.point_screen_first_visits},
   {"point_screen_second_visits",result.point_screen_second_visits},
  {"verified_signatures",signatures},{"mismatches",0},{"refinement_visit_budget",result.refinement_visit_budget},{"cover_visit_budget",result.cover_visit_budget}};
}
}
int main(int argc,char**argv){
 if(argc==2&&std::string(argv[1])=="--prefix-reuse-checks"){try{std::cout<<check_prefix_grouping().dump(2)<<'\n';return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
 if(argc==2&&std::string(argv[1])=="--prefix-support-checks"){try{std::cout<<check_program_support().dump(2)<<'\n';return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
 if(argc==2&&std::string(argv[1])=="--witness-mapping-checks"){try{std::cout<<check_witness_mapping().dump(2)<<'\n';return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
 if(argc==2&&std::string(argv[1])=="--help"){
  std::cout<<"adaptive_real_region_benchmark --model FILE --training GHDTSET1 --library SO --gate-helper SO --gate-qualification JSON --gate-snapshot FRESH_JSON [--count 8] [--cells 2] [--region-json JSON --sample-indices I,J,...] [--repetitions 3] [--policy fixed|dynamic|both] [--method relational|unary|two_point] [--gpu-bytes 2147483648] [--verification-limit 1048576] [--prefix-reuse-pairs 0] [--prefix-gap-evidence 0|1] [--witness-lookups 0] [--initial-states 1024] [--inspect-states 0] [--output JSON]\nRequires CUDA_INJECTION64_PATH and GH_CUPTI_MODULE_CAPTURE_DIR before process startup; optional DP_CUPTI_MODULE_CAPTURE_DIR must resolve to the same directory. Frozen proposals require explicit zero-based sample indices instead of --count/--cells. Their geometry is reconstructed from the current qualified source; saved labels and authority fields are ignored. Training values provide native warmup only in frozen mode. The selected method (default relational) is compared disabled/enabled. Relational/unary comparisons retain the maintained two-point screen in both arms and keep the other optional bound method disabled. Only --method two_point explicitly disables/enables the screen while both optional bound methods remain disabled. The two-point screen samples two points in the saved witness intersected with the current projected region only after an uncertified common cover. Both points must pass the unchanged native gap/window rule with different labels to skip rival proof search; construction still proceeds. Both walks share the existing cover visit budget, so second-point work can displace deeper proofs. Every selected region is converted fully; verification limit rejects the requested fixture size before conversion. No test labels or training operations. Optional --prefix-reuse-pairs N enables post-run exact metadata grouping with at most N reported representative pairs; all retained-state counts are uncapped and 0 disables it. Includes separate one-coordinate interpolation geometry counts with exact other prefix words; --prefix-gap-evidence 1 additionally tracks full projected-continuation gap/window evidence in an ephemeral one-byte-per-state sidecar, default off and requiring prefix grouping. It changes no proof decision and enables separate qualified endpoint counts plus a GPU census of each program's possible output classes; native outcomes and restored evidence remain unknown. The example limit applies separately to each summary. --prefix-reuse-checks exercises this grouping entirely on CPU. The standalone --prefix-support-checks mode uses CUDA to check cross-word output masks and malformed graph rejection. The standalone --witness-mapping-checks mode exercises diagnostic indexing and conservative unknown cases. Optional --witness-lookups N inspects native labels within each retained expanded state witness context after root verification (0 disables); N limits lookups per state, with unfinished or unmapped contexts reported unknown. Optional --initial-states N reserves 1..1073741823 initial state slots within the same GPU memory budget (default 1024); it does not change production defaults. Optional --inspect-states N records retained expanded states that collapsed to a class leaf, once per baseline policy after the first measured conversion; diagnostics are outside conversion timing. Fixed-policy inspections also replay the same retained states at 1x/4x/16x proof budgets using private scratch; dynamic per-state historical budgets are unavailable.\n";return 0;
 }
 try{
  const auto options=parse(argc,argv);cu(cudaSetDevice(0));
  auto source=class_conversion_native::read_source(options.model);
  need(source.identity==native_softprob_gap::source_sha&&source.features==54&&source.outputs==7&&source.objective==class_conversion_native::NativeObjective::softprob,"benchmark requires the existing qualified source unchanged");
  const auto library_sha=dpnative::sha256(dpnative::read_text(options.library));need(library_sha==native_softprob_gap::library_sha,"native library bytes differ");
  const auto frozen=read_frozen(options,source.identity);
  auto training=read_training(options.training);need(!frozen.boxes.empty()||options.count<=training.rows,"region count exceeds training rows");
  a::Budget memory{options.byte_budget};
  a::Buffer<float>training_values(memory,training.values.size());training_values.upload(training.values);training.values.clear();training.values.shrink_to_fit();
  Native native(options.library,source.bytes,u32(source.features),u32(source.outputs));
  (void)native.predict(training_values.data,1,true);(void)native.predict(training_values.data,1,false);
  auto qualified=class_study::try_qualify_native_gap(source.identity,options.library,options.helper,options.snapshot,options.qualification,7,"multi:softprob");
  need(qualified.gate&&qualified.gate->enabled(),"authentic native gate unavailable: "+qualified.reason);
  const auto gate_binding=qualified.gate->binding_sha256();
  const auto check_gate=[&]{need(qualified.gate->enabled()&&qualified.gate->binding_sha256()==gate_binding&&qualified.gate->matches_source(source.identity,7,"multi:softprob",library_sha),"native gate lost or changed");};
  SourceStorage device(memory,source);
  std::vector<std::vector<u32>>groups(2);for(u32 f=10;f<14;++f)groups[0].push_back(f);for(u32 f=14;f<54;++f)groups[1].push_back(f);
  const auto metadata=d::prepare_domain_metadata(54,groups,false);DomainStorage domain(memory,metadata);
  a::Buffer<u64>keys(memory,source.feature.size());threshold_keys<<<(device.view.nodes+127)/128,128>>>(device.view,domain.view,keys.data);gpu_sync();
  auto begin=thrust::device_pointer_cast(keys.data);thrust::sort(thrust::cuda::par,begin,begin+keys.size);const auto end=thrust::unique(thrust::cuda::par,begin,begin+keys.size);gpu_sync();const u32 key_count=u32(end-begin);
  a::Buffer<FrozenBox>proposals(memory,std::max<std::size_t>(1,frozen.boxes.size()));a::Buffer<u32>root_max(memory,10);
  if(!frozen.boxes.empty()){proposals.upload(frozen.boxes);root_max.upload(frozen.root_max);}
  RegionStorage box(memory,domain.view);a::Buffer<u32>starts(memory,metadata.numeric_features),counts(memory,metadata.numeric_features);a::Buffer<RegionInfo>info(memory,1);
  constexpr u32 tile=4096;a::Buffer<float>rows(memory,u64(tile)*device.view.features);a::Buffer<u32>bad(memory,1);
  J regions=J::array();const auto wall_start=Clock::now();
  for(u32 id=0;id<options.count;++id){
   if(frozen.boxes.empty())make_region<<<1,1>>>(device.view,domain.view,keys.data,key_count,training_values.data,training.rows,id,options.count,options.cells,box.view(),starts.data,counts.data,info.data);
   else make_frozen_region<<<1,1>>>(device.view,domain.view,keys.data,key_count,proposals.data,root_max.data,id,box.view(),starts.data,counts.data,info.data);gpu_sync();
   const auto region=info.download(1)[0];need(!region.bad,"invalid selected region: "+std::to_string(region.bad));need(region.signatures<=options.verification_limit,"requested region exceeds explicit verification limit; choose a smaller region or raise --verification-limit");
   a::Buffer<u32>expected(memory,region.signatures);bad.zero();
   for(u64 offset=0;offset<region.signatures;offset+=tile){const u32 count=u32(std::min<u64>(tile,region.signatures-offset));
    signature_rows<<<(count+127)/128,128>>>(device.view,domain.view,keys.data,key_count,box.view(),starts.data,counts.data,offset,count,rows.data);gpu_sync();
    const float*p=native.predict(rows.data,count,false);decode_expected<<<(count+127)/128,128>>>(p,count,7,expected.data+offset,bad.data);gpu_sync();}
   need(bad.download(1)[0]==0,"native signature output invalid");J measurements=J::array();
   for(bool dynamic:{false,true}){if((options.policy=="fixed"&&dynamic)||(options.policy=="dynamic"&&!dynamic))continue;
    for(bool enabled:{false,true})(void)one_conversion(memory,device,domain,box,starts.data,counts.data,keys.data,key_count,region.signatures,expected.data,native,check_gate,options.method,enabled,dynamic,options.byte_budget,options.initial_states);
    std::array<std::vector<J>,2>samples;
    for(u32 repetition=0;repetition<options.repetitions;++repetition)for(u32 offset=0;offset<2;++offset){const u32 mode=(repetition+offset)%2;samples[mode].push_back(one_conversion(memory,device,domain,box,starts.data,counts.data,keys.data,key_count,region.signatures,expected.data,native,check_gate,options.method,mode!=0,dynamic,options.byte_budget,options.initial_states,repetition==0&&mode==0?options.inspect_states:0,repetition==0&&mode==0?options.witness_lookups:0,repetition==0&&mode==0?options.prefix_reuse_pairs:0,repetition==0&&mode==0&&options.prefix_gap_evidence));}
    for(u32 mode=0;mode<2;++mode){std::vector<double>times;for(const auto&r:samples[mode])times.push_back(r.at("elapsed_ms"));auto sorted=times;std::sort(sorted.begin(),sorted.end());
     auto result=samples[mode].front();result.erase("elapsed_ms");result["median_elapsed_ms"]=sorted[sorted.size()/2];result["elapsed_ms_samples"]=times;for(auto&sample:samples[mode]){sample.erase("state_inspection");sample.erase("native_witness_inspection");sample.erase("prefix_reuse_opportunity");}result["work_samples"]=samples[mode];measurements.push_back(std::move(result));}
   }
   J region_provenance={{"selection","uniform_training_row"},{"training_row",region.row}};
   if(!frozen.boxes.empty()){const auto&b=frozen.boxes[id];region_provenance={{"selection","untrusted_frozen_rank_box"},{"sample_index",b.sample_index},{"original_task_id",b.task_id},{"lower_source_threshold_ranks",std::vector<u32>(b.lo,b.lo+10)},{"upper_source_threshold_ranks",std::vector<u32>(b.hi,b.hi+10)},{"snapshot_confers_authority",false}};}
   regions.push_back({{"region_id",id},{"training_row",frozen.boxes.empty()?J(region.row):J(nullptr)},{"region_provenance",std::move(region_provenance)},{"signatures",region.signatures},{"lower_fp32_order_keys",box.lower.download(metadata.numeric_features)},
    {"upper_fp32_order_keys",box.upper.download(metadata.numeric_features)},{"one_hot_masks",box.allowed.download(metadata.mask_words)},{"results",std::move(measurements)}});
  }
  J input_provenance={{"mode","uniform_training_rows"},{"training_values_used_for_region_selection",true}};
  if(!frozen.boxes.empty())input_provenance={{"mode","frozen_source_threshold_region_proposals"},{"proposal_path",options.region_json.string()},{"proposal_sha256",frozen.sha},{"sample_indices",options.sample_indices},{"training_values_used_for_region_selection",false},{"training_values_used_for_native_warmup",true},{"thresholds_reconstructed_from_qualified_source",true},{"snapshot_confers_authority",false},{"saved_labels_and_authorization_ignored",true}};
  const J report={{"format","adaptive-real-region-benchmark-2"},{"complete",true},{"whole_448tree_conversion_complete",false},
   {"scope","complete restricted-domain conversions of the unchanged qualified 448-tree source"},{"source_path",options.model.string()},{"source_sha256",source.identity},
   {"training_path",options.training.string()},{"labels_read",false},{"training_performed",false},{"source_trees",source.roots.size()},{"source_nodes",source.feature.size()},
   {"native_gate",qualified.receipt},{"method",options.method},{"comparison","selected method disabled versus enabled; relational/unary arms retain the maintained two-point screen; other optional bounds disabled"},{"method_counter_scope","relational/unary count direct state proofs; point_screen counts after-common-cover attempts, qualified witnesses and mixed refusals; both point visit counts also contribute to cover_visits"},{"input_provenance",std::move(input_provenance)},{"region_count",options.count},{"requested_cells_per_numeric_feature",frozen.boxes.empty()?J(options.cells):J(nullptr)},{"repetitions",options.repetitions},
   {"timing_scope","initialization and complete shared frontier; excludes source setup and exhaustive native signature comparison"},
   {"disk_io_inside_trials",false},{"prefix_gap_evidence",options.prefix_gap_evidence},{"prefix_reuse_pair_limit",options.prefix_reuse_pairs},{"witness_lookup_limit_per_state",options.witness_lookups},{"initial_state_capacity",options.initial_states},{"state_inspection_enabled",options.inspect_states>0},
   {"comparative_timing_eligible",options.inspect_states==0&&options.witness_lookups==0&&options.prefix_reuse_pairs==0},{"inspection_timing_caveat","post-run GPU readback can perturb subsequent trial caches; inspection runs provide diagnostic evidence only"},
   {"canonical_graph_bytes_scope","64-byte canonical header plus 16 bytes per reachable graph node; not packed runtime size"},
   {"native_allocations_in_owned_peak",false},{"elapsed_ms_including_warmups_and_signature_checks",milliseconds(wall_start)},{"regions",std::move(regions)}};
  const auto text=report.dump(2)+"\n";if(!options.output.empty()){std::ofstream out(options.output);need(bool(out),"cannot write output summary");out<<text;need(bool(out),"summary write failed");}std::cout<<text;return 0;
 }catch(const std::exception&e){std::cerr<<J{{"complete",false},{"whole_448tree_conversion_complete",false},{"error",e.what()}}.dump()<<'\n';return 2;}
}
