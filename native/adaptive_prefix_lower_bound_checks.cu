// Focused GPU arithmetic/guard tests for a range primitive, not native class
// qualification or cache admission. Source evaluation below is test-only CUDA.
#include "class_conversion/adaptive_prefix_lower_bound.cuh"
#include <cstring>
#include <iostream>

namespace a=class_conversion_adaptive;
namespace p=a::prefix_lower_bound;
using U=a::u32;
struct Report {U checks=0,failures=0,first_failure=0,source_rows=0;};
__device__ void expect(Report* r,bool ok,U code){++r->checks;if(!ok){++r->failures;if(!r->first_failure)r->first_failure=code;}}
struct Fixture {
  static constexpr U Capacity=32;
  int feature[Capacity],left[Capacity],right[Capacity],roots[Capacity],channels[Capacity];
  float cut[Capacity],value[Capacity];unsigned char missing[Capacity];
  U minimum[Capacity];a::EngineView e{};
  __device__ void init(U classes=3){
    e={};e.source.feature=feature;e.source.left=left;e.source.right=right;e.source.roots=roots;e.source.channels=channels;
    e.source.cut=cut;e.source.value=value;e.source.missing=missing;e.source.classes=classes;e.source.features=2;e.minimum=minimum;
  }
  __device__ U leaf(float v){U i=e.source.nodes++;feature[i]=-1;left[i]=right[i]=-1;cut[i]=0;value[i]=v;missing[i]=0;minimum[i]=__float_as_uint(v);return i;}
  __device__ U stump(U axis,float lo,float hi){U i=e.source.nodes++;feature[i]=int(axis);cut[i]=0;missing[i]=0;value[i]=0;
    left[i]=int(leaf(lo));right[i]=int(leaf(hi));minimum[i]=__float_as_uint(fminf(lo,hi));return i;}
  __device__ void factor(U root,U channel){const U t=e.source.trees++;roots[t]=int(root);channels[t]=int(channel);}
};
__device__ __noinline__ p::Result call_check(const Fixture* f,U channel,U word){return p::check(f->e,f->roots,channel,word);}
__device__ p::Result at(const Fixture* f,U channel,float prefix){return call_check(f,channel,__float_as_uint(prefix));}
// Independent tiny source interpreter: actual leaf operands and original source
// channel order, without reading static minima or the range implementation.
__device__ float exact_source_channel(const Fixture& f,U channel,float prefix,U signature,bool& finite){
  float score=prefix;finite=isfinite(score);
  for(U t=0;t<f.e.source.trees;++t)if(f.roots[t]>=0&&U(f.channels[t])==channel){
    U node=U(f.roots[t]);while(f.left[node]>=0){const float x=(signature&(1u<<U(f.feature[node])))?1.f:-1.f;
      node=U(x<f.cut[node]?f.left[node]:f.right[node]);}
    score=__fadd_rn(score,f.value[node]);finite&=isfinite(score);
  }return score;
}
__global__ void checks(Fixture* f,Report* r){
  if(blockIdx.x||threadIdx.x)return;
  f->init();auto out=at(f,2,-10.f);
  expect(r,out.complete&&out.lower_window_met&&out.lower_word==__float_as_uint(-10.f)&&!out.root_visits&&!out.additions,1);
  out=at(f,2,nextafterf(-10.f,-INFINITY));
  expect(r,out.complete&&!out.lower_window_met&&out.rejection==p::Rejection::below_window,2);
  out=call_check(f,2,0x80000000u);expect(r,out.lower_window_met&&out.lower_word==0x80000000u,3);
  const U nonfinite_words[3]={0x7f800000u,0xff800000u,0x7fc00001u};
  for(U i=0;i<3;++i){const U word=nonfinite_words[i];
    out=call_check(f,2,word);expect(r,!out.complete&&!out.lower_window_met&&out.rejection==p::Rejection::nonfinite_prefix,4);
  }
  // A finite intermediate outside the final score window is allowed to recover.
  f->init();f->factor(f->leaf(-5),1);f->factor(f->leaf(10),1);out=at(f,1,-9);
  expect(r,out.complete&&out.lower_window_met&&__uint_as_float(out.lower_word)==-4&&out.additions==2&&out.root_visits==2,10);
  f->init();f->factor(f->leaf(100),1);f->factor(f->leaf(-100),1);out=at(f,1,0);
  expect(r,out.lower_window_met&&out.lower_word==0,11);
  // Original order matters: (2^24 + 1) - 2^24 - 10.5 is -10.5 in RN32,
  // whereas moving -2^24 first gives -9.5 and would falsely pass this window.
  f->init(9);const U cancel=f->leaf(-16777216.f),unit=f->leaf(1.f),tail=f->leaf(-10.5f);
  f->factor(unit,8);f->factor(f->leaf(999),3);f->factor(cancel,8);f->factor(tail,8);
  out=at(f,8,16777216.f);bool finite=true;const auto actual=exact_source_channel(*f,8,16777216.f,0,finite);++r->source_rows;
  expect(r,finite&&out.complete&&!out.lower_window_met&&out.lower_word==__float_as_uint(actual)&&actual==-10.5f,20);
  expect(r,out.root_visits==4&&out.additions==3,21);
  // Inventory order, not root ID, defines the fold; a consumed factor does not add.
  f->roots[0]=-1;out=at(f,8,16777216.f);
  expect(r,out.complete&&!out.lower_window_met&&out.additions==2&&out.root_visits==4,22);
  // A class count/channel index beyond the historical seven is ordinary input.
  f->init(70);f->factor(f->leaf(2),69);out=at(f,69,-9);
  expect(r,out.lower_window_met&&__uint_as_float(out.lower_word)==-7&&out.additions==1,30);
  f->e.source.native_margin_classes=5;
  out=at(f,5,0);expect(r,!out.complete&&out.rejection==p::Rejection::invalid_channel_shape,31);
  out=at(f,4,0);expect(r,out.lower_window_met&&out.additions==0&&out.root_visits==1,32);
  f->e.source.native_margin_classes=71;out=at(f,0,0);expect(r,out.rejection==p::Rejection::invalid_channel_shape,33);
  f->init(0);out=at(f,0,0);expect(r,out.rejection==p::Rejection::invalid_channel_shape,34);
  // Intermediate overflow must fail even when a later opposite operand exists.
  f->init();f->factor(f->leaf(__uint_as_float(0x7f7fffffu)),0);f->factor(f->leaf(-__uint_as_float(0x7f7fffffu)),0);
  out=at(f,0,__uint_as_float(0x7f7fffffu));
  expect(r,!out.complete&&!out.lower_window_met&&out.rejection==p::Rejection::nonfinite_addition&&out.additions==1&&out.root_visits==1,40);
  f->value[0]=-__uint_as_float(0x7f7fffffu);f->minimum[0]=__float_as_uint(f->value[0]);
  out=at(f,0,-__uint_as_float(0x7f7fffffu));expect(r,out.rejection==p::Rejection::nonfinite_addition&&!out.complete,41);
  for(U i=0;i<3;++i){const U word=nonfinite_words[i];
    f->init();f->factor(f->leaf(0),0);f->minimum[0]=word;out=at(f,0,0);
    expect(r,out.rejection==p::Rejection::nonfinite_minimum&&!out.additions&&out.root_visits==1,42);
  }
  f->init();f->factor(f->leaf(0),0);f->roots[0]=1;out=at(f,0,0);expect(r,out.rejection==p::Rejection::invalid_root,50);
  f->roots[0]=-2;out=at(f,0,0);expect(r,out.rejection==p::Rejection::invalid_root,51);
  f->roots[0]=0;f->channels[0]=-1;out=at(f,0,0);expect(r,out.rejection==p::Rejection::invalid_channel,52);
  f->channels[0]=3;out=at(f,0,0);expect(r,out.rejection==p::Rejection::invalid_channel,53);
  f->channels[0]=0;auto e=f->e;e.minimum=nullptr;out=p::check(e,f->roots,0,0);expect(r,out.rejection==p::Rejection::missing_source_arrays,54);
  e=f->e;e.source.channels=nullptr;out=p::check(e,f->roots,0,0);expect(r,out.rejection==p::Rejection::missing_source_arrays,55);
  out=p::check(f->e,nullptr,0,0);expect(r,out.rejection==p::Rejection::missing_source_arrays,56);
  // Actual tiny trees with four exhaustive branch signatures. Lower operands
  // are independent static leaf minima; no target grid is used by the primitive.
  f->init(4);f->factor(f->stump(0,-1,2),3);f->factor(f->stump(1,-2,1),3);
  f->factor(f->stump(0,50,-50),1);f->factor(f->leaf(1),3);
  out=at(f,3,-5);expect(r,out.complete&&out.lower_window_met&&__uint_as_float(out.lower_word)==-7&&out.root_visits==4&&out.additions==3,60);
  for(U signature=0;signature<4;++signature){bool low_finite,upper_finite;
    const float target=exact_source_channel(*f,3,-5,signature,low_finite);
    const float donor=exact_source_channel(*f,3,0,signature,upper_finite);r->source_rows+=2;
    expect(r,low_finite&&upper_finite&&__uint_as_float(out.lower_word)<=target&&target<=donor,61+signature);
  }
  // The primitive is deliberately not a qualification authority: an absent
  // engine class gate changes no mathematical range result and proves no class.
  f->e.qualified_gap=false;out=at(f,3,-5);expect(r,out.lower_window_met,70);
  // Incompatible factor minima can be too pessimistic. Failure means fallback,
  // not that the actual lower-window property is false on this guard.
  f->init();f->factor(f->stump(0,-8,8),1);f->factor(f->stump(0,8,-8),1);
  out=at(f,1,-3);expect(r,out.complete&&!out.lower_window_met&&__uint_as_float(out.lower_word)==-19,71);
  for(U signature=0;signature<2;++signature){bool finite_trace;
    const float target=exact_source_channel(*f,1,-3,signature,finite_trace);++r->source_rows;
    expect(r,finite_trace&&target==-3,72+signature);
  }
}
int main(int argc,char** argv){try{
  if(argc==2&&!std::strcmp(argv[1],"--help")){
    std::cout<<"Checks original-order finite prefix lower bounds on tiny CUDA fixtures; no native qualification or cache admission. Running without --help uses CUDA.\n";return 0;
  }
  a::require(argc==1,"unknown prefix lower-bound check argument");
  a::Budget budget{8ull*1024*1024};a::Buffer<Fixture> fixture(budget,1);a::Buffer<Report> report(budget,1);report.zero();
  checks<<<1,1>>>(fixture.data,report.data);a::synchronize();const auto result=report.download(1)[0];
  if(result.failures)throw std::runtime_error("prefix lower-bound assertion "+std::to_string(result.first_failure)+"; failures="+std::to_string(result.failures));
  std::cout<<"{\"checks\":"<<result.checks<<",\"failures\":0,\"independent_source_rows\":"<<result.source_rows
    <<",\"CUDA_executed\":true,\"native_RuntimeGate_qualified\":false,\"cache_admission\":false}\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}}
