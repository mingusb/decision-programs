#include "validation_support.hpp"
#include "ghb/booster.hpp"
#include "ghb/deeper_histogram.cuh"
#include "ghb/split_search.cuh"
#include <algorithm>
#include <array>
#include <limits>
#include <tuple>

using namespace diagnostic;
using ghb::gpu::Stats;
using ghb::gpu::Split;
namespace {
constexpr unsigned rows = 64, columns = 3, outputs = 3, batch = 2, capacity = 2, bins = 5;
constexpr unsigned total_bins = columns * bins;
constexpr Stats guard{1234567.25, -7654321.5, 0xfedcba9876543210ULL};
const Split split_guard{1234567, 7654321, 37, -101.25, -0.0, 55.125, -31.75};
bool same(const Stats& a, const Stats& b) {
  return diagnostic::same(a.gradient, b.gradient) && diagnostic::same(a.hessian, b.hessian) && a.count == b.count;
}
bool same(const Split& a, const Split& b) {
  return a.feature == b.feature && a.threshold == b.threshold && a.missing_left == b.missing_left &&
      diagnostic::same(a.gain, b.gain) && diagnostic::same(a.value, b.value) &&
      diagnostic::same(a.left_value, b.left_value) && diagnostic::same(a.right_value, b.right_value);
}
void splits_json(std::ostream& out, const std::vector<Split>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    const auto& v = values[i]; out << '[' << v.feature << ',' << v.threshold << ',' << v.missing_left << ',';
    number(out, v.gain); out << ','; number(out, v.value); out << ',';
    number(out, v.left_value); out << ','; number(out, v.right_value); out << ']';
  }
  out << ']';
}
void stats_json(std::ostream& out, const std::vector<Stats>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << '['; number(out, values[i].gradient); out << ',';
    number(out, values[i].hessian); out << ',' << values[i].count << ']';
  }
  out << ']';
}
struct RefStats { long double g{}, h{}; unsigned long long n{}; };
RefStats add(RefStats a, RefStats b) { return {a.g+b.g, a.h+b.h, a.n+b.n}; }
RefStats subtract(RefStats a, RefStats b) { return {a.g-b.g, std::max(0.L,a.h-b.h), a.n-b.n}; }
long double leaf(RefStats s) { return -s.g / (s.h + 1); }
long double benefit(RefStats s, long double v) { return -v * (s.g + .5L * (s.h+1) * v); }
struct RefCandidate {
  unsigned feature{}, threshold{}, missing{};
  bool feasible{}, eligible{};
  long double gain{}, parent{}, left{}, right{}, absolute_terms{};
  long double gain_budget{}, leaf_budget{};
  std::array<long double,6> input_statistics{};
};
// Validation oracle only: every threshold/direction from the actual downloaded
// FP64 histogram is evaluated independently. These are NOT GPU candidate scores.
std::vector<RefCandidate> candidates(const Stats* actual, const std::vector<ghb::FeatureType>& types) {
  std::vector<RefCandidate> result;
  for (unsigned f = 0; f < columns; ++f) {
    std::array<RefStats, bins> cells{}; RefStats total{}; long double absolute_gradient{};
    for (unsigned b = 0; b < bins; ++b) {
      const auto v = actual[f*bins+b]; cells[b] = {v.gradient,v.hessian,v.count}; total=add(total,cells[b]);
      absolute_gradient+=std::abs(v.gradient);
    }
    RefStats prefix{};
    for (unsigned threshold = 0; threshold < bins; ++threshold) {
      if (threshold) prefix = types[f] == ghb::FeatureType::numeric ? add(prefix,cells[threshold]) : cells[threshold];
      for (unsigned missing = 0; missing < 2; ++missing) {
        auto left = missing ? add(prefix,cells[0]) : prefix, right = subtract(total,left);
        RefCandidate candidate{f,threshold,missing};
        candidate.parent=leaf(total); candidate.left=leaf(left); candidate.right=leaf(right);
        const auto l=benefit(left,candidate.left), r=benefit(right,candidate.right), p=benefit(total,candidate.parent);
        candidate.gain=l+r-p; candidate.absolute_terms=std::abs(l)+std::abs(r)+std::abs(p);
        candidate.feasible=left.n>=1 && right.n>=1;
        candidate.eligible=candidate.feasible && candidate.gain>0 && std::isfinite(candidate.gain);
        // Bounded fixture has exact integral H and lambda=1, so denominators
        // are >=1. Scale by input magnitudes, not cancellation-prone gain.
        // These conservative diagnostic budgets are not quality allowances or
        // a general formal forward-error proof for arbitrary training inputs.
        candidate.gain_budget=512*std::numeric_limits<double>::epsilon()*(1+absolute_gradient*absolute_gradient);
        candidate.leaf_budget=128*std::numeric_limits<double>::epsilon()*(1+absolute_gradient);
        candidate.input_statistics={total.g,total.h,left.g,left.h,right.g,right.h};
        result.push_back(candidate);
      }
    }
  }
  return result;
}
bool identity_before(const RefCandidate& a,const RefCandidate& b) {
  return std::tie(a.feature,a.threshold,a.missing)<std::tie(b.feature,b.threshold,b.missing);
}
// All contenders are considered. A report may be numerically ambiguous, but an
// arbitrary inferior split or missing a clearly positive split is a failure.
bool audit_choice(const Split& actual,const std::vector<RefCandidate>& all,int feature,bool exact) {
  std::vector<const RefCandidate*> possible;
  for(const auto& c:all)if(feature<0||c.feature==unsigned(feature))possible.push_back(&c);
  require(!possible.empty(),"nonempty reference candidate domain");
  require(actual.feature>=-1,"exact GPU leaf sentinel");
  const RefCandidate* chosen=nullptr;
  if(actual.feature>=0) {
    for(const auto* c:possible)if(c->feature==unsigned(actual.feature)&&c->threshold==actual.threshold&&c->missing==actual.missing_left)chosen=c;
    require(chosen&&chosen->feasible,"exact candidate identity and child eligibility");
    require(std::isfinite(actual.gain)&&actual.gain>0,"finite strictly positive GPU split gain");
    require(std::abs(actual.gain-chosen->gain)<=chosen->gain_budget,"GPU split gain independent reference budget");
    for(auto pair:{std::pair<long double,long double>{actual.value,chosen->parent},{actual.left_value,chosen->left},{actual.right_value,chosen->right}})
      require(std::isfinite(pair.first)&&std::abs(pair.first-pair.second)<=chosen->leaf_budget,"independent parent/child leaf reference budget");
  } else {
    const auto& p=*possible.front();
    require(actual.threshold==0&&actual.missing_left==0&&actual.gain==0&&actual.left_value==0&&actual.right_value==0,
            "exact no-split metadata and unused values");
    require(std::isfinite(actual.value)&&std::abs(actual.value-p.parent)<=p.leaf_budget,"independent no-split leaf reference budget");
  }
  bool ambiguous=chosen&&chosen->gain<=chosen->gain_budget;
  for(const auto* c:possible)if(c->feasible) {
    if(!chosen) {
      require(c->gain<=c->gain_budget,"GPU omitted a decisively positive eligible split");
      if(c->gain>0||c->gain+c->gain_budget>0)ambiguous=true;
    } else {
      require(c->gain-chosen->gain<=c->gain_budget+chosen->gain_budget,"GPU selected a decisively inferior candidate");
      const bool identical=c->feature==chosen->feature&&c->threshold==chosen->threshold&&c->missing==chosen->missing;
      if(!identical&&std::abs(c->gain-chosen->gain)<=c->gain_budget+chosen->gain_budget) {
        ambiguous=true;
        // Dyadic fixture sums are exact. Identical arithmetic inputs yield the
        // same score, so the documented total ordering must break this tie.
        if(exact&&c->input_statistics==chosen->input_statistics)
          require(!identity_before(*c,*chosen),"exact identity tie-break for identical candidate statistics");
      }
    }
  }
  return ambiguous;
}
Split ordered_winner(const Split* feature_candidates) {
  Split best;best.value=feature_candidates[0].value;
  for(unsigned f=0;f<columns;++f) {
    const auto& value=feature_candidates[f];
    if(value.feature<0)continue;
    if(best.feature<0||value.gain>best.gain||(value.gain==best.gain&&
       std::tie(value.feature,value.threshold,value.missing_left)<std::tie(best.feature,best.threshold,best.missing_left)))best=value;
  }
  return best;
}
void reference_selftest() {
  std::vector<Stats> histogram(total_bins);
  constexpr double gradients[columns][bins]{{-2,-8,-4,4,8},{-2,8,-8,4,-4},{-2,4,-4,-8,8}};
  for(unsigned f=0;f<columns;++f)for(unsigned b=0;b<bins;++b)histogram[f*bins+b]={gradients[f][b],4,4};
  const auto all=candidates(histogram.data(),std::vector<ghb::FeatureType>(columns,ghb::FeatureType::numeric));
  std::vector<RefCandidate> ranked;
  for(const auto& c:all)if(c.feature==0&&c.eligible)ranked.push_back(c);
  std::sort(ranked.begin(),ranked.end(),[](const auto& a,const auto& b){return a.gain!=b.gain?a.gain>b.gain:identity_before(a,b);});
  require(ranked.size()>1,"self-test contender fixture");
  auto as_split=[](const RefCandidate& c){return Split{int(c.feature),c.threshold,c.missing,double(c.gain),double(c.parent),double(c.left),double(c.right)};};
  const auto good=as_split(ranked.front());audit_choice(good,all,0,true);
  auto rejected=[&](Split value,const char* label){
    bool failed=false;try{audit_choice(value,all,0,true);}catch(const std::runtime_error&){failed=true;}
    require(failed,label);
  };
  auto inferior=std::find_if(ranked.begin()+1,ranked.end(),[&](const auto& c){return ranked.front().gain-c.gain>ranked.front().gain_budget+c.gain_budget;});
  require(inferior!=ranked.end(),"self-test decisively inferior contender exists");
  rejected(as_split(*inferior),"self-test rejects inferior but individually correct split");
  Split omitted;omitted.value=good.value;rejected(omitted,"self-test rejects omitted positive split");
  auto bad_leaf=good;bad_leaf.left_value+=1;rejected(bad_leaf,"self-test rejects corrupted child leaf");
  auto bad_identity=good;bad_identity.threshold=bins;rejected(bad_identity,"self-test rejects invalid threshold");
  auto bad_gain=good;bad_gain.gain+=1;rejected(bad_gain,"self-test rejects corrupted gain");
  std::cout<<"reference audit self-tests passed: "<<checks<<'\n';
}
struct Snapshot {
  std::vector<double> g,h;
  std::vector<Stats> histogram;
  std::vector<Split> per_feature,winner;
  std::vector<int> assignment;
};
std::string difference(const Snapshot& a, const Snapshot& b) {
  for (std::size_t i=0;i<a.g.size();++i) if(!diagnostic::same(a.g[i],b.g[i]) || !diagnostic::same(a.h[i],b.h[i])) return "derivatives";
  for (std::size_t i=0;i<a.histogram.size();++i) if(!same(a.histogram[i],b.histogram[i])) return "histogram";
  for (std::size_t i=0;i<a.per_feature.size();++i) if(!same(a.per_feature[i],b.per_feature[i])) return "per_feature_candidates";
  for (std::size_t i=0;i<a.winner.size();++i) if(!same(a.winner[i],b.winner[i])) return "winner";
  if(a.assignment!=b.assignment) return "row_assignment";
  return "none";
}
struct Fixture {
  bool exact; unsigned begin,live;
  Stream stream;
  std::vector<std::uint16_t> host_bins;
  std::vector<ghb::FeatureType> types;
  std::vector<double> margins;
  std::vector<float> targets;
  Device<std::uint16_t> packed;
  Device<unsigned> offsets,active,count;
  Device<ghb::FeatureType> feature_types;
  Device<double> prediction,g,h,tree_prediction;
  Device<float> target;
  Device<int> assignment,left_map,right_map;
  Device<Stats> histogram;
  Device<Split> per_feature,winner;
  Device<ghb::gpu::OutputBatch> selector;
  Device<ghb::Node> tree;
  ghb::gpu::DataView data;
  ghb::gpu::SplitConfig config{1,1,0,0,0};
  Graph root_graph,deeper_graph,prediction_graph;
  Fixture(bool exact_fixture,unsigned tile_begin) : exact(exact_fixture),begin(tile_begin),live(std::min(batch,outputs-begin)),
      host_bins(rows*columns),types{ghb::FeatureType::numeric,ghb::FeatureType::categorical,ghb::FeatureType::numeric},
      margins(rows*outputs),targets(rows*outputs),packed(host_bins.size()),offsets(columns+1),active(batch),count(1),
      feature_types(columns),prediction(margins.size()),g(rows*batch),h(rows*batch),tree_prediction(rows*outputs),
      target(targets.size()),assignment(rows*batch),left_map(capacity),right_map(capacity),
      histogram(batch*capacity*total_bins),per_feature(batch*capacity*columns),winner(batch*capacity),selector(1),tree(batch*3) {
    for(unsigned r=0;r<rows;++r) {
      host_bins[r]=r%13==0?0:1+(r%4);
      host_bins[rows+r]=r%11==0?0:1+((r/4)%4);
      host_bins[2*rows+r]=r%17==0?0:1+((r*3+r/5)%4);
      for(unsigned o=0;o<outputs;++o) {
        const auto i=r*outputs+o;
        const double signal=(host_bins[r]<=2?-2.:2.)+o*.25+(r%3)*.0625;
        if(exact) targets[i]=float(-signal);
        else {
          // All bin patterns see a mixture of large signed and small terms.
          constexpr double values[]{0x1p50,-0x1p50,.0625};
          margins[i]=values[(r/4+o)%3]+signal;
        }
      }
    }
    packed.put(host_bins,stream.value); offsets.put({0,5,10,15},stream.value); feature_types.put(types,stream.value);
    prediction.put(margins,stream.value); target.put(targets,stream.value);
    count.put({live},stream.value); selector.put({{begin,live,0,live}},stream.value);
    left_map.put({0,-1},stream.value); right_map.put({1,-1},stream.value);
    data={packed.value,offsets.value,feature_types.value,rows,columns,total_bins,bins};
    reset(true);
    root_graph.capture(stream.value,[&] {
      check(ghb::gpu::gradients_tile(ghb::Objective::squared_error,prediction.value,target.value,nullptr,
          g.value,h.value,rows,outputs,begin,live,stream.value));
      launch_histogram_and_split();
    });
    deeper_graph.capture(stream.value,[&] { launch_histogram_and_split(); });
    prediction_graph.capture(stream.value,[&] {
      check(cudaMemsetAsync(tree_prediction.value,0,tree_prediction.size*sizeof(double),stream.value));
      for(unsigned o=0;o<live;++o) check(ghb::gpu::add_tree(data,tree.value+o*3,3,begin+o,outputs,tree_prediction.value,stream.value));
    });
  }
  void launch_histogram_and_split() {
    check(ghb::gpu::deeper_histogram(data,assignment.value,g.value,h.value,batch,0,batch,capacity,active.value,
        histogram.value,ghb::gpu::DeeperHistogramPolicy::global,0,stream.value,selector.value));
    check(ghb::gpu::find_splits_warp_batched_active(data,histogram.value,batch,capacity,active.value,config,false,
        per_feature.value,winner.value,stream.value,count.value));
  }
  void reset(bool root) {
    histogram.put(std::vector<Stats>(histogram.size,guard),stream.value);
    per_feature.put(std::vector<Split>(per_feature.size,split_guard),stream.value);
    winner.put(std::vector<Split>(winner.size,split_guard),stream.value);
    if(root) {
      g.put(std::vector<double>(g.size,-12345.25),stream.value); h.put(std::vector<double>(h.size,-12345.25),stream.value);
      assignment.put(std::vector<int>(assignment.size,0),stream.value);
      active.put(std::vector<unsigned>(batch,1),stream.value);
    }
  }
  Snapshot snapshot() {
    return {g.get(stream.value),h.get(stream.value),histogram.get(stream.value),per_feature.get(stream.value),
            winner.get(stream.value),assignment.get(stream.value)};
  }
  std::vector<unsigned> route(const Snapshot& root) {
    std::vector<unsigned> next(batch);
    for(unsigned o=0;o<live;++o) {
      const auto& split=root.winner[o*capacity];
      next[o]=split.feature>=0?2:0;
      if(split.feature>=0) check(ghb::gpu::route(data,assignment.value+o*rows,winner.value+o*capacity,
          left_map.value,right_map.value,stream.value));
      else check(cudaMemsetAsync(assignment.value+o*rows,255,rows*sizeof(int),stream.value));
    }
    const auto actual=assignment.get(stream.value);
    for(unsigned o=0;o<live;++o) for(unsigned r=0;r<rows;++r) {
      const auto& s=root.winner[o*capacity]; int expected=-1;
      if(s.feature>=0) {
        const unsigned b=host_bins[s.feature*rows+r];
        const bool left=b==0?s.missing_left:types[s.feature]==ghb::FeatureType::numeric?b<=s.threshold:b==s.threshold;
        expected=left?0:1;
      }
      require(actual[o*rows+r]==expected,"exact GPU row routing");
    }
    active.put(next,stream.value); return next;
  }
  void predict_root(const Snapshot& root,std::ostream& out) {
    std::vector<ghb::Node> nodes(tree.size);
    for(unsigned o=0;o<live;++o) {
      const auto& s=root.winner[o*capacity]; auto& n=nodes[o*3];
      n.value=s.value;
      if(s.feature>=0) { n.feature=s.feature;n.threshold=s.threshold;n.missing_left=s.missing_left;n.left=1;n.right=2;
        nodes[o*3+1].value=s.left_value; nodes[o*3+2].value=s.right_value; }
    }
    tree.put(nodes,stream.value); prediction_graph.launch(stream.value);
    const auto actual=tree_prediction.get(stream.value);
    for(unsigned r=0;r<rows;++r) for(unsigned o=0;o<outputs;++o) {
      double expected=0;
      if(o>=begin&&o<begin+live) {
        const auto& s=root.winner[(o-begin)*capacity]; expected=s.value;
        if(s.feature>=0) { const auto b=host_bins[s.feature*rows+r];
          const bool left=b==0?s.missing_left:types[s.feature]==ghb::FeatureType::numeric?b<=s.threshold:b==s.threshold;
          expected=left?s.left_value:s.right_value; }
      }
      require(diagnostic::same(actual[r*outputs+o],0.0+expected),"exact root-tree GPU prediction");
    }
    out << ",\"root_tree_prediction\":";array(out,actual);
  }
  void record(const Snapshot& s,const std::vector<unsigned>& live_nodes,const Snapshot* baseline,
              unsigned repeat,const char* stage,std::ostream& out) {
    long double maximum_error{}; unsigned unordered_differences{};
    for(unsigned r=0;r<rows;++r) for(unsigned o=0;o<live;++o) {
      const unsigned index=r*live+o, source=r*outputs+begin+o;
      require(diagnostic::same(s.g[index],margins[source]-targets[source])&&s.h[index]==1,"exact squared-error derivative formula");
    }
    for(unsigned i=rows*live;i<rows*batch;++i)
      require(s.g[i]==-12345.25&&s.h[i]==-12345.25,"exact unused derivative tail");
    for(unsigned o=0;o<batch;++o) for(unsigned n=0;n<capacity;++n) for(unsigned f=0;f<columns;++f) for(unsigned b=0;b<bins;++b) {
      const auto index=(o*capacity+n)*total_bins+f*bins+b; const auto actual=s.histogram[index];
      if(o>=live||n>=live_nodes[o]) { require(same(actual,guard),"exact inactive histogram storage");continue; }
      long double gsum{},hsum{},absolute{};unsigned long long count{};
      for(unsigned r=0;r<rows;++r) if(s.assignment[o*rows+r]==int(n)&&host_bins[f*rows+r]==b) {
        const auto i=r*live+o; gsum+=s.g[i];hsum+=s.h[i];absolute+=std::abs(s.g[i]);++count;
      }
      require(actual.count==count,"exact histogram count against independent row reference");
      require(actual.hessian==hsum,"exact dyadic Hessian sum");
      require(std::isfinite(actual.gradient),"finite histogram gradient");
      const auto error=std::abs(actual.gradient-gsum);maximum_error=std::max(maximum_error,error);
      if(exact) require(actual.gradient==gsum,"zero-allowance dyadic gradient sum");
      else {
        const auto bound=4*(count+1)*std::numeric_limits<double>::epsilon()*absolute;
        require(error<=bound,"adversarial FP64 accumulation error budget (not quality allowance)");
      }
      if(baseline&&!same(actual,baseline->histogram[index]))++unordered_differences;
    }
    for(unsigned o=0;o<batch;++o) for(unsigned n=0;n<capacity;++n) if(o>=live||n>=live_nodes[o]) {
      require(same(s.winner[o*capacity+n],split_guard),"exact inactive winner storage");
      for(unsigned f=0;f<columns;++f) require(same(s.per_feature[(o*capacity+n)*columns+f],split_guard),"exact inactive candidate storage");
    }
    const auto divergence=baseline?difference(s,*baseline):"baseline";
    if(exact&&baseline) require(divergence=="none","zero-allowance exact fixture repeated stage");
    out << "{\"fixture\":" << quote(exact?"dyadic_exact":"cancellation_order_sensitive")
        << ",\"tile_begin\":"<<begin<<",\"live_outputs\":"<<live<<",\"repeat\":"<<repeat
        << ",\"stage\":"<<quote(stage)<<",\"first_repeat_divergence\":"<<quote(divergence)
        << ",\"changed_histogram_cells\":"<<unordered_differences<<",\"max_gradient_reference_error\":";
    number(out,maximum_error);out<<",\"derivatives_g\":";array(out,s.g);out<<",\"derivatives_h\":";array(out,s.h);
    out<<",\"row_assignment\":";array(out,s.assignment);out<<",\"histogram_g_h_count\":";stats_json(out,s.histogram);
    out<<",\"gpu_per_feature_candidates\":";splits_json(out,s.per_feature);out<<",\"gpu_winners\":";splits_json(out,s.winner);
    out<<",\"reference_candidates_source\":\"long-double validation from actual FP64 histogram; not GPU per-threshold scores\""
       <<",\"nodes\":[";
    bool first=true;
    for(unsigned o=0;o<live;++o) for(unsigned n=0;n<live_nodes[o];++n) {
      if(!first)out<<',';
      first=false;
      const auto all=candidates(s.histogram.data()+(o*capacity+n)*total_bins,types);
      const auto& actual=s.winner[o*capacity+n];
      const auto* actual_features=s.per_feature.data()+(o*capacity+n)*columns;
      require(same(actual,ordered_winner(actual_features)),"exact GPU winner and tie order from all GPU per-feature candidates");
      std::array<bool,columns+1> ambiguous{};
      for(unsigned f=0;f<columns;++f)ambiguous[f]=audit_choice(actual_features[f],all,int(f),exact);
      ambiguous[columns]=audit_choice(actual,all,-1,exact);
      std::vector<RefCandidate> ranked;
      out<<"{\"output\":"<<begin+o<<",\"node\":"<<n<<",\"all_candidates\":[";
      for(std::size_t i=0;i<all.size();++i) {
        const auto& c=all[i];if(i)out<<',';
        out<<'['<<c.feature<<','<<c.threshold<<','<<c.missing<<','<<(c.eligible?"true":"false")<<',';
        number(out,c.gain);out<<',';number(out,c.parent);out<<',';number(out,c.left);out<<',';number(out,c.right);
        out<<',';number(out,c.gain_budget);out<<',';number(out,c.leaf_budget);out<<']';
        if(c.eligible)ranked.push_back(c);
      }
      std::sort(ranked.begin(),ranked.end(),[](const auto& a,const auto& b) {
        if(a.gain!=b.gain)return a.gain>b.gain;
        return std::tie(a.feature,a.threshold,a.missing)<std::tie(b.feature,b.threshold,b.missing);
      });
      out<<"],\"reference_best_id\":";
      if(ranked.empty())out<<"null";else out<<'['<<ranked[0].feature<<','<<ranked[0].threshold<<','<<ranked[0].missing<<']';
      out<<",\"reference_best_runner_up_margin\":";
      if(ranked.size()<2)out<<"null";else number(out,ranked[0].gain-ranked[1].gain);
      const bool match=ranked.empty()?actual.feature<0:actual.feature==int(ranked[0].feature)&&
          actual.threshold==ranked[0].threshold&&actual.missing_left==ranked[0].missing;
      out<<",\"reference_winner_matches_gpu\":"<<(match?"true":"false")
         <<",\"reference_rank_status\":"<<quote(ambiguous[columns]?"ambiguous_within_numerical_budget":"decisive")
         <<",\"per_feature_reference_ambiguous\":[";
      for(unsigned f=0;f<columns;++f){if(f)out<<',';out<<(ambiguous[f]?"true":"false");}
      out<<"],\"exact_gpu_candidate_winner_order_passed\":true}";
    }
    out<<']';
  }
};
void stages(std::ostream& out) {
  out<<'[';bool first=true;
  for(bool exact:{true,false})for(unsigned begin:{0u,2u}) {
    Fixture fixture(exact,begin); Snapshot root_baseline,deeper_baseline;
    for(unsigned repeat=0;repeat<3;++repeat) {
      fixture.reset(true);fixture.root_graph.launch(fixture.stream.value);
      auto root=fixture.snapshot();
      if(!first)out<<',';
      first=false;
      fixture.record(root,std::vector<unsigned>(batch,1),repeat?&root_baseline:nullptr,repeat,"root",out);
      fixture.predict_root(root,out);out<<'}';
      const auto active=fixture.route(root);fixture.reset(false);fixture.deeper_graph.launch(fixture.stream.value);
      auto deeper=fixture.snapshot();out<<',';
      fixture.record(deeper,active,repeat?&deeper_baseline:nullptr,repeat,"deeper_frontier",out);out<<'}';
      if(!repeat){root_baseline=std::move(root);deeper_baseline=std::move(deeper);}
    }
  }
  out<<']';
}
void lifecycle(std::ostream& out) {
  out<<'[';
  constexpr unsigned shapes[]{1,3,17,2};
  for(unsigned iteration=0;iteration<4;++iteration) {
    std::ostringstream detail;
    {
      ghb::Dataset data;data.rows=65+iteration*7;data.columns=3;data.outputs=shapes[iteration];
      data.values.resize(data.rows*data.columns);data.targets.resize(data.rows*data.outputs);data.weights.resize(data.rows);
      for(unsigned r=0;r<data.rows;++r) {
        data.weights[r]=r%19==0?0.f:1.f;
        for(unsigned f=0;f<data.columns;++f)data.values[r*data.columns+f]=float((r*(f+3)+f)%17)/4;
        for(unsigned o=0;o<data.outputs;++o)data.targets[r*data.outputs+o]=iteration%2?
            float((r+o)%5<2):float(int((r+o)%7)-3)/2;
      }
      ghb::TrainConfig config;config.objective=iteration%2?ghb::Objective::binary_logistic:ghb::Objective::squared_error;
      config.rounds=2;config.max_depth=2;config.max_bins=9;config.min_leaf_rows=2;config.output_tile_size=2;
      config.tree_export_batch_size=iteration<2?0:2;config.nvtx=false;config.histogram=ghb::HistogramPolicy::global;
      config.tree_build=ghb::TreeBuildPolicy::output_batch;
      config.tree_execution=iteration%2?ghb::TreeExecution::graph:ghb::TreeExecution::stream;
      if(iteration==3){config.optimization_order=4;config.max_leaf_value=1;}
      const auto trained=ghb::train(data,config);
      const auto before=trained.model.predict_gpu(data,true),again=trained.model.predict_gpu(data,true);
      const auto cpu=trained.model.predict(data,true);
      require(before.size()==std::size_t(data.rows)*data.outputs,"lifecycle prediction shape");
      std::stringstream bytes;trained.model.save(bytes);const std::string serialized=bytes.str();
      const auto restored=ghb::Model::load(bytes);const auto after=restored.predict_gpu(data,true);
      std::stringstream roundtrip;restored.save(roundtrip);
      require(serialized==roundtrip.str(),"exact model serialization roundtrip");
      require(before.size()==again.size()&&before.size()==after.size()&&before.size()==cpu.size(),"exact all prediction reference extents");
      require(trained.training_loss.size()==config.rounds+1,"exact lifecycle loss history extent");
      long double max_cpu_error{};
      for(std::size_t i=0;i<before.size();++i){
        require(std::isfinite(before[i]),"finite lifecycle prediction");
        require(diagnostic::same(before[i],again[i])&&diagnostic::same(before[i],after[i]),"exact prediction repeat and serialization");
        max_cpu_error=std::max(max_cpu_error,std::abs(static_cast<long double>(before[i])-cpu[i]));
        require(std::isfinite(cpu[i])&&std::abs(before[i]-cpu[i])<=1e-12*std::max(1.,std::abs(cpu[i])),"CPU validation prediction budget");
      }
      for(auto loss:trained.training_loss)require(std::isfinite(loss)&&loss>=0,"finite lifecycle loss");
      detail<<"{\"iteration\":"<<iteration<<",\"rows\":"<<data.rows<<",\"outputs\":"<<data.outputs
            <<",\"order\":"<<config.optimization_order<<",\"graph\":"<<(iteration%2?"true":"false")
            <<",\"export_batch_requested\":"<<config.tree_export_batch_size<<",\"export_batch_effective\":"<<trained.tree_export_batch_size
            <<",\"serialization_bytes\":"<<serialized.size()<<",\"max_cpu_prediction_error\":";
      number(detail,max_cpu_error);detail<<",\"training_loss\":";array(detail,trained.training_loss);
      detail<<",\"gpu_raw_predictions\":";array(detail,before);
    } // Public models, training state and temporary fixtures destroyed before next shape.
    check(cudaDeviceSynchronize());std::size_t free{},total{};check(cudaMemGetInfo(&free,&total));
    if(iteration)out<<',';
    out<<detail.str()<<",\"free_device_bytes_after_destroy\":"<<free<<'}';
  }
  out<<']';
}
}
int main(int argc,char** argv) {
  if(argc==2&&std::string(argv[1])=="--self-test-reference") {
    try{reference_selftest();return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  std::string output,mode="all";
  for(int i=1;i<argc;++i) {
    const std::string arg=argv[i];
    if(arg=="--help") {std::cout<<"ghb_validate_gpu --output NEW.json [--mode all|stage|lifecycle]\nghb_validate_gpu --self-test-reference (CPU only)\nOptional correctness diagnostics; no timing or quality promotion.\n";return 0;}
    if((arg=="--output"||arg=="--mode")&&i+1<argc){(arg=="--output"?output:mode)=argv[++i];}
    else{std::cerr<<"invalid arguments\n";return 2;}
  }
  if(output.empty()||(mode!="all"&&mode!="stage"&&mode!="lifecycle")){std::cerr<<"expected --output NEW.json and valid mode\n";return 2;}
  try {
    Report report(output);std::ostringstream observations;
    try {
      const auto metadata=device_metadata();observations<<"{\"stages\":";
      if(mode!="lifecycle")stages(observations);else observations<<"[]";
      observations<<",\"lifecycle\":";if(mode!="stage")lifecycle(observations);else observations<<"[]";observations<<'}';
      report.write("{\"schema\":1,\"kind\":\"gpu_numerical_lifecycle_validation\",\"passed\":true,\"device\":"+metadata+
          ",\"checks\":"+std::to_string(checks)+",\"scope\":\"bounded diagnostic fixtures; no predictive-quality equivalence claim\","
          "\"stage_shape\":{\"rows\":64,\"features\":3,\"bins_per_feature\":5,\"outputs\":3,\"tile_capacity\":2,\"frontier_capacity\":2},"
          "\"histogram_fields\":[\"gradient\",\"hessian\",\"count\"],"
          "\"gpu_split_fields\":[\"feature\",\"threshold\",\"missing_left\",\"gain\",\"parent_value\",\"left_value\",\"right_value\"],"
          "\"reference_candidate_fields\":[\"feature\",\"threshold\",\"missing_left\",\"eligible\",\"gain\",\"parent_value\",\"left_value\",\"right_value\",\"gain_budget\",\"leaf_budget\"],"
          "\"numerical_rank_policy\":\"all contenders checked; ambiguous intervals are not exact rank preservation; quality gates unchanged\","
          "\"observations\":"+observations.str()+"}\n");
      std::cout<<"GPU diagnostic gates passed: "<<checks<<'\n';return 0;
    }catch(const std::exception& e){report.write("{\"schema\":1,\"passed\":false,\"error\":"+quote(e.what())+
        ",\"partial_observations\":"+quote(observations.str())+"}\n");throw;}
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
