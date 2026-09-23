#include "gh/learn.cuh"
#include "gh/model.cuh"
#include "gh/detail/learning.cuh"
#include "check.cuh"
#include <cuda/std/bit>
#include <cmath>
#include <math_constants.h>

namespace gh::test {
namespace {
constexpr u32 good_cases = 35, bad_cases = 41, max_rows = 65, max_outputs = 33;
constexpr u32 large_rows = 2048, large_columns = 11;
constexpr u32 max_columns = 33, node_capacity = 4096, tree_capacity = 128;
constexpr u64 workspace_bytes = 4ULL << 20;
constexpr double guard = -987654.25;
__device__ float values[max_rows * max_columns], targets[max_rows * max_outputs], weights[max_rows];
__device__ std::uint16_t bins[max_rows * max_columns];
__device__ Feature features[max_columns];
__device__ float metadata[8192];
__device__ u32 offsets[max_columns + 1];
__device__ Node nodes[node_capacity + 2];
__device__ Tree trees[tree_capacity + 2];
__device__ double base[max_outputs + 2], margins[max_rows * max_outputs + 2], losses[6];
__device__ u64 output_offsets[max_outputs + 3], input_hash;
__device__ __align__(16) std::byte scratch[workspace_bytes + 32];
__device__ Schema schema;
__device__ Model model;
__device__ Dataset data;
__device__ Training training;
__device__ TuningRecord tuning[max_outputs];
__device__ TrainConfig config;
__device__ Status status;
__device__ observe::Stamp trace_records[2048];
__device__ u32 trace_count;
__device__ float large_values[large_rows*large_columns], large_targets[large_rows];
__device__ std::uint16_t large_bins[large_rows*large_columns];
__device__ double large_margins[large_rows+2];

__device__ u64 bits(double x) { return cuda::std::bit_cast<u64>(x); }
__device__ Workspace workspace() { return {scratch + 16,workspace_bytes}; }
__device__ bool close(double a, double b) { return isfinite(a) && isfinite(b) && fabs(a-b) <= 2e-12 * (1 + fabs(b)); }
__device__ void math_checks() {
  TrainConfig c; c.l2 = 0; c.max_leaf_value = 4;
  detail::Stats<3> cubic{{1,1,1},1};
  auto value = detail::leaf(cubic,c);
  GH_CHECK(value.value == -2 && close(value.benefit,4./3));
  cubic.d[2] = 0; value = detail::leaf(cubic,c);
  GH_CHECK(value.value == -1 && value.benefit == .5);
  cubic.d[2] = 2 - 0x1p-39; value = detail::leaf(cubic,c);
  GH_CHECK(value.value == -1); // denominator 2^-40 is below 1e-12.
  cubic.d[2] = 2 - 0x1p-38; value = detail::leaf(cubic,c);
  GH_CHECK(value.value == -4); // denominator 2^-39 allows the clipped proposal.
  cubic.d[2] = 3; value = detail::leaf(cubic,c); GH_CHECK(value.value == -1);
  cubic = {{0,1,0},1}; value = detail::leaf(cubic,c);
  GH_CHECK(bits(value.value) == bits(-0.) && value.benefit == 0);
  cubic = {{CUDART_INF,1,0},1}; value = detail::leaf(cubic,c);
  GH_CHECK(value.value == 0 && value.benefit == 0);
  cubic = {{1,CUDART_INF,0},1}; value = detail::leaf(cubic,c);
  GH_CHECK(value.value == 0 && value.benefit == 0);
  cubic = {{1,1,CUDART_NAN},1}; value = detail::leaf(cubic,c);
  GH_CHECK(value.value == 0 && value.benefit == 0);
  c.max_leaf_value = 1;
  cubic = {{0x1.fffffffffffffp1023,0x1p-1022,0},1}; value = detail::leaf(cubic,c);
  GH_CHECK(value.value == -1 && value.benefit == 0x1.fffffffffffffp1023);
  cubic = {{0x1.fffffffffffffp1023,1,2},1}; value = detail::leaf(cubic,c);
  GH_CHECK(value.value == -1 && value.benefit == 0x1.fffffffffffffp1023);
  detail::Stats<4> quartic{{1,1,0,-3},1}; c.max_leaf_value = 4;
  value = detail::leaf(quartic,c); GH_CHECK(value.value == -2 && value.benefit == 2);
  c.max_leaf_value = .5; value = detail::leaf(quartic,c);
  GH_CHECK(value.value == -.5 && value.benefit == .3828125);
  c.max_leaf_value = 1; quartic = {{0x1p600,1,0,1},1}; value = detail::leaf(quartic,c);
  GH_CHECK(value.value == -1 && value.benefit == 0x1p600);
  quartic = {{1,0,0,0},1}; value = detail::leaf(quartic,c);
  GH_CHECK(value.value == 0 && value.benefit == 0);
  const detail::Choice a{2,3,1,4,0,0,0}, feature{1,9,1,4,0,0,0}, threshold{2,2,1,4,0,0,0}, missing{2,3,0,4,0,0,0};
  GH_CHECK(detail::better(a,feature).feature == 1);
  GH_CHECK(detail::better(a,threshold).threshold == 2);
  GH_CHECK(detail::better(a,missing).missing_left == 0);
}
__device__ void unbatched() {
  config.batched_roots = config.batched_root_splits = false;
  config.root_counts = RootCounts::per_output;
}
__device__ u64 fingerprint() {
  u64 h = 0;
  for (u32 i = 0; i < max_rows * max_columns; ++i) h ^= mix(u64(cuda::std::bit_cast<u32>(values[i])) + (u64(i) << 32)) ^ mix(bins[i] + u64(i) * 65537);
  for (u32 i = 0; i < max_rows * max_outputs; ++i) h ^= mix(u64(cuda::std::bit_cast<u32>(targets[i])) + u64(i) * 123456789);
  for (u32 i = 0; i < max_rows; ++i) h ^= mix(u64(cuda::std::bit_cast<u32>(weights[i])) + u64(i) * 987654321);
  for (u32 f = 0; f < schema.columns; ++f)
    h ^= mix(features[f].begin + (u64(features[f].count) << 32)) ^ mix(u64(features[f].type) + offsets[f]);
  for (u64 i = 0; i < schema.metadata_count; ++i) h ^= mix(cuda::std::bit_cast<u32>(metadata[i]) + i * 31337);
  h ^= mix(offsets[schema.columns]);
  return h;
}
__device__ void initialize(u32 c) {
  config = {}; config.rounds = 1; config.max_depth = 1; config.min_leaf_rows = 1;
  config.learning_rate = 1; config.l2 = 0; config.histogram = Histogram::global;
  u32 rows = 8, columns = 2, outputs = 1;
  switch (c) {
    case 1: unbatched(); config.splits = SplitPolicy::block256; break;
    case 2: config.batched_root_splits = false; config.root_counts = RootCounts::per_output; break;
    case 3: config.root_counts = RootCounts::shared; break;
    case 4: unbatched(); config.histogram = Histogram::shared; config.splits = SplitPolicy::block256; break;
    case 5: outputs = 3; config.output_tile = 2; config.tree_build = TreeBuild::output_batch; break;
    case 6: outputs = 33; config.tree_build = TreeBuild::output_batch; break;
    case 7: outputs = 7; config.output_tile = 4; break;
    case 8: config.max_depth = 2; columns = 1; break;
    case 9: config.max_depth = 3; config.rounds = 2; columns = 1; unbatched(); config.histogram = Histogram::shared; break;
    case 10: config.max_depth = 0; break;
    case 11: config.min_leaf_rows = 5; break;
    case 12: config.min_child_hessian = 5; break;
    case 13: config.min_gain = 100; break;
    case 14: config.l2 = 4; break;
    case 15: config.min_gain = 16; break;
    case 16: config.max_leaf_value = .5; config.learning_rate = .25; break;
    case 17: config.rounds = 0; outputs = 3; break;
    case 18: config.min_leaf_rows = 4; break;
    case 19: config.objective = Objective::binary_logistic; break;
    case 20: case 21: case 23:
      config.objective = Objective::binary_logistic; config.order = c == 20 ? 3 : 4;
      config.max_leaf_value = .5; config.tree_build = TreeBuild::output_batch;
      outputs = c == 23 ? 3 : 1; config.rounds = c == 23 ? 2 : 1; break;
    case 22: config.objective = Objective::binary_logistic; outputs = 3; config.tree_build = TreeBuild::output_batch; break;
    case 24: config.objective = Objective::multiclass_softmax; config.classes = 3; rows = 12;
      config.rounds = 2; config.tree_build = TreeBuild::output_batch; config.output_tile = 2; break;
    case 25: config.objective = Objective::multiclass_softmax; config.classes = 33; rows = 33;
      config.max_depth = 0; config.tree_build = TreeBuild::output_batch; break;
    case 26: columns = 1; break;
    case 27: columns = 33; rows = 65; outputs = 3; config.splits = SplitPolicy::warp_wide; break;
    case 28: columns = 1; outputs = 3; config.max_depth = 2; config.rounds = 3; config.learning_rate = 2; break;
    case 29: config.histogram = Histogram::automatic; outputs = 3; config.rounds = 2; unbatched(); break;
    case 30: columns = 33; rows = 65; outputs = 3; config.splits = SplitPolicy::warp_wide; break;
    case 31: rows = 1; config.max_depth = 2; break;
    case 32: config.rounds = 0; config.max_histogram_bytes = 1; break;
    case 33: config.rounds = 0; config.max_depth = 7; config.histogram = Histogram::shared;
      rows = 65; columns = 1; break;
    case 34: config.rounds = 0; config.root_counts = RootCounts::shared; columns = 1; break;
  }
  for (auto& x : values) x = -123;
  for (auto& x : targets) x = -123;
  for (auto& x : weights) x = 1;
  for (auto& x : bins) x = 65535;
  u32 total = 0, meta = 0, maximum = 0;
  for (u32 f = 0; f < columns; ++f) {
    const bool categorical = c == 26 || f == 1;
    const u32 count = c == 34 ? 8190 : ((c == 27 || c == 33) && f == 0) ? 40 : categorical ? 3 : 2;
    features[f] = {meta,count,categorical ? FeatureType::categorical : FeatureType::numeric};
    for (u32 k = 0; k < count; ++k) metadata[meta + k] = float(k);
    meta += count; offsets[f] = total;
    const u32 n = count + (categorical ? 1 : 2); total += n; maximum = max(maximum,n);
    for (u32 r = 0; r < rows; ++r) {
      u32 b = r % 4;
      if (c == 8 || c == 9 || c == 28) b = r < 2 ? (r == 0 ? 0 : 1) : r < 4 ? 2 : 3;
      if (c == 27 && f == 0) b = r % 2 ? 41 : 1;
      bins[u64(f) * rows + r] = b; values[u64(r) * columns + f] = b ? float(b - 1) : CUDART_NAN_F;
    }
  }
  offsets[columns] = total;
  for (u32 r = 0; r < rows; ++r) {
    weights[r] = c == 18 && r >= 4 ? 0 : 1;
    for (u32 o = 0; o < outputs; ++o) {
      float y = r % 4 < 2 ? -2 : 2;
      if (c == 8 || c == 9 || c == 28) y = r < 2 ? -4 : r < 4 ? 0 : 2;
      if (c == 28) y += 0x1p23f;
      if (config.objective == Objective::binary_logistic) y = float((r % 4 >= 2) != bool(o & 1));
      else if (config.objective == Objective::multiclass_softmax) y = float(r % config.classes);
      else if (o & 1) y = -y;
      targets[u64(r) * outputs + o] = y;
    }
  }
  schema = {{features,columns},{metadata,meta},{offsets,u64(columns)+1},columns,total,maximum,meta};
  data = {{values,u64(rows)*columns},{targets,u64(rows)*outputs},{weights,rows},rows,columns,outputs};
  if (c == 31) data.weights = {};
  for (auto& x : nodes) x = {-1,-1,-1,0,0,guard};
  for (auto& x : trees) x = {123,456,789};
  for (auto& x : base) x = guard;
  for (auto& x : margins) x = guard;
  for (auto& x : losses) x = guard;
  for (auto& x : output_offsets) x = UINT64_MAX;
  for (u32 i = 0; i < 16; ++i) scratch[i] = scratch[workspace_bytes + 16 + i] = std::byte{0x5a};
  model = {schema,{nodes+1,node_capacity},{trees+1,tree_capacity},{base+1,max_outputs},
    {output_offsets+1,max_outputs+1},0,0,1,Objective::squared_error};
  training = {}; training.model = &model;
  training.margins = {margins+1,max_rows*max_outputs}; training.loss = {losses+1,4};
  training.tuning = {tuning,max_outputs};
  trace_count = 0;
  if constexpr (observe::enabled) training.trace = {{trace_records, 2048}, &trace_count};
  status = {};
}

// Direct row partitions intentionally share no histogram/prefix/leaf helper.
struct Sum { double residual{}, hessian{}; u32 count{}; };
__device__ double step(Sum s) {
  const double denominator = s.hessian + config.l2;
  double value = denominator > 0 ? (s.residual == 0 ? -0. : s.residual / denominator) : 0;
  if (config.max_leaf_value > 0) value = fmax(-config.max_leaf_value,fmin(config.max_leaf_value,value));
  return value;
}
__device__ double benefit(Sum s) {
  const double value = step(s);
  return value * s.residual - .5 * (s.hessian + config.l2) * value * value;
}
__device__ bool left(u32 bin, FeatureType type, u32 threshold, u32 missing) {
  return bin == 0 ? bool(missing) : type == FeatureType::categorical ? bin == threshold : bin <= threshold;
}
__device__ void exhaustive_split(u32 output) {
  Sum total;
  for (u32 r = 0; r < data.rows; ++r) {
    total.residual += weights[r] * (targets[u64(r)*data.outputs+output] - model.base.data[output]);
    total.hessian += weights[r]; ++total.count;
  }
  int feature = -1; u32 threshold = 0, missing = 0;
  double best = config.min_gain, lv = 0, rv = 0;
  if (config.max_depth && total.count >= 2 * config.min_leaf_rows)
    for (u32 f = 0; f < data.columns; ++f) for (u32 t = 0; t < offsets[f+1]-offsets[f]; ++t)
      for (u32 m = 0; m < 2; ++m) {
        Sum l, r;
        for (u32 row = 0; row < data.rows; ++row) {
          Sum& s = left(bins[u64(f)*data.rows+row],features[f].type,t,m) ? l : r;
          s.residual += weights[row] * (targets[u64(row)*data.outputs+output] - model.base.data[output]);
          s.hessian += weights[row]; ++s.count;
        }
        if (l.count < config.min_leaf_rows || r.count < config.min_leaf_rows ||
            l.hessian < config.min_child_hessian || r.hessian < config.min_child_hessian) continue;
        const double gain = benefit(l) + benefit(r) - benefit(total);
        if (gain > best) { best = gain; feature = f; threshold = t; missing = m; lv = step(l); rv = step(r); }
      }
  const Tree tree = model.trees.data[output];
  const Node root = model.nodes.data[tree.begin];
  GH_CHECK(root.feature == feature);
  if (feature < 0) { GH_CHECK(tree.count == 1); GH_CHECK(bits(root.value) == bits(config.learning_rate*step(total))); }
  else {
    GH_CHECK(tree.count == 3 && root.threshold == threshold && root.missing_left == missing);
    GH_CHECK(bits(model.nodes.data[tree.begin+root.left].value) == bits(config.learning_rate*lv));
    GH_CHECK(bits(model.nodes.data[tree.begin+root.right].value) == bits(config.learning_rate*rv));
  }
}
__device__ double traversal(u32 row, u32 output, u32 rounds) {
  double result = model.base.data[output];
  for (u32 round = 0; round < rounds; ++round) {
    const Tree tree = model.trees.data[u64(output)*config.rounds+round];
    GH_CHECK(tree.output == output && tree.count && tree.begin + tree.count <= model.node_count);
    u32 index = 0, visited = 0;
    while (true) {
      GH_CHECK(index < tree.count && ++visited <= tree.count);
      const Node node = model.nodes.data[tree.begin+index];
      if (node.feature < 0) { result = __dadd_rn(result,node.value); break; }
      GH_CHECK(u32(node.feature) < data.columns);
      const u32 b = bins[u64(node.feature)*data.rows+row];
      index = left(b,features[node.feature].type,node.threshold,node.missing_left) ? node.left : node.right;
    }
  }
  return result;
}
__device__ void check_base_loss() {
  double mass = 0;
  for (u32 r = 0; r < data.rows; ++r) mass += weights[r];
  for (u32 o = 0; o < model.outputs; ++o) {
    double sum = 0;
    for (u32 r = 0; r < data.rows; ++r) sum += weights[r] *
      (config.objective == Objective::multiclass_softmax ? double(targets[r] == o) : targets[u64(r)*data.outputs+o]);
    double expected = sum / mass;
    if (config.objective == Objective::binary_logistic) { expected = fmin(1-1e-12,fmax(1e-12,expected)); expected = log(expected)-log1p(-expected); }
    if (config.objective == Objective::multiclass_softmax) expected = log(fmax(1e-12,expected));
    GH_CHECK(close(model.base.data[o],expected));
  }
  for (u32 round = 0; round <= config.rounds; ++round) {
    double expected = 0;
    for (u32 r = 0; r < data.rows; ++r) {
      if (weights[r] == 0) continue;
      if (config.objective == Objective::multiclass_softmax) {
        double maximum = -CUDART_INF, denominator = 0;
        for (u32 o = 0; o < model.outputs; ++o) maximum = fmax(maximum,traversal(r,o,round));
        for (u32 o = 0; o < model.outputs; ++o) denominator += exp(traversal(r,o,round)-maximum);
        expected += weights[r] * (maximum - traversal(r,u32(targets[r]),round) + log(denominator));
      } else for (u32 o = 0; o < model.outputs; ++o) {
        const double margin = traversal(r,o,round), y = targets[u64(r)*data.outputs+o];
        expected += weights[r] * (config.objective == Objective::squared_error ? .5*(margin-y)*(margin-y) :
          fmax(margin,0.) - y*margin + log1p(exp(-fabs(margin))));
      }
    }
    expected /= mass * (config.objective == Objective::multiclass_softmax ? 1 : model.outputs);
    GH_CHECK(close(training.loss.data[round],expected));
  }
}
__device__ void guards() {
  GH_CHECK(nodes[0].value == guard && nodes[node_capacity+1].value == guard);
  GH_CHECK(trees[0].begin == 123 && trees[tree_capacity+1].begin == 123);
  GH_CHECK(base[0] == guard && base[max_outputs+1] == guard);
  GH_CHECK(margins[0] == guard && margins[max_rows*max_outputs+1] == guard);
  GH_CHECK(losses[0] == guard && losses[5] == guard);
  GH_CHECK(output_offsets[0] == UINT64_MAX && output_offsets[max_outputs+2] == UINT64_MAX);
  for (u32 i = 0; i < 16; ++i) GH_CHECK(scratch[i] == std::byte{0x5a} && scratch[workspace_bytes+16+i] == std::byte{0x5a});
  GH_CHECK(fingerprint() == input_hash);
}
__global__ void start(u32 c);
__global__ void large_start();
__device__ void next(u32 c) {
  if (c+1 < good_cases+bad_cases) { start<<<1,1,0,cudaStreamTailLaunch>>>(c+1); submitted(cudaGetLastError()); }
  else { large_start<<<1,256,0,cudaStreamTailLaunch>>>(); submitted(cudaGetLastError()); }
}
__global__ void validated(u32 c) { succeeded(status); guards(); next(c); }
__device__ u64 middle(const u64* observations) {
  for (u32 i = 0; i < 5; ++i) {
    u32 lower = 0, higher = 0;
    for (u32 j = 0; j < 5; ++j) { lower += observations[j] < observations[i]; higher += observations[j] > observations[i]; }
    if (lower <= 2 && higher <= 2) return observations[i];
  }
  GH_CHECK(false); return 0;
}
__global__ void trained(u32 c) {
  GH_CHECK(status.done == 1); guards();
  if (c >= good_cases) {
    GH_CHECK(status.errors && !(status.errors & runtime));
    if (c-good_cases == 40) {
      GH_CHECK(status.errors & numeric);
      for (u32 r = 0; r < data.rows; ++r) GH_CHECK(training.margins.data[r] == 0);
    }
    next(c); return;
  }
  if (status.errors) printf("learn fixture %u returned errors=%u\n",c,status.errors);
  succeeded(status);
  if constexpr (observe::enabled) {
    GH_CHECK(trace_count >= 6 && trace_count <= 2048 && !(trace_count % 2));
    for (u32 i = 0; i < trace_count; i += 2) {
      GH_CHECK(trace_records[i].stage == trace_records[i + 1].stage);
      GH_CHECK(!trace_records[i].end && trace_records[i + 1].end);
      GH_CHECK(trace_records[i + 1].ticks > trace_records[i].ticks);
      if (i) GH_CHECK(trace_records[i].ticks >= trace_records[i - 1].ticks);
    }
  } else GH_CHECK(trace_count == 0);
  GH_CHECK(model.outputs == (config.objective == Objective::multiclass_softmax ? config.classes : data.outputs));
  GH_CHECK(model.objective == config.objective && model.tree_count == u64(model.outputs)*config.rounds);
  GH_CHECK(model.node_count <= node_capacity && training.workspace_bytes <= workspace_bytes);
  GH_CHECK(training.histogram_bytes <= config.max_histogram_bytes);
  if (!config.rounds) GH_CHECK(!training.histogram_bytes && !training.derivative_bytes && !training.tree_state_bytes &&
      !training.frontier_capacity && !training.tree_capacity && !training.output_capacity);
  for (u32 o = 0; o < model.outputs; ++o) {
    GH_CHECK(tuning[o].measured == (c == 29));
    if (c == 29) {
      GH_CHECK(tuning[o].output == o && (tuning[o].selected == Histogram::global || tuning[o].selected == Histogram::shared));
      for (u32 i = 0; i < 5; ++i) GH_CHECK(tuning[o].global_ticks[i] && tuning[o].shared_ticks[i]);
      GH_CHECK(tuning[o].selected == (middle(tuning[o].shared_ticks) < middle(tuning[o].global_ticks) ? Histogram::shared : Histogram::global));
    }
  }
  for (u32 o = 0; o <= model.outputs; ++o) GH_CHECK(model.output_offsets.data[o] == u64(o)*config.rounds);
  for (u32 r = 0; r < data.rows; ++r) for (u32 o = 0; o < model.outputs; ++o)
    GH_CHECK(bits(training.margins.data[u64(r)*model.outputs+o]) == bits(traversal(r,o,config.rounds)));
  if (config.rounds == 1 && config.max_depth <= 1 && config.objective == Objective::squared_error && data.rows == 8)
    for (u32 o = 0; o < model.outputs; ++o) exhaustive_split(o);
  if (c == 8) for (u32 r = 0; r < data.rows; ++r) GH_CHECK(training.margins.data[r] == targets[r]);
  if (c == 20 || c == 21) for (u32 r = 0; r < data.rows; ++r) GH_CHECK(training.margins.data[r] == (targets[r] ? .5 : -.5));
  check_base_loss();
  status = {}; submitted(validate_model(&model,workspace(),&status));
  validated<<<1,1,0,cudaStreamTailLaunch>>>(c); submitted(cudaGetLastError());
}
__global__ void start(u32 c) {
  initialize(c < good_cases ? c : 0);
  Workspace work = workspace();
  Array<const std::uint16_t> input_bins{bins,u64(data.rows)*data.columns};
  if (c >= good_cases) switch (c-good_cases) {
    case 0: data.rows = 0; break;
    case 1: data.outputs = 0; break;
    case 2: --input_bins.size; break;
    case 3: --data.targets.size; break;
    case 4: --data.weights.size; break;
    case 5: training.margins.size = data.rows-1; break;
    case 6: training.loss.size = config.rounds; break;
    case 7: model.base.size = 0; break;
    case 8: model.trees.size = 0; break;
    case 9: model.nodes.size = 0; break;
    case 10: work.bytes = 1; break;
    case 11: config.max_histogram_bytes = 1; break;
    case 12: config.max_device_bytes = 1; break;
    case 13: targets[0] = cuda::std::bit_cast<float>(0x7fc00000u); break;
    case 14: weights[0] = CUDART_INF_F; break;
    case 15: weights[0] = -1; break;
    case 16: for (u32 r = 0; r < data.rows; ++r) weights[r] = 0; break;
    case 17: config.objective = Objective::binary_logistic; break;
    case 18: config.objective = Objective::multiclass_softmax; config.classes = 3; targets[0] = 3; break;
    case 19: config.order = 1; break;
    case 20: config.order = 3; break;
    case 21: config.histogram = Histogram(99); break;
    case 22: config.splits = SplitPolicy(99); break;
    case 23: config.tree_build = TreeBuild(99); break;
    case 24: config.root_counts = RootCounts(99); break;
    case 25: config.l2 = -1; break;
    case 26: config.learning_rate = 0; break;
    case 27: config.min_child_hessian = CUDART_NAN; break;
    case 28: config.max_depth = 31; break;
    case 29: config.min_leaf_rows = 0; break;
    case 30: config.output_tile = 0; break;
    case 31: config.objective = Objective(99); break;
    case 32: config.tree_build = TreeBuild::output_batch; config.batched_root_splits = false; break;
    case 33: config.batched_roots = false; break;
    case 34: model.output_offsets.size = 1; break;
    case 35: config.objective = Objective::binary_logistic; config.order = 3; config.max_leaf_value = .5; break;
    case 36: config.objective = Objective::binary_logistic; config.order = 4; config.tree_build = TreeBuild::output_batch; break;
    case 37: config.objective = Objective::binary_logistic; config.order = 3; config.max_leaf_value = .5;
      config.tree_build = TreeBuild::output_batch; config.histogram = Histogram::shared; break;
    case 38: config.objective = Objective::binary_logistic; config.order = 4; config.max_leaf_value = .5;
      config.tree_build = TreeBuild::output_batch; config.splits = SplitPolicy::warp_wide; break;
    case 39: targets[0] = CUDART_NAN_F; weights[0] = 0; break;
    case 40:
      config.learning_rate = 0x1p1023; config.max_depth = 2;
      for (u32 r = 0; r < data.rows; ++r) {
        const u32 group = r / 4, bit = r & 1;
        targets[r] = 3.f * (float(group)+float(bit)-1.f);
        bins[r] = 1+group; bins[data.rows+r] = 1+bit;
        values[u64(r)*2] = float(group); values[u64(r)*2+1] = float(bit);
      }
      break;
  }
  input_hash = fingerprint();
  submitted(train(data,&schema,input_bins,config,&training,work,&status));
  trained<<<1,1,0,cudaStreamTailLaunch>>>(c); submitted(cudaGetLastError());
}
__global__ void large_validated() {
  succeeded(status);
  GH_CHECK(nodes[0].value == guard && nodes[node_capacity+1].value == guard);
  for (u32 i = 0; i < 16; ++i) GH_CHECK(scratch[i] == std::byte{0x5a} && scratch[workspace_bytes+16+i] == std::byte{0x5a});
  printf("learn checks passed: higher-order landmarks, %u small success fixtures, %u rejection fixtures, one 2048-live-node frontier; calibration records checked\n",good_cases,bad_cases);
}
__global__ void large_validate() {
  status = {}; submitted(validate_model(&model,workspace(),&status));
  large_validated<<<1,1,0,cudaStreamTailLaunch>>>(); submitted(cudaGetLastError());
}
__global__ void large_checked() {
  succeeded(status);
  GH_CHECK(model.base.data[0] == 1023.5 && model.tree_count == 1 && model.node_count == 4095);
  GH_CHECK(training.frontier_capacity == 2048 && training.tree_capacity == 4095);
  GH_CHECK(model.trees.data[0].begin == 0 && model.trees.data[0].count == 4095 && model.trees.data[0].output == 0);
  GH_CHECK(training.loss.data[0] == (double(large_rows)*large_rows-1)/24 && training.loss.data[1] == 0);
  GH_CHECK(large_margins[0] == guard && large_margins[large_rows+1] == guard);
  for (u32 i = blockIdx.x*blockDim.x+threadIdx.x; i < 4095; i += gridDim.x*blockDim.x) {
    const Node n = model.nodes.data[i];
    if (i < 2047) {
      u32 depth = 0; for (u32 x = i+1; x > 1; x >>= 1) ++depth;
      GH_CHECK(n.feature == int(depth) && n.threshold == 1 && n.missing_left == 0);
      GH_CHECK(n.left == int(2*i+1) && n.right == int(2*i+2));
    } else GH_CHECK(n.feature == -1 && n.value == double(i-2047)-1023.5);
  }
  for (u32 r = blockIdx.x*blockDim.x+threadIdx.x; r < large_rows; r += gridDim.x*blockDim.x) {
    GH_CHECK(training.margins.data[r] == r && large_targets[r] == r);
    for (u32 f = 0; f < large_columns; ++f) {
      const u32 bit = (r >> (10-f)) & 1;
      GH_CHECK(large_bins[u64(f)*large_rows+r] == 1+bit && large_values[u64(r)*large_columns+f] == bit);
    }
  }
  if (!blockIdx.x && !threadIdx.x) { large_validate<<<1,1,0,cudaStreamTailLaunch>>>(); submitted(cudaGetLastError()); }
}
__global__ void large_start() {
  if (!threadIdx.x) {
    initialize(0); config.max_depth = 12;
    for (u32 f = 0; f < large_columns; ++f) {
      features[f] = {f,1,FeatureType::numeric}; metadata[f] = .5f; offsets[f] = 3*f;
    }
    offsets[large_columns] = 3*large_columns;
    schema = {{features,large_columns},{metadata,large_columns},{offsets,large_columns+1},large_columns,3*large_columns,3,large_columns};
    data = {{large_values,large_rows*large_columns},{large_targets,large_rows},{},large_rows,large_columns,1};
    training.margins = {large_margins+1,large_rows}; large_margins[0] = large_margins[large_rows+1] = guard;
  }
  __syncthreads();
  for (u32 r = threadIdx.x; r < large_rows; r += blockDim.x) {
    large_targets[r] = float(r);
    for (u32 f = 0; f < large_columns; ++f) {
      const u32 bit = (r >> (10-f)) & 1;
      large_values[u64(r)*large_columns+f] = float(bit); large_bins[u64(f)*large_rows+r] = 1+bit;
    }
  }
  __syncthreads();
  if (!threadIdx.x) {
    submitted(train(data,&schema,{large_bins,large_rows*large_columns},config,&training,workspace(),&status));
    large_checked<<<16,256,0,cudaStreamTailLaunch>>>(); submitted(cudaGetLastError());
  }
}
}
__global__ void run() { math_checks(); start<<<1,1>>>(0); submitted(cudaGetLastError()); }
}
