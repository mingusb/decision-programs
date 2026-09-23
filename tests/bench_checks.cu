#include "gh/count.cuh"
#include "gh/data.cuh"
#include "gh/learn.cuh"
#include "gh/model.cuh"
#include "../bench/frozen_count.cuh"
#include "check.cuh"
#include <cuda/std/bit>
#include <cmath>

namespace gh::test {
namespace {
constexpr u32 cases = 8, count_bins = 16384, rows = 8192, columns = 8, outputs = 32, rounds = 3;
constexpr u64 max_count = 64ULL << 20, cells = u64(rows)*outputs, feature_cells = u64(rows)*columns;
constexpr u64 scratch_bytes = 16ULL << 20, wire_capacity = 64ULL << 10, node_capacity = 1440;
constexpr u64 count_guard = 0xfedcba9876543210ULL;
constexpr double guard = -987654.25;
__device__ __align__(16) u32 ids[max_count+8];
__device__ u64 counts[count_bins+2];
__device__ float values[feature_cells+2], targets[cells+2];
__device__ std::uint16_t bins[feature_cells+2];
__device__ Feature features[columns+2];
__device__ float metadata[columns*3+2];
__device__ u32 offsets[columns+3];
__device__ Node nodes[node_capacity+2];
__device__ Tree trees[outputs*rounds+2];
__device__ double base[outputs+2], margins[cells+2], prediction[cells+2], losses[rounds+3];
__device__ u64 output_offsets[outputs+3], wire_size;
__device__ std::byte wire[wire_capacity+2];
__device__ __align__(16) std::byte scratch[scratch_bytes+32];
__device__ Schema schema;
__device__ Model forest;
__device__ Training training;
__device__ Status operation[5], timing_status[cases];
__device__ observe::Sample raw[cases][observe::samples];
__device__ observe::Summary summaries[cases];
__device__ u64 fit_bytes, validation_bytes, export_bytes;

__device__ bool pipeline_case(u32 id) { return id == 4 || id == 5; }
__device__ u64 count_size(u32 id) { return id == 3 || id == 7 ? max_count : 16ULL << 20; }
__device__ float feature_value(u32 row, u32 feature) { return (row >> feature) & 1 ? 1.f : -1.f; }
__device__ float target_value(u32 row, u32 output) { return feature_value(row,output%columns)*float(1+output%3); }
__device__ Workspace workspace() { return {scratch+16,scratch_bytes}; }
__device__ Dataset dataset() { return {{values+1,feature_cells},{targets+1,cells},{},rows,columns,outputs}; }
__device__ count::Config count_config(u32 id, u32 variant) {
  count::Config result;
  result.algorithm = count::Algorithm::shared_atomic;
  result.input_type = count::InputType::u32; result.counter_type = count::CounterType::u64;
  result.local_counter = count::LocalCounter::u32;
  result.size = count_size(id); result.bins = count_bins;
  const bool alternative = (id == 2 || id == 3) && variant;
  result.policy = alternative ? 14 : 15; result.blocks = alternative ? 192 : 48;
  if (id >= 6) result.output_clear = count::OutputClear::kernel;
  return result;
}
__device__ TrainConfig train_config(u32 id, u32 variant) {
  TrainConfig result;
  result.rounds = rounds; result.max_depth = 3; result.min_leaf_rows = 1;
  result.learning_rate = 1; result.l2 = 0; result.histogram = Histogram::global;
  const bool batched = id == 5 && variant;
  result.tree_build = batched ? TreeBuild::output_batch : TreeBuild::per_output;
  result.root_counts = batched ? RootCounts::global : RootCounts::per_output;
  result.batched_roots = batched; result.batched_root_splits = batched;
  return result;
}
__global__ void fixtures() {
  const u64 first = u64(blockIdx.x)*blockDim.x+threadIdx.x, stride = u64(gridDim.x)*blockDim.x;
  for (u64 i = first; i < max_count+8; i += stride)
    ids[i] = i >= 4 && i < max_count+4 ? u32((13*(i-4)+7)%count_bins) : 0xfedcba98u;
  for (u64 i = first; i < feature_cells+2; i += stride)
    values[i] = i && i <= feature_cells ? feature_value(u32((i-1)/columns),u32((i-1)%columns)) : float(guard);
  for (u64 i = first; i < cells+2; i += stride)
    targets[i] = i && i <= cells ? target_value(u32((i-1)/outputs),u32((i-1)%outputs)) : float(guard);
}
__global__ void validate_inputs() {
  const u64 first = u64(blockIdx.x)*blockDim.x+threadIdx.x, stride = u64(gridDim.x)*blockDim.x;
  for (u64 i = first; i < max_count; i += stride) GH_CHECK(ids[i+4] < count_bins);
  for (u64 i = first; i < feature_cells; i += stride)
    GH_CHECK(values[i+1] == feature_value(u32(i/columns),u32(i%columns)));
  for (u64 i = first; i < cells; i += stride)
    GH_CHECK(targets[i+1] == target_value(u32(i/outputs),u32(i%outputs)));
}
// Poison reusable destinations before the start marker, preserving resident inputs.
__global__ void reset(u32 id) {
  const u64 first = u64(blockIdx.x)*blockDim.x+threadIdx.x, stride = u64(gridDim.x)*blockDim.x;
  if (!first) {
    for (auto& s : operation) s = {};
    schema = {{features+1,columns},{metadata+1,columns*3},{offsets+1,columns+1}};
    forest = {schema,{nodes+1,node_capacity},{trees+1,outputs*rounds},
      {base+1,outputs},{output_offsets+1,outputs+1}};
    training = {}; training.model = &forest; training.margins = {margins+1,cells};
    training.loss = {losses+1,rounds+1}; wire_size = 0;
  }
  for (u64 i = first; i < 16; i += stride) scratch[i] = scratch[scratch_bytes+16+i] = std::byte{0xa5};
  if (!pipeline_case(id)) {
    for (u64 i = first; i < count_bins+2; i += stride) counts[i] = count_guard;
    return;
  }
  for (u64 i = first; i < cells+2; i += stride) margins[i] = prediction[i] = guard;
  for (u64 i = first; i < feature_cells+2; i += stride) bins[i] = 65535;
  for (u64 i = first; i < columns+2; i += stride) features[i] = {777,777,FeatureType::categorical};
  for (u64 i = first; i < columns*3+2; i += stride) metadata[i] = float(guard);
  for (u64 i = first; i < columns+3; i += stride) offsets[i] = 777;
  for (u64 i = first; i < node_capacity+2; i += stride) nodes[i] = {-1,-1,-1,0,0,guard};
  for (u64 i = first; i < outputs*rounds+2; i += stride) trees[i] = {777,777,777};
  for (u64 i = first; i < outputs+2; i += stride) base[i] = guard;
  for (u64 i = first; i < outputs+3; i += stride) output_offsets[i] = 777;
  for (u64 i = first; i < rounds+3; i += stride) losses[i] = guard;
  for (u64 i = first; i < wire_capacity+2; i += stride) wire[i] = std::byte{0xa5};
}

__global__ void pipeline(u32 id, u32 variant, u32 stage) {
  if (stage) succeeded(operation[stage-1]);
  switch (stage) {
    case 0: submitted(fit_schema(dataset(),{},4,&schema,{bins+1,feature_cells},workspace(),operation)); break;
    case 1:
      fit_bytes = operation[0].required_bytes;
      submitted(train(dataset(),&schema,{bins+1,feature_cells},train_config(id,variant),&training,workspace(),operation+1)); break;
    case 2: submitted(validate_model(&forest,workspace(),operation+2)); break;
    case 3:
      validation_bytes = operation[2].required_bytes;
      submitted(predict(&forest,{bins+1,feature_cells},rows,{prediction+1,cells},false,operation+3)); break;
    default:
      submitted(encode_model(&forest,{wire+1,wire_capacity},{&wire_size,1},workspace(),operation+4)); return;
  }
  pipeline<<<1,1,0,cudaStreamTailLaunch>>>(id,variant,stage+1);
  submitted(cudaGetLastError());
}
// The entire API call executes behind the start marker, including GPU planning.
__global__ void execute(u32 id, u32 variant) {
  if (!id) submitted(finish(operation));
  else if (!pipeline_case(id)) {
    const auto config = count_config(id,variant);
    const Array<const std::byte> input{reinterpret_cast<const std::byte*>(ids+4),config.size*sizeof(u32)};
    const Array<std::byte> output{reinterpret_cast<std::byte*>(counts+1),count_bins*sizeof(u64)};
    submitted(id >= 6 && !variant
      ? bench::frozen_count::count(config.size,input,output,workspace(),operation)
      : count::count(config,input,output,workspace(),operation));
  } else {
    pipeline<<<1,1>>>(id,variant,0); submitted(cudaGetLastError());
  }
}

// Independent byte reader: no production codec/parser is the export oracle.
__device__ u64 read_wire(u64 at, u32 width) {
  GH_CHECK(at <= wire_size && width <= wire_size-at);
  u64 result = 0;
  for (u32 b = 0; b < width; ++b) result |= u64(wire[1+at+b]) << (8*b);
  return result;
}
__device__ void check_export() {
  GH_CHECK(wire_size == 5664 && read_wire(0,8) == 0x4c45444f4d424847ULL);
  GH_CHECK(read_wire(8,4) == 1 && read_wire(12,4) == 0);
  GH_CHECK(read_wire(16,4) == outputs && read_wire(20,4) == columns && read_wire(24,8) == outputs*rounds);
  for (u32 o = 0; o < outputs; ++o) GH_CHECK(read_wire(32+8*o,8) == 0);
  u64 position = 32+8*outputs;
  for (u32 f = 0; f < columns; ++f, position += 16) {
    GH_CHECK(read_wire(position,4) == 0 && read_wire(position+4,4) == 1 && read_wire(position+8,4) == 0);
    GH_CHECK(read_wire(position+12,4) == 0xbf800000u);
  }
  for (u32 t = 0; t < outputs*rounds; ++t) {
    const auto tree = forest.trees.data[t];
    GH_CHECK(read_wire(position,4) == tree.output && read_wire(position+4,4) == tree.count);
    position += 8;
    for (u32 j = 0; j < tree.count; ++j, position += 28) {
      const auto n = forest.nodes.data[tree.begin+j];
      GH_CHECK(read_wire(position,4) == cuda::std::bit_cast<u32>(n.feature));
      GH_CHECK(read_wire(position+4,4) == cuda::std::bit_cast<u32>(n.left));
      GH_CHECK(read_wire(position+8,4) == cuda::std::bit_cast<u32>(n.right));
      GH_CHECK(read_wire(position+12,4) == n.threshold && read_wire(position+16,4) == n.missing_left);
      GH_CHECK(read_wire(position+20,8) == cuda::std::bit_cast<u64>(n.value));
    }
  }
  GH_CHECK(position == wire_size);
}
__global__ void check_payload(u32 id) {
  const u64 first = u64(blockIdx.x)*blockDim.x+threadIdx.x, stride = u64(gridDim.x)*blockDim.x;
  if (!first) for (u32 i = 0; i < 16; ++i)
    GH_CHECK(scratch[i] == std::byte{0xa5} && scratch[scratch_bytes+16+i] == std::byte{0xa5});
  if (!id) return;
  if (!pipeline_case(id)) {
    for (u64 i = first; i < count_size(id); i += stride) GH_CHECK(ids[i+4] == (13*i+7)%count_bins);
    for (u64 i = first; i < count_bins; i += stride) GH_CHECK(counts[i+1] == count_size(id)/count_bins);
    if (!first) {
      GH_CHECK(counts[0] == count_guard && counts[count_bins+1] == count_guard);
      for (u32 i = 0; i < 4; ++i) GH_CHECK(ids[i] == 0xfedcba98u && ids[max_count+4+i] == 0xfedcba98u);
    }
    return;
  }
  for (u64 i = first; i < cells; i += stride) {
    const double expected = target_value(u32(i/outputs),u32(i%outputs));
    GH_CHECK(targets[i+1] == expected && margins[i+1] == expected && prediction[i+1] == expected);
  }
  for (u64 i = first; i < feature_cells; i += stride) {
    GH_CHECK(values[i+1] == feature_value(u32(i/columns),u32(i%columns)));
    GH_CHECK(bins[i+1] == (((i%rows) >> (i/rows)) & 1 ? 2 : 1));
  }
  for (u64 i = first; i < wire_capacity+2; i += stride)
    if (!i || i > wire_size) GH_CHECK(wire[i] == std::byte{0xa5});
  if (first) return;
  GH_CHECK(schema.columns == columns && schema.metadata_count == columns && schema.total_bins == columns*3 && schema.max_feature_bins == 3);
  for (u32 f = 0; f < columns; ++f) {
    GH_CHECK(features[f+1].type == FeatureType::numeric && features[f+1].begin == f && features[f+1].count == 1);
    GH_CHECK(metadata[f+1] == -1 && offsets[f+1] == f*3);
  }
  GH_CHECK(offsets[columns+1] == columns*3);
  GH_CHECK(forest.outputs == outputs && forest.tree_count == outputs*rounds && forest.node_count == outputs*5);
  for (u32 o = 0; o < outputs; ++o) {
    GH_CHECK(base[o+1] == 0 && output_offsets[o+1] == o*rounds);
    for (u32 r = 0; r < rounds; ++r) {
      const auto t = forest.trees.data[o*rounds+r];
      GH_CHECK(t.output == o && t.count == (r ? 1 : 3));
    }
  }
  GH_CHECK(output_offsets[outputs+1] == outputs*rounds);
  GH_CHECK(losses[1] == 145./64.);
  for (u32 r = 1; r <= rounds; ++r) GH_CHECK(losses[r+1] == 0);
  GH_CHECK(values[0] == guard && values[feature_cells+1] == guard && targets[0] == guard && targets[cells+1] == guard);
  GH_CHECK(margins[0] == guard && margins[cells+1] == guard && prediction[0] == guard && prediction[cells+1] == guard);
  GH_CHECK(bins[0] == 65535 && bins[feature_cells+1] == 65535);
  GH_CHECK(features[0].begin == 777 && features[columns+1].begin == 777);
  GH_CHECK(metadata[0] == guard && metadata[columns*3+1] == guard);
  GH_CHECK(offsets[0] == 777 && offsets[columns+2] == 777);
  GH_CHECK(nodes[0].value == guard && nodes[node_capacity+1].value == guard);
  GH_CHECK(trees[0].begin == 777 && trees[outputs*rounds+1].begin == 777);
  GH_CHECK(base[0] == guard && base[outputs+1] == guard);
  GH_CHECK(output_offsets[0] == 777 && output_offsets[outputs+2] == 777);
  GH_CHECK(losses[0] == guard && losses[rounds+2] == guard);
  check_export();
}

__global__ void sample(u32 id, u32 ordinal);
__global__ void summary_check(u32 id) {
  succeeded(timing_status[id]);
  const auto s = summaries[id];
  GH_CHECK(isfinite(s.ratio) && s.lower > 0 && s.lower <= s.ratio && s.upper >= s.ratio);
  printf("GH_BENCH_SUMMARY case=%u protocol=globaltimer-cdp-tail-v1 pairs=15 ratio_A_over_B=%.17g lower95=%.17g upper95=%.17g log_sd=%.17g minimum_ticks=%llu\n",
    id,s.ratio,s.lower,s.upper,s.log_stddev,static_cast<unsigned long long>(s.minimum_ticks));
  if (id+1 < cases) sample<<<1,1,0,cudaStreamTailLaunch>>>(id+1,0);
  else printf("GH_GPU_ACTIVITY benchmark cases=8 raw_samples=288 checks=pass instrumentation=%u\n",unsigned(observe::enabled));
  submitted(cudaGetLastError());
}
__global__ void after_checks(u32 id, u32 ordinal) {
  const auto slot = observe::schedule(ordinal);
  printf("GH_BENCH_CHECK case=%u ordinal=%u checked=1\n",id,ordinal);
  if (pipeline_case(id) && ordinal < 2) {
    export_bytes = operation[4].required_bytes;
    printf("GH_BENCH_MEMORY case=%u variant=%u arena=%llu fit=%llu train=%llu hist=%llu derivatives=%llu tree_state=%llu validation=%llu export=%llu wire=%llu\n",
      id,slot.variant,static_cast<unsigned long long>(scratch_bytes),static_cast<unsigned long long>(fit_bytes),
      static_cast<unsigned long long>(training.workspace_bytes),static_cast<unsigned long long>(training.histogram_bytes),
      static_cast<unsigned long long>(training.derivative_bytes),static_cast<unsigned long long>(training.tree_state_bytes),
      static_cast<unsigned long long>(validation_bytes),static_cast<unsigned long long>(export_bytes),static_cast<unsigned long long>(wire_size));
  }
  if (ordinal+1 < observe::samples) sample<<<1,1,0,cudaStreamTailLaunch>>>(id,ordinal+1);
  else {
    submitted(observe::summarize({raw[id],observe::samples},summaries+id,timing_status+id));
    summary_check<<<1,1,0,cudaStreamTailLaunch>>>(id);
  }
  submitted(cudaGetLastError());
}
__global__ void checked(u32 id, u32 ordinal) {
  const auto slot = observe::schedule(ordinal);
  const auto sample_value = raw[id][ordinal];
  // Preserve this raw observation even when the following correctness gate fails.
  printf("GH_BENCH_RAW case=%u ordinal=%u pair=%u variant=%u warmup=%u begin=%llu end=%llu\n",
    id,ordinal,slot.pair,slot.variant,unsigned(slot.warmup),static_cast<unsigned long long>(sample_value.begin),
    static_cast<unsigned long long>(sample_value.end));
  for (u32 stage = 0; stage < (pipeline_case(id) ? 5u : 1u); ++stage) succeeded(operation[stage]);
  GH_CHECK(timing_status[id].errors == 0 && raw[id][ordinal].end > raw[id][ordinal].begin);
  if (ordinal) GH_CHECK(raw[id][ordinal].begin >= raw[id][ordinal-1].end);
  check_payload<<<512,256>>>(id);
  after_checks<<<1,1,0,cudaStreamTailLaunch>>>(id,ordinal);
  submitted(cudaGetLastError());
}
__global__ void sample(u32 id, u32 ordinal) {
  if (!ordinal) {
    printf("GH_BENCH_CASE case=%u scope=%s instrumentation=%u warmups_per_variant=3 measured_pairs=15\n",
      id,id == 0 ? "empty-CDP-completion" : !pipeline_case(id) ? "clear-count-completion-CDP-stream" : "fit-train-validate-predict-modelbytes-CDP-stream",unsigned(observe::enabled));
    for (u32 v = 0; v < 2; ++v) {
      if (id && !pipeline_case(id)) {
        const auto c = count_config(id,v);
        GH_CHECK(count::supported(c) && count::required_bytes(c) == 0);
        printf("GH_BENCH_CONFIG case=%u variant=%u backend=%s family=shared_atomic input=u32 output=u64 local=u32 elements=%llu bins=%u policy=%u blocks=%u clear=%s scratch=0 cache=resident-no-flush selection=explicit\n",
          id,v,id >= 6 && !v ? "frozen" : "fresh",static_cast<unsigned long long>(c.size),c.bins,c.policy,c.blocks,
          c.output_clear == count::OutputClear::kernel ? "kernel" : "runtime");
      } else if (pipeline_case(id)) {
        const auto c = train_config(id,v);
        printf("GH_BENCH_CONFIG case=%u variant=%u rows=8192 features=8 outputs=32 rounds=3 depth=3 histogram=global split=warp32 tree_build=%s batched_roots=%u root_counts=%s\n",
          id,v,c.tree_build == TreeBuild::output_batch ? "output_batch" : "per_output",unsigned(c.batched_roots),c.root_counts == RootCounts::global ? "global" : "per_output");
      }
    }
  }
  reset<<<128,256>>>(id);
  submitted(observe::boundary({raw[id],observe::samples},ordinal,false,timing_status+id));
  execute<<<1,1>>>(id,observe::schedule(ordinal).variant);
  submitted(cudaGetLastError());
  submitted(observe::boundary({raw[id],observe::samples},ordinal,true,timing_status+id));
  checked<<<1,1,0,cudaStreamTailLaunch>>>(id,ordinal);
  submitted(cudaGetLastError());
}
__global__ void inputs_ready() {
  validate_inputs<<<1024,256>>>();
  sample<<<1,1,0,cudaStreamTailLaunch>>>(0,0);
  submitted(cudaGetLastError());
}
}
__global__ void run() {
  printf("GH_BENCH_PROTOCOL globaltimer-cdp-tail-v1 raw-ticks no-empty-subtraction no-host-timing ranking_allowed=%u\n",unsigned(!observe::enabled));
  fixtures<<<1024,256>>>();
  inputs_ready<<<1,1,0,cudaStreamTailLaunch>>>();
  submitted(cudaGetLastError());
}
}
