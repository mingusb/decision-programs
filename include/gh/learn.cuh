#pragma once
#include "gh/types.cuh"
#include "gh/observe.cuh"

namespace gh {
enum class Histogram : u32 { global, shared, automatic };
enum class SplitPolicy : u32 { block256, warp32, warp_wide };
enum class TreeBuild : u32 { per_output, output_batch };
enum class RootCounts : u32 { per_output, global, shared };
struct TrainConfig {
  Objective objective{Objective::squared_error};
  u32 classes{2}, rounds{100}, max_depth{6}, min_leaf_rows{10};
  double learning_rate{0.1}, l2{1}, min_child_hessian{1e-8}, min_gain{}, max_leaf_value{};
  u32 order{2}, output_tile{32};
  Histogram histogram{Histogram::automatic};
  SplitPolicy splits{SplitPolicy::warp32};
  TreeBuild tree_build{TreeBuild::per_output};
  RootCounts root_counts{RootCounts::global};
  bool batched_roots{true}, batched_root_splits{true};
  u64 max_histogram_bytes{512ULL << 20}, max_device_bytes{4ULL << 30};
};
struct TuningRecord {
  u32 output{};
  Histogram selected{};
  u64 global_ticks[5]{}, shared_ticks[5]{};
  bool measured{};
};
struct Training {
  Model* model{};
  Array<double> margins, loss;
  observe::Trace trace;
  Array<TuningRecord> tuning;
  u64 workspace_bytes{}, histogram_bytes{}, derivative_bytes{}, tree_state_bytes{};
  u32 frontier_capacity{}, tree_capacity{}, output_capacity{};
};

// Bins and schema are already fitted on training rows. The caller supplies the
// model's base/tree/node/offset buffers and margins/loss; their capacities are
// checked on GPU. The schema arrays are retained by the resulting model.
// loss has rounds+1 entries; model node storage is compact, descriptor order is
// output then round, and node segments may be permuted. Workspace is reusable
// only after tail completion. On failure partial output is not a valid model.
__device__ cudaError_t train(Dataset data, const Schema* schema,
    Array<const std::uint16_t> bins, TrainConfig config, Training* output,
    Workspace workspace, Status* status);
}
