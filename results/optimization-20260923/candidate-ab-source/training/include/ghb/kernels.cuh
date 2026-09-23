#pragma once
#include "ghb/booster.hpp"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ghb::gpu {

// Feature-major packed bin IDs; offsets has columns+1 entries. Each feature
// contributes its complete histogram including missing bin zero.
struct DataView {
  const std::uint16_t* bins{};
  const std::uint32_t* offsets{};
  const FeatureType* types{};
  std::uint32_t rows{}, columns{}, total_bins{};
  std::uint32_t max_feature_bins{};
};
struct Stats { double gradient{}, hessian{}; unsigned long long count{}; };
struct Split {
  std::int32_t feature{-1};
  std::uint32_t threshold{}, missing_left{};
  double gain{}, value{}, left_value{}, right_value{};
};
struct SplitConfig {
  std::uint32_t min_leaf_rows{};
  double l2{}, min_child_hessian{}, min_gain{}, max_leaf_value{};
};

// All launches are asynchronous on stream, allocate no memory, and return
// launch errors. Device buffers must remain alive until stream completion.
cudaError_t initialize_predictions(double* predictions, const double* base,
                                  std::uint32_t rows, std::uint32_t outputs,
                                  cudaStream_t stream);
cudaError_t gradients(Objective objective, const double* predictions,
                      const float* targets, const float* weights,
                      double* gradient, double* hessian,
                      std::uint32_t rows, std::uint32_t outputs,
                      cudaStream_t stream);
// Full row-major predictions/targets, compact row-major gradient output tile.
// Multiclass targets still contain one label per row. A caller updating
// multiclass margins between tiles must retain the pre-round margins itself.
cudaError_t gradients_tile(Objective objective, const double* predictions,
                           const float* targets, const float* weights,
                           double* gradient, double* hessian,
                           std::uint32_t rows, std::uint32_t outputs,
                           std::uint32_t output_begin, std::uint32_t output_count,
                           cudaStream_t stream);
// assignments[row] is an active-node index in [0,nodes), or -1 to skip.
// Gradient/Hessian use row-major outputs. Every call overwrites histograms.
cudaError_t histogram(DataView data, const std::int32_t* assignments,
                      const double* gradient, const double* hessian,
                      std::uint32_t outputs, std::uint32_t output,
                      std::uint32_t nodes, Stats* histograms,
                      HistogramPolicy policy, cudaStream_t stream);
bool shared_supported(DataView data, std::uint32_t nodes);
// One candidate per (node, feature); then one winner per node. Caller supplies
// nodes*columns candidates and nodes winners. force_leaf stops splitting.
cudaError_t find_splits(DataView data, const Stats* histograms,
                       std::uint32_t nodes, SplitConfig config, bool force_leaf,
                       Split* candidates, Split* winners, cudaStream_t stream);
// Child maps contain -1 for terminal nodes or an index in the next active
// level. Terminal rows are deactivated; predictions updated separately.
cudaError_t route(DataView data, std::int32_t* assignments, const Split* winners,
                  const std::int32_t* left_map, const std::int32_t* right_map,
                  cudaStream_t stream);
// Traverse a finished tree and add its leaf values to one prediction output.
cudaError_t add_tree(DataView data, const Node* nodes, std::uint32_t node_count,
                     std::uint32_t output, std::uint32_t outputs,
                     double* predictions, cudaStream_t stream);
cudaError_t transform(Objective objective, double* predictions,
                      std::uint32_t rows, std::uint32_t outputs,
                      cudaStream_t stream);
// Writes `blocks` partial sums of weighted per-row objective values. Squared
// error uses half the sum over outputs; classification uses stable log loss.
cudaError_t loss(Objective objective, const double* predictions,
                 const float* targets, const float* weights,
                 std::uint32_t rows, std::uint32_t outputs,
                 double* partial_loss, std::uint32_t blocks,
                 cudaStream_t stream);

} // namespace ghb::gpu
