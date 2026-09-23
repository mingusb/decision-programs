#pragma once
#include "ghb/kernels.cuh"

namespace ghb::gpu {

// All fields are device-owned during a complete tree build. Host reads them
// only at the completed-tree export boundary. Status is a sticky bitmask.
struct TreeState {
  std::uint32_t active_nodes{}, next_active_nodes{}, node_count{}, status{};
  std::uint32_t old_node_count{}, split_count{};
};
struct TreeParameters {
  std::uint32_t derivative_outputs{}, derivative_output{}, output{}, root_output{};
};
enum TreeStatus : unsigned {
  tree_frontier_overflow = 1, tree_node_overflow = 2,
  tree_invalid_split = 4, tree_nonfinite_leaf = 8
};

// Existing arithmetic kernels with device-owned active count. capacity sizes
// storage/launches; only *active entries are read/written. No host event reads.
cudaError_t histogram_active(DataView, const std::int32_t*, const double*, const double*,
                             std::uint32_t outputs, std::uint32_t output,
                             std::uint32_t capacity, const std::uint32_t* active,
                             Stats*, HistogramPolicy, cudaStream_t, const TreeParameters* selector = nullptr);
cudaError_t find_splits_active(DataView, const Stats*, std::uint32_t capacity,
                               const std::uint32_t* active, SplitConfig, bool force_leaf,
                               Split* candidates, Split* winners, cudaStream_t);

// Optional histogram cache triplet and winner cache pair are independently
// all-or-none. Either group requires selector. Caller guarantees root_output
// is in both requested cache batches; histogram cache/destination have
// total_bins cells and winner destination has one Split. All buffers remain
// live through stream completion; cache and destination must not overlap.
// A winner-only initialization never accesses the histogram buffers.
cudaError_t resident_initialize(std::uint32_t rows, std::int32_t* assignments,
                                Node* nodes, std::int32_t* frontier, TreeState*, cudaStream_t,
                                const Stats* root_cache = nullptr, Stats* root_destination = nullptr,
                                std::uint32_t total_bins = 0, const TreeParameters* selector = nullptr,
                                const Split* root_winner_cache = nullptr, Split* root_winner_destination = nullptr);
// offsets: frontier_capacity uint32 values; block_counts:
// ceil(frontier_capacity/1024) uint32 values. Stable order matches host scan.
cudaError_t resident_materialize(DataView, const Split*, const std::int32_t* frontier,
                                 std::int32_t* next_frontier, std::int32_t* left_map,
                                 std::int32_t* right_map, Node*, TreeState*,
                                 std::uint32_t frontier_capacity, std::uint32_t node_capacity,
                                 std::uint32_t* offsets, std::uint32_t* block_counts,
                                 bool expand_children, double learning_rate, cudaStream_t);
cudaError_t resident_route(DataView, std::int32_t* assignments, const Split*,
                           const std::int32_t* left_map, const std::int32_t* right_map,
                           TreeState*, cudaStream_t);
cudaError_t resident_advance(TreeState*, cudaStream_t);
cudaError_t resident_predict(DataView, const Node*, const TreeState*, std::uint32_t output,
                             std::uint32_t outputs, double* predictions, cudaStream_t,
                             const TreeParameters* selector = nullptr);
cudaError_t finalize_loss(const double* partials, std::uint32_t blocks, const double* weight_sum,
                          std::uint32_t normalized_outputs, double* loss, cudaStream_t);

} // namespace ghb::gpu
