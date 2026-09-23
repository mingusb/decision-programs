#pragma once
#include "ghb/resident.cuh"

namespace ghb::gpu {

// Direct-count counterpart of find_splits. Independent root outputs can be
// presented as nodes with histograms[nodes][total_bins], candidates[nodes]
// [columns] and winners[nodes]. No device active-count buffer is needed.
// Arithmetic, feature fallback and scratch contracts match the active form.
cudaError_t find_splits_warp(DataView, const Stats*, std::uint32_t nodes,
                             SplitConfig, bool force_leaf, Split* candidates,
                             Split* winners, cudaStream_t);

// Opt-in split search using one full warp per feature/node when both the
// maximum feature histogram and feature count fit in 32 lanes. Larger shapes
// use our existing split implementation. Contracts and scratch sizes match
// find_splits_active; all work is asynchronous and allocation-free.
cudaError_t find_splits_warp_active(DataView, const Stats*, std::uint32_t capacity,
                                    const std::uint32_t* active, SplitConfig,
                                    bool force_leaf, Split* candidates,
                                    Split* winners, cudaStream_t);

// Independent output-major frontiers. Histogram, candidate and winner layouts
// are [batch_capacity][node_capacity][total_bins], [B][C][columns], and [B][C].
// active[B] is device-owned and clamped to C. Optional device batch_count is
// clamped to B; null selects all B. Inactive outputs/nodes remain untouched.
// Buffer extents and nonaliasing are caller contracts. Launches are asynchronous,
// allocation-free, and graph-replay masks may change without reconstruction.
// Both forms retain their existing per-feature arithmetic and tie rules; warp
// dispatch retains the owned block fallback for bins/features greater than 32.
cudaError_t find_splits_batched_active(DataView, const Stats*, std::uint32_t batch_capacity,
                                       std::uint32_t node_capacity, const std::uint32_t* active,
                                       SplitConfig, bool force_leaf, Split* candidates,
                                       Split* winners, cudaStream_t,
                                       const std::uint32_t* batch_count = nullptr);
cudaError_t find_splits_warp_batched_active(DataView, const Stats*, std::uint32_t batch_capacity,
                                            std::uint32_t node_capacity, const std::uint32_t* active,
                                            SplitConfig, bool force_leaf, Split* candidates,
                                            Split* winners, cudaStream_t,
                                            const std::uint32_t* batch_count = nullptr);

} // namespace ghb::gpu
