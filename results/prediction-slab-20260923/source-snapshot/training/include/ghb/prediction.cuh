#pragma once

#include "ghb/kernels.cuh"

#include <cstdint>

namespace ghb::gpu {

struct PredictionTree {
  std::uint64_t node_begin{};
  std::uint32_t node_count{}, reserved{};
};

// Trees are stably grouped by output. output_offsets contains outputs+1
// nondecreasing indices, starts at zero and ends at tree_count. Every descriptor
// names a valid tree within nodes; child indices remain relative to that tree.
// This is a validated-model entry point: device metadata contents are the
// caller's responsibility. Every output preserves the original FP64 add order.
// Empty forests are valid; nodes/trees may then be null. Base scores, offsets
// and the output are always required. No allocation or synchronization occurs.
// All buffers must remain alive until stream completion; output must not alias
// inputs. Scratch is not used. Capture on a caller-owned stream is supported.
cudaError_t predict_forest(DataView data, const Node* nodes,
                           std::uint64_t node_count, const PredictionTree* trees,
                           std::uint64_t tree_count,
                           const std::uint64_t* output_offsets,
                           const double* base, std::uint32_t outputs,
                           double* predictions, cudaStream_t stream);

} // namespace ghb::gpu
