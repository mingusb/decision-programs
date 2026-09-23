#pragma once
#include "ghb/kernels.cuh"

namespace ghb::gpu {

// Experimental primitives; the trainer's one-tree workspace is unchanged.
enum class DeeperHistogramPolicy : std::uint32_t { global, shared1, shared4, shared8 };

// Returns whether the host shape fits the selected <=48KiB shared-memory
// policy. Global has no shared-memory requirement. No CUDA work is performed.
bool deeper_histogram_supported(DataView data, std::uint32_t node_capacity,
                                DeeperHistogramPolicy policy);

// assignments[batch_count][rows] uses an independent contiguous row assignment
// array for each output. G/H are row-major with derivative_stride entries/row;
// first_output selects their contiguous output range. active[batch_count] is
// device-resident and may change between graph replays. Each count is clamped
// to node_capacity (1..INT32_MAX). Negative/out-of-active-range assignments are skipped.
//
// output[batch_count][node_capacity][total_bins]: overwrite each active range,
// preserve inactive capacity. Missing bin zero and zero derivatives still count
// rows exactly in uint64. FP64 addition order is unspecified. Device bin IDs,
// offsets, finite G/H and buffer extents are caller-validated. No aliasing.
//
// row_chunks is zero for global, 1..256 for a shared policy. Shared policies
// group 1/4/8 outputs per feature/chunk CTA; unsupported shapes reject without
// falling back. This asynchronous operation allocates and synchronizes nothing.
cudaError_t deeper_histogram(DataView data, const std::int32_t* assignments,
                            const double* gradient, const double* hessian,
                            std::uint32_t derivative_stride, std::uint32_t first_output,
                            std::uint32_t batch_count, std::uint32_t node_capacity,
                            const std::uint32_t* active, Stats* output,
                            DeeperHistogramPolicy policy, std::uint32_t row_chunks,
                            cudaStream_t stream);

} // namespace ghb::gpu
