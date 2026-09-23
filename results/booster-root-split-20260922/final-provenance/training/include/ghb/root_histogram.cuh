#pragma once
#include "ghb/kernels.cuh"

namespace ghb::gpu {

// All rows belong to the root. Input derivatives are row-major with explicit
// stride; output is [batch_count][data.total_bins]. Missing bin zero and rows
// with zero derivatives still contribute one exact uint64 count per feature.
// Clears every output cell, then accumulates FP64 values in unspecified atomic
// order. No allocation or synchronization; buffers must survive stream work.
// Device bin IDs/offsets and finite derivatives are validated by the caller.
cudaError_t root_histogram(DataView data, const double* gradient,
                           const double* hessian, std::uint32_t derivative_stride,
                           std::uint32_t first_output, std::uint32_t batch_count,
                           Stats* output, cudaStream_t stream);

} // namespace ghb::gpu
