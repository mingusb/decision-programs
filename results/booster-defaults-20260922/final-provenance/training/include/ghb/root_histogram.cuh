#pragma once
#include "ghb/kernels.cuh"

namespace ghb::gpu {

enum class RootCountKernel : std::uint32_t { global, shared };

// Counts every retained row exactly once per feature, including missing bins
// and zero-weight rows. Counts are immutable across output/round consumers only
// while bins and participating rows stay unchanged. No allocation or host wait.
// Shared setup uses at most 48 KiB; unsupported explicit requests reject.
bool root_counts_shared_supported(DataView data);
cudaError_t root_counts(DataView data, unsigned long long* counts,
                        RootCountKernel policy, cudaStream_t stream);

// All rows belong to the root. Input derivatives are row-major with explicit
// stride; output is [batch_count][data.total_bins]. Missing bin zero and rows
// with zero derivatives still contribute one exact uint64 count per feature.
// Clears every output cell, then accumulates FP64 values in unspecified atomic
// order. No allocation or synchronization; buffers must survive stream work.
// Device bin IDs/offsets and finite derivatives are validated by the caller.
// Optional immutable cached_counts[total_bins] seeds counts instead of repeating
// their atomic accumulation. It must describe this exact DataView and must not
// overlap output or inputs. FP64 accumulation order remains unspecified.
cudaError_t root_histogram(DataView data, const double* gradient,
                           const double* hessian, std::uint32_t derivative_stride,
                           std::uint32_t first_output, std::uint32_t batch_count,
                           Stats* output, cudaStream_t stream,
                           const unsigned long long* cached_counts = nullptr);

} // namespace ghb::gpu
