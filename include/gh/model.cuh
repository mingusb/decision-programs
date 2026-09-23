#pragma once
#include "gh/types.cuh"

namespace gh {
// GPU submission effects; all pointers must name live disjoint device storage.
// Caller clears Status and observes done/errors only after tail completion.
__device__ cudaError_t validate_model(const Model*, Workspace, Status*);
// Model must have passed validation and remain unchanged. Bins are feature-major;
// output is row-major. Zero rows are a no-op; final-value checks follow transform.
__device__ cudaError_t predict(const Model*, Array<const std::uint16_t> bins,
                             u32 rows, Array<double> output, bool raw, Status*);
// Model array capacities are supplied by caller. Decode writes active extents
// and stable output grouping, then validates before successful completion.
__device__ cudaError_t decode_model(Array<const std::byte>, Model*, Workspace, Status*);
// Previously validated unchanged Model. written has one element; no host codec.
__device__ cudaError_t encode_model(const Model*, Array<std::byte>, Array<u64> written,
                                  Workspace, Status*);
}
