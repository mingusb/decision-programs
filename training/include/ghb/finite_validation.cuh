#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace ghb::gpu {

// Asynchronous effects: clear status, then reduce final-output nonfiniteness.
// Enqueue AFTER the requested objective transform on the same stream. Input
// bits are unchanged. Successful completion leaves status 0 iff all values are
// finite, otherwise 1. Submission success does not imply checked completion.
//
// No allocation, download, wait or device query occurs. Null status, null input
// for a nonzero extent, byte overflow and extents above the SM86 full-cover grid
// bound (256 * 0x7fffffff elements) reject before effects. Empty input permits
// null values and only clears status; status remains required.
//
// Caller owns sufficiently sized/aligned, nonoverlapping device allocations and
// orders all producers on stream. Both allocations must be on its device and
// stay alive until completion, including after a submission error. Concurrent
// calls require distinct status words and no writer to any input. Inspect CUDA
// completion and then status before exposing a successfully validated output.
cudaError_t validate_finite(const double* values, std::uint64_t elements,
                            std::uint32_t* status, cudaStream_t stream);

} // namespace ghb::gpu
