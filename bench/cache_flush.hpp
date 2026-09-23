#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>

// The caller supplies a separate buffer >= max(64 MiB, 8 * device L2 size).
// Initialize once before timing; evict_l2 then reads/writes every word on the
// benchmark stream. This is capacity eviction, not an architectural invalidate.
cudaError_t initialize_eviction_buffer(void* buffer, std::size_t bytes,
                                      int multiprocessors, cudaStream_t stream);
cudaError_t evict_l2(void* buffer, std::size_t bytes,
                     int multiprocessors, cudaStream_t stream);
