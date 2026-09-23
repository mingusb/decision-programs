#pragma once
#include "gh/core.cuh"

namespace gh::bench::frozen_count {
// Benchmark-only fixed p15/b48, 16384 bins, u32 input/local and u64 output.
// Accepts exactly 16 Mi or 64 Mi elements; caller validates every ID <16384.
// Arrays/status are disjoint, naturally aligned, live device storage; status
// starts zero. The same exclusive-workspace/completion contract as core applies.
__device__ cudaError_t count(u64 size, Array<const std::byte> input,
    Array<std::byte> output, Workspace workspace, Status* status);
cudaError_t initialize_runtime();
}
