#pragma once
#include "gh/types.cuh"

namespace gh {
struct DatasetRecord { Dataset data; Objective objective{}; u32 classes{}; };

// One GPU coordinator thread; caller-zeroed status and global backing storage
// remain live until finish's tail completion. Array sizes are capacities.
__device__ cudaError_t fit_schema(Dataset, Array<const FeatureType>, u32 max_bins,
    Schema*, Array<std::uint16_t> bins, Workspace, Status*,
    RadixPolicy = RadixPolicy::radix8);
__device__ cudaError_t encode(Dataset, const Schema*, Array<std::uint16_t>, Status*);
__device__ cudaError_t decode_dataset(Array<const std::byte>, Array<float> values,
    Array<float> targets, DatasetRecord*, Status*);
__device__ cudaError_t encode_dataset(const DatasetRecord*, Array<std::byte>, Status*);
}
