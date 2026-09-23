#pragma once
#include "gh/core.cuh"

namespace gh {
enum class CsvKind : u32 { targets, predictions, multiclass };
// Values are row-major. Weights are permitted for targets only; empty means one.
// written and all descriptors/backing arrays are global and remain alive through
// completion. Output is raw bytes without a trailing NUL. Partial failure is invalid.
__device__ cudaError_t encode_csv(Array<const double> values, u32 rows, u32 outputs,
    CsvKind kind, Array<const double> weights, Array<std::byte> bytes,
    Array<u64> written, Workspace workspace, Status* status);
}
