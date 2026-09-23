#pragma once
#include "gh/core.cuh"

namespace gh {
// Headerless row-major little-endian binary64, exactly rows*outputs*8 bytes.
// outputs>0; zero rows are valid. All successful values are finite. Byte buffers
// may be unaligned; double buffers require natural alignment. Regions are disjoint.
// Decode's source extent is exact; encode's destination size is capacity. Caller
// zeroes Status and waits for its completion tail before consuming/reusing storage.
__device__ cudaError_t decode_predictions(Array<const std::byte> bytes, u32 rows,
    u32 outputs, Array<double> values, Status* status);
__device__ cudaError_t encode_predictions(Array<const double> values, u32 rows,
    u32 outputs, Array<std::byte> bytes, Status* status);
}
