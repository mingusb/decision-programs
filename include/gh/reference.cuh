#pragma once
#include "gh/metrics.cuh"

namespace gh {
// Synchronous GPU conversion of one complete bounded JSON number token. Exact
// RN-even, including signed zero/subnormals/overflow; output unchanged on false.
__device__ bool parse_decimal(Array<const char>, double*);
// Completed real-data metrics on frozen predictions; all report backing arrays
// disjoint/global. Imports the frozen schema and aligns computed legacy keys.
// Caller zeroes Status and waits for tail completion before consuming results.
__device__ cudaError_t decode_metric_reference(Array<const std::byte>, const MetricReport* computed,
    MetricReport* reference, MetricReport* aligned, Status*);
}
