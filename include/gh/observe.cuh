#pragma once
#include "gh/core.cuh"

#ifndef GH_OBSERVE
#define GH_OBSERVE 0
#endif
namespace gh::observe {
inline constexpr bool enabled = GH_OBSERVE != 0;
enum class Stage : u32 { validate, base, gradient, histogram, split, route, loss, export_model };
struct Stamp { u64 ticks{}; Stage stage{}; u32 iteration{}; bool end{}; };
struct Trace { Array<Stamp> records; u32* count{}; };
__device__ cudaError_t enqueue(Trace, Stage, u32 iteration, bool end, Status*);
template<bool Enabled = enabled>
__device__ cudaError_t mark(Trace trace, Stage stage, u32 iteration, bool end, Status* status) {
  if constexpr (Enabled) return enqueue(trace,stage,iteration,end,status);
  return cudaSuccess;
}

inline constexpr u32 warmups = 3, pairs = 15, samples = 2 * (warmups + pairs);
struct Slot { u32 variant, pair; bool warmup; };
__device__ constexpr Slot schedule(u32 ordinal) {
  const u32 pair = ordinal / 2;
  return {(ordinal % 2) ^ (pair % 2), pair < warmups ? pair : pair - warmups, pair < warmups};
}
struct Sample { u64 begin{}, end{}; };
struct Summary { double ratio{}, lower{}, upper{}, log_stddev{}; u64 minimum_ticks{}; };
// Start queues on device-null; end queues after prior completion tails. Start
// the next sample only from a subsequent tail continuation, never a parent loop.
__device__ cudaError_t boundary(Array<Sample>, u32 ordinal, bool end, Status*);
// Launch from the final sample's continuation. Does not alter raw samples.
__device__ cudaError_t summarize(Array<const Sample>, Summary*, Status*);
}
