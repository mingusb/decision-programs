#include "gh/observe.cuh"
#include <cmath>

namespace gh::observe {
namespace {
__device__ u64 tick() { u64 value; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value)); return value; }
__global__ void append(Trace trace, Stage stage, u32 iteration, bool end, Status* status) {
  if (!trace.count || !contains(trace.records,trace.records.size)) { fail(status,extent); return; }
  const u32 slot = *trace.count;
  if (slot >= trace.records.size || slot == UINT32_MAX) { fail(status,capacity); return; }
  trace.records.data[slot] = {tick(),stage,iteration,end};
  *trace.count = slot + 1;
}
__global__ void stamp(Array<Sample> raw, u32 ordinal, bool end, Status* status) {
  if (ordinal >= samples || !contains(raw,u64(ordinal)+1)) { fail(status,capacity); return; }
  if (end) raw.data[ordinal].end = tick(); else raw.data[ordinal].begin = tick();
}
__global__ void paired(Array<const Sample> raw, Summary* output, Status* status) {
  if (!output || !contains(raw,samples)) { fail(status,capacity); return; }
  u64 minimum = UINT64_MAX;
  for (u32 i = 0; i < samples; ++i) {
    const auto value = raw.data[i];
    if (value.end <= value.begin || (i && value.begin < raw.data[i-1].end)) { fail(status,numeric); return; }
    minimum = min(minimum,value.end-value.begin);
  }
  double logs[pairs], mean = 0;
  for (u32 pair = 0; pair < pairs; ++pair) {
    const u32 first = 2 * (warmups + pair);
    const u32 a = first + schedule(first).variant, b = first + 1 - schedule(first).variant;
    const double x = double(raw.data[a].end - raw.data[a].begin);
    const double y = double(raw.data[b].end - raw.data[b].begin);
    logs[pair] = log(x/y);
    mean += logs[pair] / pairs;
  }
  double variance = 0;
  for (const double value : logs) variance += (value-mean)*(value-mean) / (pairs-1);
  const double sd = sqrt(variance), margin = 2.1447866879169273 * sd / sqrt(double(pairs));
  *output = {exp(mean),exp(mean-margin),exp(mean+margin),sd,minimum};
}
__device__ cudaError_t checked(Status* status) {
  const auto result = cudaGetLastError();
  if (result != cudaSuccess) fail(status,runtime);
  return result;
}
}
__device__ cudaError_t enqueue(Trace trace, Stage stage, u32 iteration, bool end, Status* status) {
  if (!status) return cudaErrorInvalidValue;
  append<<<1,1>>>(trace,stage,iteration,end,status);
  return checked(status);
}
__device__ cudaError_t boundary(Array<Sample> raw, u32 ordinal, bool end, Status* status) {
  if (!status) return cudaErrorInvalidValue;
  stamp<<<1,1,0,end ? cudaStreamTailLaunch : nullptr>>>(raw,ordinal,end,status);
  return checked(status);
}
__device__ cudaError_t summarize(Array<const Sample> raw, Summary* output, Status* status) {
  if (!status) return cudaErrorInvalidValue;
  paired<<<1,1>>>(raw,output,status);
  const auto launched = checked(status), completed = finish(status);
  return launched != cudaSuccess ? launched : completed;
}
}
