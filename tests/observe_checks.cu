#include "gh/observe.cuh"
#include "check.cuh"
#include <cmath>

namespace gh::test {
namespace {
using namespace gh::observe;
__device__ Stamp stages[2];
__device__ u32 stage_count;
__device__ Status trace_status, full_status, boundary_status, empty_status, sample_status, summary_status[6];
__device__ Sample synthetic[6][samples], measured[samples];
__device__ Sample empty;
__device__ Summary results[6], timing;
__device__ u64 work[64], clock_begin[64], clock_end[64];
__device__ u32 clock_sm[64], disabled_value;
__global__ void work_and_clocks(u32 ordinal) {
  u64 a,b; u32 sm;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(a));
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  u64 value = ordinal + blockIdx.x;
  for (u32 i = 0; i < 256; ++i) value = mix(value);
  work[blockIdx.x] = value;
  __nanosleep(10000);
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(b));
  clock_begin[blockIdx.x] = a; clock_end[blockIdx.x] = b; clock_sm[blockIdx.x] = sm;
}
__global__ void fixture() {
  for (u32 k = 0; k < 6; ++k) {
    u64 cursor = 100;
    for (u32 i = 0; i < samples; ++i) {
      const auto slot = schedule(i);
      u64 duration = slot.variant ? 100 : 200;
      if (k == 1) duration *= 7;
      if (k == 2) duration = slot.variant ? 200 : 100;
      if (k == 3 && !slot.warmup) {
        const bool high = slot.pair % 2 == 0;
        duration = slot.pair == 14 ? 100 : slot.variant ? 100 : high ? 200 : 50;
      }
      synthetic[k][i] = {cursor,cursor + duration};
      cursor += duration + 5;
    }
    if (k == 4) synthetic[k][8].end = synthetic[k][8].begin;
    if (k == 5) synthetic[k][8].begin = synthetic[k][7].end - 1;
  }
}
__global__ void run_sample(u32 ordinal);
__global__ void empty_check() {
  succeeded(empty_status);
  GH_CHECK(empty.end > empty.begin);
  printf("GH_RAW_EMPTY globaltimer-cdp-tail-v1 begin=%llu end=%llu\n",
    static_cast<unsigned long long>(empty.begin),static_cast<unsigned long long>(empty.end));
  run_sample<<<1,1,0,cudaStreamTailLaunch>>>(0);
  submitted(cudaGetLastError());
}
__global__ void empty_sample() {
  submitted(boundary({&empty,1},0,false,&empty_status));
  submitted(finish(&empty_status));
  submitted(boundary({&empty,1},0,true,&empty_status));
  empty_check<<<1,1,0,cudaStreamTailLaunch>>>();
  submitted(cudaGetLastError());
}
__global__ void final_check() {
  succeeded(sample_status);
  GH_CHECK(timing.ratio > 0 && timing.lower <= timing.ratio && timing.upper >= timing.ratio);
  GH_CHECK(timing.minimum_ticks > 0);
  for (u32 i = 0; i < samples; ++i) {
    GH_CHECK(measured[i].end > measured[i].begin);
    if (i) GH_CHECK(measured[i].begin >= measured[i-1].end);
    printf("GH_RAW_SAMPLE globaltimer-cdp-tail-v1 ordinal=%u variant=%u warmup=%u begin=%llu end=%llu\n",
      i,schedule(i).variant,unsigned(schedule(i).warmup),
      static_cast<unsigned long long>(measured[i].begin),static_cast<unsigned long long>(measured[i].end));
  }
  printf("GH_GPU_ACTIVITY observe samples=36 checks=pass; timing qualification, no ranking\n");
}
__global__ void checked_sample(u32 ordinal) {
  succeeded(trace_status);
  bool other_sm = false;
  for (u32 i = 0; i < 64; ++i) {
    GH_CHECK(clock_end[i] > clock_begin[i]);
    GH_CHECK(clock_begin[i] >= measured[ordinal].begin && clock_end[i] <= measured[ordinal].end);
    other_sm |= clock_sm[i] != clock_sm[0];
    u64 expected = ordinal + i;
    for (u32 j = 0; j < 256; ++j) expected = mix(expected);
    GH_CHECK(work[i] == expected);
  }
  GH_CHECK(other_sm);
  if (ordinal + 1 < samples) run_sample<<<1,1,0,cudaStreamTailLaunch>>>(ordinal+1);
  else {
    submitted(summarize({measured,samples},&timing,&sample_status));
    final_check<<<1,1,0,cudaStreamTailLaunch>>>();
  }
  submitted(cudaGetLastError());
}
__global__ void run_sample(u32 ordinal) {
  trace_status = {};
  submitted(boundary({measured,samples},ordinal,false,&sample_status));
  work_and_clocks<<<64,1>>>(ordinal);
  submitted(cudaGetLastError());
  submitted(finish(&trace_status));
  submitted(boundary({measured,samples},ordinal,true,&sample_status));
  checked_sample<<<1,1,0,cudaStreamTailLaunch>>>(ordinal);
  submitted(cudaGetLastError());
}
__global__ void summary_check() {
  for (u32 i = 0; i < 4; ++i) succeeded(summary_status[i]);
  for (u32 i = 4; i < 6; ++i) GH_CHECK(summary_status[i].done && summary_status[i].errors == numeric);
  GH_CHECK(fabs(results[0].ratio - 2) < 1e-12);
  GH_CHECK(fabs(results[0].upper - results[0].lower) < 1e-12);
  GH_CHECK(fabs(results[1].ratio - results[0].ratio) < 1e-12);
  GH_CHECK(results[1].minimum_ticks == 7 * results[0].minimum_ticks);
  GH_CHECK(fabs(results[2].ratio - 0.5) < 1e-12);
  GH_CHECK(fabs(results[3].ratio - 1) < 1e-12);
  GH_CHECK(results[3].lower < 1 && results[3].upper > 1);
  GH_CHECK(fabs(results[3].log_stddev - log(2.0)) < 1e-12);
  empty_sample<<<1,1,0,cudaStreamTailLaunch>>>();
  submitted(cudaGetLastError());
}
__global__ void trace_check() {
  succeeded(trace_status);
  GH_CHECK(full_status.done && full_status.errors == capacity);
  GH_CHECK(boundary_status.done && boundary_status.errors == capacity);
  GH_CHECK(disabled_value == 73 && stage_count == 2);
  GH_CHECK(stages[0].stage == Stage::gradient && stages[1].stage == Stage::gradient);
  GH_CHECK(!stages[0].end && stages[1].end && stages[0].iteration == 9 && stages[1].iteration == 9);
  GH_CHECK(stages[1].ticks > stages[0].ticks);
  fixture<<<1,1>>>();
  for (u32 k = 0; k < 6; ++k) submitted(summarize({synthetic[k],samples},results+k,summary_status+k));
  summary_check<<<1,1,0,cudaStreamTailLaunch>>>();
  submitted(cudaGetLastError());
}
}
// Deliberately public symbol for offline PTX/SASS erasure inspection; exercised.
__global__ void disabled_probe() {
  mark<false>({},Stage::gradient,0,false,nullptr);
  disabled_value = 73;
}
__global__ void run() {
  u32 seen[2]{};
  for (u32 i = 0; i < samples; ++i) {
    const auto slot = schedule(i);
    GH_CHECK(slot.variant < 2 && slot.pair < (slot.warmup ? warmups : pairs));
    if (!slot.warmup) ++seen[slot.variant];
    if (i % 2) GH_CHECK(slot.variant != schedule(i-1).variant);
    if (i >= 2) GH_CHECK(slot.variant != schedule(i-2).variant);
  }
  GH_CHECK(seen[0] == 15 && seen[1] == 15);
  disabled_probe<<<1,1>>>();
  submitted(mark<true>({{stages,2},&stage_count},Stage::gradient,9,false,&trace_status));
  work_and_clocks<<<64,1>>>(0);
  submitted(mark<true>({{stages,2},&stage_count},Stage::gradient,9,true,&trace_status));
  submitted(mark<true>({{stages,2},&stage_count},Stage::loss,9,true,&full_status));
  submitted(boundary({},0,false,&boundary_status));
  submitted(finish(&boundary_status));
  submitted(finish(&trace_status)); submitted(finish(&full_status));
  trace_check<<<1,1,0,cudaStreamTailLaunch>>>();
  submitted(cudaGetLastError());
}
}
