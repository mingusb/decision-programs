#pragma once
#include "gh/core.cuh"
#include <cassert>
#include <cstdio>

#define GH_CHECK(condition) do { if (!(condition)) { \
  printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); assert(false); \
} } while (false)

namespace gh::test {
__global__ void run();
__device__ inline void submitted(cudaError_t result) {
  if (result != cudaSuccess) printf("CUDA submission error %d\n", static_cast<int>(result));
  GH_CHECK(result == cudaSuccess);
}
__device__ inline void succeeded(const Status& status) {
  if (status.done != 1 || status.errors) printf("Status done=%u errors=%u required_bytes=%llu\n", status.done, status.errors, static_cast<unsigned long long>(status.required_bytes));
  GH_CHECK(status.done == 1); GH_CHECK(status.errors == 0);
}
__device__ inline u64 mix(u64 value) {
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}
__device__ inline u64 timestamp() {
  u64 result; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(result)); return result;
}
}
