#include "gh/core.cuh"
namespace gh {
__global__ void complete(Status* status) { if (!threadIdx.x && !blockIdx.x) status->done = 1; }
__device__ cudaError_t finish(Status* status) {
  const auto prior = cudaGetLastError();
  if (prior != cudaSuccess) fail(status, runtime);
  complete<<<1, 1, 0, cudaStreamTailLaunch>>>(status);
  const auto next = cudaGetLastError();
  if (next != cudaSuccess) fail(status, runtime);
  return prior != cudaSuccess ? prior : next;
}
}
