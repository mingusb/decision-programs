// Deliberately faulty diagnostic canary; NEVER part of a production build.
// Modes independently verify global/shared initcheck and compiler memcheck.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
__global__ void canary(int* output, const int* input, int mode) {
  __shared__ int local[32];
  const int i = threadIdx.x;
  if (mode == 0) output[i] = input[i];
  if (mode == 1) output[i] = local[i];
  if (mode == 2) output[i + 32] = i;
  if (mode == 3) output[i] = i;
}
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  int mode = !std::strcmp(argv[1],"global") ? 0 : !std::strcmp(argv[1],"shared") ? 1 : !std::strcmp(argv[1],"bounds") ? 2 : 3;
  int *input{}, *output{};
  if(cudaMalloc(&input,128)!=cudaSuccess || cudaMalloc(&output,128)!=cudaSuccess) return 3;
  canary<<<1,32>>>(output,input,mode);
  auto error=cudaDeviceSynchronize();
  cudaFree(input);cudaFree(output);
  if(error!=cudaSuccess) {std::fprintf(stderr,"%s\n",cudaGetErrorString(error));return 4;}
  return 0;
}
