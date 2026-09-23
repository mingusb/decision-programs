#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>

// This interface deliberately contains no types from either archived library.
// Both adapters receive the same values and retain their own prepared Config.
struct PairedConfig {
  std::size_t n;
  unsigned bins;
  int tuning;
  int blocks;
  bool graph;
  bool kernel_clear;
};

struct BackendApi {
  const char* name;
  cudaError_t (*create)(const PairedConfig*, void**);
  cudaError_t (*launch)(const void*, const void*, void*, cudaStream_t);
  void (*destroy)(void*);
};

extern "C" const BackendApi* paired_old_api();
extern "C" const BackendApi* paired_new_api();
