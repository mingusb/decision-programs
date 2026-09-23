#include "backend.hpp"

#include <gh/histogram.hpp>
#include <new>

#ifndef GH_BACKEND_ENTRY
#error "Compile this adapter with a unique GH_BACKEND_ENTRY."
#endif
#ifndef GH_BACKEND_NAME
#error "Compile this adapter with a GH_BACKEND_NAME."
#endif

namespace {

cudaError_t create(const PairedConfig* requested, void** handle) {
  if (!handle) return cudaErrorInvalidValue;
  *handle = nullptr;
  if (!requested) return cudaErrorInvalidValue;

  gh::Config config;
  config.algorithm = gh::Algorithm::shared_atomic;
  config.input_type = gh::InputType::u32;
  config.counter_type = gh::CounterType::u32;
  config.local_counter = gh::LocalCounter::native;
  config.size = requested->n;
  config.bins = requested->bins;
  config.tuning = requested->tuning;
  config.blocks = requested->blocks;
  config.launch = requested->graph ? gh::LaunchMode::graph : gh::LaunchMode::stream;
  config.output_clear = requested->kernel_clear ? gh::OutputClear::kernel : gh::OutputClear::runtime;
  config.cache = gh::CacheMode::warm;
  if (!gh::supported(config)) return cudaErrorInvalidValue;
  auto status = gh::prepare(static_cast<const gh::Config&>(config));
  if (status != cudaSuccess) return status;
  std::size_t bytes = 0;
  status = gh::workspace_bytes(config, bytes);
  if (status != cudaSuccess) return status;
  if (bytes != 0) return cudaErrorNotSupported;
  try {
    *handle = new gh::Config(config);
  } catch (const std::bad_alloc&) {
    return cudaErrorMemoryAllocation;
  }
  return cudaSuccess;
}

cudaError_t launch(const void* handle, const void* input, void* output, cudaStream_t stream) {
  if (!handle) return cudaErrorInvalidValue;
  return gh::histogram(*static_cast<const gh::Config*>(handle), input, output,
                       nullptr, 0, stream);
}

void destroy(void* handle) {
  delete static_cast<gh::Config*>(handle);
}

const BackendApi api{GH_BACKEND_NAME, create, launch, destroy};

}  // namespace

extern "C" const BackendApi* GH_BACKEND_ENTRY() {
  return &api;
}
