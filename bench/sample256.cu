#include "references.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <limits>

// The two kernels and their policy constants are unchanged NVIDIA sample code.
// Only the sample's allocating, process-global host interface is replaced below.
#include "../third_party/nvidia_histogram/histogram256_kernels.cuh"

namespace ghbench::detail {
using gh::Config;
using gh::CounterType;
using gh::InputType;
using gh::LocalCounter;
namespace {
constexpr unsigned partial_count = 240;  // Original PARTIAL_HISTOGRAM256_COUNT.
constexpr std::size_t partial_bytes = partial_count * HISTOGRAM256_BIN_COUNT * sizeof(uint);

bool aligned_uint(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignof(uint) == 0;
}
}  // namespace

bool sample256_supported(const Config& config) {
  // The original sample histograms bytes by loading four at a time as uint.
  // Accepting u32 bin IDs would count their constituent bytes and change semantics.
  return config.input_type == InputType::u8 && config.counter_type == CounterType::u32 &&
         config.local_counter == LocalCounter::native && config.bins == HISTOGRAM256_BIN_COUNT &&
         config.size <= std::numeric_limits<uint>::max() && config.size % sizeof(uint) == 0;
}

cudaError_t sample256_workspace_bytes(const Config& config, std::size_t& bytes) {
  bytes = 0;
  if (!sample256_supported(config)) return cudaErrorInvalidValue;
  if (config.size != 0) bytes = partial_bytes;
  return cudaSuccess;
}

cudaError_t launch_sample256(const Config& config, const void* input, void* output,
                            void* workspace, std::size_t bytes, cudaStream_t stream) {
  if (!sample256_supported(config) || output == nullptr || !aligned_uint(output))
    return cudaErrorInvalidValue;
  if (config.size == 0)
    return cudaMemsetAsync(output, 0, HISTOGRAM256_BIN_COUNT * sizeof(uint), stream);
  if (input == nullptr || workspace == nullptr || bytes < partial_bytes ||
      !aligned_uint(input) || !aligned_uint(workspace))
    return cudaErrorInvalidValue;

  // This includes all initialization and both reductions: kernel one clears its
  // shared histograms and overwrites every partial; kernel two overwrites output.
  // The original fixed launch policy intentionally ignores Config::tuning/blocks.
  histogram256Kernel<<<partial_count, HISTOGRAM256_THREADBLOCK_SIZE, 0, stream>>>(
      static_cast<uint*>(workspace), const_cast<uint*>(static_cast<const uint*>(input)),
      static_cast<uint>(config.size / sizeof(uint)));
  auto status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  mergeHistogram256Kernel<<<HISTOGRAM256_BIN_COUNT, MERGE_THREADBLOCK_SIZE, 0, stream>>>(
      static_cast<uint*>(output), static_cast<uint*>(workspace), partial_count);
  return cudaGetLastError();
}

}  // namespace ghbench::detail
