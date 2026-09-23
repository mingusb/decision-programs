#pragma once

#include "gh/histogram.hpp"

// NVIDIA baselines belong only to benchmark executables. They are not production
// algorithms, selectors, or fallbacks; gh::Config supplies workload/CSV metadata.
namespace ghbench {
enum class Reference { none, cub, nvidia_sample256 };

inline constexpr const char* name(Reference reference) {
  switch (reference) {
    case Reference::none: return "none";
    case Reference::cub: return "cub";
    case Reference::nvidia_sample256: return "nvidia_sample256";
  }
  return "unknown";
}

bool supported(Reference reference, const gh::Config& config);
cudaError_t workspace_bytes(Reference reference, const gh::Config& config, std::size_t& bytes);
cudaError_t histogram(Reference reference, const gh::Config& config,
                      const void* input, void* output, void* workspace,
                      std::size_t bytes, cudaStream_t stream);

namespace detail {
bool sample256_supported(const gh::Config& config);
cudaError_t sample256_workspace_bytes(const gh::Config& config, std::size_t& bytes);
cudaError_t launch_sample256(const gh::Config& config, const void* input, void* output,
                            void* workspace, std::size_t bytes, cudaStream_t stream);
}
}  // namespace ghbench
