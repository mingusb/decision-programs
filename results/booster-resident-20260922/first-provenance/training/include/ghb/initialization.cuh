#pragma once
#include "ghb/booster.hpp"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ghb::gpu {
enum InitializationStatus : unsigned {
  invalid_weight = 1, invalid_target = 2, no_positive_weight = 4,
  invalid_base_score = 8
};

// Scratch contains chunks * (outputs + 1) doubles, including weight partials.
// targets is Nxoutputs for independent objectives, N for multiclass.
// All arithmetic and dense target/weight validation run on the supplied stream.
// Returns launch status; device status must be checked at the setup boundary.
// No allocation or synchronization; graph-capture compatible.
cudaError_t initialize_training(Objective objective, const float* targets,
                                const float* weights, std::uint32_t rows,
                                std::uint32_t outputs, double* base_scores,
                                double* weight_sum, double* partials,
                                std::uint32_t chunks, unsigned* status,
                                cudaStream_t stream);
} // namespace ghb::gpu
