#pragma once
#include <cstdint>

namespace ghb::gpu {
// Device selector, uploaded and validated by the host before each output tile.
// Compact independent derivatives use output_count as their actual row stride;
// multiclass keeps its complete frozen pre-round derivative row stride.
struct OutputBatch {
  std::uint32_t output_begin{}, output_count{}, derivative_begin{}, derivative_stride{};
};
} // namespace ghb::gpu
