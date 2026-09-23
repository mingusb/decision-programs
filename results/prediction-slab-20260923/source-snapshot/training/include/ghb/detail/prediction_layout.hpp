#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ghb::detail {

inline constexpr std::size_t prediction_slab_alignment = 256;

namespace prediction_layout_detail {
inline std::size_t product(std::size_t count, std::size_t width, const char* name) {
  if (width && count > std::numeric_limits<std::size_t>::max() / width)
    throw std::invalid_argument(std::string(name) + " size overflow");
  return count * width;
}
inline std::size_t sum(std::size_t a, std::size_t b, const char* name) {
  if (a > std::numeric_limits<std::size_t>::max() - b)
    throw std::invalid_argument(std::string(name) + " size overflow");
  return a + b;
}
} // namespace prediction_layout_detail

struct PredictionSlabRegion { std::size_t offset{}, bytes{}; };
struct PredictionSlabLayout {
  PredictionSlabRegion base, nodes, descriptors, offsets;
  std::size_t output_offset_count{};
  std::size_t device_bytes{}, host_blocks{}, host_bytes{};
  std::size_t prediction_bytes{}, reserved_bytes{};

  // The returned remainder is only the quantizer's payload allowance. Neither
  // a free-memory snapshot nor this arithmetic promises allocation success.
  std::size_t quantizer_budget(std::size_t free_bytes) const {
    if (reserved_bytes >= free_bytes)
      throw std::invalid_argument("inference payload exceeds available GPU memory");
    return free_bytes - reserved_bytes;
  }
};

// Pure size arithmetic: no CUDA headers, calls, allocations or model traversal.
// The caller still validates model semantics and supplies its actual ABI sizes.
// Synthetic extreme counts can therefore test overflow without huge models.
template<std::size_t NodeBytes, std::size_t DescriptorBytes>
inline PredictionSlabLayout make_prediction_slab_layout(std::size_t outputs,
    std::size_t node_count, std::size_t tree_count, std::size_t prediction_elements) {
  static_assert(NodeBytes && DescriptorBytes);
  using prediction_layout_detail::product;
  using prediction_layout_detail::sum;
  PredictionSlabLayout result;
  result.output_offset_count = sum(outputs, 1, "inference output offsets");
  auto region = [&](std::size_t count, std::size_t width) {
    const auto bytes = product(count, width, "inference slab region");
    if (!bytes) return PredictionSlabRegion{};
    const auto offset = sum(result.device_bytes,
        (prediction_slab_alignment - result.device_bytes % prediction_slab_alignment) % prediction_slab_alignment,
        "inference slab alignment");
    result.device_bytes = sum(offset, bytes, "inference slab extent");
    return PredictionSlabRegion{offset, bytes};
  };
  result.base = region(outputs, sizeof(double));
  result.nodes = region(node_count, NodeBytes);
  result.descriptors = region(tree_count, DescriptorBytes);
  result.offsets = region(result.output_offset_count, sizeof(std::uint64_t));
  result.prediction_bytes = product(prediction_elements, sizeof(double), "inference predictions");
  result.reserved_bytes = sum(result.prediction_bytes, result.device_bytes, "inference slab workspace");
  result.host_blocks = sum(result.device_bytes / prediction_slab_alignment,
      result.device_bytes % prediction_slab_alignment != 0, "inference host slab blocks");
  result.host_bytes = product(result.host_blocks, prediction_slab_alignment, "inference host slab extent");
  return result;
}

} // namespace ghb::detail
