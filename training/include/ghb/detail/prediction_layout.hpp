#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ghb::detail {

inline constexpr std::size_t prediction_slab_alignment = 256;
struct PredictionSlabRegion { std::size_t offset{}, bytes{}; };

namespace prediction_layout_detail {
constexpr std::size_t product(std::size_t count, std::size_t width, const char* name) {
  if (width && count > std::numeric_limits<std::size_t>::max() / width)
    throw std::invalid_argument(std::string(name) + " size overflow");
  return count * width;
}
constexpr std::size_t sum(std::size_t a, std::size_t b, const char* name) {
  if (a > std::numeric_limits<std::size_t>::max() - b)
    throw std::invalid_argument(std::string(name) + " size overflow");
  return a + b;
}
struct Placement { PredictionSlabRegion region; std::size_t end; };
constexpr Placement place(std::size_t cursor, std::size_t count, std::size_t width) {
  const auto bytes = product(count, width, "inference slab region");
  if (!bytes) return {{}, cursor};
  const auto offset = sum(cursor,
      (prediction_slab_alignment - cursor % prediction_slab_alignment) % prediction_slab_alignment,
      "inference slab alignment");
  return {{offset, bytes}, sum(offset, bytes, "inference slab extent")};
}
} // namespace prediction_layout_detail

struct PredictionSlabLayout {
  PredictionSlabRegion base, nodes, descriptors, offsets;
  std::size_t output_offset_count{};
  std::size_t device_bytes{}, host_blocks{}, host_bytes{};
  std::size_t prediction_bytes{}, reserved_bytes{};

  // The returned remainder is only the quantizer's payload allowance. Neither
  // a free-memory snapshot nor this arithmetic promises allocation success.
  constexpr std::size_t quantizer_budget(std::size_t free_bytes) const {
    if (reserved_bytes >= free_bytes)
      throw std::invalid_argument("inference payload exceeds available GPU memory");
    return free_bytes - reserved_bytes;
  }
};

// Pure size arithmetic: no CUDA headers, calls, allocations or model traversal.
// The caller still validates model semantics and supplies its actual ABI sizes.
// Synthetic extreme counts can therefore test overflow without huge models.
template<std::size_t NodeBytes, std::size_t DescriptorBytes>
constexpr PredictionSlabLayout make_prediction_slab_layout(std::size_t outputs,
    std::size_t node_count, std::size_t tree_count, std::size_t prediction_elements) {
  static_assert(NodeBytes && DescriptorBytes);
  using prediction_layout_detail::product;
  using prediction_layout_detail::sum;
  using prediction_layout_detail::place;
  const auto offset_count = sum(outputs, 1, "inference output offsets");
  const auto base = place(0, outputs, sizeof(double));
  const auto nodes = place(base.end, node_count, NodeBytes);
  const auto descriptors = place(nodes.end, tree_count, DescriptorBytes);
  const auto offsets = place(descriptors.end, offset_count, sizeof(std::uint64_t));
  const auto prediction_bytes = product(prediction_elements, sizeof(double), "inference predictions");
  const auto reserved_bytes = sum(prediction_bytes, offsets.end, "inference slab workspace");
  const auto host_blocks = sum(offsets.end / prediction_slab_alignment,
      offsets.end % prediction_slab_alignment != 0, "inference host slab blocks");
  const auto host_bytes = product(host_blocks, prediction_slab_alignment, "inference host slab extent");
  return {.base = base.region, .nodes = nodes.region, .descriptors = descriptors.region,
          .offsets = offsets.region, .output_offset_count = offset_count,
          .device_bytes = offsets.end, .host_blocks = host_blocks, .host_bytes = host_bytes,
          .prediction_bytes = prediction_bytes, .reserved_bytes = reserved_bytes};
}

} // namespace ghb::detail
