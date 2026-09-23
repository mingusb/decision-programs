#pragma once

#include "ghb/kernels.cuh"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ghb {

// Owns the feature-major device bins, offsets and feature types in view.
// Preparation is complete on return. The caller may then discard Dataset;
// subsequent asynchronous users must finish before destroying this object.
// Bytes count owned device allocation payload (including alignment), excluding
// CUDA bookkeeping and host input/model metadata. peak_bytes includes scratch.
class QuantizedData {
public:
  gpu::DataView view{};
  std::vector<Feature> features;
  std::size_t resident_bytes{}, peak_bytes{};

  QuantizedData() noexcept = default;
  ~QuantizedData();
  QuantizedData(const QuantizedData&) = delete;
  QuantizedData& operator=(const QuantizedData&) = delete;
  QuantizedData(QuantizedData&& other) noexcept;
  QuantizedData& operator=(QuantizedData&& other) noexcept;

private:
  void* allocation_{};
  friend struct QuantizerAccess;
};

// Exact quantiles of distinct finite values, complete sorted categories,
// missing bin zero, and canonical positive zero. memory_limit is a strict
// bound on this operation's simultaneously owned device payload, not a query
// of free memory and not a bound including other caller-owned allocations.
// CUDA work uses stream; metadata/status export makes these synchronous host
// operations. Stream capture is rejected. Invalid inputs/budgets throw
// std::invalid_argument; CUDA failures throw std::runtime_error.
QuantizedData fit_quantize(const Dataset& data, std::uint32_t max_bins,
                          std::size_t memory_limit, cudaStream_t stream,
                          QuantizePolicy policy = QuantizePolicy::radix8);
QuantizedData encode_quantize(const Dataset& data,
                             const std::vector<Feature>& features,
                             std::size_t memory_limit, cudaStream_t stream);
// Explicit inference scheduling policy. Both modes complete preparation before
// returning; final_status checks cumulative input status after all tiles, so
// invalid input can take longer to reject. Unknown policies are rejected.
QuantizedData encode_quantize(const Dataset& data,
                             const std::vector<Feature>& features,
                             std::size_t memory_limit, cudaStream_t stream,
                             EncodingPolicy policy);

} // namespace ghb
