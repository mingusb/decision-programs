#pragma once
#include "ghb/higher_order_math.cuh"
#include "ghb/output_batch.cuh"

namespace ghb::gpu {

// Binary logistic loss only. Compact row-major derivative tiles; q may be null
// for Order=3. All buffer extents/finite inputs are caller-owned contracts.
// Asynchronous launches, no allocation or host synchronization. Order=3/4 only.
template<unsigned Order>
cudaError_t higher_gradients_tile(const double* predictions, const float* targets,
    const float* weights, double* g, double* h, double* t, double* q,
    unsigned rows, unsigned outputs, unsigned output_begin, unsigned output_count,
    cudaStream_t stream);

// The selector must remain live/immutable through completion. Empty or invalid
// selected derivative ranges leave output untouched; tail capacity is preserved.
// Counts are exact, floating sums unordered; T/Q retain their signs.
template<unsigned Order>
cudaError_t higher_root_histogram_batch(DataView, const double* g, const double* h,
    const double* t, const double* q, unsigned derivative_capacity,
    unsigned batch_capacity, std::size_t output_stride, const OutputBatch*,
    HigherStats<Order>*, cudaStream_t,
    const unsigned long long* cached_counts = nullptr);

template<unsigned Order>
cudaError_t higher_deeper_histogram(DataView, const int* assignments,
    const double* g, const double* h, const double* t, const double* q,
    unsigned derivative_capacity, unsigned batch_capacity, unsigned frontier_capacity,
    const unsigned* active, const OutputBatch*, HigherStats<Order>*, cudaStream_t);

// Same Split ABI and deterministic comparison rules as the order-2 path.
// The positive max_leaf_value bounds unshrunk proposals; see experiment contract.
template<unsigned Order>
cudaError_t higher_find_splits_batched_active(DataView, const HigherStats<Order>*,
    unsigned batch_capacity, unsigned frontier_capacity, const unsigned* active,
    SplitConfig, bool force_leaf, Split* candidates, Split* winners, SplitPolicy,
    cudaStream_t, const unsigned* batch_count = nullptr);

} // namespace ghb::gpu
