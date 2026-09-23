#include "ghb/root_histogram.cuh"

#include <algorithm>
#include <limits>

namespace ghb::gpu {
namespace {
using Index = unsigned long long;
constexpr unsigned threads = 256, max_grid = 65535;
constexpr unsigned count_chunk_rows = 4096, shared_bytes = 48 * 1024;

bool valid_data(DataView data) {
  return data.rows && data.columns && data.columns <= unsigned(INT32_MAX) &&
         data.bins && data.offsets && data.types && data.total_bins >= data.columns &&
         data.max_feature_bins && data.max_feature_bins <= 65536 &&
         data.max_feature_bins <= data.total_bins &&
         Index(data.rows) * data.columns <= std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t);
}

__global__ void count_global(DataView data, unsigned long long* counts) {
  for (Index row = Index(blockIdx.x) * blockDim.x + threadIdx.x;
       row < data.rows; row += Index(gridDim.x) * blockDim.x) {
    for (unsigned feature = 0; feature < data.columns; ++feature)
      atomicAdd(counts + data.offsets[feature] + data.bins[Index(feature) * data.rows + row], 1ULL);
  }
}

__global__ void count_shared(DataView data, unsigned chunks, unsigned long long* counts) {
  extern __shared__ unsigned local[];
  const Index tasks = Index(data.columns) * chunks;
  for (Index task = blockIdx.x; task < tasks; task += gridDim.x) {
    const unsigned feature = unsigned(task / chunks);
    const unsigned chunk = unsigned(task % chunks);
    const unsigned begin = data.offsets[feature], bins = data.offsets[feature + 1] - begin;
    for (unsigned bin = threadIdx.x; bin < bins; bin += blockDim.x) local[bin] = 0;
    __syncthreads();
    const Index first_row = Index(chunk) * count_chunk_rows;
    const Index last_row = min(first_row + count_chunk_rows, Index(data.rows));
    for (Index row = first_row + threadIdx.x; row < last_row; row += blockDim.x)
      atomicAdd(local + data.bins[Index(feature) * data.rows + row], 1U);
    __syncthreads();
    for (unsigned bin = threadIdx.x; bin < bins; bin += blockDim.x) {
      const unsigned count = local[bin];
      if (count) atomicAdd(counts + begin + bin, static_cast<unsigned long long>(count));
    }
    // A CTA may service another feature/chunk. Its next initialization must
    // not overwrite local cells still being read by this chunk's flush.
    __syncthreads();
  }
}

__global__ void seed_counts(unsigned total_bins, unsigned batch_count,
                            const unsigned long long* counts, Stats* output) {
  const Index cells = Index(total_bins) * batch_count;
  for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x;
       i < cells; i += Index(gridDim.x) * blockDim.x)
    output[i] = Stats{0, 0, counts[i % total_bins]};
}

template<unsigned Width, bool CachedCounts>
__global__ void accumulate(DataView data, const double* gradient,
                            const double* hessian, unsigned derivative_stride,
                            unsigned first_output, unsigned batch_count,
                            unsigned output_groups, Stats* output) {
  const Index work = Index(data.rows) * output_groups * Width;
  // Every subgroup is complete, including the final warp. This exact peer
  // mask is independent of scheduling through the conditional derivative load.
  const unsigned mask = (0xffffffffu >> (32 - Width)) << ((threadIdx.x & 31u) & ~(Width - 1u));
  for (Index index = Index(blockIdx.x) * blockDim.x + threadIdx.x;
       index < work; index += Index(gridDim.x) * blockDim.x) {
    const unsigned lane = threadIdx.x & (Width - 1);
    const Index group = index / Width;
    const Index row = group / output_groups;
    const unsigned local_output = unsigned(group % output_groups) * Width + lane;
    const bool valid = local_output < batch_count;
    double g{}, h{};
    if (valid) {
      const Index derivative = row * derivative_stride + first_output + local_output;
      g = gradient[derivative]; h = hessian[derivative];
    }
    // Width divides the warp/block size. Work is padded to complete output
    // subgroups, so every leader remains active, even for a partial last batch.
    for (unsigned feature = 0; feature < data.columns; ++feature) {
      unsigned bin{};
      if (!lane) bin = data.bins[Index(feature) * data.rows + row];
      bin = __shfl_sync(mask, bin, 0, Width);
      if (valid) {
        Stats* destination = output + Index(local_output) * data.total_bins + data.offsets[feature] + bin;
        if (g != 0) atomicAdd(&destination->gradient, g);
        if (h != 0) atomicAdd(&destination->hessian, h);
        if constexpr (!CachedCounts) atomicAdd(&destination->count, 1ULL);
      }
    }
  }
}

template<unsigned Width, bool CachedCounts>
void launch(DataView data, const double* gradient, const double* hessian,
            unsigned stride, unsigned first, unsigned count, Stats* output,
            cudaStream_t stream) {
  const unsigned groups = 1 + (count - 1) / Width;
  const Index work = Index(data.rows) * groups * Width;
  const unsigned grid = unsigned(std::min<Index>((work + threads - 1) / threads, max_grid));
  accumulate<Width, CachedCounts><<<grid, threads, 0, stream>>>(data, gradient, hessian, stride, first, count, groups, output);
}

template<bool CachedCounts>
void launch_width(DataView data, const double* gradient, const double* hessian,
                  unsigned stride, unsigned first, unsigned count, Stats* output,
                  cudaStream_t stream) {
  if (count == 1) launch<1, CachedCounts>(data, gradient, hessian, stride, first, count, output, stream);
  else if (count <= 2) launch<2, CachedCounts>(data, gradient, hessian, stride, first, count, output, stream);
  else if (count <= 4) launch<4, CachedCounts>(data, gradient, hessian, stride, first, count, output, stream);
  else if (count <= 8) launch<8, CachedCounts>(data, gradient, hessian, stride, first, count, output, stream);
  else if (count <= 16) launch<16, CachedCounts>(data, gradient, hessian, stride, first, count, output, stream);
  else launch<32, CachedCounts>(data, gradient, hessian, stride, first, count, output, stream);
}
} // namespace

bool root_counts_shared_supported(DataView data) {
  return valid_data(data) && data.max_feature_bins <= shared_bytes / sizeof(unsigned);
}

cudaError_t root_counts(DataView data, unsigned long long* counts,
                        RootCountKernel policy, cudaStream_t stream) {
  if (!valid_data(data) || !counts ||
      (policy != RootCountKernel::global && policy != RootCountKernel::shared) ||
      (policy == RootCountKernel::shared && !root_counts_shared_supported(data)) ||
      Index(data.total_bins) > std::numeric_limits<std::size_t>::max() / sizeof(unsigned long long))
    return cudaErrorInvalidValue;
  const auto error = cudaMemsetAsync(counts, 0, std::size_t(data.total_bins) * sizeof(unsigned long long), stream);
  if (error != cudaSuccess) return error;
  if (policy == RootCountKernel::global) {
    const unsigned grid = unsigned(std::min<Index>((Index(data.rows) + threads - 1) / threads, max_grid));
    count_global<<<grid, threads, 0, stream>>>(data, counts);
  } else {
    const unsigned chunks = 1 + (data.rows - 1) / count_chunk_rows;
    const unsigned grid = unsigned(std::min<Index>(Index(data.columns) * chunks, max_grid));
    count_shared<<<grid, threads, std::size_t(data.max_feature_bins) * sizeof(unsigned), stream>>>(data, chunks, counts);
  }
  return cudaGetLastError();
}

cudaError_t root_histogram(DataView data, const double* gradient,
                           const double* hessian, unsigned derivative_stride,
                           unsigned first_output, unsigned batch_count,
                           Stats* output, cudaStream_t stream,
                           const unsigned long long* cached_counts) {
  constexpr auto bytes_limit = std::numeric_limits<std::size_t>::max();
  if (!valid_data(data) ||
      !gradient || !hessian || !output || !derivative_stride || !batch_count ||
      first_output >= derivative_stride || batch_count > derivative_stride - first_output ||
      Index(data.rows) * derivative_stride > bytes_limit / sizeof(double) ||
      Index(data.rows) * data.columns > bytes_limit / sizeof(std::uint16_t) ||
      Index(batch_count) * data.total_bins > bytes_limit / sizeof(Stats)) return cudaErrorInvalidValue;
  if (cached_counts) {
    const Index cells = Index(batch_count) * data.total_bins;
    const unsigned grid = unsigned(std::min<Index>((cells + threads - 1) / threads, max_grid));
    seed_counts<<<grid, threads, 0, stream>>>(data.total_bins, batch_count, cached_counts, output);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    launch_width<true>(data, gradient, hessian, derivative_stride, first_output, batch_count, output, stream);
  } else {
    const auto error = cudaMemsetAsync(output, 0, std::size_t(batch_count) * data.total_bins * sizeof(Stats), stream);
    if (error != cudaSuccess) return error;
    launch_width<false>(data, gradient, hessian, derivative_stride, first_output, batch_count, output, stream);
  }
  return cudaGetLastError();
}
} // namespace ghb::gpu
