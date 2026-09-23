#include "ghb/deeper_histogram.cuh"

#include <algorithm>
#include <limits>

namespace ghb::gpu {
namespace {
using Index = unsigned long long;
constexpr unsigned threads = 256, max_grid = 65535, shared_limit = 48 * 1024;

unsigned width_for(DeeperHistogramPolicy policy) {
  switch (policy) {
    case DeeperHistogramPolicy::global: return 0;
    case DeeperHistogramPolicy::shared1: return 1;
    case DeeperHistogramPolicy::shared4: return 4;
    case DeeperHistogramPolicy::shared8: return 8;
  }
  return unsigned(-1);
}
bool data_valid(DataView data) {
  constexpr auto limit = std::numeric_limits<std::size_t>::max();
  return data.rows && data.columns && data.columns <= unsigned(INT32_MAX) &&
         data.bins && data.offsets && data.types && data.total_bins >= data.columns &&
         data.max_feature_bins && data.max_feature_bins <= 65536 &&
         data.max_feature_bins <= data.total_bins &&
         Index(data.rows) * data.columns <= limit / sizeof(std::uint16_t);
}
unsigned grid_for(Index count) {
  return unsigned(std::min<Index>((count + threads - 1) / threads, max_grid));
}
__global__ void clear(DataView data, unsigned capacity, unsigned outputs,
                      const unsigned* active, Stats* output) {
  const Index per_output = Index(capacity) * data.total_bins;
  for (Index local_output = blockIdx.y; local_output < outputs; local_output += gridDim.y) {
    const Index cells = Index(min(capacity, active[local_output])) * data.total_bins;
    for (Index index = Index(blockIdx.x) * blockDim.x + threadIdx.x;
         index < cells; index += Index(gridDim.x) * blockDim.x)
      output[local_output * per_output + index] = Stats{};
  }
}

template<unsigned Width>
__global__ void global_accumulate(DataView data, const int* assignments,
                                 const double* gradient, const double* hessian,
                                 unsigned stride, unsigned first, unsigned count,
                                 unsigned groups, unsigned capacity,
                                 const unsigned* active, Stats* output) {
  const Index work = Index(data.rows) * groups * Width;
  const unsigned lane = threadIdx.x & (Width - 1);
  const unsigned mask = (0xffffffffu >> (32 - Width)) << ((threadIdx.x & 31u) & ~(Width - 1u));
  for (Index index = Index(blockIdx.x) * blockDim.x + threadIdx.x;
       index < work; index += Index(gridDim.x) * blockDim.x) {
    const Index group = index / Width;
    const Index row = group / groups;
    const unsigned local_output = unsigned(group % groups) * Width + lane;
    int node = -1;
    unsigned nodes{};
    if (local_output < count) {
      nodes = min(capacity, active[local_output]);
      if (nodes) node = assignments[Index(local_output) * data.rows + row];
    }
    const bool valid = node >= 0 && unsigned(node) < nodes;
    // A complete subgroup agrees to skip, retaining valid shuffle masks even
    // when its leader's own output/node is inactive.
    if constexpr (Width == 1) { if (!valid) continue; }
    else if (!__any_sync(mask, valid)) continue;
    double g{}, h{};
    if (valid) {
      const Index derivative = row * stride + first + local_output;
      g = gradient[derivative]; h = hessian[derivative];
    }
    for (unsigned feature = 0; feature < data.columns; ++feature) {
      unsigned bin{};
      if (!lane) bin = data.bins[Index(feature) * data.rows + row];
      if constexpr (Width != 1) bin = __shfl_sync(mask, bin, 0, Width);
      if (valid) {
        Stats* destination = output + (Index(local_output) * capacity + unsigned(node)) * data.total_bins +
                             data.offsets[feature] + bin;
        if (g != 0) atomicAdd(&destination->gradient, g);
        if (h != 0) atomicAdd(&destination->hessian, h);
        atomicAdd(&destination->count, 1ULL);
      }
    }
  }
}

template<unsigned Width>
__global__ void shared_accumulate(DataView data, const int* assignments,
                                 const double* gradient, const double* hessian,
                                 unsigned stride, unsigned first, unsigned count,
                                 unsigned groups, unsigned capacity, unsigned chunks,
                                 const unsigned* active, Stats* output) {
  extern __shared__ unsigned long long shared_words[];
  auto* local = reinterpret_cast<Stats*>(shared_words);
  const Index tasks = Index(data.columns) * groups * chunks;
  constexpr unsigned rows_per_block = threads / Width;
  const unsigned lane = threadIdx.x & (Width - 1);
  const unsigned mask = (0xffffffffu >> (32 - Width)) << ((threadIdx.x & 31u) & ~(Width - 1u));
  for (Index task = blockIdx.x; task < tasks; task += gridDim.x) {
    const unsigned chunk = unsigned(task % chunks);
    const Index feature_group = task / chunks;
    const unsigned output_begin = unsigned(feature_group % groups) * Width;
    const unsigned feature = unsigned(feature_group / groups);
    const unsigned offset = data.offsets[feature];
    const unsigned bins = data.offsets[feature + 1] - offset;
    const unsigned cells = Width * capacity * bins;
    const unsigned local_output = output_begin + lane;
    const unsigned nodes = local_output < count ? min(capacity, active[local_output]) : 0;
    for (unsigned cell = threadIdx.x; cell < cells; cell += blockDim.x) local[cell] = Stats{};
    __syncthreads();
    for (Index row = Index(chunk) * rows_per_block + threadIdx.x / Width;
         row < data.rows; row += Index(chunks) * rows_per_block) {
      int node = -1;
      if (nodes) node = assignments[Index(local_output) * data.rows + row];
      const bool valid = node >= 0 && unsigned(node) < nodes;
      if constexpr (Width == 1) { if (!valid) continue; }
      else if (!__any_sync(mask, valid)) continue;
      unsigned bin{};
      if (!lane) bin = data.bins[Index(feature) * data.rows + row];
      if constexpr (Width != 1) bin = __shfl_sync(mask, bin, 0, Width);
      if (valid) {
        Stats* destination = local + (lane * capacity + unsigned(node)) * bins + bin;
        const Index derivative = row * stride + first + local_output;
        const double g = gradient[derivative], h = hessian[derivative];
        if (g != 0) atomicAdd(&destination->gradient, g);
        if (h != 0) atomicAdd(&destination->hessian, h);
        atomicAdd(&destination->count, 1ULL);
      }
    }
    __syncthreads();
    for (unsigned cell = threadIdx.x; cell < cells; cell += blockDim.x) {
      const Stats value = local[cell];
      if (!value.count) continue;
      const unsigned node_output = cell / bins;
      const unsigned target_output = output_begin + node_output / capacity;
      Stats* destination = output + (Index(target_output) * capacity + node_output % capacity) * data.total_bins +
                           offset + cell % bins;
      if (value.gradient != 0) atomicAdd(&destination->gradient, value.gradient);
      if (value.hessian != 0) atomicAdd(&destination->hessian, value.hessian);
      atomicAdd(&destination->count, value.count);
    }
    __syncthreads();
  }
}

template<unsigned Width>
void launch_global(DataView data, const int* assignments, const double* gradient,
                   const double* hessian, unsigned stride, unsigned first,
                   unsigned count, unsigned capacity, const unsigned* active,
                   Stats* output, cudaStream_t stream) {
  const unsigned groups = 1 + (count - 1) / Width;
  global_accumulate<Width><<<grid_for(Index(data.rows) * groups * Width), threads, 0, stream>>>(
      data, assignments, gradient, hessian, stride, first, count, groups, capacity, active, output);
}
template<unsigned Width>
void launch_shared(DataView data, const int* assignments, const double* gradient,
                   const double* hessian, unsigned stride, unsigned first,
                   unsigned count, unsigned capacity, unsigned chunks,
                   const unsigned* active, Stats* output, cudaStream_t stream) {
  const unsigned groups = 1 + (count - 1) / Width;
  const unsigned grid = unsigned(std::min<Index>(Index(data.columns) * groups * chunks, max_grid));
  const std::size_t bytes = std::size_t(Width) * capacity * data.max_feature_bins * sizeof(Stats);
  shared_accumulate<Width><<<grid, threads, bytes, stream>>>(data, assignments, gradient, hessian,
      stride, first, count, groups, capacity, chunks, active, output);
}
} // namespace

bool deeper_histogram_supported(DataView data, unsigned capacity, DeeperHistogramPolicy policy) {
  const unsigned width = width_for(policy);
  if (!data_valid(data) || !capacity || capacity > unsigned(INT32_MAX) || width == unsigned(-1) ||
      Index(capacity) * data.total_bins > std::numeric_limits<std::size_t>::max() / sizeof(Stats)) return false;
  return !width || Index(width) * capacity * data.max_feature_bins <= shared_limit / sizeof(Stats);
}

cudaError_t deeper_histogram(DataView data, const int* assignments,
                            const double* gradient, const double* hessian,
                            unsigned stride, unsigned first, unsigned count,
                            unsigned capacity, const unsigned* active, Stats* output,
                            DeeperHistogramPolicy policy, unsigned chunks, cudaStream_t stream) {
  constexpr auto limit = std::numeric_limits<std::size_t>::max();
  if (!deeper_histogram_supported(data, capacity, policy) || !assignments || !gradient || !hessian ||
      !active || !output || !stride || !count || first >= stride || count > stride - first ||
      Index(data.rows) * stride > limit / sizeof(double) ||
      Index(count) * data.rows > limit / sizeof(int) ||
      Index(capacity) * data.total_bins > limit / sizeof(Stats) / count ||
      (policy == DeeperHistogramPolicy::global ? chunks != 0 : !chunks || chunks > 256)) return cudaErrorInvalidValue;
  if (policy != DeeperHistogramPolicy::global &&
      Index(data.columns) * (1 + (count - 1) / width_for(policy)) >
          std::numeric_limits<Index>::max() / chunks) return cudaErrorInvalidValue;
  const dim3 clear_grid(grid_for(Index(capacity) * data.total_bins), std::min(count, max_grid));
  clear<<<clear_grid, threads, 0, stream>>>(data, capacity, count, active, output);
  const auto error = cudaGetLastError(); if (error != cudaSuccess) return error;
  if (policy == DeeperHistogramPolicy::global) {
    if (count == 1) launch_global<1>(data, assignments, gradient, hessian, stride, first, count, capacity, active, output, stream);
    else if (count <= 2) launch_global<2>(data, assignments, gradient, hessian, stride, first, count, capacity, active, output, stream);
    else if (count <= 4) launch_global<4>(data, assignments, gradient, hessian, stride, first, count, capacity, active, output, stream);
    else if (count <= 8) launch_global<8>(data, assignments, gradient, hessian, stride, first, count, capacity, active, output, stream);
    else if (count <= 16) launch_global<16>(data, assignments, gradient, hessian, stride, first, count, capacity, active, output, stream);
    else launch_global<32>(data, assignments, gradient, hessian, stride, first, count, capacity, active, output, stream);
  } else if (policy == DeeperHistogramPolicy::shared1)
    launch_shared<1>(data, assignments, gradient, hessian, stride, first, count, capacity, chunks, active, output, stream);
  else if (policy == DeeperHistogramPolicy::shared4)
    launch_shared<4>(data, assignments, gradient, hessian, stride, first, count, capacity, chunks, active, output, stream);
  else launch_shared<8>(data, assignments, gradient, hessian, stride, first, count, capacity, chunks, active, output, stream);
  return cudaGetLastError();
}
} // namespace ghb::gpu
