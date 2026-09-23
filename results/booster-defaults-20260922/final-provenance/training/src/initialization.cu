#include "ghb/initialization.cuh"
#include <algorithm>
#include <cmath>
#include <limits>

namespace ghb::gpu {
namespace {
using Index = unsigned long long;
constexpr unsigned threads = 256;

__device__ double reduce(double value, double* shared) {
  for (unsigned distance = 16; distance; distance >>= 1)
    value += __shfl_down_sync(0xffffffff, value, distance);
  if ((threadIdx.x & 31) == 0) shared[threadIdx.x >> 5] = value;
  __syncthreads();
  if (threadIdx.x < 32) {
    value = threadIdx.x < 8 ? shared[threadIdx.x] : 0;
    for (unsigned distance = 16; distance; distance >>= 1)
      value += __shfl_down_sync(0xffffffff, value, distance);
  }
  return value;
}
__device__ bool valid_weight(double weight) { return isfinite(weight) && weight >= 0; }
__device__ bool valid_target(double target, Objective objective) {
  return isfinite(target) && (objective != Objective::binary_logistic || target == 0 || target == 1);
}

__global__ void weights_partial(const float* weights, unsigned rows, unsigned outputs,
                                unsigned chunks, double* partials, unsigned* status) {
  __shared__ double sums[8];
  double value = 0;
  for (Index row = Index(blockIdx.x) * blockDim.x + threadIdx.x; row < rows;
       row += Index(chunks) * blockDim.x) {
    const double weight = weights ? weights[row] : 1;
    if (!valid_weight(weight)) atomicOr(status, unsigned(invalid_weight));
    else value += weight;
  }
  value = reduce(value, sums);
  if (!threadIdx.x) partials[Index(chunks) * outputs + blockIdx.x] = value;
}

__global__ void narrow_partial(Objective objective, const float* targets, const float* weights,
                               unsigned rows, unsigned outputs, unsigned chunks,
                               double* partials, unsigned* status) {
  __shared__ double sums[8];
  for (Index task = blockIdx.x; task < Index(outputs) * chunks; task += gridDim.x) {
    const unsigned output = task / chunks, chunk = task % chunks;
    double value = 0;
    for (Index row = Index(chunk) * blockDim.x + threadIdx.x; row < rows;
         row += Index(chunks) * blockDim.x) {
      const double target = targets[row * outputs + output];
      const double weight = weights ? weights[row] : 1;
      if (!valid_target(target, objective)) atomicOr(status, unsigned(invalid_target));
      else if (valid_weight(weight)) value += weight * target;
    }
    value = reduce(value, sums);
    if (!threadIdx.x) partials[Index(chunk) * outputs + output] = value;
    __syncthreads();
  }
}

__global__ void wide_partial(Objective objective, const float* targets, const float* weights,
                             unsigned rows, unsigned outputs, unsigned chunks,
                             double* partials, unsigned* status) {
  __shared__ double sums[8][32];
  const unsigned lane = threadIdx.x & 31, row_lane = threadIdx.x >> 5;
  const Index output_tiles = (Index(outputs) + 31) / 32;
  for (Index task = blockIdx.x; task < output_tiles * chunks; task += gridDim.x) {
    const unsigned chunk = task % chunks;
    const Index output = (task / chunks) * 32 + lane;
    double value = 0;
    if (output < outputs) {
      for (Index row = Index(chunk) * 8 + row_lane; row < rows; row += Index(chunks) * 8) {
        const double target = targets[row * outputs + output];
        const double weight = weights ? weights[row] : 1;
        if (!valid_target(target, objective)) atomicOr(status, unsigned(invalid_target));
        else if (valid_weight(weight)) value += weight * target;
      }
    }
    sums[row_lane][lane] = value;
    __syncthreads();
    for (unsigned distance = 4; distance; distance >>= 1) {
      if (row_lane < distance) sums[row_lane][lane] += sums[row_lane + distance][lane];
      __syncthreads();
    }
    if (!row_lane && output < outputs) partials[Index(chunk) * outputs + output] = sums[0][lane];
    __syncthreads();
  }
}

__global__ void multiclass_partial(const float* targets, const float* weights,
                                   unsigned rows, unsigned outputs, unsigned chunks,
                                   double* partials, unsigned* status) {
  __shared__ double sums[8];
  double weight_sum = 0;
  double* classes = partials + Index(blockIdx.x) * outputs;
  for (Index output = threadIdx.x; output < outputs; output += blockDim.x) classes[output] = 0;
  __syncthreads();
  for (Index row = Index(blockIdx.x) * blockDim.x + threadIdx.x; row < rows;
       row += Index(chunks) * blockDim.x) {
    const double weight = weights ? weights[row] : 1;
    const double target = targets[row];
    const bool weight_ok = valid_weight(weight);
    const bool target_ok = isfinite(target) && target >= 0 && target < outputs && floor(target) == target;
    if (!weight_ok) atomicOr(status, unsigned(invalid_weight));
    else weight_sum += weight;
    if (!target_ok) atomicOr(status, unsigned(invalid_target));
    if (weight_ok && target_ok && weight > 0) atomicAdd(classes + static_cast<unsigned>(target), weight);
  }
  weight_sum = reduce(weight_sum, sums);
  if (!threadIdx.x) partials[Index(chunks) * outputs + blockIdx.x] = weight_sum;
}

__global__ void finish_weights(const double* partials, unsigned chunks, unsigned outputs,
                               double* weight_sum, unsigned* status) {
  __shared__ double sums[8];
  double value = 0;
  for (unsigned chunk = threadIdx.x; chunk < chunks; chunk += blockDim.x)
    value += partials[Index(chunks) * outputs + chunk];
  value = reduce(value, sums);
  if (!threadIdx.x) {
    *weight_sum = value;
    if (!(value > 0) || !isfinite(value)) atomicOr(status, unsigned(no_positive_weight));
  }
}
__global__ void finish_base(Objective objective, const double* partials, unsigned chunks,
                            unsigned outputs, const double* weight_sum, double* base,
                            unsigned* status) {
  for (Index output = Index(blockIdx.x) * blockDim.x + threadIdx.x; output < outputs;
       output += Index(gridDim.x) * blockDim.x) {
    double sum = 0;
    for (unsigned chunk = 0; chunk < chunks; ++chunk) sum += partials[Index(chunk) * outputs + output];
    double value = 0;
    if (*weight_sum > 0 && isfinite(*weight_sum)) {
      const double mean = sum / *weight_sum;
      if (objective == Objective::squared_error) value = mean;
      else if (objective == Objective::binary_logistic) {
        const double p = fmin(1 - 1e-12, fmax(1e-12, mean));
        value = log(p) - log1p(-p);
      } else value = log(fmax(mean, 1e-12));
    }
    if (!isfinite(value)) atomicOr(status, unsigned(invalid_base_score));
    base[output] = value;
  }
}
} // namespace

cudaError_t initialize_training(Objective objective, const float* targets,
                                const float* weights, unsigned rows, unsigned outputs,
                                double* base_scores, double* weight_sum, double* partials,
                                unsigned chunks, unsigned* status, cudaStream_t stream) {
  if (!targets || !base_scores || !weight_sum || !partials || !status || !rows || !outputs ||
      !chunks || chunks > 65535 ||
      Index(chunks) * (Index(outputs) + 1) > std::numeric_limits<std::size_t>::max() / sizeof(double) ||
      Index(rows) * outputs > std::numeric_limits<std::size_t>::max() / sizeof(double) ||
      (objective != Objective::squared_error && objective != Objective::binary_logistic &&
       objective != Objective::multiclass_softmax) ||
      (objective == Objective::multiclass_softmax && outputs < 2)) return cudaErrorInvalidValue;
  auto error = cudaMemsetAsync(status, 0, sizeof(unsigned), stream);
  if (error != cudaSuccess) return error;
  if (objective == Objective::multiclass_softmax) {
    multiclass_partial<<<chunks, threads, 0, stream>>>(targets, weights, rows, outputs, chunks, partials, status);
    error = cudaGetLastError();
  } else {
    weights_partial<<<chunks, threads, 0, stream>>>(weights, rows, outputs, chunks, partials, status);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    if (outputs < 8) {
      const auto grid = unsigned(std::min<Index>(Index(chunks) * outputs, 65535));
      narrow_partial<<<grid, threads, 0, stream>>>(objective, targets, weights, rows, outputs, chunks, partials, status);
    } else {
      const auto grid = unsigned(std::min<Index>(Index(chunks) * ((Index(outputs) + 31) / 32), 65535));
      wide_partial<<<grid, threads, 0, stream>>>(objective, targets, weights, rows, outputs, chunks, partials, status);
    }
    error = cudaGetLastError();
  }
  if (error != cudaSuccess) return error;
  finish_weights<<<1, threads, 0, stream>>>(partials, chunks, outputs, weight_sum, status);
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  const auto grid = unsigned(std::min<Index>((Index(outputs) + threads - 1) / threads, 65535));
  finish_base<<<grid, threads, 0, stream>>>(objective, partials, chunks, outputs, weight_sum, base_scores, status);
  return cudaGetLastError();
}
} // namespace ghb::gpu
