#include "ghb/root_histogram.cuh"

#include <algorithm>
#include <limits>

namespace ghb::gpu {
namespace {
using Index = unsigned long long;
constexpr unsigned threads = 256, max_grid = 65535;

template<unsigned Width>
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
        atomicAdd(&destination->count, 1ULL);
      }
    }
  }
}

template<unsigned Width>
void launch(DataView data, const double* gradient, const double* hessian,
            unsigned stride, unsigned first, unsigned count, Stats* output,
            cudaStream_t stream) {
  const unsigned groups = 1 + (count - 1) / Width;
  const Index work = Index(data.rows) * groups * Width;
  const unsigned grid = unsigned(std::min<Index>((work + threads - 1) / threads, max_grid));
  accumulate<Width><<<grid, threads, 0, stream>>>(data, gradient, hessian, stride, first, count, groups, output);
}
} // namespace

cudaError_t root_histogram(DataView data, const double* gradient,
                           const double* hessian, unsigned derivative_stride,
                           unsigned first_output, unsigned batch_count,
                           Stats* output, cudaStream_t stream) {
  constexpr auto bytes_limit = std::numeric_limits<std::size_t>::max();
  if (!data.rows || !data.columns || data.columns > unsigned(INT32_MAX) ||
      !data.bins || !data.offsets || !data.types ||
      data.total_bins < data.columns || !data.max_feature_bins ||
      data.max_feature_bins > 65536 || data.max_feature_bins > data.total_bins ||
      !gradient || !hessian || !output || !derivative_stride || !batch_count ||
      first_output >= derivative_stride || batch_count > derivative_stride - first_output ||
      Index(data.rows) * derivative_stride > bytes_limit / sizeof(double) ||
      Index(data.rows) * data.columns > bytes_limit / sizeof(std::uint16_t) ||
      Index(batch_count) * data.total_bins > bytes_limit / sizeof(Stats)) return cudaErrorInvalidValue;
  const auto error = cudaMemsetAsync(output, 0, std::size_t(batch_count) * data.total_bins * sizeof(Stats), stream);
  if (error != cudaSuccess) return error;
  if (batch_count == 1) launch<1>(data, gradient, hessian, derivative_stride, first_output, batch_count, output, stream);
  else if (batch_count <= 2) launch<2>(data, gradient, hessian, derivative_stride, first_output, batch_count, output, stream);
  else if (batch_count <= 4) launch<4>(data, gradient, hessian, derivative_stride, first_output, batch_count, output, stream);
  else if (batch_count <= 8) launch<8>(data, gradient, hessian, derivative_stride, first_output, batch_count, output, stream);
  else if (batch_count <= 16) launch<16>(data, gradient, hessian, derivative_stride, first_output, batch_count, output, stream);
  else launch<32>(data, gradient, hessian, derivative_stride, first_output, batch_count, output, stream);
  return cudaGetLastError();
}
} // namespace ghb::gpu
