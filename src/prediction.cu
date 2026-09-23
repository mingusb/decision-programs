#include "gh/prediction.cuh"
#include <cuda/std/bit>

namespace gh {
namespace {
template<bool Decode>
__global__ void payload(const std::byte* source, std::byte* target,
                        const double* values, double* output, u64 count, Status* status) {
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += u64(gridDim.x) * blockDim.x) {
    u64 bits = 0;
    if constexpr (Decode) {
      #pragma unroll
      for (u32 k = 0; k < 8; ++k) bits |= u64(source[8*i+k]) << (8*k);
    } else bits = cuda::std::bit_cast<u64>(values[i]);
    if ((bits & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL) { fail(status,numeric); continue; }
    if constexpr (Decode) output[i] = cuda::std::bit_cast<double>(bits);
    else {
      #pragma unroll
      for (u32 k = 0; k < 8; ++k) target[8*i+k] = std::byte(bits >> (8*k));
    }
  }
}
__device__ bool dimensions(u32 rows, u32 outputs, u64& count, Status* status) {
  if (!outputs) { fail(status,gh::shape); return false; }
  count = u64(rows)*outputs;
  if (!mul_fits(count,8)) { fail(status,extent); return false; }
  status->required_bytes = count*8;
  return true;
}
__device__ cudaError_t complete_payload(Status* status) {
  const auto launched = cudaGetLastError();
  if (launched != cudaSuccess) fail(status,runtime);
  const auto completed = finish(status);
  return launched != cudaSuccess ? launched : completed;
}
}
__device__ cudaError_t decode_predictions(Array<const std::byte> bytes, u32 rows,
    u32 outputs, Array<double> values, Status* status) {
  if (!status) return cudaErrorInvalidValue;
  u64 count{};
  if (!dimensions(rows,outputs,count,status)) return finish(status);
  if (bytes.size != status->required_bytes) { fail(status,input); return finish(status); }
  if (!contains(bytes,status->required_bytes) || !contains(values,count)) { fail(status,capacity); return finish(status); }
  if (count) payload<true><<<u32(min(ceil_div(count,256),u64(65535))),256>>>(bytes.data,nullptr,nullptr,values.data,count,status);
  return complete_payload(status);
}
__device__ cudaError_t encode_predictions(Array<const double> values, u32 rows,
    u32 outputs, Array<std::byte> bytes, Status* status) {
  if (!status) return cudaErrorInvalidValue;
  u64 count{};
  if (!dimensions(rows,outputs,count,status)) return finish(status);
  if (!contains(values,count) || !contains(bytes,status->required_bytes)) { fail(status,capacity); return finish(status); }
  if (count) payload<false><<<u32(min(ceil_div(count,256),u64(65535))),256>>>(nullptr,bytes.data,values.data,nullptr,count,status);
  return complete_payload(status);
}
}
