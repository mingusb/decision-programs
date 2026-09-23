// Deliberately compile the current production bodies into this standalone target.
// Link src/core.cu only, never gh (which would define data entrypoints twice).
#include "../src/data.cu"
#include "check.cuh"

namespace gh::data_leaf {
constexpr u32 rows = 1025, features = 2, blocks = 2, unique_blocks = 5;
constexpr u32 cells = rows * features, histogram_cells = 512 * features;
constexpr u32 canary = 0xded1ca7e;
struct Buffers {
  u32 keys[2][cells + 2];
  u32 histogram[histogram_cells + 2];
  u32 scan_totals[features * 2 + 2];
  u32 unique[features * (unique_blocks + 1) + 2];
};
__device__ Buffers buffers;

__device__ u32 domain(u32 feature) { return feature ? 241 : 251; }
__device__ u32 key(u32 feature, u32 value) {
  return feature ? (value << 24) | (((value * 13) % 241) << 8) | ((value * 17) % 16)
                 : value * 0x01010101u;
}
__device__ u32 original(u32 feature, u32 row) {
  if (row % 97 == 0) return UINT32_MAX;
  const u32 value = feature ? (row * 37 + 11) % 241 : (row * 73 + 19) % 251;
  return key(feature, value);
}
__global__ void fixture() {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cells + 2) {
    buffers.keys[0][i] = i && i <= cells ? original((i - 1) / rows, (i - 1) % rows) : canary;
    buffers.keys[1][i] = canary;
  }
  if (i < histogram_cells + 2) buffers.histogram[i] = canary;
  if (i < features * 2 + 2) buffers.scan_totals[i] = canary;
  if (i < features * (unique_blocks + 1) + 2) buffers.unique[i] = canary;
}
__device__ void guards() {
  for (u32 side = 0; side < 2; ++side)
    GH_CHECK(buffers.keys[side][0] == canary && buffers.keys[side][cells + 1] == canary);
  GH_CHECK(buffers.histogram[0] == canary && buffers.histogram[histogram_cells + 1] == canary);
  GH_CHECK(buffers.scan_totals[0] == canary && buffers.scan_totals[features * 2 + 1] == canary);
  GH_CHECK(buffers.unique[0] == canary && buffers.unique[features * (unique_blocks + 1) + 1] == canary);
}
template<u32 Bits, u32 Shift, bool Prefix>
__global__ void check_histogram(const u32* source) {
  constexpr u32 radix = 1u << Bits, length = radix * blocks;
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (!i) guards();
  if (i >= features * length) return;
  const u32 feature = i / length, digit = (i % length) / blocks, block = i % blocks;
  u32 expected = 0;
  for (u32 row = 0; row < rows; ++row) {
    const u32 current = (source[feature * rows + row] >> Shift) & (radix - 1);
    if constexpr (Prefix) expected += current < digit || (current == digit && row < block * 1024);
    else expected += row / 1024 == block && current == digit;
  }
  GH_CHECK(buffers.histogram[i + 1] == expected);
  if constexpr (Bits == 4) {
    for (u32 j = i + features * length + 1; j < histogram_cells + 1; j += features * length)
      GH_CHECK(buffers.histogram[j] == canary);
  }
}
template<u32 Processed> __device__ u32 low(u32 value) {
  if constexpr (Processed == 32) return value;
  else return value & ((1u << Processed) - 1);
}
template<u32 Processed> __device__ u32 rank(u32 feature, u32 row) {
  const u32 value = low<Processed>(original(feature, row));
  u32 before = 0;
  for (u32 other = 0; other < rows; ++other) {
    const u32 current = low<Processed>(original(feature, other));
    before += current < value || (current == value && other < row);
  }
  return before;
}
template<u32 Bits, u32 Shift> __global__ void check_pass() {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (!i) guards();
  if (i >= cells) return;
  constexpr u32 source = (Shift / Bits) % 2, destination = 1 - source;
  const u32 feature = i / rows, row = i % rows, expected = original(feature, row);
  GH_CHECK(buffers.keys[destination][1 + feature * rows + rank<Shift + Bits>(feature, row)] == expected);
  if constexpr (Shift == 0) GH_CHECK(buffers.keys[source][i + 1] == expected);
  else GH_CHECK(buffers.keys[source][1 + feature * rows + rank<Shift>(feature, row)] == expected);
}
__global__ void prepare_unique() {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cells) buffers.keys[1][i + 1] = canary;
}
// Count analytic distinct-key starts preceding a sorted-position boundary.
__device__ u32 unique_before(u32 feature, u32 end) {
  u32 distinct = 0;
  for (u32 value = 0; value < domain(feature); ++value) {
    const u32 candidate = key(feature, value);
    u32 first = 0, present = 0;
    for (u32 row = 0; row < rows; ++row) {
      const u32 actual = original(feature, row);
      first += actual < candidate; present += actual == candidate;
    }
    GH_CHECK(present);
    distinct += first < end;
  }
  return distinct;
}
template<bool Prefix> __global__ void check_unique_counts() {
  const u32 i = threadIdx.x;
  if (!i) guards();
  if (i >= features * (unique_blocks + 1)) return;
  const u32 feature = i / (unique_blocks + 1), block = i % (unique_blocks + 1);
  const u32 begin = min(block * 256, rows);
  const u32 expected = Prefix ? unique_before(feature, begin) : block == unique_blocks ? 0 :
    unique_before(feature, min(begin + 256, rows)) - unique_before(feature, begin);
  GH_CHECK(buffers.unique[i + 1] == expected);
}
template<u32 Bits> __global__ void check_compact() {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (!i) guards();
  if (i < cells) {
    const u32 feature = i / rows, value = i % rows;
    GH_CHECK(buffers.keys[1][i + 1] == (value < domain(feature) ? key(feature, value) : canary));
  }
  if (!i) printf("GH_GPU_ACTIVITY data_leaf radix=%u rows=1025 features=2 passes=%u stable_counts_prefix_unique=pass\n", Bits, 32 / Bits);
}

// All expressions below are fixed runtime submission/constant address marshalling.
// No host code computes fixtures, observed extents, or expected outcomes.
#define GH_DATA_LEAF(...) do { __VA_ARGS__; const auto error = cudaGetLastError(); \
  if (error != cudaSuccess) return error; } while (false)
template<u32 Bits, u32 Shift = 0> cudaError_t digits(Buffers* p) {
  constexpr u32 length = (1u << Bits) * blocks;
  const auto* source = p->keys[(Shift / Bits) % 2] + 1;
  auto* destination = p->keys[1 - (Shift / Bits) % 2] + 1;
  GH_DATA_LEAF(radix_counts<Bits><<<4, 256>>>(source, p->histogram + 1, rows, blocks, Shift));
  GH_DATA_LEAF(check_histogram<Bits, Shift, false><<<4, 256>>>(source));
  if constexpr (Bits == 4) {
    GH_DATA_LEAF(scan_tiles<<<2, 256>>>(p->histogram + 1, length, 1, nullptr));
  } else {
    GH_DATA_LEAF(scan_tiles<<<4, 256>>>(p->histogram + 1, length, 2, p->scan_totals + 1));
    GH_DATA_LEAF(scan_tiles<<<2, 256>>>(p->scan_totals + 1, 2, 1, nullptr));
    GH_DATA_LEAF(scan_offsets<<<4, 256>>>(p->histogram + 1, p->scan_totals + 1, length, 2, 1024));
  }
  GH_DATA_LEAF(check_histogram<Bits, Shift, true><<<4, 256>>>(source));
  GH_DATA_LEAF(radix_move<Bits><<<4, 256>>>(source, destination, p->histogram + 1, rows, blocks, Shift));
  GH_DATA_LEAF(check_pass<Bits, Shift><<<9, 256>>>());
  if constexpr (Shift + Bits < 32) return digits<Bits, Shift + Bits>(p);
  return cudaSuccess;
}
template<u32 Bits> cudaError_t run(Buffers* p) {
  GH_DATA_LEAF(fixture<<<9, 256>>>());
  const auto error = digits<Bits>(p);
  if (error != cudaSuccess) return error;
  GH_DATA_LEAF(prepare_unique<<<9, 256>>>());
  GH_DATA_LEAF(unique_keys<false><<<10, 256>>>(p->keys[0] + 1, p->unique + 1, nullptr, rows, unique_blocks));
  GH_DATA_LEAF(check_unique_counts<false><<<1, 32>>>());
  GH_DATA_LEAF(scan_tiles<<<2, 256>>>(p->unique + 1, 6, 1, nullptr));
  GH_DATA_LEAF(check_unique_counts<true><<<1, 32>>>());
  GH_DATA_LEAF(unique_keys<true><<<10, 256>>>(p->keys[0] + 1, p->unique + 1, p->keys[1] + 1, rows, unique_blocks));
  GH_DATA_LEAF(check_compact<Bits><<<9, 256>>>());
  return cudaSuccess;
}
#undef GH_DATA_LEAF
}

int main() {
  gh::data_leaf::Buffers* storage{};
  auto error = cudaGetSymbolAddress(reinterpret_cast<void**>(&storage), gh::data_leaf::buffers);
  if (error == cudaSuccess) error = gh::data_leaf::run<4>(storage);
  if (error == cudaSuccess) error = gh::data_leaf::run<8>(storage);
  const auto completed = cudaDeviceSynchronize();
  return static_cast<int>(error == cudaSuccess ? completed : error);
}
