#include "gh/csv.cuh"
#include "check.cuh"
#include <math_constants.h>

namespace gh::test {
namespace {
struct Example { double value; char text[32]; };
__device__ const Example examples[]{
  {0., "0"}, {-0., "-0"}, {1., "1"}, {-1., "-1"},
  {.1, "0.10000000000000001"}, {.0001, "0.0001"}, {.00001, "1.0000000000000001e-05"},
  {1e16, "10000000000000000"}, {1e17, "1e+17"},
  {0x1p-1074, "4.9406564584124654e-324"}, {0x1p-1022, "2.2250738585072014e-308"},
  {0x1.fffffffffffffp1023, "1.7976931348623157e+308"},
  {0x1.0000000000001p0, "1.0000000000000002"},
  {9007199254740991., "9007199254740991"},
  {0x1.0000000000001p-4, "0.062500000000000014"},
  {0x1p-25, "2.9802322387695312e-08"}, {0x1.8p-24, "8.9406967163085938e-08"},
  {-0x1p-1074, "-4.9406564584124654e-324"},
  {-1e-4, "-0.0001"}, {1e15, "1000000000000000"}
};
constexpr u32 example_count = sizeof(examples) / sizeof(Example), cases = 16;
constexpr u64 scratch_bytes = 8ULL << 20, output_bytes = 2ULL << 20;
__device__ double values[65537], weights[2];
__device__ __align__(16) std::byte scratch[scratch_bytes + 32];
__device__ std::byte output[output_bytes + 2];
__device__ u64 written;
__device__ Status status;
__device__ void byte(u64& position, char c) {
  GH_CHECK(position < written);
  GH_CHECK(output[++position] == std::byte(c));
}
__device__ void literal(u64& position, const char* text) { while (*text) byte(position, *text++); }
__device__ void row_id(u64& position, u32 row) {
  u32 power = 1;
  while (power <= row / 10) power *= 10;
  do { byte(position, char('0' + row / power % 10)); power /= 10; } while (power);
}
__global__ void next(u32);
__global__ void verify(u32 c) {
  if (c < 7) {
    if (status.errors) printf("csv success case %u errors %u required %llu\n", c, status.errors, (unsigned long long)status.required_bytes);
    succeeded(status);
    u64 position = 0;
    if (c == 0) {
      literal(position, "row_id,prediction\n");
      for (u32 row = 0; row < example_count; ++row) {
        row_id(position, row); byte(position, ','); literal(position, examples[row].text); byte(position, '\n');
      }
    } else if (c == 1) literal(position, "row_id,target_0,target_1,weight\n0,0.25,0.5,0.5\n1,0.25,1,1\n");
    else if (c == 2) literal(position, "row_id,prediction_0,prediction_1\n0,0.25,0.5\n1,0.25,1\n");
    else if (c == 3) literal(position, "row_id,p0,p1,p2\n0,0.25,0.5,0.25\n");
    else if (c == 4) literal(position, "row_id,target_0,target_1,weight\n");
    else if (c == 5) {
      literal(position, "row_id,prediction\n");
      for (u32 row = 0; row < 65537; ++row) {
        row_id(position, row); byte(position, ','); byte(position, char('0' + row % 10)); byte(position, '\n');
      }
    } else literal(position, "row_id,prediction\n0,0.25\n");
    GH_CHECK(position == written && status.required_bytes == written);
    GH_CHECK(output[written + 1] == std::byte{0x6d});
  } else {
    GH_CHECK(status.done && status.errors);
    GH_CHECK(written == UINT64_MAX || c == 11);
  }
  GH_CHECK(output[0] == std::byte{0x6d} && output[output_bytes + 1] == std::byte{0x6d});
  for (u32 i = 0; i < 16; ++i) GH_CHECK(scratch[i] == std::byte{0x5a} && scratch[scratch_bytes + 16 + i] == std::byte{0x5a});
  if (c + 1 < cases) { next<<<1, 1, 0, cudaStreamTailLaunch>>>(c + 1); submitted(cudaGetLastError()); }
  else printf("PASS csv: exact precision-17 literals, ties, subnormals, headers, hierarchical compaction and rejections\n");
}
__global__ void next(u32 c) {
  u32 rows = 1, outputs = 1; CsvKind kind = CsvKind::predictions;
  for (u64 i = 0; i < output_bytes + 2; ++i) output[i] = std::byte{0x6d};
  for (u32 i = 0; i < 16; ++i) scratch[i] = scratch[scratch_bytes + 16 + i] = std::byte{0x5a};
  values[0] = .25; values[1] = .5; values[2] = .25; values[3] = 1;
  weights[0] = .5; weights[1] = 1;
  Array<const double> weight;
  if (c == 0) { rows = example_count; for (u32 i = 0; i < rows; ++i) values[i] = examples[i].value; }
  if (c == 1 || c == 2) { rows = outputs = 2; if (c == 1) { kind = CsvKind::targets; weight = {weights, 2}; } }
  if (c == 3) { outputs = 3; kind = CsvKind::multiclass; }
  if (c == 4) { rows = 0; outputs = 2; kind = CsvKind::targets; }
  if (c == 5) { rows = 65537; for (u32 i = 0; i < rows; ++i) values[i] = i % 10; }
  Array<const double> input{values, u64(rows) * outputs};
  Array<std::byte> bytes{output + 1, output_bytes};
  Workspace work{scratch + 16, scratch_bytes};
  if (c == 6) bytes.size = 25;
  if (c == 7) { kind = CsvKind::targets; weights[0] = -1; weight = {weights, 1}; }
  if (c == 8) values[0] = CUDART_NAN;
  if (c == 9) weight = {weights, 1};
  if (c == 10) work.bytes = 1;
  if (c == 11) bytes.size = 24;
  if (c == 12) bytes = {reinterpret_cast<std::byte*>(UINT64_MAX - 7), 32};
  if (c == 13) { rows = outputs = UINT32_MAX; input.size = UINT64_MAX; }
  if (c == 14) outputs = 0;
  if (c == 15) { rows = 0; outputs = UINT32_MAX; input = {}; }
  status = {}; written = UINT64_MAX;
  submitted(encode_csv(input, rows, outputs, kind, weight, bytes, {&written, 1}, work, &status));
  verify<<<1, 1, 0, cudaStreamTailLaunch>>>(c); submitted(cudaGetLastError());
}
}
__global__ void run() { next<<<1, 1>>>(0); submitted(cudaGetLastError()); }
}
