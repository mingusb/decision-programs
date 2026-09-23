#include "check.cuh"
#include "gh/data.cuh"
#include <cmath>

namespace gh::test {
namespace {
constexpr u32 cells_capacity = 300000, metadata_capacity = 70000, fit_cases = 26;
__device__ float values[cells_capacity], metadata[metadata_capacity + 2];
__device__ Feature descriptors[65];
__device__ FeatureType types[65];
__device__ u32 offsets[66];
__device__ std::uint16_t bins[cells_capacity + 2];
__device__ __align__(16) std::byte scratch_bytes[4 << 20];
__device__ Schema schema;
__device__ Status status;

struct Case { u32 rows, columns, max_bins; };
__device__ Case shape_for(u32 test) {
  switch (test) {
    case 0: return {1, 1, 32};
    case 1: return {31, 3, 32};
    case 2: return {32, 31, 32};
    case 3: return {33, 32, 32};
    case 4: return {257, 33, 32};
    case 5: return {1025, 65, 32};
    case 6: return {4097, 65, 32};
    case 7: return {65537, 3, 32};
    case 8: return {257, 33, 32};
    case 9: return {1025, 3, 2};
    case 10: return {65535, 1, 65536};
    case 11: return {65536, 1, 65536};
    case 14: return {1025, 65, 32};
    default: return {33, 3, 32};
  }
}
__device__ FeatureType feature_type(u32 test, u32 f) {
  if (test == 10 || test == 11) return FeatureType::categorical;
  if (test == 0 || test == 9) return FeatureType::numeric;
  return f % 3 == 1 ? FeatureType::categorical : FeatureType::numeric;
}
__device__ float fixture_value(u32 test, u32 row, u32 feature, bool query = false) {
  if (test == 10 || test == 11) return query && row % 17 == 0 ? -1.0f : float(row);
  if (test == 8 || row % 29 == 0) return __uint_as_float(0x7fc00001u + row % 127);
  if ((test == 12 || test == 25) && row == 32 && feature == 2)
    return __uint_as_float(test == 12 ? 0x7f800000u : 0xff800000u);
  if (query && row % 17 == 0) return 9999.0f;
  if (row % 19 == 0) return __uint_as_float(row % 2 ? 0x80000000u : 0u);
  const u32 width = feature_type(test, feature) == FeatureType::categorical ? 11 : 73;
  return float(int((row * 17 + feature * 13) % width) - int(width / 2));
}
__global__ void fixture(u32 test, bool query) {
  const auto c = shape_for(test);
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < u64(c.rows) * c.columns; i += u64(gridDim.x) * blockDim.x)
    values[i] = fixture_value(test, u32(i / c.columns), u32(i % c.columns), query);
}
__global__ void fit_start(u32 id);
__global__ void encode_start(u32 id);
__global__ void codec_start(u32 id);

// Independent insertion-sort oracle on a bounded distinct domain. Large
// category fixtures have an analytic ordered sequence instead of quadratic work.
__global__ void check_feature(u32 id, bool query) {
  const u32 test = id % fit_cases, f = blockIdx.x;
  const auto c = shape_for(test);
  const auto type = feature_type(test, f);
  float distinct[80];
  u32 count = 0;
  if (test == 10) count = c.rows;
  else for (u32 row = 0; row < c.rows; ++row) {
    float x = fixture_value(test, row, f);
    if (!isfinite(x)) continue;
    if (x == 0) x = 0.0f;
    u32 at = 0;
    while (at < count && distinct[at] < x) ++at;
    if (at < count && distinct[at] == x) continue;
    GH_CHECK(count < 80);
    for (u32 j = count; j > at; --j) distinct[j] = distinct[j - 1];
    distinct[at] = x; ++count;
  }
  const u32 intervals = min(count, c.max_bins - 1);
  const u32 expected = type == FeatureType::categorical ? count : intervals ? intervals - 1 : 0;
  const auto feature = schema.features.data[f];
  GH_CHECK(feature.type == type && feature.count == expected);
  GH_CHECK(offsets[f + 1] - offsets[f] == expected + (type == FeatureType::numeric ? 2 : 1));
  for (u32 k = 0; k < expected; ++k) {
    const u32 rank = type == FeatureType::categorical ? k : u32(u64(k + 1) * count / intervals - 1);
    const float value = test == 10 ? float(rank) : distinct[rank];
    GH_CHECK(__float_as_uint(metadata[1 + feature.begin + k]) == __float_as_uint(value));
  }
  for (u32 row = 0; row < c.rows; ++row) {
    const float x = fixture_value(test, row, f, query);
    GH_CHECK(__float_as_uint(values[u64(row) * c.columns + f]) == __float_as_uint(x));
    u32 expected_bin = 0;
    if (isfinite(x)) {
      if (test == 10) expected_bin = x >= 0 ? row + 1 : 0;
      else if (type == FeatureType::categorical) {
        for (u32 k = 0; k < count; ++k) if (distinct[k] == x) expected_bin = k + 1;
      } else {
        expected_bin = 1;
        for (u32 k = 1; k < intervals; ++k)
          expected_bin += distinct[u32(u64(k) * count / intervals - 1)] < x;
      }
    }
    GH_CHECK(bins[1 + u64(f) * c.rows + row] == expected_bin);
  }
}
__device__ u32 expected_failure(u32 test) {
  switch (test) {
    case 11: case 15: case 18: case 24: return capacity;
    case 12: case 13: case 25: return input;
    case 16: case 17: case 19: case 20: case 22: case 23: return shape;
    case 21: return unsupported;
    default: return 0;
  }
}
__device__ void guards() {
  GH_CHECK(bins[0] == 0xdead && bins[cells_capacity + 1] == 0xbeef);
  GH_CHECK(__float_as_uint(metadata[0]) == 0x7fa12345u);
  GH_CHECK(__float_as_uint(metadata[metadata_capacity + 1]) == 0x7fa54321u);
}
__global__ void fit_check(u32 id, bool query) {
  const u32 test = id % fit_cases;
  GH_CHECK(status.done == 1);
  guards();
  if (const u32 error = expected_failure(test); error) {
    GH_CHECK(status.errors & error);
    if (test == 15) GH_CHECK(status.required_bytes > 16);
    fit_start<<<1, 1, 0, cudaStreamTailLaunch>>>(id + 1);
    return;
  }
  succeeded(status);
  const auto c = shape_for(test);
  GH_CHECK(schema.columns == c.columns && schema.total_bins == offsets[c.columns]);
  GH_CHECK(schema.metadata_count == descriptors[c.columns - 1].begin + descriptors[c.columns - 1].count);
  GH_CHECK(offsets[0] == 0);
  if (test == 14) GH_CHECK(status.required_bytes <= 65536);
  check_feature<<<c.columns, 1>>>(id, query);
  if (query) fit_start<<<1, 1, 0, cudaStreamTailLaunch>>>(id + 1);
  else encode_start<<<1, 1, 0, cudaStreamTailLaunch>>>(id);
}
__global__ void encode_start(u32 id) {
  const u32 test = id % fit_cases;
  const auto c = shape_for(test);
  status = {};
  fixture<<<64, 256>>>(test, true);
  const Dataset d{{values, u64(c.rows) * c.columns}, {}, {}, c.rows, c.columns, 1};
  submitted(encode(d, &schema, {bins + 1, cells_capacity}, &status));
  fit_check<<<1, 1, 0, cudaStreamTailLaunch>>>(id, true);
}
__global__ void fit_start(u32 id) {
  if (id == fit_cases * 2) { codec_start<<<1, 1, 0, cudaStreamTailLaunch>>>(0); return; }
  const u32 test = id % fit_cases;
  auto c = shape_for(test);
  status = {};
  schema = {{descriptors, test == 16 ? 0u : 65u}, {metadata + 1, test == 18 ? 0u : metadata_capacity},
    {offsets, test == 17 ? 0u : 66u}};
  for (u32 f = 0; f < c.columns; ++f) types[f] = test == 13 ? FeatureType(99) : feature_type(test, f);
  bins[0] = 0xdead; bins[cells_capacity + 1] = 0xbeef;
  metadata[0] = __uint_as_float(0x7fa12345u); metadata[metadata_capacity + 1] = __uint_as_float(0x7fa54321u);
  fixture<<<64, 256>>>(test, false);
  const Dataset d{{test == 22 ? nullptr : values, u64(c.rows) * c.columns}, {}, {},
    test == 23 ? 0u : c.rows, c.columns, 1};
  const auto policy = test == 21 ? RadixPolicy(3) : id < fit_cases ? RadixPolicy::radix4 : RadixPolicy::radix8;
  const Workspace w{scratch_bytes + (test == 24), test == 15 ? 16u : test == 14 ? 65536u : sizeof(scratch_bytes) - 16};
  submitted(fit_schema(d, {types, test == 0 ? 0u : c.columns}, test == 20 ? 1u : c.max_bins,
    &schema, {bins + 1, test == 19 ? 0u : cells_capacity}, w, &status, policy));
  fit_check<<<1, 1, 0, cudaStreamTailLaunch>>>(id, false);
}

__device__ std::byte encoded[2049], repeated[2049];
__device__ float codec_values[21], codec_targets[14], decoded_values[23], decoded_targets[16];
__device__ DatasetRecord record, decoded;
__device__ u32 word(const std::byte* p) {
  return u32(p[0]) | u32(p[1]) << 8 | u32(p[2]) << 16 | u32(p[3]) << 24;
}
__device__ void set_word(std::byte* p, u32 x) {
  p[0] = std::byte(x); p[1] = std::byte(x >> 8); p[2] = std::byte(x >> 16); p[3] = std::byte(x >> 24);
}
__global__ void schema_fail_start(u32 id);
__global__ void codec_roundtrip(u32 id, u64 size) {
  succeeded(status);
  for (u64 i = 0; i < size; ++i) GH_CHECK(encoded[i + 1] == repeated[i + 1]);
  codec_start<<<1, 1, 0, cudaStreamTailLaunch>>>(id + 1);
}
__global__ void codec_decoded(u32 id, u64 size) {
  GH_CHECK(status.done == 1);
  if (id >= 5) {
    GH_CHECK(status.errors != 0);
    codec_start<<<1, 1, 0, cudaStreamTailLaunch>>>(id + 1);
    return;
  }
  succeeded(status);
  GH_CHECK(decoded.data.rows == 7 && decoded.data.columns == 3 && decoded.data.outputs == record.data.outputs);
  GH_CHECK(decoded.objective == record.objective && decoded.classes == record.classes && decoded.data.weights.size == 0);
  for (u32 i = 0; i < 21; ++i) GH_CHECK(__float_as_uint(decoded_values[i + 1]) == __float_as_uint(codec_values[i]));
  for (u32 i = 0; i < 7 * record.data.outputs; ++i) GH_CHECK(__float_as_uint(decoded_targets[i + 1]) == __float_as_uint(codec_targets[i]));
  GH_CHECK(decoded_values[0] == -999 && decoded_values[22] == -999);
  GH_CHECK(decoded_targets[0] == -999 && decoded_targets[15] == -999);
  status = {};
  submitted(encode_dataset(&decoded, {repeated + 1, sizeof(repeated) - 1}, &status));
  codec_roundtrip<<<1, 1, 0, cudaStreamTailLaunch>>>(id, size);
}
__global__ void codec_encoded(u32 id) {
  GH_CHECK(status.done == 1);
  if (id == 3 || id == 4) {
    GH_CHECK(status.errors & (id == 3 ? shape : capacity));
    codec_start<<<1, 1, 0, cudaStreamTailLaunch>>>(id + 1);
    return;
  }
  succeeded(status);
  u64 size = status.required_bytes;
  auto* bytes = encoded + 1;
  GH_CHECK(word(bytes) == 0x44424847u && word(bytes + 4) == 0x31303053u);
  GH_CHECK(word(bytes + 8) == 1 && word(bytes + 12) == 7 && word(bytes + 16) == 3);
  GH_CHECK(word(bytes + 20) == record.data.outputs && word(bytes + 24) == u32(record.objective));
  GH_CHECK(word(bytes + 28) == record.classes);
  for (u32 i = 0; i < 21; ++i) GH_CHECK(word(bytes + 32 + i * 4) == __float_as_uint(codec_values[i]));
  for (u32 i = 0; i < 7 * record.data.outputs; ++i) GH_CHECK(word(bytes + 32 + (21 + i) * 4) == __float_as_uint(codec_targets[i]));
  if (id == 5) bytes[0] = std::byte(0);
  if (id == 6) set_word(bytes + 8, 2);
  if (id == 7) --size;
  if (id == 8) ++size;
  if (id == 9) set_word(bytes + 12, 0);
  if (id == 10) set_word(bytes + 24, 7);
  if (id == 11) set_word(bytes + 32, 0x7f800000u);
  if (id == 12) set_word(bytes + 32 + 21 * 4, 0x7fc00001u);
  if (id == 13) set_word(bytes + 24, u32(Objective::binary_logistic));
  if (id == 15) { set_word(bytes + 12, UINT32_MAX); set_word(bytes + 16, UINT32_MAX); }
  status = {};
  submitted(decode_dataset({bytes, size}, {decoded_values + 1, id == 14 ? 0u : 21u},
    {decoded_targets + 1, 14}, &decoded, &status));
  codec_decoded<<<1, 1, 0, cudaStreamTailLaunch>>>(id, size);
}
__global__ void codec_start(u32 id) {
  if (id == 16) { schema_fail_start<<<1, 1, 0, cudaStreamTailLaunch>>>(0); return; }
  status = {};
  const auto objective = id < 3 ? Objective(id) : Objective::squared_error;
  const u32 outputs = objective == Objective::multiclass_softmax ? 1 : 2;
  for (u32 i = 0; i < 21; ++i) codec_values[i] = i == 0 ? __uint_as_float(0xffc00017u) : i == 1 ? -0.0f : float(int(i) - 10);
  for (u32 i = 0; i < 7 * outputs; ++i)
    codec_targets[i] = objective == Objective::binary_logistic ? float(i % 2) : objective == Objective::multiclass_softmax ? float(i % 3) : float(int(i) - 5);
  decoded_values[0] = decoded_values[22] = -999; decoded_targets[0] = decoded_targets[15] = -999;
  record = {{{codec_values, 21}, {codec_targets, u64(7) * outputs},
    {codec_targets, id == 3 ? 7u : 0u}, 7, 3, outputs}, objective, objective == Objective::multiclass_softmax ? 3u : 0u};
  submitted(encode_dataset(&record, {encoded + 1, id == 4 ? 8u : sizeof(encoded) - 1}, &status));
  codec_encoded<<<1, 1, 0, cudaStreamTailLaunch>>>(id);
}

__global__ void schema_fail_checked(u32 id) {
  GH_CHECK(status.done == 1);
  if (!id) succeeded(status); else GH_CHECK(status.errors != 0);
  GH_CHECK(bins[0] == 0xdead && bins[1] == 0x5555);
  schema_fail_start<<<1, 1, 0, cudaStreamTailLaunch>>>(id + 1);
}
__global__ void schema_fail_start(u32 id) {
  if (id == 8) { printf("GPU data checks passed: 52 fits, fitted/query encoding, 16 codec cases, 8 schema cases\n"); return; }
  status = {};
  descriptors[0] = {0, 2, FeatureType::numeric}; metadata[1] = -1; metadata[2] = 1;
  offsets[0] = 0; offsets[1] = 4;
  schema = {{descriptors, 1}, {metadata + 1, 2}, {offsets, 2}, 1, 4, 4, 2};
  bins[0] = 0xdead; bins[1] = 0x5555;
  if (id == 1) descriptors[0].begin = UINT64_MAX;
  if (id == 2) metadata[2] = -2;
  if (id == 3) metadata[2] = __uint_as_float(0x7f800000u);
  if (id == 4) offsets[1] = 9;
  if (id == 5) schema.features.size = 0;
  if (id == 6) schema.columns = 2;
  if (id == 7) schema.metadata_count = 3;
  submitted(encode({{}, {}, {}, 0, 1, 1}, &schema, {bins + 1, 0}, &status));
  schema_fail_checked<<<1, 1, 0, cudaStreamTailLaunch>>>(id);
}
}
__global__ void run() { fit_start<<<1, 1>>>(0); }
}
