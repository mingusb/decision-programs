#include "gh/data.cuh"
#include <cmath>

namespace gh {
namespace {
constexpr u32 threads = 256, tile_keys = 1024, missing = 0xffffffffu;
__device__ u32 grid(u64 n) { return u32(min(ceil_div(n, threads), u64(65535))); }
__device__ bool finite(float x) { return (__float_as_uint(x) & 0x7fffffffu) < 0x7f800000u; }
__device__ bool matrix(Dataset d, bool nonempty) {
  const u64 cells = u64(d.rows) * d.columns;
  return (!nonempty || d.rows) && d.columns && d.columns <= INT32_MAX &&
    mul_fits(cells, sizeof(float)) && contains(d.values, cells);
}
__device__ cudaError_t observed(Status* s) {
  const auto e = cudaGetLastError();
  if (e != cudaSuccess) fail(s, runtime);
  return e;
}
__device__ cudaError_t close(Status* s, cudaError_t e = cudaSuccess) {
  const auto last = finish(s);
  return e == cudaSuccess ? last : e;
}

struct Scratch {
  Arena arena;
  u32 *a, *b, *histogram, *unique, *scan;
  u32 blocks, unique_blocks;
};
__device__ Scratch scratch(Workspace w, u32 rows, u32 features, u32 radix) {
  Arena a{w};
  const u32 blocks = u32(ceil_div(rows, tile_keys));
  const u32 unique_blocks = u32(ceil_div(rows, threads));
  auto* first = a.take<u32>(u64(rows) * features);
  auto* second = a.take<u32>(u64(rows) * features);
  auto* histogram = a.take<u32>(u64(blocks) * radix * features);
  auto* unique = a.take<u32>(u64(unique_blocks + 1) * features);
  u64 scan_words = 0;
  for (u32 n = blocks * radix; n > threads;) {
    n = u32(ceil_div(n, threads));
    scan_words += u64(n) * features;
  }
  auto* scan = a.take<u32>(scan_words);
  return {a, first, second, histogram, unique, scan, blocks, unique_blocks};
}

__global__ void make_keys(Dataset d, u32 first, u32 count, u32* keys, Status* s) {
  __shared__ u32 tile[32][33];
  for (u64 row_base = u64(blockIdx.x) * 32; row_base < d.rows; row_base += u64(gridDim.x) * 32) {
    for (u32 y = threadIdx.y; y < 32; y += 8) {
      const u64 row = row_base + y;
      u32 key = missing;
      if (row < d.rows && threadIdx.x < count) {
        u32 bits = __float_as_uint(d.values.data[row * d.columns + first + threadIdx.x]);
        const u32 magnitude = bits & 0x7fffffffu;
        if (magnitude < 0x7f800000u) {
          if (!magnitude) bits = 0;
          key = bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
        } else if (magnitude == 0x7f800000u) fail(s, input);
      }
      tile[y][threadIdx.x] = key;
    }
    __syncthreads();
    const u64 row = row_base + threadIdx.x;
    for (u32 f = threadIdx.y; f < count; f += 8)
      if (row < d.rows) keys[u64(f) * d.rows + row] = tile[threadIdx.x][f];
    __syncthreads();
  }
}

struct Prefix { u32 before, total; };
// Fixed 256-thread integer scan; every lane participates, including zero tails.
__device__ Prefix block_prefix(u32 value, u32* warp_totals) {
  const u32 lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  u32 sum = value;
  for (u32 step = 1; step < 32; step *= 2) {
    const u32 other = __shfl_up_sync(missing, sum, step);
    if (lane >= step) sum += other;
  }
  if (lane == 31) warp_totals[warp] = sum;
  __syncthreads();
  if (!warp) {
    u32 all = lane < 8 ? warp_totals[lane] : 0;
    for (u32 step = 1; step < 32; step *= 2) {
      const u32 other = __shfl_up_sync(missing, all, step);
      if (lane >= step) all += other;
    }
    if (lane < 8) warp_totals[lane] = all;
  }
  __syncthreads();
  const Prefix result{sum - value + (warp ? warp_totals[warp - 1] : 0), warp_totals[7]};
  __syncthreads();
  return result;
}

__global__ void scan_tiles(u32* values, u32 length, u32 chunks, u32* totals) {
  __shared__ u32 warps[8];
  const u32 feature = blockIdx.x / chunks, chunk = blockIdx.x % chunks;
  const u64 index = u64(chunk) * threads + threadIdx.x, base = u64(feature) * length;
  const auto prefix = block_prefix(index < length ? values[base + index] : 0, warps);
  if (index < length) values[base + index] = prefix.before;
  if (!threadIdx.x && totals) totals[u64(feature) * chunks + chunk] = prefix.total;
}
__global__ void scan_offsets(u32* values, const u32* totals, u32 length, u32 chunks, u64 cells) {
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < cells; i += u64(gridDim.x) * threads)
    values[i] += totals[(i / length) * chunks + (i % length) / threads];
}
__device__ cudaError_t scan(u32* values, u32 length, u32 features, u32* workspace, Status* s) {
  u32* levels[5]{values};
  u32 lengths[5]{length};
  u32 depth = 0;
  while (true) {
    const u32 chunks = u32(ceil_div(lengths[depth], threads));
    u32* next = chunks > 1 ? workspace : nullptr;
    scan_tiles<<<features * chunks, threads>>>(levels[depth], lengths[depth], chunks, next);
    if (const auto e = observed(s); e != cudaSuccess) return e;
    if (chunks == 1) break;
    levels[++depth] = next;
    lengths[depth] = chunks;
    workspace += u64(chunks) * features;
  }
  while (depth) {
    --depth;
    const u64 cells = u64(lengths[depth]) * features;
    scan_offsets<<<grid(cells), threads>>>(levels[depth], levels[depth + 1],
      lengths[depth], lengths[depth + 1], cells);
    if (const auto e = observed(s); e != cudaSuccess) return e;
  }
  return cudaSuccess;
}

template<u32 Bits>
__global__ void radix_counts(const u32* keys, u32* counts, u32 rows, u32 blocks, u32 shift) {
  constexpr u32 radix = 1u << Bits;
  __shared__ u32 local[radix];
  const u32 lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const u32 feature = blockIdx.x / blocks, block = blockIdx.x % blocks;
  for (u32 d = threadIdx.x; d < radix; d += threads) local[d] = 0;
  __syncthreads();
  for (u32 item = 0; item < 4; ++item) {
    const u64 row = u64(block) * tile_keys + warp * 128 + item * 32 + lane;
    const bool live = row < rows;
    const u32 digit = ((live ? keys[u64(feature) * rows + row] : 0) >> shift) & (radix - 1);
    const u32 active = __ballot_sync(missing, live);
    const u32 peers = __match_any_sync(missing, digit) & active;
    if (live && lane == u32(__ffs(peers) - 1)) atomicAdd(local + digit, u32(__popc(peers)));
  }
  __syncthreads();
  for (u32 d = threadIdx.x; d < radix; d += threads)
    counts[(u64(feature) * radix + d) * blocks + block] = local[d];
}

template<u32 Bits>
__global__ void radix_move(const u32* source, u32* destination, const u32* prefixes,
                           u32 rows, u32 blocks, u32 shift) {
  constexpr u32 radix = 1u << Bits;
  __shared__ u32 warp_counts[8 * radix], bases[radix], ordered[tile_keys], totals[8];
  const u32 lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const u32 feature = blockIdx.x / blocks, block = blockIdx.x % blocks;
  const u64 start = u64(block) * tile_keys;
  u32 keys[4], ranks[4];
  for (u32 i = threadIdx.x; i < 8 * radix; i += threads) warp_counts[i] = 0;
  __syncthreads();
  for (u32 item = 0; item < 4; ++item) {
    const u64 row = start + warp * 128 + item * 32 + lane;
    const bool live = row < rows;
    keys[item] = live ? source[u64(feature) * rows + row] : 0;
    const u32 digit = (keys[item] >> shift) & (radix - 1);
    const u32 active = __ballot_sync(missing, live);
    const u32 peers = __match_any_sync(missing, digit) & active;
    ranks[item] = warp_counts[warp * radix + digit] + __popc(peers & ((1u << lane) - 1));
    __syncwarp(); // All readers precede the elected writer for this digit.
    if (live && lane == u32(__ffs(peers) - 1)) warp_counts[warp * radix + digit] += __popc(peers);
    __syncwarp();
  }
  __syncthreads();
  u32 count = 0;
  if (threadIdx.x < radix)
    for (u32 w = 0; w < 8; ++w) count += warp_counts[w * radix + threadIdx.x];
  const auto prefix = block_prefix(count, totals);
  if (threadIdx.x < radix) bases[threadIdx.x] = prefix.before;
  __syncthreads();
  for (u32 item = 0; item < 4; ++item) {
    const u32 digit = (keys[item] >> shift) & (radix - 1);
    u32 rank = ranks[item];
    for (u32 w = 0; w < warp; ++w) rank += warp_counts[w * radix + digit];
    if (start + warp * 128 + item * 32 + lane < rows) ordered[bases[digit] + rank] = keys[item];
  }
  __syncthreads();
  for (u32 i = threadIdx.x; i < min(u64(tile_keys), u64(rows) - start); i += threads) {
    const u32 key = ordered[i], digit = (key >> shift) & (radix - 1);
    const u32 target = prefixes[(u64(feature) * radix + digit) * blocks + block] + i - bases[digit];
    destination[u64(feature) * rows + target] = key;
  }
}

__device__ bool first_key(const u32* keys, u64 row, u32 rows) {
  return row < rows && keys[row] != missing && (!row || keys[row] != keys[row - 1]);
}
template<bool Compact>
__global__ void unique_keys(const u32* keys, u32* prefixes, u32* values, u32 rows, u32 blocks) {
  __shared__ u32 totals[8];
  const u32 feature = blockIdx.x / blocks, block = blockIdx.x % blocks;
  const u64 row = u64(block) * threads + threadIdx.x;
  const auto* column = keys + u64(feature) * rows;
  const bool first = first_key(column, row, rows);
  const auto prefix = block_prefix(first, totals);
  const u64 base = u64(feature) * (blocks + 1);
  if constexpr (Compact) {
    if (first) values[u64(feature) * rows + prefixes[base + block] + prefix.before] = column[row];
  } else if (!threadIdx.x) {
    prefixes[base + block] = prefix.total;
    if (!block) prefixes[base + blocks] = 0;
  }
}

__global__ void plan_metadata(Schema* schema, Array<const FeatureType> types, const u32* unique,
                              u32 blocks, u32 first, u32 count, u32 max_bins, Status* s) {
  if (s->errors) return;
  u64 metadata = schema->metadata_count, bins = schema->total_bins;
  for (u32 f = 0; f < count; ++f) {
    const auto type = types.size ? types.data[first + f] : FeatureType::numeric;
    const u32 distinct = unique[u64(f) * (blocks + 1) + blocks];
    if (type == FeatureType::categorical && distinct >= max_bins) { fail(s, capacity); return; }
    const u32 intervals = min(distinct, max_bins - 1);
    const u32 n = type == FeatureType::categorical ? distinct : intervals ? intervals - 1 : 0;
    metadata += n;
    bins += n + (type == FeatureType::numeric ? 2 : 1);
  }
  if (bins > UINT32_MAX) { fail(s, extent); return; }
  if (!contains(schema->metadata, metadata)) { fail(s, capacity); return; }
  for (u32 f = 0; f < count; ++f) {
    const auto type = types.size ? types.data[first + f] : FeatureType::numeric;
    const u32 distinct = unique[u64(f) * (blocks + 1) + blocks];
    const u32 intervals = min(distinct, max_bins - 1);
    const u32 n = type == FeatureType::categorical ? distinct : intervals ? intervals - 1 : 0;
    schema->features.data[first + f] = {schema->metadata_count, n, type};
    schema->metadata_count += n;
    const u32 feature_bins = n + (type == FeatureType::numeric ? 2 : 1);
    schema->total_bins += feature_bins;
    schema->offsets.data[first + f + 1] = schema->total_bins;
    schema->max_feature_bins = max(schema->max_feature_bins, feature_bins);
  }
}
__global__ void select_metadata(const u32* distinct, const u32* prefixes, u32 rows, u32 blocks,
                                Schema* schema, u32 first, u32 count, u32 max_bins, Status* s) {
  if (s->errors) return;
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < u64(count) * (max_bins - 1);
       i += u64(gridDim.x) * threads) {
    const u32 f = u32(i / (max_bins - 1)), k = u32(i % (max_bins - 1));
    const auto feature = schema->features.data[first + f];
    if (k >= feature.count) continue;
    const u32 total = prefixes[u64(f) * (blocks + 1) + blocks];
    const u32 rank = feature.type == FeatureType::categorical ? k : u32(u64(k + 1) * total / (feature.count + 1) - 1);
    const u32 key = distinct[u64(f) * rows + rank];
    schema->metadata.data[feature.begin + k] = __uint_as_float(key & 0x80000000u ? key ^ 0x80000000u : ~key);
  }
}

__global__ void validate_schema(const Schema* schema, u32 columns, Status* s) {
  const auto v = *schema;
  if (v.columns != columns || !contains(v.features, columns) ||
      !contains(v.offsets, u64(columns) + 1) || !contains(v.metadata, v.metadata_count) ||
      !v.max_feature_bins || v.max_feature_bins > 65536) { fail(s, shape); return; }
  for (u64 f = u64(blockIdx.x) * threads + threadIdx.x; f < columns; f += u64(gridDim.x) * threads) {
    const auto feature = v.features.data[f];
    const bool number = feature.type == FeatureType::numeric;
    if ((!number && feature.type != FeatureType::categorical) ||
        feature.count > (number ? 65534u : 65535u) || feature.begin > v.metadata_count ||
        feature.count > v.metadata_count - feature.begin) { fail(s, model); continue; }
    const u32 bins = feature.count + (number ? 2 : 1);
    if (v.offsets.data[f] > UINT32_MAX - bins || v.offsets.data[f + 1] != v.offsets.data[f] + bins ||
        bins > v.max_feature_bins || (!f && (v.offsets.data[0] || feature.begin)) ||
        (f && (v.features.data[f - 1].begin > feature.begin ||
          v.features.data[f - 1].count != feature.begin - v.features.data[f - 1].begin)) ||
        (f + 1 == columns && (v.offsets.data[f + 1] != v.total_bins || feature.begin + feature.count != v.metadata_count)))
      fail(s, model);
    for (u32 k = 0; k < feature.count; ++k) {
      const float x = v.metadata.data[feature.begin + k];
      if (!finite(x) || (k && !(v.metadata.data[feature.begin + k - 1] < x))) fail(s, model);
    }
  }
}
__global__ void encode_rows(Dataset d, const Schema* schema, std::uint16_t* bins, Status* s) {
  __shared__ float tile[32][33];
  __shared__ bool active;
  if (!threadIdx.x && !threadIdx.y) active = !s->errors;
  __syncthreads();
  if (!active) return;
  const u64 row_tiles = ceil_div(d.rows, 32), tasks = row_tiles * ceil_div(d.columns, 32);
  for (u64 task = blockIdx.x; task < tasks; task += gridDim.x) {
    const u64 row_base = (task % row_tiles) * 32;
    const u32 first = u32(task / row_tiles) * 32;
    for (u32 y = threadIdx.y; y < 32; y += 8)
      if (row_base + y < d.rows && first + threadIdx.x < d.columns)
        tile[y][threadIdx.x] = d.values.data[(row_base + y) * d.columns + first + threadIdx.x];
    __syncthreads();
    const u64 row = row_base + threadIdx.x;
    for (u32 f = threadIdx.y; f < 32 && first + f < d.columns; f += 8) {
      if (row >= d.rows) continue;
      const auto feature = schema->features.data[first + f];
      const float x = tile[threadIdx.x][f];
      const u32 magnitude = __float_as_uint(x) & 0x7fffffffu;
      u32 bin = 0;
      if (magnitude < 0x7f800000u) {
        u32 lo = 0, hi = feature.count;
        while (lo < hi) {
          const u32 mid = lo + (hi - lo) / 2;
          if (schema->metadata.data[feature.begin + mid] < x) lo = mid + 1; else hi = mid;
        }
        if (feature.type == FeatureType::numeric ||
            (lo < feature.count && schema->metadata.data[feature.begin + lo] == x)) bin = lo + 1;
      } else if (magnitude == 0x7f800000u) fail(s, input);
      bins[u64(first + f) * d.rows + row] = std::uint16_t(bin);
    }
    __syncthreads();
  }
}
__device__ cudaError_t submit_encoding(Dataset d, const Schema* schema, std::uint16_t* bins, Status* s) {
  if (!d.rows) return cudaSuccess;
  const u64 tiles = ceil_div(d.rows, 32) * ceil_div(d.columns, 32);
  encode_rows<<<u32(min(tiles, u64(65535))), dim3(32, 8)>>>(d, schema, bins, s);
  return observed(s);
}

template<u32 Bits>
__device__ cudaError_t submit_fit(Dataset d, Array<const FeatureType> types, u32 max_bins,
    Schema* schema, std::uint16_t* bins, Scratch w, u32 capacity, Status* s) {
  for (u32 first = 0; first < d.columns;) {
    const u32 count = min(capacity, d.columns - first);
    make_keys<<<u32(min(ceil_div(d.rows, 32), u64(65535))), dim3(32, 8)>>>(d, first, count, w.a, s);
    if (const auto e = observed(s); e != cudaSuccess) return e;
    u32* source = w.a;
    u32* target = w.b;
    for (u32 shift = 0; shift < 32; shift += Bits) {
      radix_counts<Bits><<<count * w.blocks, threads>>>(source, w.histogram, d.rows, w.blocks, shift);
      if (const auto e = observed(s); e != cudaSuccess) return e;
      if (const auto e = scan(w.histogram, w.blocks * (1u << Bits), count, w.scan, s); e != cudaSuccess) return e;
      radix_move<Bits><<<count * w.blocks, threads>>>(source, target, w.histogram, d.rows, w.blocks, shift);
      if (const auto e = observed(s); e != cudaSuccess) return e;
      auto* previous = source; source = target; target = previous;
    }
    unique_keys<false><<<count * w.unique_blocks, threads>>>(source, w.unique, target, d.rows, w.unique_blocks);
    if (const auto e = observed(s); e != cudaSuccess) return e;
    if (const auto e = scan(w.unique, w.unique_blocks + 1, count, w.scan, s); e != cudaSuccess) return e;
    unique_keys<true><<<count * w.unique_blocks, threads>>>(source, w.unique, target, d.rows, w.unique_blocks);
    if (const auto e = observed(s); e != cudaSuccess) return e;
    plan_metadata<<<1, 1>>>(schema, types, w.unique, w.unique_blocks, first, count, max_bins, s);
    if (const auto e = observed(s); e != cudaSuccess) return e;
    select_metadata<<<grid(u64(count) * (max_bins - 1)), threads>>>(target, w.unique, d.rows,
      w.unique_blocks, schema, first, count, max_bins, s);
    if (const auto e = observed(s); e != cudaSuccess) return e;
    first += count;
  }
  return submit_encoding(d, schema, bins, s);
}

__device__ u32 get_word(const std::byte* bytes) {
  u32 result = 0;
  for (u32 i = 0; i < 4; ++i) result |= u32(bytes[i]) << (8 * i);
  return result;
}
__device__ void put_word(std::byte* bytes, u32 value) {
  for (u32 i = 0; i < 4; ++i) bytes[i] = std::byte(value >> (8 * i));
}
__device__ bool record_shape(u32 rows, u32 columns, u32 outputs, Objective objective, u32 classes,
                              u64& cells, u64& bytes) {
  if (!rows || !columns || !outputs || u32(objective) > 2 ||
      (objective == Objective::multiclass_softmax && (outputs != 1 || classes < 2))) return false;
  cells = u64(rows) * (u64(columns) + outputs);
  if (!mul_fits(rows, u64(columns) + outputs) || !mul_fits(cells, 4) || !add_fits(cells * 4, 32)) return false;
  bytes = cells * 4 + 32;
  return true;
}
__device__ bool target_valid(float x, Objective objective, u32 classes) {
  return finite(x) && (objective != Objective::binary_logistic || x == 0 || x == 1) &&
    (objective != Objective::multiclass_softmax || (x >= 0 && double(x) < classes && floorf(x) == x));
}
template<bool Decode>
__global__ void dataset_payload(const std::byte* input_bytes, std::byte* output_bytes,
    DatasetRecord record, float* values, float* targets, u64 cells, Status* s) {
  const u64 feature_cells = u64(record.data.rows) * record.data.columns;
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < cells; i += u64(gridDim.x) * threads) {
    u32 bits;
    if constexpr (Decode) bits = get_word(input_bytes + 32 + i * 4);
    else bits = __float_as_uint(i < feature_cells ? record.data.values.data[i] : record.data.targets.data[i - feature_cells]);
    const float value = __uint_as_float(bits);
    if (i < feature_cells ? (bits & 0x7fffffffu) == 0x7f800000u : !target_valid(value, record.objective, record.classes)) fail(s, input);
    if constexpr (Decode) {
      if (i < feature_cells) values[i] = value; else targets[i - feature_cells] = value;
    } else put_word(output_bytes + 32 + i * 4, bits);
  }
}
constexpr u64 magic = 0x3130305344424847ULL;
}

__device__ cudaError_t fit_schema(Dataset d, Array<const FeatureType> types, u32 max_bins,
    Schema* schema, Array<std::uint16_t> bins, Workspace workspace, Status* s, RadixPolicy policy) {
  if (!s) return cudaErrorInvalidValue;
  if (!contains(Array<Schema>{schema, 1}, 1) || !matrix(d, true) || max_bins < 2 || max_bins > 65536 ||
      (types.size && !contains(types, d.columns)) || !contains(bins, u64(d.rows) * d.columns) ||
      !contains(schema->features, d.columns) || !contains(schema->offsets, u64(d.columns) + 1)) {
    fail(s, shape); return finish(s);
  }
  if (policy != RadixPolicy::radix8 && policy != RadixPolicy::radix4) { fail(s, unsupported); return finish(s); }
  for (u32 f = 0; types.size && f < d.columns; ++f)
    if (types.data[f] != FeatureType::numeric && types.data[f] != FeatureType::categorical) {
      fail(s, input); return finish(s);
    }
  u32 tile = min(d.columns, 32u);
  auto w = scratch(workspace, d.rows, tile, 1u << u32(policy));
  while (tile > 1 && (!w.arena.valid || w.arena.used > workspace.bytes))
    w = scratch(workspace, d.rows, --tile, 1u << u32(policy));
  if (!w.arena.fits(s)) return finish(s);
  schema->columns = d.columns; schema->metadata_count = 0;
  schema->total_bins = schema->max_feature_bins = 0; schema->offsets.data[0] = 0;
  const auto result = policy == RadixPolicy::radix8
    ? submit_fit<8>(d, types, max_bins, schema, bins.data, w, tile, s)
    : submit_fit<4>(d, types, max_bins, schema, bins.data, w, tile, s);
  return close(s, result);
}
__device__ cudaError_t encode(Dataset d, const Schema* schema, Array<std::uint16_t> bins, Status* s) {
  if (!s) return cudaErrorInvalidValue;
  if (!contains(Array<const Schema>{schema, 1}, 1) || !matrix(d, false) || !contains(bins, u64(d.rows) * d.columns)) { fail(s, shape); return finish(s); }
  validate_schema<<<grid(d.columns), threads>>>(schema, d.columns, s);
  auto result = observed(s);
  if (result == cudaSuccess) result = submit_encoding(d, schema, bins.data, s);
  return close(s, result);
}
__device__ cudaError_t decode_dataset(Array<const std::byte> bytes, Array<float> values,
    Array<float> targets, DatasetRecord* record, Status* s) {
  if (!s) return cudaErrorInvalidValue;
  if (!contains(Array<DatasetRecord>{record, 1}, 1) || !contains(bytes, 32)) { fail(s, shape); return finish(s); }
  for (u32 i = 0; i < 8; ++i)
    if (bytes.data[i] != std::byte(magic >> (i * 8))) { fail(s, input); return finish(s); }
  const u32 version = get_word(bytes.data + 8), rows = get_word(bytes.data + 12);
  const u32 columns = get_word(bytes.data + 16), outputs = get_word(bytes.data + 20);
  const auto objective = Objective(get_word(bytes.data + 24));
  const u32 classes = get_word(bytes.data + 28);
  u64 cells{}, required{};
  if (version != 1 || !record_shape(rows, columns, outputs, objective, classes, cells, required) || bytes.size != required) {
    fail(s, input); return finish(s);
  }
  if (!contains(values, u64(rows) * columns) || !contains(targets, u64(rows) * outputs)) { fail(s, capacity); return finish(s); }
  *record = {{{values.data, u64(rows) * columns}, {targets.data, u64(rows) * outputs}, {}, rows, columns, outputs}, objective, classes};
  dataset_payload<true><<<grid(cells), threads>>>(bytes.data, nullptr, *record, values.data, targets.data, cells, s);
  return close(s, observed(s));
}
__device__ cudaError_t encode_dataset(const DatasetRecord* record, Array<std::byte> bytes, Status* s) {
  if (!s) return cudaErrorInvalidValue;
  if (!contains(Array<const DatasetRecord>{record, 1}, 1)) { fail(s, shape); return finish(s); }
  const auto value = *record;
  const auto d = value.data;
  u64 cells{}, required{};
  if (!record_shape(d.rows, d.columns, d.outputs, value.objective, value.classes, cells, required) ||
      !contains(d.values, u64(d.rows) * d.columns) || !contains(d.targets, u64(d.rows) * d.outputs) || d.weights.size) {
    fail(s, shape); return finish(s);
  }
  s->required_bytes = required;
  if (!contains(bytes, required)) { fail(s, capacity); return finish(s); }
  for (u32 i = 0; i < 8; ++i) bytes.data[i] = std::byte(magic >> (i * 8));
  put_word(bytes.data + 8, 1); put_word(bytes.data + 12, d.rows); put_word(bytes.data + 16, d.columns);
  put_word(bytes.data + 20, d.outputs); put_word(bytes.data + 24, u32(value.objective)); put_word(bytes.data + 28, value.classes);
  dataset_payload<false><<<grid(cells), threads>>>(nullptr, bytes.data, value, nullptr, nullptr, cells, s);
  return close(s, observed(s));
}
}
