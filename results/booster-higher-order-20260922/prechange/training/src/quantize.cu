#include "ghb/quantize.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ghb {
namespace {
constexpr unsigned kThreads = 256, kItems = 4, kTile = kThreads * kItems;
constexpr unsigned kScanTile = 512, kMaxFeatures = 32;
constexpr unsigned kMissing = 0xffffffffu;
constexpr unsigned kInfinity = 1, kCategoryOverflow = 2;

[[noreturn]] void invalid(const char* message) { throw std::invalid_argument(message); }
void check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
std::size_t add(std::size_t a, std::size_t b) {
  if (a > std::numeric_limits<std::size_t>::max() - b) invalid("quantization allocation size overflow");
  return a + b;
}
std::size_t multiply(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b) invalid("quantization allocation size overflow");
  return a * b;
}
unsigned ceiling(unsigned value, unsigned divisor) { return value / divisor + (value % divisor != 0); }
std::size_t aligned4(std::size_t value) { return add(value, 3) & ~std::size_t(3); }

struct Buffer {
  void* pointer{};
  explicit Buffer(std::size_t bytes) { check(cudaMalloc(&pointer, bytes), "allocate quantization scratch"); }
  ~Buffer() { if (pointer) cudaFree(pointer); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};
// On failure, drain before host staging/model vectors or scratch can die.
struct Drain {
  cudaStream_t stream;
  bool active{true};
  ~Drain() { if (active) cudaStreamSynchronize(stream); }
};

void validate_input(const Dataset& data) {
  if (!data.rows || !data.columns) invalid("quantization requires nonempty rows and columns");
  if (data.values.size() != multiply(data.rows, data.columns)) invalid("quantization input matrix shape mismatch");
  if (!data.feature_types.empty() && data.feature_types.size() != data.columns) invalid("quantization feature type shape mismatch");
  for (auto type : data.feature_types)
    if (type != FeatureType::numeric && type != FeatureType::categorical) invalid("invalid quantization feature type");
}
unsigned feature_bins(const Feature& feature) {
  return unsigned((feature.type == FeatureType::numeric ? feature.cuts.size() + 2 : feature.categories.size() + 1));
}
void validate_features(const Dataset& data, const std::vector<Feature>& features) {
  if (features.size() != data.columns) invalid("quantization model feature count mismatch");
  std::uint64_t bins = 0;
  for (unsigned f = 0; f < data.columns; ++f) {
    const auto& feature = features[f];
    if (feature.type != FeatureType::numeric && feature.type != FeatureType::categorical) invalid("invalid model feature type");
    if (!data.feature_types.empty() && data.feature_types[f] != feature.type) invalid("quantization model feature type mismatch");
    const bool numeric = feature.type == FeatureType::numeric;
    if ((numeric && !feature.categories.empty()) || (!numeric && !feature.cuts.empty())) invalid("mixed feature metadata");
    const auto& values = numeric ? feature.cuts : feature.categories;
    if (values.size() > (numeric ? 65534u : 65535u)) invalid("quantization metadata exceeds uint16 bin capacity");
    for (std::size_t i = 0; i < values.size(); ++i)
      if (!std::isfinite(values[i]) || (i && !(values[i - 1] < values[i]))) invalid("feature metadata must be finite and strictly increasing");
    bins += feature_bins(feature);
    if (bins > std::numeric_limits<unsigned>::max()) invalid("quantization histogram offsets overflow");
  }
}
void reject_capture(cudaStream_t stream) {
  cudaStreamCaptureStatus status{};
  check(cudaStreamIsCapturing(stream, &status), "query quantization stream capture");
  if (status != cudaStreamCaptureStatusNone) invalid("quantization preparation cannot execute during stream capture");
}

struct Layout {
  std::size_t bytes{}, input{}, key_a{}, key_b{}, histogram{}, unique{}, sizes{}, metadata{}, status{};
  std::vector<std::size_t> scan;
  unsigned features{}, meta_stride{}, blocks{}, digits{};
  std::size_t take(std::size_t count, std::size_t size = 4) {
    const auto start = bytes;
    bytes = add(bytes, multiply(count, size));
    return start;
  }
};
Layout layout(unsigned rows, unsigned features, unsigned max_metadata, unsigned digits, bool fit) {
  Layout l; l.features = features; l.meta_stride = std::max(1u, max_metadata);
  l.blocks = ceiling(rows, kTile); l.digits = digits;
  l.status = l.take(1);
  l.input = l.take(multiply(rows, features));
  l.sizes = l.take(features);
  l.metadata = l.take(multiply(l.meta_stride, features));
  if (fit) {
    l.key_a = l.take(multiply(rows, features)); l.key_b = l.take(multiply(rows, features));
    const unsigned histogram_length = l.blocks * digits;
    l.histogram = l.take(multiply(histogram_length, features));
    l.unique = l.take(multiply(std::size_t(l.blocks) + 1, features));
    for (unsigned length = histogram_length; length > kScanTile;) {
      length = ceiling(length, kScanTile);
      l.scan.push_back(l.take(multiply(length, features)));
    }
  }
  return l;
}
Layout choose_layout(unsigned rows, unsigned columns, unsigned max_metadata, unsigned digits,
                     bool fit, std::size_t resident, std::size_t limit) {
  if (resident >= limit) invalid("quantization device memory budget cannot hold resident data and scratch");
  for (unsigned features = std::min(columns, kMaxFeatures); features; --features) {
    auto candidate = layout(rows, features, max_metadata, digits, fit);
    if (candidate.bytes <= limit - resident) return candidate;
  }
  invalid("quantization device memory budget cannot hold one complete feature tile");
}
template<class T> T* at(void* pointer, std::size_t offset) {
  return reinterpret_cast<T*>(static_cast<unsigned char*>(pointer) + offset);
}

__device__ unsigned ordered(float value, unsigned* status) {
  unsigned bits = __float_as_uint(value);
  const unsigned magnitude = bits & 0x7fffffffu;
  if (magnitude >= 0x7f800000u) {
    if (magnitude == 0x7f800000u) atomicOr(status, kInfinity);
    return kMissing;
  }
  if (magnitude == 0) bits = 0; // float equality treats both zeros as one value.
  return bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
}
__device__ float unordered(unsigned key) {
  return __uint_as_float(key & 0x80000000u ? key ^ 0x80000000u : ~key);
}

__global__ void transpose_keys(const float* input, unsigned* keys, unsigned rows,
                               unsigned features, unsigned* status) {
  __shared__ unsigned tile[32][33];
  const unsigned row_base = blockIdx.x * 32;
  for (unsigned j = threadIdx.y; j < 32; j += 8) {
    const std::size_t row = std::size_t(row_base) + j;
    tile[j][threadIdx.x] = row < rows && threadIdx.x < features
      ? ordered(input[row * features + threadIdx.x], status) : kMissing;
  }
  __syncthreads();
  const std::size_t row = std::size_t(row_base) + threadIdx.x;
  for (unsigned f = threadIdx.y; f < features; f += 8)
    if (row < rows) keys[std::size_t(f) * rows + row] = tile[threadIdx.x][f];
}

// 256-thread exclusive sum. Every thread calls it; scratch has nine words.
__device__ unsigned block_prefix(unsigned value, unsigned* scratch, unsigned& total) {
  const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  unsigned sum = value;
  for (unsigned delta = 1; delta < 32; delta <<= 1) {
    const auto other = __shfl_up_sync(0xffffffffu, sum, delta);
    if (lane >= delta) sum += other;
  }
  if (lane == 31) scratch[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    unsigned prefix = lane < 8 ? scratch[lane] : 0;
    const unsigned original = prefix;
    for (unsigned delta = 1; delta < 32; delta <<= 1) {
      const auto other = __shfl_up_sync(0xffffffffu, prefix, delta);
      if (lane >= delta) prefix += other;
    }
    if (lane < 8) scratch[lane] = prefix - original;
    if (lane == 7) scratch[8] = prefix;
  }
  __syncthreads();
  total = scratch[8];
  const unsigned result = scratch[warp] + sum - value;
  // Callers may immediately reuse scratch on the next stripe.
  __syncthreads();
  return result;
}

template<unsigned Bits>
__global__ void radix_count(const unsigned* keys, unsigned* counts, unsigned rows,
                            unsigned blocks, unsigned shift) {
  constexpr unsigned radix = 1u << Bits;
  __shared__ unsigned histogram[radix];
  for (unsigned d = threadIdx.x; d < radix; d += kThreads) histogram[d] = 0;
  __syncthreads();
  const unsigned feature = blockIdx.x / blocks, block = blockIdx.x % blocks;
  const std::size_t start = std::size_t(block) * kTile;
  const unsigned lane = threadIdx.x & 31;
  #pragma unroll
  for (unsigned item = 0; item < kItems; ++item) {
    const auto index = start + item * kThreads + threadIdx.x;
    const bool valid = index < rows;
    const unsigned key = valid ? keys[std::size_t(feature) * rows + index] : kMissing;
    const unsigned digit = (key >> shift) & (radix - 1);
    const unsigned valid_lanes = __ballot_sync(0xffffffffu, valid);
    const unsigned peers = __match_any_sync(0xffffffffu, digit) & valid_lanes;
    if (valid && lane == unsigned(__ffs(peers) - 1)) atomicAdd(histogram + digit, unsigned(__popc(peers)));
  }
  __syncthreads();
  for (unsigned d = threadIdx.x; d < radix; d += kThreads)
    counts[(std::size_t(feature) * radix + d) * blocks + block] = histogram[d];
}

__global__ void scan_blocks(unsigned* data, unsigned length, unsigned chunks, unsigned* sums) {
  __shared__ unsigned values[kScanTile];
  const unsigned feature = blockIdx.x / chunks, chunk = blockIdx.x % chunks;
  const std::size_t base = std::size_t(chunk) * kScanTile;
  const auto offset = std::size_t(feature) * length;
  for (unsigned j = threadIdx.x; j < kScanTile; j += kThreads)
    values[j] = base + j < length ? data[offset + base + j] : 0;
  __syncthreads();
  for (unsigned stride = 1; stride < kScanTile; stride <<= 1) {
    const unsigned index = (threadIdx.x + 1) * (stride << 1) - 1;
    if (index < kScanTile) values[index] += values[index - stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (sums) sums[std::size_t(feature) * chunks + chunk] = values[kScanTile - 1];
    values[kScanTile - 1] = 0;
  }
  __syncthreads();
  for (unsigned stride = kScanTile >> 1; stride; stride >>= 1) {
    const unsigned index = (threadIdx.x + 1) * (stride << 1) - 1;
    if (index < kScanTile) {
      const unsigned left = values[index - stride];
      values[index - stride] = values[index]; values[index] += left;
    }
    __syncthreads();
  }
  for (unsigned j = threadIdx.x; j < kScanTile; j += kThreads)
    if (base + j < length) data[offset + base + j] = values[j];
}
__global__ void add_scan_offsets(unsigned* data, const unsigned* sums, unsigned length,
                                 unsigned chunks, std::size_t total) {
  for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += std::size_t(gridDim.x) * blockDim.x) {
    const auto feature = index / length;
    const auto chunk = (index % length) / kScanTile;
    data[index] += sums[feature * chunks + chunk];
  }
}
void scan(unsigned* data, unsigned length, unsigned features, const Layout& l,
          void* scratch, cudaStream_t stream, unsigned level = 0) {
  const unsigned chunks = ceiling(length, kScanTile);
  unsigned* sums = chunks > 1 ? at<unsigned>(scratch, l.scan.at(level)) : nullptr;
  scan_blocks<<<features * chunks, kThreads, 0, stream>>>(data, length, chunks, sums);
  check(cudaGetLastError(), "launch quantization prefix scan");
  if (chunks > 1) {
    scan(sums, chunks, features, l, scratch, stream, level + 1);
    const auto total = multiply(length, features);
    const unsigned grid = unsigned(std::min<std::size_t>((total + kThreads - 1) / kThreads, 65535));
    add_scan_offsets<<<grid, kThreads, 0, stream>>>(data, sums, length, chunks, total);
    check(cudaGetLastError(), "launch quantization scan offsets");
  }
}

template<unsigned Bits>
__global__ void radix_scatter(const unsigned* input, unsigned* output, const unsigned* prefixes,
                              unsigned rows, unsigned blocks, unsigned shift) {
  constexpr unsigned radix = 1u << Bits;
  __shared__ unsigned histogram[radix], bases[radix], warp_counts[8 * radix];
  __shared__ unsigned local[kTile], scan_scratch[9];
  unsigned keys[kItems];
  const unsigned feature = blockIdx.x / blocks, block = blockIdx.x % blocks;
  const std::size_t start = std::size_t(block) * kTile;
  const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (unsigned d = threadIdx.x; d < radix; d += kThreads) histogram[d] = 0;
  __syncthreads();
  #pragma unroll
  for (unsigned item = 0; item < kItems; ++item) {
    const auto index = start + item * kThreads + threadIdx.x;
    const bool valid = index < rows;
    keys[item] = valid ? input[std::size_t(feature) * rows + index] : kMissing;
    const unsigned digit = (keys[item] >> shift) & (radix - 1);
    const unsigned mask = __ballot_sync(0xffffffffu, valid);
    const unsigned peers = __match_any_sync(0xffffffffu, digit) & mask;
    if (valid && lane == unsigned(__ffs(peers) - 1)) atomicAdd(histogram + digit, unsigned(__popc(peers)));
  }
  __syncthreads();
  unsigned total;
  const unsigned base = block_prefix(threadIdx.x < radix ? histogram[threadIdx.x] : 0, scan_scratch, total);
  if (threadIdx.x < radix) { bases[threadIdx.x] = base; histogram[threadIdx.x] = 0; }
  __syncthreads();
  #pragma unroll
  for (unsigned item = 0; item < kItems; ++item) {
    for (unsigned j = threadIdx.x; j < 8 * radix; j += kThreads) warp_counts[j] = 0;
    __syncthreads();
    const bool valid = start + item * kThreads + threadIdx.x < rows;
    const unsigned digit = (keys[item] >> shift) & (radix - 1);
    const unsigned mask = __ballot_sync(0xffffffffu, valid);
    const unsigned peers = __match_any_sync(0xffffffffu, digit) & mask;
    if (valid && lane == unsigned(__ffs(peers) - 1)) warp_counts[warp * radix + digit] = unsigned(__popc(peers));
    __syncthreads();
    if (valid) {
      unsigned rank = histogram[digit] + unsigned(__popc(peers & ((1u << lane) - 1)));
      for (unsigned w = 0; w < warp; ++w) rank += warp_counts[w * radix + digit];
      local[bases[digit] + rank] = keys[item];
    }
    __syncthreads();
    for (unsigned d = threadIdx.x; d < radix; d += kThreads) {
      unsigned count = 0;
      for (unsigned w = 0; w < 8; ++w) count += warp_counts[w * radix + d];
      histogram[d] += count;
    }
    __syncthreads();
  }
  const auto remaining = std::size_t(rows) - start;
  const unsigned count = unsigned(remaining < kTile ? remaining : kTile);
  for (unsigned index = threadIdx.x; index < count; index += kThreads) {
    const unsigned key = local[index], digit = (key >> shift) & (radix - 1);
    const unsigned target = prefixes[(std::size_t(feature) * radix + digit) * blocks + block] + index - bases[digit];
    output[std::size_t(feature) * rows + target] = key;
  }
}

__device__ bool unique_key(const unsigned* keys, std::size_t index, unsigned rows) {
  return index < rows && keys[index] != kMissing && (index == 0 || keys[index] != keys[index - 1]);
}
__global__ void count_unique(const unsigned* keys, unsigned* counts, unsigned rows, unsigned blocks) {
  __shared__ unsigned scratch[9];
  const unsigned feature = blockIdx.x / blocks, block = blockIdx.x % blocks;
  keys += std::size_t(feature) * rows;
  unsigned count = 0, total;
  for (unsigned item = 0; item < kItems; ++item)
    count += unique_key(keys, std::size_t(block) * kTile + item * kThreads + threadIdx.x, rows);
  block_prefix(count, scratch, total);
  if (threadIdx.x == 0) {
    counts[std::size_t(feature) * (blocks + 1) + block] = total;
    if (block == 0) counts[std::size_t(feature) * (blocks + 1) + blocks] = 0;
  }
}
__global__ void extract_metadata(const unsigned* keys, const unsigned* prefixes,
                                 const FeatureType* types, unsigned rows, unsigned blocks,
                                 unsigned max_bins, unsigned meta_stride, float* metadata,
                                 unsigned* sizes, unsigned* status) {
  __shared__ unsigned local[kTile], scratch[9], first_cut, last_cut;
  const unsigned feature = blockIdx.x / blocks, block = blockIdx.x % blocks;
  keys += std::size_t(feature) * rows;
  prefixes += std::size_t(feature) * (blocks + 1);
  metadata += std::size_t(feature) * meta_stride;
  const unsigned uniques = prefixes[blocks], begin = prefixes[block], end = prefixes[block + 1];
  const bool categorical = types[feature] == FeatureType::categorical;
  const unsigned intervals = min(uniques, max_bins - 1);
  if (threadIdx.x == 0 && block == 0) {
    sizes[feature] = categorical ? uniques : (intervals ? intervals - 1 : 0);
    if (categorical && uniques >= max_bins) atomicOr(status, kCategoryOverflow);
  }
  if (!uniques || (categorical && uniques >= max_bins)) return;
  unsigned running = 0;
  for (unsigned item = 0; item < kItems; ++item) {
    const auto index = std::size_t(block) * kTile + item * kThreads + threadIdx.x;
    const unsigned flag = unique_key(keys, index, rows);
    unsigned total;
    const unsigned rank = block_prefix(flag, scratch, total);
    if (flag) local[running + rank] = keys[index];
    running += total;
  }
  __syncthreads();
  if (categorical) {
    for (unsigned rank = threadIdx.x; rank < end - begin; rank += kThreads)
      metadata[begin + rank] = unordered(local[rank]);
  } else {
    if (threadIdx.x == 0) {
      const auto lower = (static_cast<unsigned long long>(begin) + 1) * intervals;
      const auto upper = (static_cast<unsigned long long>(end) + 1) * intervals;
      first_cut = max(1u, unsigned((lower + uniques - 1) / uniques));
      last_cut = min(intervals, unsigned((upper + uniques - 1) / uniques));
    }
    __syncthreads();
    for (unsigned cut = first_cut + threadIdx.x; cut < last_cut; cut += kThreads) {
      const auto rank = unsigned(static_cast<unsigned long long>(cut) * uniques / intervals - 1);
      metadata[cut - 1] = unordered(local[rank - begin]);
    }
  }
}

__global__ void encode_tile(const float* input, const float* metadata, const unsigned* sizes,
                            const FeatureType* types, std::uint16_t* output, unsigned rows,
                            unsigned features, unsigned meta_stride, unsigned* status) {
  __shared__ float tile[32][33];
  const std::size_t row_base = std::size_t(blockIdx.x) * 32;
  for (unsigned j = threadIdx.y; j < 32; j += 8) {
    const auto row = row_base + j;
    if (threadIdx.x < features && row < rows) tile[j][threadIdx.x] = input[row * features + threadIdx.x];
  }
  __syncthreads();
  const auto row = row_base + threadIdx.x;
  for (unsigned feature = threadIdx.y; feature < features; feature += 8) {
    if (row >= rows) continue;
    const float value = tile[threadIdx.x][feature];
    const unsigned magnitude = __float_as_uint(value) & 0x7fffffffu;
    unsigned bin = 0;
    if (magnitude < 0x7f800000u) {
      const float* values = metadata + std::size_t(feature) * meta_stride;
      const unsigned size = sizes[feature];
      unsigned lo = 0, hi = size;
      while (lo < hi) {
        const unsigned middle = lo + (hi - lo) / 2;
        if (values[middle] < value) lo = middle + 1; else hi = middle;
      }
      if (types[feature] == FeatureType::numeric || (lo < size && values[lo] == value)) bin = lo + 1;
    } else if (magnitude == 0x7f800000u) atomicOr(status, kInfinity);
    output[std::size_t(feature) * rows + row] = std::uint16_t(bin);
  }
}

template<unsigned Bits>
void sort_keys(unsigned* a, unsigned* b, unsigned* counts, unsigned rows, unsigned features,
               const Layout& l, void* scratch, cudaStream_t stream) {
  for (unsigned shift = 0; shift < 32; shift += Bits) {
    radix_count<Bits><<<features * l.blocks, kThreads, 0, stream>>>(a, counts, rows, l.blocks, shift);
    check(cudaGetLastError(), "launch quantization radix histogram");
    scan(counts, l.blocks * (1u << Bits), features, l, scratch, stream);
    radix_scatter<Bits><<<features * l.blocks, kThreads, 0, stream>>>(a, b, counts, rows, l.blocks, shift);
    check(cudaGetLastError(), "launch quantization radix scatter");
    std::swap(a, b);
  }
  // Both configured pass counts are even: sorted output is the original a.
}
void upload_tile(const Dataset& data, unsigned begin, unsigned features, float* destination, cudaStream_t stream) {
  const auto width = multiply(features, sizeof(float));
  if (begin == 0 && features == data.columns) {
    check(cudaMemcpyAsync(destination, data.values.data(), multiply(data.rows, width),
                          cudaMemcpyHostToDevice, stream), "upload contiguous quantization float matrix");
    return;
  }
  check(cudaMemcpy2DAsync(destination, width, data.values.data() + begin,
                          multiply(data.columns, sizeof(float)), width, data.rows,
                          cudaMemcpyHostToDevice, stream), "upload quantization float tile");
}
void validate_status(unsigned status) {
  if (status & kInfinity) invalid("quantization input contains infinity");
  if (status & kCategoryOverflow) invalid("categorical cardinality exceeds max_bins after reserving missing bin zero");
}

} // namespace

struct QuantizerAccess {
  static std::size_t resident_size(const Dataset& data) {
    return add(add(aligned4(multiply(multiply(data.rows, data.columns), sizeof(std::uint16_t))),
                   multiply(std::size_t(data.columns) + 1, sizeof(unsigned))), multiply(data.columns, sizeof(FeatureType)));
  }
  static void allocate(QuantizedData& result, const Dataset& data) {
    result.resident_bytes = resident_size(data);
    check(cudaMalloc(&result.allocation_, result.resident_bytes), "allocate resident quantized data");
    const auto bins_bytes = aligned4(multiply(multiply(data.rows, data.columns), sizeof(std::uint16_t)));
    result.view = {at<std::uint16_t>(result.allocation_, 0), at<unsigned>(result.allocation_, bins_bytes),
                   at<FeatureType>(result.allocation_, bins_bytes + multiply(std::size_t(data.columns) + 1, sizeof(unsigned))),
                   data.rows, data.columns, 0, 0};
  }
};

QuantizedData::~QuantizedData() { if (allocation_) cudaFree(allocation_); }
QuantizedData::QuantizedData(QuantizedData&& other) noexcept
  : view(std::exchange(other.view, {})), features(std::move(other.features)),
    resident_bytes(std::exchange(other.resident_bytes, 0)), peak_bytes(std::exchange(other.peak_bytes, 0)),
    allocation_(std::exchange(other.allocation_, nullptr)) {}
QuantizedData& QuantizedData::operator=(QuantizedData&& other) noexcept {
  if (this != &other) {
    if (allocation_) cudaFree(allocation_);
    view = std::exchange(other.view, {}); features = std::move(other.features);
    resident_bytes = std::exchange(other.resident_bytes, 0); peak_bytes = std::exchange(other.peak_bytes, 0);
    allocation_ = std::exchange(other.allocation_, nullptr);
  }
  return *this;
}

QuantizedData fit_quantize(const Dataset& data, std::uint32_t max_bins, std::size_t memory_limit,
                          cudaStream_t stream, QuantizePolicy policy) {
  validate_input(data);
  if (max_bins < 2 || max_bins > 65536) invalid("quantization max_bins must be in [2,65536]");
  if (policy != QuantizePolicy::radix8 && policy != QuantizePolicy::radix4) invalid("invalid quantization radix policy");
  const unsigned digits = policy == QuantizePolicy::radix8 ? 256 : 16;
  const auto l = choose_layout(data.rows, data.columns, max_bins - 1, digits, true,
                               QuantizerAccess::resident_size(data), memory_limit);
  reject_capture(stream);
  QuantizedData result;
  result.features.resize(data.columns);
  std::vector<FeatureType> types(data.columns, FeatureType::numeric);
  if (!data.feature_types.empty()) types = data.feature_types;
  for (unsigned f = 0; f < data.columns; ++f) result.features[f].type = types[f];
  std::vector<unsigned> sizes(l.features), offsets(std::size_t(data.columns) + 1);
  unsigned status = 0;
  QuantizerAccess::allocate(result, data);
  Buffer buffer(l.bytes);
  result.peak_bytes = add(result.resident_bytes, l.bytes);
  Drain drain{stream};
  auto* input = at<float>(buffer.pointer, l.input);
  auto* key_a = at<unsigned>(buffer.pointer, l.key_a);
  auto* key_b = at<unsigned>(buffer.pointer, l.key_b);
  auto* histogram = at<unsigned>(buffer.pointer, l.histogram);
  auto* unique = at<unsigned>(buffer.pointer, l.unique);
  auto* metadata = at<float>(buffer.pointer, l.metadata);
  auto* device_sizes = at<unsigned>(buffer.pointer, l.sizes);
  auto* device_status = at<unsigned>(buffer.pointer, l.status);
  check(cudaMemsetAsync(device_status, 0, sizeof(unsigned), stream), "clear quantization status");
  check(cudaMemcpyAsync(const_cast<FeatureType*>(result.view.types), types.data(), multiply(data.columns, sizeof(FeatureType)),
                         cudaMemcpyHostToDevice, stream), "upload quantization feature types");
  for (unsigned begin = 0; begin < data.columns;) {
    const unsigned features = std::min(l.features, data.columns - begin);
    upload_tile(data, begin, features, input, stream);
    transpose_keys<<<ceiling(data.rows, 32), dim3(32, 8), 0, stream>>>(input, key_a, data.rows, features, device_status);
    check(cudaGetLastError(), "launch quantization layout conversion");
    if (policy == QuantizePolicy::radix8) sort_keys<8>(key_a, key_b, histogram, data.rows, features, l, buffer.pointer, stream);
    else sort_keys<4>(key_a, key_b, histogram, data.rows, features, l, buffer.pointer, stream);
    count_unique<<<features * l.blocks, kThreads, 0, stream>>>(key_a, unique, data.rows, l.blocks);
    check(cudaGetLastError(), "launch quantization unique count");
    scan(unique, l.blocks + 1, features, l, buffer.pointer, stream);
    extract_metadata<<<features * l.blocks, kThreads, 0, stream>>>(key_a, unique, result.view.types + begin,
      data.rows, l.blocks, max_bins, l.meta_stride, metadata, device_sizes, device_status);
    check(cudaGetLastError(), "launch quantization exact metadata extraction");
    check(cudaMemcpyAsync(sizes.data(), device_sizes, multiply(features, sizeof(unsigned)), cudaMemcpyDeviceToHost, stream), "export quantization metadata sizes");
    check(cudaMemcpyAsync(&status, device_status, sizeof(unsigned), cudaMemcpyDeviceToHost, stream), "export quantization status");
    check(cudaStreamSynchronize(stream), "finish quantization metadata sizes");
    validate_status(status);
    encode_tile<<<ceiling(data.rows, 32), dim3(32, 8), 0, stream>>>(input, metadata, device_sizes, result.view.types + begin,
      const_cast<std::uint16_t*>(result.view.bins) + std::size_t(begin) * data.rows, data.rows, features, l.meta_stride, device_status);
    check(cudaGetLastError(), "launch fitted quantization encoding");
    for (unsigned f = 0; f < features; ++f) {
      auto& feature = result.features[begin + f];
      auto& values = feature.type == FeatureType::numeric ? feature.cuts : feature.categories;
      if (sizes[f] > l.meta_stride) throw std::runtime_error("GPU quantization returned invalid metadata size");
      values.resize(sizes[f]);
      if (!values.empty()) check(cudaMemcpyAsync(values.data(), metadata + std::size_t(f) * l.meta_stride,
        multiply(values.size(), sizeof(float)), cudaMemcpyDeviceToHost, stream), "export fitted feature metadata");
      const unsigned bins = feature_bins(feature);
      if (bins > std::numeric_limits<unsigned>::max() - offsets[begin + f]) invalid("quantization histogram offsets overflow");
      offsets[begin + f + 1] = offsets[begin + f] + bins;
      result.view.max_feature_bins = std::max(result.view.max_feature_bins, bins);
    }
    check(cudaStreamSynchronize(stream), "finish fitted feature tile");
    begin += features;
  }
  result.view.total_bins = offsets.back();
  check(cudaMemcpyAsync(const_cast<unsigned*>(result.view.offsets), offsets.data(), multiply(offsets.size(), sizeof(unsigned)),
                         cudaMemcpyHostToDevice, stream), "upload fitted feature offsets");
  check(cudaStreamSynchronize(stream), "finish fitted quantized data");
  drain.active = false;
  return result;
}

QuantizedData encode_quantize(const Dataset& data, const std::vector<Feature>& features,
                             std::size_t memory_limit, cudaStream_t stream) {
  validate_input(data); validate_features(data, features);
  unsigned max_metadata = 0;
  for (const auto& feature : features)
    max_metadata = std::max(max_metadata, unsigned(feature.type == FeatureType::numeric ? feature.cuts.size() : feature.categories.size()));
  const auto l = choose_layout(data.rows, data.columns, max_metadata, 0, false,
                               QuantizerAccess::resident_size(data), memory_limit);
  reject_capture(stream);
  QuantizedData result; result.features = features;
  std::vector<FeatureType> types(data.columns);
  std::vector<unsigned> sizes(l.features), offsets(std::size_t(data.columns) + 1);
  for (unsigned f = 0; f < data.columns; ++f) {
    types[f] = features[f].type;
    const unsigned bins = feature_bins(features[f]);
    offsets[f + 1] = offsets[f] + bins;
    result.view.max_feature_bins = std::max(result.view.max_feature_bins, bins);
  }
  const unsigned maximum_bins = result.view.max_feature_bins;
  unsigned status = 0;
  QuantizerAccess::allocate(result, data);
  result.view.total_bins = offsets.back(); result.view.max_feature_bins = maximum_bins;
  Buffer buffer(l.bytes); result.peak_bytes = add(result.resident_bytes, l.bytes);
  Drain drain{stream};
  auto* input = at<float>(buffer.pointer, l.input);
  auto* metadata = at<float>(buffer.pointer, l.metadata);
  auto* device_sizes = at<unsigned>(buffer.pointer, l.sizes);
  auto* device_status = at<unsigned>(buffer.pointer, l.status);
  check(cudaMemsetAsync(device_status, 0, sizeof(unsigned), stream), "clear inference quantization status");
  check(cudaMemcpyAsync(const_cast<FeatureType*>(result.view.types), types.data(), multiply(data.columns, sizeof(FeatureType)),
                         cudaMemcpyHostToDevice, stream), "upload inference feature types");
  check(cudaMemcpyAsync(const_cast<unsigned*>(result.view.offsets), offsets.data(), multiply(offsets.size(), sizeof(unsigned)),
                         cudaMemcpyHostToDevice, stream), "upload inference feature offsets");
  for (unsigned begin = 0; begin < data.columns;) {
    const unsigned count = std::min(l.features, data.columns - begin);
    upload_tile(data, begin, count, input, stream);
    for (unsigned f = 0; f < count; ++f) {
      const auto& feature = features[begin + f];
      const auto& values = feature.type == FeatureType::numeric ? feature.cuts : feature.categories;
      sizes[f] = unsigned(values.size());
      if (!values.empty()) check(cudaMemcpyAsync(metadata + std::size_t(f) * l.meta_stride, values.data(),
        multiply(values.size(), sizeof(float)), cudaMemcpyHostToDevice, stream), "upload inference feature metadata");
    }
    check(cudaMemcpyAsync(device_sizes, sizes.data(), multiply(count, sizeof(unsigned)), cudaMemcpyHostToDevice, stream), "upload inference metadata sizes");
    encode_tile<<<ceiling(data.rows, 32), dim3(32, 8), 0, stream>>>(input, metadata, device_sizes, result.view.types + begin,
      const_cast<std::uint16_t*>(result.view.bins) + std::size_t(begin) * data.rows, data.rows, count, l.meta_stride, device_status);
    check(cudaGetLastError(), "launch inference quantization encoding");
    check(cudaMemcpyAsync(&status, device_status, sizeof(unsigned), cudaMemcpyDeviceToHost, stream), "export inference quantization status");
    check(cudaStreamSynchronize(stream), "finish inference quantization tile");
    validate_status(status);
    begin += count;
  }
  drain.active = false;
  return result;
}

} // namespace ghb
