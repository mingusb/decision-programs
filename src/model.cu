#include "gh/model.cuh"
#include <cuda/std/bit>
#include <cmath>

namespace gh {
namespace {
constexpr u64 budget = 1ULL << 30;
constexpr u32 threads = 256;
template<class T> __device__ bool enough(Array<T> a, u64 n) {
  return mul_fits(n, sizeof(T)) && contains(a, n);
}
__device__ unsigned grid(u64 n) { return unsigned(n ? (ceil_div(n, threads) < 65535 ? ceil_div(n, threads) : 65535) : 1); }
__device__ unsigned tree_grid(u64 n) { return unsigned(n && n < 65535 ? n : n ? 65535 : 1); }
__device__ void launch_status(Status* s) { if (cudaGetLastError() != cudaSuccess) fail(s, runtime); }
__device__ bool finite(double x) { return (cuda::std::bit_cast<u64>(x) & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL; }
__device__ u32 feature_bins(Feature f) { return f.count + (f.type == FeatureType::numeric ? 2 : 1); }
__device__ bool header(const Model* p, Status* s) {
  if (!p || reinterpret_cast<std::uintptr_t>(p) % alignof(Model)) { fail(s, shape); return false; }
  const Model& m = *p;
  if (u32(m.objective) > 2 || !m.outputs || m.outputs > INT32_MAX ||
      (m.objective == Objective::multiclass_softmax && m.outputs < 2) ||
      !m.schema.columns || m.schema.columns > INT32_MAX ||
      !m.schema.max_feature_bins || m.schema.max_feature_bins > 65536) {
    fail(s, model); return false;
  }
  if (!enough(m.base, m.outputs) || !enough(m.nodes, m.node_count) ||
      !enough(m.trees, m.tree_count) || !enough(m.output_offsets, u64(m.outputs) + 1) ||
      !enough(m.schema.features, m.schema.columns) ||
      !enough(m.schema.offsets, u64(m.schema.columns) + 1) ||
      !enough(m.schema.metadata, m.schema.metadata_count)) { fail(s, capacity); return false; }
  return true;
}

// Nonnegative binary floating accumulator with a 64-bit significand. Integer
// guard/sticky rounding reproduces the archived binary80 addition precision.
struct Bound { u64 mantissa{}; int exponent{}; };
__device__ Bound positive(u64 bits) {
  const u64 fraction = bits & 0xfffffffffffffULL;
  const int exponent = int(bits >> 52);
  if (exponent) return {((1ULL << 52) | fraction) << 11, exponent - 1086};
  if (!fraction) return {};
  const int shift = __clzll(fraction);
  return {fraction << shift, -1074 - shift};
}
__device__ Bound add_bound(Bound a, Bound b) {
  if (!a.mantissa) return b;
  if (!b.mantissa) return a;
  if (a.exponent < b.exponent) { const auto t = a; a = b; b = t; }
  const unsigned d = unsigned(a.exponent - b.exponent);
  const u64 integer = d < 64 ? b.mantissa >> d : 0;
  const u64 sum = a.mantissa + integer;
  const bool carry = sum < a.mantissa;
  const bool remainder = d >= 64 ? b.mantissa != 0 : d && (b.mantissa & ((1ULL << d) - 1));
  bool guard = false, sticky = false;
  if (carry) {
    guard = sum & 1; sticky = remainder;
    a.mantissa = (1ULL << 63) | (sum >> 1); ++a.exponent;
  } else {
    a.mantissa = sum;
    if (d && d <= 64) {
      guard = (b.mantissa >> (d - 1)) & 1;
      sticky = d > 1 && (b.mantissa & ((1ULL << (d - 1)) - 1));
    }
  }
  if (guard && (sticky || (a.mantissa & 1)) && !++a.mantissa) {
    a.mantissa = 1ULL << 63; ++a.exponent;
  }
  return a;
}

struct ValidationMemory { u32* parent; u32* next; u64* maximum; u32* control; };
__device__ ValidationMemory validation_memory(Arena& a, const Model& m) {
  return {a.take<u32>(m.node_count), a.take<u32>(m.node_count),
          a.take<u64>(m.tree_count), a.take<u32>(2)};
}
__global__ void initialize_validation(Model m, ValidationMemory w) {
  const u64 first = u64(blockIdx.x) * blockDim.x + threadIdx.x, stride = u64(gridDim.x) * blockDim.x;
  for (u64 i = first; i < m.node_count; i += stride) w.parent[i] = 0;
  for (u64 i = first; i < m.tree_count; i += stride) w.maximum[i] = 0;
  if (first < 2) w.control[first] = 0;
}
__global__ void validate_schema(Model m, ValidationMemory w, Status* s) {
  const u64 first = u64(blockIdx.x) * blockDim.x + threadIdx.x, stride = u64(gridDim.x) * blockDim.x;
  for (u64 i = first; i < m.outputs; i += stride) {
    if (!finite(m.base.data[i]) || m.output_offsets.data[i] > m.output_offsets.data[i + 1] ||
        m.output_offsets.data[i + 1] > m.tree_count) fail(s, model);
  }
  if (!first && (m.output_offsets.data[0] || m.output_offsets.data[m.outputs] != m.tree_count ||
      m.schema.offsets.data[0] || m.schema.offsets.data[m.schema.columns] != m.schema.total_bins)) fail(s, model);
  for (u64 i = first; i < m.schema.columns; i += stride) {
    const auto f = m.schema.features.data[i];
    const u32 extra = f.type == FeatureType::numeric ? 2 : 1;
    if (u32(f.type) > 1 || f.count > 65536 - extra || f.begin > m.schema.metadata_count ||
        f.count > m.schema.metadata_count - f.begin) { fail(s, model); continue; }
    if ((!i && f.begin) || (i && (!add_fits(m.schema.features.data[i - 1].begin,
        m.schema.features.data[i - 1].count) || f.begin != m.schema.features.data[i - 1].begin +
        m.schema.features.data[i - 1].count)) || (i + 1 == m.schema.columns &&
        f.begin + f.count != m.schema.metadata_count) ||
        u64(m.schema.offsets.data[i]) + f.count + extra != m.schema.offsets.data[i + 1]) fail(s, model);
    atomicMax(w.control + 1, f.count + extra);
    for (u32 j = 0; j < f.count; ++j) {
      const float value = m.schema.metadata.data[f.begin + j];
      if (!finite(value) || (j && !(m.schema.metadata.data[f.begin + j - 1] < value))) fail(s, model);
    }
  }
}
__global__ void claim_segments(Model m, ValidationMemory w, Status* s) {
  if (s->errors) return;
  for (u64 t = blockIdx.x; t < m.tree_count; t += gridDim.x) {
    const Tree tree = m.trees.data[t];
    if (!tree.count || tree.count > INT32_MAX || tree.output >= m.outputs ||
        tree.begin > m.node_count || tree.count > m.node_count - tree.begin ||
        t < m.output_offsets.data[tree.output] || t >= m.output_offsets.data[tree.output + 1]) {
      if (!threadIdx.x) fail(s, model); continue;
    }
    if (!threadIdx.x) atomicMax(w.control, tree.count);
    for (u64 j = threadIdx.x; j < tree.count; j += blockDim.x)
      if (atomicCAS(w.parent + tree.begin + j, 0, 1)) fail(s, model);
  }
}
__global__ void check_coverage(Model m, ValidationMemory w, Status* s) {
  if (s->errors) return;
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < m.node_count; i += u64(gridDim.x) * blockDim.x) {
    if (w.parent[i] != 1) fail(s, model);
    w.parent[i] = UINT32_MAX;
  }
}
__global__ void claim_parents(Model m, ValidationMemory w, Status* s) {
  if (s->errors) return;
  for (u64 t = blockIdx.x; t < m.tree_count; t += gridDim.x) {
    const Tree tree = m.trees.data[t];
    for (u64 j = threadIdx.x; j < tree.count; j += blockDim.x) {
      const Node n = m.nodes.data[tree.begin + j];
      if (!j && atomicCAS(w.parent + tree.begin, UINT32_MAX, 0) != UINT32_MAX) fail(s, model);
      if (!finite(n.value) || n.missing_left > 1) { fail(s, model); continue; }
      if (n.feature == -1) {
        if (n.left != -1 || n.right != -1 || n.threshold) fail(s, model);
        atomicMax(reinterpret_cast<unsigned long long*>(w.maximum + t),
                  static_cast<unsigned long long>(cuda::std::bit_cast<u64>(n.value) & 0x7fffffffffffffffULL));
      } else {
        if (n.feature < 0 || u32(n.feature) >= m.schema.columns || n.left < 0 || n.right < 0 ||
            u32(n.left) >= tree.count || u32(n.right) >= tree.count ||
            n.threshold >= feature_bins(m.schema.features.data[n.feature])) { fail(s, model); continue; }
        if (atomicCAS(w.parent + tree.begin + u32(n.left), UINT32_MAX, u32(j)) != UINT32_MAX ||
            atomicCAS(w.parent + tree.begin + u32(n.right), UINT32_MAX, u32(j)) != UINT32_MAX) fail(s, model);
      }
    }
  }
}
__global__ void jump_parents(Model m, const u32* parent, u32* next, Status* s) {
  if (s->errors) return;
  for (u64 t = blockIdx.x; t < m.tree_count; t += gridDim.x) {
    const auto tree = m.trees.data[t];
    for (u64 j = threadIdx.x; j < tree.count; j += blockDim.x) {
      const u32 p = parent[tree.begin + j];
      if (p >= tree.count) { fail(s, model); continue; }
      next[tree.begin + j] = parent[tree.begin + p];
    }
  }
}
__global__ void check_graph_bounds(Model m, ValidationMemory w, const u32* parent, Status* s) {
  if (s->errors) return;
  const u64 first = u64(blockIdx.x) * blockDim.x + threadIdx.x, stride = u64(gridDim.x) * blockDim.x;
  for (u64 i = first; i < m.node_count; i += stride) if (parent[i]) fail(s, model);
  if (!first && w.control[1] != m.schema.max_feature_bins) fail(s, model);
  for (u64 output = first; output < m.outputs; output += stride) {
    Bound bound = positive(cuda::std::bit_cast<u64>(m.base.data[output]) & 0x7fffffffffffffffULL);
    for (u64 t = m.output_offsets.data[output]; t < m.output_offsets.data[output + 1]; ++t) {
      bound = add_bound(bound, positive(w.maximum[t]));
      if (bound.exponent > 960 || (bound.exponent == 960 && bound.mantissa > 0xfffffffffffff800ULL)) {
        fail(s, numeric); break;
      }
    }
  }
}
__global__ void finish_validation(Model m, ValidationMemory w, Status* s) {
  if (s->errors) return;
  u32* parent = w.parent;
  u32* next = w.next;
  for (u64 span = 1; span < w.control[0]; span *= 2) {
    jump_parents<<<tree_grid(m.tree_count), threads>>>(m, parent, next, s);
    launch_status(s);
    const auto temporary = parent; parent = next; next = temporary;
  }
  check_graph_bounds<<<grid(m.node_count > m.outputs ? m.node_count : m.outputs), threads>>>(m, w, parent, s);
  launch_status(s);
}

__global__ void check_bins(Model m, const std::uint16_t* bins, u32 rows, Status* s) {
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < u64(rows) * m.schema.columns;
       i += u64(gridDim.x) * blockDim.x)
    if (u32(bins[i]) >= feature_bins(m.schema.features.data[i / rows])) fail(s, input);
}
__global__ void margins(Model m, const std::uint16_t* bins, u32 rows, double* out, Status* s) {
  if (s->errors) return;
  for (u64 task = u64(blockIdx.x) * blockDim.x + threadIdx.x; task < u64(rows) * m.outputs;
       task += u64(gridDim.x) * blockDim.x) {
    const u32 output = u32(task / rows), row = u32(task % rows);
    double value = m.base.data[output];
    for (u64 t = m.output_offsets.data[output]; t < m.output_offsets.data[output + 1]; ++t) {
      const Tree tree = m.trees.data[t];
      u32 node = 0;
      for (u32 step = 0; step < tree.count; ++step) {
        const auto n = m.nodes.data[tree.begin + node];
        if (n.feature == -1) { value = __dadd_rn(value, n.value); break; }
        const auto bin = bins[u64(n.feature) * rows + row];
        const bool left = !bin ? n.missing_left != 0 :
            m.schema.features.data[n.feature].type == FeatureType::numeric ? bin <= n.threshold : bin == n.threshold;
        node = u32(left ? n.left : n.right);
      }
    }
    out[u64(row) * m.outputs + output] = value;
  }
}
__global__ void sigmoid(double* out, u64 size, Status* s) {
  if (s->errors) return;
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < size; i += u64(gridDim.x) * blockDim.x) {
    const double x = out[i], e = exp(x >= 0 ? -x : x);
    out[i] = x >= 0 ? 1.0 / (1.0 + e) : e / (1.0 + e);
  }
}
__global__ void softmax(double* out, u32 rows, u32 outputs, Status* s) {
  if (s->errors) return;
  const u32 lane = threadIdx.x % 32;
  for (u64 row = u64(blockIdx.x) * 8 + threadIdx.x / 32; row < rows; row += u64(gridDim.x) * 8) {
    const u64 begin = row * outputs;
    double maximum = -INFINITY;
    for (u64 column = lane; column < outputs; column += 32) maximum = fmax(maximum, out[begin + column]);
    for (unsigned offset = 16; offset; offset /= 2) maximum = fmax(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    double denominator = 0;
    for (u64 column = lane; column < outputs; column += 32)
      denominator = __dadd_rn(denominator, exp(out[begin + column] - maximum));
    for (unsigned offset = 16; offset; offset /= 2) denominator = __dadd_rn(denominator, __shfl_down_sync(0xffffffff, denominator, offset));
    denominator = __shfl_sync(0xffffffff, denominator, 0);
    for (u64 column = lane; column < outputs; column += 32)
      out[begin + column] = exp(out[begin + column] - maximum) / denominator;
  }
}
__global__ void final_values(const double* out, u64 size, Status* s) {
  if (s->errors) return;
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < size; i += u64(gridDim.x) * blockDim.x)
    if (!finite(out[i])) fail(s, numeric);
}

struct Reader {
  Array<const std::byte> bytes;
  u64 position{};
  bool valid{true};
  __device__ u64 get(unsigned width) {
    if (position > bytes.size || width > bytes.size - position) { valid = false; return 0; }
    u64 value = 0;
    for (unsigned j = 0; j < width; ++j) value |= u64(bytes.data[position++]) << (8 * j);
    return value;
  }
  __device__ void skip(u64 n) {
    if (position > bytes.size || n > bytes.size - position) valid = false;
    else position += n;
  }
};
__device__ u64 load(Array<const std::byte> b, u64 at, unsigned width) { Reader r{b, at}; return r.get(width); }
__device__ void store(Array<std::byte> b, u64 at, u64 value, unsigned width) {
  for (unsigned j = 0; j < width; ++j) b.data[at + j] = std::byte((value >> (8 * j)) & 255);
}
struct Frame { u32 objective{}, outputs{}, columns{}; u64 trees{}, metadata{}, nodes{}; };
__device__ bool account(u64& total, u64 count, u64 width) {
  if (!mul_fits(count, width) || count * width > budget - total) return false;
  total += count * width; return true;
}
__device__ bool frame(Array<const std::byte> bytes, Frame& f) {
  if (!contains(bytes, bytes.size) || bytes.size < 32 || bytes.size > budget) return false;
  Reader r{bytes};
  if (r.get(8) != 0x4c45444f4d424847ULL || r.get(4) != 1) return false;
  f.objective = u32(r.get(4)); f.outputs = u32(r.get(4)); f.columns = u32(r.get(4)); f.trees = r.get(8);
  if (f.objective > 2 || !f.outputs || f.outputs > INT32_MAX || !f.columns || f.columns > INT32_MAX ||
      (f.objective == 2 && f.outputs < 2)) return false;
  u64 allocation = 0;
  if (!account(allocation, f.outputs, 8) || !account(allocation, f.columns, 56) || !account(allocation, f.trees, 32)) return false;
  r.skip(u64(f.outputs) * 8);
  for (u32 i = 0; i < f.columns && r.valid; ++i) {
    const u32 type = u32(r.get(4)), cuts = u32(r.get(4)), categories = u32(r.get(4));
    if (type > 1 || cuts > 65534 || categories > 65535 || (type ? cuts : categories)) return false;
    const u32 count = cuts + categories;
    if (!account(allocation, count, 4)) return false;
    f.metadata += count; r.skip(u64(count) * 4);
  }
  for (u64 t = 0; t < f.trees && r.valid; ++t) {
    const u32 output = u32(r.get(4)), count = u32(r.get(4));
    if (output >= f.outputs || !count || count > INT32_MAX || !account(allocation, count, 32)) return false;
    f.nodes += count; r.skip(u64(count) * 28);
  }
  return r.valid && r.position == bytes.size;
}
struct WireTree { u64 position, begin; u32 count, output; };
struct CodecMemory { u64* features; WireTree* trees; u64* cursor; };
__device__ CodecMemory codec_memory(Arena& a, u32 columns, u64 trees, u32 outputs) {
  return {a.take<u64>(columns), a.take<WireTree>(trees), a.take<u64>(outputs)};
}
template<bool Decode> __global__ void codec_base(Model m, Array<std::byte> bytes) {
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < m.outputs; i += u64(gridDim.x) * blockDim.x) {
    if constexpr (Decode) m.base.data[i] = cuda::std::bit_cast<double>(load({bytes.data, bytes.size}, 32 + 8 * i, 8));
    else store(bytes, 32 + 8 * i, cuda::std::bit_cast<u64>(m.base.data[i]), 8);
  }
}
template<bool Decode> __global__ void codec_features(Model m, Array<std::byte> bytes, const u64* positions) {
  for (u64 f = blockIdx.x; f < m.schema.columns; f += gridDim.x) {
    const Feature feature = m.schema.features.data[f];
    const u64 at = positions[f];
    if constexpr (!Decode) if (!threadIdx.x) {
      store(bytes, at, u32(feature.type), 4);
      store(bytes, at + 4, feature.type == FeatureType::numeric ? feature.count : 0, 4);
      store(bytes, at + 8, feature.type == FeatureType::categorical ? feature.count : 0, 4);
    }
    for (u64 j = threadIdx.x; j < feature.count; j += blockDim.x) {
      if constexpr (Decode) m.schema.metadata.data[feature.begin + j] = cuda::std::bit_cast<float>(u32(load({bytes.data, bytes.size}, at + 12 + 4 * j, 4)));
      else store(bytes, at + 12 + 4 * j, cuda::std::bit_cast<u32>(m.schema.metadata.data[feature.begin + j]), 4);
    }
  }
}
template<bool Decode> __global__ void codec_nodes(Model m, Array<std::byte> bytes, const WireTree* wire) {
  for (u64 t = blockIdx.x; t < m.tree_count; t += gridDim.x) {
    const auto tree = wire[t];
    if constexpr (!Decode) if (!threadIdx.x) { store(bytes, tree.position, tree.output, 4); store(bytes, tree.position + 4, tree.count, 4); }
    for (u64 j = threadIdx.x; j < tree.count; j += blockDim.x) {
      const u64 at = tree.position + 8 + j * 28;
      if constexpr (Decode) {
        const Array<const std::byte> b{bytes.data, bytes.size};
        m.nodes.data[tree.begin + j] = {cuda::std::bit_cast<std::int32_t>(u32(load(b, at, 4))),
          cuda::std::bit_cast<std::int32_t>(u32(load(b, at + 4, 4))), cuda::std::bit_cast<std::int32_t>(u32(load(b, at + 8, 4))),
          u32(load(b, at + 12, 4)), u32(load(b, at + 16, 4)), cuda::std::bit_cast<double>(load(b, at + 20, 8))};
      } else {
        const auto n = m.nodes.data[tree.begin + j];
        store(bytes, at, cuda::std::bit_cast<u32>(n.feature), 4); store(bytes, at + 4, cuda::std::bit_cast<u32>(n.left), 4);
        store(bytes, at + 8, cuda::std::bit_cast<u32>(n.right), 4); store(bytes, at + 12, n.threshold, 4);
        store(bytes, at + 16, n.missing_left, 4); store(bytes, at + 20, cuda::std::bit_cast<u64>(n.value), 8);
      }
    }
  }
}
template<bool Decode> __device__ void codec_payload(Model m, Array<std::byte> b, CodecMemory w, Status* s) {
  codec_base<Decode><<<grid(m.outputs), threads>>>(m, b); launch_status(s);
  codec_features<Decode><<<tree_grid(m.schema.columns), threads>>>(m, b, w.features); launch_status(s);
  if (m.tree_count) { codec_nodes<Decode><<<tree_grid(m.tree_count), threads>>>(m, b, w.trees); launch_status(s); }
}
}

__device__ cudaError_t validate_model(const Model* p, Workspace workspace, Status* s) {
  if (!s) return cudaErrorInvalidValue;
  if (!header(p, s)) return finish(s);
  const Model m = *p;
  Arena arena{workspace}; const auto w = validation_memory(arena, m);
  if (!arena.fits(s)) return finish(s);
  initialize_validation<<<grid(m.node_count > m.tree_count ? m.node_count : m.tree_count), threads>>>(m, w); launch_status(s);
  validate_schema<<<grid(m.outputs > m.schema.columns ? m.outputs : m.schema.columns), threads>>>(m, w, s); launch_status(s);
  claim_segments<<<tree_grid(m.tree_count), threads>>>(m, w, s); launch_status(s);
  check_coverage<<<grid(m.node_count), threads>>>(m, w, s); launch_status(s);
  claim_parents<<<tree_grid(m.tree_count), threads>>>(m, w, s); launch_status(s);
  finish_validation<<<1, 1, 0, cudaStreamTailLaunch>>>(m, w, s); launch_status(s);
  return finish(s);
}
__device__ cudaError_t predict(const Model* p, Array<const std::uint16_t> bins, u32 rows,
                             Array<double> out, bool raw, Status* s) {
  if (!s) return cudaErrorInvalidValue;
  if (!header(p, s)) return finish(s);
  const Model m = *p;
  if (!rows) return finish(s);
  const u64 size = u64(rows) * m.outputs;
  if (!enough(bins, u64(rows) * m.schema.columns) || !enough(out, size)) { fail(s, capacity); return finish(s); }
  check_bins<<<grid(u64(rows) * m.schema.columns), threads>>>(m, bins.data, rows, s); launch_status(s);
  margins<<<grid(size), threads>>>(m, bins.data, rows, out.data, s); launch_status(s);
  if (!raw && m.objective == Objective::binary_logistic) { sigmoid<<<grid(size), threads>>>(out.data, size, s); launch_status(s); }
  if (!raw && m.objective == Objective::multiclass_softmax) { softmax<<<grid(u64(rows) * 32), threads>>>(out.data, rows, m.outputs, s); launch_status(s); }
  final_values<<<grid(size), threads>>>(out.data, size, s); launch_status(s);
  return finish(s);
}
__device__ cudaError_t decode_model(Array<const std::byte> bytes, Model* p, Workspace workspace, Status* s) {
  if (!s) return cudaErrorInvalidValue;
  Frame f;
  if (!p || reinterpret_cast<std::uintptr_t>(p) % alignof(Model)) { fail(s, shape); return finish(s); }
  if (!frame(bytes, f)) { fail(s, model); return finish(s); }
  if (!enough(p->base, f.outputs) || !enough(p->schema.features, f.columns) ||
      !enough(p->schema.metadata, f.metadata) || !enough(p->schema.offsets, u64(f.columns) + 1) ||
      !enough(p->trees, f.trees) || !enough(p->nodes, f.nodes) || !enough(p->output_offsets, u64(f.outputs) + 1)) {
    fail(s, capacity); return finish(s);
  }
  Arena arena{workspace}; const auto w = codec_memory(arena, f.columns, f.trees, f.outputs);
  Model m = *p; m.outputs = f.outputs; m.objective = Objective(f.objective); m.tree_count = f.trees; m.node_count = f.nodes;
  m.schema.columns = f.columns; m.schema.metadata_count = f.metadata;
  Arena validation{workspace}; (void)validation_memory(validation, m);
  const u64 required = arena.used > validation.used ? arena.used : validation.used;
  if (!arena.valid || !validation.valid) { fail(s, extent); return finish(s); }
  Arena combined{workspace}; combined.used = required;
  if (!combined.fits(s)) return finish(s);
  Reader r{bytes, 32 + u64(f.outputs) * 8};
  u64 metadata_begin = 0, node_begin = 0, total_bins = 0;
  m.schema.max_feature_bins = 0;
  for (u32 i = 0; i < f.columns; ++i) {
    w.features[i] = r.position;
    const auto type = FeatureType(u32(r.get(4))); const u32 cuts = u32(r.get(4)), categories = u32(r.get(4));
    const u32 count = cuts + categories, nbins = count + (type == FeatureType::numeric ? 2 : 1);
    m.schema.features.data[i] = {metadata_begin, count, type};
    m.schema.offsets.data[i] = u32(total_bins);
    metadata_begin += count; total_bins += nbins;
    if (nbins > m.schema.max_feature_bins) m.schema.max_feature_bins = nbins;
    r.skip(u64(count) * 4);
  }
  if (total_bins > UINT32_MAX) { fail(s, model); return finish(s); }
  m.schema.total_bins = u32(total_bins); m.schema.offsets.data[f.columns] = u32(total_bins);
  for (u32 o = 0; o < f.outputs; ++o) { m.output_offsets.data[o] = 0; w.cursor[o] = 0; }
  m.output_offsets.data[f.outputs] = 0;
  for (u64 t = 0; t < f.trees; ++t) {
    const u64 at = r.position; const u32 output = u32(r.get(4)), count = u32(r.get(4));
    w.trees[t] = {at, node_begin, count, output}; node_begin += count;
    ++m.output_offsets.data[output + 1]; r.skip(u64(count) * 28);
  }
  for (u32 o = 0; o < f.outputs; ++o) m.output_offsets.data[o + 1] += m.output_offsets.data[o];
  for (u64 t = 0; t < f.trees; ++t) {
    const auto tree = w.trees[t];
    m.trees.data[m.output_offsets.data[tree.output] + w.cursor[tree.output]++] = {tree.begin, tree.count, tree.output};
  }
  *p = m;
  codec_payload<true>(m, {const_cast<std::byte*>(bytes.data), bytes.size}, w, s);
  // Same-thread NULL-stream ordering completes payload readers before validation
  // reuses their framing arena. Header/descriptor values were written above.
  const auto result = validate_model(p, workspace, s);
  s->required_bytes = required;
  return result;
}
__device__ cudaError_t encode_model(const Model* p, Array<std::byte> bytes, Array<u64> written,
                                  Workspace workspace, Status* s) {
  if (!s) return cudaErrorInvalidValue;
  if (!header(p, s)) return finish(s);
  const Model m = *p;
  Arena arena{workspace}; const auto w = codec_memory(arena, m.schema.columns, m.tree_count, 0);
  if (!arena.fits(s)) return finish(s);
  u64 position = 32;
  if (!account(position, m.outputs, 8)) { fail(s, extent); return finish(s); }
  for (u32 i = 0; i < m.schema.columns; ++i) {
    w.features[i] = position;
    if (!account(position, 1, 12) || !account(position, m.schema.features.data[i].count, 4)) { fail(s, extent); return finish(s); }
  }
  for (u64 t = 0; t < m.tree_count; ++t) {
    const Tree tree = m.trees.data[t]; w.trees[t] = {position, tree.begin, tree.count, tree.output};
    if (!account(position, 1, 8) || !account(position, tree.count, 28)) { fail(s, extent); return finish(s); }
  }
  if (!contains(bytes, position) || !contains(written, 1)) { fail(s, capacity); return finish(s); }
  store(bytes, 0, 0x4c45444f4d424847ULL, 8); store(bytes, 8, 1, 4); store(bytes, 12, u32(m.objective), 4);
  store(bytes, 16, m.outputs, 4); store(bytes, 20, m.schema.columns, 4); store(bytes, 24, m.tree_count, 8);
  written.data[0] = position;
  codec_payload<false>(m, bytes, w, s);
  return finish(s);
}
}
