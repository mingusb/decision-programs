#pragma once
#include "ghb/deeper_histogram.cuh"
#include "ghb/resident.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace deeper_test {
inline std::size_t checks{};
inline void require(bool condition, const std::string& text) {
  ++checks;
  if (!condition) throw std::runtime_error(text);
}
inline void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
struct Stream {
  cudaStream_t value{};
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamSynchronize(value); cudaStreamDestroy(value); }
};
template<class T> struct Device {
  T* p{};
  std::size_t count{};
  explicit Device(std::size_t n) : count(n) { check(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
  ~Device() { cudaFree(p); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  void put(const std::vector<T>& host, cudaStream_t stream) {
    require(host.size() == count, "device upload shape");
    check(cudaMemcpyAsync(p, host.data(), count * sizeof(T), cudaMemcpyHostToDevice, stream));
    check(cudaStreamSynchronize(stream));
  }
  std::vector<T> get(cudaStream_t stream) const {
    std::vector<T> host(count);
    check(cudaMemcpyAsync(host.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    return host;
  }
};
struct Graph {
  cudaGraph_t value{};
  cudaGraphExec_t executable{};
  ~Graph() { if (executable) cudaGraphExecDestroy(executable); if (value) cudaGraphDestroy(value); }
};
using Policy = ghb::gpu::DeeperHistogramPolicy;
using Stats = ghb::gpu::Stats;
struct Shape {
  unsigned rows, columns, outputs, capacity, bins;
  unsigned inactive_percent{};
  bool skew{}, zero{}, irregular{}, ragged_active{};
  unsigned first{2};
  unsigned padding{3};
};
struct Variant {
  std::string name;
  bool sequential{};
  ghb::HistogramPolicy scalar{ghb::HistogramPolicy::global};
  Policy policy{Policy::global};
  unsigned chunks{};
};
struct Reference {
  long double gradient{}, hessian{}, absolute_g{}, absolute_h{};
  unsigned long long count{};
};
inline bool equal(const Stats& a, const Stats& b) {
  return a.gradient == b.gradient && a.hessian == b.hessian && a.count == b.count;
}
inline Stats sentinel() { return Stats{1234567.25, -9876543.75, 0xfedcba9876543210ULL}; }
inline void close(double actual, long double expected, long double absolute_sum,
                  unsigned long long count, const std::string& context) {
  const long double bound = 8 * (count + 1) * std::numeric_limits<double>::epsilon() * std::max(1.L, absolute_sum);
  require(std::isfinite(actual) && std::abs(static_cast<long double>(actual) - expected) <= bound, context);
}
inline unsigned hash(unsigned x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16);
}
struct Fixture {
  Shape shape;
  unsigned stride, total{};
  Stream stream;
  std::vector<unsigned> offsets, bin_counts, active;
  std::vector<std::uint16_t> encoded;
  std::vector<ghb::FeatureType> types;
  std::vector<int> assignment;
  std::vector<double> gradient, hessian;
  std::vector<Reference> expected;
  Device<std::uint16_t> d_bins;
  Device<unsigned> d_offsets, d_active;
  Device<ghb::FeatureType> d_types;
  Device<int> d_assignment;
  Device<double> d_g, d_h;
  Device<Stats> d_result;
  ghb::gpu::DataView data;

  static unsigned total_bins(const Shape& s) {
    unsigned result{};
    for (unsigned f = 0; f < s.columns; ++f)
      result += s.irregular && f % 3 ? 1 + (f * 13u) % s.bins : s.bins;
    return result;
  }
  explicit Fixture(Shape s) : shape(s), stride(s.first + s.outputs + s.padding), total(total_bins(s)),
      offsets(s.columns + 1), bin_counts(s.columns), active(s.outputs),
      encoded(std::size_t(s.rows) * s.columns), types(s.columns),
      assignment(std::size_t(s.rows) * s.outputs), gradient(std::size_t(s.rows) * stride), hessian(gradient.size()),
      expected(std::size_t(s.outputs) * s.capacity * total),
      d_bins(encoded.size()), d_offsets(offsets.size()), d_active(active.size()), d_types(types.size()),
      d_assignment(assignment.size()), d_g(gradient.size()), d_h(hessian.size()), d_result(expected.size() + 2) {
    for (unsigned f = 0; f < s.columns; ++f) {
      bin_counts[f] = s.irregular && f % 3 ? 1 + (f * 13u) % s.bins : s.bins;
      offsets[f + 1] = offsets[f] + bin_counts[f];
      types[f] = f % 2 ? ghb::FeatureType::categorical : ghb::FeatureType::numeric;
      for (unsigned row = 0; row < s.rows; ++row) {
        const unsigned bin = row % 11 == 0 ? 0 : s.skew && row % 17 ? bin_counts[f] - 1
            : hash(row + 103u * f) % bin_counts[f];
        encoded[std::size_t(f) * s.rows + row] = std::uint16_t(bin);
      }
    }
    for (unsigned row = 0; row < s.rows; ++row) for (unsigned out = 0; out < stride; ++out) {
      const auto index = std::size_t(row) * stride + out;
      const double weight = s.zero || row % 7 == 0 ? 0 : row % 3 ? .5 : 2.25;
      gradient[index] = weight * (int(hash(row * 37u + out * 93u) % 127) - 63) / 7.0;
      hessian[index] = weight * (1 + hash(row * 71u + out * 5u) % 29) / 13.0;
    }
    for (unsigned out = 0; out < s.outputs; ++out) {
      active[out] = !s.ragged_active ? s.capacity : out % 5 == 0 ? 0 : out % 5 == 1 ? s.capacity + 2 :
                    1 + hash(out) % s.capacity;
      for (unsigned row = 0; row < s.rows; ++row) {
        int node = int(hash(row * 17u + out * 719u) % s.capacity);
        if (hash(row * 43u + out * 107u) % 100 < s.inactive_percent) node = -1;
        else if (s.ragged_active && row % 29 == 0) node = int(std::min(active[out], s.capacity));
        assignment[std::size_t(out) * s.rows + row] = node;
      }
    }
    data = {d_bins.p, d_offsets.p, d_types.p, s.rows, s.columns, total, s.bins};
    d_bins.put(encoded, stream.value); d_offsets.put(offsets, stream.value); d_types.put(types, stream.value);
    d_g.put(gradient, stream.value); d_h.put(hessian, stream.value);
    d_assignment.put(assignment, stream.value); d_active.put(active, stream.value);
    reference(); reset();
  }
  void reference() {
    std::fill(expected.begin(), expected.end(), Reference{});
    for (unsigned out = 0; out < shape.outputs; ++out) for (unsigned row = 0; row < shape.rows; ++row) {
      const int node = assignment[std::size_t(out) * shape.rows + row];
      if (node < 0 || unsigned(node) >= std::min(active[out], shape.capacity)) continue;
      const auto source = std::size_t(row) * stride + shape.first + out;
      for (unsigned f = 0; f < shape.columns; ++f) {
        auto& r = expected[(std::size_t(out) * shape.capacity + unsigned(node)) * total + offsets[f] +
                           encoded[std::size_t(f) * shape.rows + row]];
        r.gradient += gradient[source]; r.hessian += hessian[source];
        r.absolute_g += std::abs(gradient[source]); r.absolute_h += std::abs(hessian[source]); ++r.count;
      }
    }
  }
  void reset() { d_result.put(std::vector<Stats>(expected.size() + 2, sentinel()), stream.value); }
  void launch(const Variant& v) {
    if (v.sequential) {
      for (unsigned out = 0; out < shape.outputs; ++out)
        check(ghb::gpu::histogram_active(data, d_assignment.p + std::size_t(out) * shape.rows,
            d_g.p, d_h.p, stride, shape.first + out, shape.capacity, d_active.p + out,
            d_result.p + 1 + std::size_t(out) * shape.capacity * total, v.scalar, stream.value));
    } else check(ghb::gpu::deeper_histogram(data, d_assignment.p, d_g.p, d_h.p, stride, shape.first,
        shape.outputs, shape.capacity, d_active.p, d_result.p + 1, v.policy, v.chunks, stream.value));
  }
  void verify(const std::string& context) {
    const auto host = d_result.get(stream.value);
    require(equal(host.front(), sentinel()) && equal(host.back(), sentinel()), context + " guards");
    for (unsigned out = 0; out < shape.outputs; ++out) for (unsigned node = 0; node < shape.capacity; ++node)
      for (unsigned bin = 0; bin < total; ++bin) {
        const auto index = (std::size_t(out) * shape.capacity + node) * total + bin;
        const auto& value = host[index + 1]; const auto& r = expected[index];
        if (node >= std::min(active[out], shape.capacity)) {
          require(equal(value, sentinel()), context + " inactive capacity changed"); continue;
        }
        require(value.count == r.count, context + " count");
        close(value.gradient, r.gradient, r.absolute_g, r.count, context + " gradient");
        close(value.hessian, r.hessian, r.absolute_h, r.count, context + " hessian");
        if (shape.zero || !r.count) require(value.gradient == 0 && value.hessian == 0, context + " zero");
      }
  }
  std::vector<Variant> variants(bool extra_chunks = true) const {
    std::vector<Variant> result{{"sequential-global", true, ghb::HistogramPolicy::global}};
    if (ghb::gpu::shared_supported(data, shape.capacity))
      result.push_back({"sequential-shared", true, ghb::HistogramPolicy::shared});
    result.push_back({"batched-global", false, ghb::HistogramPolicy::global, Policy::global});
    const unsigned base = std::min(1u + (shape.rows - 1) / 4096, 256u);
    for (const auto& item : {std::pair{Policy::shared1, "shared1"}, {Policy::shared4, "shared4"}, {Policy::shared8, "shared8"}}) {
      if (!ghb::gpu::deeper_histogram_supported(data, shape.capacity, item.first)) continue;
      result.push_back({std::string("batched-") + item.second + "-chunks" + std::to_string(base), false,
                        ghb::HistogramPolicy::global, item.first, base});
      const unsigned more = std::min(base * 4, 256u);
      if (extra_chunks && more != base)
        result.push_back({std::string("batched-") + item.second + "-chunks" + std::to_string(more), false,
                          ghb::HistogramPolicy::global, item.first, more});
    }
    return result;
  }
};
} // namespace deeper_test
