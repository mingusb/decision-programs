#include "ghb/root_histogram.cuh"
#include "ghb/resident.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::size_t checks{};
void require(bool value, const std::string& message) {
  ++checks;
  if (!value) throw std::runtime_error(message);
}
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Stream {
  cudaStream_t value{};
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamSynchronize(value); cudaStreamDestroy(value); }
};
template<class T> struct Device {
  T* p{};
  std::size_t n{};
  explicit Device(std::size_t count) : n(count) { check(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
  ~Device() { cudaFree(p); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  void put(const std::vector<T>& values, cudaStream_t stream) {
    require(values.size() == n, "upload shape");
    check(cudaMemcpyAsync(p, values.data(), n * sizeof(T), cudaMemcpyHostToDevice, stream));
    check(cudaStreamSynchronize(stream));
  }
  std::vector<T> get(cudaStream_t stream) const {
    std::vector<T> result(n);
    check(cudaMemcpyAsync(result.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    return result;
  }
};
struct Reference {
  long double g{}, h{}, absolute_g{}, absolute_h{};
  unsigned long long count{};
};
void close(double value, long double expected, long double absolute_sum,
           unsigned long long count, const char* message) {
  // Conservative forward-error bound for sequential double additions, allowing
  // either atomic ordering. The independent reference accumulates long doubles.
  const long double tolerance = 8 * (count + 1) * std::numeric_limits<double>::epsilon() *
                                std::max(1.L, absolute_sum);
  require(std::isfinite(value) && std::abs(static_cast<long double>(value) - expected) <= tolerance, message);
}
bool same(const ghb::gpu::Stats& a, const ghb::gpu::Stats& b) {
  return a.gradient == b.gradient && a.hessian == b.hessian && a.count == b.count;
}

void histogram_case(Stream& stream, unsigned rows, unsigned stride, unsigned first,
                    unsigned count, std::vector<unsigned> bins, bool skew, bool all_zero,
                    bool graph) {
  const unsigned columns = unsigned(bins.size());
  std::vector<unsigned> offsets(columns + 1);
  for (unsigned feature = 0; feature < columns; ++feature) offsets[feature + 1] = offsets[feature] + bins[feature];
  const unsigned total = offsets.back();
  std::vector<ghb::FeatureType> types(columns);
  std::vector<std::uint16_t> encoded(std::size_t(rows) * columns);
  std::vector<double> gradient(std::size_t(rows) * stride), hessian(gradient.size());
  for (unsigned feature = 0; feature < columns; ++feature) {
    types[feature] = feature % 2 ? ghb::FeatureType::categorical : ghb::FeatureType::numeric;
    for (unsigned row = 0; row < rows; ++row) {
      const unsigned value = row % 11 == 0 ? 0 : skew && row % 17 ? bins[feature] - 1
          : unsigned((std::uint64_t(row) * 103 + feature * 13) % bins[feature]);
      encoded[std::size_t(feature) * rows + row] = std::uint16_t(value);
    }
  }
  for (unsigned row = 0; row < rows; ++row) {
    for (unsigned output = 0; output < stride; ++output) {
      const auto index = std::size_t(row) * stride + output;
      const double weight = all_zero || row % 7 == 0 ? 0 : row % 3 ? .5 : 2.25;
      gradient[index] = weight * (int((row * 17 + output * 29) % 127) - 63) / 7.0;
      hessian[index] = weight * (1 + (row * 11 + output * 3) % 29) / 13.0;
    }
  }
  std::vector<Reference> expected(std::size_t(count) * total);
  for (unsigned output = 0; output < count; ++output) {
    for (unsigned row = 0; row < rows; ++row) {
      const auto source = std::size_t(row) * stride + first + output;
      for (unsigned feature = 0; feature < columns; ++feature) {
        auto& value = expected[std::size_t(output) * total + offsets[feature] + encoded[std::size_t(feature) * rows + row]];
        value.g += gradient[source]; value.h += hessian[source];
        value.absolute_g += std::abs(gradient[source]); value.absolute_h += std::abs(hessian[source]); ++value.count;
      }
    }
  }
  Device<std::uint16_t> d_bins(encoded.size()); Device<unsigned> d_offsets(offsets.size());
  Device<ghb::FeatureType> d_types(types.size());
  Device<double> d_g(gradient.size()), d_h(hessian.size());
  Device<ghb::gpu::Stats> d_result(expected.size() + 2), d_scalar(total);
  Device<int> d_assignment(rows);
  d_bins.put(encoded, stream.value); d_offsets.put(offsets, stream.value); d_types.put(types, stream.value);
  d_g.put(gradient, stream.value); d_h.put(hessian, stream.value);
  d_assignment.put(std::vector<int>(rows, 0), stream.value);
  const ghb::gpu::Stats sentinel{1234567.25, -9876543.75, 0xfedcba9876543210ULL};
  d_result.put(std::vector<ghb::gpu::Stats>(expected.size() + 2, sentinel), stream.value);
  const ghb::gpu::DataView data{d_bins.p, d_offsets.p, d_types.p, rows, columns, total,
                              *std::max_element(bins.begin(), bins.end())};
  auto run = [&] { check(ghb::gpu::root_histogram(data, d_g.p, d_h.p, stride, first, count, d_result.p + 1, stream.value)); };
  if (graph) {
    cudaGraph_t captured{}; cudaGraphExec_t executable{};
    check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
    run();
    check(cudaStreamEndCapture(stream.value, &captured));
    check(cudaGraphInstantiate(&executable, captured, 0));
    for (unsigned replay = 0; replay < 3; ++replay) check(cudaGraphLaunch(executable, stream.value));
    check(cudaStreamSynchronize(stream.value));
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(captured));
  } else { run(); run(); }
  const auto actual = d_result.get(stream.value);
  require(same(actual.front(), sentinel) && same(actual.back(), sentinel), "root histogram guard overwrite");
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const auto& reference = expected[i]; const auto& value = actual[i + 1];
    require(value.count == reference.count, "root count differs from CPU reference");
    close(value.gradient, reference.g, reference.absolute_g, reference.count, "root gradient differs from CPU reference");
    close(value.hessian, reference.h, reference.absolute_h, reference.count, "root Hessian differs from CPU reference");
    if (all_zero || !reference.count) require(value.gradient == 0 && value.hessian == 0, "zero derivatives/empty bins not zero");
  }
  // Match all selected scalar outputs for small batches, and representative
  // boundaries for wide batches; each batched cell still has its own CPU check.
  for (unsigned output = 0; output < count; ++output) {
    if (count > 17 && output != 0 && output != count - 1 && output != count / 2 && output != 31 && output != 32) continue;
    check(ghb::gpu::histogram(data, d_assignment.p, d_g.p, d_h.p, stride, first + output,
                              1, d_scalar.p, ghb::HistogramPolicy::global, stream.value));
    const auto scalar = d_scalar.get(stream.value);
    for (unsigned bin = 0; bin < total; ++bin) {
      const auto index = std::size_t(output) * total + bin;
      const auto& reference = expected[index]; const auto& value = actual[index + 1];
      require(value.count == scalar[bin].count, "batched/scalar count mismatch");
      close(value.gradient, scalar[bin].gradient, reference.absolute_g, reference.count * 2, "batched/scalar gradient mismatch");
      close(value.hessian, scalar[bin].hessian, reference.absolute_h, reference.count * 2, "batched/scalar Hessian mismatch");
    }
  }
}

void initialization(Stream& stream) {
  // Histogram copy larger than assignments, then the reverse, with guards and
  // a nonzero cache selector. A later root must never consume cache entry zero.
  for (const auto rows : {3u, 1027u}) {
    constexpr unsigned total = 519, outputs = 3;
    std::vector<ghb::gpu::Stats> cache(total * outputs);
    for (unsigned i = 0; i < cache.size(); ++i) cache[i] = {double(i) / 8, double(i) / 16, i};
    Device<ghb::gpu::Stats> d_cache(cache.size()), d_destination(total + 2);
    Device<int> assignments(rows + 2), frontier(3);
    Device<ghb::Node> nodes(2); Device<ghb::gpu::TreeState> state(1);
    Device<ghb::gpu::TreeParameters> selector(1);
    d_cache.put(cache, stream.value);
    const ghb::gpu::Stats sentinel{1234, 5678, 9012};
    d_destination.put(std::vector<ghb::gpu::Stats>(total + 2, sentinel), stream.value);
    assignments.put(std::vector<int>(rows + 2, -17), stream.value); frontier.put(std::vector<int>(3, -29), stream.value);
    const ghb::Node node_guard{-1, -1, -1, 0, 0, 123.25};
    nodes.put(std::vector<ghb::Node>(2, node_guard), stream.value);
    selector.put({ghb::gpu::TreeParameters{outputs, 1, 1, 2}}, stream.value);
    check(ghb::gpu::resident_initialize(rows, assignments.p + 1, nodes.p, frontier.p + 1, state.p, stream.value,
                                       d_cache.p, d_destination.p + 1, total, selector.p));
    const auto copied = d_destination.get(stream.value);
    const auto assigned = assignments.get(stream.value);
    require(same(copied.front(), sentinel) && same(copied.back(), sentinel), "initializer copy guard overwrite");
    for (unsigned i = 0; i < total; ++i) require(same(copied[i + 1], cache[2 * total + i]), "initializer cache selection mismatch");
    require(assigned.front() == -17 && assigned.back() == -17, "initializer assignment guard overwrite");
    for (unsigned row = 1; row <= rows; ++row) require(assigned[row] == 0, "initializer assignment mismatch");
    const auto frontiers = frontier.get(stream.value);
    const auto states = state.get(stream.value);
    const auto trees = nodes.get(stream.value);
    require(frontiers[0] == -29 && frontiers[1] == 0 && frontiers[2] == -29, "initializer frontier mismatch");
    require(states[0].active_nodes == 1 && states[0].node_count == 1 && !states[0].status &&
            !states[0].next_active_nodes && !states[0].old_node_count && !states[0].split_count, "initializer state mismatch");
    require(trees[0].feature == -1 && trees[0].left == -1 && trees[0].right == -1 && trees[0].value == 0 &&
            trees[1].value == node_guard.value, "initializer node mismatch");
    for (unsigned mask = 1; mask < 15; ++mask) {
      const auto status = ghb::gpu::resident_initialize(rows, assignments.p + 1, nodes.p, frontier.p + 1, state.p, stream.value,
          mask & 1 ? d_cache.p : nullptr, mask & 2 ? d_destination.p + 1 : nullptr,
          mask & 4 ? total : 0, mask & 8 ? selector.p : nullptr);
      require(status == cudaErrorInvalidValue, "partial cache initializer arguments accepted");
    }
    cudaGraph_t captured{}; cudaGraphExec_t executable{};
    check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
    check(ghb::gpu::resident_initialize(rows, assignments.p + 1, nodes.p, frontier.p + 1, state.p, stream.value,
                                       d_cache.p, d_destination.p + 1, total, selector.p));
    check(cudaStreamEndCapture(stream.value, &captured));
    check(cudaGraphInstantiate(&executable, captured, 0));
    for (unsigned selected : {0u, 2u, 1u}) {
      selector.put({ghb::gpu::TreeParameters{outputs, 1, 1, selected}}, stream.value);
      check(cudaGraphLaunch(executable, stream.value));
      const auto replayed = d_destination.get(stream.value);
      require(same(replayed.front(), sentinel) && same(replayed.back(), sentinel), "initializer replay guard overwrite");
      for (unsigned i = 0; i < total; ++i)
        require(same(replayed[i + 1], cache[selected * total + i]), "initializer replay used stale selector");
    }
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(captured));
    check(ghb::gpu::resident_initialize(rows, assignments.p + 1, nodes.p, frontier.p + 1, state.p, stream.value));
    check(cudaStreamSynchronize(stream.value));
  }
}

void invalid_arguments(Stream& stream) {
  Device<std::uint16_t> bins(1); Device<unsigned> offsets(2); Device<ghb::FeatureType> types(1);
  Device<double> g(1), h(1); Device<ghb::gpu::Stats> result(1);
  const ghb::gpu::DataView valid{bins.p, offsets.p, types.p, 1, 1, 1, 1};
  auto invalid = [&](ghb::gpu::DataView data, const double* gradient, const double* hessian,
                     unsigned stride, unsigned first, unsigned count, ghb::gpu::Stats* output) {
    require(ghb::gpu::root_histogram(data, gradient, hessian, stride, first, count, output, stream.value) == cudaErrorInvalidValue,
            "invalid root arguments accepted");
  };
  invalid(valid, nullptr, h.p, 1, 0, 1, result.p); invalid(valid, g.p, nullptr, 1, 0, 1, result.p);
  invalid(valid, g.p, h.p, 1, 0, 1, nullptr); invalid(valid, g.p, h.p, 0, 0, 1, result.p);
  invalid(valid, g.p, h.p, 1, 0, 0, result.p); invalid(valid, g.p, h.p, 1, 1, 1, result.p);
  invalid(valid, g.p, h.p, 7, 6, 2, result.p);
  for (unsigned test = 0; test < 10; ++test) {
    auto data = valid;
    switch (test) {
      case 0: data.rows = 0; break; case 1: data.columns = 0; break;
      case 2: data.columns = unsigned(INT32_MAX) + 1u; break; case 3: data.bins = nullptr; break;
      case 4: data.offsets = nullptr; break; case 5: data.types = nullptr; break;
      case 6: data.total_bins = 0; break; case 7: data.max_feature_bins = 0; break;
      case 8: data.max_feature_bins = 65537; data.total_bins = 65537; break;
      case 9: data.max_feature_bins = 2; break;
    }
    invalid(data, g.p, h.p, 1, 0, 1, result.p);
  }
  auto huge = valid; huge.rows = std::numeric_limits<unsigned>::max();
  invalid(huge, g.p, h.p, std::numeric_limits<unsigned>::max(), 0, 1, result.p);
  huge = valid; huge.total_bins = std::numeric_limits<unsigned>::max();
  invalid(huge, g.p, h.p, std::numeric_limits<unsigned>::max(), 0, std::numeric_limits<unsigned>::max(), result.p);
}
} // namespace

int main() {
  try {
    int devices{}; const auto status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && !devices)) {
      std::cout << "SKIP: no CUDA GPU available\n"; return 77;
    }
    check(status); Stream stream;
    invalid_arguments(stream);
    for (unsigned count : {1u, 2u, 3u, 4u, 5u, 8u, 9u, 16u, 17u, 31u, 32u, 33u, 65u, 129u}) {
      histogram_case(stream, 257, count + 7, 3, count, {1, 2, 3, 16, 17, 32, 65}, false, false, count == 17 || count == 65);
      histogram_case(stream, 33, count, 0, count, {1, 2, 16, 32}, true, false, false);
    }
    histogram_case(stream, 1, 35, 1, 33, {1, 3, 257}, false, true, false);
    histogram_case(stream, 4099, 21, 2, 17, {1, 16, 32, 256}, true, false, false);
    histogram_case(stream, 1031, 69, 1, 65, {1, 17}, true, true, true);
    histogram_case(stream, 513, 5, 2, 3, {65536}, false, false, false);
    initialization(stream);
    std::cout << "root histogram: " << checks << " checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "root histogram failure: " << error.what() << '\n'; return 1;
  }
}
