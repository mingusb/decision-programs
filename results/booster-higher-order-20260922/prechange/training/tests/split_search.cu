#include "ghb/split_search.cuh"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using ghb::gpu::Split;
using ghb::gpu::Stats;
std::size_t checks{};
void require(bool condition, const std::string& message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
struct Stream {
  cudaStream_t value{};
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamSynchronize(value); cudaStreamDestroy(value); }
};
template<class T> struct Device {
  T* value{};
  std::size_t size{};
  explicit Device(std::size_t count) : size(count) { check(cudaMalloc(reinterpret_cast<void**>(&value), count * sizeof(T))); }
  ~Device() { cudaFree(value); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  void put(const std::vector<T>& host, cudaStream_t stream) {
    require(host.size() == size, "buffer size");
    check(cudaMemcpyAsync(value, host.data(), size * sizeof(T), cudaMemcpyHostToDevice, stream));
    check(cudaStreamSynchronize(stream));
  }
  std::vector<T> get(cudaStream_t stream) const {
    std::vector<T> host(size);
    check(cudaMemcpyAsync(host.data(), value, size * sizeof(T), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    return host;
  }
};
void equal(const Split& actual, const Split& reference, const std::string& context) {
  require(actual.feature == reference.feature, context + " feature");
  require(actual.threshold == reference.threshold, context + " threshold");
  require(actual.missing_left == reference.missing_left, context + " missing direction");
  const double a[]{actual.gain, actual.value, actual.left_value, actual.right_value};
  const double b[]{reference.gain, reference.value, reference.left_value, reference.right_value};
  for (unsigned i = 0; i < 4; ++i)
    require(std::bit_cast<std::uint64_t>(a[i]) == std::bit_cast<std::uint64_t>(b[i]),
            context + " floating field " + std::to_string(i));
}
Split sentinel() { return Split{1234567, 7654321, 37, -101.25, -0.0, 55.125, -31.75}; }

struct Fixture {
  unsigned columns, bins, capacity;
  Stream stream;
  Device<std::uint16_t> data_bins;
  Device<unsigned> offsets, active;
  Device<ghb::FeatureType> types;
  Device<Stats> histogram;
  Device<Split> old_candidates, new_candidates, old_winners, new_winners;
  ghb::gpu::DataView data{};
  std::vector<unsigned> host_offsets;
  Fixture(unsigned c, unsigned b, unsigned n, bool irregular = false)
      : columns(c), bins(b), capacity(n), data_bins(1), offsets(c + 1), active(1), types(c),
        histogram(std::size_t(n) * c * b), old_candidates(std::size_t(n) * c + 2),
        new_candidates(std::size_t(n) * c + 2), old_winners(n + 2), new_winners(n + 2), host_offsets(c + 1) {
    for (unsigned f = 0; f < c; ++f)
      host_offsets[f + 1] = host_offsets[f] + (irregular && f + 1 != c ? 1 + (f * 13) % b : b);
    offsets.put(host_offsets, stream.value);
    std::vector<ghb::FeatureType> host_types(c);
    for (unsigned f = 0; f < c; ++f) host_types[f] = f % 2 ? ghb::FeatureType::categorical : ghb::FeatureType::numeric;
    types.put(host_types, stream.value);
    data = {data_bins.value, offsets.value, types.value, 256, c, host_offsets.back(), b};
    reset(n);
  }
  void reset(unsigned count) {
    active.put({count}, stream.value);
    old_candidates.put(std::vector<Split>(old_candidates.size, sentinel()), stream.value);
    new_candidates.put(std::vector<Split>(new_candidates.size, sentinel()), stream.value);
    old_winners.put(std::vector<Split>(old_winners.size, sentinel()), stream.value);
    new_winners.put(std::vector<Split>(new_winners.size, sentinel()), stream.value);
  }
  void launch(bool warp, ghb::gpu::SplitConfig config, bool force_leaf = false) {
    if (warp) check(ghb::gpu::find_splits_warp_active(data, histogram.value, capacity, active.value, config,
                                                     force_leaf, new_candidates.value + 1, new_winners.value + 1, stream.value));
    else check(ghb::gpu::find_splits_active(data, histogram.value, capacity, active.value, config,
                                           force_leaf, old_candidates.value + 1, old_winners.value + 1, stream.value));
  }
  void launch_roots(bool warp, bool batched, unsigned count, ghb::gpu::SplitConfig config,
                    bool force_leaf = false) {
    if (batched) {
      const auto search = warp ? ghb::gpu::find_splits_warp : ghb::gpu::find_splits;
      check(search(data, histogram.value, count, config, force_leaf,
                   new_candidates.value + 1, new_winners.value + 1, stream.value));
    } else {
      const auto search = warp ? ghb::gpu::find_splits_warp_active : ghb::gpu::find_splits_active;
      for (unsigned output = 0; output < count; ++output)
        check(search(data, histogram.value + std::size_t(output) * data.total_bins, 1, active.value,
                     config, force_leaf, old_candidates.value + 1 + std::size_t(output) * columns,
                     old_winners.value + 1 + output, stream.value));
    }
  }
  void compare(unsigned count, const std::string& label) {
    const auto ac = new_candidates.get(stream.value), bc = old_candidates.get(stream.value);
    const auto aw = new_winners.get(stream.value), bw = old_winners.get(stream.value);
    for (std::size_t i = 0; i < ac.size(); ++i) {
      equal(ac[i], bc[i], label + " candidate " + std::to_string(i));
      if (!i || i > std::size_t(std::min(count, capacity)) * columns) equal(ac[i], sentinel(), label + " untouched candidate");
    }
    for (std::size_t i = 0; i < aw.size(); ++i) {
      equal(aw[i], bw[i], label + " winner " + std::to_string(i));
      if (!i || i > std::min(count, capacity)) equal(aw[i], sentinel(), label + " untouched winner");
    }
  }
  void populate(unsigned mode, std::uint64_t seed = 0x578f37915ULL) {
    std::vector<Stats> stats(histogram.size);
    auto random = [&] { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };
    for (unsigned node = 0; node < capacity; ++node) for (unsigned f = 0; f < columns; ++f) {
      const unsigned count = host_offsets[f + 1] - host_offsets[f];
      for (unsigned b = 0; b < count; ++b) {
        const auto index = std::size_t(node) * data.total_bins + host_offsets[f] + b;
        const auto rows = (256 * (b + 1)) / count - (256 * b) / count;
        const int numerator = int(random() % 193) - 96;
        Stats value{double(numerator) / 7.0, double(1 + random() % 31) / 13.0, rows};
        if (mode == 1) value = {b % 2 ? -0.0 : 0.0, b % 3 ? 0.0 : -0.0, rows};
        if (mode == 2) {
          const double magnitudes[]{0x1p53, 1.0, -0x1p53, -1.0, 0x1p-40, -0x1p-40, 0.0};
          value.gradient = magnitudes[b % 7];
          value.hessian = b % 3 == 0 ? 0.0 : double(rows);
        }
        if (mode == 3) value = b == 0 ? Stats{7.5, 3.0, 256} : Stats{}; // All rows missing.
        if (mode == 4) value = b % 5 == 0 ? Stats{0.0, 0.0, rows} : value; // Zero-weight rows still count.
        if (mode == 5) value = {b < count / 2 ? -double(rows) : double(rows), double(rows), rows};
        if (mode == 6) value = {double(numerator) * 0x1p-900, double(rows) * 0x1p-900, rows};
        stats[index] = value;
      }
    }
    histogram.put(stats, stream.value);
  }
};

void cases() {
  for (unsigned bins : {1u, 2u, 16u, 31u, 32u, 33u}) for (unsigned columns : {1u, 16u, 32u, 33u}) {
    Fixture f(columns, bins, 5, columns % 2 == 0);
    for (unsigned mode = 0; mode < 7; ++mode) {
      f.populate(mode);
      for (unsigned variation = 0; variation < 4; ++variation) {
        ghb::gpu::SplitConfig config{variation == 1 ? 100u : 1u, variation == 2 ? 0.0 : 1.0,
                                     variation == 1 ? 10.0 : 0.0, variation == 2 ? 1.0 : 0.0,
                                     variation == 3 ? .125 : 0.0};
        const unsigned active = variation == 0 ? 5 : variation == 1 ? 3 : variation == 2 ? 0 : 8;
        f.reset(active);
        f.launch(false, config); f.launch(true, config);
        const std::string label = "B" + std::to_string(bins) + " F" + std::to_string(columns) + " mode" + std::to_string(mode);
        f.compare(active, label);
        if (variation == 0) {
          f.reset(active); f.launch(false, config, true); f.launch(true, config, true);
          f.compare(active, label + " force leaf");
        }
      }
    }
  }
  // Deliberately equal gains across features and missing routes. Also probe
  // the exact incumbent gain and its neighboring representable thresholds.
  Fixture tie(16, 3, 2);
  std::vector<Stats> stats(tie.histogram.size);
  for (std::size_t i = 0; i < stats.size(); i += 3) {
    stats[i] = {}; stats[i + 1] = {-4.0, 4.0, 4}; stats[i + 2] = {4.0, 4.0, 4};
  }
  tie.histogram.put(stats, tie.stream.value);
  const ghb::gpu::SplitConfig config{1, 0, 0, 0, 0};
  tie.launch(false, config); tie.launch(true, config); tie.compare(2, "exact ties");
  const Split winner = tie.new_winners.get(tie.stream.value)[1];
  require(winner.feature == 0 && winner.threshold == 1 && winner.missing_left == 0, "deterministic exact tie");
  require(winner.gain == 4.0 && winner.left_value == 1.0 && winner.right_value == -1.0, "independent exact split reference");
  for (const double gain : {std::nextafter(winner.gain, 0.0), winner.gain, std::nextafter(winner.gain, 10.0)}) {
    auto boundary = config; boundary.min_gain = gain;
    tie.reset(2); tie.launch(false, boundary); tie.launch(true, boundary); tie.compare(2, "gain boundary");
    require((tie.new_winners.get(tie.stream.value)[1].feature >= 0) == (gain < winner.gain), "strict gain boundary");
  }
  // Device active count can change without graph reconstruction.
  tie.reset(2);
  cudaGraph_t graph{}; cudaGraphExec_t executable{};
  check(cudaStreamBeginCapture(tie.stream.value, cudaStreamCaptureModeThreadLocal));
  tie.launch(false, config); tie.launch(true, config);
  check(cudaStreamEndCapture(tie.stream.value, &graph));
  check(cudaGraphInstantiate(&executable, graph, 0));
  for (unsigned active : {0u, 1u, 2u, 1u}) {
    tie.reset(active); check(cudaGraphLaunch(executable, tie.stream.value)); tie.compare(active, "graph active count");
  }
  check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
  auto invalid = config; invalid.min_gain = std::numeric_limits<double>::quiet_NaN();
  require(ghb::gpu::find_splits_warp_active(tie.data, tie.histogram.value, 2, tie.active.value,
      invalid, false, tie.new_candidates.value, tie.new_winners.value, tie.stream.value) == cudaErrorInvalidValue, "invalid gain rejected");
  require(ghb::gpu::find_splits_warp_active(tie.data, tie.histogram.value, 0, tie.active.value,
      config, false, tie.new_candidates.value, tie.new_winners.value, tie.stream.value) == cudaErrorInvalidValue, "zero capacity rejected");
  require(ghb::gpu::find_splits_warp_active(tie.data, tie.histogram.value, 2, nullptr,
      config, false, tie.new_candidates.value, tie.new_winners.value, tie.stream.value) == cudaErrorInvalidValue, "null active rejected");
}

void batched_cases() {
  // Each cached output is an independent root. Compare every field against
  // separate per-root launches over the identical fixed statistical inputs.
  for (unsigned bins : {2u, 16u, 32u, 33u}) for (unsigned columns : {1u, 16u, 33u}) {
    Fixture f(columns, bins, 17, columns == 16);
    for (unsigned mode = 0; mode < 7; ++mode) {
      f.populate(mode, 982451653 + mode);
      const ghb::gpu::SplitConfig config{mode == 3 ? 100u : 1u, mode == 1 ? 0.0 : 1.0,
                                         mode == 4 ? 10.0 : 0.0, mode == 2 ? 1.0 : 0.0,
                                         mode == 5 ? .125 : 0.0};
      for (unsigned count : {1u, 3u, 17u}) for (bool warp : {false, true}) for (bool leaf : {false, true}) {
        f.reset(1);
        f.launch_roots(warp, false, count, config, leaf);
        f.launch_roots(warp, true, count, config, leaf);
        f.compare(count, "batched roots B" + std::to_string(bins) + " F" + std::to_string(columns) +
                         " T" + std::to_string(count) + " mode" + std::to_string(mode));
      }
    }
  }
  Fixture f(16, 32, 7);
  const ghb::gpu::SplitConfig config{1, 1, 0, 0, .5};
  f.reset(1); f.populate(0);
  cudaGraph_t graph{}; cudaGraphExec_t executable{};
  check(cudaStreamBeginCapture(f.stream.value, cudaStreamCaptureModeThreadLocal));
  f.launch_roots(true, false, 7, config); f.launch_roots(true, true, 7, config);
  check(cudaStreamEndCapture(f.stream.value, &graph));
  check(cudaGraphInstantiate(&executable, graph, 0));
  for (unsigned mode : {0u, 2u, 3u, 5u}) {
    f.reset(1); f.populate(mode);
    check(cudaGraphLaunch(executable, f.stream.value)); f.compare(7, "batched root graph replay");
  }
  check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
  auto invalid = config; invalid.l2 = std::numeric_limits<double>::infinity();
  require(ghb::gpu::find_splits_warp(f.data, f.histogram.value, 0, config, false,
      f.new_candidates.value, f.new_winners.value, f.stream.value) == cudaErrorInvalidValue, "zero root batch rejected");
  require(ghb::gpu::find_splits_warp(f.data, f.histogram.value, 7, invalid, false,
      f.new_candidates.value, f.new_winners.value, f.stream.value) == cudaErrorInvalidValue, "invalid batch l2 rejected");
  require(ghb::gpu::find_splits_warp(f.data, nullptr, 7, config, false,
      f.new_candidates.value, f.new_winners.value, f.stream.value) == cudaErrorInvalidValue, "null root histograms rejected");
  require(ghb::gpu::find_splits_warp(f.data, f.histogram.value, unsigned(INT32_MAX) + 1, config, false,
      f.new_candidates.value, f.new_winners.value, f.stream.value) == cudaErrorInvalidValue, "oversized root batch rejected");
}

void deeper_batched_cases() {
  constexpr unsigned outputs = 7, capacity = 5;
  const unsigned shapes[][2]{{16, 32}, {16, 33}, {33, 16}, {1, 1}, {3, 257}};
  for (const auto& shape : shapes) {
    Fixture f(shape[0], shape[1], outputs * capacity, shape[0] == 16);
    Device<unsigned> active(outputs), reference_active(outputs), batch_count(1);
    std::vector<unsigned> host_active(outputs), host_reference(outputs);
    const ghb::gpu::SplitConfig config{1, 1, 0, 0, .125};
    auto reset = [&](unsigned count, unsigned phase) {
      f.reset(f.capacity);
      const unsigned counts[]{0, 1, 3, capacity, capacity + 3};
      for (unsigned output = 0; output < outputs; ++output) {
        host_active[output] = counts[(output + phase) % 5];
        host_reference[output] = output < std::min(outputs, count) ? host_active[output] : 0;
      }
      active.put(host_active, f.stream.value);
      reference_active.put(host_reference, f.stream.value);
      batch_count.put({count}, f.stream.value);
    };
    auto launch = [&](bool warp, bool force_leaf, bool with_count = true) {
      for (unsigned output = 0; output < outputs; ++output)
        check(ghb::gpu::find_splits_active(f.data,
            f.histogram.value + std::size_t(output) * capacity * f.data.total_bins,
            capacity, reference_active.value + output, config, force_leaf,
            f.old_candidates.value + 1 + std::size_t(output) * capacity * f.columns,
            f.old_winners.value + 1 + output * capacity, f.stream.value));
      const auto search = warp ? ghb::gpu::find_splits_warp_batched_active
                               : ghb::gpu::find_splits_batched_active;
      check(search(f.data, f.histogram.value, outputs, capacity, active.value, config, force_leaf,
                   f.new_candidates.value + 1, f.new_winners.value + 1, f.stream.value,
                   with_count ? batch_count.value : nullptr));
    };
    auto compare = [&](const std::string& label) {
      const auto ac = f.new_candidates.get(f.stream.value), bc = f.old_candidates.get(f.stream.value);
      const auto aw = f.new_winners.get(f.stream.value), bw = f.old_winners.get(f.stream.value);
      for (std::size_t index = 0; index < ac.size(); ++index) {
        equal(ac[index], bc[index], label + " candidate");
        if (!index || index + 1 == ac.size()) equal(ac[index], sentinel(), label + " candidate guard");
        else {
          const auto node = (index - 1) / f.columns;
          if (node % capacity >= std::min(capacity, host_reference[node / capacity]))
            equal(ac[index], sentinel(), label + " inactive candidate");
        }
      }
      for (std::size_t index = 0; index < aw.size(); ++index) {
        equal(aw[index], bw[index], label + " winner");
        if (!index || index + 1 == aw.size()) equal(aw[index], sentinel(), label + " winner guard");
        else {
          const auto node = index - 1;
          if (node % capacity >= std::min(capacity, host_reference[node / capacity]))
            equal(aw[index], sentinel(), label + " inactive winner");
        }
      }
    };
    for (unsigned mode : {0u, 1u, 2u, 3u, 4u, 5u, 6u}) {
      f.populate(mode, 7302019 + mode);
      for (unsigned count : {0u, 3u, outputs + 4}) for (bool warp : {false, true})
        for (bool force_leaf : {false, true}) {
          reset(count, mode); launch(warp, force_leaf); compare("independent deeper frontiers");
        }
      reset(outputs, mode); launch(true, false, false); compare("all-output default mask");
    }
    // Both device masks change after capture. Nonzero active counts outside the
    // output tail ensure the separate batch selector, rather than active=0,
    // is responsible for preserving those output slots.
    reset(outputs, 0);
    cudaGraph_t graph{}; cudaGraphExec_t executable{};
    check(cudaStreamBeginCapture(f.stream.value, cudaStreamCaptureModeThreadLocal));
    launch(true, false);
    check(cudaStreamEndCapture(f.stream.value, &graph));
    check(cudaGraphInstantiate(&executable, graph, 0));
    unsigned phase{};
    for (unsigned count : {0u, 1u, 3u, outputs, outputs + 2, 2u}) {
      reset(count, phase); f.populate(phase++ % 7);
      check(cudaGraphLaunch(executable, f.stream.value)); compare("deeper split graph masks");
    }
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
    for (const auto search : {ghb::gpu::find_splits_batched_active, ghb::gpu::find_splits_warp_batched_active}) {
      require(search(f.data, f.histogram.value, 0, capacity, active.value, config, false,
          f.new_candidates.value, f.new_winners.value, f.stream.value, nullptr) == cudaErrorInvalidValue,
          "zero output capacity rejected");
      require(search(f.data, f.histogram.value, outputs, 0, active.value, config, false,
          f.new_candidates.value, f.new_winners.value, f.stream.value, nullptr) == cudaErrorInvalidValue,
          "zero node capacity rejected");
      require(search(f.data, f.histogram.value, outputs, capacity, nullptr, config, false,
          f.new_candidates.value, f.new_winners.value, f.stream.value, nullptr) == cudaErrorInvalidValue,
          "null batch active rejected");
      require(search(f.data, nullptr, outputs, capacity, active.value, config, false,
          f.new_candidates.value, f.new_winners.value, f.stream.value, nullptr) == cudaErrorInvalidValue,
          "null batch histograms rejected");
      require(search(f.data, f.histogram.value, outputs, unsigned(INT32_MAX) + 1, active.value, config, false,
          f.new_candidates.value, f.new_winners.value, f.stream.value, nullptr) == cudaErrorInvalidValue,
          "signed node index capacity rejected");
      require(search(f.data, f.histogram.value, UINT32_MAX, INT32_MAX, active.value, config, false,
          f.new_candidates.value, f.new_winners.value, f.stream.value, nullptr) == cudaErrorInvalidValue,
          "batch storage overflow rejected");
      auto invalid = config; invalid.max_leaf_value = std::numeric_limits<double>::quiet_NaN();
      require(search(f.data, f.histogram.value, outputs, capacity, active.value, invalid, false,
          f.new_candidates.value, f.new_winners.value, f.stream.value, nullptr) == cudaErrorInvalidValue,
          "batch nonfinite clipping rejected");
    }
  }
  // Cross the grid cap for both candidate and winner kernels with small,
  // independent one-node outputs. One flattened old launch is an exact oracle.
  constexpr unsigned many_outputs = 65539;
  Fixture f(1, 1, many_outputs);
  Device<unsigned> active(many_outputs);
  active.put(std::vector<unsigned>(many_outputs, 1), f.stream.value);
  f.populate(1);
  const ghb::gpu::SplitConfig config{1, 1, 0, 0, 0};
  for (const auto search : {ghb::gpu::find_splits_batched_active, ghb::gpu::find_splits_warp_batched_active}) {
    f.reset(many_outputs);
    check(ghb::gpu::find_splits(f.data, f.histogram.value, many_outputs, config, false,
                               f.old_candidates.value + 1, f.old_winners.value + 1, f.stream.value));
    check(search(f.data, f.histogram.value, many_outputs, 1, active.value, config, false,
                  f.new_candidates.value + 1, f.new_winners.value + 1, f.stream.value, nullptr));
    f.compare(many_outputs, "deeper split grid stride");
  }
}

void winner_initialization() {
  constexpr unsigned rows = 513, bins = 37, outputs = 3;
  Stream stream;
  Device<int> assignments(rows + 2), frontier(3);
  Device<ghb::Node> nodes(3);
  Device<ghb::gpu::TreeState> state(1);
  Device<ghb::gpu::TreeParameters> selector(1);
  Device<Split> winner_cache(outputs), destination(3);
  Device<Stats> histogram_cache(outputs * bins), histogram_destination(bins + 2);
  const std::vector<Split> winners{{0, 1, 0, 4.0, -0.0, 1.0, -1.0},
                                   {-1, 0, 0, 0.0, .25, -0.0, 0.0},
                                   {15, 31, 1, 0x1p-700, -.125, .5, -.5}};
  std::vector<Stats> histograms(outputs * bins);
  for (unsigned i = 0; i < histograms.size(); ++i) histograms[i] = {double(i) / 7, double(i) / 13, i};
  winner_cache.put(winners, stream.value); histogram_cache.put(histograms, stream.value);
  const Stats stats_sentinel{-.0, -17.25, 918273645};
  const ghb::Node node_sentinel{19, 21, 23, 25, 1, -31.75};
  auto reset = [&] {
    assignments.put(std::vector<int>(rows + 2, -782), stream.value);
    frontier.put(std::vector<int>(3, -981), stream.value);
    nodes.put(std::vector<ghb::Node>(3, node_sentinel), stream.value);
    state.put({ghb::gpu::TreeState{17, 19, 23, 29, 31, 37}}, stream.value);
    destination.put(std::vector<Split>(3, sentinel()), stream.value);
    histogram_destination.put(std::vector<Stats>(bins + 2, stats_sentinel), stream.value);
  };
  auto launch = [&](bool with_histogram) {
    check(ghb::gpu::resident_initialize(rows, assignments.value + 1, nodes.value + 1, frontier.value + 1,
        state.value, stream.value, with_histogram ? histogram_cache.value : nullptr,
        with_histogram ? histogram_destination.value + 1 : nullptr, with_histogram ? bins : 0,
        selector.value, winner_cache.value, destination.value + 1));
  };
  auto compare = [&](unsigned selected, bool with_histogram) {
    const auto actual = destination.get(stream.value);
    equal(actual[0], sentinel(), "winner initializer left guard");
    equal(actual[1], winners[selected], "winner initializer selected output");
    equal(actual[2], sentinel(), "winner initializer right guard");
    const auto histogram = histogram_destination.get(stream.value);
    for (unsigned i = 0; i < histogram.size(); ++i) {
      const auto expected = with_histogram && i && i <= bins ? histograms[selected * bins + i - 1] : stats_sentinel;
      require(std::bit_cast<std::uint64_t>(histogram[i].gradient) == std::bit_cast<std::uint64_t>(expected.gradient) &&
              std::bit_cast<std::uint64_t>(histogram[i].hessian) == std::bit_cast<std::uint64_t>(expected.hessian) &&
              histogram[i].count == expected.count, "winner initializer histogram copy/no-copy contract");
    }
    const auto a = assignments.get(stream.value);
    for (unsigned i = 0; i < a.size(); ++i) require(a[i] == (i && i <= rows ? 0 : -782), "winner initializer assignment/guard");
    const auto f = frontier.get(stream.value);
    require(f[0] == -981 && f[1] == 0 && f[2] == -981, "winner initializer frontier/guards");
    const auto n = nodes.get(stream.value);
    for (unsigned i : {0u, 2u}) require(n[i].feature == node_sentinel.feature && n[i].left == node_sentinel.left &&
        n[i].right == node_sentinel.right && n[i].threshold == node_sentinel.threshold && n[i].missing_left == node_sentinel.missing_left &&
        n[i].value == node_sentinel.value, "winner initializer node guards");
    require(n[1].feature == -1 && n[1].left == -1 && n[1].right == -1 && n[1].threshold == 0 &&
            n[1].missing_left == 0 && n[1].value == 0, "winner initializer empty root");
    const auto s = state.get(stream.value)[0];
    require(s.active_nodes == 1 && s.next_active_nodes == 0 && s.node_count == 1 && s.status == 0 &&
            s.old_node_count == 0 && s.split_count == 0, "winner initializer state reset");
  };
  for (bool with_histogram : {false, true}) {
    reset(); selector.put({ghb::gpu::TreeParameters{3, 2, 2, 2}}, stream.value);
    launch(with_histogram); compare(2, with_histogram);
    cudaGraph_t graph{}; cudaGraphExec_t executable{};
    check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
    launch(with_histogram);
    check(cudaStreamEndCapture(stream.value, &graph)); check(cudaGraphInstantiate(&executable, graph, 0));
    for (unsigned selected : {0u, 2u, 1u, 0u}) {
      reset(); selector.put({ghb::gpu::TreeParameters{3, selected, selected, selected}}, stream.value);
      check(cudaGraphLaunch(executable, stream.value)); compare(selected, with_histogram);
    }
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
  }
  for (unsigned mask = 1; mask < 7; ++mask) {
    require(ghb::gpu::resident_initialize(rows, assignments.value + 1, nodes.value + 1, frontier.value + 1,
        state.value, stream.value, nullptr, nullptr, 0, mask & 4 ? selector.value : nullptr,
        mask & 1 ? winner_cache.value : nullptr, mask & 2 ? destination.value + 1 : nullptr) == cudaErrorInvalidValue,
        "partial winner initializer arguments rejected");
  }
}

struct Graph {
  cudaGraph_t value{};
  cudaGraphExec_t executable{};
  ~Graph() { if (executable) cudaGraphExecDestroy(executable); if (value) cudaGraphDestroy(value); }
};
void root_batch_benchmarks() {
  constexpr unsigned repetitions = 32, samples = 7;
  const ghb::gpu::SplitConfig config{1, 1, 0, 0, .75};
  std::cout << ",\"root_batch_boundary\":\"resident output-major root histograms to all feature candidates and root winners; sequential 2*T launches versus batched 2 launches; includes all candidates/winners writes, excludes allocation, transfers, graph construction and tree initialization\",\"root_batch_repetitions\":"
            << repetitions << ",\"root_batches\":[";
  bool first = true;
  for (unsigned bins : {16u, 32u, 33u}) for (unsigned outputs : {1u, 3u, 16u, 33u}) for (bool warp : {false, true}) {
    Fixture f(16, bins, outputs); f.reset(1); f.populate(0, 982451653 + bins + outputs);
    f.launch_roots(warp, false, outputs, config); f.launch_roots(warp, true, outputs, config);
    f.compare(outputs, "root batch benchmark validation");
    for (bool graph_mode : {false, true}) {
      Graph graphs[2];
      if (graph_mode) for (unsigned variant = 0; variant < 2; ++variant) {
        check(cudaStreamBeginCapture(f.stream.value, cudaStreamCaptureModeThreadLocal));
        for (unsigned i = 0; i < repetitions; ++i) f.launch_roots(warp, variant != 0, outputs, config);
        check(cudaStreamEndCapture(f.stream.value, &graphs[variant].value));
        check(cudaGraphInstantiate(&graphs[variant].executable, graphs[variant].value, 0));
      }
      cudaEvent_t start{}, end{}; check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
      std::vector<float> timing[2]; std::vector<unsigned> order;
      for (unsigned iteration = 0; iteration < samples + 2; ++iteration) for (unsigned position = 0; position < 2; ++position) {
        const unsigned variant = (position + iteration) % 2;
        check(cudaEventRecord(start, f.stream.value));
        if (graph_mode) check(cudaGraphLaunch(graphs[variant].executable, f.stream.value));
        else for (unsigned i = 0; i < repetitions; ++i) f.launch_roots(warp, variant != 0, outputs, config);
        check(cudaEventRecord(end, f.stream.value)); check(cudaEventSynchronize(end));
        float milliseconds{}; check(cudaEventElapsedTime(&milliseconds, start, end));
        if (iteration >= 2) { timing[variant].push_back(milliseconds * 1000 / repetitions); order.push_back(variant); }
      }
      check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
      f.compare(outputs, "root batch benchmark post-timing validation");
      if (!first) std::cout << ','; first = false;
      std::cout << "{\"bins\":" << bins << ",\"columns\":16,\"outputs\":" << outputs << ",\"split_policy\":\""
                << (warp ? "warp32" : "block256") << "\",\"launch\":\"" << (graph_mode ? "graph" : "stream") << "\",\"order\":[";
      for (std::size_t i = 0; i < order.size(); ++i) { if (i) std::cout << ','; std::cout << order[i]; }
      std::cout << ']';
      for (unsigned variant = 0; variant < 2; ++variant) {
        std::cout << ",\"" << (variant ? "batched_us" : "sequential_us") << "\":[";
        for (std::size_t i = 0; i < timing[variant].size(); ++i) { if (i) std::cout << ','; std::cout << timing[variant][i]; }
        std::cout << ']';
      }
      std::cout << '}';
    }
  }
  std::cout << ']';
}
void benchmark() {
  constexpr unsigned repetitions = 128, samples = 7;
  const ghb::gpu::SplitConfig config{1, 1, 0, 0, .75};
  std::cout << std::setprecision(10) << "{\"boundary\":\"resident histogram to feature candidates and node winners; two launches; allocation, transfers, graph construction excluded\",\"repetitions\":" << repetitions
            << ",\"samples\":" << samples << ",\"cases\":[";
  bool first = true;
  for (unsigned bins : {16u, 32u, 33u}) for (unsigned nodes : {1u, 2u, 16u, 64u}) {
    Fixture f(16, bins, nodes); f.populate(0, 58162593 + bins + nodes);
    f.launch(false, config); f.launch(true, config); f.compare(nodes, "benchmark validation");
    for (bool graph_mode : {false, true}) {
      Graph graphs[2];
      if (graph_mode) for (unsigned variant = 0; variant < 2; ++variant) {
        check(cudaStreamBeginCapture(f.stream.value, cudaStreamCaptureModeThreadLocal));
        for (unsigned i = 0; i < repetitions; ++i) f.launch(variant != 0, config);
        check(cudaStreamEndCapture(f.stream.value, &graphs[variant].value));
        check(cudaGraphInstantiate(&graphs[variant].executable, graphs[variant].value, 0));
      }
      cudaEvent_t start{}, end{}; check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
      std::vector<float> timing[2]; std::vector<unsigned> order;
      for (unsigned iteration = 0; iteration < samples + 2; ++iteration) for (unsigned position = 0; position < 2; ++position) {
        const unsigned variant = (position + iteration) % 2;
        check(cudaEventRecord(start, f.stream.value));
        if (graph_mode) check(cudaGraphLaunch(graphs[variant].executable, f.stream.value));
        else for (unsigned i = 0; i < repetitions; ++i) f.launch(variant != 0, config);
        check(cudaEventRecord(end, f.stream.value)); check(cudaEventSynchronize(end));
        float milliseconds{}; check(cudaEventElapsedTime(&milliseconds, start, end));
        if (iteration >= 2) { timing[variant].push_back(milliseconds * 1000 / repetitions); order.push_back(variant); }
      }
      check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
      f.compare(nodes, "benchmark post-timing validation");
      if (!first) std::cout << ','; first = false;
      std::cout << "{\"bins\":" << bins << ",\"columns\":16,\"nodes\":" << nodes << ",\"launch\":\"" << (graph_mode ? "graph" : "stream") << "\",\"order\":[";
      for (std::size_t i = 0; i < order.size(); ++i) { if (i) std::cout << ','; std::cout << order[i]; }
      std::cout << ']';
      for (unsigned variant = 0; variant < 2; ++variant) {
        std::cout << ",\"" << (variant ? "warp_us" : "baseline_us") << "\":[";
        for (std::size_t i = 0; i < timing[variant].size(); ++i) { if (i) std::cout << ','; std::cout << timing[variant][i]; }
        std::cout << ']';
      }
      std::cout << '}';
    }
  }
  std::cout << ']';
  root_batch_benchmarks();
  std::cout << "}\n";
}
} // namespace

int main(int argc, char** argv) {
  try {
    const bool bench = argc == 2 && std::string(argv[1]) == "--bench";
    if (argc > 1 && !bench) throw std::runtime_error("usage: ghb_split_search_tests [--bench]");
    int devices{}; const auto error = cudaGetDeviceCount(&devices);
    if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver || (error == cudaSuccess && !devices)) {
      std::cerr << "CUDA device unavailable\n"; return 77;
    }
    check(error);
    if (bench) benchmark();
    else { cases(); batched_cases(); deeper_batched_cases(); winner_initialization(); std::cout << "split search: " << checks << " checks passed\n"; }
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "split search failed: " << exception.what() << '\n'; return 1;
  }
}
