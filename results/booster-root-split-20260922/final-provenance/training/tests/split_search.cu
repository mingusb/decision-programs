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

struct Graph {
  cudaGraph_t value{};
  cudaGraphExec_t executable{};
  ~Graph() { if (executable) cudaGraphExecDestroy(executable); if (value) cudaGraphDestroy(value); }
};
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
  std::cout << "]}\n";
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
    else { cases(); std::cout << "split search: " << checks << " checks passed\n"; }
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "split search failed: " << exception.what() << '\n'; return 1;
  }
}
