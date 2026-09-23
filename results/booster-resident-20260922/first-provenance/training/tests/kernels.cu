#include "ghb/kernels.cuh"
#include "ghb/resident.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::size_t checks{};
void require(bool value, const std::string& message) {
  ++checks;
  if (!value) throw std::runtime_error(message);
}
void close(double actual, double expected, const std::string& message, double tolerance = 2e-11) {
  require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance * std::max({1.0, std::abs(actual), std::abs(expected)}),
          message + ": expected " + std::to_string(expected) + ", got " + std::to_string(actual));
}
void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Stream {
  cudaStream_t value{};
  Stream() { cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamSynchronize(value); cudaStreamDestroy(value); }
};
template<class T> struct Device {
  T* pointer{};
  std::size_t size{};
  explicit Device(std::size_t n) : size(n) {
    if (n) cuda_check(cudaMalloc(reinterpret_cast<void**>(&pointer), n * sizeof(T)));
  }
  ~Device() { cudaFree(pointer); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  void put(const std::vector<T>& values, cudaStream_t stream) {
    require(values.size() == size, "test buffer shape mismatch");
    if (size) cuda_check(cudaMemcpyAsync(pointer, values.data(), size * sizeof(T), cudaMemcpyHostToDevice, stream));
    cuda_check(cudaStreamSynchronize(stream)); // Test callers may pass temporary host vectors.
  }
  std::vector<T> get(cudaStream_t stream) const {
    std::vector<T> result(size);
    if (size) cuda_check(cudaMemcpyAsync(result.data(), pointer, size * sizeof(T), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    return result;
  }
};

double sigmoid(double margin) {
  return margin >= 0 ? 1.0 / (1.0 + std::exp(-margin)) : std::exp(margin) / (1.0 + std::exp(margin));
}
std::vector<double> probabilities(const std::vector<double>& margins, std::uint32_t rows, std::uint32_t outputs,
                                  ghb::Objective objective) {
  auto result = margins;
  for (std::uint32_t row = 0; row < rows; ++row) {
    if (objective == ghb::Objective::binary_logistic) {
      for (std::uint32_t output = 0; output < outputs; ++output) result[row * outputs + output] = sigmoid(margins[row * outputs + output]);
    } else if (objective == ghb::Objective::multiclass_softmax) {
      double maximum = *std::max_element(margins.begin() + row * outputs, margins.begin() + (row + 1) * outputs);
      double sum{};
      for (std::uint32_t output = 0; output < outputs; ++output) sum += std::exp(margins[row * outputs + output] - maximum);
      for (std::uint32_t output = 0; output < outputs; ++output) result[row * outputs + output] = std::exp(margins[row * outputs + output] - maximum) / sum;
    }
  }
  return result;
}

void objectives(Stream& stream) {
  constexpr std::uint32_t rows = 5;
  for (auto objective : {ghb::Objective::squared_error, ghb::Objective::binary_logistic, ghb::Objective::multiclass_softmax}) {
    const std::uint32_t outputs = objective == ghb::Objective::binary_logistic ? 2 : 3;
    std::vector<double> base(outputs), margins(rows * outputs);
    for (std::uint32_t k = 0; k < outputs; ++k) base[k] = .125 * (int(k) - 1);
    for (std::size_t i = 0; i < margins.size(); ++i) margins[i] = .25 * (int(i % 11) - 5);
    margins[0] = 1000; margins[1] = -1000;
    std::vector<float> weights{0, .5f, 2, 1, 3};
    const bool multiclass = objective == ghb::Objective::multiclass_softmax;
    std::vector<float> targets(multiclass ? rows : rows * outputs);
    for (std::size_t i = 0; i < targets.size(); ++i)
      targets[i] = objective == ghb::Objective::squared_error ? float(int(i % 7) - 3) : float(i % (multiclass ? outputs : 2));
    Device<double> d_base(outputs), d_predictions(margins.size()), d_g(margins.size() + 2), d_h(margins.size() + 2), d_loss(5);
    Device<float> d_targets(targets.size()), d_weights(weights.size());
    d_base.put(base, stream.value); d_targets.put(targets, stream.value); d_weights.put(weights, stream.value);
    cuda_check(ghb::gpu::initialize_predictions(d_predictions.pointer, d_base.pointer, rows, outputs, stream.value));
    auto initial = d_predictions.get(stream.value);
    for (std::size_t i = 0; i < initial.size(); ++i) close(initial[i], base[i % outputs], "base prediction initialization", 0);
    auto expected_probabilities = probabilities(margins, rows, outputs, objective);
    for (bool weighted : {false, true}) {
      d_predictions.put(margins, stream.value);
      d_g.put(std::vector<double>(margins.size() + 2, 1234567), stream.value);
      d_h.put(std::vector<double>(margins.size() + 2, -7654321), stream.value);
      cuda_check(ghb::gpu::gradients(objective, d_predictions.pointer, d_targets.pointer, weighted ? d_weights.pointer : nullptr,
                                   d_g.pointer + 1, d_h.pointer + 1, rows, outputs, stream.value));
      auto gradient = d_g.get(stream.value), hessian = d_h.get(stream.value);
      close(gradient.front(), 1234567, "gradient leading guard", 0); close(gradient.back(), 1234567, "gradient trailing guard", 0);
      close(hessian.front(), -7654321, "Hessian leading guard", 0); close(hessian.back(), -7654321, "Hessian trailing guard", 0);
      double expected_loss{};
      for (std::uint32_t row = 0; row < rows; ++row) {
        const double weight = weighted ? weights[row] : 1;
        for (std::uint32_t output = 0; output < outputs; ++output) {
          const auto index = row * outputs + output;
          const double target = multiclass ? double(output == std::uint32_t(targets[row])) : targets[index];
          const double prediction = expected_probabilities[index];
          const double curvature = objective == ghb::Objective::squared_error ? 1.0 : std::max((multiclass ? 2.0 : 1.0) * prediction * (1 - prediction), 1e-16);
          close(gradient[index + 1], weight * (prediction - target), "weighted objective gradient");
          close(hessian[index + 1], weight * curvature, "weighted objective Hessian", curvature == 1e-16 ? 1e-27 : 2e-11);
          if (weight == 0) require(gradient[index + 1] == 0 && hessian[index + 1] == 0, "zero weights must remain zero after Hessian flooring");
          if (objective == ghb::Objective::squared_error) expected_loss += .5 * weight * std::pow(margins[index] - target, 2);
          else if (objective == ghb::Objective::binary_logistic)
            expected_loss += weight * (std::max(margins[index], 0.0) + std::log1p(std::exp(-std::abs(margins[index]))) - target * margins[index]);
        }
        if (multiclass) {
          double maximum = *std::max_element(margins.begin() + row * outputs, margins.begin() + (row + 1) * outputs), sum{};
          for (std::uint32_t output = 0; output < outputs; ++output) sum += std::exp(margins[row * outputs + output] - maximum);
          expected_loss += weight * (maximum - margins[row * outputs + std::uint32_t(targets[row])] + std::log(sum));
        }
      }
      d_loss.put(std::vector<double>(5, 1e100), stream.value);
      cuda_check(ghb::gpu::loss(objective, d_predictions.pointer, d_targets.pointer, weighted ? d_weights.pointer : nullptr,
                              rows, outputs, d_loss.pointer, 5, stream.value));
      auto losses = d_loss.get(stream.value);
      close(std::accumulate(losses.begin(), losses.end(), 0.0), expected_loss, "weighted raw objective sum");
    }
    cuda_check(ghb::gpu::transform(objective, d_predictions.pointer, rows, outputs, stream.value));
    auto transformed = d_predictions.get(stream.value);
    for (std::size_t i = 0; i < transformed.size(); ++i) close(transformed[i], expected_probabilities[i], "probability transformation");
  }
}

void tiled_gradients(Stream& stream) {
  constexpr std::uint32_t rows = 5;
  for (const std::uint32_t outputs : {7u, 65u}) {
    const std::uint32_t tile_size = outputs == 7 ? 3 : 16;
    for (auto objective : {ghb::Objective::squared_error, ghb::Objective::binary_logistic}) {
      std::vector<double> margins(rows * outputs);
      std::vector<float> targets(rows * outputs), weights{0, .5f, 1, 2, 3};
      for (std::uint32_t row = 0; row < rows; ++row) for (std::uint32_t output = 0; output < outputs; ++output) {
        margins[row * outputs + output] = .125 * (int(row) - int(output % 9));
        targets[row * outputs + output] = objective == ghb::Objective::squared_error ? .25f * (int((row + output) % 7) - 3) : float((row + output) % 2);
      }
      Device<double> d_margin(margins.size()), d_full_g(margins.size()), d_full_h(margins.size());
      Device<float> d_targets(targets.size()), d_weights(rows);
      d_margin.put(margins, stream.value); d_targets.put(targets, stream.value); d_weights.put(weights, stream.value);
      cuda_check(ghb::gpu::gradients(objective, d_margin.pointer, d_targets.pointer, d_weights.pointer,
                                   d_full_g.pointer, d_full_h.pointer, rows, outputs, stream.value));
      const auto full_g = d_full_g.get(stream.value), full_h = d_full_h.get(stream.value);
      for (std::uint32_t begin = 0; begin < outputs; begin += tile_size) {
        const std::uint32_t count = std::min(tile_size, outputs - begin);
        Device<double> d_g(rows * count + 2), d_h(rows * count + 2);
        d_g.put(std::vector<double>(rows * count + 2, 1234567), stream.value);
        d_h.put(std::vector<double>(rows * count + 2, -7654321), stream.value);
        cuda_check(ghb::gpu::gradients_tile(objective, d_margin.pointer, d_targets.pointer, d_weights.pointer,
                                           d_g.pointer + 1, d_h.pointer + 1, rows, outputs, begin, count, stream.value));
        auto gradient = d_g.get(stream.value), hessian = d_h.get(stream.value);
        close(gradient.front(), 1234567, "tiled gradient leading guard", 0); close(gradient.back(), 1234567, "tiled gradient trailing guard", 0);
        close(hessian.front(), -7654321, "tiled Hessian leading guard", 0); close(hessian.back(), -7654321, "tiled Hessian trailing guard", 0);
        for (std::uint32_t row = 0; row < rows; ++row) for (std::uint32_t k = 0; k < count; ++k) {
          const auto source = row * outputs + begin + k, destination = row * count + k + 1;
          const double prediction = objective == ghb::Objective::squared_error ? margins[source] : sigmoid(margins[source]);
          const double curvature = objective == ghb::Objective::squared_error ? 1.0 : std::max(prediction * (1 - prediction), 1e-16);
          close(gradient[destination], weights[row] * (prediction - targets[source]), "tile output index gradient");
          close(hessian[destination], weights[row] * curvature, "tile output index Hessian");
          if (weights[row] == 0) require(gradient[destination] == 0 && hessian[destination] == 0, "tiled zero-weight derivative was not zero");
          close(gradient[destination], full_g[source], "tile/full gradient agreement");
          close(hessian[destination], full_h[source], "tile/full Hessian agreement");
        }
      }
    }
  }
}

struct PackedData {
  std::uint32_t rows;
  std::vector<std::uint16_t> bins;
  std::vector<std::uint32_t> offsets;
  std::vector<ghb::FeatureType> types;
  Device<std::uint16_t> d_bins;
  Device<std::uint32_t> d_offsets;
  Device<ghb::FeatureType> d_types;
  ghb::gpu::DataView view{};
  PackedData(std::uint32_t count, std::vector<std::uint16_t> values, std::vector<std::uint32_t> boundaries,
             std::vector<ghb::FeatureType> kinds, cudaStream_t stream)
      : rows(count), bins(std::move(values)), offsets(std::move(boundaries)), types(std::move(kinds)),
        d_bins(bins.size()), d_offsets(offsets.size()), d_types(types.size()) {
    d_bins.put(bins, stream); d_offsets.put(offsets, stream); d_types.put(types, stream);
    view.bins = d_bins.pointer; view.offsets = d_offsets.pointer; view.types = d_types.pointer;
    view.rows = rows; view.columns = types.size(); view.total_bins = offsets.back();
    for (std::size_t f = 0; f < types.size(); ++f) view.max_feature_bins = std::max(view.max_feature_bins, offsets[f + 1] - offsets[f]);
  }
};

void histograms(Stream& stream) {
  constexpr std::uint32_t rows = 97, outputs = 3, nodes = 4;
  std::vector<std::uint16_t> bins(rows * 3);
  std::vector<std::uint32_t> offsets{0, 5, 9, 15};
  std::vector<std::int32_t> assignment(rows);
  std::vector<double> gradient(rows * outputs), hessian(rows * outputs);
  for (std::uint32_t row = 0; row < rows; ++row) {
    assignment[row] = int(row % 4) - 1; // Includes inactive rows and an entirely empty fourth node.
    for (std::uint32_t f = 0; f < 3; ++f) bins[f * rows + row] = (row * (f + 2) + f) % (offsets[f + 1] - offsets[f]);
    for (std::uint32_t output = 0; output < outputs; ++output) {
      gradient[row * outputs + output] = row % 13 == 0 ? 0 : .25 * (int((row * 3 + output) % 17) - 8);
      hessian[row * outputs + output] = row % 13 == 0 ? 0 : .125 * ((row + output) % 5 + 1);
    }
  }
  PackedData data(rows, bins, offsets, {ghb::FeatureType::numeric, ghb::FeatureType::categorical, ghb::FeatureType::numeric}, stream.value);
  Device<std::int32_t> d_assignment(rows); Device<double> d_g(gradient.size()), d_h(hessian.size());
  Device<ghb::gpu::Stats> d_hist(nodes * offsets.back() + 2);
  d_assignment.put(assignment, stream.value); d_g.put(gradient, stream.value); d_h.put(hessian, stream.value);
  require(ghb::gpu::shared_supported(data.view, nodes), "small frontier must support shared histograms");
  auto too_large = data.view; too_large.max_feature_bins = 65536;
  require(!ghb::gpu::shared_supported(too_large, nodes), "oversized shared frontier was accepted");
  for (std::uint32_t output = 0; output < outputs; ++output) {
    std::vector<ghb::gpu::Stats> expected(nodes * offsets.back());
    for (std::uint32_t row = 0; row < rows; ++row) if (assignment[row] >= 0) {
      for (std::uint32_t f = 0; f < 3; ++f) {
        auto& value = expected[assignment[row] * offsets.back() + offsets[f] + bins[f * rows + row]];
        value.gradient += gradient[row * outputs + output]; value.hessian += hessian[row * outputs + output]; ++value.count;
      }
    }
    for (auto policy : {ghb::HistogramPolicy::global, ghb::HistogramPolicy::shared}) {
      ghb::gpu::Stats sentinel{123456, -654321, 0xabcdef};
      d_hist.put(std::vector<ghb::gpu::Stats>(expected.size() + 2, sentinel), stream.value);
      for (int repeat = 0; repeat < 2; ++repeat) {
        cuda_check(ghb::gpu::histogram(data.view, d_assignment.pointer, d_g.pointer, d_h.pointer,
                                      outputs, output, nodes, d_hist.pointer + 1, policy, stream.value));
        auto actual = d_hist.get(stream.value);
        for (const auto index : {std::size_t(0), actual.size() - 1}) {
          close(actual[index].gradient, sentinel.gradient, "histogram guard gradient", 0);
          close(actual[index].hessian, sentinel.hessian, "histogram guard Hessian", 0);
          require(actual[index].count == sentinel.count, "histogram guard count changed");
        }
        for (std::size_t index = 0; index < expected.size(); ++index) {
          close(actual[index + 1].gradient, expected[index].gradient, "histogram gradient");
          close(actual[index + 1].hessian, expected[index].hessian, "histogram Hessian");
          require(actual[index + 1].count == expected[index].count, "histogram count excludes wrong rows or outputs");
        }
      }
      Device<std::uint32_t> d_active(1);
      Device<ghb::gpu::TreeParameters> selector(1);
      selector.put({{outputs, output, 0}}, stream.value);
      for (const auto active : {0u, 2u}) {
        d_active.put({active}, stream.value);
        d_hist.put(std::vector<ghb::gpu::Stats>(expected.size() + 2, sentinel), stream.value);
        cuda_check(ghb::gpu::histogram_active(data.view, d_assignment.pointer, d_g.pointer, d_h.pointer,
            outputs, 0, nodes, d_active.pointer, d_hist.pointer + 1, policy, stream.value, selector.pointer));
        const auto actual = d_hist.get(stream.value);
        for (std::size_t index = 0; index < actual.size(); ++index) {
          const auto reference = index > 0 && index <= std::size_t(active) * offsets.back() ? expected[index - 1] : sentinel;
          close(actual[index].gradient, reference.gradient, "device-active histogram gradient/guard");
          close(actual[index].hessian, reference.hessian, "device-active histogram Hessian/guard");
          require(actual[index].count == reference.count, "device-active histogram count or inactive tail changed");
        }
      }
    }
  }
}

struct Totals { double g{}, h{}; std::uint64_t count{}; };
double leaf(Totals totals, const ghb::gpu::SplitConfig& config) {
  const double denominator = totals.h + config.l2;
  double value = denominator > 0 ? -totals.g / denominator : 0;
  if (config.max_leaf_value > 0) value = std::clamp(value, -config.max_leaf_value, config.max_leaf_value);
  return value;
}
double benefit(Totals totals, const ghb::gpu::SplitConfig& config) {
  const double value = leaf(totals, config);
  return -totals.g * value - .5 * (totals.h + config.l2) * value * value;
}
bool left_of(std::uint16_t bin, ghb::FeatureType type, std::uint32_t threshold, bool missing_left) {
  return bin == 0 ? missing_left : type == ghb::FeatureType::categorical ? bin == threshold : bin <= threshold;
}
Totals total_rows(const std::vector<std::uint32_t>& rows, const std::vector<double>& gradient, const std::vector<double>& hessian) {
  Totals result;
  for (auto row : rows) { result.g += gradient[row]; result.h += hessian[row]; ++result.count; }
  return result;
}
ghb::gpu::Split brute_force(const PackedData& data, const std::vector<std::uint32_t>& rows,
                           const std::vector<double>& gradient, const std::vector<double>& hessian,
                           const ghb::gpu::SplitConfig& config, int only_feature = -1, bool force_leaf = false) {
  ghb::gpu::Split best;
  const auto parent = total_rows(rows, gradient, hessian);
  best.value = leaf(parent, config);
  if (force_leaf) return best;
  for (std::uint32_t feature = 0; feature < data.types.size(); ++feature) {
    if (only_feature >= 0 && feature != std::uint32_t(only_feature)) continue;
    for (std::uint32_t threshold = 0; threshold < data.offsets[feature + 1] - data.offsets[feature]; ++threshold) {
      for (std::uint32_t missing_left = 0; missing_left < 2; ++missing_left) {
        std::vector<std::uint32_t> left, right;
        for (auto row : rows) (left_of(data.bins[feature * data.rows + row], data.types[feature], threshold, missing_left) ? left : right).push_back(row);
        const auto l = total_rows(left, gradient, hessian), r = total_rows(right, gradient, hessian);
        if (l.count < config.min_leaf_rows || r.count < config.min_leaf_rows || l.h < config.min_child_hessian || r.h < config.min_child_hessian) continue;
        const double gain = benefit(l, config) + benefit(r, config) - benefit(parent, config);
        if (gain > config.min_gain && (best.feature < 0 || gain > best.gain))
          best = {std::int32_t(feature), threshold, missing_left, gain, best.value, leaf(l, config), leaf(r, config)};
      }
    }
  }
  return best;
}

void splitting_and_routing(Stream& stream) {
  constexpr std::uint32_t rows = 18, nodes = 3;
  PackedData data(rows,
      {0,0,1,1,2,2,3,3,4, 1,2,3,4,1,2,3,4,2,
       1,2,1,2,3,1,2,3,1, 0,0,1,1,2,2,3,3,2,
       1,2,3,1,2,3,1,2,3, 2,3,1,2,3,1,2,3,1},
      {0,5,9,13}, {ghb::FeatureType::numeric, ghb::FeatureType::categorical, ghb::FeatureType::numeric}, stream.value);
  std::vector<double> gradients{-5,-4,-4,-3,-2,-1,4,5,6, 5,4,-4,-5,4,5,3,4,10000}, hessians(rows, 1);
  std::vector<std::int32_t> assignments(rows, -1);
  for (std::uint32_t row = 0; row < 17; ++row) assignments[row] = row < 9 ? 0 : 1;
  std::vector<ghb::gpu::Stats> histogram(nodes * data.view.total_bins);
  std::vector<std::vector<std::uint32_t>> active(nodes);
  for (std::uint32_t row = 0; row < rows; ++row) if (assignments[row] >= 0) {
    active[assignments[row]].push_back(row);
    for (std::uint32_t f = 0; f < data.types.size(); ++f) {
      auto& stats = histogram[assignments[row] * data.view.total_bins + data.offsets[f] + data.bins[f * rows + row]];
      stats.gradient += gradients[row]; stats.hessian += hessians[row]; ++stats.count;
    }
  }
  Device<ghb::gpu::Stats> d_hist(histogram.size());
  Device<ghb::gpu::Split> d_candidates(nodes * data.types.size()), d_winners(nodes);
  d_hist.put(histogram, stream.value);
  const ghb::gpu::SplitConfig normal{2, 1, .01, 0, 0};
  std::vector<ghb::gpu::SplitConfig> cases{normal, {100,1,.01,0,0}, {2,1,100,0,0}, {2,1,.01,1e9,0}, {2,3,.01,0,.4}, {1,0,0,0,0}};
  for (std::size_t test = 0; test <= cases.size(); ++test) {
    const bool forced = test == cases.size(); const auto config = forced ? normal : cases[test];
    cuda_check(ghb::gpu::find_splits(data.view, d_hist.pointer, nodes, config, forced, d_candidates.pointer, d_winners.pointer, stream.value));
    auto candidates = d_candidates.get(stream.value), winners = d_winners.get(stream.value);
    for (std::uint32_t node = 0; node < nodes; ++node) {
      auto expected = brute_force(data, active[node], gradients, hessians, config, -1, forced);
      close(winners[node].value, expected.value, "regularized/clipped parent leaf value");
      require((winners[node].feature < 0) == (expected.feature < 0), "split constraints or forced leaf violated");
      if (expected.feature >= 0) {
        require(winners[node].feature >= 0 && std::size_t(winners[node].feature) < data.types.size(), "winner feature out of range");
        require(winners[node].threshold < data.offsets[winners[node].feature + 1] - data.offsets[winners[node].feature] && winners[node].missing_left <= 1,
                "winner threshold/missing direction out of range");
        close(winners[node].gain, expected.gain, "split quadratic objective reduction");
        std::vector<std::uint32_t> chosen_left, chosen_right;
        for (auto row : active[node]) (left_of(data.bins[winners[node].feature * rows + row], data.types[winners[node].feature],
                                               winners[node].threshold, winners[node].missing_left) ? chosen_left : chosen_right).push_back(row);
        const auto left = total_rows(chosen_left, gradients, hessians), right = total_rows(chosen_right, gradients, hessians);
        require(left.count >= config.min_leaf_rows && right.count >= config.min_leaf_rows &&
                left.h >= config.min_child_hessian && right.h >= config.min_child_hessian, "chosen partition violates child constraints");
        close(winners[node].gain, benefit(left, config) + benefit(right, config) - benefit(total_rows(active[node], gradients, hessians), config), "chosen partition's actual gain");
        close(winners[node].left_value, leaf(left, config), "left clipped Newton value");
        close(winners[node].right_value, leaf(right, config), "right clipped Newton value");
      }
      if (!forced) for (std::uint32_t feature = 0; feature < data.types.size(); ++feature) {
        const auto reference = brute_force(data, active[node], gradients, hessians, config, feature);
        const auto actual = candidates[node * data.types.size() + feature];
        require((actual.feature < 0) == (reference.feature < 0), "per-feature candidate validity differs");
        if (reference.feature >= 0) close(actual.gain, reference.gain, "per-feature brute-force gain");
      }
    }
    if (test == 0) {
      require(winners[0].feature == 0 && winners[0].threshold == 2 && winners[0].missing_left == 1, "fixture must learn numerical missing-left split");
      require(winners[1].feature == 1 && winners[1].threshold == 1 && winners[1].missing_left == 0, "fixture must learn categorical missing-right split");
    }
  }
  Device<std::uint32_t> d_active(1);
  ghb::gpu::Split split_guard; split_guard.value = 1234567;
  for (auto count : {0u, 2u}) {
    d_active.put({count}, stream.value);
    d_winners.put(std::vector<ghb::gpu::Split>(nodes, split_guard), stream.value);
    d_candidates.put(std::vector<ghb::gpu::Split>(nodes * data.types.size(), split_guard), stream.value);
    cuda_check(ghb::gpu::find_splits_active(data.view, d_hist.pointer, nodes, d_active.pointer,
        normal, false, d_candidates.pointer, d_winners.pointer, stream.value));
    const auto actual = d_winners.get(stream.value), candidates = d_candidates.get(stream.value);
    for (std::uint32_t node = 0; node < nodes; ++node) {
      if (node < count) {
        const auto expected = brute_force(data, active[node], gradients, hessians, normal);
        require(actual[node].feature == expected.feature && actual[node].threshold == expected.threshold && actual[node].missing_left == expected.missing_left,
                "device-active winner differs from brute-force partition");
        close(actual[node].gain, expected.gain, "device-active winner gain");
        close(actual[node].value, expected.value, "device-active parent value");
      } else {
        require(actual[node].value == split_guard.value, "device-active split touched inactive winner");
        for (std::size_t f = 0; f < data.types.size(); ++f)
          require(candidates[node * data.types.size() + f].value == split_guard.value, "device-active split touched inactive candidate");
      }
    }
  }
  std::vector<ghb::gpu::Split> winners(nodes);
  winners[0].feature = 0; winners[0].threshold = 2; winners[0].missing_left = 1;
  winners[1].feature = 1; winners[1].threshold = 1; winners[1].missing_left = 0;
  assignments[0] = -1; assignments.back() = 2;
  const std::vector<std::int32_t> left_map{7,8,-1}, right_map{9,-1,-1};
  Device<std::int32_t> d_assignments(rows), d_left(nodes), d_right(nodes);
  d_assignments.put(assignments, stream.value); d_left.put(left_map, stream.value); d_right.put(right_map, stream.value); d_winners.put(winners, stream.value);
  cuda_check(ghb::gpu::route(data.view, d_assignments.pointer, d_winners.pointer, d_left.pointer, d_right.pointer, stream.value));
  auto routed = d_assignments.get(stream.value);
  for (std::uint32_t row = 0; row < rows; ++row) {
    auto expected = std::int32_t(-1);
    if (assignments[row] >= 0 && winners[assignments[row]].feature >= 0) {
      const auto& split = winners[assignments[row]];
      const bool left = left_of(data.bins[split.feature * rows + row], data.types[split.feature], split.threshold, split.missing_left);
      expected = (left ? left_map : right_map)[assignments[row]];
    }
    require(routed[row] == expected, "frontier routing/deactivation mismatch");
  }
  const std::vector<ghb::Node> tree{{0,1,2,2,1,0}, {-1,-1,-1,0,0,-2}, {1,3,4,1,0,0}, {-1,-1,-1,0,0,3}, {-1,-1,-1,0,0,5}};
  Device<ghb::Node> d_tree(tree.size()); Device<double> d_prediction(rows * 2);
  std::vector<double> initial(rows * 2, .25); d_tree.put(tree, stream.value); d_prediction.put(initial, stream.value);
  cuda_check(ghb::gpu::add_tree(data.view, d_tree.pointer, tree.size(), 1, 2, d_prediction.pointer, stream.value));
  auto prediction = d_prediction.get(stream.value);
  for (std::uint32_t row = 0; row < rows; ++row) {
    const double expected = left_of(data.bins[row], ghb::FeatureType::numeric, 2, true) ? -2 :
                            left_of(data.bins[rows + row], ghb::FeatureType::categorical, 1, false) ? 3 : 5;
    close(prediction[row * 2], .25, "tree changed another output", 0);
    close(prediction[row * 2 + 1], .25 + expected, "tree traversal prediction", 0);
  }
  Device<ghb::gpu::TreeState> state(1); Device<ghb::gpu::TreeParameters> selector(1);
  state.put({{0,0,std::uint32_t(tree.size()),0,0,0}}, stream.value);
  selector.put({{1,0,1}}, stream.value); d_prediction.put(initial, stream.value);
  cuda_check(ghb::gpu::resident_predict(data.view, d_tree.pointer, state.pointer, 0, 2, d_prediction.pointer, stream.value, selector.pointer));
  const auto resident_prediction = d_prediction.get(stream.value);
  for (std::size_t i = 0; i < prediction.size(); ++i)
    close(resident_prediction[i], prediction[i], "resident selector-based tree prediction", 0);
  state.put({{0,0,std::uint32_t(tree.size()),ghb::gpu::tree_invalid_split,0,0}}, stream.value);
  cuda_check(ghb::gpu::resident_predict(data.view, d_tree.pointer, state.pointer, 1, 2, d_prediction.pointer, stream.value));
  const auto blocked = d_prediction.get(stream.value);
  for (std::size_t i = 0; i < prediction.size(); ++i)
    close(blocked[i], prediction[i], "invalid resident tree changed prediction", 0);
}

void resident_frontiers(Stream& stream) {
  // Sparse split flags straddle the 1,024-item scan boundary. Expected IDs are
  // constructed by ordinary append in stable frontier order, independently of
  // the device's scan/offset representation.
  for (const auto capacity : {1u, 17u, 1025u, 2051u}) {
    const auto active = capacity == 1 ? 1 : capacity - 3, rows = capacity + 4;
    std::vector<std::uint16_t> keys(rows);
    for (std::uint32_t row = 0; row < rows; ++row) keys[row] = row % 5;
    PackedData data(rows, keys, {0,5}, {ghb::FeatureType::numeric}, stream.value);
    std::vector<ghb::gpu::Split> splits(capacity);
    std::vector<std::int32_t> frontier(capacity, -17), next(capacity, -17), left(capacity, -17), right(capacity, -17), assignments(rows);
    std::uint32_t split_count{};
    for (std::uint32_t node = 0; node < active; ++node) {
      frontier[node] = active - 1 - node;
      splits[node].value = .25 * (int(node % 13) - 6);
      if (capacity > 1 && node % 3 == 1) {
        splits[node].feature = 0; splits[node].threshold = 2; splits[node].missing_left = node % 2;
        splits[node].left_value = -double(node + 1); splits[node].right_value = double(node + 2); splits[node].gain = 1;
        ++split_count;
      }
    }
    const auto node_capacity = active + 2 * split_count + 4;
    ghb::Node guard; guard.value = 987654321.25;
    Device<ghb::Node> d_nodes(node_capacity + 2);
    Device<ghb::gpu::Split> d_splits(capacity); Device<ghb::gpu::TreeState> state(1);
    Device<std::int32_t> d_frontier(capacity), d_next(capacity), d_left(capacity), d_right(capacity), d_assignment(rows);
    Device<std::uint32_t> offsets(capacity + 2), block_counts((capacity + 1023) / 1024 + 2);
    d_splits.put(splits, stream.value); d_frontier.put(frontier, stream.value);
    for (bool expand : {false, true}) {
      d_nodes.put(std::vector<ghb::Node>(node_capacity + 2, guard), stream.value);
      d_next.put(next, stream.value); d_left.put(left, stream.value); d_right.put(right, stream.value);
      offsets.put(std::vector<std::uint32_t>(offsets.size, 0x12345678), stream.value);
      block_counts.put(std::vector<std::uint32_t>(block_counts.size, 0x76543210), stream.value);
      state.put({{active,0,active,0,0,0}}, stream.value);
      cuda_check(ghb::gpu::resident_materialize(data.view, d_splits.pointer, d_frontier.pointer, d_next.pointer,
          d_left.pointer, d_right.pointer, d_nodes.pointer + 1, state.pointer, capacity, node_capacity,
          offsets.pointer + 1, block_counts.pointer + 1, expand, .5, stream.value));
      const auto actual_state = state.get(stream.value)[0];
      require(actual_state.status == 0 && actual_state.node_count == active + 2 * split_count &&
              actual_state.next_active_nodes == (expand ? 2 * split_count : 0), "resident materialization counts");
      const auto nodes = d_nodes.get(stream.value);
      const auto next_frontier = d_next.get(stream.value);
      const auto left_map = d_left.get(stream.value), right_map = d_right.get(stream.value);
      std::uint32_t ordinal{};
      for (std::uint32_t index = 0; index < active; ++index) {
        const auto& node = nodes[frontier[index] + 1]; const auto& split = splits[index];
        if (split.feature < 0) {
          require(node.feature == -1, "resident terminal node is not a leaf");
          close(node.value, .5 * split.value, "resident scaled parent leaf", 0);
          require(left_map[index] == -1 && right_map[index] == -1, "leaf has active child maps");
        } else {
          const auto child = active + 2 * ordinal;
          require(node.feature == 0 && node.threshold == 2 && node.missing_left == split.missing_left &&
                  node.left == int(child) && node.right == int(child + 1), "resident stable child order or predicate");
          require(nodes[child + 1].feature == -1 && nodes[child + 2].feature == -1, "new children must be initialized as leaves");
          close(nodes[child + 1].value, .5 * split.left_value, "resident left child value", 0);
          close(nodes[child + 2].value, .5 * split.right_value, "resident right child value", 0);
          if (expand) {
            require(next_frontier[2 * ordinal] == int(child) && next_frontier[2 * ordinal + 1] == int(child + 1), "resident next frontier order");
            require(left_map[index] == int(2 * ordinal) && right_map[index] == int(2 * ordinal + 1), "resident child assignment map");
          } else require(left_map[index] == -1 && right_map[index] == -1, "terminal-depth split exposed another frontier");
          ++ordinal;
        }
      }
      require(nodes.front().value == guard.value && nodes.back().value == guard.value, "resident node guards changed");
      const auto scanned = offsets.get(stream.value), blocks = block_counts.get(stream.value);
      require(scanned.front() == 0x12345678 && scanned.back() == 0x12345678 && blocks.front() == 0x76543210 && blocks.back() == 0x76543210,
              "resident scan guards changed");
      for (std::uint32_t row = 0; row < rows; ++row) assignments[row] = row % 11 == 0 ? -1 : std::int32_t(row % active);
      d_assignment.put(assignments, stream.value);
      cuda_check(ghb::gpu::resident_route(data.view, d_assignment.pointer, d_splits.pointer, d_left.pointer, d_right.pointer, state.pointer, stream.value));
      const auto routed = d_assignment.get(stream.value);
      for (std::uint32_t row = 0; row < rows; ++row) {
        auto expected = assignments[row];
        if (expand && split_count && expected >= 0) {
          const auto& split = splits[expected];
          expected = split.feature < 0 ? -1 : left_of(keys[row], ghb::FeatureType::numeric, split.threshold, split.missing_left)
              ? left_map[expected] : right_map[expected];
        }
        require(routed[row] == expected, "resident row routing differs from CPU predicate");
      }
      cuda_check(ghb::gpu::resident_advance(state.pointer, stream.value));
      require(state.get(stream.value)[0].active_nodes == (expand ? 2 * split_count : 0), "resident frontier advance");
    }
    if (split_count) {
      state.put({{active,0,active,0,0,0}}, stream.value);
      cuda_check(ghb::gpu::resident_materialize(data.view, d_splits.pointer, d_frontier.pointer, d_next.pointer,
          d_left.pointer, d_right.pointer, d_nodes.pointer + 1, state.pointer, capacity, active,
          offsets.pointer + 1, block_counts.pointer + 1, true, .5, stream.value));
      require(state.get(stream.value)[0].status & ghb::gpu::tree_node_overflow, "resident node-capacity overflow was not reported");
    }
    // Initialization must clear stale status/frontier counts after any failure.
    cuda_check(ghb::gpu::resident_initialize(rows, d_assignment.pointer, d_nodes.pointer + 1, d_frontier.pointer, state.pointer, stream.value));
    const auto initialized = state.get(stream.value)[0];
    require(initialized.active_nodes == 1 && initialized.node_count == 1 && initialized.status == 0 && initialized.next_active_nodes == 0,
            "resident initialization did not reset state");
    for (const auto assignment : d_assignment.get(stream.value)) require(assignment == 0, "resident initialization omitted a row");
    require(d_frontier.get(stream.value)[0] == 0 && d_nodes.get(stream.value)[1].feature == -1, "resident initialization did not create a root leaf");
  }
}

void wide_numeric_scans(Stream& stream) {
  constexpr std::uint32_t rows = 16;
  for (const auto bins : {257u, 2048u, 2049u, 65536u}) {
    const std::uint16_t low = bins == 257 ? 255 : std::uint16_t(bins / 2);
    const std::uint16_t high = std::uint16_t(bins - 1);
    std::vector<std::uint16_t> keys(rows);
    std::vector<double> gradient(rows), hessian(rows, 1);
    std::vector<ghb::gpu::Stats> reference(bins);
    for (std::uint32_t row = 0; row < rows; ++row) {
      keys[row] = row < 4 ? 0 : row < 10 ? low : high;
      gradient[row] = row < 10 ? -3 : 5;
      auto& stats = reference[keys[row]]; stats.gradient += gradient[row]; stats.hessian += 1; ++stats.count;
    }
    PackedData data(rows, keys, {0,bins}, {ghb::FeatureType::numeric}, stream.value);
    Device<ghb::gpu::Stats> d_hist(bins);
    Device<ghb::gpu::Split> d_candidates(1), d_winners(1);
    d_hist.put(reference, stream.value);
    const ghb::gpu::SplitConfig config{2,1,.01,0,0};
    cuda_check(ghb::gpu::find_splits(data.view, d_hist.pointer, 1, config, false,
                                    d_candidates.pointer, d_winners.pointer, stream.value));
    const auto winner = d_winners.get(stream.value)[0];
    // Only three occupied values exist. All other numeric thresholds have the
    // same partition as one of {0,low,high}; separating ten negative from six
    // positive gradients is the unique best non-equivalent partition.
    require(winner.feature == 0 && winner.threshold == low && winner.missing_left == 1,
            "numeric scan missed occupied bins across scan-block boundaries");
    close(winner.gain, benefit({-30,10,10}, config) + benefit({30,6,6}, config), "wide numeric gain");
    close(winner.left_value, 30.0 / 11, "wide numeric left leaf");
    close(winner.right_value, -30.0 / 7, "wide numeric right leaf");
    require(ghb::gpu::shared_supported(data.view, 1) == (bins <= 2048), "shared exact resource boundary");
    if (bins == 2048) {
      require(!ghb::gpu::shared_supported(data.view, 2), "shared frontier budget must include every node");
      Device<double> d_g(rows), d_h(rows); Device<std::int32_t> d_assignments(rows);
      d_g.put(gradient, stream.value); d_h.put(hessian, stream.value);
      d_assignments.put(std::vector<std::int32_t>(rows, 0), stream.value);
      for (const auto policy : {ghb::HistogramPolicy::global, ghb::HistogramPolicy::shared}) {
        cuda_check(ghb::gpu::histogram(data.view, d_assignments.pointer, d_g.pointer, d_h.pointer,
                                      1, 0, 1, d_hist.pointer, policy, stream.value));
        const auto actual = d_hist.get(stream.value);
        for (std::uint32_t bin = 0; bin < bins; ++bin) {
          close(actual[bin].gradient, reference[bin].gradient, "2048-bin histogram gradient");
          close(actual[bin].hessian, reference[bin].hessian, "2048-bin histogram Hessian");
          require(actual[bin].count == reference[bin].count, "2048-bin histogram count");
        }
      }
    }
  }
}
} // namespace

int main() {
  try {
    int devices{};
    const auto status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
      std::cout << "SKIP: no CUDA device/driver\n"; return 77;
    }
    cuda_check(status);
    Stream stream;
    objectives(stream); tiled_gradients(stream); histograms(stream); splitting_and_routing(stream); resident_frontiers(stream); wide_numeric_scans(stream);
    std::cout << "Passed " << checks << " independent gradient/loss/histogram/split/routing checks.\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << "kernel correctness: " << error.what() << '\n'; return 1; }
}
