#include "ghb/booster.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::size_t checks{};
void require(bool value, const std::string& message) {
  ++checks;
  if (!value) throw std::runtime_error(message);
}
void close(double actual, double expected, const std::string& message, double tolerance = 2e-9) {
  require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance * std::max({1.0, std::abs(actual), std::abs(expected)}),
          message + ": expected " + std::to_string(expected) + ", got " + std::to_string(actual));
}
template<class Function> void rejected(Function&& function, const char* message) {
  bool threw{};
  try { function(); } catch (const std::exception&) { threw = true; }
  require(threw, message);
}
void same(const std::vector<double>& actual, const std::vector<double>& expected, const std::string& message, double tolerance = 2e-9) {
  require(actual.size() == expected.size(), message + " shape mismatch");
  for (std::size_t i = 0; i < actual.size(); ++i) close(actual[i], expected[i], message, tolerance);
}
double sigmoid(double margin) {
  return margin >= 0 ? 1.0 / (1.0 + std::exp(-margin)) : std::exp(margin) / (1.0 + std::exp(margin));
}

// A deterministic held-out set uses shifted numeric values and different row
// order, while retaining the same simple target-generating rule. It is a
// correctness fixture, not evidence about generalization on real datasets.
ghb::Dataset fixture(ghb::Objective objective, std::uint32_t outputs = 1, std::uint32_t rows = 120, bool held_out = false) {
  ghb::Dataset data;
  data.rows = rows; data.columns = 3; data.outputs = objective == ghb::Objective::multiclass_softmax ? 1 : outputs;
  data.feature_types = {ghb::FeatureType::numeric, ghb::FeatureType::categorical, ghb::FeatureType::numeric};
  data.values.resize(rows * data.columns); data.weights.resize(rows); data.targets.resize(rows * data.outputs);
  const auto missing = std::numeric_limits<float>::quiet_NaN();
  for (std::uint32_t row = 0; row < rows; ++row) {
    const auto index = held_out ? row * 5 + 3 : row;
    float x = float(int((index * 7) % 23) - 11) / 5 + (held_out ? .015f : 0.f);
    float category = float(10 * (1 + (index / 4) % 3));
    if (index % 17 == 0) x = missing;
    if (index % 13 == 0) category = missing;
    if (held_out && row < 2) category = 99; // Unseen categories follow the learned missing route.
    const double x_signal = std::isnan(x) ? .7 : x > 0 ? 1 : -1;
    const double category_signal = std::isnan(category) || category == 99 ? -.5 : category == 10 ? -1 : category == 20 ? .25 : 1;
    data.values[row * 3] = x; data.values[row * 3 + 1] = category; data.values[row * 3 + 2] = float((index * 11) % 19) / 19;
    data.weights[row] = row % 29 == 0 ? 0.f : row % 3 == 0 ? 2.f : 1.f;
    if (objective == ghb::Objective::multiclass_softmax) {
      data.targets[row] = std::isnan(category) || category == 99 ? float(x_signal > 0 ? 2 : 0) : float(category / 10 - 1);
    } else for (std::uint32_t output = 0; output < outputs; ++output) {
      const double sign = output % 2 ? -1 : 1;
      data.targets[row * outputs + output] = objective == ghb::Objective::squared_error
          ? float(sign * (1 + output % 3) * x_signal + .3 * (int(output % 5) - 2) * category_signal + .01 * output)
          : float(sign * x_signal + .25 * (int(output % 3) - 1) * category_signal > 0);
    }
  }
  return data;
}

std::vector<double> output_losses(const ghb::Dataset& data, ghb::Objective objective, std::uint32_t outputs,
                                  const std::vector<double>& margins) {
  require(margins.size() == std::size_t(data.rows) * outputs, "loss margin shape");
  std::vector<long double> sums(objective == ghb::Objective::multiclass_softmax ? 1 : outputs);
  long double total_weight{};
  for (std::uint32_t row = 0; row < data.rows; ++row) {
    const double weight = data.weights.empty() ? 1 : data.weights[row]; total_weight += weight;
    if (objective == ghb::Objective::multiclass_softmax) {
      double maximum = *std::max_element(margins.begin() + row * outputs, margins.begin() + (row + 1) * outputs);
      long double denominator{};
      for (std::uint32_t output = 0; output < outputs; ++output) denominator += std::exp(static_cast<long double>(margins[row * outputs + output] - maximum));
      sums[0] += weight * (maximum - margins[row * outputs + std::uint32_t(data.targets[row])] + std::log(denominator));
    } else for (std::uint32_t output = 0; output < outputs; ++output) {
      const auto index = row * outputs + output;
      const long double prediction = margins[index], target = data.targets[index];
      sums[output] += weight * (objective == ghb::Objective::squared_error ? .5L * (prediction - target) * (prediction - target) :
          std::max(prediction, 0.L) + std::log1p(std::exp(-std::abs(prediction))) - target * prediction);
    }
  }
  std::vector<double> result;
  for (auto value : sums) result.push_back(double(value / total_weight));
  return result;
}
double loss(const ghb::Dataset& data, const ghb::Model& model) {
  auto per_output = output_losses(data, model.objective, model.outputs, model.predict(data, true));
  return std::accumulate(per_output.begin(), per_output.end(), 0.0) / per_output.size();
}
ghb::TrainConfig configuration(ghb::Objective objective) {
  ghb::TrainConfig config;
  config.objective = objective; config.classes = 3; config.rounds = 10; config.max_depth = 3; config.max_bins = 32;
  config.min_leaf_rows = 2; config.learning_rate = .3; config.l2 = .5; config.min_gain = 1e-9;
  config.histogram = ghb::HistogramPolicy::global; config.nvtx = false;
  return config;
}

void verify_probabilities(const ghb::Model& model, const ghb::Dataset& data) {
  const auto raw = model.predict(data, true), predictions = model.predict(data);
  same(model.predict_gpu(data, true), raw, "CPU/GPU raw prediction agreement");
  same(model.predict_gpu(data), predictions, "CPU/GPU transformed prediction agreement");
  require(predictions.size() == std::size_t(data.rows) * model.outputs, "prediction dimensions");
  for (std::uint32_t row = 0; row < data.rows; ++row) {
    double sum{};
    for (std::uint32_t output = 0; output < model.outputs; ++output) {
      const auto index = row * model.outputs + output;
      if (model.objective == ghb::Objective::squared_error) close(predictions[index], raw[index], "regression identity transform");
      else {
        require(predictions[index] >= 0 && predictions[index] <= 1, "classification probability outside [0,1]");
        if (model.objective == ghb::Objective::binary_logistic) close(predictions[index], sigmoid(raw[index]), "independent sigmoid transform");
        sum += predictions[index];
      }
    }
    if (model.objective == ghb::Objective::multiclass_softmax) close(sum, 1, "softmax probability normalization");
  }
}

std::string serialized(const ghb::Model& model) {
  std::ostringstream output(std::ios::binary); model.save(output); return output.str();
}
ghb::Model deserialized(const std::string& bytes) {
  std::istringstream input(bytes, std::ios::binary); return ghb::Model::load(input);
}
void roundtrip(const ghb::Model& model, const ghb::Dataset& data) {
  const auto bytes = serialized(model);
  const auto restored = deserialized(bytes);
  same(restored.predict(data, true), model.predict(data, true), "serialized CPU margins", 0);
  same(restored.predict_gpu(data), model.predict(data), "serialized GPU predictions");
  require(serialized(restored) == bytes, "model serialization roundtrip changed bytes");
}

void equivalent_models(const ghb::Model& actual, const ghb::Model& expected) {
  require(actual.objective == expected.objective && actual.outputs == expected.outputs, "execution policy changed objective/outputs");
  same(actual.base_scores, expected.base_scores, "execution policy changed weighted base scores");
  require(actual.features.size() == expected.features.size(), "execution policy changed feature count");
  for (std::size_t f = 0; f < actual.features.size(); ++f)
    require(actual.features[f].type == expected.features[f].type && actual.features[f].cuts == expected.features[f].cuts &&
            actual.features[f].categories == expected.features[f].categories, "execution policy changed feature representation");
  require(actual.trees.size() == expected.trees.size(), "execution policy changed tree count");
  for (std::size_t t = 0; t < actual.trees.size(); ++t) {
    const auto& a = actual.trees[t]; const auto& b = expected.trees[t];
    require(a.output == b.output && a.nodes.size() == b.nodes.size(), "execution policy changed tree dimensions");
    for (std::size_t n = 0; n < a.nodes.size(); ++n) {
      const auto& x = a.nodes[n]; const auto& y = b.nodes[n];
      require(x.feature == y.feature && x.left == y.left && x.right == y.right && x.threshold == y.threshold && x.missing_left == y.missing_left,
              "execution policy changed tree topology or routing");
      close(x.value, y.value, "execution policy changed leaf values", 2e-8);
    }
  }
}

void resident_stages(const ghb::TrainingResult& result, const ghb::TrainConfig& config) {
  namespace gi = ghb::instrumentation;
  const std::size_t trees = std::size_t(config.rounds) * result.model.outputs;
  std::vector<unsigned> builds(trees), histograms(trees);
  std::size_t build_count{}, histogram_count{};
  for (const auto& sample : result.samples) {
    require(sample.host_end_ns >= sample.host_start_ns, "resident stage host interval reversed");
    if (sample.timing == gi::Timing::gpu)
      require(sample.gpu_ms && std::isfinite(*sample.gpu_ms) && *sample.gpu_ms >= 0, "resident GPU timing missing");
    if (sample.stage == gi::Stage::download || sample.stage == gi::Stage::upload)
      require(sample.context.depth < 0, "tree level contains a recorded host/device transfer");
    if (sample.stage == gi::Stage::tree_build || sample.stage == gi::Stage::histogram) {
      require(sample.context.round >= 0 && std::uint32_t(sample.context.round) < config.rounds && sample.context.output >= 0 &&
              std::uint32_t(sample.context.output) < result.model.outputs, "tree stage has invalid round/output context");
      const auto index = std::size_t(sample.context.round) * result.model.outputs + sample.context.output;
      if (sample.stage == gi::Stage::tree_build) {
        require(sample.timing == gi::Timing::gpu, "tree_build must time GPU work");
        ++builds[index]; ++build_count;
      } else if (sample.timing == gi::Timing::gpu) {
        require(sample.context.depth >= 0 && std::uint32_t(sample.context.depth) < std::max(config.max_depth, 1u),
                "terminal leaves triggered an unnecessary histogram level");
        ++histograms[index]; ++histogram_count;
      }
    }
    if (config.tree_execution == ghb::TreeExecution::graph)
      require(sample.stage != gi::Stage::histogram && sample.stage != gi::Stage::split_search && sample.stage != gi::Stage::route,
              "graph replay recorded individual captured tree stages");
  }
  require(build_count == trees && std::all_of(builds.begin(), builds.end(), [](auto n) { return n == 1; }),
          "each output tree requires exactly one tree_build sample");
  if (config.tree_execution == ghb::TreeExecution::stream)
    require(histogram_count == trees * std::max(config.max_depth, 1u) &&
            std::all_of(histograms.begin(), histograms.end(), [&](auto n) { return n == std::max(config.max_depth, 1u); }),
            "stream histogram schedule omitted or duplicated a level");
  else require(histogram_count == 0, "graph mode should expose the complete tree scope only");
}

void resident_execution() {
  for (const auto objective : {ghb::Objective::squared_error, ghb::Objective::binary_logistic, ghb::Objective::multiclass_softmax}) {
    const auto data = fixture(objective, objective == ghb::Objective::multiclass_softmax ? 1 : 7, 72);
    const auto held_out = fixture(objective, data.outputs, 31, true);
    auto config = configuration(objective); config.rounds = 2; config.max_depth = 2;
    config.output_tile_size = 3; config.record_stages = true;
    config.tree_execution = ghb::TreeExecution::stream;
    const auto stream = ghb::train(data, config);
    resident_stages(stream, config);
    config.tree_execution = ghb::TreeExecution::graph;
    const auto graph = ghb::train(data, config);
    resident_stages(graph, config);
    equivalent_models(graph.model, stream.model);
    same(graph.training_loss, stream.training_loss, "stream/graph objective history", 2e-8);
    same(graph.model.predict(held_out, true), stream.model.predict(held_out, true), "stream/graph held-out margins", 2e-8);
    verify_probabilities(graph.model, held_out); roundtrip(graph.model, held_out);
    for (const auto* result : {&stream, &graph}) {
      const auto tile = objective == ghb::Objective::multiclass_softmax ? result->model.outputs : 3u;
      require(result->gradient_bytes == std::size_t(data.rows) * tile * 2 * sizeof(double), "derivative tile allocation does not match reported bytes");
      require(result->device_bytes > 0 && result->device_bytes <= config.max_device_bytes, "resident payload violates device budget");
      require(result->preparation_peak_bytes > 0 && result->preparation_peak_bytes <= config.max_device_bytes, "feature preparation violates device budget");
      auto base = result->model; base.trees.clear();
      close(result->training_loss.front(), loss(data, base), "resident initial loss normalization");
      close(result->training_loss.back(), loss(data, result->model), "resident final loss normalization");
    }
    // A second exact quantizer must retain every feature cut, category and tree.
    config.quantize_policy = ghb::QuantizePolicy::radix4;
    config.record_stages = false;
    const auto radix4 = ghb::train(data, config);
    equivalent_models(radix4.model, graph.model);
    same(radix4.training_loss, graph.training_loss, "radix policy objective history", 2e-8);
  }
  // Depth zero still computes one root histogram, without a child frontier.
  auto config = configuration(ghb::Objective::squared_error);
  config.rounds = 1; config.max_depth = 0; config.record_stages = true;
  for (auto execution : {ghb::TreeExecution::stream, ghb::TreeExecution::graph}) {
    config.tree_execution = execution;
    const auto result = ghb::train(fixture(config.objective, 2, 32), config);
    resident_stages(result, config);
    for (const auto& tree : result.model.trees) require(tree.nodes.size() == 1, "depth-zero resident tree split");
  }
}

void end_to_end() {
  for (const auto objective : {ghb::Objective::squared_error, ghb::Objective::binary_logistic, ghb::Objective::multiclass_softmax}) {
    const std::uint32_t outputs = objective == ghb::Objective::squared_error ? 3 : 1;
    const auto data = fixture(objective, outputs), held_out = fixture(objective, outputs, 81, true);
    auto config = configuration(objective);
    if (objective == ghb::Objective::squared_error) config.histogram = ghb::HistogramPolicy::shared;
    if (objective == ghb::Objective::binary_logistic) config.histogram = ghb::HistogramPolicy::autotune;
    config.record_stages = objective == ghb::Objective::multiclass_softmax;
    auto trained = ghb::train(data, config);
    auto base = trained.model; base.trees.clear();
    require(trained.training_loss.size() == config.rounds + 1, "loss history must include initial and every completed round");
    close(trained.training_loss.front(), loss(data, base), "independent initial weighted objective");
    close(trained.training_loss.back(), loss(data, trained.model), "independent final weighted objective");
    require(trained.training_loss.back() < .8 * trained.training_loss.front(), "training failed to learn deterministic fixture");
    require(loss(held_out, trained.model) < .9 * loss(held_out, base), "held-out fixture loss did not improve over the base model");
    for (auto value : trained.training_loss) require(std::isfinite(value) && value >= 0, "invalid objective history value");
    require(trained.model.trees.size() == std::size_t(config.rounds) * trained.model.outputs, "one tree per output per round");
    for (auto value : {trained.quantize_ms, trained.upload_ms, trained.training_ms, trained.total_ms})
      require(std::isfinite(value) && value >= 0, "invalid trainer timing");
    if (config.histogram == ghb::HistogramPolicy::autotune) {
      require(!trained.tuning.empty(), "autotune did not record its policy measurements");
      for (const auto& decision : trained.tuning) {
        require(decision.selected == ghb::HistogramPolicy::global || decision.selected == ghb::HistogramPolicy::shared, "autotune selected an unresolved or external implementation");
        require(std::isfinite(decision.global_ms) && decision.global_ms >= 0 && std::isfinite(decision.shared_ms) && decision.shared_ms >= 0,
                "invalid histogram tuning timings");
      }
    }
    require(config.record_stages == !trained.samples.empty(), "recording enable/disable contract");
    if (config.record_stages) {
      bool saw_gradient{}, saw_histogram{}, saw_split{}, saw_evaluate{};
      for (const auto& sample : trained.samples) {
        require(sample.host_end_ns >= sample.host_start_ns, "reversed host instrumentation interval");
        if (sample.timing == ghb::instrumentation::Timing::gpu)
          require(sample.gpu_ms && std::isfinite(*sample.gpu_ms) && *sample.gpu_ms >= 0, "missing GPU stage timing");
        saw_gradient |= sample.stage == ghb::instrumentation::Stage::gradients;
        saw_histogram |= sample.stage == ghb::instrumentation::Stage::histogram;
        saw_split |= sample.stage == ghb::instrumentation::Stage::split_search;
        saw_evaluate |= sample.stage == ghb::instrumentation::Stage::evaluate;
      }
      require(saw_gradient && saw_histogram && saw_split && saw_evaluate, "requested training stages were not instrumented");
    }
    verify_probabilities(trained.model, held_out); roundtrip(trained.model, held_out);
  }
}

void zero_rounds_and_leaf_only() {
  for (auto objective : {ghb::Objective::squared_error, ghb::Objective::binary_logistic, ghb::Objective::multiclass_softmax}) {
    const auto data = fixture(objective, objective == ghb::Objective::multiclass_softmax ? 1 : 3, 60);
    auto config = configuration(objective); config.rounds = 0;
    const auto initial = ghb::train(data, config);
    require(initial.model.trees.empty() && initial.training_loss.size() == 1, "zero rounds must return only a base model");
    const auto prediction = initial.model.predict(data);
    const double total = std::accumulate(data.weights.begin(), data.weights.end(), 0.0);
    for (std::uint32_t output = 0; output < initial.model.outputs; ++output) {
      double weighted{};
      for (std::uint32_t row = 0; row < data.rows; ++row)
        weighted += data.weights[row] * (objective == ghb::Objective::multiclass_softmax ? double(data.targets[row] == output) : data.targets[row * data.outputs + output]);
      close(prediction[output], weighted / total, "weighted constant base prediction");
    }
    close(initial.training_loss.front(), loss(data, initial.model), "zero-round objective");
    verify_probabilities(initial.model, data);
    config.rounds = 2; config.max_depth = 0;
    const auto leaf_only = ghb::train(data, config);
    for (const auto& tree : leaf_only.model.trees)
      require(tree.nodes.size() == 1 && tree.nodes[0].feature == -1, "depth zero created a split");
  }
  ghb::Feature numeric; numeric.cuts = {0, 2};
  require(numeric.bins() == 4 && numeric.encode(-1) == 1 && numeric.encode(0) == 1 && numeric.encode(1) == 2 && numeric.encode(2) == 2 && numeric.encode(3) == 3,
          "numerical cuts must use inclusive lower-bound binning");
  require(numeric.encode(std::numeric_limits<float>::quiet_NaN()) == 0, "NaN must encode as missing");
  ghb::Feature category; category.type = ghb::FeatureType::categorical; category.categories = {10,20,30};
  require(category.bins() == 4 && category.encode(20) == 2 && category.encode(99) == 0, "categorical lookup/unseen handling");
}

void categorical_missing_only() {
  ghb::Dataset data; data.rows = 12; data.columns = 1; data.feature_types = {ghb::FeatureType::categorical};
  for (std::uint32_t row = 0; row < data.rows; ++row) {
    data.values.push_back(row < 4 ? std::numeric_limits<float>::quiet_NaN() : row < 8 ? 10.f : 20.f);
    data.targets.push_back(row < 4 ? -2.f : 2.f);
  }
  auto config = configuration(ghb::Objective::squared_error);
  config.rounds = 1; config.max_depth = 1; config.learning_rate = 1; config.l2 = 0; config.min_leaf_rows = 1;
  const auto trained = ghb::train(data, config);
  require(trained.model.trees[0].nodes[0].threshold == 0 && trained.model.trees[0].nodes[0].missing_left == 1,
          "categorical missing-only split was not retained");
  auto prediction = trained.model.predict(data);
  for (std::uint32_t row = 0; row < data.rows; ++row) close(prediction[row], data.targets[row], "categorical missing-only fit");
  auto unseen = data; unseen.rows = 1; unseen.values = {99}; unseen.targets.clear();
  close(trained.model.predict(unseen)[0], -2, "unseen category follows missing-only branch");
  roundtrip(trained.model, data);
}

void output_tiles() {
  for (auto objective : {ghb::Objective::squared_error, ghb::Objective::binary_logistic}) {
    const auto data = fixture(objective, 7, 72), held_out = fixture(objective, 7, 31, true);
    auto config = configuration(objective); config.rounds = 3; config.max_depth = 2; config.max_bins = 24;
    config.output_tile_size = 3;
    const auto uneven = ghb::train(data, config);
    require(uneven.model.outputs == 7, "independent outputs were collapsed");
    for (const auto width : {1u, 7u}) {
      config.output_tile_size = width;
      const auto other = ghb::train(data, config);
      same(other.model.predict(held_out, true), uneven.model.predict(held_out, true), "tile-size-independent model predictions", 2e-8);
      same(other.training_loss, uneven.training_loss, "tile-size-independent loss history", 2e-8);
    }
    verify_probabilities(uneven.model, held_out); roundtrip(uneven.model, held_out);
    const auto wide_data = fixture(objective, 65, 96);
    config.rounds = 2; config.max_depth = 1; config.output_tile_size = 16; config.max_bins = 24;
    // Covers persistent targets/margins, one bounded derivative tile, and ample
    // small-frontier storage; it deliberately cannot hold all 65 derivatives.
    config.max_device_bytes = std::size_t(wide_data.rows) * wide_data.outputs * (sizeof(float) + sizeof(double))
                            + std::size_t(wide_data.rows) * config.output_tile_size * 2 * sizeof(double) + 32768;
    const auto wide = ghb::train(wide_data, config);
    require(wide.model.outputs == 65 && wide.model.trees.size() == 130, "wide output coverage");
    auto base = wide.model; base.trees.clear();
    const auto before = output_losses(wide_data, objective, 65, base.predict(wide_data, true));
    const auto after = output_losses(wide_data, objective, 65, wide.model.predict(wide_data, true));
    for (std::uint32_t output = 0; output < 65; ++output)
      require(after[output] < before[output], "an output beyond the first derivative tile failed to learn");
    close(wide.training_loss.back(), std::accumulate(after.begin(), after.end(), 0.0) / 65, "wide independent objective average");
    same(wide.model.predict_gpu(wide_data, true), wide.model.predict(wide_data, true), "wide CPU/GPU predictions");
    auto full_derivatives = config; full_derivatives.output_tile_size = 65;
    rejected([&] { ghb::train(wide_data, full_derivatives); }, "device budget failed to distinguish bounded and full derivative workspace");
  }
}

void compact_constant_models() {
  auto data = fixture(ghb::Objective::squared_error, 4, 128);
  for (std::uint32_t row = 0; row < data.rows; ++row) for (std::uint32_t output = 0; output < data.outputs; ++output)
    data.targets[row * data.outputs + output] = float(output + 1);
  auto config = configuration(ghb::Objective::squared_error);
  config.rounds = 2; config.max_depth = 20; config.min_leaf_rows = 1;
  config.max_device_bytes = 1ULL << 20; config.max_histogram_bytes = 256ULL << 10;
  config.tree_execution = ghb::TreeExecution::graph;
  const auto trained = ghb::train(data, config);
  require(trained.device_bytes <= config.max_device_bytes && trained.preparation_peak_bytes <= config.max_device_bytes,
          "deep constant model exceeded a bounded device budget");
  for (const auto& tree : trained.model.trees) {
    require(tree.nodes.size() == 1 && tree.nodes[0].feature == -1, "constant targets created unnecessary split nodes");
    require(tree.nodes.capacity() < 128, "a one-node tree retained its worst-case node capacity");
  }
  close(trained.training_loss.back(), 0, "constant-target objective", 0);
  const auto predictions = trained.model.predict(data);
  for (std::size_t index = 0; index < predictions.size(); ++index) close(predictions[index], data.targets[index], "constant-target prediction", 0);
}

std::uint32_t read_u32(const std::string& bytes, std::size_t offset) {
  require(offset + 4 <= bytes.size(), "model fixture offset");
  std::uint32_t value{};
  for (int i = 0; i < 4; ++i) value |= std::uint32_t(static_cast<unsigned char>(bytes[offset + i])) << (8 * i);
  return value;
}
void write_u32(std::string& bytes, std::size_t offset, std::uint32_t value) {
  require(offset + 4 <= bytes.size(), "model mutation offset");
  for (int i = 0; i < 4; ++i) bytes[offset + i] = char((value >> (8 * i)) & 255);
}

void invalid_inputs_and_models() {
  const auto data = fixture(ghb::Objective::squared_error, 1, 48);
  auto config = configuration(ghb::Objective::squared_error); config.rounds = 2; config.max_depth = 2;
  const std::vector<std::function<void(ghb::Dataset&)>> bad_data{
    [](auto& x) { x.rows = 0; x.values.clear(); x.targets.clear(); x.weights.clear(); },
    [](auto& x) { x.values.pop_back(); }, [](auto& x) { x.targets.pop_back(); },
    [](auto& x) { x.weights.pop_back(); }, [](auto& x) { x.weights[0] = -1; },
    [](auto& x) { std::fill(x.weights.begin(), x.weights.end(), 0); },
    [](auto& x) { x.weights[0] = std::numeric_limits<float>::quiet_NaN(); },
    [](auto& x) { x.targets[0] = std::numeric_limits<float>::infinity(); },
    [](auto& x) { x.values[0] = std::numeric_limits<float>::infinity(); },
    [](auto& x) { x.outputs = 0; }, [](auto& x) { x.feature_types.pop_back(); },
    [](auto& x) { x.feature_types[0] = static_cast<ghb::FeatureType>(99); }
  };
  for (const auto& mutate : bad_data) { auto bad = data; mutate(bad); rejected([&] { ghb::train(bad, config); }, "invalid dataset accepted"); }
  const std::vector<std::function<void(ghb::TrainConfig&)>> bad_configs{
    [](auto& x) { x.learning_rate = 0; }, [](auto& x) { x.l2 = -1; },
    [](auto& x) { x.min_child_hessian = -1; }, [](auto& x) { x.min_gain = std::numeric_limits<double>::quiet_NaN(); },
    [](auto& x) { x.max_leaf_value = -1; }, [](auto& x) { x.min_leaf_rows = 0; },
    [](auto& x) { x.max_bins = 1; }, [](auto& x) { x.max_bins = 65537; },
    [](auto& x) { x.max_depth = 31; }, [](auto& x) { x.output_tile_size = 0; },
    [](auto& x) { x.max_device_bytes = 1; }, [](auto& x) { x.max_histogram_bytes = 1; },
    [](auto& x) { x.objective = static_cast<ghb::Objective>(99); },
    [](auto& x) { x.histogram = static_cast<ghb::HistogramPolicy>(99); },
    [](auto& x) { x.tree_execution = static_cast<ghb::TreeExecution>(99); },
    [](auto& x) { x.quantize_policy = static_cast<ghb::QuantizePolicy>(99); }
  };
  for (const auto& mutate : bad_configs) { auto bad = config; mutate(bad); rejected([&] { ghb::train(data, bad); }, "invalid configuration or insufficient budget accepted"); }
  auto binary = fixture(ghb::Objective::binary_logistic); binary.targets[0] = .25f;
  rejected([&] { ghb::train(binary, configuration(ghb::Objective::binary_logistic)); }, "non-binary label accepted");
  auto multiclass = fixture(ghb::Objective::multiclass_softmax); multiclass.targets[0] = 3;
  rejected([&] { ghb::train(multiclass, configuration(ghb::Objective::multiclass_softmax)); }, "out-of-range multiclass label accepted");
  auto cardinality = config; cardinality.max_bins = 3;
  rejected([&] { ghb::train(data, cardinality); }, "categorical cardinality ignored reserved missing bin");
  const auto model = ghb::train(data, config).model;
  const auto bytes = serialized(model);
  rejected([&] { deserialized(bytes.substr(0, bytes.size() - 1)); }, "truncated model accepted");
  rejected([&] { deserialized(bytes + "junk"); }, "trailing model bytes accepted");
  auto corrupt = bytes; corrupt[0] ^= 1;
  rejected([&] { deserialized(corrupt); }, "invalid model magic accepted");
  corrupt = bytes; write_u32(corrupt, 8, 99);
  rejected([&] { deserialized(corrupt); }, "unsupported model version accepted");
  corrupt = bytes; write_u32(corrupt, 24, 0xffffffff); write_u32(corrupt, 28, 0xffffffff);
  rejected([&] { deserialized(corrupt); }, "unbounded serialized allocation accepted");
  corrupt = bytes;
  const auto nan_bits = std::bit_cast<std::uint64_t>(std::numeric_limits<double>::quiet_NaN());
  for (int i = 0; i < 8; ++i) corrupt[32 + i] = char((nan_bits >> (8 * i)) & 255);
  rejected([&] { deserialized(corrupt); }, "nonfinite serialized base score accepted");
  std::size_t feature_offset = 32 + std::size_t(model.outputs) * sizeof(double), tree_offset = feature_offset;
  for (std::size_t feature = 0; feature < model.features.size(); ++feature)
    tree_offset += 12 + 4 * std::size_t(read_u32(bytes, tree_offset + 4) + read_u32(bytes, tree_offset + 8));
  require(model.trees[0].nodes[0].feature >= 0, "malformed-model fixture requires a split tree");
  corrupt = bytes; write_u32(corrupt, tree_offset + 8 + 4, 0);
  rejected([&] { deserialized(corrupt); }, "serialized tree cycle accepted");
  corrupt = bytes; write_u32(corrupt, feature_offset, 99);
  rejected([&] { deserialized(corrupt); }, "unknown serialized feature type accepted");
  for (const auto& mutate : std::vector<std::function<void(ghb::Model&)>>{
      [](auto& x) { x.base_scores[0] = std::numeric_limits<double>::infinity(); },
      [](auto& x) { x.trees[0].nodes[0].left = 0; },
      [](auto& x) { x.trees[0].nodes.push_back(ghb::Node{}); },
      [](auto& x) { x.trees[0].output = x.outputs; },
      [](auto& x) { x.features[0].cuts = {2,1}; }}) {
    auto bad = model; mutate(bad);
    rejected([&] { bad.predict(data); }, "malformed in-memory model accepted for prediction");
    rejected([&] { serialized(bad); }, "malformed in-memory model accepted for serialization");
  }
  auto prediction_only = data; prediction_only.targets.clear(); prediction_only.weights.clear();
  same(model.predict(prediction_only), model.predict(data), "prediction should not require labels/weights", 0);
  auto empty = prediction_only; empty.rows = 0; empty.values.clear();
  require(model.predict(empty).empty() && model.predict_gpu(empty).empty(), "empty inference must return no predictions");
  auto wrong = prediction_only; wrong.values.pop_back();
  rejected([&] { model.predict(wrong); }, "malformed prediction matrix accepted");
}
} // namespace

int main() {
  try {
    int devices{};
    const auto status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
      std::cout << "SKIP: no CUDA device/driver\n"; return 77;
    }
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    zero_rounds_and_leaf_only(); categorical_missing_only(); end_to_end(); resident_execution(); output_tiles(); compact_constant_models(); invalid_inputs_and_models();
    std::cout << "Passed " << checks << " independent custom trainer, held-out prediction, output-tiling, and model-validation checks.\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << "booster correctness: " << error.what() << '\n'; return 1; }
}
