#include "ghb/booster.hpp"
#include "ghb/batch_resident.cuh"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::size_t checks{};
std::string case_context;
void require(bool condition, const std::string& message) { ++checks; if (!condition) throw std::runtime_error(message + " [" + case_context + "]"); }
std::string node_fields(const ghb::Node& node) {
  std::ostringstream out;
  out << std::setprecision(17) << "feature=" << node.feature << ",threshold=" << node.threshold
      << ",missing_left=" << node.missing_left << ",left=" << node.left << ",right=" << node.right << ",value=" << node.value;
  return out.str();
}
// These bounds scale with the finite FP64 reductions in a small correctness
// fixture. They do not change the separate zero-allowance quality evaluator.
double tolerance(unsigned rows, unsigned rounds) {
  return 64 * std::numeric_limits<double>::epsilon() * (rows + 1) * (rounds + 1);
}
void close(double a, double b, double allowed, const std::string& message) {
  require(std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= allowed * std::max({1.0, std::abs(a), std::abs(b)}), message);
}
void same(const std::vector<double>& a, const std::vector<double>& b, double allowed, const std::string& message) {
  require(a.size() == b.size(), message + " extent");
  for (std::size_t i = 0; i < a.size(); ++i) close(a[i], b[i], allowed, message + " index " + std::to_string(i));
}
template<class Function> void rejected(Function&& function, const std::string& message) {
  bool failed{};
  try { function(); } catch (const std::exception&) { failed = true; }
  require(failed, message);
}
std::uint32_t hash(std::uint32_t value) {
  value ^= value >> 16; value *= 0x7feb352dU; value ^= value >> 15; value *= 0x846ca68bU; return value ^ (value >> 16);
}
ghb::Dataset fixture(ghb::Objective objective, unsigned outputs, unsigned rows = 257, bool held_out = false) {
  ghb::Dataset data;
  data.rows = rows; data.columns = 3; data.outputs = objective == ghb::Objective::multiclass_softmax ? 1 : outputs;
  data.feature_types = {ghb::FeatureType::numeric, ghb::FeatureType::categorical, ghb::FeatureType::numeric};
  data.values.resize(std::size_t(rows) * data.columns); data.weights.resize(rows); data.targets.resize(std::size_t(rows) * data.outputs);
  for (unsigned row = 0; row < rows; ++row) {
    const unsigned index = row + (held_out ? 1009 : 0);
    const float raw_x = float(int(hash(index + 11) % 10007) - 5003) / 1009;
    const float raw_z = float(int(hash(index + 97) % 10009) - 5004) / 997;
    const unsigned category = hash(index + 257) % 7;
    const float x = index % 17 == 0 ? std::numeric_limits<float>::quiet_NaN() : raw_x;
    float cat = index % 23 == 0 ? std::numeric_limits<float>::quiet_NaN() : float(category * 13 + 3);
    if (held_out && row % 31 == 0) cat = 999; // Independent unseen-category route.
    data.values[std::size_t(row) * 3] = x;
    data.values[std::size_t(row) * 3 + 1] = cat;
    data.values[std::size_t(row) * 3 + 2] = raw_z;
    data.weights[row] = row % 29 == 0 ? 0 : .125f + float(hash(index + 503) % 1009) / 997;
    if (objective == ghb::Objective::multiclass_softmax) {
      // Every class occurs, while weights distinguish candidate gains.
      data.targets[row] = float((row * 7 + row / outputs) % outputs);
    } else for (unsigned output = 0; output < outputs; ++output) {
      const double missing_signal = std::isnan(x) ? .337 : raw_x;
      const double score = (output % 2 ? -1 : 1) * missing_signal + .219 * raw_z +
                           .137 * (int(category) - 3) + .031 * (int(output % 5) - 2);
      data.targets[std::size_t(row) * outputs + output] = objective == ghb::Objective::squared_error
          ? float(score + .087 * std::sin(raw_z * (1 + output % 3)) + .0001 * hash(index + output) / UINT32_MAX)
          : float(score > .017 * (int(output % 7) - 3));
    }
  }
  return data;
}
ghb::TrainConfig configuration(ghb::Objective objective, unsigned outputs, unsigned depth, unsigned bins) {
  ghb::TrainConfig config;
  config.objective = objective; config.classes = outputs; config.rounds = 2; config.max_depth = depth; config.max_bins = bins;
  config.min_leaf_rows = 7; config.learning_rate = .173; config.l2 = 1.137; config.min_gain = 1e-7;
  config.max_leaf_value = .719; config.histogram = ghb::HistogramPolicy::global;
  config.output_tile_size = 16; config.tree_export_batch_size = 3; config.nvtx = false;
  return config;
}
double reference_loss(const ghb::Dataset& data, const ghb::Model& model, const std::vector<double>& margins) {
  long double sum{}, weight_sum{};
  const unsigned outputs = model.outputs;
  for (unsigned row = 0; row < data.rows; ++row) {
    const long double weight = data.weights.empty() ? 1 : data.weights[row]; weight_sum += weight;
    if (model.objective == ghb::Objective::multiclass_softmax) {
      const auto begin = margins.begin() + std::size_t(row) * outputs;
      const long double maximum = *std::max_element(begin, begin + outputs);
      long double exponential{};
      for (unsigned output = 0; output < outputs; ++output) exponential += std::exp(static_cast<long double>(begin[output]) - maximum);
      sum += weight * (maximum + std::log(exponential) - begin[unsigned(data.targets[row])]);
    } else for (unsigned output = 0; output < outputs; ++output) {
      const auto index = std::size_t(row) * outputs + output;
      const long double margin = margins[index], target = data.targets[index];
      sum += weight * (model.objective == ghb::Objective::squared_error ? .5L * (margin - target) * (margin - target) :
          std::max(0.L, margin) + std::log1p(std::exp(-std::abs(margin))) - margin * target);
    }
  }
  return double(sum / weight_sum / (model.objective == ghb::Objective::multiclass_softmax ? 1 : outputs));
}
void compare_models(const ghb::Model& actual, const ghb::Model& reference, double allowed) {
  require(actual.objective == reference.objective && actual.outputs == reference.outputs, "objective/output contract");
  same(actual.base_scores, reference.base_scores, 0, "exact GPU base scores");
  require(actual.features.size() == reference.features.size(), "feature metadata extent");
  for (std::size_t feature = 0; feature < actual.features.size(); ++feature) {
    const auto& a = actual.features[feature]; const auto& b = reference.features[feature];
    require(a.type == b.type && a.cuts == b.cuts && a.categories == b.categories, "exact feature binning");
  }
  require(actual.trees.size() == reference.trees.size(), "tree count");
  for (std::size_t index = 0; index < actual.trees.size(); ++index) {
    const auto& a = actual.trees[index]; const auto& b = reference.trees[index];
    require(a.output == b.output && a.nodes.size() == b.nodes.size(), "tree order/dimensions at " + std::to_string(index));
    for (std::size_t node = 0; node < a.nodes.size(); ++node) {
      const auto& x = a.nodes[node]; const auto& y = b.nodes[node];
      require(x.feature == y.feature && x.threshold == y.threshold && x.missing_left == y.missing_left &&
              x.left == y.left && x.right == y.right, "tree topology at " + std::to_string(index) + "/" + std::to_string(node) +
              " actual{" + node_fields(x) + "} expected{" + node_fields(y) + "}");
      close(x.value, y.value, allowed, "leaf value");
    }
  }
}
void predictions_and_objectives(const ghb::TrainingResult& result, const ghb::TrainingResult& reference,
                                 const ghb::Dataset& data, const ghb::Dataset& validation, unsigned rounds) {
  const double allowed = tolerance(data.rows, rounds);
  double training_difference{}, validation_difference{};
  for (const auto* dataset : {&data, &validation}) {
    const auto actual = result.model.predict(*dataset, true), expected = reference.model.predict(*dataset, true);
    for (std::size_t i = 0; i < actual.size(); ++i)
      (dataset == &data ? training_difference : validation_difference) =
          std::max(dataset == &data ? training_difference : validation_difference, std::abs(actual[i] - expected[i]));
  }
  std::ostringstream differences;
  differences << std::setprecision(17) << " max_train_margin_delta=" << training_difference << " max_validation_margin_delta=" << validation_difference;
  case_context += differences.str();
  compare_models(result.model, reference.model, allowed);
  same(result.training_loss, reference.training_loss, allowed, "training objective trajectory");
  require(result.training_loss.size() == std::size_t(rounds) + 1, "objective history count");
  for (const auto* fixture_data : {&data, &validation}) {
    const auto raw = result.model.predict(*fixture_data, true), probability = result.model.predict(*fixture_data);
    same(raw, reference.model.predict(*fixture_data, true), allowed, "baseline margin comparison");
    same(probability, reference.model.predict(*fixture_data), allowed, "baseline probability comparison");
    same(result.model.predict_gpu(*fixture_data, true), raw, 64 * std::numeric_limits<double>::epsilon() * (rounds + 1), "CPU/GPU raw prediction reference");
    same(result.model.predict_gpu(*fixture_data), probability, 256 * std::numeric_limits<double>::epsilon() * (rounds + 1), "CPU/GPU transformed reference");
    for (unsigned row = 0; row < fixture_data->rows; ++row) {
      long double sum{};
      for (unsigned output = 0; output < result.model.outputs; ++output) {
        const auto index = std::size_t(row) * result.model.outputs + output;
        require(std::isfinite(raw[index]), "finite margin");
        if (result.model.objective != ghb::Objective::squared_error) require(probability[index] >= 0 && probability[index] <= 1, "valid probability");
        if (result.model.objective == ghb::Objective::binary_logistic) {
          const double expected = raw[index] >= 0 ? 1 / (1 + std::exp(-raw[index])) : std::exp(raw[index]) / (1 + std::exp(raw[index]));
          close(probability[index], expected, 16 * std::numeric_limits<double>::epsilon(), "independent sigmoid");
        }
        sum += probability[index];
      }
      if (result.model.objective == ghb::Objective::multiclass_softmax)
        close(double(sum), 1, 64 * std::numeric_limits<double>::epsilon() * result.model.outputs, "softmax row normalization");
    }
  }
  close(result.training_loss.back(), reference_loss(data, result.model, result.model.predict(data, true)), allowed, "independent weighted objective");
  std::ostringstream serialized(std::ios::binary); result.model.save(serialized);
  std::istringstream input(serialized.str(), std::ios::binary); const auto restored = ghb::Model::load(input);
  same(restored.predict(validation, true), result.model.predict(validation, true), 0, "serialized independent trees");
}
std::size_t total_bins(const ghb::Model& model) {
  std::size_t bins{}; for (const auto& feature : model.features) bins += feature.bins(); return bins;
}
void memory_contract(const ghb::TrainingResult& result, const ghb::Dataset& data, const ghb::TrainConfig& config,
                     bool full_frontier = true) {
  const std::size_t batch = result.tree_batch_size, outputs = result.model.outputs, bins = total_bins(result.model);
  const std::size_t expected_counts = config.root_counts == ghb::RootCountPolicy::per_output ? 0 : bins * sizeof(unsigned long long);
  require(batch && batch <= std::min<std::size_t>(outputs, config.output_tile_size), "bounded tree batch");
  require(result.root_count_bytes == expected_counts, "exact root count cache bytes");
  require(result.root_histogram_bytes == 0 && result.root_split_bytes == 0, "batch builder removes separate root caches");
  require(result.root_histogram_batch_size == batch, "root/tree batch agreement");
  const auto hist = result.histogram_bytes - expected_counts;
  require(hist % (batch * bins * sizeof(ghb::gpu::Stats)) == 0, "integral histogram capacity");
  const std::size_t capacity = hist / (batch * bins * sizeof(ghb::gpu::Stats));
  const std::size_t leaves = std::max<std::size_t>(1, data.rows / config.min_leaf_rows);
  const std::size_t desired = std::min<std::size_t>(std::size_t(1) << (config.max_depth ? config.max_depth - 1 : 0), leaves);
  const std::size_t nodes = std::min<std::size_t>((std::size_t(1) << (config.max_depth + 1)) - 1, 2 * leaves - 1);
  if (full_frontier) require(capacity == desired, "batch planner preserves desired frontier");
  require(capacity && capacity <= desired, "bounded frontier");
  const std::size_t history = config.record_stages && config.tree_execution == ghb::TreeExecution::stream ? std::max(1U, config.max_depth) : 0;
  const std::size_t state_bytes = batch * (std::size_t(data.rows) * sizeof(int) + nodes * sizeof(ghb::Node) +
      capacity * (4 * sizeof(int) + sizeof(unsigned)) + (1 + (capacity - 1) / 1024) * sizeof(unsigned) + sizeof(ghb::gpu::TreeState) + sizeof(unsigned));
  require(result.tree_state_bytes == state_bytes, "exact retained state bytes");
  const std::size_t derivative_outputs = config.objective == ghb::Objective::multiclass_softmax ? outputs : batch;
  require(result.gradient_bytes == std::size_t(data.rows) * derivative_outputs * 2 * sizeof(double), "derivative snapshot/tile bytes");
  const std::size_t quantized = ((std::size_t(data.rows) * data.columns * sizeof(std::uint16_t) + 3) & ~std::size_t(3)) +
      (std::size_t(data.columns) + 1) * sizeof(unsigned) + std::size_t(data.columns) * sizeof(ghb::FeatureType);
  const std::size_t loss_blocks = std::min(4096U, 1 + (data.rows - 1) / 256);
  const std::size_t init_partials = std::min(256U, 1 + (data.rows - 1) / 1024) * (outputs + 1);
  const std::size_t expected_device = quantized + std::size_t(data.rows) * outputs * sizeof(double) +
      (data.targets.size() + data.weights.size()) * sizeof(float) + outputs * sizeof(double) +
      (loss_blocks + init_partials + 2) * sizeof(double) + sizeof(unsigned) + sizeof(ghb::gpu::OutputBatch) +
      result.gradient_bytes + result.histogram_bytes + batch * capacity * (data.columns + 1) * sizeof(ghb::gpu::Split) +
      state_bytes + batch * history * sizeof(unsigned);
  require(result.device_bytes == expected_device, "exact complete device payload");
  require(std::max(result.device_bytes, result.preparation_peak_bytes) <= config.max_device_bytes, "preparation/training device budget");
  require(result.histogram_bytes <= config.max_histogram_bytes, "histogram budget");
  const std::size_t exports = std::min<std::size_t>({config.tree_export_batch_size, batch, (64ULL << 20) / (nodes * sizeof(ghb::Node))});
  require(result.tree_export_batch_size == exports, "bounded pinned node chunk");
  require(result.pinned_export_bytes == exports * nodes * sizeof(ghb::Node) + batch * sizeof(ghb::gpu::TreeState) +
      sizeof(ghb::gpu::OutputBatch) + batch * history * sizeof(unsigned), "pinned export payload");
}
unsigned frontier_at(const ghb::Tree& tree, unsigned depth) {
  std::vector<unsigned> frontier{0};
  for (unsigned level = 0; level < depth; ++level) {
    std::vector<unsigned> next;
    for (const unsigned index : frontier) {
      const auto& node = tree.nodes[index];
      if (node.feature >= 0) { next.push_back(unsigned(node.left)); next.push_back(unsigned(node.right)); }
    }
    frontier = std::move(next);
  }
  return unsigned(frontier.size());
}
void stages(const ghb::TrainingResult& result, const ghb::TrainConfig& config) {
  namespace gi = ghb::instrumentation;
  if (!config.record_stages) { require(result.samples.empty(), "disabled instrumentation has no samples"); return; }
  const unsigned batch = result.tree_batch_size, outputs = result.model.outputs, levels = std::max(1U, config.max_depth);
  const std::size_t batches = (outputs + batch - 1) / batch, expected = std::size_t(config.rounds) * batches;
  std::size_t builds{}, histograms{}, splits{}, routes{}, prediction{}, exports{}, operations{};
  for (const auto& sample : result.samples) {
    require(sample.host_end_ns >= sample.host_start_ns, "monotonic host scope");
    if (sample.timing == gi::Timing::gpu) require(sample.gpu_ms && std::isfinite(*sample.gpu_ms) && *sample.gpu_ms >= 0, "completed GPU timing");
    if (sample.stage == gi::Stage::upload || sample.stage == gi::Stage::download)
      require(sample.context.depth < 0, "no recorded host/device transfer inside a tree level");
    const auto& context = sample.context;
    if (context.round < 0 || context.output < 0) continue;
    require(unsigned(context.round) < config.rounds && unsigned(context.output) < outputs, "valid tile context");
    const unsigned width = std::min(batch, outputs - unsigned(context.output));
    require(unsigned(context.output) % batch == 0 && context.operations == width, "full/short tile instrumentation width");
    if (sample.stage == gi::Stage::tree_build) { ++builds; operations += context.operations; }
    if (sample.stage == gi::Stage::download) ++exports;
    if (sample.stage == gi::Stage::prediction) ++prediction;
    if (sample.stage == gi::Stage::histogram || sample.stage == gi::Stage::split_search || sample.stage == gi::Stage::route) {
      require(config.tree_execution == ghb::TreeExecution::stream && context.depth >= 0 && unsigned(context.depth) < levels, "per-level stream scope only");
      unsigned active{};
      for (unsigned output = unsigned(context.output); output < unsigned(context.output) + width; ++output)
        active += frontier_at(result.model.trees[std::size_t(context.round) * outputs + output], unsigned(context.depth));
      require(context.active_nodes == active, "GPU frontier history agrees with exported topology");
      histograms += sample.stage == gi::Stage::histogram; splits += sample.stage == gi::Stage::split_search; routes += sample.stage == gi::Stage::route;
    }
  }
  require(builds == expected && exports == expected && operations == std::size_t(config.rounds) * outputs, "complete tile build/export counts");
  const bool stream = config.tree_execution == ghb::TreeExecution::stream;
  require(histograms == (stream ? expected * levels : 0) && splits == histograms && routes == histograms, "complete level stage counts");
  require(prediction == (stream ? expected : 0), "one prediction stage per completed tile");
}

void cases() {
  for (const auto objective : {ghb::Objective::squared_error, ghb::Objective::binary_logistic, ghb::Objective::multiclass_softmax}) {
    const unsigned outputs = 7;
    const auto data = fixture(objective, outputs), validation = fixture(objective, outputs, 73, true);
    for (const unsigned depth : {0U, 2U, 5U}) for (const unsigned bins : {17U, 64U}) {
      auto config = configuration(objective, outputs, depth, bins);
      const auto baseline = ghb::train(data, config);
      for (const auto execution : {ghb::TreeExecution::stream, ghb::TreeExecution::graph}) {
        case_context = "objective=" + std::to_string(unsigned(objective)) + " outputs=7 depth=" + std::to_string(depth) +
            " bins=" + std::to_string(bins) + " execution=" + std::to_string(unsigned(execution));
        config.tree_build = ghb::TreeBuildPolicy::output_batch; config.tree_execution = execution; config.record_stages = true;
        const auto result = ghb::train(data, config);
        require(result.tree_batch_size == outputs, "unconstrained output batch width");
        predictions_and_objectives(result, baseline, data, validation, config.rounds); memory_contract(result, data, config); stages(result, config);
      }
    }
    // This is a true short third tile, including multiclass frozen derivatives.
    const auto wide = fixture(objective, 33), held = fixture(objective, 33, 41, true);
    auto config = configuration(objective, 33, 2, 17);
    const auto baseline = ghb::train(wide, config);
    for (const auto execution : {ghb::TreeExecution::stream, ghb::TreeExecution::graph}) {
      case_context = "objective=" + std::to_string(unsigned(objective)) + " outputs=33 depth=2 bins=17 execution=" + std::to_string(unsigned(execution));
      config.tree_build = ghb::TreeBuildPolicy::output_batch; config.tree_execution = execution; config.record_stages = true;
      config.tree_export_batch_size = execution == ghb::TreeExecution::stream ? 0 : 16;
      const auto result = ghb::train(wide, config);
      require(result.tree_batch_size == 16, "wide batch capacity");
      predictions_and_objectives(result, baseline, wide, held, config.rounds); memory_contract(result, wide, config); stages(result, config);
    }
  }
  for (const auto objective : {ghb::Objective::squared_error, ghb::Objective::binary_logistic}) {
    case_context = "scalar objective=" + std::to_string(unsigned(objective)) + " depth=5 bins=64 graph";
    const auto data = fixture(objective, 1), held = fixture(objective, 1, 47, true);
    auto config = configuration(objective, 1, 5, 64);
    const auto baseline = ghb::train(data, config);
    config.tree_build = ghb::TreeBuildPolicy::output_batch; config.tree_execution = ghb::TreeExecution::graph;
    const auto result = ghb::train(data, config);
    predictions_and_objectives(result, baseline, data, held, config.rounds); memory_contract(result, data, config); stages(result, config);
  }
}
void budgets_and_policies() {
  case_context = "budget/policy cases";
  auto data = fixture(ghb::Objective::squared_error, 33);
  auto config = configuration(ghb::Objective::squared_error, 33, 5, 17);
  config.tree_build = ghb::TreeBuildPolicy::output_batch; config.tree_execution = ghb::TreeExecution::graph;
  const auto unrestricted = ghb::train(data, config);
  const auto bins = total_bins(unrestricted.model);
  const std::size_t frontier = std::size_t(1) << (config.max_depth - 1);
  config.max_histogram_bytes = unrestricted.root_count_bytes + 3 * frontier * bins * sizeof(ghb::gpu::Stats);
  const auto constrained = ghb::train(data, config);
  require(constrained.tree_batch_size == 3, "histogram budget shrinks tile before frontier");
  memory_contract(constrained, data, config);
  predictions_and_objectives(constrained, unrestricted, data, fixture(config.objective, 33, 43, true), config.rounds);
  auto one_less = config; --one_less.max_histogram_bytes;
  const auto smaller = ghb::train(data, one_less);
  require(smaller.tree_batch_size == 2, "one-byte budget boundary chooses next smaller tile"); memory_contract(smaller, data, one_less);
  auto no_root = config; no_root.max_histogram_bytes = unrestricted.root_count_bytes + bins * sizeof(ghb::gpu::Stats) - 1;
  rejected([&] { ghb::train(data, no_root); }, "insufficient root histogram budget rejected");
  auto device_limited = config; device_limited.max_histogram_bytes = 512ULL << 20;
  device_limited.max_device_bytes = std::max(constrained.device_bytes, constrained.preparation_peak_bytes);
  const auto device_result = ghb::train(data, device_limited);
  require(device_result.tree_batch_size >= 3 && device_result.tree_batch_size < 16, "device budget also limits retained output state");
  memory_contract(device_result, data, device_limited);
  for (const auto counts : {ghb::RootCountPolicy::per_output, ghb::RootCountPolicy::reuse_shared}) {
    auto alternate = configuration(config.objective, 33, 2, 17); alternate.tree_build = ghb::TreeBuildPolicy::output_batch;
    alternate.root_counts = counts; alternate.histogram = ghb::HistogramPolicy::shared;
    const auto reference = ghb::train(data, configuration(config.objective, 33, 2, 17));
    const auto result = ghb::train(data, alternate);
    predictions_and_objectives(result, reference, data, fixture(config.objective, 33, 43, true), alternate.rounds);
    memory_contract(result, data, alternate);
  }
  auto empty = configuration(config.objective, 33, 2, 17); empty.tree_build = ghb::TreeBuildPolicy::output_batch; empty.rounds = 0;
  const auto zero = ghb::train(data, empty);
  require(zero.model.trees.empty() && zero.training_loss.size() == 1, "zero-round batch trainer"); memory_contract(zero, data, empty);
  for (const auto& change : std::vector<std::function<void(ghb::TrainConfig&)>>{
      [](auto& c) { c.tree_build = static_cast<ghb::TreeBuildPolicy>(999); },
      [](auto& c) { c.root_histogram = ghb::RootHistogramPolicy::per_tree; },
      [](auto& c) { c.split_batch = ghb::SplitBatchPolicy::per_tree; },
      [](auto& c) { c.output_tile_size = 0; }}) {
    auto invalid = config; change(invalid); rejected([&] { ghb::train(data, invalid); }, "invalid batch configuration rejected");
  }
}
} // namespace

int main() {
  int devices{}; if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
  try { cases(); budgets_and_policies(); std::cout << "batch training checks passed: " << checks << '\n'; return 0; }
  catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
