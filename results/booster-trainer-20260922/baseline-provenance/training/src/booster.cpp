#include "ghb/booster.hpp"
#include "ghb/kernels.cuh"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <istream>
#include <limits>
#include <map>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ghb {
namespace {
using Clock = std::chrono::steady_clock;
namespace gi = instrumentation;
constexpr std::uint32_t kMaximumBins = 65536;
constexpr std::uint64_t kModelBudget = 1ULL << 30;

double milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
[[noreturn]] void invalid(const std::string& what) { throw std::invalid_argument(what); }
void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
std::size_t product(std::size_t a, std::size_t b, const char* name) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b) invalid(std::string(name) + " size overflow");
  return a * b;
}
std::size_t sum(std::size_t a, std::size_t b, const char* name) {
  if (a > std::numeric_limits<std::size_t>::max() - b) invalid(std::string(name) + " size overflow");
  return a + b;
}
bool objective_valid(Objective value) {
  return value == Objective::squared_error || value == Objective::binary_logistic || value == Objective::multiclass_softmax;
}
bool type_valid(FeatureType value) { return value == FeatureType::numeric || value == FeatureType::categorical; }
void finite_nonnegative(double value, const char* name) {
  if (!std::isfinite(value) || value < 0) invalid(std::string(name) + " must be finite and nonnegative");
}

struct Stream {
  cudaStream_t value{};
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "create training stream"); }
  ~Stream() { if (value) { cudaStreamSynchronize(value); cudaStreamDestroy(value); } }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  void wait() const { check(cudaStreamSynchronize(value), "wait for training stream"); }
};
struct DrainStream {
  cudaStream_t stream{};
  ~DrainStream() { cudaStreamSynchronize(stream); }
};
template<class T> struct Device {
  T* data{};
  std::size_t count{};
  explicit Device(std::size_t n = 0) : count(n) {
    if (n) check(cudaMalloc(reinterpret_cast<void**>(&data), product(n, sizeof(T), "device allocation")), "allocate persistent device buffer");
  }
  ~Device() { if (data) cudaFree(data); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  std::size_t bytes() const { return product(count, sizeof(T), "device buffer"); }
  void upload(const T* source, std::size_t n, cudaStream_t stream) const {
    if (n > count) throw std::logic_error("device upload exceeds persistent buffer");
    if (n) check(cudaMemcpyAsync(data, source, product(n, sizeof(T), "upload"), cudaMemcpyHostToDevice, stream), "upload device buffer");
  }
};
template<class T> struct Pinned {
  T* data{};
  std::size_t count{};
  explicit Pinned(std::size_t n) : count(n) {
    if (n) check(cudaMallocHost(reinterpret_cast<void**>(&data), product(n, sizeof(T), "pinned allocation")), "allocate persistent pinned buffer");
  }
  ~Pinned() { if (data) cudaFreeHost(data); }
  Pinned(const Pinned&) = delete;
  Pinned& operator=(const Pinned&) = delete;
};
struct Timer {
  cudaEvent_t start{}, end{};
  explicit Timer(bool enabled) {
    if (enabled) {
      check(cudaEventCreate(&start), "create tuning start event");
      const auto status = cudaEventCreate(&end);
      if (status != cudaSuccess) { cudaEventDestroy(start); start = nullptr; check(status, "create tuning end event"); }
    }
  }
  ~Timer() { if (start) cudaEventDestroy(start); if (end) cudaEventDestroy(end); }
  template<class F> double measure(cudaStream_t stream, F&& operation) {
    check(cudaEventRecord(start, stream), "record tuning start");
    operation();
    check(cudaEventRecord(end, stream), "record tuning end");
    check(cudaEventSynchronize(end), "wait for tuning sample");
    float value{};
    check(cudaEventElapsedTime(&value, start, end), "read tuning sample");
    if (!std::isfinite(value) || value < 0) throw std::runtime_error("invalid histogram tuning interval");
    return value;
  }
};

template<class Record, class F>
void stage(Record& recorder, gi::Stage name, const gi::Context& context, gi::Timing timing,
           cudaStream_t stream, F&& operation) {
  const auto ticket = recorder.begin(name, context, timing, stream);
  try { operation(); }
  catch (...) { try { recorder.end(ticket); } catch (...) {} throw; }
  recorder.end(ticket);
}
template<class Record> struct HostScope {
  Record& recorder;
  gi::Ticket ticket;
  bool open{true};
  HostScope(Record& record, gi::Stage name, const gi::Context& context)
      : recorder(record), ticket(record.begin(name, context, gi::Timing::host, nullptr)) {}
  void finish() { recorder.end(ticket); open = false; }
  ~HostScope() { if (open) { try { recorder.end(ticket); } catch (...) {} } }
};
template<class Record> void collect(Record& recorder, TrainingResult& result) {
  if constexpr (!std::is_same_v<Record, gi::NullRecorder>) {
    auto samples = recorder.collect(false);
    if (!samples) throw std::logic_error("training stages still pending after stream completion");
    result.samples.insert(result.samples.end(), samples->begin(), samples->end());
    recorder.reset(); // IDs and the recorder's timestamp epoch are retained.
  }
}

void validate_input(const Dataset& data, bool training) {
  if (!data.columns || (training && !data.rows)) invalid("dataset requires features and training requires at least one row");
  if (data.columns > std::uint32_t(std::numeric_limits<std::int32_t>::max())) invalid("too many features");
  if (data.values.size() != product(data.rows, data.columns, "dataset")) invalid("feature matrix shape mismatch");
  if (!data.feature_types.empty() && data.feature_types.size() != data.columns) invalid("feature type shape mismatch");
  for (auto type : data.feature_types) if (!type_valid(type)) invalid("unknown feature type");
  for (float value : data.values) if (std::isinf(value)) invalid("infinite feature value; use NaN for missing data");
  if (!training) return;
  if (!data.outputs) invalid("dataset outputs must be positive");
  if (!data.weights.empty() && data.weights.size() != data.rows) invalid("weight shape mismatch");
  long double weights{};
  for (std::uint32_t row = 0; row < data.rows; ++row) {
    const double weight = data.weights.empty() ? 1.0 : data.weights[row];
    finite_nonnegative(weight, "weight"); weights += weight;
  }
  if (!(weights > 0) || !std::isfinite(weights)) invalid("training requires positive finite total weight");
}
void validate_config(const Dataset& data, const TrainConfig& config) {
  if (!objective_valid(config.objective)) invalid("unsupported objective");
  if (config.histogram != HistogramPolicy::global && config.histogram != HistogramPolicy::shared && config.histogram != HistogramPolicy::autotune)
    invalid("unknown histogram policy");
  if (config.max_depth > 30) invalid("max_depth must be at most 30 for signed node indices");
  if (config.max_bins < 2 || config.max_bins > kMaximumBins) invalid("max_bins must be in [2,65536], including missing bin zero");
  if (!config.min_leaf_rows || !config.max_histogram_bytes || !config.max_device_bytes || !config.output_tile_size)
    invalid("min_leaf_rows, output_tile_size and memory budgets must be positive");
  finite_nonnegative(config.learning_rate, "learning_rate");
  if (!config.learning_rate) invalid("learning_rate must be positive");
  finite_nonnegative(config.l2, "l2"); finite_nonnegative(config.min_child_hessian, "min_child_hessian");
  finite_nonnegative(config.min_gain, "min_gain"); finite_nonnegative(config.max_leaf_value, "max_leaf_value");
  const auto target_outputs = config.objective == Objective::multiclass_softmax ? 1U : data.outputs;
  if (config.objective == Objective::multiclass_softmax && data.outputs != 1) invalid("multiclass targets must have one class index per row");
  if (config.objective == Objective::multiclass_softmax && config.classes < 2) invalid("multiclass requires at least two classes");
  if (data.targets.size() != product(data.rows, target_outputs, "targets")) invalid("target matrix shape mismatch");
  for (float target : data.targets) {
    if (!std::isfinite(target)) invalid("targets must be finite");
    if (config.objective == Objective::binary_logistic && target != 0 && target != 1) invalid("binary targets must be zero or one");
    if (config.objective == Objective::multiclass_softmax && (target < 0 || double(target) >= config.classes || std::floor(target) != target))
      invalid("multiclass targets must be integer class indices");
  }
}

void validate_feature(const Feature& feature) {
  if (!type_valid(feature.type)) invalid("model has unknown feature type");
  if (feature.type == FeatureType::numeric && !feature.categories.empty()) invalid("numeric feature contains categories");
  if (feature.type == FeatureType::categorical && !feature.cuts.empty()) invalid("categorical feature contains numeric cuts");
  const auto& values = feature.type == FeatureType::numeric ? feature.cuts : feature.categories;
  if (values.size() > kMaximumBins - (feature.type == FeatureType::numeric ? 2U : 1U)) invalid("feature exceeds packed bin capacity");
  for (std::size_t i = 0; i < values.size(); ++i)
    if (!std::isfinite(values[i]) || (i && !(values[i - 1] < values[i]))) invalid("model feature values must be finite, strictly increasing");
}
void validate_model(const Model& model) {
  if (!objective_valid(model.objective) || !model.outputs || model.outputs > std::uint32_t(std::numeric_limits<std::int32_t>::max())) invalid("invalid model objective/output count");
  if (model.objective == Objective::multiclass_softmax && model.outputs < 2) invalid("multiclass model must have at least two outputs");
  if (model.base_scores.size() != model.outputs) invalid("model base score shape mismatch");
  for (double score : model.base_scores) if (!std::isfinite(score)) invalid("nonfinite model base score");
  if (model.features.empty() || model.features.size() > std::size_t(std::numeric_limits<std::int32_t>::max())) invalid("invalid model feature count");
  std::uint64_t bins{};
  for (const auto& feature : model.features) { validate_feature(feature); bins += feature.bins(); }
  if (bins > std::numeric_limits<std::uint32_t>::max()) invalid("model histogram offsets overflow");
  std::vector<long double> margin_bounds(model.outputs);
  for (std::uint32_t output = 0; output < model.outputs; ++output) margin_bounds[output] = std::abs(model.base_scores[output]);
  for (const auto& tree : model.trees) {
    if (tree.output >= model.outputs || tree.nodes.empty() || tree.nodes.size() > std::size_t(std::numeric_limits<std::int32_t>::max())) invalid("invalid tree output/node count");
    std::vector<unsigned char> seen(tree.nodes.size(), 0);
    std::vector<std::uint32_t> pending{0};
    long double maximum_leaf{};
    while (!pending.empty()) {
      const auto index = pending.back(); pending.pop_back();
      if (seen[index]++) invalid("tree contains a cycle or shared child");
      const auto& node = tree.nodes[index];
      if (!std::isfinite(node.value) || node.missing_left > 1) invalid("invalid tree value/missing direction");
      if (node.feature == -1) {
        if (node.left != -1 || node.right != -1 || node.threshold != 0) invalid("terminal node contains a split");
        maximum_leaf = std::max(maximum_leaf, static_cast<long double>(std::abs(node.value)));
        continue;
      }
      if (node.feature < 0 || std::size_t(node.feature) >= model.features.size() || node.left < 0 || node.right < 0 ||
          std::size_t(node.left) >= tree.nodes.size() || std::size_t(node.right) >= tree.nodes.size()) invalid("invalid tree feature/child index");
      const auto& feature = model.features[std::size_t(node.feature)];
      // Threshold zero plus missing_left is a valid missing-only left split
      // for both numerical and categorical features; present bins start at 1.
      if (node.threshold >= feature.bins()) invalid("split threshold outside feature bin domain");
      pending.push_back(std::uint32_t(node.right)); pending.push_back(std::uint32_t(node.left));
    }
    if (std::find(seen.begin(), seen.end(), 0) != seen.end()) invalid("tree contains unreachable nodes");
    margin_bounds[tree.output] += maximum_leaf;
    if (margin_bounds[tree.output] > std::numeric_limits<double>::max()) invalid("model can overflow prediction margins");
  }
}
void validate_prediction(const Model& model, const Dataset& data) {
  validate_model(model); validate_input(data, false);
  if (data.columns != model.features.size()) invalid("prediction feature count mismatch");
  if (!data.feature_types.empty()) for (std::size_t f = 0; f < model.features.size(); ++f)
    if (data.feature_types[f] != model.features[f].type) invalid("prediction feature type mismatch");
}

struct Encoded {
  std::vector<std::uint16_t> bins;
  std::vector<std::uint32_t> offsets;
  std::vector<FeatureType> types;
  std::uint32_t total_bins{}, max_feature_bins{};
};
Encoded encode(const Dataset& data, const std::vector<Feature>& features) {
  Encoded result;
  result.bins.resize(product(data.rows, data.columns, "packed feature matrix"));
  result.offsets.reserve(std::size_t(data.columns) + 1); result.offsets.push_back(0);
  result.types.reserve(data.columns);
  for (std::uint32_t f = 0; f < data.columns; ++f) {
    const auto count = features[f].bins();
    if (count > std::numeric_limits<std::uint32_t>::max() - result.total_bins) invalid("feature histogram offsets overflow");
    result.total_bins += count; result.max_feature_bins = std::max(result.max_feature_bins, count);
    result.offsets.push_back(result.total_bins); result.types.push_back(features[f].type);
    for (std::uint32_t row = 0; row < data.rows; ++row)
      result.bins[std::size_t(f) * data.rows + row] = features[f].encode(data.values[std::size_t(row) * data.columns + f]);
  }
  return result;
}
std::vector<Feature> fit_features(const Dataset& data, std::uint32_t max_bins) {
  std::vector<Feature> result(data.columns);
  std::vector<float> values; values.reserve(data.rows);
  for (std::uint32_t f = 0; f < data.columns; ++f) {
    auto& feature = result[f];
    feature.type = data.feature_types.empty() ? FeatureType::numeric : data.feature_types[f];
    values.clear();
    for (std::uint32_t row = 0; row < data.rows; ++row) {
      const float value = data.values[std::size_t(row) * data.columns + f];
      if (!std::isnan(value)) values.push_back(value);
    }
    std::sort(values.begin(), values.end()); values.erase(std::unique(values.begin(), values.end()), values.end());
    if (feature.type == FeatureType::categorical) {
      if (values.size() >= max_bins) invalid("categorical cardinality exceeds max_bins after reserving missing bin zero");
      feature.categories = values;
    } else if (values.size() > 1) {
      const std::size_t intervals = std::min<std::size_t>(values.size(), max_bins - 1);
      feature.cuts.reserve(intervals - 1);
      for (std::size_t boundary = 1; boundary < intervals; ++boundary) {
        // Quantiles of distinct observed values, with no dropped categories or
        // midpoint overflow. The final observed value is never a cut.
        const std::size_t index = product(boundary, values.size(), "quantile") / intervals - 1;
        feature.cuts.push_back(values[index]);
      }
    }
  }
  return result;
}
std::vector<double> base_scores(const Dataset& data, const TrainConfig& config, std::uint32_t outputs, long double weight_sum) {
  std::vector<long double> totals(outputs);
  for (std::uint32_t row = 0; row < data.rows; ++row) {
    const long double weight = data.weights.empty() ? 1 : data.weights[row];
    if (config.objective != Objective::multiclass_softmax) {
      for (std::uint32_t output = 0; output < outputs; ++output) totals[output] += weight * data.targets[std::size_t(row) * outputs + output];
    } else totals[std::size_t(data.targets[row])] += weight;
  }
  std::vector<double> result(outputs);
  for (std::uint32_t output = 0; output < outputs; ++output) {
    const double mean = double(totals[output] / weight_sum);
    if (config.objective == Objective::squared_error) result[output] = mean;
    else if (config.objective == Objective::binary_logistic) {
      const double p = std::clamp(mean, 1e-12, 1 - 1e-12); result[output] = std::log(p) - std::log1p(-p);
    } else result[output] = std::log(std::max(mean, 1e-12));
  }
  return result;
}
void transform_host(Objective objective, std::vector<double>& predictions, std::uint32_t outputs) {
  if (objective == Objective::binary_logistic) for (double& margin : predictions) {
    const double e = std::exp(-std::abs(margin)); margin = margin >= 0 ? 1 / (1 + e) : e / (1 + e);
  }
  else if (objective == Objective::multiclass_softmax) for (std::size_t row = 0; row < predictions.size(); row += outputs) {
    const double maximum = *std::max_element(predictions.begin() + std::ptrdiff_t(row), predictions.begin() + std::ptrdiff_t(row + outputs));
    double denominator{};
    for (std::uint32_t output = 0; output < outputs; ++output) denominator += predictions[row + output] = std::exp(predictions[row + output] - maximum);
    for (std::uint32_t output = 0; output < outputs; ++output) predictions[row + output] /= denominator;
  }
}

struct PackedDevice {
  Device<std::uint16_t> bins;
  Device<std::uint32_t> offsets;
  Device<FeatureType> types;
  gpu::DataView view;
  PackedDevice(const Dataset& data, const Encoded& packed)
      : bins(packed.bins.size()), offsets(packed.offsets.size()), types(packed.types.size()),
        view{bins.data, offsets.data, types.data, data.rows, data.columns, packed.total_bins, packed.max_feature_bins} {}
  void upload(const Encoded& packed, cudaStream_t stream) {
    bins.upload(packed.bins.data(), packed.bins.size(), stream);
    offsets.upload(packed.offsets.data(), packed.offsets.size(), stream);
    types.upload(packed.types.data(), packed.types.size(), stream);
  }
};

std::size_t recorder_capacity(const TrainConfig& config, std::uint32_t outputs) {
  const auto capacity = sum(16, product(product(outputs, std::size_t(config.max_depth) + 1, "recorder"), 10, "recorder"), "recorder");
  if (capacity > (1U << 20)) invalid("per-round instrumentation exceeds one million reserved scopes; disable recording or reduce outputs/depth");
  return capacity;
}

template<class Record> TrainingResult train_impl(const Dataset& data, const TrainConfig& config) {
  const auto total_start = Clock::now();
  TrainingResult result;
  auto& model = result.model;
  model.objective = config.objective;
  model.outputs = config.objective == Objective::multiclass_softmax ? config.classes : data.outputs;
  if (model.outputs > std::uint32_t(std::numeric_limits<std::int32_t>::max())) invalid("output count exceeds context index range");
  if (config.rounds > std::uint32_t(std::numeric_limits<std::int32_t>::max())) invalid("round count exceeds context index range");
  long double weight_sum{};
  for (std::uint32_t row = 0; row < data.rows; ++row) weight_sum += data.weights.empty() ? 1.0L : data.weights[row];
  model.base_scores = base_scores(data, config, model.outputs, weight_sum);
  model.trees.reserve(product(config.rounds, model.outputs, "tree count"));
  result.training_loss.reserve(std::size_t(config.rounds) + 1);
  const std::size_t capacity = [&] { if constexpr (std::is_same_v<Record, gi::NullRecorder>) return std::size_t(0); else return recorder_capacity(config, model.outputs); }();
  Record recorder(capacity, config.nvtx);
  gi::Context context;
  context.rows = data.rows; context.features = data.columns; context.stream_id = 1;
  Encoded packed;
  auto phase = Clock::now();
  stage(recorder, gi::Stage::quantize, context, gi::Timing::host, nullptr, [&] {
    model.features = fit_features(data, config.max_bins); packed = encode(data, model.features);
  });
  result.quantize_ms = milliseconds(phase);
  context.bins = packed.total_bins;
  const std::uint64_t leaves = std::max<std::uint64_t>(1, data.rows / config.min_leaf_rows);
  // Final-depth leaf values are already supplied by their parents' winning
  // splits; only levels that can still split need histograms and row routing.
  const auto histogram_depth = config.max_depth ? config.max_depth - 1 : 0;
  const std::uint64_t level_bound = std::min<std::uint64_t>(1ULL << histogram_depth, leaves);
  const std::uint64_t node_bound = std::min<std::uint64_t>((1ULL << (config.max_depth + 1)) - 1, leaves * 2 - 1);
  if (node_bound > std::uint64_t(std::numeric_limits<std::int32_t>::max())) invalid("tree capacity exceeds signed node indices");
  const auto histogram_per_node = product(packed.total_bins, sizeof(gpu::Stats), "histogram");
  const auto active_capacity = std::min<std::uint64_t>(level_bound, config.max_histogram_bytes / histogram_per_node);
  if (!active_capacity) invalid("max_histogram_bytes cannot hold even the root histogram");
  const auto nodes_capacity = std::uint32_t(active_capacity);
  const auto tree_capacity = std::uint32_t(node_bound);
  const auto predictions_count = product(data.rows, model.outputs, "predictions");
  const auto derivative_outputs = model.objective == Objective::multiclass_softmax ? model.outputs : std::min(model.outputs, config.output_tile_size);
  const auto derivative_count = product(data.rows, derivative_outputs, "tiled derivatives");
  const auto histogram_count = product(nodes_capacity, packed.total_bins, "histogram stats");
  const auto candidate_count = product(nodes_capacity, data.columns, "split candidates");
  const auto loss_blocks = std::min<std::uint32_t>(4096, 1 + (data.rows - 1) / 256);
  std::size_t device_bytes{};
  auto account = [&](std::size_t count, std::size_t element) { device_bytes = sum(device_bytes, product(count, element, "device workspace"), "device workspace"); };
  account(packed.bins.size(), sizeof(std::uint16_t)); account(packed.offsets.size(), sizeof(std::uint32_t)); account(packed.types.size(), sizeof(FeatureType));
  account(predictions_count, sizeof(double)); account(derivative_count, 2 * sizeof(double));
  account(data.targets.size(), sizeof(float)); account(data.weights.size(), sizeof(float));
  account(model.outputs, sizeof(double)); account(histogram_count, sizeof(gpu::Stats)); account(candidate_count, sizeof(gpu::Split));
  account(nodes_capacity, sizeof(gpu::Split) + 2 * sizeof(std::int32_t)); account(data.rows, sizeof(std::int32_t));
  account(tree_capacity, sizeof(Node)); account(loss_blocks, sizeof(double));
  result.device_bytes = device_bytes;
  result.gradient_bytes = product(derivative_count, 2 * sizeof(double), "derivative memory");
  result.histogram_bytes = product(histogram_count, sizeof(gpu::Stats), "histogram memory");
  if (device_bytes > config.max_device_bytes) invalid("persistent trainer workspace exceeds max_device_bytes; reduce output_tile_size, rows, outputs or histogram capacity");
  context.scratch_bytes = device_bytes;
  HostScope preparation(recorder, gi::Stage::initialize, context);
  std::size_t free_bytes{}, total_bytes{};
  check(cudaMemGetInfo(&free_bytes, &total_bytes), "query trainer memory budget");
  if (device_bytes > free_bytes) invalid("persistent trainer workspace exceeds currently available GPU memory");
  Stream stream;
  PackedDevice device_data(data, packed);
  Device<double> predictions(predictions_count), gradient(derivative_count), hessian(derivative_count), base(model.outputs), partial_loss(loss_blocks);
  Device<float> targets(data.targets.size()), weights(data.weights.size());
  Device<std::int32_t> assignments(data.rows), left_map(nodes_capacity), right_map(nodes_capacity);
  Device<gpu::Stats> histograms(histogram_count);
  Device<gpu::Split> candidates(candidate_count), winners(nodes_capacity);
  Device<Node> tree_nodes(tree_capacity);
  Pinned<gpu::Split> host_winners(nodes_capacity);
  Pinned<std::int32_t> host_left(nodes_capacity), host_right(nodes_capacity);
  Pinned<double> host_loss(loss_blocks);
  Timer tuner(config.histogram == HistogramPolicy::autotune);
  DrainStream drain{stream.value};
  std::vector<std::int32_t> frontier, next_frontier;
  frontier.reserve(nodes_capacity); next_frontier.reserve(nodes_capacity);
  std::vector<Node> node_workspace;
  node_workspace.reserve(tree_capacity);
  preparation.finish();
  phase = Clock::now();
  stage(recorder, gi::Stage::upload, context, gi::Timing::gpu, stream.value, [&] {
    device_data.upload(packed, stream.value); targets.upload(data.targets.data(), data.targets.size(), stream.value);
    weights.upload(data.weights.data(), data.weights.size(), stream.value); base.upload(model.base_scores.data(), model.outputs, stream.value);
  });
  stream.wait(); result.upload_ms = milliseconds(phase);
  stage(recorder, gi::Stage::initialize, context, gi::Timing::gpu, stream.value, [&] {
    check(gpu::initialize_predictions(predictions.data, base.data, data.rows, model.outputs, stream.value), "initialize predictions");
  });
  auto evaluate = [&] {
    stage(recorder, gi::Stage::evaluate, context, gi::Timing::gpu, stream.value, [&] {
      check(gpu::loss(model.objective, predictions.data, targets.data, weights.data, data.rows, model.outputs, partial_loss.data, loss_blocks, stream.value), "evaluate training loss");
    });
    stage(recorder, gi::Stage::download, context, gi::Timing::gpu, stream.value, [&] {
      check(cudaMemcpyAsync(host_loss.data, partial_loss.data, partial_loss.bytes(), cudaMemcpyDeviceToHost, stream.value), "download objective partial sums");
    });
    stream.wait();
    stage(recorder, gi::Stage::evaluate, context, gi::Timing::host, nullptr, [&] {
      long double loss{};
      for (std::uint32_t block = 0; block < loss_blocks; ++block) {
        if (!std::isfinite(host_loss.data[block]) || host_loss.data[block] < 0) throw std::runtime_error("nonfinite or negative objective partial sum");
        loss += host_loss.data[block];
      }
      const long double denominator = weight_sum * (model.objective == Objective::multiclass_softmax ? 1U : model.outputs);
      const double normalized = double(loss / denominator);
      if (!std::isfinite(normalized)) throw std::runtime_error("nonfinite normalized training objective");
      result.training_loss.push_back(normalized);
    });
  };
  evaluate(); collect(recorder, result);
  // Cache is local to this dataset/config/layout/device invocation. Include
  // output and depth; timing at one output/frontier does not tune another.
  using TuningKey = std::tuple<std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>;
  std::map<TuningKey, HistogramPolicy> policies;
  const gpu::SplitConfig split_config{config.min_leaf_rows, config.l2, config.min_child_hessian, config.min_gain, config.max_leaf_value};
  const auto training_start = Clock::now();
  for (std::uint32_t round = 0; round < config.rounds; ++round) {
    context.round = std::int32_t(round); context.depth = -1; context.output = -1; context.active_nodes = 0;
    // Softmax outputs are coupled: retain their full pre-round derivatives.
    // Independent regression/binary outputs instead reuse bounded NxC tiles.
    if (model.objective == Objective::multiclass_softmax) {
      auto work = context; work.logical_write_bytes = result.gradient_bytes;
      stage(recorder, gi::Stage::gradients, work, gi::Timing::gpu, stream.value, [&] {
        check(gpu::gradients(model.objective, predictions.data, targets.data, weights.data, gradient.data, hessian.data,
                             data.rows, model.outputs, stream.value), "compute pre-round softmax derivatives");
      });
    }
    for (std::uint32_t tile_begin = 0; tile_begin < model.outputs;) {
      const auto tile_count = std::min(derivative_outputs, model.outputs - tile_begin);
      if (model.objective != Objective::multiclass_softmax) {
        auto work = context; work.output = std::int32_t(tile_begin);
        work.logical_read_bytes = product(product(data.rows, tile_count, "derivative reads"), sizeof(double) + sizeof(float), "derivative reads") + weights.bytes();
        work.logical_write_bytes = product(product(data.rows, tile_count, "derivative writes"), 2 * sizeof(double), "derivative writes");
        stage(recorder, gi::Stage::gradients, work, gi::Timing::gpu, stream.value, [&] {
          check(gpu::gradients_tile(model.objective, predictions.data, targets.data, weights.data, gradient.data, hessian.data,
                                   data.rows, model.outputs, tile_begin, tile_count, stream.value), "compute tiled objective derivatives");
        });
      }
    for (std::uint32_t local_output = 0; local_output < tile_count; ++local_output) {
      const auto output = tile_begin + local_output;
      context.output = std::int32_t(output); context.depth = -1; context.active_nodes = 1;
      // Each completed tree retains this vector. Reserving the worst-case
      // capacity here would multiply unused host memory by rounds * outputs.
      Tree tree; tree.output = output; tree.nodes.swap(node_workspace); tree.nodes.clear(); tree.nodes.emplace_back();
      frontier.clear(); frontier.push_back(0);
      stage(recorder, gi::Stage::initialize, context, gi::Timing::gpu, stream.value, [&] {
        check(cudaMemsetAsync(assignments.data, 0, assignments.bytes(), stream.value), "initialize row assignments");
      });
      for (std::uint32_t depth = 0; !frontier.empty(); ++depth) {
        const auto active = std::uint32_t(frontier.size());
        context.depth = std::int32_t(depth); context.active_nodes = active;
        auto run_histogram = [&](HistogramPolicy policy) {
          check(gpu::histogram(device_data.view, assignments.data, gradient.data, hessian.data, tile_count, local_output,
                               active, histograms.data, policy, stream.value), "build gradient histogram");
        };
        HistogramPolicy policy = config.histogram;
        if (policy == HistogramPolicy::shared && !gpu::shared_supported(device_data.view, active))
          invalid("forced shared histogram exceeds this frontier's supported shared-memory capacity");
        if (policy == HistogramPolicy::autotune) {
          const TuningKey key{active, output, depth, tile_count};
          const auto cached = policies.find(key);
          if (cached != policies.end()) policy = cached->second;
          else if (!gpu::shared_supported(device_data.view, active)) policy = HistogramPolicy::global;
          else {
            auto tuning_context = context; tuning_context.operations = 14;
            stage(recorder, gi::Stage::histogram, tuning_context, gi::Timing::host, nullptr, [&] {
              for (unsigned warmup = 0; warmup < 2; ++warmup) { run_histogram(HistogramPolicy::global); run_histogram(HistogramPolicy::shared); }
              stream.wait();
              std::array<double, 5> global{}, shared{};
              for (unsigned repetition = 0; repetition < global.size(); ++repetition) {
                if (repetition % 2) {
                  shared[repetition] = tuner.measure(stream.value, [&] { run_histogram(HistogramPolicy::shared); });
                  global[repetition] = tuner.measure(stream.value, [&] { run_histogram(HistogramPolicy::global); });
                } else {
                  global[repetition] = tuner.measure(stream.value, [&] { run_histogram(HistogramPolicy::global); });
                  shared[repetition] = tuner.measure(stream.value, [&] { run_histogram(HistogramPolicy::shared); });
                }
              }
              TuningRecord measured;
              measured.active_nodes = active; measured.output = output;
              measured.round = round; measured.depth = depth; measured.output_tile_size = tile_count;
              measured.global_samples_ms = global; measured.shared_samples_ms = shared;
              std::sort(global.begin(), global.end()); std::sort(shared.begin(), shared.end());
              policy = shared[2] < global[2] ? HistogramPolicy::shared : HistogramPolicy::global;
              measured.global_ms = global[2]; measured.shared_ms = shared[2]; measured.selected = policy;
              result.tuning.push_back(measured);
            });
          }
          policies.emplace(key, policy);
        }
        stage(recorder, gi::Stage::histogram, context, gi::Timing::gpu, stream.value, [&] { run_histogram(policy); });
        stage(recorder, gi::Stage::split_search, context, gi::Timing::gpu, stream.value, [&] {
          check(gpu::find_splits(device_data.view, histograms.data, active, split_config, depth == config.max_depth,
                                 candidates.data, winners.data, stream.value), "find best splits");
        });
        stage(recorder, gi::Stage::download, context, gi::Timing::gpu, stream.value, [&] {
          check(cudaMemcpyAsync(host_winners.data, winners.data, product(active, sizeof(gpu::Split), "split download"),
                                 cudaMemcpyDeviceToHost, stream.value), "download split decisions");
        });
        stream.wait();
        stage(recorder, gi::Stage::route, context, gi::Timing::host, nullptr, [&] {
          next_frontier.clear();
          const auto scaled_leaf = [&](double value) {
            const double scaled = config.learning_rate * value;
            if (!std::isfinite(scaled)) throw std::runtime_error("learning rate overflows a leaf value");
            return scaled;
          };
          for (std::uint32_t index = 0; index < active; ++index) {
            const auto& split = host_winners.data[index];
            if (!std::isfinite(split.value) || !std::isfinite(split.gain) || !std::isfinite(split.left_value) || !std::isfinite(split.right_value))
              throw std::runtime_error("split search returned a nonfinite decision");
            const auto node_index = std::size_t(frontier[index]);
            host_left.data[index] = host_right.data[index] = -1;
            if (split.feature == -1) { tree.nodes[node_index].value = scaled_leaf(split.value); continue; }
            if (depth >= config.max_depth || split.feature < 0 || std::size_t(split.feature) >= model.features.size() || split.missing_left > 1 ||
                split.threshold >= model.features[std::size_t(split.feature)].bins()) throw std::runtime_error("split search returned an invalid split");
            const bool expand_children = depth + 1 < config.max_depth;
            if (expand_children && next_frontier.size() + 2 > nodes_capacity)
              invalid("required frontier exceeds max_histogram_bytes; increase the explicit histogram budget");
            if (tree.nodes.size() + 2 > tree_capacity) throw std::logic_error("tree exceeds row/depth node capacity");
            const auto left = std::int32_t(tree.nodes.size()), right = left + 1;
            tree.nodes[node_index] = Node{split.feature, left, right, split.threshold, split.missing_left, 0};
            Node left_node, right_node;
            left_node.value = scaled_leaf(split.left_value); right_node.value = scaled_leaf(split.right_value);
            tree.nodes.push_back(left_node); tree.nodes.push_back(right_node);
            if (expand_children) {
              host_left.data[index] = std::int32_t(next_frontier.size()); next_frontier.push_back(left);
              host_right.data[index] = std::int32_t(next_frontier.size()); next_frontier.push_back(right);
            }
          }
        });
        if (!next_frontier.empty()) {
          stage(recorder, gi::Stage::upload, context, gi::Timing::gpu, stream.value, [&] {
            left_map.upload(host_left.data, active, stream.value); right_map.upload(host_right.data, active, stream.value);
          });
          stage(recorder, gi::Stage::route, context, gi::Timing::gpu, stream.value, [&] {
            check(gpu::route(device_data.view, assignments.data, winners.data, left_map.data, right_map.data, stream.value), "route active rows");
          });
        }
        frontier.swap(next_frontier);
      }
      context.depth = -1; context.active_nodes = 0;
      // One reusable worst-case construction workspace; retained trees only
      // own their actual nodes. Compact before upload so the asynchronous
      // copy's source remains stable while the workspace builds another tree.
      Tree completed; completed.output = output;
      stage(recorder, gi::Stage::route, context, gi::Timing::host, nullptr, [&] {
        completed.nodes.assign(tree.nodes.begin(), tree.nodes.end());
        node_workspace.swap(tree.nodes);
        model.trees.push_back(std::move(completed));
      });
      const auto& retained = model.trees.back();
      stage(recorder, gi::Stage::upload, context, gi::Timing::gpu, stream.value, [&] { tree_nodes.upload(retained.nodes.data(), retained.nodes.size(), stream.value); });
      stage(recorder, gi::Stage::prediction, context, gi::Timing::gpu, stream.value, [&] {
        check(gpu::add_tree(device_data.view, tree_nodes.data, std::uint32_t(retained.nodes.size()), output, model.outputs, predictions.data, stream.value), "apply completed tree");
      });
    }
      tile_begin += tile_count;
    }
    context.output = -1; context.depth = -1; context.active_nodes = 0;
    evaluate(); collect(recorder, result);
  }
  stream.wait();
  result.training_ms = milliseconds(training_start);
  validate_model(model);
  result.total_ms = milliseconds(total_start);
  return result;
}

// Portable little-endian model format with a cumulative 1 GiB allocation/read
// bound. No object padding, pointers, or platform size_t are serialized.
class Writer {
  std::ostream& out_;
  std::uint64_t bytes_{};
 public:
  explicit Writer(std::ostream& out) : out_(out) {}
  void raw(const char* bytes, std::size_t count) {
    if (count > kModelBudget - bytes_) invalid("serialized model exceeds 1 GiB safety bound");
    out_.write(bytes, std::streamsize(count)); bytes_ += count;
    if (!out_) throw std::runtime_error("model write failed");
  }
  template<class U> void unsigned_value(U value) {
    static_assert(std::is_unsigned_v<U>);
    std::array<char, sizeof(U)> bytes{};
    for (std::size_t i = 0; i < sizeof(U); ++i) bytes[i] = char((value >> (8 * i)) & 255);
    raw(bytes.data(), bytes.size());
  }
  void f32(float value) { unsigned_value(std::bit_cast<std::uint32_t>(value)); }
  void f64(double value) { unsigned_value(std::bit_cast<std::uint64_t>(value)); }
  void i32(std::int32_t value) { unsigned_value(std::bit_cast<std::uint32_t>(value)); }
};
class Reader {
  std::istream& in_;
  std::uint64_t bytes_{}, allocation_{};
 public:
  explicit Reader(std::istream& in) : in_(in) {}
  void raw(char* bytes, std::size_t count) {
    if (count > kModelBudget - bytes_) invalid("serialized model exceeds 1 GiB safety bound");
    in_.read(bytes, std::streamsize(count)); bytes_ += count;
    if (!in_) invalid("truncated or unreadable model");
  }
  template<class U> U unsigned_value() {
    static_assert(std::is_unsigned_v<U>);
    std::array<char, sizeof(U)> bytes{}; raw(bytes.data(), bytes.size()); U value{};
    for (std::size_t i = 0; i < sizeof(U); ++i) value |= U(static_cast<unsigned char>(bytes[i])) << (8 * i);
    return value;
  }
  void allocation(std::size_t count, std::size_t element) {
    const auto n = product(count, element, "model allocation");
    if (n > kModelBudget - allocation_) invalid("model allocation exceeds 1 GiB safety bound");
    allocation_ += n;
  }
  float f32() { return std::bit_cast<float>(unsigned_value<std::uint32_t>()); }
  double f64() { return std::bit_cast<double>(unsigned_value<std::uint64_t>()); }
  std::int32_t i32() { return std::bit_cast<std::int32_t>(unsigned_value<std::uint32_t>()); }
  void end() { if (in_.peek() != std::char_traits<char>::eof()) invalid("trailing bytes after serialized model"); }
};
} // namespace

std::uint32_t Feature::bins() const {
  const auto extra = type == FeatureType::numeric ? 2U : 1U;
  const auto count = type == FeatureType::numeric ? cuts.size() : categories.size();
  if (!type_valid(type) || count > kMaximumBins - extra) invalid("invalid feature bin count");
  return std::uint32_t(count) + extra;
}
std::uint16_t Feature::encode(float value) const {
  if (std::isnan(value)) return 0;
  if (std::isinf(value)) invalid("infinite feature value");
  if (type == FeatureType::numeric) {
    if (cuts.size() > kMaximumBins - 2) invalid("numeric feature exceeds bin capacity");
    return std::uint16_t(1 + std::size_t(std::lower_bound(cuts.begin(), cuts.end(), value) - cuts.begin()));
  }
  if (type == FeatureType::categorical) {
    if (categories.size() > kMaximumBins - 1) invalid("categorical feature exceeds bin capacity");
    const auto it = std::lower_bound(categories.begin(), categories.end(), value);
    return it != categories.end() && *it == value ? std::uint16_t(1 + std::size_t(it - categories.begin())) : 0;
  }
  invalid("unknown feature type");
}
TrainingResult train(const Dataset& data, const TrainConfig& config) {
  const auto start = Clock::now();
  validate_input(data, true); validate_config(data, config);
  auto result = config.record_stages ? train_impl<gi::Recorder>(data, config) : train_impl<gi::NullRecorder>(data, config);
  result.total_ms = milliseconds(start); // Includes input checks and device cleanup.
  return result;
}
std::vector<double> Model::predict(const Dataset& data, bool raw) const {
  validate_prediction(*this, data);
  std::vector<double> result(product(data.rows, outputs, "prediction output"));
  for (std::uint32_t row = 0; row < data.rows; ++row) {
    for (std::uint32_t output = 0; output < outputs; ++output) result[std::size_t(row) * outputs + output] = base_scores[output];
    for (const auto& tree : trees) {
      std::size_t index{};
      while (tree.nodes[index].feature != -1) {
        const auto& node = tree.nodes[index];
        const auto feature = std::size_t(node.feature);
        const auto bin = features[feature].encode(data.values[std::size_t(row) * data.columns + feature]);
        const bool left = !bin ? bool(node.missing_left) : features[feature].type == FeatureType::categorical ? bin == node.threshold : bin <= node.threshold;
        index = std::size_t(left ? node.left : node.right);
      }
      result[std::size_t(row) * outputs + tree.output] += tree.nodes[index].value;
    }
  }
  for (double value : result) if (!std::isfinite(value)) throw std::runtime_error("prediction margin overflow");
  if (!raw) transform_host(objective, result, outputs);
  return result;
}
std::vector<double> Model::predict_gpu(const Dataset& data, bool raw) const {
  validate_prediction(*this, data);
  std::vector<double> result(product(data.rows, outputs, "prediction output"));
  if (!data.rows) return result;
  const auto packed = encode(data, features);
  std::size_t max_nodes{};
  for (const auto& tree : trees) max_nodes = std::max(max_nodes, tree.nodes.size());
  Stream stream;
  PackedDevice device_data(data, packed);
  Device<double> predictions(result.size()), base(outputs);
  Device<Node> nodes(max_nodes);
  DrainStream drain{stream.value};
  device_data.upload(packed, stream.value); base.upload(base_scores.data(), outputs, stream.value);
  check(gpu::initialize_predictions(predictions.data, base.data, data.rows, outputs, stream.value), "initialize inference predictions");
  for (const auto& tree : trees) {
    nodes.upload(tree.nodes.data(), tree.nodes.size(), stream.value);
    check(gpu::add_tree(device_data.view, nodes.data, std::uint32_t(tree.nodes.size()), tree.output, outputs, predictions.data, stream.value), "predict tree");
  }
  if (!raw) check(gpu::transform(objective, predictions.data, data.rows, outputs, stream.value), "transform prediction margins");
  check(cudaMemcpyAsync(result.data(), predictions.data, predictions.bytes(), cudaMemcpyDeviceToHost, stream.value), "download predictions");
  stream.wait();
  for (double value : result) if (!std::isfinite(value)) throw std::runtime_error("nonfinite GPU prediction");
  return result;
}
void Model::save(std::ostream& out) const {
  validate_model(*this);
  Writer writer(out); writer.raw("GHBMODEL", 8); writer.unsigned_value<std::uint32_t>(1);
  writer.unsigned_value(std::uint32_t(objective)); writer.unsigned_value(outputs);
  writer.unsigned_value(std::uint32_t(features.size())); writer.unsigned_value(std::uint64_t(trees.size()));
  for (double value : base_scores) writer.f64(value);
  for (const auto& feature : features) {
    writer.unsigned_value(std::uint32_t(feature.type));
    writer.unsigned_value(std::uint32_t(feature.cuts.size())); writer.unsigned_value(std::uint32_t(feature.categories.size()));
    for (float value : feature.cuts) writer.f32(value);
    for (float value : feature.categories) writer.f32(value);
  }
  for (const auto& tree : trees) {
    writer.unsigned_value(tree.output); writer.unsigned_value(std::uint32_t(tree.nodes.size()));
    for (const auto& node : tree.nodes) {
      writer.i32(node.feature); writer.i32(node.left); writer.i32(node.right); writer.unsigned_value(node.threshold);
      writer.unsigned_value(node.missing_left); writer.f64(node.value);
    }
  }
}
Model Model::load(std::istream& in) {
  Reader reader(in); std::array<char, 8> magic{}; reader.raw(magic.data(), magic.size());
  if (std::memcmp(magic.data(), "GHBMODEL", 8) || reader.unsigned_value<std::uint32_t>() != 1) invalid("unknown model format/version");
  Model model; model.objective = Objective(reader.unsigned_value<std::uint32_t>()); model.outputs = reader.unsigned_value<std::uint32_t>();
  const auto feature_count = reader.unsigned_value<std::uint32_t>();
  const auto tree_count = reader.unsigned_value<std::uint64_t>();
  if (!objective_valid(model.objective) || !model.outputs || !feature_count || model.outputs > std::uint32_t(std::numeric_limits<std::int32_t>::max()) ||
      feature_count > std::uint32_t(std::numeric_limits<std::int32_t>::max()) || tree_count > std::numeric_limits<std::size_t>::max()) invalid("invalid serialized model shape");
  reader.allocation(model.outputs, sizeof(double)); reader.allocation(feature_count, sizeof(Feature)); reader.allocation(std::size_t(tree_count), sizeof(Tree));
  model.base_scores.resize(model.outputs); model.features.resize(feature_count); model.trees.resize(std::size_t(tree_count));
  for (double& value : model.base_scores) value = reader.f64();
  for (auto& feature : model.features) {
    feature.type = FeatureType(reader.unsigned_value<std::uint32_t>());
    const auto cuts = reader.unsigned_value<std::uint32_t>(), categories = reader.unsigned_value<std::uint32_t>();
    if (!type_valid(feature.type) || cuts > kMaximumBins - 2 || categories > kMaximumBins - 1 ||
        (feature.type == FeatureType::numeric ? categories != 0 : cuts != 0)) invalid("invalid serialized feature metadata");
    reader.allocation(cuts, sizeof(float)); reader.allocation(categories, sizeof(float));
    feature.cuts.resize(cuts); feature.categories.resize(categories);
    for (float& value : feature.cuts) value = reader.f32();
    for (float& value : feature.categories) value = reader.f32();
    validate_feature(feature);
  }
  for (auto& tree : model.trees) {
    tree.output = reader.unsigned_value<std::uint32_t>(); const auto count = reader.unsigned_value<std::uint32_t>();
    if (!count || count > std::uint32_t(std::numeric_limits<std::int32_t>::max()) || tree.output >= model.outputs) invalid("invalid serialized tree shape");
    reader.allocation(count, sizeof(Node)); tree.nodes.resize(count);
    for (auto& node : tree.nodes) {
      node.feature = reader.i32(); node.left = reader.i32(); node.right = reader.i32();
      node.threshold = reader.unsigned_value<std::uint32_t>(); node.missing_left = reader.unsigned_value<std::uint32_t>(); node.value = reader.f64();
    }
  }
  reader.end(); validate_model(model); return model;
}
} // namespace ghb
