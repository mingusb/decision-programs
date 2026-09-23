#include "ghb/booster.hpp"
#include "ghb/kernels.cuh"
#include "ghb/resident.cuh"
#include "ghb/quantize.hpp"
#include "ghb/initialization.cuh"
#include "ghb/root_histogram.cuh"
#include "ghb/split_search.cuh"
#include "ghb/batch_resident.cuh"
#include "ghb/deeper_histogram.cuh"
#include "ghb/higher_order.cuh"
#include "ghb/prediction.cuh"

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
#include <optional>
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
struct TreeGraph {
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  ~TreeGraph() { if (executable) cudaGraphExecDestroy(executable); if (graph) cudaGraphDestroy(graph); }
  template<class F> void capture(cudaStream_t stream, F&& operation) {
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin complete-tree graph capture");
    try { operation(); }
    catch (...) { cudaGraph_t discarded{}; cudaStreamEndCapture(stream, &discarded); if (discarded) cudaGraphDestroy(discarded); throw; }
    check(cudaStreamEndCapture(stream, &graph), "end complete-tree graph capture");
    check(cudaGraphInstantiate(&executable, graph, 0), "instantiate complete-tree graph");
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

void validate_input(const Dataset& data, bool training, bool dense_reference = false) {
  if (!data.columns || (training && !data.rows)) invalid("dataset requires features and training requires at least one row");
  if (data.columns > std::uint32_t(std::numeric_limits<std::int32_t>::max())) invalid("too many features");
  if (data.values.size() != product(data.rows, data.columns, "dataset")) invalid("feature matrix shape mismatch");
  if (!data.feature_types.empty() && data.feature_types.size() != data.columns) invalid("feature type shape mismatch");
  for (auto type : data.feature_types) if (!type_valid(type)) invalid("unknown feature type");
  if (dense_reference) for (float value : data.values) if (std::isinf(value)) invalid("infinite feature value; use NaN for missing data");
  if (!training) return;
  if (!data.outputs) invalid("dataset outputs must be positive");
  if (!data.weights.empty() && data.weights.size() != data.rows) invalid("weight shape mismatch");

}
void validate_config(const Dataset& data, const TrainConfig& config) {
  if (!objective_valid(config.objective)) invalid("unsupported objective");
  if (config.optimization_order < 2 || config.optimization_order > 4) invalid("optimization_order must be 2, 3 or 4");
  if (config.optimization_order > 2 &&
      (config.objective != Objective::binary_logistic || config.tree_build != TreeBuildPolicy::output_batch ||
       config.histogram == HistogramPolicy::shared || !(config.max_leaf_value > 0)))
    invalid("higher orders require binary logistic, output_batch, global/auto histogram and positive max_leaf_value");
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
  if (config.tree_execution != TreeExecution::stream && config.tree_execution != TreeExecution::graph) invalid("unknown tree execution mode");
  if (config.tree_build != TreeBuildPolicy::per_output && config.tree_build != TreeBuildPolicy::output_batch) invalid("unknown tree build policy");
  if (config.tree_build == TreeBuildPolicy::output_batch && (config.root_histogram != RootHistogramPolicy::batched || config.split_batch != SplitBatchPolicy::batched_root))
    invalid("output-batch construction requires batched roots and batched root splits");
  if (config.quantize_policy != QuantizePolicy::radix8 && config.quantize_policy != QuantizePolicy::radix4) invalid("unknown quantization policy");
  if (config.root_histogram != RootHistogramPolicy::per_tree && config.root_histogram != RootHistogramPolicy::batched) invalid("unknown root histogram policy");
  if (config.split_policy != SplitPolicy::block256 && config.split_policy != SplitPolicy::warp32 &&
      config.split_policy != SplitPolicy::warp_wide) invalid("unknown split policy");
  if (config.split_policy == SplitPolicy::warp_wide && config.optimization_order != 2)
    invalid("warp-wide split policy requires optimization order 2");
  if (config.root_counts != RootCountPolicy::per_output && config.root_counts != RootCountPolicy::reuse_global && config.root_counts != RootCountPolicy::reuse_shared) invalid("unknown root count policy");
  if (config.split_batch != SplitBatchPolicy::per_tree && config.split_batch != SplitBatchPolicy::batched_root) invalid("unknown split batching policy");
  if ((config.root_counts != RootCountPolicy::per_output || config.split_batch == SplitBatchPolicy::batched_root) && config.root_histogram != RootHistogramPolicy::batched)
    invalid("count reuse and batched root splits require batched root histograms");
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
void validate_prediction(const Model& model, const Dataset& data, bool dense_reference = false) {
  validate_model(model); validate_input(data, false, dense_reference);
  if (data.columns != model.features.size()) invalid("prediction feature count mismatch");
  if (!data.feature_types.empty()) for (std::size_t f = 0; f < model.features.size(); ++f)
    if (data.feature_types[f] != model.features[f].type) invalid("prediction feature type mismatch");
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

std::size_t recorder_capacity(const TrainConfig& config, std::uint32_t outputs) {
  const auto capacity = sum(16, product(product(outputs, std::size_t(config.max_depth) + 1, "recorder"), 10, "recorder"), "recorder");
  if (capacity > (1U << 20)) invalid("per-round instrumentation exceeds one million reserved scopes; disable recording or reduce outputs/depth");
  return capacity;
}

template<class Record> TrainingResult train_impl(const Dataset& data, const TrainConfig& config) {
  constexpr bool recording = !std::is_same_v<Record, gi::NullRecorder>;
  const auto total_start = Clock::now();
  TrainingResult result;
  auto& model = result.model;
  model.objective = config.objective;
  model.outputs = config.objective == Objective::multiclass_softmax ? config.classes : data.outputs;
  if (model.outputs > std::uint32_t(std::numeric_limits<std::int32_t>::max()) ||
      config.rounds > std::uint32_t(std::numeric_limits<std::int32_t>::max())) invalid("output/round count exceeds context index range");
  model.base_scores.resize(model.outputs);
  model.trees.reserve(product(config.rounds, model.outputs, "tree count"));
  result.training_loss.reserve(std::size_t(config.rounds) + 1);
  Record recorder(recording ? recorder_capacity(config, model.outputs) : 0, config.nvtx);
  gi::Context context;
  context.rows = data.rows; context.features = data.columns; context.stream_id = 1;
  Stream stream;
  QuantizedData quantized;
  auto phase = Clock::now();
  stage(recorder, gi::Stage::quantize, context, gi::Timing::host, nullptr, [&] {
    quantized = fit_quantize(data, config.max_bins, config.max_device_bytes, stream.value, config.quantize_policy);
    model.features = std::move(quantized.features);
  });
  result.quantize_ms = milliseconds(phase);
  const auto view = quantized.view;
  context.bins = view.total_bins;
  const std::uint64_t leaves = std::max<std::uint64_t>(1, data.rows / config.min_leaf_rows);
  const auto histogram_depth = config.max_depth ? config.max_depth - 1 : 0;
  const auto levels = std::max(1U, config.max_depth);
  const std::uint64_t level_bound = std::min<std::uint64_t>(1ULL << histogram_depth, leaves);
  const std::uint64_t node_bound = std::min<std::uint64_t>((1ULL << (config.max_depth + 1)) - 1, leaves * 2 - 1);
  if (node_bound > std::uint64_t(std::numeric_limits<std::int32_t>::max())) invalid("tree capacity exceeds signed node indices");
  const auto histogram_per_node = product(view.total_bins, sizeof(gpu::Stats), "histogram");
  const bool batched_roots = config.root_histogram == RootHistogramPolicy::batched;
  const bool reuse_counts = config.root_counts != RootCountPolicy::per_output;
  const bool batched_splits = config.split_batch == SplitBatchPolicy::batched_root;
  const auto root_batch_capacity = batched_roots ? std::min(model.outputs, config.output_tile_size) : 0U;
  const auto root_cache_count = product(root_batch_capacity, view.total_bins, "root histogram cache");
  result.root_histogram_batch_size = root_batch_capacity;
  result.root_histogram_bytes = product(root_cache_count, sizeof(gpu::Stats), "root histogram cache");
  result.root_count_bytes = reuse_counts ? product(view.total_bins, sizeof(unsigned long long), "root count cache") : 0;
  const auto root_reserved_bytes = sum(result.root_histogram_bytes, result.root_count_bytes, "root histogram storage");
  if (config.root_counts == RootCountPolicy::reuse_shared && !gpu::root_counts_shared_supported(view))
    invalid("shared root counts cannot hold the feature bins; select reuse-global");
  if (root_reserved_bytes >= config.max_histogram_bytes)
    invalid("max_histogram_bytes cannot hold root cache and tree histogram");
  const auto active_capacity = std::min<std::uint64_t>(level_bound,
      (config.max_histogram_bytes - root_reserved_bytes) / histogram_per_node);
  if (!active_capacity) invalid("max_histogram_bytes cannot hold the root histogram");
  const auto nodes_capacity = std::uint32_t(active_capacity), tree_capacity = std::uint32_t(node_bound);
  const auto scan_blocks = 1U + (nodes_capacity - 1) / 1024;
  if (config.histogram == HistogramPolicy::shared && !gpu::shared_supported(view, nodes_capacity))
    invalid("forced shared policy cannot support the configured frontier capacity; choose global/auto or reduce capacity");
  const auto predictions_count = product(data.rows, model.outputs, "predictions");
  const auto derivative_outputs = model.objective == Objective::multiclass_softmax ? model.outputs : std::min(model.outputs, config.output_tile_size);
  const auto derivative_count = product(data.rows, derivative_outputs, "tiled derivatives");
  const auto histogram_count = product(nodes_capacity, view.total_bins, "histogram stats");
  const auto candidate_count = product(std::max(nodes_capacity, batched_splits ? root_batch_capacity : 0U), data.columns, "split candidates");
  const auto root_winner_count = batched_splits ? root_batch_capacity : 0U;
  result.root_split_bytes = product(sum(candidate_count - product(nodes_capacity, data.columns, "tree candidates"), root_winner_count, "added root split scratch"), sizeof(gpu::Split), "added root split scratch");
  const auto loss_blocks = std::min<std::uint32_t>(4096, 1 + (data.rows - 1) / 256);
  const auto init_chunks = std::min<std::uint32_t>(256, 1 + (data.rows - 1) / 1024);
  const auto init_count = product(init_chunks, std::size_t(model.outputs) + 1, "initialization partials");
  const auto history_count = recording && config.tree_execution == TreeExecution::stream ? levels : 0U;
  // Fixed-capacity host exports exchange extra transfer bytes for fewer waits.
  // The cap applies to pinned nodes; status/selector/history are reported too.
  constexpr std::size_t pinned_node_limit = 64ULL << 20;
  const auto tree_bytes = product(tree_capacity, sizeof(Node), "tree export capacity");
  const auto requested_export_batch = std::min({config.tree_export_batch_size, config.output_tile_size,
                                               derivative_outputs, model.outputs});
  const auto export_batch = tree_bytes <= pinned_node_limit
      ? std::min<std::size_t>(requested_export_batch, pinned_node_limit / tree_bytes) : 0;
  const auto export_slots = std::max<std::size_t>(1, export_batch);
  const auto export_node_count = product(export_batch, tree_capacity, "pinned tree export nodes");
  result.tree_export_batch_size = std::uint32_t(export_batch);
  result.pinned_export_bytes = sum(product(export_node_count, sizeof(Node), "pinned tree export nodes"),
    product(export_slots, sum(sizeof(gpu::TreeState) + sizeof(gpu::TreeParameters),
      product(history_count, sizeof(unsigned), "pinned tree history"), "pinned tree metadata"),
      "pinned export slots"), "pinned export payload");
  std::size_t device_bytes = quantized.resident_bytes;
  auto account = [&](std::size_t count, std::size_t element) { device_bytes = sum(device_bytes, product(count, element, "device workspace"), "device workspace"); };
  account(predictions_count, sizeof(double)); account(derivative_count, 2 * sizeof(double));
  account(data.targets.size(), sizeof(float)); account(data.weights.size(), sizeof(float));
  account(model.outputs, sizeof(double)); account(histogram_count, sizeof(gpu::Stats)); account(candidate_count, sizeof(gpu::Split));
  account(root_cache_count, sizeof(gpu::Stats));
  account(reuse_counts ? view.total_bins : 0U, sizeof(unsigned long long));
  account(root_winner_count, sizeof(gpu::Split));
  account(nodes_capacity, sizeof(gpu::Split) + 4 * sizeof(std::int32_t) + sizeof(std::uint32_t));
  account(data.rows, sizeof(std::int32_t)); account(tree_capacity, sizeof(Node));
  account(loss_blocks + std::size_t(init_count) + 2, sizeof(double)); // partial loss, init partials, weight, scalar loss
  account(scan_blocks + history_count + 1, sizeof(unsigned)); // scan counts, optional history, initialization status
  account(1, sizeof(gpu::TreeState) + sizeof(gpu::TreeParameters));
  result.device_bytes = device_bytes;
  result.gradient_bytes = product(derivative_count, 2 * sizeof(double), "derivative memory");
  result.histogram_bytes = sum(product(histogram_count, sizeof(gpu::Stats), "histogram memory"), root_reserved_bytes, "histogram memory");
  result.preparation_peak_bytes = quantized.peak_bytes;
  if (device_bytes > config.max_device_bytes) invalid("persistent trainer workspace exceeds max_device_bytes");
  context.scratch_bytes = device_bytes - quantized.resident_bytes;
  HostScope preparation(recorder, gi::Stage::initialize, context);
  std::size_t free_bytes{}, total_bytes{};
  check(cudaMemGetInfo(&free_bytes, &total_bytes), "query trainer memory budget");
  if (device_bytes - quantized.resident_bytes > free_bytes) invalid("persistent trainer workspace exceeds available GPU memory");
  Device<double> predictions(predictions_count), gradient(derivative_count), hessian(derivative_count), base(model.outputs);
  Device<double> partial_loss(loss_blocks), scalar_loss(1), weight_sum(1), initial_partials(init_count);
  Device<float> targets(data.targets.size()), weights(data.weights.size());
  Device<int> assignments(data.rows), frontier_a(nodes_capacity), frontier_b(nodes_capacity), left_map(nodes_capacity), right_map(nodes_capacity);
  Device<unsigned> scan_offsets(nodes_capacity), scan_counts(scan_blocks), initialization_status(1), level_history(history_count);
  Device<gpu::Stats> histograms(histogram_count), root_cache(root_cache_count);
  Device<unsigned long long> root_counts(reuse_counts ? view.total_bins : 0U);
  Device<gpu::Split> candidates(candidate_count), winners(nodes_capacity), root_winners(root_winner_count);
  Device<Node> tree_nodes(tree_capacity);
  Device<gpu::TreeState> tree_state(1);
  Device<gpu::TreeParameters> selector(1);
  Pinned<gpu::TreeState> host_state(export_slots);
  Pinned<gpu::TreeParameters> host_selector(export_slots);
  Pinned<Node> host_nodes(export_node_count);
  Pinned<double> host_loss(1), host_weight(1);
  Pinned<unsigned> host_status(1), host_history(product(export_slots, history_count, "pinned tree history"));
  Timer tuner(config.histogram == HistogramPolicy::autotune && !batched_roots);
  std::array<std::unique_ptr<TreeGraph>, 2> graphs;
  DrainStream drain{stream.value};
  std::vector<unsigned> round_history(product(model.outputs, history_count, "instrumentation history"));
  preparation.finish();
  phase = Clock::now();
  stage(recorder, gi::Stage::upload, context, gi::Timing::gpu, stream.value, [&] {
    targets.upload(data.targets.data(), data.targets.size(), stream.value);
    weights.upload(data.weights.data(), data.weights.size(), stream.value);
  });
  stream.wait(); result.upload_ms = milliseconds(phase);
  stage(recorder, gi::Stage::initialize, context, gi::Timing::gpu, stream.value, [&] {
    check(gpu::initialize_training(model.objective, targets.data, weights.data, data.rows, model.outputs,
                                  base.data, weight_sum.data, initial_partials.data, init_chunks,
                                  initialization_status.data, stream.value), "validate targets/weights and compute GPU base scores");
  });
  stage(recorder, gi::Stage::download, context, gi::Timing::host, nullptr, [&] {
    check(cudaMemcpyAsync(model.base_scores.data(), base.data, base.bytes(), cudaMemcpyDeviceToHost, stream.value), "export base scores");
    check(cudaMemcpyAsync(host_weight.data, weight_sum.data, sizeof(double), cudaMemcpyDeviceToHost, stream.value), "export total weight");
    check(cudaMemcpyAsync(host_status.data, initialization_status.data, sizeof(unsigned), cudaMemcpyDeviceToHost, stream.value), "export initialization status");
    stream.wait();
    if (*host_status.data) invalid("GPU target/weight/base validation failed, status=" + std::to_string(*host_status.data));
    if (!std::isfinite(*host_weight.data) || !(*host_weight.data > 0)) invalid("invalid GPU total weight");
  });
  stage(recorder, gi::Stage::initialize, context, gi::Timing::gpu, stream.value, [&] {
    // Full-capacity copies include the unused tail. Initialize it once, outside
    // boosting; later builders overwrite live nodes and leave a defined tail.
    // Compact exports also copy Node alignment padding, which member stores do
    // not initialize. Define the complete representation once for every mode.
    check(cudaMemsetAsync(tree_nodes.data, 0, tree_nodes.bytes(), stream.value), "initialize tree export storage");
    check(gpu::initialize_predictions(predictions.data, base.data, data.rows, model.outputs, stream.value), "initialize predictions");
  });
  auto evaluate = [&] {
    stage(recorder, gi::Stage::evaluate, context, gi::Timing::gpu, stream.value, [&] {
      check(gpu::loss(model.objective, predictions.data, targets.data, weights.data, data.rows, model.outputs,
                      partial_loss.data, loss_blocks, stream.value), "evaluate objective partial sums");
      check(gpu::finalize_loss(partial_loss.data, loss_blocks, weight_sum.data,
                              model.objective == Objective::multiclass_softmax ? 1U : model.outputs,
                              scalar_loss.data, stream.value), "reduce and normalize objective on GPU");
    });
    stage(recorder, gi::Stage::download, context, gi::Timing::host, nullptr, [&] {
      check(cudaMemcpyAsync(host_loss.data, scalar_loss.data, sizeof(double), cudaMemcpyDeviceToHost, stream.value), "export objective scalar");
      stream.wait();
      if (!std::isfinite(*host_loss.data) || *host_loss.data < 0) throw std::runtime_error("invalid GPU objective scalar");
      result.training_loss.push_back(*host_loss.data);
    });
  };
  evaluate(); collect(recorder, result);
  using TuningKey = std::pair<unsigned, unsigned>;
  std::map<TuningKey, HistogramPolicy> root_policies;
  const gpu::SplitConfig split_config{config.min_leaf_rows, config.l2, config.min_child_hessian, config.min_gain, config.max_leaf_value};
  // This fixed host submission loop has no data-dependent host branches. Device
  // counts guard every level; the graph uses the same bounded sequence.
  auto submit_tree = [&]<class R>(R& stages, HistogramPolicy root_policy) {
    auto work = context; work.depth = -1; work.active_nodes = 1;
    stage(stages, gi::Stage::initialize, work, gi::Timing::gpu, stream.value, [&] {
      check(gpu::resident_initialize(data.rows, assignments.data, tree_nodes.data, frontier_a.data, tree_state.data, stream.value,
                                     batched_roots && !batched_splits ? root_cache.data : nullptr, batched_roots && !batched_splits ? histograms.data : nullptr,
                                     batched_roots && !batched_splits ? view.total_bins : 0U, batched_roots ? selector.data : nullptr,
                                     batched_splits ? root_winners.data : nullptr, batched_splits ? winners.data : nullptr), "initialize resident tree");
    });
    int* frontier = frontier_a.data; int* next = frontier_b.data;
    for (unsigned depth = 0; depth < levels; ++depth) {
      const auto level_capacity = std::min<unsigned long long>(nodes_capacity, 1ULL << depth);
      work.depth = int(depth); work.active_nodes = 0; // Replaced from exported device history when recording.
      if (history_count) check(cudaMemcpyAsync(level_history.data + depth, &tree_state.data->active_nodes,
                                               sizeof(unsigned), cudaMemcpyDeviceToDevice, stream.value), "record device frontier size");
      const auto policy = config.histogram == HistogramPolicy::shared ? HistogramPolicy::shared : depth == 0 ? root_policy : HistogramPolicy::global;
      if (!(batched_roots && depth == 0)) stage(stages, gi::Stage::histogram, work, gi::Timing::gpu, stream.value, [&] {
        check(gpu::histogram_active(view, assignments.data, gradient.data, hessian.data, derivative_outputs, 0,
                                    unsigned(level_capacity), &tree_state.data->active_nodes, histograms.data,
                                    policy, stream.value, selector.data), "resident gradient histogram");
      });
      if (!(batched_splits && depth == 0)) stage(stages, gi::Stage::split_search, work, gi::Timing::gpu, stream.value, [&] {
        const auto split_search = config.split_policy == SplitPolicy::warp_wide ? gpu::find_splits_warp_wide_active :
          config.split_policy == SplitPolicy::warp32 ? gpu::find_splits_warp_active : gpu::find_splits_active;
        check(split_search(view, histograms.data, unsigned(level_capacity), &tree_state.data->active_nodes,
                                      split_config, config.max_depth == 0, candidates.data, winners.data, stream.value), "resident split search");
      });
      stage(stages, gi::Stage::route, work, gi::Timing::gpu, stream.value, [&] {
        check(gpu::resident_materialize(view, winners.data, frontier, next, left_map.data, right_map.data,
                                        tree_nodes.data, tree_state.data, nodes_capacity, tree_capacity,
                                        scan_offsets.data, scan_counts.data, depth + 1 < config.max_depth,
                                        config.learning_rate, stream.value), "materialize compact device frontier");
        check(gpu::resident_route(view, assignments.data, winners.data, left_map.data, right_map.data, tree_state.data, stream.value), "resident row routing");
        check(gpu::resident_advance(tree_state.data, stream.value), "advance device frontier");
      });
      std::swap(frontier, next);
    }
    work.depth = -1; work.active_nodes = 0;
    stage(stages, gi::Stage::prediction, work, gi::Timing::gpu, stream.value, [&] {
      check(gpu::resident_predict(view, tree_nodes.data, tree_state.data, 0, model.outputs, predictions.data, stream.value, selector.data), "apply resident tree");
    });
  };
  const auto training_start = Clock::now();
  if (reuse_counts && config.rounds) {
    auto work = context; work.depth = 0; work.active_nodes = 1; work.output = -1;
    work.logical_write_bytes = result.root_count_bytes;
    stage(recorder, gi::Stage::histogram, work, gi::Timing::gpu, stream.value, [&] {
      check(gpu::root_counts(view, root_counts.data, config.root_counts == RootCountPolicy::reuse_shared ? gpu::RootCountKernel::shared : gpu::RootCountKernel::global, stream.value), "compute invariant root counts");
    });
  }
  for (unsigned round = 0; round < config.rounds; ++round) {
    context.round = int(round); context.output = -1; context.depth = -1; context.active_nodes = 0;
    if (model.objective == Objective::multiclass_softmax) {
      auto work = context; work.logical_write_bytes = result.gradient_bytes;
      stage(recorder, gi::Stage::gradients, work, gi::Timing::gpu, stream.value, [&] {
        check(gpu::gradients(model.objective, predictions.data, targets.data, weights.data, gradient.data, hessian.data,
                             data.rows, model.outputs, stream.value), "compute pre-round softmax derivatives");
      });
    }
    for (unsigned tile_begin = 0; tile_begin < model.outputs;) {
      const unsigned tile_count = std::min(batched_roots ? root_batch_capacity : derivative_outputs, model.outputs - tile_begin);
      const unsigned derivative_stride = model.objective == Objective::multiclass_softmax ? model.outputs : tile_count;
      const unsigned derivative_begin = model.objective == Objective::multiclass_softmax ? tile_begin : 0U;
      if (model.objective != Objective::multiclass_softmax) {
        auto work = context; work.output = int(tile_begin);
        work.logical_read_bytes = product(product(data.rows, tile_count, "derivative reads"), sizeof(double) + sizeof(float), "derivative reads") + weights.bytes();
        work.logical_write_bytes = product(product(data.rows, tile_count, "derivative writes"), 2 * sizeof(double), "derivative writes");
        stage(recorder, gi::Stage::gradients, work, gi::Timing::gpu, stream.value, [&] {
          check(gpu::gradients_tile(model.objective, predictions.data, targets.data, weights.data, gradient.data, hessian.data,
                                    data.rows, model.outputs, tile_begin, tile_count, stream.value), "compute tiled objective derivatives");
        });
      }
      if (batched_roots) {
        auto work = context; work.output = int(tile_begin); work.depth = 0; work.active_nodes = 1;
        work.operations = tile_count;
        work.logical_write_bytes = product(product(tile_count, view.total_bins, "root cache writes"), sizeof(gpu::Stats), "root cache writes");
        stage(recorder, gi::Stage::histogram, work, gi::Timing::gpu, stream.value, [&] {
          check(gpu::root_histogram(view, gradient.data, hessian.data, derivative_stride, derivative_begin,
                                     tile_count, root_cache.data, stream.value, reuse_counts ? root_counts.data : nullptr), "batched root histograms");
        });
      }
      if (batched_splits) {
        auto work = context; work.output = int(tile_begin); work.depth = 0; work.active_nodes = 1; work.operations = tile_count;
        stage(recorder, gi::Stage::split_search, work, gi::Timing::gpu, stream.value, [&] {
          const auto split_search = config.split_policy == SplitPolicy::warp_wide ? gpu::find_splits_warp_wide :
            config.split_policy == SplitPolicy::warp32 ? gpu::find_splits_warp : gpu::find_splits;
          check(split_search(view, root_cache.data, tile_count, split_config, config.max_depth == 0,
                             candidates.data, root_winners.data, stream.value), "batched root split decisions");
        });
      }
      unsigned pending_exports = 0;
      std::optional<HostScope<Record>> export_scope;
      auto flush_exports = [&] {
        if (!pending_exports) return;
        stream.wait();
        // Validate the whole completed batch before retaining any model nodes.
        for (unsigned slot = 0; slot < pending_exports; ++slot) {
          const auto& state = host_state.data[slot];
          if (state.status) invalid("GPU tree construction failed, status=" + std::to_string(state.status));
          if (!state.node_count || state.node_count > tree_capacity) throw std::runtime_error("invalid exported tree size");
        }
        for (unsigned slot = 0; slot < pending_exports; ++slot) {
          Tree tree; tree.output = host_selector.data[slot].output;
          const Node* first = host_nodes.data + std::size_t(slot) * tree_capacity;
          tree.nodes.assign(first, first + host_state.data[slot].node_count);
          model.trees.push_back(std::move(tree));
          if (history_count) {
            const unsigned* first_history = host_history.data + std::size_t(slot) * history_count;
            std::copy(first_history, first_history + history_count,
              round_history.begin() + std::ptrdiff_t(std::size_t(host_selector.data[slot].output) * history_count));
          }
        }
        export_scope->finish();
        export_scope.reset();
        pending_exports = 0;
      };
      for (unsigned local_output = 0; local_output < tile_count; ++local_output) {
        const unsigned output = tile_begin + local_output;
        context.output = int(output); context.depth = -1; context.active_nodes = 1;
        HistogramPolicy root_policy = batched_roots ? HistogramPolicy::global : config.histogram;
        if (root_policy == HistogramPolicy::autotune) {
          const TuningKey key{output, tile_count};
          const auto cached = root_policies.find(key);
          if (cached != root_policies.end()) root_policy = cached->second;
          else {
            root_policy = HistogramPolicy::global;
            if (gpu::shared_supported(view, 1)) {
              auto work = context; work.depth = 0; work.operations = 14;
              stage(recorder, gi::Stage::histogram, work, gi::Timing::host, nullptr, [&] {
                check(gpu::resident_initialize(data.rows, assignments.data, tree_nodes.data, frontier_a.data, tree_state.data, stream.value), "initialize root calibration");
                auto run_histogram = [&](HistogramPolicy policy) {
                  check(gpu::histogram(view, assignments.data, gradient.data, hessian.data, tile_count, local_output,
                                       1, histograms.data, policy, stream.value), "calibrate actual root histogram");
                };
                for (unsigned i = 0; i < 2; ++i) { run_histogram(HistogramPolicy::global); run_histogram(HistogramPolicy::shared); }
                stream.wait();
                TuningRecord measured; measured.active_nodes = 1; measured.output = output;
                measured.round = round; measured.depth = 0; measured.output_tile_size = tile_count;
                for (unsigned i = 0; i < 5; ++i) {
                  auto global = [&] { measured.global_samples_ms[i] = tuner.measure(stream.value, [&] { run_histogram(HistogramPolicy::global); }); };
                  auto shared = [&] { measured.shared_samples_ms[i] = tuner.measure(stream.value, [&] { run_histogram(HistogramPolicy::shared); }); };
                  if (i % 2) { shared(); global(); } else { global(); shared(); }
                }
                auto global = measured.global_samples_ms, shared = measured.shared_samples_ms;
                std::sort(global.begin(), global.end()); std::sort(shared.begin(), shared.end());
                measured.global_ms = global[2]; measured.shared_ms = shared[2];
                root_policy = shared[2] < global[2] ? HistogramPolicy::shared : HistogramPolicy::global;
                measured.selected = root_policy; result.tuning.push_back(measured);
              });
            }
            root_policies.emplace(key, root_policy);
          }
        }
        auto* queued_selector = host_selector.data + (export_batch ? pending_exports : 0);
        *queued_selector = gpu::TreeParameters{derivative_stride, derivative_begin + local_output, output, local_output};
        stage(recorder, gi::Stage::upload, context, gi::Timing::gpu, stream.value, [&] { selector.upload(queued_selector, 1, stream.value); });
        const auto graph_index = root_policy == HistogramPolicy::shared ? 1U : 0U;
        if (config.tree_execution == TreeExecution::graph && !graphs[graph_index]) {
          stage(recorder, gi::Stage::initialize, context, gi::Timing::host, nullptr, [&] {
            auto graph = std::make_unique<TreeGraph>(); gi::NullRecorder disabled;
            graph->capture(stream.value, [&] { submit_tree(disabled, root_policy); });
            graphs[graph_index] = std::move(graph);
          });
        }
        stage(recorder, gi::Stage::tree_build, context, gi::Timing::gpu, stream.value, [&] {
          if (config.tree_execution == TreeExecution::graph) check(cudaGraphLaunch(graphs[graph_index]->executable, stream.value), "replay complete resident tree");
          else submit_tree(recorder, root_policy);
        });
        if (export_batch) {
          if (!pending_exports) {
            auto work = context;
            work.operations = std::min<std::size_t>(export_batch, tile_count - local_output);
            work.logical_read_bytes = product(work.operations,
              sum(tree_bytes, sum(sizeof(gpu::TreeState), level_history.bytes(), "tree export metadata"),
                  "tree export payload"), "batched export payload");
            // One host scope spans this batch's enqueue, intervening tree work,
            // single wait and model export. Its duration overlaps tree scopes.
            export_scope.emplace(recorder, gi::Stage::download, work);
          }
          const auto slot = pending_exports;
          check(cudaMemcpyAsync(host_state.data + slot, tree_state.data, sizeof(gpu::TreeState),
                                cudaMemcpyDeviceToHost, stream.value), "queue resident tree status export");
          if (history_count) check(cudaMemcpyAsync(host_history.data + std::size_t(slot) * history_count,
            level_history.data, level_history.bytes(), cudaMemcpyDeviceToHost, stream.value), "queue instrumentation frontier history");
          check(cudaMemcpyAsync(host_nodes.data + std::size_t(slot) * tree_capacity, tree_nodes.data, tree_bytes,
                                cudaMemcpyDeviceToHost, stream.value), "queue bounded tree node export");
          ++pending_exports;
          if (pending_exports == export_batch || local_output + 1 == tile_count) flush_exports();
        } else {
          // Existing compact path: status/count wait, then exact node copy wait.
          stage(recorder, gi::Stage::download, context, gi::Timing::host, nullptr, [&] {
            check(cudaMemcpyAsync(host_state.data, tree_state.data, sizeof(gpu::TreeState), cudaMemcpyDeviceToHost, stream.value), "export resident tree status");
            if (history_count) check(cudaMemcpyAsync(host_history.data, level_history.data, level_history.bytes(), cudaMemcpyDeviceToHost, stream.value), "export instrumentation frontier history");
            stream.wait();
            if (host_state.data->status) invalid("GPU tree construction failed, status=" + std::to_string(host_state.data->status));
            if (!host_state.data->node_count || host_state.data->node_count > tree_capacity) throw std::runtime_error("invalid exported tree size");
            Tree tree; tree.output = output; tree.nodes.resize(host_state.data->node_count);
            // Store first: exported host backing survives the outer stream drain
            // even if a subsequent CUDA operation throws.
            model.trees.push_back(std::move(tree));
            auto& retained = model.trees.back();
            check(cudaMemcpyAsync(retained.nodes.data(), tree_nodes.data, product(retained.nodes.size(), sizeof(Node), "model export"), cudaMemcpyDeviceToHost, stream.value), "export compact model nodes");
            stream.wait();
            if (history_count) std::copy(host_history.data, host_history.data + history_count, round_history.begin() + std::ptrdiff_t(std::size_t(output) * history_count));
          });
        }
      }
      tile_begin += tile_count;
    }
    context.output = -1; context.depth = -1; context.active_nodes = 0;
    evaluate();
    if constexpr (recording) {
      auto samples = recorder.collect(false);
      if (!samples) throw std::logic_error("completed resident round has pending events");
      if (history_count) for (auto& sample : *samples) {
        auto& work = sample.context;
        if (work.output >= 0 && unsigned(work.output) < model.outputs && work.depth >= 0 && unsigned(work.depth) < levels)
          work.active_nodes = round_history[std::size_t(work.output) * history_count + unsigned(work.depth)];
      }
      result.samples.insert(result.samples.end(), samples->begin(), samples->end()); recorder.reset();
    }
  }
  stream.wait(); result.training_ms = milliseconds(training_start);
  validate_model(model); result.total_ms = milliseconds(total_start);
  return result;
}

#include "batch_training.inc"

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
  auto result = config.optimization_order == 3
      ? (config.record_stages ? train_batch_impl<gi::Recorder, 3>(data, config) : train_batch_impl<gi::NullRecorder, 3>(data, config))
      : config.optimization_order == 4
      ? (config.record_stages ? train_batch_impl<gi::Recorder, 4>(data, config) : train_batch_impl<gi::NullRecorder, 4>(data, config))
      : config.tree_build == TreeBuildPolicy::output_batch
      ? (config.record_stages ? train_batch_impl<gi::Recorder>(data, config) : train_batch_impl<gi::NullRecorder>(data, config))
      : (config.record_stages ? train_impl<gi::Recorder>(data, config) : train_impl<gi::NullRecorder>(data, config));
  result.total_ms = milliseconds(start); // Includes input checks and device cleanup.
  return result;
}
std::vector<double> Model::predict(const Dataset& data, bool raw) const {
  validate_prediction(*this, data, true);
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
std::vector<double> Model::predict_gpu(const Dataset& data, bool raw, PredictionPolicy policy) const {
  if (policy != PredictionPolicy::per_tree && policy != PredictionPolicy::fused_output)
    invalid("unknown GPU prediction policy");
  validate_prediction(*this, data);
  std::vector<double> result(product(data.rows, outputs, "prediction output"));
  if (!data.rows) return result;
  const bool fused = policy == PredictionPolicy::fused_output;
  std::size_t node_count{};
  for (const auto& tree : trees)
    node_count = fused ? sum(node_count, tree.nodes.size(), "packed inference nodes") : std::max(node_count, tree.nodes.size());
  const auto descriptor_count = fused ? trees.size() : 0;
  const auto offset_count = fused ? sum(outputs, 1, "inference output offsets") : 0;
  Stream stream;
  std::size_t free_bytes{}, total_bytes{};
  check(cudaMemGetInfo(&free_bytes, &total_bytes), "query inference memory budget");
  const auto prediction_bytes = sum(sum(product(sum(result.size(), outputs, "inference predictions"), sizeof(double), "inference predictions"),
                                    product(node_count, sizeof(Node), "inference nodes"), "inference workspace"),
                                    sum(product(descriptor_count, sizeof(gpu::PredictionTree), "inference descriptors"),
                                        product(offset_count, sizeof(std::uint64_t), "inference offsets"), "inference metadata"), "inference workspace");
  if (prediction_bytes >= free_bytes) invalid("inference payload exceeds available GPU memory");
  auto quantized = encode_quantize(data, features, free_bytes - prediction_bytes, stream.value);
  Device<double> predictions(result.size()), base(outputs);
  Device<Node> nodes(node_count);
  Device<gpu::PredictionTree> descriptors(descriptor_count);
  Device<std::uint64_t> offsets(offset_count);
  // Host storage precedes DrainStream so asynchronous uploads remain valid
  // through stream completion, including any later exception.
  std::vector<Node> packed_nodes;
  std::vector<gpu::PredictionTree> packed_trees;
  std::vector<std::uint64_t> packed_offsets;
  DrainStream drain{stream.value};
  base.upload(base_scores.data(), outputs, stream.value);
  if (fused) {
    packed_offsets.resize(offset_count);
    for (const auto& tree : trees) ++packed_offsets[std::size_t(tree.output) + 1];
    for (std::size_t output = 1; output < offset_count; ++output)
      packed_offsets[output] += packed_offsets[output - 1];
    std::vector<std::uint64_t> next(packed_offsets);
    std::vector<std::size_t> order(descriptor_count);
    for (std::size_t index = 0; index < trees.size(); ++index)
      order[next[trees[index].output]++] = index;
    packed_nodes.reserve(node_count); packed_trees.reserve(descriptor_count);
    for (const auto index : order) {
      const auto& tree = trees[index];
      packed_trees.push_back({packed_nodes.size(), std::uint32_t(tree.nodes.size()), 0});
      packed_nodes.insert(packed_nodes.end(), tree.nodes.begin(), tree.nodes.end());
    }
    nodes.upload(packed_nodes.data(), packed_nodes.size(), stream.value);
    descriptors.upload(packed_trees.data(), packed_trees.size(), stream.value);
    offsets.upload(packed_offsets.data(), packed_offsets.size(), stream.value);
    check(gpu::predict_forest(quantized.view, nodes.data, node_count, descriptors.data, descriptor_count,
                             offsets.data, base.data, outputs, predictions.data, stream.value), "predict ordered forest");
  } else {
    check(gpu::initialize_predictions(predictions.data, base.data, data.rows, outputs, stream.value), "initialize inference predictions");
    for (const auto& tree : trees) {
      nodes.upload(tree.nodes.data(), tree.nodes.size(), stream.value);
      check(gpu::add_tree(quantized.view, nodes.data, std::uint32_t(tree.nodes.size()), tree.output, outputs, predictions.data, stream.value), "predict tree");
    }
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
