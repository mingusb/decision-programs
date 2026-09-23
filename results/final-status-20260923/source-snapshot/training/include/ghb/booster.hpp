#pragma once

#include "ghb/instrumentation.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace ghb {

enum class Objective : std::uint32_t { squared_error, binary_logistic, multiclass_softmax };
enum class PredictionPolicy : std::uint32_t { per_tree, fused_output, fused_output_slab };
enum class EncodingPolicy : std::uint32_t { per_tile, final_status };
enum class FeatureType : std::uint32_t { numeric, categorical };
enum class HistogramPolicy : std::uint32_t { global, shared, autotune };
enum class TreeExecution : std::uint32_t { stream, graph };
enum class TreeBuildPolicy : std::uint32_t { per_output, output_batch };
enum class QuantizePolicy : std::uint32_t { radix8, radix4 };
enum class RootHistogramPolicy : std::uint32_t { per_tree, batched };
enum class SplitPolicy : std::uint32_t { block256, warp32, warp_wide };
enum class RootCountPolicy : std::uint32_t { per_output, reuse_global, reuse_shared };
enum class SplitBatchPolicy : std::uint32_t { per_tree, batched_root };

// Host input. Values, regression targets, and independent binary targets are
// row-major. Multiclass targets contain one class index per row. Weights are
// optional, nonnegative, and shared by all outputs of a row.
struct Dataset {
  std::uint32_t rows{}, columns{}, outputs{1};
  std::vector<float> values, targets, weights;
  std::vector<FeatureType> feature_types; // empty means all numeric
};

struct TrainConfig {
  Objective objective{Objective::squared_error};
  std::uint32_t classes{2}, rounds{100}, max_depth{6}, max_bins{256};
  std::uint32_t min_leaf_rows{10};
  double learning_rate{0.1}, l2{1.0}, min_child_hessian{1e-8}, min_gain{0.0};
  double max_leaf_value{0.0}; // zero disables clipping
  // Experimental orders 3/4: binary logistic, output_batch, global/auto
  // histograms and positive max_leaf_value required. Order 2 retains its path.
  std::uint32_t optimization_order{2};
  // per_output auto calibrates uncached roots; output_batch auto uses the
  // measured global deeper policy without per-output calibration launches.
  HistogramPolicy histogram{HistogramPolicy::autotune};
  TreeExecution tree_execution{TreeExecution::stream};
  TreeBuildPolicy tree_build{TreeBuildPolicy::per_output};
  QuantizePolicy quantize_policy{QuantizePolicy::radix8};
  // Measured root policies; FP64 atomics retain their existing unordered sums.
  // Per-tree roots require explicit per_output counts and per_tree split batches.
  RootHistogramPolicy root_histogram{RootHistogramPolicy::batched};
  SplitPolicy split_policy{SplitPolicy::warp32};
  RootCountPolicy root_counts{RootCountPolicy::reuse_global};
  SplitBatchPolicy split_batch{SplitBatchPolicy::batched_root};
  // Zero exports each compact tree. Positive values request bounded pinned
  // batches; effective width may be smaller to satisfy tile/storage limits.
  std::uint32_t tree_export_batch_size{0};
  std::size_t max_histogram_bytes{512ULL << 20};
  // Independent regression/binary outputs share bounded gradient workspace.
  // Multiclass uses a full pre-round gradient snapshot because outputs couple.
  std::uint32_t output_tile_size{32};
  std::size_t max_device_bytes{4ULL << 30};
  bool record_stages{false}, nvtx{true};
};

// Bin zero is missing. Numeric bins are 1 + lower_bound(cuts, value).
// Numeric cuts exclude the maximum training value. Categorical bins are
// 1 + the index in categories; unseen categories map to missing.
struct Feature {
  FeatureType type{FeatureType::numeric};
  std::vector<float> cuts, categories;
  std::uint32_t bins() const;
  std::uint16_t encode(float value) const;
};

// A node is terminal when feature == -1. Numerical left: bin <= threshold;
// categorical left: bin == threshold; missing uses missing_left.
struct Node {
  std::int32_t feature{-1}, left{-1}, right{-1};
  std::uint32_t threshold{}, missing_left{};
  double value{}; // includes learning rate
};
struct Tree { std::uint32_t output{}; std::vector<Node> nodes; };
struct Model {
  Objective objective{Objective::squared_error};
  std::uint32_t outputs{1};
  std::vector<double> base_scores;
  std::vector<Feature> features;
  std::vector<Tree> trees;
  // Row-major raw margins, or probabilities for classification.
  std::vector<double> predict(const Dataset& data, bool raw = false) const;
  // Fused policies pack the complete forest and require more device memory.
  // fused_output_slab also combines model uploads in one aligned allocation.
  // Both preserve per-output tree addition order; per_tree remains the default.
  // final_status keeps encoding on one stream and checks input status at the end.
  std::vector<double> predict_gpu(const Dataset& data, bool raw = false,
                                PredictionPolicy policy = PredictionPolicy::per_tree,
                                EncodingPolicy encoding_policy = EncodingPolicy::per_tile) const;
  void save(std::ostream& out) const;
  static Model load(std::istream& in);
};

struct TuningRecord {
  std::uint32_t active_nodes{}, output{};
  double global_ms{}, shared_ms{};
  HistogramPolicy selected{HistogramPolicy::global};
  std::uint32_t round{}, depth{}, output_tile_size{};
  std::array<double, 5> global_samples_ms{}, shared_samples_ms{};
};
struct TrainingResult {
  Model model;
  std::vector<double> training_loss;
  std::vector<TuningRecord> tuning;
  std::vector<instrumentation::Sample> samples;
  // Payload bytes owned by the trainer on the GPU, excluding CUDA/recorder
  // bookkeeping; gradient_bytes includes every selected derivative plane.
  std::size_t device_bytes{}, gradient_bytes{}, histogram_bytes{};
  // Included in histogram_bytes and device_bytes, not an additional payload.
  std::size_t root_histogram_bytes{};
  // Counts are included in histogram_bytes/device_bytes; added split scratch
  // (including enlarged shared candidate storage) is included in device_bytes.
  std::size_t root_count_bytes{}, root_split_bytes{};
  std::uint32_t root_histogram_batch_size{};
  // Output-batch construction retains independent state for this many outputs.
  // State payload is included in device_bytes; zero for the legacy path.
  std::uint32_t tree_batch_size{1};
  std::size_t tree_state_bytes{};
  // Feature-fit peak; total owned device payload peak is the maximum of this
  // and device_bytes because preparation scratch is released before training.
  std::size_t preparation_peak_bytes{};
  std::uint32_t tree_export_batch_size{};
  std::size_t pinned_export_bytes{};
  // Initial objective followed by one value per completed boosting round.
  // Regression is half weighted mean squared error, averaged over outputs;
  // independent binary log loss is also averaged over outputs.
  double quantize_ms{}, upload_ms{}, training_ms{}, total_ms{};
};

TrainingResult train(const Dataset& data, const TrainConfig& config = {});

} // namespace ghb
