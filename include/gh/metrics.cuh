#pragma once
#include "gh/types.cuh"

namespace gh {
enum class MetricProfile : u32 { synthetic, real_data };
enum class MetricTask : u32 { regression, binary, multiclass, multilabel };
enum class MetricName : u32 {
  mse, rmse, mae, r2, logloss, accuracy, brier, auc, f1, average_precision,
  hamming_loss, exact_match, micro_f1, macro_f1, micro_ap, macro_ap,
  micro_auc, macro_auc, precision_at_1, precision_at_3, precision_at_5
};
constexpr u32 aggregate_output = UINT32_MAX;
struct MetricInput {
  Array<const double> targets, predictions, weights;
  u32 rows{}, outputs{};
  MetricTask task{};
  MetricProfile profile{};
  double probability_tolerance{1e-6};
};
struct Metric { MetricName name{}; u32 output{aggregate_output}; double value{}; u32 available{}; };
struct MetricReport {
  Array<Metric> metrics;
  u32 count{}, outputs{}, ap_outputs{}, auc_outputs{};
  MetricTask task{};
  MetricProfile profile{};
};
enum class MetricVerdict : u32 { pass, regression, not_applicable, invalid };
struct MetricGate { u32 checked{}, regressions{}, unavailable{}, invalid{}; };
struct SignalInput { Array<const double> labels, predictions; u32 size{}; };
enum class ThresholdMode : u32 { validation, frozen };
struct OperatingPoint {
  double threshold{}, recall{}, false_positive_rate{}, precision{}, f1{};
  u64 true_positive{}, false_positive{}, false_negative{}, true_negative{};
  u64 positives{}, negatives{};
  u32 meets_five_percent_fpr{};
};
struct SignalReport { OperatingPoint fixed, selected; ThresholdMode origin{}; };

// All computation and submission are device-only. Caller supplies capacities,
// zero Status and global disjoint backing arrays live through tail completion.
__device__ cudaError_t evaluate_metrics(MetricInput, MetricReport*, Workspace, Status*);
__device__ cudaError_t compare_metrics(const MetricReport*, const MetricReport*,
    Array<MetricVerdict>, MetricGate*, Status*);
__device__ cudaError_t signal_metrics(SignalInput, ThresholdMode, double frozen_threshold,
    SignalReport*, Workspace, Status*);
}
