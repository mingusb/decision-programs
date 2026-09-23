#include "ghb/booster.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
std::size_t product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b) throw std::invalid_argument("benchmark shape overflows address space");
  return a * b;
}
void finite_nonnegative(double value, const char* field) {
  if (!std::isfinite(value) || value < 0) throw std::runtime_error(std::string("invalid benchmark metric: ") + field);
}
struct Options {
  std::uint32_t rows{65536}, test_rows{8192}, features{32}, outputs{1};
  std::uint64_t seed{2026092201};
  ghb::TrainConfig config;
  std::filesystem::path directory;
  Options() { config.rounds = 10; config.max_depth = 5; config.max_bins = 64; }
};
template<class T> T integer(std::string_view value) {
  T result{};
  auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc{} || end != value.data() + value.size())
    throw std::invalid_argument("invalid unsigned integer argument");
  return result;
}
Options parse(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view key = argv[i];
    if (key == "--help") {
      std::cout << "ghb_bench [--objective regression|binary|multiclass] [--outputs K]\n"
                   "  [--rows N] [--test-rows N] [--features F] [--classes K]\n"
                   "  [--rounds N] [--depth N] [--bins N] [--output-tile N]\n"
                   "  [--histogram auto|global|shared] [--instrumentation off|timing|nvtx]\n"
                   "  [--tree-execution stream|graph]\n"
                   "  [--quantize-policy radix8|radix4]\n"
                   "  [--tree-export-batch N] (0=compact export)\n"
                   "  [--seed N] [--max-device-bytes N] [--output-dir NEW_DIRECTORY]\n";
      std::exit(0);
    }
    if (++i == argc) throw std::invalid_argument("missing argument value");
    const std::string_view value = argv[i];
    if (key == "--rows") options.rows = integer<std::uint32_t>(value);
    else if (key == "--test-rows") options.test_rows = integer<std::uint32_t>(value);
    else if (key == "--features") options.features = integer<std::uint32_t>(value);
    else if (key == "--outputs") options.outputs = integer<std::uint32_t>(value);
    else if (key == "--classes") options.config.classes = integer<std::uint32_t>(value);
    else if (key == "--rounds") options.config.rounds = integer<std::uint32_t>(value);
    else if (key == "--depth") options.config.max_depth = integer<std::uint32_t>(value);
    else if (key == "--bins") options.config.max_bins = integer<std::uint32_t>(value);
    else if (key == "--output-tile") options.config.output_tile_size = integer<std::uint32_t>(value);
    else if (key == "--tree-export-batch") options.config.tree_export_batch_size = integer<std::uint32_t>(value);
    else if (key == "--max-device-bytes") options.config.max_device_bytes = integer<std::size_t>(value);
    else if (key == "--seed") options.seed = integer<std::uint64_t>(value);
    else if (key == "--output-dir") options.directory = value;
    else if (key == "--objective") {
      if (value == "regression") options.config.objective = ghb::Objective::squared_error;
      else if (value == "binary") options.config.objective = ghb::Objective::binary_logistic;
      else if (value == "multiclass") options.config.objective = ghb::Objective::multiclass_softmax;
      else throw std::invalid_argument("invalid objective");
    } else if (key == "--histogram") {
      if (value == "auto") options.config.histogram = ghb::HistogramPolicy::autotune;
      else if (value == "global") options.config.histogram = ghb::HistogramPolicy::global;
      else if (value == "shared") options.config.histogram = ghb::HistogramPolicy::shared;
      else throw std::invalid_argument("invalid histogram policy");
    } else if (key == "--tree-execution") {
      if (value == "stream") options.config.tree_execution = ghb::TreeExecution::stream;
      else if (value == "graph") options.config.tree_execution = ghb::TreeExecution::graph;
      else throw std::invalid_argument("invalid tree execution policy");
    } else if (key == "--quantize-policy") {
      if (value == "radix8") options.config.quantize_policy = ghb::QuantizePolicy::radix8;
      else if (value == "radix4") options.config.quantize_policy = ghb::QuantizePolicy::radix4;
      else throw std::invalid_argument("invalid quantization policy");
    } else if (key == "--instrumentation") {
      if (value != "off" && value != "timing" && value != "nvtx")
        throw std::invalid_argument("invalid instrumentation mode");
      options.config.record_stages = value != "off";
      options.config.nvtx = value == "nvtx";
    } else throw std::invalid_argument("unknown option: " + std::string(key));
  }
  if (!options.rows || !options.test_rows || !options.features || !options.outputs)
    throw std::invalid_argument("rows, test rows, features, and outputs must be positive");
  if (options.config.objective == ghb::Objective::multiclass_softmax && options.outputs != 1)
    throw std::invalid_argument("multiclass uses --classes; --outputs is for independent targets");
  if (options.config.max_bins < 2 || options.config.max_bins > 65536 || options.config.max_depth > 30 ||
      !options.config.output_tile_size || !options.config.max_device_bytes)
    throw std::invalid_argument("invalid bin/depth/tile/memory bounds");
  if (options.config.objective == ghb::Objective::multiclass_softmax && options.config.classes < 2)
    throw std::invalid_argument("multiclass requires at least two classes");
  const auto prediction_outputs = options.config.objective == ghb::Objective::multiclass_softmax ? options.config.classes : options.outputs;
  if (product(product(std::max(options.rows, options.test_rows), prediction_outputs), sizeof(double)) > options.config.max_device_bytes)
    throw std::invalid_argument("dense predictions alone exceed max_device_bytes");
  return options;
}
const char* objective_name(ghb::Objective objective) {
  switch (objective) {
    case ghb::Objective::squared_error: return "regression";
    case ghb::Objective::binary_logistic: return "binary";
    case ghb::Objective::multiclass_softmax: return "multiclass";
  }
  throw std::invalid_argument("unknown objective");
}
const char* policy_name(ghb::HistogramPolicy policy) {
  switch (policy) {
    case ghb::HistogramPolicy::global: return "global";
    case ghb::HistogramPolicy::shared: return "shared";
    case ghb::HistogramPolicy::autotune: return "auto";
  }
  throw std::invalid_argument("unknown policy");
}
const char* execution_name(ghb::TreeExecution execution) {
  switch (execution) {
    case ghb::TreeExecution::stream: return "stream";
    case ghb::TreeExecution::graph: return "graph";
  }
  throw std::invalid_argument("unknown tree execution policy");
}
const char* quantize_name(ghb::QuantizePolicy policy) {
  switch (policy) {
    case ghb::QuantizePolicy::radix8: return "radix8";
    case ghb::QuantizePolicy::radix4: return "radix4";
  }
  throw std::invalid_argument("unknown quantization policy");
}
std::string quote(std::string_view value) {
  std::ostringstream out; out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
    else out << c;
  }
  out << '"'; return out.str();
}
ghb::Dataset generate(const Options& options, std::uint32_t rows, std::uint64_t seed) {
  ghb::Dataset data;
  data.rows = rows; data.columns = options.features; data.outputs = options.outputs;
  data.values.resize(product(rows, data.columns));
  data.targets.resize(product(rows, data.outputs));
  data.weights.resize(rows);
  data.feature_types.resize(data.columns, ghb::FeatureType::numeric);
  const bool categorical = data.columns >= 4;
  if (categorical) data.feature_types.back() = ghb::FeatureType::categorical;
  const auto category_count = std::min(4U, options.config.max_bins - 1);
  std::mt19937_64 rng(seed), weights_rng(seed ^ 0xd1b54a32d192ed03ULL);
  auto unit = [&]() { return float(rng() >> 40) * (1.0f / 16777216.0f); };
  const auto mix = [](std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL; value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL; return value ^ (value >> 31);
  };
  const auto hash_unit = [&](std::uint64_t value) { return float(mix(value) >> 40) * (1.0f / 16777216.0f); };
  for (std::uint32_t row = 0; row < rows; ++row) {
    for (std::uint32_t f = 0; f < data.columns; ++f) {
      float value = 2 * unit() - 1;
      if (categorical && f + 1 == data.columns) value = float(rng() % category_count);
      if (f % 7 == 0 && rng() % 31 == 0) value = std::numeric_limits<float>::quiet_NaN();
      data.values[std::size_t(row) * data.columns + f] = value;
    }
    // Features and weights are unchanged when objective/output count changes.
    // Keep a positive total weight even for a one-row holdout.
    data.weights[row] = row % 97 == 96 ? 0.0f : 0.5f + float(weights_rng() >> 40) * (1.0f / 16777216.0f);
    for (std::uint32_t output = 0; output < data.outputs; ++output) {
      // Output-specific functions avoid short modular threshold cycles.
      const auto numeric_columns = data.columns - std::uint32_t(categorical);
      const auto f = std::uint32_t(mix(output) % numeric_columns);
      float x = data.values[std::size_t(row) * data.columns + f];
      const bool missing = std::isnan(x);
      if (missing) x = 0;
      const float threshold = hash_unit(std::uint64_t(output) + 0x632be59bd9b4e019ULL) - 0.5f;
      const float category = categorical ? data.values[std::size_t(row) * data.columns + data.columns - 1] : 0;
      const float category_term = categorical ? (category == float(mix(output + 0x12345ULL) % category_count) ? 0.35f : -0.15f) : 0;
      const float score = (x > threshold ? 0.7f : -0.7f) + 0.5f * x - 0.3f * threshold + category_term + (missing ? 0.6f : 0);
      float target;
      if (options.config.objective == ghb::Objective::squared_error)
        target = score + 0.02f * (hash_unit(seed ^ (std::uint64_t(row) << 32) ^ output) - 0.5f);
      else if (options.config.objective == ghb::Objective::binary_logistic) target = score > 0 ? 1 : 0;
      else target = float(std::min(options.config.classes - 1,
          std::uint32_t(std::max(0.0, (double(x) + 1) * 0.5 * options.config.classes))));
      data.targets[std::size_t(row) * data.outputs + output] = target;
    }
  }
  return data;
}
double objective_loss(const ghb::Dataset& data, const std::vector<double>& margins,
                      ghb::Objective objective, std::uint32_t outputs) {
  if (margins.size() != product(data.rows, outputs)) throw std::runtime_error("loss prediction shape mismatch");
  for (double value : margins) if (!std::isfinite(value)) throw std::runtime_error("nonfinite loss margin");
  long double sum = 0, weight_sum = 0;
  for (std::uint32_t row = 0; row < data.rows; ++row) {
    const double weight = data.weights.empty() ? 1 : data.weights[row];
    weight_sum += weight;
    if (!weight) continue;
    if (objective == ghb::Objective::multiclass_softmax) {
      const auto label = static_cast<std::uint32_t>(data.targets[row]);
      const auto first = margins.begin() + std::ptrdiff_t(std::size_t(row) * outputs);
      const double maximum = *std::max_element(first, first + outputs);
      long double denominator{};
      for (std::uint32_t output = 0; output < outputs; ++output) denominator += std::exp(first[output] - maximum);
      sum += weight * (std::log(denominator) + (maximum - first[label]));
    } else for (std::uint32_t output = 0; output < outputs; ++output) {
      const auto index = std::size_t(row) * outputs + output;
      const double p = margins[index], y = data.targets[index];
      if (objective == ghb::Objective::squared_error) sum += weight * 0.5L * (p - y) * (p - y);
      else sum += weight * ((y == 1 ? std::max(-p, 0.0) : std::max(p, 0.0)) + std::log1p(std::exp(-std::abs(p))));
    }
  }
  if (!(weight_sum > 0)) throw std::runtime_error("held-out total weight is not positive");
  const double loss = double(sum / weight_sum / (objective == ghb::Objective::multiclass_softmax ? 1 : outputs));
  finite_nonnegative(loss, "heldout_loss"); return loss;
}
void save_csv(const std::filesystem::path& directory, const ghb::Dataset& data,
              const std::vector<double>& predicted, const std::vector<double>& base,
              const ghb::Model& model) {
  std::ofstream targets(directory / "targets.csv"), predictions(directory / "predictions.csv"), baseline(directory / "baseline.csv");
  targets.exceptions(std::ios::badbit | std::ios::failbit);
  predictions.exceptions(std::ios::badbit | std::ios::failbit);
  baseline.exceptions(std::ios::badbit | std::ios::failbit);
  targets << std::setprecision(17) << "row_id";
  const bool multiclass = model.objective == ghb::Objective::multiclass_softmax;
  const std::uint32_t target_outputs = multiclass ? 1 : model.outputs;
  for (std::uint32_t k = 0; k < target_outputs; ++k)
    targets << ',' << (target_outputs == 1 ? "target" : "target_" + std::to_string(k));
  targets << ",weight\n";
  for (auto* out : {&predictions, &baseline}) {
    *out << std::setprecision(17) << "row_id";
    for (std::uint32_t k = 0; k < model.outputs; ++k)
      *out << ',' << (multiclass ? "p" + std::to_string(k) : model.outputs == 1 ? "prediction" : "prediction_" + std::to_string(k));
    *out << '\n';
  }
  for (std::uint32_t row = 0; row < data.rows; ++row) {
    targets << row; predictions << row; baseline << row;
    for (std::uint32_t k = 0; k < target_outputs; ++k) targets << ',' << data.targets[std::size_t(row) * target_outputs + k];
    targets << ',' << (data.weights.empty() ? 1.0f : data.weights[row]) << '\n';
    for (std::uint32_t k = 0; k < model.outputs; ++k) {
      predictions << ',' << predicted[std::size_t(row) * model.outputs + k];
      baseline << ',' << base[std::size_t(row) * model.outputs + k];
    }
    predictions << '\n'; baseline << '\n';
  }
  targets.close(); predictions.close(); baseline.close();
}
void samples_json(std::ostream& out, const std::vector<ghb::instrumentation::Sample>& samples) {
  out << '[';
  for (std::size_t i = 0; i < samples.size(); ++i) {
    if (i) out << ',';
    const auto& s = samples[i]; const auto& c = s.context;
    out << "{\"id\":" << s.id << ",\"stage\":" << quote(ghb::instrumentation::name(s.stage))
        << ",\"timing\":\"" << (s.timing == ghb::instrumentation::Timing::gpu ? "gpu" : "host")
        << "\",\"host_start_ns\":" << s.host_start_ns << ",\"host_end_ns\":" << s.host_end_ns
        << ",\"gpu_ms\":";
    if (s.gpu_ms) out << *s.gpu_ms; else out << "null";
    out << ",\"context\":{\"round\":" << c.round << ",\"depth\":" << c.depth << ",\"output\":" << c.output
        << ",\"repetition\":" << c.repetition << ",\"stream_id\":" << c.stream_id
        << ",\"active_nodes\":" << c.active_nodes << ",\"rows\":" << c.rows << ",\"features\":" << c.features
        << ",\"bins\":" << c.bins << ",\"scratch_bytes\":" << c.scratch_bytes
        << ",\"logical_read_bytes\":" << c.logical_read_bytes << ",\"logical_write_bytes\":" << c.logical_write_bytes
        << ",\"operations\":" << c.operations << "}}";
  }
  out << ']';
}
int run(int argc, char** argv) {
  const auto start = Clock::now();
  const Options options = parse(argc, argv);
  if (options.config.objective == ghb::Objective::multiclass_softmax && options.config.classes < 2)
    throw std::invalid_argument("multiclass requires at least two classes");
  if (!options.directory.empty()) {
    if (!std::filesystem::create_directory(options.directory)) throw std::invalid_argument("output directory must be new");
    std::ofstream(options.directory / ".incomplete") << "Training/validation in progress\n";
  }
  const auto training = generate(options, options.rows, options.seed);
  const auto testing = generate(options, options.test_rows, options.seed ^ 0x9e3779b97f4a7c15ULL);
  int device = 0, runtime = 0, driver = 0;
  cudaDeviceProp properties{};
  auto check = [](cudaError_t error) { if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); };
  check(cudaGetDevice(&device)); check(cudaGetDeviceProperties(&properties, device));
  check(cudaRuntimeGetVersion(&runtime)); check(cudaDriverGetVersion(&driver));
  check(cudaFree(nullptr)); // Initialize the CUDA context outside train() timings.
  auto result = ghb::train(training, options.config);
  const auto prediction_start = Clock::now();
  const auto predicted = result.model.predict_gpu(testing);
  const double prediction_ms = elapsed(prediction_start);
  const auto reference = result.model.predict(testing);
  if (reference.size() != predicted.size()) throw std::runtime_error("CPU/GPU prediction shape mismatch");
  double max_prediction_error = 0;
  for (std::size_t i = 0; i < predicted.size(); ++i) {
    if (!std::isfinite(predicted[i]) || !std::isfinite(reference[i])) throw std::runtime_error("nonfinite prediction");
    max_prediction_error = std::max(max_prediction_error, std::abs(predicted[i] - reference[i]));
    if (std::abs(predicted[i] - reference[i]) > 1e-10 * (1 + std::abs(reference[i])))
      throw std::runtime_error("CPU/GPU prediction mismatch");
  }
  std::stringstream serialized(std::ios::in | std::ios::out | std::ios::binary);
  result.model.save(serialized);
  auto restored = ghb::Model::load(serialized);
  if (restored.predict(testing) != reference) throw std::runtime_error("model serialization mismatch");
  ghb::Model baseline_model = result.model; baseline_model.trees.clear();
  const auto baseline = baseline_model.predict(testing);
  // Evaluate stable objective formulas from raw margins. Probability clipping
  // would silently cap classification loss for confident wrong predictions.
  const double baseline_loss = objective_loss(testing, baseline_model.predict(testing, true), result.model.objective, result.model.outputs);
  const double heldout_loss = objective_loss(testing, result.model.predict(testing, true), result.model.objective, result.model.outputs);
  const double end_to_end = elapsed(start);
  for (double value : {result.quantize_ms, result.upload_ms, result.training_ms, result.total_ms, prediction_ms, end_to_end})
    finite_nonnegative(value, "wall_time");
  if (result.training_loss.size() != std::size_t(options.config.rounds) + 1) throw std::runtime_error("training loss coverage mismatch");
  for (double value : result.training_loss) finite_nonnegative(value, "training_loss");
  for (const auto& tuning : result.tuning) {
    finite_nonnegative(tuning.global_ms, "global_ms"); finite_nonnegative(tuning.shared_ms, "shared_ms");
    for (double value : tuning.global_samples_ms) finite_nonnegative(value, "global_samples_ms");
    for (double value : tuning.shared_samples_ms) finite_nonnegative(value, "shared_samples_ms");
  }
  for (const auto& sample : result.samples) if (sample.gpu_ms) finite_nonnegative(*sample.gpu_ms, "sample.gpu_ms");
  std::ostringstream json; json << std::setprecision(17);
  json << "{\"schema_version\":1,\"kind\":\"ghb.training\",\"generator_version\":2,\"objective\":" << quote(objective_name(options.config.objective))
       << ",\"rows\":" << options.rows << ",\"test_rows\":" << options.test_rows << ",\"features\":" << options.features
       << ",\"outputs\":" << result.model.outputs << ",\"seed\":" << options.seed
       << ",\"test_seed\":" << (options.seed ^ 0x9e3779b97f4a7c15ULL)
       << ",\"rounds\":" << options.config.rounds << ",\"max_depth\":" << options.config.max_depth
       << ",\"max_bins\":" << options.config.max_bins << ",\"output_tile_size\":" << options.config.output_tile_size
       << ",\"max_device_bytes\":" << options.config.max_device_bytes << ",\"histogram\":" << quote(policy_name(options.config.histogram))
       << ",\"tree_execution\":" << quote(execution_name(options.config.tree_execution))
       << ",\"quantize_policy\":" << quote(quantize_name(options.config.quantize_policy))
       << ",\"tree_export_batch_requested\":" << options.config.tree_export_batch_size
       << ",\"tree_export_batch_effective\":" << result.tree_export_batch_size
       << ",\"max_histogram_bytes\":" << options.config.max_histogram_bytes << ",\"learning_rate\":" << options.config.learning_rate
       << ",\"l2\":" << options.config.l2 << ",\"min_leaf_rows\":" << options.config.min_leaf_rows
       << ",\"min_child_hessian\":" << options.config.min_child_hessian << ",\"min_gain\":" << options.config.min_gain
       << ",\"max_leaf_value\":" << options.config.max_leaf_value
       << ",\"instrumentation\":" << quote(!options.config.record_stages ? "off" : options.config.nvtx ? "nvtx" : "timing")
       << ",\"gpu\":" << quote(properties.name) << ",\"cuda_runtime\":" << runtime << ",\"cuda_driver\":" << driver
       << ",\"memory\":{\"device_payload_bytes\":" << result.device_bytes << ",\"gradient_hessian_bytes\":" << result.gradient_bytes
       << ",\"histogram_bytes\":" << result.histogram_bytes << ",\"preparation_peak_bytes\":" << result.preparation_peak_bytes
       << ",\"owned_device_peak_bytes\":" << std::max(result.preparation_peak_bytes, result.device_bytes)
       << ",\"pinned_export_bytes\":" << result.pinned_export_bytes << "}"
       << ",\"timing\":{\"quantize_ms\":" << result.quantize_ms << ",\"upload_ms\":" << result.upload_ms
       << ",\"training_ms\":" << result.training_ms << ",\"total_train_ms\":" << result.total_ms
       << ",\"gpu_predict_wall_ms\":" << prediction_ms << ",\"through_validation_wall_ms\":" << end_to_end << "}"
       << ",\"validation\":{\"cpu_gpu_max_abs_error\":" << max_prediction_error << ",\"serialization_equal\":true}"
       << ",\"heldout\":{\"baseline_loss\":" << baseline_loss << ",\"loss\":" << heldout_loss << "}"
       << ",\"trees\":" << result.model.trees.size() << ",\"training_loss\":[";
  for (std::size_t i = 0; i < result.training_loss.size(); ++i) { if (i) json << ','; json << result.training_loss[i]; }
  json << "],\"tuning\":[";
  for (std::size_t i = 0; i < result.tuning.size(); ++i) {
    if (i) json << ',';
    const auto& t = result.tuning[i];
    json << "{\"active_nodes\":" << t.active_nodes << ",\"output\":" << t.output << ",\"global_ms\":" << t.global_ms
         << ",\"shared_ms\":" << t.shared_ms << ",\"selected\":" << quote(policy_name(t.selected))
         << ",\"round\":" << t.round << ",\"depth\":" << t.depth << ",\"output_tile_size\":" << t.output_tile_size
         << ",\"global_samples_ms\":[";
    for (std::size_t j = 0; j < t.global_samples_ms.size(); ++j) { if (j) json << ','; json << t.global_samples_ms[j]; }
    json << "],\"shared_samples_ms\":[";
    for (std::size_t j = 0; j < t.shared_samples_ms.size(); ++j) { if (j) json << ','; json << t.shared_samples_ms[j]; }
    json << "]}";
  }
  json << "],\"samples\":"; samples_json(json, result.samples); json << "}\n";
  if (!options.directory.empty()) {
    save_csv(options.directory, testing, predicted, baseline, result.model);
    std::ofstream model(options.directory / "model.ghb", std::ios::binary);
    model.exceptions(std::ios::badbit | std::ios::failbit); result.model.save(model); model.close();
    std::ofstream report(options.directory / "result.json"); report.exceptions(std::ios::badbit | std::ios::failbit);
    report << json.str(); report.close();
    std::filesystem::remove(options.directory / ".incomplete");
  }
  std::cout << json.str();
  return 0;
}
} // namespace
int main(int argc, char** argv) {
  try { return run(argc, argv); }
  catch (const std::exception& error) { std::cerr << "ghb_bench: " << error.what() << '\n'; return 1; }
}
