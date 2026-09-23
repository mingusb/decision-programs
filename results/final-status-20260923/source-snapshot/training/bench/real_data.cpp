#include "ghb/booster.hpp"
#include "profiling.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/resource.h>

// Experiment-only file adapter. The raw feature matrix is passed to train(),
// whose production feature fitting and binning execute on the GPU.
namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
std::size_t multiply(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::invalid_argument("fixture extent overflow");
  return a * b;
}
template<class T> T number(std::string_view value) {
  T result{};
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc{} || end != value.data() + value.size())
    throw std::invalid_argument("invalid numeric argument");
  return result;
}
struct Input {
  ghb::Dataset data;
  ghb::Objective objective;
  std::uint32_t classes;
};
Input load(const std::filesystem::path& path) {
  static_assert(std::endian::native == std::endian::little);
  static_assert(sizeof(float) == 4 && sizeof(double) == 8);
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open fixture: " + path.string());
  std::array<char, 8> magic{};
  std::array<std::uint32_t, 6> header{};
  in.read(magic.data(), magic.size());
  in.read(reinterpret_cast<char*>(header.data()), sizeof(header));
  if (!in || std::string_view(magic.data(), magic.size()) != "GHBDS001" || header[0] != 1)
    throw std::invalid_argument("invalid fixture header");
  const auto [version, rows, columns, outputs, objective, classes] = header;
  (void)version;
  if (!rows || !columns || !outputs || objective > 2 ||
      (objective == 2 && (outputs != 1 || classes < 2)))
    throw std::invalid_argument("invalid fixture shape or objective");
  const auto feature_cells = multiply(rows, columns), target_cells = multiply(rows, outputs);
  const auto feature_bytes = multiply(feature_cells, sizeof(float));
  const auto target_bytes = multiply(target_cells, sizeof(float));
  if (target_bytes > std::numeric_limits<std::size_t>::max() - 32 ||
      feature_bytes > std::numeric_limits<std::size_t>::max() - target_bytes - 32 ||
      std::filesystem::file_size(path) != 32 + feature_bytes + target_bytes)
    throw std::invalid_argument("fixture length does not match header");
  Input result{{rows, columns, outputs, {}, {}, {}, {}}, static_cast<ghb::Objective>(objective), classes};
  result.data.values.resize(feature_cells);
  result.data.targets.resize(target_cells);
  in.read(reinterpret_cast<char*>(result.data.values.data()), static_cast<std::streamsize>(feature_bytes));
  in.read(reinterpret_cast<char*>(result.data.targets.data()), static_cast<std::streamsize>(target_bytes));
  if (!in) throw std::runtime_error("truncated fixture payload");
  for (float x : result.data.values) if (std::isinf(x)) throw std::invalid_argument("infinite fixture feature");
  for (float y : result.data.targets) {
    if (!std::isfinite(y)) throw std::invalid_argument("nonfinite fixture target");
    if (objective == 1 && y != 0 && y != 1) throw std::invalid_argument("nonbinary fixture target");
    if (objective == 2 && (y < 0 || y >= classes || std::floor(y) != y))
      throw std::invalid_argument("invalid fixture class index");
  }
  return result;
}
struct Options {
  std::filesystem::path train, evaluation, directory;
  ghb::TrainConfig config;
  Options() {
    config.rounds = 25; config.max_depth = 3; config.max_bins = 32;
    config.min_leaf_rows = 1;
    config.record_stages = false; config.nvtx = false;
    config.tree_execution = ghb::TreeExecution::graph;
    config.tree_export_batch_size = 16;
  }
};
Options parse(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view key = argv[i];
    if (key == "--help") {
      std::cout << "ghb_real_bench --train FIXTURE --evaluation FIXTURE --output-dir NEW_DIRECTORY\n"
                   "  [--rounds N] [--depth N] [--bins N] [--learning-rate X] [--l2 X]\n"
                   "  [--optimization-order 2|3|4] [--max-leaf-value X] [--instrumentation off|timing|nvtx]\n"
                   "  [--output-tile N] [--tree-export-batch N] [--max-device-bytes N]\n"
                   "  [--max-histogram-bytes N] [--tree-build per-output|output-batch]\n"
                   "  [--tree-execution stream|graph] [--histogram auto|global|shared]\n"
                   "  [--split-policy block256|warp32|warp-wide]\n";
      std::exit(0);
    }
    if (++i == argc) throw std::invalid_argument("missing argument value");
    const std::string_view value = argv[i];
    auto& c = options.config;
    if (key == "--train") options.train = value;
    else if (key == "--evaluation") options.evaluation = value;
    else if (key == "--output-dir") options.directory = value;
    else if (key == "--rounds") c.rounds = number<std::uint32_t>(value);
    else if (key == "--depth") c.max_depth = number<std::uint32_t>(value);
    else if (key == "--bins") c.max_bins = number<std::uint32_t>(value);
    else if (key == "--learning-rate") c.learning_rate = number<double>(value);
    else if (key == "--l2") c.l2 = number<double>(value);
    else if (key == "--optimization-order") c.optimization_order = number<std::uint32_t>(value);
    else if (key == "--max-leaf-value") c.max_leaf_value = number<double>(value);
    else if (key == "--instrumentation") {
      if (value != "off" && value != "timing" && value != "nvtx") throw std::invalid_argument("invalid instrumentation mode");
      c.record_stages = value != "off"; c.nvtx = value == "nvtx";
    }
    else if (key == "--output-tile") c.output_tile_size = number<std::uint32_t>(value);
    else if (key == "--tree-export-batch") c.tree_export_batch_size = number<std::uint32_t>(value);
    else if (key == "--max-device-bytes") c.max_device_bytes = number<std::size_t>(value);
    else if (key == "--max-histogram-bytes") c.max_histogram_bytes = number<std::size_t>(value);
    else if (key == "--tree-build") {
      if (value == "per-output") c.tree_build = ghb::TreeBuildPolicy::per_output;
      else if (value == "output-batch") c.tree_build = ghb::TreeBuildPolicy::output_batch;
      else throw std::invalid_argument("invalid tree-build policy");
    } else if (key == "--tree-execution") {
      if (value == "stream") c.tree_execution = ghb::TreeExecution::stream;
      else if (value == "graph") c.tree_execution = ghb::TreeExecution::graph;
      else throw std::invalid_argument("invalid execution policy");
    } else if (key == "--split-policy") {
      if (value == "block256") c.split_policy = ghb::SplitPolicy::block256;
      else if (value == "warp32") c.split_policy = ghb::SplitPolicy::warp32;
      else if (value == "warp-wide") c.split_policy = ghb::SplitPolicy::warp_wide;
      else throw std::invalid_argument("invalid split policy");
    } else if (key == "--histogram") {
      if (value == "auto") c.histogram = ghb::HistogramPolicy::autotune;
      else if (value == "global") c.histogram = ghb::HistogramPolicy::global;
      else if (value == "shared") c.histogram = ghb::HistogramPolicy::shared;
      else throw std::invalid_argument("invalid histogram policy");
    } else throw std::invalid_argument("unknown option: " + std::string(key));
  }
  if (options.train.empty() || options.evaluation.empty() || options.directory.empty())
    throw std::invalid_argument("train, evaluation and output-dir are required");
  if (options.config.record_stages && options.config.nvtx && !ghb::instrumentation::Recorder::nvtx_available())
    throw std::invalid_argument("NVTX requested but this build disabled stage annotations");
  return options;
}
} // namespace

int main(int argc, char** argv) try {
  const auto o = parse(argc, argv);
  if (std::filesystem::exists(o.directory)) throw std::invalid_argument("output directory already exists");
  const auto load_start = Clock::now();
  const auto train = load(o.train), evaluation = load(o.evaluation);
  const double load_ms = milliseconds(load_start);
  if (train.data.columns != evaluation.data.columns || train.data.outputs != evaluation.data.outputs ||
      train.objective != evaluation.objective || train.classes != evaluation.classes)
    throw std::invalid_argument("train/evaluation contracts differ");
  auto config = o.config;
  config.objective = train.objective; config.classes = train.classes;
  std::filesystem::create_directories(o.directory);
  const auto context_start = Clock::now();
  check(cudaFree(nullptr)); // Context initialization is reported separately from training.
  int device{}; check(cudaGetDevice(&device));
  cudaDeviceProp properties{}; check(cudaGetDeviceProperties(&properties, device));
  check(cudaDeviceSynchronize());
  const double context_ms = milliseconds(context_start);
  gh::profiling::CaptureRange training_capture("train");
  const auto train_start = Clock::now();
  auto result = ghb::train(train.data, config);
  check(cudaDeviceSynchronize());
  const double train_wall_ms = milliseconds(train_start);
  training_capture.finish();
  gh::profiling::CaptureRange prediction_capture("predict");
  const auto prediction_start = Clock::now();
  const auto predictions = result.model.predict_gpu(evaluation.data);
  check(cudaDeviceSynchronize());
  const double prediction_wall_ms = milliseconds(prediction_start);
  prediction_capture.finish();
  const auto output_count = train.objective == ghb::Objective::multiclass_softmax ? train.classes : train.data.outputs;
  if (predictions.size() != multiply(evaluation.data.rows, output_count) ||
      std::any_of(predictions.begin(), predictions.end(), [](double x) { return !std::isfinite(x); }))
    throw std::runtime_error("invalid GPU prediction result");
  const auto save_start = Clock::now();
  {
    std::ofstream model(o.directory / "model.ghb", std::ios::binary);
    result.model.save(model);
    if (!model) throw std::runtime_error("cannot write model");
    std::ofstream p(o.directory / "predictions.f64", std::ios::binary);
    p.write(reinterpret_cast<const char*>(predictions.data()),
            static_cast<std::streamsize>(multiply(predictions.size(), sizeof(double))));
    if (!p) throw std::runtime_error("cannot write predictions");
  }
  const double serialization_ms = milliseconds(save_start);
  std::size_t nodes{};
  for (const auto& tree : result.model.trees) nodes += tree.nodes.size();
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage)) throw std::runtime_error("cannot read process RSS");
  std::ofstream out(o.directory / "metrics.json");
  out << std::setprecision(17)
      << "{\n  \"implementation\": \"custom\",\n  \"backend\": \"cuda\",\n  \"prediction_backend\": \"cuda\",\n"
      << "  \"device\": \"" << properties.name << "\",\n"
      << "  \"objective\": " << static_cast<unsigned>(train.objective) << ",\n"
      << "  \"train_rows\": " << train.data.rows << ", \"evaluation_rows\": " << evaluation.data.rows
      << ", \"features\": " << train.data.columns << ", \"outputs\": " << output_count << ",\n"
      << "  \"parameters\": {\"rounds\": " << config.rounds << ", \"depth\": " << config.max_depth
      << ", \"bins\": " << config.max_bins << ", \"learning_rate\": " << config.learning_rate
      << ", \"l2\": " << config.l2 << ", \"output_tile\": " << config.output_tile_size
      << ", \"optimization_order\": " << config.optimization_order << ", \"max_leaf_value\": " << config.max_leaf_value
      << ", \"tree_build\": \"" << (config.tree_build == ghb::TreeBuildPolicy::per_output ? "per-output" : "output-batch")
      << "\", \"tree_execution\": \"" << (config.tree_execution == ghb::TreeExecution::graph ? "graph" : "stream")
      << "\", \"histogram\": \"" << (config.histogram == ghb::HistogramPolicy::autotune ? "auto" : config.histogram == ghb::HistogramPolicy::shared ? "shared" : "global")
      << "\", \"split_policy\": \"" << (config.split_policy == ghb::SplitPolicy::warp_wide ? "warp-wide" : config.split_policy == ghb::SplitPolicy::warp32 ? "warp32" : "block256")
      << "\", \"min_leaf_rows\": " << config.min_leaf_rows << ", \"min_child_hessian\": " << config.min_child_hessian
      << ", \"max_device_bytes\": " << config.max_device_bytes << ", \"max_histogram_bytes\": " << config.max_histogram_bytes << "},\n"
      << "  \"timing_ms\": {\"load\": " << load_ms << ", \"context\": " << context_ms << ", \"training_wall\": " << train_wall_ms
      << ", \"training_internal_total\": " << result.total_ms << ", \"quantize\": " << result.quantize_ms
      << ", \"upload\": " << result.upload_ms << ", \"training\": " << result.training_ms
      << ", \"prediction_wall\": " << prediction_wall_ms << ", \"serialization\": " << serialization_ms << "},\n"
      << "  \"memory\": {\"owned_device_bytes\": " << result.device_bytes
      << ", \"preparation_peak_bytes\": " << result.preparation_peak_bytes
      << ", \"histogram_bytes\": " << result.histogram_bytes << ", \"gradient_bytes\": " << result.gradient_bytes
      << ", \"tree_state_bytes\": " << result.tree_state_bytes << ", \"process_peak_rss_bytes\": " << std::size_t(usage.ru_maxrss) * 1024 << "},\n"
      << "  \"tree_batch_size\": " << result.tree_batch_size << ", \"trees\": " << result.model.trees.size()
      << ", \"nodes\": " << nodes << ", \"model_bytes\": " << std::filesystem::file_size(o.directory / "model.ghb") << ",\n"
      << "  \"training_loss_initial\": " << result.training_loss.front()
      << ", \"training_loss_final\": " << result.training_loss.back() << ",\n  \"training_loss\": [";
  for (std::size_t i = 0; i < result.training_loss.size(); ++i) { if (i) out << ','; out << result.training_loss[i]; }
  out << "]\n}\n";
  if (!out) throw std::runtime_error("cannot write metrics");
  std::cout << "training_wall_ms=" << train_wall_ms << " prediction_wall_ms=" << prediction_wall_ms << '\n';
  return 0;
} catch (const std::exception& error) {
  std::cerr << "real-data benchmark: " << error.what() << '\n';
  return 1;
}
