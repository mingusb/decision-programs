#include "ghb/booster.hpp"
#include "ghb/prediction.cuh"
#include "profiling.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
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
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
struct Options {
  std::string model, features, output, policy{"both"};
  unsigned rows{4096}, pairs{15}, warmup{3};
  bool raw{};
};
void check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
std::size_t add(std::size_t a, std::size_t b) {
  if (a > std::numeric_limits<std::size_t>::max() - b) throw std::invalid_argument("size overflow");
  return a + b;
}
std::size_t multiply(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b) throw std::invalid_argument("size overflow");
  return a * b;
}
unsigned number(std::string_view value) {
  unsigned result{};
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
    throw std::invalid_argument("invalid unsigned integer");
  return result;
}
Options options(int argc, char** argv) {
  Options result;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--raw") { result.raw = true; continue; }
    if (i + 1 == argc) throw std::invalid_argument("missing option value: " + argument);
    const std::string value = argv[++i];
    if (argument == "--model") result.model = value;
    else if (argument == "--features-bin") result.features = value;
    else if (argument == "--output") result.output = value;
    else if (argument == "--rows") result.rows = number(value);
    else if (argument == "--pairs") result.pairs = number(value);
    else if (argument == "--warmup") result.warmup = number(value);
    else if (argument == "--policy") result.policy = value;
    else throw std::invalid_argument("unknown option: " + argument);
  }
  if (result.model.empty() || !result.pairs || result.pairs > 10000 || result.warmup > 1000 ||
      (result.policy != "both" && result.policy != "per-tree" && result.policy != "fused-output" &&
       result.policy != "compare-slab" && result.policy != "compare-slab-per-tree" && result.policy != "fused-output-slab" &&
       result.policy != "compare-encoding" && result.policy != "fused-output-final-status"))
    throw std::invalid_argument("require --model; pairs 1..10000, warmup 0..1000, policy both|per-tree|fused-output|compare-slab|compare-slab-per-tree|fused-output-slab|compare-encoding|fused-output-final-status");
  if (!result.output.empty() && std::filesystem::exists(result.output))
    throw std::invalid_argument("output already exists");
  return result;
}
std::string quote(std::string_view value) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result{"\""};
  for (const unsigned char c : value) {
    if (c == '"' || c == '\\') { result += '\\'; result += char(c); }
    else if (c < 32) { result += "\\u00"; result += digits[c >> 4]; result += digits[c & 15]; }
    else result += char(c);
  }
  return result + '"';
}
std::uint64_t fingerprint(const std::vector<float>& values) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (const float value : values) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    for (unsigned byte = 0; byte < 4; ++byte) { hash ^= (bits >> (byte * 8)) & 255U; hash *= 1099511628211ULL; }
  }
  return hash;
}
ghb::Dataset input(const ghb::Model& model, const Options& arguments) {
  ghb::Dataset data;
  data.rows = arguments.rows; data.columns = unsigned(model.features.size()); data.outputs = model.outputs;
  data.values.resize(multiply(data.rows, data.columns));
  for (const auto& feature : model.features) data.feature_types.push_back(feature.type);
  if (!arguments.features.empty()) {
    const auto bytes = multiply(data.values.size(), sizeof(float));
    if (std::filesystem::file_size(arguments.features) != bytes || bytes > std::size_t(std::numeric_limits<std::streamsize>::max()))
      throw std::invalid_argument("features-bin must contain exactly rows*features native FP32 values");
    std::ifstream stream(arguments.features, std::ios::binary);
    if (!stream || (bytes && !stream.read(reinterpret_cast<char*>(data.values.data()), std::streamsize(bytes))))
      throw std::runtime_error("cannot read feature file");
    return data;
  }
  for (unsigned row = 0; row < data.rows; ++row) {
    for (unsigned column = 0; column < data.columns; ++column) {
      std::uint64_t key = (std::uint64_t(row) + 1) * 0x9e3779b97f4a7c15ULL + std::uint64_t(column) * 0xbf58476d1ce4e5b9ULL;
      key ^= key >> 30; key *= 0xbf58476d1ce4e5b9ULL; key ^= key >> 27;
      const auto& feature = model.features[column];
      float value = float(int(key % 1024) - 512) / 32;
      const auto& metadata = feature.type == ghb::FeatureType::numeric ? feature.cuts : feature.categories;
      if (!metadata.empty()) {
        value = metadata[std::size_t(key % metadata.size())];
        if (key % 5 == 0) {
          const float next = std::nextafter(value, std::numeric_limits<float>::infinity());
          if (std::isfinite(next)) value = next;
        }
      }
      if (key % 31 == 0) value = std::numeric_limits<float>::quiet_NaN();
      data.values[std::size_t(row) * data.columns + column] = value;
    }
  }
  return data;
}
void exact(const std::vector<double>& actual, const std::vector<double>& expected) {
  if (actual.size() != expected.size()) throw std::runtime_error("prediction extent changed");
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (std::bit_cast<std::uint64_t>(actual[i]) != std::bit_cast<std::uint64_t>(expected[i]))
      throw std::runtime_error("zero-allowance FP64 comparison failed at " + std::to_string(i));
}
struct Variant {
  ghb::PredictionPolicy prediction{};
  ghb::EncodingPolicy encoding{ghb::EncodingPolicy::per_tile};
  bool operator==(const Variant&) const = default;
};
const char* name(Variant policy) {
  if (policy.encoding == ghb::EncodingPolicy::final_status && policy.prediction == ghb::PredictionPolicy::fused_output)
    return "fused_output_final_status";
  if (policy.encoding != ghb::EncodingPolicy::per_tile) throw std::logic_error("unknown benchmark encoding variant");
  switch (policy.prediction) {
    case ghb::PredictionPolicy::per_tree: return "per_tree";
    case ghb::PredictionPolicy::fused_output: return "fused_output";
    case ghb::PredictionPolicy::fused_output_slab: return "fused_output_slab";
  }
  throw std::logic_error("unknown prediction policy");
}
struct Sample { unsigned pair{}, position{}; Variant policy{}; double milliseconds{}; };
double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values.size() % 2 ? values[values.size() / 2] : .5 * (values[values.size() / 2 - 1] + values[values.size() / 2]);
}
} // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
      std::cout << "Usage: ghb_prediction_bench --model FILE [--features-bin FILE] [--rows N] [--pairs N] [--warmup N]\n"
                   "  [--policy both|per-tree|fused-output|compare-slab|compare-slab-per-tree|fused-output-slab|compare-encoding|fused-output-final-status] [--raw] [--output NEW_JSON]\n"
                   "  both compares per-tree/fused-output (default); compare-slab compares fused-output/fused-output-slab.\n"
                   "  compare-slab-per-tree compares per-tree/fused-output-slab.\n"
                   "  compare-encoding compares fused-output with per-tile/final-status encoding; all older modes use per-tile.\n"
                   "Loads one frozen model; optional input is native row-major FP32. Otherwise creates a deterministic\n"
                   "model-derived inference fixture. Timings include complete predict_gpu setup, transfers and cleanup.\n";
      return 0;
    }
    const auto arguments = options(argc, argv);
    std::ifstream source(arguments.model, std::ios::binary);
    if (!source) throw std::runtime_error("cannot open model");
    const auto model = ghb::Model::load(source);
    const auto data = input(model, arguments);
    check(cudaFree(nullptr), "warm CUDA context");
    int device{}, runtime{}, driver{}; cudaDeviceProp property{};
    check(cudaGetDevice(&device), "get device"); check(cudaGetDeviceProperties(&property, device), "get device properties");
    check(cudaRuntimeGetVersion(&runtime), "runtime version"); check(cudaDriverGetVersion(&driver), "driver version");
    using Policy = ghb::PredictionPolicy;
    using Encoding = ghb::EncodingPolicy;
    constexpr Variant per_tree{Policy::per_tree}, fused{Policy::fused_output}, slab{Policy::fused_output_slab};
    constexpr Variant final_status{Policy::fused_output, Encoding::final_status};
    std::vector<Variant> policies;
    if (arguments.policy == "both") policies = {per_tree, fused};
    else if (arguments.policy == "compare-slab") policies = {fused, slab};
    else if (arguments.policy == "compare-slab-per-tree") policies = {per_tree, slab};
    else if (arguments.policy == "compare-encoding") policies = {fused, final_status};
    else if (arguments.policy == "per-tree") policies = {per_tree};
    else if (arguments.policy == "fused-output") policies = {fused};
    else if (arguments.policy == "fused-output-final-status") policies = {final_status};
    else policies = {slab};
    const bool fused_checked = std::find(policies.begin(), policies.end(), fused) != policies.end();
    const bool slab_checked = std::any_of(policies.begin(), policies.end(), [](auto p) { return p.prediction == Policy::fused_output_slab; });
    const bool final_status_checked = std::find(policies.begin(), policies.end(), final_status) != policies.end();
    // Preserve the frozen reference for both forms before recording samples.
    std::vector<double> expected;
    for (const bool raw : {true, false}) {
      auto reference = model.predict_gpu(data, raw, Policy::per_tree);
      for (const auto policy : policies)
        if (policy != per_tree) exact(model.predict_gpu(data, raw, policy.prediction, policy.encoding), reference);
      if (raw == arguments.raw) expected = std::move(reference);
    }
    for (unsigned warm = 0; warm < arguments.warmup; ++warm)
      for (const auto policy : policies) exact(model.predict_gpu(data, arguments.raw, policy.prediction, policy.encoding), expected);
    std::vector<Sample> samples;
    // Optional whole-operation profiling excludes reference prechecks and
    // warmups. With capture disabled this helper makes no CUDA/NVTX calls.
    gh::profiling::CaptureRange capture(arguments.policy == "both" ? "predict_paired_samples" :
        arguments.policy == "compare-slab" ? "predict_slab_paired_samples" :
        arguments.policy == "compare-slab-per-tree" ? "predict_slab_per_tree_paired_samples" :
        arguments.policy == "compare-encoding" ? "predict_encoding_paired_samples" :
        arguments.policy == "fused-output-final-status" ? "predict_fused_output_final_status_samples" :
        arguments.policy == "per-tree" ? "predict_per_tree_samples" :
        arguments.policy == "fused-output" ? "predict_fused_output_samples" : "predict_fused_output_slab_samples");
    for (unsigned pair = 0; pair < arguments.pairs; ++pair) {
      if (pair && policies.size() == 2) std::reverse(policies.begin(), policies.end());
      for (unsigned position = 0; position < policies.size(); ++position) {
        const auto policy = policies[position];
        const auto start = Clock::now();
        const auto actual = model.predict_gpu(data, arguments.raw, policy.prediction, policy.encoding);
        const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        // Retain each observation in the process log even if a later exactness
        // gate or CUDA operation aborts before the final JSON is written.
        std::cerr << std::setprecision(17) << "{\"pair\":" << pair << ",\"position\":" << position
                  << ",\"policy\":" << quote(name(policy)) << ",\"milliseconds\":" << elapsed << "}\n";
        exact(actual, expected); // Comparison is outside the measurement.
        samples.push_back({pair, position, policy, elapsed});
      }
    }
    capture.finish();
    std::size_t nodes{}, maximum_nodes{};
    for (const auto& tree : model.trees) { nodes = add(nodes, tree.nodes.size()); maximum_nodes = std::max(maximum_nodes, tree.nodes.size()); }
    const auto node_bytes = multiply(nodes, sizeof(ghb::Node));
    const auto descriptor_bytes = multiply(model.trees.size(), sizeof(ghb::gpu::PredictionTree));
    const auto offset_bytes = multiply(add(model.outputs, 1), sizeof(std::uint64_t));
    const auto prediction_bytes = multiply(multiply(data.rows, model.outputs), sizeof(double));
    const auto base_bytes = multiply(model.outputs, sizeof(double));
    const auto packed_bytes = add(node_bytes, add(descriptor_bytes, offset_bytes));
    const auto packing_auxiliary_bytes = add(multiply(model.trees.size(), sizeof(std::size_t)), offset_bytes);
    constexpr std::size_t slab_alignment = 256;
    std::size_t slab_bytes{};
    auto slab_region = [&](std::size_t bytes) {
      if (!bytes) return std::size_t{};
      const auto offset = add(slab_bytes, (slab_alignment - slab_bytes % slab_alignment) % slab_alignment);
      slab_bytes = add(offset, bytes);
      return offset;
    };
    const auto slab_base_offset = slab_region(base_bytes);
    const auto slab_node_offset = slab_region(node_bytes);
    const auto slab_descriptor_offset = slab_region(descriptor_bytes);
    const auto slab_offset_offset = slab_region(offset_bytes);
    const auto slab_host_bytes = multiply(add(slab_bytes / slab_alignment, slab_bytes % slab_alignment != 0), slab_alignment);
    std::ofstream destination;
    if (!arguments.output.empty()) {
      destination.open(arguments.output);
      if (!destination) throw std::runtime_error("cannot create output JSON");
    }
    std::ostream& out = arguments.output.empty() ? std::cout : destination;
    out << std::setprecision(17)
        << "{\"schema\":\"ghb.prediction_benchmark.v1\",\"model\":" << quote(arguments.model)
        << ",\"features_file\":" << quote(arguments.features)
        << ",\"input_kind\":" << quote(arguments.features.empty() ? "deterministic_model_derived_v1" : "native_row_major_fp32")
        << ",\"feature_fnv1a64\":" << quote(std::to_string(fingerprint(data.values)))
        << ",\"rows\":" << data.rows << ",\"features\":" << data.columns << ",\"outputs\":" << model.outputs
        << ",\"trees\":" << model.trees.size() << ",\"nodes\":" << nodes << ",\"maximum_tree_nodes\":" << maximum_nodes
        << ",\"raw\":" << (arguments.raw ? "true" : "false") << ",\"gpu\":" << quote(property.name)
        << ",\"compute_major\":" << property.major << ",\"compute_minor\":" << property.minor
        << ",\"cuda_runtime\":" << runtime << ",\"cuda_driver\":" << driver
        << ",\"warmup_per_policy\":" << arguments.warmup << ",\"pairs\":" << arguments.pairs
        << ",\"requested_policy\":" << quote(arguments.policy) << ",\"reference_policy\":\"per_tree\""
        << ",\"bitwise_frozen_reference_passed\":true"
        << ",\"fused_reference_checked\":" << (fused_checked ? "true" : "false")
        << ",\"slab_reference_checked\":" << (slab_checked ? "true" : "false")
        << ",\"final_status_reference_checked\":" << (final_status_checked ? "true" : "false")
        << ",\"reference_encoding_policy\":\"per_tile\""
        << ",\"timing_boundary\":\"complete predict_gpu including validation, host packing, allocation, quantization, uploads, kernels, download and device cleanup; excludes context warmup, model/input loading and result validation\""
        << ",\"payload_excluding_quantizer\":{\"applies_to_nonempty_rows\":true,\"prediction_bytes\":" << prediction_bytes
        << ",\"base_bytes\":" << base_bytes << ",\"packed_nodes_bytes\":" << node_bytes
        << ",\"packed_descriptors_bytes\":" << descriptor_bytes << ",\"packed_offsets_bytes\":" << offset_bytes
        << ",\"fused_host_packing_payload_bytes\":" << packed_bytes
        << ",\"fused_host_packing_auxiliary_bytes\":" << packing_auxiliary_bytes
        << ",\"fused_device_payload_bytes\":" << add(add(prediction_bytes, base_bytes), packed_bytes)
        << ",\"per_tree_device_payload_bytes\":" << add(add(prediction_bytes, base_bytes), multiply(maximum_nodes, sizeof(ghb::Node)))
        << ",\"per_tree_model_upload_bytes\":" << add(base_bytes, node_bytes)
        << ",\"fused_model_upload_bytes\":" << add(base_bytes, packed_bytes)
        << ",\"slab_alignment_bytes\":" << slab_alignment
        << ",\"fused_slab_model_padding_bytes\":" << slab_bytes - add(base_bytes, packed_bytes)
        << ",\"fused_slab_host_packing_payload_bytes\":" << slab_bytes
        << ",\"fused_slab_host_allocation_bytes\":" << slab_host_bytes
        << ",\"fused_slab_host_packing_auxiliary_bytes\":" << packing_auxiliary_bytes
        << ",\"fused_slab_device_payload_bytes\":" << add(prediction_bytes, slab_bytes)
        << ",\"fused_slab_model_upload_bytes\":" << slab_bytes
        << ",\"fused_slab_region_offsets\":{\"base\":" << slab_base_offset << ",\"nodes\":" << slab_node_offset
        << ",\"descriptors\":" << slab_descriptor_offset << ",\"offsets\":" << slab_offset_offset << "}}";
    const auto final_length_bytes = data.rows ? multiply(data.columns, sizeof(unsigned)) : 0;
    const auto per_tile_length_bound = data.rows ? multiply(std::min(data.columns, 32U), sizeof(unsigned)) : 0;
    out << ",\"encoding_host_lengths\":{\"final_status_payload_bytes\":" << final_length_bytes
        << ",\"per_tile_payload_upper_bound_bytes\":" << per_tile_length_bound
        << ",\"additional_payload_lower_bound_bytes\":" << final_length_bytes - per_tile_length_bound
        << ",\"scope\":\"Length-vector payload only, excluding allocator overhead. Actual per-tile width depends on the available memory budget and is not exported by Model::predict_gpu; zero-row calls allocate neither vector.\"}";
    out << ",\"samples\":[";
    for (std::size_t index = 0; index < samples.size(); ++index) {
      const auto& sample = samples[index];
      if (index) out << ',';
      out << "{\"pair\":" << sample.pair << ",\"position\":" << sample.position << ",\"policy\":" << quote(name(sample.policy))
          << ",\"milliseconds\":" << sample.milliseconds << '}';
    }
    out << "],\"median_ms\":{";
    bool separator = false;
    for (const auto policy : {per_tree, fused, slab, final_status}) {
      std::vector<double> values;
      for (const auto& sample : samples) if (sample.policy == policy) values.push_back(sample.milliseconds);
      if (values.empty()) continue;
      if (separator) out << ',';
      separator = true; out << quote(name(policy)) << ':' << median(std::move(values));
    }
    out << "}}\n";
    if (!out) throw std::runtime_error("cannot write benchmark JSON");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "prediction benchmark failed: " << error.what() << '\n'; return 1;
  }
}
