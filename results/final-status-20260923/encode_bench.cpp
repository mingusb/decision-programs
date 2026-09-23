// Explicit CPU validation and stage benchmark, never production preprocessing.
// Prepared before execution; see EXPERIMENT_RUNBOOK.md for the timing contract.
#include "ghb/quantize.hpp"
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
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
struct Options { std::string model, output; unsigned rows{257}, width{1}, pairs{15}, warmup{3}; };
struct Sample { unsigned pair, position; ghb::EncodingPolicy policy; double milliseconds; };
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void check(cudaError_t error) { if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); }
unsigned number(std::string_view text) {
  unsigned value{};
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  require(result.ec == std::errc{} && result.ptr == text.data() + text.size(), "invalid unsigned number");
  return value;
}
Options options(int argc, char** argv) {
  Options result;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    require(i + 1 < argc, "missing option value");
    const std::string value = argv[++i];
    if (argument == "--model") result.model = value;
    else if (argument == "--output") result.output = value;
    else if (argument == "--rows") result.rows = number(value);
    else if (argument == "--tile-width") result.width = number(value);
    else if (argument == "--pairs") result.pairs = number(value);
    else if (argument == "--warmup") result.warmup = number(value);
    else throw std::invalid_argument("unknown option: " + argument);
  }
  require(!result.model.empty() && !result.output.empty(), "model and output required");
  require(result.rows && result.rows <= 65536 && result.width && result.width <= 32 &&
          result.pairs && result.pairs <= 10000 && result.warmup <= 1000, "invalid bounded benchmark extent");
  require(!std::filesystem::exists(result.output), "output already exists");
  return result;
}
std::string quote(std::string_view text) {
  std::string result{"\""};
  for (const unsigned char value : text) {
    require(value >= 32, "control character in JSON string");
    if (value == '"' || value == '\\') result += '\\';
    result += char(value);
  }
  return result + '"';
}
const char* name(ghb::EncodingPolicy policy) {
  return policy == ghb::EncodingPolicy::per_tile ? "per_tile" : "final_status";
}
struct Stream {
  cudaStream_t value{};
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
};
// Same deterministic fixture contract as training/bench/prediction.cpp.
ghb::Dataset fixture(const ghb::Model& model, unsigned rows) {
  ghb::Dataset data;
  data.rows = rows; data.columns = unsigned(model.features.size()); data.outputs = model.outputs;
  require(data.columns <= 4096 && std::size_t(rows) * data.columns <= (16ULL << 20), "fixture exceeds validation bound");
  data.values.resize(std::size_t(rows) * data.columns);
  for (const auto& feature : model.features) data.feature_types.push_back(feature.type);
  for (unsigned row = 0; row < rows; ++row) for (unsigned column = 0; column < data.columns; ++column) {
    std::uint64_t key = (std::uint64_t(row) + 1) * 0x9e3779b97f4a7c15ULL + std::uint64_t(column) * 0xbf58476d1ce4e5b9ULL;
    key ^= key >> 30; key *= 0xbf58476d1ce4e5b9ULL; key ^= key >> 27;
    const auto& feature = model.features[column];
    const auto& metadata = feature.type == ghb::FeatureType::numeric ? feature.cuts : feature.categories;
    float value = float(int(key % 1024) - 512) / 32;
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
  return data;
}
std::uint64_t fingerprint(const ghb::Dataset& data) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (const float value : data.values) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    for (unsigned byte = 0; byte < 4; ++byte) { hash ^= (bits >> (byte * 8)) & 255U; hash *= 1099511628211ULL; }
  }
  return hash;
}
std::size_t resident_bytes(const ghb::Dataset& data) {
  const auto bins = std::size_t(data.rows) * data.columns * sizeof(std::uint16_t);
  return ((bins + 3) & ~std::size_t(3)) + (std::size_t(data.columns) + 1) * sizeof(std::uint32_t) + data.columns * sizeof(ghb::FeatureType);
}
std::size_t metadata_stride(const std::vector<ghb::Feature>& features) {
  std::size_t maximum = 1;
  for (const auto& feature : features)
    maximum = std::max(maximum, feature.type == ghb::FeatureType::numeric ? feature.cuts.size() : feature.categories.size());
  return maximum;
}
std::size_t peak_bytes(const ghb::Dataset& data, const std::vector<ghb::Feature>& features, unsigned width) {
  return resident_bytes(data) + sizeof(unsigned) + std::size_t(width) *
    (std::size_t(data.rows) * sizeof(float) + sizeof(unsigned) + metadata_stride(features) * sizeof(float));
}
// Explicit independent CPU lower_bound reference; only used outside timing.
std::vector<std::uint16_t> reference(const ghb::Dataset& data, const std::vector<ghb::Feature>& features) {
  std::vector<std::uint16_t> result(data.values.size());
  for (unsigned feature = 0; feature < data.columns; ++feature) {
    const auto& meta = features[feature];
    const bool numeric = meta.type == ghb::FeatureType::numeric;
    const auto& values = numeric ? meta.cuts : meta.categories;
    for (unsigned row = 0; row < data.rows; ++row) {
      const float value = data.values[std::size_t(row) * data.columns + feature];
      const auto found = std::lower_bound(values.begin(), values.end(), value);
      const auto bin = std::isnan(value) ? 0 : numeric ? 1 + found - values.begin() :
        found == values.end() || *found != value ? 0 : 1 + found - values.begin();
      result[std::size_t(feature) * data.rows + row] = std::uint16_t(bin);
    }
  }
  return result;
}
template<class T> std::vector<T> download(const T* source, std::size_t count) {
  std::vector<T> result(count);
  if (count) check(cudaMemcpy(result.data(), source, count * sizeof(T), cudaMemcpyDeviceToHost));
  return result;
}
void same_floats(const std::vector<float>& actual, const std::vector<float>& expected) {
  require(actual.size() == expected.size(), "metadata extent differs");
  for (std::size_t i = 0; i < actual.size(); ++i)
    require(std::bit_cast<std::uint32_t>(actual[i]) == std::bit_cast<std::uint32_t>(expected[i]), "metadata bits differ");
}
void verify(const ghb::QuantizedData& actual, const ghb::Dataset& data, const std::vector<ghb::Feature>& features,
            const std::vector<std::uint16_t>& expected, std::size_t peak) {
  require(actual.view.rows == data.rows && actual.view.columns == data.columns, "view shape differs");
  require(actual.resident_bytes == resident_bytes(data) && actual.peak_bytes == peak, "owned device payload differs");
  require(actual.features.size() == features.size(), "feature count differs");
  require(download(actual.view.bins, expected.size()) == expected, "CPU-reference bin exactness failed");
  const auto offsets = download(actual.view.offsets, std::size_t(data.columns) + 1);
  const auto types = download(actual.view.types, data.columns);
  require(offsets.front() == 0, "invalid first histogram offset");
  unsigned maximum{};
  for (unsigned f = 0; f < data.columns; ++f) {
    require(actual.features[f].type == features[f].type && types[f] == features[f].type, "feature type differs");
    same_floats(actual.features[f].cuts, features[f].cuts);
    same_floats(actual.features[f].categories, features[f].categories);
    const auto bins = unsigned(features[f].type == ghb::FeatureType::numeric ? features[f].cuts.size() + 2 : features[f].categories.size() + 1);
    require(offsets[f + 1] == offsets[f] + bins, "histogram offsets differ");
    maximum = std::max(maximum, bins);
  }
  require(actual.view.total_bins == offsets.back() && actual.view.max_feature_bins == maximum, "view bounds differ");
}
double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values.size() % 2 ? values[values.size() / 2] : .5 * (values[values.size() / 2 - 1] + values[values.size() / 2]);
}
} // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
      std::cout << "Usage: encode_bench --model FILE --output NEW_JSON [--rows N] [--tile-width 1..32] [--pairs N] [--warmup N]\n";
      return 0;
    }
    const auto args = options(argc, argv);
    std::ifstream stream(args.model, std::ios::binary);
    require(bool(stream), "cannot open model");
    const auto model = ghb::Model::load(stream);
    const auto data = fixture(model, args.rows);
    require(args.width <= data.columns, "requested tile wider than input");
    const auto expected = reference(data, model.features);
    const auto limit = peak_bytes(data, model.features, args.width);
    check(cudaFree(nullptr));
    int device{}, runtime{}, driver{}; cudaDeviceProp property{};
    check(cudaGetDevice(&device)); check(cudaGetDeviceProperties(&property, device));
    check(cudaRuntimeGetVersion(&runtime)); check(cudaDriverGetVersion(&driver));
    const Stream work;
    const auto verify_call = [&](ghb::EncodingPolicy policy) {
      const auto actual = ghb::encode_quantize(data, model.features, limit, work.value, policy);
      verify(actual, data, model.features, expected, limit);
    };
    for (const auto policy : {ghb::EncodingPolicy::per_tile, ghb::EncodingPolicy::final_status}) {
      bool rejected{};
      try { (void)ghb::encode_quantize(data, model.features, peak_bytes(data, model.features, 1) - 1, work.value, policy); }
      catch (const std::invalid_argument&) { rejected = true; }
      require(rejected, "one byte below minimum budget was accepted");
      verify_call(policy);
    }
    for (unsigned warm = 0; warm < args.warmup; ++warm)
      for (const auto policy : {ghb::EncodingPolicy::per_tile, ghb::EncodingPolicy::final_status}) verify_call(policy);
    std::vector<Sample> samples;
    for (unsigned pair = 0; pair < args.pairs; ++pair) for (unsigned position = 0; position < 2; ++position) {
      const auto policy = (pair + position) % 2 ? ghb::EncodingPolicy::final_status : ghb::EncodingPolicy::per_tile;
      const auto start = Clock::now();
      const auto actual = ghb::encode_quantize(data, model.features, limit, work.value, policy);
      const double milliseconds = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
      std::cerr << std::setprecision(17) << "{\"pair\":" << pair << ",\"position\":" << position
                << ",\"policy\":" << quote(name(policy)) << ",\"milliseconds\":" << milliseconds << "}\n";
      verify(actual, data, model.features, expected, limit);
      samples.push_back({pair, position, policy, milliseconds});
      // Destruction follows validation, outside stage timing, for both policies.
    }
    std::ofstream out(args.output, std::ios::out);
    require(bool(out), "cannot write output");
    out << std::setprecision(17) << "{\"schema\":\"ghb.encode_stage.v1\",\"model\":" << quote(args.model)
        << ",\"rows\":" << data.rows << ",\"features\":" << data.columns << ",\"tile_width\":" << args.width
        << ",\"encoding_tiles\":" << (data.columns + args.width - 1) / args.width
        << ",\"metadata_stride\":" << metadata_stride(model.features)
        << ",\"resident_bytes\":" << resident_bytes(data) << ",\"peak_bytes\":" << limit << ",\"memory_limit\":" << limit
        << ",\"per_tile_host_lengths_bytes\":" << args.width * sizeof(unsigned)
        << ",\"final_status_host_lengths_bytes\":" << data.columns * sizeof(unsigned)
        << ",\"input_kind\":\"deterministic_model_derived_v1\",\"feature_fnv1a64\":" << quote(std::to_string(fingerprint(data)))
        << ",\"device\":" << quote(property.name) << ",\"cuda_runtime\":" << runtime << ",\"cuda_driver\":" << driver
        << ",\"pairs\":" << args.pairs << ",\"warmup_per_policy\":" << args.warmup
        << ",\"bitwise_cpu_reference_passed\":true,\"one_byte_below_minimum_rejected\":true"
        << ",\"timing_boundary\":\"encode_quantize entry through synchronous return, including validation, host metadata copies, device allocation, uploads, encoding and scratch cleanup; owned resident result destruction, explicit CPU reference, downloads for verification, context and stream setup are excluded; separate stage evidence, not complete prediction\",\"samples\":[";
    for (std::size_t i = 0; i < samples.size(); ++i) {
      const auto& sample = samples[i];
      if (i) out << ',';
      out << "{\"pair\":" << sample.pair << ",\"position\":" << sample.position << ",\"policy\":" << quote(name(sample.policy))
          << ",\"milliseconds\":" << sample.milliseconds << '}';
    }
    out << "],\"median_ms\":{";
    bool comma{};
    for (const auto policy : {ghb::EncodingPolicy::per_tile, ghb::EncodingPolicy::final_status}) {
      std::vector<double> values;
      for (const auto& sample : samples) if (sample.policy == policy) values.push_back(sample.milliseconds);
      if (comma) out << ',';
      comma = true; out << quote(name(policy)) << ':' << median(std::move(values));
    }
    out << "}}\n";
    require(bool(out), "cannot finish output");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "encode stage benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
