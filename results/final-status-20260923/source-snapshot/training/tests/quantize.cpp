#include "ghb/quantize.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
std::size_t checks{};
constexpr std::size_t budget = 512ULL << 20;
constexpr auto nan = std::numeric_limits<float>::quiet_NaN();
void require(bool condition, const char* message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}
void cuda_check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template<class Function> void rejected(Function&& function, const char* message) {
  bool failed = false;
  try { function(); } catch (const std::invalid_argument&) { failed = true; }
  require(failed, message);
}
struct Stream {
  cudaStream_t value{};
  Stream() { cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
};
template<class T> std::vector<T> download(const T* pointer, std::size_t size) {
  std::vector<T> values(size);
  if (size) cuda_check(cudaMemcpy(values.data(), pointer, size * sizeof(T), cudaMemcpyDeviceToHost));
  return values;
}

// Deliberately CPU-only validation reference: comparison sort/unique and
// independent numeric/category lookups, never production preprocessing.
std::vector<ghb::Feature> reference(const ghb::Dataset& data, unsigned max_bins) {
  std::vector<ghb::Feature> features(data.columns);
  for (unsigned f = 0; f < data.columns; ++f) {
    auto& feature = features[f];
    feature.type = data.feature_types.empty() ? ghb::FeatureType::numeric : data.feature_types[f];
    std::vector<float> unique;
    for (unsigned r = 0; r < data.rows; ++r) {
      const float value = data.values[std::size_t(r) * data.columns + f];
      if (!std::isnan(value)) unique.push_back(value == 0.f ? 0.f : value);
    }
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    if (feature.type == ghb::FeatureType::categorical) {
      if (unique.size() >= max_bins) throw std::invalid_argument("reference category overflow");
      feature.categories = std::move(unique);
    } else if (unique.size() > 1) {
      const auto intervals = std::min<std::size_t>(unique.size(), max_bins - 1);
      for (std::size_t j = 1; j < intervals; ++j) feature.cuts.push_back(unique[j * unique.size() / intervals - 1]);
    }
  }
  return features;
}
void same_floats(const std::vector<float>& actual, const std::vector<float>& expected) {
  require(actual.size() == expected.size(), "feature metadata length mismatch");
  for (std::size_t i = 0; i < actual.size(); ++i)
    require(std::bit_cast<std::uint32_t>(actual[i]) == std::bit_cast<std::uint32_t>(expected[i]), "feature metadata bits mismatch");
}
unsigned bins(const ghb::Feature& feature) {
  return unsigned(feature.type == ghb::FeatureType::numeric ? feature.cuts.size() + 2 : feature.categories.size() + 1);
}
std::uint16_t encode(const ghb::Feature& feature, float value) {
  if (std::isnan(value)) return 0;
  if (feature.type == ghb::FeatureType::categorical) {
    const auto found = std::lower_bound(feature.categories.begin(), feature.categories.end(), value);
    return found == feature.categories.end() || *found != value ? 0 : std::uint16_t(1 + found - feature.categories.begin());
  }
  return std::uint16_t(1 + std::lower_bound(feature.cuts.begin(), feature.cuts.end(), value) - feature.cuts.begin());
}
std::size_t resident_size(const ghb::Dataset& data) {
  const auto raw = std::size_t(data.rows) * data.columns * sizeof(std::uint16_t);
  return ((raw + 3) & ~std::size_t(3)) + (std::size_t(data.columns) + 1) * sizeof(std::uint32_t) + data.columns * sizeof(ghb::FeatureType);
}
void verify(const ghb::QuantizedData& actual, const ghb::Dataset& data,
            const std::vector<ghb::Feature>& expected, std::size_t limit = budget) {
  require(actual.view.rows == data.rows && actual.view.columns == data.columns, "GPU view shape mismatch");
  require(actual.features.size() == expected.size(), "exported feature count mismatch");
  require(actual.resident_bytes == resident_size(data), "resident allocation accounting mismatch");
  require(actual.peak_bytes >= actual.resident_bytes && actual.peak_bytes <= limit, "peak allocation outside memory budget");
  const auto offsets = download(actual.view.offsets, std::size_t(data.columns) + 1);
  const auto types = download(actual.view.types, data.columns);
  const auto packed = download(actual.view.bins, std::size_t(data.rows) * data.columns);
  require(offsets.front() == 0, "offsets must start at zero");
  unsigned maximum = 0;
  for (unsigned f = 0; f < data.columns; ++f) {
    require(actual.features[f].type == expected[f].type && types[f] == expected[f].type, "feature type mismatch");
    same_floats(actual.features[f].cuts, expected[f].cuts);
    same_floats(actual.features[f].categories, expected[f].categories);
    require(offsets[f + 1] == offsets[f] + bins(expected[f]), "feature offset mismatch");
    maximum = std::max(maximum, bins(expected[f]));
    for (unsigned r = 0; r < data.rows; ++r)
      require(packed[std::size_t(f) * data.rows + r] == encode(expected[f], data.values[std::size_t(r) * data.columns + f]), "packed feature bin mismatch");
  }
  require(actual.view.total_bins == offsets.back() && actual.view.max_feature_bins == maximum, "view bin bounds mismatch");
}

// D2 contract: both explicit policies must preserve the original four-argument
// encoder's device payload/layout, metadata bits and independently checked bins.
std::size_t verify_encodings(const ghb::Dataset& data, const std::vector<ghb::Feature>& features,
                            std::size_t limit, cudaStream_t stream) {
  const auto legacy = ghb::encode_quantize(data, features, limit, stream);
  verify(legacy, data, features, limit);
  for (const auto policy : {ghb::EncodingPolicy::per_tile, ghb::EncodingPolicy::final_status}) {
    const auto actual = ghb::encode_quantize(data, features, limit, stream, policy);
    verify(actual, data, features, limit);
    require(actual.resident_bytes == legacy.resident_bytes && actual.peak_bytes == legacy.peak_bytes,
            "encoding policy changed device allocation accounting");
    require(download(actual.view.bins, std::size_t(data.rows) * data.columns) ==
            download(legacy.view.bins, std::size_t(data.rows) * data.columns), "encoding policy changed bin bits");
  }
  return legacy.peak_bytes;
}

ghb::Dataset fixture(unsigned rows, unsigned columns) {
  ghb::Dataset data;
  data.rows = rows; data.columns = columns;
  data.values.resize(std::size_t(rows) * columns);
  data.feature_types.resize(columns);
  for (unsigned f = 0; f < columns; ++f) {
    data.feature_types[f] = f % 7 == 1 || f % 7 == 5 ? ghb::FeatureType::categorical : ghb::FeatureType::numeric;
    for (unsigned r = 0; r < rows; ++r) {
      float value;
      switch (f % 7) {
        case 0: value = float(int((std::uint64_t(r) * 7919 + 17) % 1031) - 515) / 8; break;
        case 1: value = float(int(r % 17) - 8) * 3; break;
        case 2: value = r % 2 ? -0.f : 0.f; break;
        case 3: value = nan; break;
        case 4: value = r + 19 < rows ? -3.f : float(r % 19); break;
        case 5: value = nan; break;
        default: value = 2.5f; break;
      }
      if (r % 29 == 0) value = nan;
      data.values[std::size_t(r) * columns + f] = value;
    }
  }
  return data;
}

void tails_and_semantics(cudaStream_t stream) {
  for (unsigned rows : {1u, 31u, 32u, 33u, 255u, 256u, 257u, 1023u, 1024u, 1025u, 4099u}) {
    const auto data = fixture(rows, 7);
    const auto expected = reference(data, 257);
    for (auto policy : {ghb::QuantizePolicy::radix8, ghb::QuantizePolicy::radix4}) {
      const auto actual = ghb::fit_quantize(data, 257, budget, stream, policy);
      verify(actual, data, expected);
      verify_encodings(data, actual.features, budget, stream);
    }
  }
  ghb::Dataset extremes;
  extremes.rows = 12; extremes.columns = 1;
  extremes.values = {nan, -0.f, 0.f, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
    -std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::denorm_min(),
    -1.f, 1.f, nan, -0.f, 0.f};
  for (auto policy : {ghb::QuantizePolicy::radix8, ghb::QuantizePolicy::radix4}) {
    for (unsigned maximum : {2u, 3u, 256u, 65536u}) {
      const auto actual = ghb::fit_quantize(extremes, maximum, budget, stream, policy);
      verify(actual, extremes, reference(extremes, maximum));
      verify_encodings(extremes, actual.features, budget, stream);
    }
  }
  // Frequent zero must not change uniform DISTINCT ranks or category IDs.
  ghb::Dataset skew; skew.rows = 10003; skew.columns = 1;
  skew.values.assign(skew.rows, 0.f);
  skew.values[10000] = 10; skew.values[10001] = 20; skew.values[10002] = 30;
  auto fitted = ghb::fit_quantize(skew, 3, budget, stream);
  require(fitted.features[0].cuts == std::vector<float>{10}, "distinct quantiles were replaced by row quantiles");
  verify(fitted, skew, reference(skew, 3));
}

void cardinality_and_hierarchy(cudaStream_t stream) {
  for (unsigned count : {257u, 65535u, 65536u}) {
    ghb::Dataset data;
    data.rows = count; data.columns = 1; data.values.resize(count);
    // Reverse order exercises all float ordering transformations and tails.
    for (unsigned r = 0; r < count; ++r) data.values[r] = float(int(count - r) - 32768);
    for (auto policy : {ghb::QuantizePolicy::radix8, ghb::QuantizePolicy::radix4}) {
      auto fitted = ghb::fit_quantize(data, 65536, budget, stream, policy);
      verify(fitted, data, reference(data, 65536));
      if (policy == ghb::QuantizePolicy::radix8) verify_encodings(data, fitted.features, budget, stream);
      data.feature_types = {ghb::FeatureType::categorical};
      if (count < 65536) {
        fitted = ghb::fit_quantize(data, 65536, budget, stream, policy);
        verify(fitted, data, reference(data, 65536));
        if (policy == ghb::QuantizePolicy::radix8) verify_encodings(data, fitted.features, budget, stream);
      } else rejected([&] { ghb::fit_quantize(data, 65536, budget, stream, policy); }, "65536 categories must overflow missing-reserved uint16 bins");
      data.feature_types.clear();
    }
  }
  // 1,025 row blocks exercise >512 unique-count entries, and >512 parent
  // scan entries in radix8's digit-major histogram hierarchy.
  ghb::Dataset large; large.rows = 1048593; large.columns = 1;
  large.values.resize(large.rows);
  for (unsigned r = 0; r < large.rows; ++r)
    large.values[r] = r % 997 == 0 ? nan : float(int((std::uint64_t(r) * 104729) % 65537) - 32768);
  const auto expected = reference(large, 257);
  for (auto policy : {ghb::QuantizePolicy::radix8, ghb::QuantizePolicy::radix4}) {
    const auto fitted = ghb::fit_quantize(large, 257, budget, stream, policy);
    verify(fitted, large, expected);
  }
}

void tiling_budgets_and_moves(cudaStream_t stream) {
  const auto data = fixture(1051, 35);
  const auto expected = reference(data, 257);
  for (auto policy : {ghb::QuantizePolicy::radix8, ghb::QuantizePolicy::radix4}) {
    auto full = ghb::fit_quantize(data, 257, budget, stream, policy);
    verify(full, data, expected);
    auto single = fixture(data.rows, 1);
    const auto probe = ghb::fit_quantize(single, 257, budget, stream, policy);
    const auto exact_budget = resident_size(data) + probe.peak_bytes - probe.resident_bytes;
    auto tiled = ghb::fit_quantize(data, 257, exact_budget, stream, policy);
    verify(tiled, data, expected, exact_budget);
    require(tiled.peak_bytes == exact_budget, "one-feature budget accounting mismatch");
    rejected([&] { ghb::fit_quantize(data, 257, exact_budget - 1, stream, policy); }, "one byte below minimum fit scratch must fail");
    ghb::QuantizedData moved(std::move(tiled));
    require(!tiled.view.bins && tiled.resident_bytes == 0 && tiled.peak_bytes == 0, "move construction retained source ownership");
    verify(moved, data, expected, exact_budget);
    full = std::move(moved);
    require(!moved.view.bins && moved.resident_bytes == 0, "move assignment retained source ownership");
    verify(full, data, expected, exact_budget);
    auto* self = &full;
    full = std::move(*self);
    verify(full, data, expected, exact_budget);

    auto held_out = data;
    for (unsigned r = 0; r < data.rows; ++r) for (unsigned f = 0; f < data.columns; ++f)
      held_out.values[std::size_t(r) * data.columns + f] = r % 9 == 0 ? nan : expected[f].type == ghb::FeatureType::categorical
        ? (r % 2 ? 123456.f : 0.f) : float(int(r % 83) - 41) / 4;
    const auto one_encoded = ghb::encode_quantize(single, {expected.front()}, budget, stream);
    // First numeric dictionary has the maximum metadata length in this fixture.
    const auto encode_budget = resident_size(held_out) + one_encoded.peak_bytes - one_encoded.resident_bytes;
    require(verify_encodings(held_out, expected, encode_budget, stream) == encode_budget,
            "exact minimum encode payload differs");
    rejected([&] { ghb::encode_quantize(held_out, expected, encode_budget - 1, stream); }, "one byte below minimum encode scratch must fail");
    for (const auto encoding : {ghb::EncodingPolicy::per_tile, ghb::EncodingPolicy::final_status})
      rejected([&] { ghb::encode_quantize(held_out, expected, encode_budget - 1, stream, encoding); },
               "explicit policy accepted one byte below minimum encode scratch");
  }
}

// These test-only byte counts implement the documented payload contract:
// resident bins/offsets/types plus status, tile input, lengths and dictionary.
std::size_t encoding_peak(const ghb::Dataset& data, const std::vector<ghb::Feature>& features,
                          unsigned tile_width) {
  std::size_t maximum = 1;
  for (const auto& feature : features)
    maximum = std::max(maximum, feature.type == ghb::FeatureType::numeric ? feature.cuts.size() : feature.categories.size());
  return resident_size(data) + sizeof(unsigned) + std::size_t(tile_width) *
      (std::size_t(data.rows) * sizeof(float) + sizeof(unsigned) + maximum * sizeof(float));
}

void infinity_and_recovery(ghb::Dataset& data, const std::vector<ghb::Feature>& features,
                           cudaStream_t stream) {
  // At the default width32 these are the early, middle and tail encoding tiles.
  // The middle column is an empty categorical dictionary, so status must still
  // record infinity even when its otherwise-valid inputs map to missing.
  for (const auto policy : {ghb::EncodingPolicy::per_tile, ghb::EncodingPolicy::final_status}) {
    for (const unsigned column : {0U, 33U, 66U}) {
      for (const float infinity : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}) {
        const auto index = std::size_t(column == 0 ? 0 : column == 33 ? data.rows / 2 : data.rows - 1) * data.columns + column;
        const auto saved = data.values[index];
        data.values[index] = infinity;
        rejected([&] { ghb::encode_quantize(data, features, budget, stream, policy); },
                 "multi-tile encoding lost an infinity status");
        data.values[index] = saved;
        cuda_check(cudaGetLastError());
        const auto recovered = ghb::encode_quantize(data, features, budget, stream, policy);
        verify(recovered, data, features);
      }
    }
  }
}

void multi_tile_encoding(cudaStream_t stream) {
  auto data = fixture(257, 67);
  const auto features = reference(data, 257);
  for (const unsigned width : {1U, 7U, 32U}) {
    const auto exact = encoding_peak(data, features, width);
    require(verify_encodings(data, features, exact, stream) == exact, "encoding tile budget did not select exact width");
    if (width > 1)
      require(verify_encodings(data, features, exact - 1, stream) == encoding_peak(data, features, width - 1),
              "one-byte budget reduction did not shrink encoding tile");
  }
  const auto minimum = encoding_peak(data, features, 1);
  for (const auto policy : {ghb::EncodingPolicy::per_tile, ghb::EncodingPolicy::final_status})
    rejected([&] { ghb::encode_quantize(data, features, minimum - 1, stream, policy); },
             "minimum-budget encoding rejection differs");
  infinity_and_recovery(data, features, stream);
  const auto saved_first = data.values.front();
  data.values.front() = 3.25f;
  verify_encodings(data, features, encoding_peak(data, features, 7), stream);
  data.values.front() = saved_first;
  verify_encodings(data, features, encoding_peak(data, features, 7), stream);

  auto empty_data = fixture(33, 65);
  std::vector<ghb::Feature> empty_features(empty_data.columns);
  for (unsigned f = 0; f < empty_data.columns; ++f) empty_features[f].type = empty_data.feature_types[f];
  require(verify_encodings(empty_data, empty_features, budget, stream) == encoding_peak(empty_data, empty_features, 32),
          "all-empty metadata scratch accounting differs");

  // Dataset has an owning vector, not an external-pointer overload. Register
  // that existing caller-owned storage to exercise pinned asynchronous input
  // where CUDA supports host registration; retain an explicit unsupported
  // diagnostic rather than claiming that branch ran.
  int device{}, supported{};
  cuda_check(cudaGetDevice(&device));
  cuda_check(cudaDeviceGetAttribute(&supported, cudaDevAttrHostRegisterSupported, device));
  if (!supported) {
    std::cout << "DIAGNOSTIC UNSUPPORTED: pinned encoding input, cudaDevAttrHostRegisterSupported=0\n";
    return;
  }
  const auto registration = cudaHostRegister(data.values.data(), data.values.size() * sizeof(float), cudaHostRegisterDefault);
  if (registration == cudaErrorNotSupported) {
    (void)cudaGetLastError(); // Clear this explicitly diagnosed unsupported request.
    std::cout << "DIAGNOSTIC UNSUPPORTED: pinned encoding input, cudaHostRegister returned not-supported\n";
    return;
  }
  cuda_check(registration);
  struct Registration {
    void* pointer;
    ~Registration() { if (pointer) cudaHostUnregister(pointer); }
  } registered{data.values.data()};
  verify_encodings(data, features, budget, stream);
  infinity_and_recovery(data, features, stream);
  for (unsigned row = 0; row < data.rows; ++row)
    for (unsigned f = 0; f < data.columns; ++f)
      data.values[std::size_t(row) * data.columns + f] = row % 7 == 0 ? nan :
          (features[f].type == ghb::FeatureType::categorical ? float(row % 2 ? 123456 : 0) : float(int(row % 31) - 15));
  verify_encodings(data, features, encoding_peak(data, features, 7), stream);
  cuda_check(cudaHostUnregister(registered.pointer));
  registered.pointer = nullptr;
  std::cout << "pinned encoding input and changed-input recovery checks passed\n";
}

void invalid_and_capture(cudaStream_t stream) {
  const auto good = fixture(1051, 7);
  const auto features = reference(good, 257);
  rejected([&] { ghb::fit_quantize(good, 1, budget, stream); }, "max_bins below range accepted");
  rejected([&] { ghb::fit_quantize(good, 65537, budget, stream); }, "max_bins above range accepted");
  rejected([&] { ghb::fit_quantize(good, 257, budget, stream, static_cast<ghb::QuantizePolicy>(999)); }, "invalid radix policy accepted");
  rejected([&] { ghb::fit_quantize(good, 257, 0, stream); }, "zero fit memory budget accepted");
  rejected([&] { ghb::encode_quantize(good, features, 0, stream); }, "zero encode memory budget accepted");
  rejected([&] { ghb::encode_quantize(good, features, budget, stream, ghb::EncodingPolicy(99)); },
           "unknown encoding policy accepted");
  ghb::Dataset tiny; tiny.rows = 3; tiny.columns = 1;
  tiny.values = {-0.f, 0.f, nan}; tiny.feature_types = {ghb::FeatureType::categorical};
  const auto zero_category = ghb::fit_quantize(tiny, 2, budget, stream);
  verify(zero_category, tiny, reference(tiny, 2));
  require(zero_category.features.front().categories.size() == 1 &&
          std::bit_cast<std::uint32_t>(zero_category.features.front().categories.front()) == 0,
          "signed zeros must form one canonical positive-zero category");
  tiny.values[1] = 1;
  rejected([&] { ghb::fit_quantize(tiny, 2, budget, stream); }, "two categories plus missing accepted with max_bins2");
  auto bad = good; bad.values.pop_back();
  rejected([&] { ghb::fit_quantize(bad, 257, budget, stream); }, "bad matrix shape accepted");
  bad = good; bad.rows = 0; bad.values.clear();
  rejected([&] { ghb::fit_quantize(bad, 257, budget, stream); }, "zero rows accepted");
  bad = good; bad.feature_types.pop_back();
  rejected([&] { ghb::fit_quantize(bad, 257, budget, stream); }, "bad feature type shape accepted");
  bad = good; bad.feature_types[0] = static_cast<ghb::FeatureType>(99);
  rejected([&] { ghb::fit_quantize(bad, 257, budget, stream); }, "unknown feature type accepted");
  for (float infinity : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}) {
    bad = good; bad.values.back() = infinity;
    rejected([&] { ghb::fit_quantize(bad, 257, budget, stream); }, "fit accepted infinity");
    rejected([&] { ghb::encode_quantize(bad, features, budget, stream); }, "encode accepted infinity");
  }
  auto bad_features = features; bad_features.pop_back();
  rejected([&] { ghb::encode_quantize(good, bad_features, budget, stream); }, "missing model feature accepted");
  bad_features = features; bad_features[0].cuts = {0, 0};
  rejected([&] { ghb::encode_quantize(good, bad_features, budget, stream); }, "duplicate metadata accepted");
  bad_features = features; bad_features[0].cuts = {nan};
  rejected([&] { ghb::encode_quantize(good, bad_features, budget, stream); }, "nonfinite metadata accepted");
  bad_features = features; bad_features[0].categories = {1};
  rejected([&] { ghb::encode_quantize(good, bad_features, budget, stream); }, "mixed metadata accepted");
  bad = good; bad.feature_types[0] = ghb::FeatureType::categorical;
  rejected([&] { ghb::encode_quantize(bad, features, budget, stream); }, "model feature type mismatch accepted");

  for (const auto policy : {ghb::EncodingPolicy::per_tile, ghb::EncodingPolicy::final_status}) {
    rejected([&] { ghb::encode_quantize(good, features, 0, stream, policy); }, "explicit encoding zero budget accepted");
    bad = good; bad.values.pop_back();
    rejected([&] { ghb::encode_quantize(bad, features, budget, stream, policy); }, "explicit encoding bad matrix accepted");
    bad = good; bad.rows = 0; bad.values.clear();
    rejected([&] { ghb::encode_quantize(bad, features, budget, stream, policy); }, "explicit encoding zero rows accepted");
    bad = good; bad.feature_types.pop_back();
    rejected([&] { ghb::encode_quantize(bad, features, budget, stream, policy); }, "explicit encoding bad feature-type shape accepted");
    bad = good; bad.feature_types[0] = ghb::FeatureType(99);
    rejected([&] { ghb::encode_quantize(bad, features, budget, stream, policy); }, "explicit encoding unknown feature type accepted");
    bad_features = features; bad_features.pop_back();
    rejected([&] { ghb::encode_quantize(good, bad_features, budget, stream, policy); }, "explicit encoding missing metadata accepted");
    bad_features = features; bad_features[0].cuts = {0, 0};
    rejected([&] { ghb::encode_quantize(good, bad_features, budget, stream, policy); }, "explicit encoding duplicate metadata accepted");
    bad_features = features; bad_features[0].cuts = {nan};
    rejected([&] { ghb::encode_quantize(good, bad_features, budget, stream, policy); }, "explicit encoding nonfinite metadata accepted");
    bad_features = features; bad_features[0].categories = {1};
    rejected([&] { ghb::encode_quantize(good, bad_features, budget, stream, policy); }, "explicit encoding mixed metadata accepted");
    bad_features = features; bad_features[0].type = ghb::FeatureType(99);
    rejected([&] { ghb::encode_quantize(good, bad_features, budget, stream, policy); }, "explicit encoding invalid model feature type accepted");
    bad = good; bad.feature_types[0] = ghb::FeatureType::categorical;
    rejected([&] { ghb::encode_quantize(bad, features, budget, stream, policy); }, "explicit encoding model type mismatch accepted");
  }

  cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  rejected([&] { ghb::fit_quantize(good, 257, budget, stream); }, "fit during capture accepted");
  rejected([&] { ghb::encode_quantize(good, features, budget, stream); }, "encode during capture accepted");
  for (const auto policy : {ghb::EncodingPolicy::per_tile, ghb::EncodingPolicy::final_status})
    rejected([&] { ghb::encode_quantize(good, features, budget, stream, policy); }, "explicit encoding during capture accepted");
  cudaGraph_t graph{};
  cuda_check(cudaStreamEndCapture(stream, &graph));
  if (graph) cuda_check(cudaGraphDestroy(graph));
  // Rejection must leave the stream reusable and no stale CUDA error.
  const auto after = ghb::fit_quantize(good, 257, budget, stream);
  verify(after, good, features);
  verify_encodings(good, features, budget, stream);
  cuda_check(cudaGetLastError());
}
} // namespace

int main() {
  static_assert(!std::is_copy_constructible_v<ghb::QuantizedData>);
  static_assert(!std::is_copy_assignable_v<ghb::QuantizedData>);
  static_assert(std::is_nothrow_move_constructible_v<ghb::QuantizedData>);
  static_assert(std::is_nothrow_move_assignable_v<ghb::QuantizedData>);
  try {
    int count = 0;
    const auto error = cudaGetDeviceCount(&count);
    if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver || (error == cudaSuccess && count == 0)) {
      std::cout << "quantization tests skipped: no CUDA device/driver\n"; return 77;
    }
    cuda_check(error);
    Stream stream;
    tails_and_semantics(stream.value);
    cardinality_and_hierarchy(stream.value);
    tiling_budgets_and_moves(stream.value);
    multi_tile_encoding(stream.value);
    invalid_and_capture(stream.value);
    std::cout << "quantization tests passed: " << checks << " checks\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "quantization test failure: " << error.what() << '\n'; return 1;
  }
}
