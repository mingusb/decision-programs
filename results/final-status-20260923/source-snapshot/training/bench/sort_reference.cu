// NVIDIA algorithms are permitted in this isolated diagnostic only.
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/version.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
static_assert(CUB_VERSION == 300402, "This frozen experiment requires the inspected CUB 3.4.2 headers");
constexpr unsigned warmups = 2, samples = 5;
constexpr std::array<const char*, 3> names{"segmented_u32", "per_column_u32", "feature_tagged_u64"};
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Stream {
  cudaStream_t value{};
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamSynchronize(value); cudaStreamDestroy(value); }
};
struct Event {
  cudaEvent_t value{};
  Event() { check(cudaEventCreate(&value)); }
  ~Event() { cudaEventDestroy(value); }
};
template<class T> struct Device {
  T* value{};
  explicit Device(std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::invalid_argument("device allocation size overflow");
    check(cudaMalloc(reinterpret_cast<void**>(&value), std::max<std::size_t>(count, 1) * sizeof(T)));
  }
  ~Device() { cudaFree(value); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
};
struct Options {
  std::uint32_t rows{65536}, features{32};
  std::uint64_t cardinality{}, seed{2026092201};
};
template<class T> T integer(std::string_view text) {
  T value{};
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size()) throw std::invalid_argument("invalid unsigned integer");
  return value;
}
Options parse(int argc, char** argv) {
  Options result;
  for (int i = 1; i < argc; ++i) {
    const std::string_view key = argv[i];
    if (key == "--help") {
      std::cout << "ghb_sort_reference [--rows N] [--features F] [--cardinality K] [--seed N]\n"
                   "K=0 uses the full uint32 domain; otherwise K<=4294967296 bounds distinct keys.\n"
                   "Requires rows*features<=INT_MAX; 2 warmup sweeps and 5 samples per candidate.\n"
                   "Sort-only GPU event timings exclude allocations, transfers, and validation.\n";
      std::exit(0);
    }
    if (++i == argc) throw std::invalid_argument("missing argument value");
    const std::string_view value = argv[i];
    if (key == "--rows") result.rows = integer<std::uint32_t>(value);
    else if (key == "--features") result.features = integer<std::uint32_t>(value);
    else if (key == "--cardinality") result.cardinality = integer<std::uint64_t>(value);
    else if (key == "--seed") result.seed = integer<std::uint64_t>(value);
    else throw std::invalid_argument("unknown option: " + std::string(key));
  }
  if (!result.rows || !result.features || std::uint64_t(result.rows) * result.features > std::uint64_t(INT32_MAX))
    throw std::invalid_argument("positive rows/features and rows*features<=INT_MAX required");
  if (result.cardinality > (1ULL << 32)) throw std::invalid_argument("cardinality exceeds uint32 domain");
  return result;
}
std::uint64_t next_random(std::uint64_t& state) {
  auto value = (state += 0x9e3779b97f4a7c15ULL);
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}
std::string quote(std::string_view value) {
  std::ostringstream out; out << '"';
  for (const unsigned char character : value) {
    if (character == '"' || character == '\\') out << '\\' << character;
    else if (character < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(character) << std::dec;
    else out << character;
  }
  out << '"'; return out.str();
}

__global__ void pack(const std::uint32_t* input, std::uint64_t* output, unsigned count, unsigned rows) {
  for (std::uint64_t index = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += std::uint64_t(gridDim.x) * blockDim.x)
    output[index] = (std::uint64_t(index / rows) << 32) | input[index];
}
__global__ void strip(const std::uint64_t* input, std::uint32_t* output, unsigned count) {
  for (std::uint64_t index = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += std::uint64_t(gridDim.x) * blockDim.x)
    output[index] = std::uint32_t(input[index]);
}

int run(int argc, char** argv) {
  const auto options = parse(argc, argv);
  const auto count = std::size_t(options.rows) * options.features;
  const int item_count = int(count), tag_end_bit = 32 + std::bit_width(options.features - 1);
  const auto key_bytes = count * sizeof(std::uint32_t);
  std::vector<std::uint32_t> input(count), reference, downloaded(count), distinct(options.features);
  auto random_state = options.seed;
  std::uint64_t fingerprint = 14695981039346656037ULL;
  for (auto& key : input) {
    const auto random = next_random(random_state);
    key = options.cardinality == 0 ? std::uint32_t(random) : options.cardinality == 1 ? 0u
        : std::uint32_t((random % options.cardinality) * UINT32_MAX / (options.cardinality - 1));
    // Portable byte order, diagnostic fingerprint only; not a cryptographic hash.
    for (unsigned byte = 0; byte < 4; ++byte) { fingerprint ^= (key >> (8 * byte)) & 255u; fingerprint *= 1099511628211ULL; }
  }
  reference = input;
  for (std::uint32_t feature = 0; feature < options.features; ++feature) {
    const auto first = reference.begin() + std::size_t(feature) * options.rows, last = first + options.rows;
    std::sort(first, last);
    distinct[feature] = 1;
    for (auto value = first + 1; value != last; ++value) distinct[feature] += *value != *(value - 1);
  }
  int device{}, runtime{}, driver{}; cudaDeviceProp properties{};
  check(cudaGetDevice(&device)); check(cudaGetDeviceProperties(&properties, device));
  check(cudaRuntimeGetVersion(&runtime)); check(cudaDriverGetVersion(&driver));
  Stream stream; Event begin, end;
  Device<std::uint32_t> keys(count), output(count);
  Device<std::uint64_t> tagged(count), sorted_tagged(count);
  Device<std::int32_t> offsets(std::size_t(options.features) + 1);
  std::vector<std::int32_t> host_offsets(std::size_t(options.features) + 1);
  for (std::uint32_t feature = 0; feature <= options.features; ++feature)
    host_offsets[feature] = std::int32_t(std::uint64_t(feature) * options.rows);
  check(cudaMemcpyAsync(keys.value, input.data(), key_bytes, cudaMemcpyHostToDevice, stream.value));
  check(cudaMemcpyAsync(offsets.value, host_offsets.data(), host_offsets.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream.value));
  check(cudaStreamSynchronize(stream.value));
  std::array<std::size_t, 3> scratch{};
  check(cub::DeviceSegmentedRadixSort::SortKeys(nullptr, scratch[0], keys.value, output.value, item_count,
        options.features, offsets.value, offsets.value + 1, 0, 32, stream.value));
  check(cub::DeviceRadixSort::SortKeys(nullptr, scratch[1], keys.value, output.value, int(options.rows), 0, 32, stream.value));
  check(cub::DeviceRadixSort::SortKeys(nullptr, scratch[2], tagged.value, sorted_tagged.value, item_count, 0, tag_end_bit, stream.value));
  const auto workspace_bytes = std::max<std::size_t>(1, *std::max_element(scratch.begin(), scratch.end()));
  Device<unsigned char> workspace(workspace_bytes);
  const unsigned blocks = unsigned(std::min<std::size_t>((count + 255) / 256, 65535));
  auto invoke = [&](unsigned candidate) {
    auto bytes = scratch[candidate];
    if (candidate == 0) {
      check(cub::DeviceSegmentedRadixSort::SortKeys(workspace.value, bytes, keys.value, output.value, item_count,
            options.features, offsets.value, offsets.value + 1, 0, 32, stream.value));
    } else if (candidate == 1) {
      for (std::uint32_t feature = 0; feature < options.features; ++feature) {
        const auto offset = std::size_t(feature) * options.rows;
        bytes = scratch[candidate];
        check(cub::DeviceRadixSort::SortKeys(workspace.value, bytes, keys.value + offset, output.value + offset,
              int(options.rows), 0, 32, stream.value));
      }
    } else {
      pack<<<blocks, 256, 0, stream.value>>>(keys.value, tagged.value, unsigned(count), options.rows);
      check(cudaGetLastError());
      check(cub::DeviceRadixSort::SortKeys(workspace.value, bytes, tagged.value, sorted_tagged.value, item_count, 0, tag_end_bit, stream.value));
      strip<<<blocks, 256, 0, stream.value>>>(sorted_tagged.value, output.value, unsigned(count));
      check(cudaGetLastError());
    }
  };
  unsigned validated{};
  auto validate = [&] {
    check(cudaMemcpyAsync(downloaded.data(), output.value, key_bytes, cudaMemcpyDeviceToHost, stream.value));
    check(cudaStreamSynchronize(stream.value));
    const auto mismatch = std::mismatch(downloaded.begin(), downloaded.end(), reference.begin());
    if (mismatch.first != downloaded.end()) throw std::runtime_error("sort differs from CPU reference at key " + std::to_string(mismatch.first - downloaded.begin()));
    ++validated;
  };
  std::array<std::array<float, samples>, 3> milliseconds{};
  std::array<std::array<unsigned, 3>, samples> order{};
  const unsigned rotation = options.seed % 3;
  for (unsigned sweep = 0; sweep < warmups + samples; ++sweep) {
    for (unsigned position = 0; position < 3; ++position) {
      const auto candidate = (rotation + (sweep % 2 ? 2 - position : position)) % 3;
      // A missing/partial write must not inherit the previous candidate's
      // correct output. This test-only preparation is outside the event span.
      check(cudaMemsetAsync(output.value, 0xa5, key_bytes, stream.value));
      check(cudaStreamSynchronize(stream.value));
      if (sweep < warmups) { invoke(candidate); check(cudaStreamSynchronize(stream.value)); }
      else {
        const auto sample = sweep - warmups;
        order[sample][position] = candidate;
        check(cudaEventRecord(begin.value, stream.value)); invoke(candidate); check(cudaEventRecord(end.value, stream.value));
        check(cudaEventSynchronize(end.value));
        check(cudaEventElapsedTime(&milliseconds[candidate][sample], begin.value, end.value));
        if (!std::isfinite(milliseconds[candidate][sample]) || milliseconds[candidate][sample] < 0)
          throw std::runtime_error("invalid GPU event sample");
      }
      validate();
    }
  }
  check(cudaMemcpyAsync(downloaded.data(), keys.value, key_bytes, cudaMemcpyDeviceToHost, stream.value));
  check(cudaStreamSynchronize(stream.value));
  if (downloaded != input) throw std::runtime_error("sort modified immutable common input");
  const auto offset_bytes = host_offsets.size() * sizeof(std::int32_t);
  const std::array<std::size_t, 3> payload{2 * key_bytes + offset_bytes + std::max<std::size_t>(scratch[0], 1),
      2 * key_bytes + std::max<std::size_t>(scratch[1], 1), 6 * key_bytes + std::max<std::size_t>(scratch[2], 1)};
  std::ostringstream json; json << std::setprecision(17);
  json << "{\"schema_version\":1,\"kind\":\"ghb.benchmark_only.sort_reference\",\"cub_version\":" << CUB_VERSION
       << ",\"generator\":\"splitmix64_rank_domain_v1\",\"seed\":" << options.seed
       << ",\"rows\":" << options.rows << ",\"features\":" << options.features << ",\"keys\":" << count
       << ",\"cardinality_upper_bound\":" << options.cardinality << ",\"zero_cardinality_means_full_u32\":true"
       << ",\"input_fnv1a64\":" << quote(std::to_string(fingerprint)) << ",\"gpu\":" << quote(properties.name)
       << ",\"compute_capability\":" << quote(std::to_string(properties.major) + "." + std::to_string(properties.minor))
       << ",\"cuda_runtime\":" << runtime << ",\"cuda_driver\":" << driver
       << ",\"warmup_sweeps\":" << warmups << ",\"samples_per_candidate\":" << samples
       << ",\"timing_boundary\":\"resident_feature_major_u32_to_sorted_feature_major_u32\""
       << ",\"timing_excludes\":[\"allocation\",\"scratch_query\",\"upload\",\"download\",\"CPU_validation\",\"event_setup\",\"test_output_poison\"]"
       << ",\"timing_includes_tag_pack_strip\":true,\"comparison_scope\":\"sort_only_not_full_quantization_or_training\""
       << ",\"memory\":{\"common_input_bytes\":" << key_bytes << ",\"common_output_bytes\":" << key_bytes
       << ",\"combined_experiment_payload_bytes\":" << 6 * key_bytes + offset_bytes + workspace_bytes
       << ",\"shared_workspace_bytes\":" << workspace_bytes << ",\"excludes_runtime_bookkeeping\":true}"
       << ",\"validation\":{\"exact_per_feature_cpu_std_sort\":true,\"input_unchanged\":true,\"checked_outputs\":" << validated << "}"
       << ",\"observed_distinct_per_feature\":[";
  for (std::size_t i = 0; i < distinct.size(); ++i) { if (i) json << ','; json << distinct[i]; }
  json << "],\"sample_execution_order\":[";
  for (unsigned sample = 0; sample < samples; ++sample) {
    if (sample) json << ',';
    json << '[';
    for (unsigned position = 0; position < 3; ++position) { if (position) json << ','; json << quote(names[order[sample][position]]); }
    json << ']';
  }
  json << "],\"candidates\":[";
  for (unsigned candidate = 0; candidate < 3; ++candidate) {
    if (candidate) json << ',';
    auto ordered = milliseconds[candidate]; std::sort(ordered.begin(), ordered.end());
    json << "{\"name\":" << quote(names[candidate]) << ",\"scratch_bytes\":" << scratch[candidate]
         << ",\"isolated_device_payload_bytes\":" << payload[candidate] << ",\"sort_begin_bit\":0,\"sort_end_bit\":" << (candidate == 2 ? tag_end_bit : 32)
         << ",\"median_ms\":" << ordered[samples / 2] << ",\"raw_ms\":[";
    for (unsigned sample = 0; sample < samples; ++sample) { if (sample) json << ','; json << milliseconds[candidate][sample]; }
    json << "]}";
  }
  json << "]}\n"; std::cout << json.str();
  return 0;
}
} // namespace
int main(int argc, char** argv) {
  try { return run(argc, argv); }
  catch (const std::exception& error) { std::cerr << "ghb_sort_reference: " << error.what() << '\n'; return 1; }
}
