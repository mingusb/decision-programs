#include "gh/histogram.hpp"

#include <algorithm>
#include <climits>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
int checks = 0;

void require(bool condition, const std::string& message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

cudaDeviceProp measured_device() {
  cudaDeviceProp device{};
  std::strcpy(device.name, "NVIDIA RTX A5000 Laptop GPU");
  device.major = 8;
  device.minor = 6;
  device.multiProcessorCount = 48;
  device.maxThreadsPerBlock = 1024;
  device.maxGridSize[0] = INT_MAX;
  device.sharedMemPerBlock = 48 * 1024;
  device.sharedMemPerBlockOptin = 96 * 1024;
  return device;
}

bool same_config(const gh::Config& a, const gh::Config& b) {
  return a.algorithm == b.algorithm && a.input_type == b.input_type &&
         a.counter_type == b.counter_type && a.local_counter == b.local_counter &&
         a.size == b.size && a.bins == b.bins && a.blocks == b.blocks &&
         a.tuning == b.tuning && a.output_clear == b.output_clear &&
         a.launch == b.launch && a.cache == b.cache;
}

gh::Config graph_request() {
  gh::Config request;
  request.size = std::size_t{1} << 24;
  request.bins = 4096;
  request.counter_type = gh::CounterType::u64;
  request.launch = gh::LaunchMode::graph;
  return request;
}

bool production_algorithm(gh::Algorithm algorithm) {
  switch (algorithm) {
    case gh::Algorithm::global_atomic:
    case gh::Algorithm::warp_aggregated:
    case gh::Algorithm::shared_atomic:
    case gh::Algorithm::shared_rle:
    case gh::Algorithm::shared_warp:
    case gh::Algorithm::shared_partial:
    case gh::Algorithm::bitplane:
      return true;
    default:
      return false;
  }
}

void require_workload_preserved(const gh::Config& request, const gh::Config& chosen) {
  require(chosen.size == request.size && chosen.bins == request.bins &&
              chosen.input_type == request.input_type && chosen.counter_type == request.counter_type &&
              chosen.launch == request.launch && chosen.cache == request.cache,
          "automatic resolution changed the workload or declared execution mode");
}

gh::Config require_custom(const gh::Config& request, const cudaDeviceProp& device) {
  const auto chosen = gh::default_config(request, device);
  require(production_algorithm(chosen.algorithm) && gh::supported(chosen),
          "valid workload did not resolve to a supported custom implementation");
  require_workload_preserved(request, chosen);
  require(chosen.output_clear == (request.launch == gh::LaunchMode::graph
              ? gh::OutputClear::kernel : gh::OutputClear::runtime),
          "automatic output clearing differs from the declared launch mode");
  const auto policy = gh::tuning_catalog[chosen.tuning];
  require(device.maxThreadsPerBlock == 0 || policy.threads <= device.maxThreadsPerBlock,
          "selected policy exceeds the supplied thread limit");
  require(device.maxGridSize[0] == 0 || chosen.blocks <= device.maxGridSize[0],
          "selected grid exceeds the supplied grid limit");
  if (chosen.algorithm == gh::Algorithm::shared_atomic) {
    const std::size_t local_bytes = chosen.local_counter == gh::LocalCounter::u32
        ? sizeof(unsigned) : gh::counter_bytes(chosen.counter_type);
    const std::size_t actual = static_cast<std::size_t>(chosen.bins) * policy.replicas * local_bytes;
    require(actual <= policy.shared_limit, "selected histogram exceeds its declared shared limit");
    require(policy.shared_limit > 48 * 1024
                ? device.sharedMemPerBlockOptin >= policy.shared_limit
                : actual <= device.sharedMemPerBlock,
            "selected histogram exceeds supplied device shared capacity");
  }
  return chosen;
}

void test_resolution_contract() {
  const auto device = measured_device();
  const auto request = graph_request();
  require(request.algorithm == gh::Algorithm::automatic, "Config must default to automatic selection");
  const auto chosen = require_custom(request, device);
  require(chosen.algorithm == gh::Algorithm::shared_atomic && chosen.tuning == 10 &&
              chosen.blocks == 48 && chosen.local_counter == gh::LocalCounter::u32,
          "measured large-bin graph default changed unexpectedly");
  require(same_config(gh::default_config(request, device), chosen), "selection is not deterministic");
  require(same_config(gh::default_config(chosen, device), chosen), "explicit resolution is not idempotent");
  auto reset = chosen;
  reset.algorithm = gh::Algorithm::automatic;
  ++reset.size;
  require(gh::supported(reset), "reset automatic must accept prior vector/local-u32 policy knobs");
  require_custom(reset, device);

  auto stream = request;
  stream.size = std::size_t{1} << 20;
  stream.counter_type = gh::CounterType::u32;
  stream.launch = gh::LaunchMode::stream;
  const auto stream_chosen = require_custom(stream, device);
  require(stream_chosen.tuning == 10 && stream_chosen.blocks == 48,
          "measured stream default changed unexpectedly");

  auto byte = request;
  byte.input_type = gh::InputType::u8;
  byte.counter_type = gh::CounterType::u32;
  byte.bins = 256;
  const auto byte_chosen = require_custom(byte, device);
  require(byte_chosen.algorithm == gh::Algorithm::shared_atomic &&
              byte_chosen.tuning == 10 && byte_chosen.blocks == 96,
          "large-byte default must use the recorded custom finalist");
}

void test_generic_coverage() {
  const auto device = measured_device();
  for (auto input : {gh::InputType::u8, gh::InputType::u32})
    for (auto counter : {gh::CounterType::u32, gh::CounterType::u64})
      for (auto launch : {gh::LaunchMode::stream, gh::LaunchMode::graph})
        for (auto cache : {gh::CacheMode::warm, gh::CacheMode::cold})
          for (std::size_t size : {std::size_t{0}, std::size_t{1}, std::size_t{4095},
                                  std::size_t{4097}, (std::size_t{1} << 20) + 1})
            for (unsigned bins : {1u, 255u, 256u, 4095u, 16385u,
                                  static_cast<unsigned>(INT_MAX - 1)}) {
              if (input == gh::InputType::u8 && bins > 256) continue;
              gh::Config request;
              request.input_type = input; request.counter_type = counter;
              request.launch = launch; request.cache = cache;
              request.size = size; request.bins = bins;
              const auto selected = require_custom(request, device);
              require(selected.blocks <= device.multiProcessorCount * 4,
                      "generic grid is not modestly bounded by the SM count");
              const auto policy = gh::tuning_catalog[selected.tuning];
              const std::size_t tile = static_cast<std::size_t>(policy.threads) * policy.items;
              const std::size_t tiles = size / tile + (size % tile != 0);
              require(static_cast<std::size_t>(selected.blocks) <= std::max<std::size_t>(tiles, 1),
                      "generic grid creates more CTAs than input tiles");
            }
}

void test_hardware_and_resource_constraints() {
  const auto request = graph_request();
  const auto measured = require_custom(request, measured_device());
  auto different = measured_device();
  std::strcpy(different.name, "Other GPU");
  const auto generic = require_custom(request, different);
  require(generic.tuning != measured.tuning || generic.blocks != measured.blocks,
          "a different hardware name unexpectedly reused the exact measured policy");
  require_custom(request, cudaDeviceProp{});
  different = measured_device();
  different.major = 9;
  require_custom(request, different);
  different = measured_device();
  different.multiProcessorCount = 7;
  require(require_custom(request, different).blocks <= 28,
          "generic grid does not respect the supplied SM count");
  different.maxThreadsPerBlock = 128;
  different.maxGridSize[0] = 3;
  require_custom(request, different);

  auto large_bins = request;
  large_bins.bins = 16384;
  const auto optin = require_custom(large_bins, measured_device());
  require(gh::tuning_catalog[optin.tuning].shared_limit == 96 * 1024,
          "large-bin local-u32 profile should exercise opt-in storage");
  different = measured_device();
  different.sharedMemPerBlockOptin = 96 * 1024 - 1;
  const auto no_optin = require_custom(large_bins, different);
  require(no_optin.algorithm == gh::Algorithm::global_atomic,
          "insufficient declared opt-in capacity should use the global implementation");
  different.sharedMemPerBlock = 0;
  different.sharedMemPerBlockOptin = 0;
  require(require_custom(request, different).algorithm == gh::Algorithm::global_atomic,
          "zero shared capacity must not select shared storage");
}

void test_extreme_counts() {
  auto device = measured_device();
  std::strcpy(device.name, "Generic single-SM GPU");
  device.multiProcessorCount = 1;
  gh::Config request;
  request.input_type = gh::InputType::u8;
  request.counter_type = gh::CounterType::u64;
  request.bins = 1;
  request.size = std::size_t{std::numeric_limits<unsigned>::max()} + 1;
  const auto narrow = require_custom(request, device);
  require(narrow.local_counter == gh::LocalCounter::u32,
          "safe per-CTA local-u32 counting was not selected");

  // Four bounded CTAs cannot hold this all-one-bin count in u32 locals.
  request.size = (std::size_t{std::numeric_limits<unsigned>::max()} + 1) * 4;
  const auto wide = require_custom(request, device);
  require(wide.algorithm == gh::Algorithm::shared_atomic && wide.local_counter == gh::LocalCounter::native,
          "unsafe local-u32 counts must fall back to native shared counters when they fit");
  request.size = static_cast<std::size_t>(PTRDIFF_MAX);
  require(require_custom(request, device).local_counter == gh::LocalCounter::native,
          "maximum byte extent must retain 64-bit accumulation");

  request.input_type = gh::InputType::u32;
  request.size = static_cast<std::size_t>(PTRDIFF_MAX) / sizeof(unsigned);
  request.bins = static_cast<unsigned>(INT_MAX - 1);
  const auto global = require_custom(request, device);
  require(global.algorithm == gh::Algorithm::global_atomic && global.local_counter == gh::LocalCounter::native,
          "huge dense domains must retain the native global implementation");
  device.multiProcessorCount = INT_MAX;
  require_custom(request, device);  // SM-to-grid multiplication must not overflow int.

  request.counter_type = gh::CounterType::u32;
  request.size = std::size_t{std::numeric_limits<unsigned>::max()} + 1;
  const auto invalid = gh::default_config(request, device);
  require(production_algorithm(invalid.algorithm) && !gh::supported(invalid),
          "automatic selection must not disguise an invalid u32 output extent");
  require_workload_preserved(request, invalid);
}

void test_manual_passthrough() {
  auto manual = graph_request();
  manual.algorithm = gh::Algorithm::shared_rle;
  manual.local_counter = gh::LocalCounter::u32;
  manual.tuning = 3;
  manual.blocks = 73;
  manual.cache = gh::CacheMode::cold;
  manual.output_clear = gh::OutputClear::runtime;
  require(gh::supported(manual), "manual fixture must be structurally supported");
  require(same_config(gh::default_config(manual, measured_device()), manual),
          "manual algorithm settings must not be retuned");
  require(same_config(gh::default_config(manual, cudaDeviceProp{}), manual),
          "hardware selection must not modify an explicit algorithm");
  manual.blocks = 0;
  require(same_config(gh::default_config(manual, measured_device()), manual) && !gh::supported(manual),
          "invalid manual settings must remain visible to validation");
}
}  // namespace

int main() try {
  // Metadata-only selector tests. No CUDA calls are made, but this executable
  // links gh/CUDA and must not run during the user's GPU-use prohibition.
  test_resolution_contract();
  test_generic_coverage();
  test_hardware_and_resource_constraints();
  test_extreme_counts();
  test_manual_passthrough();
  std::cout << "PASS: " << checks << " CPU-only default selection checks\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "FAIL: " << error.what() << '\n';
  return 1;
}
