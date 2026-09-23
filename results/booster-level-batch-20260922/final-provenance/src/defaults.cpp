#include <gh/histogram.hpp>

#include <algorithm>
#include <climits>
#include <cstring>

namespace gh {
namespace {

int bounded_grid(std::size_t size, int tuning, const cudaDeviceProp& device) {
  const auto policy = tuning_catalog[tuning];
  const std::size_t tile = static_cast<std::size_t>(policy.threads) * policy.items;
  // Quotient/remainder avoids size+tile-1 overflow at the public size limit.
  const std::size_t tiles = size / tile + (size % tile != 0);
  const std::size_t sms = static_cast<std::size_t>(std::max(device.multiProcessorCount, 1));
  std::size_t limit = std::min(sms * 4, static_cast<std::size_t>(INT_MAX));
  if (device.maxGridSize[0] > 0)
    limit = std::min(limit, static_cast<std::size_t>(device.maxGridSize[0]));
  return static_cast<int>(std::max<std::size_t>(1, std::min(tiles, limit)));
}

bool resources_fit(const Config& config, const cudaDeviceProp& device) {
  if (!supported(config)) return false;
  const auto policy = tuning_catalog[config.tuning];
  if (device.maxThreadsPerBlock > 0 && policy.threads > device.maxThreadsPerBlock) return false;
  if (device.maxGridSize[0] > 0 && config.blocks > device.maxGridSize[0]) return false;
  if (config.algorithm != Algorithm::shared_atomic) return true;
  const std::size_t local_bytes = config.local_counter == LocalCounter::u32
      ? sizeof(unsigned) : counter_bytes(config.counter_type);
  const std::size_t actual_shared = static_cast<std::size_t>(config.bins) * policy.replicas * local_bytes;
  if (policy.shared_limit > 48 * 1024)
    return device.sharedMemPerBlockOptin >= policy.shared_limit && actual_shared <= policy.shared_limit;
  return actual_shared <= device.sharedMemPerBlock;
}

Config generic_config(const Config& config, const cudaDeviceProp& device) {
  Config fallback = config;
  fallback.algorithm = Algorithm::global_atomic;
  fallback.tuning = device.maxThreadsPerBlock > 0 && device.maxThreadsPerBlock < 256 ? 0 : 2;
  fallback.blocks = bounded_grid(config.size, fallback.tuning, device);
  fallback.local_counter = LocalCounter::native;
  fallback.output_clear = config.launch == LaunchMode::graph
      ? OutputClear::kernel : OutputClear::runtime;
  if (config.size == 0 || !supported(fallback)) return fallback;

  // Heuristic coverage outside the measured table. Shared storage is preferred
  // when it fits; the generic grid is bounded rather than extrapolating measured
  // winners. Narrow locals are chosen only after supported() proves their exact
  // maximum CTA count. Native locals/global atomics cover larger count domains.
  for (LocalCounter local : {LocalCounter::u32, LocalCounter::native}) {
    if (local == LocalCounter::u32 && config.counter_type != CounterType::u64) continue;
    for (int tuning : {fallback.tuning == 0 ? 0 : 7, 14}) {
      Config candidate = fallback;
      candidate.algorithm = Algorithm::shared_atomic;
      candidate.tuning = tuning;
      candidate.blocks = bounded_grid(config.size, tuning, device);
      candidate.local_counter = local;
      if (resources_fit(candidate, device)) return candidate;
    }
  }
  return fallback;
}

}  // namespace

Config default_config(const Config& config, const cudaDeviceProp& device) {
  if (config.algorithm != Algorithm::automatic) return config;

  const Config fallback = generic_config(config, device);
  if (std::strcmp(device.name, "NVIDIA RTX A5000 Laptop GPU") != 0 ||
      device.major != 8 || device.minor != 6 || device.multiProcessorCount != 48)
    return fallback;

  // Exact shape keys only. These are measured defaults, not a distribution
  // classifier; one policy serves every input ordering/value distribution.
  struct Entry {
    std::size_t size;
    unsigned bins;
    InputType input;
    CounterType counter;
    LaunchMode launch;
    CacheMode cache;
    int tuning;
    int blocks;
    LocalCounter local;
  };
  constexpr std::size_t one_million = 1u << 20;
  constexpr std::size_t sixteen_million = 1u << 24;
  constexpr Entry entries[] = {
      {one_million, 4096, InputType::u32, CounterType::u32, LaunchMode::stream,
       CacheMode::warm, 10, 48, LocalCounter::native},
      {4096, 256, InputType::u8, CounterType::u32, LaunchMode::graph,
       CacheMode::warm, 1, 96, LocalCounter::native},
      {one_million, 256, InputType::u8, CounterType::u32, LaunchMode::graph,
       CacheMode::warm, 11, 48, LocalCounter::native},
      // Best custom finalist from the byte validation; this entry did not pass
      // the old reference-retention margin. Production remains custom-only.
      {sixteen_million, 256, InputType::u8, CounterType::u32, LaunchMode::graph,
       CacheMode::warm, 10, 96, LocalCounter::native},
      {one_million, 8, InputType::u32, CounterType::u32, LaunchMode::graph,
       CacheMode::warm, 6, 192, LocalCounter::native},
      {one_million, 256, InputType::u32, CounterType::u32, LaunchMode::graph,
       CacheMode::warm, 11, 48, LocalCounter::native},
      {sixteen_million, 4096, InputType::u32, CounterType::u32, LaunchMode::graph,
       CacheMode::warm, 11, 48, LocalCounter::native},
      {sixteen_million, 4096, InputType::u32, CounterType::u64, LaunchMode::graph,
       CacheMode::warm, 10, 48, LocalCounter::u32},
      {sixteen_million, 8192, InputType::u32, CounterType::u64, LaunchMode::graph,
       CacheMode::warm, 15, 48, LocalCounter::u32},
      {sixteen_million, 16384, InputType::u32, CounterType::u64, LaunchMode::graph,
       CacheMode::warm, 15, 48, LocalCounter::u32},
      {one_million, 4096, InputType::u32, CounterType::u64, LaunchMode::graph,
       CacheMode::cold, 10, 48, LocalCounter::u32},
  };
  for (const auto& entry : entries) {
    if (config.size != entry.size || config.bins != entry.bins ||
        config.input_type != entry.input || config.counter_type != entry.counter ||
        config.launch != entry.launch || config.cache != entry.cache)
      continue;
    Config selected = fallback;
    selected.algorithm = Algorithm::shared_atomic;
    selected.tuning = entry.tuning;
    selected.blocks = entry.blocks;
    selected.local_counter = entry.local;
    return resources_fit(selected, device) ? selected : fallback;
  }
  return fallback;
}

}  // namespace gh
