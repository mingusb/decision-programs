#include <gh/histogram.hpp>

#include <algorithm>
#include <climits>

namespace gh {
namespace {

constexpr std::size_t shared_memory_limit = 48 * 1024;

bool shared_algorithm(Algorithm algorithm) {
  return algorithm == Algorithm::shared_atomic || algorithm == Algorithm::shared_rle ||
         algorithm == Algorithm::shared_warp || algorithm == Algorithm::shared_partial ||
         algorithm == Algorithm::shared_overflow;
}

std::size_t local_bytes(const Config& config) {
  return config.local_counter == LocalCounter::u32 ? sizeof(unsigned) : counter_bytes(config.counter_type);
}

std::size_t max_block_items(const Config& config) {
  const auto policy = tuning_catalog[config.tuning];
  const std::size_t tile = static_cast<std::size_t>(policy.threads) * policy.items;
  const std::size_t stride = static_cast<std::size_t>(config.blocks) * tile;
  // The first CTA receives the most input: one tile per complete grid round,
  // followed by at most one tile from the remainder. The result is <= size,
  // so neither multiplication nor addition can overflow a valid input extent.
  return (config.size / stride) * tile + std::min(config.size % stride, tile);
}

}  // namespace

bool supported(const Config& config) {
  if (config.input_type != InputType::u8 && config.input_type != InputType::u32) return false;
  if (config.counter_type != CounterType::u32 && config.counter_type != CounterType::u64) return false;
  if (config.local_counter != LocalCounter::native && config.local_counter != LocalCounter::u32) return false;
  if (config.output_clear != OutputClear::runtime && config.output_clear != OutputClear::kernel) return false;
  if (config.launch != LaunchMode::stream && config.launch != LaunchMode::graph) return false;
  if (config.cache != CacheMode::warm && config.cache != CacheMode::cold) return false;
  if (config.bins == 0 || config.bins > static_cast<unsigned>(INT_MAX - 1)) return false;
  if (config.input_type == InputType::u8 && config.bins > 256) return false;
  if (config.counter_type == CounterType::u32 && config.size > UINT_MAX) return false;
  if (config.size > static_cast<std::size_t>(PTRDIFF_MAX) / input_bytes(config.input_type)) return false;
  if (config.blocks <= 0 || config.tuning < 0 ||
      config.tuning >= static_cast<int>(sizeof(tuning_catalog) / sizeof(tuning_catalog[0]))) return false;
  // Resolution replaces policy knobs. Treat automatic as a scalar global
  // policy here, including when a caller resets a previously resolved config.
  const auto policy = tuning_catalog[config.algorithm == Algorithm::automatic ? 2 : config.tuning];
  if (policy.threads <= 0 || policy.threads > 1024) return false;
  if (policy.shared_limit > shared_memory_limit && !shared_algorithm(config.algorithm)) return false;
  if (policy.load != LoadPolicy::scalar && !shared_algorithm(config.algorithm) &&
      config.algorithm != Algorithm::bitplane) return false;
  if (config.algorithm != Algorithm::automatic && config.local_counter == LocalCounter::u32) {
    if (config.counter_type != CounterType::u64) return false;
    if (shared_algorithm(config.algorithm)) {
      if (max_block_items(config) > UINT_MAX) return false;
    } else if (config.algorithm == Algorithm::global_atomic ||
               config.algorithm == Algorithm::warp_aggregated ||
               config.algorithm == Algorithm::global_window) {
      // All blocks update the same narrow scratch histogram. Even a single-bin
      // input must fit; the shared algorithms' per-block bound does not apply.
      if (config.size > UINT_MAX || static_cast<std::size_t>(config.bins) >
          static_cast<std::size_t>(PTRDIFF_MAX) / sizeof(unsigned)) return false;
    } else {
      return false;
    }
  }
  switch (config.algorithm) {
    case Algorithm::automatic:
    case Algorithm::global_atomic:
    case Algorithm::warp_aggregated:
      break;
    case Algorithm::shared_atomic:
    case Algorithm::shared_rle:
    case Algorithm::shared_warp:
    case Algorithm::shared_partial:
      if (static_cast<std::size_t>(config.bins) * policy.replicas *
              local_bytes(config) > policy.shared_limit) return false;
      if (config.algorithm == Algorithm::shared_partial &&
          static_cast<std::size_t>(config.blocks) > static_cast<std::size_t>(PTRDIFF_MAX) /
              config.bins / local_bytes(config)) return false;
      break;
    case Algorithm::shared_overflow:
      if (config.input_type != InputType::u32 || config.counter_type != CounterType::u64 ||
          config.local_counter != LocalCounter::u32 || config.bins <= 24576 ||
          (config.tuning != 14 && config.tuning != 15)) return false;
      break;
    case Algorithm::global_window:
      if (config.input_type != InputType::u32 || config.counter_type != CounterType::u64 ||
          config.local_counter != LocalCounter::u32 || config.tuning >= 6 ||
          config.window_bins == 0 || config.window_bins > static_cast<unsigned>(INT_MAX - 1))
        return false;
      break;
    case Algorithm::bitplane: {
      if (config.bins > 256) return false;
      unsigned capacity = 1;
      while (capacity < config.bins) capacity *= 2;
      if (static_cast<std::size_t>(policy.threads / 32) * capacity * counter_bytes(config.counter_type) >
          shared_memory_limit) return false;
      break;
    }
    default:
      return false;
  }
  return true;
}

}  // namespace gh
