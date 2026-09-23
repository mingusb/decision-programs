#pragma once
#include "gh/core.cuh"

namespace gh::count {
enum class InputType { u8, u32 };
enum class CounterType { u32, u64 };
enum class LocalCounter { native, u32 };
enum class LoadPolicy { scalar, full_tile, vector4 };
enum class OutputClear { runtime, kernel };
enum class LaunchMode { stream, graph };
enum class CacheMode { warm, cold };
enum class Algorithm { global_atomic, warp_aggregated, shared_atomic, shared_rle,
  shared_warp, shared_partial, bitplane, automatic, shared_overflow, global_window };

struct Tuning { unsigned threads{}, items{}, replicas{}; LoadPolicy load{}; u32 shared_limit{48 * 1024}; };
inline constexpr unsigned tuning_count = 16;
__host__ __device__ constexpr Tuning tuning(unsigned index) {
  constexpr Tuning policies[] = {
    {128,4,1}, {256,4,1}, {256,8,1}, {256,16,1}, {128,8,4}, {256,8,4},
    {256,8,1,LoadPolicy::full_tile}, {256,8,1,LoadPolicy::vector4},
    {256,16,1,LoadPolicy::full_tile}, {256,16,1,LoadPolicy::vector4},
    {512,8,1,LoadPolicy::vector4}, {1024,8,1,LoadPolicy::vector4},
    {256,8,8,LoadPolicy::vector4}, {512,8,16,LoadPolicy::vector4},
    {256,8,1,LoadPolicy::vector4,96 * 1024}, {512,8,1,LoadPolicy::vector4,96 * 1024}};
  return index < tuning_count ? policies[index] : Tuning{};
}
struct Config {
  Algorithm algorithm{Algorithm::automatic};
  InputType input_type{InputType::u32};
  CounterType counter_type{CounterType::u32};
  LocalCounter local_counter{LocalCounter::native};
  u64 size{};
  u32 bins{256}, blocks{192}, policy{2};
  OutputClear output_clear{OutputClear::runtime};
  LaunchMode launch{LaunchMode::stream};
  CacheMode cache{CacheMode::warm};
  u32 window_bins{524288};
};
// Runtime identity/capabilities are supplied infrastructure facts. Defaults
// describe the pinned target; callers on another device must supply its facts.
struct Hardware {
  bool a5000_laptop{true};
  u32 major{8}, minor{6}, sms{48}, max_threads{1024}, max_grid{2147483647};
  u32 shared_bytes{48 * 1024}, shared_optin_bytes{96 * 1024};
};
__device__ bool supported(Config config);
__device__ Config resolve(Config config, Hardware hardware = {});
// Requires a resolved supported configuration; no allocation or CUDA effects.
__device__ u64 required_bytes(Config config);
__device__ cudaError_t count(Config config, Array<const std::byte> input,
    Array<std::byte> output, Workspace workspace, Status* status, Hardware hardware = {});
// Mandatory current-context setup for policy14/15. Only CUDA attribute calls;
// perform before launches/capture, and again after context reset/device change.
cudaError_t initialize_runtime();
}  // namespace gh::count
