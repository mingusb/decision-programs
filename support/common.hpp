#pragma once
#include "gh/histogram.hpp"
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

inline void check(cudaError_t status, const char* expr, int line) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(expr) + " at line " + std::to_string(line) + ": " + cudaGetErrorString(status));
}
#define CUDA_CHECK(x) check((x), #x, __LINE__)

struct DeviceBuffer {
  void* data = nullptr;
  std::size_t bytes;
  explicit DeviceBuffer(std::size_t n) : bytes(std::max<std::size_t>(n, 1)) {
    CUDA_CHECK(cudaMalloc(&data, bytes));
  }
  ~DeviceBuffer() { cudaFree(data); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
struct Stream {
  cudaStream_t value{};
  Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
};
struct Event {
  cudaEvent_t value{};
  Event() { CUDA_CHECK(cudaEventCreate(&value)); }
  ~Event() { cudaEventDestroy(value); }
};

inline constexpr gh::Algorithm algorithms[] = {
  gh::Algorithm::global_atomic, gh::Algorithm::warp_aggregated,
  gh::Algorithm::shared_atomic, gh::Algorithm::shared_rle, gh::Algorithm::shared_warp,
  gh::Algorithm::shared_partial, gh::Algorithm::bitplane, gh::Algorithm::shared_overflow};

struct Dataset {
  std::vector<unsigned> keys;
  std::vector<std::uint64_t> expected;
  std::vector<unsigned char> packed;
  Dataset(std::size_t n, unsigned bins, gh::InputType input_type,
          const std::string& distribution, const std::string& order, std::uint64_t seed)
      : keys(n), expected(bins) {
    if (!bins || (input_type == gh::InputType::u8 && bins > 256))
      throw std::runtime_error("invalid bin count for input type");
    // An explicit benchmark distribution: preserve the historical hot99 RNG
    // sequence while moving only the forced dominant value. No input sampling.
    unsigned hot_bin = bins - 1;
    const bool located_hot99 = distribution.starts_with("hot99@");
    if (located_hot99) {
      const char* first = distribution.data() + 6;
      const char* last = distribution.data() + distribution.size();
      const auto parsed = std::from_chars(first, last, hot_bin);
      if (parsed.ec != std::errc{} || parsed.ptr != last || hot_bin >= bins)
        throw std::runtime_error("hot99@BIN requires a valid decimal bin ID");
    }
    std::mt19937_64 rng(seed);
    for (auto& key : keys) {
      unsigned candidate = static_cast<unsigned>(rng() % bins);
      if (distribution == "uniform") key = candidate;
      else if (distribution == "single") key = bins - 1;
      else if (distribution == "two") key = (rng() & 1) ? 0 : bins - 1;
      else if (distribution == "hot90" || distribution == "hot99" || located_hot99) {
        const unsigned percentage = distribution == "hot90" ? 90 : 99;
        key = rng() % 100 < percentage ? hot_bin : candidate;
      } else throw std::runtime_error("unknown distribution: " + distribution);
      ++expected[key];
    }
    if (order == "sorted") std::sort(keys.begin(), keys.end());
    else if (order == "roundrobin") {
      auto remaining = expected;
      std::deque<unsigned> active;
      for (unsigned b = 0; b < bins; ++b) if (remaining[b]) active.push_back(b);
      for (auto& key : keys) {
        key = active.front(); active.pop_front();
        if (--remaining[key]) active.push_back(key);
      }
    } else if (order != "shuffled") throw std::runtime_error("unknown order: " + order);
    if (input_type == gh::InputType::u8) {
      packed.resize(n);
      std::transform(keys.begin(), keys.end(), packed.begin(), [](unsigned x) { return static_cast<unsigned char>(x); });
    }
  }
  const void* data(gh::InputType type) const { return type == gh::InputType::u8 ? static_cast<const void*>(packed.data()) : keys.data(); }
};

inline void verify_output(const gh::Config& config, const void* output,
                          const std::vector<std::uint64_t>& expected) {
  std::vector<std::uint64_t> actual(config.bins);
  if (config.counter_type == gh::CounterType::u32) {
    std::vector<unsigned> small(config.bins);
    CUDA_CHECK(cudaMemcpy(small.data(), output, small.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
    std::copy(small.begin(), small.end(), actual.begin());
  } else {
    CUDA_CHECK(cudaMemcpy(actual.data(), output, actual.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
  }
  for (unsigned b = 0; b < config.bins; ++b) {
    if (actual[b] != expected[b])
      throw std::runtime_error(std::string(gh::name(config.algorithm)) + " mismatch bin " + std::to_string(b) +
                               " actual=" + std::to_string(actual[b]) + " expected=" + std::to_string(expected[b]) +
                               " n=" + std::to_string(config.size) + " bins=" + std::to_string(config.bins) +
                               " tuning=" + std::to_string(config.tuning));
  }
  if (std::accumulate(actual.begin(), actual.end(), std::uint64_t{0}) != config.size)
    throw std::runtime_error("histogram count conservation failed");
}
