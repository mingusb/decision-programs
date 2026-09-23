#include "ghb/booster.hpp"
#include <cuda_runtime_api.h>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
bool observing{};
std::size_t allocation_calls{}, copy_calls{}, fail_allocation{}, fail_copy{}, checks{};
std::size_t host_calls{}, fail_host{};
std::array<void*, 64> live{};
bool bookkeeping_error{};
void require(bool condition, const char* message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}
void begin(std::size_t allocation = 0, std::size_t copy = 0, std::size_t host = 0) {
  require(!observing, "nested diagnostic");
  for (auto* pointer : live) require(!pointer, "device allocation survived preceding call");
  allocation_calls = copy_calls = host_calls = 0;
  fail_allocation = allocation; fail_copy = copy; fail_host = host;
  observing = true;
}
void end() {
  observing = false;
  require(!bookkeeping_error, "allocation tracking failure");
  for (auto* pointer : live) require(!pointer, "prediction leaked tracked allocation");
}
void exact(const std::vector<double>& a, const std::vector<double>& b) {
  require(a.size() == b.size(), "prediction extent changed");
  for (std::size_t i = 0; i < a.size(); ++i)
    require(std::bit_cast<std::uint64_t>(a[i]) == std::bit_cast<std::uint64_t>(b[i]),
            "prediction bits changed after injected failure");
}
}

extern "C" cudaError_t __real_cudaMalloc(void**, std::size_t);
extern "C" cudaError_t __real_cudaFree(void*);
extern "C" cudaError_t __real_cudaMemcpyAsync(void*, const void*, std::size_t, cudaMemcpyKind, cudaStream_t);
extern "C" void* __real__ZnamSt11align_val_t(std::size_t, std::align_val_t);
extern "C" void* __wrap__ZnamSt11align_val_t(std::size_t bytes, std::align_val_t alignment) {
  if (observing && ++host_calls == fail_host) throw std::bad_alloc();
  return __real__ZnamSt11align_val_t(bytes, alignment);
}
extern "C" cudaError_t __wrap_cudaMalloc(void** pointer, std::size_t bytes) {
  if (observing && ++allocation_calls == fail_allocation) return cudaErrorMemoryAllocation;
  const auto status = __real_cudaMalloc(pointer, bytes);
  if (observing && status == cudaSuccess && *pointer) {
    if (reinterpret_cast<std::uintptr_t>(*pointer) % 256 != 0) bookkeeping_error = true;
    bool found = false;
    for (auto& entry : live) if (!entry) { entry = *pointer; found = true; break; }
    if (!found) bookkeeping_error = true;
  }
  return status;
}
extern "C" cudaError_t __wrap_cudaFree(void* pointer) {
  const auto status = __real_cudaFree(pointer);
  if (observing && pointer) {
    bool found = false;
    for (auto& entry : live) if (entry == pointer) {
      if (status == cudaSuccess) entry = nullptr;
      found = true; break;
    }
    if (!found || status != cudaSuccess) bookkeeping_error = true;
  }
  return status;
}
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* destination, const void* source,
    std::size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
  if (observing && ++copy_calls == fail_copy) return cudaErrorInvalidValue;
  return __real_cudaMemcpyAsync(destination, source, bytes, kind, stream);
}

int main() {
  try {
    require(cudaFree(nullptr) == cudaSuccess, "CUDA context unavailable");
    ghb::Model model;
    model.objective = ghb::Objective::binary_logistic;
    model.outputs = 3;
    model.base_scores = {-0.0, .25, -.5};
    model.features = {{ghb::FeatureType::numeric, {-1, 0, 1}, {}},
                      {ghb::FeatureType::categorical, {}, {10, 20}}};
    ghb::Dataset data;
    data.rows = 33; data.columns = 2;
    data.feature_types = {ghb::FeatureType::numeric, ghb::FeatureType::categorical};
    for (unsigned row = 0; row < data.rows; ++row) {
      data.values.push_back(float(int(row % 5) - 2));
      data.values.push_back(row % 2 ? 10.f : 20.f);
    }
    ghb::Node leaf; leaf.value = .125;
    for (const bool forest : {false, true}) {
      if (forest) model.trees = {{2, {leaf}}, {0, {leaf}}, {2, {leaf}}};
      const auto reference = model.predict_gpu(data, false, ghb::PredictionPolicy::per_tree);
      const auto raw_reference = model.predict_gpu(data, true, ghb::PredictionPolicy::per_tree);
      auto predict = [&] { return model.predict_gpu(data, false, ghb::PredictionPolicy::fused_output_slab); };
      begin(); const auto actual = predict();
      const auto allocations = allocation_calls, copies = copy_calls, hosts = host_calls;
      end(); exact(actual, reference);
      require(allocations == 4, "expected four whole-call E allocations");
      require(copies == 9, "expected all nine quantizer/model/result asynchronous copies");
      require(hosts == 1, "expected one aligned host slab allocation");
      for (unsigned type = 0; type < 3; ++type) {
        const auto count = type == 0 ? allocations : type == 1 ? copies : hosts;
        for (std::size_t position = 1; position <= count; ++position) {
          begin(type == 0 ? position : 0, type == 1 ? position : 0, type == 2 ? position : 0);
          bool threw = false;
          try { (void)predict(); }
          catch (const std::runtime_error&) { threw = type != 2; }
          catch (const std::bad_alloc&) { threw = type == 2; }
          const auto observed = type == 0 ? allocation_calls : type == 1 ? copy_calls : host_calls;
          end();
          require(threw, "injected API error did not propagate");
          require(observed == position, "injection position not reached exactly");
          begin(); const auto recovered = predict(); end(); exact(recovered, reference);
          begin(); const auto raw_recovered = model.predict_gpu(data, true, ghb::PredictionPolicy::fused_output_slab);
          end(); exact(raw_recovered, raw_reference);
        }
      }
      std::cout << "forest=" << forest << " allocation_positions=" << allocations
                << " copy_positions=" << copies << " host_positions=" << hosts << '\n';
    }
    std::cout << "fault validation passed: " << checks << " checks\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fault validation failed: " << error.what() << '\n';
    return 1;
  }
}
