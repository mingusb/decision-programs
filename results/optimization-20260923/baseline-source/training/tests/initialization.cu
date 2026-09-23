#include "ghb/initialization.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::size_t checks{};
void require(bool condition, const std::string& message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}
void close(double actual, long double expected, const char* message) {
  require(std::isfinite(actual) && std::abs(static_cast<long double>(actual) - expected)
      <= 3e-12L * std::max(1.L, std::abs(expected)), message);
}
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Stream {
  cudaStream_t value{};
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamSynchronize(value); cudaStreamDestroy(value); }
};
template<class T> struct Device {
  T* value{};
  std::size_t size;
  explicit Device(std::size_t count) : size(count) {
    check(cudaMalloc(reinterpret_cast<void**>(&value), size * sizeof(T)));
  }
  ~Device() { cudaFree(value); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  void put(const std::vector<T>& source, cudaStream_t stream) {
    require(source.size() == size, "test upload shape");
    check(cudaMemcpyAsync(value, source.data(), size * sizeof(T), cudaMemcpyHostToDevice, stream));
    check(cudaStreamSynchronize(stream));
  }
  std::vector<T> get(cudaStream_t stream) const {
    std::vector<T> result(size);
    check(cudaMemcpyAsync(result.data(), value, size * sizeof(T), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    return result;
  }
};

// Reference sums rows directly in extended precision. It shares neither the
// device reduction layout nor its chunk/reduction order.
std::vector<long double> reference(ghb::Objective objective, const std::vector<float>& targets,
                                   const std::vector<float>& weights, std::uint32_t rows,
                                   std::uint32_t outputs) {
  std::vector<long double> result(outputs);
  long double total{};
  for (std::uint32_t row = 0; row < rows; ++row) {
    const long double weight = weights.empty() ? 1 : weights[row];
    total += weight;
    for (std::uint32_t output = 0; output < outputs; ++output)
      result[output] += weight * (objective == ghb::Objective::multiclass_softmax
          ? static_cast<long double>(targets[row] == output) : targets[std::size_t(row) * outputs + output]);
  }
  for (auto& value : result) {
    value /= total;
    if (objective == ghb::Objective::binary_logistic) {
      // The public contract clamps in binary64 before computing log odds.
      const double probability = std::clamp(static_cast<double>(value), 1e-12, 1.0 - 1e-12);
      value = std::log(probability / (1.0 - probability));
    } else if (objective == ghb::Objective::multiclass_softmax)
      value = std::log(std::max(value, 1e-12L));
  }
  result.push_back(total);
  return result;
}

void valid_case(Stream& stream, ghb::Objective objective, std::uint32_t rows, std::uint32_t outputs,
                const std::vector<float>& targets, const std::vector<float>& weights,
                std::uint32_t chunks, bool graph) {
  constexpr double canary = 987654321.25;
  Device<float> d_targets(targets.size()), d_weights(rows);
  Device<double> base(outputs + 2), sum(3), scratch(std::size_t(chunks) * (outputs + 1) + 2);
  Device<unsigned> status(3);
  d_targets.put(targets, stream.value);
  d_weights.put(weights.empty() ? std::vector<float>(rows, 1) : weights, stream.value);
  base.put(std::vector<double>(base.size, canary), stream.value);
  sum.put(std::vector<double>(sum.size, canary), stream.value);
  scratch.put(std::vector<double>(scratch.size, canary), stream.value);
  status.put({1234567, 0xffffffffu, 7654321}, stream.value);
  auto invoke = [&] {
    return ghb::gpu::initialize_training(objective, d_targets.value, weights.empty() ? nullptr : d_weights.value,
        rows, outputs, base.value + 1, sum.value + 1, scratch.value + 1, chunks, status.value + 1, stream.value);
  };
  if (graph) {
    cudaGraph_t captured{}; cudaGraphExec_t executable{};
    check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
    check(invoke());
    check(cudaStreamEndCapture(stream.value, &captured));
    check(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
    check(cudaGraphLaunch(executable, stream.value));
    check(cudaStreamSynchronize(stream.value));
    // A second replay catches accumulation into stale partial/base storage.
    check(cudaGraphLaunch(executable, stream.value));
    check(cudaStreamSynchronize(stream.value));
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(captured));
  } else { check(invoke()); check(invoke()); }
  const auto actual = base.get(stream.value), weight_sum = sum.get(stream.value), partial = scratch.get(stream.value);
  const auto flags = status.get(stream.value);
  require(flags[1] == 0, "valid initialization did not reset device status");
  require(flags.front() == 1234567 && flags.back() == 7654321, "status canary changed");
  for (const auto* guarded : {&actual, &weight_sum, &partial})
    require(guarded->front() == canary && guarded->back() == canary, "initialization output canary changed");
  const auto expected = reference(objective, targets, weights, rows, outputs);
  for (std::uint32_t output = 0; output < outputs; ++output) close(actual[output + 1], expected[output], "weighted base score");
  close(weight_sum[1], expected.back(), "weight sum must count each row once, not each output");
}

void valid_inputs(Stream& stream) {
  for (auto objective : {ghb::Objective::squared_error, ghb::Objective::binary_logistic, ghb::Objective::multiclass_softmax}) {
    const std::uint32_t rows = 1031, outputs = objective == ghb::Objective::multiclass_softmax ? 7 : 65;
    std::vector<float> targets(objective == ghb::Objective::multiclass_softmax ? rows : std::size_t(rows) * outputs), weights(rows);
    for (std::uint32_t row = 0; row < rows; ++row) {
      weights[row] = row % 13 == 0 ? 0.f : row % 3 == 0 ? .25f : 3.f;
      if (objective == ghb::Objective::multiclass_softmax) targets[row] = float(row % (outputs - 1)); // Last class absent.
      else for (std::uint32_t output = 0; output < outputs; ++output)
        targets[std::size_t(row) * outputs + output] = objective == ghb::Objective::squared_error
            ? float(int((row * 11 + output * 7) % 31) - 12) * .125f
            : output == 0 ? 0.f : output == 1 ? 1.f : float((row + output * 3) % 5 == 0);
    }
    for (auto chunks : {1u, 3u, 17u})
      valid_case(stream, objective, rows, outputs, targets, weights, chunks, chunks == 3);
    valid_case(stream, objective, rows, outputs, targets, {}, 3, false);
    if (objective != ghb::Objective::multiclass_softmax) {
      std::vector<float> narrow(std::size_t(rows) * 7);
      for (std::uint32_t row = 0; row < rows; ++row)
        std::copy_n(targets.begin() + std::size_t(row) * outputs, 7, narrow.begin() + std::size_t(row) * 7);
      valid_case(stream, objective, rows, 7, narrow, weights, 3, true);
    }
  }
  const auto maximum = std::numeric_limits<float>::max();
  valid_case(stream, ghb::Objective::squared_error, 2, 2,
             {maximum, -maximum, -maximum, maximum}, {1, 3}, 11, false);
  valid_case(stream, ghb::Objective::multiclass_softmax, 5, 4,
             {2, 2, 2, 2, 2}, {1, 0, 2, 4, .5f}, 11, true);
}

void invalid_inputs(Stream& stream) {
  constexpr std::uint32_t rows = 4, outputs = 3, chunks = 3;
  Device<float> targets(rows * outputs), weights(rows);
  Device<double> base(outputs), sum(1), scratch(chunks * (outputs + 1));
  Device<unsigned> status(1);
  auto invoke = [&](ghb::Objective objective) {
    return ghb::gpu::initialize_training(objective, targets.value, weights.value, rows, outputs,
        base.value, sum.value, scratch.value, chunks, status.value, stream.value);
  };
  auto check_flags = [&](ghb::Objective objective, const std::vector<float>& labels,
                          const std::vector<float>& row_weights, unsigned required) {
    targets.put(labels, stream.value); weights.put(row_weights, stream.value);
    status.put({0xffffffffu}, stream.value); check(invoke(objective));
    const auto flag = status.get(stream.value)[0];
    require((flag & required) == required, "invalid input failed to set required device status");
    require((flag & ~15u) == 0, "initialization left stale or unknown status bits");
  };
  const std::vector<float> good(rows * outputs, 0), weighted{0, 1, 2, 3};
  for (const auto bad : {-1.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
    auto bad_weights = weighted; bad_weights[1] = bad;
    check_flags(ghb::Objective::squared_error, good, bad_weights, ghb::gpu::invalid_weight);
  }
  check_flags(ghb::Objective::squared_error, good, std::vector<float>(rows, 0), ghb::gpu::no_positive_weight);
  for (auto objective : {ghb::Objective::squared_error, ghb::Objective::binary_logistic, ghb::Objective::multiclass_softmax}) {
    for (const auto bad : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
      auto labels = good; labels[0] = bad; // Row zero has zero weight: still invalid.
      check_flags(objective, labels, weighted, ghb::gpu::invalid_target);
    }
    if (objective != ghb::Objective::squared_error) for (const auto bad : {-.5f, .5f, 3.f}) {
      auto labels = good; labels[0] = bad;
      check_flags(objective, labels, weighted, ghb::gpu::invalid_target);
    }
    targets.put(good, stream.value); weights.put(weighted, stream.value); check(invoke(objective));
    require(status.get(stream.value)[0] == 0, "valid call did not clear preceding invalid status");
  }
  auto invalid_call = [&](ghb::Objective objective, const float* y, std::uint32_t n, std::uint32_t k,
                          double* b, double* w, double* p, std::uint32_t c, unsigned* s) {
    require(ghb::gpu::initialize_training(objective, y, weights.value, n, k, b, w, p, c, s, stream.value)
            == cudaErrorInvalidValue, "invalid initialization launch arguments accepted");
  };
  for (unsigned mutation = 0; mutation < 10; ++mutation)
    invalid_call(mutation == 0 ? static_cast<ghb::Objective>(99) : mutation == 9 ? ghb::Objective::multiclass_softmax : ghb::Objective::squared_error,
        mutation == 1 ? nullptr : targets.value, mutation == 2 ? 0 : rows,
        mutation == 3 ? 0 : mutation == 9 ? 1 : outputs,
        mutation == 4 ? nullptr : base.value, mutation == 5 ? nullptr : sum.value,
        mutation == 6 ? nullptr : scratch.value, mutation == 7 ? 0 : chunks,
        mutation == 8 ? nullptr : status.value);
}
} // namespace

int main() {
  try {
    int devices{}; const auto status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
      std::cout << "SKIP: no CUDA device/driver\n"; return 77;
    }
    check(status);
    Stream stream; valid_inputs(stream); invalid_inputs(stream);
    std::cout << "Passed " << checks << " independent weighted initialization, validation, and capture checks.\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << "initialization correctness: " << error.what() << '\n'; return 1; }
}
