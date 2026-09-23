#include "references.hpp"

#include <cub/device/device_histogram.cuh>
#include <limits>

namespace ghbench {
namespace {
template <typename Function>
cudaError_t dispatch_types(const gh::Config& config, Function function) {
  if (config.input_type == gh::InputType::u8) {
    if (config.counter_type == gh::CounterType::u32)
      return function.template operator()<unsigned char, unsigned>();
    return function.template operator()<unsigned char, unsigned long long>();
  }
  if (config.counter_type == gh::CounterType::u32)
    return function.template operator()<unsigned, unsigned>();
  return function.template operator()<unsigned, unsigned long long>();
}

template <typename Input, typename Counter>
cudaError_t cub_histogram(const gh::Config& config, const void* input, void* output,
                          void* workspace, std::size_t& bytes, cudaStream_t stream) {
  return cub::DeviceHistogram::HistogramEven(
      workspace, bytes, static_cast<const Input*>(input), static_cast<Counter*>(output),
      static_cast<int>(config.bins + 1), 0u, config.bins, config.size, stream);
}
}  // namespace

bool supported(Reference reference, const gh::Config& config) {
  if (reference != Reference::cub && reference != Reference::nvidia_sample256) return false;
  if (config.input_type != gh::InputType::u8 && config.input_type != gh::InputType::u32) return false;
  if (config.counter_type != gh::CounterType::u32 && config.counter_type != gh::CounterType::u64) return false;
  if (config.local_counter != gh::LocalCounter::native) return false;
  if (config.output_clear != gh::OutputClear::runtime && config.output_clear != gh::OutputClear::kernel) return false;
  if (config.launch != gh::LaunchMode::stream && config.launch != gh::LaunchMode::graph) return false;
  if (config.cache != gh::CacheMode::warm && config.cache != gh::CacheMode::cold) return false;
  if (!config.bins || config.bins > static_cast<unsigned>(std::numeric_limits<int>::max() - 1)) return false;
  if (config.input_type == gh::InputType::u8 && config.bins > 256) return false;
  if (config.counter_type == gh::CounterType::u32 && config.size > std::numeric_limits<unsigned>::max()) return false;
  if (config.size > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) /
                        gh::input_bytes(config.input_type)) return false;
  if (config.blocks <= 0 || config.tuning < 0 ||
      config.tuning >= static_cast<int>(gh::tuning_count)) return false;
  // Preserve the prior benchmark's accepted scalar metadata policies. NVIDIA's
  // implementations choose their own launch policy and ignore these knobs.
  const auto policy = gh::tuning_catalog[config.tuning];
  if (policy.load != gh::LoadPolicy::scalar || policy.shared_limit > 48 * 1024) return false;
  return reference == Reference::cub || detail::sample256_supported(config);
}

cudaError_t workspace_bytes(Reference reference, const gh::Config& config, std::size_t& bytes) {
  bytes = 0;
  if (!supported(reference, config)) return cudaErrorInvalidValue;
  if (reference == Reference::nvidia_sample256)
    return detail::sample256_workspace_bytes(config, bytes);
  if (!config.size) return cudaSuccess;
  return dispatch_types(config, [&]<typename Input, typename Counter>() {
    return cub_histogram<Input, Counter>(config, nullptr, nullptr, nullptr, bytes, nullptr);
  });
}

cudaError_t histogram(Reference reference, const gh::Config& config,
                      const void* input, void* output, void* workspace,
                      std::size_t bytes, cudaStream_t stream) {
  if (!supported(reference, config) || output == nullptr || (config.size && input == nullptr))
    return cudaErrorInvalidValue;
  if (reference == Reference::nvidia_sample256)
    return detail::launch_sample256(config, input, output, workspace, bytes, stream);
  if (!config.size)
    return cudaMemsetAsync(output, 0, static_cast<std::size_t>(config.bins) *
                          gh::counter_bytes(config.counter_type), stream);
  if (workspace == nullptr || !bytes) return cudaErrorInvalidValue;
  return dispatch_types(config, [&]<typename Input, typename Counter>() {
    return cub_histogram<Input, Counter>(config, input, output, workspace, bytes, stream);
  });
}
}  // namespace ghbench
