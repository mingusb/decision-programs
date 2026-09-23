#include "ghb/instrumentation.hpp"

#include <type_traits>

using namespace ghb::instrumentation;
static_assert(std::is_empty_v<NullRecorder>);
static_assert(std::is_trivially_copyable_v<NullRecorder>);
static_assert(!NullRecorder::nvtx_available());
static_assert([] {
  NullRecorder recorder;
  const auto ticket = recorder.begin(Stage::histogram);
  recorder.end(ticket);
  recorder.reset();
  return recorder.ready() && recorder.size() == 0 && recorder.capacity() == 0;
}());

int main() {
  NullRecorder recorder(100, true);
  // Runtime exercise of the same API, linked without Recorder or CUDA runtime.
  for (int index = 0; index < 100; ++index) {
    const auto ticket = recorder.begin(Stage::histogram, {}, Timing::gpu);
    recorder.end(ticket);
  }
  const auto samples = recorder.collect(true);
  return samples && samples->empty() && recorder.ready() ? 0 : 1;
}
