#include "ghb/instrumentation.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

using namespace ghb::instrumentation;
namespace {

std::size_t checks{};
void require(bool condition, const char* message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}
void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template <typename Function>
void rejected(Function function) {
  bool threw = false;
  try { function(); } catch (const std::exception&) { threw = true; }
  require(threw, "invalid instrumentation operation was accepted");
}

struct Stream {
  cudaStream_t value{};
  Stream() { cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
};
struct Buffer {
  void* data{};
  std::size_t bytes;
  explicit Buffer(std::size_t bytes) : bytes(bytes) { cuda_check(cudaMalloc(&data, bytes)); }
  ~Buffer() { cudaFree(data); }
};
struct Graph {
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  ~Graph() {
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
  }
  void end_capture(cudaStream_t stream) {
    cuda_check(cudaStreamEndCapture(stream, &graph));
    cuda_check(cudaGraphInstantiateWithFlags(&executable, graph, 0));
  }
};
struct Gate {
  std::atomic<bool> entered{false}, released{false};
  cudaStream_t stream;
  explicit Gate(cudaStream_t stream) : stream(stream) {
    cuda_check(cudaLaunchHostFunc(stream, [](void* pointer) {
      auto& gate = *static_cast<Gate*>(pointer);
      gate.entered.store(true, std::memory_order_release);
      while (!gate.released.load(std::memory_order_acquire)) std::this_thread::yield();
    }, this));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!entered.load(std::memory_order_acquire)) {
      if (std::chrono::steady_clock::now() > deadline) {
        released.store(true, std::memory_order_release);
        cudaStreamSynchronize(stream);
        throw std::runtime_error("GPU stream gate did not start");
      }
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }
  void release() { released.store(true, std::memory_order_release); }
  ~Gate() { release(); cudaStreamSynchronize(stream); }
};

void lifecycle(bool nvtx) {
  Recorder empty(0, nvtx);
  require(empty.ready() && empty.size() == 0 && empty.capacity() == 0, "empty recorder state");
  require(empty.collect(false)->empty(), "empty collection");
  rejected([&] { empty.begin(Stage::histogram, {}, Timing::host); });

  Recorder recorder(2, nvtx), other(1, nvtx);
  Stream stream;
  rejected([&] { recorder.begin(Stage::count, {}, Timing::host, stream.value); });
  rejected([&] { recorder.begin(static_cast<Stage>(999), {}, Timing::host, stream.value); });
  rejected([&] { recorder.begin(Stage::upload, {}, static_cast<Timing>(999), stream.value); });
  require(recorder.size() == 0 && recorder.ready(), "rejected begin consumed capacity");
  Context context;
  context.round = 7; context.depth = 2; context.output = 3; context.repetition = 4;
  context.active_nodes = 8; context.features = 9; context.bins = 10; context.stream_id = 11;
  context.rows = 123; context.logical_read_bytes = 234; context.logical_write_bytes = 345;
  context.scratch_bytes = 456; context.operations = 16;
  const auto first = recorder.begin(Stage::quantize, context, Timing::host, stream.value);
  require(!recorder.ready(), "open scope reported ready");
  rejected([&] { recorder.collect(false); });
  rejected([&] { recorder.collect(true); });
  rejected([&] { recorder.reset(); });
  rejected([&] { other.end(first); });
  auto forged = first; ++forged.index;
  rejected([&] { recorder.end(forged); });
  recorder.end(first);
  rejected([&] { recorder.end(first); });
  const auto second = recorder.begin(Stage::checkpoint, {}, Timing::host, stream.value);
  rejected([&] { recorder.begin(Stage::upload, {}, Timing::host, stream.value); });
  recorder.end(second);
  require(recorder.size() == 2 && recorder.ready(), "closed host scopes not ready");
  auto collected = recorder.collect(false);
  require(collected && collected->size() == 2, "host collection count");
  const auto& sample = collected->front();
  require(sample.id != collected->back().id, "sample IDs reused");
  require(sample.stage == Stage::quantize && sample.timing == Timing::host && !sample.gpu_ms,
          "host sample type");
  require(sample.host_end_ns >= sample.host_start_ns, "host clock ordering");
  require(sample.context.round == 7 && sample.context.depth == 2 && sample.context.output == 3 &&
      sample.context.repetition == 4 && sample.context.active_nodes == 8 && sample.context.features == 9 &&
      sample.context.bins == 10 && sample.context.stream_id == 11 && sample.context.rows == 123 &&
      sample.context.logical_read_bytes == 234 && sample.context.logical_write_bytes == 345 &&
      sample.context.scratch_bytes == 456 && sample.context.operations == 16, "sample context changed");
  require(recorder.collect(true)->front().host_end_ns == sample.host_end_ns, "collection mutated sample");
  const auto previous_id = collected->back().id;
  recorder.reset();
  require(recorder.size() == 0 && recorder.capacity() == 2, "reset changed capacity");
  const auto fresh = recorder.begin(Stage::evaluate, {}, Timing::host, stream.value);
  rejected([&] { recorder.end(first); });
  recorder.end(fresh);
  require(recorder.collect(false)->front().id > previous_id, "reset reused sample identity");
  bool wrong_thread_rejected = false;
  std::thread wrong_thread([&] {
    try { recorder.ready(); } catch (const std::logic_error&) { wrong_thread_rejected = true; }
  });
  wrong_thread.join();
  require(wrong_thread_rejected, "cross-thread recorder access accepted");
}

void pending_and_streams(bool nvtx) {
  Stream first, second;
  Buffer one(1 << 20), two(1 << 20);
  Recorder recorder(3, nvtx);
  Context context; context.stream_id = 1;
  const auto a = recorder.begin(Stage::upload, context, Timing::gpu, first.value);
  cuda_check(cudaMemsetAsync(one.data, 0x31, one.bytes, first.value));
  {
    Gate gate(first.value);
    recorder.end(a);
    require(!recorder.ready(), "blocked end event reported ready");
    cuda_check(cudaPeekAtLastError());
    require(!recorder.collect(false), "nonblocking collect returned pending timestamps");
    cuda_check(cudaPeekAtLastError());
    rejected([&] { recorder.reset(); });
    cuda_check(cudaPeekAtLastError());
    require(recorder.size() == 1, "failed reset discarded pending sample");
    context.stream_id = 2;
    const auto b = recorder.begin(Stage::download, context, Timing::gpu, second.value);
    cuda_check(cudaMemsetAsync(two.data, 0x72, two.bytes, second.value));
    cuda_check(cudaGetLastError());
    recorder.end(b);
    gate.release();
  }
  const auto samples = recorder.collect(true);
  require(samples && samples->size() == 2 && recorder.ready(), "multi-stream collection failed");
  require(samples->at(0).context.stream_id == 1 && samples->at(1).context.stream_id == 2,
          "stream metadata changed");
  for (const auto& sample : *samples)
    require(sample.gpu_ms && std::isfinite(*sample.gpu_ms) && *sample.gpu_ms >= 0,
            "invalid collected GPU interval");
  std::array<unsigned char, 16> copied{};
  cuda_check(cudaMemcpy(copied.data(), one.data, copied.size(), cudaMemcpyDeviceToHost));
  require(std::all_of(copied.begin(), copied.end(), [](auto byte) { return byte == 0x31; }), "first stream result");
  cuda_check(cudaMemcpy(copied.data(), two.data, copied.size(), cudaMemcpyDeviceToHost));
  require(std::all_of(copied.begin(), copied.end(), [](auto byte) { return byte == 0x72; }), "second stream result");
  recorder.reset();
  const auto x = recorder.begin(Stage::upload, {}, Timing::gpu, first.value);
  const auto y = recorder.begin(Stage::download, {}, Timing::gpu, second.value);
  cuda_check(cudaMemsetAsync(one.data, 0, one.bytes, first.value));
  cuda_check(cudaMemsetAsync(two.data, 0, two.bytes, second.value));
  recorder.end(y); recorder.end(x);
  require(recorder.collect(true)->size() == 2, "overlapping scopes require stack order");
}

void capture_and_replay(bool nvtx) {
  Stream stream;
  Buffer input(1 << 20), output(1 << 20);
  Recorder recorder(4, nvtx);
  Graph graph;
  const auto spanning = recorder.begin(Stage::initialize, {}, Timing::gpu, stream.value);
  cuda_check(cudaStreamSynchronize(stream.value));
  cuda_check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
  rejected([&] { recorder.begin(Stage::prediction, {}, Timing::gpu, stream.value); });
  rejected([&] { recorder.begin(Stage::quantize, {}, Timing::host, stream.value); });
  rejected([&] { recorder.end(spanning); });
  require(recorder.size() == 1, "capture rejection consumed sample slots");
  cuda_check(cudaMemcpyAsync(output.data, input.data, input.bytes, cudaMemcpyDeviceToDevice, stream.value));
  graph.end_capture(stream.value);
  // The rejected instrumentation calls must not invalidate capture or make
  // the existing ticket unusable after capture has ended.
  cuda_check(cudaMemsetAsync(input.data, 0x19, input.bytes, stream.value));
  cuda_check(cudaGraphLaunch(graph.executable, stream.value));
  recorder.end(spanning);
  require(recorder.collect(true)->size() == 1, "capture rejection poisoned recorder");
  recorder.reset();

  for (int phase = 0; phase < 3; ++phase) {
    const unsigned char pattern = static_cast<unsigned char>(0x20 + phase);
    cuda_check(cudaMemsetAsync(input.data, pattern, input.bytes, stream.value));
    Context context; context.repetition = phase;
    const auto ticket = recorder.begin(Stage::prediction, context, Timing::gpu, stream.value);
    cuda_check(cudaGraphLaunch(graph.executable, stream.value));
    recorder.end(ticket);
    std::array<unsigned char, 16> actual{};
    cuda_check(cudaMemcpyAsync(actual.data(), output.data, actual.size(), cudaMemcpyDeviceToHost, stream.value));
    cuda_check(cudaStreamSynchronize(stream.value));
    require(std::all_of(actual.begin(), actual.end(), [pattern](auto byte) { return byte == pattern; }),
            "graph replay did not use changed input");
  }
  const auto samples = recorder.collect(false);
  require(samples && samples->size() == 3, "graph replay intervals missing");
  for (std::size_t index = 0; index < samples->size(); ++index) {
    const auto& sample = samples->at(index);
    require(sample.context.repetition == static_cast<int>(index) && sample.gpu_ms &&
        std::isfinite(*sample.gpu_ms) && *sample.gpu_ms >= 0, "graph replay sample metadata");
    if (index) require(sample.id != samples->at(index - 1).id, "graph replay reused an event slot");
  }
}

}  // namespace

int main() try {
  int devices{};
  const auto available = cudaGetDeviceCount(&devices);
  if (available == cudaErrorNoDevice || available == cudaErrorInsufficientDriver ||
      (available == cudaSuccess && devices == 0)) {
    std::cout << "SKIP: CUDA device or driver unavailable.\n";
    return 77;
  }
  cuda_check(available);
  for (bool nvtx : {false, true}) {
    lifecycle(nvtx);
    pending_and_streams(nvtx);
    capture_and_replay(nvtx);
  }
  require(name(Stage::histogram) == "histogram" && name(Stage::histogram_subtract) == "histogram_subtract",
          "stage names changed");
  require(name(Stage::count) == "unknown", "invalid stage name");
  std::cout << "PASS: " << checks << " instrumentation lifecycle, deferred timing, capture, replay, and stream checks.\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "FAIL: " << error.what() << '\n';
  return 1;
}
