#include "deeper_histogram_fixture.hpp"
#include <iostream>

namespace {
using namespace deeper_test;
void cases() {
  const std::vector<Shape> shapes{
      {1, 1, 1, 1, 1},
      {33, 7, 2, 2, 16, 0, false, false, true, true},
      {259, 5, 3, 2, 32, 75, true, false, true, true},
      {1027, 4, 4, 8, 16, 0, false, false, true, false},
      {1031, 4, 7, 2, 32, 0, true, true, true, true},
      {517, 3, 8, 2, 64, 75, false, false, true, true},
      {1024, 7, 16, 2, 32, 0, false, false, true, false},
      {271, 3, 17, 2, 32, 100, true, false, true, false},
      {4097, 3, 33, 2, 16, 75, true, false, true, true},
      {521, 3, 3, 32, 64, 0, false, false, true, true},
      {257, 3, 3, 8, 256, 75, false, false, true, true},
      {47, 2, 35, 2, 257, 0, true, false, true, false},
      // shared1 with four chunks has 66820 tasks, exercising grid-stride CTA
      // scratch reuse beyond the 65535 launch cap with a tiny input matrix.
      {1, 257, 65, 1, 1},
  };
  for (const auto s : shapes) {
    Fixture f(s);
    for (const auto& v : f.variants()) {
      f.reset(); f.launch(v); f.launch(v); f.verify(v.name);
      if (v.sequential) continue;
      Graph graph;
      check(cudaStreamBeginCapture(f.stream.value, cudaStreamCaptureModeThreadLocal));
      f.launch(v);
      check(cudaStreamEndCapture(f.stream.value, &graph.value));
      check(cudaGraphInstantiate(&graph.executable, graph.value, 0));
      const auto original = f.active;
      // Replay one captured graph with changed device active counts, including
      // zero counts. No host selector or launch dimensions are changed.
      for (unsigned replay = 0; replay < 3; ++replay) {
        for (unsigned out = 0; out < s.outputs; ++out)
          f.active[out] = replay == 1 ? 0 : replay == 2 ? s.capacity : original[out];
        f.d_active.put(f.active, f.stream.value); f.reference(); f.reset();
        check(cudaGraphLaunch(graph.executable, f.stream.value)); f.verify(v.name + " graph replay");
      }
      f.active = original; f.d_active.put(f.active, f.stream.value); f.reference();
    }
  }
}
void invalid() {
  Fixture f({17, 2, 3, 2, 16});
  auto call = [&](ghb::gpu::DataView data, unsigned stride, unsigned first,
                  unsigned count, unsigned capacity, Policy policy, unsigned chunks,
                  const int* assignments, const unsigned* active, Stats* output) {
    return ghb::gpu::deeper_histogram(data, assignments, f.d_g.p, f.d_h.p,
        stride, first, count, capacity, active, output, policy, chunks, f.stream.value);
  };
  auto bad = [&](ghb::gpu::DataView data, unsigned stride, unsigned first,
                 unsigned count, unsigned capacity, Policy policy, unsigned chunks) {
    require(call(data, stride, first, count, capacity, policy, chunks,
                 f.d_assignment.p, f.d_active.p, f.d_result.p + 1) == cudaErrorInvalidValue, "invalid host shape accepted");
  };
  bad(f.data, 0, 0, 1, 2, Policy::global, 0);
  bad(f.data, f.stride, f.stride, 1, 2, Policy::global, 0);
  bad(f.data, f.stride, 2, f.stride, 2, Policy::global, 0);
  bad(f.data, f.stride, 2, 0, 2, Policy::global, 0);
  bad(f.data, f.stride, 2, 3, 0, Policy::global, 0);
  bad(f.data, f.stride, 2, 3, unsigned(INT32_MAX) + 1, Policy::global, 0);
  bad(f.data, f.stride, 2, 3, 2, Policy::global, 1);
  bad(f.data, f.stride, 2, 3, 2, Policy::shared4, 0);
  bad(f.data, f.stride, 2, 3, 2, Policy::shared4, 257);
  bad(f.data, f.stride, 2, 3, 2, Policy(99), 0);
  bad(f.data, f.stride, 2, 3, 257, Policy::shared8, 1);
  auto data = f.data; data.rows = 0; bad(data, f.stride, 2, 3, 2, Policy::global, 0);
  data = f.data; data.bins = nullptr; bad(data, f.stride, 2, 3, 2, Policy::global, 0);
  data = f.data; data.max_feature_bins = 65537; bad(data, f.stride, 2, 3, 2, Policy::global, 0);
  for (unsigned pointer = 0; pointer < 3; ++pointer)
    require(call(f.data, f.stride, 2, 3, 2, Policy::global, 0,
                 pointer == 0 ? nullptr : f.d_assignment.p, pointer == 1 ? nullptr : f.d_active.p,
                 pointer == 2 ? nullptr : f.d_result.p + 1) == cudaErrorInvalidValue, "null pointer accepted");
  require(!ghb::gpu::deeper_histogram_supported(f.data, 257, Policy::shared8), "unsupported shared shape");
  for (const auto& value : f.d_result.get(f.stream.value))
    require(equal(value, sentinel()), "invalid requests changed destination");
}
} // namespace
int main() {
  try {
    int devices{}; const auto error = cudaGetDeviceCount(&devices);
    if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver || (error == cudaSuccess && !devices)) return 77;
    check(error); cases(); invalid();
    std::cout << "deeper histogram: " << checks << " checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "deeper histogram failed: " << error.what() << '\n'; return 1;
  }
}
