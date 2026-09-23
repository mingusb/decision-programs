#include "../tests/deeper_histogram_fixture.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>

namespace {
using namespace deeper_test;
void benchmark(unsigned selected) {
  const std::vector<Shape> shapes{
      {4096, 16, 1, 2, 32},
      {4096, 16, 3, 2, 32, 75, false, false, true, true},
      {4096, 16, 16, 2, 32},
      {4096, 16, 16, 2, 32, 0, true},
      {4096, 16, 16, 2, 32, 75},
      {4096, 16, 16, 2, 32, 0, false, true},
      {1024, 16, 16, 2, 16},
      {1024, 16, 16, 8, 16, 0, false, false, true, true},
      {4096, 16, 33, 2, 32, 0, false, false, true, true},
      {4096, 16, 16, 8, 32},
      {4096, 16, 16, 32, 64, 75, false, false, true, true},
      {4096, 16, 16, 8, 256, 0, true},
      {65536, 32, 1, 2, 64},
      {65536, 16, 16, 2, 32},
      {65536, 16, 16, 8, 16, 75},
      {4096, 32, 16, 2, 32},
  };
  constexpr unsigned samples = 7, warmups = 2;
  if (selected != unsigned(-1) && selected >= shapes.size()) throw std::runtime_error("case index out of range");
  std::cout << std::setprecision(10)
      << "{\"schema_version\":1,\"kind\":\"ghb.deeper_histogram\","
      << "\"boundary\":\"resident feature-major bins, output-major assignments and row-major G/H to active histogram ranges; clear plus accumulation included; no packing or transpose; allocation, transfers, graph construction and CPU validation excluded\","
      << "\"semantics\":\"independent output assignments and active counts; uint64 counts exact; FP64 atomic order unspecified; inactive histogram capacity preserved\","
      << "\"integration\":\"primitive only; independent-tree trainer state retention is not implemented by this benchmark\","
      << "\"samples\":" << samples << ",\"warmup_samples\":" << warmups << ",\"cases\":[";
  bool first_case = true;
  for (unsigned case_index = 0; case_index < shapes.size(); ++case_index) {
    if (selected != unsigned(-1) && selected != case_index) continue;
    auto s = shapes[case_index];
    // Match the production independent-output tile's compact derivative stride.
    // Correctness fixtures separately exercise nonzero first output and padding.
    s.first = 0; s.padding = 0;
    Fixture f(s);
    const auto variants = f.variants();
    for (const auto& variant : variants) { f.reset(); f.launch(variant); f.verify("pre-benchmark " + variant.name); }
    const unsigned repetitions = s.rows >= 65536 ? 2 : 8;
    for (bool graph_mode : {false, true}) {
      std::vector<std::unique_ptr<Graph>> graphs;
      if (graph_mode) for (const auto& variant : variants) {
        auto graph = std::make_unique<Graph>();
        check(cudaStreamBeginCapture(f.stream.value, cudaStreamCaptureModeThreadLocal));
        for (unsigned repeat = 0; repeat < repetitions; ++repeat) f.launch(variant);
        check(cudaStreamEndCapture(f.stream.value, &graph->value));
        check(cudaGraphInstantiate(&graph->executable, graph->value, 0));
        graphs.push_back(std::move(graph));
      }
      cudaEvent_t start{}, stop{}; check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
      std::vector<std::vector<double>> device_us(variants.size()), wall_us(variants.size());
      std::vector<unsigned> order;
      f.reset();
      for (unsigned sample = 0; sample < samples + warmups; ++sample)
        for (unsigned position = 0; position < variants.size(); ++position) {
          const unsigned variant = (sample % 2 ? unsigned(variants.size()) - 1 - position : position);
          const auto before = std::chrono::steady_clock::now();
          check(cudaEventRecord(start, f.stream.value));
          if (graph_mode) check(cudaGraphLaunch(graphs[variant]->executable, f.stream.value));
          else for (unsigned repeat = 0; repeat < repetitions; ++repeat) f.launch(variants[variant]);
          check(cudaEventRecord(stop, f.stream.value)); check(cudaEventSynchronize(stop));
          const auto after = std::chrono::steady_clock::now();
          float milliseconds{}; check(cudaEventElapsedTime(&milliseconds, start, stop));
          if (sample >= warmups) {
            order.push_back(variant);
            device_us[variant].push_back(milliseconds * 1000. / repetitions);
            wall_us[variant].push_back(std::chrono::duration<double, std::micro>(after - before).count() / repetitions);
          }
        }
      check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
      for (const auto& variant : variants) { f.reset(); f.launch(variant); f.verify("post-benchmark " + variant.name); }
      if (!first_case) std::cout << ','; first_case = false;
      std::cout << "{\"case\":" << case_index << ",\"rows\":" << s.rows << ",\"columns\":" << s.columns
          << ",\"outputs\":" << s.outputs << ",\"capacity\":" << s.capacity << ",\"max_bins\":" << s.bins
          << ",\"derivative_stride\":" << f.stride << ",\"first_output\":" << s.first
          << ",\"total_bins\":" << f.total << ",\"inactive_percent\":" << s.inactive_percent
          << ",\"skew\":" << (s.skew ? "true" : "false") << ",\"zero_derivatives\":" << (s.zero ? "true" : "false")
          << ",\"irregular_features\":" << (s.irregular ? "true" : "false")
          << ",\"ragged_active\":" << (s.ragged_active ? "true" : "false")
          << ",\"launch\":\"" << (graph_mode ? "graph" : "stream") << "\",\"repetitions\":" << repetitions
          << ",\"assignment_bytes\":" << f.assignment.size() * sizeof(int)
          << ",\"histogram_bytes\":" << f.expected.size() * sizeof(Stats)
          << ",\"derivative_bytes\":" << (f.gradient.size() + f.hessian.size()) * sizeof(double)
          << ",\"order\":[";
      for (std::size_t i = 0; i < order.size(); ++i) { if (i) std::cout << ','; std::cout << order[i]; }
      std::cout << "],\"variants\":[";
      for (std::size_t v = 0; v < variants.size(); ++v) {
        if (v) std::cout << ',';
        std::cout << "{\"name\":\"" << variants[v].name << "\",\"row_chunks\":" << variants[v].chunks << ",\"device_us\":[";
        for (std::size_t i = 0; i < device_us[v].size(); ++i) { if (i) std::cout << ','; std::cout << device_us[v][i]; }
        std::cout << "],\"wall_us\":[";
        for (std::size_t i = 0; i < wall_us[v].size(); ++i) { if (i) std::cout << ','; std::cout << wall_us[v][i]; }
        std::cout << "]}";
      }
      std::cout << "]}" << std::flush;
    }
  }
  std::cout << "],\"validation_checks\":" << checks << "}\n";
}
} // namespace
int main(int argc, char** argv) {
  try {
    unsigned selected = unsigned(-1);
    if (argc == 3 && std::string(argv[1]) == "--case") {
      std::size_t parsed{};
      const std::string text = argv[2]; const auto value = std::stoul(text, &parsed);
      if (parsed != text.size() || value > std::numeric_limits<unsigned>::max()) throw std::runtime_error("invalid case index");
      selected = unsigned(value);
    } else if (argc != 1) throw std::runtime_error("usage: ghb_deeper_histogram_bench [--case INDEX]");
    int devices{}; const auto error = cudaGetDeviceCount(&devices);
    if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver || (error == cudaSuccess && !devices)) return 77;
    check(error); benchmark(selected); return 0;
  } catch (const std::exception& error) {
    std::cerr << "deeper histogram benchmark failed: " << error.what() << '\n'; return 1;
  }
}
