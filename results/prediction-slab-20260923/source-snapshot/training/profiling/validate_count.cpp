#include "validation_support.hpp"
#include "gh/histogram.hpp"
#include <algorithm>
#include <cstdint>
#include <memory>

using namespace diagnostic;
namespace {
struct Instance {
  gh::Config config;
  Stream stream;
  Device<std::uint32_t> input;
  Device<unsigned long long> output;
  std::vector<std::uint32_t> host;
  std::vector<unsigned long long> expected;
  Graph graph;
  Instance(unsigned rows, unsigned bins) : input(rows), output(bins), host(rows), expected(bins) {
    config.algorithm = gh::Algorithm::global_window;
    config.input_type = gh::InputType::u32; config.counter_type = gh::CounterType::u64;
    config.local_counter = gh::LocalCounter::u32; config.size = rows; config.bins = bins;
    config.window_bins = 128; config.tuning = 2; config.blocks = 8;
    config.output_clear = gh::OutputClear::kernel; config.launch = gh::LaunchMode::graph;
    check(gh::prepare(config));
  }
  void fill(unsigned seed) {
    std::fill(expected.begin(), expected.end(), 0);
    for (std::size_t i = 0; i < host.size(); ++i) {
      host[i] = (unsigned(i) * 37u + seed * 17u + unsigned(i / 7)) % config.bins;
      ++expected[host[i]];
    }
    input.put(host, stream.value);
    output.put(std::vector<unsigned long long>(config.bins, 0xfedcba9876543210ULL), stream.value);
  }
  std::size_t bytes() { std::size_t n{}; check(gh::workspace_bytes(config, n)); return n; }
  void capture(void* scratch, std::size_t bytes) {
    graph.capture(stream.value, [&] { check(gh::histogram(config, input.value, output.value, scratch, bytes, stream.value)); });
  }
  void validate() {
    const auto actual = output.get(stream.value);
    require(actual == expected, "exact uint64 histogram reference");
  }
};
void run(std::ostream& out) {
  out << '[';
  for (unsigned shared = 0; shared != 2; ++shared) {
    Instance a(4097, 257), b(2053, 129);
    Device<unsigned char> scratch_a(std::max(a.bytes(), b.bytes()));
    Device<unsigned char> scratch_b(b.bytes());
    a.fill(0); b.fill(1);
    a.capture(scratch_a.value, scratch_a.size);
    b.capture(shared ? scratch_a.value : scratch_b.value, shared ? scratch_a.size : scratch_b.size);
    Event ready;
    for (unsigned iteration = 0; iteration != 4; ++iteration) {
      // Input bits change at stable captured addresses, after prior work completed.
      a.fill(iteration * 2); b.fill(iteration * 2 + 1);
      a.graph.launch(a.stream.value);
      if (shared) {
        check(cudaEventRecord(ready.value, a.stream.value));
        check(cudaStreamWaitEvent(b.stream.value, ready.value, 0));
      }
      b.graph.launch(b.stream.value);
      a.validate(); b.validate();
    }
    // Graph destruction precedes input/scratch release; streams are complete here.
    if (shared) out << ',';
    out << "{\"shared_workspace\":" << (shared ? "true" : "false")
        << ",\"ordering\":" << quote(shared ? "CUDA event hand-off" : "independent buffers and streams")
        << ",\"replays_per_instance\":4,\"input_changed_each_replay\":true,\"exact_counts_passed\":true"
        << ",\"workspace_bytes\":" << scratch_a.size << '}';
    // Graphs hold scratch addresses; explicitly destroy while storage is live.
    check(cudaGraphExecDestroy(a.graph.executable)); a.graph.executable = nullptr;
    check(cudaGraphDestroy(a.graph.value)); a.graph.value = nullptr;
    check(cudaGraphExecDestroy(b.graph.executable)); b.graph.executable = nullptr;
    check(cudaGraphDestroy(b.graph.value)); b.graph.value = nullptr;
  }
  out << ']';
}
}
int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    std::cout << "ghb_validate_count --output NEW.json\nValid-contract concurrency correctness only; no performance measurement.\n";
    return 0;
  }
  if (argc != 3 || std::string(argv[1]) != "--output") { std::cerr << "expected --output NEW.json\n"; return 2; }
  try {
    Report report(argv[2]); std::ostringstream observations;
    try {
      const auto metadata = device_metadata(); run(observations);
      report.write("{\"schema\":1,\"kind\":\"count_lifecycle_validation\",\"passed\":true,\"device\":" + metadata +
                   ",\"checks\":" + std::to_string(checks) + ",\"cases\":" + observations.str() + "}\n");
      std::cout << "count lifecycle exact gates passed: " << checks << '\n'; return 0;
    } catch (const std::exception& e) {
      report.write("{\"schema\":1,\"passed\":false,\"error\":" + quote(e.what()) +
                   ",\"partial_observations\":" + quote(observations.str()) + "}\n"); throw;
    }
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
