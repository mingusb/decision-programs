#include "ghb/instrumentation.hpp"
#include "common.hpp"
#include "profiling.hpp"

#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <type_traits>

namespace gi = ghb::instrumentation;
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

struct Options {
  std::uint64_t n{1ULL << 24}, seed{20260922401ULL};
  unsigned bins{1U << 20}, window{524288}, repetitions{21}, batch{4}, warmup_ms{2000};
  int tuning{2}, blocks{96};
  std::string variant{"global_window"}, instrumentation{"timing"}, launch{"graph"};
};

std::uint64_t integer(const std::string& value) {
  std::uint64_t result{};
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
    throw std::invalid_argument("expected an unsigned decimal integer: " + value);
  return result;
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help") {
      std::cout << "ghb_instrumentation_bench [--n N] [--bins B] [--seed N]\n"
        "  [--variant global_window|global] [--tuning 0..5] [--blocks N]\n"
        "  [--window-bins N] [--repetitions N] [--batch N] [--warmup-ms N]\n"
        "  [--instrumentation off|timing|nvtx] [--launch stream|graph]\n"
        "JSON stdout; exact CPU output validation for every repeated batch.\n"
        "Only existing count kernels are executed. No trainer is simulated.\n"
        "Outer event timings include opt-in instrumentation overhead.\n";
      std::exit(0);
    }
    if (++i == argc) throw std::invalid_argument("missing value for " + key);
    const std::string value = argv[i];
    if (key == "--variant") o.variant = value;
    else if (key == "--instrumentation") o.instrumentation = value;
    else if (key == "--launch") o.launch = value;
    else {
      const auto n = integer(value);
      if (key == "--n") o.n = n;
      else if (key == "--seed") o.seed = n;
      else {
        if (n > unsigned(std::numeric_limits<int>::max())) throw std::invalid_argument("option too large: " + key);
        if (key == "--bins") o.bins = unsigned(n);
        else if (key == "--window-bins") o.window = unsigned(n);
        else if (key == "--repetitions") o.repetitions = unsigned(n);
        else if (key == "--batch") o.batch = unsigned(n);
        else if (key == "--warmup-ms") o.warmup_ms = unsigned(n);
        else if (key == "--tuning") o.tuning = int(n);
        else if (key == "--blocks") o.blocks = int(n);
        else throw std::invalid_argument("unknown option: " + key);
      }
    }
  }
  if (!o.n || o.n > UINT32_MAX || !o.bins || o.bins >= unsigned(INT_MAX) ||
      !o.window || o.window >= unsigned(INT_MAX) || !o.repetitions || o.repetitions > 4096 ||
      !o.batch || o.batch > 4096 || o.warmup_ms > 60000 || o.tuning > 5 || !o.blocks)
    throw std::invalid_argument("invalid workload or measurement bounds");
  if (o.variant != "global_window" && o.variant != "global") throw std::invalid_argument("unsupported variant");
  if (o.instrumentation != "off" && o.instrumentation != "timing" && o.instrumentation != "nvtx")
    throw std::invalid_argument("unsupported instrumentation mode");
  if (o.launch != "stream" && o.launch != "graph") throw std::invalid_argument("unsupported launch mode");
  if (std::uint64_t(o.bins) * sizeof(std::uint64_t) * o.repetitions > (512ULL << 20))
    throw std::invalid_argument("validation snapshots exceed the 512 MiB pinned-memory budget; reduce repetitions or bins");
  const auto scans = o.variant == "global_window" ? (std::uint64_t(o.bins) + o.window - 1) / o.window : 1;
  if ((o.n * 4) > UINT64_MAX / scans / o.batch)
    throw std::invalid_argument("logical traffic metadata exceeds u64 range");
  return o;
}

struct Pinned {
  void* data{};
  explicit Pinned(std::size_t bytes) { CUDA_CHECK(cudaMallocHost(&data, bytes)); }
  ~Pinned() { cudaFreeHost(data); }
  Pinned(const Pinned&) = delete;
  Pinned& operator=(const Pinned&) = delete;
};

struct DrainOnFailure {
  cudaStream_t stream;
  bool completed{};
  ~DrainOnFailure() { if (!completed) cudaStreamSynchronize(stream); }
};

struct Graph {
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  ~Graph() {
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
  }
  template<class Launch> void capture(cudaStream_t stream, unsigned batch, Launch launch) {
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    try {
      for (unsigned i = 0; i < batch; ++i) launch();
    } catch (...) {
      cudaStreamEndCapture(stream, &graph);
      throw;
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
  }
};

std::string quote(std::string_view value) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
    else out << c;
  }
  out << '"';
  return out.str();
}

template<class Record> int run(const Options& o) {
  constexpr bool enabled = !std::is_same_v<Record, gi::NullRecorder>;
  const auto total_start = Clock::now();
  Record recorder(3 + 2 * o.repetitions, o.instrumentation == "nvtx");
  if (o.instrumentation == "nvtx" && !Record::nvtx_available())
    throw std::runtime_error("NVTX requested but this build disabled it");
  gi::Context context;
  context.rows = o.n; context.features = 1; context.bins = o.bins; context.stream_id = 1;
  auto setup = recorder.begin(gi::Stage::initialize, context, gi::Timing::host);
  cudaDeviceProp gpu{};
  CUDA_CHECK(cudaGetDeviceProperties(&gpu, 0));
  int driver{}, runtime{};
  CUDA_CHECK(cudaDriverGetVersion(&driver)); CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  std::size_t free_before{}, total_memory{};
  CUDA_CHECK(cudaMemGetInfo(&free_before, &total_memory));
  gh::Config config;
  config.algorithm = o.variant == "global_window" ? gh::Algorithm::global_window : gh::Algorithm::global_atomic;
  config.input_type = gh::InputType::u32; config.counter_type = gh::CounterType::u64;
  config.local_counter = gh::LocalCounter::u32; config.output_clear = gh::OutputClear::kernel;
  config.launch = o.launch == "graph" ? gh::LaunchMode::graph : gh::LaunchMode::stream;
  config.size = o.n; config.bins = o.bins; config.window_bins = o.window;
  config.tuning = o.tuning; config.blocks = o.blocks;
  CUDA_CHECK(gh::prepare(config));
  std::size_t scratch_bytes{};
  CUDA_CHECK(gh::workspace_bytes(config, scratch_bytes));
  context.scratch_bytes = scratch_bytes;
  Dataset dataset(o.n, o.bins, gh::InputType::u32, "uniform", "shuffled", o.seed);
  Stream stream;
  const std::size_t input_bytes = o.n * sizeof(unsigned), output_bytes = std::size_t(o.bins) * sizeof(std::uint64_t);
  DeviceBuffer input(input_bytes), output(output_bytes), scratch(scratch_bytes);
  Pinned snapshots(output_bytes * o.repetitions);
  std::deque<Event> starts(o.repetitions), ends(o.repetitions);
  Event run_start, run_end;
  std::vector<double> operation_ms(o.repetitions);
  std::size_t free_after{}, unused_total{};
  CUDA_CHECK(cudaMemGetInfo(&free_after, &unused_total));
  recorder.end(setup);
  const double prepare_ms = elapsed(total_start);
  // On exceptional submission exits, finish queued copies before any pinned
  // snapshots or device buffers are released. The successful path disarms this.
  DrainOnFailure drain{stream.value};

  auto phase_start = Clock::now();
  auto upload_context = context; upload_context.logical_read_bytes = input_bytes; upload_context.logical_write_bytes = input_bytes;
  auto upload = recorder.begin(gi::Stage::upload, upload_context, gi::Timing::gpu, stream.value);
  CUDA_CHECK(cudaMemcpyAsync(input.data, dataset.keys.data(), input_bytes, cudaMemcpyHostToDevice, stream.value));
  recorder.end(upload);
  CUDA_CHECK(cudaStreamSynchronize(stream.value));
  const double upload_ms = elapsed(phase_start);

  auto launch_one = [&] { CUDA_CHECK(gh::histogram(config, input.data, output.data, scratch.data, scratch_bytes, stream.value)); };
  phase_start = Clock::now();
  Graph graph;
  if (o.launch == "graph") graph.capture(stream.value, o.batch, launch_one);
  const double capture_ms = elapsed(phase_start);
  auto launch_batch = [&] {
    if (graph.executable) CUDA_CHECK(cudaGraphLaunch(graph.executable, stream.value));
    else for (unsigned b = 0; b < o.batch; ++b) launch_one();
  };
  // Validate once before measurement, including poisoned output and scratch.
  phase_start = Clock::now();
  CUDA_CHECK(cudaMemsetAsync(output.data, 0xa5, output_bytes, stream.value));
  CUDA_CHECK(cudaMemsetAsync(scratch.data, 0xa5, scratch_bytes, stream.value));
  launch_batch();
  CUDA_CHECK(cudaStreamSynchronize(stream.value));
  verify_output(config, output.data, dataset.expected);
  const double preflight_ms = elapsed(phase_start);
  phase_start = Clock::now();
  while (elapsed(phase_start) < o.warmup_ms) {
    launch_batch();
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
  }
  const double warmup_ms = elapsed(phase_start);

  // Each replay gets distinct inner and outer events and a pinned snapshot.
  // There is no per-position wait or event read in this submission loop.
  gh::profiling::CaptureRange capture("count");
  const auto submit_start = Clock::now();
  CUDA_CHECK(cudaEventRecord(run_start.value, stream.value));
  double readback_submit_ms{};
  for (unsigned i = 0; i < o.repetitions; ++i) {
    auto work = context;
    work.repetition = int(i); work.operations = o.batch;
    const std::uint64_t scans = o.variant == "global_window" ? (std::uint64_t(o.bins) + o.window - 1) / o.window : 1;
    work.logical_read_bytes = input_bytes * scans * o.batch;
    work.logical_write_bytes = output_bytes * o.batch;
    CUDA_CHECK(cudaEventRecord(starts[i].value, stream.value));
    const auto histogram = recorder.begin(gi::Stage::histogram, work, gi::Timing::gpu, stream.value);
    launch_batch();
    recorder.end(histogram);
    CUDA_CHECK(cudaEventRecord(ends[i].value, stream.value));
    auto copy = context; copy.repetition = int(i);
    copy.logical_read_bytes = output_bytes; copy.logical_write_bytes = output_bytes;
    const auto readback_start = Clock::now();
    const auto download = recorder.begin(gi::Stage::download, copy, gi::Timing::gpu, stream.value);
    CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(snapshots.data) + output_bytes * i,
                               output.data, output_bytes, cudaMemcpyDeviceToHost, stream.value));
    recorder.end(download);
    readback_submit_ms += elapsed(readback_start);
  }
  CUDA_CHECK(cudaEventRecord(run_end.value, stream.value));
  const double submit_ms = elapsed(submit_start);
  phase_start = Clock::now();
  CUDA_CHECK(cudaEventSynchronize(run_end.value));
  drain.completed = true;
  const double completion_ms = elapsed(phase_start);
  capture.finish();

  phase_start = Clock::now();
  float device_span{};
  CUDA_CHECK(cudaEventElapsedTime(&device_span, run_start.value, run_end.value));
  for (unsigned i = 0; i < o.repetitions; ++i) {
    float ms{};
    CUDA_CHECK(cudaEventElapsedTime(&ms, starts[i].value, ends[i].value));
    operation_ms[i] = double(ms) / o.batch;
    if (!std::isfinite(operation_ms[i]) || operation_ms[i] <= 0)
      throw std::runtime_error("invalid outer event interval");
  }
  double collection_ms = elapsed(phase_start);
  phase_start = Clock::now();
  const auto evaluate = recorder.begin(gi::Stage::evaluate, context, gi::Timing::host);
  const auto* all_counts = static_cast<const std::uint64_t*>(snapshots.data);
  for (unsigned i = 0; i < o.repetitions; ++i)
    for (unsigned bin = 0; bin < o.bins; ++bin)
      if (all_counts[std::size_t(i) * o.bins + bin] != dataset.expected[bin])
        throw std::runtime_error("histogram validation failed at repetition " + std::to_string(i) + ", bin " + std::to_string(bin));
  recorder.end(evaluate);
  const double validation_ms = elapsed(phase_start);
  phase_start = Clock::now();
  auto samples = recorder.collect(false);
  if (!samples) throw std::logic_error("completed run has pending recorder events");
  collection_ms += elapsed(phase_start);
  const double total_ms = elapsed(total_start);
  auto sorted = operation_ms;
  std::sort(sorted.begin(), sorted.end());
  const double median = (sorted[(sorted.size() - 1) / 2] + sorted[sorted.size() / 2]) / 2;

  std::cout << std::setprecision(17) << std::boolalpha;
  std::cout << "{\n\"schema\":1,\"kind\":\"ghb.instrumentation\",\"benchmark\":\"count_histogram_component\",\n"
    << "\"implementation\":" << quote(o.variant + ":" + std::to_string(o.tuning) + ":" + std::to_string(o.blocks) + ":u32:kernel")
    << ",\"instrumented\":" << enabled << ",\"nvtx\":" << (enabled && o.instrumentation == "nvtx")
    << ",\"build\":{\"id\":" << quote(GHB_BUILD_ID) << "},\n"
    << "\"environment\":{\"gpu\":" << quote(gpu.name) << ",\"sm\":" << gpu.major * 10 + gpu.minor
    << ",\"driver_api\":" << driver << ",\"runtime\":" << runtime << ",\"total_memory_bytes\":" << total_memory << "},\n"
    << "\"workload\":{\"rows\":" << o.n << ",\"features\":1,\"bins\":" << o.bins << ",\"seed\":" << o.seed
    << ",\"input\":\"u32\",\"counter\":\"u64\",\"distribution\":\"uniform\",\"cache\":\"warm\",\"launch\":" << quote(o.launch)
    << ",\"repetitions\":" << o.repetitions << ",\"batch\":" << o.batch << ",\"warmup_ms\":" << o.warmup_ms << ",\"window_bins\":" << o.window << "},\n"
    << "\"validation\":{\"passed\":true,\"checked_bins\":" << o.bins << ",\"checked_repetitions\":" << o.repetitions << ",\"mismatched_bins\":0},\n"
    << "\"memory\":{\"input_bytes\":" << input_bytes << ",\"output_bytes\":" << output_bytes << ",\"scratch_bytes\":" << scratch_bytes
    << ",\"validation_snapshot_bytes\":" << output_bytes * o.repetitions << ",\"free_before_bytes\":" << free_before << ",\"free_after_alloc_bytes\":" << free_after << "},\n"
    << "\"timing\":{\"prepare_wall_ms\":" << prepare_ms << ",\"upload_wall_ms\":" << upload_ms << ",\"capture_wall_ms\":" << capture_ms
    << ",\"preflight_wall_ms\":" << preflight_ms << ",\"warmup_wall_ms\":" << warmup_ms << ",\"submit_wall_ms\":" << submit_ms
    << ",\"completion_wall_ms\":" << completion_ms << ",\"collection_wall_ms\":" << collection_ms << ",\"device_span_ms\":" << device_span
    << ",\"readback_submit_wall_ms\":" << readback_submit_ms << ",\"validation_wall_ms\":" << validation_ms << ",\"end_to_end_wall_ms\":" << total_ms
    << ",\"operation_median_ms\":" << median << ",\"operation_ms\":[";
  for (std::size_t i = 0; i < operation_ms.size(); ++i) std::cout << (i ? "," : "") << operation_ms[i];
  std::cout << "]},\n\"samples\":[";
  for (std::size_t i = 0; i < samples->size(); ++i) {
    const auto& s = (*samples)[i]; const auto& c = s.context;
    std::cout << (i ? "," : "") << "\n{\"id\":" << s.id << ",\"stage\":" << quote(gi::name(s.stage))
      << ",\"timing\":" << quote(s.timing == gi::Timing::gpu ? "gpu" : "host")
      << ",\"context\":{\"round\":" << c.round << ",\"depth\":" << c.depth << ",\"output\":" << c.output << ",\"repetition\":" << c.repetition
      << ",\"active_nodes\":" << c.active_nodes << ",\"features\":" << c.features << ",\"bins\":" << c.bins << ",\"stream_id\":" << c.stream_id
      << ",\"rows\":" << c.rows << ",\"logical_read_bytes\":" << c.logical_read_bytes << ",\"logical_write_bytes\":" << c.logical_write_bytes
      << ",\"scratch_bytes\":" << c.scratch_bytes << ",\"operations\":" << c.operations << "},\"host_start_ns\":" << s.host_start_ns
      << ",\"host_end_ns\":" << s.host_end_ns << ",\"gpu_ms\":";
    if (s.gpu_ms) std::cout << *s.gpu_ms; else std::cout << "null";
    std::cout << '}';
  }
  std::cout << "\n]}\n";
  return 0;
}

int main(int argc, char** argv) try {
  const auto options = parse(argc, argv);
  if (options.instrumentation == "off") return run<gi::NullRecorder>(options);
  return run<gi::Recorder>(options);
} catch (const std::exception& error) {
  std::cerr << "ghb_instrumentation_bench: " << error.what() << '\n';
  return 1;
}
