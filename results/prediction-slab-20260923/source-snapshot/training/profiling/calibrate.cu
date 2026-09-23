#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

// Isolated diagnostic primitive benchmark: see CALIBRATION_DESIGN.md.
namespace {
void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template<class T> struct Buffer {
  T* data = nullptr;
  explicit Buffer(std::size_t n) { check(cudaMalloc(&data, n * sizeof(T))); }
  ~Buffer() { if (data) cudaFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};
struct Event {
  cudaEvent_t value{};
  Event() { check(cudaEventCreate(&value)); }
  ~Event() { cudaEventDestroy(value); }
};
struct Stream {
  cudaStream_t value{};
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
};
constexpr unsigned bins = 1024;
__host__ __device__ unsigned key(unsigned i, unsigned pattern) {
  const unsigned mixed = i * 1664525u + 1013904223u;
  if (pattern == 1 || (pattern == 2 && i % 10 != 0)) return 0;
  return mixed & (bins - 1);
}
template<class Key> __global__ void make_keys(Key* keys, unsigned n, unsigned pattern) {
  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) keys[i] = static_cast<Key>(key(i, pattern));
}
template<class T, class Key> __global__ void atomic_updates(const Key* keys, unsigned n, T* counts) {
  const T increment = std::is_same_v<T, double> ? T(0.125) : T(1);
  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) atomicAdd(counts + keys[i], increment);
}
struct Options {
  std::string output, type = "all", keys = "all", pattern = "all";
  unsigned n = 262144, repetitions = 7, blocks_per_sm = 0;
};
unsigned positive(const std::string& text) {
  std::size_t used = 0;
  const auto n = std::stoull(text, &used);
  if (used != text.size() || !n || n > 67108864) throw std::runtime_error("invalid bounded positive integer");
  return static_cast<unsigned>(n);
}
Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help") {
      std::cout << "--output NEW.json [--n N] [--repetitions 1..100] "
                   "[--type all|u32|u64|f64] [--keys all|u16|u32] "
                   "[--pattern all|uniform|hot|skew] [--blocks-per-sm 1|4]\n";
      std::exit(0);
    }
    if (i + 1 == argc) throw std::runtime_error("missing argument value");
    const std::string value = argv[++i];
    if (arg == "--output") o.output = value;
    else if (arg == "--type") o.type = value;
    else if (arg == "--keys") o.keys = value;
    else if (arg == "--pattern") o.pattern = value;
    else if (arg == "--n") o.n = positive(value);
    else if (arg == "--repetitions") o.repetitions = positive(value);
    else if (arg == "--blocks-per-sm") o.blocks_per_sm = positive(value);
    else throw std::runtime_error("unknown option: " + arg);
  }
  if (o.output.empty() || o.repetitions > 100 ||
      (o.type != "all" && o.type != "u32" && o.type != "u64" && o.type != "f64") ||
      (o.keys != "all" && o.keys != "u16" && o.keys != "u32") ||
      (o.pattern != "all" && o.pattern != "uniform" && o.pattern != "hot" && o.pattern != "skew") ||
      (o.blocks_per_sm && o.blocks_per_sm != 1 && o.blocks_per_sm != 4))
    throw std::runtime_error("invalid options; use --help");
  return o;
}
template<class T, class Key>
bool measure(const Options& o, const cudaDeviceProp& device, unsigned pattern,
             const char* pattern_name, unsigned blocks_per_sm, const char* type,
             const char* key_type, std::ostream& output, bool& first) {
  Stream stream;
  Event start, end;
  Buffer<Key> keys(o.n);
  Buffer<T> counts(bins);
  const unsigned blocks = device.multiProcessorCount * blocks_per_sm;
  make_keys<<<blocks, 256, 0, stream.value>>>(keys.data, o.n, pattern);
  check(cudaGetLastError());
  check(cudaStreamSynchronize(stream.value));
  auto operation = [&] {
    check(cudaMemsetAsync(counts.data, 0, bins * sizeof(T), stream.value));
    atomic_updates<<<blocks, 256, 0, stream.value>>>(keys.data, o.n, counts.data);
    check(cudaGetLastError());
  };
  for (unsigned i = 0; i < 2; ++i) operation();
  check(cudaStreamSynchronize(stream.value));
  std::vector<double> gpu, host, mismatch_by_repetition;
  std::vector<T> actual(bins), expected(bins, T(0));
  const T increment = std::is_same_v<T, double> ? T(0.125) : T(1);
  for (unsigned i = 0; i < o.n; ++i) expected[key(i, pattern)] += increment;
  unsigned mismatches = 0;
  for (unsigned i = 0; i < o.repetitions; ++i) {
    const auto before = std::chrono::steady_clock::now();
    check(cudaEventRecord(start.value, stream.value));
    operation();
    check(cudaEventRecord(end.value, stream.value));
    check(cudaEventSynchronize(end.value));
    host.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - before).count());
    float elapsed = 0;
    check(cudaEventElapsedTime(&elapsed, start.value, end.value));
    gpu.push_back(elapsed);
    check(cudaMemcpy(actual.data(), counts.data, bins * sizeof(T), cudaMemcpyDeviceToHost));
    unsigned repetition_mismatches = 0;
    for (unsigned b = 0; b < bins; ++b) repetition_mismatches += actual[b] != expected[b];
    mismatch_by_repetition.push_back(repetition_mismatches);
    mismatches += repetition_mismatches;
  }
  if (!first) output << ",\n";
  first = false;
  output << "{\"type\":\"" << type << "\",\"key_type\":\"" << key_type
         << "\",\"pattern\":\"" << pattern_name << "\",\"blocks_per_sm\":" << blocks_per_sm
         << ",\"blocks\":" << blocks << ",\"key_bytes\":" << std::uint64_t(o.n) * sizeof(Key)
         << ",\"output_bytes\":" << bins * sizeof(T) << ",\"mismatched_bins\":" << mismatches;
  auto write_array = [&](const char* name, const std::vector<double>& values) {
    output << ",\"" << name << "\":[";
    for (std::size_t i = 0; i < values.size(); ++i) { if (i) output << ','; output << values[i]; }
    output << ']';
  };
  write_array("clear_and_accumulate_device_ms", gpu);
  write_array("synchronized_host_ms", host);
  write_array("mismatched_bins_by_repetition", mismatch_by_repetition);
  output << '}'; output.flush();
  return !mismatches;
}
template<class T>
bool run_type(const Options& o, const cudaDeviceProp& device, const char* type,
              std::ostream& output, bool& first) {
  bool good = true;
  const char* patterns[] = {"uniform", "hot", "skew"};
  for (unsigned pattern = 0; pattern < 3; ++pattern) {
    if (o.pattern != "all" && o.pattern != patterns[pattern]) continue;
    for (const unsigned blocks : {1u, 4u}) {
      if (o.blocks_per_sm && o.blocks_per_sm != blocks) continue;
      if (o.keys == "all" || o.keys == "u16")
        good &= measure<T, std::uint16_t>(o, device, pattern, patterns[pattern], blocks, type, "u16", output, first);
      if (o.keys == "all" || o.keys == "u32")
        good &= measure<T, std::uint32_t>(o, device, pattern, patterns[pattern], blocks, type, "u32", output, first);
    }
  }
  return good;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse(argc, argv);
    const int fd = open(options.output.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
    if (fd < 0) throw std::runtime_error("cannot create new output (existing evidence is never overwritten)");
    close(fd);
    std::ofstream output(options.output);
    output << std::setprecision(17);
    cudaDeviceProp device{};
    check(cudaGetDeviceProperties(&device, 0));
    int runtime = 0, driver = 0;
    check(cudaRuntimeGetVersion(&runtime)); check(cudaDriverGetVersion(&driver));
    const bool capture = std::getenv("GH_PROFILE_CAPTURE") && std::string(std::getenv("GH_PROFILE_CAPTURE")) == "1";
    output << "{\"schema\":\"gh.hardware-calibration.v1\",\"device\":" << std::quoted(device.name)
           << ",\"sm\":" << device.major * 10 + device.minor << ",\"sm_count\":" << device.multiProcessorCount
           << ",\"l2_bytes\":" << device.l2CacheSize << ",\"global_bytes\":" << device.totalGlobalMem
           << ",\"runtime_version\":" << runtime << ",\"driver_version\":" << driver
           << ",\"n\":" << options.n << ",\"bins\":" << bins << ",\"warmups\":2,\"repetitions\":" << options.repetitions
           << ",\"capture_requested\":" << (capture ? "true" : "false")
           << ",\"scope\":\"clear plus native global atomic accumulation; no production widening stage\",\"cases\":[\n";
    if (capture) check(cudaProfilerStart());
    bool first = true, good = true;
    if (options.type == "all" || options.type == "u32") good &= run_type<unsigned>(options, device, "u32", output, first);
    if (options.type == "all" || options.type == "u64") good &= run_type<unsigned long long>(options, device, "u64", output, first);
    if (options.type == "all" || options.type == "f64") good &= run_type<double>(options, device, "f64", output, first);
    if (capture) check(cudaProfilerStop());
    output << "\n],\"status\":\"" << (good ? "passed" : "failed") << "\"}\n";
    output.close();
    if (!output) throw std::runtime_error("failed writing calibration evidence");
    std::cout << (good ? "passed" : "failed") << ": " << options.output << '\n';
    return good ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "calibration failed: " << error.what() << '\n';
    return 1;
  }
}
