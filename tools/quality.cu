#include "gh/data.cuh"
#include "gh/metrics.cuh"
#include "gh/model.cuh"
#include "gh/reference.cuh"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gh::quality {
constexpr u64 arena_bytes = 512ULL << 20;
constexpr u32 threads = 256, metric_capacity = 16;
enum class Phase : u32 { dataset, model, prepared, encoded, predicted, compared,
                         reference_metrics, legacy_import, legacy_gate, candidate_metrics, gate };
struct State {
  Array<const std::byte> model_bytes, prediction_bytes, quality_bytes;
  Workspace storage, scratch;
  u64 permanent{}, cells{};
  DatasetRecord dataset;
  Model model;
  Array<float> values, targets;
  Array<std::uint16_t> bins;
  Array<double> reference, candidate, labels;
  MetricInput input;
  Metric metrics[2][metric_capacity];
  MetricReport reports[2];
  MetricVerdict verdicts[metric_capacity];
  MetricGate gate;
  Metric legacy_metrics[2][metric_capacity];
  MetricReport legacy_reports[2];
  MetricVerdict legacy_verdicts[metric_capacity];
  MetricGate legacy_gate;
  bool legacy_pass{};
  Status status;
  unsigned long long bit_differences{}, value_differences{}, decisions{}, nonfinite{};
  unsigned long long first_difference{UINT64_MAX}, maximum_absolute_bits{};
};
__device__ State state{};
__global__ void advance(Phase);

__device__ bool require(bool condition, const char* what) {
  if (!condition) { printf("quality failure: %s\n", what); assert(condition); }
  return condition;
}
__device__ bool submitted(cudaError_t error) {
  if (error != cudaSuccess) printf("quality CUDA submission failure: %u\n", u32(error));
  return require(error == cudaSuccess, "CUDA submission");
}
__device__ bool completed(Phase phase) {
  const auto s = state.status;
  if (!s.done || s.errors)
    printf("quality phase=%u done=%u errors=%u required_bytes=%llu\n", u32(phase), s.done,
      s.errors, static_cast<unsigned long long>(s.required_bytes));
  return require(s.done && !s.errors, "phase completion");
}
__device__ void then(cudaError_t error, Phase phase) {
  if (!submitted(error)) return;
  advance<<<1, 1, 0, cudaStreamTailLaunch>>>(phase);
  submitted(cudaGetLastError());
}
__device__ u32 grid(u64 n) { return u32(min(ceil_div(n, threads), u64(65535))); }
__device__ unsigned long long bits(double x) { return __double_as_longlong(x); }
__device__ bool finite(double x) { return (bits(x) & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL; }
__device__ bool suffix(Arena arena) {
  if (!arena.valid || !add_fits(arena.used, 15)) return require(false, "arena extent");
  arena.used = (arena.used + 15) & ~u64(15);
  if (!arena.fits(&state.status)) return require(false, "512 MiB arena capacity");
  state.permanent = arena.used;
  state.scratch = {state.storage.data + arena.used, state.storage.bytes - arena.used};
  return true;
}
__global__ void convert() {
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < state.cells; i += u64(gridDim.x) * threads) {
    u64 word = 0;
    for (u32 b = 0; b < 8; ++b) word |= u64(state.prediction_bytes.data[i * 8 + b]) << (8 * b);
    state.reference.data[i] = __longlong_as_double(word);
  }
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < state.labels.size; i += u64(gridDim.x) * threads)
    state.labels.data[i] = double(state.dataset.data.targets.data[i]);
}
__global__ void compare_predictions() {
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < state.cells; i += u64(gridDim.x) * threads) {
    const double a = state.reference.data[i], b = state.candidate.data[i];
    if (!finite(a)) atomicAdd(&state.nonfinite, 1ULL);
    if (bits(a) != bits(b)) {
      atomicAdd(&state.bit_differences, 1ULL);
      atomicMin(&state.first_difference, static_cast<unsigned long long>(i));
    }
    if (a != b) atomicAdd(&state.value_differences, 1ULL);
    if (finite(a) && finite(b)) atomicMax(&state.maximum_absolute_bits, bits(fabs(__dsub_rn(a, b))));
    if (state.model.objective == Objective::binary_logistic && ((a >= .5) != (b >= .5)))
      atomicAdd(&state.decisions, 1ULL);
  }
  if (state.model.objective == Objective::multiclass_softmax) {
    for (u64 r = u64(blockIdx.x) * threads + threadIdx.x; r < state.dataset.data.rows; r += u64(gridDim.x) * threads) {
      u32 a = 0, b = 0;
      for (u32 o = 1; o < state.model.outputs; ++o) {
        if (state.reference.data[r * state.model.outputs + o] > state.reference.data[r * state.model.outputs + a]) a = o;
        if (state.candidate.data[r * state.model.outputs + o] > state.candidate.data[r * state.model.outputs + b]) b = o;
      }
      if (a != b) atomicAdd(&state.decisions, 1ULL);
    }
  }
}
__device__ const char* metric_name(MetricName name) {
  switch (name) {
    case MetricName::mse: return "mse";
    case MetricName::rmse: return "rmse";
    case MetricName::mae: return "mae";
    case MetricName::r2: return "r2";
    case MetricName::logloss: return "logloss";
    case MetricName::accuracy: return "accuracy";
    case MetricName::brier: return "brier";
    case MetricName::auc: return "auc";
    case MetricName::f1: return "f1";
    case MetricName::average_precision: return "average_precision";
    case MetricName::hamming_loss: return "hamming_loss";
    case MetricName::exact_match: return "exact_match";
    case MetricName::micro_f1: return "micro_f1";
    case MetricName::macro_f1: return "macro_f1";
    case MetricName::micro_ap: return "micro_ap";
    case MetricName::macro_ap: return "macro_ap";
    case MetricName::micro_auc: return "micro_auc";
    case MetricName::macro_auc: return "macro_auc";
    case MetricName::precision_at_1: return "precision_at_1";
    case MetricName::precision_at_3: return "precision_at_3";
    case MetricName::precision_at_5: return "precision_at_5";
  }
  return "invalid";
}
__device__ void prediction_report() {
  printf("prediction cells=%llu bit_differences=%llu value_differences=%llu decision_differences=%llu nonfinite_reference=%llu max_abs=%.17g\n",
    static_cast<unsigned long long>(state.cells), state.bit_differences, state.value_differences,
    state.decisions, state.nonfinite, __longlong_as_double(state.maximum_absolute_bits));
  if (state.bit_differences) {
    const u64 i = state.first_difference;
    const double a = state.reference.data[i], b = state.candidate.data[i];
    printf("first_difference index=%llu row=%llu output=%u old=%.17g new=%.17g old_bits=%016llx new_bits=%016llx\n",
      static_cast<unsigned long long>(i), static_cast<unsigned long long>(i / state.model.outputs),
      u32(i % state.model.outputs), a, b, bits(a), bits(b));
  }
}
__device__ void metric_report() {
  for (u32 i = 0; i < state.reports[0].count; ++i) {
    const auto a = state.metrics[0][i], b = state.metrics[1][i];
    printf("metric name=%s output=%u new_name=%s new_output=%u old_available=%u new_available=%u old=%.17g new=%.17g old_bits=%016llx new_bits=%016llx verdict=%u\n",
      metric_name(a.name), a.output, metric_name(b.name), b.output, a.available, b.available,
      a.value, b.value, bits(a.value), bits(b.value), u32(state.verdicts[i]));
  }
  printf("same_engine_quality checked=%u regressions=%u unavailable=%u invalid=%u old_ap_outputs=%u new_ap_outputs=%u old_auc_outputs=%u new_auc_outputs=%u\n",
    state.gate.checked, state.gate.regressions, state.gate.unavailable, state.gate.invalid,
    state.reports[0].ap_outputs, state.reports[1].ap_outputs, state.reports[0].auc_outputs, state.reports[1].auc_outputs);
  if (!state.quality_bytes.size) printf("legacy_json_metric_arithmetic=NOT_CHECKED\n");
  printf("metric_gate done=%u errors=%u\n", state.status.done, state.status.errors);
  const bool pass = state.status.done && !state.status.errors && !state.bit_differences &&
    !state.nonfinite && !state.gate.regressions && !state.gate.invalid;
  printf("frozen_prediction_and_same_engine_quality=%s\n", pass ? "PASS" : "FAIL");
  require(pass && (!state.quality_bytes.size || state.legacy_pass), "strict conformance gates");
}
__device__ void legacy_report() {
  u32 differences = 0;
  for (u32 i = 0; i < state.legacy_reports[0].count; ++i) {
    const auto a = state.legacy_metrics[0][i], b = state.legacy_metrics[1][i];
    const bool differs = a.available != b.available || (a.available && bits(a.value) != bits(b.value));
    differences += differs;
    printf("legacy_metric name=%s available=%u computed_available=%u legacy=%.17g computed=%.17g legacy_bits=%016llx computed_bits=%016llx bit_difference=%u verdict=%u\n",
      metric_name(a.name), a.available, b.available, a.value, b.value, bits(a.value), bits(b.value), u32(differs), u32(state.legacy_verdicts[i]));
  }
  const auto a = state.legacy_reports[0], b = state.legacy_reports[1];
  const bool metadata = a.ap_outputs == b.ap_outputs && a.auc_outputs == b.auc_outputs;
  const bool valid = state.status.done && !state.status.errors && !state.legacy_gate.invalid;
  const bool exact = valid && metadata && !differences;
  const bool quality = valid && !state.legacy_gate.regressions;
  printf("legacy_eligibility ap=%u computed_ap=%u auc=%u computed_auc=%u\n", a.ap_outputs, b.ap_outputs, a.auc_outputs, b.auc_outputs);
  printf("legacy_json_metric_arithmetic=%s differences=%u\n", exact ? "PASS" : "FAIL", differences);
  printf("legacy_zero_allowance_quality=%s checked=%u regressions=%u unavailable=%u invalid=%u errors=%u\n",
    quality ? "PASS" : "FAIL", state.legacy_gate.checked, state.legacy_gate.regressions,
    state.legacy_gate.unavailable, state.legacy_gate.invalid, state.status.errors);
  state.legacy_pass = exact && quality;
}
__global__ void advance(Phase phase) {
  if (phase == Phase::gate) { metric_report(); return; }
  if (phase == Phase::legacy_gate) {
    legacy_report(); state.status = {};
    state.input.predictions = {state.candidate.data, state.candidate.size};
    then(evaluate_metrics(state.input, &state.reports[1], state.scratch, &state.status), Phase::candidate_metrics);
    return;
  }
  if (!completed(phase)) return;
  state.status = {};
  switch (phase) {
    case Phase::dataset:
      then(decode_model(state.model_bytes, &state.model, state.scratch, &state.status), Phase::model);
      return;
    case Phase::model: {
      const auto d = state.dataset.data; const auto m = state.model;
      const bool multiclass = m.objective == Objective::multiclass_softmax;
      if (!require(d.columns == m.schema.columns && state.dataset.objective == m.objective &&
          (multiclass ? d.outputs == 1 && state.dataset.classes == m.outputs : d.outputs == m.outputs), "model/dataset compatibility")) return;
      state.cells = u64(d.rows) * m.outputs;
      if (!require(state.cells && mul_fits(state.cells, 8) && state.prediction_bytes.size == state.cells * 8,
          "prediction wire extent")) return;
      Arena arena{state.storage, state.permanent};
      state.bins = {arena.take<std::uint16_t>(u64(d.rows) * d.columns), u64(d.rows) * d.columns};
      state.reference = {arena.take<double>(state.cells), state.cells};
      state.candidate = {arena.take<double>(state.cells), state.cells};
      state.labels = {arena.take<double>(d.targets.size), d.targets.size};
      if (!suffix(arena)) return;
      state.status = {};
      const auto task = m.objective == Objective::squared_error ? MetricTask::regression :
        multiclass ? MetricTask::multiclass : m.outputs == 1 ? MetricTask::binary : MetricTask::multilabel;
      state.input = {{state.labels.data, state.labels.size}, {}, {}, d.rows, m.outputs, task, MetricProfile::real_data};
      for (u32 i = 0; i < 2; ++i) state.reports[i].metrics = {state.metrics[i], metric_capacity};
      printf("quality objective=%u task=%u rows=%u features=%u outputs=%u trees=%llu nodes=%llu permanent_bytes=%llu scratch_bytes=%llu\n",
        u32(m.objective), u32(task), d.rows, d.columns, m.outputs, static_cast<unsigned long long>(m.tree_count),
        static_cast<unsigned long long>(m.node_count), static_cast<unsigned long long>(state.permanent),
        static_cast<unsigned long long>(state.scratch.bytes));
      convert<<<grid(state.cells), threads>>>();
      if (submitted(cudaGetLastError())) then(finish(&state.status), Phase::prepared);
      return;
    }
    case Phase::prepared:
      then(encode(state.dataset.data, &state.model.schema, state.bins, &state.status), Phase::encoded);
      return;
    case Phase::encoded:
      then(predict(&state.model, {state.bins.data, state.bins.size}, state.dataset.data.rows,
        state.candidate, false, &state.status), Phase::predicted);
      return;
    case Phase::predicted:
      compare_predictions<<<grid(state.cells), threads>>>();
      if (submitted(cudaGetLastError())) then(finish(&state.status), Phase::compared);
      return;
    case Phase::compared:
      prediction_report();
      state.input.predictions = {state.reference.data, state.reference.size};
      then(evaluate_metrics(state.input, &state.reports[0], state.scratch, &state.status), Phase::reference_metrics);
      return;
    case Phase::reference_metrics:
      if (state.quality_bytes.size) {
        for (u32 i = 0; i < 2; ++i) state.legacy_reports[i].metrics = {state.legacy_metrics[i], metric_capacity};
        then(decode_metric_reference(state.quality_bytes, &state.reports[0], &state.legacy_reports[0],
          &state.legacy_reports[1], &state.status), Phase::legacy_import);
        return;
      }
      state.input.predictions = {state.candidate.data, state.candidate.size};
      then(evaluate_metrics(state.input, &state.reports[1], state.scratch, &state.status), Phase::candidate_metrics);
      return;
    case Phase::legacy_import:
      then(compare_metrics(&state.legacy_reports[0], &state.legacy_reports[1], {state.legacy_verdicts, metric_capacity},
        &state.legacy_gate, &state.status), Phase::legacy_gate);
      return;
    case Phase::candidate_metrics:
      then(compare_metrics(&state.reports[0], &state.reports[1], {state.verdicts, metric_capacity}, &state.gate,
        &state.status), Phase::gate);
      return;
    case Phase::gate:
    case Phase::legacy_gate:
      return;
  }
}
__global__ void run(Array<const std::byte> data, Array<const std::byte> model,
                    Array<const std::byte> predictions, Array<const std::byte> quality, Workspace storage) {
  state.model_bytes = model; state.prediction_bytes = predictions;
  state.quality_bytes = quality;
  state.storage = storage; state.status = {};
  Arena arena{storage};
  state.values = {arena.take<float>(data.size / 4), data.size / 4};
  state.targets = {arena.take<float>(data.size / 4), data.size / 4};
  auto& m = state.model;
  m.schema.features = {arena.take<Feature>(model.size / 12), model.size / 12};
  m.schema.metadata = {arena.take<float>(model.size / 4), model.size / 4};
  m.schema.offsets = {arena.take<u32>(model.size / 12 + 1), model.size / 12 + 1};
  m.nodes = {arena.take<Node>(model.size / 28), model.size / 28};
  m.trees = {arena.take<Tree>(model.size / 8), model.size / 8};
  m.base = {arena.take<double>(model.size / 8), model.size / 8};
  m.output_offsets = {arena.take<u64>(model.size / 8 + 1), model.size / 8 + 1};
  if (!suffix(arena)) return;
  state.status = {};
  then(decode_dataset(data, state.values, state.targets, &state.dataset, &state.status), Phase::dataset);
}
}

// Host effects are restricted to opaque file bytes and CUDA runtime resources.
struct Blob {
  std::byte* device{};
  std::size_t bytes{};
  ~Blob() { if (device) cudaFree(device); }
  bool read(const char* path) {
    const int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return false; }
    struct stat info{};
    if (fstat(fd, &info) || info.st_size <= 0) { fprintf(stderr, "cannot size input: %s\n", path); close(fd); return false; }
    bytes = static_cast<std::size_t>(info.st_size);
    void* mapped = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) { perror(path); return false; }
    auto error = cudaMalloc(&device, bytes);
    if (error == cudaSuccess) error = cudaMemcpy(device, mapped, bytes, cudaMemcpyHostToDevice);
    munmap(mapped, bytes);
    if (error != cudaSuccess) fprintf(stderr, "transport: %s: %s\n", path, cudaGetErrorString(error));
    return error == cudaSuccess;
  }
};
int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) { fprintf(stderr, "usage: quality DATA.ghb MODEL.ghb PREDICTIONS.f64 [QUALITY.json]\n"); return 2; }
  Blob data, model, predictions, quality;
  if (!data.read(argv[1]) || !model.read(argv[2]) || !predictions.read(argv[3])) return 1;
  if (argc == 5 && !quality.read(argv[4])) return 1;
  std::byte* arena{};
  auto error = cudaMalloc(&arena, gh::quality::arena_bytes);
  if (error != cudaSuccess) { fprintf(stderr, "arena: %s\n", cudaGetErrorString(error)); return int(error); }
  gh::quality::run<<<1, 1>>>({data.device, data.bytes}, {model.device, model.bytes},
    {predictions.device, predictions.bytes}, {quality.device, quality.bytes}, {arena, gh::quality::arena_bytes});
  error = cudaGetLastError();
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) fprintf(stderr, "quality completion: %s\n", cudaGetErrorString(error));
  cudaFree(arena);
  return int(error);
}
