#include "check.cuh"
#include "gh/metrics.cuh"
#include "gh/detail/metric_math.cuh"
#include <cuda/std/bit>
#include <cmath>

namespace gh::test {
namespace {
constexpr u32 capacity_cells = 70000, cases = 35;
__device__ double target[capacity_cells], prediction[capacity_cells], weights[capacity_cells];
__device__ Metric entries[70], candidate_entries[70];
__device__ MetricReport report, candidate;
__device__ MetricVerdict verdict[70];
__device__ MetricGate gate;
__device__ Status status;
__device__ Status numerical_status[1024];
__device__ SignalReport signal;
__device__ __align__(16) std::byte scratch[5 << 20];

// Independent small CPython-style partial expansion. Tests only, bounded to
// 128 operands; production exact summation uses integer limbs instead.
__device__ double expansion(const double* values, u32 size) {
  double partial[128]; u32 count = 0;
  for (u32 k = 0; k < size; ++k) {
    double x = values[k]; u32 used = 0;
    for (u32 j = 0; j < count; ++j) {
      double y = partial[j];
      if (fabs(x) < fabs(y)) { const double t = x; x = y; y = t; }
      const double hi = __dadd_rn(x, y), lo = __dsub_rn(y, __dsub_rn(hi, x));
      if (lo) partial[used++] = lo;
      x = hi;
    }
    partial[used++] = x; count = used;
  }
  double hi = 0, lo = 0;
  if (count) {
    hi = partial[--count];
    while (count) {
      const double x = hi, y = partial[--count];
      hi = __dadd_rn(x, y); lo = __dsub_rn(y, __dsub_rn(hi, x));
      if (lo) break;
    }
    if (count && ((lo < 0 && partial[count - 1] < 0) || (lo > 0 && partial[count - 1] > 0))) {
      const double y = __dmul_rn(lo, 2.0), x = __dadd_rn(hi, y);
      if (y == __dsub_rn(x, hi)) hi = x;
    }
  }
  return hi;
}
__global__ void numerical_checks() {
  const u32 test = blockIdx.x * blockDim.x + threadIdx.x;
  double values[128]; const u32 n = test % 128 + 1;
  for (u32 i = 0; i < n; ++i) {
    const u64 bits = mix(u64(test) * 128 + i);
    const u32 exponent = test % 3 == 0 ? 0 : test % 3 == 1 ? 1023 + i % 3 : u32(bits % 2000);
    values[i] = cuda::std::bit_cast<double>((u64(exponent) << 52) | (bits & 0xfffffffffffffULL));
  }
  Status& local = numerical_status[test]; local = {};
  const double actual = detail::positive_sum(n, [&](u32 i) { return values[i]; }, &local);
  GH_CHECK(local.errors == 0);
  GH_CHECK(cuda::std::bit_cast<u64>(actual) == cuda::std::bit_cast<u64>(expansion(values, n)));
  if (test) return;
  const auto one = [](u32) { return 1.0; };
  const double maximum = cuda::std::bit_cast<double>(0x7fefffffffffffffULL);
  GH_CHECK(detail::scaled_mean<2>(2, [=](u32) { return maximum; }, one, 2, &local) == maximum);
  GH_CHECK(detail::scaled_mean<1>(2, [=](u32) { return maximum; }, one, 2, &local) == maximum);
  const double tiny = cuda::std::bit_cast<double>(1ULL);
  GH_CHECK(detail::scaled_mean<2>(2, [=](u32) { return tiny; }, one, 2, &local) == tiny);
  GH_CHECK(detail::scaled_mean<1>(2, [=](u32 i) { return i ? 2.0 : INFINITY; },
    [](u32 i) { return i ? 1.0 : 0.0; }, 1, &local) == 2);
  // Weighted scale exponents exceed binary64 while the final mean is finite.
  GH_CHECK(detail::scaled_mean<2>(2, [=](u32) { return 0x1p1000; },
    [](u32) { return 0x1p1000; }, 0x1p1001, &local) == 0x1p1000);
  GH_CHECK(local.errors == 0);
  detail::PositiveSum overflow; overflow.add(maximum, &local); overflow.add(maximum, &local);
  overflow.value(&local); GH_CHECK(local.errors == numeric);
  local = {}; detail::PositiveSum invalid; invalid.add(-1, &local); GH_CHECK(local.errors == numeric);
  local = {};
  GH_CHECK(detail::positive_sum(3, [](u32 i) { return i ? 0x1p-53 : 1.0; }, &local) == 0x1.0000000000001p0);
  GH_CHECK(detail::positive_sum(2, [](u32 i) { return i ? 0x1p-53 : 1.0; }, &local) == 1.0);
  GH_CHECK(cuda::std::bit_cast<u64>(detail::positive_sum(2, [](u32) { return -0.0; }, &local)) == 0);
  // NumPy's eight-lane leaf differs deliberately from a sequential fold.
  GH_CHECK(detail::numpy_sum(9, [](u32 i) { return i == 0 ? 0x1p53 : i == 8 ? -0x1p53 : 1.0; }) == 6.0);
  GH_CHECK(detail::column_sum(9, 2, [](u32 i) { return i == 0 ? 0x1p53 : i == 8 ? -0x1p53 : 1.0; }) == 0.0);
  for (u32 n : {0u, 1u, 7u, 8u, 127u, 128u, 129u, 257u, 1025u})
    GH_CHECK(detail::numpy_sum(n, [](u32) { return 1.0; }) == n);
}

__device__ MetricInput input_for(u32 id) {
  MetricInput d{{target, capacity_cells}, {prediction, capacity_cells}, {}, 4, 1,
    MetricTask::binary, MetricProfile::real_data};
  if (id == 0 || id == 1) { d.task = MetricTask::regression; d.outputs = 2; }
  if (id == 0 || id == 3 || id == 5 || id == 8 || id == 10 || id == 12 || id == 13 || id == 14 || id == 16 || id == 17)
    d.profile = MetricProfile::synthetic;
  if (id == 0 || id == 3 || id == 12 || id == 13 || id == 14 || id == 15 || id == 16 || id == 17) d.weights = {weights, 4};
  if (id == 4 || id == 5 || id == 6 || id == 23 || id == 24 || id == 25) { d.task = MetricTask::multilabel; d.outputs = 5; }
  if (id == 7 || id == 8 || id == 9 || id == 10 || id == 18) { d.task = MetricTask::multiclass; d.outputs = 2; }
  if (id == 11) { d.task = MetricTask::regression; d.rows = 1; }
  if (id >= 26 && id <= 28) d.rows = id == 26 ? 257 : id == 27 ? 1025 : 65537;
  if (id >= 29 && id <= 32) {
    d.rows = 17;
    if (id >= 31) { d.outputs = 3; d.task = MetricTask::multilabel; }
    if (id == 30 || id == 32) { d.profile = MetricProfile::synthetic; d.weights = {weights,17}; }
  }
  if (id >= 33) d.task = MetricTask::regression;
  if (id == 19) d.rows = 0;
  if (id == 20) d.predictions.size = 0;
  return d;
}
__global__ void fixture(u32 id) {
  const auto d = input_for(id);
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < u64(d.rows) * d.outputs; i += u64(gridDim.x) * blockDim.x) {
    const u32 row = u32(i / d.outputs), col = u32(i % d.outputs);
    double y = row == 1 || row == 2, p = row == 0 ? .125 : row == 3 ? .875 : .5;
    if (d.task == MetricTask::regression) { y = double(row * 2 + col); p = y + (id == 0 ? 2 : 1); }
    if (d.task == MetricTask::multilabel) { y = col == 0 ? 0 : col == 1 ? 1 : double(row == 1 || row == 2); p = .5; }
    if (d.task == MetricTask::multiclass) { y = row % 2; p = .5; }
    if (id == 6) y = 0;
    if (id == 9 || id == 10) p += col == row % 2 ? 0x1p-24 : 0;
    if (id == 18) p = .6;
    if (id == 23) y = 1;
    if (id == 24) { y = col == 0; p = .5; }
    if (id == 25) { y = (row + col) % 2; p = y; }
    if (id >= 26 && id <= 28) { y = row % 2; p = y; }
    if (id >= 29 && id <= 32) { y = double((row * 7 + col * 3) % 5 < 2); p = double((row * 3 + col * 2) % 5) / 4; }
    if (id >= 33) { y = 2; p = id == 33 ? 2 : 3; }
    if (id == 16 && i == 0) p = INFINITY;
    if (id == 17 && i == 0) y = .25;
    prediction[i] = p;
    if (d.task != MetricTask::multiclass) target[i] = y;
    if (i < d.rows) {
      if (d.task == MetricTask::multiclass) target[i] = i % 2;
      weights[i] = id == 3 ? (i == 1 || i == 3 ? 2.0 : 1.0) : 1.0;
      if (id == 30 || id == 32) weights[i] = i % 4 ? ldexp(1.0, int(i % 9) - 4) : 0.0;
      if (id == 12) weights[i] = 0;
      if (id == 13 && i == 0) weights[i] = -1;
      if (id == 14 && i == 0) weights[i] = INFINITY;
    }
  }
}
__device__ Metric get(MetricName name, u32 output = aggregate_output) {
  for (u32 i = 0; i < report.count; ++i) if (entries[i].name == name && entries[i].output == output) return entries[i];
  GH_CHECK(false); return {};
}
__device__ void exact(MetricName name, double value, u32 output = aggregate_output) {
  const auto metric = get(name, output); GH_CHECK(metric.available == 1); GH_CHECK(metric.value == value);
}
struct RankAnswer { double ap, auc; };
// Exhaust every known score threshold; no production sorting/scan is reused.
// At most 51 cells and five groups keep reference reductions below eight terms.
__device__ RankAnswer rank_oracle(MetricInput d, u32 output, bool pooled) {
  const u32 size = pooled ? d.rows * d.outputs : d.rows;
  u32 tp[5], fp[5], count = 0, positives = 0;
  for (u32 k = 0; k < size; ++k) positives += target[pooled ? k : k * d.outputs + output] == 1;
  const u32 negatives = size - positives;
  for (int level = 4; level >= 0; --level) {
    u32 p = 0, n = 0, tied = 0;
    for (u32 k = 0; k < size; ++k) {
      const u32 i = pooled ? k : k * d.outputs + output;
      tied += prediction[i] == double(level) / 4;
      if (prediction[i] >= double(level) / 4) { p += target[i] == 1; n += target[i] == 0; }
    }
    if (tied) { tp[count] = p; fp[count++] = n; }
  }
  double ap = 0, auc = 0, previous_fpr = 0, previous_tpr = 0;
  for (u32 k = count; k; --k) {
    const u32 j = k - 1;
    const double recall = __ddiv_rn(double(tp[j]), positives);
    const double next = j ? __ddiv_rn(double(tp[j - 1]), positives) : 0;
    ap = __dadd_rn(ap, __dmul_rn(__dsub_rn(next, recall), __ddiv_rn(double(tp[j]), tp[j] + fp[j])));
  }
  for (u32 k = 0; k < count; ++k) {
    if (k && k + 1 < count && int(tp[k + 1]) - 2 * int(tp[k]) + int(tp[k - 1]) == 0 &&
        int(fp[k + 1]) - 2 * int(fp[k]) + int(fp[k - 1]) == 0) continue;
    const double fpr = __ddiv_rn(double(fp[k]), negatives), tpr = __ddiv_rn(double(tp[k]), positives);
    auc = __dadd_rn(auc, __ddiv_rn(__dmul_rn(__dsub_rn(fpr, previous_fpr), __dadd_rn(tpr, previous_tpr)), 2.0));
    previous_fpr = fpr; previous_tpr = tpr;
  }
  if (d.profile == MetricProfile::synthetic) {
    double mass[64], terms[5];
    for (u32 k = 0; k < size; ++k) mass[k] = target[k * d.outputs + output] == 1 ? weights[k] : 0;
    const double positive = expansion(mass, size);
    for (u32 k = 0; k < size; ++k) mass[k] = target[k * d.outputs + output] == 0 ? weights[k] : 0;
    const double negative = expansion(mass, size);
    double before = 0, correction = 0;
    for (u32 level = 0; level < 5; ++level) {
      for (u32 k = 0; k < size; ++k) mass[k] = prediction[k * d.outputs + output] == double(level) / 4 && target[k * d.outputs + output] == 1 ? __ddiv_rn(weights[k], positive) : 0;
      const double p = expansion(mass, size);
      for (u32 k = 0; k < size; ++k) mass[k] = prediction[k * d.outputs + output] == double(level) / 4 && target[k * d.outputs + output] == 0 ? __ddiv_rn(weights[k], negative) : 0;
      const double n = expansion(mass, size);
      terms[level] = __dmul_rn(p, __dadd_rn(before, __dmul_rn(.5, n)));
      const double change = __dsub_rn(n, correction), updated = __dadd_rn(before, change);
      correction = __dsub_rn(__dsub_rn(updated, before), change); before = updated;
    }
    auc = expansion(terms, 5);
  }
  return {-ap, auc};
}
__global__ void start(u32 id);
__global__ void gate_start(u32 id);
__global__ void check(u32 id) {
  GH_CHECK(status.done == 1);
  const auto d = input_for(id);
  if (id >= 12 && id <= 22) {
    const u32 error = id <= 14 || id == 16 || id == 17 || id == 18 ? input : id >= 21 ? capacity : shape;
    GH_CHECK(status.errors & error);
    if (id == 21) GH_CHECK(status.required_bytes > 1);
  } else {
    if (status.errors) printf("metric case %u errors=%u required_bytes=%llu\n", id, status.errors, static_cast<unsigned long long>(status.required_bytes));
    succeeded(status);
    for (u32 i = 0; i < report.count; ++i) if (entries[i].available) GH_CHECK(detail::finite_metric(entries[i].value));
    if (id == 0) { exact(MetricName::rmse, 2); exact(MetricName::mae, 2); for (u32 k = 0; k < 2; ++k) { exact(MetricName::rmse, 2, k); exact(MetricName::mae, 2, k); } }
    if (id == 1) { exact(MetricName::mse, 1); exact(MetricName::rmse, 1); exact(MetricName::mae, 1); exact(MetricName::r2, __dsub_rn(1.0, .2)); }
    if (id == 2) { exact(MetricName::auc, .5); exact(MetricName::average_precision, __ddiv_rn(2.0, 3.0)); exact(MetricName::f1, .8); exact(MetricName::accuracy, .75); exact(MetricName::brier, .3203125); }
    if (id == 3) { exact(MetricName::auc, __ddiv_rn(1.0, 3.0)); exact(MetricName::accuracy, __ddiv_rn(2.0, 3.0)); exact(MetricName::brier, .3828125); }
    if (id == 4) {
      exact(MetricName::micro_ap, .5); exact(MetricName::micro_auc, .5); exact(MetricName::macro_auc, .5);
      exact(MetricName::macro_ap, .625); GH_CHECK(report.ap_outputs == 4 && report.auc_outputs == 3);
      exact(MetricName::precision_at_1, 0); exact(MetricName::precision_at_3, .5); exact(MetricName::precision_at_5, .5);
      exact(MetricName::hamming_loss, .5); exact(MetricName::exact_match, 0); exact(MetricName::brier, .25);
    }
    if (id == 5) { GH_CHECK(!get(MetricName::auc).available); exact(MetricName::auc, .5, 2); exact(MetricName::brier, .25); exact(MetricName::accuracy, .5); }
    if (id == 6) { GH_CHECK(!get(MetricName::macro_ap).available && !get(MetricName::macro_auc).available && !get(MetricName::micro_auc).available); exact(MetricName::micro_ap, 0); exact(MetricName::macro_f1, 0); }
    if (id == 7 || id == 8) { exact(MetricName::accuracy, .5); exact(MetricName::logloss, -log(.5)); if (id == 7) { exact(MetricName::brier, .5); exact(MetricName::macro_f1, __ddiv_rn(1.0, 3.0)); } }
    if (id == 9 || id == 10) { exact(MetricName::accuracy, 1); const double p = .5 + 0x1p-24; exact(MetricName::logloss, -log(id == 10 ? __ddiv_rn(p, 1 + 0x1p-24) : p)); }
    if (id == 11) { GH_CHECK(!get(MetricName::r2).available); exact(MetricName::rmse, 1); }
    if (id == 23) { exact(MetricName::macro_ap, 1); exact(MetricName::micro_ap, 1); GH_CHECK(!get(MetricName::macro_auc).available); }
    if (id == 24) { exact(MetricName::precision_at_1, 1); exact(MetricName::precision_at_3, __ddiv_rn(1.0, 3.0)); exact(MetricName::precision_at_5, .2); }
    if (id >= 25 && id <= 28) { exact(id == 25 ? MetricName::micro_auc : MetricName::auc, 1); exact(id == 25 ? MetricName::micro_ap : MetricName::average_precision, 1); exact(MetricName::brier, 0); exact(MetricName::hamming_loss, 0); exact(MetricName::exact_match, 1); }
    if (id == 29 || id == 30) { const auto reference = rank_oracle(d, 0, false); exact(MetricName::auc, reference.auc); if (id == 29) exact(MetricName::average_precision, reference.ap); }
    if (id == 31) {
      const auto pooled = rank_oracle(d, 0, true); exact(MetricName::micro_ap, pooled.ap); exact(MetricName::micro_auc, pooled.auc);
      double ap = 0, auc = 0;
      for (u32 k = 0; k < 3; ++k) { const auto reference = rank_oracle(d, k, false); ap = __dadd_rn(ap, reference.ap); auc = __dadd_rn(auc, reference.auc); }
      exact(MetricName::macro_ap, __ddiv_rn(ap, 3.0)); exact(MetricName::macro_auc, __ddiv_rn(auc, 3.0));
    }
    if (id == 32) for (u32 k = 0; k < 3; ++k) exact(MetricName::auc, rank_oracle(d,k,false).auc, k);
    if (id >= 33) { exact(MetricName::r2, id == 33 ? 1 : 0); exact(MetricName::mse, id == 33 ? 0 : 1); }
  }
  if (id + 1 == cases) gate_start<<<1, 1, 0, cudaStreamTailLaunch>>>(0);
  else start<<<1, 1, 0, cudaStreamTailLaunch>>>(id + 1);
  submitted(cudaGetLastError());
}
__global__ void start(u32 id) {
  status = {}; report = {{entries, id == 22 ? 0u : 70u}};
  fixture<<<64, 256>>>(id); submitted(cudaGetLastError());
  submitted(evaluate_metrics(input_for(id), &report, {scratch, id == 21 ? 1u : sizeof(scratch)}, &status));
  check<<<1, 1, 0, cudaStreamTailLaunch>>>(id); submitted(cudaGetLastError());
}

__global__ void signal_start(u32 id);
__global__ void gate_check(u32 id) {
  GH_CHECK(status.done);
  if (id == 2) GH_CHECK(status.errors == shape);
  else if (id == 1) { GH_CHECK(status.errors == input); GH_CHECK(gate.invalid == 2); GH_CHECK(verdict[0] == MetricVerdict::invalid && verdict[1] == MetricVerdict::invalid); }
  else {
    succeeded(status); GH_CHECK(gate.checked == 4 && gate.regressions == 2 && gate.unavailable == 1 && !gate.invalid);
    GH_CHECK(verdict[0] == MetricVerdict::pass && verdict[1] == MetricVerdict::regression && verdict[2] == MetricVerdict::regression && verdict[3] == MetricVerdict::not_applicable && verdict[4] == MetricVerdict::pass);
  }
  if (id == 2) signal_start<<<1, 1, 0, cudaStreamTailLaunch>>>(0);
  else gate_start<<<1, 1, 0, cudaStreamTailLaunch>>>(id + 1);
  submitted(cudaGetLastError());
}
__global__ void gate_start(u32 id) {
  status = {}; report = {{entries,70},5,1}; candidate = {{candidate_entries,70},5,1};
  entries[0] = {MetricName::mse,aggregate_output,-0.0,1}; candidate_entries[0] = {MetricName::mse,aggregate_output,0.0,1};
  entries[1] = {MetricName::r2,aggregate_output,0x1p1023,1}; candidate_entries[1] = {MetricName::r2,aggregate_output,-0x1p1023,1};
  entries[2] = {MetricName::mae,aggregate_output,1,1}; candidate_entries[2] = {MetricName::mae,aggregate_output,nextafter(1.0,2.0),1};
  entries[3] = {MetricName::auc,aggregate_output,0,0}; candidate_entries[3] = entries[3];
  entries[4] = {MetricName::accuracy,aggregate_output,1,1}; candidate_entries[4] = entries[4];
  if (id == 1) { candidate_entries[0].available = 0; candidate_entries[1].value = INFINITY; }
  if (id == 2) candidate.ap_outputs = 1;
  submitted(compare_metrics(&report,&candidate,{verdict,70},&gate,&status));
  gate_check<<<1,1,0,cudaStreamTailLaunch>>>(id); submitted(cudaGetLastError());
}
__global__ void signal_check(u32 id) {
  GH_CHECK(status.done);
  if (id == 3) GH_CHECK(status.errors == input);
  else {
    succeeded(status);
    if (id <= 1) {
      GH_CHECK(signal.fixed.true_positive == 2 && signal.fixed.false_positive == 3);
      GH_CHECK(signal.selected.threshold == .6 && signal.selected.true_positive == 2 && signal.selected.false_positive == 2);
      GH_CHECK(signal.selected.meets_five_percent_fpr && !signal.fixed.meets_five_percent_fpr);
      GH_CHECK(signal.selected.recall == 1 && signal.selected.false_positive_rate == .05 && signal.selected.precision == .5);
    } else { GH_CHECK(signal.selected.threshold == nextafter(.9, double(INFINITY))); GH_CHECK(signal.selected.true_positive == 0 && signal.selected.false_positive == 0); }
  }
  if (id < 3) signal_start<<<1,1,0,cudaStreamTailLaunch>>>(id+1);
  else printf("GH_GPU_ACTIVITY metrics cases=35 gates=3 signals=4 positive-sum-oracles=1024 checks=pass\n");
  submitted(cudaGetLastError());
}
__global__ void signal_start(u32 id) {
  status = {};
  for (u32 i = 0; i < 42; ++i) {
    target[i] = id == 3 ? 0 : double(i < 2);
    prediction[i] = i == 0 ? .8 : i == 1 || i == 3 ? .6 : i == 2 ? .7 : i == 4 ? .55 : .1;
    if (id == 2) prediction[i] = i < 2 ? .1 : .9;
  }
  submitted(signal_metrics({{target,42},{prediction,42},42}, id == 1 ? ThresholdMode::frozen : ThresholdMode::validation, .6, &signal,{scratch,sizeof(scratch)},&status));
  signal_check<<<1,1,0,cudaStreamTailLaunch>>>(id); submitted(cudaGetLastError());
}
}
__global__ void run() {
  numerical_checks<<<32,32>>>(); submitted(cudaGetLastError());
  start<<<1,1,0,cudaStreamTailLaunch>>>(0); submitted(cudaGetLastError());
}
}
