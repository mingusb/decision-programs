#include "check.cuh"
#include "gh/reference.cuh"

namespace gh::test {
namespace {
struct Number { const char* text; u64 bits; bool accepted{true}; };
__device__ const Number numbers[]{
  {"0", 0}, {"-0", 0x8000000000000000ULL}, {"-0.0e+308", 0x8000000000000000ULL},
  {"0.1", 0x3fb999999999999aULL}, {"0.10000000000000001", 0x3fb999999999999aULL},
  {"0.10000000000000002", 0x3fb999999999999bULL}, {"1.0000000000000001", 0x3ff0000000000000ULL},
  {"1.0000000000000002", 0x3ff0000000000001ULL}, {"5e-324", 1}, {"-5e-324", 0x8000000000000001ULL},
  {"4.9406564584124654e-324", 1}, {"2.4703282292062327e-324", 0}, {"2.4703282292062328e-324", 1},
  {"-2.4703282292062327e-324", 0x8000000000000000ULL}, {"1e-342", 0},
  {"2.2250738585072011e-308", 0x000fffffffffffffULL}, {"2.2250738585072014e-308", 0x0010000000000000ULL},
  {"1.7976931348623157e308", 0x7fefffffffffffffULL}, {"1.7976931348623158e308", 0x7fefffffffffffffULL},
  {"1.7976931348623159e308", 0x7ff0000000000000ULL}, {"-1.7976931348623159e308", 0xfff0000000000000ULL},
  {"9007199254740993", 0x4340000000000000ULL}, {"9007199254740995", 0x4340000000000002ULL},
  {"9999999999999999999", 0x43e158e460913d00ULL},
  {"-9007199254740993", 0xc340000000000000ULL}, {"1.25E+2", 0x405f400000000000ULL},
  {"1e309", 0, false}, {"1e-343", 0, false}, {"12345678901234567890", 0, false},
  {"01", 0, false}, {"+1", 0, false}, {".1", 0, false}, {"1.", 0, false},
  {"1e", 0, false}, {"1e+", 0, false}, {"NaN", 0, false}, {"Infinity", 0, false},
  {" 1", 0, false}, {"1 ", 0, false}, {"--1", 0, false}, {"1e9999999999", 0, false}, {"", 0, false}
};
__device__ u64 length(const char* text) { u64 n = 0; while (text[n]) ++n; return n; }
__global__ void number_checks() {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < sizeof(numbers) / sizeof(numbers[0])) {
    const auto value = numbers[i]; double out = 17;
    GH_CHECK(parse_decimal({value.text, length(value.text)}, &out) == value.accepted);
    GH_CHECK(u64(__double_as_longlong(out)) == (value.accepted ? value.bits : 0x4031000000000000ULL));
  }
  // Integer lattices have known spacing: two at 2^53 and 2048 at 2^63.
  // The latter also exercises the supported 19-digit mantissa boundary.
  if (i < 2048) {
    for (u32 scale = 0; scale < 2; ++scale) {
      char reversed[32], text[32]; u32 n = 0; u64 value = (1ULL << (scale ? 63 : 53)) + i;
      do { reversed[n++] = char('0' + value % 10); value /= 10; } while (value);
      for (u32 j = 0; j < n; ++j) text[j] = reversed[n - j - 1];
      const u64 expected = scale ? 0x43e0000000000000ULL + u64(i > 1024) :
        0x4340000000000000ULL + i / 2 + ((i & 1) && ((i / 2) & 1));
      double out{}; GH_CHECK(parse_decimal({text, n}, &out));
      GH_CHECK(u64(__double_as_longlong(out)) == expected);
    }
  }
}
__device__ char json[8192];
__device__ u32 json_size;
__device__ Metric computed_values[16], reference_values[16], aligned_values[16];
__device__ MetricReport computed, reference, aligned;
__device__ MetricVerdict verdicts[16];
__device__ MetricGate gate;
__device__ Status status;
__device__ const char* regression = "\"mse\":0.5,\"rmse\":0.5,\"mae\":0.5,\"r2\":0.5,\"selection_metric\":\"mse\",\"selection_value\":0.5";
__device__ const char* binary = "\"log_loss\":0.5,\"brier\":0.5,\"hamming_loss\":0.5,\"exact_match_accuracy\":0.5,\"accuracy\":0.5,\"f1\":0.5,\"roc_auc\":0.5,\"average_precision\":0.5,\"selection_metric\":\"log_loss\",\"selection_value\":0.5,\"log_clip_epsilon\":1e-15";
__device__ const char* multiclass = "\"log_loss\":0.5,\"brier\":0.5,\"accuracy\":0.5,\"macro_f1\":0.5,\"selection_metric\":\"log_loss\",\"selection_value\":0.5,\"log_clip_epsilon\":1e-15";
__device__ const char* multilabel = "\"log_loss\":0.5,\"brier\":0.5,\"hamming_loss\":0.5,\"exact_match_accuracy\":0.5,\"micro_f1\":0.5,\"macro_f1\":0.5,\"micro_ap\":0.5,\"macro_ap\":0.5,\"macro_auc\":0.5,\"precision_at_1\":0.5,\"precision_at_3\":0.5,\"precision_at_5\":0.5,\"macro_ap_labels\":4,\"macro_auc_labels\":3,\"selection_metric\":\"log_loss\",\"selection_value\":0.5,\"log_clip_epsilon\":1e-15";
__device__ void append(const char* text) { while (*text) { GH_CHECK(json_size < sizeof(json)); json[json_size++] = *text++; } }
__device__ void replace(const char* old, const char* value) {
  const u32 n = u32(length(old)), m = u32(length(value));
  for (u32 i = 0; i + n <= json_size; ++i) {
    u32 j = 0; while (j < n && json[i + j] == old[j]) ++j;
    if (j != n) continue;
    GH_CHECK(json_size + m - n < sizeof(json));
    if (m > n) for (u32 k = json_size; k > i + n; --k) json[k + m - n - 1] = json[k - 1];
    else for (u32 k = i + n; k < json_size; ++k) json[k + m - n] = json[k];
    for (u32 k = 0; k < m; ++k) json[i + k] = value[k];
    json_size += m - n; return;
  }
  GH_CHECK(false);
}
__device__ void add(MetricName name, bool available = true) {
  computed_values[computed.count++] = {name, aggregate_output, available ? .5 : 0.0, u32(available)};
}
__global__ void setup(u32 id);
__device__ void next(u32 id) {
  if (id == 27) printf("reference checks passed: 42 decimal landmarks, 4096 integer rounding cases, 27 JSON cases\n");
  else { setup<<<1, 1, 0, cudaStreamTailLaunch>>>(id); submitted(cudaGetLastError()); }
}
__global__ void checked_gate(u32 id) {
  GH_CHECK(status.done == 1);
  GH_CHECK(status.errors == (id == 25 ? u32(shape) : u32(input)));
  next(id + 1);
}
__global__ void checked(u32 id) {
  GH_CHECK(status.done == 1);
  if (id >= 8 && id <= 24) GH_CHECK(status.errors == (id == 15 ? u32(capacity) : u32(input)));
  else {
    succeeded(status);
    GH_CHECK(reference.count == (computed.task == MetricTask::multilabel ? 12 : computed.count));
    GH_CHECK(reference.count == aligned.count && reference.outputs == computed.outputs);
    for (u32 i = 0; i < reference.count; ++i) {
      const auto a = reference_values[i], b = aligned_values[i];
      GH_CHECK(a.name == b.name && a.output == aggregate_output && a.name != MetricName::micro_auc);
      GH_CHECK(a.available == b.available || id == 26);
      GH_CHECK(u64(__double_as_longlong(a.value)) == (a.available ? 0x3fe0000000000000ULL : 0));
    }
    if (id == 3) GH_CHECK(reference.ap_outputs == 4 && reference.auc_outputs == 3);
    if (id == 5) GH_CHECK(reference.ap_outputs == 0 && reference.auc_outputs == 0);
    if (id >= 25) {
      status = {}; submitted(compare_metrics(&reference, &aligned, {verdicts, 16}, &gate, &status));
      checked_gate<<<1, 1, 0, cudaStreamTailLaunch>>>(id); submitted(cudaGetLastError()); return;
    }
  }
  next(id + 1);
}
__global__ void setup(u32 id) {
  status = {}; json_size = 0; computed = {}; reference = {}; aligned = {};
  computed.metrics = {computed_values, 16}; reference.metrics = {reference_values, id == 15 ? 0u : 16u}; aligned.metrics = {aligned_values, 16};
  const bool multi = id == 3 || id == 5 || id == 13 || id == 25;
  const bool bin = id == 1 || id == 7 || id == 12 || id == 26;
  computed.task = multi ? MetricTask::multilabel : bin ? MetricTask::binary : id == 2 ? MetricTask::multiclass : MetricTask::regression;
  computed.outputs = multi ? 5 : id == 2 ? 3 : 1; computed.profile = MetricProfile::real_data;
  const char* body = multi ? multilabel : bin ? binary : id == 2 ? multiclass : regression;
  if (computed.task == MetricTask::regression) {
    add(MetricName::mse); add(MetricName::rmse); add(MetricName::mae); add(MetricName::r2, id != 4);
  } else {
    add(MetricName::logloss); add(MetricName::brier);
    if (id == 2) { add(MetricName::accuracy); add(MetricName::macro_f1); }
    else {
      add(MetricName::hamming_loss); add(MetricName::exact_match);
      if (bin) { add(MetricName::accuracy); add(MetricName::f1); add(MetricName::auc, id != 7 && id != 26); add(MetricName::average_precision); }
      else {
        add(MetricName::micro_f1); add(MetricName::macro_f1); add(MetricName::micro_ap); add(MetricName::micro_auc);
        add(MetricName::macro_ap, id != 5); add(MetricName::macro_auc, id != 5);
        add(MetricName::precision_at_1); add(MetricName::precision_at_3); add(MetricName::precision_at_5);
        computed.ap_outputs = id == 5 ? 0 : 4; computed.auc_outputs = id == 5 ? 0 : id == 25 ? 2 : 3;
      }
    }
  }
  append("{\"reference\":\"CPU float64 common metric implementation\",\"fixture_sha256\":\"");
  for (u32 i = 0; i < 64; ++i) append("0");
  append("\",\"training_fixture_sha256\":\""); for (u32 i = 0; i < 64; ++i) append("1");
  append("\",\"predictions_sha256\":\""); for (u32 i = 0; i < 64; ++i) append("a");
  append("\",\"metrics\":{"); append(body);
  if (id == 8) append(",\"mse\":0.5");
  if (id == 9) append(",\"bogus\":0.5");
  if (id == 24) append(",");
  append("},\"training_mean_baseline\":{"); append(body); append("}");
  if (id == 17) { append(",\"metrics\":{"); append(body); append("}"); }
  append("}");
  if (id == 4) replace("\"r2\":0.5", "\"r2\":null");
  if (id == 5) {
    replace("\"macro_ap\":0.5", "\"macro_ap\":null"); replace("\"macro_auc\":0.5", "\"macro_auc\":null");
    replace("\"macro_ap_labels\":4", "\"macro_ap_labels\":0"); replace("\"macro_auc_labels\":3", "\"macro_auc_labels\":0");
  }
  if (id == 6) replace("\"mse\":", "\"\\u006dse\":");
  if (id == 7) replace("\"roc_auc\":0.5", "\"roc_auc\":null");
  if (id == 10) replace("\"mae\":0.5,", "");
  if (id == 11) replace("\"selection_value\":0.5", "\"selection_value\":0.25");
  if (id == 12) replace("1e-15", "1e-14");
  if (id == 13) replace("\"macro_ap_labels\":4", "\"macro_ap_labels\":6");
  if (id == 14) --computed.count;
  if (id == 16) append(" true");
  if (id == 18) replace("0000", "z000");
  if (id == 19) replace("\"mse\":", "\"mse\\u0000\":");
  if (id == 20) replace("\"selection_value\":0.5", "\"selection_value\":null");
  if (id == 21) replace("\"mse\":0.5", "\"mse\":1.8e308");
  if (id == 22) replace("CPU float64", "GPU float64");
  if (id == 23) replace("\"mse\":0.5", "\"mse\":01");
  submitted(decode_metric_reference({reinterpret_cast<const std::byte*>(json), json_size}, &computed, &reference, &aligned, &status));
  checked<<<1, 1, 0, cudaStreamTailLaunch>>>(id); submitted(cudaGetLastError());
}
}
__global__ void run() {
  GH_CHECK(!parse_decimal({}, nullptr));
  GH_CHECK(decode_metric_reference({}, nullptr, nullptr, nullptr, nullptr) == cudaErrorInvalidValue);
  number_checks<<<8, 256>>>(); submitted(cudaGetLastError());
  next(0);
}
}
