#include "gh/reference.cuh"

namespace gh {
namespace {
constexpr u32 limbs = 36;
struct Integer {
  u32 word[limbs]{};
  __device__ int bits() const {
    for (int i = limbs - 1; i >= 0; --i) if (word[i]) return i * 32 + 32 - __clz(word[i]);
    return 0;
  }
  __device__ bool times_five() {
    u64 carry = 0;
    for (u32 i = 0; i < limbs; ++i) { const u64 v = u64(word[i]) * 5 + carry; word[i] = u32(v); carry = v >> 32; }
    return !carry;
  }
  __device__ bool shift(u32 n) {
    if (n > limbs * 32 || bits() + n > limbs * 32) return false;
    const u32 whole = n / 32, part = n % 32;
    for (int i = limbs - 1; i >= 0; --i) {
      u32 v = u32(i) >= whole ? word[i - whole] << part : 0;
      if (part && u32(i) > whole) v |= word[i - whole - 1] >> (32 - part);
      word[i] = v;
    }
    return true;
  }
  __device__ int compare(const Integer& b) const {
    for (int i = limbs - 1; i >= 0; --i) if (word[i] != b.word[i]) return word[i] > b.word[i] ? 1 : -1;
    return 0;
  }
  __device__ void subtract(const Integer& b) {
    u64 borrow = 0;
    for (u32 i = 0; i < limbs; ++i) {
      const u64 a = word[i], sub = u64(b.word[i]) + borrow;
      word[i] = u32(a - sub); borrow = a < sub;
    }
  }
};
__device__ bool digit(char c) { return c >= '0' && c <= '9'; }
__device__ u64 bits(double x) { return __double_as_longlong(x); }
__device__ bool finite(double x) { return (bits(x) & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL; }
__device__ bool decimal_parts(Array<const char> text, u64& mantissa, int& exponent, bool& negative) {
  if (!text.size || text.size > 768 || !contains(text, text.size)) return false;
  u64 p = 0; negative = text.data[p] == '-'; p += negative;
  if (p == text.size || !digit(text.data[p])) return false;
  u32 significant = 0, fraction = 0;
  const auto append = [&](char c) {
    if (mantissa || c != '0') { if (++significant > 19) return false; mantissa = mantissa * 10 + u32(c - '0'); }
    return true;
  };
  if (text.data[p] == '0') ++p;
  else while (p < text.size && digit(text.data[p])) if (!append(text.data[p++])) return false;
  if (p < text.size && text.data[p] == '.') {
    ++p; const u64 first = p;
    while (p < text.size && digit(text.data[p])) { ++fraction; if (!append(text.data[p++])) return false; }
    if (p == first) return false;
  }
  int explicit_exponent = 0;
  if (p < text.size && (text.data[p] == 'e' || text.data[p] == 'E')) {
    ++p; bool minus = false;
    if (p < text.size && (text.data[p] == '+' || text.data[p] == '-')) minus = text.data[p++] == '-';
    const u64 first = p;
    while (p < text.size && digit(text.data[p])) {
      explicit_exponent = explicit_exponent * 10 + text.data[p++] - '0';
      if (explicit_exponent > 4096) return false;
    }
    if (p == first) return false;
    if (minus) explicit_exponent = -explicit_exponent;
  }
  exponent = explicit_exponent - int(fraction);
  return p == text.size && exponent >= -342 && exponent <= 308;
}
struct Text { char data[80]{}; u32 size{}; };
__device__ bool equal(const Text& a, const char* b) {
  u32 i = 0; while (i < a.size && b[i] && a.data[i] == b[i]) ++i;
  return i == a.size && !b[i];
}
struct Json {
  Array<const std::byte> bytes;
  u64 at{};
  __device__ char peek() const { return at < bytes.size ? char(bytes.data[at]) : '\0'; }
  __device__ void spaces() { while (peek() == ' ' || peek() == '\n' || peek() == '\r' || peek() == '\t') ++at; }
  __device__ bool take(char c) { spaces(); if (peek() != c || at == bytes.size) return false; ++at; return true; }
  __device__ bool literal(const char* text) {
    while (*text) if (at == bytes.size || peek() != *text++) return false; else ++at;
    return true;
  }
  __device__ bool string(Text& text) {
    text.size = 0;
    if (!take('"')) return false;
    while (at < bytes.size && peek() != '"') {
      u32 c = u32(static_cast<unsigned char>(peek())); ++at;
      if (c < 32 || c > 127) return false;
      if (c == '\\') {
        if (at == bytes.size) return false;
        c = u32(static_cast<unsigned char>(peek())); ++at;
        if (c == 'u') {
          c = 0;
          for (u32 i = 0; i < 4; ++i) {
            const char h = peek(); const int v = digit(h) ? h - '0' : h >= 'a' && h <= 'f' ? h - 'a' + 10 : h >= 'A' && h <= 'F' ? h - 'A' + 10 : -1;
            if (at == bytes.size || v < 0) return false;
            ++at; c = c * 16 + u32(v);
          }
          if (c > 127) return false;
        } else if (c == 'b') c = '\b'; else if (c == 'f') c = '\f';
        else if (c == 'n') c = '\n'; else if (c == 'r') c = '\r'; else if (c == 't') c = '\t';
        else if (c != '"' && c != '\\' && c != '/') return false;
      }
      if (text.size == sizeof(text.data)) return false;
      text.data[text.size++] = char(c);
    }
    return take('"');
  }
  __device__ bool number(double& value, bool& available) {
    spaces(); available = peek() != 'n';
    if (!available) { value = 0; return literal("null"); }
    const u64 first = at;
    while (digit(peek()) || peek() == '-' || peek() == '+' || peek() == '.' || peek() == 'e' || peek() == 'E') ++at;
    return parse_decimal({reinterpret_cast<const char*>(bytes.data + first), at - first}, &value) && finite(value);
  }
};
__device__ int metric_key(const Text& key) {
  if (equal(key, "mse")) return int(MetricName::mse);
  if (equal(key, "rmse")) return int(MetricName::rmse);
  if (equal(key, "mae")) return int(MetricName::mae);
  if (equal(key, "r2")) return int(MetricName::r2);
  if (equal(key, "log_loss")) return int(MetricName::logloss);
  if (equal(key, "accuracy")) return int(MetricName::accuracy);
  if (equal(key, "brier")) return int(MetricName::brier);
  if (equal(key, "roc_auc")) return int(MetricName::auc);
  if (equal(key, "f1")) return int(MetricName::f1);
  if (equal(key, "average_precision")) return int(MetricName::average_precision);
  if (equal(key, "hamming_loss")) return int(MetricName::hamming_loss);
  if (equal(key, "exact_match_accuracy")) return int(MetricName::exact_match);
  if (equal(key, "micro_f1")) return int(MetricName::micro_f1);
  if (equal(key, "macro_f1")) return int(MetricName::macro_f1);
  if (equal(key, "micro_ap")) return int(MetricName::micro_ap);
  if (equal(key, "macro_ap")) return int(MetricName::macro_ap);
  if (equal(key, "macro_auc")) return int(MetricName::macro_auc);
  if (equal(key, "precision_at_1")) return int(MetricName::precision_at_1);
  if (equal(key, "precision_at_3")) return int(MetricName::precision_at_3);
  if (equal(key, "precision_at_5")) return int(MetricName::precision_at_5);
  return -1;
}
__device__ u32 flag(MetricName name) { return 1u << u32(name); }
struct Set {
  double values[21]{}, selection_value{}, clip{};
  u32 present{}, available{}, extras{}, ap_outputs{}, auc_outputs{};
  int selection{-1};
};
__device__ bool unique(u32& set, u32 bit) { if (set & bit) return false; set |= bit; return true; }
__device__ bool metrics(Json& json, Set& set) {
  if (!json.take('{')) return false;
  do {
    Text key; if (!json.string(key) || !json.take(':')) return false;
    const int name = metric_key(key);
    if (name >= 0) {
      if (!unique(set.present, 1u << name)) return false;
      bool available; if (!json.number(set.values[name], available)) return false;
      if (available) set.available |= 1u << name;
    } else if (equal(key, "selection_metric")) {
      Text value; if (!unique(set.extras, 1) || !json.string(value)) return false;
      set.selection = metric_key(value);
    } else {
      const u32 bit = equal(key, "selection_value") ? 2 : equal(key, "log_clip_epsilon") ? 4 :
        equal(key, "macro_ap_labels") ? 8 : equal(key, "macro_auc_labels") ? 16 : 0;
      double value; bool available;
      if (!bit || !unique(set.extras, bit) || !json.number(value, available) || !available) return false;
      if (bit == 2) set.selection_value = value;
      else if (bit == 4) set.clip = value;
      else {
        if (value < 0 || value > UINT32_MAX || double(u32(value)) != value) return false;
        if (bit == 8) set.ap_outputs = u32(value); else set.auc_outputs = u32(value);
      }
    }
    if (json.take('}')) return true;
  } while (json.take(','));
  return false;
}
__device__ bool hash(const Text& text) {
  if (text.size != 64) return false;
  for (u32 i = 0; i < 64; ++i) if (!digit(text.data[i]) && !(text.data[i] >= 'a' && text.data[i] <= 'f')) return false;
  return true;
}
__device__ bool decode(Json& json, Set& values, Set& baseline) {
  u32 seen = 0;
  if (!json.take('{')) return false;
  do {
    Text key; if (!json.string(key) || !json.take(':')) return false;
    const u32 bit = equal(key, "reference") ? 1 : equal(key, "fixture_sha256") ? 2 :
      equal(key, "training_fixture_sha256") ? 4 : equal(key, "predictions_sha256") ? 8 :
      equal(key, "metrics") ? 16 : equal(key, "training_mean_baseline") ? 32 : 0;
    if (!bit || !unique(seen, bit)) return false;
    if (bit >= 16) { if (!metrics(json, bit == 16 ? values : baseline)) return false; }
    else {
      Text value; if (!json.string(value) || (bit == 1 ? !equal(value, "CPU float64 common metric implementation") : !hash(value))) return false;
    }
    if (json.take('}')) { json.spaces(); return seen == 63 && json.at == json.bytes.size; }
  } while (json.take(','));
  return false;
}
__device__ u32 required(MetricTask task, u32 outputs) {
  if (task == MetricTask::regression) return flag(MetricName::mse) | flag(MetricName::rmse) | flag(MetricName::mae) | flag(MetricName::r2);
  u32 mask = flag(MetricName::logloss) | flag(MetricName::brier);
  if (task == MetricTask::multiclass) return mask | flag(MetricName::accuracy) | flag(MetricName::macro_f1);
  mask |= flag(MetricName::hamming_loss) | flag(MetricName::exact_match);
  if (outputs == 1) return mask | flag(MetricName::accuracy) | flag(MetricName::f1) | flag(MetricName::auc) | flag(MetricName::average_precision);
  mask |= flag(MetricName::micro_f1) | flag(MetricName::macro_f1) | flag(MetricName::micro_ap) |
    flag(MetricName::macro_ap) | flag(MetricName::macro_auc) | flag(MetricName::precision_at_1);
  if (outputs >= 3) mask |= flag(MetricName::precision_at_3);
  if (outputs >= 5) mask |= flag(MetricName::precision_at_5);
  return mask;
}
__device__ bool schema(const Set& set, const MetricReport& report, u32 mask) {
  const bool regression = report.task == MetricTask::regression;
  const bool multi = report.task == MetricTask::multilabel && report.outputs > 1;
  const u32 extras = regression ? 3 : multi ? 31 : 7;
  const auto selected = regression ? MetricName::mse : MetricName::logloss;
  return set.present == mask && set.extras == extras && set.selection == int(selected) &&
    (set.available & flag(selected)) && bits(set.selection_value) == bits(set.values[u32(selected)]) &&
    (regression || bits(set.clip) == bits(1e-15)) && set.ap_outputs <= report.outputs && set.auc_outputs <= set.ap_outputs &&
    (!multi || (bool(set.available & flag(MetricName::macro_ap)) == bool(set.ap_outputs) &&
      bool(set.available & flag(MetricName::macro_auc)) == bool(set.auc_outputs)));
}
}

__device__ bool parse_decimal(Array<const char> text, double* output) {
  if (!output) return false;
  u64 mantissa = 0; int decimal = 0; bool negative = false;
  if (!decimal_parts(text, mantissa, decimal, negative)) return false;
  const u64 sign = u64(negative) << 63;
  if (!mantissa) { *output = __longlong_as_double(sign); return true; }
  Integer numerator, denominator; numerator.word[0] = u32(mantissa); numerator.word[1] = u32(mantissa >> 32); denominator.word[0] = 1;
  for (int i = 0; i < (decimal < 0 ? -decimal : decimal); ++i)
    if (!(decimal < 0 ? denominator.times_five() : numerator.times_five())) return false;
  int k = numerator.bits() - denominator.bits();
  Integer trial = k >= 0 ? denominator : numerator;
  if (!trial.shift(k >= 0 ? u32(k) : u32(-k))) return false;
  if (k >= 0 ? numerator.compare(trial) < 0 : trial.compare(denominator) < 0) --k;
  int exponent = k + decimal;
  if (exponent > 1023) { *output = __longlong_as_double(sign | 0x7ff0000000000000ULL); return true; }
  if (exponent < -1075) { *output = __longlong_as_double(sign); return true; }
  const int quantum = exponent >= -1022 ? exponent - 52 : -1074, shift = decimal - quantum;
  if (!(shift >= 0 ? numerator.shift(u32(shift)) : denominator.shift(u32(-shift)))) return false;
  u64 significand = 0;
  const int top = numerator.bits() - denominator.bits();
  if (top > 53) return false;
  for (int b = top; b >= 0; --b) {
    trial = denominator; if (!trial.shift(u32(b))) return false;
    if (numerator.compare(trial) >= 0) { numerator.subtract(trial); significand |= 1ULL << b; }
  }
  if (!numerator.shift(1)) return false;
  const int halfway = numerator.compare(denominator);
  significand += halfway > 0 || (halfway == 0 && (significand & 1));
  u64 result;
  if (exponent < -1022) result = significand;
  else {
    if (significand == 1ULL << 53) { significand >>= 1; ++exponent; }
    result = exponent > 1023 ? 0x7ff0000000000000ULL : (u64(exponent + 1023) << 52) | (significand & 0xfffffffffffffULL);
  }
  *output = __longlong_as_double(sign | result);
  return true;
}
__device__ cudaError_t decode_metric_reference(Array<const std::byte> bytes, const MetricReport* computed,
    MetricReport* reference, MetricReport* aligned, Status* status) {
  if (!status) return cudaErrorInvalidValue;
  if (!contains(bytes, bytes.size) || !bytes.size || bytes.size > (1u << 20) ||
      !contains(Array<const MetricReport>{computed, 1}, 1) || !contains(Array<MetricReport>{reference, 1}, 1) ||
      !contains(Array<MetricReport>{aligned, 1}, 1)) { fail(status, shape); return finish(status); }
  const auto report = *computed;
  if (report.profile != MetricProfile::real_data || u32(report.task) > 3 || !report.outputs ||
      (report.task == MetricTask::binary && report.outputs != 1) || (report.task == MetricTask::multiclass && report.outputs < 2) ||
      !report.count || report.count > 21 || !contains(report.metrics, report.count)) { fail(status, shape); return finish(status); }
  Set values, baseline; Json json{bytes};
  const u32 mask = required(report.task, report.outputs);
  if (!decode(json, values, baseline) || !schema(values, report, mask) || !schema(baseline, report, mask)) {
    fail(status, input); return finish(status);
  }
  const u32 count = __popc(mask);
  if (!contains(reference->metrics, count) || !contains(aligned->metrics, count)) { fail(status, capacity); return finish(status); }
  u32 seen = 0;
  for (u32 i = 0; i < report.count; ++i) {
    const auto m = report.metrics.data[i];
    if (u32(m.name) > 20 || m.output != aggregate_output || m.available > 1 ||
        !unique(seen, flag(m.name)) || (m.available && !finite(m.value))) { fail(status, input); return finish(status); }
  }
  const u32 extra = report.task == MetricTask::multilabel && report.outputs > 1 ? flag(MetricName::micro_auc) : 0;
  if (seen != (mask | extra)) { fail(status, input); return finish(status); }
  reference->count = aligned->count = 0;
  reference->outputs = aligned->outputs = report.outputs;
  reference->task = aligned->task = report.task;
  reference->profile = aligned->profile = report.profile;
  reference->ap_outputs = values.ap_outputs; reference->auc_outputs = values.auc_outputs;
  aligned->ap_outputs = report.ap_outputs; aligned->auc_outputs = report.auc_outputs;
  for (u32 i = 0; i < report.count; ++i) {
    const auto m = report.metrics.data[i];
    if (!(mask & flag(m.name))) continue;
    reference->metrics.data[reference->count++] = {m.name, aggregate_output, values.values[u32(m.name)], u32(bool(values.available & flag(m.name)))};
    aligned->metrics.data[aligned->count++] = m;
  }
  return finish(status);
}
}
