#include "gh/metrics.cuh"
#include "gh/detail/metric_math.cuh"
#include <cmath>

namespace gh {
namespace {
using detail::finite_metric;
using detail::numpy_sum;
using detail::positive_sum;
using detail::scaled_mean;
constexpr u32 threads = 256;
constexpr double clip = 1e-15;
__device__ u32 grid(u64 n) { return u32(min(ceil_div(n, threads), u64(65535))); }
__device__ bool observed(Status* s) {
  const auto e = cudaGetLastError();
  if (e != cudaSuccess) fail(s, runtime);
  return e == cudaSuccess;
}
template<class T> __device__ bool pointer(T* p) { return p && reinterpret_cast<std::uintptr_t>(p) % alignof(T) == 0; }
__device__ bool independent(MetricTask t) { return t == MetricTask::binary || t == MetricTask::multilabel; }
__device__ double weight(MetricInput d, u32 row) { return d.weights.size ? d.weights.data[row] : 1.0; }
__device__ double square(double x) { return __dmul_rn(x, x); }
__device__ double f1(u64 tp, u64 fp, u64 fn) { return tp ? __ddiv_rn(double(2 * tp), double(2 * tp + fp + fn)) : 0; }

struct Pair { u32 positives{}, groups{}; };
struct Group { u32 end{}, positives{}; };
struct RankMemory { u32 *a, *b; Pair *prefix, *scratch; Group* groups; u32* counts; };
__device__ u64 scan_size(u32 length, u32 groups) {
  u64 size = 0;
  while (length > threads) { length = u32(ceil_div(length, threads)); size += u64(length) * groups; }
  return size;
}
__device__ RankMemory ranking_memory(Arena& arena, u32 length, u32 groups) {
  const u64 n = u64(length) * groups;
  return {arena.take<u32>(n), arena.take<u32>(n), arena.take<Pair>(n),
    arena.take<Pair>(max(scan_size(length, groups), scan_size(u32(n), 1))),
    arena.take<Group>(n), arena.take<u32>(groups)};
}
__device__ bool before(const double* p, u32 a, u32 b) {
  return p[a] > p[b] || (p[a] == p[b] && a < b);
}
__global__ void rank_indices(u32* out, u32 length, u32 groups, u32 outputs, bool pooled) {
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < u64(length) * groups; i += u64(gridDim.x) * threads)
    out[i] = pooled ? u32(i) : u32((i % length) * outputs + i / length);
}
// Each thread merges eight adjacent output positions after a merge-path search.
__global__ void merge_indices(const double* p, const u32* in, u32* out, u32 length,
                              u32 groups, u64 width, Status* s) {
  if (s->errors) return;
  const u64 runs = ceil_div(length, width * 2), chunks = ceil_div(width * 2, 8);
  const u64 jobs_per_group = runs * chunks;
  for (u64 job = u64(blockIdx.x) * threads + threadIdx.x; job < jobs_per_group * groups; job += u64(gridDim.x) * threads) {
    const u64 group = job / jobs_per_group, local = job % jobs_per_group;
    const u64 first = (local / chunks) * width * 2, diagonal = (local % chunks) * 8;
    const u32 a = u32(min(width, u64(length) - first));
    const u32 b = u32(min(width, u64(length) - first - a));
    if (diagonal >= u64(a) + b) continue;
    const u64 base = group * length + first;
    u32 lo = u32(diagonal > b ? diagonal - b : 0), hi = u32(min(diagonal, u64(a)));
    while (lo < hi) {
      const u32 middle = lo + (hi - lo) / 2, right = u32(diagonal) - middle;
      if (middle < a && right && before(p, in[base + middle], in[base + a + right - 1])) lo = middle + 1;
      else hi = middle;
    }
    u32 left = lo, right = u32(diagonal) - lo;
    for (u32 k = 0; k < 8 && diagonal + k < u64(a) + b; ++k) {
      const bool take_left = left < a && (right == b || before(p, in[base + left], in[base + a + right]));
      out[base + diagonal + k] = take_left ? in[base + left++] : in[base + a + right++];
    }
  }
}
__device__ Pair plus(Pair a, Pair b) { return {a.positives + b.positives, a.groups + b.groups}; }
__device__ Pair inclusive(Pair x) {
  for (u32 step = 1; step < 32; step *= 2) {
    const Pair previous{__shfl_up_sync(0xffffffffu, x.positives, step), __shfl_up_sync(0xffffffffu, x.groups, step)};
    if (threadIdx.x % 32 >= step) x = plus(previous, x);
  }
  return x;
}
__global__ void rank_flags(const double* p, const double* y, const u32* order, Pair* flags,
                           u32 length, u32 groups, Status* s) {
  if (s->errors) return;
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < u64(length) * groups; i += u64(gridDim.x) * threads)
    flags[i] = {u32(y[order[i]] == 1), u32(i % length + 1 == length || p[order[i]] != p[order[i + 1]])};
}
__global__ void scan_pairs(Pair* data, u32 length, u32 blocks, Pair* sums, Status* s) {
  if (s->errors) return;
  __shared__ Pair totals[8];
  const u32 group = blockIdx.x / blocks, block = blockIdx.x % blocks, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const u64 at = u64(block) * threads + threadIdx.x, base = u64(group) * length;
  const Pair original = at < length ? data[base + at] : Pair{};
  Pair value = inclusive(original);
  if (lane == 31) totals[warp] = value;
  __syncthreads();
  if (!warp) {
    const Pair all = inclusive(lane < 8 ? totals[lane] : Pair{});
    if (lane < 8) totals[lane] = all;
  }
  __syncthreads();
  if (warp) value = plus(totals[warp - 1], value);
  if (at < length) data[base + at] = {value.positives - original.positives, value.groups - original.groups};
  if (!threadIdx.x && sums) sums[u64(group) * blocks + block] = totals[7];
}
__global__ void add_pair_offsets(Pair* values, const Pair* totals, u32 length, u32 blocks, u64 n, Status* s) {
  if (s->errors) return;
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < n; i += u64(gridDim.x) * threads)
    values[i] = plus(values[i], totals[(i / length) * blocks + (i % length) / threads]);
}
__global__ void compact_groups(const double* p, const double* y, const u32* order, const Pair* prefix,
                               Group* out, u32* counts, u32 length, u32 groups, Status* s) {
  if (s->errors) return;
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < u64(length) * groups; i += u64(gridDim.x) * threads) {
    const u32 local = u32(i % length);
    if (local + 1 == length || p[order[i]] != p[order[i + 1]]) {
      out[(i / length) * length + prefix[i].groups] = {local + 1, prefix[i].positives + u32(y[order[i]] == 1)};
      if (local + 1 == length) counts[i / length] = prefix[i].groups + 1;
    }
  }
}
__device__ u32* rank(const double* p, const double* y, u32 length, u32 groups,
                    u32 outputs, bool pooled, RankMemory w, Status* s) {
  const u64 n = u64(length) * groups;
  rank_indices<<<grid(n), threads>>>(w.a, length, groups, outputs, pooled);
  if (!observed(s)) return nullptr;
  u32* source = w.a; u32* destination = w.b;
  for (u64 width = 1; width < length; width *= 2) {
    const u64 jobs = ceil_div(length, width * 2) * ceil_div(width * 2, 8) * groups;
    merge_indices<<<grid(jobs), threads>>>(p, source, destination, length, groups, width, s);
    if (!observed(s)) return nullptr;
    auto* temporary = source; source = destination; destination = temporary;
  }
  rank_flags<<<grid(n), threads>>>(p, y, source, w.prefix, length, groups, s);
  if (!observed(s)) return nullptr;
  Pair* levels[5]{w.prefix}; u32 lengths[5]{length}, depth = 0;
  auto* scratch = w.scratch;
  while (true) {
    const u32 blocks = u32(ceil_div(lengths[depth], threads));
    auto* next = blocks > 1 ? scratch : nullptr;
    scan_pairs<<<groups * blocks, threads>>>(levels[depth], lengths[depth], blocks, next, s);
    if (!observed(s)) return nullptr;
    if (blocks == 1) break;
    levels[++depth] = next; lengths[depth] = blocks; scratch += u64(blocks) * groups;
  }
  while (depth) {
    --depth;
    add_pair_offsets<<<grid(u64(lengths[depth]) * groups), threads>>>(levels[depth], levels[depth + 1],
      lengths[depth], lengths[depth + 1], u64(lengths[depth]) * groups, s);
    if (!observed(s)) return nullptr;
  }
  compact_groups<<<grid(n), threads>>>(p, y, source, w.prefix, w.groups, w.counts, length, groups, s);
  return observed(s) ? source : nullptr;
}

struct Row { double norm{}, brier{}; u32 chosen{}, exact{}, top[3]{}; };
struct Column {
  double rmse{}, mae{}, r2{}, loss{}, accuracy{}, brier{}, auc{}, ap{};
  u64 tp{}, fp{}, fn{}, positives{};
  u32 auc_available{};
};
__global__ void validate_inputs(MetricInput d, Status* s) {
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < u64(d.rows) * d.outputs; i += u64(gridDim.x) * threads) {
    const double p = d.predictions.data[i];
    if (!finite_metric(p) || (d.task != MetricTask::regression && (p < 0 || p > 1))) fail(s, input);
    if (d.task != MetricTask::multiclass || i < d.rows) {
      const double y = d.targets.data[i];
      if (!finite_metric(y) || (independent(d.task) && y != 0 && y != 1) ||
          (d.task == MetricTask::multiclass && (y < 0 || y >= d.outputs || floor(y) != y))) fail(s, input);
    }
    if (i < d.rows && d.weights.size && (!finite_metric(d.weights.data[i]) || d.weights.data[i] < 0)) fail(s, input);
  }
}
__global__ void weight_total(MetricInput d, double* total, Status* s) {
  if (s->errors) return;
  *total = d.weights.size ? positive_sum(d.rows, [=](u32 i) { return weight(d, i); }, s) : double(d.rows);
  if (!(*total > 0)) fail(s, input);
}
__global__ void row_statistics(MetricInput d, Row* out, Status* s) {
  if (s->errors) return;
  for (u64 row = u64(blockIdx.x) * threads + threadIdx.x; row < d.rows; row += u64(gridDim.x) * threads) {
    Row result{}; result.norm = 1; result.exact = 1;
    const u64 begin = row * d.outputs;
    u32 top[5]{}; u32 used = 0;
    for (u32 k = 0; k < d.outputs; ++k) {
      if (d.task == MetricTask::multiclass && d.predictions.data[begin + k] > d.predictions.data[begin + result.chosen]) result.chosen = k;
      if (independent(d.task)) {
        result.exact &= (d.predictions.data[begin + k] >= .5) == (d.targets.data[begin + k] == 1);
        if (d.profile == MetricProfile::real_data && d.task == MetricTask::multilabel) {
          u32 at = 0;
          while (at < used && d.predictions.data[begin + top[at]] >= d.predictions.data[begin + k]) ++at;
          if (at < 5) {
            for (u32 j = min(used, 4u); j > at; --j) top[j] = top[j - 1];
            top[at] = k; used = min(used + 1, 5u);
          }
        }
      }
    }
    if (d.task == MetricTask::multiclass) {
      const auto probability = [=](u32 k) { return d.predictions.data[begin + k]; };
      result.norm = d.profile == MetricProfile::synthetic ? positive_sum(d.outputs, probability, s) : numpy_sum(d.outputs, probability);
      if (!(result.norm > 0) || fabs(result.norm - 1) > d.probability_tolerance) fail(s, input);
      if (d.profile == MetricProfile::real_data) result.norm = 1;
      result.exact = result.chosen == u32(d.targets.data[row]);
      if (d.profile == MetricProfile::real_data) result.brier = numpy_sum(d.outputs, [=](u32 k) {
          return square(__dsub_rn(d.predictions.data[begin + k], double(k == u32(d.targets.data[row]))));
        });
    }
    if (used) {
      u32 hits = 0;
      for (u32 k = 0; k < used; ++k) {
        hits += d.targets.data[begin + top[k]] == 1;
        if (k == 0) result.top[0] = hits;
        if (k == 2) result.top[1] = hits;
        if (k == 4) result.top[2] = hits;
      }
    }
    out[row] = result;
  }
}
__device__ double correct_probability(MetricInput d, const Row* rows, u32 row, u32 output) {
  if (d.task == MetricTask::multiclass)
    return __ddiv_rn(d.predictions.data[u64(row) * d.outputs + u32(d.targets.data[row])], rows[row].norm);
  const u64 i = u64(row) * d.outputs + output;
  return d.targets.data[i] == 1 ? d.predictions.data[i] : __dsub_rn(1.0, d.predictions.data[i]);
}
__device__ double synthetic_loss(double p) { return -log(fmin(1 - clip, fmax(clip, p))); }
__global__ void column_statistics(MetricInput d, const Row* rows, const double* total, Column* out, Status* s) {
  if (s->errors) return;
  for (u64 output = u64(blockIdx.x) * threads + threadIdx.x; output < d.outputs; output += u64(gridDim.x) * threads) {
    Column result{};
    const auto w = [=](u32 row) { return weight(d, row); };
    if (d.task == MetricTask::regression) {
      const auto error = [=](u32 row) { return fabs(__dsub_rn(d.targets.data[u64(row) * d.outputs + output], d.predictions.data[u64(row) * d.outputs + output])); };
      if (d.profile == MetricProfile::synthetic) {
        result.rmse = scaled_mean<2>(d.rows, error, w, *total, s);
        result.mae = scaled_mean<1>(d.rows, error, w, *total, s);
      } else {
        const double mean = __ddiv_rn(detail::column_sum(d.rows, d.outputs, [=](u32 row) {
          return d.targets.data[u64(row) * d.outputs + output]; }), d.rows);
        const double numerator = detail::column_sum(d.rows, d.outputs, [=](u32 row) { return square(error(row)); });
        const double denominator = detail::column_sum(d.rows, d.outputs, [=](u32 row) {
          return square(__dsub_rn(d.targets.data[u64(row) * d.outputs + output], mean)); });
        result.r2 = !numerator ? 1 : !denominator ? 0 : __dsub_rn(1.0, __ddiv_rn(numerator, denominator));
      }
    } else {
      if (d.profile == MetricProfile::real_data) for (u32 row = 0; row < d.rows; ++row) {
        const bool positive = d.task == MetricTask::multiclass ? d.targets.data[row] == output : d.targets.data[u64(row) * d.outputs + output] == 1;
        const bool predicted = d.task == MetricTask::multiclass ? rows[row].chosen == output : d.predictions.data[u64(row) * d.outputs + output] >= .5;
        result.positives += positive; result.tp += positive && predicted;
        result.fp += !positive && predicted; result.fn += positive && !predicted;
      }
      if (d.profile == MetricProfile::synthetic && independent(d.task)) {
        result.loss = scaled_mean<1>(d.rows, [=](u32 row) { return synthetic_loss(correct_probability(d, rows, row, u32(output))); }, w, *total, s);
        result.brier = scaled_mean<1>(d.rows, [=](u32 row) {
          return square(__dsub_rn(d.predictions.data[u64(row) * d.outputs + output], d.targets.data[u64(row) * d.outputs + output])); }, w, *total, s);
        result.accuracy = __ddiv_rn(positive_sum(d.rows, [=](u32 row) {
          const u64 i = u64(row) * d.outputs + output;
          return (d.predictions.data[i] >= .5) == (d.targets.data[i] == 1) ? weight(d, row) : 0.0;
        }, s), *total);
      }
    }
    out[output] = result;
  }
}

__device__ double weighted_auc(MetricInput d, const u32* order, const Group* groups, u32 count, Status* s, bool& available) {
  const double positive = positive_sum(d.rows, [=](u32 row) {
    const u32 i = order[row]; return d.targets.data[i] == 1 ? weight(d, i / d.outputs) : 0.0;
  }, s);
  const double negative = positive_sum(d.rows, [=](u32 row) {
    const u32 i = order[row]; return d.targets.data[i] == 0 ? weight(d, i / d.outputs) : 0.0;
  }, s);
  available = positive > 0 && negative > 0;
  if (!available) return 0;
  detail::PositiveSum concordance;
  double before_negative = 0, correction = 0;
  for (u32 reverse = count; reverse; --reverse) {
    const u32 g = reverse - 1, begin = g ? groups[g - 1].end : 0, end = groups[g].end;
    const double p = positive_sum(end - begin, [=](u32 k) {
      const u32 i = order[begin + k];
      return d.targets.data[i] == 1 ? __ddiv_rn(weight(d, i / d.outputs), positive) : 0.0;
    }, s);
    const double n = positive_sum(end - begin, [=](u32 k) {
      const u32 i = order[begin + k];
      return d.targets.data[i] == 0 ? __ddiv_rn(weight(d, i / d.outputs), negative) : 0.0;
    }, s);
    concordance.add(__dmul_rn(p, __dadd_rn(before_negative, __dmul_rn(.5, n))), s);
    const double increment = __dsub_rn(n, correction), updated = __dadd_rn(before_negative, increment);
    correction = __dsub_rn(__dsub_rn(updated, before_negative), increment); before_negative = updated;
  }
  return fmin(1.0, fmax(0.0, concordance.value(s)));
}
__global__ void ranking_statistics(MetricInput d, const u32* order, RankMemory w,
                                   Column* out, bool pooled, Status* s) {
  if (s->errors) return;
  const u32 columns = pooled ? 1 : d.outputs, length = pooled ? d.rows * d.outputs : d.rows;
  for (u64 output = u64(blockIdx.x) * threads + threadIdx.x; output < columns; output += u64(gridDim.x) * threads) {
    const auto* groups = w.groups + output * length;
    const u32 count = w.counts[output], positives = groups[count - 1].positives, negatives = length - positives;
    const u32 slot = pooled ? d.outputs : u32(output);
    out[slot].positives = positives; out[slot].auc_available = positives && negatives;
    if (d.profile == MetricProfile::synthetic) {
      bool available;
      out[slot].auc = weighted_auc(d, order + output * length, groups, count, s, available);
      out[slot].auc_available = available;
      continue;
    }
    out[slot].ap = positives ? fmax(0.0, -numpy_sum(count, [=](u32 k) {
      const u32 g = count - 1 - k;
      const double current = __ddiv_rn(double(groups[g].positives), positives);
      const double next = g ? __ddiv_rn(double(groups[g - 1].positives), positives) : 0;
      return __dmul_rn(__dsub_rn(next, current), __ddiv_rn(double(groups[g].positives), groups[g].end));
    })) : 0;
    if (positives && negatives) {
      // Prefix storage has no readers after compaction; reuse it for ROC areas.
      auto* areas = reinterpret_cast<double*>(w.prefix) + output * length;
      double previous_fpr = 0, previous_tpr = 0;
      u32 used = 0;
      for (u32 g = 0; g < count; ++g) {
        const auto point = groups[g];
        bool keep = !g || g + 1 == count;
        if (!keep) {
          const auto a = groups[g - 1], b = groups[g + 1];
          keep = std::int64_t(b.positives) - 2 * std::int64_t(point.positives) + a.positives ||
            (std::int64_t(b.end) - b.positives) - 2 * (std::int64_t(point.end) - point.positives) + (std::int64_t(a.end) - a.positives);
        }
        if (!keep) continue;
        const double fpr = __ddiv_rn(double(point.end - point.positives), negatives), tpr = __ddiv_rn(double(point.positives), positives);
        areas[used++] = __ddiv_rn(__dmul_rn(__dsub_rn(fpr, previous_fpr), __dadd_rn(tpr, previous_tpr)), 2.0);
        previous_fpr = fpr; previous_tpr = tpr;
      }
      out[slot].auc = numpy_sum(used, [=](u32 i) { return areas[i]; });
    }
  }
}

struct Writer {
  MetricReport* report; Status* status;
  __device__ void add(MetricName name, double value, bool available = true, u32 output = aggregate_output) {
    if (available && !finite_metric(value)) fail(status, numeric);
    report->metrics.data[report->count++] = {name, output, value, u32(available)};
  }
};
__global__ void aggregate_statistics(MetricInput d, const Row* rows, const Column* columns,
    const double* total, double* compact, MetricReport* report, Status* s) {
  if (s->errors) return;
  Writer write{report, s};
  const u32 n = d.rows * d.outputs;
  const auto unit = [](u32) { return 1.0; };
  if (d.profile == MetricProfile::synthetic) {
    if (d.task == MetricTask::regression) {
      write.add(MetricName::rmse, scaled_mean<2>(d.outputs, [=](u32 k) { return columns[k].rmse; }, unit, d.outputs, s));
      write.add(MetricName::mae, scaled_mean<1>(d.outputs, [=](u32 k) { return columns[k].mae; }, unit, d.outputs, s));
      for (u32 k = 0; k < d.outputs; ++k) {
        write.add(MetricName::rmse, columns[k].rmse, true, k); write.add(MetricName::mae, columns[k].mae, true, k);
      }
    } else if (d.task == MetricTask::multiclass) {
      write.add(MetricName::logloss, scaled_mean<1>(d.rows, [=](u32 row) { return synthetic_loss(correct_probability(d, rows, row, 0)); },
        [=](u32 row) { return weight(d, row); }, *total, s));
      write.add(MetricName::accuracy, __ddiv_rn(positive_sum(d.rows, [=](u32 row) { return rows[row].exact ? weight(d, row) : 0.0; }, s), *total));
    } else {
      bool auc_available = true;
      for (u32 k = 0; k < d.outputs; ++k) auc_available &= columns[k].auc_available != 0;
      write.add(MetricName::logloss, scaled_mean<1>(d.outputs, [=](u32 k) { return columns[k].loss; }, unit, d.outputs, s));
      write.add(MetricName::accuracy, scaled_mean<1>(d.outputs, [=](u32 k) { return columns[k].accuracy; }, unit, d.outputs, s));
      write.add(MetricName::brier, scaled_mean<1>(d.outputs, [=](u32 k) { return columns[k].brier; }, unit, d.outputs, s));
      write.add(MetricName::auc, auc_available ? scaled_mean<1>(d.outputs, [=](u32 k) { return columns[k].auc; }, unit, d.outputs, s) : 0, auc_available);
      if (d.task == MetricTask::multilabel) for (u32 k = 0; k < d.outputs; ++k) {
        write.add(MetricName::logloss, columns[k].loss, true, k); write.add(MetricName::accuracy, columns[k].accuracy, true, k);
        write.add(MetricName::brier, columns[k].brier, true, k); write.add(MetricName::auc, columns[k].auc, columns[k].auc_available, k);
      }
    }
    return;
  }
  if (d.task == MetricTask::regression) {
    const double mse = __ddiv_rn(numpy_sum(n, [=](u32 i) { return square(__dsub_rn(d.predictions.data[i], d.targets.data[i])); }), n);
    write.add(MetricName::mse, mse); write.add(MetricName::rmse, sqrt(mse));
    write.add(MetricName::mae, __ddiv_rn(numpy_sum(n, [=](u32 i) { return fabs(__dsub_rn(d.predictions.data[i], d.targets.data[i])); }), n));
    write.add(MetricName::r2, __ddiv_rn(numpy_sum(d.outputs, [=](u32 k) { return columns[k].r2; }), d.outputs), d.rows > 1);
    return;
  }
  if (d.task == MetricTask::multiclass) {
    write.add(MetricName::logloss, -__ddiv_rn(numpy_sum(d.rows, [=](u32 row) {
      return log(fmin(1.0, fmax(clip, d.predictions.data[u64(row) * d.outputs + u32(d.targets.data[row])]))); }), d.rows));
    u64 correct = 0; for (u32 row = 0; row < d.rows; ++row) correct += rows[row].exact;
    write.add(MetricName::accuracy, __ddiv_rn(double(correct), d.rows));
    write.add(MetricName::macro_f1, __ddiv_rn(numpy_sum(d.outputs, [=](u32 k) { return f1(columns[k].tp, columns[k].fp, columns[k].fn); }), d.outputs));
    write.add(MetricName::brier, __ddiv_rn(numpy_sum(d.rows, [=](u32 row) { return rows[row].brier; }), d.rows));
    return;
  }
  write.add(MetricName::logloss, -__ddiv_rn(numpy_sum(n, [=](u32 i) {
    const double p = fmin(1 - clip, fmax(clip, d.predictions.data[i])), y = d.targets.data[i];
    return __dadd_rn(__dmul_rn(y, log(p)), __dmul_rn(__dsub_rn(1.0, y), log1p(-p)));
  }), n));
  write.add(MetricName::brier, __ddiv_rn(numpy_sum(n, [=](u32 i) { return square(__dsub_rn(d.predictions.data[i], d.targets.data[i])); }), n));
  u64 tp = 0, fp = 0, fn = 0, exact = 0;
  for (u32 k = 0; k < d.outputs; ++k) { tp += columns[k].tp; fp += columns[k].fp; fn += columns[k].fn; }
  for (u32 row = 0; row < d.rows; ++row) exact += rows[row].exact;
  write.add(MetricName::hamming_loss, __ddiv_rn(double(fp + fn), n));
  write.add(MetricName::exact_match, __ddiv_rn(double(exact), d.rows));
  if (d.outputs == 1) {
    write.add(MetricName::accuracy, __ddiv_rn(double(n - fp - fn), n)); write.add(MetricName::f1, f1(tp, fp, fn));
    write.add(MetricName::auc, columns[0].auc, columns[0].auc_available); write.add(MetricName::average_precision, columns[0].ap);
  } else {
    write.add(MetricName::micro_f1, f1(tp, fp, fn));
    write.add(MetricName::macro_f1, __ddiv_rn(numpy_sum(d.outputs, [=](u32 k) { return f1(columns[k].tp, columns[k].fp, columns[k].fn); }), d.outputs));
    write.add(MetricName::micro_ap, columns[d.outputs].ap); write.add(MetricName::micro_auc, columns[d.outputs].auc, columns[d.outputs].auc_available);
    u32 present = 0, varying = 0;
    for (u32 k = 0; k < d.outputs; ++k) if (columns[k].positives) compact[present++] = columns[k].ap;
    write.add(MetricName::macro_ap, present ? __ddiv_rn(numpy_sum(present, [=](u32 k) { return compact[k]; }), present) : 0, present);
    for (u32 k = 0; k < d.outputs; ++k) if (columns[k].auc_available) compact[varying++] = columns[k].auc;
    write.add(MetricName::macro_auc, varying ? __ddiv_rn(numpy_sum(varying, [=](u32 k) { return compact[k]; }), varying) : 0, varying);
    report->ap_outputs = present; report->auc_outputs = varying;
    const MetricName names[3]{MetricName::precision_at_1, MetricName::precision_at_3, MetricName::precision_at_5};
    for (u32 j = 0; j < 3; ++j) {
      const u32 k = 2 * j + 1;
      if (k > d.outputs) continue;
      u64 hits = 0; for (u32 row = 0; row < d.rows; ++row) hits += rows[row].top[j];
      write.add(names[j], __ddiv_rn(double(hits), double(u64(d.rows) * k)));
    }
  }
}

__device__ u64 metric_count(MetricInput d) {
  if (d.profile == MetricProfile::synthetic) {
    if (d.task == MetricTask::regression) return 2 + 2ULL * d.outputs;
    if (d.task == MetricTask::multiclass) return 2;
    return d.task == MetricTask::binary ? 4 : 4 + 4ULL * d.outputs;
  }
  if (!independent(d.task)) return 4;
  return d.outputs == 1 ? 8 : 11 + u64(d.outputs >= 3) + u64(d.outputs >= 5);
}
__device__ bool valid_shape(MetricInput d) {
  const u64 n = u64(d.rows) * d.outputs;
  return d.rows && d.outputs && n <= UINT32_MAX && u32(d.task) <= 3 && u32(d.profile) <= 1 &&
    (!independent(d.task) || u64(d.outputs) * ceil_div(d.rows, threads) <= INT32_MAX) &&
    (d.task != MetricTask::binary || d.outputs == 1) && (d.task != MetricTask::multiclass || d.outputs >= 2) &&
    finite_metric(d.probability_tolerance) && d.probability_tolerance >= 0 && d.probability_tolerance < 1 &&
    contains(d.predictions, n) && contains(d.targets, d.task == MetricTask::multiclass ? d.rows : n) &&
    (!d.weights.size || (d.profile == MetricProfile::synthetic && contains(d.weights, d.rows)));
}

__device__ OperatingPoint point(u64 positives, u64 negatives, u64 tp, u64 fp, double threshold) {
  return {threshold, __ddiv_rn(double(tp), double(positives)), __ddiv_rn(double(fp), double(negatives)),
    tp + fp ? __ddiv_rn(double(tp), double(tp + fp)) : 0, f1(tp, fp, positives - tp),
    tp, fp, positives - tp, negatives - fp, positives, negatives, u32(20 * fp <= negatives)};
}
__global__ void signal_result(SignalInput d, ThresholdMode mode, double threshold, const u32* order,
                              RankMemory w, SignalReport* report, Status* s) {
  if (s->errors) return;
  const u32 count = w.counts[0], positives = w.groups[count - 1].positives, negatives = d.size - positives;
  if (!positives || !negatives) { fail(s, input); return; }
  u32 fixed_tp = 0, fixed_fp = 0, selected_tp = 0, selected_fp = 0;
  double selected = mode == ThresholdMode::validation ? nextafter(d.predictions.data[order[0]], double(INFINITY)) : threshold;
  for (u32 g = 0; g < count; ++g) {
    const auto v = w.groups[g]; const u32 fp = v.end - v.positives;
    const double score = d.predictions.data[order[v.end - 1]];
    if (score >= .5) { fixed_tp = v.positives; fixed_fp = fp; }
    if (mode == ThresholdMode::validation) {
      if (20ULL * fp <= negatives && v.positives > selected_tp) { selected_tp = v.positives; selected_fp = fp; selected = score; }
    } else if (score >= threshold) { selected_tp = v.positives; selected_fp = fp; }
  }
  *report = {point(positives, negatives, fixed_tp, fixed_fp, .5), point(positives, negatives, selected_tp, selected_fp, selected), mode};
}
__device__ bool minimize(MetricName name) {
  return name == MetricName::mse || name == MetricName::rmse || name == MetricName::mae ||
    name == MetricName::logloss || name == MetricName::brier || name == MetricName::hamming_loss;
}
__global__ void metric_gate(const MetricReport* reference, const MetricReport* candidate,
    MetricVerdict* verdicts, MetricGate* result, Status* s) {
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < reference->count; i += u64(gridDim.x) * threads) {
    const auto a = reference->metrics.data[i], b = candidate->metrics.data[i];
    if (a.name != b.name || a.output != b.output || u32(a.name) > u32(MetricName::precision_at_5) ||
        a.available > 1 || b.available > 1 || a.available != b.available ||
        (a.available && (!finite_metric(a.value) || !finite_metric(b.value)))) {
      verdicts[i] = MetricVerdict::invalid; atomicAdd(&result->invalid, 1); fail(s, input);
    } else if (!a.available) { verdicts[i] = MetricVerdict::not_applicable; atomicAdd(&result->unavailable, 1); }
    else {
      const bool worse = minimize(a.name) ? b.value > a.value : b.value < a.value;
      verdicts[i] = worse ? MetricVerdict::regression : MetricVerdict::pass;
      atomicAdd(&result->checked, 1); if (worse) atomicAdd(&result->regressions, 1);
    }
  }
}
}

__device__ cudaError_t evaluate_metrics(MetricInput d, MetricReport* report, Workspace workspace, Status* s) {
  if (!s) return cudaErrorInvalidValue;
  if (!pointer(report) || !valid_shape(d)) { fail(s, shape); return finish(s); }
  const u64 required = metric_count(d);
  if (required > UINT32_MAX || !contains(report->metrics, required)) { fail(s, capacity); return finish(s); }
  Arena arena{workspace};
  RankMemory rankings{};
  if (independent(d.task)) rankings = ranking_memory(arena, d.rows, d.outputs);
  auto* rows = arena.take<Row>(d.task == MetricTask::regression ? 0 : d.rows);
  auto* columns = arena.take<Column>(u64(d.outputs) + 1);
  auto* total = arena.take<double>(1);
  auto* compact = arena.take<double>(d.outputs);
  if (!arena.fits(s)) return finish(s);
  report->count = report->ap_outputs = report->auc_outputs = 0;
  report->outputs = d.outputs; report->task = d.task; report->profile = d.profile;
  validate_inputs<<<grid(u64(d.rows) * d.outputs), threads>>>(d, s); observed(s);
  weight_total<<<1, 1>>>(d, total, s); observed(s);
  if (d.task != MetricTask::regression) { row_statistics<<<grid(d.rows), threads>>>(d, rows, s); observed(s); }
  if (d.task != MetricTask::multiclass || d.profile == MetricProfile::real_data) {
    column_statistics<<<grid(d.outputs), threads>>>(d, rows, total, columns, s); observed(s);
  }
  if (independent(d.task)) {
    if (const auto* sorted = rank(d.predictions.data, d.targets.data, d.rows, d.outputs, d.outputs, false, rankings, s)) {
      ranking_statistics<<<grid(d.outputs), threads>>>(d, sorted, rankings, columns, false, s); observed(s);
    }
    if (d.profile == MetricProfile::real_data && d.outputs > 1) {
      if (const auto* sorted = rank(d.predictions.data, d.targets.data, d.rows * d.outputs, 1, d.outputs, true, rankings, s)) {
        ranking_statistics<<<1, 1>>>(d, sorted, rankings, columns, true, s); observed(s);
      }
    }
  }
  aggregate_statistics<<<1, 1>>>(d, rows, columns, total, compact, report, s); observed(s);
  return finish(s);
}
__device__ cudaError_t compare_metrics(const MetricReport* reference, const MetricReport* candidate,
    Array<MetricVerdict> verdicts, MetricGate* result, Status* s) {
  if (!s) return cudaErrorInvalidValue;
  if (!pointer(reference) || !pointer(candidate) || !pointer(result)) { fail(s, shape); return finish(s); }
  if (!reference->count || reference->count != candidate->count || reference->outputs != candidate->outputs ||
      reference->task != candidate->task || reference->profile != candidate->profile ||
      reference->ap_outputs != candidate->ap_outputs || reference->auc_outputs != candidate->auc_outputs ||
      !contains(reference->metrics, reference->count) || !contains(candidate->metrics, candidate->count) || !contains(verdicts, reference->count)) {
    fail(s, shape); return finish(s);
  }
  *result = {};
  metric_gate<<<grid(reference->count), threads>>>(reference, candidate, verdicts.data, result, s); observed(s);
  return finish(s);
}
__device__ cudaError_t signal_metrics(SignalInput d, ThresholdMode mode, double threshold,
    SignalReport* report, Workspace workspace, Status* s) {
  if (!s) return cudaErrorInvalidValue;
  const MetricInput input{d.labels, d.predictions, {}, d.size, 1, MetricTask::binary, MetricProfile::real_data};
  if (!pointer(report) || !valid_shape(input) || u32(mode) > 1 || (mode == ThresholdMode::frozen && !finite_metric(threshold))) {
    fail(s, shape); return finish(s);
  }
  Arena arena{workspace}; const auto w = ranking_memory(arena, d.size, 1);
  if (!arena.fits(s)) return finish(s);
  validate_inputs<<<grid(d.size), threads>>>(input, s); observed(s);
  if (const auto* sorted = rank(d.predictions.data, d.labels.data, d.size, 1, 1, true, w, s)) {
    signal_result<<<1, 1>>>(d, mode, threshold, sorted, w, report, s); observed(s);
  }
  return finish(s);
}
}
