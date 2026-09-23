#include "gh/csv.cuh"
#include <cuda/std/bit>

namespace gh {
namespace {
constexpr u32 threads = 256;
struct Text { char data[25]{}; u32 size{}; };
__device__ u32 grid(u64 n) { return u32(min(u64{65535}, ceil_div(n, threads))); }
__device__ bool launched(Status* s) {
  const auto error = cudaGetLastError();
  if (error != cudaSuccess) fail(s, runtime);
  return error == cudaSuccess;
}
__device__ void append(Text& out, char c) { out.data[out.size++] = c; }
__device__ void integer(Text& out, u32 value, u32 minimum = 1) {
  char digits[10]; u32 n = 0;
  do { digits[n++] = char('0' + value % 10); value /= 10; } while (value || n < minimum);
  while (n) append(out, digits[--n]);
}
__device__ u64 header_size(u32 outputs, CsvKind kind) {
  const bool numbered = outputs > 1 || kind == CsvKind::multiclass;
  u64 bytes = 7 + (kind == CsvKind::targets ? 7 : 0) +
    u64(outputs) * (kind == CsvKind::targets ? 7 : kind == CsvKind::multiclass ? 2 : 11);
  if (numbered) {
    bytes += outputs; // At least one digit per column.
    for (u64 power = 10; power < outputs; power *= 10) bytes += outputs - power;
    if (kind != CsvKind::multiclass) bytes += outputs;
  }
  return bytes;
}
__device__ Text decimal(double value, Status* status) {
  Text out;
  const u64 bits = cuda::std::bit_cast<u64>(value);
  const u32 exponent = u32((bits >> 52) & 2047);
  if (exponent == 2047) { fail(status, input); return out; }
  if (bits >> 63) append(out, '-');
  const u64 mantissa = (bits & 0xfffffffffffffULL) | (exponent ? 1ULL << 52 : 0);
  if (!mantissa) { append(out, '0'); return out; }
  u32 limbs[86]{}; u32 used = 0;
  for (u64 x = mantissa; x; x /= 1000000000) limbs[used++] = u32(x % 1000000000);
  const int binary_exponent = exponent ? int(exponent) - 1075 : -1074;
  u32 remaining = u32(abs(binary_exponent));
  while (remaining) {
    const u32 step = min(remaining, binary_exponent < 0 ? 12u : 29u);
    u32 factor = 1;
    for (u32 i = 0; i < step; ++i) factor *= binary_exponent < 0 ? 5u : 2u;
    u64 carry = 0;
    for (u32 i = 0; i < used; ++i) {
      const u64 product = u64(limbs[i]) * factor + carry;
      limbs[i] = u32(product % 1000000000); carry = product / 1000000000;
    }
    if (carry) limbs[used++] = u32(carry);
    remaining -= step;
  }
  u32 leading = 0;
  for (u32 x = limbs[used - 1]; x; x /= 10) ++leading;
  const u32 digits = (used - 1) * 9 + leading;
  auto digit = [&](u32 index) {
    const u32 position = digits - 1 - index, limb = position / 9;
    u32 divisor = 1;
    for (u32 i = 0; i < position % 9; ++i) divisor *= 10;
    return limbs[limb] / divisor % 10;
  };
  char significant[17]; u32 count = min(digits, 17u);
  for (u32 i = 0; i < count; ++i) significant[i] = char('0' + digit(i));
  int power = int(digits) - 1 - (binary_exponent < 0 ? -binary_exponent : 0);
  if (digits > 17) {
    const u32 guard = digit(17); bool sticky = false;
    for (u32 i = 18; i < digits && !sticky; ++i) sticky = digit(i) != 0;
    if (guard > 5 || (guard == 5 && (sticky || ((significant[16] - '0') & 1)))) {
      int i = 16;
      while (i >= 0 && significant[i] == '9') significant[i--] = '0';
      if (i >= 0) ++significant[i];
      else { significant[0] = '1'; ++power; }
    }
  }
  while (count > 1 && significant[count - 1] == '0') --count;
  if (power < -4 || power >= 17) {
    append(out, significant[0]);
    if (count > 1) append(out, '.');
    for (u32 i = 1; i < count; ++i) append(out, significant[i]);
    append(out, 'e'); append(out, power < 0 ? '-' : '+'); integer(out, u32(abs(power)), 2);
  } else if (power < 0) {
    append(out, '0'); append(out, '.');
    for (int i = -1; i > power; --i) append(out, '0');
    for (u32 i = 0; i < count; ++i) append(out, significant[i]);
  } else {
    for (u32 i = 0; i <= u32(power); ++i) append(out, i < count ? significant[i] : '0');
    if (count > u32(power) + 1) append(out, '.');
    for (u32 i = u32(power) + 1; i < count; ++i) append(out, significant[i]);
  }
  return out;
}
__global__ void format(Array<const double> values, Array<const double> weights,
    u32 rows, u32 outputs, CsvKind kind, Text* text, u64* lengths, Status* status) {
  const u64 columns = u64(outputs) + 1 + (kind == CsvKind::targets);
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < u64(rows) * columns; i += u64(gridDim.x) * threads) {
    const u32 row = u32(i / columns);
    const u64 column = i % columns;
    Text out;
    if (!column) integer(out, row);
    else if (column <= outputs) out = decimal(values.data[u64(row) * outputs + column - 1], status);
    else {
      const double w = weights.size ? weights.data[row] : 1;
      if (w < 0) fail(status, input);
      out = decimal(w, status);
    }
    append(out, column + 1 == columns ? '\n' : ',');
    text[i] = out; lengths[i] = out.size;
  }
}
__global__ void prefix(u64* lengths, u64 count, u64* totals) {
  __shared__ u64 sums[threads];
  for (u64 block = blockIdx.x; block < ceil_div(count, threads); block += gridDim.x) {
    const u64 i = block * threads + threadIdx.x;
    sums[threadIdx.x] = i < count ? lengths[i] : 0;
    __syncthreads();
    for (u32 stride = 1; stride < threads; stride *= 2) {
      const u64 value = threadIdx.x >= stride ? sums[threadIdx.x - stride] : 0;
      __syncthreads();
      sums[threadIdx.x] += value;
      __syncthreads();
    }
    if (i < count) lengths[i] = sums[threadIdx.x];
    if (!threadIdx.x && totals) totals[block] = sums[threads - 1];
    __syncthreads();
  }
}
__global__ void add_offsets(u64* lengths, u64 count, const u64* totals) {
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < count; i += u64(gridDim.x) * threads)
    if (i >= threads) lengths[i] += totals[i / threads - 1];
}
__device__ void put(Array<std::byte> out, u64& position, char c) {
  if (position < out.size) out.data[position] = std::byte(c);
  ++position;
}
__global__ void header(u32 outputs, CsvKind kind, Array<std::byte> out,
    Array<u64> written, u64* header_bytes, const u64* offsets, u64 count, Status* s) {
  if (s->errors) return;
  u64 p = 0;
  for (const char* x = "row_id"; *x; ++x) put(out, p, *x);
  for (u64 column = 0; column < outputs; ++column) {
    put(out, p, ',');
    const char* name = kind == CsvKind::targets ? "target" : kind == CsvKind::multiclass ? "p" : "prediction";
    for (; *name; ++name) put(out, p, *name);
    if (outputs > 1 || kind == CsvKind::multiclass) {
      if (kind != CsvKind::multiclass) put(out, p, '_');
      Text number; integer(number, u32(column));
      for (u32 i = 0; i < number.size; ++i) put(out, p, number.data[i]);
    }
  }
  if (kind == CsvKind::targets) for (const char* x = ",weight"; *x; ++x) put(out, p, *x);
  put(out, p, '\n'); *header_bytes = p;
  const u64 payload = count ? offsets[count - 1] : 0;
  if (!add_fits(p, payload)) { fail(s, extent); return; }
  written.data[0] = s->required_bytes = p + payload;
  if (!contains(out, p + payload)) fail(s, capacity);
}
__global__ void compact(const Text* text, const u64* offsets, u64 count,
    const u64* header_bytes, Array<std::byte> out, Status* s) {
  if (s->errors) return;
  for (u64 i = u64(blockIdx.x) * threads + threadIdx.x; i < count; i += u64(gridDim.x) * threads) {
    const u64 p = *header_bytes + (i ? offsets[i - 1] : 0);
    for (u32 k = 0; k < text[i].size; ++k) out.data[p + k] = std::byte(text[i].data[k]);
  }
}
}
__device__ cudaError_t encode_csv(Array<const double> values, u32 rows, u32 outputs,
    CsvKind kind, Array<const double> weights, Array<std::byte> bytes,
    Array<u64> written, Workspace workspace, Status* status) {
  if (!status) return cudaErrorInvalidValue;
  if (!outputs || u32(kind) > u32(CsvKind::multiclass) || !contains(values, u64(rows) * outputs) ||
      !contains(written, 1) || !contains(bytes, bytes.size) ||
      ((weights.data || weights.size) && (kind != CsvKind::targets || !contains(weights, rows)))) {
    fail(status, shape); return finish(status);
  }
  const u64 columns = u64(outputs) + 1 + (kind == CsvKind::targets);
  if (bytes.size < header_size(outputs, kind)) {
    status->required_bytes = header_size(outputs, kind); fail(status, capacity); return finish(status);
  }
  if (!mul_fits(rows, columns) || !mul_fits(u64(rows) * columns, 25)) { fail(status, extent); return finish(status); }
  const u64 count = u64(rows) * columns;
  Arena arena{workspace};
  auto* text = arena.take<Text>(count); auto* header_bytes = arena.take<u64>(1);
  u64* levels[9]; u64 counts[9]{count}; u32 depth = 0;
  levels[0] = arena.take<u64>(count);
  while (counts[depth] > threads) { counts[depth + 1] = ceil_div(counts[depth], threads); ++depth; levels[depth] = arena.take<u64>(counts[depth]); }
  if (!arena.fits(status)) return finish(status);
  if (count) {
    format<<<grid(count), threads>>>(values, weights, rows, outputs, kind, text, levels[0], status);
    if (!launched(status)) return finish(status);
    for (u32 i = 0; i <= depth; ++i) {
      prefix<<<grid(counts[i]), threads>>>(levels[i], counts[i], i < depth ? levels[i + 1] : nullptr);
      if (!launched(status)) return finish(status);
    }
    for (u32 i = depth; i; --i) {
      add_offsets<<<grid(counts[i - 1]), threads>>>(levels[i - 1], counts[i - 1], levels[i]);
      if (!launched(status)) return finish(status);
    }
  }
  header<<<1, 1>>>(outputs, kind, bytes, written, header_bytes, levels[0], count, status);
  if (!launched(status)) return finish(status);
  if (count) compact<<<grid(count), threads>>>(text, levels[0], count, header_bytes, bytes, status);
  return finish(status);
}
}
