// GH_SOURCE_CATEGORY: production
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#ifndef GH_MODE
#define GH_MODE 0
#endif
#ifndef GH_IMPLEMENTATION
#define GH_IMPLEMENTATION 0
#endif
#ifndef GH_DATA_LEAF_IMPLEMENTATION
#define GH_DATA_LEAF_IMPLEMENTATION 0
#endif
#ifdef __CUDACC__
#include <cuda/std/bit>
#include <cmath>
#include <climits>
#include <limits>
#include <math_constants.h>

// GH_SOURCE_CATEGORY: production
// core
namespace gh {
using u32 = std::uint32_t;
using u64 = std::uint64_t;
template<class T> struct Array { T* data{}; u64 size{}; };
struct Workspace { std::byte* data{}; u64 bytes{}; };
enum Error : u32 { shape = 1, extent = 2, capacity = 4, input = 8,
                   model = 16, numeric = 32, unsupported = 64, runtime = 128 };
struct Status { u32 errors{}, done{}; u64 required_bytes{}; };

__device__ inline void fail(Status* status, Error error) {
  atomicOr(&status->errors, static_cast<u32>(error));
}
__host__ __device__ constexpr bool add_fits(u64 a, u64 b) { return b <= UINT64_MAX - a; }
__host__ __device__ constexpr bool mul_fits(u64 a, u64 b) { return !b || a <= UINT64_MAX / b; }
__host__ __device__ constexpr u64 ceil_div(u64 n, u64 d) { return n / d + (n % d != 0); }
template<class T> __device__ bool contains(Array<T> a, u64 n) {
  const auto address = reinterpret_cast<std::uintptr_t>(a.data);
  return n <= a.size && mul_fits(n, sizeof(T)) &&
    (!n || (a.data && address % alignof(T) == 0 && add_fits(address, n * sizeof(T))));
}

// A local layout cursor: size calculation and partitioning use the same code.
// Null backing storage computes required bytes; dereference requires capacity.
struct Arena {
  Workspace storage;
  u64 used{};
  bool valid{true};
  template<class T> __device__ T* take(u64 n) {
    if (!n) return nullptr;
    constexpr u64 mask = alignof(T) - 1;
    if (!valid || !add_fits(used, mask) || !mul_fits(n, sizeof(T))) {
      valid = false; return nullptr;
    }
    const u64 begin = (used + mask) & ~mask;
    const u64 bytes = n * sizeof(T);
    if (!add_fits(begin, bytes)) { valid = false; return nullptr; }
    used = begin + bytes;
    return storage.data && used <= storage.bytes && add_fits(reinterpret_cast<std::uintptr_t>(storage.data), used)
      ? reinterpret_cast<T*>(storage.data + begin) : nullptr;
  }
  __device__ bool fits(Status* status) const {
    status->required_bytes = used;
    if (!valid) { fail(status, extent); return false; }
    if (!add_fits(reinterpret_cast<std::uintptr_t>(storage.data), used)) { fail(status, extent); return false; }
    if (used > storage.bytes || (used && (!storage.data || reinterpret_cast<std::uintptr_t>(storage.data) % 16))) {
      fail(status, capacity); return false;
    }
    return true;
  }
};

// Called by one GPU coordinator thread. Status is caller-initialized and alive
// through child completion. A tail continuation observes child results safely.
__global__ void complete(Status* status);
__device__ cudaError_t finish(Status* status);
}

// GH_SOURCE_CATEGORY: production
// types
namespace gh {
enum class Objective : u32 { squared_error, binary_logistic, multiclass_softmax };
enum class FeatureType : u32 { numeric, categorical };
enum class RadixPolicy : u32 { radix8 = 8, radix4 = 4 };
struct Feature { u64 begin{}; u32 count{}; FeatureType type{}; };
struct Schema {
  Array<Feature> features;
  Array<float> metadata;
  Array<u32> offsets;
  u32 columns{}, total_bins{}, max_feature_bins{};
  u64 metadata_count{};
};
struct Dataset {
  Array<const float> values, targets, weights;
  u32 rows{}, columns{}, outputs{1};
};
struct Node {
  std::int32_t feature{-1}, left{-1}, right{-1};
  u32 threshold{}, missing_left{};
  double value{};
};
struct Tree { u64 begin{}; u32 count{}, output{}; };
struct Model {
  Schema schema;
  Array<Node> nodes;
  Array<Tree> trees;
  Array<double> base;
  Array<u64> output_offsets;
  u64 node_count{}, tree_count{};
  u32 outputs{1};
  Objective objective{};
};
static_assert(sizeof(Node) == 32 && offsetof(Node, value) == 24);
static_assert(sizeof(Feature) == 16 && sizeof(Tree) == 16);
}

// GH_SOURCE_CATEGORY: production
// count
namespace gh::count {
enum class InputType { u8, u32 };
enum class CounterType { u32, u64 };
enum class LocalCounter { native, u32 };
enum class LoadPolicy { scalar, full_tile, vector4 };
enum class OutputClear { runtime, kernel };
enum class LaunchMode { stream, graph };
enum class CacheMode { warm, cold };
enum class Algorithm { global_atomic, warp_aggregated, shared_atomic, shared_rle,
  shared_warp, shared_partial, bitplane, automatic, shared_overflow, global_window };

struct Tuning { unsigned threads{}, items{}, replicas{}; LoadPolicy load{}; u32 shared_limit{48 * 1024}; };
inline constexpr unsigned tuning_count = 16;
__host__ __device__ constexpr Tuning tuning(unsigned index) {
  constexpr Tuning policies[] = {
    {128,4,1}, {256,4,1}, {256,8,1}, {256,16,1}, {128,8,4}, {256,8,4},
    {256,8,1,LoadPolicy::full_tile}, {256,8,1,LoadPolicy::vector4},
    {256,16,1,LoadPolicy::full_tile}, {256,16,1,LoadPolicy::vector4},
    {512,8,1,LoadPolicy::vector4}, {1024,8,1,LoadPolicy::vector4},
    {256,8,8,LoadPolicy::vector4}, {512,8,16,LoadPolicy::vector4},
    {256,8,1,LoadPolicy::vector4,96 * 1024}, {512,8,1,LoadPolicy::vector4,96 * 1024}};
  return index < tuning_count ? policies[index] : Tuning{};
}
struct Config {
  Algorithm algorithm{Algorithm::automatic};
  InputType input_type{InputType::u32};
  CounterType counter_type{CounterType::u32};
  LocalCounter local_counter{LocalCounter::native};
  u64 size{};
  u32 bins{256}, blocks{192}, policy{2};
  OutputClear output_clear{OutputClear::runtime};
  LaunchMode launch{LaunchMode::stream};
  CacheMode cache{CacheMode::warm};
  u32 window_bins{524288};
};
// Runtime identity/capabilities are supplied infrastructure facts. Defaults
// describe the pinned target; callers on another device must supply its facts.
struct Hardware {
  bool a5000_laptop{true};
  u32 major{8}, minor{6}, sms{48}, max_threads{1024}, max_grid{2147483647};
  u32 shared_bytes{48 * 1024}, shared_optin_bytes{96 * 1024};
};
__device__ bool supported(Config config);
__device__ Config resolve(Config config, Hardware hardware = {});
// Requires a resolved supported configuration; no allocation or CUDA effects.
__device__ u64 required_bytes(Config config);
__device__ cudaError_t count(Config config, Array<const std::byte> input,
    Array<std::byte> output, Workspace workspace, Status* status, Hardware hardware = {});
// Mandatory current-context setup for policy14/15. Only CUDA attribute calls;
// perform before launches/capture, and again after context reset/device change.
cudaError_t initialize_runtime();
}  // namespace gh::count

// GH_SOURCE_CATEGORY: production
// data
namespace gh {
struct DatasetRecord { Dataset data; Objective objective{}; u32 classes{}; };

// One GPU coordinator thread; caller-zeroed status and global backing storage
// remain live until finish's tail completion. Array sizes are capacities.
__device__ cudaError_t fit_schema(Dataset, Array<const FeatureType>, u32 max_bins,
    Schema*, Array<std::uint16_t> bins, Workspace, Status*,
    RadixPolicy = RadixPolicy::radix8);
__device__ cudaError_t encode(Dataset, const Schema*, Array<std::uint16_t>, Status*);
__device__ cudaError_t decode_dataset(Array<const std::byte>, Array<float> values,
    Array<float> targets, DatasetRecord*, Status*);
__device__ cudaError_t encode_dataset(const DatasetRecord*, Array<std::byte>, Status*);
}

// GH_SOURCE_CATEGORY: production
// model
namespace gh {
// GPU submission effects; all pointers must name live disjoint device storage.
// Caller clears Status and observes done/errors only after tail completion.
__device__ cudaError_t validate_model(const Model*, Workspace, Status*);
// Model must have passed validation and remain unchanged. Bins are feature-major;
// output is row-major. Zero rows are a no-op; final-value checks follow transform.
__device__ cudaError_t predict(const Model*, Array<const std::uint16_t> bins,
                             u32 rows, Array<double> output, bool raw, Status*);
// Model array capacities are supplied by caller. Decode writes active extents
// and stable output grouping, then validates before successful completion.
__device__ cudaError_t decode_model(Array<const std::byte>, Model*, Workspace, Status*);
// Previously validated unchanged Model. written has one element; no host codec.
__device__ cudaError_t encode_model(const Model*, Array<std::byte>, Array<u64> written,
                                  Workspace, Status*);
}

// GH_SOURCE_CATEGORY: production
// prediction
namespace gh {
// Headerless row-major little-endian binary64, exactly rows*outputs*8 bytes.
// outputs>0; zero rows are valid. All successful values are finite. Byte buffers
// may be unaligned; double buffers require natural alignment. Regions are disjoint.
// Decode's source extent is exact; encode's destination size is capacity. Caller
// zeroes Status and waits for its completion tail before consuming/reusing storage.
__device__ cudaError_t decode_predictions(Array<const std::byte> bytes, u32 rows,
    u32 outputs, Array<double> values, Status* status);
__device__ cudaError_t encode_predictions(Array<const double> values, u32 rows,
    u32 outputs, Array<std::byte> bytes, Status* status);
}

// GH_SOURCE_CATEGORY: production
// csv
namespace gh {
enum class CsvKind : u32 { targets, predictions, multiclass };
// Values are row-major. Weights are permitted for targets only; empty means one.
// written and all descriptors/backing arrays are global and remain alive through
// completion. Output is raw bytes without a trailing NUL. Partial failure is invalid.
__device__ cudaError_t encode_csv(Array<const double> values, u32 rows, u32 outputs,
    CsvKind kind, Array<const double> weights, Array<std::byte> bytes,
    Array<u64> written, Workspace workspace, Status* status);
}

// GH_SOURCE_CATEGORY: production
// metrics
namespace gh {
enum class MetricProfile : u32 { synthetic, real_data };
enum class MetricTask : u32 { regression, binary, multiclass, multilabel };
enum class MetricName : u32 {
  mse, rmse, mae, r2, logloss, accuracy, brier, auc, f1, average_precision,
  hamming_loss, exact_match, micro_f1, macro_f1, micro_ap, macro_ap,
  micro_auc, macro_auc, precision_at_1, precision_at_3, precision_at_5
};
constexpr u32 aggregate_output = UINT32_MAX;
struct MetricInput {
  Array<const double> targets, predictions, weights;
  u32 rows{}, outputs{};
  MetricTask task{};
  MetricProfile profile{};
  double probability_tolerance{1e-6};
};
struct Metric { MetricName name{}; u32 output{aggregate_output}; double value{}; u32 available{}; };
struct MetricReport {
  Array<Metric> metrics;
  u32 count{}, outputs{}, ap_outputs{}, auc_outputs{};
  MetricTask task{};
  MetricProfile profile{};
};
enum class MetricVerdict : u32 { pass, regression, not_applicable, invalid };
struct MetricGate { u32 checked{}, regressions{}, unavailable{}, invalid{}; };
struct SignalInput { Array<const double> labels, predictions; u32 size{}; };
enum class ThresholdMode : u32 { validation, frozen };
struct OperatingPoint {
  double threshold{}, recall{}, false_positive_rate{}, precision{}, f1{};
  u64 true_positive{}, false_positive{}, false_negative{}, true_negative{};
  u64 positives{}, negatives{};
  u32 meets_five_percent_fpr{};
};
struct SignalReport { OperatingPoint fixed, selected; ThresholdMode origin{}; };

// All computation and submission are device-only. Caller supplies capacities,
// zero Status and global disjoint backing arrays live through tail completion.
__device__ cudaError_t evaluate_metrics(MetricInput, MetricReport*, Workspace, Status*);
__device__ cudaError_t compare_metrics(const MetricReport*, const MetricReport*,
    Array<MetricVerdict>, MetricGate*, Status*);
__device__ cudaError_t signal_metrics(SignalInput, ThresholdMode, double frozen_threshold,
    SignalReport*, Workspace, Status*);
}

// GH_SOURCE_CATEGORY: production
// reference
namespace gh {
// Synchronous GPU conversion of one complete bounded JSON number token. Exact
// RN-even, including signed zero/subnormals/overflow; output unchanged on false.
__device__ bool parse_decimal(Array<const char>, double*);
// Completed real-data metrics on frozen predictions; all report backing arrays
// disjoint/global. Imports the frozen schema and aligns computed legacy keys.
// Caller zeroes Status and waits for tail completion before consuming results.
__device__ cudaError_t decode_metric_reference(Array<const std::byte>, const MetricReport* computed,
    MetricReport* reference, MetricReport* aligned, Status*);
}

// GH_SOURCE_CATEGORY: production
// observe
#ifndef GH_OBSERVE
#define GH_OBSERVE 0
#endif
namespace gh::observe {
inline constexpr bool enabled = GH_OBSERVE != 0;
enum class Stage : u32 { validate, base, gradient, histogram, split, route, loss, export_model };
struct Stamp { u64 ticks{}; Stage stage{}; u32 iteration{}; bool end{}; };
struct Trace { Array<Stamp> records; u32* count{}; };
__device__ cudaError_t enqueue(Trace, Stage, u32 iteration, bool end, Status*);
template<bool Enabled = enabled>
__device__ cudaError_t mark(Trace trace, Stage stage, u32 iteration, bool end, Status* status) {
  if constexpr (Enabled) return enqueue(trace,stage,iteration,end,status);
  return cudaSuccess;
}

inline constexpr u32 warmups = 3, pairs = 15, samples = 2 * (warmups + pairs);
struct Slot { u32 variant, pair; bool warmup; };
__device__ constexpr Slot schedule(u32 ordinal) {
  const u32 pair = ordinal / 2;
  return {(ordinal % 2) ^ (pair % 2), pair < warmups ? pair : pair - warmups, pair < warmups};
}
struct Sample { u64 begin{}, end{}; };
struct Summary { double ratio{}, lower{}, upper{}, log_stddev{}; u64 minimum_ticks{}; };
// Start queues on device-null; end queues after prior completion tails. Start
// the next sample only from a subsequent tail continuation, never a parent loop.
__device__ cudaError_t boundary(Array<Sample>, u32 ordinal, bool end, Status*);
// Launch from the final sample's continuation. Does not alter raw samples.
__device__ cudaError_t summarize(Array<const Sample>, Summary*, Status*);
}

// GH_SOURCE_CATEGORY: production
// learn
namespace gh {
enum class Histogram : u32 { global, shared, automatic };
enum class SplitPolicy : u32 { block256, warp32, warp_wide };
enum class TreeBuild : u32 { per_output, output_batch };
enum class RootCounts : u32 { per_output, global, shared };
struct TrainConfig {
  Objective objective{Objective::squared_error};
  u32 classes{2}, rounds{100}, max_depth{6}, min_leaf_rows{10};
  double learning_rate{0.1}, l2{1}, min_child_hessian{1e-8}, min_gain{}, max_leaf_value{};
  u32 order{2}, output_tile{32};
  Histogram histogram{Histogram::automatic};
  SplitPolicy splits{SplitPolicy::warp32};
  TreeBuild tree_build{TreeBuild::per_output};
  RootCounts root_counts{RootCounts::global};
  bool batched_roots{true}, batched_root_splits{true};
  u64 max_histogram_bytes{512ULL << 20}, max_device_bytes{4ULL << 30};
};
struct TuningRecord {
  u32 output{};
  Histogram selected{};
  u64 global_ticks[5]{}, shared_ticks[5]{};
  bool measured{};
};
struct Training {
  Model* model{};
  Array<double> margins, loss;
  observe::Trace trace;
  Array<TuningRecord> tuning;
  u64 workspace_bytes{}, histogram_bytes{}, derivative_bytes{}, tree_state_bytes{};
  u32 frontier_capacity{}, tree_capacity{}, output_capacity{};
};

// Bins and schema are already fitted on training rows. The caller supplies the
// model's base/tree/node/offset buffers and margins/loss; their capacities are
// checked on GPU. The schema arrays are retained by the resulting model.
// loss has rounds+1 entries; model node storage is compact, descriptor order is
// output then round, and node segments may be permuted. Workspace is reusable
// only after tail completion. On failure partial output is not a valid model.
__device__ cudaError_t train(Dataset data, const Schema* schema,
    Array<const std::uint16_t> bins, TrainConfig config, Training* output,
    Workspace workspace, Status* status);
}

// GH_SOURCE_CATEGORY: production
// detail/learning
namespace gh::detail {
template<unsigned Order> struct Stats {
  double d[Order]{};
  unsigned long long count{};
};
static_assert(sizeof(Stats<2>) == 24 && sizeof(Stats<3>) == 32 && sizeof(Stats<4>) == 40);
struct Choice {
  std::int32_t feature{-1};
  u32 threshold{}, missing_left{};
  double gain{}, value{}, left{}, right{};
};
struct Leaf { double value{}, benefit{}; };
__device__ inline double sigmoid(double x) {
  if (x >= 0) return 1.0 / (1.0 + exp(-x));
  const double e = exp(x); return e / (1.0 + e);
}
template<unsigned O> __device__ Stats<O> add(Stats<O> a, const Stats<O>& b) {
#pragma unroll
  for (unsigned k = 0; k < O; ++k) a.d[k] += b.d[k];
  a.count += b.count; return a;
}
template<unsigned O> __device__ Stats<O> subtract(Stats<O> a, const Stats<O>& b) {
#pragma unroll
  for (unsigned k = 0; k < O; ++k) a.d[k] -= b.d[k];
  a.d[1] = fmax(0.0, a.d[1]); a.count -= b.count; return a;
}
template<unsigned Mode, class T> __device__ T shuffle(T x, unsigned delta) {
  if constexpr (Mode == 0) return __shfl_down_sync(0xffffffff, x, delta);
  if constexpr (Mode == 1) return __shfl_up_sync(0xffffffff, x, delta);
  if constexpr (Mode == 2) return __shfl_sync(0xffffffff, x, delta);
}
template<unsigned Mode, unsigned O> __device__ Stats<O> shuffle(Stats<O> x, unsigned delta) {
#pragma unroll
  for (unsigned k = 0; k < O; ++k) x.d[k] = shuffle<Mode>(x.d[k], delta);
  x.count = shuffle<Mode>(x.count, delta); return x;
}
template<unsigned O> __device__ Stats<O> warp_sum(Stats<O> x) {
  for (unsigned d = 16; d; d >>= 1) x = add(x, shuffle<0>(x, d));
  return x;
}
template<unsigned O> __device__ Stats<O> warp_scan(Stats<O> x) {
  for (unsigned d = 1; d < 32; d <<= 1) {
    auto y = shuffle<1>(x, d);
    if ((threadIdx.x & 31) >= d) x = add(x, y);
  }
  return x;
}
__device__ inline double warp_sum(double x) {
  for (unsigned d = 16; d; d >>= 1) x += shuffle<0>(x, d);
  return x;
}
__device__ inline double warp_max(double x) {
  for (unsigned d = 16; d; d >>= 1) x = fmax(x, shuffle<0>(x, d));
  return x;
}
__device__ inline double block_sum(double x, double* scratch) {
  x = warp_sum(x);
  if (!(threadIdx.x & 31)) scratch[threadIdx.x >> 5] = x;
  __syncthreads();
  if (threadIdx.x < 32) x = warp_sum(threadIdx.x < 8 ? scratch[threadIdx.x] : 0.0);
  return x;
}
template<unsigned O, unsigned Block> __device__ Stats<O> block_sum(Stats<O> x, Stats<O>* scratch) {
  x = warp_sum(x);
  if constexpr (Block == 32) {
    // The order-2 narrow split emulates the original two-level block schedule.
    if constexpr (O == 2) x = warp_sum(threadIdx.x == 0 ? x : Stats<O>{});
    return shuffle<2>(x, 0);
  } else {
    if (!(threadIdx.x & 31)) scratch[threadIdx.x >> 5] = x;
    __syncthreads();
    if (threadIdx.x < 32) {
      x = warp_sum(threadIdx.x < Block / 32 ? scratch[threadIdx.x] : Stats<O>{});
      if (!threadIdx.x) scratch[0] = x;
    }
    __syncthreads(); x = scratch[0]; __syncthreads(); return x;
  }
}
template<unsigned O, unsigned Block> __device__ Stats<O> block_scan(Stats<O> x, Stats<O>* scratch, Stats<O>& total) {
  x = warp_scan(x);
  if constexpr (Block == 32) { total = shuffle<2>(x, 31); return x; }
  const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 31) scratch[warp] = x;
  __syncthreads();
  if (!warp) {
    auto prefix = warp_scan(lane < Block / 32 ? scratch[lane] : Stats<O>{});
    if (lane < Block / 32) scratch[lane] = prefix;
  }
  __syncthreads();
  if (warp) x = add(scratch[warp - 1], x);
  total = scratch[Block / 32 - 1]; __syncthreads(); return x;
}
template<unsigned O> __device__ double benefit(const Stats<O>& s, double v, double l2) {
  if constexpr (O == 2) return -v * (s.d[0] + 0.5 * (s.d[1] + l2) * v);
  double tail = s.d[2] / 6.0;
  if constexpr (O == 4) tail += v * (s.d[3] / 24.0);
  return -v * (s.d[0] + v * (0.5 * (s.d[1] + l2) + v * tail));
}
template<unsigned O> __device__ Leaf leaf(const Stats<O>& s, const TrainConfig& c) {
  const double a = s.d[1] + c.l2;
  if (!(a > 0)) return {};
  double v = -s.d[0] / a;
  if (c.max_leaf_value > 0) v = fmin(c.max_leaf_value, fmax(-c.max_leaf_value, v));
  Leaf result{v, benefit(s, v, c.l2)};
  if constexpr (O > 2) {
    if (!isfinite(s.d[0]) || !isfinite(a) || !(c.max_leaf_value > 0) || !isfinite(c.max_leaf_value)) return {};
    if (!isfinite(result.benefit) || result.benefit < 0) result = {};
    const double x = s.d[0] / a, ratio = x * (s.d[2] / a);
    if (!isfinite(x) || !isfinite(ratio)) return result;
    double numerator = 1, denominator = 1 - 0.5 * ratio;
    if constexpr (O == 4) {
      const double fourth = x * x * (s.d[3] / a);
      if (!isfinite(fourth)) return result;
      numerator = 1 - 0.5 * ratio; denominator = 1 - ratio + fourth / 6;
    }
    if (!isfinite(numerator) || !isfinite(denominator) || !(denominator > 1e-12)) return result;
    double proposed = -x * numerator / denominator;
    if (!isfinite(proposed) || !((s.d[0] > 0 && proposed < 0) || (s.d[0] < 0 && proposed > 0))) return result;
    proposed = fmin(c.max_leaf_value, fmax(-c.max_leaf_value, proposed));
    const double score = benefit(s, proposed, c.l2);
    if (isfinite(score) && score > result.benefit) result = {proposed, score};
  }
  return result;
}
__device__ inline Choice better(Choice a, Choice b) {
  if (b.feature < 0) return a;
  if (a.feature < 0 || b.gain > a.gain) return b;
  if (b.gain < a.gain) return a;
  if (b.feature != a.feature) return b.feature < a.feature ? b : a;
  if (b.threshold != a.threshold) return b.threshold < a.threshold ? b : a;
  return b.missing_left < a.missing_left ? b : a;
}
__device__ inline bool goes_left(u32 bin, FeatureType type, u32 threshold, u32 missing_left) {
  return !bin ? missing_left != 0 : type == FeatureType::categorical ? bin == threshold : bin <= threshold;
}
}

// GH_SOURCE_CATEGORY: production
// detail/metric_math
namespace gh::detail {
__device__ inline bool finite_metric(double x) {
  return (cuda::std::bit_cast<u64>(x) & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}
// Actual fsum operands in the synthetic protocol are finite and nonnegative.
// Three 26-bit pieces per input; <=UINT32_MAX inputs cannot overflow a limb.
struct PositiveSum {
  u64 limbs[84]{};
  __device__ void add(double x, Status* s) {
    if (!finite_metric(x) || x < 0) { fail(s, numeric); return; }
    const u64 bits = cuda::std::bit_cast<u64>(x);
    const u32 exponent = u32((bits >> 52) & 2047);
    const u64 significand = (bits & 0xfffffffffffffULL) | (exponent ? 1ULL << 52 : 0);
    const u32 shift = exponent ? exponent - 1 : 0, first = shift / 26, offset = shift % 26;
    for (u32 k = 0; k < 3; ++k) {
      const int right = int(k * 26) - int(offset);
      limbs[first + k] += (right >= 0 ? significand >> right : significand << -right) & 0x3ffffffULL;
    }
  }
  __device__ double value(Status* s) {
    u64 carry = 0;
    for (u32 i = 0; i < 84; ++i) {
      const u64 v = limbs[i] + carry; limbs[i] = v & 0x3ffffffULL; carry = v >> 26;
    }
    int last = 83;
    while (last >= 0 && !limbs[last]) --last;
    if (last < 0) return 0;
    u32 highest = u32(last * 26 + 63 - __clzll(limbs[last]));
    const u32 shift = highest > 52 ? highest - 52 : 0, first = shift / 26, offset = shift % 26;
    u64 significant = ((limbs[first] >> offset) | (limbs[first + 1] << (26 - offset)) |
      (limbs[first + 2] << (52 - offset))) & 0x1fffffffffffffULL;
    if (shift) {
      const u32 bit = shift - 1, word = bit / 26, within = bit % 26;
      const bool guard = (limbs[word] >> within) & 1;
      bool sticky = (limbs[word] & ((1ULL << within) - 1)) != 0;
      for (u32 i = 0; i < word; ++i) sticky |= limbs[i] != 0;
      if (guard && (sticky || (significant & 1))) {
        if (++significant == (1ULL << 53)) { significant >>= 1; ++highest; }
      }
    }
    if (highest > 2097 || carry) { fail(s, numeric); return 0; }
    const u64 bits = highest < 52 ? significant : (u64(highest - 51) << 52) | (significant & 0xfffffffffffffULL);
    return cuda::std::bit_cast<double>(bits);
  }
};
template<class Get> __device__ double positive_sum(u32 count, Get get, Status* s) {
  PositiveSum sum;
  for (u32 i = 0; i < count; ++i) sum.add(get(i), s);
  return sum.value(s);
}

// NumPy's 128-value leaves and eight ordered accumulators. The explicit stack
// is bounded by the u32 count; this is an arithmetic schedule, not a generic sum.
template<class Get> __device__ double numpy_sum(u32 count, Get get) {
  u32 starts[32], lengths[32];
  double left_values[32];
  bool right[32];
  u32 begin = 0, length = count, depth = 0;
  double result;
  while (true) {
    while (length > 128) {
      const u32 left = (length / 2) & ~7u;
      starts[depth] = begin + left; lengths[depth] = length - left; right[depth++] = false;
      length = left;
    }
    if (length < 8) {
      result = -0.0;
      for (u32 i = 0; i < length; ++i) result = __dadd_rn(result, get(begin + i));
    } else {
      double parts[8];
      for (u32 k = 0; k < 8; ++k) parts[k] = get(begin + k);
      u32 i = 8;
      for (; i < (length & ~7u); i += 8)
        for (u32 k = 0; k < 8; ++k) parts[k] = __dadd_rn(parts[k], get(begin + i + k));
      result = __dadd_rn(__dadd_rn(__dadd_rn(parts[0], parts[1]), __dadd_rn(parts[2], parts[3])),
        __dadd_rn(__dadd_rn(parts[4], parts[5]), __dadd_rn(parts[6], parts[7])));
      for (; i < length; ++i) result = __dadd_rn(result, get(begin + i));
    }
    while (depth && right[depth - 1]) { --depth; result = __dadd_rn(left_values[depth], result); }
    if (!depth) return __dadd_rn(0.0, result); // NumPy's reduction identity.
    right[depth - 1] = true; left_values[depth - 1] = result;
    begin = starts[depth - 1]; length = lengths[depth - 1];
  }
}
template<class Get> __device__ double column_sum(u32 count, u32 columns, Get get) {
  if (columns == 1) return numpy_sum(count, get);
  double sum = 0;
  for (u32 i = 0; i < count; ++i) sum = __dadd_rn(sum, get(i));
  return sum;
}

template<u32 Power, class ErrorAt, class WeightAt>
__device__ double scaled_mean(u32 count, ErrorAt error_at, WeightAt weight_at, double total, Status* s) {
  int denominator_exponent;
  const double denominator = frexp(total, &denominator_exponent);
  int maximum_exponent = INT32_MIN;
  double maximum_error = 0;
  for (u32 i = 0; i < count; ++i) {
    const double weight = weight_at(i);
    if (weight <= 0) continue;
    const double error = error_at(i);
    if (!finite_metric(error) || error < 0) { fail(s, numeric); return 0; }
    maximum_error = fmax(maximum_error, error);
    if (!error) continue;
    int exponent, weight_exponent;
    frexp(error, &exponent); frexp(weight, &weight_exponent);
    maximum_exponent = max(maximum_exponent, int(Power) * exponent + weight_exponent - denominator_exponent);
  }
  if (maximum_exponent == INT32_MIN) return 0;
  PositiveSum sum;
  for (u32 i = 0; i < count; ++i) {
    const double weight = weight_at(i);
    if (weight <= 0) continue;
    const double error = error_at(i);
    if (!error) continue;
    int exponent, weight_exponent;
    const double magnitude = frexp(error, &exponent), wm = frexp(weight, &weight_exponent);
    const double powered = Power == 2 ? __dmul_rn(magnitude, magnitude) : magnitude;
    const double term = __ddiv_rn(__dmul_rn(powered, wm), denominator);
    sum.add(ldexp(term, int(Power) * exponent + weight_exponent - denominator_exponent - maximum_exponent), s);
  }
  double result;
  if constexpr (Power == 2) {
    int exponent = maximum_exponent / 2, remainder = maximum_exponent % 2;
    if (remainder < 0) { --exponent; remainder += 2; }
    result = ldexp(sqrt(ldexp(sum.value(s), remainder)), exponent);
  } else result = ldexp(sum.value(s), maximum_exponent);
  return fmin(result, maximum_error);
}
}

namespace gh::data_impl {
inline constexpr u32 threads = 256, tile_keys = 1024, missing = 0xffffffffu;
template<u32 Bits> __global__ void radix_counts(const u32*,u32*,u32,u32,u32);
template<u32 Bits> __global__ void radix_move(const u32*,u32*,const u32*,u32,u32,u32);
template<bool Compact> __global__ void unique_keys(const u32*,u32*,u32*,u32,u32);
__global__ void scan_tiles(u32*,u32,u32,u32*);
__global__ void scan_offsets(u32*,const u32*,u32,u32,u64);
}

// GH_SOURCE_CATEGORY: tests
// Shared GPU assertions
#if !GH_IMPLEMENTATION && GH_MODE > 0
#include <cassert>
#include <cstdio>
#define GH_CHECK(condition) do { if (!(condition)) { \
  printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); assert(false); \
} } while (false)

namespace gh::test {
__device__ inline void submitted(cudaError_t result) {
  if (result != cudaSuccess) printf("CUDA submission error %d\n", static_cast<int>(result));
  GH_CHECK(result == cudaSuccess);
}
__device__ inline void succeeded(const Status& status) {
  if (status.done != 1 || status.errors) printf("Status done=%u errors=%u required_bytes=%llu\n", status.done, status.errors, static_cast<unsigned long long>(status.required_bytes));
  GH_CHECK(status.done == 1); GH_CHECK(status.errors == 0);
}
__device__ inline u64 mix(u64 value) {
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}
__device__ inline u64 timestamp() {
  u64 result; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(result)); return result;
}
}
#endif

// GH_SOURCE_CATEGORY: tooling
// Isolated frozen benchmark declarations
#if GH_MODE == 11
namespace gh::bench::frozen_count {
// Benchmark-only fixed p15/b48, 16384 bins, u32 input/local and u64 output.
// Accepts exactly 16 Mi or 64 Mi elements; caller validates every ID <16384.
// Arrays/status are disjoint, naturally aligned, live device storage; status
// starts zero. The same exclusive-workspace/completion contract as core applies.
__device__ cudaError_t count(u64 size, Array<const std::byte> input,
    Array<std::byte> output, Workspace workspace, Status* status);
cudaError_t initialize_runtime();
}
#endif

#endif // __CUDACC__

// GH_SOURCE_CATEGORY: tooling
// Host bootstrap boundary
// Host boundary: arguments are opaque resident buffers, byte extents and a
// compile-time executable mode. CUDA wrappers only bootstrap the selected GPU
// computation; callers retain every buffer until CUDA completion. No host
// fixtures, validation, decisions, metrics or benchmark statistics cross here.
#include <cuda_runtime_api.h>
#include <cstddef>

inline constexpr std::size_t gh_quality_arena_bytes = 512ULL << 20;
extern "C" {
cudaError_t gh_initialize(int mode);
cudaError_t gh_launch(int mode);
cudaError_t gh_after(int mode);
cudaError_t gh_quality_launch(const void* data, std::size_t data_n,
    const void* model, std::size_t model_n, const void* predictions,
    std::size_t predictions_n, const void* quality, std::size_t quality_n,
    void* arena, std::size_t arena_n);
}
