#include "ghb/higher_order.cuh"
#include "ghb/root_histogram.cuh"
#include "ghb/batch_resident.cuh"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ghb;
using namespace ghb::gpu;
std::size_t checks{}, observed_loss_increases{};
std::string context;
void require(bool value, const std::string& message) {
  ++checks;
  if (!value) throw std::runtime_error(context + ": " + message);
}
void check(cudaError_t value) {
  if (value != cudaSuccess) throw std::runtime_error(context + ": " + cudaGetErrorString(value));
}
void near(double actual, long double expected, const std::string& message, long double tolerance = 3e-11L,
          long double floor = 1) {
  const long double error = std::abs(static_cast<long double>(actual) - expected);
  if (!(std::isfinite(actual) && std::isfinite(expected) &&
        error <= tolerance * std::max(floor, std::abs(expected)))) {
    std::ostringstream details;
    details << message << std::setprecision(20) << " actual=" << actual << " expected=" << expected
            << " error=" << error;
    require(false, details.str());
  }
  ++checks;
}
struct Stream {
  cudaStream_t value{};
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamSynchronize(value); cudaStreamDestroy(value); }
};
template<class T> struct Device {
  T* data{};
  std::size_t size{};
  explicit Device(std::size_t n) : size(n) { if (n) check(cudaMalloc(reinterpret_cast<void**>(&data), n * sizeof(T))); }
  ~Device() { if (data) cudaFree(data); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  void put(const std::vector<T>& values, cudaStream_t stream) {
    require(values.size() == size, "test upload extent");
    if (size) check(cudaMemcpyAsync(data, values.data(), size * sizeof(T), cudaMemcpyHostToDevice, stream));
    check(cudaStreamSynchronize(stream));
  }
  std::vector<T> get(cudaStream_t stream) const {
    std::vector<T> result(size);
    if (size) check(cudaMemcpyAsync(result.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    return result;
  }
};
struct Graph {
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  template<class Function> Graph(cudaStream_t stream, Function&& submit) {
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)); submit();
    check(cudaStreamEndCapture(stream, &graph)); check(cudaGraphInstantiate(&executable, graph, 0));
  }
  ~Graph() { cudaGraphExecDestroy(executable); cudaGraphDestroy(graph); }
};

// Independent identities using long-double exponentials and hyperbolic
// functions; the production derivative helper is deliberately never called.
std::array<long double, 4> logistic(double margin, float target, float weight) {
  if (weight == 0) return {};
  const long double z = margin, w = weight;
  const long double h = w / (2 + 2 * std::cosh(z));
  const long double c = std::cosh(z / 2);
  return {target == 1 ? -w / (1 + std::exp(z)) : w / (1 + std::exp(-z)),
          h, -h * std::tanh(z / 2), h * (1 - 1.5L / (c * c))};
}
template<unsigned Order> void derivatives(Stream& stream) {
  context = "order " + std::to_string(Order) + " derivatives";
  constexpr unsigned rows = 19, outputs = 37;
  const double margins[]{-1000, -700, -40, -8, -1, -.125, 0, .125, 1, 8, 40, 700, 1000};
  std::vector<double> prediction(rows * outputs);
  std::vector<float> target(prediction.size()), weights(rows);
  for (unsigned r = 0; r < rows; ++r) {
    weights[r] = r % 5 == 0 ? 0 : r % 3 == 0 ? .5f : r % 3 == 1 ? 1 : 2;
    for (unsigned o = 0; o < outputs; ++o) {
      prediction[r * outputs + o] = margins[(r * 7 + o) % std::size(margins)];
      target[r * outputs + o] = float((r + o) & 1);
    }
  }
  Device<double> dp(prediction.size()), dg(prediction.size() + 2), dh(dg.size), dt(dg.size), dq(dg.size);
  Device<float> dy(target.size()), dw(weights.size());
  dp.put(prediction, stream.value); dy.put(target, stream.value); dw.put(weights, stream.value);
  for (bool weighted : {false, true}) for (const auto tile : {std::array<unsigned, 2>{0, outputs}, {5, 17}, {36, 1}}) {
    const std::vector<double> guard(dg.size, 1234567.25);
    dg.put(guard, stream.value); dh.put(guard, stream.value); dt.put(guard, stream.value); dq.put(guard, stream.value);
    check(higher_gradients_tile<Order>(dp.data, dy.data, weighted ? dw.data : nullptr, dg.data + 1,
      dh.data + 1, dt.data + 1, Order == 4 ? dq.data + 1 : nullptr, rows, outputs, tile[0], tile[1], stream.value));
    const std::array<std::vector<double>, 4> actual{dg.get(stream.value), dh.get(stream.value), dt.get(stream.value), dq.get(stream.value)};
    for (unsigned d = 0; d < 4; ++d) for (std::size_t i = 0; i < dg.size; ++i) {
      if (d >= Order || i == 0 || i > rows * tile[1]) require(actual[d][i] == guard[i], "derivative inactive/guard write");
      else {
        const unsigned row = unsigned(i - 1) / tile[1], output = tile[0] + unsigned(i - 1) % tile[1];
        const auto expected = logistic(prediction[row * outputs + output], target[row * outputs + output], weighted ? weights[row] : 1);
        // Casting specifies the representable double reference at underflow.
        near(actual[d][i], static_cast<double>(expected[d]), "independent logistic derivative", 5e-13L, 1e-300L);
      }
    }
  }
  require(higher_gradients_tile<Order>(dp.data, dy.data, dw.data, dg.data, dh.data, dt.data,
    Order == 4 ? dq.data : nullptr, rows, outputs, outputs, 1, stream.value) == cudaErrorInvalidValue, "reject derivative output overflow");
  require(higher_gradients_tile<Order>(dp.data, dy.data, dw.data, dg.data, dh.data, nullptr,
    Order == 4 ? dq.data : nullptr, rows, outputs, 0, 1, stream.value) == cudaErrorInvalidValue, "reject missing third derivative");
  if constexpr (Order == 4) require(higher_gradients_tile<Order>(dp.data, dy.data, dw.data, dg.data, dh.data,
    dt.data, nullptr, rows, outputs, 0, 1, stream.value) == cudaErrorInvalidValue, "reject missing fourth derivative");
}

struct Packed {
  unsigned rows, columns, bins;
  std::vector<std::uint16_t> keys;
  std::vector<unsigned> offsets;
  std::vector<FeatureType> types;
  Device<std::uint16_t> device_keys;
  Device<unsigned> device_offsets;
  Device<FeatureType> device_types;
  DataView view;
  Packed(unsigned n, unsigned f, unsigned b, cudaStream_t stream)
      : rows(n), columns(f), bins(b), keys(std::size_t(n) * f), offsets(f + 1), types(f),
        device_keys(keys.size()), device_offsets(offsets.size()), device_types(types.size()) {
    for (unsigned feature = 0; feature < f; ++feature) {
      offsets[feature] = feature * b;
      types[feature] = feature % 3 == 1 ? FeatureType::categorical : FeatureType::numeric;
      for (unsigned row = 0; row < n; ++row)
        keys[std::size_t(feature) * n + row] = row % 13 == 0 ? 0 : 1 + (row * (2 * feature + 1) + feature) % (b - 1);
    }
    offsets[f] = f * b;
    device_keys.put(keys, stream); device_offsets.put(offsets, stream); device_types.put(types, stream);
    view = {device_keys.data, device_offsets.data, device_types.data, n, f, f * b, b};
  }
};
template<unsigned Order> HigherStats<Order> sentinel() {
  HigherStats<Order> value; value.gradient = 1234567.25; value.hessian = -7654321.5; value.count = 0xfedcba9876543210ULL;
  for (unsigned i = 0; i < Order - 2; ++i) value.extra[i] = -23456.25 - i;
  return value;
}
template<unsigned Order> void same_stats(const HigherStats<Order>& a, const HigherStats<Order>& b) {
  require(a.gradient == b.gradient && a.hessian == b.hessian && a.count == b.count, "exact dyadic histogram G/H/count or guard");
  for (unsigned i = 0; i < Order - 2; ++i) require(a.extra[i] == b.extra[i], "exact signed higher histogram or guard");
}
template<unsigned Order> void histograms(Stream& stream) {
  constexpr unsigned rows = 257, batch = 33, capacity = 3, stride_max = 37;
  for (unsigned bins : {3u, 33u}) {
    context = "order " + std::to_string(Order) + " histograms bins " + std::to_string(bins);
    Packed data(rows, 3, bins, stream.value);
    const auto per_output = std::size_t(capacity) * data.view.total_bins;
    std::vector<double> g(rows * stride_max), h(g.size()), t(g.size()), q(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
      g[i] = (int(i % 31) - 15) / 8.; h[i] = (i % 7) / 8.;
      t[i] = (int(i % 17) - 8) / 32.; q[i] = (int(i % 13) - 6) / 64.;
    }
    std::vector<int> assignments(batch * rows);
    std::vector<unsigned> active(batch);
    for (unsigned o = 0; o < batch; ++o) {
      active[o] = o % 4 == 0 ? 0 : o % 4 == 1 ? 1 : o % 4 == 2 ? capacity : capacity + 7;
      for (unsigned r = 0; r < rows; ++r) assignments[o * rows + r] = int((r + o) % (capacity + 2)) - 1;
    }
    Device<double> dg(g.size()), dh(h.size()), dt(t.size()), dq(q.size());
    dg.put(g, stream.value); dh.put(h, stream.value); dt.put(t, stream.value); dq.put(q, stream.value);
    Device<int> da(assignments.size()); Device<unsigned> dn(active.size()); Device<OutputBatch> selector(1);
    da.put(assignments, stream.value); dn.put(active, stream.value);
    Device<unsigned long long> counts(data.view.total_bins);
    check(root_counts(data.view, counts.data, RootCountKernel::global, stream.value));
    Device<HigherStats<Order>> output(batch * per_output + 2);
    const auto guard = sentinel<Order>();
    for (unsigned mode = 0; mode < 3; ++mode) {
      auto launch = [&] {
        if (mode < 2) check(higher_root_histogram_batch<Order>(data.view, dg.data, dh.data, dt.data,
          Order == 4 ? dq.data : nullptr, stride_max, batch, per_output, selector.data, output.data + 1,
          stream.value, mode ? counts.data : nullptr));
        else check(higher_deeper_histogram<Order>(data.view, da.data, dg.data, dh.data, dt.data,
          Order == 4 ? dq.data : nullptr, stride_max, batch, capacity, dn.data, selector.data, output.data + 1, stream.value));
      };
      Graph graph(stream.value, launch);
      for (bool replay : {false, true}) for (OutputBatch chosen : {
          OutputBatch{0,batch,0,batch}, OutputBatch{99,3,0,3}, OutputBatch{7,17,5,stride_max},
          OutputBatch{0,batch+4,0,stride_max}, OutputBatch{0,0,0,1}, OutputBatch{0,3,0,0},
          OutputBatch{0,3,0,stride_max+1}, OutputBatch{0,3,stride_max-1,stride_max}}) {
        auto expected = std::vector<HigherStats<Order>>(output.size, guard);
        const unsigned live = std::min(batch, chosen.output_count);
        const bool valid = chosen.derivative_stride && chosen.derivative_stride <= stride_max &&
          chosen.derivative_begin < chosen.derivative_stride && live <= chosen.derivative_stride - chosen.derivative_begin;
        for (unsigned o = 0; valid && o < live; ++o) {
          const unsigned nodes = mode < 2 ? 1 : std::min(capacity, active[o]);
          for (std::size_t cell = 0; cell < std::size_t(nodes) * data.view.total_bins; ++cell) expected[1 + o * per_output + cell] = {};
          for (unsigned r = 0; r < rows; ++r) {
            const int node = mode < 2 ? 0 : assignments[o * rows + r];
            if (node < 0 || unsigned(node) >= nodes) continue;
            const auto source = std::size_t(r) * chosen.derivative_stride + chosen.derivative_begin + o;
            for (unsigned f = 0; f < data.columns; ++f) {
              auto& value = expected[1 + o * per_output + std::size_t(node) * data.view.total_bins + data.offsets[f] + data.keys[f * rows + r]];
              value.gradient += g[source]; value.hessian += h[source]; value.extra[0] += t[source];
              if constexpr (Order == 4) value.extra[1] += q[source];
              ++value.count;
            }
          }
        }
        output.put(std::vector<HigherStats<Order>>(output.size, guard), stream.value); selector.put({chosen}, stream.value);
        if (replay) check(cudaGraphLaunch(graph.executable, stream.value)); else launch();
        const auto actual = output.get(stream.value);
        for (std::size_t i = 0; i < actual.size(); ++i) same_stats(actual[i], expected[i]);
      }
    }
    require(higher_root_histogram_batch<Order>(data.view, dg.data, dh.data, dt.data, Order == 4 ? dq.data : nullptr,
      stride_max, batch, data.view.total_bins - 1, selector.data, output.data, stream.value) == cudaErrorInvalidValue, "reject short histogram output stride");
    require(higher_deeper_histogram<Order>(data.view, da.data, dg.data, dh.data, dt.data, Order == 4 ? dq.data : nullptr,
      stride_max, batch, 0, dn.data, selector.data, output.data, stream.value) == cudaErrorInvalidValue, "reject zero deeper capacity");
  }
}

struct Totals {
  std::array<long double, 4> derivative{};
  unsigned long long count{};
};
template<unsigned Order> void add(Totals& total, const HigherStats<Order>& value) {
  total.derivative[0] += value.gradient; total.derivative[1] += value.hessian;
  for (unsigned i = 2; i < Order; ++i) total.derivative[i] += value.extra[i - 2];
  total.count += value.count;
}
long double benefit(const Totals& total, long double step, double l2, unsigned order) {
  long double result{}, power = 1, factorial = 1;
  for (unsigned i = 0; i < order; ++i) {
    power *= step; factorial *= i + 1;
    result -= (total.derivative[i] + (i == 1 ? l2 : 0)) * power / factorial;
  }
  return result;
}
struct Estimate { long double value{}, benefit{}; };
// Derive proposals from the reciprocal power series of the leaf derivative,
// not the normalized rational expression or Horner score used on the GPU.
Estimate leaf(const Totals& total, const SplitConfig& config, unsigned order) {
  const long double g = total.derivative[0], a = total.derivative[1] + config.l2;
  if (!(a > 0)) return {};
  const long double radius = config.max_leaf_value;
  auto clip = [&](long double v) { return std::clamp(v, -radius, radius); };
  Estimate best;
  const long double n = clip(-g / a), score = benefit(total, n, config.l2, order);
  if (std::isfinite(score) && score >= 0) best = {n, score};
  if (g == 0) return best;
  const long double t = total.derivative[2], q = total.derivative[3];
  const long double denominator = order == 3 ? (2*a*a-g*t)/(2*a*a) : (6*a*a*a-6*g*a*t+g*g*q)/(6*a*a*a);
  if (!(denominator > 1e-12L)) return best;
  const std::array<long double, 4> f{g,a,t/2,q/6};
  std::array<long double, 4> reciprocal{1/g,0,0,0};
  for (unsigned i = 1; i < order; ++i) {
    for (unsigned j = 1; j <= i; ++j) reciprocal[i] -= f[j] * reciprocal[i-j];
    reciprocal[i] /= g;
  }
  long double proposed = reciprocal[order-2] / reciprocal[order-1];
  if (!std::isfinite(proposed) || !((g > 0 && proposed < 0) || (g < 0 && proposed > 0))) return best;
  proposed = clip(proposed);
  const auto higher_score = benefit(total, proposed, config.l2, order);
  if (std::isfinite(higher_score) && higher_score > best.benefit) best = {proposed, higher_score};
  return best;
}
bool goes_left(unsigned bin, FeatureType type, unsigned threshold, unsigned missing_left) {
  return bin == 0 ? missing_left != 0 : type == FeatureType::categorical ? bin == threshold : bin <= threshold;
}
template<unsigned Order> Split reference_split(const Packed& data, const HigherStats<Order>* histogram,
    const SplitConfig& config, bool force_leaf, int only_feature = -1) {
  Totals parent;
  for (unsigned bin = 0; bin < data.bins; ++bin) add(parent, histogram[bin]);
  const auto parent_leaf = leaf(parent, config, Order);
  Split best; best.value = double(parent_leaf.value);
  long double best_gain{};
  if (force_leaf || parent.count < 2ULL * config.min_leaf_rows) return best;
  for (unsigned f = 0; f < data.columns; ++f) {
    if (only_feature >= 0 && f != unsigned(only_feature)) continue;
    for (unsigned threshold = 0; threshold < data.bins; ++threshold) for (unsigned missing = 0; missing < 2; ++missing) {
      Totals left, right;
      for (unsigned bin = 0; bin < data.bins; ++bin)
        add(goes_left(bin, data.types[f], threshold, missing) ? left : right, histogram[data.offsets[f] + bin]);
      if (left.count < config.min_leaf_rows || right.count < config.min_leaf_rows ||
          left.derivative[1] < config.min_child_hessian || right.derivative[1] < config.min_child_hessian) continue;
      const auto l = leaf(left, config, Order), r = leaf(right, config, Order);
      const auto gain = l.benefit + r.benefit - parent_leaf.benefit;
      if (gain > config.min_gain && (best.feature < 0 || gain > best_gain)) {
        best_gain = gain;
        best = {int(f), threshold, missing, double(gain), double(parent_leaf.value), double(l.value), double(r.value)};
      }
    }
  }
  return best;
}
void equal_split(const Split& a, const Split& b) {
  require(a.feature == b.feature && a.threshold == b.threshold && a.missing_left == b.missing_left, "bitwise block/warp split structure or guard");
  for (const auto pair : {std::array<double,2>{a.gain,b.gain}, {a.value,b.value}, {a.left_value,b.left_value}, {a.right_value,b.right_value}})
    require(std::bit_cast<std::uint64_t>(pair[0]) == std::bit_cast<std::uint64_t>(pair[1]), "bitwise block/warp split value or guard");
}
template<unsigned Order> void splits(Stream& stream) {
  constexpr unsigned rows = 129, batch = 7, capacity = 3;
  for (unsigned bins : {5u, 33u}) for (unsigned features : {3u, 33u}) {
    context = "order " + std::to_string(Order) + " splits bins " + std::to_string(bins) + " features " + std::to_string(features);
    Packed data(rows, features, bins, stream.value);
    std::vector<HigherStats<Order>> histogram(std::size_t(batch) * capacity * data.view.total_bins);
    for (unsigned o = 0; o < batch; ++o) for (unsigned r = 0; r < rows; ++r) {
      HigherStats<Order> value;
      value.gradient = (data.keys[r] <= bins / 2 ? -1 : 1) * (.5 + o / 8.) + (r % 7) / 16.;
      value.hessian = .5 + (r % 5) / 8.; value.extra[0] = (int((r*3+o)%11)-5)/64.;
      if constexpr (Order == 4) value.extra[1] = (int((r*7+o)%9)-4)/128.;
      value.count = 1;
      for (unsigned f = 0; f < features; ++f) {
        auto& cell = histogram[(std::size_t(o)*capacity+r%capacity)*data.view.total_bins+data.offsets[f]+data.keys[f*rows+r]];
        cell.gradient += value.gradient; cell.hessian += value.hessian; cell.count += value.count;
        for (unsigned i = 0; i < Order - 2; ++i) cell.extra[i] += value.extra[i];
      }
    }
    Device<HigherStats<Order>> dh(histogram.size()); dh.put(histogram, stream.value);
    Device<unsigned> active(batch), live(1);
    const std::vector<unsigned> counts{3,0,1,8,2,3,1}; active.put(counts, stream.value);
    Device<Split> candidates(std::size_t(batch)*capacity*features+2), winners(std::size_t(batch)*capacity+2);
    Split guard; guard.feature = -7; guard.gain = -999; guard.value = 1234; guard.left_value = -91; guard.right_value = 88;
    for (unsigned mode = 0; mode < 3; ++mode) {
      const SplitConfig config{mode == 2 ? 1000u : 2u, .5, .01, 0, .75};
      const bool force = mode == 1;
      std::vector<Split> expected(histogram.size()/data.view.total_bins), expected_candidates(expected.size()*features);
      for (unsigned node = 0; node < expected.size(); ++node) {
        const auto* source = histogram.data()+std::size_t(node)*data.view.total_bins;
        expected[node] = reference_split(data, source, config, force);
        for (unsigned f = 0; f < features; ++f) expected_candidates[node*features+f] = reference_split(data, source, config, force, int(f));
      }
      auto launch = [&](SplitPolicy policy) { check(higher_find_splits_batched_active<Order>(data.view, dh.data, batch, capacity, active.data,
        config, force, candidates.data+1, winners.data+1, policy, stream.value, live.data)); };
      Graph block_graph(stream.value, [&] { launch(SplitPolicy::block256); });
      Graph warp_graph(stream.value, [&] { launch(SplitPolicy::warp32); });
      for (unsigned selected : {0u, 3u, batch, batch+1}) {
        live.put({selected}, stream.value);
        std::vector<Split> block_candidates, block_winners;
        for (auto policy : {SplitPolicy::block256, SplitPolicy::warp32}) {
          candidates.put(std::vector<Split>(candidates.size, guard), stream.value); winners.put(std::vector<Split>(winners.size, guard), stream.value);
          check(cudaGraphLaunch(policy == SplitPolicy::block256 ? block_graph.executable : warp_graph.executable, stream.value));
          const auto actual_candidates = candidates.get(stream.value), actual = winners.get(stream.value);
          equal_split(actual.front(),guard); equal_split(actual.back(),guard);
          equal_split(actual_candidates.front(),guard); equal_split(actual_candidates.back(),guard);
          for (unsigned node = 0; node < expected.size(); ++node) {
            const bool enabled = node/capacity < std::min(batch,selected) && node%capacity < std::min(capacity,counts[node/capacity]);
            if (!enabled) {
              equal_split(actual[node+1], guard);
              for (unsigned f = 0; f < features; ++f) equal_split(actual_candidates[1+node*features+f],guard);
              continue;
            }
            auto validate = [&](const Split& chosen, const Split& reference) {
              near(chosen.value, reference.value, "parent safeguarded proposal");
              require((chosen.feature >= 0) == (reference.feature >= 0), "split validity against independent oracle");
              if (chosen.feature < 0) return;
              require(unsigned(chosen.feature) < features && chosen.threshold < bins && chosen.missing_left <= 1, "legal split fields");
              near(chosen.gain, reference.gain, "best independent same-order split gain");
              Totals left, right, parent;
              const auto* source = histogram.data()+std::size_t(node)*data.view.total_bins+data.offsets[chosen.feature];
              for (unsigned bin = 0; bin < bins; ++bin) {
                add(parent, source[bin]);
                add(goes_left(bin,data.types[chosen.feature],chosen.threshold,chosen.missing_left)?left:right,source[bin]);
              }
              const auto l=leaf(left,config,Order), r=leaf(right,config,Order), p=leaf(parent,config,Order);
              near(chosen.left_value,l.value,"left safeguarded proposal"); near(chosen.right_value,r.value,"right safeguarded proposal");
              near(chosen.gain,l.benefit+r.benefit-p.benefit,"actual partition same-order gain");
            };
            validate(actual[node+1],expected[node]);
            for (unsigned f=0;f<features;++f) validate(actual_candidates[1+node*features+f],expected_candidates[node*features+f]);
          }
          if (policy == SplitPolicy::block256) { block_candidates=actual_candidates; block_winners=actual; }
          else {
            for (std::size_t i=0;i<actual.size();++i) equal_split(actual[i],block_winners[i]);
            for (std::size_t i=0;i<actual_candidates.size();++i) equal_split(actual_candidates[i],block_candidates[i]);
          }
        }
      }
    }
    const auto after=dh.get(stream.value);
    for(std::size_t i=0;i<after.size();++i) same_stats(after[i],histogram[i]);
    SplitConfig invalid{1,1,0,0,0};
    require(higher_find_splits_batched_active<Order>(data.view,dh.data,batch,capacity,active.data,invalid,false,
      candidates.data,winners.data,SplitPolicy::warp32,stream.value,live.data)==cudaErrorInvalidValue,"reject missing positive proposal bound");
  }
}

Dataset fixture(unsigned outputs, bool heldout=false) {
  Dataset data; data.rows=heldout?67:129; data.columns=3; data.outputs=outputs;
  data.feature_types={FeatureType::numeric,FeatureType::categorical,FeatureType::numeric};
  data.values.resize(data.rows*data.columns); data.targets.resize(data.rows*outputs); data.weights.resize(data.rows);
  for(unsigned r=0;r<data.rows;++r) {
    const int x=int((r*17+(heldout?3:0))%101)-50;
    data.values[r*3]=r%13==0?std::numeric_limits<float>::quiet_NaN():float(x)/16;
    data.values[r*3+1]=r%17==0?std::numeric_limits<float>::quiet_NaN():heldout&&r%7==0?99.f:float(r%5);
    data.values[r*3+2]=r%19==0?std::numeric_limits<float>::quiet_NaN():float(int((r*7)%79)-39)/8;
    data.weights[r]=r%11==0?0:r%3==0?.5f:r%3==1?1:2;
    for(unsigned o=0;o<outputs;++o) data.targets[r*outputs+o]=float(x+int(r%5)*3>int(o%9)*3-12);
  }
  return data;
}
long double objective(const Dataset& data,const std::vector<double>& margins) {
  long double sum{},weight{};
  for(unsigned r=0;r<data.rows;++r) {
    const long double w=data.weights.empty()?1:data.weights[r]; weight+=w;
    for(unsigned o=0;o<data.outputs;++o) {
      const auto i=std::size_t(r)*data.outputs+o; const long double z=margins[i];
      sum+=w*(std::max(z,0.L)+std::log1p(std::exp(-std::abs(z)))-data.targets[i]*z);
    }
  }
  return sum/(weight*data.outputs);
}
template<unsigned Order> void memory(const TrainingResult& result,const Dataset& data,const TrainConfig& config) {
  const std::size_t batch=result.tree_batch_size,outputs=data.outputs;
  std::size_t bins{}; for(const auto& f:result.model.features) bins+=f.bins();
  const std::size_t count_bytes=config.root_counts==RootCountPolicy::per_output?0:bins*sizeof(unsigned long long);
  const std::size_t leaves=std::max(1u,data.rows/config.min_leaf_rows);
  const std::size_t capacity=std::min<std::size_t>(std::size_t(1)<<(config.max_depth?config.max_depth-1:0),leaves);
  const std::size_t nodes=std::min<std::size_t>((std::size_t(1)<<(config.max_depth+1))-1,2*leaves-1);
  require(batch&&batch<=std::min(outputs,std::size_t(config.output_tile_size)),"bounded output width");
  require(result.gradient_bytes==std::size_t(data.rows)*batch*Order*sizeof(double),"exact higher derivative payload");
  require(result.root_count_bytes==count_bytes&&result.histogram_bytes==count_bytes+batch*capacity*bins*sizeof(HigherStats<Order>),"exact higher histogram payload and full frontier");
  require(result.root_histogram_bytes==0&&result.root_split_bytes==0&&result.root_histogram_batch_size==batch,"no separate batch root cache");
  const std::size_t history=config.record_stages&&config.tree_execution==TreeExecution::stream?std::max(1u,config.max_depth):0;
  const std::size_t state=batch*(std::size_t(data.rows)*sizeof(int)+nodes*sizeof(Node)+capacity*(4*sizeof(int)+sizeof(unsigned))+
    ((capacity+1023)/1024)*sizeof(unsigned)+sizeof(TreeState)+sizeof(unsigned));
  const std::size_t quantized=((std::size_t(data.rows)*data.columns*sizeof(std::uint16_t)+3)&~std::size_t(3))+
    (data.columns+1)*sizeof(unsigned)+data.columns*sizeof(FeatureType);
  const std::size_t loss=std::min(4096u,1+(data.rows-1)/256),init=std::min(256u,1+(data.rows-1)/1024)*(outputs+1);
  const std::size_t bytes=quantized+std::size_t(data.rows)*outputs*sizeof(double)+(data.targets.size()+data.weights.size())*sizeof(float)+
    outputs*sizeof(double)+(loss+init+2)*sizeof(double)+sizeof(unsigned)+sizeof(OutputBatch)+result.gradient_bytes+result.histogram_bytes+
    batch*capacity*(data.columns+1)*sizeof(Split)+state+batch*history*sizeof(unsigned);
  require(result.tree_state_bytes==state&&result.device_bytes==bytes,"exact complete owned payload");
  require(result.histogram_bytes<=config.max_histogram_bytes&&std::max(result.device_bytes,result.preparation_peak_bytes)<=config.max_device_bytes,"memory budgets");
  const std::size_t exports=std::min<std::size_t>({config.tree_export_batch_size,batch,(64ULL<<20)/(nodes*sizeof(Node))});
  require(result.tree_export_batch_size==exports&&result.pinned_export_bytes==exports*nodes*sizeof(Node)+batch*sizeof(TreeState)+
    sizeof(OutputBatch)+batch*history*sizeof(unsigned),"bounded pinned export payload");
}
void validate_model_result(const TrainingResult& result,const Dataset& data,const TrainConfig& config) {
  const auto held=fixture(data.outputs,true);
  require(result.model.objective==Objective::binary_logistic&&result.model.outputs==data.outputs,"model objective/output metadata");
  require(result.model.trees.size()==std::size_t(config.rounds)*data.outputs,"one tree per round/output");
  for(std::size_t i=0;i<result.model.trees.size();++i) {
    require(result.model.trees[i].output==i%data.outputs,"round/output tree order");
    for(const auto& node:result.model.trees[i].nodes) require(std::isfinite(node.value)&&std::abs(node.value)<=config.learning_rate*config.max_leaf_value,
      "finite materialized value within actual margin-step bound");
  }
  for(const auto* input:{&data,&held}) for(bool raw:{false,true}) {
    const auto cpu=result.model.predict(*input,raw),gpu=result.model.predict_gpu(*input,raw);
    require(cpu.size()==gpu.size(),"prediction shape");
    for(std::size_t i=0;i<cpu.size();++i) near(gpu[i],cpu[i],"independent exported CPU/GPU prediction",2e-13L);
  }
  require(result.training_loss.size()==std::size_t(config.rounds)+1,"per-round objective history");
  for(unsigned round=0;round<=config.rounds;++round) {
    auto prefix=result.model; prefix.trees.resize(std::size_t(round)*data.outputs);
    near(result.training_loss[round],objective(data,prefix.predict(data,true)),"independent weighted loss at each round",2e-12L);
    if(round&&result.training_loss[round]>result.training_loss[round-1]) ++observed_loss_increases;
  }
  std::ostringstream serialized(std::ios::binary); result.model.save(serialized);
  std::istringstream input(serialized.str(),std::ios::binary); const auto restored=Model::load(input);
  require(restored.predict(held,true)==result.model.predict(held,true),"exact saved-model inference round trip");
  std::size_t builds{},operations{};
  for(const auto& sample:result.samples) if(sample.stage==instrumentation::Stage::tree_build) {
    ++builds; operations+=sample.context.operations;
    require(sample.context.output>=0&&sample.context.operations>0&&sample.context.operations<=result.tree_batch_size,"meaningful tile stage context");
  }
  require(builds==std::size_t(config.rounds)*((data.outputs+result.tree_batch_size-1)/result.tree_batch_size)&&
    operations==std::size_t(config.rounds)*data.outputs,"complete tree-build instrumentation");
}
template<unsigned Order> void training() {
  TrainConfig config; config.objective=Objective::binary_logistic; config.optimization_order=Order;
  config.tree_build=TreeBuildPolicy::output_batch; config.histogram=HistogramPolicy::global;
  config.rounds=2; config.min_leaf_rows=2; config.learning_rate=.125; config.l2=.5; config.max_leaf_value=1;
  config.output_tile_size=3; config.tree_export_batch_size=2; config.record_stages=true; config.nvtx=false;
  for(const auto shape:{std::array<unsigned,3>{1,0,17},{7,3,17},{33,2,64}}) for(auto execution:{TreeExecution::stream,TreeExecution::graph}) {
    context="order "+std::to_string(Order)+" trainer outputs "+std::to_string(shape[0])+" depth "+std::to_string(shape[1])+" execution "+std::to_string(unsigned(execution));
    const auto data=fixture(shape[0]); config.max_depth=shape[1]; config.max_bins=shape[2]; config.tree_execution=execution;
    config.histogram=execution==TreeExecution::stream?HistogramPolicy::global:HistogramPolicy::autotune;
    const auto result=train(data,config); memory<Order>(result,data,config); validate_model_result(result,data,config);
  }
  context="order "+std::to_string(Order)+" constrained and zero-round trainer";
  const auto data=fixture(7); config.max_depth=3; config.max_bins=17; config.tree_execution=TreeExecution::stream;
  config.output_tile_size=7; config.rounds=0;
  const auto empty=train(data,config); memory<Order>(empty,data,config); validate_model_result(empty,data,config);
  std::size_t bins{}; for(const auto& f:empty.model.features) bins+=f.bins();
  config.rounds=1; config.max_histogram_bytes=empty.root_count_bytes+2*4*bins*sizeof(HigherStats<Order>);
  const auto limited=train(data,config); require(limited.tree_batch_size==2,"histogram budget reduces width before frontier");
  memory<Order>(limited,data,config); validate_model_result(limited,data,config);
  --config.max_histogram_bytes;
  const auto narrower=train(data,config); require(narrower.tree_batch_size==1,"one-byte histogram boundary");
  memory<Order>(narrower,data,config); validate_model_result(narrower,data,config);
}
void rejections() {
  context="unsupported higher-order training configurations";
  const auto data=fixture(1);
  TrainConfig base; base.objective=Objective::binary_logistic; base.optimization_order=3; base.tree_build=TreeBuildPolicy::output_batch;
  base.histogram=HistogramPolicy::global; base.max_leaf_value=1; base.rounds=0;
  for(unsigned variation=0;variation<8;++variation) {
    auto config=base;
    switch(variation) {
      case 0:config.optimization_order=1;break;
      case 1:config.optimization_order=5;break;
      case 2:config.objective=Objective::squared_error;break;
      case 3:config.objective=Objective::multiclass_softmax;break;
      case 4:config.tree_build=TreeBuildPolicy::per_output;break;
      case 5:config.histogram=HistogramPolicy::shared;break;
      case 6:config.max_leaf_value=0;break;
      case 7:config.max_leaf_value=std::numeric_limits<double>::infinity();break;
    }
    bool rejected{}; try { (void)train(data,config); } catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,"explicit invalid-combination rejection "+std::to_string(variation));
  }
}
} // namespace

int main() {
  try {
    int devices{}; const auto status=cudaGetDeviceCount(&devices);
    if(status==cudaErrorNoDevice||status==cudaErrorInsufficientDriver||(status==cudaSuccess&&!devices)) {
      std::cout<<"SKIP: no CUDA GPU available\n"; return 77;
    }
    check(status); Stream stream;
    derivatives<3>(stream); derivatives<4>(stream);
    histograms<3>(stream); histograms<4>(stream);
    splits<3>(stream); splits<4>(stream);
    training<3>(); training<4>(); rejections();
    std::cout<<"Passed "<<checks<<" higher-order GPU derivative/histogram/split/selector/trainer checks; actual per-round loss increases observed: "
             <<observed_loss_increases<<".\n";
    return 0;
  } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
