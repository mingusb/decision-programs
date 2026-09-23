#include "ghb/batch_resident.cuh"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using ghb::gpu::BatchResidentView;
using ghb::gpu::OutputBatch;
using ghb::gpu::Split;
using ghb::gpu::TreeState;
std::size_t checks{};
void require(bool condition, const std::string& message) { ++checks; if (!condition) throw std::runtime_error(message); }
void check(cudaError_t error) { if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); }
struct Stream {
  cudaStream_t value{};
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamSynchronize(value); cudaStreamDestroy(value); }
};
template<class T> struct Buffer {
  T* allocation{};
  T* value{};
  std::size_t size{};
  explicit Buffer(std::size_t count) : size(count) {
    check(cudaMalloc(reinterpret_cast<void**>(&allocation), (count + 2) * sizeof(T))); value = allocation + 1;
  }
  ~Buffer() { cudaFree(allocation); }
  Buffer(const Buffer&) = delete;
  void put(const std::vector<T>& values, T guard, cudaStream_t stream) {
    require(values.size() == size, "host fixture extent");
    std::vector<T> payload;
    payload.reserve(size + 2); payload.push_back(guard);
    for (const auto& item : values) payload.push_back(item);
    payload.push_back(guard);
    check(cudaMemcpyAsync(allocation, payload.data(), payload.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    check(cudaStreamSynchronize(stream));
  }
  void fill(T value, cudaStream_t stream) { put(std::vector<T>(size, value), value, stream); }
  std::vector<T> get(cudaStream_t stream) const {
    std::vector<T> payload(size + 2);
    check(cudaMemcpyAsync(payload.data(), allocation, payload.size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream)); return payload;
  }
};
ghb::Node node_sentinel() { return {431, 43, 47, 59, 61, -0.0}; }
TreeState state_sentinel() { return {43, 47, 53, 59, 61, 67}; }
Split split_sentinel() { return {731, 733, 739, -0.0, 743.5, 751.5, 757.5}; }
bool same_double(double a, double b) { return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b); }
void equal(const ghb::Node& a, const ghb::Node& b, const std::string& where) {
  require(a.feature == b.feature && a.left == b.left && a.right == b.right && a.threshold == b.threshold &&
          a.missing_left == b.missing_left && same_double(a.value, b.value), where + " node fields");
}
void equal(const TreeState& a, const TreeState& b, const std::string& where) {
  require(a.active_nodes == b.active_nodes && a.next_active_nodes == b.next_active_nodes && a.node_count == b.node_count &&
          a.status == b.status && a.old_node_count == b.old_node_count && a.split_count == b.split_count, where + " state fields");
}
void equal(const Split& a, const Split& b, const std::string& where) {
  require(a.feature == b.feature && a.threshold == b.threshold && a.missing_left == b.missing_left &&
          same_double(a.gain, b.gain) && same_double(a.value, b.value) && same_double(a.left_value, b.left_value) &&
          same_double(a.right_value, b.right_value), where + " split fields");
}
template<class T> void equal(const T& a, const T& b, const std::string& where) { require(a == b, where); }
template<class T> void compare(const Buffer<T>& a, const Buffer<T>& b, cudaStream_t stream, const std::string& where) {
  const auto av = a.get(stream), bv = b.get(stream);
  require(av.size() == bv.size(), where + " extent");
  for (std::size_t i = 0; i < av.size(); ++i) equal(av[i], bv[i], where + " index " + std::to_string(i));
}
struct Workspace {
  unsigned outputs, rows, capacity, node_capacity, blocks;
  Buffer<int> assignments, frontier, next, left, right;
  Buffer<ghb::Node> nodes;
  Buffer<TreeState> states;
  Buffer<unsigned> active, offsets, counts;
  Workspace(unsigned b, unsigned n, unsigned c, unsigned p)
      : outputs(b), rows(n), capacity(c), node_capacity(p), blocks(1 + (c - 1) / 1024),
        assignments(std::size_t(b) * n), frontier(std::size_t(b) * c), next(std::size_t(b) * c),
        left(std::size_t(b) * c), right(std::size_t(b) * c), nodes(std::size_t(b) * p), states(b), active(b),
        offsets(std::size_t(b) * c), counts(std::size_t(b) * blocks) {}
  BatchResidentView view() {
    return {outputs, capacity, node_capacity, assignments.value, nodes.value, frontier.value, next.value,
            left.value, right.value, states.value, active.value, offsets.value, counts.value};
  }
  void reset(cudaStream_t stream) {
    assignments.fill(-811, stream); frontier.fill(-821, stream); next.fill(-823, stream);
    left.fill(-827, stream); right.fill(-829, stream); nodes.fill(node_sentinel(), stream);
    states.fill(state_sentinel(), stream); active.fill(839, stream); offsets.fill(853, stream); counts.fill(857, stream);
  }
  void compare_to(const Workspace& reference, cudaStream_t stream, const std::string& label) const {
    compare(assignments, reference.assignments, stream, label + " assignments");
    compare(frontier, reference.frontier, stream, label + " frontier");
    compare(next, reference.next, stream, label + " next frontier");
    compare(left, reference.left, stream, label + " left map");
    compare(right, reference.right, stream, label + " right map");
    compare(nodes, reference.nodes, stream, label + " nodes");
    compare(states, reference.states, stream, label + " states");
    compare(offsets, reference.offsets, stream, label + " offsets");
    compare(counts, reference.counts, stream, label + " block prefixes");
  }
  void compare_active(unsigned live, cudaStream_t stream, const std::string& label) const {
    const auto s = states.get(stream);
    const auto a = active.get(stream);
    require(a.front() == 839 && a.back() == 839, label + " active guards");
    for (unsigned i = 0; i < outputs; ++i)
      require(a[i + 1] == (i < live ? s[i + 1].active_nodes : 0), label + " active mirror/tail");
  }
};
struct Fixture {
  unsigned outputs, rows, capacity, node_capacity;
  Stream stream;
  Workspace actual, reference;
  Buffer<std::uint16_t> bins;
  Buffer<unsigned> offsets;
  Buffer<ghb::FeatureType> types;
  Buffer<Split> winners;
  Buffer<OutputBatch> selector;
  ghb::gpu::DataView data{};
  Fixture(unsigned c, unsigned b = 4, unsigned n = 67, unsigned p = 0)
      : outputs(b), rows(n), capacity(c), node_capacity(p ? p : 3 * c + 2), actual(b, n, c, node_capacity),
        reference(b, n, c, node_capacity), bins(std::size_t(n) * 2), offsets(3), types(2),
        winners(std::size_t(b) * c), selector(1) {
    std::vector<std::uint16_t> host_bins(bins.size);
    for (unsigned f = 0; f < 2; ++f) for (unsigned row = 0; row < n; ++row) host_bins[std::size_t(f) * n + row] = (row * 3 + f) % 8;
    bins.put(host_bins, std::uint16_t(991), stream.value); offsets.put({0, 8, 16}, 997, stream.value);
    types.put({ghb::FeatureType::numeric, ghb::FeatureType::categorical}, ghb::FeatureType::numeric, stream.value);
    data = {bins.value, offsets.value, types.value, n, 2, 16, 8};
  }
  void select(unsigned count, unsigned begin = 1) { selector.put({OutputBatch{begin, count, 0, count}}, OutputBatch{}, stream.value); }
  void initialize_reference(unsigned live) {
    for (unsigned output = 0; output < live; ++output)
      check(ghb::gpu::resident_initialize(rows, reference.assignments.value + std::size_t(output) * rows,
          reference.nodes.value + std::size_t(output) * node_capacity, reference.frontier.value + std::size_t(output) * capacity,
          reference.states.value + output, stream.value));
  }
  void materialize_reference(unsigned live, bool expand, double rate) {
    for (unsigned output = 0; output < live; ++output) {
      const auto base = std::size_t(output) * capacity;
      check(ghb::gpu::resident_materialize(data, winners.value + base, reference.frontier.value + base,
          reference.next.value + base, reference.left.value + base, reference.right.value + base,
          reference.nodes.value + std::size_t(output) * node_capacity, reference.states.value + output,
          capacity, node_capacity, reference.offsets.value + base, reference.counts.value + std::size_t(output) * reference.blocks,
          expand, rate, stream.value));
    }
  }
  void route_reference(unsigned live) {
    for (unsigned output = 0; output < live; ++output) {
      const auto base = std::size_t(output) * capacity;
      check(ghb::gpu::resident_route(data, reference.assignments.value + std::size_t(output) * rows, winners.value + base,
          reference.left.value + base, reference.right.value + base, reference.states.value + output, stream.value));
      check(ghb::gpu::resident_advance(reference.states.value + output, stream.value));
    }
  }
  void prepare(unsigned mode) {
    actual.reset(stream.value); reference.reset(stream.value);
    std::vector<Split> host_winners(winners.size);
    std::vector<TreeState> states(outputs);
    std::vector<int> frontier(std::size_t(outputs) * capacity), assignments(std::size_t(outputs) * rows);
    for (unsigned output = 0; output < outputs; ++output) {
      const unsigned live_nodes = output == 0 ? capacity : output == 1 ? (capacity + 1) / 2 : output == 2 ? 0 : capacity;
      states[output] = {live_nodes, 123, capacity, 0, 456, 789};
      if (mode == 2 && output == 0) states[output].active_nodes = capacity + 1;
      if (mode == 3 && output == 0) states[output].status = ghb::gpu::tree_node_overflow;
      for (unsigned node = 0; node < capacity; ++node) {
        const std::size_t index = std::size_t(output) * capacity + node;
        frontier[index] = int(node);
        host_winners[index] = {node % 3 == 1 ? -1 : int((node + output) % 2), 1 + (node + output) % 6,
                               (node + output) % 2, double(node + 1) / 8, -double(node + 1) / 16,
                               double(node + 3) / 32, -double(node + 5) / 64};
        if (mode == 1) host_winners[index].feature = -1;
        if (mode == 4 && node == 0) host_winners[index].feature = -2;
        if (mode == 5 && node == 0) host_winners[index].threshold = 8;
        if (mode == 6 && node == 0) host_winners[index].missing_left = 2;
        if (mode == 7 && node == 0) host_winners[index].gain = std::numeric_limits<double>::quiet_NaN();
        if (mode == 8 && node == 0) host_winners[index].left_value = std::numeric_limits<double>::max();
        if (mode == 9 && node == 0) host_winners[index].feature = 2;
      }
      for (unsigned row = 0; row < rows; ++row)
        assignments[std::size_t(output) * rows + row] = row % 11 == 0 ? -1 : row % 13 == 0 ? int(capacity + 5) :
            live_nodes ? int((row * 7) % live_nodes) : -1;
    }
    winners.put(host_winners, split_sentinel(), stream.value);
    for (Workspace* work : {&actual, &reference}) {
      work->states.put(states, state_sentinel(), stream.value);
      work->frontier.put(frontier, -821, stream.value);
      work->assignments.put(assignments, -811, stream.value);
    }
  }
};

void materialize_cases() {
  for (unsigned capacity : {1U, 2U, 3U, 17U, 1024U, 1025U, 2051U}) {
    Fixture f(capacity);
    for (unsigned mode = 0; mode < 10; ++mode) for (bool expand : {false, true}) {
      const unsigned live = mode % 3 == 0 ? 4 : mode % 3 == 1 ? 1 : 3;
      f.prepare(mode); f.select(live);
      const auto original_winners = f.winners.get(f.stream.value);
      const double rate = mode == 8 ? 2.0 : 0.125;
      f.materialize_reference(live, expand, rate);
      check(ghb::gpu::resident_batch_materialize(f.data, f.winners.value, f.actual.view(), f.selector.value, expand, rate, f.stream.value));
      f.actual.compare_to(f.reference, f.stream.value, "materialize C" + std::to_string(capacity) + " M" + std::to_string(mode));
      f.route_reference(live);
      check(ghb::gpu::resident_batch_route(f.data, f.winners.value, f.actual.view(), f.selector.value, f.stream.value));
      check(ghb::gpu::resident_batch_advance(f.actual.view(), f.selector.value, f.stream.value));
      f.actual.compare_to(f.reference, f.stream.value, "route/advance");
      f.actual.compare_active(live, f.stream.value, "route/advance");
      const auto retained_winners = f.winners.get(f.stream.value);
      for (std::size_t i = 0; i < original_winners.size(); ++i)
        equal(retained_winners[i], original_winners[i], "read-only winner/guard");
    }
  }
  // Insufficient node storage must report overflow without partial node writes.
  Fixture overflow(17, 4, 67, 17);
  overflow.prepare(0); overflow.select(4);
  overflow.materialize_reference(4, false, .125);
  check(ghb::gpu::resident_batch_materialize(overflow.data, overflow.winners.value, overflow.actual.view(),
      overflow.selector.value, false, .125, overflow.stream.value));
  overflow.actual.compare_to(overflow.reference, overflow.stream.value, "node overflow");
}

void graph_and_prediction() {
  Fixture f(8, 4, 259);
  constexpr unsigned prediction_outputs = 11;
  Buffer<double> predictions(std::size_t(f.rows) * prediction_outputs), reference_predictions(predictions.size);
  std::vector<Split> host_winners(f.winners.size);
  for (unsigned output = 0; output < f.outputs; ++output)
    host_winners[std::size_t(output) * f.capacity] = {int(output % 2), 3, output % 2, .75, -.25,
                                                   double(output + 1) / 8, -double(output + 3) / 16};
  f.winners.put(host_winners, split_sentinel(), f.stream.value);
  f.select(4);
  auto launch = [&] {
    check(ghb::gpu::resident_batch_initialize(f.rows, f.actual.view(), f.selector.value, f.stream.value));
    check(ghb::gpu::resident_batch_materialize(f.data, f.winners.value, f.actual.view(), f.selector.value, true, .125, f.stream.value));
    check(ghb::gpu::resident_batch_route(f.data, f.winners.value, f.actual.view(), f.selector.value, f.stream.value));
    check(ghb::gpu::resident_batch_advance(f.actual.view(), f.selector.value, f.stream.value));
    check(ghb::gpu::resident_batch_predict(f.data, f.actual.view(), f.selector.value, prediction_outputs, predictions.value, f.stream.value));
  };
  cudaGraph_t graph{}; cudaGraphExec_t executable{};
  check(cudaStreamBeginCapture(f.stream.value, cudaStreamCaptureModeGlobal)); launch();
  check(cudaStreamEndCapture(f.stream.value, &graph)); check(cudaGraphInstantiate(&executable, graph, 0));
  for (const unsigned live : {4U, 1U, 0U, 3U, 4U}) {
    f.actual.reset(f.stream.value); f.reference.reset(f.stream.value);
    const unsigned begin = live == 1 ? 9 : 2;
    f.select(live, begin);
    std::vector<double> initial(predictions.size);
    for (std::size_t i = 0; i < initial.size(); ++i) initial[i] = double(int(i % 31) - 15) / 32;
    predictions.put(initial, -987.25, f.stream.value); reference_predictions.put(initial, -987.25, f.stream.value);
    check(cudaGraphLaunch(executable, f.stream.value));
    f.initialize_reference(live); f.materialize_reference(live, true, .125); f.route_reference(live);
    for (unsigned output = 0; output < live; ++output)
      check(ghb::gpu::resident_predict(f.data, f.reference.nodes.value + std::size_t(output) * f.node_capacity,
          f.reference.states.value + output, begin + output, prediction_outputs, reference_predictions.value, f.stream.value));
    f.actual.compare_to(f.reference, f.stream.value, "graph selector live=" + std::to_string(live));
    f.actual.compare_active(live, f.stream.value, "graph active mirror");
    const auto actual = predictions.get(f.stream.value), expected = reference_predictions.get(f.stream.value);
    for (std::size_t i = 0; i < actual.size(); ++i) require(same_double(actual[i], expected[i]), "graph prediction bit equality/guards");
    // Independent one-split reference checks numeric, categorical and missing branches.
    const auto bins = f.bins.get(f.stream.value);
    for (unsigned output = 0; output < live; ++output) for (unsigned row = 0; row < f.rows; ++row) {
      const auto split = host_winners[std::size_t(output) * f.capacity];
      const auto bin = bins[1 + std::size_t(split.feature) * f.rows + row];
      const bool left = !bin ? split.missing_left != 0 : split.feature == 1 ? bin == split.threshold : bin <= split.threshold;
      const std::size_t index = std::size_t(row) * prediction_outputs + begin + output;
      const double expected_value = initial[index] + .125 * (left ? split.left_value : split.right_value);
      require(same_double(actual[index + 1], expected_value), "independent branch prediction");
    }
  }
  check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
}

void arguments_and_empty() {
  Fixture f(3);
  f.prepare(0); f.select(0);
  check(ghb::gpu::resident_batch_materialize(f.data, f.winners.value, f.actual.view(), f.selector.value, true, .125, f.stream.value));
  check(ghb::gpu::resident_batch_route(f.data, f.winners.value, f.actual.view(), f.selector.value, f.stream.value));
  check(ghb::gpu::resident_batch_advance(f.actual.view(), f.selector.value, f.stream.value));
  f.actual.compare_to(f.reference, f.stream.value, "empty selector"); f.actual.compare_active(0, f.stream.value, "empty active");
  auto view = f.actual.view();
  require(ghb::gpu::resident_batch_initialize(0, view, f.selector.value, f.stream.value) == cudaErrorInvalidValue, "zero rows rejected");
  require(ghb::gpu::resident_batch_initialize(f.rows, view, nullptr, f.stream.value) == cudaErrorInvalidValue, "null selector rejected");
  require(ghb::gpu::resident_batch_materialize(f.data, f.winners.value, view, f.selector.value, true, 0, f.stream.value) == cudaErrorInvalidValue, "zero rate rejected");
  require(ghb::gpu::resident_batch_materialize(f.data, f.winners.value, view, f.selector.value, true, std::numeric_limits<double>::infinity(), f.stream.value) == cudaErrorInvalidValue, "infinite rate rejected");
  view.frontier_capacity = unsigned(INT32_MAX) + 1U;
  require(ghb::gpu::resident_batch_advance(view, f.selector.value, f.stream.value) == cudaErrorInvalidValue, "signed frontier extent rejected");
  view = f.actual.view(); view.node_capacity = unsigned(INT32_MAX) + 1U;
  require(ghb::gpu::resident_batch_advance(view, f.selector.value, f.stream.value) == cudaErrorInvalidValue, "signed node extent rejected");
  view = f.actual.view(); view.output_capacity = 0;
  require(ghb::gpu::resident_batch_advance(view, f.selector.value, f.stream.value) == cudaErrorInvalidValue, "zero output capacity rejected");
}
} // namespace

int main() {
  int devices{};
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
  try {
    materialize_cases(); graph_and_prediction(); arguments_and_empty();
    std::cout << "batch resident checks passed: " << checks << '\n'; return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
