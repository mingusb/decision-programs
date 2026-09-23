#include "ghb/booster.hpp"
#include "ghb/prediction.cuh"

#include <cuda_runtime_api.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
std::size_t checks{};
void require(bool condition, const std::string& message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}
void exact(const std::vector<double>& actual, const std::vector<double>& expected,
           const std::string& label) {
  require(actual.size() == expected.size(), label + " extent");
  for (std::size_t i = 0; i < actual.size(); ++i)
    require(std::bit_cast<std::uint64_t>(actual[i]) == std::bit_cast<std::uint64_t>(expected[i]),
            label + " differing FP64 bits at " + std::to_string(i));
}
template<class Function> void rejected(Function&& function, const char* label) {
  bool threw = false;
  try { function(); } catch (const std::invalid_argument&) { threw = true; }
  require(threw, label);
}
ghb::Node leaf(double value) { ghb::Node node; node.value = value; return node; }
ghb::Model model(ghb::Objective objective, unsigned outputs, bool forest = true) {
  ghb::Model result;
  result.objective = objective; result.outputs = outputs;
  result.features = {{ghb::FeatureType::numeric, {-1, 0, 1}, {}},
                     {ghb::FeatureType::categorical, {}, {10, 20, 30}},
                     {ghb::FeatureType::numeric, {}, {}}};
  result.base_scores.resize(outputs);
  for (unsigned output = 0; output < outputs; ++output)
    result.base_scores[output] = output == 0 ? -0.0 : (int(output % 7) - 3) * .125;
  if (!forest) return result;
  // Deliberately interleave outputs differently in successive rounds. Values
  // expose incorrect reassociation and moving the base after the tree sum.
  constexpr double values[]{0x1p53, 1, -0x1p53, .125};
  for (unsigned round = 0; round < 4; ++round) {
    for (unsigned index = 0; index < outputs; ++index) {
      const unsigned output = (outputs - 1 - index + round) % outputs;
      if (outputs > 3 && output % 7 == 6) continue; // Outputs with no trees.
      ghb::Tree tree; tree.output = output;
      tree.nodes = {{0, 4, 1, round % 4, round % 2, 0},
                    {1, 2, 3, round % 4, (round + 1) % 2, 0},
                    leaf(values[round]), leaf(-values[round]), leaf(values[round] * .5)};
      result.trees.push_back(std::move(tree));
    }
  }
  return result;
}
ghb::Dataset data(unsigned rows) {
  ghb::Dataset result;
  result.rows = rows; result.columns = 3;
  result.feature_types = {ghb::FeatureType::numeric, ghb::FeatureType::categorical, ghb::FeatureType::numeric};
  result.values.resize(std::size_t(rows) * 3);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float numeric[]{-2, -1, std::nextafter(-1.f, 0.f), -0.f, +0.f,
                        std::nextafter(0.f, 1.f), 1, 2, nan};
  const float categorical[]{10, 20, 30, 99, nan};
  for (unsigned row = 0; row < rows; ++row) {
    result.values[std::size_t(row) * 3] = numeric[row % 9];
    result.values[std::size_t(row) * 3 + 1] = categorical[(row / 3) % 5];
    result.values[std::size_t(row) * 3 + 2] = row % 3 == 0 ? nan : float(row);
  }
  return result;
}
void compare(const ghb::Model& frozen, const ghb::Dataset& input, const std::string& label) {
  for (const bool raw : {true, false}) {
    const auto reference = frozen.predict_gpu(input, raw, ghb::PredictionPolicy::per_tree);
    for (const auto policy : {ghb::PredictionPolicy::fused_output, ghb::PredictionPolicy::fused_output_slab}) {
      const std::string policy_label = policy == ghb::PredictionPolicy::fused_output ? " fused_output" : " fused_output_slab";
      const auto fused = frozen.predict_gpu(input, raw, policy);
      exact(fused, reference, label + policy_label + (raw ? " raw" : " transformed"));
      if (raw) exact(fused, frozen.predict(input, true), label + policy_label + " CPU raw reference");
    }
  }
}

// Direct-entry contract tests only: caller-owned pre-binned data and a valid
// packed forest; no production compute or policy changes. Check stream calls,
// captured replay with changed bins/base at fixed addresses, exact ordered
// margins, guarded outputs, immutable inputs, and host argument rejection.
// Device metadata contents remain the documented caller responsibility; these
// tests do not deliberately submit malformed device descriptors or aliases.
void cuda_check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
struct DirectStream {
  cudaStream_t value{};
  DirectStream() { cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~DirectStream() { cudaStreamSynchronize(value); cudaStreamDestroy(value); }
};
template<class T> struct DirectBuffer {
  T* pointer{};
  std::size_t size{};
  explicit DirectBuffer(std::size_t count) : size(count) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)));
  }
  ~DirectBuffer() { cudaFree(pointer); }
  DirectBuffer(const DirectBuffer&) = delete;
  DirectBuffer& operator=(const DirectBuffer&) = delete;
  void put(const std::vector<T>& values, cudaStream_t stream) {
    require(values.size() == size, "direct fixture upload extent");
    cuda_check(cudaMemcpyAsync(pointer, values.data(), size * sizeof(T), cudaMemcpyHostToDevice, stream));
    cuda_check(cudaStreamSynchronize(stream));
  }
  std::vector<T> get(cudaStream_t stream) const {
    std::vector<T> result(size);
    cuda_check(cudaMemcpyAsync(result.data(), pointer, size * sizeof(T), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    return result;
  }
};
struct DirectGraph {
  cudaStream_t stream{};
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  explicit DirectGraph(cudaStream_t value) : stream(value) {}
  ~DirectGraph() {
    cudaStreamSynchronize(stream);
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
  }
};
void exact_nodes(const std::vector<ghb::Node>& actual, const std::vector<ghb::Node>& expected) {
  require(actual.size() == expected.size(), "direct immutable node extent");
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const auto& a = actual[i]; const auto& e = expected[i];
    require(a.feature == e.feature && a.left == e.left && a.right == e.right && a.threshold == e.threshold &&
            a.missing_left == e.missing_left && std::bit_cast<std::uint64_t>(a.value) == std::bit_cast<std::uint64_t>(e.value),
            "direct immutable node fields and guards");
  }
}
void direct_api_cases() {
  constexpr unsigned rows = 37, outputs = 5;
  const double tiny = std::numeric_limits<double>::denorm_min(), guard = -918273.25;
  ghb::Model frozen;
  frozen.objective = ghb::Objective::squared_error; frozen.outputs = outputs;
  frozen.features = {{ghb::FeatureType::numeric, {0}, {}}, {ghb::FeatureType::categorical, {}, {10, 20}}};
  frozen.base_scores.resize(outputs);
  // Original model order interleaves outputs. Output 0's 2^53,+1,-2^53
  // sequence distinguishes ordered addition from summing trees separately.
  frozen.trees = {{0, {leaf(0x1p53)}}, {2, {leaf(tiny)}},
    {1, {{1, 1, 2, 2, 0, 0}, leaf(-0.0), leaf(.25)}},
    {0, {{0, 1, 2, 1, 1, 0}, leaf(1), leaf(2)}},
    {2, {leaf(tiny)}}, {0, {leaf(-0x1p53)}}, {3, {leaf(-0.0)}}};
  const ghb::Node node_guard{91, 92, 93, 94, 95, guard};
  const ghb::gpu::PredictionTree descriptor_guard{918273, 837261, 726153};
  std::vector<ghb::Node> packed_nodes{node_guard};
  std::vector<ghb::gpu::PredictionTree> packed_trees{descriptor_guard};
  std::vector<std::uint64_t> packed_offsets(outputs + 3, 987654321ULL);
  for (unsigned output = 0; output < outputs; ++output) {
    packed_offsets[1 + output] = packed_trees.size() - 1;
    for (const auto& tree : frozen.trees) if (tree.output == output) {
      packed_trees.push_back({packed_nodes.size() - 1, std::uint32_t(tree.nodes.size()), 0});
      packed_nodes.insert(packed_nodes.end(), tree.nodes.begin(), tree.nodes.end());
    }
  }
  packed_offsets[1 + outputs] = packed_trees.size() - 1;
  packed_nodes.push_back(node_guard); packed_trees.push_back(descriptor_guard);
  const auto forest_offsets = packed_offsets;
  std::vector<std::uint32_t> feature_offsets{987654321U, 0, 3, 6, 987654321U};
  std::vector<ghb::FeatureType> feature_types{ghb::FeatureType(91), ghb::FeatureType::numeric,
                                           ghb::FeatureType::categorical, ghb::FeatureType(92)};
  DirectStream stream;
  DirectBuffer<std::uint16_t> bins(rows * 2 + 2);
  DirectBuffer<std::uint32_t> offsets(feature_offsets.size());
  DirectBuffer<ghb::FeatureType> types(feature_types.size());
  DirectBuffer<ghb::Node> nodes(packed_nodes.size());
  DirectBuffer<ghb::gpu::PredictionTree> trees(packed_trees.size());
  DirectBuffer<std::uint64_t> output_offsets(packed_offsets.size());
  DirectBuffer<double> base(outputs + 2), prediction(rows * outputs + 2);
  offsets.put(feature_offsets, stream.value); types.put(feature_types, stream.value);
  nodes.put(packed_nodes, stream.value); trees.put(packed_trees, stream.value);
  output_offsets.put(packed_offsets, stream.value);
  struct Call {
    ghb::gpu::DataView data;
    const ghb::Node* nodes; std::uint64_t node_count;
    const ghb::gpu::PredictionTree* trees; std::uint64_t tree_count;
    const std::uint64_t* offsets; const double* base;
    unsigned outputs; double* prediction;
  };
  const Call valid{{bins.pointer + 1, offsets.pointer + 1, types.pointer + 1, rows, 2, 6, 3},
    nodes.pointer + 1, packed_nodes.size() - 2, trees.pointer + 1, packed_trees.size() - 2,
    output_offsets.pointer + 1, base.pointer + 1, outputs, prediction.pointer + 1};
  auto invoke = [&](const Call& c) {
    return ghb::gpu::predict_forest(c.data, c.nodes, c.node_count, c.trees, c.tree_count,
                                  c.offsets, c.base, c.outputs, c.prediction, stream.value);
  };
  ghb::Dataset input;
  input.rows = rows; input.columns = 2; input.outputs = outputs;
  input.feature_types = {ghb::FeatureType::numeric, ghb::FeatureType::categorical};
  input.values.resize(rows * 2);
  std::vector<std::uint16_t> host_bins(rows * 2 + 2, 0xfdb9);
  std::vector<double> host_base(outputs + 2, guard);
  auto prepare = [&](unsigned phase) {
    const double bases[][outputs]{{0, -0.0, 0, -0.0, -0.0}, {-1, .125, tiny, +0.0, .5}, {2, -.125, 2 * tiny, -0.0, -.5}};
    frozen.base_scores.assign(bases[phase], bases[phase] + outputs);
    for (unsigned output = 0; output < outputs; ++output) host_base[1 + output] = bases[phase][output];
    for (unsigned row = 0; row < rows; ++row) {
      const unsigned numeric = (row + phase) % 3, categorical = (row / 2 + phase) % 3;
      host_bins[1 + row] = std::uint16_t(numeric);
      host_bins[1 + rows + row] = std::uint16_t(categorical);
      input.values[row * 2] = numeric ? float(numeric - 1) : std::numeric_limits<float>::quiet_NaN();
      input.values[row * 2 + 1] = categorical ? float(categorical * 10) :
        row % 2 ? 99.f : std::numeric_limits<float>::quiet_NaN();
    }
    bins.put(host_bins, stream.value); base.put(host_base, stream.value);
    prediction.put(std::vector<double>(rows * outputs + 2, guard), stream.value);
  };
  auto immutable = [&] {
    require(bins.get(stream.value) == host_bins, "direct immutable bins and guards");
    require(offsets.get(stream.value) == feature_offsets, "direct immutable feature offsets and guards");
    require(types.get(stream.value) == feature_types, "direct immutable feature types and guards");
    exact(base.get(stream.value), host_base, "direct immutable base and guards");
    require(output_offsets.get(stream.value) == packed_offsets, "direct immutable output offsets and guards");
    exact_nodes(nodes.get(stream.value), packed_nodes);
    const auto actual = trees.get(stream.value);
    for (std::size_t i = 0; i < actual.size(); ++i)
      require(actual[i].node_begin == packed_trees[i].node_begin && actual[i].node_count == packed_trees[i].node_count &&
              actual[i].reserved == packed_trees[i].reserved, "direct immutable descriptors and guards");
  };
  auto compare_direct = [&](unsigned phase) {
    const auto result = prediction.get(stream.value);
    std::vector<double> expected(rows * outputs + 2, guard);
    const auto cpu = frozen.predict(input, true); // independent raw-feature traversal and encoding
    for (std::size_t i = 0; i < cpu.size(); ++i) expected[1 + i] = cpu[i];
    exact(result, expected, "direct ordered margins and output canaries");
    for (unsigned row = 0; row < rows; ++row) {
      require(std::bit_cast<std::uint64_t>(result[1 + row * outputs + 2]) == phase + 2,
              "direct exact subnormal accumulation");
      require(std::bit_cast<std::uint64_t>(result[1 + row * outputs + 3]) == (phase == 1 ? 0ULL : 1ULL << 63),
              "direct signed-zero leaf addition");
      if (phase == 0) require(result[1 + row * outputs] == (host_bins[1 + row] == 2 ? 2.0 : 0.0),
                              "direct independent cancellation reference");
    }
    immutable();
  };
  prepare(0); cuda_check(invoke(valid)); compare_direct(0);
  {
    DirectGraph graph(stream.value);
    cuda_check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
    cuda_check(invoke(valid));
    cuda_check(cudaStreamEndCapture(stream.value, &graph.graph));
    cuda_check(cudaGraphInstantiate(&graph.executable, graph.graph, 0));
    for (unsigned phase : {1u, 2u, 0u}) {
      prepare(phase); cuda_check(cudaGraphLaunch(graph.executable, stream.value)); compare_direct(phase);
    }
  }
  // Empty forest accepts null storage but still copies each base score exactly.
  auto empty = valid; empty.nodes = nullptr; empty.node_count = 0; empty.trees = nullptr; empty.tree_count = 0;
  for (unsigned output = 0; output <= outputs; ++output) packed_offsets[1 + output] = 0;
  output_offsets.put(packed_offsets, stream.value); prepare(1); cuda_check(invoke(empty));
  std::vector<double> expected(rows * outputs + 2, guard);
  for (unsigned row = 0; row < rows; ++row) for (unsigned output = 0; output < outputs; ++output)
    expected[1 + row * outputs + output] = host_base[1 + output];
  exact(prediction.get(stream.value), expected, "direct empty forest base and guards"); immutable();
  packed_offsets = forest_offsets;
  output_offsets.put(packed_offsets, stream.value);

  // Every malformed host call must reject before launching. The small live
  // buffers below are deliberately reused with oversized extents; no enormous
  // allocation or malformed device-metadata execution is required.
  auto reject = [&](auto change, const char* label) {
    auto invalid = valid; change(invalid);
    require(invoke(invalid) == cudaErrorInvalidValue, label);
  };
  reject([](auto& c) { c.data.rows = 0; }, "direct zero rows rejected");
  reject([](auto& c) { c.data.columns = 0; }, "direct zero features rejected");
  reject([](auto& c) { c.data.columns = std::uint32_t(INT32_MAX) + 1; }, "direct signed feature overflow rejected");
  reject([](auto& c) { c.data.bins = nullptr; }, "direct null bins rejected");
  reject([](auto& c) { c.data.offsets = nullptr; }, "direct null feature offsets rejected");
  reject([](auto& c) { c.data.types = nullptr; }, "direct null feature types rejected");
  reject([](auto& c) { c.data.total_bins = 1; }, "direct inconsistent total bins rejected");
  reject([](auto& c) { c.data.max_feature_bins = 0; }, "direct empty max bins rejected");
  reject([](auto& c) { c.data.max_feature_bins = 65537; }, "direct packed bin overflow rejected");
  reject([](auto& c) { c.data.max_feature_bins = 7; }, "direct inconsistent max bins rejected");
  reject([](auto& c) { c.outputs = 0; }, "direct zero outputs rejected");
  reject([](auto& c) { c.offsets = nullptr; }, "direct null output offsets rejected");
  reject([](auto& c) { c.base = nullptr; }, "direct null base rejected");
  reject([](auto& c) { c.prediction = nullptr; }, "direct null predictions rejected");
  reject([](auto& c) { c.nodes = nullptr; }, "direct nonempty null nodes rejected");
  reject([](auto& c) { c.trees = nullptr; }, "direct nonempty null trees rejected");
  reject([](auto& c) { c.node_count = 0; }, "direct nonempty missing node count rejected");
  reject([](auto& c) { c.tree_count = 0; }, "direct empty forest with nodes rejected");
  reject([](auto& c) { c.node_count = std::numeric_limits<std::size_t>::max() / sizeof(ghb::Node) + 1; },
         "direct node byte overflow rejected");
  reject([](auto& c) { c.tree_count = std::numeric_limits<std::size_t>::max() / sizeof(ghb::gpu::PredictionTree) + 1; },
         "direct descriptor byte overflow rejected");
  reject([](auto& c) { c.data.rows = UINT32_MAX; c.outputs = UINT32_MAX; }, "direct prediction byte overflow rejected");
  exact(prediction.get(stream.value), expected, "direct host rejection leaves outputs untouched"); immutable();
}
} // namespace

int main() {
  try {
    int devices{};
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
      std::cout << "SKIP: CUDA device unavailable\n"; return 77;
    }
    for (const auto objective : {ghb::Objective::squared_error, ghb::Objective::binary_logistic,
                                 ghb::Objective::multiclass_softmax}) {
      for (const unsigned outputs : {1U, 3U, 33U, 65U}) {
        if (objective == ghb::Objective::multiclass_softmax && outputs == 1) continue;
        compare(model(objective, outputs), data(257), "mixed feature forest");
        compare(model(objective, outputs, false), data(33), "zero tree forest");
      }
    }
    const auto tail_model = model(ghb::Objective::binary_logistic, 3);
    for (const unsigned rows : {0U, 1U, 31U, 32U, 255U, 256U})
      compare(tail_model, data(rows), "row tail");

    auto zeros = model(ghb::Objective::squared_error, 4, false);
    zeros.base_scores = {-0.0, -0.0, +0.0, std::numeric_limits<double>::denorm_min()};
    zeros.trees = {{0, {leaf(-0.0)}}, {1, {leaf(+0.0)}}, {2, {leaf(-0.0)}},
                  {3, {leaf(std::numeric_limits<double>::denorm_min())}}};
    compare(zeros, data(1), "signed zero and subnormal");
    for (const auto policy : {ghb::PredictionPolicy::fused_output, ghb::PredictionPolicy::fused_output_slab}) {
      const auto zero_values = zeros.predict_gpu(data(1), true, policy);
      require(std::bit_cast<std::uint64_t>(zero_values[0]) == (1ULL << 63), "negative zero retained");
      require(std::bit_cast<std::uint64_t>(zero_values[1]) == 0, "positive zero addition retained");
      require(std::bit_cast<std::uint64_t>(zero_values[3]) == 2, "subnormal addition retained");
    }

    auto wide = model(ghb::Objective::multiclass_softmax, 1024, false);
    wide.trees = {{1023, {leaf(.1)}}, {0, {leaf(-.125)}}, {1023, {leaf(.2)}}};
    compare(wide, data(33), "wide sparse output forest");
    const auto input = data(1);
    rejected([&] { tail_model.predict_gpu(input, true, ghb::PredictionPolicy(99)); }, "invalid policy accepted");
    rejected([&] { tail_model.predict_gpu(input, true, ghb::PredictionPolicy(3)); }, "policy after slab accepted");
    auto bad_model = tail_model; bad_model.trees[0].nodes[0].left = 999;
    for (const auto policy : {ghb::PredictionPolicy::per_tree, ghb::PredictionPolicy::fused_output,
                              ghb::PredictionPolicy::fused_output_slab}) {
      rejected([&] { bad_model.predict_gpu(input, true, policy); }, "invalid tree accepted");
      auto infinite = input; infinite.values[0] = std::numeric_limits<float>::infinity();
      rejected([&] { tail_model.predict_gpu(infinite, true, policy); }, "infinite feature accepted");
    }
    direct_api_cases();
    std::cout << "prediction exact checks passed: " << checks << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "prediction test failed: " << error.what() << '\n'; return 1;
  }
}
