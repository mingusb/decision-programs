#include "ghb/prediction.cuh"

#include <algorithm>
#include <climits>
#include <limits>

namespace ghb::gpu {
namespace {
using Index = unsigned long long;
constexpr unsigned threads = 256;

__global__ void ordered_forest(DataView data, const Node* nodes,
                               const PredictionTree* trees,
                               const std::uint64_t* output_offsets,
                               const double* base, unsigned outputs,
                               double* predictions) {
  const Index size = Index(data.rows) * outputs;
  // Output-major work keeps adjacent lanes on adjacent feature-major rows.
  for (Index task = Index(blockIdx.x) * blockDim.x + threadIdx.x; task < size;
       task += Index(gridDim.x) * blockDim.x) {
    const unsigned output = task / data.rows;
    const unsigned row = task % data.rows;
    double margin = base[output];
    for (std::uint64_t tree = output_offsets[output]; tree < output_offsets[output + 1]; ++tree) {
      const PredictionTree descriptor = trees[tree];
      const Node* tree_nodes = nodes + descriptor.node_begin;
      int current = 0;
      for (unsigned step = 0; step < descriptor.node_count && current >= 0 &&
           unsigned(current) < descriptor.node_count; ++step) {
        const Node node = tree_nodes[current];
        if (node.feature < 0) {
          // Retain every addition, including +/-0, in the reference order.
          margin = __dadd_rn(margin, node.value);
          break;
        }
        if (unsigned(node.feature) >= data.columns) break;
        const unsigned bin = data.bins[Index(node.feature) * data.rows + row];
        const bool left = !bin ? node.missing_left != 0 :
            data.types[node.feature] == FeatureType::numeric ? bin <= node.threshold : bin == node.threshold;
        current = left ? node.left : node.right;
      }
    }
    predictions[Index(row) * outputs + output] = margin;
  }
}
} // namespace

cudaError_t predict_forest(DataView data, const Node* nodes,
                           std::uint64_t node_count, const PredictionTree* trees,
                           std::uint64_t tree_count,
                           const std::uint64_t* output_offsets,
                           const double* base, unsigned outputs,
                           double* predictions, cudaStream_t stream) {
  if (!data.rows || !data.columns || data.columns > unsigned(INT32_MAX) ||
      !data.bins || !data.offsets || !data.types || data.total_bins < data.columns ||
      !data.max_feature_bins || data.max_feature_bins > 65536 ||
      data.max_feature_bins > data.total_bins || !outputs ||
      !output_offsets || !base || !predictions ||
      (tree_count && (!trees || !nodes || !node_count)) || (!tree_count && node_count) ||
      node_count > std::numeric_limits<std::size_t>::max() / sizeof(Node) ||
      tree_count > std::numeric_limits<std::size_t>::max() / sizeof(PredictionTree) ||
      Index(outputs) + 1 > std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t) ||
      Index(data.rows) * outputs > std::numeric_limits<std::size_t>::max() / sizeof(double))
    return cudaErrorInvalidValue;
  const auto grid = unsigned(std::min<Index>((Index(data.rows) * outputs + threads - 1) / threads, 65535));
  ordered_forest<<<grid, threads, 0, stream>>>(data, nodes, trees, output_offsets, base, outputs, predictions);
  return cudaGetLastError();
}

} // namespace ghb::gpu
