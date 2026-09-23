#include "ghb/resident.cuh"
#include <algorithm>
#include <cmath>
#include <limits>

namespace ghb::gpu {
namespace {
constexpr unsigned kThreads = 256, kTile = 1024;
using Index = unsigned long long;
unsigned grid(Index n) { return unsigned(std::min<Index>((n + kThreads - 1) / kThreads, 65535)); }

__global__ void initialize(unsigned rows, int* assignments, Node* nodes, int* frontier, TreeState* state) {
  for (Index row = Index(blockIdx.x) * blockDim.x + threadIdx.x; row < rows; row += Index(gridDim.x) * blockDim.x)
    assignments[row] = 0;
  if (!blockIdx.x && !threadIdx.x) { *state = TreeState{1, 0, 1, 0, 0, 0}; nodes[0] = Node{}; frontier[0] = 0; }
}

__global__ void scan_splits(DataView data, const Split* winners, TreeState* state,
                            unsigned capacity, unsigned width, unsigned* offsets,
                            unsigned* block_counts, double rate) {
  __shared__ unsigned scan[kTile];
  const Index base = Index(blockIdx.x) * kTile;
  const unsigned active = state->active_nodes;
  for (unsigned local = threadIdx.x; local < width; local += blockDim.x) {
    const Index index = base + local;
    unsigned flag{};
    if (index < capacity && index < active) {
      const Split split = winners[index];
      if (!isfinite(split.value) || !isfinite(split.left_value) || !isfinite(split.right_value) || !isfinite(split.gain) ||
          !isfinite(rate * split.value) || !isfinite(rate * split.left_value) || !isfinite(rate * split.right_value)) {
        atomicOr(&state->status, unsigned(tree_nonfinite_leaf));
      } else if (split.feature < -1 || split.missing_left > 1 ||
                 (split.feature >= 0 && (unsigned(split.feature) >= data.columns ||
                   split.threshold >= data.offsets[split.feature + 1] - data.offsets[split.feature]))) {
        atomicOr(&state->status, unsigned(tree_invalid_split));
      } else flag = split.feature >= 0;
    }
    scan[local] = flag;
  }
  __syncthreads();
  for (unsigned stride = 1; stride < width; stride <<= 1) {
    for (unsigned index = (threadIdx.x + 1) * stride * 2 - 1; index < width; index += blockDim.x * stride * 2)
      scan[index] += scan[index - stride];
    __syncthreads();
  }
  if (!threadIdx.x) { block_counts[blockIdx.x] = scan[width - 1]; scan[width - 1] = 0; }
  __syncthreads();
  for (unsigned stride = width / 2; stride; stride >>= 1) {
    for (unsigned index = (threadIdx.x + 1) * stride * 2 - 1; index < width; index += blockDim.x * stride * 2) {
      const unsigned value = scan[index - stride]; scan[index - stride] = scan[index]; scan[index] += value;
    }
    __syncthreads();
  }
  for (unsigned local = threadIdx.x; local < width; local += blockDim.x)
    if (base + local < capacity) offsets[base + local] = scan[local];
}
__global__ void prefix_blocks(TreeState* state, unsigned capacity, unsigned node_capacity,
                              unsigned* counts, unsigned blocks, bool expand) {
  if (threadIdx.x) return;
  unsigned total{};
  for (unsigned i = 0; i < blocks; ++i) { const unsigned value = counts[i]; counts[i] = total; total += value; }
  if (state->active_nodes > capacity) state->status |= tree_frontier_overflow;
  if (Index(state->node_count) + Index(total) * 2 > node_capacity) state->status |= tree_node_overflow;
  if (expand && Index(total) * 2 > capacity) state->status |= tree_frontier_overflow;
  state->old_node_count = state->node_count; state->split_count = total;
  state->next_active_nodes = !state->status && expand ? total * 2 : 0;
  if (!state->status) state->node_count += total * 2;
}
__global__ void materialize(const Split* winners, const int* frontier, int* next_frontier,
                            int* left_map, int* right_map, Node* nodes, const TreeState* state,
                            unsigned capacity, const unsigned* offsets, const unsigned* counts,
                            bool expand, double rate) {
  if (state->status) return;
  for (Index index = Index(blockIdx.x) * blockDim.x + threadIdx.x;
       index < state->active_nodes && index < capacity; index += Index(gridDim.x) * blockDim.x) {
    const Split split = winners[index];
    left_map[index] = right_map[index] = -1;
    if (split.feature < 0) {
      Node leaf; leaf.value = rate * split.value; nodes[frontier[index]] = leaf; continue;
    }
    const unsigned ordinal = counts[index / kTile] + offsets[index];
    const unsigned left = state->old_node_count + ordinal * 2, right = left + 1;
    nodes[frontier[index]] = Node{split.feature, int(left), int(right), split.threshold, split.missing_left, 0};
    Node left_node, right_node; left_node.value = rate * split.left_value; right_node.value = rate * split.right_value;
    nodes[left] = left_node; nodes[right] = right_node;
    if (expand) {
      left_map[index] = int(ordinal * 2); right_map[index] = int(ordinal * 2 + 1);
      next_frontier[ordinal * 2] = int(left); next_frontier[ordinal * 2 + 1] = int(right);
    }
  }
}
__global__ void route_rows(DataView data, int* assignments, const Split* winners,
                           const int* left_map, const int* right_map, const TreeState* state) {
  if (state->status || !state->next_active_nodes) return;
  for (Index row = Index(blockIdx.x) * blockDim.x + threadIdx.x; row < data.rows; row += Index(gridDim.x) * blockDim.x) {
    const int node = assignments[row];
    if (node < 0 || unsigned(node) >= state->active_nodes) continue;
    const Split split = winners[node];
    if (split.feature < 0) { assignments[row] = -1; continue; }
    const auto bin = data.bins[Index(split.feature) * data.rows + row];
    const bool left = !bin ? split.missing_left != 0 : data.types[split.feature] == FeatureType::categorical ? bin == split.threshold : bin <= split.threshold;
    assignments[row] = left ? left_map[node] : right_map[node];
  }
}
__global__ void advance(TreeState* state) { if (!threadIdx.x) state->active_nodes = state->status ? 0 : state->next_active_nodes; }
__global__ void predict_tree(DataView data, const Node* nodes, const TreeState* state,
                             unsigned output, unsigned outputs, double* predictions, const TreeParameters* selector) {
  if (selector) output = selector->output;
  if (output >= outputs) return;
  if (state->status) return;
  for (Index row = Index(blockIdx.x) * blockDim.x + threadIdx.x; row < data.rows; row += Index(gridDim.x) * blockDim.x) {
    unsigned current{}, visited{};
    while (current < state->node_count && visited++ < state->node_count) {
      const Node node = nodes[current];
      if (node.feature < 0) { predictions[row * outputs + output] += node.value; break; }
      const auto bin = data.bins[Index(node.feature) * data.rows + row];
      const bool left = !bin ? node.missing_left != 0 : data.types[node.feature] == FeatureType::categorical ? bin == node.threshold : bin <= node.threshold;
      current = unsigned(left ? node.left : node.right);
    }
  }
}
__global__ void reduce_loss(const double* partials, unsigned blocks, const double* weight,
                            unsigned outputs, double* loss) {
  __shared__ double values[kThreads];
  double total{};
  for (unsigned i = threadIdx.x; i < blocks; i += blockDim.x) total += partials[i];
  values[threadIdx.x] = total; __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride) values[threadIdx.x] += values[threadIdx.x + stride];
    __syncthreads();
  }
  if (!threadIdx.x) *loss = values[0] / (*weight * outputs);
}
} // namespace
cudaError_t resident_initialize(unsigned rows, int* assignments, Node* nodes, int* frontier, TreeState* state, cudaStream_t stream) {
  if (!rows || !assignments || !nodes || !frontier || !state) return cudaErrorInvalidValue;
  initialize<<<grid(rows), kThreads, 0, stream>>>(rows, assignments, nodes, frontier, state); return cudaGetLastError();
}
cudaError_t resident_materialize(DataView data, const Split* winners, const int* frontier, int* next_frontier,
                                 int* left_map, int* right_map, Node* nodes, TreeState* state, unsigned capacity,
                                 unsigned node_capacity, unsigned* offsets, unsigned* counts, bool expand,
                                 double rate, cudaStream_t stream) {
  if (!capacity || capacity > unsigned(INT32_MAX) || !node_capacity || node_capacity > unsigned(INT32_MAX) ||
      !winners || !frontier || !next_frontier || !left_map || !right_map || !nodes || !state || !offsets || !counts ||
      !data.offsets || !data.columns || !std::isfinite(rate) || rate <= 0) return cudaErrorInvalidValue;
  unsigned width = 1; while (width < std::min(capacity, kTile)) width <<= 1;
  const unsigned blocks = 1 + (capacity - 1) / kTile;
  scan_splits<<<blocks, std::min(kThreads, width), 0, stream>>>(data, winners, state, capacity, width, offsets, counts, rate);
  auto error = cudaGetLastError(); if (error != cudaSuccess) return error;
  prefix_blocks<<<1, 1, 0, stream>>>(state, capacity, node_capacity, counts, blocks, expand);
  error = cudaGetLastError(); if (error != cudaSuccess) return error;
  materialize<<<grid(capacity), kThreads, 0, stream>>>(winners, frontier, next_frontier, left_map, right_map, nodes, state, capacity, offsets, counts, expand, rate);
  return cudaGetLastError();
}
cudaError_t resident_route(DataView data, int* assignments, const Split* winners, const int* left, const int* right, TreeState* state, cudaStream_t stream) {
  if (!data.rows || !data.bins || !data.types || !assignments || !winners || !left || !right || !state) return cudaErrorInvalidValue;
  route_rows<<<grid(data.rows), kThreads, 0, stream>>>(data, assignments, winners, left, right, state); return cudaGetLastError();
}
cudaError_t resident_advance(TreeState* state, cudaStream_t stream) {
  if (!state) return cudaErrorInvalidValue;
  advance<<<1, 1, 0, stream>>>(state); return cudaGetLastError();
}
cudaError_t resident_predict(DataView data, const Node* nodes, const TreeState* state, unsigned output, unsigned outputs, double* predictions, cudaStream_t stream, const TreeParameters* selector) {
  if (!data.rows || !data.bins || !data.types || !nodes || !state || !outputs || output >= outputs || !predictions) return cudaErrorInvalidValue;
  predict_tree<<<grid(data.rows), kThreads, 0, stream>>>(data, nodes, state, output, outputs, predictions, selector); return cudaGetLastError();
}
cudaError_t finalize_loss(const double* partials, unsigned blocks, const double* weight, unsigned outputs, double* loss, cudaStream_t stream) {
  if (!partials || !blocks || !weight || !outputs || !loss) return cudaErrorInvalidValue;
  reduce_loss<<<1, kThreads, 0, stream>>>(partials, blocks, weight, outputs, loss); return cudaGetLastError();
}
} // namespace ghb::gpu
