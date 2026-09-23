#include "ghb/batch_resident.cuh"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ghb::gpu {
namespace {
constexpr unsigned kThreads = 256, kTile = 1024;
using Index = unsigned long long;
unsigned grid(Index count) { return unsigned(std::min<Index>((count + kThreads - 1) / kThreads, 65535)); }
unsigned outputs_grid(unsigned count) { return std::min(count, 65535U); }
__device__ unsigned live_count(BatchResidentView view, const OutputBatch* batch) {
  return min(view.output_capacity, batch->output_count);
}
bool extent(unsigned first, unsigned second, std::size_t bytes) {
  return Index(first) * second <= std::numeric_limits<std::size_t>::max() / bytes;
}
bool shape(BatchResidentView v) {
  return v.output_capacity && v.frontier_capacity && v.frontier_capacity <= unsigned(INT32_MAX) &&
         v.node_capacity && v.node_capacity <= unsigned(INT32_MAX) &&
         extent(v.output_capacity, v.frontier_capacity, sizeof(Split)) &&
         extent(v.output_capacity, v.node_capacity, sizeof(Node));
}

__global__ void initialize_batch(unsigned rows, BatchResidentView v, const OutputBatch* batch) {
  const unsigned live = live_count(v, batch);
  for (Index output = blockIdx.y; output < v.output_capacity; output += gridDim.y) {
    if (output >= live) {
      if (!blockIdx.x && !threadIdx.x) v.active[output] = 0;
      continue;
    }
    for (Index row = Index(blockIdx.x) * blockDim.x + threadIdx.x; row < rows;
         row += Index(gridDim.x) * blockDim.x) v.assignments[Index(output) * rows + row] = 0;
    if (!blockIdx.x && !threadIdx.x) {
      v.states[output] = TreeState{1, 0, 1, 0, 0, 0};
      v.active[output] = 1;
      v.nodes[Index(output) * v.node_capacity] = Node{};
      v.frontier[Index(output) * v.frontier_capacity] = 0;
    }
  }
}

__device__ unsigned split_flag(DataView data, const Split split, TreeState* state, double rate) {
  if (!isfinite(split.value) || !isfinite(split.left_value) || !isfinite(split.right_value) || !isfinite(split.gain) ||
      !isfinite(rate * split.value) || !isfinite(rate * split.left_value) || !isfinite(rate * split.right_value)) {
    atomicOr(&state->status, unsigned(tree_nonfinite_leaf));
  } else if (split.feature < -1 || split.missing_left > 1 ||
             (split.feature >= 0 && (unsigned(split.feature) >= data.columns ||
               split.threshold >= data.offsets[split.feature + 1] - data.offsets[split.feature]))) {
    atomicOr(&state->status, unsigned(tree_invalid_split));
  } else return split.feature >= 0;
  return 0;
}
// All participating threads execute every barrier, including empty frontiers.
__device__ unsigned exclusive_scan(unsigned* scan, unsigned width) {
  __syncthreads();
  for (unsigned stride = 1; stride < width; stride <<= 1) {
    for (unsigned index = (threadIdx.x + 1) * stride * 2 - 1; index < width; index += blockDim.x * stride * 2)
      scan[index] += scan[index - stride];
    __syncthreads();
  }
  const unsigned total = scan[width - 1];
  // Every thread must finish reading the total before thread zero clears it.
  __syncthreads();
  if (!threadIdx.x) scan[width - 1] = 0;
  __syncthreads();
  for (unsigned stride = width / 2; stride; stride >>= 1) {
    for (unsigned index = (threadIdx.x + 1) * stride * 2 - 1; index < width; index += blockDim.x * stride * 2) {
      const unsigned value = scan[index - stride]; scan[index - stride] = scan[index]; scan[index] += value;
    }
    __syncthreads();
  }
  return total;
}
__device__ void finish_state(TreeState* state, unsigned capacity, unsigned node_capacity,
                              unsigned total, bool expand) {
  if (state->active_nodes > capacity) state->status |= tree_frontier_overflow;
  if (Index(state->node_count) + Index(total) * 2 > node_capacity) state->status |= tree_node_overflow;
  if (expand && Index(total) * 2 > capacity) state->status |= tree_frontier_overflow;
  state->old_node_count = state->node_count; state->split_count = total;
  state->next_active_nodes = !state->status && expand ? total * 2 : 0;
  if (!state->status) state->node_count += total * 2;
}
__device__ void write_node(const Split split, unsigned index, unsigned ordinal,
                            unsigned output, BatchResidentView v, bool expand, double rate) {
  const Index node_base = Index(output) * v.node_capacity;
  const Index slot = Index(output) * v.frontier_capacity + index;
  const TreeState state = v.states[output];
  v.left_map[slot] = v.right_map[slot] = -1;
  if (split.feature < 0) {
    Node leaf; leaf.value = rate * split.value; v.nodes[node_base + v.frontier[slot]] = leaf; return;
  }
  const unsigned left = state.old_node_count + ordinal * 2, right = left + 1;
  v.nodes[node_base + v.frontier[slot]] = Node{split.feature, int(left), int(right), split.threshold, split.missing_left, 0};
  Node left_node, right_node;
  left_node.value = rate * split.left_value; right_node.value = rate * split.right_value;
  v.nodes[node_base + left] = left_node; v.nodes[node_base + right] = right_node;
  if (expand) {
    v.left_map[slot] = int(ordinal * 2); v.right_map[slot] = int(ordinal * 2 + 1);
    const Index next = Index(output) * v.frontier_capacity + ordinal * 2;
    v.next_frontier[next] = int(left); v.next_frontier[next + 1] = int(right);
  }
}

__global__ void materialize_small(DataView data, const Split* winners, BatchResidentView v,
                                  const OutputBatch* batch, unsigned width, bool expand, double rate) {
  __shared__ unsigned scan[kTile];
  const unsigned live = live_count(v, batch);
  for (Index output = blockIdx.x; output < live; output += gridDim.x) {
    TreeState* state = v.states + output;
    const Index base = Index(output) * v.frontier_capacity;
    for (unsigned local = threadIdx.x; local < width; local += blockDim.x)
      scan[local] = local < v.frontier_capacity && local < state->active_nodes
                      ? split_flag(data, winners[base + local], state, rate) : 0;
    const unsigned total = exclusive_scan(scan, width);
    for (unsigned local = threadIdx.x; local < v.frontier_capacity; local += blockDim.x)
      v.offsets[base + local] = scan[local];
    if (!threadIdx.x) {
      v.block_counts[output] = 0;
      finish_state(state, v.frontier_capacity, v.node_capacity, total, expand);
    }
    __syncthreads();
    if (!state->status)
      for (unsigned local = threadIdx.x; local < state->active_nodes && local < v.frontier_capacity; local += blockDim.x)
        write_node(winners[base + local], local, scan[local], output, v, expand, rate);
    __syncthreads();
  }
}
__global__ void scan_batch(DataView data, const Split* winners, BatchResidentView v,
                            const OutputBatch* batch, unsigned width, unsigned blocks, double rate) {
  __shared__ unsigned scan[kTile];
  const unsigned live = live_count(v, batch);
  for (Index output = blockIdx.y; output < live; output += gridDim.y) {
    TreeState* state = v.states + output;
    const Index base = Index(output) * v.frontier_capacity;
    const Index block_base = Index(blockIdx.x) * kTile;
    for (unsigned local = threadIdx.x; local < width; local += blockDim.x) {
      const Index index = block_base + local;
      scan[local] = index < v.frontier_capacity && index < state->active_nodes
                      ? split_flag(data, winners[base + index], state, rate) : 0;
    }
    const unsigned total = exclusive_scan(scan, width);
    if (!threadIdx.x) v.block_counts[Index(output) * blocks + blockIdx.x] = total;
    for (unsigned local = threadIdx.x; local < width; local += blockDim.x)
      if (block_base + local < v.frontier_capacity) v.offsets[base + block_base + local] = scan[local];
    __syncthreads();
  }
}
__global__ void prefix_batch(BatchResidentView v, const OutputBatch* batch, unsigned blocks, bool expand) {
  const unsigned live = live_count(v, batch);
  for (Index output = Index(blockIdx.x) * blockDim.x + threadIdx.x; output < live; output += Index(gridDim.x) * blockDim.x) {
    unsigned total{};
    unsigned* counts = v.block_counts + Index(output) * blocks;
    for (unsigned i = 0; i < blocks; ++i) { const unsigned value = counts[i]; counts[i] = total; total += value; }
    finish_state(v.states + output, v.frontier_capacity, v.node_capacity, total, expand);
  }
}
__global__ void materialize_batch(const Split* winners, BatchResidentView v, const OutputBatch* batch,
                                   unsigned blocks, bool expand, double rate) {
  const unsigned live = live_count(v, batch);
  for (Index output = blockIdx.y; output < live; output += gridDim.y) {
    const TreeState state = v.states[output];
    if (state.status) continue;
    const Index base = Index(output) * v.frontier_capacity;
    for (Index index = Index(blockIdx.x) * blockDim.x + threadIdx.x;
         index < state.active_nodes && index < v.frontier_capacity; index += Index(gridDim.x) * blockDim.x) {
      const unsigned ordinal = v.block_counts[Index(output) * blocks + index / kTile] + v.offsets[base + index];
      write_node(winners[base + index], unsigned(index), ordinal, output, v, expand, rate);
    }
  }
}
__global__ void route_batch(DataView data, const Split* winners, BatchResidentView v, const OutputBatch* batch) {
  const unsigned live = live_count(v, batch);
  for (Index output = blockIdx.y; output < live; output += gridDim.y) {
    const TreeState state = v.states[output];
    if (state.status || !state.next_active_nodes) continue;
    const Index base = Index(output) * v.frontier_capacity;
    int* assignments = v.assignments + Index(output) * data.rows;
    for (Index row = Index(blockIdx.x) * blockDim.x + threadIdx.x; row < data.rows; row += Index(gridDim.x) * blockDim.x) {
      const int node = assignments[row];
      if (node < 0 || unsigned(node) >= state.active_nodes || unsigned(node) >= v.frontier_capacity) continue;
      const Split split = winners[base + node];
      if (split.feature < 0) { assignments[row] = -1; continue; }
      const auto bin = data.bins[Index(split.feature) * data.rows + row];
      const bool left = !bin ? split.missing_left != 0 : data.types[split.feature] == FeatureType::categorical
                          ? bin == split.threshold : bin <= split.threshold;
      assignments[row] = left ? v.left_map[base + node] : v.right_map[base + node];
    }
  }
}
__global__ void advance_batch(BatchResidentView v, const OutputBatch* batch) {
  const unsigned live = live_count(v, batch);
  for (Index output = Index(blockIdx.x) * blockDim.x + threadIdx.x; output < v.output_capacity;
       output += Index(gridDim.x) * blockDim.x) {
    if (output >= live) { v.active[output] = 0; continue; }
    TreeState& state = v.states[output];
    state.active_nodes = state.status ? 0 : state.next_active_nodes;
    v.active[output] = state.active_nodes;
  }
}
__global__ void predict_batch(DataView data, BatchResidentView v, const OutputBatch* batch,
                               unsigned outputs, double* predictions) {
  const unsigned live = live_count(v, batch);
  for (Index output = blockIdx.y; output < live; output += gridDim.y) {
    const Index prediction_output = Index(batch->output_begin) + output;
    const TreeState state = v.states[output];
    if (prediction_output >= outputs || state.status) continue;
    const Node* nodes = v.nodes + Index(output) * v.node_capacity;
    for (Index row = Index(blockIdx.x) * blockDim.x + threadIdx.x; row < data.rows; row += Index(gridDim.x) * blockDim.x) {
      unsigned current{}, visited{};
      while (current < state.node_count && current < v.node_capacity && visited++ < state.node_count) {
        const Node node = nodes[current];
        if (node.feature < 0) { predictions[row * outputs + prediction_output] += node.value; break; }
        const auto bin = data.bins[Index(node.feature) * data.rows + row];
        const bool left = !bin ? node.missing_left != 0 : data.types[node.feature] == FeatureType::categorical
                            ? bin == node.threshold : bin <= node.threshold;
        current = unsigned(left ? node.left : node.right);
      }
    }
  }
}
} // namespace

cudaError_t resident_batch_initialize(unsigned rows, BatchResidentView v, const OutputBatch* batch, cudaStream_t stream) {
  if (!shape(v) || !rows || !extent(v.output_capacity, rows, sizeof(int)) || !batch ||
      !v.assignments || !v.nodes || !v.frontier || !v.states || !v.active) return cudaErrorInvalidValue;
  initialize_batch<<<dim3(grid(rows), outputs_grid(v.output_capacity)), kThreads, 0, stream>>>(rows, v, batch);
  return cudaGetLastError();
}
cudaError_t resident_batch_materialize(DataView data, const Split* winners, BatchResidentView v,
                                       const OutputBatch* batch, bool expand, double rate, cudaStream_t stream) {
  if (!shape(v) || !batch || !winners || !v.frontier || !v.next_frontier || !v.left_map || !v.right_map ||
      !v.nodes || !v.states || !v.offsets || !v.block_counts || !data.offsets || !data.columns ||
      !std::isfinite(rate) || rate <= 0) return cudaErrorInvalidValue;
  unsigned width = 1; while (width < std::min(v.frontier_capacity, kTile)) width <<= 1;
  const unsigned blocks = 1 + (v.frontier_capacity - 1) / kTile;
  if (v.frontier_capacity <= kTile) {
    materialize_small<<<outputs_grid(v.output_capacity), std::min(kThreads, width), 0, stream>>>(data, winners, v, batch, width, expand, rate);
    return cudaGetLastError();
  }
  scan_batch<<<dim3(blocks, outputs_grid(v.output_capacity)), kThreads, 0, stream>>>(data, winners, v, batch, width, blocks, rate);
  auto error = cudaGetLastError(); if (error != cudaSuccess) return error;
  prefix_batch<<<grid(v.output_capacity), kThreads, 0, stream>>>(v, batch, blocks, expand);
  error = cudaGetLastError(); if (error != cudaSuccess) return error;
  materialize_batch<<<dim3(grid(v.frontier_capacity), outputs_grid(v.output_capacity)), kThreads, 0, stream>>>(winners, v, batch, blocks, expand, rate);
  return cudaGetLastError();
}
cudaError_t resident_batch_route(DataView data, const Split* winners, BatchResidentView v,
                                 const OutputBatch* batch, cudaStream_t stream) {
  if (!shape(v) || !batch || !data.rows || !extent(v.output_capacity, data.rows, sizeof(int)) || !data.bins ||
      !data.types || !winners || !v.assignments || !v.left_map || !v.right_map || !v.states) return cudaErrorInvalidValue;
  route_batch<<<dim3(grid(data.rows), outputs_grid(v.output_capacity)), kThreads, 0, stream>>>(data, winners, v, batch);
  return cudaGetLastError();
}
cudaError_t resident_batch_advance(BatchResidentView v, const OutputBatch* batch, cudaStream_t stream) {
  if (!shape(v) || !batch || !v.states || !v.active) return cudaErrorInvalidValue;
  advance_batch<<<grid(v.output_capacity), kThreads, 0, stream>>>(v, batch); return cudaGetLastError();
}
cudaError_t resident_batch_predict(DataView data, BatchResidentView v, const OutputBatch* batch,
                                   unsigned outputs, double* predictions, cudaStream_t stream) {
  if (!shape(v) || !batch || !data.rows || !outputs || !extent(data.rows, outputs, sizeof(double)) ||
      !data.bins || !data.types || !v.nodes || !v.states || !predictions) return cudaErrorInvalidValue;
  predict_batch<<<dim3(grid(data.rows), outputs_grid(v.output_capacity)), kThreads, 0, stream>>>(data, v, batch, outputs, predictions);
  return cudaGetLastError();
}
} // namespace ghb::gpu
