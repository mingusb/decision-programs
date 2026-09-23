#pragma once
#include "ghb/output_batch.cuh"
#include "ghb/resident.cuh"

namespace ghb::gpu {

// Independent output-major storage. For B=output_capacity, C=frontier_capacity,
// P=node_capacity, N=data.rows: assignments[B*N], nodes[B*P], frontier,
// next_frontier,left_map,right_map,offsets[B*C], states[B], active[B], and
// block_counts[B*ceil(C/1024)]. Winners supplied separately have B*C entries.
// No buffers alias. Node/frontier indices are local to their output.
// active mirrors states[].active_nodes after initialize/advance. The host owns
// only pointer setup; all counts, status and decisions remain device-resident.
struct BatchResidentView {
  std::uint32_t output_capacity{}, frontier_capacity{}, node_capacity{};
  std::int32_t* assignments{};
  Node* nodes{};
  std::int32_t* frontier{};
  std::int32_t* next_frontier{};
  std::int32_t* left_map{};
  std::int32_t* right_map{};
  TreeState* states{};
  std::uint32_t* active{};
  std::uint32_t* offsets{};
  std::uint32_t* block_counts{};
};

// Device selector and all storage remain live through stream completion.
// output_count<=output_capacity is a host-validated selector contract; kernels
// defensively clamp it. Full, short and empty selectors support graph replay.
// Inactive outputs are untouched except active is zeroed by initialize/advance.
// Calls allocate nothing, perform no host synchronization and return launch
// errors. All pointer extents are caller-owned and host size products checked.
cudaError_t resident_batch_initialize(std::uint32_t rows, BatchResidentView,
                                      const OutputBatch*, cudaStream_t);
// Stable integer child ordering and floating arithmetic match resident_materialize.
// A single CTA/output handles C<=1024; larger C uses scan/prefix/write stages.
cudaError_t resident_batch_materialize(DataView, const Split*, BatchResidentView,
                                       const OutputBatch*, bool expand_children,
                                       double learning_rate, cudaStream_t);
cudaError_t resident_batch_route(DataView, const Split*, BatchResidentView,
                                 const OutputBatch*, cudaStream_t);
cudaError_t resident_batch_advance(BatchResidentView, const OutputBatch*, cudaStream_t);
// predictions[N*outputs] is row-major; output_begin selects the output columns.
// The host validates output_begin+output_count<=outputs before selector upload.
cudaError_t resident_batch_predict(DataView, BatchResidentView, const OutputBatch*,
                                   std::uint32_t outputs, double* predictions,
                                   cudaStream_t);

} // namespace ghb::gpu
