# Bounded output-tile tree construction

2026-09-22. Record before algorithm implementation. New build/evidence:
build/booster-level-batch and results/booster-level-batch-20260922. Preserve the
promoted root defaults, original count code, and every prior sealed build.

## Evidence and candidate selection

Previous complete-operation experiments give batched global deeper histograms
the lowest sample median in all14 multi-output shapes, in stream and graph mode.
The combined trainer trace still spends33.3% of kernel time in deeper histograms,
23.4% in frontier management and18.8% in split candidates/winners. These are
diagnostic fractions, not end-to-end speedup estimates. Shared policies add
loads, shared64-bit CAS work and bank conflicts; do not make them a default.

Primary sources checked for task batching/hardware applicability:
- XGBoost feature/input task split grids and warp evaluation, with a different
  quantized-integer numerical contract:
  https://raw.githubusercontent.com/dmlc/xgboost/56f951e7419a6f66f4568865e1d7835bcb6dbbf1/src/tree/gpu_hist/evaluate_splits.cu
- LightGBM CUDA histogram task construction and shared/global tradeoffs:
  https://raw.githubusercontent.com/microsoft/LightGBM/a0afb85496b1f483710d2a861330321eab794fff/src/treelearner/cuda/cuda_histogram_constructor.cu
- SM86 resource constraints, including the16-block and48-warp limits:
  https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
No external algorithm source is incorporated. Our existing arithmetic is reused.

Selected architecture builds one level across a bounded output tile, retaining
independent trees and partitions. It removes per-output launch serialization
before changing local arithmetic. Root statistics write directly into the first
node slot of each retained histogram; remove the separate root Stats/winner caches
and their copies in this path. Reuse immutable root counts as already measured.

Split candidates/winners use output-major independent tasks and per-output active
masks. Retain exact existing per-feature FP arithmetic and tie rules, including
owned block fallback. Packing multiple feature warps into a larger CTA is a
separate candidate; do not conflate that unmeasured rewrite with this integration.

Resident state uses output-major assignments/nodes/frontiers. For capacity<=1024,
one CTA/output can scan integer split flags, check status/capacity and materialize
nodes without inter-CTA dependencies, replacing three materialization launches.
For larger capacities retain a batched scan/prefix/materialize sequence. Keep
route and advance separate: route reads old active counts, and globally changing
them inside row CTAs would race. Row routing and prediction initially use output
grid tasks, preserving per-output arithmetic. Grouped-output prediction stores
may improve coalescing but are a separately measured future layout candidate.

## Exact contracts

Inputs remain feature-major packed uint16 bins, raw GPU-resident targets/weights,
row-major predictions and FP64 derivatives. Counts are exact uint64, G/H atomics
unordered FP64. No sampling, histogram subtraction, approximate gradients, shared
tree structures or changed objectives. Multiclass computes one full pre-round
derivative snapshot before any output prediction updates. Independent objectives
compute compact derivative tiles before their trees. Per-output predictions are
updated once per round; inter-output batching introduces no new dependency.

Common device OutputBatch has output_begin, output_count, derivative_begin and
derivative_stride. The host uploads a validated selector before work; storage
remains live until completed-tile export. Kernels guard live outputs and device
active counts; no host reads or allocation occur between levels. Reusable graph
launches use the same selector for full and short tiles. Every output retains
independent status/frontier counts; inactive capacity must not leak stale state.

For B outputs, N rows, F features, H total bins, C frontier capacity and P tree
capacity: assignments[B][N], histograms[B][C][H], candidates[B][C][F], winners[B][C],
four frontier/map arrays[B][C], offsets[B][C], scan counts[B][ceil(C/1024)],
nodes[B][P], states[B], packed active[B]. Root histograms use stride C*H and
overwrite only each live output's first H cells. The shared selector carries
derivative strides because compact final tiles have a smaller row stride.
Device pointer extents/overflow and explicit memory budgets must be checked.

Host memory planning is setup, not training computation. Prefer reducing B to
fit full desired frontier capacity before reducing C; if even B1 requires a
smaller frontier, retain explicit overflow failure rather than silently truncate
trees. Account fixed dense predictions/targets, multiclass full derivatives,
per-output state/histograms/independent derivatives, count cache, temporary arrays,
initialization/loss buffers and optional recording. Available GPU memory also
bounds allocations. Exports are bounded and happen only for completed tiles;
compact export may read counts then exact nodes, while bounded full-capacity
export trades bytes for a single wait. Model serialization order stays round/output.

Expose a separate tree-build policy initially. Existing per-output production
path remains the comparison, and defaults change only with measured support.
Invalid/incompatible explicit policy combinations reject. Shared deeper mode
remains explicit and subject to existing shared-capacity limits.

## Fair experiments and acceptance

First validate primitive masks, independent frontiers, stable node numbering,
guards, malformed splits/status, zero/short tiles, graph selectors, fallback and
overflow against independent CPU or existing exact references. Then full training
for all objectives, depths0/2/5, bins below/above32, scalar/33/129/1024/4096 outputs,
both submission modes and constrained tile budgets. Serial uninstrumented forward
and reverse comparisons include all setup, retained storage, exports and prediction.
Use Nsight to explain launches and remaining stalls, not to rank implementations.
Run applicable Compute Sanitizer checks. Preserve raw failures and source/binary
identity. Recompute held-out metrics without modifying the existing evaluator.
Acceptance clarification recorded after integration: the user's acceptance of
observed rounding-scale changes applies to the preceding
root-default promotion. It is not a blanket allowance for this new architecture;
strict zero-allowance reports retain their actual status. A new promotion decision
requires its own complete numerical and quality evidence.

Real-data experiments use separate training/validation/test fixtures prepared
offline. Production consumes raw floats and performs its own GPU binning/training.
Compare external GPU boosters only as isolated benchmark references, with pinned
versions, explicit supported objective strategies, comparable tuning budgets,
complete timings, memory and inference measurements. Unsupported modes must be
reported rather than substituted silently. Do not infer general superiority from
histogram or synthetic speedups; compare held-out quality and time to that quality.

## Follow-up candidates identified during integration; not implemented

The current integration deliberately retains the existing per-tree prediction
traversal to isolate batching. Routing already discovers when each row reaches a
leaf. A separate candidate can add that leaf's already-materialized, rounded
`Node.value` to the corresponding prediction at termination, exactly once per
output/round. This could remove the final tree traversal and launch. Read the
stored leaf value (or prove separately rounded multiplication); folding
learning_rate*leaf + prediction into an FMA would change the existing numerical
contract. Preserve pre-round multiclass derivatives, inactive/status handling,
early leaves, depth0, and all exported tree fields. This needs its own recorded
experiment and all-input correctness checks before changing the implementation.

The public inference path uploads/applies one exported tree at a time. A separate
GPU forest representation could upload all nodes plus tree/output offsets once,
then traverse trees in existing per-output addition order in one or a bounded
number of kernels. A reusable resident model object could also avoid subsequent
model uploads. Candidate thread mappings must compare row-coalesced feature bins
against output-coalesced prediction stores and divergent per-output trees. Do not
assume a published inference layout wins on these independent large-output
forests; compare complete public inference and resident inference separately,
including preparation/storage and exact model-order rounding. Current benchmark
inference reports describe the existing public API, not this unmeasured design.
