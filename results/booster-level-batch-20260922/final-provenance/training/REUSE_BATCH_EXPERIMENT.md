# Exact count reuse, split batching, and deeper histogram experiment

2026-09-22. Recorded before implementation. Preserve finalized root/split
provenance and original counting code/defaults. New build: build/booster-reuse;
new evidence: results/booster-reuse-20260922. CUDA C++23, owned algorithms only.

## Evidence and selection

The last whole-training comparison improved large-output cases, but scalar and
fresh 129-output confirmation did not improve uniformly. All strict quality
failures remain failed. Nsight's combined trace attributes 27.6% of recorded
kernel time to deeper histograms, 22.4% to split candidates and 11.2% to roots.
Batched root atomics still show instruction/memory queue pressure; small warp
split grids have only 16 blocks. Percentages are diagnostic, not speedup bounds
for complete training including host/export/preparation.

Primary sources checked before this pass:
- Pinned XGBoost histogram batching maps independent node/target tasks to CTAs;
  its target-major quantized integer statistics differ from our FP64 contract:
  https://raw.githubusercontent.com/dmlc/xgboost/56f951e7419a6f66f4568865e1d7835bcb6dbbf1/src/tree/gpu_hist/histogram.cu
- Pinned XGBoost split code uses feature/input task grids followed by independent
  winners, supporting batching rather than reducing root feature parallelism:
  https://raw.githubusercontent.com/dmlc/xgboost/56f951e7419a6f66f4568865e1d7835bcb6dbbf1/src/tree/gpu_hist/evaluate_splits.cu
- Shared-local histogram construction plus global merging is distribution and
  architecture dependent. Maxwell evidence is not an SM86 ranking:
  https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
- Resource limits must match SM86, including 48 resident warps/SM and 16 CTAs/SM:
  https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html

No source algorithm implementations are incorporated. CUDA warp intrinsics and
runtime are infrastructure; CUB/Thrust/NVIDIA algorithm code remain references.

## C: immutable root counts

Input: retained feature-major uint16 bins, exact feature offsets, positive rows
and valid device buffers. Output: uint64 count[total_bins]. Every row contributes
once per feature, including missing bin zero and zero-weight rows. No sampling,
row exclusion, output-specific binning or mutable bins is supported in this
reuse contract. Counts are therefore invariant across outputs and rounds.

Compare an owned global row kernel and an owned per-feature shared kernel for
one-time setup. Shared chunks contain at most 4096 rows, so uint32 shared counts
are exact; final counts are uint64. Shared footprint is 4*max_feature_bins bytes,
limited to 48KiB; explicit unsupported requests reject. Do not label either
setup policy fastest before measuring complete training with setup included.

Cache costs 8*total_bins bytes, retained for training and counted in histogram
and device budgets. Cached root batches replace memset with seed Stats{0,0,count}
and run the same output-subgroup accumulation with count atomics compiled out.
This removes per-output/per-round integer increments but retains G/H atomics,
histogram writes, and cache consumption. The cache/output must not alias.
Original per-output count accumulation remains available for attribution.
Exact count reuse is not a claim of identical FP64 atomic ordering.

## S: batched root split decisions

Root histograms already have layout [output][total_bins], the same layout as
independent node histograms. Evaluate all roots in an output batch using the
existing candidate/winner arithmetic, with output interpreted as node. Add a
direct-count warp entry point; block256 batching already exists. No active-count
upload or extra kernel is needed. Grid grows F -> T*F, then 1 -> T winners;
two launches per output become two per tile.

Scratch needs T*F candidates plus T winners, each Split 48 bytes. Enlarge the
existing per-tree candidate buffer to max(node_capacity,T)*F and reuse it,
because root candidates need not survive winner selection. Only T winners are
separately retained; all added bytes are included in the device budget.
Initializer copies the selected cached winner using the existing
device selector. Skip both the root Stats copy and per-tree root split kernels.
Node creation/routing, deeper splits and independent scalar-tree semantics stay
unchanged. Multiclass keeps its full frozen pre-round derivative snapshot.
All fields must compare bitwise on identical inputs across sequential/batched
block/warp execution, including fallback, missing/category/clipping/tie/leaf cases.
The unchanged FP arithmetic DAG is required; no parent-benefit rearrangement.
Cache lifetimes and pinned selector reuse follow the same stream/export ordering.

## D: deeper histogram batching versus small shared histograms

Before changing tree-state lifetime/architecture, build bounded, allocation-free
primitives and measure complete histogram operations. Inputs: bins as above,
row-major G/H with explicit stride/output range, assignments[output][row], each
output's device active count and a shared node capacity. Assignments may differ
across outputs; negative assignments are inactive. Output:
Stats[output][node_capacity][total_bins]. Clear each output's active node range
(clamped to capacity), preserve inactive capacity like histogram_active, and accumulate
exact u64 counts with FP64 unordered atomics. No row pack/transpose is omitted.

D-global maps rows/output subgroups to lanes, reusing bins and G/H across features.
D-shared uses feature/output-group/row-chunk CTAs, with group widths 1/4/8 and
24*group_width*node_capacity*max_feature_bins <=48KiB shared memory. Compare
ceil(N/4096) row chunks (capped at256) and four times that count (capped at256);
more chunks expose parallelism but add merge atomics. Shared FP64
updates and global merge may lose to direct global updates; shared also reloads
G/H and assignments per feature. Compare against existing sequential global and
shared operations, including clearing, in both stream and graph execution.

Retaining all output states for deeper production batching adds assignments,
frontiers, nodes and histograms, unlike root batching. At T16,N4096 alone the
extra assignments cost 240KiB; histogram storage grows with T*capacity*total_bins.
A primitive result does not establish an end-to-end win. Use the already-owned
shared trainer path to measure deeper shared histograms end-to-end while roots
remain batched. A full state-retention rewrite must be justified by measured
primitive and complete-training evidence, not assumed from occupancy.

## Experiments and gates

Independent flags for C and S; defaults remain per-output counts and per-tree
splits. Require batched roots for these cache consumers, reject incompatible
configuration. Include all cache/scratch costs, short tiles, scalar controls,
multiclass K>tile, depth zero and >32-bin fallback. Full-training comparisons:
previous combined path, C only, S only, C+S, and deeper forced-shared variation.
Use serial alternate-order timings, matching seeds and a fresh seed. Four-repeat
baseline controls distinguish pre-existing FP variability without relaxing any
zero-allowance gate or treating a prior failed comparison as passing.

Primitive D matrix covers N1024/4096/65536, T1/3/16/33, capacity2/8/32, bins
16/32/64/256, irregular features, zero derivatives, missing/skew and 0/75% inactive
rows. Independent CPU references check counts and FP error bounds; guards and
changing active counts/graph selectors are tested. CTest and applicable Compute
Sanitizer checks are mandatory. Preserve raw failures, commands, source/binary
hashes and old frozen evidence. Nsight explains results; profiler timings never
rank candidates. All applicable held-out metrics use the unchanged evaluator.
