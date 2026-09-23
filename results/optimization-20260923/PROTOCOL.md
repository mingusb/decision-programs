# Continuous optimization: first measured candidates

2026-09-23 UTC. Goal remains sustained improvement of the complete custom GPU
histogram/booster, including large output counts. The preceding goal work made
progress: new diagnostics and verified evidence are present. This iteration
uses them to select and validate algorithm changes, not to declare the broad
goal finished.

Baseline: `build/diagnostics-runtime-booster`, resource diagnostics enabled,
compiler time tracing disabled. Fifty current source files and three baseline
binary identities were recorded before edits in `baseline-manifest.json` and
`baseline-source/`. Counting implementations/defaults remain frozen. GPU jobs
run serially. Existing historical quality failures remain failed.

## Work selection

Fresh Nsight Systems capture: complete one-round Delicious training and
prediction using all 500 features and 983 labels, existing train/validation
fixtures, depth3, max32bins, output tile32, output-batch graph execution, global
histogram and bounded exports. The source-derived numeric bin contract is
verified against saved model metadata. Timelines explain work; profiler times
do not rank candidates. Primary reduction/occupancy references:
[Ampere guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html) and
[NVIDIA warp reduction implementation](https://github.com/NVIDIA/cccl/blob/main/cub/cub/warp/warp_reduce.cuh).
The latter is a research reference only, never linked into production.

Candidate A: the pre-existing
`training/WIDE_FEATURE_SPLIT_EXPERIMENT.md` identifies unnecessary 256-thread
candidate blocks on wide-feature histograms fitting32lanes. Reuse the owned
warp arithmetic and existing block winner; do not simply remove the feature
guard from a winner that visits only32features. Exact fixed-histogram fieldwise
results, masks, guards, errors, signed zeros and fallback behavior are required.
Introduce explicit `warp-wide`; retain current `warp32` default. No parent
formula, histogram update, comparison ordering or model architecture change.

Candidate B: frozen-model prediction with stable per-output tree packing and
one sequential FP64 accumulator per row/output. Remove per-tree uploads/launches
while keeping every leaf addition in original order and retaining probability
transforms. Host model packing is setup, not training or feature preprocessing.
Packing, device storage, upload and complete prediction are included in timing;
extra model storage is declared. An explicit policy retains the existing path.
Design and alternatives must be written before implementation.

## Gates and fair experiments

1. Candidate A: mandatory exact split tests before timings; complete two-stage
   operation in preallocated storage, forward/reverse paired order, stream and
   graph, raw repeated timings plus synchronized host boundary. Include wide
   and narrow controls, bins3/16/32 and33fallback, small/multiple active nodes
   and outputs, partial selectors and capacity guards. No stage-only winner.
2. Candidate A end-to-end: same candidate binary, `warp32` versus `warp-wide`,
   Delicious train/validation, one round/depth3/max32bins/tile32/global/output-
   batch graph. Two warmup jobs then at least five alternating-order pairs.
   Include preparation, upload, all boosting work, export, and prediction.
   Candidate selection uses validation only. Preserved test sets are not used
   for parameter selection or represented as newly untouched data.
3. Exact fixed-statistic fixtures and normal trainer correctness tests must
   pass. Evaluate saved validation predictions with the existing independent
   common evaluator, including all applicable aggregate/per-label/signal
   metrics. Preserve strict zero-allowance failures and baseline repeat
   controls separately from implementation correctness and performance.
4. Candidate B uses the identical saved model and evaluation data for both
   policies; never retrain to compare prediction arithmetic. Require bitwise
   raw/transformed outputs and independent traversal checks across objectives,
   missing/categorical/tail/base-only models. Measure complete prediction with
   setup included, paired warm repetitions; report storage increase explicitly.
5. Use new NCU sections/resource reports to explain changed launch geometry,
   barriers, registers, FP64 work and memory traffic. Use Nsight Systems to
   verify launch/transfer removal. Run relevant memcheck/racecheck/synccheck/
   initcheck checks. Do not instrument performance rankings or promote a path
   before its actual gates are satisfied. Compilation time tracing stays in
   separate build directories because its code-generation effect is measured.

No global fastest claim follows from one GPU/workload. Candidates remain
explicit until promotion is supported; failed experiments are retained and
inform the next iteration rather than being hidden or redefining the goal.
