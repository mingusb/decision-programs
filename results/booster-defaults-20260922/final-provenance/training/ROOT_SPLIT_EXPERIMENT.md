# Root batching and small-bin split experiment — 2026-09-22

Recorded before implementation. These are independent, opt-in candidates; no
counting policy, training default, or finalized evidence is replaced.

## Evidence and candidate selection

The finalized resident trainer Nsight Systems capture attributes 45.6% of kernel
time to weighted histograms and 24.0% to split candidates on the 129-output case
(`results/booster-resident-20260922/REPORT.md`). The global row kernel exposes only
4 blocks for 1024 rows, or 16 for 4096 rows, on our 48-SM RTX A5000 Laptop SM86.
Adjacent lanes read different rows of one output from row-major derivatives.

Candidate R batches depth-zero histograms across the existing output tile. A
power-of-two output subgroup (up to 32 lanes) shares a bin load through a shuffle;
adjacent lanes load adjacent outputs and reuse their FP64 derivative pair across
features. Larger tiles use multiple subgroups. Direct global atomics retain the
existing mathematical statistics. This removes per-tree root submissions and
exposes more parallelism. It adds a bounded output-major root cache and a copy
fused into the existing tree initialization. It does not remove the count atomic
stream yet. Clear plus accumulation plus consumption and cache memory all count
in the comparison.

Alternatives considered: feature/output-group shared privatization reduces global
atomics but reloads derivatives across features and increases shared capacity;
output-major derivatives plus per-output CTAs trades a transpose for coalescing;
compact row partitions add scan/scatter but help deeper inactive rows, not roots;
parent-minus-child statistics remove deeper work but add retained memory and FP64
cancellation; warp aggregation has few same-output peers in a wide-output warp.
Count reuse across outputs/rounds and regression Hessian reuse are separate later
experiments. Root split batching is also deferred to isolate attribution.

Candidate S uses a single 32-thread CTA per feature/node when all feature
histograms have at most 32 bins. It preserves the two-stage candidate/winner
operation and falls back to our original 256-thread implementation for larger
features. A warp winner is used only for <=32 features. It removes idle warps,
shared split reductions and block barriers without changing histogram layout.
One-CTA-per-node fusion could remove a launch but collapse a small root from 16
feature CTAs to one, so is deferred. Moving parent benefit outside individual
candidate scoring could change rounded ties and is excluded.

Primary implementation/hardware references checked before implementation:
- XGBoost pinned split evaluator uses 32-thread feature CTAs and explains its
  measured barrier/broadcast advantage; this is design evidence, not our ranking:
  https://raw.githubusercontent.com/dmlc/xgboost/56f951e7419a6f66f4568865e1d7835bcb6dbbf1/src/tree/gpu_hist/evaluate_splits.cu
- XGBoost pinned histograms batch target tasks with target-major derivatives and
  quantized integer statistics, a different numerical/layout contract:
  https://raw.githubusercontent.com/dmlc/xgboost/56f951e7419a6f66f4568865e1d7835bcb6dbbf1/src/tree/gpu_hist/histogram.cu
- SM86 limits must be used instead of extrapolating A100 resource/FP64 behavior:
  https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html

No NVIDIA algorithm code is incorporated. CUDA runtime and warp intrinsics are
infrastructure; CUB/Thrust remain absent from production.

## Contracts

R inputs: retained feature-major uint16 bins, exact offsets, frozen row-major
FP64 gradients/Hessians, explicit derivative stride, first output and batch size.
R output: batch-major [output][total_bins] Stats (FP64 G/H and uint64 count),
including missing bin zero and zero-weight rows. All root rows participate.
Integer counts are exact; FP64 atomic addition ordering can change. No quantizing
statistics, model architecture change, split-rule change or approximation.
Multiclass derivatives remain the full pre-round snapshot, while root batches
are bounded by output_tile_size. The cached histogram is copied into the normal
root workspace during tree initialization, with no additional launch.
Cache bytes = 24 * batch_capacity * total_bins. Both histogram and total device
budgets include this payload; insufficient budgets reject, never silently change
the algorithm. Cache lifetime covers all consumers before the next batch writes.
With R selected, root calibration is skipped because R explicitly selects global
root atomics. Existing global/shared/auto policy still controls deeper levels
(auto uses the existing global deeper path).

S inputs/outputs, clipping, missing handling, categorical equality, leaf fallback,
gain threshold and deterministic tie rules equal the existing split API. It is
asynchronous and allocation-free, respecting device active count and capacity.
Preserve useful-lane addition order, including zero additions of the original
two-level reduction. Fieldwise bit equality on identical histograms is required
for the <=32 specialization, including signed-zero/cancellation fixtures. No
claim of training bit equality follows from unchanged atomic histograms.

## Fair experiment and gates

Retain build/booster-resident and its finalized provenance. Build in a new
directory; preserve every command, binary/source hashes, raw output and failure.
Run all GPU workloads serially. Compare same-binary baseline, S, R and R+S using
identical seeds, instrumentation off, alternating order and fresh-seed repeats.
Primary cases: scalar 65536x32, 129 regression outputs at 4096x16, 1024 binary
outputs at 4096x16, 4096 binary outputs at 1024x16. Cover partial tiles, multiclass,
depth zero, >32-bin fallback, missing/categories/zero weights and strict budgets
in correctness tests. Measure total training and preparation/whole-call times,
not just kernels. Record root-cache bytes and effective batch size.

Test R against explicit CPU statistical references and the owned per-output GPU
primitive with exact counts and stated FP64 error checks. Test S fieldwise on
fixed histograms and complete two-launch microbenchmarks. Run CTest, applicable
Compute Sanitizer memory/race/synchronization checks and end-to-end CPU/GPU model
validation. Evaluate all held-out metrics using the existing zero-allowance
evaluator; a regression stays a failure even at roundoff scale. Nsight explains
launch/occupancy/coalescing/atomic/barrier changes; profiler timings never rank
uninstrumented variants. Defaults stay unchanged unless separate evidence
supports promotion. No universal fastest or quality-preservation claim.
