# Parent leaf sharing within one feature warp

Recorded 2026-09-23 before implementation. This document proposes a bounded
candidate C after the independent wide-split and fused-inference experiments.
It changes no code or default. Its reference is candidate A, `warp-wide`, with
all other dispatch, arithmetic, memory and submission choices frozen. It is
not a performance result or a claim that sharing is faster on SM86.

## Observed duplicated work and numerical boundary

Fresh retained evidence is in
`results/optimization-20260923/ncu-wide-split-baseline/profile.ncu-repz` and its
`manifest.json`; selected counters are in `baseline-ncu-selected.json`. The
capture uses Delicious training, 500 features, one round, depth three, output
tile 32, output-batch construction and graph submission. It profiles the
existing 256-thread `split_candidates<true>` implementation, not candidate A.
The grid has 64,000 allocated feature tasks, of which 16,000 execute the parent
leaf calculation in this captured launch. Dynamic masks explain the difference.

Offline `ncu --import ... --page source --print-source cuda,sass --csv` ties
the following program counters to `kernels.cu` in the retained build. PCs are
specific to that capture and must not be treated as future-build identifiers.

| Work | Captured SASS location | Instructions / predicated thread instructions |
| --- | --- | --- |
| Parent H+l2 (`kernels.cu:253`) | `0xb00ecfe10`, `DADD R24, R22, c[0x0][0x1a0]` | 128,000 / 4,096,000 |
| Parent division reciprocal | `0xb00ecfe90`, `MUFU.RCP64H R13, R25` | 128,000 / 4,096,000 |
| Representative division refinement | `0xb00ecfed0`, `DFMA R14, -R24, R12, 1` | 128,000 / 4,096,000 |
| Parent inner factor (`kernels.cu:260`) | `0xb00ed2c60`, `DFMA R42, R42, R14, UR12` | 16,000 / 16,000 |
| Another parent inner-factor site | `0xb00ed3770`, `DFMA R24, R20, R14, UR12` | 16,000 / 16,000 |
| Final gain contribution | `0xb00ed2c80`, `DFMA R42, R42, R14, R26` | 16,000 / 16,000 |
| Another final gain site | `0xb00ed3790`, `DFMA R20, R24, R14, R20` | 16,000 / 16,000 |

The parent-solve sites execute with 32 active lanes in each of eight warps:
256 identical scalar evaluations per active feature task. The captured block
kernel uses 72 registers/thread, 48.08% achieved occupancy, 82.08% FP64-pipeline
activity and 0.145888 eligible warps/cycle. These are explanatory counters, not
time fractions that can be summed or a speedup bound.

Candidate A already replaces those eight warps with one; do not credit that
removed work to C. Current candidate-A resources in
`results/optimization-20260923/compiler/report.json` are 82 registers/thread
for `warp_candidates<true>` and 80 for `<false>`, with zero shared memory, local
memory, stack and reported spills. A source-counter capture of A is still
required to verify its dynamic parent-solve count before interpreting C.

The final gain contribution is fused. Computing the entire parent benefit once
would introduce a rounded product before subtraction where the baseline can
use one DFMA. That changes the operation graph even when the symbolic formula
is unchanged. C shares only the already rounded parent leaf value. It leaves
`benefit`, `consider`, child leaves, parent inner factors and the final fused
gain expression in their current locations and forms.

## Exact operation contract and proposed change

Use the same CUDA C++23 order-2 warp-candidate input/output contract as
`WARP_WIDE_SPLIT_EXPERIMENT.md`: validated DataView and feature offsets/types,
FP64 gradient/Hessian and exact u64 counts, existing SplitConfig, device active
counts and optional batch selector. Preserve all overflow checks, finite-input
requirements, scratch bounds and nonaliasing requirements. Higher-order
derivative paths and feature histograms larger than 32 bins retain their
existing behavior. No counting code, model layout, quantization, prediction,
tree growth or learned objective changes belong to this candidate.

Every live feature task is still a complete 32-thread CTA. Retain the original
histogram summation, its second zero-warp reduction and its prefix zero carry.
After `total = broadcast(local)`, all 32 lanes hold the same total bits. Lane
zero alone calls the existing `leaf(total, config)` helper, including its
denominator test, division and clipping. Initialize other lanes' temporary to
positive zero, then have **every lane** execute the same full-mask lane-zero
FP64 shuffle outside the lane-zero branch. Assign that result to `best.value`.
No other expression, helper implementation or comparison order is changed.

The warp shuffle copies the selected lane's representation; it performs no
floating-point arithmetic. The installed CUDA13.4 implementation at
`/usr/local/cuda-13.4/include/sm_30_intrinsics.hpp:501` bit-casts the double to
an integer, shuffles it, and bit-casts it back. The integer implementation
transfers both 32-bit halves. This preserves signed zero and subnormal payloads.

Parent leaf calculation is still required for force-leaf, insufficient-count
and no-eligible-split cases: a no-split candidate must export its parent leaf.
Do not move sharing inside `if (!force_leaf ...)`, `if (bin < bins)` or the
lane-zero missing-threshold branch. All lanes named by the full mask must reach
the shuffle, including lanes without a bin. Task bounds and active-output/node
skips are uniform for this single-warp CTA. Do not replace the declared mask
with an opportunistic active mask that could hide a participation bug.

Each feature retains its own reduced parent statistics. No cross-feature or
cross-node reuse is permitted: independently accumulated feature histograms can
have different FP64 residuals, so mathematically equal node totals need not
have equal bits. Neither a reusable parent-benefit buffer nor a new parent
reduction kernel is part of C.

The output contract is zero differing bits in every defined Split field for
the same fixed histogram, including gain, parent/child values, signed zeros,
feature/threshold/missing direction, force-leaf and no-split metadata. Inactive
outputs/nodes and histogram inputs remain untouched. Two launches, original
candidate/winner scratch, stream ordering, graph replay and absence of host
waits/allocations/atomics all remain unchanged.

## Work, memory, occupancy and divergence analysis

At source level, C changes 32 identical parent solves per active feature task
to one scalar solve plus a double broadcast. No histogram load, reduction,
candidate store, winner read, child solve or gain evaluation is removed. It
adds a lane-selection predicate or branch, reconvergence, and two 32-bit
shuffle operations in the installed implementation. Verify the resulting SASS
rather than assume a particular compiler branch lowering.

This is **not a 32-fold reduction in issued warp instructions or a promised
speedup**. Both policies may issue the parent instruction sequence once per
warp; C changes its active-lane mask. Whether this reduces FP64 execution cost
depends on hardware execution and compiler lowering. A scalar parent division
still has its dependency chain, and all other lanes then wait for its shuffle.
The extra branch/shuffle can lose on small or denominator-zero workloads.
Register allocation is per thread; inactive arithmetic does not automatically
free registers in other lanes. A shorter live range may help, but the new
temporary or control flow may instead increase registers or introduce spills.

On SM86 a one-warp CTA is capped by at most 16 CTAs/SM and 48 warps/SM, so the
block-count limit alone caps these kernels at 16 resident warps, or one third
of the warp ceiling. This candidate does not change CTA size and cannot remove
that structural limit. Candidate A's measured occupancy, instruction issue,
FP64 activity, register allocation and spills must be compared directly with
C. Packing multiple feature warps into a CTA is a separate future experiment.

Compared alternatives are unchanged per-lane computation (A), one value shared
through registers (C), shared-memory broadcast (unnecessary storage and memory
ordering for one warp), full benefit sharing (forbidden rounding change), and
cross-feature sharing (forbidden histogram-residual change). No external
algorithm implementation is imported; CUDA shuffle is infrastructure.

Primary references, checked against the current documentation:

- [CUDA warp shuffles and participation rules](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-shuffle-functions):
  indexed lane exchange requires participating source/destination lanes and
  matching masks at the collective. It is not a shared-memory fence.
- [CUDA FP64 fused operations](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html):
  a fused multiply-add rounds once; separated multiply/add rounds differently.
- [Ampere occupancy limits](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#occupancy):
  SM86 limits are 48 warps and 16 blocks/SM, with 64K 32-bit registers. A100's
  limits are different and do not apply to this RTX A5000 Laptop experiment.

## Required validation before ranking

Add an explicit isolated policy/specialization for C; keep A and all production
defaults intact. Build with the same compiler, architecture, optimization and
arithmetic flags. Never turn off contraction or enable fast math to make a
comparison pass. Runtime builds must have compiler time tracing disabled,
because our NVCC13.4 isolation study found time tracing changed generated code.

First inspect both variants' generated code. Map the lane-zero parent division
and clipping path, confirm the FP64 broadcast, and verify that the final gain
DFMA and its input computation/order are retained for every threshold/missing
branch. Matching source expressions alone is insufficient if NVCC changes
contraction while restructuring control flow. Preserve cubins, SASS, resources,
source/build/compiler hashes and any failed comparison.

Run exact A-versus-C candidate and winner comparisons on identical fixed
histograms, with the original block implementation as an additional independent
implementation reference. Reuse A's full shape/mask/graph/tie/cancellation
matrix and histogram/sentinel guards, then add focused parent-leaf fixtures:

- Positive and negative zero gradients; positive Hessians; positive/negative
  zero Hessian and l2 combinations giving a zero denominator. The original
  `!(denominator > 0)` branch must return positive zero exactly.
- Positive denominators adjacent to zero, subnormal gradients/results, very
  small normal values and mixed-magnitude cancellation. Retain the original
  division behavior without reciprocal approximations.
- Clipping disabled and enabled; parent values exactly at each clipping
  boundary and immediately on either side; positive/negative large values.
- Force-leaf, too few rows, all-missing, empty/no-eligible split, exact ties,
  and irregular 1/2/3/16/32-bin features. Ensure lane zero is a valid shuffle
  source even when most lanes have no histogram bin.
- Gain values adjacent to min_gain and tied/nearly tied candidates whose rank
  changes under a one-bit perturbation. Include retained cancellation cases
  that detect a separately rounded parent-benefit product.
- Changed full/short/empty/over-capacity graph selectors, independent streams
  with separate scratch, signed feature/extent guards, and a >32-bin fallback
  control that must retain A's implementation and outputs.

Require zero different defined-field bits, no guard/input corruption, and zero
applicable sanitizer findings (memcheck, initcheck and synccheck; racecheck as
applicable). Run exact dyadic full-training model/objective tests and the
existing CPU/GPU prediction validation. For arbitrary unordered-FP training,
retain baseline-repeat controls and zero-allowance quality results separately;
fixed-histogram equality does not prove identical independently trained models.

## Independent bounded performance experiment

Complete and preserve A's results first. The C comparison is A versus C in one
frozen build, not original block256 versus C. Initially use six resident shapes
`(F, bins, outputs, node_capacity)`:
`(32,3,16,8)`, `(500,3,1,1)`, `(500,3,16,1)`, `(500,3,16,8)`,
`(500,32,16,8)`, and `(500,33,16,8)` as the block-fallback control. Both variants
must write equal-sized separate scratch buffers and pass exact pre/post checks.

For each shape, measure stream and graph submission, eight complete
candidate-plus-winner operations per sample, three warmups and 15 retained
paired samples with alternating order. Record CUDA-event time and synchronized
host time for the complete operation. Allocation, identical fixed input upload,
validation and graph construction are excluded equally and labeled. Preserve
every observation, policy order, clock/temperature telemetry and failure. Run
serially without a profiler. No candidate-only kernel timing establishes the
complete split-operation ranking.

Only after that pass, measure the existing complete Delicious validation
training protocol with all 500 features and all 983 outputs. Freeze dataset,
round/depth/bin limits, output tile, submission/export mode, histogram policy,
memory budget and inference policy for A and C. Include setup and completed
model export in the established total-training boundary; do not combine C with
an inference or initialization change. Keep all objective/quality metrics and
baseline-repeat evidence. Do not choose the policy using held-out test metrics.

For a case-specific speedup claim require the upper endpoint of a 95% paired
bootstrap interval for the median C/A time ratio below 1.0, with zero exactness
failures. A default promotion additionally needs no measured regression on the
agreed controls and no strict quality-gate failure. Failed or inconclusive gates
stay visible; resource/code size regressions must also be reported.

Finally collect matched NCU source counters for A and C on the same active
Delicious split stage: parent-solve predicated threads and warp instructions,
added shuffle/branch work, final gain DFMA, registers/spills, occupancy, eligible
warps, FP64 activity and stall reasons. Normalize counts by actual live feature
tasks, not allocated grid capacity. Nsight Systems verifies unchanged launch,
copy and synchronization boundaries. These profiles explain independently
measured times and are never substituted for uninstrumented rankings.
