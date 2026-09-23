# Profiling-guided optimization — 2026-09-23

Status: A/B correctness, idle timing and quality campaigns completed. Candidate
E prediction validation is in progress. No production default is promoted. The active
continuous-optimization goal remains open. The five profiling integrations
preceding this iteration are documented in the sealed
[profiling expansion report](../profiling-expansion-20260923/REPORT.md).

## Implemented candidates

**A: explicit `warp-wide` split policy.** For features with at most 32 bins,
reuse the existing warp candidate arithmetic and, above 32 features, the existing
block winner. Narrow-feature and larger-bin paths retain their prior owned
implementations. This is CUDA C++23 and introduces no NVIDIA algorithm-library
dependency. [Contract and pre-code selection](../../training/WARP_WIDE_SPLIT_EXPERIMENT.md).

**B: explicit `PredictionPolicy::fused_output`.** Pack the immutable host model
stably by output, upload the forest once, and let each CUDA thread traverse its
trees sequentially with the original FP64 addition order. Quantization and
probability transforms are retained. Host packing is model setup, not a CPU
feature-processing/training stage. The complete forest must fit in device
memory; the default per-tree path retains its smaller workspace contract.
[Contract and pre-code selection](../../training/FUSED_PREDICTION_EXPERIMENT.md).

Counting implementations, policies and defaults are untouched. Training still
defaults to `warp32`; prediction still defaults to `per_tree`.

## Evidence and correctness

The fresh baseline Nsight Systems trace uses all 500 Delicious features and 983
outputs. Every fitted feature has 3 bins. Split candidates account for 30.01% of
recorded training kernel time; deeper/root histogram accumulation account for
53.24%/15.85%. These shares select work, not performance rankings. Nsight Compute
reports 72 registers, 48.08% achieved occupancy and 82.08% FP64 pipeline activity for
the captured block candidate. Baseline raw evidence is under
`nsys-delicious-baseline/` and `ncu-wide-split-baseline/`.

- All 19 CTest cases passed (`ctest.log`), including new exact primitive,
  graph-selector, overflow, signed-zero and dyadic full-trainer comparisons.
- Prediction's 373,762 checks cover all objectives, output/row tails, empty
  forests, sparse 1024-output models, categorical/unseen/missing values, signed
  zeros, subnormals and cancellation, with frozen-model GPU and CPU references.
  The expanded direct API checks replay graphs after changing bins/base values,
  verify buffer guards/input immutability, and reject21 malformed host calls
  before dispatch. Both the expanded normal and sanitizer runs passed.
- Prediction passed Compute Sanitizer memcheck with 128-byte allocation padding
  and full leak checking, plus global/shared initcheck: zero errors and leaks.
- Split tests passed 17,414,808 checks under padded memcheck/full leak checking:
  zero errors and leaks. A bounded 500-feature, one-output, one-node fixture
  passed racecheck, synccheck and global/shared initcheck in both stream and
  graph mode, with 17,273 exact field checks per diagnostic. The larger
  16-output/eight-node racecheck timed out at 120 seconds and remains a failed
  diagnostic attempt; the bounded passes do not claim that larger coverage.
- All 312 existing code/constant sections in the booster archive match the frozen
  baseline byte for byte under unique full-demangled symbol mapping. Two new
  sections belong to the forest kernel. Its compiler report shows 30 registers,
  no spills and no barriers. See `compiler-demangled-comparison.json` and
  `compiler-lib/`; compiler time tracing is disabled for runtime builds.

An isolated Nsight capture brackets only one complete fused prediction; it
excludes correctness prechecks and warmups. On the same saved 983-tree model,
the complete path has 18 kernel launches instead of 1,001: 16 encoding launches,
one ordered-forest launch and one sigmoid launch. H2D copies fall from 1,518 to 538;
remaining per-feature metadata transfers are a concrete next target. The
prediction output is 20,320,576 bytes; additional packed-forest device payload
versus the old maximum-tree buffer is 446,640 bytes. These are work/storage
observations, not timing claims. See `prediction-transfer-comparison.json` and
`nsys-prediction-fused/kernel-summary.json`.

## Quality gate: failed, retained

The independent CPU evaluator audits all 983 labels, with 982 defined AUCs and
one null AUC; other metrics for that label still participate. All five matched
`warp32`/`warp-wide` training pairs fail the zero-allowance quality gate. Failed
label counts are 112, 84, 131, 121, 117. Every pair also fails at its reference-selected
validation operating threshold. Pair 0 loses log loss, Brier, micro/macro AP,
macro AUC and precision@3/@5, and changes 49 tree structures.

The separate identical-policy repeat control also fails quality gates and
changes 65 tree structures. It exposes baseline variability; it does not waive
candidate failures or establish that every difference is caused by one source.
Fixed-histogram split equivalence and arbitrary-run training repeatability are
different contracts. No allowance has been introduced. Full reports, corruption
controls, hashes, metric directions and provenance limitations are retained in
[quality-summary.json](quality-summary.json) and `quality-pair0..4.json`.

## Performance observations are provisional

The exploratory campaigns retain every paired timing, warmup, command, model,
prediction and available telemetry. GPU jobs ran serially, but some CPU audits
and compilation overlapped the initial measurements. Windows-native counters
later identified `cod` using 78.75–84.74% of its GPU 3D engine; the user confirmed
the game was active and requested correctness work for now. The initial
prediction median paired ratio of 0.40007 is therefore not an idle-device speed
claim or a promotion gate. See [timing limitations](EXPLORATORY_TIMING_LIMITS.md).

The full split-operation benchmark stopped progressing and was terminated after
more than five minutes. Its incomplete output and host backtrace are retained;
the host was in `cuEventSynchronize`. Buffered output prevents assigning the
stall to a precise cell. Bounded, progress-logged case 4 reproductions subsequently
passed exact pre/post checks in both stream and graph modes. This does not
reclassify the interrupted full run as passed or prove its cause. Diagnostic
timings are excluded from selection.

## Separate idle confirmation

The user subsequently released the GPU. Windows-native counters then reported
no engine above5% during two pre-run samples. No heavy CPU audit or build ran
during the following serial, unprofiled confirmation campaigns. Their artifacts
are separate `*-idle` directories; historical failures remain unchanged.

All26 split-operation cells completed with exact pre/post checks. For500
three-bin features, complete candidate-plus-winner device operations improved
by approximately2.62–3.10x across the tested output/frontier/stream/graph cells.
The direct narrow-policy control was approximately1.00x. The five paired
complete training runs had median paired candidate/reference ratio0.83537
(95% paired-bootstrap interval0.82421–0.87026); train+predict+serialize ratio was
0.84117 (0.83048–0.86652). These are local performance results; quality failures
still block promoting the split policy.

Frozen-model prediction completed18 matrix cases plus actual Delicious
validation, with15 alternating pairs/case and exact raw/transformed results.
The actual983-output validation case had median paired speedup1.993x, with a
candidate/reference ratio interval0.4074–0.5301. Its separate policy medians
were33.36ms and14.51ms; a ratio of those two medians differs from the median of
paired ratios. The1024-output model improved23.54x/2.20x/1.26x for32/4096/65536
rows respectively. The65-output model improved3.10x/1.86x/1.17x. These are
casewise intervals and measurements, not universal rankings.

Small cases are mixed or inconclusive. The three-tree/three-output model does
not establish a gain; its4096-row median paired ratio is about1.031 with an
interval0.9935–1.3289. Several scalar large-row cases also lack a confident
improvement. The default remains unchanged. Full observations and uncertainty
are in `performance-idle-summary.json`; neither statistical intervals nor speed
waive a strict quality failure.

Fresh candidate Nsight Compute confirms32-thread blocks,82 registers/thread,
about31.98% achieved occupancy and84.61% FP64 pipeline activity, with no recorded
local/shared spilling. Compared with the old256-thread candidate, occupancy
falls while redundant thread work is removed; the profiler counters explain
the independently timed result rather than rank it.

## Next work

The separate idle quality audit also fails all five training pairs, with
73, 71, 69, 80 and 63 labels regressing under zero allowance. Maximum probability
difference across those pairs is 0.2377384124. The identical-policy repeat also
fails; this does not relax the candidate gate. All five evidence pairs are valid,
and the campaign's 56 before/after identities agree. Later E source edits are
recorded separately from the preserved A/B snapshot. Full metric failures and
hashes are in [quality-idle-summary.json](quality-idle-summary.json).

A bounded independent count/high-precision oracle examined five first divergent
decisions in three outputs from the original campaign. All selected features
were mathematically co-optimal. For output 573, different tied feature choices
route one validation row to existing leaf increments -0.00369 or +3.62287,
explaining its 0.4033 probability change. This is a training-partition tie with
different held-out routing; saved leaf arithmetic does not explain that change.
No device histogram snapshots were saved, so floating-point accumulation is a
supported explanation for the tie selection, not a proven mechanism. The
oracle does not waive any quality failure. See the
[bounded diagnostic report](divergence-oracle/REPORT.md).

Pre-code analysis also considers sharing only the parent leaf within a warp
(not the rounding-sensitive full parent benefit), and reducing per-feature
inference metadata uploads/synchronization. Fewer predicated lanes do not by
themselves guarantee fewer issued warp instructions or a speedup. These remain
separate experiments until measured.

Candidate E256 now has a pre-code contract and selection record:
`MODEL_SLAB_SELECTION.md` and
`training/INFERENCE_MODEL_SLAB_EXPERIMENT.md`. It combines four fused-prediction
model allocations/uploads into one aligned slab while retaining the same CUDA
kernel. It is implemented explicitly and undergoing validation to test the
small-call overhead hypothesis; no improvement or promotion is assumed. The complete AB source and
binary snapshot is preserved under `candidate-ab-source/` before E edits.
