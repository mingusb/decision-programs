# GPU output-tile tree construction

Status: complete. The new output-batch policy remains opt-in. Correctness and
evidence audits pass; the documented stronger semantic and quality gates retain
their failed status.

The new `TreeBuildPolicy::output_batch` constructs one tree level across a bounded
tile of independent outputs. The measured multi-output synthetic training stage
is 1.33–3.69× faster than the same binary's per-output path; complete training,
including setup, is 1.25–3.32× faster. Scalar training is effectively unchanged.
These are two-sample medians for the recorded workloads, not general speed bounds.

The new policy remains explicit. Strong semantic comparisons exposed discrete
split instability in both the existing trainer and the batched trainer. They do
not establish preservation of the learned function for all legal inputs. The
previously promoted root defaults and measured counting implementation are
unchanged; the user's earlier acceptance of rounding-scale metric variation is
not extended to these newly observed differences.

## Implementation and scope

CUDA C++23 production code retains GPU-resident feature fitting/binning,
derivatives, assignments, histograms, split decisions, frontier/tree construction
and prediction. Host activity supplies setup, launch submission and completed-tile
model export. External boosters and their CPU validation/prediction references
are isolated benchmark programs, not production algorithms or fallbacks.

Independent output trees and pre-round multiclass derivatives are preserved.
Roots write directly into retained strided histograms; root cache copies and
separate cached winners disappear in this path. Histograms, split searches,
integer frontier scans, materialization, routing and prediction operate across
the output tile, with active counts and full/tail selection read on the GPU.
Frontiers through 1,024 slots use the fused scan/materialization path; larger
frontiers retain the staged scan/prefix/write sequence. No CUB or Thrust algorithm
is introduced.

Retaining multiple assignments/frontiers/histograms consumes additional storage.
The planner accounts for all owned allocations and reduces tile width before
frontier capacity. If the resulting frontier is too small for the actual tree,
training fails explicitly instead of silently truncating the model. Full/short
tiles, compact derivatives, graph reuse, memory budgets and bounded exports are
validated. Automatic deeper-histogram selection uses the measured owned global
implementation in this new architecture.

The pre-code candidate record is
[LEVEL_BATCH_EXPERIMENT.md](../../training/LEVEL_BATCH_EXPERIMENT.md), including
the checked primary implementation sources and hardware applicability. Detailed
implementation and validation evidence is in [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Uninstrumented synthetic measurements

All 56 runs use the same production binary, alternate comparison order, and
execute serially on the RTX A5000 Laptop GPU (SM86). Seven shapes cover scalar,
33 deep outputs, 129 outputs, 1,024/4,096 binary outputs, 17-class softmax and a
40-feature/256-bin block-split fallback. Both stream and graph submission are
included. Profilers and CPU quality audits run outside ranked timings.

| Shape | Stream training speedup | Graph training speedup | Stream complete-training speedup | Graph complete-training speedup |
|---|---:|---:|---:|---:|
| Scalar | 1.006× | 1.000× | 1.078× | 1.029× |
| 33 deep outputs | 2.454× | 1.791× | 1.868× | 1.435× |
| 129 outputs | 2.882× | 2.398× | 2.350× | 2.033× |
| 1,024 binary outputs | 3.694× | 2.770× | 3.323× | 2.511× |
| 4,096 binary outputs | 3.645× | 3.339× | 3.151× | 2.865× |
| 17 classes | 3.003× | 1.831× | 1.931× | 1.369× |
| 33 outputs, block fallback | 1.355× | 1.334× | 1.268× | 1.255× |

[The synthetic audit](quality-synthetic/summary.md) preserves every raw pair,
metric and strict gate. Two observations per cell describe these runs; they do
not provide confidence intervals or demonstrate a universal ranking.

## Correctness and numerical limits

All 13 CTest suites and 13 applicable Compute Sanitizer checks pass. The sanitizer
checks cover memory, races and synchronization in the new/extended primitives,
plus full-trainer memory checking. Mandatory exact-statistics fixtures pass
zero-tolerance cross-policy model/function certificates for regression, binary
and multiclass learning. General fixtures validate each exported model against
independent CPU traversal and weighted objective references, including every
boosting round. These checks validate the implementation's actual contract.

The stronger arbitrary-run semantic diagnostic remains **failed**. Tiny changes
in unordered FP64 sums can select different thresholds across an empty local
training bin. A preserved unchanged-baseline repeat has identical training
margins but a held-out margin difference of **0.050911730395435617**. Batched versus
baseline training reproduces the issue. The general contract run also records a
maximum cross-model held-out margin difference of **0.24877399999999994**; this is
an observation, not a bound or merely rounding noise.

The synthetic held-out rows have much smaller cross-policy differences: maximum
saved prediction difference about 1e-15, metric deterioration at most 3e-16, and
no changed classification decisions. Nevertheless, 17/28 cross-policy strict
quality gates and 16/28 within-policy repeat gates fail with zero allowance.
All 56 independent model/prediction checks and model-versus-constant-baseline
quality gates pass. None of those outcomes overrides another gate.

For the deep synthetic models, a concrete legal raw float32 input produces a
**0.04660918** cross-policy margin difference; another witness reproduces that
scale in an unchanged-baseline repeat. The maximum computed cross-policy
all-bin margin bound is 0.08898427. The witnesses are retained in
[deep-region-witness.md](quality-synthetic/deep-region-witness.md), including
model hashes, feature values and paths. This is why favorable ordinary held-out
metrics do not justify an all-input equivalence claim or automatic promotion.

Original failed tests, binaries, fixtures and model pairs are retained. The
[validation contract correction](../../training/BATCH_TRAINING_VALIDATION.md)
explains the distinct mandatory correctness suite and still-failing stronger
diagnostic. No gain tolerance, split tie change, deterministic reduction,
objective change or relaxed quality threshold was introduced to make a failure
pass.

## What Nsight explains

The [Systems audit](systems-audit.md) matches 1,442 emitted samples to NVTX scopes.
For the same 387 output trees, batching reduces boosting kernel launches from
**9,817 to 655** and graph enqueues from **387 to 27**. Both paths have 27 completed
export scopes/waits. Every traced tree scope contains zero explicit host waits,
device-to-host copies and device allocations/frees. The comparison includes
root work outside the old per-tree scopes, preventing an unfair boundary choice.

The [Compute audit](COMPUTE.md) identifies remaining candidates, not new wins:

- The selected global deeper histogram has 72.99% occupancy but 94.64% of
  scheduler cycles with no eligible warp; scattered accesses/cache latency merit
  investigation. Its 72% excessive theoretical global sectors are not measured
  external DRAM bytes, and DRAM bandwidth is not saturated in this capture.
- One warp per split block limits occupancy through the SM86 block ceiling.
  The measured short-scoreboard pressure is not evidence of shared bank
  conflicts: this kernel performs no shared loads or stores. Packing independent
  feature warps and reducing dependency chains are separate candidates.
- Frontier materialization is a small operation: 16 four-thread blocks on 48
  SMs. Removing compatible work/launches is more promising than inflating work
  merely to raise occupancy.

These are first-matching-kernel diagnostic captures. Their multipass durations
are not used to rank implementations.

## Real-data comparison

All **80 validation jobs and 60 selected test jobs completed successfully**. Each implementation
received four predefined configurations per dataset, selected on validation
loss only, followed by three test repetitions in alternating order. No test
score influenced configuration selection. All 983 Delicious labels are retained.

The references are XGBoost 3.4.1, LightGBM 4.7.0 built with CUDA for SM86, and
CatBoost 1.2.10. These are native framework comparisons, with distinct binning,
objectives, regularization and tree architectures. Complete public training
includes preparation, upload/binning and fitting; loading, imports and initial
CUDA context setup are excluded uniformly. No training warm-up is performed.

Median complete training time, seconds (all three samples and ranges are in
[the full real-data report](real/summary/REPORT.md)):

| Dataset | Per-output | Output-batch | XGBoost | LightGBM | CatBoost |
|---|---:|---:|---:|---:|---:|
| Wine | 0.0508 | 0.0510 | 0.2258 | 1.1510 | 0.8838 |
| MAGIC | 0.0687 | 0.0628 | 0.2952 | 1.1425 | 0.9023 |
| Letter (26 classes) | 1.2526 | 0.4838 | 3.9318 | 36.3872 | 1.5696 |
| Delicious (983 labels) | 49.2720 | 28.3867 | 13.3433 | 91.0095 | 9.8783 |

Median test loss; smaller is better. Wine uses MSE; the other datasets use
natural-log loss. The unchanged evaluator also reports all applicable
classification, ranking, regression and multilabel metrics in the raw JSON.

| Dataset | Per-output | Output-batch | XGBoost | LightGBM | CatBoost |
|---|---:|---:|---:|---:|---:|
| Wine | 0.486619266 | 0.48726513 | 0.500257671 | 0.494804301 | 0.49038766 |
| MAGIC | 0.315845961 | 0.315817439 | 0.316585129 | 0.309261429 | 0.330468828 |
| Letter (26 classes) | 0.325142393 | 0.326060192 | 0.325123343 | 0.161326784 | 0.608241285 |
| Delicious (983 labels) | 0.0677669255 | 0.067767558 | 0.0677635688 | 0.0671238882 | 0.23482477 |

Output batching is **2.59x faster** than the per-output path on Letter and
**1.74x faster** on Delicious; scalar performance is broadly similar. The new
path is faster than the three external trainers on the first three datasets in
this grid. On Delicious, **XGBoost trains 2.13x faster** than output batching and
has slightly lower test log loss. LightGBM has lower test log loss on MAGIC,
Letter and Delicious. These outcomes do not establish general superiority.

CatBoost is quickest on Delicious but has much worse loss under this short
5/10-round budget. Its saved multilabel parameters use `boost_from_average=false`,
whereas our trainer initializes from label means. Its shared symmetric vector
leaves also differ from independent scalar trees. The campaign does not isolate
those causes or establish converged CatBoost quality. LightGBM retains every
label through a fully timed sequential binary-relevance wrapper. CUDA backend
evidence beyond requested flags is in [LIGHTGBM_BACKEND.md](real/LIGHTGBM_BACKEND.md).

Public prediction on Delicious takes median **169.58 ms** for output batching,
**163.61 ms** for the per-output models, and **419.30 ms** for XGBoost. The public
API includes input/model preparation and returned predictions; this is not a
resident-kernel-only latency. Native framework conversion to the saved float64
artifact occurs after its prediction timer; our API already returns double.
LightGBM and CatBoost multiclass/multilabel predictions use explicitly labeled
CPU reference APIs, and do not rank GPU inference kernels.

The 28 matched custom configuration/repetition pairs have **22 failed all-metric
zero-allowance gates** and **17 failed loss-only gates**. Maximum individual
prediction differences are 0.1794409 regression units on Wine and probability
differences of 0.23553372 on MAGIC, 0.25709104 on Letter and **0.59643733** on
Delicious. Close aggregate losses therefore do not establish preservation for
individual predictions or labels. These observations reinforce the decision to
keep the new path explicit.

The [separate repeat audit](real/repeat-audit.md) compares test repetition zero
against repetitions one and two within each custom policy: 16 pairs, all with
changed topology, 10 failed all-metric gates and four failed loss-only gates.
Delicious probability differences reach **0.50581958 within the per-output path**
and **0.29054072 within the batched path**. This documents variability; it does
not excuse any cross-policy regression or establish its cause.

The largest cross-policy witness is Delicious validation row 2446, label 965
(`TAG_wikipedia`), with actual target one. Per-output predicts
**0.6711769262937208**, while output-batch predicts **0.07473959381333392**.
This label has 109 positives among 10,336 training rows. The complete matched
pair changes 25 threshold decisions across 23 validation rows. The audit retains
the fixture, prediction, model and capture hashes and independently checks the
saved values. This is a material probability/decision change, not a raw-margin
rounding artifact.

A separate sampled-memory campaign reruns the selected Delicious configurations.
These timings are excluded from rankings. The requested sampling interval is
100 ms plus query overhead. Device-wide values include driver/context/other
process allocations and can miss short peaks; they are not process allocator
peaks or directly comparable to owned payload accounting.

| Implementation | Before run, MiB | Sampled peak, MiB | After run, MiB |
|---|---:|---:|---:|
| custom-per-output | 1068 | 1804 | 1306 |
| custom-output-batch | 1306 | 1679 | 1319 |
| xgboost | 1319 | 1978 | 1352 |
| lightgbm | 1352 | 1742 | 1328 |
| catboost | 1328 | 15811 | 1161 |

The custom batched trainer reports 139,530,344 owned device payload bytes for
this selected workload. That excludes CUDA allocator/driver/context bookkeeping
and must not be presented as a framework-comparable device peak.

The real-data provenance audit verifies all 140 cases, 700 captured artifact
hashes, 680 frozen source hashes, 280 fixture identities, 56 custom binary
identities and 60 cached-evaluator receipts. CPU quality evaluation always runs
after the GPU process exits. The test-only cache reuses only an immutable
training-mean baseline from the unchanged evaluator, with four original misses
and 56 audited exact hits; candidate metrics are recomputed unchanged every time.
All 13 earlier capability-smoke failures remain retained. See
[PROTOCOL.md](real/PROTOCOL.md), [summary.json](real/summary/summary.json) and
[the cache audit](real/test-cache-audit.json) for complete scope and provenance.

## Decisions and next experiments

Keep the new policy opt-in while characterizing and addressing split instability.
The failure also occurs in the baseline, so a fix must address the shared
statistical/split contract rather than assume that output scheduling alone caused
it. Empty-bin split canonicalization and deterministic/reproducible statistics
are distinct candidates whose semantic and performance costs must be recorded
before implementation.

One workload-specific follow-up is documented in
[WIDE_FEATURE_SPLIT_EXPERIMENT.md](../../training/WIDE_FEATURE_SPLIT_EXPERIMENT.md).
Delicious has 500 binary features with very few bins, but the current warp split
dispatch falls back when feature count exceeds 32. That guard protects the
single-load warp winner; the per-feature warp candidate does not require it.
The first candidate pairs the existing warp candidates with the existing wider
block winner. A strided warp winner is a separate candidate. Simply deleting the
guard would silently ignore higher-index features. Neither candidate is
implemented or measured, and the 16-feature profiler capture does not establish
the bottleneck on the 500-feature dataset.

Other performance candidates are histogram layout/atomic-traffic reduction,
multiple independent feature warps per split block, prediction updates when
routing reaches a leaf, and resident whole-forest inference. Each needs its own
contract, pre-code comparison and complete-operation measurements. In particular,
prediction fusion must read the already-rounded stored leaf value to avoid
silently introducing a fused multiply-add or changing tree-order rounding.

## Reproduction and preservation

The counting preservation check verifies all 12 originally frozen files/binaries.
Both preceding experiment artifact seals and their frozen source/build manifests
also verify without differences. Commands, stdout/stderr, failed attempts,
profiler reports, models, fixtures, predictions and quality results are retained
in this directory. The final source/build snapshot is `final-provenance/`, with
its `manifest.sha256`; the complete evidence manifest is `artifacts.sha256`.
`seal-verification.json` records successful verification, file counts and manifest
digests. The seal excludes itself, its own receipt and transient Python bytecode.
