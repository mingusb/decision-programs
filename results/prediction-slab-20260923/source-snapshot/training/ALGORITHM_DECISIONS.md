# Algorithm selection before implementation

Decision date: 22 September 2026. Target: RTX A5000 Laptop, SM86, 48 SMs,
16 GiB, CUDA 13.4.59/C++23. This document records selections made before their
implementation, not universal winners. GPU-resident preparation and tree
building now have implementation evidence in the
[resident report](../results/booster-resident-20260922/REPORT.md), followed by the
[root batching and split report](../results/booster-root-split-20260922/REPORT.md).
The original selection analyses below remain the rationale for those experiments.
The frozen hybrid reference and its measurements are in
[the baseline report](../results/booster-trainer-20260922/REPORT.md).
The current default promotion is recorded in
[DEFAULT_POLICY_DECISION.md](DEFAULT_POLICY_DECISION.md), with validation in
[the default report](../results/booster-defaults-20260922/REPORT.md). Earlier
experiment sections retain the policies and gates in effect when selected.

## Exact GPU feature preparation

The existing numerical contract is unusual: cuts have uniformly spaced ranks
among **distinct** finite values, not among rows or row weights. If U distinct
values produce I=min(U,max_bins-1) intervals, cut j uses distinct index
floor(j*U/I)-1. Categories require complete sorted enumeration. NaNs are missing;
positive and negative zero are one value. Any replacement must first respect
this contract, or explicitly establish a different modeling experiment.

| Candidate | Main benefit | Main cost or boundary |
|---|---|---|
| Keys-only Onesweep-family radix sorting, then dedup/cuts/encoding | General exact path, reduced main-array sorting traffic | Two key buffers, lookback/scan state, retained original input, encoding pass |
| Exact hash deduplication, then sort distinct keys | Sort work scales with distinct cardinality U | Random atomic probes, capacity/overflow handling; poor case can be U≈N |
| Key/row-ID sort with bin scatter | May avoid later value-search encoding | Doubles key payload and adds scattered output writes |
| Distinct extraction plus exact multiselection | Potentially avoids a full distinct-key sort for few cut ranks | Dedup still required; many requested cuts retain much partition work |
| Approximate samples/sketches | Less preprocessing/storage | Changes cuts and potentially accuracy; not an exact replacement |

The first two are the primary exact contenders. The
[Onesweep paper](https://arxiv.org/abs/2206.01784) supplies a strong sorting design;
its A100/historical-library performance is not an A5000 end-to-end quantizer
result. Installed CUB 3.4.2 already includes Onesweep dispatch, so its current
sort must be a benchmark reference rather than an assumed weaker historical
baseline. [Versioned reference source](https://github.com/NVIDIA/cccl/blob/v3.4.2/cub/cub/device/device_radix_sort.cuh).

Exact GPU hashing/deduplication has concrete implementation precedent in
[cuDF](https://developer.nvidia.com/blog/supercharging-deduplication-in-pandas-using-rapids-cudf/).
That is evidence for a candidate, not for this full quantization pipeline.
[SampleSelect](https://icl.utk.edu/files/publications/2019/icl-utk-1310-2019.pdf)
provides an exact selection/multiselection alternative whose applicability depends
on requested ranks and distinct cardinality.

For 32-bit keys with four 8-bit passes, Onesweep's main sorting-array traffic
estimate is 4*(2*4+1)*N = 36N bytes per feature. This excludes transpose, prefix
metadata, deduplication, final encoding, allocation and transfers. Carrying 32-bit
row IDs increases the corresponding estimate to 68N bytes. Those omitted costs
must be measured, not hidden behind a sorting-only result.

Feature tiling preserves each feature's complete distinct set. Independently
binning row chunks does not; row chunks need an exact union/merge. Compare shared-
memory tiled layout conversion with fused loading/encoding paths. Include the
4-byte float upload required by GPU fitting versus the baseline's 2-byte packed
bin upload. Measure both device-resident and host-Dataset-to-ready-device-bin
boundaries. [Transpose design evidence](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/).

Required tests: high cardinality, U=16/256/65535, repeated values plus rare tails,
categorical overflow, NaNs, all-missing/constant columns, ±0, irregular dimensions,
and memory-budget limits. Compare every cut, category and encoded bin against an
independent CPU reference. Freeze configurations before fresh-seed confirmation.

## GPU-resident tree building

Our histograms already process multiple active nodes. The repeated CPU dependency
is the winner download, stream wait, node/map construction and map upload at each
level. The first architecture experiment removes that dependency while preserving
scalar-tree split rules. Nodes, frontier counts, maps and decisions remain on the
device; host transfers serve setup and model export.

For 1/16/64 active nodes, compare a small stable block scan for frontier compaction
with direct fixed-slot mapping. Fixed slots avoid compaction but can waste dense
histogram work/memory. Device-wide row partitioning is a separate, larger scan
problem; do not impose a general scan on a tiny frontier.
[Decoupled-look-back scan](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back)
is relevant evidence for large scans, with roughly one input read and one output
write per element; its NVIDIA implementation remains benchmark-only.

Compare ordinary bounded graph execution with
[conditional graph control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
before selecting persistent cooperative workers. Evaluate graph construction,
replay, empty-frontier work, dynamic shared-memory requirements, occupancy and
forward progress. A host-free loop is not automatically the fastest loop.

The next histogram candidates are compact row partitions, batching across output
tiles, and smaller-child construction plus parent-minus-child subtraction.
Subtraction preserves integer counts within bounds but changes floating-point
rounding. Fixed-point accumulation changes arbitrary real gradients during
quantization, even if subsequent integer accumulation is exact. These require
separate numerical/quality experiments. The current
[XGBoost histogram code](https://github.com/dmlc/xgboost/blob/56f951e7419a6f66f4568865e1d7835bcb6dbbf1/src/tree/gpu_hist/histogram.cuh)
and [quantizer](https://github.com/dmlc/xgboost/blob/56f951e7419a6f66f4568865e1d7835bcb6dbbf1/src/tree/gpu_hist/quantiser.cuh)
are precedents, not local performance winners.

For 32 features ×64 bins and 24-byte statistics, dense histograms require
48 KiB/768 KiB/3 MiB per output at 1/16/64 nodes. Shared per-feature state is
1.5/24/96 KiB. Our current shared limit is 48 KiB. SM86's larger opt-in allocation
can accommodate the last case but can sharply limit block residency; benchmark
it instead of assuming the larger allocation wins.
[Hardware limits](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html).

## Large multi-output architecture

Compare three candidates before settling the model architecture:

1. Independent scalar trees, with bounded output tiles and batched device work.
2. Full-gradient vector-leaf trees, sharing row partitions across outputs.
3. Reduced-gradient split search with full-dimensional leaf values.

The first preserves independent partitions but repeats feature/assignment work.
Vector leaves share structure and counts; full statistics still grow with output
count. Reduced split gradients reduce that cost but approximate split scoring.
At 1,024 outputs, naive simultaneous scalar histograms for the preceding workload
would require 48 MiB/768 MiB/3 GiB; assignment storage additionally scales with
rows×output tile. Memory layout and batching are architectural choices.

[Current XGBoost vector-leaf CUDA results](https://xgboost.ai/2026/08/25/introducing-the-xgboost-vector-leaf-model)
support shared partitions as a serious candidate, with dataset-dependent
speed/quality tradeoffs. Its
[reduced-gradient interface](https://xgboost.readthedocs.io/en/stable/tutorials/multioutput.html)
separates split gradients from value gradients. The
[SketchBoost paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a36c3dbe676fa8445715a31a90c66ab3-Abstract-Conference.html)
provides approximate-scoring speed/quality evidence on its tested datasets;
retaining full leaf values does not make approximate split selection exact.

Use both compatible and conflicting output partitions, independent validation
and held-out test data, and time-to-matched-quality comparisons. Equal rounds
give different model capacities and are not enough. Include real high-dimensional
datasets before making NLP claims; the existing 4,096-output synthetic fixture
has only 64 held-out rows.

## Promotion gate

Retain raw timing arrays, executable/source identity, memory payloads, device
telemetry, dataset/split identity and failures. Run GPU measurements serially;
separate Nsight diagnostics from uninstrumented rankings. Measure the entire
operation, including clear/convert/scan/scatter/merge, and full training time.
Changes in precision, floating-point order, binning semantics or tree structure
must be explicit. The current tile-width comparison already fails an exact
zero-allowance per-output metric gate at floating-point rounding scale; it must
not be described as exact preservation.

## Initial statistics and dense validation: implementation experiment

Selected before implementation: fused GPU target validation and weighted sums,
with bounded chunk partials and a second reduction. Independent outputs use a
row-parallel scalar kernel for narrow matrices and a 32-output ×8-row block for
wide matrices, preserving coalesced reads instead of gathering one strided output
at a time. Compare narrow and tiled paths at their boundary. Multiclass uses
chunk-private class sums rather than scanning all rows once per class; work is
O(rows + chunks×classes), not O(rows×classes). Validate weights once in the same
preparation phase. No dense values are downloaded for CPU validation or means.

Scratch is 8×chunks×(outputs+1) bytes; default chunks are bounded by256. Weight
and target sums remain doubles. Two-stage reduction gives an explicit bounded
workspace and no device-wide spin barrier; a single atomic accumulator per output
would save partial storage but increase contention and nondeterministic ordering.
This selection follows the same owned warp/block reduction pattern already used
in the measured loss kernels. It is a candidate to measure, not a fastest claim.
GPU reduction order differs from the old CPU long-double sums, so compare against
an independent high-precision reference and retain any quality differences.

## Immutable root counts and independent root split batches

The next selection is recorded in
[REUSE_BATCH_EXPERIMENT.md](REUSE_BATCH_EXPERIMENT.md), before the corresponding
code changes. Build it in `build/booster-reuse`; retain all prior frozen binaries
and finalized evidence. The previous combined trace identifies deeper histograms,
small split grids and root atomic pressure as reasons to compare these candidates.
Those diagnostic observations are not end-to-end speedup predictions.

Root row membership and packed bins are invariant across outputs and boosting
rounds in the current trainer. Therefore one GPU uint64 histogram can provide
the exact count field for every root. The chosen experiment compares owned global
and shared setup kernels, then seeds each root statistics cache with those counts
and zero gradients/Hessians. Removing repeated integer atomic updates saves work;
the one-time count setup, seeding, retained cache and complete training must all
be measured. This does not reuse weighted Hessians or alter FP64 precision, and
it does not establish identical ordering of the remaining floating-point atomics.

Cached root statistics already use the node-major layout accepted by split
search. Interpreting outputs as independent nodes batches feature tasks without
changing each feature's arithmetic or deterministic tie rules. Candidate and
winner operations remain separate to retain feature-level parallelism. The
chosen design batches existing block/warp implementations, caches the resulting
root winners, and consumes each winner during tree initialization. It removes
the per-tree root Stats copy and split launches. Fixed-input fieldwise bit
comparisons, including fallback, signed-zero, cancellation, force-leaf and
changing-selector graph cases, test the split contract independently of atomic
histogram variability.

Public controls are `RootCountPolicy::{per_output,reuse_global,reuse_shared}` and
`SplitBatchPolicy::{per_tree,batched_root}`. Both changes require batched root
histograms. CLI equivalents are `--root-counts` and `--split-batch`; per-output
counts and per-tree splits remained defaults during this experiment. Setup and
cached winners remain on the GPU. Multiclass retains its pre-round derivative snapshot.

With T root outputs per tile, H total bins, F features and C frontier capacity,
counts add 8*H bytes to the histogram and device budgets. Batched split candidates
reuse the existing C*F buffer enlarged to max(C,T)*F. Cached winners require T
additional Split cells, giving incremental storage
48*((max(C,T)-C)*F+T) bytes under the device budget. The reported
`root_count_bytes` and `root_split_bytes` are included in the appropriate totals.
Budget failures reject the requested configuration rather than selecting an
unmeasured alternative.

The larger deeper-level alternative remains a separate primitive experiment:
batch output-specific row assignments and active-node histograms with owned
global or shared groups of 1/4/8 outputs. It includes clearing and merging and
preserves inactive capacity. Production integration would require retaining
multiple assignments, frontiers, trees and histograms at once, so a kernel-only
win would not justify that architectural change. In parallel, complete training
can compare the existing shared deeper path under the same batched-root controls.

The integrated implementation has passed the 11 CTest suites with the count probe
disabled. Complete-operation timings, Nsight explanations, applicable sanitizer
checks and strict held-out quality comparisons are separate evidence gates.
At this selection stage, the decision made no speed claim and promotion remained
pending. Baseline-versus-baseline repeats measure variability without relaxing
the zero-allowance gate or reclassifying previous failures.

## Current default promotion after measurement

The completed [reuse/batching report](../results/booster-reuse-20260922/REPORT.md)
supports promoting the combined root policy. The user explicitly accepts the
observed rounding-scale differences as practically immaterial for this decision.
The defaults are now `RootHistogramPolicy::batched`, `SplitPolicy::warp32`,
`RootCountPolicy::reuse_global`, and `SplitBatchPolicy::batched_root`.
Histogram auto selection, stream tree execution, output tile size 32, compact
export (batch size 0), and radix8 preparation retain their existing defaults.
The warp policy keeps the owned block implementation for shapes above its
32-feature/32-bin boundary; shared count setup and deeper batching primitives
are not promoted.

This is a policy promotion using measured evidence and the user's acceptance of
its numerical differences. Historical zero-allowance failures remain failed;
neither the evaluator nor a general numerical allowance changes. New speed or
quality claims still require their own evidence. Count-cache invariance requires
fixed bins and all-row roots, which is the current trainer contract. Future
sampling or row filtering must revisit that contract.

Root caches and retained winners consume the explicit budgets described above.
Changing these defaults can cause formerly fitting tight-budget configurations
to reject or overflow frontier capacity; the trainer does not silently truncate
trees. The complete legacy CLI/API combinations are documented in
[TRAINER.md](TRAINER.md). Use `build/booster-defaults` and new evidence under
`results/booster-defaults-20260922`; previous builds and finalized reports remain
frozen. The [promotion decision](DEFAULT_POLICY_DECISION.md) records the selected
validation before the change, and [its report](../results/booster-defaults-20260922/REPORT.md)
records the outcome.

## Output-tile tree construction experiment

The next implementation integrates the measured batched global deeper histogram
with independent batched split tasks and GPU frontier management. Selection,
contracts, memory planning and experiment gates were recorded before coding in
[LEVEL_BATCH_EXPERIMENT.md](LEVEL_BATCH_EXPERIMENT.md). A fused integer
scan/materialization handles frontiers through1024; larger frontiers retain a
batched scan/prefix/write sequence. Root cache copies disappear by writing roots
into retained per-output histograms. Per-output routing/prediction arithmetic and
all independent-tree objective semantics remain the reference.

This path is a separate `TreeBuildPolicy::output_batch`, initially explicit.
No broader fastest claim or default promotion follows from primitive timings.
The current combined root defaults and original counting defaults stay intact.
The output-batch `auto` histogram setting chooses the measured global candidate;
there is no per-output histogram calibration in this architecture.

### Integrated evidence and disposition

The integrated path passes 13 CTest suites and 13 applicable sanitizer checks.
The 56 serial synthetic runs measure 1.33–3.69x faster multi-output boosting and
1.25–3.32x faster complete training, with scalar boosting effectively unchanged.
Nsight Systems records 9,817 versus 655 boosting kernels for the same 387 trees;
no explicit host waits or device-to-host copies occur inside the tree scopes.
These observations are workload-specific; profiler durations are not rankings.

Keep `TreeBuildPolicy::per_output` as the default. Stronger arbitrary-run function
comparisons failed, including a repeated unchanged baseline with identical
training margins but a 0.05091173 held-out margin difference. The general trainer
check observed differences as large as 0.248774. Locally empty bins can allow
rounding-sensitive gain comparisons to select different thresholds. Baseline
variability does not turn a candidate's failed quality gate into a pass, and the
user's earlier root-default acceptance is not a blanket tolerance for this issue.
Mandatory exact-statistics fixtures, exported-model correctness, arbitrary-run
semantic diagnostics and zero-allowance quality gates remain distinct.

The completed real-data campaign adds 80 validation and 60 selected test runs.
Output batching improves complete training 2.59x on Letter and 1.74x on Delicious,
but XGBoost is 2.13x faster on Delicious. Matched custom comparisons fail 22/28
all-metric gates; an individual probability differs by 0.59643733. Repeated
unchanged per-output runs can also differ by 0.50581958. The largest preserved
cross-policy witness changes a positive label's prediction from 0.671177 to
0.074740. This is not covered by the previous rounding-scale acceptance.

The [complete experiment report](../results/booster-level-batch-20260922/REPORT.md)
contains raw timing/quality links, real-data comparisons and preserved failures.
The original counting implementation and preceding root defaults remain intact.
The [wide-feature split follow-up](WIDE_FEATURE_SPLIT_EXPERIMENT.md) documents an
unimplemented candidate for the 500-feature, low-bin workload: decouple the warp
candidate kernel from its current 32-feature winner restriction. It requires its
own exactness, quality, complete-operation and real-workload profiling evidence.

## Closed-form higher-order logistic experiment — 2026-09-22

The contract and candidate choice were recorded in
[HIGHER_ORDER_EXPERIMENT.md](HIGHER_ORDER_EXPERIMENT.md) before implementation.
CUDA C++23 orders 3 and 4 now compute stable binary logistic derivatives and
bounded Halley/Householder leaf proposals with same-order Taylor split scores.
They require explicit output batching and a positive leaf bound; independent
multilabel outputs are supported, coupled softmax tensors and arbitrary-order
automatic differentiation are not. This changes optimization and derivative
arithmetic; it is not an exact-preservation claim.

The [completed experiment](../results/booster-higher-order-20260922/REPORT.md)
contains 62 real-data fits, 27 measured synthetic fits, three synthetic warm-ups,
correctness/sanitizer checks and serial Nsight diagnostics. Main selected
Delicious macro AP improves slightly at clipped settings, while training takes
1.50x/2.07x longer for orders 3/4. MAGIC selected AP declines. Larger-cap checks
show mixed outcomes and preserve the clipping confound. Twenty of 22 main
matched comparisons fail the all-common-metric quality gate; 21 fail at least
one quality or signal gate. No default is promoted and all count code is intact.

Nsight identifies histogram accumulation and split scoring as approximately
99% of recorded kernel time on the full 983-label Delicious one-round profile.
Root reduction requests rise with the extra fields; split registers increase
72/90/96 and achieved occupancy falls from about 48% to 33%. The captured
split FP64 pipeline activity is 75–82%, with no observed spilling. These are
diagnostics, not profiler-based speed rankings. Removing unnecessary wide-feature
split work is a candidate for a separate measured experiment, not a claimed win.

## Wide split and ordered prediction candidates — 2026-09-23

Pre-code contracts are recorded in
[WARP_WIDE_SPLIT_EXPERIMENT.md](WARP_WIDE_SPLIT_EXPERIMENT.md) and
[FUSED_PREDICTION_EXPERIMENT.md](FUSED_PREDICTION_EXPERIMENT.md). The explicit
`warp-wide` policy connects unchanged warp candidates to the unchanged block
winner for wide feature sets. The explicit `fused_output` prediction policy
stably packs one frozen forest and preserves every per-output FP64 addition.
Counting code, training defaults and prediction defaults remain unchanged.

The ongoing [experiment report](../results/optimization-20260923/REPORT.md)
records19 passing CTest cases, exact primitive/frozen-model checks, sanitizer
receipts and312 unchanged baseline device code/constant sections. Nsight
confirms1,001→18 complete inference kernel launches on the983-tree Delicious
model. This is a reduction in work, not a profiler timing ranking.

All five independently trained split-policy pairs fail the strict quality and
signal gates; failed-label counts are112,84,131,121,117 of983 labels. An
identical-policy repeat also fails. Baseline variability does not waive the
candidate failures. Neither candidate is promoted. Gaming was confirmed during
exploratory timings; performance confirmation is deferred at the user's request
while correctness work continues. The interrupted full split benchmark remains
failed evidence despite successful bounded stream/graph reproductions.
