# Algorithm selection before implementation

Decision date: 22 September 2026. Target: RTX A5000 Laptop, SM86, 48 SMs,
16 GiB, CUDA 13.4.59/C++23. This document selects experiments, not winners.
No GPU replacements described here have been implemented yet. The frozen hybrid
reference and its measurements are in
[the baseline report](../results/booster-trainer-20260922/REPORT.md).

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
