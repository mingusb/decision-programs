# Explicit wide-feature warp candidate experiment

Recorded 2026-09-23 before implementation. This activates candidate A from
[WIDE_FEATURE_SPLIT_EXPERIMENT.md](WIDE_FEATURE_SPLIT_EXPERIMENT.md), under the
explicit `SplitPolicy::warp_wide` / `--split-policy warp-wide` selection. It is
not a default change or a measured promotion. CUDA C++23; no counting code,
histogram arithmetic, learned objective, model format, or numerical tolerance
changes are authorized by this experiment.

## Evidence and choice

The retained Delicious profile uses 10,336 rows, 500 mostly three-bin features,
983 outputs, tile 16, depth three. Order-2 histogram accumulation and split
candidates account for 70.737% and 28.212% of summed kernel durations respectively
(`results/booster-higher-order-20260922/REPORT.md:536`). The root split candidate
uses 256-thread blocks, 72 registers/thread and 81.76% FP64 pipeline activity.
These diagnostic shares identify a relevant cost, not an achievable speedup.

Current `split_search.cu` warp candidates already index arbitrary features, but
the associated warp winner loads only 32 features. The combined guard sends
wide-feature inputs through the full 256-thread candidate calculation. Candidate
A reuses the **unchanged existing warp candidate kernel**, then the **unchanged
existing block winner kernel** for features above 32. This removes unused warps,
block barriers and shared candidate-reduction storage for small feature
histograms without changing candidate arithmetic or global scratch boundaries.
Two launches, histogram reads, candidate stores and winner reads remain.

Alternatives remain separate: strided warp winners, packing feature warps into
one CTA, sharing parent arithmetic, histogram aggregation/subtraction, and
routing/prediction fusion. This experiment does not combine them. One-warp CTAs
can themselves hit the SM86 16-block / 48-warp residency ceiling; therefore fewer
threads is not a local performance conclusion.

Primary hardware reference: NVIDIA's [Ampere tuning guide, occupancy]
(https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#occupancy), which
distinguishes SM86 from A100/SM80. CUDA's [Programming Guide]
(https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) defines the
existing warp shuffle synchronization contract. The relevant implementation
references are our `training/src/split_search.cu` (`warp_candidates`,
`warp_winners`, `launch_warp`) and `training/src/kernels.cu` (`split_winners`).
No external algorithm implementation is imported.

## Exact input, output and device contract

Expose direct, active and batched `find_splits_warp_wide*` APIs with the existing
warp API signatures and layouts. Inputs are finite Stats fields (FP64 G/H,
uint64 count), valid ordered feature offsets/types/bin IDs, existing SplitConfig,
and direct or device active counts. Features must fit signed Split.feature;
reject columns greater than INT32_MAX before dispatch. Retain all extent and
configuration checks. Overflow does not launch a kernel.

For max feature bins <=32 and columns >32, use warp candidates plus block
winners. For columns <=32 retain the existing warp path exactly. For max feature
bins >32 retain the existing owned block fallback. `warp32`, `block256`, and the
default retain their current dispatch. Higher-order training rejects `warp-wide`
at configuration validation: extending HigherStats kernels is a separate choice.

For identical fixed histograms, every defined candidate/winner field must equal
the block reference bitwise, including signed-zero FP fields, clipping, numeric
and categorical thresholds, missing direction, force-leaf and invalid candidates.
Keep the original two-level zero-contribution sum and zero-prefix addition.
Histogram inputs and inactive outputs/nodes are untouched. Batched layouts remain
Stats[B][C][H], Split[B][C][F], Split[B][C]; active[B] and optional batch_count
clamp to allocated bounds. Full, short, empty and changed graph selectors work.
Pointers/scratch do not alias; all work is asynchronous on the caller stream,
allocation-free, host-wait-free, and uses no new atomics.

Expose a small internal block-winner launcher to connect the existing translation
units. It validates candidate/winner sizes and supports direct, active and
batched masks while invoking the existing kernel specialization unchanged.

## Validation and fair comparison

Extend exact split primitive checks to features 32/33/64/255/256/257/500/1024,
bins 1/2/3/16/32 plus 33-bin fallback, irregular numeric/categorical features,
high-index winners/ties, all-invalid/force-leaf, clipping, cancellation and
signed zero. Compare every field, prefix/suffix guards, inactive regions and
histogram immutability. Test direct/active/batched APIs, full/tail/empty and
changed graph masks, grid-stride execution and overflow/invalid calls.

Root builds and runs GPU workloads serially. Preserve failures and source/binary
identities; run applicable memcheck/racecheck/synccheck. Benchmark complete
candidate-plus-winner operations, including both launches and scratch accesses,
with warmups, alternating order, repeated raw timings and stream/graph submission.
Include small-feature and >32-bin fallback controls. End-to-end comparison uses
the same model/data/configuration, complete training boundary, output export,
memory accounting and quality metrics. Existing exact dyadic tests and independent
GPU/CPU objective/prediction checks remain mandatory. Arbitrary unordered-FP
training repeatability and strict zero-allowance quality failures remain distinct
and cannot be waived by passing the fixed-histogram stage contract.

Freeze validation-selection data and parameters before examining confirmation
quality. Previously viewed test data are not a fresh held-out selection set.
Use Nsight Systems for complete launch/stage behavior and Nsight Compute for
candidate/winner registers, barriers, occupancy, FP64 and stall counters. Inspect
generated code for unchanged arithmetic and for unintended spills. Profiler
durations explain results; only uninstrumented complete operations and whole
training can support promotion. No candidate is called fastest before those gates.

The existing `ghb_split_search_tests --bench-wide` entry point is the bounded
operation benchmark (normal invocation performs correctness only). It compares
block256 against warp-wide over 12 selected F/bin/output/frontier shapes, each
in stream and graph mode: F=33/257/500/1024, bins=3/16/32/33, B=1/16, C=1/8.
Each mode uses eight complete operations per sample, two unrecorded warmups and
seven recorded alternating-order samples per variant. JSON preserves every
device-event and synchronized-host observation. Both variants write separate,
equally sized candidate/winner buffers and are checked bitwise before and after
timing; histogram immutability and output guards are also checked.

Mandatory normal tests additionally compare complete serialized models and
per-round objectives on one-round exact-derivative fixtures for all three
objectives, both construction policies, numeric/categorical/missing inputs and
short output tiles. Independent CPU/GPU raw predictions are checked exactly on
those one-tree fixtures. Arbitrary-run nondeterminism is not inferred from them.
