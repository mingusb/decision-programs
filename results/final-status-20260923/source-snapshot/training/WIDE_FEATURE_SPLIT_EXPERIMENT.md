# Wide-feature, small-bin split-search experiment

2026-09-22. **Analysis only: neither candidate below is implemented or measured,
and no dispatch/default is promoted by this document.** Record a new isolated
build and evidence directory before any implementation. Preserve the completed
output-batch experiment, its failures and the original counting implementation.

## Observed workload and current dispatch

The current real-data protocol includes all 500 binary numeric Delicious features
and all 983 independent labels. Sparse absent entries are observed zero, not
missing: see
`results/booster-level-batch-20260922/real/PROTOCOL.md:46` and `:56`.
The measured custom output-batch system did not outperform XGBoost on that
workload. This motivates profiling its actual path; it does not identify the
split kernel as the cause of the complete-training gap.

Under our numeric encoding contract (`training/include/ghb/booster.hpp`, Feature
comments), a nonconstant binary feature has bins for missing, observed zero and
observed one: only three bins. Constant numeric features need no more than two.
Verify the retained fitted metadata when recording the experiment. A requested
`max_bins` above three does not create extra distinct values in binary columns.

Current owned implementation references:

- `training/src/split_search.cu:195`: the batched warp entry point falls back when
  either `max_feature_bins > 32` **or** `columns > 32`. The direct and active
  entry points have the same guard at lines 177 and 186.
- `training/src/split_search.cu:98`: `warp_candidates` already maps each
  independent `(node,feature)` task to one warp CTA. Feature indexing at lines
  103–107 does not require total feature count to fit one warp. Its histogram
  bin count must fit 32 lanes.
- `training/src/split_search.cu:109`: the second reduction includes the original
  zero-warp contributions, preserving the existing FP arithmetic and signed-zero
  behavior. The explicit zero prefix carry at line 123 also remains significant
  to the exact-arithmetic contract.
- `training/src/split_search.cu:141`: `warp_winners` seeds the leaf result from
  feature zero, then loads only `candidates[node*columns + threadIdx.x]` once.
  It therefore covers at most 32 features. The launch helper couples candidate
  and winner dispatch at lines 164–169.
- `training/src/kernels.cu:343`: the owned block winner already supports larger
  feature counts using a strided load loop at line 355 and its existing block
  comparison reduction. Batched launch wiring is at lines 619–639.
- `training/src/split_search.cu:51` and `training/src/kernels.cu:262`: winner
  ordering is gain, then feature, threshold and missing direction. Valid split
  gains are finite and strictly above the configured nonnegative minimum.

Thus the total-feature-count restriction protects the current winner kernel,
not the per-feature warp candidate calculation. At `F=500`, it currently sends
both stages through the 256-thread implementation even when each feature has
only two or three bins. **Removing the guard alone would be incorrect:** the
current warp winner would silently omit candidates 32 through 499.

## Exact operation contract

Inputs are the existing GPU DataView and fixed output-major histograms
`Stats[B][C][H]`, per-output device active counts and optional device batch count.
Stats retain FP64 gradient/Hessian fields and exact uint64 row counts. Feature
offsets are valid, ordered and include missing bin zero; feature types and bin
IDs retain current meanings. Active/output masks clamp to the allocated shape,
with full, short and empty graph replays supported. Counts do not overflow and
numerical inputs obey the existing finite production contract.

Keep SplitConfig, force-leaf behavior, clipping, child-count/Hessian thresholds,
missing/categorical handling and tie rules unchanged. For an identical fixed
histogram, **every candidate and winner field must match the current owned block
reference bitwise**, including leaf fields, signed zero, all-invalid/force-leaf
cases and candidates at high feature indices. Inactive nodes and outputs remain
untouched. Histogram input is read-only.

Retain `candidates[B][C][F]` and `winners[B][C]`, 48 bytes per Split on the current
ABI. Retain two launches for the complete candidates-plus-winners operation,
existing stream ordering, no host synchronization, no allocation and no extra
atomics. Validate size products and signed feature indices. In particular,
removing the `F<=32` dispatch guard must add an explicit `F<=INT32_MAX` check to
the warp launch validation: the old guard implicitly guaranteed this bound,
while Split.feature is signed.

No histogram construction, reduction ordering, parent-benefit formula, model
architecture, objective or deterministic-tie policy is changed. CUDA C++23 and
owned algorithms only; no NVIDIA algorithm implementation or fallback enters
production. Original counting kernels/policies/defaults remain untouched.

## Candidates and expected tradeoffs

**A — decouple candidate and winner dispatch first.** For feature histograms
fitting 32 lanes, run the existing warp candidate arithmetic regardless of the
total feature count, then use the existing owned 256-thread winner reduction for
`F>32`. Keep the present path for its already supported shapes and the block
candidate fallback for larger feature histograms. Expose/reuse the owned winner
launcher without importing another implementation or changing its arithmetic.

This candidate preserves the existing global scratch layout, traffic boundary,
launch count and winner reduction. It removes candidate-stage block barriers,
shared reduction storage and seven largely unused warps for each tiny feature
histogram. It does not remove feature work, candidate writes or winner reads.
One-warp CTAs can also be limited by CTA residency; fewer threads alone do not
establish a win. Measure the complete operation.

**B — independently compare a strided warp winner.** Each lane visits features
`lane, lane+32, ...`, retains its best candidate, then runs the existing warp
winner comparison. Seed every lane's default leaf from feature zero as before.
The winner reduction copies/compares already computed Split values; it does not
sum FP statistics. The existing ordering can select the same valid candidate
regardless of comparison grouping, but this requires explicit proof and bitwise
tests, including the all-invalid leaf case. Do not conflate this reduction change
with candidate A.

For 500 features, each warp lane would inspect up to 16 candidates, while the
existing 256-thread winner inspects up to two per thread and uses shared-memory
barriers. Their memory traffic is similar; register pressure, parallelism,
barriers and small active-node grids determine the result. Neither is currently
ranked. Retain A's existing winner as the reference even if B is implemented.

Packing several independent feature warps into a CTA or fusing partial winner
selection could change residency and reduce intermediate traffic. That is a
separate later architecture experiment, with its own scratch/parallelism and
arithmetic analysis. Do not silently include it in A or B.

## Bounded validation and measurement plan

First compare the complete candidates-plus-winners operation on fixed,
preallocated histograms. Include actual Delicious-like tiny bins and boundaries:

- Feature counts 32, 33, 64, 255, 256, 257, 500 and 1,024.
- Maximum feature bins 2, 3, 16 and 32, plus 33 to verify the retained fallback;
  include irregular feature sizes and numeric/categorical/missing cases.
- Frontier capacities 1, 2, 8 and 16; output capacities 1, 3, 16 and 33; zero,
  partial and full active counts; full/short/empty graph selectors.
- Best and exactly tied candidates beyond lane 31 and thread 255; cancellation,
  signed-zero, force-leaf, all-invalid and clipping cases. Compare all fields,
  output guards, untouched inactive entries and read-only histogram guards.

Run serial correctness and applicable memcheck/racecheck/synccheck first. Preserve
all failures and source/binary identities. Time both stages together, including
all scratch reads/writes and their ordering; exclude only the same explicit
setup from every variant. Use warmups, alternating forward/reverse variant order,
repeated raw uninstrumented observations and stream/graph execution. A candidate
kernel-only timing does not rank the complete split operation.

Then run complete Delicious training with the existing fitted-input protocol,
all 500 features and all 983 labels, fixed rounds/depth/bin limits, memory budgets,
submission mode, exports and prediction boundary. Include quantization/setup,
retained memory, all training work and completed model handling in the established
whole-operation totals. Compare current production dispatch, A and B on the same
source/binary build where practical; do not compare against a changing baseline.
Include a small-feature control so changing dispatch cannot silently regress the
already supported path.

Keep the real-data split identities and tuning budget frozen. Use validation
fixtures for candidate/parameter selection. Do not inspect held-out test metrics
to choose feature thresholds, launch settings, datasets, seeds or stopping rules.
After selection is fixed, apply the existing test protocol once for the declared
confirmation; prior published test results are already known and are not a fresh
untouched selection set. Any new tuning based on those results needs an explicitly
new held-out confirmation plan, not a claim of fresh test generalization.

Exact fixed-histogram tests and the mandatory exact dyadic full-training fixtures
must pass. For general unordered-FP training, retain independent GPU/CPU model
prediction and per-round objective checks, strict zero-allowance metric audits,
baseline-repeat controls and the preserved stronger all-bin diagnostics. Do not
waive quality regressions because the baseline also varies, or claim identical
learned functions solely from exact split-stage inputs/outputs.

Use Nsight Systems/Compute on the **500-feature workload** to explain candidate
and winner launch shapes, dynamic active work, registers/spills, residency,
eligible-warps/stalls, shared-memory/barrier removal and global candidate traffic.
Existing 16-feature synthetic traces do not establish Delicious's bottleneck.
Profiled durations remain diagnostic; only uninstrumented complete-operation and
whole-training observations rank candidates. Promotion requires measured support
under the declared performance, memory, correctness and quality gates.
