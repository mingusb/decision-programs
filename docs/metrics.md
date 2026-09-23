# GPU quality profiles and strict comparison

Preimplementation decision, 2026-09-23. Targets/predictions/optional weights are
resident binary64 arrays; callers perform any float conversion on GPU. Inputs,
result capacities, scratch and status are global allocations with exclusive
outputs and immutable inputs through tail completion. Each API runs in one GPU
coordinator thread and calls finish. There is no CPU metric, parsing, selection
or reference computation. File/CSV transport is not this resident metric API.
MetricTask distinguishes binary from one-output multilabel because the latter
retains per-output synthetic metric records. Dataset/model formats are separate
GPU codecs. The initial capacity bound is UINT32_MAX prediction cells, sufficient
for addressable resident binary64 predictions on the target SM86 GPU. Independent
ranking also checks `outputs*ceil(rows/256)<=INT32_MAX` before effects, matching
SM86's grid-x bound for the segmented scan. Capacities do not imply allocation
ownership; input, scratch, report records and status may not overlap.

## Frozen settings are separate profiles

Contract sources are the archived `training/tools/evaluate.py`, pinned
`results/booster-level-batch-20260922/real/evaluate.py`, NumPy 2.5.3 /
scikit-learn 1.9.1 environment and `training/tools/higher_order_campaign.py`.

Synthetic: nonnegative weights with positive finite sum; finite targets and
predictions; probabilities in [0,1]; labels exactly 0/1 or integral class indices.
Multiclass row sums use fsum, must pass absolute tolerance (<1), and accepted
rows are divided by their sum. Log loss clips the *correct-class probability*
to [1e-15,1-1e-15]. Regression uses exponent-scaled weighted MAE/RMSE; zero-weight
overflowing errors are excluded. Aggregate output means use the same scaled
operation, so subnormal values survive both row and output averaging. Binary
accuracy is inclusive at .5. Weighted AUC uses normalized class weights,
half-credit ties, compensated cumulative negative mass, then fsum of concordance
terms. Multilabel aggregate AUC is unavailable if any output AUC is unavailable.
Regression and multilabel retain every per-output metric.

Real data: weights are rejected because the frozen evaluator is unweighted.
Multiclass row sums must pass tolerance but are not renormalized; selected-class
probabilities clip to [1e-15,1]. Binary loss uses separately clipped p,
`y*log(p)+(1-y)*log1p(-p)`. Regression emits MSE/RMSE/MAE and uniformly averaged
per-output R2, including force-finite constant-target behavior. Multiclass emits
accuracy, macro F1 across all declared classes, and summed-per-row Brier.
Independent labels emit Hamming loss, exact match, scalar F1/AP/AUC or micro and
macro metrics. Macro AP includes positive-present outputs, macro AUC includes
both-class outputs; eligibility counts are report metadata, not quality scores.
Stable output-index ties define precision@1/3/5. Micro AUC is also explicit to
cover the accepted capability inventory. Undefined values are typed unavailable,
never silently converted to zero or discarded from comparison; AP with no
positives is the reference's zero. Real R2 with fewer than two rows is unavailable.

## Arithmetic and algorithm selection

[CPython math.fsum](https://github.com/python/cpython/blob/3.14/Modules/mathmodule.c)
uses error-free partials and final rounding. Actual synthetic fsum operands here
are nonnegative. Choose an exact base-2^26 integer accumulator covering the full
binary64 exponent range, followed by explicit round-to-nearest-even conversion.
With at most UINT32_MAX terms, each of 84 u64 limbs stays below 2^58 before carry.
This reproduces the relevant nonnegative finite-sum rounding and overflow
contract; it is not advertised as a generic signed-fsum replacement. Scaled-error
construction retains each frexp/multiply/divide/ldexp operation and the final
convex-maximum clamp. Intermediate FP64 arithmetic uses explicit RN intrinsics
where contraction would change the reference expression.

[NumPy's reduction source](https://github.com/numpy/numpy/blob/v2.5.3/numpy/_core/src/umath/loops_utils.h.src)
defines a distinct eight-accumulator/128-element pairwise sum. Preserve that
schedule for contiguous reductions and the slow-axis traversal for column
reductions, rather than replacing real metrics with fsum. Use a bounded explicit
stack, not runtime recursion. CUDA libdevice log/log1p and the frozen host/vector
math libraries require conformance evidence; matching formulas alone does not
establish bitwise transcendental agreement. No numerical discrepancy may be
waived by widening the zero gate.

[scikit-learn ranking](https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/metrics/_ranking.py)
defines tied thresholds, AP as the precision-recall step integral, and ROC
collinearity removal before trapezoidal integration. Preserve these actual
operations, including subtracting rounded recalls, rather than substituting an
algebraically equivalent AP formula.

Select an owned stable parallel merge-path sort of row indices, then an integer
segmented prefix scan of positives/tie ends and compact threshold records. Reuse
that storage for the pooled micro/signal ranking. Radix sorting is relevant for
fixed binary64 keys, but factoring the new quantizer while its validation is
running would couple independent gates; a separate radix backend is not dormant
production code. Merge partitioning costs binary searches plus linear merge
work per pass, O(N log N) traffic; stable ties need no floating reduction.
Per-output final arithmetic retains reference order; scalar compatibility
reductions can bottleneck few-output cases and must be included in timing.
Parallel integer accumulation is a later scheduling comparison if this bounded
first implementation is limited there. No O(N^2) production pair enumeration.

Research supplement: [GPU Merge Path](https://davidbader.net/publication/2012-gm-ba/2012-gm-ba.pdf)
provides independently partitionable equal-length merge ranges; the new kernel
uses eight outputs per partition. It sorts u32 indices and loads binary64 keys
indirectly, so coalescing and repeated merge-pass traffic need measurement.
[Onesweep](https://arxiv.org/abs/2206.01784) reduces radix digit-pass traffic using
single-pass prefix propagation. Its reported A100/u32 sorting wins do not rank
this SM86/binary64, segmented, tied-score workload. A fresh radix8 implementation
would need eight key-digit passes, stable index payloads, per-segment histograms
and a progress-safe prefix protocol. Compare it only as a complete ranking
operation, including scratch and launch costs, if merge traffic dominates.

Signal selection sorts pooled scores descending, never splits ties, and chooses
maximum TP satisfying exact integer 20*FP<=negatives, then minimum FP and highest
threshold. Predict-none starts at nextafter(max_score,+infinity) and wins if no
eligible group has a positive TP. Frozen mode consumes its supplied threshold
unchanged; held-out data never retunes it. Both modes report the .5 point too.

Strict gates compare finite candidate/reference values directly in their native
direction, avoiding overflowing subtraction. Signed zeros compare equal. Both
unavailable produces not-applicable; asymmetric availability, mismatched keys/
profiles/eligibility metadata, and nonfinite available values are errors.

## Required experiment

GPU-only fixtures pin weighted ties, subnormals, extreme error/weight exponents,
zero-weight overflow exclusions, normalization differences, AP/ROC ties and
collinear points, class absence, constant targets, stable top-k, threshold ties,
and adjacent/extreme/zero gate values. Tiny exhaustive ranking and integer/rational
oracles are independent of production sort/scan. Complete measurements include
validation, scratch setup, sort/scan, metric arithmetic and tail completion;
report few/many outputs, rows, ties, weights and scratch bytes. Root runs tests,
sanitizers and measurements serially. Compilation and conformance are separate
gates; no metric or algorithm is claimed complete/fastest before they pass.

## Compile-only receipt

Source and checks compile with `nvcc -std=c++23 -O3 -arch=sm_86 -rdc=true`, with
default contraction policy and explicit RN intrinsics at the required arithmetic
steps. The first source compile exposed a mixed double/float `nextafter` overload;
using a double infinity argument fixed it without relaxed-constexpr flags. Raw
logs, objects and hashes are in `/tmp/gh-metrics-compile` for root preservation.
The final checked suite has 35 metric cases, three gate cases, four signal cases,
and 1024 independent small partial-expansion summation oracles. Five-threshold
exhaustive ranking oracles cover real/synthetic and pooled/per-output paths;
large 257/1025/65537-row cases cover merge and scan boundaries analytically.

Ptxas reports zero spills in arithmetic kernels, but their explicit accumulator
and traversal storage uses 672–1632-byte local frames and up to 208 registers.
CDP coordinator functions spill parameters. These are measured code-generation
properties, not runtime performance evidence. GPU execution, sanitizer results,
frozen NumPy dispatch arithmetic and host-versus-libdevice transcendental
conformance are still pending; the tests do not establish those missing gates.
