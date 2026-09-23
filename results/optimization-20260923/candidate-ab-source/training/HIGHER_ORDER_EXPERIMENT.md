# Safeguarded higher-order binary boosting experiment

2026-09-22. This contract is recorded before the algorithm edits. The user has
authorized implementing the higher-order experiment. It is an experimental
algorithm change, not an arithmetic-preserving replacement or a fastest-code
claim. Preserve the measured order-2 implementation, defaults, counting kernels,
previous reports, failures and saved models.

## Scope and exact contracts

The initial higher-order path supports binary logistic loss, including independent
multilabel outputs, with derivative order 3 or 4. It requires output-batch tree
construction, global/auto histogram policy and a finite positive
`max_leaf_value`. Unsupported higher-order configurations reject explicitly;
there is no silent framework or CPU training fallback. Squared-error and
multiclass-softmax objectives reject orders greater than 2 in this experiment.
Squared-error derivatives above order 2 vanish; softmax requires a separate
coupled-objective contract. Every existing order-2 path remains available.

Differentiate the smooth per-row loss with respect to its scalar prediction
margin, with targets, nonnegative row weights, binned features and tree partitions
held fixed. Binary targets retain the existing exact 0/1 validation. Do not
differentiate integer bins, discrete split selection, routing or the whole
training program.

For each active output/node/bin, accumulate FP64 G, H, T and (order 4 only) Q,
which are weighted sums of the first through fourth loss derivatives. Counts are
exact unsigned 64-bit values with existing row-count semantics. T and Q are
signed: never clamp them on addition or subtraction. As in the current
implementation, subtraction clamps H at zero to handle roundoff. FP64 atomic
addition remains unordered; neither bitwise reproducibility nor model-function
equivalence across training runs is promised.

Use `HigherStats<Order>` with fields `double gradient, hessian`,
`double extra[Order-2]`, and `unsigned long long count`; extra[0] is T and
extra[1] is Q. The derivative helper returns count zero; histogram construction
assigns/counts rows. Histogram layout remains output-major
`[batch_capacity][node_capacity][total_bins]`; active masks, tail masks and
buffer lifetime rules are inherited from the output-batch contract.

For a margin z, target y and weight w, let u=exp(-abs(z)), v=1/(1+u).
Compute p and q=1-p without subtracting nearly equal values: for z>=0 use
p=v, q=u*v; for z<0 use p=u*v, q=v. With c=u*v*v, use

```
G = w * (y == 1 ? -q : p)
H = w * c
T = H * (1 - 2*p)
Q = H * (1 - 6*c)
```

Zero weight yields zero derivatives. These formulas are the analytical
derivatives of stable logistic loss, evaluated in floating point; they do not
make underflow or rounding disappear. Higher orders use unfloored H. This and
stable saturated-margin evaluation differ from the retained order-2 code's
`max(p*(1-p),1e-16)` Hessian and `p-y` residual. The current softmax path uses a
diagonal upper bound `2*p*(1-p)`, not an exact full Hessian, and is not extended
by appending these scalar statistics.

## Leaf proposals, safeguards and consistent split scoring

Let A=H+lambda. If G or A is nonfinite, or A<=0, return the zero leaf with zero
benefit. Form the Newton proposal n=-G/A, clipped to
`[-max_leaf_value,+max_leaf_value]`. A nonfinite raw proposal can be clipped to
the finite interval, but the selected value and its score must be finite.

At the current margin, one Halley proposal using loss derivatives through order
3 is

```
s3 = -2*G*A / (2*A*A - G*T).
```

One fourth-order Householder proposal using loss derivatives through order 4 is

```
s4 = -3*G*(2*A*A-G*T) / (6*A*A*A-6*G*A*T+G*G*Q).
```

These follow by applying Householder root finding to the derivative of the
regularized leaf loss. They are single local proposals, not exact minimizers of
an unrestricted cubic/quartic. To avoid unnecessary powers of A, evaluate with
the raw ratio x=G/A, r=x*(T/A), and q=x*x*(Q/A):

```
s3 = -x / (1 - r/2)
s4 = -x * (1 - r/2) / (1 - r + q/6).
```

The normalized denominator and intermediates must be finite, and the denominator
must exceed 1e-12. Reject nonfinite or non-descent higher-order proposals, then
clip to the same finite interval. T=Q=0 must reduce to the clipped Newton result.
There is no iterative solver, halving loop, curvature gate or extra row-wise
acceptance pass in this initial candidate.

Evaluate zero, clipped Newton and the valid clipped higher-order proposal using
the same regularized Taylor benefit at the requested order:

```
B3(s) = -(G*s + A*s*s/2 + T*s*s*s/6)
B4(s) = -(G*s + A*s*s/2 + T*s*s*s/6 + Q*s*s*s*s/24).
```

Select the greatest finite nonnegative benefit; choose Newton on a benefit tie
with the higher-order proposal. If neither nonzero proposal has an acceptable
score, select zero. Parent and child leaves all use this same evaluator, and
split gain is `B(left)+B(right)-B(parent)`. Preserve existing minimum-count,
minimum-Hessian and minimum-gain requirements and feature/threshold/missing-side
tie rules. Do not substitute an order-2 optimal-gain formula for the benefit of
a higher-order proposal.

As in the existing trainer, these are unshrunk leaf values and scores. The
materialized node stores `learning_rate*s`; consequently `max_leaf_value` bounds
the unshrunk step, and the actual margin increment is bounded by
`learning_rate*max_leaf_value`. This is not a changed learning-rate convention.

A cubic with nonzero T is unbounded on one side without a finite domain. A
quartic can also be unbounded: logistic Q=-w/8 at p=1/2. The finite interval and
candidate checks bound this experiment and ensure nonnegative *surrogate*
benefit; they do not guarantee actual training-loss decrease or better held-out
quality. Record actual per-round losses and preserve adverse results.

## Candidate choice and alternatives considered

The existing implementation already spends substantial work in histogram
accumulation, split evaluation and launch/frontier management. Reuse its output
batching and GPU-resident state, with compile-time order specialization and no
new host training stage. Add the derivative fields to the same histogram pass;
do not create a separate pass/kernel for each derivative. Root count reuse,
output masks and existing integer counts retain their contracts.

For these known scalar losses, fused analytical formulas reuse one exponential
and the common probability/curvature terms. This is the selected first
implementation candidate, not an automatic-differentiation frontend. A templated
Taylor-jet loss evaluator is a future candidate for custom losses; it would
propagate r+1 coefficients per live value with typically O(r^2) primitive work,
potentially increasing registers and spills. Compiler-generated derivatives
are another candidate, but require separate toolchain and numerical validation.

Primary sources checked for this decision:

* [Pachebat and Ivanov, High-Order Optimization of Gradient Boosted Decision Trees](https://arxiv.org/abs/2211.11367):
  GPU binary-cross-entropy experiments motivate testing time to quality. The
  corrected fourth-order denominator above uses G squared times Q; printed
  Equation 14 omits that square. Derive split benefits directly rather than
  transcribing the paper's polynomial simplifications. Its measurements do not
  establish our hardware/workload ranking or improved detection metrics.
* [Dangel et al., Collapsing Taylor Mode Automatic Differentiation, NeurIPS 2025](https://arxiv.org/pdf/2505.13644):
  directional Taylor coefficients avoid explicitly materializing complete
  derivative tensors. Its PDE experiments are not a local booster ranking.
* [Moses et al., GPU differentiation through Enzyme, SC 2021](https://c.wsmoses.com/papers/EnzymeGPU.pdf):
  establishes compiler AD of CUDA device code; it does not establish compatibility
  or fastest order-4 derivatives for this CUDA C++23 build.
* [Clad CUDA documentation](https://clad.readthedocs.io/en/latest/user/UsingCladOnCUDACode.html):
  supports differentiating CUDA functions but documents shared-memory and
  synchronization restrictions; differentiating the whole histogram is not the
  selected approach.
* [Grapiglia and Nesterov, Tensor Methods for Minimizing Convex Functions](https://optimization-online.org/wp-content/uploads/2019/04/7181.pdf):
  high-order optimization uses regularized Taylor models and sufficient-decrease
  conditions. Its guarantees do not automatically apply to greedy tree building
  or to the single safeguarded proposal specified here.

The formulas above are direct algebraic derivations. In particular the
fourth-order proposal can be independently reconstructed from the reciprocal
power series of F(s)=G+A*s+T*s^2/2+Q*s^3/6: if 1/F(s)=sum b_j*s^j,
Halley gives b1/b2 and fourth-order Householder gives b2/b3.

Other alternatives are repeated exact-loss Newton refinement of fixed leaves,
true-loss acceptance/backtracking, higher-order scoring only for a shortlist of
splits, and retaining ordinary splits while refining final leaves. Those can
avoid expanding all histograms but require additional row passes or a distinct
selection experiment. They are not silently folded into this candidate.

## Work, memory and experiment

For batch width B, frontier capacity C, total feature bins J and N rows:

| Payload/work | Order 2 | Order 3 | Order 4 |
|---|---:|---:|---:|
| Histogram cell bytes | 24 | 32 | 40 |
| Histogram allocation | 24BCJ | 32BCJ | 40BCJ |
| Derivative scratch bytes | 16NB | 24NB | 32NB |
| Floating-point fields accumulated | 2 | 3 | 4 |
| Count fields accumulated | 1 | 1 | 1 |

Root counts already reused do not gain extra count atomics. New fields increase
clear/store/read traffic, FP64 atomic updates, split-scan traffic and register
use. Root/deeper scratch planning must account for actual order-specific sizes;
a fixed memory budget may reduce B or C. Table ratios are not runtime predictions.
Original counting kernels and order-2 structures are not enlarged.

The new histogram kernels retain the measured subgroup widths (1/2/4/8/16/32)
for coalesced derivative loads and shared feature-bin loads across outputs.
Split dispatch retains the current 32-bin/32-feature warp boundary, with the
256-thread implementation otherwise; the separate wide-feature dispatch
experiment is not included. Launch counts and GPU-resident frontier work remain
the same, allowing a comparison of the higher-order algorithm in that architecture.
Before final validation, source review selected removal of the extra all-zero
second warp reduction for higher-order one-warp candidates. It has no useful
summands; the new algorithm has no signed-zero identity contract against order 2.
The frozen order-2 arithmetic retains that reduction. This is an explicit small
implementation difference in the optimizer comparison, not isolated derivative
arithmetic timing.

Validation precedes performance claims. Independent CPU references check stable
loss derivatives, factorial conventions, reciprocal-series Householder algebra,
zero/degenerate curvature, denominator rejection, bounded proposals, score ties,
signed T/Q and consistent split scores. GPU tests then cover derivatives,
histograms, splits, guards, tails, graph replay, scalar and multilabel training,
saved-model prediction and applicable sanitizers. Use analytically exact/dyadic
fixtures where possible and explicit floating-point tolerances for numerical
oracles; never change the zero-allowance quality evaluator to make a failure pass.

Run GPU measurements serially in an isolated evidence directory, retaining
commands, build/source hashes, raw samples, memory payloads, losses, models,
predictions and failures. Compare orders 2/3/4 with the same positive
`max_leaf_value=1` as a clipping control, and include a separately labeled
unchanged unclipped order-2 baseline. Keep initialization, dataset membership,
binning, objectives, tree budget and other parameters fixed for paired tests.
Also disclose the higher-order path's stable/unfloored derivative difference;
matching the clipping control does not isolate that derivative change by itself.

Use scalar binary and wide multilabel synthetic fixtures, then the fixed real
MAGIC and Delicious train/validation/test splits. Compare complete training and
time to held-out quality with equal tuning allowance, not only derivative-kernel
speed or training loss. Include all applicable log loss, Brier, accuracy and
ranking metrics, every strict failure, and repeated-run variability. Profile
after uninstrumented timing to explain derivative arithmetic, FP64 atomics,
memory traffic, register pressure/spills, occupancy and launch counts. Profiler
durations do not rank the implementations. No default promotion or universal
speed/accuracy claim follows from implementing the candidate.
