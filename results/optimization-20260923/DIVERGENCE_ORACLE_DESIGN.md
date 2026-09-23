# Saved-model divergence oracle

2026-09-23. This is an explicit CPU validation reference, not a production
training implementation. No CUDA calls, training runs, or timing comparisons.
Heavy computation waits until the root agent finishes isolated performance runs.

## Contract and bounded selection

Read the unchanged Delicious train/validation fixtures and three one-round,
depth-3 models: `split-campaign/pair0-warp32`, `pair0-warp-wide`, and
`pair1-warp32`. Require binary independent outputs, unit row weights, 500
features, 983 outputs, identical fitted quantization, the captured order-2
configuration, and exactly one tree per output. Preserve artifact, source,
fixture and base-score bit identities. Existing exported models and metrics
remain untouched.

Examine at most three outputs: output 573, where the previous paired audit
observed the maximum probability change at validation row 526; the earliest
output with a root decision difference, if any; and the earliest remaining
output with a structural difference. Find the first differing node in breadth
first order for each selected output and comparison. Reconstruct membership
using the actual saved ancestor decisions. Only call two node inputs matched
when those ancestors and their row memberships agree.

Selection clarification before score computation: "earliest" uses the primary
pair0 warp32 versus pair0 warp-wide comparison; the baseline repeat is examined
on the same selected outputs. Source-only inspection selects outputs 573, 87
(earliest root difference), and 8 (earliest remaining decision difference).
Decision identity uses feature, threshold, missing direction and leaf status;
physical child-array indices alone are not a semantic decision difference.

## Independent numerical reference

`training/src/kernels.cu` initializes every row/output to its saved base margin.
In this first independent-logistic round, each output consequently has one
probability p, two label-dependent gradient values, and one Hessian. There is no
row weighting in the native fixture loader. For a node containing n rows and k
positive targets, a mathematical high-precision reference therefore uses
G = n*p-k and H = n*max(p*(1-p), 1e-16). Obtain p from the exact stored binary64
margin, then evaluate the sigmoid at 80 decimal digits. Separately evaluate an
exact integer-count sum of host-binary64 derivative values, explicitly named a
host reference and never claimed to reproduce the unrecorded CUDA exp result.

For the recorded lambda=1 and unclipped leaves, the independent optimum is
w=-G/(H+lambda), benefit=G^2/(2*(H+lambda)). Rank every legal feature/threshold/
missing-direction candidate with the recorded row/Hessian guards and minimum
gain zero; preserve feature/threshold/missing-direction tie order. The standard
regularized second-order formula is given in
[Chen and Guestrin, section 2.2](https://arxiv.org/html/1603.02754v3#S2.SS2), and
matches the local leaf/benefit definitions after substitution. This formula
provides an independently expressed oracle rather than replaying the device
arithmetic instruction sequence.

Record all candidate counts and high-precision scores for the bounded nodes,
the oracle winner and runner-up, score margins, the ranks of saved decisions,
and exact equal-count equivalence classes. Recheck selected scores at 120
digits. Identical integer sufficient statistics imply a mathematical tie;
small nonzero margins remain quantified observations, not an arbitrary
tolerance that converts a failed zero-allowance gate to a pass.

## Saved-model cross-checks and limits

Quantize with the saved feature metadata using numeric lower-bound semantics;
missing values map to bin 0 and categorical unknowns follow the saved contract.
Check reconstructed row counts against their integer sums. Check saved leaf
values against the corresponding count-based optimum times the recorded
learning rate, retaining exact differences. At the affected validation row,
show each traversed split, reached leaf, exported base margin, leaf increment,
and CPU transformed probability versus the saved GPU probability. This
cross-check must explain a large probability difference through actual model
decisions and leaf values, not merely infer it from metric differences.

GPU histogram snapshots are absent. High-precision counts cannot prove the
exact FP64 atomic sums or their instruction ordering. Distinguish an ideal
mathematical tie, a small score separation, and a clearly inferior ideal
candidate; only the latter warrants additional corruption/contract scrutiny,
and none alone establishes what the device received. Preserve errors and
unsupported assumptions in the report.

Count-derived first-round statistics may be a later design hypothesis if the
observations support it. They change floating-point summation and are not an
exact-preserving replacement. Do not implement or promote that change here.
