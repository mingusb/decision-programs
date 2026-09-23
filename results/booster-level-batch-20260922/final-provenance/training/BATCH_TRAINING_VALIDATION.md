# Output-batch trainer validation contracts

2026-09-22. This records a test-contract correction after preserved failures,
not a change to production split selection or the zero-allowance quality gate.

`tests/batch_training.cpp` has two distinct responsibilities:

1. The mandatory correctness suite checks the implementation's stated contract.
   For all objectives, depths 0/2/5, bins 17/64, weighted/missing/categorical
   inputs, full/short output tiles and both submission modes, each exported
   model's GPU predictions must agree with its CPU traversal. Its recorded GPU
   objective at every boosting round must agree with an independent weighted
   CPU objective of that round's exported forest. Feature metadata, base scores,
   output order/count, serialization, memory budgets/accounting, frontier
   history, stage operations and bounded exports are checked independently.
2. `--comparison-diagnostic` retains the stronger cross-run model/quality
   comparisons and exhaustive legal-bin semantic certificates. The exact
   tolerances and failing assertions are retained. `--targeted-repeat` compares
   the unchanged per-output trainer against itself; `--targeted-batch` compares
   output-batch construction against it. `--diagnostic-dir PATH` preserves both
   models in binary and JSON form, fixtures, configurations and failing paths.
   Existing diagnostic directories are never overwritten.

The mandatory suite also requires zero-tolerance cross-policy model/whole-bin
function agreement where statistics are mathematically invariant to ordering:

- One-round regression has equal-weight opposite dyadic targets, exact zero
  base scores and exactly representable gradient/Hessian sums.
- One-round binary learning has equal-weight complementary labels, exact zero
  base scores, probability 1/2 and Hessian 1/4 before weighting.
- One-round eight-class learning has 32 rows per class with unit weights, equal
  base margins, probability 1/8 and Hessian 7/32. This also tests the frozen
  multiclass derivative snapshot across tiles.

Those exact fixtures use depth 5, bins 17/64, output tile width 3, full/short
tiles and stream/graph execution. Every legal encoded feature tuple is checked,
including missing/unseen-category bin zero. Fixed-histogram split primitive
tests remain bitwise comparisons, including tie handling and fallback.

## Why arbitrary cross-run equality is a separate diagnostic

The first full-training tests incorrectly assumed unordered FP64 histogram
accumulation would always preserve the complete learned function. A stronger
exhaustive certificate exposed a failure that held-out rows sometimes missed.
The unchanged per-output trainer then reproduced the same failure against
itself on the multiclass fixture. Production code was unchanged throughout.

The preserved baseline-repeat comparison has identical training margins but a
maximum held-out margin difference of **0.050911730395435617**. In tree 9,
thresholds 7 and 8 give the same observed local training partition. Their
floating gain comparison can choose different thresholds, and a legal encoded
input with bins `[8,0,0]` reaches leaf values about -0.02631238 and +0.02459935.
The optimized-versus-baseline comparison exhibited the same discrepancy. This
is a discrete split-instability observation, not merely a tiny prediction-rounding
difference, and baseline variability does not excuse an optimization's quality
regression.

Evidence is retained under
`results/booster-level-batch-20260922/baseline-repeat-failure/` and
`results/booster-level-batch-20260922/batch-targeted-failure/`, with preceding
failed sources/binaries and SHA256 receipts in the same experiment directory.
The diagnostic still fails on this evidence. The default suite reports observed
cross-run margin differences without claiming arbitrary unordered runs are
equivalent. Held-out metric audits and strict zero-allowance gate results remain
separate, unchanged requirements for evaluating the candidate.

No new tie canonicalization, gain tolerance, deterministic reduction, objective
change or production fallback was introduced to make tests pass.
