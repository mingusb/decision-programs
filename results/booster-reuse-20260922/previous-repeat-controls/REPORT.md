# Previous same-policy repeat controls

The unchanged previous implementations already produce different predictions
across repeat runs. **13 of 16 directional zero-allowance comparisons fail**,
including all 12 large-output comparisons. The largest recorded deterioration
is **3e-16 nats in an individual output's log loss**. These findings do not
change the allowance or convert any failed optimization comparison into a pass.

This is a CPU audit of frozen evidence in
results/booster-root-split-20260922, performed before the current count-reuse
campaign. Each row below compares the previous a and b runs of one unchanged
configuration, in both directions. There are **eight pairs of runs**, not
16 independent replicate pairs. **base** means the previous per-tree roots and
256-thread split search; **both** means the previous batched roots and warp split
search. Neither includes the new count reuse or batched root split decisions.

| Case and unchanged policy | Failed directions / 2 | Regressed metrics, a→b / b→a | Maximum positive metric deterioration across both directions | Differing prediction values / total | Maximum absolute prediction difference |
|---|---:|---:|---:|---:|---:|
| Scalar, base | 1 | 0 / 2 | 7e-17 target units, MAE | 3,315 / 8,192 | 2.220446049250313e-16 |
| Scalar, both | 0 | 0 / 0 | 0 | 3,290 / 8,192 | 2.220446049250313e-16 |
| 129 regression outputs, base | 2 | 33 / 32 | 2e-16 target units, MAE/RMSE | 19,540 / 33,024 | 5.551115123125783e-16 |
| 129 regression outputs, both | 2 | 34 / 31 | 2e-16 target units, MAE/RMSE | 19,901 / 33,024 | 4.996003610813204e-16 |
| 1,024 binary outputs, base | 2 | 258 / 220 | 3e-16 nats, log loss | 40,064 / 262,144 | 2.7755575615628914e-16 |
| 1,024 binary outputs, both | 2 | 254 / 244 | 2e-16 nats, log loss | 39,269 / 262,144 | 2.7755575615628914e-16 |
| 4,096 binary outputs, base | 2 | 430 / 413 | 3e-16 nats, log loss | 18,681 / 262,144 | 2.220446049250313e-16 |
| 4,096 binary outputs, both | 2 | 411 / 421 | 3e-16 nats, log loss | 18,920 / 262,144 | 2.220446049250313e-16 |

Each scalar comparison checks four metrics; each 129-output comparison checks
260; the 1,024- and 4,096-output comparisons check 4,100 and 16,388 respectively.
These totals include aggregate and individual-output metrics. Scalar aggregate
and output-zero metrics describe the same output and must not be interpreted
as independent observations. Every applicable metric participates in the gate.

For example, the largest deterioration occurs in the
[1024-base-b → 1024-base-a comparison](1024-base-a-vs-1024-base-b.json)
for logloss_output_205 and logloss_output_707, each at 3e-16 nats.
The largest Brier-score deterioration is 1.2e-16 squared-probability units.
All binary decision changes are **zero**, and accuracy/AUC metric deterioration
is zero in these controls. Those facts apply to these held-out predictions;
they are not a guarantee for other data.

The scalar both pair passes in both directions while 3,290 predictions differ.
Thus equal measured metrics do not establish bitwise equality. Conversely,
large-output pairs can fail in both directions because different individual
outputs regress in each direction; neither run necessarily dominates the other.

## Evidence and interpretation

The [audit summary](summary.json) records allowance zero, 16 comparisons,
13 failures and no recorded errors. Full adjacent comparison JSON files retain
metric values, exact decimal deterioration, source-CSV verification and
prediction-change counts. All 16 source captures name the same executable SHA256:

369ce3e735949e32822ce617c18f22c98fea90703630ec70251d2f476a0d3ae9

Matching protocol and policy checks precede the comparison. The evaluator
rechecks source CSV hashes and recomputes their metrics; deterioration is judged
using recorded decimal values with no extra tolerance. This report was derived
from the retained JSON and capture records and launches no GPU work.

These repeats establish pre-existing run-to-run variability and its observed
scale on eight specific configuration pairs. Unordered FP64 atomic accumulation
is consistent with that variation, but these controls do not isolate its causal
contribution from every other source of floating-point variation. Two runs per
configuration do not estimate a reliable variability distribution or prove that
all future optimization differences arise from the same cause. The new campaign
adds separate repeat controls; its results must be assessed on their own merits.

## Numerical contracts for the new work

- **Count reuse is exact under its stated participation contract.** Every
  retained row contributes once per feature, including missing bins and
  zero-weight rows. Fixed bins and all-row root participation make the uint64
  counts invariant across outputs and rounds. Shared count setup uses uint32
  partials over at most 4,096 rows, then exact uint64 merging. Row sampling,
  output-specific row exclusion or mutable bins would require a new cache
  validity contract.
- **Gradient/Hessian sums remain FP64 with unspecified atomic order.**
  Removing integer count atomics can change thread timing and therefore the
  order of FP64 additions, even when the same real-valued terms are added.
  Shared weighted histograms additionally change grouping and merge order.
  Exact integer counts do not make these sums or trained models bitwise equal.
- **Split batching preserves the split computation on identical histogram
  inputs.** Fieldwise bit-equality tests compare existing and batched split
  results, including ties, signed zero and owned fallback paths. This
  conditional property does not imply identical histograms across training
  variants. A changed rounded histogram can change a near-tied split decision.
- **Quality remains a separate gate.** Exact counts, valid FP error bounds,
  bitwise split tests on fixed inputs and unchanged held-out class decisions
  do not override a failed zero-allowance metric comparison.
