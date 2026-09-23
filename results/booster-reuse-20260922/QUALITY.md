# Strict quality results

**The overall gate remains regression.** All 61 trained models pass comparison
against their own constant-base predictions, but **52 of 62 paired comparisons
fail** at zero allowance. This comprises **34 of 44 policy comparisons** and
**all 18 directional same-policy repeat comparisons**. No failure is waived.

Sources: [complete audit summary](quality/summary.json),
[per-comparison table](quality/summary.md), and
[independent metadata verification](metadata-verification.json).
The audit records no invalid evidence. Metadata verification confirms all
61 case protocols, serialized model objectives/dimensions, one executable hash,
615 distinct artifact hashes and all 123 recorded gates. Its verification
status leaves the quality status as regression.

## Comparison coverage

The current baseline is the previous combined policy: batched root histograms,
per-output count accumulation and per-tree warp split searches. Each main
policy group covers scalar, 129-output regression, 1,024-output binary and
4,096-output binary cases in two policy sweeps.

| Comparison group | Failed / checked | Interpretation |
|---|---:|---|
| Global root count reuse | 6 / 8 | Both scalar comparisons pass; all large-output comparisons fail |
| Batched root split search | 6 / 8 | Both scalar comparisons pass; all large-output comparisons fail |
| Count reuse + batched root splits | 6 / 8 | Both scalar comparisons pass; all large-output comparisons fail |
| Both + existing shared deeper histograms | 6 / 8 | Both scalar comparisons pass; all large-output comparisons fail |
| Shared root count setup | 4 / 4 | 129 and 4,096 outputs, two observations each |
| Fresh-seed combined/deeper-shared confirmations | 4 / 4 | Seed 20260922604, 129 and 4,096 outputs |
| Seventeen-class combined/deeper-shared comparisons | 0 / 2 | Both pass this held-out metric gate |
| Original scalar policy → current baseline | 2 / 2 | These are separate controls, not new count-reuse comparisons |
| Same-policy repeat controls | 18 / 18 | Three large-output workloads; baseline a versus b/c/d, in both directions |
| **All paired comparisons** | **52 / 62** | Zero allowance throughout |

The repeat controls contain four baseline runs per large-output workload.
They form nine pairs, each assessed in two directions, not 18 independent
replicate pairs. They do not cover every possible pairing of the four runs.
Passing against a constant-base predictor is a different requirement from
matching an existing trained model and does not substitute for that comparison.

## Exact metric-family maxima

Each number below is the largest positive deterioration among all corresponding
aggregate and individual-output metrics. Policy comparisons include the
original-scalar controls; repeat comparisons are listed separately. Metric units
differ and their magnitudes should not be treated as a common effect size.

| Exact family key | Unit | Policy comparisons | Repeat controls | All comparisons |
|---|---|---:|---:|---:|
| rmse | target units | 3e-16 | 2e-16 | 3e-16 |
| mae | target units | 3e-16 | 2e-16 | 3e-16 |
| logloss | nats | **4e-16** | 3e-16 | **4e-16** |
| brier | squared-probability units | 1.2e-16 | 1.2e-16 | 1.2e-16 |
| accuracy | proportion | 0 | 0 | 0 |
| auc | proportion | 0 | 0 | 0 |

The largest deterioration is
[1024-base-a → 1024-split-a](quality/1024-split-a-vs-1024-base-a.json),
exact metric key **logloss_output_707**:
0.5136599339205032 → 0.5136599339205036 nats, with recorded decimal deterioration
**4E-16** and status regression. The same comparison reaches 1.2e-16 on
**brier_output_707**. Examples of the regression-family maxima are
**rmse_output_19** in 129-base-a → 129-both-a and **mae_output_4** in
129-base-b → 129-both-b, each 3e-16 target units.

No family is omitted because its changes are small. The evaluator checks
aggregate and individual-output RMSE/MAE for regression; log loss, accuracy,
Brier score and AUC for binary/multilabel outputs; and log loss/accuracy for
multiclass. All applicable metrics participate in each gate.

## Predictions and serialized models

All 34 classification comparisons have zero changed held-out decisions, using
probability ≥0.5 for binary outputs and lowest-index argmax for multiclass.
Accuracy and all applicable AUC metrics remain equal. The probability values
can still differ, and log loss/Brier failures remain failures.

The maximum held-out prediction difference is:

- All comparisons and repeat controls: **6.3837823915946501e-16**, in
  129-base-a versus 129-base-b.
- Policy comparisons: **6.1062266354383610e-16**, in
  129-base-a versus 129-deep-shared-a.

All 62 comparisons have identical feature metadata, including exact encoded
cut/category values, and equal numeric base scores. Nonetheless, 34 comparisons
have changed split thresholds or missing directions. No comparison changes
the selected split feature, leaf-versus-split status or set of structural node
paths. The largest per-comparison counts are 174 trees with changed split fields,
174 changed thresholds, and 111 changed missing directions.

Matched leaf values differ in all comparisons. The largest difference between
leaves at the same structural path is **0.1891869869837374** in
confirm-4096-base versus confirm-4096-both. That number is not a prediction
difference: changed routing can make corresponding path positions represent
different input regions. The encoded-region comparison below accounts for
the regions actually routed by each tree.

## Bounds over every encoded input

The audit intersects paired leaf regions over every feature's encoded-bin
domain, including missing bin zero. Using exact rational representations of
stored FP64 leaves, it sums per-tree maximum differences and base-score
differences for each output, then rounds reported floating bounds upward.
The largest whole-model raw-margin bounds are:

| Scope | Maximum upward-rounded bound | Comparison |
|---|---:|---|
| All comparisons / repeat controls | 1.5681900222830336e-15 | 1024-base-a versus 1024-base-b |
| Policy comparisons | 1.5196177649556830e-15 | 1024-base-b versus 1024-counts-b |

These bound the **exact-real sum of serialized leaf values**, including every
metadata-defined encoded input combination. They exclude additional rounding
from runtime accumulation and probability transforms. They are conservative
model-function bounds, not bounds on all quality metrics, statistical
generalization guarantees, or evidence of bitwise model equality. They do not
change any strict gate outcome.

## Repeat variability and numerical semantics

The [previous frozen repeat controls](previous-repeat-controls/REPORT.md)
failed 13 of 16 directional comparisons, including all 12 large-output
directions. Their maximum metric deterioration was 3e-16 nats. The current
18 large-output repeat directions also fail, with maximum log-loss
deterioration 3e-16. Current policy comparisons reach 4e-16. These observations
establish that variation already exists within an unchanged implementation;
they do not prove that every optimization difference has the same cause or
provide an allowance derived from variability.

Count reuse preserves integer counts exactly under fixed bins and all-row root
membership. FP64 gradient/Hessian atomics remain unordered, and removing count
atomics can change their execution ordering. Shared weighted histograms also
change accumulation grouping. Bitwise split tests establish equal result fields
on identical histogram inputs; they do not establish identical independently
accumulated histograms or trained models.

The audit calls the existing evaluator with allowance **0.0** for both
model-versus-base and paired-model comparisons. It recomputes metrics from
source CSVs, verifies their hashes, and compares recorded decimal metric values
with 2,048-digit Decimal arithmetic and no additional tolerance. The audit,
both imported helpers and evaluator still match their recorded source hashes
after completion. Evaluator SHA256:

87d360e916087c2b8253467ecb81fa1099b568d0f51d1559390287168b8a0aca

The [audit command](quality-audit-command.json) retains its regression outcome;
the [metadata-verification command](metadata-verification-command.json) checks
evidence integrity separately. No quality-preserving default promotion follows
from these results.
