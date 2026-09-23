# Promotion of measured root policies

2026-09-22. Recorded before changing defaults.

The user accepts the observed floating-point differences (up to 4e-16 nats in
the completed audit) as practically negligible and requests default promotion.
This is a release-policy decision for these observed results, not a numerical
tolerance added to the evaluator. Historical zero-allowance failures remain
failed and immutable. Future material quality differences still require review.

Use the existing implementations and contracts in REUSE_BATCH_EXPERIMENT.md.
No kernel, arithmetic, model architecture, or count-histogram production policy
is changed. The measured candidates remove repeated invariant count updates
and batch independent root split work. The prior 61-case campaign and Nsight
captures support promotion: lower paired complete training times in large-output
cases, exact integer counts, fixed-input split equivalence, unchanged held-out
classification decisions, and completed correctness/sanitizer checks. Baseline
run-to-run floating-point variability and timing spread remain documented.

Candidate defaults: batched root histograms, warp32 split policy (owned block
fallback for unsupported widths), reuse-global root counts, batched root splits.
Do not default to forced shared deeper histograms: they lost the measured wide
training comparisons. Global count setup supports the full existing bin domain;
shared setup remains explicit because its shared-memory domain is bounded and
its full-training performance was mixed.

Input/output and numerical contracts are unchanged: fixed bins and all root
rows across outputs/rounds; uint64 counts, unordered FP64 G/H; independent scalar
trees and frozen multiclass derivative snapshots. The same cache costs remain
within declared histogram/device budgets. Default users now pay those costs;
explicit legacy policies remain available when needed. Invalid combinations
continue to reject rather than silently selecting another algorithm.

Before promotion, measure the candidate using the frozen previous binary under
the actual otherwise-default context: stream execution, tile32, compact export,
histogram auto. Compare the old four-policy defaults against the candidate in
opposite-order sweeps for scalar,129,1024,4096 outputs and multiclass17. This
checks applicability beyond the preceding graph/tile16/export16 campaign.
If a material repeatable default-context regression appears, retain/adjust the
default choice based on that evidence before changing it. Do not modify old
sealed builds or reports; use build/booster-defaults and a new evidence directory.

Validation after the change: run the existing CTest suite serially, preserve
explicit legacy test baselines, and exercise actual inherited defaults for all
objectives with independent predictions/loss checks. Capture no-policy CLI
smokes and compare reported settings to explicit candidate configuration.
Rebuild/validate changed callers; unchanged CUDA kernels do not require another
full profiling/sanitizer campaign. Existing performance and strict quality
evidence remain linked, with no relabeling of failed gates.

## Default-context result and choice

The frozen-binary 24-run comparison completed before the header edit. Candidate
total training time was lower in both sweep orders for all six cases, including
scalar, all three wide shapes, multiclass17 and the >32-feature/>32-bin owned
block fallback case. This supports promoting the four controls while retaining
all other defaults. Much of the actual-default improvement removes repeated
root autotuning; it must not be conflated with the prior incremental 3.5–16.2%
comparison against an already tuned graph configuration. Raw results are in
results/booster-defaults-20260922. The quality acceptance remains the user
decision above; historical strict audits are unchanged.
