# Output-tile trainer work log

Scope and pre-implementation contracts: training/LEVEL_BATCH_EXPERIMENT.md.
New production policy `TreeBuildPolicy::output_batch` retains independent state,
uses selected strided root/deeper histograms, batches split/materialization/route/
advance/prediction, and reduces tile width to satisfy memory budgets. Original
counting code and promoted per-output root defaults are unchanged.

Final core status: ctest-contract records 13/13 passing suites. Thirteen serial
Compute Sanitizer checks passed: memcheck/racecheck/synccheck for each of four
primitive suites, plus trainer memcheck. The mandatory suite validates each
model against independent exported CPU/GPU prediction and per-round objective
references, memory/accounting, state, stage history and exports. Additional
one-round dyadic regression, balanced binary and balanced eight-class fixtures
require zero-allowance cross-policy equality over every legal encoded input;
these cover depth 5, bins 17/64, tile width 3, short tails and stream/graph execution.

The initial 12/13 CTest result and all subsequent failures remain preserved.
Some topology differences passed exhaustive function certificates, but a
multiclass case failed. The unchanged per-output trainer reproduced a stronger
failure against itself: identical training margins but a held-out margin change
of 0.050911730395435617. Locally empty bin 8 allows thresholds 7/8 to agree on
observed training rows but disagree on unseen inputs. The candidate comparison also
failed. Final trainer memcheck observed a larger general held-out raw-margin
difference, 0.24877399999999994, in two comparisons outside the rounding bound.
These are meaningful split-instability observations, not negligible prediction
rounding or accuracy/loss metrics. Strong cross-run/all-bin checks remain as
explicit diagnostic modes with unchanged bounds. The strict quality evaluator
and failed gate statuses are unchanged; baseline variability grants no waiver.

Nsight Systems audit passed for all three captures and 1,442 matched stages.
For the same 387 output trees, complete-boosting kernel records including count
setup fell from 9,817 to 655, and graph enqueues from 387 to 27. Every tree scope
contains zero traced host synchronization, D2H copies, device allocations and
device frees. Exports retain 27 completed-tile waits on both graph paths.
Profiler durations are diagnostic only. See IMPLEMENTATION.md and systems-audit.md.

The new tree_build policy remains opt-in. Production sources are stable; no tie
canonicalization, deterministic reduction or objective change was introduced
to make the validation suite pass. Full test-contract details and preserved
failure provenance are linked from IMPLEMENTATION.md.

Reference capability failures are preserved, not timed evidence: XGBoost adapter
used an outdated saved-config key; CuPy prediction required absent NVRTC runtime;
CatBoost multiclass/multilabel structure search rejected NewtonL2, and its GPU
prediction API rejects multidimensional models. Adapters now use current config
key, isolated official NVRTC runtime, supported L2 multi-output score, and
explicitly labeled CPU reference prediction for those CatBoost modes. Initial
MAGIC smoke prefix was single-class; deterministic class-covering smoke rows
replace it. Full real-data fixtures and benchmark grids are unchanged.
The final capability smoke receipt succeeds; those smoke results are capability
checks, not performance or quality rankings.

Final evidence is complete in REPORT.md: 56 serial synthetic runs, 80 real-data
validation runs and 60 validation-selected test runs. The real-data provenance
audit and exact baseline-cache audit pass. Five separate Delicious memory
diagnostics complete; their timings are excluded from rankings. The final report
retains all observations, native framework scope differences and earlier failures.

The synthetic multi-output training medians improve 1.33–3.69x; real complete
training improves 2.59x on Letter and 1.74x on Delicious relative to per-output
construction. XGBoost remains faster on Delicious; LightGBM has lower selected
test log loss on MAGIC, Letter and Delicious. No universal ranking is established.

Real matched-policy checks fail 22/28 all-metric gates and 17/28 loss gates with
zero allowance. The largest probability difference is 0.59643733 on a positive
Delicious label (0.671177 versus 0.074740). Repeated unchanged per-output runs
also vary by as much as 0.50581958 probability. All 16 within-policy repeat pairs
change topology, and 10 fail an applicable metric gate. These differences are
material and do not override or excuse any failed gate. The new policy remains
opt-in; the root/count defaults remain unchanged.

WIDE_FEATURE_SPLIT_EXPERIMENT.md records an unimplemented, unmeasured follow-up.
Final source/build and artifact manifests seal this completed experiment.
