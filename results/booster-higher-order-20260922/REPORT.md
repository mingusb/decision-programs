# Safeguarded third- and fourth-order CUDA boosting

**Implemented, measured and audited.** The closed-form higher-order CUDA paths
show workload-dependent quality tradeoffs and higher training cost; they have
not earned a default promotion. At the selected clipped Delicious settings,
macro average precision rises from 0.0853403 to 0.0869077 / 0.0872968 while median
training takes 1.50x / 2.07x longer. Selected MAGIC average precision declines.
Larger leaf bounds expose further mixed results. No fastest-implementation claim
is made. All existing defaults remain unchanged: order 2, unclipped leaves
(`max_leaf_value=0`) and per-output tree construction. The experiments below
explicitly request output-batch construction, including the unclipped control.

## Implementation and support contract

The CUDA C++23 trainer now has compile-time-specialized third- and fourth-order
binary logistic optimizers, including independent multilabel outputs. The
experiment supports scalar binary learning and all 983 Delicious labels; it does
not implement coupled multiclass derivative tensors or arbitrary-order automatic
differentiation. Orders above 2 require output-batch tree construction, a global
or automatic histogram policy, and a finite positive leaf-value cap. Unsupported
combinations reject explicitly. The existing order-2 paths remain available.

Binning, derivatives, histograms, split decisions, routing, tree construction and
prediction remain on the GPU. Host fixture loading, model export and the explicit
offline CPU metric references are outside production training. The existing
counting algorithms and defaults are unchanged; NVIDIA algorithm implementations
are not production fallbacks.

The design and its alternatives were recorded in
[HIGHER_ORDER_EXPERIMENT.md](../../training/HIGHER_ORDER_EXPERIMENT.md).
The implementation is in [higher_order_math.cuh](../../training/include/ghb/higher_order_math.cuh),
[higher_order.cu](../../training/src/higher_order.cu), and
[batch_training.inc](../../training/src/batch_training.inc), with the public
configuration in [booster.hpp](../../training/include/ghb/booster.hpp).

For each output, the kernels compute weighted loss derivatives with respect to
the scalar prediction margin. They accumulate FP64 gradient G, Hessian H, third
derivative T and, for order 4, fourth derivative Q, together with exact unsigned
64-bit counts. Signed T and Q are preserved. Stable logistic formulas reuse one
exponential and shared probability/curvature terms in one derivative pass.
Higher orders use the true unfloored Hessian; retained order 2 uses its existing
Hessian floor and residual arithmetic. Consequently even a matched clipping cap
does not isolate optimization order from every derivative-arithmetic change.

With A=H+lambda, the local leaf proposals are

```
Newton:     -G / A
Halley:     -2*G*A / (2*A*A - G*T)
Householder: -3*G*(2*A*A - G*T) / (6*A*A*A - 6*G*A*T + G*G*Q)
```

The code evaluates normalized ratios to avoid unnecessary powers. The fourth
formula includes G squared times Q: the printed Equation 14 in
[Pachebat and Ivanov](https://arxiv.org/abs/2211.11367) omits that square. An
independent reciprocal-series reference checks the implemented algebra.

These are single local proposals, not exact unrestricted cubic/quartic
minimizers. Finite denominators above `1e-12`, descent checks and the finite leaf
cap guard the higher-order candidate. Zero, clipped Newton and the valid clipped
higher-order proposal are scored using the same regularized Taylor model;
Newton wins a benefit tie. Parent and child split benefits use that same order.
The cap bounds the unshrunk leaf; the applied margin change is also multiplied
by the learning rate. Nonnegative surrogate benefit does not certify actual
loss reduction or held-out improvement. Actual per-round training losses are
retained in every benchmark's `metrics.json`.

## Additional work and memory

For N rows, output tile B, frontier capacity C and J total feature bins:

| Payload or work | Order 2 | Order 3 | Order 4 |
|---|---:|---:|---:|
| Histogram cell | 24 bytes | 32 bytes | 40 bytes |
| Histogram allocation | 24BCJ | 32BCJ | 40BCJ |
| Derivative scratch | 16NB bytes | 24NB bytes | 32NB bytes |
| Floating-point fields accumulated | 2 | 3 | 4 |

Root count reuse avoids repeated count atomics. Extra derivative fields still
increase clear/store/read traffic, FP64 atomic work, split scanning and register
demand. They do not require one new histogram pass per derivative. Scratch
planning uses the actual order-specific layouts; a memory limit can reduce the
available batch/frontier capacity. These ratios describe payloads, not predicted
runtime speedups or slowdowns.

The higher-order path retains output batching, subgroup feature-bin reuse and
GPU-resident frontier state. Its one-warp reduction removes an unnecessary
second reduction of zero partials, while order-2 arithmetic remains unchanged.
The real Delicious workload has 500 features and therefore uses the existing
256-thread split path, not the narrow-feature warp-only specialization.

## Completed correctness and preservation checks

- All 13 existing CTest suites passed: [existing test receipt](ctest-existing.log).
- Both new suites passed: [new-suite receipt](ctest-higher-order-initial.log).
  Final detailed output records **19,143 independent CPU math checks** and
  **3,373,239 GPU checks** in
  [the archived final CTest log](ctest-higher-order-details.log).
  The earlier [math-test.log](math-test.log) records the earlier 18,627-check
  revision and is preserved.
- Compute Sanitizer reported zero errors for
  [memcheck](sanitizer-memcheck.log), [racecheck](sanitizer-racecheck.log),
  [synccheck](sanitizer-synccheck.log), and [initcheck](sanitizer-initcheck.log).
  Racecheck also reported zero warnings/hazards. Commands and successful return
  codes are in [sanitizer-results.json](sanitizer-results.json).
- All nine preserved counting source/build-file comparisons passed:
  [count-source-preservation.json](production-provenance/count-source-preservation.json).
  Frozen [source hashes](production-provenance/source-sha256.json) and
  [binary hashes](production-provenance/binary-sha256.json) identify the measured
  implementation.

The new tests cover analytical derivatives and independent Householder algebra,
guards, signed statistics, split scoring, selectors/tails, graph replay,
training and saved-model prediction. Numerical oracle tolerances do not relax
the separate zero-allowance quality gates. The sanitizer fixture suite observed
zero per-round training-loss increases; that is an observation on those
fixtures, not a guarantee for arbitrary training data.

## Registered experiment and interpretation

The [main protocol](real/plan/protocol.json) and
[campaign driver](../../training/tools/higher_order_campaign.py) specify
**50 real-data jobs**: eight screen cases, 24 validation-grid cases and 18
validation-selected test fits. Every GPU job runs serially in a fresh process;
the offline common evaluator runs after that GPU process finishes. Models,
predictions, losses, metric values, memory payloads, timing scopes, commands,
stdout/stderr, failures and hashes are preserved per case.

All main order-2/3/4 comparisons use cap 1, 32 bins, learning rate .1, global
histograms, output-batch construction, graph execution, output tile 16 and tree
export batch 16. The screen also includes a separately labeled **unclipped
order-2 output-batch control** for each dataset. MAGIC screening uses 25 rounds/depth 3;
Delicious screening uses five rounds/depth 2.

Each dataset/order receives four validation configurations: lambda 1 or 10
crossed with 25 or 75 rounds at depth 5 for MAGIC, or five or ten rounds at depth
3 for Delicious. Selection maximizes validation average precision for MAGIC and
macro average precision for Delicious, then breaks ties by validation log loss,
synchronized training wall time and fixed grid index. The coarse time-to-quality
comparison uses the best observed matched-cap order-2 validation selection
metric as its shared target. An order that never reaches it at the sampled
budgets must be reported as not reaching it; there is no interpolation or
extrapolated convergence claim.

Each selected setting receives three new training fits evaluated on the original
test split. **These test data were already inspected in prior development.**
This is an exploratory follow-up on reused held-out data, not fresh confirmation.
Three fits measure training variability on one split, not uncertainty from
sampling new data. All 983 Delicious labels remain included; undefined
single-class ranking metrics retain the unchanged evaluator's explicit handling.

For detection operating points, a threshold is selected only on validation to
maximize recall under the exact condition `20*false_positives <= negatives`,
without splitting tied scores. That frozen threshold is applied to each test
fit. The report must show **actual test false-positive rate alongside recall**;
the validation five-percent constraint does not guarantee a five-percent test
rate. Delicious applies this rule to pooled micro decisions across all labels,
not an individual per-label guarantee. Fixed-threshold .5 metrics are also kept.

Every applicable quality metric retains its own direction and zero allowable
regression against a matched order-2 control. Loss, ranking, classification and
operating-point regressions remain visible. Changed models or unordered FP64
atomics do not turn a failed gate into a pass. Runtime and memory comparisons
must identify their complete measured scope. No synthetic test, individual
kernel timing or single improved metric establishes a universal winner.

## Completed main real-data campaign

The completed [screen receipt](real/screen/summary.json) records eight of eight
successful jobs; the [validation receipt](real/validation/summary.json) records
24 of 24, and the [test receipt](real/test/summary.json) records 18 of 18. The
[audited summary](real/summary/summary.json) contains the selected configurations,
raw timing repetitions, all metrics, operating-point confusion counts, memory
payloads and every matched comparison. This section reports the completed main
campaign; the completed radius, synthetic, profiling and final audits follow.

### Selected-test timing and selection metrics

Every value below is the median of three fits. Times are synchronized complete
`train` and `predict_gpu` calls. Training includes GPU binning, allocations,
transfers, tree building and model export within the trainer; fixture loading,
CUDA context initialization, file serialization and CPU metric evaluation are
separate captured scopes. Prediction includes the complete public GPU prediction
call, rather than only its device kernels. Brackets show the full observed
minimum–maximum, retaining every sample; they are not confidence intervals.

| Dataset | Order | Rounds / depth / lambda | Training ms [min–max] | Prediction ms [min–max] | AP / macro AP | Log loss |
|---|---:|---:|---:|---:|---:|---:|
| MAGIC | 2 | 75 / 5 / 1 | 69.788 [62.526–192.379] | 3.908 [2.781–4.458] | 0.945036245 | 0.321548344 |
| MAGIC | 3 | 75 / 5 / 1 | 79.090 [76.306–81.711] | 2.508 [2.428–3.313] | 0.943874080 | 0.322742477 |
| MAGIC | 4 | 75 / 5 / 1 | 95.074 [90.430–97.909] | 4.190 [2.618–8.939] | 0.943655175 | 0.323126197 |
| Delicious | 2 | 10 / 3 / 10 | 28,231.774 [27,832.043–28,859.339] | 251.240 [177.471–330.682] | 0.085340289 | 0.071180478 |
| Delicious | 3 | 10 / 3 / 1 | 42,426.012 [41,897.345–43,241.046] | 235.609 [158.790–283.408] | 0.086907747 | 0.070839580 |
| Delicious | 4 | 10 / 3 / 1 | 58,439.281 [56,903.412–58,953.192] | 254.234 [169.489–255.449] | 0.087296808 | 0.070833279 |

MAGIC selected the same configuration for all orders. **Delicious selected
lambda 10 for order 2 and lambda 1 for orders 3/4.** Its selected-test table
compares outcomes of equal bounded tuning budgets; those rows are not a matched
optimizer-only control. The matched Delicious optimizer comparisons use the
validation grid. None of these tests use fresh unseen development data.

Full raw repetitions are in `datasets.<dataset>.<order>.training_wall_ms.raw`
and `prediction_wall_ms.raw` in [summary.json](real/summary/summary.json).
The order-2 MAGIC training sample of 192.379 ms is retained; the median does not
erase it. Prediction variability is substantial, so small median prediction
differences should not be interpreted as stable algorithmic wins.

### All common quality metrics

These are medians on the reused test split. Lower log loss, Brier score and
Hamming loss are better; higher values are better for the other metrics.
Classification metrics use threshold .5. MAGIC exact-match accuracy equals its
ordinary binary accuracy. Counts of labels and the log-clipping constant are
metadata, not quality objectives.

| MAGIC metric | Order 2 | Order 3 | Order 4 |
|---|---:|---:|---:|
| Log loss | 0.321548344 | 0.322742477 | 0.323126197 |
| Brier score | 0.097070821 | 0.097356450 | 0.097493726 |
| Hamming loss | 0.127793847 | 0.125164344 | 0.125953195 |
| Accuracy / exact-match accuracy | 0.872206153 | 0.874835656 | 0.874046805 |
| F1 | 0.905813953 | 0.907894737 | 0.907296303 |
| ROC AUC | 0.918912468 | 0.918194248 | 0.917800259 |
| Average precision | 0.945036245 | 0.943874080 | 0.943655175 |

| Delicious metric | Order 2, lambda 10 | Order 3, lambda 1 | Order 4, lambda 1 |
|---|---:|---:|---:|
| Log loss | 0.071180478 | 0.070839580 | 0.070833279 |
| Brier score | 0.016616935 | 0.016610666 | 0.016607530 |
| Hamming loss | 0.019070509 | 0.019095742 | 0.019084563 |
| Exact-match accuracy | 0.001255887 | 0.001255887 | 0.001255887 |
| Micro F1 | 0.027874111 | 0.024698206 | 0.026365103 |
| Macro F1 | 0.001325062 | 0.001159918 | 0.001238854 |
| Micro AP | 0.224892034 | 0.225869556 | 0.226061065 |
| Macro AP | 0.085340289 | 0.086907747 | 0.087296808 |
| Macro ROC AUC | 0.708497013 | 0.718528915 | 0.717813565 |
| Precision@1 | 0.512715856 | 0.511459969 | 0.514285714 |
| Precision@3 | 0.454631083 | 0.454945055 | 0.455154369 |
| Precision@5 | 0.420282575 | 0.418273155 | 0.418587127 |

Delicious macro AP and macro AUC have 982 defined label values; all 983 labels
remain trained and included in the other appropriate metrics. Higher orders
worsened MAGIC log loss, Brier, AUC and AP at the selected matched setting despite
better .5-threshold F1/accuracy. On Delicious, selected higher-order models
improved ranking/loss metrics but worsened Hamming loss, both F1 averages and
precision@5; order 3 also worsened precision@1. These selected Delicious
differences include the changed lambda and are not isolated optimizer effects.

### Detection operating points and actual false-positive rates

All values except thresholds are percentages. The first table uses the threshold
frozen from each selected validation model. The validation constraint was at
most 5% FPR; **the displayed test FPR is observed, not retrospectively constrained**.
Delicious statistics are pooled micro counts over all row–label pairs.

| Dataset | Order | Frozen threshold | Test recall | Test precision | Test F1 | Actual test FPR | Fits above 5% test FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAGIC | 2 | 0.873270100 | 60.2190% | 93.8092% | 73.3514% | 7.3298% | 3/3 |
| MAGIC | 3 | 0.871178810 | 61.4761% | 93.6380% | 74.2228% | 7.7038% | 3/3 |
| MAGIC | 4 | 0.871169929 | 61.6788% | 93.7731% | 74.4129% | 7.5542% | 3/3 |
| Delicious | 2 | 0.065133578 | 52.3170% | 17.0589% | 25.7285% | 4.98931% | 0/3 |
| Delicious | 3 | 0.062940869 | 52.7238% | 17.1367% | 25.8661% | 5.00051% | 3/3 |
| Delicious | 4 | 0.063194315 | 52.7686% | 17.0238% | 25.7426% | 5.04490% | 3/3 |

Fifteen of 18 test fits exceeded 5% FPR, including all nine MAGIC fits. Even the
small Delicious order-3 violation remains a violation. Higher MAGIC recall
coincided with worse FPR and precision than order 2 at these separately selected
thresholds. Delicious order 4 also had worse selected-threshold precision than
its differently regularized order-2 reference. None of these higher-order
recalls may be described as measured recall at controlled 5% test FPR.

| Dataset | Order | Recall at .5 | Precision at .5 | F1 at .5 | FPR at .5 |
|---|---:|---:|---:|---:|---:|
| MAGIC | 2 | 94.7689% | 86.7483% | 90.5814% | 26.7016% |
| MAGIC | 3 | 95.1338% | 86.8246% | 90.7895% | 26.6268% |
| MAGIC | 4 | 95.0527% | 86.7827% | 90.7296% | 26.7016% |
| Delicious | 2 | 1.42124% | 71.9328% | 2.78741% | 0.0108773% |
| Delicious | 3 | 1.25687% | 70.6816% | 2.46982% | 0.0102259% |
| Delicious | 4 | 1.34321% | 70.9649% | 2.63651% | 0.0107796% |

The selected thresholds and validation operating points are in
[selection.json](real/test/selection.json); each test case's `signal.json`,
`quality.json` and `result/metrics.json` are in [real/test](real/test).

### Strict matched comparison failures

Zero allowance remains in effect for every applicable metric, including the
additional recall, precision, F1 and FPR operating-point metrics. FPR is minimized;
the other operating-point metrics are maximized. Threshold values and confusion
count sizes are not independently treated as quality objectives.

| Comparison scope | Pairs | Common-quality failures | Signal failures | Either gate fails |
|---|---:|---:|---:|---:|
| Matched validation grid | 16 | 14 | 15 | 15 |
| Matched selected MAGIC tests | 6 | 6 | 6 | 6 |
| Main total | 22 | 20 | 21 | 21 |

There are no matched selected Delicious test pairs because lambda differs.
Separately, all four matched cap-1 screen pairs failed the common-quality gate;
two failed the signal gate. They are not silently added to or removed from the
22-pair main total.

The following counts show every common metric with a regression in the main
matched comparisons. MAGIC has 14 pairs (eight validation, six test); Delicious
has eight validation pairs. A zero means no regression in that applicable
metric; a dash means the metric does not apply. All exact case IDs, directions,
values and deltas are retained in [summary.json](real/summary/summary.json).

| Regressed common metric | MAGIC / 14 | Delicious / 8 |
|---|---:|---:|
| Log loss | 8 | 0 |
| Brier score | 10 | 4 |
| Hamming loss | 2 | 5 |
| Exact-match accuracy | 2 | 0 |
| Binary accuracy | 2 | — |
| Binary F1 | 2 | — |
| Binary ROC AUC | 12 | — |
| Binary average precision | 9 | — |
| Micro F1 | — | 5 |
| Macro F1 | — | 6 |
| Precision@1 | — | 5 |
| Precision@3 | — | 6 |
| Precision@5 | — | 2 |

Delicious micro AP, macro AP and macro AUC had zero regressions in these eight
matched validation pairs. The exhaustive additional signal regression counts
are:

| Regressed operating-point metric | MAGIC / 14 | Delicious / 8 |
|---|---:|---:|
| .5-threshold recall | 5 | 5 |
| .5-threshold precision | 2 | 3 |
| .5-threshold F1 | 2 | 5 |
| .5-threshold FPR | 2 | 2 |
| Validation-selected-threshold recall | 2 | 0 |
| Validation-selected-threshold precision | 9 | 3 |
| Validation-selected-threshold F1 | 2 | 2 |
| Validation-selected-threshold FPR | 8 | 5 |

Across all 50 saved main-campaign `training_loss` vectors, no consecutive
round increased actual training loss. That observed training behavior coexists
with the held-out failures above and does not establish a general guarantee.

### Coarse validation time to quality

The shared targets are the best observed cap-1 order-2 validation AP for MAGIC
(0.955389783) and macro AP for Delicious (0.089540668). These are independently
fitted coarse checkpoints, not a continuously monitored learning trajectory.

| Dataset | Order | Earliest sampled rounds reaching target | Lambda | Observed score | Training ms |
|---|---:|---:|---:|---:|---:|
| MAGIC | 2 | 75 | 1 | 0.955389783 | 71.692 |
| MAGIC | 3 | Not reached | — | Best: 0.955235273 | — |
| MAGIC | 4 | Not reached | — | Best: 0.955138171 | — |
| Delicious | 2 | 10 | 10 | 0.089540668 | 27,232.445 |
| Delicious | 3 | 10 | 1 | 0.091723640 | 40,707.007 |
| Delicious | 4 | 10 | 1 | 0.092254858 | 55,766.839 |

The sampled grid provides no earlier-round or lower-time target attainment for
the higher orders. They improve Delicious validation ranking at the ten-round
checkpoint but take longer to reach the shared target at the available samples.
No claim is made about unmeasured intermediate rounds or other tuning budgets.

### Recorded memory scopes

Values below are MiB (2^20 bytes), medians over the selected test fits. Owned
device bytes describe the trainer's accounted allocations; preparation peak is
the separately reported GPU preparation phase. Histogram, derivative and tree
state are components, not additional quantities to add to the owned total.
CPU peak RSS is the process observation from `getrusage`. These are not sampled
device-wide VRAM peaks and do not include a complete driver/context accounting.

| Dataset | Order | Owned device | Preparation peak | Histogram | Derivatives | Tree state | CPU peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAGIC | 2 | 0.696625 | 1.643150 | 0.119629 | 0.174164 | 0.045799 | 129.414062 |
| MAGIC | 3 | 0.822769 | 1.643150 | 0.158691 | 0.261246 | 0.045799 | 133.449219 |
| MAGIC | 4 | 0.948914 | 1.643150 | 0.197754 | 0.348328 | 0.045799 | 133.535156 |
| Delicious | 2 | 133.066505 | 13.996010 | 2.208710 | 2.523438 | 0.639893 | 231.453125 |
| Delicious | 3 | 135.060646 | 13.996010 | 2.941132 | 3.785156 | 0.639893 | 235.347656 |
| Delicious | 4 | 137.054787 | 13.996010 | 3.673553 | 5.046875 | 0.639893 | 235.285156 |

## Screen clipping confound and completed radius check

The original Delicious screen exposed a substantial clipping confound:

| Screen setting | Validation macro AP |
|---|---:|
| Order 2, unclipped, output-batch | 0.105561926 |
| Order 2, cap 1 | 0.070835394 |
| Order 3, cap 1 | 0.072294684 |
| Order 4, cap 1 | 0.072588685 |

These values come from the saved `quality.json` files in [real/screen](real/screen).
The improvement of orders 3/4 over clipped order 2 does not establish improvement
over the unclipped order-2 output-batch control. A separately registered **12-job
validation-only radius check** tested fixed caps 4 and 16 for all three orders on
both original screen shapes. Its [runner](run_radius_check.py) preserved the
original campaign, ran all fixed cells, and performed no test selection or
default promotion. This follow-up was designed after seeing the screen confound;
it is not part of the original blind comparison. All 12 jobs passed, with zero
audit failures; [the radius summary](real/radius-check/summary.json) retains
every metric, operating point, timing scope and comparison.

The following are single-fit validation observations. MAGIC uses 25 rounds,
depth 3; Delicious uses five rounds, depth 2. Both retain lambda 1, learning rate
.1 and the other original screen settings. These shallow, short-budget cells
must not be treated as replacements for the deeper, validation-selected main
test configurations.

| Dataset | Cap | Order-2 AP / macro AP | Order-3 AP / macro AP | Order-4 AP / macro AP | Training ms, orders 2 / 3 / 4 |
|---|---:|---:|---:|---:|---:|
| MAGIC | 4 | 0.933283046 | 0.932140662 | 0.934417583 | 26.513 / 28.338 / 33.562 |
| MAGIC | 16 | 0.933283046 | 0.932140662 | 0.934417583 | 30.571 / 30.598 / 32.420 |
| Delicious | 4 | 0.096065912 | 0.096763047 | 0.095130735 | 7,995.296 / 11,737.800 / 15,798.461 |
| Delicious | 16 | 0.105541913 | 0.096362371 | 0.093304366 | 7,800.957 / 11,916.765 / 15,799.592 |

Larger caps recover much of the order-2 Delicious loss of quality caused by cap
1. At cap 16, order-2 macro AP is close to its unclipped screen value, while both
higher orders have substantially lower macro AP and longer observed training
times. At cap 4, order 3 has a small macro-AP gain over order 2, alongside
regressions in other metrics. This is not a uniform higher-order benefit.

MAGIC order 4 at caps 4 and 16 improves validation AP from 0.933283046 to
0.934417583 and log loss from 0.397918250 to 0.383283967. It passes the
common-quality gate at both caps. That short-budget result coexists with worse
order-4 AP/AUC/loss at the deeper cap-1 selected test setting above; it does not
erase that result or establish superior test generalization. MAGIC order 3
improves loss at these larger caps but regresses AP and AUC.

**Six of eight matched-cap comparisons fail the common-quality gate.** The
complete failure pattern is:

| Matched candidate vs order 2 | Regressed common metrics |
|---|---|
| MAGIC, order 3, caps 4 and 16 | ROC AUC; average precision |
| MAGIC, order 4, caps 4 and 16 | None; both common-quality gates pass |
| Delicious, order 3, cap 4 | Log loss; Brier; micro AP; precision@1/@3/@5 |
| Delicious, order 4, cap 4 | Log loss; Brier; micro AP; macro AP; precision@1/@3/@5 |
| Delicious, order 3, cap 16 | Log loss; Brier; micro AP; macro AP; precision@1/@3/@5 |
| Delicious, order 4, cap 16 | Log loss; Brier; micro AP; macro AP; macro AUC; precision@1/@3/@5 |

The separately saved signal values also matter: comparing recall, precision,
F1 and FPR in their existing directions at both operating points shows at least
one signal regression in **all eight** matched-cap comparisons. In particular,
even the two MAGIC order-4 common-quality passes have lower recall, precision
and F1 at the validation-selected threshold than their matched order-2 controls.
These counts are direct comparisons of the saved radius `signal` fields; the
radius JSON's named strict gate covers common quality metrics. No tolerances
were added. Each radius cell is one fit, so this follow-up does not estimate
training variability or data uncertainty and does not select a new cap/default.

## Completed synthetic measurements

The [synthetic protocol](synthetic/protocol.json) runs binary objectives with
cap 1, output-batch graph construction, global histograms, output tile 16 and
export batch 16. Each case has 512 held-out rows. Scalar uses 262,144 training
rows, 16 features, one output, ten rounds, depth 5 and 32 bins; multi33 uses
32,768 rows, 16 features, 33 outputs and the same tree/bin budgets; wide129 uses
8,192 rows, 65 features, 129 outputs, five rounds, depth 3 and 64 bins. The
[runner](run_synthetic.py) executes one unranked multi33 warm-up per order, then
three repetitions per case/order with rotated order. All 30 process receipts
succeeded: three warm-ups plus 27 measured fits.

`total_train_ms` is the benchmark's complete training timing including
preparation and cleanup, not merely its inner training stage. The table shows
medians and full observed ranges; held-out losses were identical across the
three fits within each displayed case/order. This synthetic loss observation
does not replace the real-data detection metrics or strict quality gates.

| Case | Order | Total training ms [min–max] | Held-out log loss |
|---|---:|---:|---:|
| Scalar | 2 | 56.140 [55.216–62.615] | 0.324762257 |
| Scalar | 3 | 69.049 [66.150–74.680] | 0.324779360 |
| Scalar | 4 | 76.852 [72.189–78.908] | 0.324756511 |
| Multi33 | 2 | 122.845 [121.306–128.101] | 0.310695300 |
| Multi33 | 3 | 143.800 [140.899–149.736] | 0.310687108 |
| Multi33 | 4 | 170.033 [165.310–171.069] | 0.310645564 |
| Wide129 | 2 | 205.348 [204.910–206.673] | 0.451899908 |
| Wide129 | 3 | 291.194 [286.102–291.694] | 0.451916741 |
| Wide129 | 4 | 461.608 [457.101–462.367] | 0.451934501 |

Both higher orders have longer median total training times in all three cases.
Held-out loss changes have mixed directions: both higher orders worsen wide129;
order 3 also worsens scalar. No sampled training-loss trajectory increases in
any of the 27 measured fits. The maximum saved CPU-versus-GPU prediction
reference error is 2.220446049250313e-16 across these fits. That checks two
prediction implementations of each saved model, not equivalence between
different optimizer orders.

The synthetic summary records these memory payloads (MiB, 2^20 bytes) from the
first measured repetition for each case/order; full memory fields and all raw
results remain available. `Owned peak` is the benchmark's accounted maximum of
training payload and preparation payload, not a sampled device-wide VRAM peak.

| Case | Order | Training payload | Preparation peak | Owned peak | Histogram | Derivative scratch |
|---|---:|---:|---:|---:|---:|---:|
| Scalar | 2 | 17.207909 | 60.025581 | 60.025581 | 0.181313 | 4.000000 |
| Scalar | 3 | 19.267113 | 60.025581 | 60.025581 | 0.240517 | 6.000000 |
| Scalar | 4 | 21.326317 | 60.025581 | 60.025581 | 0.299721 | 8.000000 |
| Multi33 | 2 | 26.590538 | 7.505074 | 26.590538 | 2.845497 | 8.000000 |
| Multi33 | 3 | 31.537804 | 7.505074 | 31.537804 | 3.792763 | 12.000000 |
| Multi33 | 4 | 36.485069 | 7.505074 | 36.485069 | 4.740028 | 16.000000 |
| Wide129 | 2 | 21.891327 | 4.275528 | 21.891327 | 6.038612 | 2.000000 |
| Wide129 | 3 | 24.893768 | 4.275528 | 24.893768 | 8.041054 | 3.000000 |
| Wide129 | 4 | 27.896210 | 4.275528 | 27.896210 | 10.043495 | 4.000000 |

[synthetic/summary.json](synthetic/summary.json) preserves all timing samples,
inner-stage timings, losses, prediction-reference errors and memory fields.
The individual result directories and command receipts are in
[synthetic](synthetic). These bounded generated cases establish neither fastest
code nor superiority over other learning algorithms.

## Completed Nsight diagnostics

The successful [profiling summary](profiles-v2/summary.json) records three
Nsight Systems graph/NVTX runs and nine Nsight Compute captures, all verified
against the frozen binary, sources and fixtures. The
[corrected runner](run_profiles_v2.py) uses the actual Delicious 10,336 training
rows, 500 features and all 983 outputs, one round, depth 3, 32 bins, cap 1 and
tile 16. Compute captures the first derivative, cached-count root accumulation
and root split-candidate launch per order, in stream mode with full metrics and
uncontrolled clocks. These are diagnostic observations, not uninstrumented speed
rankings or coverage of every depth, round and output tail.

The [original attempt](profiles/summary.json) remains failed: three Systems
captures and one Compute derivative capture succeeded, then the root filter
matched no kernels because CUDA emitted explicit template-parameter casts.
The benchmark exited zero, but the missing Compute report correctly failed the
runner. The raw [no-kernel warning](profiles/ncu-o2-root/profile.stdout), failed
audit and incomplete marker are preserved. Corrected patterns were checked
against the observed Systems kernel names before the separate complete retry.
Only capture filters and output destinations changed; production code did not.

### Where recorded GPU kernel time goes

These percentages use the sum of all kernel durations in each complete Systems
trace; they exclude CPU gaps, memory-operation durations and non-kernel GPU work.
They include setup and prediction kernels, so they are not percentages of the
public training wall time. Each order launches 186 histogram accumulations,
186 split-candidate kernels and 62 derivative kernels in this one-round trace.

| Order | Histogram accumulation | Split candidates | Fused derivatives |
|---|---:|---:|---:|
| 2 | 70.737% | 28.212% | 0.113% |
| 3 | 59.673% | 39.608% | 0.070% |
| 4 | 54.723% | 44.709% | 0.056% |

Histogram accumulation plus split scoring account for 98.95%, 99.28% and 99.43%
of recorded kernel time. Computing the closed-form derivatives themselves is a
small phase. This identifies a more consequential target than micro-optimizing
the derivative formulas. It does not imply those percentages equal removable
work or that a proposed replacement will achieve any particular speedup.

The [Systems analysis](systems-analysis.md) and
[correlated trace evidence](systems-analysis.json) verify 2,899 captured kernels
per order, including 62 tree graphs of 23 kernels each. Every tree submission
scope contains one graph launch and zero host synchronization calls, device
allocations/frees, blocking copies or device-to-host transfers. GPU execution
extends beyond the short host submission range; attribution follows enqueue
correlation. Completed-tree exports occur outside those scopes, with 124
transfers totaling 495,432 bytes per capture. Their long host ranges mainly
wait for prior GPU work and are not transfer-cost measurements. Setup and
instrumentation event allocation are separately attributed before training.
The detailed Systems analysis also offers a narrower boosting-kernel denominator;
its percentages are explicitly distinct from the full-capture table above.

### What the hardware counters show

[compute-analysis.md](compute-analysis.md) and its
[raw metric extraction](compute-analysis.json) retain exact metric names, units,
source hashes and the nine captured identities.

| Root split observation | Order 2 | Order 3 | Order 4 |
|---|---:|---:|---:|
| Registers/thread | 72 | 90 | 96 |
| Register-limited blocks/SM | 3 | 2 | 2 |
| Achieved occupancy | 48.07% | 32.71% | 32.82% |
| FP64 pipeline active, elapsed cycles | 81.76% | 75.25% | 79.82% |
| Eligible warps/scheduler | 0.146 | 0.088 | 0.074 |

All nine captured kernels have zero measured register-spill instructions and
zero local-memory load/store sectors. Higher-order derivative kernels actually
use fewer registers than retained order 2 (24 versus 29), so derivative register
pressure and spilling are not supported explanations. Split arithmetic and its
register demand are supported concerns.

Root histogram reduction requests increase from 5,168,000 to 7,752,000 and
10,336,000: exactly 1.5x and 2x for the additional fields on this captured tile.
Counts remain cached. L2 throughput is 62.25%, 65.30% and 66.50% of peak while
combined DRAM read/write throughput is only 3.63%, 3.41% and 2.68%. These metrics
support investigating atomic reduction/cache work; they do not describe a
simple saturated off-chip bandwidth stream.

The 500-feature workload retains 256-thread split blocks despite roughly three
bins per feature. Root launches include four frontier-capacity slots while only
one is active. The source also forms a common parent leaf proposal before
per-bin candidate predicates. Removing inactive capacity/thread work and
checking whether the common parent arithmetic can be shared are concrete next
experiments. Source-level redundancy is not proof of the generated instruction
schedule, and none of these proposed changes has been implemented or ranked in
this experiment. The existing
[wide-feature split proposal](../../training/WIDE_FEATURE_SPLIT_EXPERIMENT.md)
requires its own correctness, exactness, quality and complete-operation gates.

## Final audit, evidence preservation and decision

The [final audit](loss-audit.json) passed with zero integrity errors and zero
pending jobs. It checks **111 saved loss curves, 2,148 finite loss values and
2,037 round transitions**, with **zero positive increases at zero allowance**.
Those curves include 62 real-data fits, 30 synthetic fits/warm-ups, 12 final
profile runs, five runs from the preserved profiling attempt and two earlier
smokes. Profile and smoke curves are explicitly diagnostic, not ranked timing
samples. Losses are the saved GPU-computed observations; the audit does not
independently recompute every round from predictions or prove future descent.

All 47 frozen production-source identities, nine preserved counting-source
identities and four frozen binary identities match. The original profiling
failure is separately verified and remains failed; the successful retry does
not convert it into a quality pass. Quality-gate failures throughout this report
remain failed despite the successful correctness and integrity checks.

The [measured source archive](production-provenance/measured-sources.zip),
[source archive audit](production-provenance/source-archive-audit.json),
[build environment](environment.json),
[configuration](production-provenance/CMakeCache.txt), raw commands and frozen
binaries preserve the implementation used for these observations. The
[artifact manifest](artifact-manifest.json) and
[seal verification](seal-verification.json) identify the final evidence files.
Hashes establish retained file identities, not independent reproducible
compilation or accuracy on fresh data.

Orders 3/4 remain explicit experimental binary/multilabel options. All counting
and trainer defaults remain unchanged. The new TrainConfig member changes its
C++ binary layout, so linked clients must rebuild; exported model format and
inference semantics are unchanged. This implementation demonstrates that the
closed forms can run inside the GPU-resident trainer. The measured costs,
clipping sensitivity and failed quality gates do not establish a uniformly
better optimizer, faster time-to-quality or a universally fastest booster.
