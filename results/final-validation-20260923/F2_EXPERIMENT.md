# F2 final-output validation experiment — prepared, not executed

Recorded 2026-09-23. The selection and exact contract precede this implementation
in `training/FUNCTIONAL_GPU_NEXT_STEP.md`. While the user is gaming, this work
does not authorize GPU queries, workloads, diagnostics or profilers. GPU
correctness, timing and generated-device-code inspection remain pending.

The comparison holds B (`fused_output`, `per_tile` encoding) fixed and changes
only final-output checking from `OutputValidationPolicy::host` to `::device`.
`--policy compare-validation` emits the existing `fused_output` label and the
experimental `fused_output_device_validation` label. The default stays host.
D2 final-status encoding and E slab packing are separate comparisons; their
effects cannot explain an F2 result. This slice does not complete the functional
migration of prediction or training.

## Frozen inputs and correctness gates

Before the authorized campaign, save source/build flags, binary and library
SHA-256 identities, model bytes, actual feature bytes and prior input identities.
Reuse the models and inputs listed in
`results/optimization-20260923/prediction-matrix-idle/observations.json` and
`prediction-actual-idle/observations.json` under that same parent directory.
Check their recorded hashes before and after execution. Synthetic feature
generation stays `deterministic_model_derived_v1`; preserve the feature hash.

The benchmark checks raw and transformed F2 results bit for bit against the
per-tree/per-tile/host-validation reference before timing. Both paired and
isolated F2 modes also check host-validated B against that reference. Every
warmup and recorded sample is checked outside its timer. Keep existing CTests
and the F2 special-value, tail, empty, overflow, fault/drain and reuse gates;
sigmoid must accept an infinite intermediate margin when its final result is
finite. Require zero differences and retain every failed gate. Actual labeled
data must retain applicable aggregate, per-label and signal-quality checks;
synthetic cases establish no signal-quality result.

## Predeclared serial timing protocol

After GPU access resumes, run all cases serially with no concurrent build,
analysis or profiler. Use three warmups per policy and 15 alternating paired
samples (30 complete calls) per case, retaining pair and position. Fifteen
pairs give an 8/7 split of which policy runs first; do not describe that as
exactly equal order counts. Do not replace failures or discard slow samples.

Use the six frozen matrix models (`regression1`, `binary1`, `independent3`,
`multiclass5`, `independent65`, `independent1024`) at rows 32, 4,096 and 65,536,
each with and without `--raw`: 36 cases. This includes narrow-output,
small-row, regression, sigmoid, softmax and the 512 MiB large-output case.
Also use actual Delicious validation (2,584 rows, 983 outputs, exact existing
FP32 feature file), with and without `--raw`: 38 cases total. Preserve the
matrix's listed model order and ascending rows, transformed then raw; append
Delicious transformed then raw. Record telemetry and desktop interference
outside each timed process, all stdout/stderr, exit status, timestamps and
partial samples. Every result filename must be new.

The invocation shape is:

```text
ghb_prediction_bench --model FROZEN_MODEL --rows ROWS --pairs 15 --warmup 3
  --policy compare-validation [--features-bin FROZEN_FP32] [--raw]
  --output NEW_RESULT.json
```

Unset `GH_PROFILE_CAPTURE` for ranking. The unchanged timer includes complete
vector-returning calls: host packing and result construction, GPU allocation,
encoding, uploads, prediction, transforms, finite checking, downloads, checked
completion and device cleanup. It excludes model/input loading, context warmup
and external result comparisons. F2 retains all result-download bytes, adds a
contiguous GPU read of the final FP64 output, and adds one four-byte device
status allocation, clear and download for nonempty rows. JSON reports the
status bytes separately from B's existing device payload; zero rows add none.
This is not a resident-output measurement.

Compute the median paired F2/B ratio and its casewise 95% percentile bootstrap
interval using 20,000 resamples and fixed seed 20260923, preserving raw arrays
and the analysis code. Only an upper endpoint below 1 supports a casewise
speedup claim. Keep inconclusive cases and regressions visible; no universal
ranking, default promotion or multiplicity-adjusted claim follows.

## Diagnostics after unprofiled timings

Capture the same binary/model/input separately with `--policy fused-output`
and `--policy fused-output-device-validation`, `--pairs 1 --warmup 3` and
`GH_PROFILE_CAPTURE=1`, using the existing CUDA-profiler/NVTX capture boundaries.
The isolated candidate still runs both reference prechecks before capture.
Confirm unchanged numerical kernels and full result-download bytes, removal of
the CPU finite scan, and the added device scan/status sequence. Inspect scan
loads, vote/atomic lowering, resources and unexpected local storage or FP work;
retain unsupported diagnostics and failures. Profile dense-invalid rejection
separately to expose status contention, without ranking profiler times. Preserve
the full F2 plan's failure-injection and memory/synchronization diagnostic gates
before any future R0 integration.
