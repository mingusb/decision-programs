# One final inference-encoding status check

Selected 2026-09-23 before production edits. This is D2 from
`INFERENCE_TRANSFER_EXPERIMENT.md`, tested independently before D1 or D1+D2.
The user has released the GPU. Recorded residual desktop rendering remains an
experimental condition, not a zero-load claim. No default is changed.

## Evidence and selection

The single-model-allocation E experiment confirmed its work reduction but did
not establish a general win: of 25 B/E shapes, 4 casewise intervals favored E,
1 favored B and 20 included equality. Preserve that candidate and its failure
to earn promotion. The source/binary baseline is the completed E snapshot at
`results/prediction-slab-20260923/source-snapshot/`, with its same-binary B
control and all 314 compiled device sections unchanged from the preceding B.

The fresh isolated B capture on Delicious shows 500 dictionary uploads, 16
encoding tiles/status downloads, 539 asynchronous-copy calls, 19 explicit
stream waits and 18 kernels per complete prediction. D2 removes 15 status
downloads/checkpoints while leaving encoding, model upload and kernel work
unchanged. D1 would instead batch 516 dictionary/length uploads into 16, but
its globally padded host representation can grow to 128 MiB for skewed
dictionaries. Compact metadata, native batched transfers and an explicit
resident predictor remain relevant separate candidates. The latter removes
more repeated work but needs a new construction/use/lifetime boundary.

Choose D2 now as the smallest independently measurable scheduling change.
This selection does not assert that it is faster than D1, E or resident
inference. Pageable runtime copies can still block; profiler API waiting is
not a prediction of removable elapsed time. Relevant primary implementation
is the current `encode_quantize` and `encode_tile` in `src/quantize.cu`; primary
runtime contracts and alternative analysis are linked in
`INFERENCE_TRANSFER_EXPERIMENT.md`.

## Exact API, memory, numerical and device contract

Add independent `EncodingPolicy::{per_tile, final_status}`. The current
four-argument `encode_quantize` remains the unmodified per-tile control.
A five-argument overload selects that control or the explicit final-status
implementation, rejecting unknown enum values before dispatch. Extend
`Model::predict_gpu` with an optional fourth encoding-policy argument whose
default is `per_tile`. Prediction policy and encoding policy remain separate.
All three existing prediction policies support the explicit choice; production
prediction/count/training defaults and `fit_quantize` remain unchanged.

Inputs are the same immutable Dataset and fitted Feature vectors. GPU bins,
feature metadata bits, offsets/types, view dimensions, owned resident bytes
and peak device payload must match the control exactly. Reuse its layout,
budget selection, kernel, grid/block sizes, feature dictionary order and
missing/unseen/category/numeric semantics. No CPU feature processing, floating
reordering, approximate math, extra kernel, atomics or new GPU scratch.

Replace the reused K-entry host length vector with an immutable F-entry vector
populated before any upload or drain guard. Upload the appropriate subrange
for each tile. Enqueue input/metadata/length uploads and encoding in the same
stream, then download cumulative status and perform one checked stream wait
and status validation after the final tile. Status is zeroed once and the
unchanged encoder only atomically ORs error bits. No earlier infinity error
may disappear. All host sources, status destination and device owners must
outlive queued work, including exceptional exits. Deactivate the drain only
after successful final synchronization and validation.

Preparation remains synchronous on return, and capture remains rejected.
Unknown policy, invalid shapes/features/budgets and infinity remain errors.
Infinity detection moves to the end; early-abort latency and precedence over
a separate later CUDA error are not preserved. Host allocation/CUDA failures
propagate with safe draining and release; no fallback is introduced. Zero-row
Model prediction preserves its no-CUDA return after ordinary validation.

For Delicious, unchanged device resident/scratch/peak payloads are
2,588,004 / 331,012 / 2,919,016 bytes. The minimum K=1 peak is 2,598,352 bytes.
Host lengths increase from 128 to 2,000 bytes. Charge that extra 1,872 bytes
and its initialization in complete-call timings. Expected B-to-D2 counts:
539 to 524 `cudaMemcpyAsync`, 19 to 4 explicit waits, 17 to 2 D2H activities;
538 H2D activities, 18 kernels and 7 allocation/free pairs stay unchanged.
These expectations exclude E's separate three-copy/allocation reduction.

## Fair experiment and acceptance

First run exact encoding checks against the current GPU path and explicit CPU
references: one/multiple/tail tiles, narrow and wide features, empty metadata,
numeric and categorical limits, NaNs/signed zero/unseen categories, exact
minimum memory and one byte below, changed-input repeated calls, capture
rejection and infinity in early/middle/late tiles. Verify both accounting and
bins/metadata. Exercise pinned external input where supported to reduce
accidental dependence on synchronous pageable staging. Retain any unsupported
diagnostic explicitly; do not represent it as a pass.

Extend frozen-model raw/transformed prediction exactness across all existing
prediction policies/objectives and zero-row/empty-forest/wide-output fixtures.
Run applicable memory, initialization and synchronization diagnostics and
bounded exceptional-path tests. Keep existing training/count tests intact.
Compare generated device sections; the host scheduling experiment must not
silently change the GPU arithmetic. No quality allowance is introduced.

Benchmark B/per-tile versus B/final-status in one binary, alternating at least
15 pairs after 3 warmups. Retain all E matrix shapes and actual Delicious data,
plus a wide-feature, larger-row and skewed-dictionary case, and a bounded
encode-only minimum-budget case. The primary boundary remains the complete
synchronous prediction call, including validation, packing, allocation,
quantization, transfers, output validation and cleanup; result comparison and
file/context setup stay outside. Encode-only timings are separate evidence.
No profiler timings rank implementations. Preserve raw observations, failed
attempts, source/build/model/input hashes and operating conditions; run GPU
work serially with agents/builds idle during uninstrumented measurements.

After correctness and timings, isolate one complete call of each policy in
Nsight to confirm the predicted work counts and unchanged kernel sequence.
Casewise paired-bootstrap intervals and exactness gates determine local wins;
inconclusive/regressing cases remain visible. Do not combine D1 or E until
their own contribution and capacity contract are independently accounted for.
