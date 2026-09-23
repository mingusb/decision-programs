# Booster instrumentation implementation and verification

The standalone [training instrumentation module](../../training/README.md) now
builds and runs. It provides deferred GPU stage timing, CPU spans, Nsight labels,
structured workload/memory records, immutable benchmark capture and audit,
diagnostic profiler/sanitizer commands, and weighted prediction-quality checks.
The runnable probe uses the existing counting library. No trainer or new GPU
algorithm was implemented in this step.

## Preservation and checks

- All seven CTest targets passed. After final Python validation fixes, all three
  CPU targets passed again: **35 observation tests, 38 evaluation tests**, and
  the disabled-recorder executable. Logs: [full run](ctest.log) and
  [final CPU run](cpu-final.log).
- The recorder suite passed **104 checks** covering pending events, stale and
  foreign tickets, capacity, capture rejection/recovery, changed-input graph
  replay, multiple streams, metadata, and CUDA error-state handling. It also
  passed [Compute Sanitizer memcheck](lifecycle-memcheck/profile.log).
- The integrated graph probe passed memcheck, racecheck, and synccheck with
  **zero errors, hazards, or warnings** in their respective summaries:
  [memcheck](memcheck/profile.log), [racecheck](racecheck/profile.log),
  [synccheck](synccheck/profile.log). Graph execution passed with instrumentation
  off, timing enabled, and NVTX enabled. [Direct-stream execution](stream-smoke/manifest.json)
  also passed repeated-output verification.
- The [disabled-path linkage](disabled-linkage.txt) contains libc and its loader,
  with no CUDA, NVTX, or recorder linkage. The type and its no-op calls are also
  checked at compile time. Caller-created argument expressions still require
  care if their work should disappear when instrumentation is disabled.
- [Counting source/library/executable hashes](counting-before.sha256) all
  [remain unchanged](counting-preservation.log). The new project imports the
  existing count archives rather than rebuilding them. No NVIDIA histogram
  reference is linked into the new probe.
- All **13 captured observation/profile directories** passed strict manifest
  and artifact audit. Executable, source and build-file snapshots are retained
  with each capture. [Final instrumentation hashes](instrumentation.sha256)
  cover the sources, binaries, and exported Nsight database.

The probe validates the final histogram from each repeated batch against exact
CPU counts. That does not independently validate every intermediate histogram
within the batch. The existing GPU counting algorithm and its earlier tests
remain separate from these instrumentation checks.

## Fresh instrumentation comparison

Four serial processes used off/timing/timing/off order, seed 20260922411,
16,777,216 uniform u32 inputs, 1,048,576 bins, u64 output, warm-cache graphs,
11 repetitions, batch four, and a requested two-second warmup. The same existing
window configuration (policy 2, 96 blocks, 524,288 counters per window) ran in
all four. Each repeated batch had its own events and pinned output snapshot.

| Capture | Median outer-event time per operation |
|---|---:|
| [1: off](calibration-1-off/result.json) | 716.920 µs |
| [2: timing](calibration-2-timing/result.json) | 626.600 µs |
| [3: timing](calibration-3-timing/result.json) | 687.400 µs |
| [4: off](calibration-4-off/result.json) | 713.528 µs |

The timing/off ratios are **0.874017** and **0.963382** for the two adjacent
pairs. They do not establish that instrumentation speeds up execution or has
zero cost. In the reverse pair, the full device interval instead increased
**1.48%** and CPU submission time increased **112%**. Clocks, process variation,
and submission effects remain uncontrolled. The forward/reverse comparisons
retain all phases and raw values: [forward](overhead-forward.json),
[reverse](overhead-reverse.json).

Independent audit verified **44 raw repetition timings, 176 timed operations,
and 50 recorder samples**, with matching executable hashes. The probe's outer
events and per-batch snapshots differ from the earlier ranking harness; these
observations are not a new comparison against NVIDIA's histogram reference.
The disabled recorder's structural absence is established separately from
empirical claims about complete-application timing.

Two initial runs remain in the record but are excluded from this comparison:
`graph-off-a` was a startup smoke observation; `graph-timing-a` overlapped initial
CTest activity and is retained for validation only. Both still pass artifact
and output-validation audit. Only the four fresh serial runs above inform this
overhead discussion.

## Nsight verification

[Nsight Systems](nsys/manifest.json) and [Nsight Compute](ncu/manifest.json) both
completed successfully. These are diagnostic records and the comparison tool
rejects them as ranking inputs. NCU collection is bounded to two launches and
can therefore include setup/preflight work; it is not a steady-state campaign.
The native `.ncu-repz` format used by the installed profiler is supported by
capture and audit, alongside `.ncu-rep`.

Independent inspection of the Systems database found exactly nine completed
`ghb` ranges, matching the probe's recorded sample IDs:

- 0: initialization; 1: upload.
- 2, 4, 6: histogram batches.
- 3, 5, 7: output downloads.
- 8: CPU evaluation.

Range IDs were distinct and registered names resolved correctly. The measured
submission window contained three graph launches and three device-to-host copies,
with event records/capture checks and **no CUDA allocation, event-query, or
synchronization calls inside that window**. Completion and timestamp collection
occurred afterward. This is not a claim about hidden allocations inside driver
or profiling-library internals.

Exports are stored outside the immutable capture directories:
[stage summary](nvtx-summary_nvtx_sum.csv), [Compute details](ncu-details.txt),
and `nsys-export.sqlite`. The database passed SQLite integrity checking and its
SHA-256 is
`b02760d88d09ae38f9a73559184f472f813b867aab6ff050149e4ffc97d02b76`.

## Prediction quality and remaining integration

The evaluator supports weighted scalar and multi-output regression, binary
classification with tied-score AUC, and multiclass classification. Every
applicable metric and regression output participates in comparison. Matching
dataset/split identities, targets, weights, numerical settings, original CSV
hashes, and recomputed metrics are required before a comparison can pass.
Missing or changed source evidence is rejected. Exact decimal differences are
retained for reviewing tolerance-boundary decisions.

These checks evaluate supplied predictions; no model was trained in this work.
Actual gradient/Hessian construction, per-node training histograms, split search,
row routing, and prediction kernels still need implementation and comparative
measurements. The instrumentation is ready to measure those stages when they
exist. Internal stage timing within an already captured whole-training graph
is not implemented; the current API safely wraps separate graph replays.
