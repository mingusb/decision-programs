# Final-status inference encoding

The explicit `EncodingPolicy::final_status` candidate is implemented and passes
the correctness and diagnostic gates below. Unprofiled performance ranking is
pending. No production default, counting kernel or training algorithm changed.
The pre-code contract is `training/FINAL_STATUS_ENCODING_EXPERIMENT.md`.

The frozen build is `build/final-status-20260923`. Its source and six build
artifacts are preserved in `source-snapshot/manifest.json` (95 files). Subsequent
functional-layout and dead-state edits belong to a separate build; current
working-tree contents are not the compiled-source identity of this experiment.

## Completed evidence

- All 20 CTests passed (5 CPU and 15 GPU), run serially; `ctest.log`.
- Quantization: 6,238,421 checks under padded memcheck, all-address-space
  initcheck and synccheck. All report zero errors. Pinned host input was
  supported and exercised, including changed-input recovery.
- Prediction: 2,251,411 exact checks under padded memcheck and all-address-space
  initcheck. Both report zero errors. Memcheck on both executables reports zero
  leaked bytes and allocations.
- Offline unique full-demangled symbol comparison: all 314 device code and
  constant sections have identical bytes to the previous E/B library. This
  does not assert identical host code or equal execution time.

The existing tests cover policy combinations, frozen raw/transformed outputs,
multi-tile metadata, minimum budgets, invalid inputs, cumulative infinity
status, capture rejection and recovery. This is not exhaustive fault injection
at every CUDA API position, physical out-of-memory or device-loss validation.

## Nsight work-count verification

Same-binary isolated complete calls on the frozen Delicious model and input:

| Work | Per tile | Final status |
|---|---:|---:|
| Explicit stream waits | 19 | 4 |
| `cudaMemcpyAsync` calls | 539 | 524 |
| Device-to-host activities | 17 | 2 |
| Host-to-device activities | 538 | 538 |
| Host-to-device bytes | 5,630,988 | 5,630,988 |
| Kernels | 18 | 18 |
| Allocation/free pairs | 7 | 7 |

Kernel names, order and launch dimensions match exactly. See
`transfer-comparison.json`, `nsys-per-tile-v2/` and `nsys-final-status/`.
These are counts, not a profiler-based performance ranking. The first capture
attempt, `nsys-per-tile/`, failed before benchmarking because the command used
`--features` instead of `--features-bin`; its failed receipt and logs remain.

## Remaining experiment

Run the predeclared same-binary paired complete-prediction matrix and separate
minimum-budget encoding stages with builds and agents idle. Preserve desktop
conditions, every raw sample, process receipt and source/model/binary identity.
The final-status policy delays error detection and uses F host length entries
instead of the control's K entries; both costs belong in its comparison.

All current observations are diagnostic/correctness evidence. No speedup or
default-promotion conclusion is established yet.
