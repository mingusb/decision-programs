# Single-allocation prediction model experiment

Status: implementation, correctness, Nsight work-count confirmation and the
unprofiled 25-shape timing matrix completed. E has not earned default promotion;
the continuous optimization goal remains active.

The explicit `PredictionPolicy::fused_output_slab` combines the ordered forest's
four model arrays into one allocation/upload with 256-byte region alignment.
The `per_tree` default and `fused_output` control remain callable and unchanged.
The contract and pre-code selection are in
`training/INFERENCE_MODEL_SLAB_EXPERIMENT.md` and
`results/optimization-20260923/MODEL_SLAB_SELECTION.md`.

## Validation

- Fresh Release CUDA C++23 / SM86 build, resource diagnostics on and compiler
  time tracing off: all 20 CTest cases passed, including 5 CPU and 15 GPU tests.
- The shared production layout helper passed 118 CPU checks for hand-derived
  offsets, absent regions, padding, overflow and strict free-budget boundaries.
- The expanded prediction suite passed 745,771 exact checks under padded
  memcheck/full leak checking and global/shared initcheck, with zero errors or
  device leaks. It covers both fused policies against per-tree and CPU raw
  references, including objective, missing/category, output/row tail, graph,
  signed-zero, subnormal and cancellation cases.
- A separate fault harness injects failures at each of four CUDA allocation,
  nine asynchronous-copy and one aligned host-slab allocation positions, for
  both empty and nonempty forests. Its strengthened version passed 17,043
  checks under padded memcheck, with zero device errors/leaks and exact raw and
  transformed recovery predictions. Original 10,603-check observations and
  source are retained in `fault-v1/` and `memcheck-fault/`.
- All 314 device code/constant sections match A/B byte for byte after unique
  full-symbol demangling; 102 names changed their compilation-specific suffix.
  See `runtime-mapped-sections.json`. This makes no host-code equality claim.

Source/build snapshots and identities are under `source-snapshot/`. The
pre-layout-helper source is separately retained in `pre-layout-extraction/`.
Fault injection validates synchronous API error propagation and tracked device
cleanup; it does not simulate device loss, physical fragmentation, every host
allocation failure or arbitrary asynchronous errors. See `FAULT_VALIDATION.md`.

## Nsight confirmation

Two isolated same-binary captures use the same frozen 983-output Delicious
model and actual 2,584 validation rows. Reference prechecks and warmups are
outside capture. The complete calls have identical 18-kernel sequences.

| Whole-call observation | Fused control | Slab candidate |
| --- | ---: | ---: |
| CUDA allocations / frees | 7 / 7 | 4 / 4 |
| Host-to-device transfer activities | 538 | 535 |
| Host-to-device bytes | 5,630,988 | 5,631,364 |
| `cudaMemcpyAsync` calls | 539 | 536 |
| `cudaMemcpy2DAsync` calls | 16 | 16 |
| Explicit stream waits | 19 | 19 |
| Device-to-host activities | 17 | 17 |

The 376 extra bytes are alignment padding. These are profiler work counts,
not performance rankings. Raw captures and the reproducible comparison are in
`nsys-fused-control/`, `nsys-slab/` and `transfer-comparison.json`.

## Timing conditions and Windows Terminal change

Before timing, native Windows counters showed Windows Terminal PID 20860 using
16–24% of a 3D engine. DXGI enumeration confirmed that engine's adapter LUID
`0x00000000_0x03bfea22` is the RTX A5000, not the integrated Intel adapter. All
agents were idle and no benchmark campaign had begun. A prepared serial driver
is `results/optimization-20260923/run_prediction_slab_campaign.py`: 25 shapes,
direct B/E and per-tree/E comparisons, 15 alternating pairs and 3 warmups each.

The user then requested configuring Terminal to avoid GPU rendering. The
documented global `experimental.rendering.software` setting was set to `true`
in the installed Terminal 1.24 settings, with an exact original backup beside
the file. The change receipt is `terminal-software-rendering-change.json`.
Only that semantic setting changed; JSON was validated before and after.
A same-byte native Windows write also triggered its settings watcher.

Before the user's Terminal restart, post-change counters still attributed
roughly 16.7–17.4% 3D-engine activity to
the Terminal process. This does not establish which remaining rendering or
composition component caused it, and the setting has **not** been demonstrated
to eliminate GPU interference. Video-decode activity from another process was
also observed. No application was stopped or restarted. Raw counter samples,
adapter mapping and change receipts are retained alongside this report.

The user subsequently restarted Terminal. Its new process was PID 14076;
software rendering remained enabled. Across four native counter samples the
largest Terminal engine value was 5.7543%, with no other sampled engine above
5%. Benchmarking then proceeded with residual desktop activity explicitly
recorded in `TIMING_CONDITIONS.json`; this is not a zero-background-load claim.

## Complete-call timing results

The first serial campaign completed 37 comparisons before its process was
interrupted. Job 38 wrote a result but no process-exit receipt; its samples are
retained and excluded from the accepted matrix irrespective of their values.
After verifying no driver/benchmark was live and all 114 original identities
still matched, `resume_campaign.py` reran only the 13 comparisons lacking full
records in a separate directory. It revalidated the original 37 result/exit/raw
receipts and all resumed identities. The combined manifest maps all 50 accepted
comparisons without rewriting any original observation. The pause and both
attempts remain visible.

All 25 shapes have 15 alternating pairs for B/E and 15 for per-tree/E, after
three warmups. All raw/transformed reference and per-sample exactness gates
passed. Timings include the complete synchronous prediction operation. The
casewise bootstrap classification is:

| Comparison | E faster | E slower | Inconclusive |
| --- | ---: | ---: | ---: |
| E versus existing fused B | 4 | 1 | 20 |
| E versus per-tree | 16 | 0 | 9 |

The one B/E regression is the three-output, three-tree, 32-row shape: median
paired E/B ratio 1.04777, 95% interval 1.02148–1.10548. Its policy medians were
0.85425 ms for B and 0.89372 ms for E. E's local wins versus B were the
three-output/65,536-row case, 65-output/32-row case, empty three-output forest
at 4,096 rows, and one-tree three-output forest at 65,536 rows.

On actual Delicious validation, E/B was inconclusive: paired ratio 1.03187,
interval 0.95513–2.03271. E/per-tree was 0.49087, interval 0.33568–0.53599;
that supports the already measured forest-fusion benefit, not an additional
slab benefit. On 1,024 outputs and 65,536 rows, E/B also remained inconclusive
(1.01967; 0.98769–1.03127), while E/per-tree was 0.78903
(0.76858–0.80281).

Intervals are casewise 95% percentile bootstraps over median paired ratios,
20,000 resamples with fixed seed; there is no multiplicity correction or
universal ranking. Residual desktop activity and the campaign interruption
limit interpretation. Smaller API counts have not established a consistent
complete-call improvement, and E remains explicit/experimental. All ratios,
timings, payloads and hashed result links are in `performance-summary.json`.

Microsoft documents the software-rendering setting here:
https://learn.microsoft.com/en-us/windows/terminal/customize-settings/rendering
Its release-1.24 renderer update path supports switching on settings reload:
https://github.com/microsoft/terminal/blob/release-1.24/src/cascadia/TerminalControl/ControlCore.cpp#L918-L954
This setting does not disable Windows desktop composition.

## Next work

D2 is selected in `training/FINAL_STATUS_ENCODING_EXPERIMENT.md`: immutable
per-feature lengths and one final quantizer status check, preserving GPU
encoding and memory layout. Its implementation/validation is in progress with
the current preparation path retained as control and default.

An additional frozen-B Nsight capture at 65,536 rows and 1,024 outputs is in
`nsys-fused-large/`. Its compiled provenance points to the E source snapshot,
not subsequent D2 edits. It will guide a separate large-output architecture
analysis; captured elapsed times are diagnostic, not ranking evidence.
