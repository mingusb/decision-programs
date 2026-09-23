# GPU-resident booster implementation and measured results

22 September 2026, RTX A5000 Laptop (SM86), WSL2, CUDA 13.4.59,
CUDA C++23. GPU workloads ran serially. Clocks were not locked; short repeated
measurements are observations, not confidence intervals or universal rankings.

## Implemented behavior

Training now fits exact feature bins, validates dense inputs, computes weighted
base scores, builds weighted histograms, chooses splits, constructs trees,
routes rows, updates predictions and reduces objective values on the GPU.
Dense feature bins remain resident. GPU prediction also encodes input features
on the device. Host setup, transfers, completed model export/serialization and
the explicit independent CPU prediction/reference tests remain.

The builder supports ordinary stream submission and reusable whole-tree CUDA
graphs. Device active counts guard the bounded level schedule. No split-winner
download, CPU child decision or map upload occurs between levels. Graph output
selectors reuse at most two graphs across thousands of outputs.

Profile-guided follow-up changes are included:

- Bounded pinned model-export batches replace two waits per tree with one wait
  per batch. The same device tree workspace is reused in stream order. Requested
  batch width is bounded by derivative/output tiles and a 64 MiB pinned-node cap.
- Contiguous complete-matrix uploads use a single contiguous copy; partial feature
  tiles retain pitched copies.
- Regression/binary loss assigns adjacent elements to adjacent threads, including
  a scalar specialization. Multiclass keeps its coupled rowwise log-sum-exp.

Exact binning compares owned radix8/four-pass and radix4/eight-pass implementations.
Both use conventional count/scan/scatter sorting, not Onesweep. The current code
does not establish that this sorter or the complete booster is the fastest known.
The implemented experiment decisions were recorded before their respective code:
[binning](../../training/QUANTIZE_EXPERIMENT.md),
[resident trees](../../training/RESIDENT_EXPERIMENT.md),
[export batching](../../training/EXPORT_BATCH_EXPERIMENT.md), and
[loss layout](../../training/LOSS_LAYOUT_EXPERIMENT.md).

The original counting kernels, policies, defaults and frozen binaries are
unchanged: every entry in [the preservation check](counting-preservation.log)
passed. The production library's [symbol inventory](production-symbols.txt)
contains no CUB/Thrust/NVIDIA histogram implementation. NVIDIA sorting is an
isolated benchmark executable, disabled by default in CMake.

## End-to-end training results

`total_train_ms` includes preprocessing, allocations, uploads, initialization,
tree building, model export, reporting and cleanup, after CUDA context startup.
Data generation, held-out prediction/validation and artifact writes are separate.
All table comparisons use matching generator-version-2 datasets and training
settings. Final scalar and 4096-output comparisons used fresh reverse-order
blocks; the 129-output confirmation uses a new seed after selecting batch 16.

| Workload | Frozen hybrid total, ms | GPU-resident total, ms | Observed result |
|---|---:|---:|---|
| Scalar regression, 65,536 rows × 32 features, 10 rounds, depth 5, 64 bins, shared histogram | 230.47–256.32 | graph/batch 1: 38.92–41.97; stream/batch 1: 39.94–40.38 | roughly 5.5–6.6× faster; stream/graph close |
| 129-output regression, 4,096 rows × 16 features, 3 rounds, depth 2, 32 bins, fresh seed | 155.92 | graph/batch 16: 83.86 | 1.86× faster, one confirmation pair |
| 4096-output binary, 1,024 rows × 16 features, 1 round, depth 2, 16 bins | 1375.51–1388.36 | graph/batch 16: 755.98–762.85 | 1.80–1.84× faster |

The multi-output runs use independent scalar trees and output tiles of 16, with
the owned global histogram forced for a clean comparison. They are synthetic
fixtures, not NLP datasets or evidence of accuracy superiority over other models.

The first resident 129-output graph implementation regressed from 133.43 ms to
146.25 ms on its matched dataset. That observation remains in
[the first campaign](campaign.log); it was not discarded. Nsight identified the
completed-tree export waits, prompting the separate batching experiment.

| Same final binary, fixed graph execution | Compact export total, ms | Batch 16 export total, ms |
|---|---:|---:|
| 129 outputs, same seed as first campaign | 133.70–153.00 | 91.40–94.35 |
| 1024 outputs, 4,096 rows, 2 rounds | 641.02–653.43 | 467.21–472.62 |

Batch 1 measured 102.50 ms and 597.00 ms respectively. Stream/batch 16 measured 136.76 ms
and 594.91 ms. Raw results, orders, seeds, memory and GPU telemetry are retained in
[the optimized campaign](campaign-optimized.log) and its per-case captures.

## Larger preparation cases

These runs use zero boosting rounds, testing preparation/initialization only.
Fitted metadata matched the frozen hybrid exactly for both radix policies.

| Rows ×features | Hybrid complete preparation/train call, ms | Optimized GPU call, ms |
|---|---:|---:|
| 1,048,576 × 8 | 919.53 | 29.20–31.59 |
| 16,777,216 × 1 | 2293.78 | 88.32–89.63 |

The GPU quantization phase for 16M× 1 fell from 112.98–119.75 ms in the first resident
candidate to 39.77–40.14 ms after the contiguous-upload change. These are whole
phase wall times including upload, allocation and metadata synchronization.
The hybrid's quantization phase excludes its later packed-bin upload, so that
phase alone is not a matching CPU/GPU boundary; the complete-call table is.

## Profiler findings

Nsight Systems records actual CUDA API activity, kernels and NVTX stages; these
diagnostic timings do not rank the uninstrumented candidates.

- The 129-output/3-round capture has 387 complete trees. Compact export required
  774 export waits. Batch 16 required 27. Whole-process stream-sync counts changed
  from 790 to 43; those include additional setup, loss and inference operations.
- Whole-tree ranges contain no per-level host waits or device-to-host transfers.
  Batch export scopes overlap intervening GPU work and must not be interpreted
  as isolated copy duration or added to tree time.
- The first 16M× 1 pitched upload spent 101.38 ms in a host API call. The contiguous
  copy removes that unnecessary pitched-copy path; host API time is not DMA time.
- Scalar loss's recorded kernel duration changed from 34.31 ms to 1.06 ms after the
  coalesced layout change. This is a diagnostic comparison with unlocked clocks.
- In the final node-level graph trace, global histograms account for 45.6% of
  recorded kernel time and split candidates 24.0%. Histogram and split work are
  still substantial optimization targets.

[The trace audit](trace-audit-verified.md) matches all 2,073 emitted stage IDs/names to
NVTX and checks SQLite integrity. Both wide captures export 95,976 bytes through
774 device-to-host copy calls: batching removes dependency waits while preserving
the copy count. These counts are scoped to completed-tree exports, not setup or
prediction. Host tree-submit ranges have zero correlated device-to-host calls
and zero runtime stream/device/event waits.

The first wide trace used graph-level tracing and omitted graph-node kernels;
the final batched trace explicitly uses node-level tracing. Their kernel totals
are therefore not directly comparable. Native `.nsys-rep`, exported SQLite,
CSV summaries, benchmark metadata and command captures are retained here.
Nsight Compute native `.ncu-repz` reports and text exports cover both radix
scatter variants and the final scalar loss kernel. It measured radix8 scatter
at 40 registers/thread, 96.09% achieved occupancy; radix4 at 41 registers/thread,
79.17%. The final scalar-loss invocation reports 73.29% DRAM throughput,
84.14% SM throughput and 96.28% achieved occupancy. Those counters explain
individual invocations, not a complete training or sort ranking.

## Correctness and quality

All 8 final CTest suites passed, including 123,864 trainer checks, 55,845 primitive
checks, 859 initialization checks, 3,600,102 exact quantization checks and the
recorder/CPU evaluator tests. Final memcheck passed on trainer, kernels,
initialization and quantization. Kernel racecheck reported zero hazards;
kernel/quantization synccheck reported zero errors. Quantization racecheck also
passed on the unchanged kernels before the upload-only change. Final sanitizer
commands record the exact executable hashes. See `ctest-optimized.log`,
`optimized-provenance/build/booster-resident/Testing/Temporary/LastTest.log`,
and `final-*-command.json`/stdout artifacts.

**The strict zero-regression quality gate does not pass for the wide models.**
[The optimized audit](quality-optimized/summary.md) keeps every failure:

- All 26 trained evidence models pass against their own constant base predictions.
- All 4 final scalar-vs-hybrid comparisons pass every applicable metric.
- All 15 wide-vs-hybrid comparisons and all 4 same-binary batch 16-vs-compact
  comparisons fail zero allowance on some metrics. Maximum deterioration is
  3e-16 in a metric's native units. No classification decisions changed.
- Feature cuts/categories are identical, but floating-point atomic/reduction
  order changes some tree splits and serialized leaf values. This is not
  bit-identical training.
- Exact rational leaf-region intersections cover missing and every retained
  encoded-bin combination, rather than only held-out rows. The largest bound on
  whole-model exact-real raw-margin discrepancy is 2.072e-15 (1.565e-15 in batching
  pairs), excluding subsequent runtime arithmetic rounding. Apparent large
  same-path leaf differences come from changed child orientations.

These numerical bounds do not convert failed zero-allowance gates into passes,
establish future-data accuracy, or prove no loss whatsoever. There was no
invalid evidence in either quality audit. The initial audit and observations
remain in `quality-audit/`; optimized results are separate in `quality-optimized/`.

## NVIDIA sorting diagnostic and remaining work

An isolated benchmark using CUB 3.4.2, NVIDIA's CUDA parallel-algorithm library,
compares three complete sorting boundaries on
identical resident unsigned 32-bit input, preserving duplicates. Each of 21
warmup/timed outputs is independently checked against per-feature CPU sorting.
Timings exclude allocation/transfers/validation; the tagged 64-bit method includes
GPU pack and strip. Five raw samples and execution orders are retained.

| Uniform full-width keys | Segmented u32 median, ms | Per-column u32 median, ms | Tagged u64 median, ms |
|---|---:|---:|---:|
| 65,536 × 32 | 0.650 | 4.570 | 0.810 |
| 1,048,576 × 8 | 8.111 | 1.632 | 3.039 |

These sorting-only diagnostics are not complete-quantizer comparisons and do
not show that our sort beats NVIDIA's. They demonstrate why sort decomposition
must depend on segment size. Owned Onesweep-family sorting, low-cardinality
hash deduplication, fewer/smaller weighted histograms, partitioned row work and
vector-leaf architectures remain measured-experiment candidates.

Current defaults remain stream execution, radix8, compact export. The measured
wide graph/batch 16 path is explicit (`--tree-execution graph --tree-export-batch 16`).
No universal default or accuracy-preserving promotion is claimed from these short fixtures.
The library still implements the documented objective/feature subset; full
framework parity, real NLP quality and broad competitive rankings are unproven.

## Reproduction and provenance

Use [the trainer guide](../../training/TRAINER.md) to build `build/booster-resident`.
`run_campaign.py` and `run_optimized_campaign.py` record the workload schedules
for the first and final campaigns. They refer to the working build; for an exact
historical replay, use the retained per-case command with the matching frozen
executable and a new output directory. Existing results must not be overwritten.
`run_diagnostic.py` captures profiler/sanitizer commands without a shell.
Quality and trace audits run on CPU artifacts only.

`first-provenance/` freezes the first resident source/build, while
`optimized-provenance/` freezes the final source/build/tests. The hybrid executable
and its original sources remain in the earlier campaign's immutable
`baseline-provenance/`. Each source snapshot has a SHA256 manifest; benchmark
captures retain binary hashes, and the final evidence manifest covers this
directory. Do not rebuild the historical `build/booster` comparison in place.
