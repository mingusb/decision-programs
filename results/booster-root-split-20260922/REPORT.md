# Batched roots and small-bin split search

2026-09-22, RTX A5000 Laptop GPU (SM86, 48 SMs), CUDA 13.4, CUDA C++23.
Two independent opt-in candidates are implemented and validated. Wide-output
training improved in several measured cases, but neither a universal speed win
nor a passing zero-allowance quality gate was established. Defaults are unchanged.

## Implementation and contracts

The pre-code comparison and experiment are in
[`training/ROOT_SPLIT_EXPERIMENT.md`](../../training/ROOT_SPLIT_EXPERIMENT.md).

- **R, `--root-histogram batched`:** compute a bounded batch of output roots in
  one clear/accumulate operation. Adjacent output lanes share feature-bin loads
  and read adjacent FP64 derivatives. Tree initialization copies the selected
  cached root into the existing workspace without an additional launch. Deeper
  histograms, independent scalar trees and split semantics are retained. The
  multiclass derivative snapshot still covers all classes before any update.
- **S, `--split-policy warp32`:** use 32-thread feature candidate and node winner
  kernels when both maximum feature bins and feature count are <=32. Other shapes
  call our existing 256-thread implementation. Missing/category behavior, clipping,
  tie rules, minimum constraints and useful-lane arithmetic order are retained.
- R keeps exact u64 counts and FP64 sums with unspecified atomic addition order.
  S requires identical result fields on identical histogram inputs; this does
  not make independently accumulated training histograms deterministic.
- Root cache bytes are included in both histogram and device budgets. Reported
  `root_histogram_bytes` is a subset of `histogram_bytes`, not extra unaccounted
  memory. Actual additional cache: 186,240 bytes for 129 outputs; 94,080 bytes for
  the 1,024/4,096-output cases; 58,200 bytes for 17 classes batched five at a time.
- Existing `per-tree`/`block256` defaults remain. R bypasses root calibration;
  deeper forced shared/global behavior remains, and auto retains global deeper
  histograms. No CPU training stage or NVIDIA algorithm dependency was added.

All original counting kernels, policies, defaults and binaries pass their
retained SHA256 manifest. `production-symbols.json` records the absence of CUB,
Thrust and DeviceHistogram symbols in the production archive. Prior finalized
resident and hybrid builds/evidence remain intact.

## Complete training measurements

Same executable, identical generated train/test inputs per comparison, graph
execution, export batch requested 16, instrumentation off. Two serial sweeps use
opposite variant orders. Every observation, including unfavorable ones, is
retained in `<case>-<mode>-<a|b>/result.json` and its capture/stdout/stderr files.
Times below are whole `train()` wall times, including GPU preparation, setup,
training, completed-tree export and final host model validation. They are not
kernel-only or profiler timings. CPU validation/inference after `train()` is
separately reported by the benchmark and excluded from this table.

| Workload | Existing | S only | R only | R + S |
|---|---:|---:|---:|---:|
| Scalar, 65,536 rows, 32 features, 10 rounds, depth 5, max 64 bins | 37.21–37.23 ms | 38.98–39.27 ms | 39.67–43.67 ms | 41.70–44.23 ms |
| 129 regression outputs, 4,096 rows, 16 features, 3 rounds, depth 2, max 32 bins | 88.76–95.54 ms | 84.73–94.21 ms | 87.31–90.30 ms | 84.44–88.70 ms |
| 1,024 binary outputs, 4,096 rows, 16 features, 2 rounds, depth 2, max 16 bins | 494.69–522.29 ms | 469.40–518.71 ms | 426.28–431.66 ms | 427.14–427.52 ms |
| 4,096 binary outputs, 1,024 rows, 16 features, 1 round, depth 2, max 16 bins | 838.37–845.48 ms | 807.49–836.92 ms | 730.56–833.48 ms | 690.90–700.93 ms |

R+S reduces paired whole-training time by 13.6–18.2% for 1,024 outputs and
17.1–17.6% for 4,096 outputs. The latter measured training-loop times are
765.80–771.25 ms existing versus 620.51–629.97 ms combined.

The scalar control is slower with R. Its 64-bin shape falls back for S, so its
S-only differences are variation between independent runs, not a specialized
kernel improvement. Scalar uses forced shared histograms; R selects global
root atomics, so it is a workload-specific policy change as well as batching.

Fresh seed 20260922603, 129 outputs: existing **94.39 ms**, S **83.68 ms**,
R **78.80 ms**, combined **98.53 ms**. Thus the combined 129-output improvement
did **not** repeat on this confirmation. The 17-class case (tile 5, preserving
full pre-round derivatives) is essentially tied: 25.12 versus 25.06 ms total;
the training loop is 13.50 versus 14.08 ms. These are retained negative/neutral
observations, not discarded outliers.

Clocks are unlocked; captures record before/after telemetry. For example the
fresh combined 129-output run ends at 1,455 MHz after starting at 1,635 MHz.
Host scheduling and WSL add variation. Two full-training observations per main
variant do not establish confidence intervals or a universal dispatch rule.

## Complete split operation

`ghb_split_search_tests --bench` measures candidate **and** winner launches from
resident fixed histograms; allocation, transfers and graph construction are
excluded. Seven retained alternating samples per variant follow two warmups;
each sample contains 128 operations. The graph therefore covers all 256 kernel
nodes, not just one kernel. Both initial and confirmation raw arrays are retained.

Confirmation medians, graph mode, 16 features:

| Bins | Nodes | Existing | Warp | Speedup |
|---|---:|---:|---:|---:|
| 16 | 1 | 19.480 us | 14.000 us | 1.39x |
| 16 | 2 | 18.976 us | 13.552 us | 1.40x |
| 16 | 16 | 49.136 us | 19.328 us | 2.54x |
| 16 | 64 | 175.584 us | 60.968 us | 2.88x |
| 32 | 1 | 17.168 us | 12.184 us | 1.41x |
| 32 | 2 | 17.408 us | 12.535 us | 1.39x |
| 32 | 16 | 49.512 us | 19.552 us | 2.53x |
| 32 | 64 | 177.864 us | 65.368 us | 2.72x |

The 33-bin fallback is 0.992–1.009x in the confirmation. Stream results and all
raw samples are in `split-benchmark-confirm.stdout`. The initial benchmark ran
while the CPU quality audit was active; confirmation ran after it finished, with
no other agent GPU workload. Full-training rankings above were also collected
before the CPU quality audit began.

## Correctness, quality and profiling

- All **10 CTest suites pass**. Split tests pass 863,548 fieldwise checks against
  identical legacy inputs, including signed zeros, cancellation, missing-only
  splits, categorical equality, ties, strict gain boundaries, clipping, inactive
  nodes, graph replay and larger-shape fallback.
- Root tests compare independent CPU statistics and our scalar GPU primitive,
  exact counts, guards, missing/zero-weight rows, skewed bins, partial/wide batches,
  65,536-bin features and graph replay with changing cache selectors.
- Trainer tests cover both execution modes, partial output/export tiles, budget
  boundaries, missing/categorical/weighted data, depth zero and an independent
  multiclass pre-round Newton-leaf reference. CPU/GPU held-out predictions and
  serialization are checked in every benchmark.
- Compute Sanitizer: root and split **memcheck, racecheck and synccheck** pass;
  full trainer memcheck passes. Every raw report/exit status is retained.
- **38/38 models beat their own constant-base models** under all applicable
  held-out metrics. However, **22/28 variant comparisons fail the zero-allowance
  quality gate**. Largest recorded metric deterioration: **3e-16**. Classification
  decisions are unchanged on these held-out rows. See
  [`quality/summary.md`](quality/summary.md); these remain failed gates.
  The independent `quality-metadata.json` verification passes: all 38 named
  configurations, captured flags, serialized model dimensions, source/artifact
  hashes, 28 pair identities and complete zero-allowance metric sets agree.
  Its metadata pass leaves the quality status as **regression**.
- The largest observed prediction difference is 5.552e-16. Exact-real bounds over
  intersections of all encoded leaf regions are <=1.718e-15 in raw margin; these
  exclude runtime arithmetic rounding. Small bounds explain scale, not a quality
  waiver or equivalence guarantee. No NLP corpus or competing learner is measured.
- Both new Nsight Systems captures include graph nodes. The verified trace audit
  matches **1,713 emitted stage/NVTX pairs**, with 387 trees and 27 export waits
  per capture, and **zero waits or device-to-host copies inside tree builds**.
  The extra 27 histogram scopes cover batched roots outside `tree_build`.
  See [`trace-audit-verified.md`](trace-audit-verified.md). The first reused audit's
  Markdown included a historical coverage note; the verified rendering removes
  that inapplicable text and repeats the same underlying checks.
- Nsight Systems observes scalar weighted accumulation calls fall from 774 to
  387, plus 27 batched roots (24 full-width and three single-output tails).
  Candidate and winner counts remain 774 each. Nsight Compute confirms increased
  batched-root parallelism, but substantial remaining memory/atomic latency and
  inefficient global accesses. See [`PROFILING.md`](PROFILING.md). Profiler times
  explain behavior and are excluded from rankings above.

## Retained evidence and next candidate

`final-provenance/manifest.sha256` freezes 56 source/build/test files. Measured
trainer SHA256:
`369ce3e735949e32822ce617c18f22c98fea90703630ec70251d2f476a0d3ae9`.
`artifacts.sha256` covers this evidence directory, excluding itself and Python
bytecode caches. Command captures preserve executable hashes and return codes;
quality exit 1 is the recorded regression result, not an infrastructure failure.
Build captures which changed the executable are expected to report an executable
change; all measured workload captures require it unchanged.

Next, compare root count reuse and a small shared-memory batch against direct
atomics, with complete clear/accumulate/consume costs, scalar controls and strict
quality gates. Counts are identical across outputs/rounds under the present
no-sampling contract, so this may remove work rather than only schedule it better.
Deeper histogram privatization/row partitioning remains independently measurable.
These next candidates are unimplemented and unranked. No new Deep Research report
is required before those local experiments.
