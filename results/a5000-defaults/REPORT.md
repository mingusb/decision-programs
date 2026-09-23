# Automatic histogram defaults

The implementation now selects measured configurations automatically for the workload shapes listed below. A default `gh::Config` and a benchmark invocation without `--algorithm` both request automatic selection. This connects the earlier tuning results to ordinary use instead of requiring callers to copy a winning configuration manually.

We compare against NVIDIA’s CUB library of optimized GPU routines; its histogram is our reference implementation. Automatic selection uses that NVIDIA histogram reference for unmeasured shapes or devices, including the measured 16M-byte case where the reference was retained.

## How selection works

The table in [src/defaults.cpp](../../src/defaults.cpp) uses only the device identity, sample count, number of bins, input/output types, and declared launch/cache modes. It does not inspect input values or ordering, classify a distribution, or detect cache residency. Every distribution with the same key receives the same configuration.

Custom entries require the exact NVIDIA RTX A5000 Laptop GPU name, compute capability 8.6, and 48 streaming multiprocessors. Policies requesting 96 KiB shared-memory capacity also check that the device supports it. Other devices and unmatched keys use the NVIDIA histogram reference. The measurements used driver 597.06 under WSL2; selection checks hardware properties, not the driver version.

Call mutable `gh::prepare(config)` before querying workspace or capturing a graph. It resolves the automatic configuration once, performs any required resource setup, and leaves an explicit configuration for execution. Preparation happens outside timing and capture. Changing the workload afterward requires resetting `algorithm` to `automatic` and preparing again. Setting an explicit algorithm preserves the caller's manual parameters.

Automatic stream configurations use runtime output clearing; automatic graph configurations use an explicit clearing kernel. This distinction follows the earlier [stream ablation](../a5000-profiled/resumed-2026-09-21/stream-ablation-summary.md), which showed that gains from kernel clearing in repeated graphs did not transfer uniformly to direct stream launches. The benchmark also retains explicit clearing and manual-variant controls.

## Installed custom choices

All entries use the shared-atomic histogram family. A policy number indexes the compile-time catalog in [histogram.hpp](../../include/gh/histogram.hpp). Here 4K=4096, 1M=1,048,576, and 16M=16,777,216 samples; arrows identify input and output types. Warm/cold describes the declared benchmark regime, not a runtime cache-state test.

| Samples and types | Bins | Launch / cache | Policy | Blocks | Local counters / clearing |
|---|---:|---|---:|---:|---|
| 1M u32→u32 | 4096 | stream / warm | 10 | 48 | native / runtime |
| 4K u8→u32 | 256 | graph / warm | 1 | 96 | native / kernel |
| 1M u8→u32 | 256 | graph / warm | 11 | 48 | native / kernel |
| 1M u32→u32 | 8 | graph / warm | 6 | 192 | native / kernel |
| 1M u32→u32 | 256 | graph / warm | 11 | 48 | native / kernel |
| 16M u32→u32 | 4096 | graph / warm | 11 | 48 | native / kernel |
| 16M u32→u64 | 4096 | graph / warm | 10 | 48 | u32 / kernel |
| 16M u32→u64 | 8192 | graph / warm | 15 | 48 | u32 / kernel |
| 16M u32→u64 | 16384 | graph / warm | 15 | 48 | u32 / kernel |
| 1M u32→u64 | 4096 | graph / cold | 10 | 48 | u32 / kernel |

Policy 1 uses 256 threads and four scalar samples per thread. Policy 6 uses 256 threads and eight samples with a full-tile fast path. Policies 10 and 11 use packed loads with eight samples per thread and 512 or 1024 threads respectively. Policy 15 uses 512 threads, packed loads, and permits up to 96 KiB of shared memory. For u64 output with u32 local counters, the existing bounds check proves that a block's assigned samples fit the local counter before execution.

The no-argument benchmark workload is 1M u32 samples, 4096 bins, u32 output, warm stream execution. It now resolves to policy 10, 48 blocks, and runtime clearing. For 1M u32 samples with 256 bins in warm graphs, policy 11 serves every distribution. Earlier per-distribution choices such as policy 6 for sorted hot99 or single-bin data are not used as an input-distribution oracle.

## Evidence behind these defaults

The preceding explicit-clearing build completed [twelve graph reconfirmation cases](../a5000-profiled/clear-policy-2026-09-21/final-comparison.md), with three invocations per case. The audit checks all 36 invocations, their frozen configurations, workload metadata, binary hashes, and raw-sample summaries. Eleven custom selections beat the strongest applicable reference in each run; the 16M-byte case retains the NVIDIA histogram reference. Those results establish the tested configurations' performance, not a universal dispatch rule.

The exact no-argument stream workload received a separate [search and validation plan](stream-4096.json): 233 search configurations, seed 12345, five samples, and batch 32; then two validation seeds, 67890 and 24680, with fifteen samples each. Its selected configuration is `shared:10:48:native:runtime`, with a median validation ratio of **3.417×** versus the reference. Validation participated in choosing that configuration and is not an independent final test.

Additional checks compare that stream choice and the common graph-256 choice against the NVIDIA histogram reference on uniform, hot99, single-bin, and two-bin shuffled inputs, using seeds 424242 and 987654. Each invocation has 21 samples and batch 32. The following ranges span the two process medians; ratios divide reference time by custom time, so values above 1 favor the custom implementation.

| Input distribution | Graph 256-bin median, µs | Graph reference/custom | Stream 4096-bin median, µs | Stream reference/custom |
|---|---:|---:|---:|---:|
| uniform | 6.464–8.480 | 1.694–2.223× | 20.288–76.864 | 1.045–3.601× |
| hot99 | 6.112 | 2.335× | 17.888–19.328 | 1.901–2.043× |
| single bin | 6.080 | 2.263–2.268× | 16.832–17.888 | 1.805–1.922× |
| two bins | 6.112 | 2.471× | 17.888–19.456 | 2.309–2.510× |

Every recorded comparison here favors the selected custom configuration over the reference, but the uniform stream result is noisy: **1.045× on one seed versus 3.601× on the other**. The weaker result is below a 1.05× margin and is slightly slower than the original scalar control in that invocation, 76.864 versus 76.288 µs. These results do not justify promising a stable 3.417× improvement or the fastest configuration for every invocation.

The [selection-validation directory](selection-validation/) preserves all sixteen commands and raw CSVs. CPU audit verified 40 rows and 840 samples, including reported medians, percentiles, extrema, executable hashes, and GPU/driver identities; 33 samples exceed twice their own row median, and none are discarded. GPU clocks were unlocked and stream timings include host submission effects. The dedicated common-policy checks cover the four shuffled distributions above; they do not establish optimality for all orderings or other data distributions.

These measurements use the preserved `build/profiled-clear-policy/histogram_bench` binary, SHA256 `5c50596f6c820152e8b3df6c0d9529e11461746b6d4000f89ad1e27639bc74c8`. See [recorded session identity](selection-validation/environment.json). They test the explicit configurations that automatic selection is intended to choose; they are not measurements of the new automatic-dispatch build.

## Final verification

The automatic-selection build completed successfully. Its benchmark SHA256 is `a91544fe771a19175b05611c51b1b08549314dcee418d624eeba58ab5444ce6d`; the matching executables and tuner are preserved under `build/defaults`.

- [CTest](ctest.log): all three suites passed, including 18,634 histogram executions, 64 CPU-only selector checks, and 16 autotuner integration tests. GPU correctness checks include preparation of automatic requests, rejection of unresolved execution/workspace queries, fallback, empty input, overwrite behavior, and existing kernel/counter boundaries.
- [Compute Sanitizer memcheck](memcheck.log): 3,110 histogram executions, zero errors. This revision changes host selection and setup, not counting kernels; racecheck and synccheck evidence for those kernels remains in the preceding clearing-policy report.
- [CLI integration](integration/): all ten custom table entries and two reference fallbacks emitted the expected concrete settings and passed the benchmark's independent CPU-count validation. Three further invocations verified manual policy, clearing, and variant overrides. Graph cases also passed the benchmark's captured-execution validation.
- [Tuner smoke check](tuner-smoke/plan.json): a real 334-configuration search with the new executable, fresh-seed validation, and [schema-4 replay](tuner-smoke/replay.log) passed. This short run checks tool compatibility; it does not replace the stored defaults with its differently batched timing choices.

The integration runs use five samples and batch four (three samples and batch two for overrides). They verify the actual default path and selected configurations; they do not establish a new performance ranking. The longer performance comparisons above remain explicitly attributed to the preserved pre-selector binary.

The defaults provide measured choices for exact supported keys and a reference fallback elsewhere. They do not establish that all inefficiencies are removed or that any one configuration is universally fastest.
