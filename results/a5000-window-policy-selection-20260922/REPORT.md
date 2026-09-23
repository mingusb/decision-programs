# Existing-kernel launch-policy selection — 22 September 2026

The existing two-pass histogram runs faster on the tested million-bin workloads
with **256 threads, eight items per thread, and 96 blocks** (policy 2). Fresh
confirmation measured **1.807–1.912×** over the earlier window configuration at
16,777,216 inputs and **1.451–1.523×** at 268,435,456 inputs. No implementation
code, production defaults, or executables were changed in this follow-up.

These are dense counting measurements on the RTX A5000 Laptop GPU, SM 86,
Windows driver 597.06 through WSL2. Inputs are uniform shuffled u32 bin IDs;
outputs are exact dense u64 counts. They are not trainer timings or predictive
accuracy results. The existing trainer directory contains two interface headers
and no working trainer implementation.

## Selection and confirmation

The frozen executable is `build/window-experiment/histogram_bench`, SHA256
`79bb396fdc50760eed6394c3ca78fe297f79934d9a0df7a45f22b2404d96b374`.
Window size is 524,288 u32 counters (2 MiB), with kernel-based clearing. Every
uninstrumented run uses warm-cache CUDA graphs and a requested 2,000 ms warmup.
Raw CSVs and corresponding validation logs are retained in this directory.

Selection tested policies 0–4, each with 48, 96, 192, 384, and 768 blocks: 25
window configurations plus the frozen narrow-global control. Policy 5 was
excluded because it duplicates policy 2 for this kernel, which ignores replica
count. Seeds 20260922301 and 20260922302 each used 21 randomized rounds, with
four complete operations per sample. Policy 2 / 96 blocks had the lowest median
in both runs: 667.392 and 673.280 microseconds. Policy 4 / 192 blocks was close
(680.960 and 676.864 microseconds); the screen does not establish a definitive
ordering between those two configurations.

Policy 2 / 96 blocks was fixed before confirmation and was not selected again
from confirmation results. Configurations below use these variant identities:

- Selected window: `global_window:2:96:u32:kernel`.
- Earlier window: `global_window:0:48:u32:kernel`.
- Narrow-global control: `global:0:48:u32:kernel`.
- NVIDIA reference: `cub:0:48:native:kernel`. This reference chooses its own
  launch configuration; the supplied policy and block fields are metadata.

All rows in the next table have 1,048,576 bins. Values are process medians in
milliseconds, not confidence intervals. Ratios were calculated within each
process. The three 16M confirmation runs used 31 rounds and batch eight; the
two 256M transfer runs used 13 rounds and batch two.

| CSV | Inputs | Selected window | Earlier window | Narrow global | NVIDIA reference | Earlier / selected |
|---|---:|---:|---:|---:|---:|---:|
| [confirmation-331](confirmation-331.csv) | 16,777,216 | 0.594176 | 1.135872 | 2.442496 | 17.018751 | 1.912× |
| [confirmation-332](confirmation-332.csv) | 16,777,216 | 0.617472 | 1.116032 | 2.457216 | 17.062016 | 1.807× |
| [confirmation-333](confirmation-333.csv) | 16,777,216 | 0.609408 | 1.123968 | 2.453376 | 17.061249 | 1.844× |
| [large-341](large-341.csv) | 268,435,456 | 11.514880 | 17.542656 | 38.578175 | 171.165695 | 1.523× |
| [large-342](large-342.csv) | 268,435,456 | 12.039168 | 17.463808 | 38.496769 | 169.946106 | 1.451× |

The selected window was 27.632–28.643× faster than the tested NVIDIA reference
at 16M inputs and 14.116–14.865× at 256M. The reference is the bundled CUB
histogram implementation (CSV version 300402), used only for benchmarking.
It reported 1,207,960,063 bytes of scratch in each of these five runs; the
selected window uses 2,097,152 bytes. These comparisons cover this reference
API, hardware, data distribution, and timing contract, not every histogram
implementation or workload.

## Boundary loss retained

Two further seeds (20260922351 and 20260922352) tested 16,777,216 inputs and
524,289 bins, with 31 rounds and batch eight. The additional narrow-global
policy 2 / 96-block comparator was specified before either boundary run. At
this boundary, the second window contains only one bin but still scans all
inputs.

| CSV | Selected window, ms | Narrow global 0:48, ms | Narrow global 2:96, ms | Selected / narrow 2:96 |
|---|---:|---:|---:|---:|
| [boundary-351](boundary-351.csv) | 0.763264 | 0.727680 | 0.586880 | 1.3005× |
| [boundary-352](boundary-352.csv) | 0.801536 | 0.729728 | 0.572928 | 1.3990× |

The selected window takes **30.05–39.90% longer** than that single-scan
comparator. No automatic selection change or universal threshold follows from
the million-bin wins. Concentrated distributions, cold-cache execution, other
devices, and training statistics were not tested in this follow-up.

## Nsight Compute diagnostics

Separate profiles used kernel replay, cache control enabled, clock control
disabled, a requested two-second warmup, 300 skipped matching launches, and two
captured counting launches. Both profiles use seed 20260922361, 16M inputs,
one million bins, and direct-stream execution. Their sections cover throughput,
launch statistics, occupancy, memory, scheduling, and warp stalls. Timing
printed by the instrumented benchmark is excluded from the tables above.

| Diagnostic | Earlier window 0:48 | Selected window 2:96 |
|---|---:|---:|
| Achieved occupancy | 8.30% | 32.72–32.76% |
| DRAM throughput, profiler percentage of peak | 75.91–78.34% | 91.37–92.25% |
| L2 hit rate | 76.69–79.19% | 79.37–79.38% |
| Reported SM clock | 1.63 GHz | 1.45 GHz |
| Reported DRAM clock | 5.99 GHz | 5.99 GHz |

The higher occupancy and memory throughput are consistent with more concurrent
work helping this workload. This is a qualitative inference: the profiles have
different SM clocks and do not isolate a causal effect. No speed ratio is
derived from profiler durations. Full exports: [selected](profile-selected.txt)
and [earlier window](profile-original.txt); corresponding `.ncu-repz` files retain
the native reports.

## Verification and trainer implications

An independent read-only audit checked all nine CSVs: **80 candidate rows,
1,816 raw samples, and 9,536 timed operations**. Raw medians, extrema, p95,
throughput, policy fields, workload identity, scratch/window sizes, and timing
metadata agree with the summaries. Each of these nine invocations exited
successfully and validated its outputs against exact CPU counts twice through
direct execution and twice through graph replay before measurement. The harness
does not check each timed sample's output afterward.

Timing includes scratch clearing, all scans and atomic updates, and dense
output conversion. Allocation, host-to-device upload, graph construction, host
submission, and warmup are excluded. Selection, confirmation, large-input,
boundary, and profiler observations remain separate. Unlocked clocks and the
previously documented WSL timing variability preclude claims of zero overhead
or universal optimality.

[Recorded hashes](hashes.txt) were rechecked after profiling and match. Existing
counting source, production defaults, and the benchmark binary remain intact.
No new sanitizer campaign was run because this follow-up changes no code; each
newly measured configuration passed the existing benchmark's correctness checks.

The next missing trainer comparison is one complete training level's histogram:
16M rows, 64 packed features, 256 bins per feature, and 1/16/64 active nodes,
producing gradient sums, Hessian sums, and exact counts. That entails over a
billion row-feature updates and a proposed 24-byte statistics record per bin.
No existing owned executable implements that contract. The counting results
above therefore remain component evidence, with no claim about training speed
or model accuracy.
