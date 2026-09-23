# Independent audit of the fixed window-rescan campaign

**PASS:** 20 invocations, 40 candidate measurements, 440 raw timing samples, and 1,760 timed histogram operations. All 20 receipts and their 60 artifacts match their recorded hashes. The frozen binary and all 22 source hashes match.

The existing CPU-only audit passed. A separate calculation reconstructed command/configuration identities, checked the exact artifact set, recomputed every CSV summary from raw timings, and checked every ratio against `analysis.json`. No GPU work was performed for this audit.

The table preserves each of the ten cases. Each timing pair is ordered by seed 2026092271, then 2026092272; each seed is a separate process. Times are microseconds per complete histogram. Speedup is baseline time divided by window time, so values below 1 mean the window candidate is slower.

| Case | Input count | Bins | Window bins / passes | Cache / launch | Baseline medians (us) | Window medians (us) | Speedup range |
|---:|---:|---:|---:|---|---|---|---|
| 0 | 16,777,216 | 1,048,576 | 1,048,576 / 1 | warm / graph | 2582.784 / 2580.224 | 2730.752 / 2546.432 | 0.945814–1.013270x |
| 1 | 16,777,216 | 1,048,576 | 524,288 / 2 | warm / graph | 2614.784 / 2588.672 | 1079.552 / 1155.072 | 2.241135–2.422101x |
| 2 | 16,777,216 | 524,287 | 524,288 / 1 | warm / graph | 710.912 / 716.032 | 721.920 / 724.736 | 0.984752–0.987990x |
| 3 | 16,777,216 | 524,288 | 524,288 / 1 | warm / graph | 715.008 / 713.984 | 720.128 / 717.824 | 0.992890–0.994651x |
| 4 | 16,777,216 | 524,289 | 524,288 / 2 | warm / graph | 698.880 / 697.856 | 1057.536 / 1136.128 | 0.614241–0.660857x |
| 5 | 16,777,216 | 786,432 | 524,288 / 2 | warm / graph | 1401.856 / 1393.408 | 984.832 / 983.808 | 1.416341–1.423447x |
| 6 | 16,777,233 | 1,048,576 | 524,288 / 2 | warm / graph | 2654.720 / 2620.416 | 1152.512 / 1080.832 | 2.303421–2.424443x |
| 7 | 268,435,456 | 1,048,576 | 524,288 / 2 | warm / graph | 39962.879 / 40273.407 | 18207.232 / 18264.065 | 2.194890–2.205063x |
| 8 | 16,777,216 | 1,048,576 | 524,288 / 2 | cold / graph | 2591.488 / 2606.080 | 1154.304 / 1254.656 | 2.077127–2.245065x |
| 9 | 16,777,216 | 1,048,576 | 524,288 / 2 | warm / stream | 2622.720 / 2507.520 | 1089.792 / 1092.096 | 2.296062–2.406624x |

## Interpretation by case

- **Case 0:** One-pass 4 MiB window control is mixed: one process is slower and one faster. This establishes neither a speedup nor zero overhead.
- **Case 1:** Two-pass 2 MiB counters improve this million-bin warm-graph workload in both processes. The gain is measured against the existing narrow kernel at the same policy and grid.
- **Case 2:** One-pass window just below the threshold is slightly slower in both processes. The small difference is descriptive, not a statistically established regression.
- **Case 3:** One-pass window at the threshold is slightly slower in both processes. Near parity does not prove zero overhead.
- **Case 4:** One bin above the threshold forces an additional full scan and is substantially slower in both processes. This excludes a universal two-pass default at this boundary.
- **Case 5:** At 786432 bins the two-pass candidate is faster in both processes. This establishes a favorable tested shape, not the exact crossover bin count.
- **Case 6:** The million-bin gain remains with a 17-element input tail in both processes; the focused correctness suite separately tests tails.
- **Case 7:** The gain remains at 268435456 inputs in both processes; the input has grown to 1 GiB and the candidate scans it twice.
- **Case 8:** The gain remains when 64 MiB is read to evict cache before each complete operation. Eviction is outside timing; this does not flush cache between the candidate's two passes.
- **Case 9:** The gain remains with stream launches and runtime clearing in both processes. The timed device operation does not include allocation, preparation, or all host-side application overhead.

The central million-bin result is 2.241–2.422x in warm graph mode at 16,777,216 inputs and 2.195–2.205x at 268,435,456 inputs. Conversely, the two-pass candidate takes 51.3% and 62.8% more time at 524,289 bins. These are separate workload-specific outcomes.

## Limits

- Only the fixed initial 20 invocations in this manifest are included. Subsequent 1 MiB/four-pass experiments and skew/profile results are outside this audit.
- Two seeds each correspond to one fresh process per case. They are descriptive replications; the 11 timing batches within a process are not 11 independent process replications. No confidence interval is claimed and cases are not pooled.
- Each baseline and candidate share one invocation, fixed scalar policy 0 (128 threads, four items), 48 blocks, and u32 scratch/u64 output. Different window sizes are separate cases/processes, each with its own baseline.
- All workloads here are uniform shuffled u32 bin IDs. No conclusion extends to hot, sorted, weighted, multi-feature, or other unmeasured workloads.
- Times include each histogram's clearing, counting scans, and output widening. Allocation, graph construction, warmup, and cache eviction are outside the timed interval. End-to-end host overhead and zero-loss preservation of other library paths are not established.
- Windows driver 597.06, WSL2, one RTX A5000 Laptop GPU (SM86), unlocked clocks, and within/cross-process timing variation limit portability and causal interpretation. Recorded telemetry does not prove the absence of other GPU work.
- The cache-fit explanation is plausible but timing alone does not establish the cause. No profiling data is used by this audit.
- Only the existing owned narrow kernel is the comparison baseline. No NVIDIA histogram ran in this campaign, and no historical NVIDIA result is multiplied into these gains.
- No production-default promotion is recorded. The severe loss at 524289 bins and small/mixed control differences prevent a universal-window or zero-overhead conclusion.

## Evidence

- [Manifest](manifest.json), [original analysis](analysis.json), [environment](environment.json), and [machine-readable independent audit](independent-audit.json).
- Benchmark SHA256: `79bb396fdc50760eed6394c3ca78fe297f79934d9a0df7a45f22b2404d96b374`.
- Per-process medians, ranges, ratios, and CSV/receipt hashes are retained in the independent JSON.
