We compare against NVIDIA’s CUB library of optimized GPU routines; its histogram is our reference implementation.

The preserved benchmark shows substantial, algorithm-dependent sensitivity to graph batch size. These results justify fixing the timing protocol before selecting kernels; they do not establish a clock, power, or cache cause.

Evidence is from `timing-before/*.csv` and `before-*.details.txt`, with exact invocations in the adjacent `.command.json` files. `before.json` identifies `build/expanded/histogram_bench` by SHA256 `37f184896e2cb15b25b64e563e34bc77fd2daa722323c15a6cdb307ebe65ed68`. This audit used CPU file analysis only. The new protocol3 implementation was reviewed as source, without compiling or executing it.

**Unprofiled timing ladder.** There are 30 invocations, 120 candidate rows, and 1,800 raw samples: three workloads, five batch sizes, two seeds, four candidates, and 15 randomized rounds per invocation. All use warm cache mode, graph launch mode, uniform shuffled input, and u32 output. The following values are per-operation medians in microseconds; each cell is `seed24680 / seed67890`, not a confidence interval.

| Workload and candidate | Batch1 | Batch2 | Batch5 | Batch20 | Batch100 |
|---|---:|---:|---:|---:|---:|
| N1048576, u32, B8; CUB | 18.432 / 18.432 | 16.896 / 16.896 | 15.360 / 15.360 | 15.002 / 14.950 | 13.343 / 14.838 |
| Same; shared:4:192 | 10.240 / 11.264 | 10.752 / 9.728 | 14.765 / 15.155 | 13.414 / 12.698 | 12.861 / 12.431 |
| Same; shared_partial:2:96 | 11.264 / 11.264 | 9.728 / 9.728 | 8.397 / 8.397 | 8.192 / 8.192 | 7.240 / 8.079 |
| Same; bitplane:1:192 | 15.360 / 15.360 | 13.824 / 13.824 | 20.275 / 20.070 | 20.787 / 19.046 | 19.725 / 19.405 |
| N4096, u8, B256; CUB | 7.168 / 7.168 | 5.120 / 5.120 | 4.301 / 6.758 | 3.840 / 3.840 | 4.239 / 4.209 |
| Same; shared:2:192 | 7.168 / 7.168 | 8.688 / 11.776 | 14.746 / 36.250 | 18.944 / 12.237 | 12.196 / 13.076 |
| Same; shared_partial:3:96 | 8.192 / 8.192 | 6.656 / 6.656 | 5.734 / 6.758 | 5.478 / 5.478 | 6.052 / 6.052 |
| Same; nvidia_sample256 | 8.192 / 8.192 | 6.656 / 6.656 | 5.939 / 6.554 | 5.530 / 5.530 | 6.164 / 6.164 |
| N1048576, u8, B256; CUB | 9.216 / 9.216 | 7.168 / 7.680 | 6.144 / 6.349 | 5.888 / 5.990 | 5.878 / 6.666 |
| Same; shared:2:192 | 10.240 / 11.264 | 9.216 / 9.728 | 13.107 / 14.746 | 12.390 / 20.224 | 13.752 / 15.524 |
| Same; shared_partial:3:96 | 11.264 / 11.264 | 9.216 / 9.216 | 8.192 / 8.397 | 7.936 / 7.936 | 7.875 / 8.858 |
| Same; nvidia_sample256 | 10.240 / 11.264 | 8.704 / 8.704 | 7.578 / 7.578 | 7.270 / 7.270 | 7.219 / 8.120 |

Measured implications:

- B8 `shared_partial:2:96` is repeatable at 8.3968us for batch5 and 8.192us for batch20 on both seeds. Its ratio against the same-invocation CUB median is approximately 1.83x in those four comparisons. It belongs in the next focused comparison even though the earlier search shortlist omitted it.
- B8 shared atomic rises from 9.728–10.752us at batch2 to 14.765–15.155us at batch5, while bitplane rises from 13.824us to 20.070–20.275us. In contrast, CUB and shared_partial improve. This is inconsistent with assuming a single multiplicative slowdown of every candidate.
- For N4096/u8/B256, shared atomic reaches 36.249599us at batch5/seed67890, versus 7.168us at batch1 for the same seed. Its batch5 median also differs by 2.46x between the two invocations/seeds. CUB changes from 7.168us at batch1 to 3.840us at batch20 on both seeds.
- There are 74 raw samples above twice their own candidate-row median (4.11% of 1,800). This is a descriptive threshold, not an outlier rejection rule. The largest excursion is B8/shared_partial/batch1/seed24680: 4,696.063995us versus an 11.264us median, or 416.91x. B8/bitplane/batch20/seed67890 has a 361.318398us maximum versus a 19.0464us median. Even batch100 retains excursions, including shared_partial at 53.56544us versus a 7.23968us median.
- With this nearest-rank calculation and only 15 samples, the reported p95 is the maximum. It does not establish a precise tail-latency distribution. Many short measurements also lie on visibly coarse timestamp increments, making batch1 particularly weak evidence for small percentage wins.

The preserved graph protocol places timing events outside `cudaGraphLaunch`. Its intervals can include submission gaps if the GPU reaches an event before the host has queued subsequent work. Batch size also changes graph topology and amortization. These are concrete measurement sensitivities; the ladder alone cannot apportion the observed variation among them, scheduling, cache state, or frequency changes. Seeds also change candidate shuffle order, so the two-seed comparisons are not controlled experiments isolating input data alone.

**Nsight Compute kernel evidence.** All eight commands use kernel replay, `--cache-control all`, `--clock-control none`, and one selected kernel. Despite the benchmark argument `--cache warm`, these captures deliberately flush caches before profiling passes. `--launch-skip 2` skips the two standalone correctness launches and selects the first round's untimed warmup kernel. The reported durations exclude the complete operation's other kernels and output initialization. They must not be compared directly with the warm-graph operation medians above.

`Long-scoreboard share` below is NCU's reported fraction of average warp cycles between issued instructions spent waiting on L1TEX dependencies; it is not a fraction of total wall time. `No eligible` is the scheduler percentage with no eligible warp. All output counters are u32.

| Capture / candidate | Workload | Kernel us | DRAM throughput % | No eligible % | Long-scoreboard share % | Registers/thread | Achieved occupancy % |
|---|---|---:|---:|---:|---:|---:|---:|
| small8-shared; shared:4:192 | N1048576/u32/B8/uniform | 16.70 | 79.54 | 87.66 | 80.5 | 37 | 37.82 |
| small8-bitplane; bitplane:3:192 | N1048576/u32/B8/uniform | 21.38 | 60.36 | 68.24 | 65.3 | 23 | 55.30 |
| byte-shared; shared:3:96 | N16777216/u8/B256/uniform | 61.22 | 78.92 | 53.49 | 42.9 | 52 | 35.10 |
| byte-cub; CUB sweep kernel | N16777216/u8/B256/uniform | 55.39 | 87.55 | 56.18 | 68.8 | 56 | 82.30 |
| hot-shared; shared:2:384 | N1048576/u32/B256/hot99 | 16.80 | 78.14 | 84.25 | 93.5 | 32 | 97.33 |
| hot-rle; shared_rle:3:96 | N1048576/u32/B256/hot99 | 16.61 | 73.65 | 78.76 | 58.8 | 59 | 33.53 |
| hot-warp; shared_warp:2:192 | N1048576/u32/B256/hot99 | 17.70 | 70.73 | 74.47 | 73.8 | 37 | 65.60 |
| large-control; shared:3:96 | N16777216/u32/B4096/uniform | 190.85 | 94.25 | 86.56 | 78.3 | 52 | 34.26 |

The bitplane NCU capture uses tuning3, whereas the timing ladder uses tuning1. Likewise, the byte NCU captures use 16MiB input rather than either byte ladder size. These reports guide bottleneck hypotheses, not direct attribution of ladder timings.

Shared-access serialization indicators from the NCU Source Counters recommendations:

| Capture | Excessive shared wavefronts | Total shared wavefronts | NCU rounded excessive share |
|---|---:|---:|---:|
| byte-shared | 1,128,607 | 1,654,431 | 68% |
| byte-cub | 1,311,874 | 1,838,466 | 71% |
| hot-shared | 336 | 39,248 | 1% |
| hot-rle | 25,157 | 42,618 | 59% |
| hot-warp | 336 | 36,176 | 1% |
| large-control | 1,314,943 | 1,863,807 | 71% |

No excessive-shared-wavefront recommendation appears in the two small8 text reports; absence of a recommendation is not a measured zero. These counts are indicators of shared-access serialization, not proof that bank conflicts dominate runtime. For example, the large control simultaneously reaches 94.25% DRAM throughput, and the three hot99 kernels finish within 1.09us despite very different shared-wavefront and occupancy metrics. The existing captures support examining load scheduling and access structure while preserving the large bandwidth-oriented control. They do not justify selecting a kernel from a single stall percentage.

Every report warns that GPU frequencies were not fixed. Reported average SM frequencies are 1.43–1.45GHz, but those profiled averages are not a time-aligned clock/power trace of the timing ladder. The CUB report also lists 82.30% achieved occupancy against 75% theoretical occupancy; treat these report-level estimates cautiously instead of using them as a precise residency proof. There is no recorded evidence here establishing thermal or power throttling as the cause of the timing excursions.

**Protocol3 source review.** The current graph event ownership and elapsed-time logic has no identified lifetime or timestamp-selection defect:

- Each candidate owns its graph through `unique_ptr`; each graph owns its timing events in a deque allocated before capture. Graph execution and graph handles are destroyed before the event members.
- `cudaEventRecordWithFlags(..., cudaEventRecordExternal)` forces timing records to exist as graph nodes. Warm mode brackets the entire batch once. Cold mode records a distinct start/end pair around each histogram after that operation's eviction; only those histogram intervals are summed and divided by batch size.
- Warmup and measurement launches use the same stream and graph. The later measurement overwrites the earlier warmup timestamps in stream order. `cudaStreamSynchronize` completes both launches before elapsed times are read, so the reads observe the latest measured records. Separate cold pairs prevent one repetition from overwriting another repetition's measurement.
- Capture alone is not treated as execution. Every graph is launched and CPU-checked twice before timing, including repeated output overwrite. Capture failure ends the capture and destroys any returned discarded graph. Optional process warmup precedes sampling and defaults to zero.
- Stream timing remains unchanged for comparison. Embedded graph events remove host submission from the measured graph intervals; they do not exclude device scheduling interference or guarantee a frequency/cache state. Runtime correctness and sanitizer checks remain necessary and are being run separately by the parent task.

For the next measurements, keep the candidate set and batch size identical across selection and confirmation, retain CUB in each invocation, and repeat a fixed data seed before changing seeds. Use the protocol3 batch ladder to check whether the candidate-specific reversal persists. Record optional warmup as an explicit ablation and retain raw samples. The separate EHH-versus-HEH whole-graph probe tests eviction by matching total kernel traffic while changing its order; its results must remain separate from these intentionally cache-flushed individual-kernel captures.
