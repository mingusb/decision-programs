# Root batching and small-bin split profiling

These are diagnostic observations on the RTX A5000 Laptop, SM86, not timings
used to rank the uninstrumented variants. Nsight Compute captured one invocation
per report with clock control disabled; reported SM frequency was 1.45 GHz and
DRAM frequency 5.47–5.50 GHz. Its cases use 4096 rows, 16 features, 16 outputs,
32 maximum bins and depth 2. The scalar root capture handles **one output**;
the batched root capture handles **16 outputs**. Neither root duration includes
the complete clear/accumulate/cache-consumption boundary.

All four native reports were imported without launching GPU work. Raw CSV metric
exports are `ncu-import-raw-*.stdout`; readable metric/rule CSV exports are
`ncu-import-*.stdout`. Their adjacent command records retain tool/executable hashes
and successful return codes. Sources: [scalar root](ncu-import-global-root.stdout),
[batched root](ncu-import-batched-root.stdout),
[block split](ncu-import-block-split.stdout),
[warp split](ncu-import-warp-split.stdout).

## Individual kernel evidence

| Metric | Scalar root | Batched root, 16 outputs | Block split candidate | Warp split candidate |
|---|---:|---:|---:|---:|
| Grid blocks ×threads/block | 16×256 | 256×256 | 16×256 | 16×32 |
| Registers/thread | 36 | 38 | 68 | 80 |
| Static shared bytes/block | 0 | 0 | 12,480 | 0 |
| Theoretical occupancy | 100% | 100% | 50% | 33.33% |
| Achieved occupancy | 16.45% | 72.93% | 16.70% | 2.12% |
| Reported waves/SM | 0.06 | 0.89 | 0.11 | 0.02 |
| Scheduler cycles with no eligible warp | 96.60% | 95.69% | 94.00% | 92.01% |
| DRAM throughput, fraction of peak | 6.84% | 3.35% | 1.60% | 1.91% |
| L2 throughput, fraction of peak | 17.65% | 43.05% | 1.49% | 1.66% |
| Local/shared spill requests | 0/0 | 0/0 | 0/0 | 0/0 |
| Profiled duration, µs | 32.22 | 185.34 | 17.79 | 12.99 |

Batched-root duration divided by its 16 outputs is 11.58 µs/output. This is only
work normalization, not a measured complete-operation speedup against running
all 16 scalar outputs. The captured scalar output need not have exactly the same
nonzero derivative pattern as every output in the batch.

Batching supplies enough blocks to occupy substantially more of the device, but
does not solve readiness stalls. Scalar-root long-scoreboard stalls account for
55.14 of 58.92 average warp cycles per issued instruction. For the batched root,
the corresponding values are 88.23 of 206.30; its raw metrics additionally report
49.91 cycles of local/global instruction-queue throttling and 55.60 cycles of
MIO throttling per issued instruction. These are per-warp diagnostic ratios,
not additive wall times. High occupancy alone therefore does not establish
efficient execution. The low DRAM percentage and high L2 hit rate (98.26% in
the batch) do not support calling this DRAM-bandwidth saturation.

The raw L1TEX global-load sector count is 15,232 for the scalar root and 98,304
for the entire batch, or 6,144 per output. Global reduction sectors, which include
the compiled atomic updates, are 172,871 versus 3,096,128, or 193,508 per batched
output. These are 32-byte cache-sector requests, not bytes read from DRAM.
The normalized load decrease is consistent with sharing bins and loading adjacent
derivative outputs. It does **not** establish better coalescing for every access:
Nsight's source-level global-access rule reports 51% excessive sectors for the
scalar capture and 72% for the batch. Output-major histogram destinations still
scatter the atomic traffic. Exact source locations and alternative atomic
layouts require a separate experiment; the profiler's estimated speedups are
not measured promises.

The warp split removes block-barrier stalls (12.01 to zero average warp cycles
per issued instruction) and all recorded shared-memory traffic. Short-scoreboard
stalls fall from 11.77 to 5.60 cycles. The latter must not be labeled shared-memory
stalls in the warp kernel: it uses no shared memory, and this category also covers
other MIO dependencies. Registers increase without spilling. Its very low
occupancy reflects launching only 16 one-warp blocks, so fewer idle warps and
barriers help this invocation despite a lower occupancy percentage. Both kernels
remain too small to fill this GPU. The split capture covers candidate evaluation,
not its separate winner kernel.

## Complete trace composition

The Nsight Systems baseline and combined captures use matching 129-output,
4096-row, 16-feature, 3-round, depth-2 cases. They include preparation, training
and held-out GPU prediction. Their summed kernel durations exclude memory-copy
operations and host/API gaps; they are not end-to-end training time. Sources:
[baseline kernel CSV](nsys-base-stats_cuda_gpu_kern_sum.csv),
[combined kernel CSV](nsys-both-stats_cuda_gpu_kern_sum.csv).

| Kernel group | Baseline diagnostic sum | Combined diagnostic sum |
|---|---:|---:|
| All reported kernels | 49.294 ms, 8194 launches | 36.170 ms, 7447 launches |
| Weighted histograms | 22.288 ms, 45.2% | 14.044 ms, 38.8% |
| Split candidates +winners | 14.688 ms, 29.8% | 10.259 ms, 28.4% |
| Frontier scan/prefix/materialize/advance | 5.640 ms, 11.4% | 5.677 ms, 15.7% |

The combined trace replaces 387 root scalar histogram invocations with 27 batched
invocations: 24 full batches of 16 outputs and three one-output tails. The 387
remaining scalar histogram calls are the deeper level. Active-histogram clears
fall from 774 to 387; root-cache clears are separate CUDA memory operations and
must not be omitted from complete-operation measurements. The cached-root copy
adds work to tree initialization: its total rises from 0.498 to 0.697 ms across
387 launches. These observations match the intended integration boundaries.

## Remaining bottlenecks

The largest combined-trace component is the deeper scalar histogram, 9.977 ms
(27.6% of all recorded kernel time). Split candidates follow at 8.100 ms (22.4%);
batched roots consume 4.067 ms (11.2%, including one-output tails). Tiny per-tree
split launches and repeated frontier-management kernels remain material after
the first two changes.

The next investigation should distinguish deeper-level histogram work from root
atomic-queue pressure. The current evidence supports examining output/node task
batching and avoiding repeated statistics, while preserving the independent-tree
contract. The warp split's underfilled grid also makes batching independent split
tasks worth assessing before more instruction-level tuning. Compact row lists
need a measured inactive-row benefit, and parent subtraction needs its separate
floating-point contract. None of these unimplemented candidates is established
as faster by these captures.
