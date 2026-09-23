# Nsight Compute audit

All three captures passed the CPU-only identity and launch audit. Each profiles
exactly the first matching launch from the same unchanged `ghb_bench` binary,
SHA256 `282474586a6ff6a7519dbbd995b5b929eaf6afd2e9c0e7b4007e62b907d4fc16`.
The command receipts, report-import paths, saved/emitted workload JSON, raw/detail
kernel identities, launch dimensions, and same-binary SM86 disassembly agree.
Full metrics, profiler rules, source/report hashes and static instruction counts
are in [compute-audit.json](compute-audit.json). The audit itself launches no GPU
work. `compute-sass.txt` was produced by CPU-only `cuobjdump --dump-sass`.

The workload is regression, 4,096 rows × 16 features × 16 outputs, one round,
depth 3, at most 32 bins, output tile 16, output-batch stream execution. The
effective deeper histogram is global. Each report selects one kernel; it does
not characterize every depth, distribution or tile width. Clock control is
explicitly `none`; Nsight warns that clocks are unmodified. Multipass profiler
durations below are diagnostic observations, **not timing rankings**.

| Selected kernel | Grid × threads/block | Registers/thread | Static shared bytes/block | Achieved / theoretical occupancy | Waves/SM | No eligible warp | Profiled duration |
|---|---:|---:|---:|---:|---:|---:|---:|
| `global_accumulate<16,true>` | 256 × 256 | 38 | 0 | 72.99% / 100% | 0.89 | 94.64% | 155.33 µs |
| `materialize_small` | 16 × 4 | 48 | 4,096 | 2.15% / 33.33% | 0.02 | 93.67% | 5.82 µs |
| `warp_candidates<true>` | 1,024 × 32 | 82 | 0 | 18.15% / 33.33% | 1.33 | 87.77% | 30.72 µs |

All three report zero local/shared spilling requests and zero measured shared
bank conflicts. A 1,024-byte driver shared-memory allocation per block is separate
from the explicit static shared-memory column. The dimensions follow the current
launch contracts: histogram row/output work is 65,536 threads; split capacity is
16 outputs × 4 frontier slots × 16 features; small materialization uses one
four-thread block per output. Inactive frontier slots return without evaluating a
split. Profiling the first matching launch does not mean every capacity slot has
useful work; actual arguments/active-node counts are not present in the CSV export.

## Deeper histogram

The histogram has substantial resident warp occupancy but almost no ready warps:
8.60 active warps and only 0.10 eligible warps per scheduler. Long-scoreboard
dependencies account for 80.11 of 160.40 average warp cycles per issued
instruction, about 49.9%. L1/TEX throughput is 63.53% of peak, L2 throughput
49.91%, external DRAM throughput only 5.11%, and L2 hit rate 97.88%. This capture
supports a cache/latency and scattered-access concern, not a conclusion that
external DRAM bandwidth is saturated.

Source counters estimate 3,210,859 global sectors against 897,792 ideal sectors:
2,313,067 excessive sectors, about 72.04%. These are theoretical source-counter
sectors, not measured DRAM transfer bytes. The static SASS includes native global
`RED.E.ADD.F64.RN` and 64-bit integer reductions, plus bin loads and shuffle
broadcasts; no explicit shared-memory accumulation or shared bank conflicts are
involved in this selected global kernel.

Candidates for a subsequent fair experiment are a layout that coalesces assignment
loads across the output tile, feature-axis subdivision, and reduced histogram
traffic/atomic work. Each must preserve row/node/output ownership and exact counts,
account for layout conversion/routing costs, and measure complete training. The
current 38-register count already permits theoretical full occupancy; merely
lowering registers does not address the dominant observation.

## Batched split candidates

The one-warp-per-block launch is limited to 16 resident blocks/warps per SM, below
the hardware's 48-warps/SM ceiling. Its register limit is 20 blocks, so the 82
registers alone are not the limiting occupancy resource for this launch shape.
The grid comprises 1.33 nominal waves; the profiler's generic tail estimate assumes
uniform block work, whereas inactive frontier slots make that assumption weak.

Short-scoreboard stalls account for 17.86 of 27.88 average warp cycles per issued
instruction, about 64.1%. The selected kernel reports **zero shared loads, stores,
and bank conflicts**. Therefore the profiler's generic suggestion to investigate
shared-memory conflicts is not evidence of that cause here. Static SASS contains
warp shuffles and `MUFU.RCP64H`/FP64 arithmetic; shuffle/math dependency chains are
plausible candidates for instruction-level investigation, not a proven causal
attribution from this report alone. FP64 pipe activity is 44.14% of sustained peak
over elapsed cycles and 66.45% over active cycles. DRAM throughput is only 2.64%.

Global sector estimates are 39,552 total versus 16,784 ideal, with 22,768 excessive
(57.57%). A measured candidate is packing several independent feature warps into
one block while retaining each warp's arithmetic order and reduction semantics.
Removing redundant candidate work or fusing winner selection may matter more than
coalescing alone. The profiler's FP32/FMA suggestions change precision or rounding
semantics and are not authorized shortcuts to exact preservation.

## Frontier materialization

Only 64 threads are launched across 16 blocks on a 48-SM GPU, with four threads
per block and 2.09 average active threads per warp. The capture has 2.15% achieved
occupancy, 0.02 waves/SM, 0.80% compute throughput, and 2.53% DRAM throughput.
This is a small control/update operation whose limited work cannot fill the GPU.
The shared scan has no measured bank conflicts; reducing its registers is not
supported as the primary remedy.

Potential followups are removing another launch through compatible fusion or
packing independent output scans while preserving synchronization and node-index
contracts. Artificially expanding work to raise occupancy is not a performance
goal. Full-operation measurements must determine whether any rewrite beats the
current fused small-frontier path.

## Systems cross-check

[systems-audit.md](systems-audit.md) independently passed 1,442 exact emitted
sample/NVTX matches across three traces. For the same 387 output trees, the
comparison and output-batch graph paths record 9,817 versus 655 boosting kernels
including immutable-count setup, and 387 versus 27 graph enqueues. Export scopes
and export stream waits remain 27. Every tree scope has zero explicit runtime
host synchronization, correlated device-to-host copy, or device allocation/free.
These observed work/launch reductions explain architectural progress; they do not
turn profiler timings into an uninstrumented speedup estimate.
