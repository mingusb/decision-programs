# Nsight Compute evidence for count reuse, split batching, and deeper histograms

These eight single-invocation captures explain work and resource behavior on the 48-SM RTX A5000 Laptop (SM86). They are not used to rank uninstrumented implementations. All capture and CSV-import commands exited zero. Native reports, command records, readable details, and raw counters remain alongside this file. Every capture used the full section set and `--clock-control none`; reported SM frequencies were 1.44–1.45 GHz and DRAM frequencies 5.48–5.49 GHz.

The deeper-histogram captures explicitly used `--cache-control none` after validation and two warmup samples. Root/split captures used the profiler default cache control (`all`) and selected their first matching invocation. Do not compare absolute cache counters or durations across these two protocols as though they were one controlled experiment. Multi-pass replay, unlocked clocks and short kernels also limit interpretation of derived ratios.

## Kernel boundaries

- Both root accumulation captures process the complete batch of 16 outputs: 4,096 rows, 16 features, at most 32 bins/feature, 485 total feature bins. “Per-output counts” means integer count atomics inside this batched kernel; it does not mean a scalar-output launch. The cached version compiles those count updates out. Neither capture includes one-time count construction, histogram zeroing/count seeding, cache consumption, or tree initialization.
- The split control evaluates one root across 16 features; the batched capture evaluates 16 roots across those features. Both use the same warp-candidate kernel. Neither includes the winner kernel, cached-winner copy, or tree materialization. The batched trainer also uses cached root counts, so histogram floating-point ordering need not match the control.
- D2 is synthetic case 2: 4,096 rows, 16 features, 16 outputs, two nodes/output, 32 bins/feature. D6 is case 6: 1,024 rows with 16 bins/feature; other dimensions are the same. Every D capture covers all 16 outputs with different output-specific assignments. The batched histogram clear is outside the captured accumulation kernel. These are primitive fixtures, not retained-tree-state training.

## Launch and occupancy

Occupancy columns show theoretical / achieved percentage. Shared bytes are the algorithm’s dynamic allocation per block; every capture has zero static algorithm shared memory. Nsight additionally reports driver-reserved shared memory, which is not included in that column. All eight captures report zero local/shared spilling requests.

| Capture | Grid blocks × threads/block | Registers/thread | Dynamic shared bytes/block | Occupancy % | Waves/SM | No eligible scheduler % |
|---|---:|---:|---:|---:|---:|---:|
| [Root: per-output counts](ncu-per-output-root-details.stdout) | 256 × 256 | 38 | 0 | 100 / 73.55 | 0.89 | 95.78 |
| [Root: cached counts](ncu-cached-root-details.stdout) | 256 × 256 | 34 | 0 | 100 / 73.53 | 0.89 | 94.15 |
| [Split: one root](ncu-per-tree-split-details.stdout) | 16 × 32 | 80 | 0 | 33.33 / 2.08 | 0.02 | 91.98 |
| [Split: 16 roots](ncu-batched-split-details.stdout) | 256 × 32 | 80 | 0 | 33.33 / 10.51 | 0.33 | 92.55 |
| [D2: global16](ncu-deeper-2-global16-details.stdout) | 256 × 256 | 36 | 0 | 100 / 79.04 | 0.89 | 93.92 |
| [D2: shared1 / 4 chunks](ncu-deeper-2-shared1-c4-details.stdout) | 1024 × 256 | 34 | 1,536 | 100 / 95.67 | 3.56 | 82.94 |
| [D6: global16](ncu-deeper-6-global16-details.stdout) | 64 × 256 | 36 | 0 | 100 / 25.65 | 0.22 | 94.23 |
| [D6: shared4 / 4 chunks](ncu-deeper-6-shared4-c4-details.stdout) | 256 × 256 | 38 | 3,072 | 100 / 79.44 | 0.89 | 81.12 |

Removing root count updates reduces register allocation from 38 to 34 while theoretical occupancy stays 100% and achieved occupancy stays near 73.5%. Its effect therefore does not require an occupancy explanation. Split batching grows the grid from 16 to 256 one-warp blocks without changing the 80-register allocation. This supplies more independent work: achieved occupancy rises from 2.08% to 10.51%, while compute throughput rises from 5.00% to 52.16% of peak. The one-warp block’s theoretical occupancy remains 33.33%; raw occupancy limits are 16 blocks/SM, 24 from registers, and 48 from warps. Generic rule text mentioning shared-memory limits must not be mistaken for algorithm shared-memory use.

The D shared policies launch four times as many blocks as their paired global kernels and report higher occupancy. That alone does not establish faster complete operations: their instruction work, repeated loads, shared updates and barriers also differ.

## Traffic and stalls

Global load/reduction counts below are L1TEX 32-byte sector requests, not DRAM bytes or source-level atomic counts. Global atomics compile to reductions because their old values are unused; the separate global `op_atom` counter is zero in every capture. Shared columns are operation-specific wavefront counters. Raw metrics are `l1tex__t_sectors_pipe_lsu_mem_global_op_{ld,red}.sum` and `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_{ld,atom}.sum`.

| Capture | Global load sectors | Global reduction sectors | Shared load wavefronts | Shared atomic wavefronts | DRAM throughput % | L2 throughput % |
|---|---:|---:|---:|---:|---:|---:|
| [Root: per-output counts](ncu-per-output-root-raw.stdout) | 98,304 | 3,096,128 | 0 | 0 | 2.45 | 43.53 |
| [Root: cached counts](ncu-cached-root-raw.stdout) | 98,304 | 2,057,132 | 0 | 0 | 3.97 | 46.38 |
| [Split: one root](ncu-per-tree-split-raw.stdout) | 2,264 | 0 | 0 | 0 | 1.37 | 2.05 |
| [Split: 16 roots](ncu-batched-split-raw.stdout) | 36,250 | 0 | 0 | 0 | 3.67 | 1.80 |
| [D2: global16](ncu-deeper-2-global16-raw.stdout) | 135,168 | 2,834,127 | 0 | 0 | 5.27 | 48.52 |
| [D2: shared1 / 4 chunks](ncu-deeper-2-shared1-c4-raw.stdout) | 2,301,179 | 149,500 | 1,413,442 | 1,952,830 | 5.17 | 37.64 |
| [D6: global16](ncu-deeper-6-global16-raw.stdout) | 33,792 | 707,363 | 0 | 0 | 4.93 | 36.80 |
| [D6: shared4 / 4 chunks](ncu-deeper-6-shared4-c4-raw.stdout) | 177,786 | 75,615 | 863,133 | 621,532 | 4.11 | 8.25 |

Cached root counts leave the captured global load sectors unchanged at 98,304 while reduction sectors decrease from 3,096,128 to 2,057,132. This is consistent with removing one update stream, rather than eliminating gradient/Hessian gathering. The full operation still pays count setup and seeding costs.

For D2, shared1 reduces global reduction sectors from 2,834,127 to 149,500 but increases global load sectors from 135,168 to 2,301,179. For D6, shared4 reduces reductions from 707,363 to 75,615 while loads increase from 33,792 to 177,786. The source reloads derivatives and assignments per feature in shared kernels; shared1 also accesses one output through a strided row-major derivative array. These observations support a traffic tradeoff, not a claim that any one counter explains the full measured difference.

Histograms show low DRAM utilization, with most valid L2 hit-rate readings near 98%; these captures do not demonstrate DRAM-bandwidth saturation. D6 shared4 reports an impossible derived L2 hit rate of **100.494773%** in raw counters (100.49% in details). It is retained as an out-of-range profiler measurement and excluded from physical hit-probability conclusions. Its cause is not established by this capture; no clipping or silent correction is applied.

The following are average warp stall cycles per issued instruction, not percentages of wall time. They must not be summed across kernels or treated as additive training-time costs. Raw names are `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`.

| Capture | Long scoreboard | Short scoreboard | LG throttle | MIO throttle | Barrier | Total warp cycles / issue |
|---|---:|---:|---:|---:|---:|---:|
| Root: per-output counts | 88.05 | 3.43 | 50.11 | 56.60 | 0.00 | 205.44 |
| Root: cached counts | 77.66 | 2.06 | 32.11 | 34.72 | 0.00 | 157.32 |
| Split: one root | 1.54 | 5.59 | 0.00 | 0.00 | 0.00 | 12.43 |
| Split: 16 roots | 1.15 | 11.64 | 0.00 | 0.00 | 0.00 | 17.98 |
| D2: global16 | 55.78 | 2.93 | 33.21 | 35.52 | 0.00 | 145.46 |
| D2: shared1 / 4 chunks | 3.87 | 27.05 | 0.17 | 0.77 | 15.08 | 67.14 |
| D6: global16 | 37.29 | 3.61 | 0.24 | 1.14 | 0.00 | 53.09 |
| D6: shared4 / 4 chunks | 2.60 | 23.28 | 0.00 | 0.23 | 4.74 | 49.12 |

Root count reuse lowers LG/MIO queue-throttle ratios alongside fewer reduction requests; long-scoreboard waiting remains large. D global kernels likewise retain long-scoreboard dependencies. Shared kernels exchange much of that waiting for short-scoreboard and barrier stalls. The split kernels have zero algorithm shared traffic and barrier stalls, so their short-scoreboard stalls cannot be labeled shared-memory stalls; this category also includes other MIO dependencies. Source-level instruction attribution would be needed to isolate individual producers.

## Shared CAS and layout evidence

[`deeper-sass-notes.md`](deeper-sass-notes.md) records compiled SM86 `ATOMS.CAST.SPIN.64` update loops for both FP64 fields and the uint64 shared count. Nsight reports 654,194 shared-load bank conflicts for D2 shared1, accounting for 46.28% of its 1,413,442 shared-load wavefronts. D6 shared4 reports 600,674 conflicts, 69.59% of 863,133 shared-load wavefronts. The wavefront counters are not counts of CAS retries; the captures establish conflicts and shared traffic but do not isolate a retry count.

An actionable layout inference follows from D6’s source. Local cells are `[output][node][bin]` with 24-byte Stats. Output stride is `2*16*24 = 768` bytes and node stride is `16*24 = 384` bytes. Both are multiples of the 128-byte bank-address cycle, while a same-row output subgroup shares its bin ID. Thus its different-output fields can target the same banks at different addresses. NVIDIA documents 32 banks with successive 32-bit words assigned to successive banks in the [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks). Making output the innermost local dimension, or another conflict-aware layout, is a concrete future candidate; it has not been implemented or measured here. Separately, uint32 local counts can be exact for this uint32-row contract and could remove the count CAS, but that candidate is also unmeasured.

The evidence supports removing redundant updates, batching independent split work, and testing shared-layout/work reductions. It does not establish that these kernels are globally fastest, that higher occupancy guarantees a win, or that all remaining inefficiencies have been removed. Complete-operation results remain in [`deeper-results.md`](deeper-results.md); whole-training and strict quality gates belong to the main report.
