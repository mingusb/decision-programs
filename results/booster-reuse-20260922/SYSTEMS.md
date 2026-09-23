# Nsight Systems: count reuse, split batches and deeper histograms

These captures explain GPU work and submission boundaries. Their durations are
**diagnostic observations, not uninstrumented performance rankings**. Each runs
the same 4,096-row, 16-feature, 129-output regression fixture for three rounds,
depth two, at most 32 bins, output tiles of 16 and graph tree execution. There
are 387 trees, 24 full output tiles and three one-output tails. All three commands
request CUDA graph **node** activity; their kernel counts have matching coverage.

| Capture | Root statistics | Root splits | Deeper histograms |
|---|---|---|---|
| `nsys-base` | Batched; count updates per output | Per tree, warp policy | Global |
| `nsys-both` | Batched; one reusable global count setup | Batched roots, warp policy | Global |
| `nsys-deep-shared` | Same as combined | Same as combined | Existing shared implementation |

This baseline already includes the previous root batching and warp split changes;
it is not the original block/per-tree-root trainer. The shared capture tests the
existing production deeper histogram path, not the separate experimental
multi-output deeper primitive.

Sources: saved [baseline configuration](nsys-base-benchmark/result.json),
[combined configuration](nsys-both-benchmark/result.json),
[shared configuration](nsys-deep-shared-benchmark/result.json), and the three
[baseline](nsys-base-command.json), [combined](nsys-both-command.json),
[shared](nsys-deep-shared-command.json) command receipts. The unchanged executable
SHA-256 is `fdf60682669fa790dc411e67d9053c0e02e6d49de13b365debc41254ba0f02a1`.

## Launches and recorded kernel composition

Times below sum the GPU kernel records in the full captured process. They exclude
memory-copy/memset operations and host/API gaps, and include preparation and
held-out GPU prediction. They are not complete training durations.

| Kernel group | Baseline calls / ms | Combined calls / ms | Shared deeper calls / ms |
|---|---:|---:|---:|
| All recorded kernels | 7,447 / 36.315 | 6,755 / 29.853 | 6,755 / 54.888 |
| Root weighted accumulation | 27 / 4.099 | 27 / 2.396 | 27 / 2.404 |
| Reusable count setup | 0 / 0 | 1 / 0.0128 | 1 / 0.0128 |
| Root count seeding | 0 / 0 | 27 / 0.0580 | 27 / 0.0576 |
| Deeper weighted accumulation | 387 / 9.942 | 387 / 9.939 | 387 / 35.015 |
| Deeper active-histogram clearing | 387 / 0.631 | 387 / 0.631 | 387 / 0.632 |
| Split candidates | 774 / 8.130 | 414 / 4.452 | 414 / 4.442 |
| Split winners | 774 / 2.170 | 414 / 1.166 | 414 / 1.164 |
| Tree initialization | 387 / 0.708 | 387 / 0.673 | 387 / 0.669 |
| Frontier scan/prefix/materialize/route/advance | 3,870 / 7.094 | 3,870 / 6.997 | 3,870 / 7.000 |

Sources: complete kernel CSVs for
[baseline](nsys-base-stats_cuda_gpu_kern_sum.csv),
[combined](nsys-both-stats_cuda_gpu_kern_sum.csv), and
[shared deeper](nsys-deep-shared-stats_cuda_gpu_kern_sum.csv).

The launch arithmetic matches the implementation. Each baseline tree graph
contains 18 kernels; each combined/shared graph contains 16. Removing two root
split kernels from each of 387 trees removes 774 graph-node launches, reducing
graph-node records from 6,966 to 6,192. Root splits then require two kernels per
tile, adding 54 launches outside the tree graphs. Count reuse adds 27 seeding
kernels and one setup kernel. The net reduction is therefore
`774 - 54 - 27 - 1 = 692` device kernel launches. The remaining 414 candidate and
winner calls are 387 deeper calls plus 27 root batches for each operation.

The host `cudaLaunchKernel` API count increases from 499 to 579 while device
kernel executions decrease: work moved outside the graphs, whereas graph replay
still uses one `cudaGraphLaunch` per tree. Runtime launch-call counts cannot be
substituted for graph-node execution counts.

## Cache costs and removed work

Both variants retain the existing 186,240-byte root Stats cache for 16 outputs
and 485 actual total bins. Count reuse adds 3,880 bytes. Split batching adds
11,520 bytes: an enlarged shared candidate workspace plus cached root winners.
The trainer's reported GPU payload grows from 7,769,956 to 7,785,356 bytes,
an increment of 15,400 bytes. These are owned payloads, excluding CUDA and
instrumentation bookkeeping.

Root accumulation still occurs 27 times. The kernel specializations in the CSVs
change from count-updating to count-reusing variants. The latter seed all root
Stats cells before accumulating gradients and Hessians, rather than eliminating
initialization work. Their 27 seed kernels and one count accumulation kernel
are included separately in the table. The count setup also clears its cache.

Baseline root-cache clears are CUDA memset operations, not kernel records.
Whole-capture memset counts change from 31 to five: the 27 root-cache memsets
are replaced by seed kernels and one count-cache memset. Their full-process
recorded sums are 0.0441 ms and 0.0046 ms; these include unrelated preparation
memsets and must not be treated as isolated root-clear timings. See the
[baseline memory-operation CSV](nsys-base-stats_cuda_gpu_mem_time_sum.csv) and
[combined memory-operation CSV](nsys-both-stats_cuda_gpu_mem_time_sum.csv).

Per-tree initialization remains one kernel. The baseline copies 485 Stats cells
into its root workspace; the combined path copies the selected cached Split
winner and skips the root Stats copy. The observed initializer totals are
0.708 and 0.673 ms across 387 trees. The source-level removed copy covers
4,504,680 bytes of Stats payload across those trees, with an associated read and
write for each byte; that is a logical traffic count, not measured DRAM traffic.
Winner consumption and assignment/state initialization still occur. The trace
does not justify omitting those costs from complete-operation comparisons.

## Why forced deeper shared histograms cost more here

The shared capture changes 387 deeper accumulations from the owned global kernel
to the owned per-feature shared kernel. Their diagnostic sum increases from
9.939 to 35.015 ms while the split, root and frontier counts stay the same.
The shared calls average 90.478 microseconds, with individual durations from
77.218 to 110.116 microseconds; the combined global calls average 25.682
microseconds, ranging from 23.136 to 29.089 microseconds.

That replacement is the conspicuous additional work interval in this capture:
it accounts for 63.8% of shared-capture kernel time. In the combined global
capture, deeper accumulation accounts for 33.3%, split candidates/winners
together for 18.8%, and frontier scan/prefix/materialize/route/advance for 23.4%.
These percentages describe the captured kernel denominator only.

Systems establishes where the added duration occurs. It does not by itself
determine whether shared atomic implementation, contention, derivative reloads,
barriers or available parallelism causes it; those require kernel-level evidence.
Likewise this result does not rank the separate output-grouped deeper histogram
primitive or justify extrapolating to another shape, distribution or device.

## Verified submission and export boundaries

The immutable previous-campaign auditor was run CPU-only against these new
artifacts. It opened SQLite read-only, verified integrity, recorded input hashes,
matched every emitted sample to an NVTX range, checked the recorder-clock
relationship, and attributed copies using CUDA correlation IDs. Its successful
[command receipt](trace-audit-command.json), [full JSON](trace-audit.json), and
[generated audit summary](trace-audit.md) are retained. The inherited summary
sentence saying “Both captures” is generic old wording; its table and JSON
contain all three verified captures below.

| Audit observation | Baseline | Combined | Shared deeper |
|---|---:|---:|---:|
| Matched samples / NVTX ranges | 870 / 870 | 898 / 898 | 898 / 898 |
| Tree-build scopes | 387 | 387 | 387 |
| Waits inside tree scopes | 0 | 0 | 0 |
| D2H enqueue calls inside tree scopes | 0 | 0 | 0 |
| Tree export scopes / trees covered | 27 / 387 | 27 / 387 | 27 / 387 |
| Export stream waits | 27 | 27 | 27 |
| Export D2H calls / bytes | 774 / 95,976 | 774 / 95,976 | 774 / 95,976 |
| Whole-capture stream waits | 43 | 43 | 43 |

All SQLite checks report `ok`; no runtime call crosses a matched NVTX boundary.
Every export scope has exactly one stream wait. Export bytes and calls have not
been reduced by this experiment, and no waits were introduced within tree build.
Whole-capture counts also include preparation, inference, validation and cleanup;
they are not standalone training counts.

The baseline has 27 root histogram scopes and no separate root split scopes
outside its tree graphs. Combined/shared captures have 28 histogram scopes:
one count setup at round/output -1, followed by the same 27 root batches. They
also have 27 root split scopes, each carrying the first output and tile operation
count. For histogram and split batches there are eight scopes with
`operations=16` plus one with `operations=1` per round. This accounts for the
28 additional matched samples and confirms short-tile metadata.

NVTX tree ranges bound host submission, not completion of all GPU work.
Correlation IDs provide transfer attribution even when a copy finishes after
its enclosing host range. Export scopes also overlap later tree submissions;
their durations and nested API totals must not be added to tree durations.
Comparative speed and strict quality conclusions belong to the uninstrumented
campaign and its separate quality audit.
