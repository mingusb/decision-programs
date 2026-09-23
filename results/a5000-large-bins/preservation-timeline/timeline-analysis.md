# Matched stream timeline diagnostic

Diagnostic only; unprofiled preservation remains unresolved.

All four reports passed command, identity, correlation, operation-count, kernel-resource, and gap-reconciliation checks. Each profile has 21 ranges of 32 timed operations; warmup operations are excluded.

Values below are microseconds per operation, except profile names. Each cell is the median across 21 sample ranges.

| Profile | Traced event | Kernel | Memset | GPU gaps | Launch API | Memset API | Between observed APIs |
|---|---:|---:|---:|---:|---:|---:|---:|
| p1-old | 40.031 | 6.486 | 1.438 | 30.412 | 10.980 | 12.151 | 1.943 |
| p2-current | 29.022 | 6.480 | 1.465 | 19.435 | 11.386 | 11.086 | 2.159 |
| p3-current | 26.812 | 6.449 | 1.448 | 18.723 | 9.950 | 10.311 | 1.851 |
| p4-old | 22.619 | 6.490 | 1.445 | 13.824 | 7.960 | 7.947 | 1.409 |

| Profile | GPU gap before next API enters | During next API | After next API returns |
|---|---:|---:|---:|
| p1-old | 0.000 | 1.278 | 30.412 |
| p2-current | 0.000 | 0.039 | 18.678 |
| p3-current | 0.000 | 0.000 | 17.648 |
| p4-old | 0.000 | 0.000 | 13.824 |

The first old profile is the slowest and the second old profile is the fastest. Counting-kernel and device-memset durations are close across binaries; the large traced differences occur in submission/API time and gaps between GPU activities. Most gap time occurs after the next operation's API has already returned, so it cannot all be labeled late host submission. This localizes the traced variability but does not establish the cause of the unprofiled regression.

## Limits

- Diagnostic instrumentation changes runtime behavior; these traces do not replace unprofiled preservation evidence.
- Each NVTX sample contains one warmup plus 32 timed operations. Timed activities are selected by runtime correlation IDs between the two CPU event-record APIs; the warmup pair is excluded.
- Device event timestamps are zero. GPU activity span covers first timed memset start through last timed kernel end, not the exact CUDA event interval.
- GPU gaps mean absence of this target stream's recorded memset/kernel activity. They do not establish whole-device idleness or identify external GPU work.
- Gap partitions describe when the next operation's API is not entered, active, or returned. They do not assign causation to application instructions, operating-system scheduling, driver queues, or profiler overhead.
- Observed API durations include instrumentation and possible waiting; cuKernelGetName is reported separately. CPU context switches and instruction sampling were disabled.
- CPU API durations overlap GPU execution; they must not be added to GPU durations as one elapsed-time decomposition.
- Nsight warns that scheduling information is absent and not all NVTX events might have been collected. This audit nevertheless finds all 21 expected benchmark ranges and every expected operation within them; raw profiler diagnostics are retained.
- Per-profile table entries are medians of per-range values divided by 32; medians of component columns need not sum exactly.
- Telemetry surrounding the profiler includes startup and export time and cannot establish clocks during the target's measured ranges.
