# Fresh Nsight Compute diagnostics

These captures diagnose individual counting kernels in the resumed session. They use kernel replay, cache-control all and unlocked clocks. They exclude output clearing and finalization costs outside the selected counting kernel, so they cannot replace unprofiled complete-operation comparisons. Historical captures came from another driver session and are not a controlled before/after baseline.

| Capture | Counting kernel, µs | DRAM peak | Registers/thread | Dynamic shared, KiB | Achieved occupancy | No eligible warp |
|---|---:|---:|---:|---:|---:|---:|
| profile-byte-wide | 5.79 | 48.45% | 23 | ≈1.00 | 63.00% | 78.76% |
| profile-large-optin | 197.89 | 90.60% | 32 | ≈64.00 | 17.06% | 95.67% |

The byte capture (`shared:11:48`, N1M/u8/B256) uses 1,024-thread blocks and approximately 1 KiB of dynamic shared memory. Its 48.45% DRAM throughput and 78.76% cycles without an eligible warp leave latency and scheduling as measurable limitations in this cache-flushed capture; this is not proof that a particular change would improve the warm-graph operation.

The opt-in capture (`shared:14:48:u32`, N16M/u32/B16384 with u64 output) uses 64 KiB of dynamic shared memory. Shared-memory capacity limits it to one 256-thread block per SM, yet it reaches 90.60% of peak DRAM throughput. Increasing occupancy alone is therefore not an established improvement.

Both reports record zero local- and shared-memory spilling requests. Nsight reports excessive shared wavefronts (68% in the byte capture and 70% in the opt-in capture), but those aggregate counts do not establish that all of the random histogram update cost is avoidable. Suggested Nsight speedups are diagnostic estimates, not measured algorithm gains.

[Extracted metrics, artifact hashes and exact commands](profile-summary.json). The original `.details.txt` and `.ncu-repz` files preserve the full evidence. Command records contain no per-profile driver/hash snapshots; this summary does not manufacture that provenance.
