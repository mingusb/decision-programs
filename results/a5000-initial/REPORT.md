# Initial A5000 results

Measured on the NVIDIA RTX A5000 Laptop GPU (SM86), driver 596.71, CUDA 13.4.59, CUB 3.4.2, CUDA C++23. These are exploratory results on one laptop GPU, not a universal fastest claim.

Each selected configuration was searched on seed 12345, validated with CUB on seeds 67890 and 24680, then measured on untouched seed 424242 in 21 randomized rounds of 20 operations. A custom candidate had to exceed 1.05x CUB on both validation seeds; otherwise the saved plan retained CUB. The independent confirmation below did not change the saved selection.

Times are median microseconds per complete device operation: clearing, counting, and every merge. Input is resident and repeatedly reused. Allocation/transfers/tuning are excluded. Clocks are not locked; laptop/WSL scheduling introduces visible outliers.

| Input / count | N | B | Distribution | Selected | CUB µs | Selected µs | Speedup |
|---|---:|---:|---|---|---:|---:|---:|
| u32 / u32 | 1,048,576 | 256 | hot99 | shared t2 grid384 | 18.07 | 15.62 | 1.16× |
| u32 / u32 | 16,777,216 | 257 | uniform | shared t2 grid96 | 549.94 | 187.60 | 2.93× |
| u32 / u64 | 16,777,216 | 4,096 | uniform | shared t3 grid384 | 2120.76 | 244.33 | 8.68× |
| u32 / u32 | 16,777,216 | 4,096 | uniform | shared t3 grid96 | 748.60 | 191.80 | 3.90× |
| u32 / u32 | 1,048,576 | 8 | uniform | cub t2 grid192 | 18.28 | 18.28 | 1.00× |
| u8 / u32 | 16,777,216 | 256 | uniform | cub t2 grid192 | 54.12 | 54.12 | 1.00× |

All rows use shuffled order. `hot99` selects the highest bin for 99% of draws and samples uniformly otherwise. Policy and grid columns for CUB are ignored; CUB chooses its own launch. Its internal offsets were verified to narrow to 32-bit for these input sizes.

The eight-bin custom candidate had a faster median in the final confirmation but failed the two-seed selection gate; its plan correctly remains CUB. The byte-input result is effectively tied. The bit-plane backend did not win the tested cases; it remains an experimental family rather than a selected default.

## Profiling evidence

- `shared-4096.metrics.txt`: Nsight Compute measured 93.68% of peak DRAM throughput (359.46 GB/s), 52 registers/thread, and no shared-memory spilling requests for the selected 32-bit-counter kernel. This is diagnostic instrumentation, not the timing used to rank candidates.
- `cub-4096.metrics.txt`: the comparable CUB sweep measured 732.70µs under profiling, versus 193.15µs for the shared kernel.
- `timeline-4096.stats.txt`: Nsight Systems captured 13 invocations per kernel; medians were 187.21µs for the custom counting kernel, 724.05µs for CUB counting, and 1.57µs for CUB initialization. Kernel-only values exclude custom clearing and must not replace the complete-operation table.
- The `.ncu-repz` and `.nsys-rep` captures are retained alongside their text exports.

## Validation and reproducibility

- Correctness suite: 12,176 executions against independent CPU counts, full-vector equality and sum checks, guarded output/scratch, partial warps, non-power-of-two bins, both types/counter widths, poisoned/reused buffers, nondefault streams, and invalid-configuration checks.
- Compute Sanitizer: 1,304 executions per tool; memcheck and synccheck reported 0 errors, racecheck reported 0 hazards. These tests do not empirically exercise output counts exceeding UINT32_MAX.
- `resources.txt` records compiler resource usage. `environment.json` records tool versions and source SHA256 values.
- Every workload JSON retains the executable hash, selected policy, raw search/validation CSV hashes, and exact commands. Replay with `python3 tools/autotune.py --replay <plan.json>`; changed binaries are rejected.
- Positive replay and deliberate stale-hash rejection were exercised.

## Next useful experiments

Expand the size/bin/order matrix and test cold-cache and CUDA Graph timing separately. For 64-bit outputs, test safely bounded 32-bit block-local counters as an additional template policy. Refine small-operation measurements before accepting a distribution-independent dispatch rule. Larger source/device coverage is required before claiming an overall performance frontier.
