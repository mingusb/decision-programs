# Nsight Compute diagnostics

These are first matching stream-mode launches on the full Delicious workload, from the successful twelve-job profiling retry. They are diagnostic observations, not uninstrumented timing rankings. Each captured derivative/root launch processes the first output tile; the benchmark trains all 983 outputs. Raw metric names, original values/units and source hashes are in [compute-analysis.json](compute-analysis.json).

## Root histogram accumulation

| Order | Global reduction requests | Reduction sectors | L2 throughput (% peak) | DRAM read + write throughput (% peak) | Achieved occupancy |
|---|---:|---:|---:|---:|---:|
| 2 | 5,168,000 | 163,957,936 | 62.25% | 3.63% | 91.36% |
| 3 | 7,752,000 | 248,064,000 | 65.30% | 3.41% | 90.26% |
| 4 | 10,336,000 | 330,752,000 | 66.50% | 2.68% | 88.95% |

The observed reduction requests increase exactly 1.5x and 2x with the added nonzero statistic fields. Counts are cached; global atomic-return requests are zero because unused-return additions appear as reduction instructions. Low DRAM throughput together with substantial L2/reduction activity argues against treating this launch as a simple off-chip bandwidth-saturation problem. Long-scoreboard stalls and additional reduction traffic support investigating histogram atomic/cache pressure; the capture does not isolate every cause of serialization.

## Root split scoring

| Order | Registers/thread | Register-limited blocks/SM | Achieved occupancy | FP64 pipeline active (% elapsed) | Eligible warps/scheduler | Spill instructions |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 72 | 3 | 48.07% | 81.76% | 0.146 | 0 |
| 3 | 90 | 2 | 32.71% | 75.25% | 0.088 | 0 |
| 4 | 96 | 2 | 32.82% | 79.82% | 0.074 | 0 |

All three captured root split launches use 32,000 blocks of 256 threads. At the root, only one of four frontier-capacity slots is active, so 8,000 feature/output blocks perform candidate work and the rest return. The source keeps the 256-thread path for 500 features, although these features have roughly three bins. Higher-order source also evaluates the parent leaf proposal before per-bin candidate predicates. Reducing unused block/thread work and testing whether the common parent score can be computed once are candidates, not measured speedups. Source-level redundancy is not by itself proof of the exact generated instruction schedule.

Measured FP64 pipeline activity of 75–82% and the register-limited occupancy drop support focusing on arithmetic work and block shape. All nine captured kernels report zero register-spill instructions and zero local-memory load/store sectors. The evidence does not support blaming spilling or introducing a precision approximation.

## Fused derivatives

| Order | Registers/thread | Achieved occupancy | FP64 pipeline active (% elapsed) |
|---|---:|---:|---:|
| 2 | 29 | 84.79% | 76.26% |
| 3 | 24 | 85.22% | 74.03% |
| 4 | 24 | 86.94% | 73.99% |

Higher derivatives reuse the exponential and write extra planes. Their register allocation is lower than the retained order-2 derivative kernel, so increased derivative-kernel register pressure is not an observed cause. Systems stage shares determine whether this small phase warrants optimization.

These captures do not characterize later boosting rounds, all frontier depths, every output tail or every shape. The full uninstrumented operation measurements remain the speed evidence. No candidate described here has been implemented or ranked by these profiler durations.
