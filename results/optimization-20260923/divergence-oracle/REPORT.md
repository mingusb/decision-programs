# Bounded saved-model divergence diagnostic

CPU validation only; no GPU execution, production changes or performance ranking.

The three runs have bitwise-identical base margins and identical binary feature quantization. All previous strict quality failures remain failed.

| Output | Compared run | First differing node | Saved features | Ideal gain | Gap to different count signature | Saved choices share ideal-best counts |
|---:|---|---|---|---:|---:|---|
| 573 | pair0-warp-wide | RR | 290 / 262 | 25.8634549855 | 9.7680024969 | True |
| 573 | pair1-warp32 | RR | 290 / 196 | 25.8634549855 | 9.7680024969 | True |
| 87 | pair0-warp-wide | root | 80 / 114 | 509.690795344 | 1.10305138118 | True |
| 87 | pair1-warp32 | root | 80 / 407 | 509.690795344 | 1.10305138118 | True |
| 8 | pair0-warp-wide | R | 238 / 85 | 5.79436755549 | 0.256643670823 | True |

For validation row526/output573:

| Run | Leaf path | Leaf increment | Saved probability | CPU probability minus saved |
|---|---|---:|---:|---:|
| pair0-warp32 | RRR | -0.0036925094163568215 | 0.019087089828201544 | 0 |
| pair0-warp-wide | RRL | 3.6228729077672899 | 0.42240555652057149 | -5.551e-17 |
| pair1-warp32 | RRL | 3.6228729077672899 | 0.42240555652057149 | -5.551e-17 |

All saved leaf values in these three outputs were independently checked from their actual training memberships; maximum absolute difference from the 80-digit count-based reference is 1.5503741784966285e-14. This is an observed discrepancy, not a new correctness tolerance.

Full candidate rankings, exact integer counts, selected partition membership hashes, 80/120-digit scores, exact-rational host-derivative checks, base bits, source hashes and prediction paths are retained in `findings.json`.

No GPU histogram snapshot exists. The oracle identifies mathematical ties and score separations; it does not prove the device's actual FP64 sums or blame a particular reduction order. Count-derived first-round statistics are an optional numerical design change, not an exact-preserving optimization.

The independent regularized score formula follows [Chen and Guestrin, section2.2](https://arxiv.org/html/1603.02754v3#S2.SS2) and the local documented leaf objective.
