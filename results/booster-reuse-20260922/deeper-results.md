# Deeper histogram complete-operation measurements

Batched global accumulation has the lowest observed sample median in all 14 multi-output shapes, in both stream and graph execution. The two scalar shapes favor shared1 with four times the existing row-chunk count. This is a primitive result, not a measured training speedup.

Across all 32 shape/mode combinations, the best batched median is 1.11–12.03 times faster than the better existing sequential policy. Stream ratios include device idle gaps caused by host submission; graph ratios provide a separate comparison with those submissions captured. The 28 global and four shared median wins do not establish statistically resolved rankings: several raw ranges overlap, including the close global/shared comparison in case 9.

Source: [`deeper-benchmark.stdout`](deeper-benchmark.stdout), SHA-256 `f7e6349b1651a4671ca157e544a66d1029b5f1003191538df71f2713999798ab`. All 32 records have seven finite positive samples per variant and complete alternating order records. The benchmark reports 63,212,563 validation checks, including CPU references before and after timings. [`deeper-summary.json`](deeper-summary.json) retains median/min/max for every tested variant's device and host-wall times, all dimensions, and descriptive range-overlap flags.

## Measurement boundary

Each operation overwrites the active histogram range and accumulates every selected output. Sequential variants perform two launches per output; batched variants perform one batched clear plus one accumulation launch. Inactive node capacity is preserved in all variants. Inputs are resident feature-major bins, output-major row assignments, and compact row-major FP64 gradients/Hessians. Integer row counts remain uint64. No packing, transpose, or auxiliary row-list construction is omitted.

Two warmup samples precede seven alternating-order samples. Each sample contains eight complete operations, or two for 65,536-row shapes. CUDA-event time divided by that count is reported below. Host-wall time includes event submission and completion waiting and is separately retained in JSON. Allocation, initial transfers, graph construction and CPU references are outside the timed boundary. Profiler output is not used.

## All shapes

| Case | Rows | Features | Outputs | Node capacity | Maximum bins | Distribution and active state |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 4,096 | 16 | 1 | 2 | 32 | all nodes active; hashed assignments and bins |
| 1 | 4,096 | 16 | 3 | 2 | 32 | approximately 75% negative assignments; irregular bins; varying active counts including zero |
| 2 | 4,096 | 16 | 16 | 2 | 32 | all nodes active; hashed assignments and bins |
| 3 | 4,096 | 16 | 16 | 2 | 32 | skewed bins |
| 4 | 4,096 | 16 | 16 | 2 | 32 | approximately 75% negative assignments |
| 5 | 4,096 | 16 | 16 | 2 | 32 | zero G/H |
| 6 | 1,024 | 16 | 16 | 2 | 16 | all nodes active; hashed assignments and bins |
| 7 | 1,024 | 16 | 16 | 8 | 16 | irregular bins; varying active counts including zero |
| 8 | 4,096 | 16 | 33 | 2 | 32 | irregular bins; varying active counts including zero |
| 9 | 4,096 | 16 | 16 | 8 | 32 | all nodes active; hashed assignments and bins |
| 10 | 4,096 | 16 | 16 | 32 | 64 | approximately 75% negative assignments; irregular bins; varying active counts including zero |
| 11 | 4,096 | 16 | 16 | 8 | 256 | skewed bins |
| 12 | 65,536 | 32 | 1 | 2 | 64 | all nodes active; hashed assignments and bins |
| 13 | 65,536 | 16 | 16 | 2 | 32 | all nodes active; hashed assignments and bins |
| 14 | 65,536 | 16 | 16 | 8 | 16 | approximately 75% negative assignments |
| 15 | 4,096 | 32 | 16 | 2 | 32 | all nodes active; hashed assignments and bins |

## Stream results

Microseconds: median [minimum, maximum] over seven raw samples. `S1/c4` denotes batched shared width 1, four row chunks. Speedup compares the best batched median with the better sequential median for the same row.

| Case | Sequential global | Sequential shared | Batched global | Best batched shared | Shared time | Best / sequential speedup |
|---:|---:|---:|---:|---|---:|---:|
| 0 | 32.00 [30.21, 83.45] | 46.21 [45.82, 53.76] | 30.59 [29.95, 31.23] | S1/c4 | 24.45 [23.81, 40.45] | 1.31x |
| 1 | 70.53 [53.25, 245.89] | 98.05 [95.49, 107.90] | 22.66 [21.89, 24.58] | S1/c4 | 33.66 [33.15, 230.52] | 3.11x |
| 2 | 559.62 [504.58, 759.81] | 722.79 [689.28, 775.52] | 136.83 [124.16, 265.22] | S1/c1 | 165.89 [158.59, 383.49] | 4.09x |
| 3 | 669.57 [648.06, 805.76] | 3563.90 [3302.27, 3696.90] | 170.75 [159.36, 327.42] | S8/c4 | 423.81 [416.74, 550.76] | 3.92x |
| 4 | 315.52 [290.92, 884.35] | 447.23 [395.39, 1283.07] | 40.45 [39.30, 67.45] | S1/c4 | 65.66 [64.26, 151.42] | 7.80x |
| 5 | 361.83 [327.17, 542.34] | 409.73 [399.36, 588.42] | 50.94 [46.72, 54.91] | S4/c4 | 58.35 [56.04, 156.80] | 7.10x |
| 6 | 355.84 [333.57, 530.82] | 398.98 [326.66, 601.47] | 43.78 [43.14, 51.33] | S1/c1 | 51.84 [51.71, 58.24] | 8.13x |
| 7 | 311.04 [267.25, 331.65] | 342.66 [286.85, 374.02] | 25.86 [25.47, 39.55] | S1/c4 | 31.74 [31.21, 38.66] | 12.03x |
| 8 | 807.94 [800.26, 1213.95] | 3246.85 [3140.10, 3493.63] | 175.74 [167.67, 270.85] | S4/c4 | 240.38 [231.40, 368.51] | 4.60x |
| 9 | 511.62 [344.94, 644.61] | 614.02 [586.24, 751.10] | 114.82 [111.23, 120.83] | S1/c4 | 116.10 [112.77, 121.58] | 4.46x |
| 10 | 309.89 [257.92, 557.18] | 358.02 [343.42, 449.79] | 39.55 [38.53, 46.20] | S1/c1 | 64.38 [63.09, 66.18] | 7.83x |
| 11 | 531.58 [426.11, 597.25] | 2255.72 [2214.66, 2483.20] | 180.74 [174.98, 370.02] | S1/c1 | 494.46 [486.77, 664.06] | 2.94x |
| 12 | 375.81 [308.22, 400.35] | 313.34 [253.44, 776.70] | 389.10 [327.17, 612.35] | S1/c64 | 280.58 [258.05, 370.18] | 1.12x |
| 13 | 7700.99 [6903.81, 8705.54] | 5364.22 [3668.48, 7301.63] | 2580.48 [2001.92, 3134.46] | S1/c64 | 3411.46 [3371.52, 4292.10] | 2.08x |
| 14 | 1656.32 [1629.70, 2086.40] | 2063.36 [1971.71, 3241.98] | 640.51 [634.37, 757.76] | S1/c16 | 1501.70 [1455.10, 2343.42] | 2.59x |
| 15 | 756.35 [727.55, 818.05] | 837.50 [788.99, 1007.49] | 265.09 [250.50, 359.17] | S1/c4 | 278.66 [267.90, 289.79] | 2.85x |

## Graph results

Microseconds: median [minimum, maximum] over seven raw samples. `S1/c4` denotes batched shared width 1, four row chunks. Speedup compares the best batched median with the better sequential median for the same row.

| Case | Sequential global | Sequential shared | Batched global | Best batched shared | Shared time | Best / sequential speedup |
|---:|---:|---:|---:|---|---:|---:|
| 0 | 26.50 [26.11, 27.01] | 42.75 [42.62, 43.39] | 26.62 [26.11, 27.14] | S1/c4 | 20.35 [20.10, 21.89] | 1.30x |
| 1 | 20.74 [20.48, 21.25] | 76.80 [76.03, 84.99] | 12.54 [12.03, 12.67] | S1/c4 | 26.88 [26.50, 144.62] | 1.65x |
| 2 | 400.13 [381.82, 408.32] | 713.86 [630.66, 807.04] | 128.11 [119.81, 216.70] | S1/c4 | 136.96 [131.97, 294.53] | 3.12x |
| 3 | 593.92 [589.18, 843.65] | 3552.64 [3279.36, 3700.35] | 155.39 [155.01, 160.77] | S8/c4 | 394.62 [389.89, 408.45] | 3.82x |
| 4 | 146.05 [145.92, 147.97] | 350.98 [340.46, 505.86] | 37.12 [36.48, 38.91] | S1/c4 | 62.59 [61.95, 62.85] | 3.93x |
| 5 | 168.32 [164.22, 224.26] | 345.22 [339.07, 461.82] | 43.90 [43.52, 52.35] | S4/c4 | 54.14 [52.86, 62.21] | 3.83x |
| 6 | 211.71 [210.56, 215.92] | 241.41 [240.90, 243.94] | 39.94 [39.81, 40.96] | S4/c4 | 46.98 [46.72, 97.65] | 5.30x |
| 7 | 134.02 [133.89, 188.03] | 183.30 [182.66, 321.52] | 22.27 [22.14, 22.78] | S1/c4 | 27.90 [27.52, 29.43] | 6.02x |
| 8 | 817.92 [707.71, 847.36] | 3097.34 [2918.40, 3291.26] | 169.86 [158.85, 251.39] | S4/c4 | 231.17 [228.74, 233.96] | 4.82x |
| 9 | 280.19 [276.10, 399.10] | 547.20 [541.03, 668.42] | 110.85 [109.57, 112.77] | S1/c4 | 111.49 [110.46, 112.90] | 2.53x |
| 10 | 113.28 [111.87, 194.82] | 287.62 [287.36, 343.04] | 35.58 [35.46, 37.38] | S1/c1 | 60.93 [59.90, 62.59] | 3.18x |
| 11 | 375.04 [363.01, 512.77] | 2211.07 [2118.02, 2353.54] | 175.87 [172.40, 251.26] | S1/c4 | 535.17 [527.10, 701.82] | 2.13x |
| 12 | 379.90 [362.50, 404.43] | 310.27 [295.92, 313.86] | 375.30 [360.86, 383.49] | S1/c64 | 279.55 [268.80, 548.86] | 1.11x |
| 13 | 8033.28 [6948.86, 8862.72] | 4792.32 [4211.71, 6002.69] | 2517.50 [1928.67, 3867.65] | S1/c64 | 3401.22 [3364.86, 4298.24] | 1.90x |
| 14 | 1630.72 [1550.85, 2389.50] | 2091.01 [2005.50, 3289.09] | 679.42 [632.83, 1099.78] | S1/c16 | 1595.90 [1478.14, 2948.10] | 2.40x |
| 15 | 705.15 [666.37, 752.38] | 766.85 [717.44, 1089.15] | 256.63 [246.77, 319.62] | S1/c4 | 280.45 [267.39, 342.27] | 2.75x |

## Representative profiling priorities

Case 2 (4,096 rows, 16 outputs, two nodes, 32 bins) favors batched global: 128.11 µs graph median versus 400.13 µs for sequential global and 136.96 µs for the best shared candidate, shared1/chunks4. The global [119.81, 216.70] and shared [131.97, 294.53] ranges overlap substantially; capture global and shared1/chunks4 to explain different atomic traffic, assignment loads and shared spin updates, without declaring the small median gap conclusive. The two shared1 stream medians both round to 165.89 µs; their tiny numerical ordering is immaterial.

Case 6 (1,024 rows, 16 outputs, two nodes, 16 bins) favors batched global: 39.94 µs graph median [39.81, 40.96], versus 211.71 µs sequential global and 46.98 µs shared4/chunks4 [46.72, 97.65]. These observed global/shared ranges do not overlap, although this is not a confidence interval. The smaller input is useful for examining launch/grid efficiency. Profile global first and shared4/chunks4 second.

For `--case 2` or `--case 6`, the mangled filter `.*global_accumulateILj16EE.*` with `--launch-skip 17 --launch-count 1` selects the first measured stream global invocation after validation and two warmup samples. `.*shared_accumulateILj1EE.*` (case 2) or `.*shared_accumulateILj4EE.*` (case 6), with skip 42, selects the four-chunk invocation. Use `--kernel-name-base mangled --set full --clock-control none --cache-control none`; confirm the launch shape in the report. Any benchmark timing produced under Nsight is diagnostic and must not enter these rankings.

## Exceptions and limits

Batched global is not uniformly preferable: on scalar case 0 its graph median is slightly slower than existing sequential global (26.624 versus 26.496 µs), and on scalar case 12 it loses to existing sequential shared (375.30 versus 310.27 µs graph). The shared1/chunks4 or chunks64 alternatives win those scalar rows. Existing sequential shared is also faster than sequential global in the 65,536-row, 16-output case 13; the comparison above therefore uses shared as the stronger baseline there.

- Synthetic output-specific hashed assignments do not model learned feature/partition correlations or highly imbalanced leaf populations.
- Primitive timing excludes tree-state retention, routing, splitting, model export and full training.
- Independent-output compact derivative strides are covered; multiclass strides larger than the output batch are correctness-covered only by generic nonzero-offset/padded tests, not this timing matrix.
- Floating-point addition order may change; CPU error-bound validation is not a zero-allowance model-quality or bitwise-equivalence gate.
- One benchmark process supplies seven interleaved samples per variant, not independent process-level repetitions or confidence intervals.
- Warm repeated inputs model reuse; clock control is not imposed by this benchmark. Range overlap is descriptive, not a significance test.
- Observed median selection across multiple policies can be optimistic; no default promotion or universal fastest claim is justified.

Shared policy selection varies with shape. Wider grouped shared histograms are not consistently best; width 1 often wins among shared candidates. The SM86 disassembly separately records FP64 and uint64 shared CAS loops. Runtime profiles are needed to attribute losses to contention, repeated derivative loads, occupancy, or launch shape. The exact uint32 local-count candidate remains unimplemented and unmeasured.

These results justify evaluating retained per-output training state so deeper work can be batched. They do not justify multiplying a whole-training time by these speedups, overlooking the additional assignment/histogram/frontier storage, promoting a default, or marking any existing strict model-quality failure as passing.
