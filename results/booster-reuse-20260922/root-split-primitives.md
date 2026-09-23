# Root count reuse and split batching: primitive measurements

Sources: [root-count CSV](root-count-benchmark.stdout), [split benchmark JSON](split-batch-benchmark.stdout), and their [root](root-count-benchmark-command.json) / [split](split-batch-benchmark-command.json) command records. Both commands exited successfully with unchanged executable hashes. [primitive-summary.json](primitive-summary.json) preserves source hashes, every raw sample, trial order and derived statistics. The benchmark implementations are frozen in [root_histogram.cu](final-provenance/training/tests/root_histogram.cu) and [split_search.cu](final-provenance/training/tests/split_search.cu).

These are unprofiled CUDA-event observations. Timed stream submission can contain host submission gaps; graph timings replay prebuilt graphs. All reported samples are retained. Tables show **median [minimum, maximum]**, not confidence intervals. Ratios divide the reference median by the candidate median; values above one indicate lower candidate time. Neither ratios nor extrema establish universal rankings or training-quality preservation.

## Count reuse: boundaries and results

The complete comparison is count setup followed by three full root histogram batches. A batch contains every output in the shape. Setup includes clearing and accumulating the immutable count cache. Every histogram includes its Stats clear or count-seeding operation plus gradient/Hessian accumulation. Allocation, uploads, tree-initializer cache consumption, splits and full training are outside this boundary. A count policy can therefore help this component while losing complete training.

Each field is independently timed: one untimed invocation, then seven timed invocations averaged, repeated for three trials. Policy order is forward/reverse/forward. `setup_plus_three_batches_ms` is a separately observed region, **not** the sum of `setup_ms` and `three_batches_ms`; the recorded differences and outliers are retained. For `per_output`, that complete region contains only three histogram batches.

All shapes have 16 numeric features. Uniform means the deterministic modular bin pattern in the frozen fixture; skew means that rows with `row % 17 != 0` are forced to missing bin zero (over 94% of rows). The benchmark itself does not download or independently validate timed histogram buffers. Separate primitive correctness tests cover the algorithms; the timing program must not be described as validating every measured output.

Count reuse with shared setup has a lower complete-boundary median in all six tested shapes. Global setup has a lower median in five of six: on 65,536 uniform rows with one output it is **0.950 ms versus 0.904 ms**, a 5.1% increase, with overlapping observed ranges. Shared setup in that case is 0.626 ms. Global reuse also loses the paired trial 1 comparison for the 4,096-row skew case (0.783 versus 0.734 ms), even though its median improves. It loses paired trials 0 and 2 of the scalar uniform case. No such observations were removed.

Every workload and timing boundary follows; values are milliseconds. N=rows, T=outputs, B=bins per feature.

| N / T / B | Distribution | Policy | Setup | One batch | Three batches | Setup + three batches | Complete ratio |
|---|---|---|---:|---:|---:|---:|---:|
| 1024 / 16 / 16 | uniform | per_output | 0.000 [0.000, 0.000] | 0.058 [0.057, 0.082] | 0.238 [0.171, 0.321] | 0.171 [0.170, 0.218] | — |
| 1024 / 16 / 16 | uniform | reuse_global | 0.015 [0.012, 0.015] | 0.032 [0.032, 0.032] | 0.098 [0.097, 0.098] | 0.107 [0.106, 0.108] | 1.599 |
| 1024 / 16 / 16 | uniform | reuse_shared | 0.014 [0.013, 0.015] | 0.033 [0.032, 0.033] | 0.098 [0.097, 0.110] | 0.104 [0.104, 0.104] | 1.640 |
| 1024 / 16 / 16 | skew | per_output | 0.000 [0.000, 0.000] | 0.074 [0.074, 0.074] | 0.224 [0.223, 0.292] | 0.224 [0.222, 0.224] | — |
| 1024 / 16 / 16 | skew | reuse_global | 0.018 [0.015, 0.025] | 0.045 [0.045, 0.046] | 0.136 [0.136, 0.138] | 0.148 [0.148, 0.148] | 1.510 |
| 1024 / 16 / 16 | skew | reuse_shared | 0.011 [0.011, 0.016] | 0.046 [0.045, 0.046] | 0.140 [0.136, 0.140] | 0.144 [0.141, 0.146] | 1.556 |
| 4096 / 16 / 32 | uniform | per_output | 0.000 [0.000, 0.000] | 0.154 [0.154, 0.165] | 0.625 [0.497, 0.786] | 0.495 [0.462, 0.570] | — |
| 4096 / 16 / 32 | uniform | reuse_global | 0.021 [0.016, 0.090] | 0.091 [0.090, 0.096] | 0.288 [0.270, 0.318] | 0.280 [0.280, 0.296] | 1.767 |
| 4096 / 16 / 32 | uniform | reuse_shared | 0.019 [0.016, 0.075] | 0.091 [0.090, 0.159] | 0.269 [0.268, 0.284] | 0.295 [0.278, 0.371] | 1.680 |
| 4096 / 16 / 32 | skew | per_output | 0.000 [0.000, 0.000] | 0.220 [0.220, 0.236] | 0.699 [0.659, 0.726] | 0.691 [0.663, 0.734] | — |
| 4096 / 16 / 32 | skew | reuse_global | 0.029 [0.025, 0.031] | 0.143 [0.140, 0.147] | 0.487 [0.425, 0.689] | 0.465 [0.441, 0.783] | 1.487 |
| 4096 / 16 / 32 | skew | reuse_shared | 0.017 [0.014, 0.018] | 0.143 [0.141, 0.148] | 0.428 [0.427, 0.465] | 0.435 [0.420, 0.500] | 1.588 |
| 65536 / 1 / 64 | uniform | per_output | 0.000 [0.000, 0.000] | 0.281 [0.264, 0.319] | 0.824 [0.796, 0.878] | 0.904 [0.789, 1.115] | — |
| 65536 / 1 / 64 | uniform | reuse_global | 0.099 [0.099, 0.101] | 0.194 [0.190, 0.196] | 0.647 [0.577, 0.652] | 0.950 [0.665, 0.959] | 0.952 |
| 65536 / 1 / 64 | uniform | reuse_shared | 0.018 [0.015, 0.021] | 0.192 [0.192, 0.192] | 0.573 [0.570, 0.617] | 0.626 [0.581, 0.663] | 1.444 |
| 65536 / 1 / 64 | skew | per_output | 0.000 [0.000, 0.000] | 1.246 [1.202, 1.290] | 4.029 [3.735, 4.075] | 4.050 [3.988, 4.051] | — |
| 65536 / 1 / 64 | skew | reuse_global | 0.284 [0.284, 0.347] | 0.979 [0.962, 1.263] | 3.038 [2.949, 3.312] | 3.482 [3.309, 3.539] | 1.163 |
| 65536 / 1 / 64 | skew | reuse_shared | 0.016 [0.015, 0.020] | 1.044 [0.967, 1.286] | 3.243 [2.994, 3.271] | 3.130 [3.049, 3.339] | 1.294 |

Global and shared count setup feed the same subsequent reuse histogram implementation. Differences between their separately timed one-batch columns are not evidence of different accumulation algorithms. For example, the 4,096-row uniform shared setup has a 0.016–0.075 ms setup spread and a 0.090–0.159 ms one-batch spread. Such variability makes component addition and small per-policy distinctions unreliable.

## Root split batching: boundaries and results

The complete split operation reads already resident output-major root statistics and writes every feature candidate and root winner. The sequential reference launches candidate and winner kernels separately for each root (2*T launches). The candidate evaluates all T roots in two launches. Allocation, transfers, graph construction, histogram building and tree initialization are excluded. The production removal of the per-tree root histogram copy is consequently outside this microbenchmark and must be assessed in complete training.

Each sample averages 32 complete operations. Two warmup iterations precede seven recorded samples per variant, with alternating variant order. Both feature candidates and winners compare fieldwise bit-identically before and after timing. This is a fixed-histogram split check; it makes no claim that separate atomic histogram runs or complete trained models are bit-identical.

All **36 multi-output cases** have lower batched medians, with reference/candidate ratios from 2.593 to 22.122. Their raw ranges are disjoint except the stream block256 case with T=3, B=33 (sequential 62.784–285.088 µs; batched 21.114–83.072 µs). At T=16 with warp32 and graph replay, the ratios are 10.211 for B=16 and 10.603 for B=32; these are split-operation ratios for the entire tile, not training speedups.

The single negative root-batching median is **T=1, B=33, warp32 policy, stream**: 27.680 µs batched versus 23.904 µs sequential (15.8% longer), with overlapping raw ranges. B=33 uses the owned block fallback. T=1 reduces no launch count and serves as a control; small positive ratios there are not evidence that output batching helps scalar work.

All 48 cases follow. Every shape has 16 features. Values are microseconds for the complete T-output operation. At B=33, rows labelled warp32 use the same owned block kernels.

| B | T | Policy | Launch | Sequential | Batched | Ratio |
|---:|---:|---|---|---:|---:|---:|
| 16 | 1 | block256 | stream | 21.248 [20.160, 22.368] | 20.896 [20.480, 22.879] | 1.017 |
| 16 | 1 | block256 | graph | 16.864 [16.832, 16.956] | 16.576 [16.536, 17.248] | 1.017 |
| 16 | 1 | warp32 | stream | 23.427 [21.280, 25.600] | 22.720 [21.312, 23.232] | 1.031 |
| 16 | 1 | warp32 | graph | 12.032 [12.000, 12.160] | 11.712 [11.712, 11.936] | 1.027 |
| 16 | 3 | block256 | stream | 80.857 [62.880, 114.528] | 26.560 [24.192, 33.568] | 3.044 |
| 16 | 3 | block256 | graph | 51.328 [51.136, 51.387] | 17.152 [17.120, 17.376] | 2.993 |
| 16 | 3 | warp32 | stream | 73.824 [61.952, 90.406] | 23.134 [21.600, 38.368] | 3.191 |
| 16 | 3 | warp32 | graph | 36.538 [36.512, 36.928] | 12.224 [12.223, 12.480] | 2.989 |
| 16 | 16 | block256 | stream | 379.584 [338.016, 620.896] | 52.448 [51.968, 93.019] | 7.237 |
| 16 | 16 | block256 | graph | 289.336 [276.443, 320.351] | 48.729 [48.512, 48.992] | 5.938 |
| 16 | 16 | warp32 | stream | 388.767 [342.112, 424.282] | 28.224 [24.064, 55.424] | 13.774 |
| 16 | 16 | warp32 | graph | 198.336 [197.664, 219.168] | 19.424 [19.136, 32.896] | 10.211 |
| 16 | 33 | block256 | stream | 1138.848 [796.768, 1982.042] | 95.680 [93.984, 122.976] | 11.903 |
| 16 | 33 | block256 | graph | 609.952 [580.155, 632.480] | 93.658 [90.496, 117.598] | 6.513 |
| 16 | 33 | warp32 | stream | 757.788 [711.867, 887.232] | 37.280 [33.952, 59.200] | 20.327 |
| 16 | 33 | warp32 | graph | 429.600 [426.304, 466.400] | 30.592 [30.560, 64.064] | 14.043 |
| 32 | 1 | block256 | stream | 23.232 [20.860, 27.928] | 22.492 [20.704, 36.926] | 1.033 |
| 32 | 1 | block256 | graph | 17.472 [17.472, 17.632] | 17.376 [17.152, 34.080] | 1.006 |
| 32 | 1 | warp32 | stream | 24.954 [21.735, 34.368] | 24.896 [22.656, 41.019] | 1.002 |
| 32 | 1 | warp32 | graph | 12.704 [12.640, 12.896] | 12.352 [12.351, 12.736] | 1.028 |
| 32 | 3 | block256 | stream | 70.688 [62.592, 88.160] | 27.264 [21.408, 35.040] | 2.593 |
| 32 | 3 | block256 | graph | 51.424 [51.264, 84.832] | 17.216 [17.152, 17.664] | 2.987 |
| 32 | 3 | warp32 | stream | 65.276 [61.536, 91.680] | 21.824 [19.616, 24.352] | 2.991 |
| 32 | 3 | warp32 | graph | 37.792 [37.312, 56.800] | 12.576 [12.448, 12.896] | 3.005 |
| 32 | 16 | block256 | stream | 428.800 [360.576, 458.528] | 54.973 [52.224, 64.959] | 7.800 |
| 32 | 16 | block256 | graph | 286.176 [277.376, 327.770] | 49.184 [49.119, 66.848] | 5.818 |
| 32 | 16 | warp32 | stream | 401.336 [344.320, 417.952] | 29.600 [24.480, 32.960] | 13.559 |
| 32 | 16 | warp32 | graph | 207.648 [199.960, 229.056] | 19.584 [19.488, 39.296] | 10.603 |
| 32 | 33 | block256 | stream | 811.840 [744.640, 1248.224] | 100.347 [94.270, 158.784] | 8.090 |
| 32 | 33 | block256 | graph | 590.304 [583.706, 632.224] | 91.808 [91.133, 160.480] | 6.430 |
| 32 | 33 | warp32 | stream | 763.104 [719.072, 830.656] | 34.496 [33.952, 37.952] | 22.122 |
| 32 | 33 | warp32 | graph | 437.440 [409.824, 464.736] | 31.072 [30.752, 60.352] | 14.078 |
| 33 | 1 | block256 | stream | 21.342 [20.059, 37.375] | 20.800 [19.776, 28.160] | 1.026 |
| 33 | 1 | block256 | graph | 16.896 [16.864, 34.368] | 16.576 [16.544, 16.832] | 1.019 |
| 33 | 1 | warp32 | stream | 23.904 [20.768, 38.559] | 27.680 [20.672, 36.064] | 0.864 **slower** |
| 33 | 1 | warp32 | graph | 16.928 [16.891, 24.256] | 16.672 [16.544, 33.120] | 1.015 |
| 33 | 3 | block256 | stream | 105.280 [62.784, 285.088] | 27.648 [21.114, 83.072] | 3.808 |
| 33 | 3 | block256 | graph | 52.128 [52.000, 80.224] | 17.184 [17.152, 17.376] | 3.034 |
| 33 | 3 | warp32 | stream | 66.816 [62.176, 93.120] | 21.344 [20.576, 41.216] | 3.130 |
| 33 | 3 | warp32 | graph | 52.256 [51.968, 84.672] | 17.248 [17.152, 17.664] | 3.030 |
| 33 | 16 | block256 | stream | 442.560 [352.224, 602.560] | 53.756 [52.064, 66.272] | 8.233 |
| 33 | 16 | block256 | graph | 274.970 [274.106, 336.800] | 48.928 [48.736, 87.264] | 5.620 |
| 33 | 16 | warp32 | stream | 412.000 [353.792, 982.784] | 53.888 [52.160, 84.800] | 7.645 |
| 33 | 16 | warp32 | graph | 300.160 [274.528, 310.432] | 48.887 [48.800, 49.312] | 6.140 |
| 33 | 33 | block256 | stream | 804.608 [740.160, 891.040] | 97.376 [94.752, 129.312] | 8.263 |
| 33 | 33 | block256 | graph | 605.920 [570.397, 632.220] | 93.376 [91.904, 126.304] | 6.489 |
| 33 | 33 | warp32 | stream | 849.020 [767.136, 1151.200] | 97.568 [95.296, 140.928] | 8.702 |
| 33 | 33 | warp32 | graph | 630.079 [570.398, 650.496] | 91.872 [91.424, 142.240] | 6.858 |

## Existing block-versus-warp control

The same file also retains 24 comparisons of the existing two-launch split operation over 1, 2, 16 or 64 active nodes. Each sample averages 128 operations, with seven samples after two warmup iterations. This is not the sequential-versus-batched-root comparison above.

Eight control medians favor the block reference. Four are B<=32 stream cases with one or two nodes; the largest is B=16, two nodes, 30.543 µs warp versus 23.256 µs block (31.3% longer). The other four are B=33 fallback controls, where both variants invoke the owned block implementation: one-node graph, two-node stream, two-node graph and 64-node graph. Their differences demonstrate observed timing variability rather than a 33-bin warp algorithm. All 24 cases and their spreads are retained below.

| B | Active nodes | Launch | Block256 | Warp32 policy | Ratio |
|---:|---:|---|---:|---:|---:|
| 16 | 1 | stream | 23.672 [23.080, 36.864] | 23.831 [21.472, 24.600] | 0.993 **slower** |
| 16 | 1 | graph | 19.528 [19.384, 26.464] | 14.088 [13.952, 24.032] | 1.386 |
| 16 | 2 | stream | 23.256 [22.832, 28.552] | 30.543 [22.639, 39.056] | 0.761 **slower** |
| 16 | 2 | graph | 19.120 [17.152, 22.224] | 13.632 [12.240, 17.200] | 1.403 |
| 16 | 16 | stream | 58.880 [52.368, 72.600] | 28.408 [23.016, 36.504] | 2.073 |
| 16 | 16 | graph | 48.983 [48.952, 56.752] | 19.440 [19.368, 33.544] | 2.520 |
| 16 | 64 | stream | 190.495 [178.640, 199.656] | 64.648 [63.944, 80.432] | 2.947 |
| 16 | 64 | graph | 185.128 [175.320, 195.424] | 67.088 [60.968, 77.624] | 2.759 |
| 32 | 1 | stream | 21.351 [20.376, 27.776] | 21.647 [20.600, 28.760] | 0.986 **slower** |
| 32 | 1 | graph | 17.160 [17.056, 25.752] | 12.360 [12.199, 18.736] | 1.388 |
| 32 | 2 | stream | 21.984 [20.832, 36.302] | 22.422 [19.872, 26.544] | 0.980 **slower** |
| 32 | 2 | graph | 17.376 [17.296, 24.096] | 12.496 [12.463, 12.584] | 1.390 |
| 32 | 16 | stream | 61.192 [52.624, 66.728] | 24.991 [23.208, 27.304] | 2.449 |
| 32 | 16 | graph | 52.816 [49.400, 54.480] | 19.608 [19.488, 26.720] | 2.694 |
| 32 | 64 | stream | 191.840 [173.016, 195.312] | 64.760 [64.200, 80.848] | 2.962 |
| 32 | 64 | graph | 186.144 [177.008, 189.368] | 61.344 [61.224, 71.496] | 3.034 |
| 33 | 1 | stream | 22.840 [20.376, 31.352] | 22.048 [20.704, 28.448] | 1.036 |
| 33 | 1 | graph | 17.008 [16.904, 23.752] | 17.016 [16.872, 17.312] | 1.000 **slower** |
| 33 | 2 | stream | 22.319 [21.960, 29.800] | 23.384 [21.535, 35.608] | 0.954 **slower** |
| 33 | 2 | graph | 18.008 [17.896, 20.111] | 18.032 [17.968, 25.240] | 0.999 **slower** |
| 33 | 16 | stream | 56.136 [53.064, 59.616] | 55.640 [52.904, 68.248] | 1.009 |
| 33 | 16 | graph | 52.656 [49.208, 65.112] | 51.912 [49.328, 63.632] | 1.014 |
| 33 | 64 | stream | 193.704 [178.936, 195.808] | 188.576 [179.359, 193.304] | 1.027 |
| 33 | 64 | graph | 181.088 [175.392, 188.168] | 187.192 [176.432, 189.000] | 0.967 **slower** |

## Scope of the conclusion

These observations support retaining root count reuse and root split batching as independently selectable candidates for the complete-training experiments. They do not justify changing scalar defaults, inferring a break-even round count from separately timed components, or claiming quality preservation. The larger deeper-histogram experiment has its own evidence. Complete-operation and training timings, independent quality gates and Nsight explanations must remain distinct.
