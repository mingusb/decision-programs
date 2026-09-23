# Independent interpretation of the synchronization screen

Independently verified all 64 process CSV files, 8,192 raw positions, 2,048 quartet ratios, and published process geometric means. Artifact hashes match the frozen receipts. Raw results agree with the published analysis to relative tolerance 1e-12. No runner or analyzer code was imported for this recomputation.

**Moving synchronization to quartet boundaries did not solve timing variation across both workloads, and increasing the batch size did not establish a general stability plateau.**

For the single-valued graph workload, quartet mode at batch 32 is the strongest candidate for a separately declared confirmation. In old/old, the two process AA means are 0.996451 and 1.000560; in new/new, 1.000004 and 0.994480. Within each of those processes, the 90th-percentile symmetric quartet deviation is about 0.87–1.47%. This does not mean all quartets are that close: one new/new quartet is 0.850126 (about 15% below one). Larger batches bring back substantial tails in both modes.

For the uniform stream workload, quartet mode at batch 256 reduces the 90th-percentile symmetric quartet deviation in every matched repetition: old/old 48.68%→25.83% and 45.77%→22.06%; new/new 42.02%→27.09% and 49.39%→25.87%. Those errors remain much too large for a small-regression claim. Its old/old process AA means are 1.006905 and 0.972848; new/new means are 0.962212 and 1.021001. Other batches show inconsistent mode effects, including an old/old batch-64 quartet-mode process with a 2.271× event-latency ratio and a 3.751× host-duration ratio relative to its matched position-mode process.

A universal batch choice is therefore unsupported. The graph batch-32/quartet and stream batch-256/quartet configurations are screening candidates with different remaining uncertainty, not established improvements or a production-default decision. The next measurement diagnostic should classify long gaps with a scheduling trace and observed clocks/host submission behavior; this screen alone cannot attribute them to Windows scheduling, WSL, clocks, or the histogram kernel.

## Matched mode effects

Ratios are quartet mode / position mode for adjacent process pairs with matching case, AA binding, batch, input seed and quartet-order seed. Each cell lists repetition 0 and repetition 1 separately. Below one means a smaller measured duration. No values are pooled across workloads or bindings.

| Case | AA binding | Batch | Event latency ratios, r0 / r1 | Host quartet duration ratios, r0 / r1 |
|---|---|---:|---:|---:|
| single | old-old | 32 | 0.998136 / 0.798375 | 0.916075 / 0.501580 |
| single | old-old | 64 | 1.005926 / 0.977590 | 0.985404 / 0.925579 |
| single | old-old | 128 | 1.009918 / 0.995890 | 0.954659 / 0.943836 |
| single | old-old | 256 | 1.031447 / 1.032242 | 1.000996 / 0.973036 |
| single | new-new | 32 | 0.982984 / 0.856981 | 0.914685 / 0.830099 |
| single | new-new | 64 | 0.985710 / 0.973129 | 0.902412 / 0.897782 |
| single | new-new | 128 | 1.033800 / 0.995662 | 1.009066 / 0.937637 |
| single | new-new | 256 | 0.970172 / 0.994186 | 0.884717 / 0.964769 |
| stream4096 | old-old | 32 | 0.909468 / 1.233275 | 0.867069 / 1.206688 |
| stream4096 | old-old | 64 | 2.271431 / 0.997803 | 3.751392 / 1.017684 |
| stream4096 | old-old | 128 | 0.936938 / 1.333747 | 0.889684 / 1.387999 |
| stream4096 | old-old | 256 | 1.055246 / 0.911542 | 1.035413 / 0.891115 |
| stream4096 | new-new | 32 | 1.055463 / 1.128745 | 1.030189 / 1.451987 |
| stream4096 | new-new | 64 | 0.903353 / 0.938944 | 0.858366 / 0.898770 |
| stream4096 | new-new | 128 | 0.853117 / 1.056713 | 0.823418 / 1.007633 |
| stream4096 | new-new | 256 | 0.989590 / 0.875656 | 0.977393 / 0.840600 |

## Within-process tail check

The table preserves both process repetitions. Symmetric deviation is `100 × (max(B/A, A/B) − 1)`; p90 uses the nearest rank among the 32 retained quartets in each process. This diagnostic exposes tails that near-one process averages can hide. It is not a confidence interval.

| Case | AA binding | Batch | Sync mode | AA process mean, r0 / r1 | p90 symmetric quartet deviation %, r0 / r1 |
|---|---|---:|---|---:|---:|
| single | old-old | 32 | position | 1.000457 / 1.000787 | 1.163 / 39.958 |
| single | old-old | 32 | quartet | 0.996451 / 1.000560 | 1.469 / 1.460 |
| single | old-old | 64 | position | 1.006617 / 1.037621 | 56.806 / 59.225 |
| single | old-old | 64 | quartet | 1.000341 / 0.936811 | 59.197 / 58.430 |
| single | old-old | 128 | position | 0.996741 / 1.008763 | 41.472 / 34.945 |
| single | old-old | 128 | quartet | 0.963787 / 0.964524 | 45.571 / 33.954 |
| single | old-old | 256 | position | 0.988564 / 1.017234 | 37.321 / 24.265 |
| single | old-old | 256 | quartet | 0.971787 / 1.015850 | 33.300 / 33.918 |
| single | new-new | 32 | position | 1.034576 / 1.000288 | 2.050 / 77.613 |
| single | new-new | 32 | quartet | 1.000004 / 0.994480 | 1.177 / 0.873 |
| single | new-new | 64 | position | 1.047244 / 0.999611 | 57.197 / 47.196 |
| single | new-new | 64 | quartet | 0.982380 / 0.946399 | 57.927 / 59.502 |
| single | new-new | 128 | position | 0.977047 / 0.985605 | 40.849 / 41.525 |
| single | new-new | 128 | quartet | 0.958302 / 0.994316 | 39.105 / 41.627 |
| single | new-new | 256 | position | 0.997896 / 0.950305 | 40.508 / 29.486 |
| single | new-new | 256 | quartet | 1.015753 / 0.977429 | 32.442 / 32.882 |
| stream4096 | old-old | 32 | position | 1.011022 / 0.956421 | 73.015 / 42.549 |
| stream4096 | old-old | 32 | quartet | 1.017675 / 0.993708 | 33.345 / 46.081 |
| stream4096 | old-old | 64 | position | 0.975046 / 0.915095 | 36.054 / 86.175 |
| stream4096 | old-old | 64 | quartet | 1.128255 / 1.064134 | 231.430 / 107.284 |
| stream4096 | old-old | 128 | position | 0.924697 / 0.964134 | 65.690 / 43.427 |
| stream4096 | old-old | 128 | quartet | 0.972059 / 1.046207 | 44.187 / 88.854 |
| stream4096 | old-old | 256 | position | 0.986429 / 1.028137 | 48.680 / 45.770 |
| stream4096 | old-old | 256 | quartet | 1.006905 / 0.972848 | 25.828 / 22.065 |
| stream4096 | new-new | 32 | position | 1.042965 / 0.965851 | 43.224 / 78.030 |
| stream4096 | new-new | 32 | quartet | 0.997034 / 0.884021 | 40.934 / 250.414 |
| stream4096 | new-new | 64 | position | 1.014600 / 0.944307 | 72.302 / 67.944 |
| stream4096 | new-new | 64 | quartet | 1.021590 / 0.986635 | 29.184 / 43.160 |
| stream4096 | new-new | 128 | position | 1.021240 / 0.910812 | 42.690 / 118.155 |
| stream4096 | new-new | 128 | quartet | 1.018054 / 1.002781 | 28.707 / 90.196 |
| stream4096 | new-new | 256 | position | 1.058160 / 1.022684 | 42.021 / 49.391 |
| stream4096 | new-new | 256 | quartet | 0.962212 / 1.021001 | 27.093 / 25.875 |

## Interpretation limits

- No pooling across cases, AA bindings, batches, or synchronization modes. Each stratum has only two process repetitions.
- AA ratios compare slots bound to the same archived implementation; they do not measure an old/new performance effect.
- Mode pairs use separate adjacent processes. Reversed order reduces a simple order confound but does not isolate scheduler, clock, thermal, or host-enqueue mechanisms.
- Event timing excludes output copies but can include GPU scheduling gaps or stream submission starvation. The screen has no scheduling trace to classify them.
- Quartet mode still copies every output to pinned host memory. The factor changes host synchronization/checking placement, not the existence of copies.
- Graph event warmup is one whole preceding untimed graph batch; stream warmup is one operation. Absolute host durations across different cases are not directly comparable.
- Descriptive screening candidates are not evidence of a confirmed plateau, equivalence, or zero performance loss.

`independent-audit.json` retains every independently recomputed quartet ratio, process statistic, matched mode pair, input CSV hash and receipt hash. Historical raw measurements and frozen artifacts were not changed.
