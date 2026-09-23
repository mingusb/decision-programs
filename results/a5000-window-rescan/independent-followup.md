# Independent follow-up audit

**PASS:** 40 processes, 112 candidate measurements, 1,392 raw timing samples, and 3,456 timed histogram operations. Frozen sources/binary, all 40 receipts and 120 linked artifacts, exact measurement-file sets, command identities, CSV summaries, and ratios passed independent checks. Both existing audit commands also passed. This audit used no GPU calls.

## Window size: two scans versus four

All eight processes use 16,777,216 uniform shuffled u32 inputs, 1,048,576 bins, u64 output, policy 0, 48 blocks, warm graph mode, and batch four. Each invocation contains the original narrow baseline and one window candidate. A 524,288-bin window is 2 MiB and needs two scans; a 262,144-bin window is 1 MiB and needs four scans.

| Process | Seed | Window / scans | Narrow median (us) | Window median (us) | Narrow / window |
|---:|---:|---|---:|---:|---:|
| 0 | 2026092291 | 524,288 / 2 | 2559.744 | 1081.088 | 2.367748x |
| 1 | 2026092291 | 262,144 / 4 | 2505.984 | 1672.192 | 1.498622x |
| 2 | 2026092291 | 262,144 / 4 | 2608.896 | 1671.424 | 1.560882x |
| 3 | 2026092291 | 524,288 / 2 | 2533.888 | 1083.904 | 2.337742x |
| 4 | 2026092292 | 262,144 / 4 | 2488.576 | 1569.536 | 1.585549x |
| 5 | 2026092292 | 524,288 / 2 | 2504.448 | 1082.624 | 2.313313x |
| 6 | 2026092292 | 524,288 / 2 | 2606.080 | 1182.720 | 2.203463x |
| 7 | 2026092292 | 262,144 / 4 | 2492.160 | 1577.728 | 1.579588x |

The two-scan candidate is faster in this comparison. For each seed, take the geometric mean of the two process medians for each window; alternatively normalize each candidate median by its same-process narrow baseline before taking that mean. The following ratios are **four-scan time / two-scan time**, so values above one favor two scans.

| Seed | Raw time ratio | Ratio after baseline normalization |
|---:|---:|---:|
| 2026092291 | 1.544402 | 1.538278 |
| 2026092292 | 1.390663 | 1.426620 |

The 2 MiB/two-scan candidate is faster than the 1 MiB/four-scan candidate in this follow-up. Each size is a separate process with a narrow baseline; normalization is descriptive and cannot remove all environment effects. Four scans take 42.7% and 53.8% more normalized time in the two seed blocks. The smaller window still beats the original narrow baseline. No optimality outside these settings is established.

Eight processes: four per seed, two per window per seed. The manifest inherited a generic limitation saying "Two fresh seeds/processes"; that is ambiguous here. There are two distinct seeds and eight total processes. ABBA is only used with the first seed and BAAB only with the second, so seed and process-order pattern are confounded. There is no same-process direct comparison of the two window sizes.

## Skew and dominant-value location

Each row below is a separate workload, with two seeds/processes. Timing pairs are ordered by seed 2026092271, then 2026092272. The baseline is shared-overflow at 24,577/32,768 bins and plain global narrow at 1,048,576 bins. Both warp candidates use policy 4 and 96 blocks; native uses u64 output atomics, while narrow uses u32 scratch followed by widening. All times are microseconds.

| Case | Bins | Dominant ID | Order | Baseline medians | Native-warp medians | Narrow-warp medians | Baseline / native-warp range |
|---:|---:|---:|---|---|---|---|---|
| 0 | 24,577 | 24,575 | shuffled | 16.896 / 15.872 | 31.744 / 28.160 | 33.280 / 29.696 | 0.532258–0.563636x |
| 1 | 24,577 | 24,575 | sorted | 15.360 / 16.384 | 28.160 / 31.232 | 29.696 / 33.280 | 0.524590–0.545455x |
| 2 | 24,577 | 24,576 | shuffled | 681.472 / 680.960 | 28.160 / 28.160 | 30.208 / 30.208 | 24.181818–24.200000x |
| 3 | 24,577 | 24,576 | sorted | 679.936 / 679.936 | 28.160 / 28.672 | 29.696 / 29.696 | 23.714285–24.145454x |
| 4 | 32,768 | 24,575 | shuffled | 17.920 / 25.600 | 32.256 / 28.672 | 33.792 / 29.696 | 0.555556–0.892857x |
| 5 | 32,768 | 24,575 | sorted | 16.896 / 45.632 | 28.672 / 29.184 | 30.208 / 30.720 | 0.589286–1.563596x |
| 6 | 32,768 | 24,576 | shuffled | 681.472 / 683.008 | 28.672 / 28.160 | 29.696 / 29.696 | 23.767857–24.254546x |
| 7 | 32,768 | 24,576 | sorted | 680.448 / 680.960 | 29.696 / 29.184 | 31.232 / 30.208 | 22.913792–23.333333x |
| 8 | 32,768 | 32,767 | shuffled | 682.496 / 680.448 | 28.160 / 28.672 | 29.696 / 29.696 | 23.732143–24.236364x |
| 9 | 32,768 | 32,767 | sorted | 682.496 / 682.496 | 29.184 / 28.672 | 30.208 / 29.696 | 23.385964–23.803572x |
| 10 | 1,048,576 | 24,575 | shuffled | 973.312 / 747.008 | 90.112 / 93.184 | 102.912 / 97.792 | 8.016484–10.801136x |
| 11 | 1,048,576 | 24,575 | sorted | 750.592 / 741.376 | 60.416 / 62.976 | 83.968 / 80.896 | 11.772357–12.423728x |
| 12 | 1,048,576 | 24,576 | shuffled | 753.152 / 753.152 | 89.600 / 88.576 | 102.912 / 105.472 | 8.405715–8.502891x |
| 13 | 1,048,576 | 24,576 | sorted | 747.008 / 746.496 | 59.392 / 59.392 | 82.432 / 83.968 | 12.568966–12.577586x |
| 14 | 1,048,576 | 1,048,575 | shuffled | 757.760 / 754.688 | 89.088 / 88.576 | 106.496 / 103.424 | 8.505747–8.520232x |
| 15 | 1,048,576 | 1,048,575 | sorted | 812.544 / 723.968 | 71.680 / 69.632 | 84.480 / 82.944 | 10.397059–11.335714x |

The hot-bin location is essential to performance. At 24577/32768 bins, moving the dominant ID from 24575 to 24576 changes the selected shared-overflow counting route and is associated with large runtime increases. Native warp wins outside the prefix, while shared-overflow wins most tested in-prefix invocations, with one winner reversal at 32768 bins and sorted input. Native warp beats the tested u32-scratch warp variant in all 32 invocations, but is not a universal histogram winner.

At 24,577 bins, moving the dominant ID from 24,575 (the last shared bin) to 24,576 (the first global bin) moves shared-overflow medians from roughly 15–17 us to roughly 680 us. At 32,768 bins, the out-of-prefix medians likewise stay around 680–683 us, while the in-prefix results vary from 16.896 to 45.632001 us. These are separate location-specific workloads and processes.

**Keep the exception:** case 5 (32,768 bins, dominant ID 24,575, sorted) favors shared-overflow at one seed and native warp at the other. Its shared-overflow median is 16.896 us in one process and 45.632001 us in the other. The audit retains both observations and does not assign a cause.

These are different workload locations and existing algorithm alternatives, not old/new versions of one implementation. The differences do not establish a newly introduced timing regression. No global_window candidate or known-hot specialization is tested in the skew campaign.

## Limits

- This report independently audits only eight window-size processes and 32 skew processes. The original 20-process audit is unchanged; Nsight profiles and timing-harness preservation experiments are excluded.
- All ratios are descriptive, based on per-process medians. Raw batches are not treated as independent process replicas; no significance test, confidence interval, zero-regression claim, or zero-overhead claim is made.
- Window-size comparisons are between separate processes, each normalized to its own narrow baseline. Seed and ABBA/BAAB order are coupled, and normalization does not prove environmental equivalence.
- The fixed scalar policy/grid and two tested window sizes establish a useful measured choice for the tested million-bin uniform workload, not a globally optimal parameterization or a universal default.
- Skew tests use 1048593 u32 values, u64 counts, warm graph execution, and hot99@BIN: 99% forced-value probability plus a uniform background that may also hit that value. Actual hot counts are random; locations preserve the historical generator sequence for a given seed and bin count.
- Existing shared-overflow, plain narrow, and warp kernels have location/order-dependent tradeoffs. One sorted in-prefix case reverses the observed winner across seeds. No observations were removed as outliers.
- No new known-hot kernel or adaptive selector is evaluated. No global_window skew comparison is present, so uniform window gains cannot be extended to skew from these data.
- Device operation times include required clearing/counting/finalization, but exclude allocation, graph setup, preparation, warmup, and cache eviction. Full application overhead and preservation of every existing path are separate questions.
- No NVIDIA reference was requested in either follow-up. Historical NVIDIA measurements are not combined with these ratios.
- One A5000 Laptop GPU under WSL2/Windows driver 597.06 with unlocked clocks was used. Recorded identity and telemetry cannot rule out other work or prove why any timing changed.
- Both manifests record production_default_promotion=false. The evidence supports retaining separate candidates and workload-sensitive evaluation, not a universal skew winner or a timing-regression verdict.

## Evidence

- [Machine-readable independent follow-up](independent-followup.json), [window-size manifest](window-sizes/manifest.json), [window-size analysis](window-sizes/analysis.json), and [skew analysis](../a5000-window-skew/analysis.json).
- Original [independent audit](independent-audit.md) and its JSON remain byte-identical.
- Benchmark SHA256: `79bb396fdc50760eed6394c3ca78fe297f79934d9a0df7a45f22b2404d96b374`.
