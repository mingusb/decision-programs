# Fixed-policy extension checks

All 15 workloads passed CPU-reference correctness and the recorded-data audit: 30 invocations, 144 candidate measurements, 1,584 raw timing samples. Policies were frozen from the uniform N=16,777,216 experiment before these measurements; no retuning or automatic-default promotion occurred.

Ratios below are comparator time / selected time, measured together in each invocation. Values below 1 mean the frozen selected policy lost. Ranges span two seed medians, not confidence intervals. Clocks were unlocked. `hot99` means 99% of input values land in one bin.

| N | Bins | Distribution/order | Cache/launch | Selected time (µs) | Native/selected | Warp native/selected |
|---:|---:|---|---|---:|---:|---:|
| 16,777,233 | 24,577 | uniform/shuffled | warm/graph | 195.840–196.352 | 2.886–3.725 | — |
| 16,777,233 | 32,768 | uniform/shuffled | warm/graph | 220.928–222.464 | 2.581–2.625 | — |
| 16,777,233 | 1,048,576 | uniform/shuffled | warm/graph | 2378.496–2505.984 | 1.800–1.902 | — |
| 268,435,456 | 24,577 | uniform/shuffled | warm/graph | 3087.360–3087.360 | 3.233–3.264 | — |
| 268,435,456 | 1,048,576 | uniform/shuffled | warm/graph | 38362.881–38424.065 | 1.878–1.878 | — |
| 1,048,593 | 24,577 | hot99/shuffled | warm/graph | 679.936–681.472 | 0.992–0.996 | 0.040–0.040 |
| 1,048,593 | 24,577 | hot99/sorted | warm/graph | 679.680–679.936 | 0.995–0.995 | 0.040–0.040 |
| 1,048,593 | 32,768 | hot99/shuffled | warm/graph | 679.168–680.192 | 0.997–0.998 | 0.040–0.040 |
| 1,048,593 | 32,768 | hot99/sorted | warm/graph | 679.680–680.448 | 0.995–0.996 | 0.040–0.040 |
| 1,048,593 | 1,048,576 | hot99/shuffled | warm/graph | 755.968–757.760 | 0.974–0.976 | 0.109–0.111 |
| 1,048,593 | 1,048,576 | hot99/sorted | warm/graph | 721.920–722.944 | 0.987–0.992 | 0.090–0.103 |
| 16,777,216 | 24,577 | uniform/shuffled | cold/graph | 202.752–203.264 | 2.497–2.499 | — |
| 16,777,216 | 24,577 | uniform/shuffled | warm/stream | 199.936–200.960 | 2.816–3.439 | — |
| 16,777,216 | 1,048,576 | uniform/shuffled | cold/graph | 2502.144–2503.168 | 1.808–1.810 | — |
| 16,777,216 | 1,048,576 | uniform/shuffled | warm/stream | 2390.528–2390.784 | 1.892–1.893 | — |

The uniform gains extend to the measured tails, 256-million-element inputs, cold-cache graphs and warm direct-stream launches. The uniform-selected policies lose badly on the tested skewed input: roughly 9–25× slower than the fixed existing warp-aggregation control. These measurements prohibit treating the new paths as universal replacements.

## Frozen configurations

| Bins | Selected policy | Native comparator |
|---:|---|---|
| 24,577 | `shared_overflow:15:24:u32:kernel` | `global:0:1536:native:kernel` |
| 32,768 | `shared_overflow:15:24:u32:kernel` | `global:2:1536:native:kernel` |
| 1,048,576 | `global:0:48:u32:kernel` | `global:0:48:native:kernel` |

The complete [audit JSON](extension-analysis.json) records all candidates, losses, frozen configurations, per-seed medians, raw samples, and artifact hashes. [Manifest](manifest.json). Reproduce the CPU-only audit with `python3 results/a5000-large-bins/run_extensions.py --audit`.
