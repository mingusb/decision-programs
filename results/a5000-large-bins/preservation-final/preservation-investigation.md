# Native-path preservation investigation

The strict audit passed, but performance preservation is unresolved for `single` and `stream4096`.

Audited 16 cases, 128 invocations, 64 adjacent pairs, and 2688 raw timing samples. Current execution was slower in 31 pairs, faster in 24, and equal in 9; 10 pairs were at least 5% slower.

Ratios below are current median / old median. All outcomes are retained.

| Case | Median ratio | Four paired ratios | Slower pairs | ≥5% slower |
|---|---:|---|---:|---:|
| small8 | 1.000018 | 0.897849, 1.005988, 0.994048, 1.255952 | 2/4 | 1 |
| smallbyte | 1.000000 | 1.000000, 1.000000, 1.000000, 1.000000 | 0/4 | 0 |
| cachedbyte | 0.982014 | 0.894040, 1.000000, 1.118519, 0.964029 | 1/4 | 1 |
| byte | 1.000964 | 0.996159, 1.009044, 1.001928, 1.000000 | 2/4 | 0 |
| hot99 | 1.002924 | 1.060302, 1.005848, 0.994186, 1.000000 | 2/4 | 1 |
| sortedhot99 | 0.997059 | 0.982558, 0.994118, 1.000000, 1.005917 | 1/4 | 0 |
| single | 1.159763 | 1.260355, 1.000000, 1.260355, 1.059172 | 3/4 | 3 |
| large4096 | 1.002603 | 1.001386, 1.003820, 0.991569, 1.081184 | 3/4 | 1 |
| large4096-u64 | 0.998544 | 0.945889, 1.001208, 1.001034, 0.996054 | 2/4 | 0 |
| large8192-u64 | 1.000429 | 0.942728, 1.001199, 0.999659, 1.007171 | 2/4 | 0 |
| large16384-u64 | 0.999917 | 1.001003, 0.933914, 0.998832, 1.001337 | 2/4 | 0 |
| cold4096-u64 | 0.997071 | 0.998540, 0.994083, 0.995601, 1.008863 | 1/4 | 0 |
| stream4096 | 1.074403 | 1.046032, 1.300000, 1.083770, 1.065037 | 4/4 | 3 |
| native-boundary | 1.004080 | 1.008878, 0.999231, 1.023239, 0.999282 | 2/4 | 0 |
| native-medium | 0.998645 | 1.013932, 0.998160, 0.999130, 0.997191 | 1/4 | 0 |
| native-million | 1.000222 | 1.000916, 1.000430, 1.000014, 0.998873 | 3/4 | 0 |

## Repeated losses

`single`: old/current/current/old medians were 5.408/6.816/5.440/5.440 µs; current/old/old/current medians were 6.816/5.408/5.408/5.728 µs. Every current invocation still reached 5.408 µs. Its current invocations contained 14, 0, 17, and 7 samples above the largest old sample (5.760 µs), so the large losses involve many elevated samples.

`stream4096`: paired slowdowns were 4.60%, 30.00%, 8.38%, and 6.50%. All four pairs lose. Matching endpoint clocks in the largest-loss pair and both second-round pairs prevent a simple recorded-clock explanation.

## Interpretation

- Audit passed all commands, binary hashes, GPU identities, workload/config metadata, raw summary values, and artifact completeness. No parsing or pairing error was found.
- Ratios compare temporally adjacent invocation medians; the 21 samples within an invocation are not independent binary-level replications. No statistical-significance test is claimed.
- Single and stream4096 show losses in both ABBA and BAAB rounds. A simple execution-order explanation is insufficient.
- Single uses an explicit shared:6:192:native:kernel graph configuration; both seeds generate the same input of all bin 255. Old medians range 5.408–5.440us, current 5.440–6.816us. Current reaches 5.408us in every invocation but elevated samples occur repeatedly.
- Single telemetry endpoints all report 1635 MHz SM, 6001 MHz memory, 62–63°C. Those snapshots do not explain its repeated losses and cannot exclude unobserved within-run interference.
- Stream4096 uses explicit shared:10:48:native:runtime. All 4 pairs lose, including both round 2 pairs with matching 1455 MHz endpoint clocks. Both executables have broad sample variation and long tails.
- Neither measurement resolves automatic defaults. src/defaults.cpp exactly matches the archived old source.
- Graph capture calls gh::histogram and supported before timing. Their added CPU branches cannot directly add per-operation time inside the captured graph interval. Kernel-address/driver/graph execution effects and external interference remain unresolved.
- Stream timing surrounds submissions, so changed host validation/dispatch and runtime submission gaps can affect event intervals. Assembly confirms changes but does not establish that they caused the observed penalty.
- The median ratios do not represent a constant slowdown: single and stream penalties vary substantially across invocations. No loss is discarded as acceptable merely because it is below 5 percent.
- Preservation remains unresolved. A bounded same-protocol repeat of single and stream4096 in a new output root is justified; combine evidence without replacing the original cohort.

No GPU activity or production edits were performed during this analysis. Raw artifacts were preserved.
