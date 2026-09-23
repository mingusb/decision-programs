# Combined native-path preservation evidence

Performance preservation remains unresolved for the stream case and the single-bin graph case. Both completed cohorts are retained.

Audited 16 configurations, 144 invocations, 3024 timing samples, and 72 adjacent old/current pairs. There are 36 slower pairs, 25 faster pairs, and 11 ties; 14 pairs are at least 5% slower.

Ratios are current invocation median / old invocation median.

| Case | Pairs | Median ratio | Range | Slower | ≥5% slower |
|---|---:|---:|---|---:|---:|
| small8 | 4 | 1.000018 | 0.897849–1.255952 | 2 | 1 |
| smallbyte | 4 | 1.000000 | 1.000000–1.000000 | 0 | 0 |
| cachedbyte | 4 | 0.982014 | 0.894040–1.118519 | 1 | 1 |
| byte | 4 | 1.000964 | 0.996159–1.009044 | 2 | 0 |
| hot99 | 4 | 1.002924 | 0.994186–1.060302 | 2 | 1 |
| sortedhot99 | 4 | 0.997059 | 0.982558–1.005917 | 1 | 0 |
| single | 8 | 1.035468 | 1.000000–1.260355 | 5 | 4 |
| large4096 | 4 | 1.002603 | 0.991569–1.081184 | 3 | 1 |
| large4096-u64 | 4 | 0.998544 | 0.945889–1.001208 | 2 | 0 |
| large8192-u64 | 4 | 1.000429 | 0.942728–1.007171 | 2 | 0 |
| large16384-u64 | 4 | 0.999917 | 0.933914–1.001337 | 2 | 0 |
| cold4096-u64 | 4 | 0.997071 | 0.994083–1.008863 | 1 | 0 |
| stream4096 | 8 | 1.149728 | 0.890244–2.138889 | 7 | 6 |
| native-boundary | 4 | 1.004080 | 0.999231–1.023239 | 2 | 0 |
| native-medium | 4 | 0.998645 | 0.997191–1.013932 | 1 | 0 |
| native-million | 4 | 1.000222 | 0.998873–1.000916 | 3 | 0 |

## Two-case repeat

| Case | Initial four ratios | Followup four ratios |
|---|---|---|
| single | 1.260355, 1.000000, 1.260355, 1.059172 | 1.011765, 1.123529, 1.000000, 1.000000 |
| stream4096 | 1.046032, 1.300000, 1.083770, 1.065037 | 1.215686, 0.890244, 1.307283, 2.138889 |

`single`: the followup median ratio fell to 1.005882; its first round crosses recorded SM-clock transitions, and its second round ties at 5.440 µs. The combined median ratio is 1.035468. This does not erase the initial losses, observed with matching endpoint clocks.

`stream4096`: the followup median ratio is 1.261484 and the combined median ratio is 1.149728. One followup pair favors current, while the other three lose. The final current invocation has a 39.424 µs median versus 18.432 µs for adjacent old; 20 of its 21 samples are approximately 33–53 µs. Both endpoint snapshots report 1455 MHz SM and 6001 MHz memory. The broad slowdown is not one isolated outlier.

The explicit variants are unchanged. Added host dispatch executes during graph capture, outside graph timing; stream submission still runs the changed host path. Matched Nsight Systems timelines can distinguish GPU execution from idle gaps and API delays, but profiled latencies must not replace the unprofiled preservation results.

## Limits

- The followup repeats the original seeds and protocol for two flagged workloads; it supplies repeatability evidence, not fresh input seeds or an independent randomized full campaign.
- Ratios compare adjacent invocation medians. Do not treat the 21 raw samples as 21 independent old/new experiments.
- Only 16 explicit configurations were covered; automatic selection was not exercised by this preservation campaign.
- The 5 percent flag is a diagnostic threshold, not an acceptance margin. Smaller losses remain visible.
- Endpoint GPU telemetry is insufficient to establish stable clocks during each sample or to isolate host, driver, GPU, and external interference.
- The current executable remains slower in 7 of 8 stream comparisons and 5 of 8 single-bin graph comparisons (three graph ties). Performance preservation is unresolved.
- Unchanged native GPU machine code does not establish unchanged execution time; host dispatch and executable layout changed.
- These results neither establish statistical significance nor prove a fixed penalty or universal regression.

Raw CSV, logs, commands, hashes, and telemetry are preserved. Analysis performed no GPU activity or production edits.
