# Matched preservation investigation

All 136 invocations passed artifact, command, binary, workload, environment, and raw-summary validation. Both cohorts and all 2,856 raw timing samples are retained. Performance acceptance remains open: the 8192-bin case retains a small positive paired bias and invocation variation is substantial.

Ratios are current / old invocation medians; values above 1 mean slower current. Follow-up cases were selected after inspecting the initial results, so combined medians are descriptive.

| Case | Initial median | Follow-up median | Combined median | Combined range | Slower pairs | >=1.05 flags |
|---|---:|---:|---:|---|---:|---:|
| small8 | 0.905797 | — | 0.905797 | 0.800948–1.005952 | 1/4 | 0 |
| smallbyte | 0.455279 | — | 0.455279 | 0.382353–1.000000 | 0/4 | 0 |
| cachedbyte | 0.996350 | 1.003704 | 1.000000 | 0.754190–1.451852 | 3/8 | 2 |
| byte | 1.001281 | — | 1.001281 | 0.997436–1.003854 | 2/4 | 0 |
| hot99 | 1.000000 | — | 1.000000 | 0.994186–1.017544 | 1/4 | 0 |
| sortedhot99 | 1.000000 | — | 1.000000 | 0.982558–1.000000 | 0/4 | 0 |
| single | 0.994152 | — | 0.994152 | 0.850000–1.000000 | 0/4 | 0 |
| large4096 | 1.000173 | — | 1.000173 | 0.999653–1.013589 | 2/4 | 0 |
| large4096-u64 | 0.998622 | 1.005183 | 0.998622 | 0.925474–1.070160 | 3/8 | 1 |
| large8192-u64 | 1.025344 | 0.999587 | 1.004890 | 0.934852–1.062809 | 6/8 | 1 |
| large16384-u64 | 0.998751 | — | 0.998751 | 0.993194–1.005165 | 2/4 | 0 |
| cold4096-u64 | 1.002216 | — | 1.002216 | 0.980994–1.007407 | 2/4 | 0 |
| stream4096 | 0.960630 | 0.978602 | 0.978602 | 0.404250–1.063752 | 3/8 | 1 |

## Retained flags

| Cohort | Case | Round / positions | Order | Old µs | Current µs | Ratio |
|---|---|---|---|---:|---:|---:|
| initial | cachedbyte | 1 / 3/4 | current→old | 4.320000 | 6.272000 | 1.451852 |
| followup | cachedbyte | 1 / 1/2 | old→current | 4.320000 | 5.632000 | 1.303704 |
| initial | large4096-u64 | 2 / 1/2 | current→old | 185.632005 | 198.655993 | 1.070160 |
| initial | large8192-u64 | 2 / 1/2 | current→old | 187.999994 | 199.808002 | 1.062809 |
| initial | stream4096 | 1 / 3/4 | current→old | 17.568000 | 18.688001 | 1.063752 |

## Invocation-order association

| Cohort | Adjacent order | Pairs | Median ratio | Slower pairs | >=1.05 flags |
|---|---|---:|---:|---:|---:|
| initial | old then current | 26 | 0.999312 | 7 | 0 |
| initial | current then old | 26 | 1.000000 | 11 | 4 |
| followup | old then current | 8 | 0.996150 | 3 | 1 |
| followup | current then old | 8 | 1.002403 | 4 | 0 |

## Findings

- Initial campaign: 104 invocations, 52 pairs, four >=1.05 flags; follow-up: 32 invocations, 16 pairs, one >=1.05 flag. Combined: 136 invocations, 68 pairs, 2856 raw samples, five flagged pairs retained.
- cachedbyte has elevated current invocation medians in both cohorts (6.272 and 5.632 us versus 4.320 us adjacent old), and an elevated old median in follow-up (5.728 us versus 4.320 us current). Its combined paired median is exactly one; its broad pair range remains visible.
- large8192-u64 initial paired median ratio is 1.025344, follow-up 0.999587, combined 1.004890. Follow-up includes an elevated old invocation median of 200.895995 us versus 187.808007 us current. Both binaries exhibit lower and upper raw timing bands; the median depends on band occupancy.
- large8192-u64 telemetry snapshots were uniformly 1635 MHz SM / 6001 MHz memory in both cohorts, with 63-65 C initial and 60-61 C follow-up. Those snapshots do not explain its per-invocation differences.
- large4096-u64 follow-up has both a slower current pair (+4.8425%) and a slower old invocation (current/old 0.925474). The combined paired median ratio is 0.998622, while the original +7.016% pair is retained.
- stream4096 pair range is very wide, including initial old invocation median 44.803999 us. Follow-up paired median is 0.978602, with three of four pairs favoring current. Stream snapshots change from 1635 to 1455 MHz during each cohort; a stable clock-causal explanation is not established.
- Apparent smallbyte and small8 gains should not be presented as implementation improvements: some old invocations were elevated, while the best old medians match current closely or exactly.

The selected 8192-bin counting kernel and u64 clear kernel have identical old/current instruction encodings and resource records in the archived SASS:

| Kernel | Encoding words | Registers | Encoding SHA256 |
|---|---:|---:|---|
| u64 clear | 80 | REG:8 | 82b3c62f7eac08645f9d18233f8ae401cd39e9b6685ad62c2846702876e64874 |
| shared15 u32-local / u64-output | 608 | REG:32 | bd51d6f7ecef9975309a7865b4e9f725ced4069d9b6be80ab0da2495edac7bdc |

## Limits

- Ratios are current invocation median divided by adjacent old invocation median; values above one indicate a slower current invocation. Raw timing samples are not paired or treated as independent replicates.
- Follow-up cases were selected because initial measurements were suspicious. Combined per-case medians are descriptive, with eight pairs for four cases and four pairs for the others; there is no aggregate speedup or statistical-equivalence claim.
- Every measured invocation and raw sample is retained. The 1.05 flag is an investigation threshold, not an acceptable-regression threshold; smaller differences remain reported.
- Seed, round/time, and outer invocation order are confounded. Pair orientation exists within both rounds. The observed order association does not establish a causal explanation.
- Before/after telemetry snapshots do not capture every timed batch and cannot rule out transient clock, scheduling, or other effects. No cause for the observed variation has been established.
- Matching device instructions and resources rules out a change to those selected kernel instructions; it does not prove complete runtime equivalence, zero regressions, or fastest-in-existence performance.
- The large8192-u64 combined median remains 0.489% slower with six of eight pairs slower. Its initial larger effect did not persist in follow-up, but performance acceptance remains open for this residual bias.

Source reports: [initial](paired-comparison.md), [follow-up](followup/paired-comparison.md). The [JSON investigation](preservation-investigation.json) contains all invocation records, raw samples, source report hashes, and selected-kernel instruction comparison.
