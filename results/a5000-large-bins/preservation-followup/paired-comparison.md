# Matched binary preservation measurements

Status: **complete** for requested rounds 1, 2.

Each ratio is current median / old median. Values above 1 indicate a slower current invocation.

| Case | Status | Pairs | Median ratio | Min ratio | Max ratio | Slower pairs | >=1.05 pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| single | complete | 4 | 1.005882 | 1.000000 | 1.123529 | 2 | 1 |
| stream4096 | complete | 4 | 1.261484 | 0.890244 | 2.138889 | 3 | 3 |

## Adjacent invocation pairs

| Case | Round | Positions | Order | Old median (µs) | Current median (µs) | Ratio | >=1.05 flag |
|---|---:|---|---|---:|---:|---:|---|
| single | 1 | 1/2 | old→current | 5.440000 | 5.504000 | 1.011765 | no |
| single | 1 | 3/4 | current→old | 5.440000 | 6.112000 | 1.123529 | yes |
| single | 2 | 1/2 | current→old | 5.440000 | 5.440000 | 1.000000 | no |
| single | 2 | 3/4 | old→current | 5.440000 | 5.440000 | 1.000000 | no |
| stream4096 | 1 | 1/2 | old→current | 19.584000 | 23.808001 | 1.215686 | yes |
| stream4096 | 1 | 3/4 | current→old | 23.615999 | 21.024000 | 0.890244 | no |
| stream4096 | 2 | 1/2 | current→old | 18.015999 | 23.552001 | 1.307283 | yes |
| stream4096 | 2 | 3/4 | old→current | 18.432001 | 39.423998 | 2.138889 | yes |

## Interpretation and provenance

- Ratios are current median / old median; values above 1 mean the current invocation was slower.
- Each ratio pairs adjacent invocations in the specified ABBA/BAAB order. Individual timing samples are not paired.
- The >=1.05 flag marks a suspicious regression for investigation. Smaller slowdowns remain reported and are not declared acceptable.
- These measurements do not establish statistical significance, prove zero regressions, or cover workloads absent from this manifest.
- Executable identities are validated from recorded hashes and paths; the analyzer does not execute or require the historical binaries.

All raw samples, complete invocation metadata, and artifact SHA256 hashes are retained in the JSON report.
CSV summary values and bandwidth were checked against raw samples and workload size. Exact commands, variants, binary identities, GPU/driver metadata, and benchmark environments were validated.

## Frozen scope

- The 13 original cases reproduce their recorded explicit configurations, not a new resolution of automatic defaults.
- Three native global/warp controls use identical batch-32 settings in both binaries; historical batch-4 latencies are not matched comparisons.
- Both seeds are fresh for this preservation campaign. Each case runs ABBA then BAAB; clocks remain unlocked.
- This is bounded preservation evidence, not proof of zero regressions or universal performance equivalence.
- The narrowed backend is evaluated separately. This runner makes no production-default promotion.
