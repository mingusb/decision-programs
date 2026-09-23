# Matched binary preservation measurements

Status: **complete** for requested rounds 1, 2.

Each ratio is current median / old median. Values above 1 indicate a slower current invocation.

| Case | Status | Pairs | Median ratio | Min ratio | Max ratio | Slower pairs | >=1.05 pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| stream4096 | complete | 4 | 0.971301 | 0.843529 | 2.423081 | 2 | 1 |

## Adjacent invocation pairs

| Case | Round | Positions | Order | Old median (µs) | Current median (µs) | Ratio | >=1.05 flag |
|---|---:|---|---|---:|---:|---:|---|
| stream4096 | 1 | 1/2 | old→current | 27.200000 | 22.944000 | 0.843529 | no |
| stream4096 | 1 | 3/4 | current→old | 19.552000 | 20.352000 | 1.040917 | no |
| stream4096 | 2 | 1/2 | current→old | 22.784000 | 20.544000 | 0.901685 | no |
| stream4096 | 2 | 3/4 | old→current | 19.904001 | 48.229001 | 2.423081 | yes |

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
