# Matched binary preservation measurements

Status: **complete** for requested rounds 1, 2.

Each ratio is current median / old median. Values above 1 indicate a slower current invocation.

| Case | Status | Pairs | Median ratio | Min ratio | Max ratio | Slower pairs | >=1.05 pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| cachedbyte | complete | 4 | 1.003704 | 0.754190 | 1.303704 | 2 | 1 |
| large4096-u64 | complete | 4 | 1.005183 | 0.925474 | 1.048425 | 2 | 0 |
| large8192-u64 | complete | 4 | 0.999587 | 0.934852 | 1.004975 | 2 | 0 |
| stream4096 | complete | 4 | 0.978602 | 0.946309 | 1.008503 | 1 | 0 |

## Adjacent invocation pairs

| Case | Round | Positions | Order | Old median (µs) | Current median (µs) | Ratio | >=1.05 flag |
|---|---:|---|---|---:|---:|---:|---|
| cachedbyte | 1 | 1/2 | old→current | 4.320000 | 5.632000 | 1.303704 | yes |
| cachedbyte | 1 | 3/4 | current→old | 4.320000 | 4.320000 | 1.000000 | no |
| cachedbyte | 2 | 1/2 | current→old | 4.320000 | 4.352000 | 1.007407 | no |
| cachedbyte | 2 | 3/4 | old→current | 5.728000 | 4.320000 | 0.754190 | no |
| large4096-u64 | 1 | 1/2 | old→current | 185.471997 | 185.087994 | 0.997930 | no |
| large4096-u64 | 1 | 3/4 | current→old | 185.248002 | 187.552005 | 1.012437 | no |
| large4096-u64 | 2 | 1/2 | current→old | 199.231997 | 184.384003 | 0.925474 | no |
| large4096-u64 | 2 | 3/4 | old→current | 188.991994 | 198.144004 | 1.048425 | no |
| large8192-u64 | 1 | 1/2 | old→current | 187.552005 | 186.496004 | 0.994370 | no |
| large8192-u64 | 1 | 3/4 | current→old | 186.527997 | 187.455997 | 1.004975 | no |
| large8192-u64 | 2 | 1/2 | current→old | 186.463997 | 187.360004 | 1.004805 | no |
| large8192-u64 | 2 | 3/4 | old→current | 200.895995 | 187.808007 | 0.934852 | no |
| stream4096 | 1 | 1/2 | old→current | 16.992001 | 16.543999 | 0.973635 | no |
| stream4096 | 1 | 3/4 | current→old | 19.231999 | 18.916000 | 0.983569 | no |
| stream4096 | 2 | 1/2 | current→old | 19.072000 | 18.048000 | 0.946309 | no |
| stream4096 | 2 | 3/4 | old→current | 18.816000 | 18.975999 | 1.008503 | no |

## Interpretation and provenance

- Ratios are current median / old median; values above 1 mean the current invocation was slower.
- Each ratio pairs adjacent invocations in the specified ABBA/BAAB order. Individual timing samples are not paired.
- The >=1.05 flag marks a suspicious regression for investigation. Smaller slowdowns remain reported and are not declared acceptable.
- These measurements do not establish statistical significance, prove zero regressions, or cover workloads absent from this manifest.
- Executable identities are validated from recorded hashes and paths; the analyzer does not execute or require the historical binaries.

All raw samples, complete invocation metadata, and artifact SHA256 hashes are retained in the JSON report.
CSV summary values and bandwidth were checked against raw samples and workload size. Exact commands, variants, binary identities, GPU/driver metadata, and benchmark environments were validated.
