# Matched binary preservation measurements

Status: **complete** for requested rounds 1, 2.

Each ratio is current median / old median. Values above 1 indicate a slower current invocation.

| Case | Status | Pairs | Median ratio | Min ratio | Max ratio | Slower pairs | >=1.05 pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| small8 | complete | 4 | 1.000018 | 0.897849 | 1.255952 | 2 | 1 |
| smallbyte | complete | 4 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 |
| cachedbyte | complete | 4 | 0.982014 | 0.894040 | 1.118519 | 1 | 1 |
| byte | complete | 4 | 1.000964 | 0.996159 | 1.009044 | 2 | 0 |
| hot99 | complete | 4 | 1.002924 | 0.994186 | 1.060302 | 2 | 1 |
| sortedhot99 | complete | 4 | 0.997059 | 0.982558 | 1.005917 | 1 | 0 |
| single | complete | 4 | 1.159763 | 1.000000 | 1.260355 | 3 | 3 |
| large4096 | complete | 4 | 1.002603 | 0.991569 | 1.081184 | 3 | 1 |
| large4096-u64 | complete | 4 | 0.998544 | 0.945889 | 1.001208 | 2 | 0 |
| large8192-u64 | complete | 4 | 1.000429 | 0.942728 | 1.007171 | 2 | 0 |
| large16384-u64 | complete | 4 | 0.999917 | 0.933914 | 1.001337 | 2 | 0 |
| cold4096-u64 | complete | 4 | 0.997071 | 0.994083 | 1.008863 | 1 | 0 |
| stream4096 | complete | 4 | 1.074403 | 1.046032 | 1.300000 | 4 | 3 |
| native-boundary | complete | 4 | 1.004080 | 0.999231 | 1.023239 | 2 | 0 |
| native-medium | complete | 4 | 0.998645 | 0.997191 | 1.013932 | 1 | 0 |
| native-million | complete | 4 | 1.000222 | 0.998873 | 1.000916 | 3 | 0 |

## Adjacent invocation pairs

| Case | Round | Positions | Order | Old median (µs) | Current median (µs) | Ratio | >=1.05 flag |
|---|---:|---|---|---:|---:|---:|---|
| small8 | 1 | 1/2 | old→current | 5.952000 | 5.344000 | 0.897849 | no |
| small8 | 1 | 3/4 | current→old | 5.344000 | 5.376000 | 1.005988 | no |
| small8 | 2 | 1/2 | current→old | 5.376000 | 5.344000 | 0.994048 | no |
| small8 | 2 | 3/4 | old→current | 5.376000 | 6.752000 | 1.255952 | yes |
| smallbyte | 1 | 1/2 | old→current | 2.496000 | 2.496000 | 1.000000 | no |
| smallbyte | 1 | 3/4 | current→old | 2.496000 | 2.496000 | 1.000000 | no |
| smallbyte | 2 | 1/2 | current→old | 2.784000 | 2.784000 | 1.000000 | no |
| smallbyte | 2 | 3/4 | old→current | 2.816000 | 2.816000 | 1.000000 | no |
| cachedbyte | 1 | 1/2 | old→current | 4.832000 | 4.320000 | 0.894040 | no |
| cachedbyte | 1 | 3/4 | current→old | 4.320000 | 4.320000 | 1.000000 | no |
| cachedbyte | 2 | 1/2 | current→old | 4.320000 | 4.832000 | 1.118519 | yes |
| cachedbyte | 2 | 3/4 | old→current | 4.448000 | 4.288000 | 0.964029 | no |
| byte | 1 | 1/2 | old→current | 49.984001 | 49.791999 | 0.996159 | no |
| byte | 1 | 3/4 | current→old | 49.536001 | 49.984001 | 1.009044 | no |
| byte | 2 | 1/2 | current→old | 49.791999 | 49.888000 | 1.001928 | no |
| byte | 2 | 3/4 | old→current | 50.048001 | 50.048001 | 1.000000 | no |
| hot99 | 1 | 1/2 | old→current | 6.368000 | 6.752000 | 1.060302 | yes |
| hot99 | 1 | 3/4 | current→old | 5.472000 | 5.504000 | 1.005848 | no |
| hot99 | 2 | 1/2 | current→old | 5.504000 | 5.472000 | 0.994186 | no |
| hot99 | 2 | 3/4 | old→current | 5.472000 | 5.472000 | 1.000000 | no |
| sortedhot99 | 1 | 1/2 | old→current | 5.504000 | 5.408000 | 0.982558 | no |
| sortedhot99 | 1 | 3/4 | current→old | 5.440000 | 5.408000 | 0.994118 | no |
| sortedhot99 | 2 | 1/2 | current→old | 5.440000 | 5.440000 | 1.000000 | no |
| sortedhot99 | 2 | 3/4 | old→current | 5.408000 | 5.440000 | 1.005917 | no |
| single | 1 | 1/2 | old→current | 5.408000 | 6.816000 | 1.260355 | yes |
| single | 1 | 3/4 | current→old | 5.440000 | 5.440000 | 1.000000 | no |
| single | 2 | 1/2 | current→old | 5.408000 | 6.816000 | 1.260355 | yes |
| single | 2 | 3/4 | old→current | 5.408000 | 5.728000 | 1.059172 | yes |
| large4096 | 1 | 1/2 | old→current | 184.735999 | 184.992000 | 1.001386 | no |
| large4096 | 1 | 3/4 | current→old | 184.287995 | 184.992000 | 1.003820 | no |
| large4096 | 2 | 1/2 | current→old | 185.984001 | 184.415996 | 0.991569 | no |
| large4096 | 2 | 3/4 | old→current | 184.863999 | 199.872002 | 1.081184 | yes |
| large4096-u64 | 1 | 1/2 | old→current | 195.743993 | 185.151994 | 0.945889 | no |
| large4096-u64 | 1 | 3/4 | current→old | 185.376003 | 185.599998 | 1.001208 | no |
| large4096-u64 | 2 | 1/2 | current→old | 185.696006 | 185.888007 | 1.001034 | no |
| large4096-u64 | 2 | 3/4 | old→current | 186.496004 | 185.760006 | 0.996054 | no |
| large8192-u64 | 1 | 1/2 | old→current | 198.911995 | 187.519997 | 0.942728 | no |
| large8192-u64 | 1 | 3/4 | current→old | 186.848000 | 187.071994 | 1.001199 | no |
| large8192-u64 | 2 | 1/2 | current→old | 187.583998 | 187.519997 | 0.999659 | no |
| large8192-u64 | 2 | 3/4 | old→current | 200.800002 | 202.240005 | 1.007171 | no |
| large16384-u64 | 1 | 1/2 | old→current | 191.487998 | 191.679999 | 1.001003 | no |
| large16384-u64 | 1 | 3/4 | current→old | 205.791995 | 192.192003 | 0.933914 | no |
| large16384-u64 | 2 | 1/2 | current→old | 191.808000 | 191.584006 | 0.998832 | no |
| large16384-u64 | 2 | 3/4 | old→current | 191.487998 | 191.744000 | 1.001337 | no |
| cold4096-u64 | 1 | 1/2 | old→current | 21.920000 | 21.888000 | 0.998540 | no |
| cold4096-u64 | 1 | 3/4 | current→old | 21.632000 | 21.504000 | 0.994083 | no |
| cold4096-u64 | 2 | 1/2 | current→old | 21.824000 | 21.728000 | 0.995601 | no |
| cold4096-u64 | 2 | 3/4 | old→current | 21.664000 | 21.856000 | 1.008863 | no |
| stream4096 | 1 | 1/2 | old→current | 20.160001 | 21.088000 | 1.046032 | no |
| stream4096 | 1 | 3/4 | current→old | 17.600000 | 22.879999 | 1.300000 | yes |
| stream4096 | 2 | 1/2 | current→old | 18.336000 | 19.872000 | 1.083770 | yes |
| stream4096 | 2 | 3/4 | old→current | 19.711999 | 20.994000 | 1.065037 | yes |
| native-boundary | 1 | 1/2 | old→current | 623.583972 | 629.119992 | 1.008878 | no |
| native-boundary | 1 | 3/4 | current→old | 623.935997 | 623.456001 | 0.999231 | no |
| native-boundary | 2 | 1/2 | current→old | 623.776019 | 638.271987 | 1.023239 | no |
| native-boundary | 2 | 3/4 | old→current | 623.839974 | 623.391986 | 0.999282 | no |
| native-medium | 1 | 1/2 | old→current | 615.552008 | 624.127984 | 1.013932 | no |
| native-medium | 1 | 3/4 | current→old | 626.272023 | 625.119984 | 0.998160 | no |
| native-medium | 2 | 1/2 | current→old | 625.472009 | 624.927998 | 0.999130 | no |
| native-medium | 2 | 3/4 | old→current | 626.559973 | 624.800026 | 0.997191 | no |
| native-million | 1 | 1/2 | old→current | 4541.247845 | 4545.407772 | 1.000916 | no |
| native-million | 1 | 3/4 | current→old | 4539.616108 | 4541.567802 | 1.000430 | no |
| native-million | 2 | 1/2 | current→old | 4541.632175 | 4541.696072 | 1.000014 | no |
| native-million | 2 | 3/4 | old→current | 4543.424129 | 4538.303852 | 0.998873 | no |

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
