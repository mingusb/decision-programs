# Matched binary preservation measurements

Status: **complete** for requested rounds 1, 2.

Each ratio is current median / old median. Values above 1 indicate a slower current invocation.

| Case | Status | Pairs | Median ratio | Min ratio | Max ratio | Slower pairs | >=1.05 pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| small8 | complete | 4 | 0.905797 | 0.800948 | 1.005952 | 1 | 0 |
| smallbyte | complete | 4 | 0.455279 | 0.382353 | 1.000000 | 0 | 0 |
| cachedbyte | complete | 4 | 0.996350 | 0.992593 | 1.451852 | 1 | 1 |
| byte | complete | 4 | 1.001281 | 0.997436 | 1.003854 | 2 | 0 |
| hot99 | complete | 4 | 1.000000 | 0.994186 | 1.017544 | 1 | 0 |
| sortedhot99 | complete | 4 | 1.000000 | 0.982558 | 1.000000 | 0 | 0 |
| single | complete | 4 | 0.994152 | 0.850000 | 1.000000 | 0 | 0 |
| large4096 | complete | 4 | 1.000173 | 0.999653 | 1.013589 | 2 | 0 |
| large4096-u64 | complete | 4 | 0.998622 | 0.997588 | 1.070160 | 1 | 1 |
| large8192-u64 | complete | 4 | 1.025344 | 1.003735 | 1.062809 | 4 | 1 |
| large16384-u64 | complete | 4 | 0.998751 | 0.993194 | 1.005165 | 2 | 0 |
| cold4096-u64 | complete | 4 | 1.002216 | 0.980994 | 1.007407 | 2 | 0 |
| stream4096 | complete | 4 | 0.960630 | 0.404250 | 1.063752 | 2 | 1 |

## Adjacent invocation pairs

| Case | Round | Positions | Order | Old median (µs) | Current median (µs) | Ratio | >=1.05 flag |
|---|---:|---|---|---:|---:|---:|---|
| small8 | 1 | 1/2 | old→current | 5.376000 | 5.408000 | 1.005952 | no |
| small8 | 1 | 3/4 | current→old | 6.624000 | 5.376000 | 0.811594 | no |
| small8 | 2 | 1/2 | current→old | 5.376000 | 5.376000 | 1.000000 | no |
| small8 | 2 | 3/4 | old→current | 6.752000 | 5.408000 | 0.800948 | no |
| smallbyte | 1 | 1/2 | old→current | 6.528000 | 2.496000 | 0.382353 | no |
| smallbyte | 1 | 3/4 | current→old | 5.664000 | 2.496000 | 0.440678 | no |
| smallbyte | 2 | 1/2 | current→old | 5.312000 | 2.496000 | 0.469880 | no |
| smallbyte | 2 | 3/4 | old→current | 2.496000 | 2.496000 | 1.000000 | no |
| cachedbyte | 1 | 1/2 | old→current | 4.320000 | 4.288000 | 0.992593 | no |
| cachedbyte | 1 | 3/4 | current→old | 4.320000 | 6.272000 | 1.451852 | yes |
| cachedbyte | 2 | 1/2 | current→old | 4.384000 | 4.352000 | 0.992701 | no |
| cachedbyte | 2 | 3/4 | old→current | 4.320000 | 4.320000 | 1.000000 | no |
| byte | 1 | 1/2 | old→current | 49.952000 | 49.952000 | 1.000000 | no |
| byte | 1 | 3/4 | current→old | 49.920000 | 49.791999 | 0.997436 | no |
| byte | 2 | 1/2 | current→old | 49.952000 | 50.080001 | 1.002562 | no |
| byte | 2 | 3/4 | old→current | 49.823999 | 50.016001 | 1.003854 | no |
| hot99 | 1 | 1/2 | old→current | 5.504000 | 5.472000 | 0.994186 | no |
| hot99 | 1 | 3/4 | current→old | 5.472000 | 5.472000 | 1.000000 | no |
| hot99 | 2 | 1/2 | current→old | 5.472000 | 5.568000 | 1.017544 | no |
| hot99 | 2 | 3/4 | old→current | 5.472000 | 5.472000 | 1.000000 | no |
| sortedhot99 | 1 | 1/2 | old→current | 5.504000 | 5.408000 | 0.982558 | no |
| sortedhot99 | 1 | 3/4 | current→old | 5.440000 | 5.440000 | 1.000000 | no |
| sortedhot99 | 2 | 1/2 | current→old | 5.440000 | 5.440000 | 1.000000 | no |
| sortedhot99 | 2 | 3/4 | old→current | 5.440000 | 5.440000 | 1.000000 | no |
| single | 1 | 1/2 | old→current | 6.400000 | 5.440000 | 0.850000 | no |
| single | 1 | 3/4 | current→old | 5.440000 | 5.440000 | 1.000000 | no |
| single | 2 | 1/2 | current→old | 5.440000 | 5.440000 | 1.000000 | no |
| single | 2 | 3/4 | old→current | 5.472000 | 5.408000 | 0.988304 | no |
| large4096 | 1 | 1/2 | old→current | 184.607998 | 184.607998 | 1.000000 | no |
| large4096 | 1 | 3/4 | current→old | 184.607998 | 184.543997 | 0.999653 | no |
| large4096 | 2 | 1/2 | current→old | 195.455998 | 198.111996 | 1.013589 | no |
| large4096 | 2 | 3/4 | old→current | 184.576005 | 184.640005 | 1.000347 | no |
| large4096-u64 | 1 | 1/2 | old→current | 186.048001 | 185.791999 | 0.998624 | no |
| large4096-u64 | 1 | 3/4 | current→old | 185.760006 | 185.312003 | 0.997588 | no |
| large4096-u64 | 2 | 1/2 | current→old | 185.632005 | 198.655993 | 1.070160 | yes |
| large4096-u64 | 2 | 3/4 | old→current | 185.632005 | 185.376003 | 0.998621 | no |
| large8192-u64 | 1 | 1/2 | old→current | 188.096002 | 189.055994 | 1.005104 | no |
| large8192-u64 | 1 | 3/4 | current→old | 188.511997 | 189.216003 | 1.003735 | no |
| large8192-u64 | 2 | 1/2 | current→old | 187.999994 | 199.808002 | 1.062809 | yes |
| large8192-u64 | 2 | 3/4 | old→current | 189.536005 | 198.175997 | 1.045585 | no |
| large16384-u64 | 1 | 1/2 | old→current | 192.767993 | 191.456005 | 0.993194 | no |
| large16384-u64 | 1 | 3/4 | current→old | 192.607999 | 192.640007 | 1.000166 | no |
| large16384-u64 | 2 | 1/2 | current→old | 192.064002 | 193.056002 | 1.005165 | no |
| large16384-u64 | 2 | 3/4 | old→current | 192.128003 | 191.615999 | 0.997335 | no |
| cold4096-u64 | 1 | 1/2 | old→current | 21.664000 | 21.760000 | 1.004431 | no |
| cold4096-u64 | 1 | 3/4 | current→old | 21.600000 | 21.760000 | 1.007407 | no |
| cold4096-u64 | 2 | 1/2 | current→old | 21.696000 | 21.696000 | 1.000000 | no |
| cold4096-u64 | 2 | 3/4 | old→current | 21.888000 | 21.472000 | 0.980994 | no |
| stream4096 | 1 | 1/2 | old→current | 15.872000 | 16.031999 | 1.010081 | no |
| stream4096 | 1 | 3/4 | current→old | 17.568000 | 18.688001 | 1.063752 | yes |
| stream4096 | 2 | 1/2 | current→old | 44.803999 | 18.112000 | 0.404250 | no |
| stream4096 | 2 | 3/4 | old→current | 20.896001 | 19.040000 | 0.911179 | no |

## Interpretation and provenance

- Ratios are current median / old median; values above 1 mean the current invocation was slower.
- Each ratio pairs adjacent invocations in the specified ABBA/BAAB order. Individual timing samples are not paired.
- The >=1.05 flag marks a suspicious regression for investigation. Smaller slowdowns remain reported and are not declared acceptable.
- These measurements do not establish statistical significance, prove zero regressions, or cover workloads absent from this manifest.
- Executable identities are validated from recorded hashes and paths; the analyzer does not execute or require the historical binaries.

All raw samples, complete invocation metadata, and artifact SHA256 hashes are retained in the JSON report.
CSV summary values and bandwidth were checked against raw samples and workload size. Exact commands, variants, binary identities, GPU/driver metadata, and benchmark environments were validated.
