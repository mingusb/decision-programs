Frozen-plan confirmation: **12/12 cases complete**. Search and validation selected each configuration before these runs. Confirmations evaluate that frozen choice; they do not independently choose a winner.

Each confirmation contains the chosen configuration, the fastest preserved scalar custom configuration from the same search (tuning<6), and every recorded reference, deduplicated when configurations coincide. The analyzer verifies CSV configuration sets, command variants, workload metadata and recorded binary hashes.

R1/R2 use seed424242 in two separate invocations; R3 uses seed987654. Both seeds are excluded from this plan's search and validation. Every invocation has21 samples and batch32. Ratios are reference-time/chosen-time or scalar-time/chosen-time; values above1 favor the frozen choice. The strongest reference is recomputed within each invocation. Scalar comparisons use the current binary and timing protocol, not historical before timings.

| Case | Chosen algorithm:tuning:blocks:local | Search configs | Strongest reference / chosen, R1 / R2 / R3 | Scalar / chosen, R1 / R2 / R3 | Chosen >2× samples; largest max/median |
|---|---|---:|---|---|---|
| small8 | shared:6:192:native | 333 | cub 2.473× / cub 2.463× / cub 2.468× | 1.022× / 1.016× / 1.016× | 0/63; 1.04× |
| smallbyte | shared:1:96:native | 334 | cub 1.477× / cub 1.477× / cub 1.477× | 1.011× / 1.011× / 1.000× | 0/63; 1.58× |
| cachedbyte | shared:11:48:native | 334 | cub 1.355× / cub 1.355× / cub 1.362× | 1.230× / 1.230× / 1.230× | 0/63; 1.06× |
| byte | cub:2:192:native | 334 | cub 1.000× / cub 1.000× / cub 1.000× | 0.992× / 0.978× / 0.993× | 0/63; 1.44× |
| hot99 | shared:11:48:native | 333 | cub 2.345× / cub 2.345× / cub 2.351× | 1.105× / 1.105× / 1.105× | 1/63; 3.12× |
| sortedhot99 | shared:6:192:native | 333 | cub 2.295× / cub 2.295× / cub 2.295× | 1.037× / 1.037× / 1.032× | 0/63; 1.03× |
| single | shared:6:192:native | 333 | cub 2.284× / cub 2.284× / cub 2.284× | 1.026× / 1.026× / 1.026× | 0/63; 1.09× |
| large4096 | shared:11:48:native | 233 | cub 4.174× / cub 4.152× / cub 4.095× | 1.005× / 1.027× / 1.000× | 0/63; 1.14× |
| large4096-u64 | shared:10:48:u32 | 425 | cub 10.335× / cub 10.272× / cub 10.133× | 1.010× / 1.013× / 1.003× | 0/63; 1.16× |
| large8192-u64 | shared:15:48:u32 | 265 | cub 20.177× / cub 20.155× / cub 19.702× | 1.038× / 1.025× / 1.001× | 0/63; 1.16× |
| large16384-u64 | shared:15:48:u32 | 73 | cub 30.415× / cub 30.647× / cub 30.579× | 2.708× / 2.745× / 2.801× | 0/63; 1.18× |
| cold4096-u64 | shared:10:48:u32 | 425 | cub 5.425× / cub 5.482× / cub 5.422× | 1.053× / 1.061× / 1.048× | 0/63; 1.64× |

| Case | Chosen load / shared-memory limit | Preserved scalar control | R1/R2 chosen median spread (max/min) |
|---|---|---|---:|
| small8 | full_tile / 48KiB | shared:0:384:native | 1.011× |
| smallbyte | scalar / 48KiB | shared:0:48:native | 1.000× |
| cachedbyte | vector4 / 48KiB | shared:2:96:native | 1.000× |
| byte | reference / N/A | shared:2:192:native | 1.012× |
| hot99 | vector4 / 48KiB | shared:2:96:native | 1.000× |
| sortedhot99 | full_tile / 48KiB | shared:1:192:native | 1.000× |
| single | full_tile / 48KiB | shared:2:192:native | 1.000× |
| large4096 | vector4 / 48KiB | shared:2:96:native | 1.007× |
| large4096-u64 | vector4 / 48KiB | shared:2:96:u32 | 1.001× |
| large8192-u64 | vector4 / 96KiB | shared:2:96:u32 | 1.000× |
| large16384-u64 | vector4 / 96KiB | global:1:384:native | 1.008× |
| cold4096-u64 | vector4 / 48KiB | shared:3:48:u32 | 1.000× |

Ratios and raw-sample excursions describe these recorded runs. They do not establish universal performance or a precise tail distribution. Two repeats of one seed provide limited repeatability evidence. Reference choice may differ across runs; the JSON records all candidates, per-run medians and samples. No samples are discarded. Telemetry snapshots are preserved without inferring a clock/power cause.
