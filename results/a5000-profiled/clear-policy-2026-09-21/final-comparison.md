Frozen-plan confirmation: **12/12 cases complete**. Search and validation selected each configuration before these runs. Confirmations evaluate that frozen choice; they do not independently choose a winner.

These are reconfirmations of configurations selected by prior plans on a different recorded binary. The new binary uses explicit kernel clearing for these graph measurements. No new search or validation selection was performed; the search counts and selection ratios belong to the prior plans. The JSON preserves prior-plan, prior-binary, prior-search, new-binary and provenance hashes. Configuration matching permits only the declared clear-policy field change.

Each confirmation contains the chosen configuration, the fastest preserved scalar custom configuration from the same search (tuning<6), and every recorded reference, deduplicated when configurations coincide. The analyzer verifies CSV configuration sets, command variants, workload metadata and recorded binary hashes.

R1/R2 use seed424242 in two separate invocations; R3 uses seed987654. Both seeds are excluded from this plan's search and validation. Every invocation has21 samples and batch32. Ratios are reference-time/chosen-time or scalar-time/chosen-time; values above1 favor the frozen choice. The strongest reference is recomputed within each invocation. Scalar comparisons use the current binary and timing protocol, not historical before timings.

| Case | Chosen algorithm:tuning:blocks:local[:clear] | Selection search configs | Strongest reference / chosen, R1 / R2 / R3 | Scalar / chosen, R1 / R2 / R3 | Chosen >2× samples; largest max/median |
|---|---|---:|---|---|---|
| small8 | shared:6:192:native:kernel | 333 | cub 2.471× / cub 2.476× / cub 2.476× | 1.021× / 1.021× / 1.021× | 0/63; 1.06× |
| smallbyte | shared:1:96:native:kernel | 334 | cub 1.477× / cub 1.477× / cub 1.477× | 1.000× / 1.000× / 1.000× | 0/63; 1.03× |
| cachedbyte | shared:11:48:native:kernel | 334 | cub 1.364× / cub 1.358× / cub 1.355× | 1.238× / 1.232× / 1.224× | 0/63; 1.06× |
| byte | cub:2:192:native:kernel | 334 | cub 1.000× / cub 1.000× / cub 1.000× | 0.985× / 0.983× / 0.988× | 0/63; 1.43× |
| hot99 | shared:11:48:native:kernel | 333 | cub 2.345× / cub 2.345× / cub 2.351× | 1.105× / 1.105× / 1.105× | 0/63; 1.01× |
| sortedhot99 | shared:6:192:native:kernel | 333 | cub 2.291× / cub 2.289× / cub 2.289× | 1.032× / 1.032× / 1.032× | 0/63; 1.07× |
| single | shared:6:192:native:kernel | 333 | cub 2.280× / cub 2.280× / cub 2.279× | 1.026× / 1.026× / 1.026× | 0/63; 1.27× |
| large4096 | shared:11:48:native:kernel | 233 | cub 4.103× / cub 4.124× / cub 4.095× | 1.009× / 1.015× / 1.008× | 0/63; 1.11× |
| large4096-u64 | shared:10:48:u32:kernel | 425 | cub 10.199× / cub 10.173× / cub 10.154× | 1.005× / 1.004× / 1.010× | 0/63; 1.14× |
| large8192-u64 | shared:15:48:u32:kernel | 265 | cub 19.970× / cub 19.942× / cub 18.876× | 1.015× / 1.016× / 0.962× | 0/63; 1.15× |
| large16384-u64 | shared:15:48:u32:kernel | 73 | cub 30.597× / cub 30.793× / cub 30.562× | 2.674× / 2.692× / 2.661× | 0/63; 1.20× |
| cold4096-u64 | shared:10:48:u32:kernel | 425 | cub 5.444× / cub 5.460× / cub 5.395× | 1.044× / 1.053× / 1.053× | 0/63; 1.65× |

| Case | Chosen load / shared-memory limit | Preserved scalar control | R1/R2 chosen median spread (max/min) |
|---|---|---|---:|
| small8 | full_tile / 48KiB | shared:0:384:native:kernel | 1.000× |
| smallbyte | scalar / 48KiB | shared:0:48:native:kernel | 1.000× |
| cachedbyte | vector4 / 48KiB | shared:2:96:native:kernel | 1.000× |
| byte | reference / N/A | shared:2:192:native:kernel | 1.001× |
| hot99 | vector4 / 48KiB | shared:2:96:native:kernel | 1.000× |
| sortedhot99 | full_tile / 48KiB | shared:1:192:native:kernel | 1.005× |
| single | full_tile / 48KiB | shared:2:192:native:kernel | 1.000× |
| large4096 | vector4 / 48KiB | shared:2:96:native:kernel | 1.000× |
| large4096-u64 | vector4 / 48KiB | shared:2:96:u32:kernel | 1.003× |
| large8192-u64 | vector4 / 96KiB | shared:2:96:u32:kernel | 1.002× |
| large16384-u64 | vector4 / 96KiB | global:1:384:native:kernel | 1.004× |
| cold4096-u64 | vector4 / 48KiB | shared:3:48:u32:kernel | 1.001× |

Ratios and raw-sample excursions describe these recorded runs. They do not establish universal performance or a precise tail distribution. Two repeats of one seed provide limited repeatability evidence. Reference choice may differ across runs; the JSON records all candidates, per-run medians and samples. No samples are discarded. Telemetry snapshots are preserved without inferring a clock/power cause.
