# Frozen custom histogram versus NVIDIA references

Validated all 26 invocations and 58 measurements across 13 cases. The custom implementation was faster than the best applicable reference on both recorded seeds in 13 cases.

Speedup = best reference median / custom median. Values above 1 favor the custom implementation. Each range includes both seeds, 424242 and 987654.

| Case | Best reference(s) | Seed 424242 speedup | Seed 987654 speedup | Speedup range | Direction changes |
|---|---|---:|---:|---|---|
| small8 | cub | 2.470588× | 2.470238× | 2.470238–2.470588× | no |
| smallbyte | cub | 1.487179× | 1.468354× | 1.468354–1.487179× | no |
| cachedbyte | cub, nvidia_sample256 | 1.228261× | 1.370370× | 1.228261–1.370370× | no |
| byte | cub | 1.049454× | 1.046183× | 1.046183–1.049454× | no |
| hot99 | cub | 2.337209× | 2.356725× | 2.337209–2.356725× | no |
| sortedhot99 | cub | 2.288235× | 2.288235× | 2.288235–2.288235× | no |
| single | cub | 2.282353× | 2.295858× | 2.282353–2.295858× | no |
| large4096 | cub | 4.147778× | 4.216127× | 4.147778–4.216127× | no |
| large4096-u64 | cub | 10.366246× | 10.345835× | 10.345835–10.366246× | no |
| large8192-u64 | cub | 20.085026× | 20.007160× | 20.007160–20.085026× | no |
| large16384-u64 | cub | 31.555444× | 35.665213× | 31.555444–35.665213× | no |
| cold4096-u64 | cub | 5.416048× | 5.413947× | 5.413947–5.416048× | no |
| stream4096 | cub | 3.476510× | 3.332263× | 3.332263–3.476510× | no |

For cachedbyte, the fastest reference changes from nvidia_sample256 at seed 424242 to CUB at seed 987654.

## Every recorded median

| Case | Seed | Algorithm | Median (µs) | Reference/custom speedup |
|---|---:|---|---:|---:|
| small8 | 424242 | shared | 5.984000 | custom |
| small8 | 424242 | cub | 14.784000 | 2.470588× |
| small8 | 987654 | shared | 5.376000 | custom |
| small8 | 987654 | cub | 13.280000 | 2.470238× |
| smallbyte | 424242 | shared | 2.496000 | custom |
| smallbyte | 424242 | cub | 3.712000 | 1.487179× |
| smallbyte | 424242 | nvidia_sample256 | 5.440000 | 2.179487× |
| smallbyte | 987654 | shared | 2.528000 | custom |
| smallbyte | 987654 | cub | 3.712000 | 1.468354× |
| smallbyte | 987654 | nvidia_sample256 | 5.472000 | 2.164557× |
| cachedbyte | 424242 | shared | 5.888000 | custom |
| cachedbyte | 424242 | cub | 7.328000 | 1.244565× |
| cachedbyte | 424242 | nvidia_sample256 | 7.232000 | 1.228261× |
| cachedbyte | 987654 | shared | 4.320000 | custom |
| cachedbyte | 987654 | cub | 5.920000 | 1.370370× |
| cachedbyte | 987654 | nvidia_sample256 | 7.232000 | 1.674074× |
| byte | 424242 | shared | 49.823999 | custom |
| byte | 424242 | cub | 52.288000 | 1.049454× |
| byte | 424242 | nvidia_sample256 | 57.376001 | 1.151574× |
| byte | 987654 | shared | 49.888000 | custom |
| byte | 987654 | cub | 52.191999 | 1.046183× |
| byte | 987654 | nvidia_sample256 | 57.184000 | 1.146248× |
| hot99 | 424242 | shared | 5.504000 | custom |
| hot99 | 424242 | cub | 12.864000 | 2.337209× |
| hot99 | 987654 | shared | 5.472000 | custom |
| hot99 | 987654 | cub | 12.896000 | 2.356725× |
| sortedhot99 | 424242 | shared | 5.440000 | custom |
| sortedhot99 | 424242 | cub | 12.448000 | 2.288235× |
| sortedhot99 | 987654 | shared | 5.440000 | custom |
| sortedhot99 | 987654 | cub | 12.448000 | 2.288235× |
| single | 424242 | shared | 5.440000 | custom |
| single | 424242 | cub | 12.416000 | 2.282353× |
| single | 987654 | shared | 5.408000 | custom |
| single | 987654 | cub | 12.416000 | 2.295858× |
| large4096 | 424242 | shared | 185.791999 | custom |
| large4096 | 424242 | cub | 770.623982 | 4.147778× |
| large4096 | 987654 | shared | 184.928000 | custom |
| large4096 | 987654 | cub | 779.680014 | 4.216127× |
| large4096-u64 | 424242 | shared | 185.056001 | custom |
| large4096-u64 | 424242 | cub | 1918.336034 | 10.366246× |
| large4096-u64 | 987654 | shared | 185.151994 | custom |
| large4096-u64 | 987654 | cub | 1915.552020 | 10.345835× |
| large8192-u64 | 424242 | shared | 187.424004 | custom |
| large8192-u64 | 424242 | cub | 3764.415979 | 20.085026× |
| large8192-u64 | 987654 | shared | 187.711999 | custom |
| large8192-u64 | 987654 | cub | 3755.584002 | 20.007160× |
| large16384-u64 | 424242 | shared | 190.464005 | custom |
| large16384-u64 | 424242 | cub | 6010.176182 | 31.555444× |
| large16384-u64 | 987654 | shared | 190.880001 | custom |
| large16384-u64 | 987654 | cub | 6807.775974 | 35.665213× |
| cold4096-u64 | 424242 | shared | 21.536000 | custom |
| cold4096-u64 | 424242 | cub | 116.640002 | 5.416048× |
| cold4096-u64 | 987654 | shared | 21.568000 | custom |
| cold4096-u64 | 987654 | cub | 116.768002 | 5.413947× |
| stream4096 | 424242 | shared | 19.072000 | custom |
| stream4096 | 424242 | cub | 66.303998 | 3.476510× |
| stream4096 | 987654 | shared | 19.936001 | custom |
| stream4096 | 987654 | cub | 66.431999 | 3.332263× |

## Interpretation and validation

- Speedup is the lowest applicable NVIDIA reference median divided by the frozen custom median from the same invocation. Values above 1 favor the custom implementation.
- Both recorded seeds and every applicable reference are retained. The best reference is selected separately for each seed; changes in its identity are flagged.
- These are finite observations of 13 predeclared workloads on one GPU/driver configuration, not statistical proof of superiority or a universal fastest-histogram claim.
- Custom configurations were frozen before these measurements. The byte case uses its predeclared custom configuration because its historical plan selected a reference.
- Recorded clear_policy is the requested custom initialization strategy. NVIDIA references retain their own initialization, as recorded in the benchmark logs.
- All variants use timing protocol 3, 21 samples, batch 32, and 200 ms requested warmup. Stream and graph timing and cold and warm cache conditions remain distinct workloads.
- CUB workspace sizes are recorded and checked for positivity and consistency between seeds; this CPU-only audit does not repeat a CUDA workspace query.

The JSON report retains every raw sample, CSV row, command record, telemetry record, full log, and artifact hash. All expected files, executable identities, GPU/driver metadata, variants, workload fields, seeds, protocol fields, and CSV summaries were checked. No GPU queries or benchmark execution were performed by the analyzer.
