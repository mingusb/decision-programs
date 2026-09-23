Comparison of the preserved external-event graph timing protocol with protocol3's embedded graph events. These are matched tuning0–5 controls. Changes below describe measured timing and variability; they are not kernel-improvement claims.

- cachedbyte: N=1048576, B=256, u8 input, u32 output, uniform/shuffled, warm cache, graph launches.
- small8: N=1048576, B=8, u32 input, u32 output, uniform/shuffled, warm cache, graph launches.
- smallbyte: N=4096, B=256, u8 input, u32 output, uniform/shuffled, warm cache, graph launches.

| Data | Rows / samples | Samples >2× own row median | Largest within-row max/median | Warmup ms |
|---|---:|---:|---:|---:|
| before | 120 / 1800 | 74 (4.11%) | 416.91× | 0 |
| after | 120 / 1800 | 108 (6.00%) | 383.91× | 0 |

Per-operation microseconds, **before → after**. Each number is the median of the seed-specific row medians; the JSON preserves every seed and raw-sample variability statistic.

| Case / candidate | Batch 1 | Batch 2 | Batch 5 | Batch 20 | Batch 100 |
|---|---:|---:|---:|---:|---:|
| cachedbyte / cub:2:192:native | 9.216 → 9.216 | 7.424 → 7.168 | 6.246 → 7.066 | 5.939 → 6.630 | 6.272 → 6.246 |
| cachedbyte / nvidia_sample256:2:192:native | 10.752 → 10.240 | 8.704 → 8.704 | 7.578 → 9.011 | 7.270 → 8.141 | 7.670 → 7.675 |
| cachedbyte / shared:2:192:native | 10.752 → 10.240 | 9.472 → 13.312 | 13.926 → 33.894 | 16.307 → 16.358 | 14.638 → 18.964 |
| cachedbyte / shared_partial:3:96:native | 11.264 → 11.264 | 9.216 → 9.216 | 8.294 → 9.318 | 7.936 → 8.858 | 8.366 → 8.371 |
| small8 / bitplane:1:192:native | 15.360 → 14.336 | 13.824 → 19.456 | 20.173 → 21.709 | 19.917 → 21.171 | 19.565 → 19.907 |
| small8 / cub:2:192:native | 18.432 → 18.432 | 16.896 → 16.384 | 15.360 → 15.155 | 14.976 → 14.874 | 14.090 → 13.993 |
| small8 / shared:4:192:native | 10.752 → 10.240 | 10.240 → 16.384 | 14.960 → 15.360 | 13.056 → 19.123 | 12.646 → 13.583 |
| small8 / shared_partial:2:96:native | 11.264 → 11.264 | 9.728 → 14.336 | 8.397 → 8.294 | 8.192 → 8.115 | 7.660 → 7.598 |
| smallbyte / cub:2:192:native | 7.168 → 7.168 | 5.120 → 5.120 | 5.530 → 4.710 | 3.840 → 4.250 | 4.224 → 3.973 |
| smallbyte / nvidia_sample256:2:192:native | 8.192 → 8.192 | 6.656 → 7.168 | 6.246 → 6.554 | 5.530 → 6.195 | 6.164 → 5.786 |
| smallbyte / shared:2:192:native | 7.168 → 8.192 | 10.232 → 6.656 | 25.498 → 20.070 | 15.590 → 13.235 | 12.636 → 16.952 |
| smallbyte / shared_partial:3:96:native | 8.192 → 8.192 | 6.656 → 7.168 | 6.246 → 6.451 | 5.478 → 6.093 | 6.052 → 5.699 |

Largest across-seed spread of row medians (max/min):

| Protocol | Case / candidate / batch | Seed medians (us) | Spread |
|---|---|---|---:|
| before | smallbyte / shared:2:192:native / 5 | 24680: 14.746, 67890: 36.250 | 2.46× |
| after | cachedbyte / shared:2:192:native / 5 | 24680: 15.770, 67890: 52.019 | 3.30× |

Each seed has one invocation per batch and protocol, so this spread is not same-seed repeatability. Data and candidate order both vary with seed. Timing-protocol changes and any warmup change prevent attributing before/after ratios to kernels. No samples are removed. Clock, power, cache and scheduling causes require separate evidence.
