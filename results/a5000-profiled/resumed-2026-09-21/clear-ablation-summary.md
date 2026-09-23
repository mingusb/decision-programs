# Output-clear ablation after resuming: driver 597.06

We compare against NVIDIA’s CUB library of optimized GPU routines; its histogram is our reference implementation.

Replacing runtime memset with the explicit clearing kernel improves the complete shared operation by **2.116×** for N1M/u32/B8, **4.456×** for N4096/u8/B256, and **2.917×** for N1M/u8/B256 (median of three paired ratios). All nine shared pairs improve; unchanged controls remain close to parity.

All 18 invocations below were rerun on the same RTX A5000 Laptop GPU with Windows driver **597.06**. They compare the archived runtime-clear binary against the kernel-clear binary within this session. Each uses uniform shuffled data, u32 output, warm graphs, timing protocol 3, seed 424242, 21 samples and batch 32. This is a paired implementation ablation on one seed; it does not select a dispatch policy.

Old/new ratios divide the runtime-clear process median by the paired kernel-clear median. CUB-normalized gain divides that ratio by the paired CUB ratio. Summary ratios are the median of the three paired ratios; displayed times are medians of three process medians.

Case names: small8 = N1M/u32/B8; smallbyte = N4096/u8/B256; cachedbyte = N1M/u8/B256. N1M is 1,048,576 samples.

| Case / variant | Runtime clear, µs | Kernel clear, µs | Paired old/new | CUB-normalized gain |
|---|---:|---:|---:|---:|
| small8 / cub:2:192:native | 14.720 | 14.720 | 1.000× | 1.000× |
| small8 / shared:4:192:native | 13.472 | 6.368 | 2.116× | 2.106× |
| small8 / shared_partial:2:96:native | 8.000 | 7.968 | 1.000× | 1.000× |
| small8 / bitplane:1:192:native | 19.648 | 11.904 | 1.651× | 1.646× |
| smallbyte / cub:2:192:native | 4.160 | 4.160 | 1.000× | 1.000× |
| smallbyte / shared:7:96:native | 12.832 | 2.880 | 4.456× | 4.456× |
| smallbyte / shared_partial:7:96:native | 5.504 | 5.504 | 1.000× | 1.000× |
| smallbyte / nvidia_sample256:2:192:native | 6.112 | 6.112 | 1.000× | 1.000× |
| cachedbyte / cub:2:192:native | 6.592 | 6.560 | 1.005× | 1.000× |
| cachedbyte / shared:7:96:native | 15.680 | 5.376 | 2.917× | 2.903× |
| cachedbyte / shared_partial:7:96:native | 7.456 | 7.424 | 1.004× | 0.999× |
| cachedbyte / nvidia_sample256:2:192:native | 8.064 | 8.064 | 1.000× | 0.995× |

| Changed variant | Raw gains, R1 / R2 / R3 | CUB-normalized gains, R1 / R2 / R3 |
|---|---|---|
| small8 / shared | 2.281× / 2.116× / 2.005× | 2.281× / 2.106× / 2.014× |
| small8 / bitplane | 1.785× / 1.651× / 1.639× | 1.785× / 1.643× / 1.646× |
| smallbyte / shared | 4.889× / 4.201× / 4.456× | 4.889× / 4.201× / 4.456× |
| cachedbyte / shared | 2.917× / 3.190× / 2.686× | 2.903× / 3.175× / 2.673× |

All 18 command records report success and their recorded SHA256 hashes match the named executables. Paired commands differ only by executable path; configuration and workload metadata match. All 72 raw sample vectors contain 21 values and reproduce median, p95, minimum and maximum (1,512 samples total; 35 exceed twice their own row median). No samples were discarded. GPU UUID and driver identity agree in all before/after telemetry snapshots. Snapshot agreement does not show that clocks were constant during execution.

Runtime-clear SHA256: `5d651b317053a23ed8fb4932066d4020d008655c7634541153c6c262e2702ad4`. Kernel-clear SHA256: `3485149e183ada05893e59d505bcebc1e86d2a484a4184ec53821216072b43c1`.

The [historical ablation](../clear-ablation-summary.md) belongs to the earlier driver session. Its measurements are not pooled with these results, and differences between sessions do not establish a driver effect.

[Full paired measurements, raw samples, hashes and telemetry](clear-ablation-summary.json); [source CSVs and command records](clear-ablation/).
