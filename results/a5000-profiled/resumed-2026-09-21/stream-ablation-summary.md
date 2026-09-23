# Output-clear ablation with direct stream launches

We compare against NVIDIA’s CUB library of optimized GPU routines; its histogram is our reference implementation.

The unconditional clearing-kernel change regresses the small8 stream case. This prompted an explicit output-clearing policy, retaining runtime clearing as the public API default. The graph-mode gains must not be generalized to direct stream launches.

All 24 processes use driver 597.06, seed 424242, warm stream mode, 21 samples, and batch 32. The archived binaries differ in output clearing; paired commands otherwise match. Rows below are medians of three process medians, while ratios are medians of paired ratios. Values above 1 favor kernel clearing. Normalization to the NVIDIA histogram reference describes concurrent control shifts; it does not identify their cause.

| Case / variant | Runtime clear, µs | Kernel clear, µs | Runtime/kernel | Reference-normalized ratio | Normalized range |
|---|---:|---:|---:|---:|---:|
| small8 / cub:2:192:native | 17.376 | 18.144 | 1.025× | 1.000× | 1.000–1.000× |
| small8 / shared:4:192:native | 13.088 | 14.336 | 0.913× | 0.891× | 0.883–0.908× |
| small8 / shared_partial:2:96:native | 15.904 | 16.384 | 0.971× | 0.942× | 0.893–1.027× |
| smallbyte / cub:2:192:native | 14.074 | 14.656 | 0.960× | 1.000× | 1.000–1.000× |
| smallbyte / shared:7:96:native | 12.832 | 13.376 | 0.949× | 0.949× | 0.938–0.999× |
| smallbyte / shared_partial:7:96:native | 14.784 | 14.304 | 1.005× | 1.005× | 0.979–1.142× |
| smallbyte / nvidia_sample256:2:192:native | 15.648 | 14.784 | 0.982× | 1.039× | 0.959–1.199× |
| cachedbyte / cub:2:192:native | 15.328 | 15.776 | 0.941× | 1.000× | 1.000–1.000× |
| cachedbyte / shared:7:96:native | 12.864 | 15.328 | 0.839× | 0.951× | 0.603–1.037× |
| cachedbyte / shared_partial:7:96:native | 16.768 | 15.808 | 1.008× | 1.142× | 0.888–1.236× |
| cachedbyte / nvidia_sample256:2:192:native | 16.576 | 14.976 | 1.027× | 1.068× | 1.031–1.164× |
| large4096 / cub:2:192:native | 773.088 | 778.560 | 0.995× | 1.000× | 1.000–1.000× |
| large4096 / shared:3:96:native | 193.088 | 194.304 | 0.999× | 0.998× | 0.997–1.021× |
| large4096 / shared_partial:3:96:native | 199.200 | 201.952 | 0.989× | 0.994× | 0.958–0.995× |

The small8 shared configuration has a normalized ratio below 1 in all three pairs. The byte cases show substantial process-to-process variation, including shifts in unchanged controls; the large4096 shared case is close to parity. Stream timings include host submission gaps, so these numbers characterize complete API operation behavior rather than isolated device kernels.

Verified 24 successful command records, 84 CSV rows, and 1764 raw samples; 62 samples exceed twice their own row median. No samples are discarded. Recorded executable hashes match preserved archives, even if the working build path later changes. GPU/driver identities match the session environment; snapshots do not establish constant clocks.

[Full audit](stream-ablation-summary.json); [raw CSVs and command records](stream-ablation/).
