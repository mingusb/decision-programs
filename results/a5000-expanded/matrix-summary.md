# Observed histogram campaign

Best observed per-cell portfolios are ex-post oracles over measured candidates, not deployable selectors. Ratios use within-cell median times. Aggregate medians/geometric means weight cells equally; they are descriptive and do not establish uncertainty or generalization.

All measured candidates, including CUB and the NVIDIA sample where present. Best custom excludes both references. Strongest reference is the faster measured CUB/sample median within that cell.

Analyzed **184 cells** and **1700 candidate measurements**.

GPU: NVIDIA RTX A5000 Laptop GPU (SM86); runtime API 13040; driver API 13020; CUB 300402. Driver API is a compatibility version, not the installed driver build.

Campaign coverage: 184/184 cells (complete); binary `37f184896e2cb15b25b64e563e34bc77fd2daa722323c15a6cdb307ebe65ed68`.

## Comparisons by cache and launch

Ratios above one favor the custom/portfolio result. The portfolio includes references, so its speedup cannot be below one by construction.

| Cache / launch | Cells | Custom ≥1.05x CUB | Custom ≥1.05x strongest ref | Custom/ref median | Custom/ref geomean | Portfolio/ref geomean |
|---|---:|---:|---:|---:|---:|---:|
| all / all | 184 | 141 | 140 | 1.306x | 1.606x | 1.640x |
| cold / graph | 8 | 5 | 5 | 2.109x | 2.295x | 2.360x |
| cold / stream | 8 | 5 | 5 | 2.077x | 2.232x | 2.244x |
| warm / graph | 84 | 61 | 61 | 1.293x | 1.595x | 1.657x |
| warm / stream | 84 | 70 | 69 | 1.331x | 1.515x | 1.522x |

## Largest observed custom/reference ratios

These rows are selected by the observed ratio; they are not an independent validation set. Analysis IDs are not campaign filenames.

| Analysis ID | N | Bins | Input/count | Distribution/order | Cache/launch | Best custom | Custom µs | CUB µs | Strongest reference | Ref µs | Custom/CUB | Custom/ref |
|---:|---:|---:|---|---|---|---|---:|---:|---|---:|---:|---:|
| 183 | 16777216 | 8192 | u32/u64 | uniform/shuffled | warm/stream | shared t3 g96 u32 | 193.741 | 4628.480 | cub | 4628.480 | 23.890x | 23.890x |
| 182 | 16777216 | 8192 | u32/u64 | uniform/shuffled | warm/graph | shared t3 g96 u32 | 191.078 | 3788.186 | cub | 3788.186 | 19.825x | 19.825x |
| 176 | 16777216 | 4096 | u32/u64 | uniform/shuffled | cold/graph | shared t3 g96 u32 | 196.813 | 2836.275 | cub | 2836.275 | 14.411x | 14.411x |
| 114 | 1048576 | 8192 | u32/u64 | uniform/shuffled | warm/graph | shared t3 g96 u32 | 14.541 | 199.066 | cub | 199.066 | 13.690x | 13.690x |
| 110 | 1048576 | 4096 | u32/u64 | uniform/shuffled | warm/graph | shared t3 g96 u32 | 10.650 | 109.978 | cub | 109.978 | 10.327x | 10.327x |
| 178 | 16777216 | 4096 | u32/u64 | uniform/shuffled | warm/graph | shared t3 g96 u32 | 186.778 | 1847.910 | cub | 1847.910 | 9.894x | 9.894x |
| 179 | 16777216 | 4096 | u32/u64 | uniform/shuffled | warm/stream | shared t3 g96 u32 | 190.669 | 1863.680 | cub | 1863.680 | 9.774x | 9.774x |
| 177 | 16777216 | 4096 | u32/u64 | uniform/shuffled | cold/stream | shared t3 g96 u32 | 199.066 | 1928.397 | cub | 1928.397 | 9.687x | 9.687x |
| 115 | 1048576 | 8192 | u32/u64 | uniform/shuffled | warm/stream | shared t3 g96 u32 | 24.576 | 200.704 | cub | 200.704 | 8.167x | 8.167x |
| 112 | 1048576 | 8192 | u32/u64 | single/shuffled | warm/graph | shared t3 g96 u32 | 10.240 | 68.608 | cub | 68.608 | 6.700x | 6.700x |
| 102 | 1048576 | 4096 | u32/u32 | uniform/shuffled | warm/graph | shared t3 g96 native | 10.240 | 68.198 | cub | 68.198 | 6.660x | 6.660x |
| 108 | 1048576 | 4096 | u32/u64 | uniform/shuffled | cold/graph | shared t3 g96 u32 | 23.552 | 116.941 | cub | 116.941 | 4.965x | 4.965x |

## Smallest observed custom/reference ratios

These rows are selected by the observed ratio; they are not an independent validation set. Analysis IDs are not campaign filenames.

| Analysis ID | N | Bins | Input/count | Distribution/order | Cache/launch | Best custom | Custom µs | CUB µs | Strongest reference | Ref µs | Custom/CUB | Custom/ref |
|---:|---:|---:|---|---|---|---|---:|---:|---|---:|---:|---:|
| 82 | 1048576 | 256 | u8/u32 | uniform/sorted | warm/graph | shared_partial t3 g96 native | 7.987 | 5.325 | cub | 5.325 | 0.667x | 0.667x |
| 20 | 4096 | 256 | u8/u32 | hot99/shuffled | warm/graph | shared_partial t3 g96 native | 5.939 | 4.096 | cub | 4.096 | 0.690x | 0.690x |
| 76 | 1048576 | 256 | u8/u32 | hot99/sorted | warm/graph | shared_partial t3 g96 native | 7.782 | 5.530 | cub | 5.530 | 0.711x | 0.711x |
| 24 | 4096 | 256 | u8/u32 | uniform/shuffled | warm/graph | shared_partial t3 g96 native | 5.734 | 4.096 | cub | 4.096 | 0.714x | 0.714x |
| 74 | 1048576 | 256 | u8/u32 | hot99/shuffled | warm/graph | shared_partial t3 g96 native | 7.987 | 5.939 | cub | 5.939 | 0.744x | 0.744x |
| 22 | 4096 | 256 | u8/u32 | hot99/sorted | warm/graph | shared_partial t3 g96 native | 5.734 | 4.301 | cub | 4.301 | 0.750x | 0.750x |
| 26 | 4096 | 256 | u8/u32 | uniform/sorted | warm/graph | shared_partial t3 g96 native | 5.734 | 4.301 | cub | 4.301 | 0.750x | 0.750x |
| 80 | 1048576 | 256 | u8/u32 | uniform/shuffled | warm/graph | shared_partial t3 g96 native | 8.192 | 6.349 | cub | 6.349 | 0.775x | 0.775x |
| 144 | 16777216 | 256 | u8/u32 | hot99/sorted | warm/graph | shared_partial t3 g96 native | 57.139 | 49.357 | cub | 49.357 | 0.864x | 0.864x |
| 142 | 16777216 | 256 | u8/u32 | hot99/shuffled | warm/graph | shared_partial t3 g96 native | 56.934 | 49.562 | cub | 49.562 | 0.871x | 0.871x |
| 148 | 16777216 | 256 | u8/u32 | uniform/shuffled | warm/graph | shared t3 g96 native | 56.934 | 49.766 | cub | 49.766 | 0.874x | 0.874x |
| 150 | 16777216 | 256 | u8/u32 | uniform/sorted | warm/graph | shared t3 g96 native | 56.525 | 49.562 | cub | 49.562 | 0.877x | 0.877x |

## Matched local counter widths for u64 outputs

Only exact algorithm/tuning/threads/items/replicas/grid matches are paired; scratch width may differ. A ratio above one favors local u32.

Matched pairs: 100. Unmatched native rows: 0; unmatched u32 rows: 40.

| Cache/launch | Algorithm | Pairs | Median native/u32 | Geomean native/u32 | Min | Max | u32 ≥1.05x faster |
|---|---|---:|---:|---:|---:|---:|---:|
| cold/graph | shared | 4 | 1.312x | 1.300x | 1.180x | 1.409x | 4 |
| cold/graph | shared_partial | 2 | 1.432x | 1.431x | 1.389x | 1.475x | 2 |
| cold/graph | shared_rle | 2 | 1.508x | 1.508x | 1.491x | 1.524x | 2 |
| cold/graph | shared_warp | 2 | 1.322x | 1.320x | 1.249x | 1.395x | 2 |
| cold/stream | shared | 4 | 1.286x | 1.279x | 1.186x | 1.367x | 4 |
| cold/stream | shared_partial | 2 | 1.424x | 1.424x | 1.404x | 1.444x | 2 |
| cold/stream | shared_rle | 2 | 1.487x | 1.486x | 1.463x | 1.510x | 2 |
| cold/stream | shared_warp | 2 | 1.340x | 1.337x | 1.242x | 1.438x | 2 |
| warm/graph | shared | 16 | 9.791x | 7.000x | 1.000x | 56.707x | 15 |
| warm/graph | shared_partial | 8 | 13.551x | 8.182x | 1.415x | 47.653x | 8 |
| warm/graph | shared_rle | 8 | 1.565x | 1.472x | 0.964x | 2.291x | 7 |
| warm/graph | shared_warp | 8 | 1.320x | 1.333x | 1.013x | 2.196x | 7 |
| warm/stream | shared | 16 | 7.309x | 5.694x | 1.084x | 40.986x | 16 |
| warm/stream | shared_partial | 8 | 10.819x | 6.224x | 1.017x | 39.817x | 7 |
| warm/stream | shared_rle | 8 | 1.365x | 1.364x | 1.048x | 1.766x | 7 |
| warm/stream | shared_warp | 8 | 1.172x | 1.191x | 1.018x | 1.478x | 6 |

JSON output retains every cell, matched pair, source file hash, and measurement protocol. Reference scope, candidate coverage, cache eviction, launch mode, and host submission effects remain part of the measurement contract.
