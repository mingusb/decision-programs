# Larger histogram scaling experiments

Status: **complete**. Audited 29/29 requested cases, 174 invocations, and 3452 candidate measurements.

Speedups below use fresh confirmation seeds and candidates measured in the same invocation. Reference/custom and default/custom ratios above 1 favor the selected custom configuration. The existing production default is resolved and measured separately for each workload, then included in confirmation.

| Inputs | Bins | Chosen algorithm:policy:grid:local:clear | NVIDIA/custom range | Default/custom range | Custom µs range |
|---:|---:|---|---:|---:|---:|
| 16,777,216 | 16,384 | shared:15:24:u32:kernel | 31.0781–31.6537 | 1.0027–1.0027 | 190.2080–190.7200 |
| 16,777,216 | 24,576 | shared:15:24:u32:kernel | 36.9960–37.6951 | 1.1864–1.1877 | 192.2560–192.2560 |
| 16,777,216 | 32,768 | global:2:1536:native:kernel | 16.3512–16.6932 | 0.9956–0.9978 | 469.7600–472.3200 |
| 16,777,216 | 65,536 | global:1:384:native:kernel | 18.8428–18.9346 | 0.9902–1.0011 | 465.9200–470.5280 |
| 33,554,432 | 16,384 | shared_rle:15:48:u32:kernel | 32.6566–32.6880 | 1.0537–1.0592 | 371.7120–371.9680 |
| 33,554,432 | 24,576 | shared:15:24:u32:kernel | 38.0667–38.6769 | 1.0816–1.0924 | 374.0160–376.3200 |
| 33,554,432 | 32,768 | global:1:1536:native:kernel | 16.6855–16.7311 | 1.0087–1.0219 | 935.9360–936.9600 |
| 33,554,432 | 65,536 | global:1:384:native:kernel | 19.2003–19.2267 | 1.0019–1.0534 | 928.0000–929.2800 |
| 67,108,864 | 16,384 | shared:15:24:u32:kernel | 33.1818–33.3600 | 1.0219–1.0261 | 735.2320–738.0480 |
| 67,108,864 | 24,576 | shared_rle:15:48:u32:kernel | 39.0273–39.0999 | 1.0366–1.0373 | 740.3520–741.8880 |
| 67,108,864 | 32,768 | global:4:96:native:kernel | 15.5708–15.7183 | 1.0203–1.0717 | 1979.1360–2005.5039 |
| 67,108,864 | 65,536 | global:2:192:native:kernel | 18.0348–18.1904 | 1.0000–1.0000 | 1964.5441–1979.9041 |
| 134,217,728 | 16,384 | shared:15:192:u32:kernel | 33.1935–33.2136 | 0.9983–1.0040 | 1480.4480–1483.2640 |
| 134,217,728 | 24,576 | shared:15:24:u32:kernel | 39.2882–39.6364 | 1.0078–1.0173 | 1465.8560–1478.1441 |
| 134,217,728 | 32,768 | global:3:96:native:kernel | 11.9512–12.4517 | 0.9596–1.0650 | 5022.9759–5240.5758 |
| 134,217,728 | 65,536 | global:2:384:native:kernel | 13.6950–13.8758 | 0.9721–1.0206 | 5160.7041–5229.5680 |
| 268,435,456 | 16,384 | shared:14:192:u32:kernel | 27.6713–35.0357 | 1.0000–1.0000 | 3198.4639–4108.2878 |
| 268,435,456 | 24,576 | shared:14:384:u32:kernel | 38.1237–38.1460 | 0.9597–0.9931 | 3058.9440–3059.2000 |
| 268,435,456 | 32,768 | global:2:48:native:kernel | 11.9550–12.1367 | 0.9965–1.0139 | 10313.2162–10505.2156 |
| 268,435,456 | 65,536 | global:4:96:native:kernel | 13.7778–13.8796 | 0.9987–1.0018 | 10274.8156–10348.0320 |
| 16,777,216 | 24,575 | shared:15:24:u32:kernel | 36.8109–36.8813 | 1.1880–1.1891 | 192.0000–192.2560 |
| 16,777,216 | 24,577 | global:0:192:native:kernel | 15.0541–15.0937 | 1.0032–1.0038 | 472.8320–472.8320 |
| 16,777,216 | 131,072 | global:2:48:native:kernel | 20.7907–20.8568 | 1.0022–1.0039 | 464.8960–464.8960 |
| 16,777,216 | 262,144 | global:0:192:native:kernel | 22.5086–22.5107 | 1.0107–1.0124 | 476.6720–477.1840 |
| 16,777,216 | 1,048,576 | warp:4:96:native:kernel | 3.6995–3.7282 | 1.0060–1.0186 | 4553.2160–4581.1200 |
| 536,870,912 | 16,384 | shared_rle:15:768:u32:kernel | 32.3137–32.3866 | 0.9946–1.0006 | 6062.8481–6081.7919 |
| 536,870,912 | 65,536 | global:2:192:native:kernel | 13.6964–13.7328 | 1.0000–1.0000 | 20989.4409–21168.8957 |
| 1,073,741,824 | 16,384 | shared_partial:15:48:u32:kernel | 32.7231–32.8941 | 0.9955–1.0000 | 11964.4155–12021.2479 |
| 1,073,741,824 | 65,536 | global:2:48:native:kernel | 13.9856–14.0192 | 1.0040–1.0050 | 40976.8944–41060.3523 |

## Search and selection

Search ranks the declared custom candidates by median latency. Validation includes the four fastest, the fastest remaining candidate from every other custom algorithm family, the production default, and the NVIDIA histogram. Selection maximizes median speedup over the NVIDIA histogram across the two validation seeds; median latency and variant string break ties. Confirmation uses two further seeds after this choice is frozen.

| Case | Fastest search custom | Validation-selected custom | Production default |
|---|---|---|---|
| n16777216-b16384 | shared:15:48:u32:kernel | shared:15:24:u32:kernel | shared:15:48:u32:kernel |
| n16777216-b24576 | shared_rle:15:24:u32:kernel | shared:15:24:u32:kernel | shared:14:192:u32:kernel |
| n16777216-b32768 | global:4:768:native:kernel | global:2:1536:native:kernel | global:2:192:native:kernel |
| n16777216-b65536 | global:4:768:native:kernel | global:1:384:native:kernel | global:2:192:native:kernel |
| n33554432-b16384 | shared:15:48:u32:kernel | shared_rle:15:48:u32:kernel | shared:14:192:u32:kernel |
| n33554432-b24576 | shared:15:24:u32:kernel | shared:15:24:u32:kernel | shared:14:192:u32:kernel |
| n33554432-b32768 | global:1:1536:native:kernel | global:1:1536:native:kernel | global:2:192:native:kernel |
| n33554432-b65536 | global:2:1536:native:kernel | global:1:384:native:kernel | global:2:192:native:kernel |
| n67108864-b16384 | shared:15:24:u32:kernel | shared:15:24:u32:kernel | shared:14:192:u32:kernel |
| n67108864-b24576 | shared:15:48:u32:kernel | shared_rle:15:48:u32:kernel | shared:14:192:u32:kernel |
| n67108864-b32768 | warp:2:384:native:kernel | global:4:96:native:kernel | global:2:192:native:kernel |
| n67108864-b65536 | global:1:96:native:kernel | global:2:192:native:kernel | global:2:192:native:kernel |
| n134217728-b16384 | shared:15:24:u32:kernel | shared:15:192:u32:kernel | shared:14:192:u32:kernel |
| n134217728-b24576 | shared:15:24:u32:kernel | shared:15:24:u32:kernel | shared:14:192:u32:kernel |
| n134217728-b32768 | global:3:96:native:kernel | global:3:96:native:kernel | global:2:192:native:kernel |
| n134217728-b65536 | global:2:384:native:kernel | global:2:384:native:kernel | global:2:192:native:kernel |
| n268435456-b16384 | shared:15:192:u32:kernel | shared:14:192:u32:kernel | shared:14:192:u32:kernel |
| n268435456-b24576 | shared:15:96:u32:kernel | shared:14:384:u32:kernel | shared:14:192:u32:kernel |
| n268435456-b32768 | global:2:96:native:kernel | global:2:48:native:kernel | global:2:192:native:kernel |
| n268435456-b65536 | global:4:96:native:kernel | global:4:96:native:kernel | global:2:192:native:kernel |
| n16777216-b24575 | shared_rle:15:24:u32:kernel | shared:15:24:u32:kernel | shared:14:192:u32:kernel |
| n16777216-b24577 | global:0:192:native:kernel | global:0:192:native:kernel | global:2:192:native:kernel |
| n16777216-b131072 | global:0:192:native:kernel | global:2:48:native:kernel | global:2:192:native:kernel |
| n16777216-b262144 | global:2:1536:native:kernel | global:0:192:native:kernel | global:2:192:native:kernel |
| n16777216-b1048576 | warp:4:96:native:kernel | warp:4:96:native:kernel | global:2:192:native:kernel |
| n536870912-b16384 | shared_partial:15:48:u32:kernel | shared_rle:15:768:u32:kernel | shared:14:192:u32:kernel |
| n536870912-b65536 | global:4:96:native:kernel | global:2:192:native:kernel | global:2:192:native:kernel |
| n1073741824-b16384 | shared:15:48:u32:kernel | shared_partial:15:48:u32:kernel | shared:14:192:u32:kernel |
| n1073741824-b65536 | global:0:48:native:kernel | global:2:48:native:kernel | global:2:192:native:kernel |

## Every confirmation comparison

| Case | Seed | Chosen µs | NVIDIA µs | Default µs | NVIDIA/custom | Default/custom |
|---|---:|---:|---:|---:|---:|---:|
| n16777216-b16384 | 196613 | 190.720007 | 6036.992073 | 191.231996 | 31.653690 | 1.002685 |
| n16777216-b16384 | 262147 | 190.208003 | 5911.295891 | 190.720007 | 31.078061 | 1.002692 |
| n16777216-b24576 | 196613 | 192.256004 | 7247.104168 | 228.351995 | 37.695073 | 1.187750 |
| n16777216-b24576 | 262147 | 192.256004 | 7112.703800 | 228.095993 | 36.996004 | 1.186418 |
| n16777216-b32768 | 196613 | 469.760001 | 7841.792107 | 467.711985 | 16.693188 | 0.995640 |
| n16777216-b32768 | 262147 | 472.319990 | 7723.008156 | 471.296012 | 16.351220 | 0.997832 |
| n16777216-b65536 | 196613 | 470.528007 | 8866.047859 | 465.920001 | 18.842763 | 0.990207 |
| n16777216-b65536 | 262147 | 465.920001 | 8822.015762 | 466.432005 | 18.934615 | 1.001099 |
| n33554432-b16384 | 196613 | 371.968001 | 12147.199631 | 391.936004 | 32.656572 | 1.053682 |
| n33554432-b16384 | 262147 | 371.711999 | 12150.527954 | 393.727988 | 32.688016 | 1.059229 |
| n33554432-b24576 | 196613 | 376.320004 | 14325.247765 | 407.040000 | 38.066666 | 1.081633 |
| n33554432-b24576 | 262147 | 374.015987 | 14465.791702 | 408.576012 | 38.676934 | 1.092403 |
| n33554432-b32768 | 196613 | 935.935974 | 15659.263611 | 956.416011 | 16.731127 | 1.021882 |
| n33554432-b32768 | 262147 | 936.959982 | 15633.664131 | 945.151985 | 16.685520 | 1.008743 |
| n33554432-b65536 | 196613 | 927.999973 | 17817.855835 | 929.791987 | 19.200276 | 1.001931 |
| n33554432-b65536 | 262147 | 929.279983 | 17867.008209 | 978.944004 | 19.226722 | 1.053444 |
| n67108864-b16384 | 196613 | 738.048017 | 24489.728928 | 754.176021 | 33.181756 | 1.021852 |
| n67108864-b16384 | 262147 | 735.231996 | 24527.360916 | 754.432023 | 33.360029 | 1.026114 |
| n67108864-b24576 | 196613 | 740.351975 | 28947.711945 | 768.000007 | 39.099932 | 1.037344 |
| n67108864-b24576 | 262147 | 741.887987 | 28953.855515 | 769.024014 | 39.027260 | 1.036577 |
| n67108864-b32768 | 196613 | 2005.503893 | 31227.392197 | 2046.207905 | 15.570846 | 1.020296 |
| n67108864-b32768 | 262147 | 1979.135990 | 31108.608246 | 2120.959997 | 15.718277 | 1.071660 |
| n67108864-b65536 | 196613 | 1964.544058 | 35735.809326 | 1964.544058 | 18.190383 | 1.000000 |
| n67108864-b65536 | 262147 | 1979.904056 | 35707.134247 | 1979.904056 | 18.034780 | 1.000000 |
| n134217728-b16384 | 196613 | 1483.263969 | 49234.687805 | 1489.151955 | 33.193477 | 1.003970 |
| n134217728-b16384 | 262147 | 1480.448008 | 49170.944214 | 1477.887988 | 33.213557 | 0.998271 |
| n134217728-b24576 | 196613 | 1478.144050 | 58073.600769 | 1489.663959 | 39.288188 | 1.007793 |
| n134217728-b24576 | 262147 | 1465.855956 | 58101.249695 | 1491.199970 | 39.636398 | 1.017290 |
| n134217728-b32768 | 196613 | 5240.575790 | 62630.912781 | 5028.607845 | 11.951151 | 0.959553 |
| n134217728-b32768 | 262147 | 5022.975922 | 62544.639587 | 5349.376202 | 12.451710 | 1.064981 |
| n134217728-b65536 | 196613 | 5229.568005 | 71618.812561 | 5083.648205 | 13.694977 | 0.972097 |
| n134217728-b65536 | 262147 | 5160.704136 | 71608.833313 | 5267.199993 | 13.875787 | 1.020636 |
| n268435456-b16384 | 196613 | 4108.287811 | 113681.663513 | 4108.287811 | 27.671300 | 1.000000 |
| n268435456-b16384 | 262147 | 3198.463917 | 112060.417175 | 3198.463917 | 35.035698 | 1.000000 |
| n268435456-b24576 | 196613 | 3059.200048 | 116696.319580 | 2935.807943 | 38.146024 | 0.959665 |
| n268435456-b24576 | 262147 | 3058.943987 | 116618.240356 | 3037.695885 | 38.123693 | 0.993054 |
| n268435456-b32768 | 196613 | 10505.215645 | 125589.500427 | 10468.095779 | 11.954966 | 0.996467 |
| n268435456-b32768 | 262147 | 10313.216209 | 125168.899536 | 10456.576347 | 12.136747 | 1.013901 |
| n268435456-b65536 | 196613 | 10348.031998 | 142572.799683 | 10334.976196 | 13.777770 | 0.998738 |
| n268435456-b65536 | 262147 | 10274.815559 | 142610.168457 | 10292.991638 | 13.879584 | 1.001769 |
| n16777216-b24575 | 196613 | 192.256004 | 7077.119827 | 228.607997 | 36.810917 | 1.189081 |
| n16777216-b24575 | 262147 | 192.000002 | 7081.215858 | 228.095993 | 36.881332 | 1.188000 |
| n16777216-b24577 | 196613 | 472.831994 | 7118.080139 | 474.368006 | 15.054142 | 1.003249 |
| n16777216-b24577 | 262147 | 472.831994 | 7136.767864 | 474.624008 | 15.093665 | 1.003790 |
| n16777216-b131072 | 196613 | 464.895993 | 9665.535927 | 466.688007 | 20.790749 | 1.003855 |
| n16777216-b131072 | 262147 | 464.895993 | 9696.255684 | 465.920001 | 20.856828 | 1.002203 |
| n16777216-b262144 | 196613 | 476.671994 | 10729.215622 | 482.560009 | 22.508592 | 1.012352 |
| n16777216-b262144 | 262147 | 477.183998 | 10741.760254 | 482.304007 | 22.510730 | 1.010730 |
| n16777216-b1048576 | 196613 | 4581.120014 | 16947.711945 | 4608.511925 | 3.699469 | 1.005979 |
| n16777216-b1048576 | 262147 | 4553.215981 | 16975.360870 | 4637.951851 | 3.728213 | 1.018610 |
| n536870912-b16384 | 196613 | 6081.791878 | 196525.054932 | 6049.024105 | 32.313676 | 0.994612 |
| n536870912-b16384 | 262147 | 6062.848091 | 196354.812622 | 6066.688061 | 32.386563 | 1.000633 |
| n536870912-b65536 | 196613 | 21168.895721 | 289938.690186 | 21168.895721 | 13.696449 | 1.000000 |
| n536870912-b65536 | 262147 | 20989.440918 | 288243.713379 | 20989.440918 | 13.732796 | 1.000000 |
| n1073741824-b16384 | 196613 | 12021.247864 | 393372.924805 | 11967.488289 | 32.723136 | 0.995528 |
| n1073741824-b16384 | 262147 | 11964.415550 | 393558.776855 | 11964.415550 | 32.894108 | 1.000000 |
| n1073741824-b65536 | 196613 | 41060.352325 | 574255.126953 | 41224.449158 | 13.985636 | 1.003996 |
| n1073741824-b65536 | 262147 | 40976.894379 | 574461.669922 | 41180.416107 | 14.019161 | 1.004967 |

## Interpretation and evidence

- Search, validation, and confirmation are separate stages. The chosen custom configuration is frozen using validation seeds before the two fresh confirmation seeds are inspected.
- Confirmation ratios compare candidates measured in the same invocation: reference/custom and production-default/custom values above one favor the chosen custom configuration.
- Every candidate and raw timing sample is retained. A best search result is a result within the declared candidate set, not a claim of global optimality.
- The automatic-default invocation uses three samples on the search seed to identify its concrete configuration; performance comparisons use its fresh same-invocation confirmation measurements.
- When the selected configuration equals the production default, both labels refer to the same measured CSV row. Their ratio of one is an identity, not an independent performance-equivalence measurement.
- Uniform shuffled u32 inputs, u64 counts, warm cache, and graph execution are the only workloads covered. These observations do not establish behavior for other distributions, types, launch modes, or devices.
- Warm cache means no explicit eviction between operations; it does not mean these larger input arrays fit in the GPU cache. Input GB/s counts input bytes only, not output or internal memory traffic.
- This campaign uses timing protocol 3, batch 4, and 200 ms requested warmup. Earlier batch-32 campaign latencies are not treated as matched comparisons.
- Clocks are unlocked. Before/after telemetry does not capture every timed batch. Two confirmation seeds do not establish statistical significance or universal superiority.
- Requested clear_policy describes custom initialization. The NVIDIA histogram retains its own initialization; its scratch metadata is checked for within-case consistency without repeating a GPU workspace query.
- No production defaults are changed by this analysis or by the recorded experiment selection.

The JSON report preserves every audited CSV row, raw timing sample, command, telemetry record, log, selection score, and artifact hash. The audit checks exact commands and candidate coverage, binary/GPU identity, workload/protocol consistency, raw-derived summaries, source hashes, and both deterministic selection stages. No GPU queries or execution are performed by this analyzer.
