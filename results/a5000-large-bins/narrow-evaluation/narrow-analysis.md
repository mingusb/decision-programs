# Narrow global-counter evaluation

Independent CPU audit passed: **4/4 cases**, **24 invocations**, **656 candidate measurements**, and **4040 raw timing samples**.

All cases use 16,777,216 uniform shuffled u32 input values, u64 output, warm graph execution, batch 4, and 200 ms requested warmup on the RTX A5000 Laptop GPU, driver 597.06. Search has exactly 140 custom configurations and one NVIDIA histogram reference per shape.

| Bins | Selected algorithm:policy:grid:local:clear | Selected µs range | Native/selected | NVIDIA/selected |
|---:|---|---:|---:|---:|
| 24,577 | global:0:1536:native:kernel | 468.735993–469.760001 | 1.000000–1.000000 | 15.243583–15.374932 |
| 65,536 | global:2:192:native:kernel | 462.336004–462.592006 | 1.000000–1.000000 | 19.063642–19.128460 |
| 262,144 | global:0:192:u32:kernel | 466.176003–466.944009 | 1.018640–1.021966 | 23.291117–23.328940 |
| 1,048,576 | global:0:48:u32:kernel | 2383.359909–2487.807989 | 1.821980–1.898926 | 6.823317–7.127175 |

At 24,577 and 65,536 bins, validation selected native counters: narrowing was not the selected improvement. A native/selected ratio of 1 for an identical policy is a self-comparison, not evidence that narrowing helped. At 262,144 bins the selected narrow policy showed a small improvement; at 1,048,576 bins it showed a larger one. The intervals are ranges of two independent confirmation-seed medians, not confidence intervals.

## Fresh confirmation measurements

| Bins | Seed | Selected µs | Native µs | Default µs | NVIDIA µs | Native/selected | Default/selected | NVIDIA/selected |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24,577 | 104395303 | 468.735993 | 468.735993 | 469.503999 | 7145.215988 | 1.000000 | 1.001638 | 15.243583 |
| 24,577 | 122949829 | 469.760001 | 469.760001 | 470.528007 | 7222.527981 | 1.000000 | 1.001635 | 15.374932 |
| 65,536 | 104395303 | 462.336004 | 462.336004 | 462.336004 | 8843.775749 | 1.000000 | 1.000000 | 19.128460 |
| 65,536 | 122949829 | 462.592006 | 462.592006 | 462.592006 | 8818.688393 | 1.000000 | 1.000000 | 19.063642 |
| 262,144 | 104395303 | 466.944009 | 475.647986 | 475.647986 | 10875.647545 | 1.018640 | 1.018640 | 23.291117 |
| 262,144 | 122949829 | 466.176003 | 476.415992 | 476.415992 | 10875.391960 | 1.021966 | 1.021966 | 23.328940 |
| 1,048,576 | 104395303 | 2487.807989 | 4532.735825 | 4598.015785 | 16975.103378 | 1.821980 | 1.848220 | 6.823317 |
| 1,048,576 | 122949829 | 2383.359909 | 4525.824070 | 4604.671955 | 16986.623764 | 1.898926 | 1.932009 | 7.127175 |

## Frozen comparisons

| Bins | Native comparator | Resolved production default |
|---:|---|---|
| 24,577 | global:0:1536:native:kernel | global:2:192:native:kernel |
| 65,536 | global:2:192:native:kernel | global:2:192:native:kernel |
| 262,144 | global:2:192:native:kernel | global:2:192:native:kernel |
| 1,048,576 | global:0:48:native:kernel | global:2:192:native:kernel |

Search finalists contain the top four custom policies plus the fastest remaining policy from each (algorithm, local counter) family, the resolved default and the NVIDIA reference. Validation selects by median paired NVIDIA/custom ratio, then custom median latency and variant. The native comparator minimizes median validation latency. Both choices are reproduced from validation alone and fixed before the two disjoint confirmation seeds.

## Provenance and limits

- Raw samples, derived summaries, exact command/candidate coverage, executable/GPU identity, metadata, and all persisted finalist/selection/confirmation hashes passed independent checks.
- Production source provenance is checked against narrow-source.tar.gz and its manifest, including the archived CSV parser. Ongoing source changes are not substituted into this historical audit.
- Source archive SHA256: `ebb2fb891735cf219dc17656d7ac7f4fc1e767b8da6340533f6b2b99ba83ae68`.
- Measured executable SHA256: `bcf042dee54cd18408e36f5852e756e88348dd5aa3d3083e145a834a41e9f648`.
- Clocks were unlocked; telemetry is sampled before/after invocations. The small 262,144-bin gain does not establish statistical significance or broader superiority.
- Confirmation compares the selected candidate with the fastest validated native candidate; this is a bounded search result, not a globally optimal native baseline or a universal speed claim.
- No production defaults are promoted or changed. Different input distributions, sizes, cache/launch modes and devices require separate validation. NVIDIA histogram remains benchmark-only.
- Audit performed no GPU queries or execution. The JSON preserves every recorded row, raw sample, command, telemetry record, log, and artifact hash.
