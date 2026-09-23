# Shared-overflow evaluation

Independent CPU audit passed: **5/5 cases**, **30 invocations**, **209 candidate measurements**, and **2191 raw timing samples**.

All cases use 16,777,216 uniform shuffled u32 input values, u64 output, warm graph execution, batch 4, and 200 ms requested warmup on the RTX A5000 Laptop GPU, driver 597.06. Each search includes exactly 14 shared-overflow configurations, frozen native/narrow controls, the actual production default, and the NVIDIA histogram reference. All comparisons below were measured within the same invocation using the same final executable. Ratios above 1 favor the denominator.

| Bins | Selected algorithm:policy:grid:local:clear | Selected µs range | Native/selected | NVIDIA/selected |
|---:|---|---:|---:|---:|
| 24,577 | shared_overflow:15:24:u32:kernel | 194.304004–194.560006 | 2.411067–2.444737 | 36.618419–36.711461 |
| 32,768 | shared_overflow:15:24:u32:kernel | 199.167997–199.936002 | 2.363636–2.383033 | 38.723651–38.939820 |
| 65,536 | shared_overflow:14:48:u32:kernel | 308.735996–308.735996 | 1.509121–1.509121 | 28.485905–28.563849 |
| 262,144 | shared_overflow:14:48:u32:kernel | 433.663994–434.175998 | 1.101535–1.106132 | 24.974647–25.081465 |
| 1,048,576 | global:0:48:u32:kernel | 2483.711958–2491.904020 | 1.813848–1.818697 | 6.809020–6.845496 |

Shared overflow won at 24,577, 32,768, 65,536 and 262,144 bins. At 1,048,576 bins, the narrow global policy won instead: 2.484–2.492 ms versus 4.359–4.366 ms for the independently selected overflow candidate. The losing overflow candidate was still measured on both confirmation seeds. Intervals are ranges of two independent confirmation-seed medians, not confidence intervals.

## Overflow results, including the loss

| Bins | Best validated overflow policy | Overflow µs range | Native/overflow | Narrow/overflow |
|---:|---|---:|---:|---:|
| 24,577 | shared_overflow:15:24:u32:kernel | 194.304004–194.560006 | 2.411067–2.444737 | 2.479578–2.502632 |
| 32,768 | shared_overflow:15:24:u32:kernel | 199.167997–199.936002 | 2.363636–2.383033 | 2.410026–2.428937 |
| 65,536 | shared_overflow:14:48:u32:kernel | 308.735996–308.735996 | 1.509121–1.509121 | 1.528192–1.529851 |
| 262,144 | shared_overflow:14:48:u32:kernel | 433.663994–434.175998 | 1.101535–1.106132 | 1.074292–1.076151 |
| 1,048,576 | shared_overflow:15:24:u32:kernel | 4358.911991–4365.824223 | 1.035300–1.036295 | 0.569801–0.570775 |

## Fresh confirmation measurements

| Bins | Seed | Selected µs | Overflow µs | Native µs | Narrow µs | Default µs | NVIDIA µs | Native/selected |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24,577 | 1500450311 | 194.304004 | 194.304004 | 468.479991 | 481.792003 | 469.503999 | 7133.183956 | 2.411067 |
| 24,577 | 1500450323 | 194.560006 | 194.560006 | 475.647986 | 486.912012 | 474.368006 | 7124.479771 | 2.444737 |
| 32,768 | 1500450311 | 199.167997 | 199.167997 | 474.624008 | 479.999989 | 473.856002 | 7712.512016 | 2.383033 |
| 32,768 | 1500450323 | 199.936002 | 199.936002 | 472.575992 | 485.632002 | 466.944009 | 7785.471916 | 2.363636 |
| 65,536 | 1500450311 | 308.735996 | 308.735996 | 465.920001 | 471.807986 | 465.920001 | 8794.624329 | 1.509121 |
| 65,536 | 1500450323 | 308.735996 | 308.735996 | 465.920001 | 472.319990 | 465.920001 | 8818.688393 | 1.509121 |
| 262,144 | 1500450311 | 434.175998 | 434.175998 | 480.255991 | 466.432005 | 480.255991 | 10843.392372 | 1.106132 |
| 262,144 | 1500450323 | 433.663994 | 433.663994 | 477.696002 | 466.688007 | 477.696002 | 10876.928329 | 1.101535 |
| 1,048,576 | 1500450311 | 2491.904020 | 4365.824223 | 4519.936085 | 2491.904020 | 4591.616154 | 16967.424393 | 1.813848 |
| 1,048,576 | 1500450323 | 2483.711958 | 4358.911991 | 4517.119884 | 2483.711958 | 4596.479893 | 17002.239227 | 1.818697 |

## Frozen comparisons

| Bins | Frozen native comparator | Frozen narrow comparator | Resolved production default |
|---:|---|---|---|
| 24,577 | global:0:1536:native:kernel | global:0:96:u32:kernel | global:2:192:native:kernel |
| 32,768 | global:2:1536:native:kernel | global:2:192:u32:kernel | global:2:192:native:kernel |
| 65,536 | global:2:192:native:kernel | global:4:192:u32:kernel | global:2:192:native:kernel |
| 262,144 | global:2:192:native:kernel | global:0:192:u32:kernel | global:2:192:native:kernel |
| 1,048,576 | global:0:48:native:kernel | global:0:48:u32:kernel | global:2:192:native:kernel |

At four shapes, comparators minimize prior narrow-study validation latency separately for native/u32 counters, with deterministic tie-breaking. Their prior source hashes and scores are checked against the independent narrow audit. For 32,768 bins, native global:2:1536 comes from the scaling study; global:2:192 native/u32 are explicit controls. **The 32,768-bin narrow control was not tuned.** Comparator variants are fixed before overflow search; only their fresh measurements enter the ratios.

Search finalists contain the top four custom policies plus the fastest remaining policy from each (algorithm, local counter) family, the resolved default and the NVIDIA reference. Validation selects by median paired NVIDIA/custom ratio, then custom median latency and variant. The overall winner and best overflow candidate are selected independently by this rule. Confirmation always contains both, both frozen comparators, the actual default and NVIDIA, deduplicating identical variants. All choices are reproduced without consulting confirmation timings.

## Provenance and limits

- Raw samples, derived summaries, exact command/candidate coverage, executable/GPU identity, metadata, and all persisted finalist/selection/confirmation hashes passed independent checks.
- Production source provenance is checked against final-source.tar.gz and its manifest, including the archived CSV parser. Ongoing source changes are not substituted into this historical audit.
- Source archive SHA256: `8a302c90ec615b5bb04806ef62792ef46d277084ec65d8cef8bc0fb43b9b593b`.
- Measured executable SHA256: `ab480a2e08254cb54a5579125ccef3d1101a776aca434ccd0e2e891ddc642ea1`.
- Clocks were unlocked; telemetry is sampled before/after invocations. Two fresh seeds do not establish statistical significance or broader superiority. Ratios between identical variants are self-comparisons.
- Native/narrow comparators were fixed from prior work rather than retuned exhaustively on the final binary. The overflow search is bounded to two policies and seven grids; these are not universal optimum claims.
- No production defaults are promoted or changed. Different input distributions, sizes, cache/launch modes and devices require separate validation. NVIDIA histogram remains benchmark-only.
- Audit performed no GPU queries or execution. The JSON preserves every recorded row, raw sample, command, telemetry record, log, and artifact hash.
