# Same-process preservation comparison

Audited 24 invocations, 768 quartet clusters, and 3072 raw timing positions.

Real-pair ratios are **new / old**; above 1 means the new backend took longer. AA controls compare slot B / slot A of the same backend. Each table entry summarizes four separate processes. These are descriptive measurements, not equivalence or significance tests.

| Case | Comparison | Event ratio: geometric mean; process range | Submission ratio: geometric mean; process range | Total host ratio: geometric mean; process range |
|---|---|---:|---:|---:|
| single | real | 1.008467; 0.974867–1.041014 | 0.967063; 0.919020–1.002834 | 0.993305; 0.960479–1.017932 |
| single | old-old | 1.007690; 0.976141–1.049758 | 1.030292; 1.022375–1.036204 | 1.024886; 0.975661–1.044586 |
| single | new-new | 0.987878; 0.958231–1.001543 | 0.956006; 0.934568–0.982395 | 0.996005; 0.973720–1.018850 |
| stream4096 | real | 0.989073; 0.935825–1.044372 | 0.986036; 0.962035–1.006836 | 0.994286; 0.940713–1.048359 |
| stream4096 | old-old | 0.999728; 0.967309–1.022429 | 1.018471; 0.993679–1.062186 | 1.002453; 0.946725–1.024841 |
| stream4096 | new-new | 0.988398; 0.944123–1.028099 | 1.003377; 0.973460–1.030968 | 0.995905; 0.951312–1.043270 |

## Every process

Ratios below summarize the 32 quartet clusters in each invocation. Reversed real-pair slot mappings are normalized to new / old. The JSON report retains every quartet ratio, ABBA/BAAB summaries, and slot latency distributions.

| Case | Comparison | Data seed | Order seed | Event geometric mean | Event median | Submission geometric mean | Total host geometric mean |
|---|---|---:|---:|---:|---:|---:|---:|
| single | new-new | 2026092201 | 2026092251 | 0.958231 | 1.002946 | 0.982395 | 0.999595 |
| single | new-new | 2026092201 | 2026092252 | 0.998159 | 0.997055 | 0.969333 | 1.018850 |
| single | new-new | 2026092202 | 2026092251 | 0.994203 | 1.000000 | 0.938585 | 0.973720 |
| single | new-new | 2026092202 | 2026092252 | 1.001543 | 1.000000 | 0.934568 | 0.992379 |
| single | new-old | 2026092201 | 2026092252 | 1.041014 | 0.998682 | 0.963643 | 0.990072 |
| single | new-old | 2026092202 | 2026092252 | 0.974867 | 1.000000 | 0.984804 | 0.960479 |
| single | old-new | 2026092201 | 2026092251 | 1.017060 | 1.000000 | 0.919020 | 1.017932 |
| single | old-new | 2026092202 | 2026092251 | 1.002071 | 1.000000 | 1.002834 | 1.005674 |
| single | old-old | 2026092201 | 2026092251 | 0.976141 | 0.998527 | 1.028179 | 0.975661 |
| single | old-old | 2026092201 | 2026092252 | 1.049758 | 1.007386 | 1.034468 | 1.039436 |
| single | old-old | 2026092202 | 2026092251 | 0.999926 | 1.000000 | 1.022375 | 1.044586 |
| single | old-old | 2026092202 | 2026092252 | 1.006326 | 1.004376 | 1.036204 | 1.041504 |
| stream4096 | new-new | 2026092201 | 2026092251 | 0.977582 | 0.982186 | 1.030968 | 0.974978 |
| stream4096 | new-new | 2026092201 | 2026092252 | 0.944123 | 0.966936 | 0.987752 | 0.951312 |
| stream4096 | new-new | 2026092202 | 2026092251 | 1.028099 | 0.983664 | 0.973460 | 1.043270 |
| stream4096 | new-new | 2026092202 | 2026092252 | 1.005796 | 0.978717 | 1.022456 | 1.016615 |
| stream4096 | new-old | 2026092201 | 2026092252 | 0.935825 | 0.969820 | 0.962035 | 0.940713 |
| stream4096 | new-old | 2026092202 | 2026092252 | 1.008817 | 1.022686 | 0.974919 | 1.001754 |
| stream4096 | old-new | 2026092201 | 2026092251 | 0.970625 | 1.003389 | 1.001044 | 0.989275 |
| stream4096 | old-new | 2026092202 | 2026092251 | 1.044372 | 1.018183 | 1.006836 | 1.048359 |
| stream4096 | old-old | 2026092201 | 2026092251 | 1.022429 | 1.020693 | 0.993679 | 1.024841 |
| stream4096 | old-old | 2026092201 | 2026092252 | 1.013188 | 1.025916 | 1.062186 | 1.020157 |
| stream4096 | old-old | 2026092202 | 2026092251 | 0.967309 | 0.972103 | 1.002207 | 0.946725 |
| stream4096 | old-old | 2026092202 | 2026092252 | 0.996871 | 1.009912 | 1.017164 | 1.020256 |

## Limits

- This is a bounded investigation of two previously flagged explicit configurations, not a proof of zero regressions.
- GPU clocks are not locked. Process/order summaries and AA controls are descriptive; they are not confidence intervals or significance tests.
- The same context, stream, input and output are used for both backends. Each graph slot has a separately captured graph.
- Graph submission time covers graph launch; stream submission time includes event recording and the batch of operations.
- Total host time covers enqueue through synchronization and can include waiting for preceding untimed warmup. It is not pure kernel latency.
- Every timed position is retained and followed by output/canary validation; no outlier deletion or fastest-repeat selection is allowed.
- Per-position output/canary checking performs an untimed device-to-host copy and stream synchronization; this differs from the historical benchmark's between-position protocol.
- The two archive backends are namespace-adapted into a single executable. This controls process differences but does not reproduce either original standalone binary's layout.
- The single-valued case generates identical data for both data seeds; those runs are process replicates, not independent input distributions.
- Position timings within a quartet are dependent. Quartet ratios, process summaries, and AA controls are retained separately; positions are not treated as independent trials.
- AA control ranges provide descriptive noise context. Overlap or lack of overlap alone does not establish equivalence, significance, or a zero-loss guarantee.

Executable SHA-256: `731f03397e404783bd29582037b9d800d4f9c6258aaf1e999ae19f681daab9b5`.

The frozen manifest, per-invocation receipts, original CSV files, command metadata, telemetry, and source provenance remain beside this report. Automatic defaults were not changed.
