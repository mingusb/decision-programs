# Independent quality and metadata audit

All **8/8** selected trained-model-versus-base comparisons passed the exact zero-allowance gate across **20,766 metrics**, including every per-output metric. Source CSVs were rehashed and all metrics recomputed by the separate CPU evaluator. Original result, prediction, target, baseline, and model files were unchanged.

| Case | Train / held-out rows | Outputs | Main held-out metric: baseline → model | Zero-allowance baseline gate |
|---|---:|---:|---:|---|
| [regression-global-a](regression-global-a/quality-comparison.json) | 65,536 / 8,192 | 1 | rmse: 0.978142960 → 0.346886080 | pass |
| [regression-shared-a](regression-shared-a/quality-comparison.json) | 65,536 / 8,192 | 1 | rmse: 0.978142960 → 0.346886080 | pass |
| [regression-auto-stages](regression-auto-stages/quality-comparison.json) | 65,536 / 8,192 | 1 | rmse: 0.978142960 → 0.346886080 | pass |
| [binary](binary/quality-comparison.json) | 32,768 / 4,096 | 1 | logloss: 0.682814238 → 0.199646729 | pass |
| [multiclass](multiclass/quality-comparison.json) | 32,768 / 4,096 | 5 | logloss: 1.609578396 → 0.480813808 | pass |
| [regression-129](regression-129/quality-comparison.json) | 4,096 / 256 | 129 | rmse: 0.944417544 → 0.703284709 | pass |
| [multilabel-1024-tile16](multilabel-1024-tile16/quality-comparison.json) | 4,096 / 256 | 1,024 | logloss: 0.648405779 → 0.488852436 | pass |
| [multilabel-4096](multilabel-4096/quality-comparison.json) | 1,024 / 64 | 4,096 | logloss: 0.646191601 → 0.558580748 | pass |

## Tile-width preservation

For 1,024 outputs, tile widths 16 and 1,024 used byte-identical targets and baseline CSVs. Their model and prediction files are **not byte-identical**. Of 262,144 predicted probabilities, **43,576 differ**, with maximum absolute difference **3.3306690738754696e-16**. All threshold decisions, per-output accuracy, per-output AUC, and aggregate metrics are identical.

However, comparing tile16 against the full-width model with **zero allowance fails**: **274 per-output metrics regress** (122 log loss, 152 Brier). The largest absolute differences are **2.220446049250313e-16** for log loss and **1.1102230246251565e-16** for Brier. These differences are retained in the [complete exact comparison](tile16-vs-tile1024-quality-comparison.json). Close numerical agreement is not exact zero-loss preservation.

Gradient/Hessian payload falls from **64 MiB to 1 MiB (64×)**. Total declared trainer device payload falls from **117,626,436 to 51,566,148 bytes (2.281×)**. The two observed training times were 1,095.959 ms and 834.325 ms; there is only one invocation per width, so this is not a confirmed speedup.

The 4,096-output run records **262,144 bytes (0.25 MiB)** of derivative workspace. Holding all derivatives would require 67,108,864 bytes (64 MiB), a **256× calculated derivative-space difference**; that full-width configuration was not measured. Its total declared payload is 50,681,316 bytes, including the full prediction and target matrices.

## Metadata checks

All **11** preserved case/result/capture sets agree on dimensions, command arguments, generator version, executable identity, and GPU/runtime identity. The audit checked all **31 tuning records / 310 raw timing samples**: each reported latency is the five-sample median and each selected policy matches the smaller median. It also checked **2,052 stage records** for unique IDs, valid timing intervals/scopes, bounded contexts, and expected instrumentation enablement. Recorded derivative sizes match the declared output tiling. CPU/GPU prediction checks and serialization checks succeeded in the original benchmark records.

## Limits

- These are deterministic synthetic generator-version2 workloads; they are not real NLP benchmarks or evidence of representative task accuracy.
- Held-out sets range from64 to8192 rows. The4096-output case uses only64 held-out rows; output count does not increase the number of independent examples.
- Baseline passes compare the trained model to its constant base prediction on the same supplied held-out rows; they do not establish equivalence to another tree implementation.
- The tile comparison retains exact zero-allowance metric failures from last-bit probability differences; no tolerance was added to force a pass.
- Device payload means declared trainer-owned buffer payload, not total CUDA/context/process VRAM. Histogram/timing metadata are checked against their declared meanings, not hardware throughput claims.
- Host stage intervals can contain GPU work or overlap other scopes. GPU and host durations must not be added into an overall time.

[Complete audit, input hashes, stage summaries, and report links](quality-audit.json).
