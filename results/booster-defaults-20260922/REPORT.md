# Promoted GPU booster defaults

The default TrainConfig and CLI now select batched root histograms, warp32 split
search, global root-count reuse and batched root split searches. The user accepts
the observed rounding-scale differences from the preceding audit for this
promotion. Historical strict failures remain unchanged; the evaluator has not
been modified. This is a release decision, not a claim of bitwise equality or
permission to ignore future material accuracy changes.

The [pre-change decision](../../training/DEFAULT_POLICY_DECISION.md) records the
scope, contracts and experiment. No training algorithm source or CUDA kernel
changed. Original counting kernels and their defaults are untouched. All prior
sealed evidence remains intact.

## Selected settings

| Setting | New default |
|---|---|
| root_histogram | batched |
| split_policy | warp32 (owned block implementation for larger shapes) |
| root_counts | reuse-global |
| split_batch | root |

Histogram auto, stream execution, output tile32, compact export and radix8
quantization remain unchanged. With batched roots, root autotuning is bypassed
and auto uses global accumulation for deeper levels. Shared count setup remains
explicit. The separate deeper batching primitive is not yet integrated into
training. Root cache/count/split storage still counts against the declared
budgets. A tight budget can reject a formerly fitting case; trees are not silently
truncated. The complete legacy settings remain selectable and are documented in
[TRAINER.md](../../training/TRAINER.md).

## Additional applicability measurement

Before editing the defaults, the frozen previous binary ran 24 cases serially:
six workloads, old versus candidate policies, in opposite-order sweeps. Unlike
the prior graph/tile16/export16 measurements, these retain actual other defaults:
auto histograms, stream execution, tile32 and compact export. Candidate total
training time was lower in both sweeps for all six workloads.

Total training wall times in milliseconds, two observations per policy:

| Case | Old defaults | Candidate | Paired reduction |
|---|---:|---:|---:|
| scalar | 53.55 / 63.86 | 45.95 / 40.37 | 14.2–36.8% |
| 129 | 525.42 / 430.19 | 172.94 / 224.31 | 47.9–67.1% |
| 1024 | 2800.08 / 2823.82 | 708.27 / 758.56 | 73.1–74.7% |
| 4096 | 6129.80 / 6631.56 | 1202.76 / 1805.21 | 72.8–80.4% |
| multiclass17 | 64.81 / 63.20 | 36.46 / 25.11 | 43.7–60.3% |
| fallback33 | 89.15 / 85.60 | 41.08 / 46.58 | 45.6–53.9% |

The earlier incremental improvement was measured against an already tuned path.
These larger ratios also remove repeated per-output root calibration: old runs
record 129/1024/4096 tuning records for those output counts; candidate runs record
none. They must not be presented as additional kernel-only speedups or combined
multiplicatively with prior ratios. Synthetic inputs, unlocked clocks, WSL and
two observations limit generalization. No profiler timings enter this table.

The main shapes match the previous scalar/129/1024/4096 protocol with a fresh
seed 20260922605. Multiclass has 17 classes. The additional fallback case has 33
regression outputs, 2048 rows, 40 features, 256 maximum bins and two depth-2 rounds,
exercising the owned block fallback. All exact flags, snapshots, binary hashes
and GPU telemetry are in the per-case captures, with summaries in
[timing-summary.json](timing-summary.json).

## Validation and acceptance

All 11 CTest suites pass in the new build. Legacy comparison configurations are
explicitly pinned in tests so default changes cannot silently replace their
references. New integration checks inherit actual defaults across 33-output
regression/binary and 33-class multiclass, including tile 32 plus its tail,
independent loss calculations, CPU/GPU predictions, cache accounting, learning,
and the multiclass frozen-derivative contract.

Three command-line checks pass: omitted policy flags, explicit new policies,
and the complete legacy policy combination. [CLI verification](cli-verification.json)
confirms inherited settings and exact implicit/explicit memory reporting.
The unchanged implementations retain their previous Nsight and ten passing
sanitizer results; this parameter-only change did not rerun that campaign.

The previous quality audit established rounding-scale metric differences and
unchanged classification decisions; the user accepts that evidence for this
promotion. An additional [context quality audit](context-quality/summary.md)
evaluates all 24 new saved-prediction sets with the unchanged metric evaluator,
retaining exact zero-allowance statuses rather than redefining them.
All 24 captures validate with no evidence errors. Eight of 12 strict comparisons
fail, with maximum metric deterioration 3e-16 nats and maximum prediction
difference 4.996003610813204e-16. Classification decisions, accuracy and AUC are
unchanged. These observations remain within the rounding scale accepted for
this promotion; the recorded audit status is still regression.

Build: `build/booster-defaults`. New trainer SHA256:
`d26b0d8fb2ad5e2d61f5d84bd2dc62dd5d5cba86203168e970e615b5560c42d9`.
Performance prechecks use the unchanged frozen previous SHA256
`fdf60682669fa790dc411e67d9053c0e02e6d49de13b365debc41254ba0f02a1`.
[Preservation checks](preservation.json) verify all 1083 prior reuse artifacts,
the original 12 counting artifacts, and unchanged training algorithm sources.

## Next work

1. Integrate bounded output-tile tree state so the measured batched global
   histogram builds deeper levels across outputs. Batch compatible deeper split
   and routing work too, while preserving independent trees and multiclass
   derivative snapshots.
2. Select tile capacity under explicit memory budgets and measure full training,
   including assignments, frontiers, histograms, tree nodes, routing and exports.
   Primitive speedups alone do not establish an end-to-end gain.
3. Benchmark time to matched held-out quality, peak memory and inference latency
   against XGBoost, LightGBM and CatBoost on real regression, classification and
   substantial multi-output datasets, using separate validation/test sets and
   comparable tuning budgets. This supplies evidence for model-quality work
   beyond the present synthetic kernel and trainer benchmarks.
