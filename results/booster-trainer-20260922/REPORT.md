# Hybrid trainer baseline — 22 September 2026

This is a measured reference implementation, **not the requested final GPU-resident
architecture**. Gradient generation, histograms, split scoring, row routing,
prediction and loss run in CUDA. Feature binning and tree metadata construction
still run on the CPU, with a split-decision download/synchronization per level.
The user has required fastest-known GPU algorithm analysis before replacing these
stages. No fastest-ever or framework-superiority claim is established here.

The original counting sources, defaults, archives and benchmark remain byte
identical to the earlier preservation manifest. This trainer has no CUB/Thrust
algorithm linkage. Original count speedups must not be transferred to this
different gradient/Hessian/count contract.

## Functional scope and verification

- Scalar/multi-output regression, independent multilabel binary classification,
  multiclass softmax, row weights, numeric/categorical features, missing values,
  CPU/GPU prediction and checked model save/load.
- Independent outputs use bounded gradient/Hessian tiles; predictions and targets
  remain dense. Multiclass retains its full pre-round derivative snapshot.
- Owned global and shared histograms; measured policy selection with five raw
  samples per candidate. Shared frontier capacity is 2,048 24-byte statistics.
- Parent split leaf values eliminate the redundant final-depth histogram pass.
  Finished model trees retain compact storage and survive asynchronous cleanup.
- Final CTest: six targets pass. Trainer checks: 22,433; kernel checks: 18,238;
  evaluator checks: 47; observation checks: 35; recorder checks: 104.
- Compute Sanitizer memcheck passes kernel and trainer suites; racecheck and
  synccheck pass the kernel suite. See `*-final.log` and exact-command JSONs.
- Nsight Systems and Compute both captured real trainer executions. All 97 NVTX
  ranges match the recorded stage IDs/names; SQLite integrity passes.

## Initial training measurements

RTX A5000 Laptop, SM86, WSL2; CUDA 13.4.59. Clocks were not fixed. All GPU
workloads were run serially. These are small initial observations, not sufficient
for universal ranking, noise bounds, or a no-regression claim.

The scalar regression comparison uses 65,536 training rows, 32 features, 64 maximum
bins, depth five, ten rounds, fixed data, and instrumentation off. Global/shared
were run in ABBA order in separate processes. Training time includes boosting
rounds, objective evaluation, and host coordination; it excludes feature fitting,
initial uploads and allocation, all included in `total_train_ms`.

| Policy/run | Boosting loop (ms) |
|---|---:|
| Global A | 38.932221 |
| Shared A | 28.532402 |
| Shared B | 26.593731 |
| Global B | 34.210482 |

Use each `result.json` for full-precision values. Observed shared/global improvement
is approximately 1.29–1.36× for these paired runs. CPU binning took approximately
183–283 ms across these and the instrumented auto run, dominating total trainer
time. This CPU work is a major architectural problem, not acceptable evidence of
an optimized all-GPU trainer.

| Output workload | Rows / held-out rows | Rounds | Boosting loop (ms) | Held-out objective: base → model |
|---|---:|---:|---:|---:|
| Binary, one output | 32,768 / 4,096 | 10 | 40.277 | 0.682814 → 0.199647 |
| Multiclass, five classes | 32,768 / 4,096 | 10 | 153.950 | 1.609578 → 0.480814 |
| Regression, 129 outputs | 4,096 / 256 | 3 | 130.683 | 0.445962 → 0.247305 |
| Multilabel, 1,024 outputs, tile 16 | 4,096 / 256 | 2 | 834.325 | 0.648406 → 0.488852 |
| Multilabel, 1,024 outputs, full tile | 4,096 / 256 | 2 | 1,095.959 | 0.648406 → 0.488852 |
| Multilabel, 4,096 outputs | 1,024 / 64 | 1 | 1,664.404 | 0.646192 → 0.558581 |

Regression objective is half weighted MSE averaged over outputs. Binary is weighted
log loss averaged over independent outputs; multiclass is weighted log loss.
Classification runs use instrumentation/autotuning; wide-output runs use global
histograms with instrumentation off. Do not compare these as equal protocols.

For 1,024 outputs, tile 16 uses 1 MiB of derivatives versus 64 MiB for the full
tile. Total persistent device payload is 51,566,148 versus 117,626,436 bytes.
The observed runtime difference is a single comparison, not a stable speedup
estimate. Targets and baseline predictions are byte identical. Model predictions
differ by at most 3.33e-16; they are **not bit-identical**.

## Quality audit

[quality-audit.md](quality-audit.md) and [quality-audit.json](quality-audit.json)
contain the independent CPU evaluation and evidence hashes. All eight model-versus-
base comparisons pass their complete metric gates: 20,766 metrics, zero regressions.
This is improvement over a constant base predictor on generated fixtures, not
comparison with established boosting systems or representative NLP data.

The stricter full-tile-to-tile-16 preservation comparison **fails** its zero-
allowance gate: 274 per-output regressions at the last floating-point bits
(122 log-loss, 152 Brier). Maximum differences are 2.22e-16 and 1.11e-16,
respectively. Aggregate metrics, per-output accuracy/AUC, and all threshold
decisions match. The failure is retained; no tolerance was added to turn it into
a pass. The 4,096-output fixture has only 64 held-out rows and establishes limited
functional scaling, not vocabulary-scale NLP quality.

## Profile findings

The Nsight Systems diagnostic uses 65,536 rows, 32 features, depth four, three
rounds, shared histograms and NVTX. Histogram kernels account for 75.1% of measured
kernel time, loss reduction 14.2%, and split candidates 8.2%. The trace includes
inference too; kernel shares are not end-to-end training shares.

The CUDA API table's approximately 213 ms `cudaFree` outlier is initial context
setup via `cudaFree(nullptr)`, outside `train()` timing. It must not be interpreted
as a training-loop deallocation bottleneck. Per-level stream waits/readbacks are
real, and the trace validates their presence.

Separate Nsight Compute root-histogram diagnostics show achieved occupancy
72.21% global and 78.90% shared, with DRAM throughput 4.48% and 5.73% respectively.
These kernels are not near bulk DRAM bandwidth on this case. Launch size, atomic
latency, histogram layout and shared-memory work need analysis. Profiler estimates
of potential speedup are hypotheses, not measurements or guarantees.

## Provenance and next algorithm decisions

`run_campaign.py` records eleven serial cases, arguments, telemetry, stdout/stderr,
whole-process time and executable identity. `baseline-provenance/` preserves the
source and executable used by the campaign before GPU architecture changes.
`quality-audit.json` verifies eleven result/capture pairs, 31 tuning records with
310 raw event samples, and 2,052 stage records.

The next design decisions must precede code: exact GPU binning versus alternative
quantile semantics; device-resident tree metadata/frontiers; row partitioning and
histogram subtraction; output layouts/batching versus vector leaves and reduced
split gradients; and launch/control-flow strategy. Each candidate needs a workload
contract, primary evidence, memory/traffic/synchronization analysis, and an
end-to-end speed plus correctness/quality comparison on this device.
