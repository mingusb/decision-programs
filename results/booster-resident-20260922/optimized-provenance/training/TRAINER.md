# Custom GPU tree trainer

The `ghb` library implements levelwise boosted trees with owned CUDA kernels.
The original count kernels, policies, and frozen count benchmarks are unchanged.
The training library links CUDA runtime and the optional stage recorder; it uses
no NVIDIA histogram implementation, CUB, Thrust, or external boosting library.
Its training performance must be measured independently of the count results.

## Build and run

```bash
cmake -S training -B build/booster-resident -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGHB_BUILD_HISTOGRAM_PROBE=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build/booster-resident -j 6
ctest --test-dir build/booster-resident --output-on-failure

build/booster-resident/ghb_bench --objective regression --rows 65536 \
  --features 32 --rounds 10 --tree-execution graph --instrumentation timing \
  --output-dir results/my-first-trainer-run

build/booster-resident/ghb_bench --objective binary --outputs 1024 \
  --output-tile 16 --rows 1024 --test-rows 256 --features 16 \
  --rounds 2 --depth 2 --bins 16 --histogram global --tree-execution graph \
  --tree-export-batch 16 \
  --output-dir results/my-wide-multilabel-run
```

The output directory must not already exist; its parent must exist. It contains
the model, held-out targets, model predictions, base-score predictions, and JSON
measurements. An `.incomplete` marker remains on failure. Training and held-out
rows are generated separately from recorded seeds, using the same deterministic
synthetic target function. Wide-output labels use different features and
thresholds. These fixtures test functionality; they are not NLP datasets or
evidence of predictive superiority over other learning systems.

## API and supported targets

See [booster.hpp](include/ghb/booster.hpp). `Dataset.values` is a row-major float
matrix. Numeric/categorical feature types can be supplied per column; an empty
type list means numeric. NaN is missing. Infinite feature values are rejected.
Weights are optional finite nonnegative row weights with a positive total.

| Objective | Targets | Model outputs |
|---|---|---|
| `squared_error` | row-major `rows × Dataset.outputs` | independent real values |
| `binary_logistic` | row-major `rows × Dataset.outputs`, each 0 or 1 | independent probabilities; supports multilabel |
| `multiclass_softmax` | one integer class label per row, `Dataset.outputs=1` | `TrainConfig.classes` probabilities summing to one |

```cpp
ghb::Dataset data;
data.rows = rows;
data.columns = features;
data.outputs = 1024;
data.values = feature_values;       // rows * features
data.targets = binary_targets;      // rows * 1024 independent 0/1 labels
ghb::TrainConfig config;
config.objective = ghb::Objective::binary_logistic;
config.output_tile_size = 16;
config.max_device_bytes = std::size_t{4} << 30;
auto trained = ghb::train(data, config);
auto probabilities = trained.model.predict_gpu(held_out);
```

Training returns the model, initial/per-round loss, histogram tuning samples,
planned GPU payload bytes, feature-preparation peak bytes, phase wall times,
and optional stage samples.
`predict(..., true)` and `predict_gpu(..., true)` return raw margins. The default
returns regression values or transformed probabilities. Model save/load uses a
versioned little-endian format with checked feature metadata and tree graphs.
Loading/saving currently has a 1 GiB model safety bound.

## Large output spaces

Independent regression and binary outputs reuse a gradient/Hessian tile. Their
derivative storage is `16 × rows × min(outputs, output_tile_size)` bytes rather
than `16 × rows × outputs`. The last partial tile is supported. Each output still
gets its own scalar trees; this is not a vector-leaf tree implementation.

Predictions remain dense doubles (`8 × rows × outputs` bytes), and targets remain
dense floats. These arrays, packed features, tree staging, histograms, split
candidates, and other owned persistent GPU payloads are checked against
`max_device_bytes` before allocation (default 4 GiB), and against reported free
GPU memory. Reducing the tile size does not reduce dense prediction/target
storage. CUDA/recorder bookkeeping and host memory are outside this payload
budget. GPU prediction also currently materializes a dense output matrix.

Multiclass outputs are coupled through softmax. The trainer preserves a full
pre-round derivative snapshot for multiclass; it does not apply independent
output tiling to that objective. Multiclass uses the diagonal Hessian convention
`2*p*(1-p)`, floored at `1e-16` before weighting. Binary uses `p*(1-p)` with the
same floor. Zero-weight rows contribute no gradient/Hessian but remain in the
exact row counts used by `min_leaf_rows`.

For NLP, this version can consume prepared numeric features/embeddings and
multi-target or multilabel data. Sparse targets, vocabulary-scale sampled
objectives, tokenization, sequence models, learned embeddings, and out-of-core
output storage are not implemented. Wide-output tests do not establish that
dense independent trees are the best architecture for those workloads.

## Trees, binning, and histograms

Features are fitted and packed on the GPU as feature-major u16 bin IDs. Bin zero represents
missing values. Numeric cuts are deterministic quantiles of distinct observed
values, fitted on training rows only; they are not weighted quantiles. Categories
are exact finite float values. Excess categorical cardinality is rejected instead
of merged silently, and unseen prediction categories follow the missing branch.

GPU preparation uses stable keys-only radix sorting, distinct-run detection,
exact cut selection and encoding. `quantize_policy` selects radix8 (four passes)
or radix4 (eight passes); both preserve the same cuts and bins. Preparation
tiles whole features to bound scratch, retaining every row in each feature's
exact fit. It exports only fitted metadata and status; dense bins stay on the
device. `preparation_peak_bytes` includes resident output and simultaneous
scratch, and must fit `max_device_bytes`. GPU inference also encodes features
on the device. See [the binning experiment](QUANTIZE_EXPERIMENT.md) for traffic
analysis and stronger sorting/deduplication contenders still to evaluate.

Splits use numeric prefixes or categorical equality. Both missing directions and
missing-versus-present splits are considered. Split gain uses the actual quadratic
objective improvement, including L2 regularization and optional leaf clipping.
Minimum leaf rows, child Hessian, and gain constraints are enforced. Learning rate
is included in stored leaf values. Categorical subset splits, monotonic/interaction
constraints, ranking losses, sampling, and ordered target statistics are not yet
implemented.

Each training bin stores a double gradient sum, double Hessian sum, and exact u64
row count (24 bytes). Two owned implementations are available: global atomics,
and per-feature shared-memory privatization. The latter holds all active nodes'
bins for that feature and visits each row once per feature. It requires
`frontier_capacity × max_feature_bins <= 2048` (48 KiB). Forced shared mode
conservatively requires the preplanned level capacities to fit, even if a
particular dataset would stop splitting sooner.
Floating-point atomic accumulation order may vary; bit-identical training across
runs/devices is not guaranteed.

Autotuning warms both alternatives and measures five alternating-order samples
per candidate on the actual root histogram. It retains raw event times and
medians and caches the root selection within the current training call by output
and tile width. Timing includes histogram clearing. Deeper, unmeasured frontiers
use the owned global implementation; root timings do not select a deep-node
policy. Calibration stays outside tree execution. This is a measured initial selection,
not an exhaustive search or a proof of global optimality. Tuning overhead is part
of reported training time. `max_histogram_bytes` bounds preallocated histogram
storage; if a required frontier exceeds it, training fails explicitly.

## Measurement and validation

`record_stages=false` selects a compile-time no-op recorder path. Enabled recording
reserves event slots outside the training loop, records real stages with round,
depth, output, and active-node context, and collects/resets after each round.
The pool is bounded to one million scopes; large-output profiling should use a
representative subset or recording disabled. Stage intervals can overlap with
host orchestration and should not be added into end-to-end wall time.

Weighted base scores, dense input validation, histogram accumulation, split
selection, frontier compaction, tree construction, routing, prediction updates
and objective reduction execute on the GPU. A bounded level schedule uses
device-owned active counts; empty levels return early. Final child leaves reuse
their parent's selected values, avoiding an extra terminal histogram. The host
exports a completed tree, with no split-winner download or child-map upload
between levels. Host dataset setup, transfers, model export/serialization and
the explicit CPU reference predictor remain available.

`tree_execution` selects ordered stream submission or a reusable CUDA graph for
one complete bounded tree. A device selector lets graph replay use successive
outputs without instantiating thousands of graphs. Graph construction is part
of total training cost; replay and model export are included in boosting time.
Graph instrumentation records a whole `tree_build`; stream instrumentation can
also expose individual levels. The graph covers a tree, not the whole training
call. See [the resident experiment](RESIDENT_EXPERIMENT.md).

`tree_export_batch_size=0` retains compact export: download the node count, then
exactly those nodes, with two waits per tree. Positive values request full-capacity
copies into bounded pinned host slots and one export wait per batch. Effective
width is limited by output/derivative tiles and a 64 MiB pinned-node cap; if one
full tree would exceed the cap, compact export remains active. The returned
effective batch and `pinned_export_bytes` report this choice separately from GPU
storage. A batch download scope overlaps queued tree work and calibration, so
its duration is not a pure transfer time. See [the export experiment](EXPORT_BATCH_EXPERIMENT.md).

Independent regression/binary loss uses adjacent lanes for adjacent output
elements, with a division-free scalar specialization. Multiclass retains a
warp-per-row log-sum-exp. Both reduce partial losses on the GPU. See
[the loss-layout experiment](LOSS_LAYOUT_EXPERIMENT.md).

GPU reductions change floating-point summation order relative to the frozen
CPU/hybrid baseline. Exact feature bins and integer counts are tested separately
from numeric tolerances and strict per-output quality comparisons. Tolerance-based
correctness does not establish a zero-loss quality result. Parent-child histogram
subtraction, vector leaves, multi-GPU training, validation-driven early stopping
and framework parity remain unimplemented.

`ghb_bench` initializes the CUDA context before calling `train`. `total_train_ms`
includes trainer setup, quantization, allocation, upload, and training;
`training_ms` covers boosting rounds including tuning, loss evaluation, and
host/device coordination. GPU prediction wall time includes packing, allocation,
upload, inference, and download. `through_validation_wall_ms` starts before data
generation and ends after prediction/serialization checks, before artifact writes.

Use the separate quality evaluator on exported held-out predictions:

```bash
python3 training/tools/evaluate.py evaluate --objective multilabel \
  --targets results/my-wide-multilabel-run/targets.csv \
  --predictions results/my-wide-multilabel-run/predictions.csv \
  --dataset-id synthetic-wide-v1 --split-id held-out \
  --output results/my-wide-multilabel-run/quality.json
```

It records aggregate and per-output metrics and checks input identity when
comparing models. For regression select `regression`; for one binary output
select `binary`; for multiclass select `multiclass --classes K`. The earlier
`observe.py` capture schema is specific to the counting probe and must not be
used to label training measurements as count-histogram results.
