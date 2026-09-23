# Custom GPU booster and instrumentation

This standalone CUDA C++23 module includes a working tree trainer, GPU/CPU
prediction, model serialization, stage measurements, and saved-prediction quality
checks. See [TRAINER.md](TRAINER.md) for the training API, multi-output support,
memory limits, and benchmark commands. The separate counting probe continues to
exercise our existing measured counting kernels. Counting timings are not used
as evidence of training performance.

The original count project, libraries, executables, defaults, and GPU kernels
are unchanged. The probe imports already built `libgh.a` and `libgh_policy.a`;
it does not rebuild them or link NVIDIA histogram references.

## Build and check

From the repository root:

```bash
cmake -S training -B build/booster-instrumentation -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/booster-instrumentation -j 6
ctest --test-dir build/booster-instrumentation --output-on-failure
```

`GHB_COUNT_BUILD` selects the existing count-library directory and defaults to
`build/window-experiment`. `GHB_BUILD_HISTOGRAM_PROBE=OFF` builds the reusable
recorder and its tests without those libraries. `GHB_WITH_NVTX=OFF` removes the
optional Nsight annotation implementation. CTest labels `cpu` and `gpu` allow
running the CPU checks without GPU access.
`GHB_BUILD_TRAINER=OFF` builds the original instrumentation-only project. Use
`build/booster` for the new trainer to retain earlier binaries as evidence.

## Stage recording

[instrumentation.hpp](include/ghb/instrumentation.hpp) defines `Recorder`,
`NullRecorder`, stage identifiers, and structured context. Stages cover
quantization, upload, initialization, gradient generation, histogram construction
and subtraction, split search, routing, prediction, evaluation, download, and
checkpointing. Context carries round, depth, output, repetition, active nodes,
rows, features, bins, stream ID, operation count, and declared memory traffic.
The caller supplies metadata; the recorder does not infer tree structure or
physical memory transactions from it.

Construct a recorder with sufficient capacity outside the hot path. Pair each
`begin` ticket with `end`, then call `collect(false)` to poll without waiting or
`collect(true)` to wait explicitly for the recorded end events. There are no
recorder-owned allocations or explicit synchronization calls in successful
`begin`/`end`. CUDA and profiling-library internals can still introduce cost.
One host thread owns each recorder; that recorder can observe multiple streams
and overlapping scopes. Do not sum overlapping GPU intervals into total runtime.

Every occurrence owns distinct events until collection/reset. Recording during
stream capture is rejected before consuming a slot; wrap a graph replay instead.
The recorder does not instrument the internal stages of an already captured
whole-training graph. Capturing a graph that reuses event handles across pending
replays would overwrite the measurements. These semantics follow the
[CUDA event API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html).

Select `NullRecorder` at compile time for the uninstrumented path. It is an empty
type with no event, clock, NVTX, allocation, or synchronization operations. The
disabled-path test links without CUDA or the recorder library. The benchmark
still has its own outer timing events and correctness checks in this mode.
Caller-side argument expressions are still evaluated by C++; costly metadata
preparation must also be conditional if it should disappear when recording is off.

With NVTX enabled, the recorder registers names once in the `ghb` domain and
emits stage ranges with the sample ID as payload. Nsight can use these names
to relate CPU submissions to GPU work. See the
[NVTX annotation API](https://nvidia.github.io/NVTX/doxygen/nv_tools_ext_8h.html).

## Reproducible component measurements

[The probe](bench/instrumentation.cpp) defaults to the independently selected
million-bin configuration: policy 2, 96 blocks, and a 524,288-counter window.
Its default input is 16,777,216 uniform shuffled u32 values with dense u64 counts.
These parameters are explicit; the production automatic selector is unchanged.

```bash
python3 training/tools/observe.py capture \
  --exe build/booster-instrumentation/ghb_instrumentation_bench \
  --output-dir results/my-booster-observation-off \
  --source training/src/instrumentation.cpp \
  --source training/bench/instrumentation.cpp \
  -- --instrumentation off --launch graph --seed 20260922401

python3 training/tools/observe.py capture \
  --exe build/booster-instrumentation/ghb_instrumentation_bench \
  --output-dir results/my-booster-observation-timing \
  --source training/src/instrumentation.cpp \
  --source training/bench/instrumentation.cpp \
  -- --instrumentation timing --launch graph --seed 20260922401

python3 training/tools/observe.py audit results/my-booster-observation-timing
python3 training/tools/observe.py compare \
  results/my-booster-observation-off results/my-booster-observation-timing
```

The recorder returns each GPU stage's full batch duration in `samples[].gpu_ms`;
`context.operations` gives the batch size. Independent outer events provide
`timing.operation_ms`, already divided by batch size. The outer interval includes
the enabled inner timing markers and any submission gaps. These are distinct
observables, not interchangeable timing arrays.

The probe preallocates event pairs and pinned output snapshots, submits all
repetitions without per-repetition waits, and reads events after completion. It
validates the final dense output of **every batch** against exact CPU counts;
it does not independently retain every intermediate invocation within a batch.
Snapshot storage is limited to 512 MiB. A cleanup guard drains queued work before
buffers are freed on exceptional exits.

Reported phases include preparation, upload, capture, preflight validation,
warmup, CPU submission, completion wait, collection, and final validation.
`readback_submit_wall_ms` is CPU enqueue time and overlaps `submit_wall_ms`.
Actual GPU download intervals are separate stage samples when enabled.
`device_span_ms` includes snapshots and stream gaps; it is not the sum of kernel
execution time. `end_to_end_wall_ms` includes setup, generation, warmup, and
validation, but ends before result sorting, JSON serialization, and destruction.
The runner separately records the complete subprocess wall duration.

Logical traffic describes declared input scans and final output writes. It
excludes atomic counter traffic and is not measured DRAM traffic. Memory records
separate input, output, scratch, and validation snapshots, plus CUDA free-memory
observations. They do not claim to account for every driver allocation.

The observation runner records exact arguments, raw output, executable/source
snapshots and hashes, available build files, platform, relevant CUDA environment
settings, and GPU telemetry before/after the process. New evidence directories
are created exclusively; interrupted or failed runs cannot audit as successes.
The build ID identifies selected sources, archives, and configuration; the
executable SHA-256 is the complete binary identity. Comparisons require matched
declared workload, seed, validation coverage, environment, and timing arguments.
Input contents are generated from the recorded seed; the probe does not currently
export a separate hash of the generated input array.

Instrumentation comparisons report observed ratios. Different builds,
implementations, clocks, and WSL scheduling can confound attribution. No sample
is dropped and no near-one ratio is promoted into a claim of zero overhead.
Use uninstrumented observations for algorithm rankings and separate profiles
to understand behavior. The original histogram benchmark remains the source
of the previously reported performance comparisons.

## Nsight and sanitizer workflows

```bash
python3 training/tools/observe.py profile --tool nsys \
  --exe build/booster-instrumentation/ghb_instrumentation_bench \
  --output-dir results/my-booster-nsys \
  -- --n 131089 --bins 32768 --repetitions 3 --batch 2 \
     --warmup-ms 0 --instrumentation nvtx --launch graph
```

The same command accepts `ncu`, `memcheck`, `racecheck`, and `synccheck` as tools.
Every profile gets its own directory, tool version, command, stdout/stderr,
native artifacts, and manifest. These runs are diagnostic; `compare` rejects
them as ranking inputs. A failed profiler or sanitizer invocation remains a
failure in the evidence record. Sanitizers use a nonzero error exit code.

## Prediction-quality instrumentation

[evaluate.py](tools/evaluate.py) evaluates saved predictions on an explicitly
named dataset and split. It supports weighted scalar/multi-output regression,
binary classification, multilabel classification, and multiclass classification. Available metrics are
RMSE, MAE, log loss, Brier score, accuracy, and binary ROC AUC as applicable.
Per-output regression metrics are retained alongside aggregate metrics.

Targets use `row_id,target[,weight]`; scalar regression and binary predictions
use `row_id,prediction`. Multi-output regression uses contiguous `target_0,...`
and matching `prediction_0,...` columns. Multiclass predictions use `row_id,p0,...`
with one probability column per class. Multilabel uses the multi-output column
format with independent 0/1 labels and probabilities; select `--objective multilabel`.
Its metrics average outputs equally, retaining each output's metrics. Aggregate
AUC is null if any output has undefined AUC. Row IDs must be unique and match in order.
Numeric inputs must be finite, weights nonnegative with positive total weight,
and class/probability values valid.

```bash
python3 training/tools/evaluate.py evaluate --objective binary \
  --targets targets.csv --predictions predictions.csv \
  --dataset-id my-dataset --split-id held-out-test --output quality.json

python3 training/tools/evaluate.py compare \
  --reference reference-quality.json --candidate quality.json \
  --max-loss-increase 0 --output quality-comparison.json
```

Comparison checks every applicable metric, including each regression output,
against the declared allowance. The allowance is absolute in each metric's own
units: 0.01 accuracy means one percentage point, while 0.01 RMSE means 0.01 target
units. Equality passes. Dataset/split identity, target hashes, numerical settings,
and metric coverage must match. Keep the source CSVs with the reports so their
hashes and metrics can be rechecked. Missing or changed source CSVs make the
comparison fail; an unverified report cannot receive a passing comparison status.

Endpoint probabilities use explicit clipping for log loss. Multiclass vectors
must sum to one within the declared tolerance (default 1e-6), then are normalized;
the tolerance and normalization policy are recorded. AUC awards half credit to
tied scores and is undefined when one class has zero total weight. Metrics
describe the supplied predictions and split; the evaluator does not establish
how predictions were trained or whether the split was kept out of training.

## Trainer integration

The trainer records actual quantization, uploads, gradient generation, per-node
histograms, split search, routing, prediction, and loss evaluation. The count
probe remains a separate measurement contract. Parent-child histogram subtraction
and checkpoint stage names are reserved; the trainer does not fabricate samples
for operations it does not perform. Comparisons with complete boosting systems
and representative real datasets remain necessary for competitive claims.
