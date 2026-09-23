# Frozen prediction conformance runner

Preimplementation contract and choice, 2026-09-23.
`quality DATA MODEL PREDICTIONS [QUALITY.json]` accepts three or four file paths.
Host code transports opaque bytes, allocates one fixed
512 MiB working arena plus the input blobs, launches one bootstrap and
returns CUDA completion status. File lengths are transport metadata; the host
does not inspect headers, choose a task, calculate a fixture, compare a value or
compute a metric. All imported objects and intermediate arrays live on GPU.

The GPU decodes GHBDS001 and GHBMODEL v1 using the production codecs, including
model validation. Conservative capacities come from wire lengths; active extents
come only from successful decoding. Bounds are checked before partitioning or
use. Decoded model and dataset must agree on feature count, objective and target
shape; multiclass uses one target column and the declared class count. Binary
models with multiple outputs select the multilabel real-data metric profile.
Targets are promoted from binary32 to binary64 by a GPU kernel. The reference
file must contain exactly `rows * model.outputs * 8` little-endian bytes.

The frozen executable's `training/bench/real_data.cpp:187` called default
`Model::predict_gpu`, then wrote its transformed binary64 values directly at
lines 200–202. The reference therefore contains probabilities for classification,
not raw margins. The new runner encodes evaluation features with the imported
schema and predicts with `raw=false`. Every binary64 bit is compared, including
signed zero. It also reports numerical inequality, nonfinite references,
maximum absolute difference and the first differing cell. These diagnostics do
not relax the exact gate; a difference between opposite finite extremes may
overflow the diagnostic maximum to infinity. A bit failure does not skip the
subsequent metrics.

Both frozen and candidate predictions are evaluated with the same resident
real-data metric engine, with all available metrics, eligibility metadata and
strict zero-allowance comparison. Every metric's identity, availability, value
bits and verdict are printed by GPU code. Final success requires bit equality,
finite references, valid metric results and no metric regression. Device asserts
remain enabled in release builds, so a failed gate yields CUDA failure without a
host quality decision. There is no tolerance waiver or CPU numerical oracle.

The three-file mode checks **frozen prediction conformance and same-engine
quality comparison**. The optional fourth file enables the separate legacy
NumPy/scikit-learn metric arithmetic gate using the exact GPU decimal importer
specified in [reference.md](reference.md). Every applicable legacy value is
compared bitwise to metrics recomputed on frozen predictions, and a separate
directional zero-allowance gate is reported. Both must pass. The legacy schema's
absence of micro_auc is explicit; that additional metric remains in the
same-engine gate. An arithmetic failure does not suppress the subsequent
candidate metric report. Compilation alone proves none of these conformance
gates. Training quality, fixed-configuration retraining, synthetic metrics and
signal operating points remain separate acceptance work.

## Algorithm and lifetime choice

Choose a single GPU coordinator and tail continuations between decode, packing,
encoding, prediction, comparison and the two metric evaluations. The documented
[CDP2 memory and tail-launch rules](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html)
require consumers of child writes to execute in a tail continuation. Every
submission and tail launch checks its immediate runtime error; each API's
completion record is checked before reuse. Inputs, descriptors and reports use
global storage, never parent local/shared addresses. The scratch suffix is
reused only after checked completion; permanent arrays occupy its disjoint
prefix. No GPU heap allocation or assumed default heap capacity is needed.

Parallel conversion and bit comparison cost linear streaming traffic, with
atomics only for diagnostic counts/minimum index/maximum error. Serial scalar
comparison would remove atomics but adds millions of dependent iterations for
Delicious; copying results to host violates the computation contract. The
existing resident metric algorithms retain their own numerical schedules.
Both metric passes reuse scratch, avoiding a second ranking workspace. No
performance ranking is claimed. A future complete-operation experiment must
include decode/validation, conversion, encoding, prediction, both metric passes
and checked completion, and report transport separately. Profilers explain
timings; they do not rank uninstrumented runs.

## Authoritative four-case matrix

Let `A=/home/b/gpu_histogram-archive-20260923/results/booster-level-batch-20260922`.
For each dataset below, use:

- data: `$A/data/fixtures/NAME/validation.ghb`
- model: `$A/real/validation/validation-NAME-g0-custom-per-output/result/model.ghb`
- predictions: the same result directory's `predictions.f64`
- optional legacy metrics: the parent case directory's `quality.json`

| NAME | task | rows | features | outputs | rounds / depth |
| --- | --- | ---: | ---: | ---: | --- |
| wine | regression | 975 | 11 | 1 | 25 / 3 |
| magic | binary | 3803 | 10 | 1 | 25 / 3 |
| letter | multiclass | 3189 | 16 | 26 | 25 / 3 |
| delicious | multilabel | 2584 | 500 | 983 | 5 / 2 |

These are matched unchanged model/prediction pairs, not mutation-test artifacts.
The adjacent `capture.json` binds commands, binary hash, evaluation fixture and
prediction hashes; `result/metrics.json` records settings. The fixture manifest
at `$A/data/fixtures/manifest.json` records source hashes and split seed
2026092207. Preserve those provenance files and raw runner stdout/stderr/exit
status alongside each run. The runner does not authenticate file identities;
the externally preserved hashes establish which bytes were tested.

The frozen settings are bins 32, learning rate .1, L2 1, per-output construction,
output tile 16 and the captured histogram/execution settings. This mode never
retrain/reselects them. Invoke the executable separately for each triple; only
the root agent runs GPU work, serially. Compilation, actual execution, frozen
bit conformance, same-engine metrics and legacy-metric conformance must have
separate recorded outcomes.

## Preparation receipt

The initial runner compiled with CUDA 13.4, C++23, O3, sm86, RDC and assertions
enabled. Raw compile output/object are in `/tmp/gh-quality-compile`; a subsequent
read-only review identified one unused stored input descriptor, which was
removed before handoff. The root integration build must certify this final
source. No GPU run or frozen conformance result is claimed by this receipt.

The later optional JSON extension and new reference module require their own
compile and GPU execution receipt; the initial runner object does not certify
those changes.
