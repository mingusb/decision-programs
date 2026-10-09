# Decision Programs command reference

`decision-programs` is the public entrypoint. Run `decision-programs --help` or
`decision-programs COMMAND --help` for arguments. Linux and WSL are supported.
Numerical training, conversion, simplification, validation, prediction and path
tracing run in C++23/CUDA. The frontend handles arguments, hashes and input-format
transport.

## Common workflow

```sh
decision-programs doctor --gpu
decision-programs demo

decision-programs train --data examples/quickstart.csv --target label \
  --rounds 2 --depth 2 --output trained
decision-programs convert --model trained/model.json --output compiled
decision-programs predict --model compiled --data examples/predict.csv
decision-programs inspect --model compiled
decision-programs explain --model compiled --data examples/predict.csv --output paths.json
decision-programs export --model compiled --output equations.json
```

Use a fresh output path for every run. `--output` and `--out` are equivalent.
`demo` creates a fresh temporary directory and prints its location when no output
is supplied. Model directories automatically select `model.canonical` and its
compact companion. A directory with multiple numbered trials requires an explicit
model file.

Training and native-source conversion require CUDA-enabled XGBoost **3.4.1**.
Library discovery uses `--library PATH`, then `XGBOOST_LIBRARY`, then the active
Python installation's `xgboost` package. Saved plans may already contain an exact
library binding. Runtime prediction/inspection/export does not require XGBoost.

### Input formats

CSV training uses `--target HEADER_NAME`, a zero-based column index, or `last`.
The target can contain nonnegative integer class IDs or strings. String labels
receive stable first-occurrence class IDs; the output's `input-transport.json`
retains the mapping. Conversion carries this mapping forward. CSV prediction
contains feature columns only, in training order. Headers are detected
automatically; `--header yes|no|auto` overrides detection. Quoted fields, escaped
quotes and CRLF are accepted. Empty, `NaN`, `NA` and `null` features become NaN;
infinity and malformed features are rejected.

Raw FP32 training avoids CSV transport:

```sh
decision-programs train --values inputs.fp32 --labels labels.u32 \
  --rows 1000 --features 4 --classes 3 --rounds 10 --depth 4 --output trained
decision-programs predict --model compiled --values inputs.fp32 --rows 1000
```

Files use little-endian FP32 features and uint32 labels. `--row-stride N` supports
padding. Alternatively pass `--data DESCRIPTOR.json` using the maintained
`dense-fp32-u32-class-labels-1` descriptor. JSON prediction input is
`{"rows":[[1,2],[null,3]]}`; pass it with `--input FILE.json` or `--data FILE.json`.

CSV transport retains immutable input files under
`$XDG_CACHE_HOME/decision-programs/inputs`, or `$HOME/.cache/decision-programs/inputs`.
`DECISION_PROGRAMS_CACHE_DIR` overrides the cache root. Source content, target,
header mode and label mapping determine the identity. Existing cached bytes must
match exactly. These stable paths let checkpoint declarations survive a new
process. Retained CSV files are input artifacts, not CUDA model working memory.

## Commands and backend scope

| Command | Behavior | Maintained backend |
| --- | --- | --- |
| `train` | FIT-only native CUDA training; rounds, depth, regularization and sampling flags | `class_model_train` |
| `convert` | Saved multiclass native-model conversion with explicit resource limits | `class_model_convert` |
| `import-tree` | Structural CLSTREE1 import with a bound origin/domain contract | `class_tree_adapter` |
| `study` | Fixed trials: train, convert, simplify, then evaluate | `class_study` |
| `hpo` | Declared native trials selected on VALID | `class_study`, `native-accuracy-search-1` |
| `combine` / `nonlinear` | OOF nonlinear teacher composition, VALID selection and optional nested holdout confirmation | `class_study`, `native-nonlinear-combination-1` |
| `rl` | Qualified experimental regional equivalent-encoding policy search | `rl_session` |
| `evaluate` | Compare runtime/native class IDs and FIT/VALID scores | `class_model_evaluate` |
| `simplify` | Exact adjacent-predicate shared-DAG rewrites | `class_model_simplify` |
| `predict`, `inspect`, `explain`, `export`, `demo` | Generic persisted-model tools through the shared Runtime | `decision_programs_model_tools` |
| `checkpoint`, `resume` | Checkpoint signaling, metadata inventory and backend recovery | Configured live backend |
| `proofs` | Formal source inventory, maintained Lean elaboration and kernel replay | Portable `CheckLean.cmake` |
| `profile` | One installed profiler/sanitizer/debugger around a public command | Selected external tool |

Internal backend names are implementation details. The installed frontend finds
them beside its binary or under `../libexec/decision-programs`.
`DECISION_PROGRAMS_BACKEND_DIR` provides an explicit alternate backend directory.

### Conversion and simplification

```sh
decision-programs convert --model trained/model.json --output compiled \
  --max-nodes 1048576 --max-states 1048576 --gpu-byte-budget 1073741824
decision-programs simplify --model compiled --source trained/model.json \
  --max-passes 4 --output simplified
```

Conversion supports `--batch-size`, `--max-expansions`, `--split-policy`, runtime
residency, checkpoint and proof-module flags. A bounded or interrupted conversion
can remain incomplete; its receipt determines completion. A completed conversion's
native-class preservation applies to the **declared conversion domain**, including
any one-hot restrictions. Runtime tools report the supported input representation
separately and establish no new native-source equivalence outside that domain.

Joint bounds and rival covers are enabled by default; `--no-joint-bounds` and
`--no-rival-covers` disable them. Relational and grouped unary bounds are optional:
use `--relational-bounds` or `--unary-bounds` to enable them, or the corresponding
`--no-...-bounds` flag to disable them explicitly. Supplying both forms of one
option is rejected. These methods share the existing native qualification rule.
The default [two-point screen](TWO_POINT_SCREEN.md) can avoid further proof search
when two qualified feasible inputs have different classes; ordinary construction
still follows, and no extra public option is needed.

Simplification retains the source binding. It may stop before a fixed point when
the pass budget is exhausted. Native comparison is a separate evaluation:

```sh
decision-programs evaluate --model simplified --source trained/model.json \
  --data evaluation.csv --target label --fit-rows 100 --output evaluation
```

Evaluation CSV string labels reuse the source/model's `input-transport.json`, or
an explicit `--label-map FILE`. Missing or unseen mappings are rejected. Numeric
labels remain direct class IDs when the trained mapping is numeric; numeric-looking
tokens in a categorical mapping retain their original mapped IDs. Evaluation needs
both FIT and VALID rows and performs no training, selection or TEST access.

### Studies and composition

```sh
decision-programs hpo --data evaluation.csv --target label --fit-rows 100 \
  --rounds 10 --depth 3 --trial '{"max_depth":2}' --trial '{"max_depth":4}' \
  --output native-search
```

Trials override the base hyperparameters. HPO selection orders candidates by VALID
errors, native model bytes, then declared order. Reused VALID scores do not measure
independent generalization.

Fixed `study` accepts `--train-data FIT_DESCRIPTOR.json` plus `--data
EVAL_DESCRIPTOR.json`, or an advanced `--plan PLAN.json`. Its construction finishes
before deferred VALID evaluation.

Composition requires completed HPO source checkpoints. The following small
workflow uses the shipped synthetic CSV, with 24 FIT rows and six VALID rows.
First prepare both teacher sources and retain their complete checkpoint. Then
select the one-round prefix baseline from those exact sources:

```sh
decision-programs hpo --data examples/quickstart.csv --target label --fit-rows 24 \
  --rounds 2 --depth 2 --trial '{"max_depth":2}' --trial '{"max_depth":3}' \
  --checkpoint teacher-checkpoint --output teacher-sources
decision-programs hpo --data examples/quickstart.csv --target label --fit-rows 24 \
  --rounds 1 --depth 2 --trial '{"max_depth":2}' --trial '{"max_depth":3}' \
  --training-checkpoint teacher-checkpoint --output teacher-prefixes
decision-programs combine --data examples/quickstart.csv --target label --fit-rows 24 \
  --teachers '[{"rounds":1,"max_depth":2},{"rounds":1,"max_depth":3}]' \
  --meta '[{"rounds":1,"max_depth":2}]' --baseline teacher-prefixes/selected-model.json \
  --training-checkpoint teacher-checkpoint --folds 2 --output composition
```

The second command reuses the completed sources instead of fitting new models.
Both it and `combine` slice the same source models to the declared teacher rounds,
so the selected prefix baseline is retained in the teacher bank. An unrelated
`train` output is not a compatible baseline merely because its parameters match.

`--training-checkpoint DIRECTORY` is repeatable for `hpo` and `combine`. Source
checkpoints must be complete native HPO checkpoints with exactly matching dataset,
FIT/VALID split, native-library binding and normalized training parameters; their
round count must cover the requested prefix. The backend verifies these bindings
and the baseline's exact identity. `--checkpoint` names a new workflow's output
checkpoint, while `--training-checkpoint` supplies completed input models.

OOF settings, baseline provenance, prefix reuse and frozen refit/final-evaluation
workflows are also available through advanced plans. These tiny commands exercise
the interface; composition results do not imply improved TEST accuracy.

### Nested holdout confirmation

`combine` optionally confirms its frozen inner VALID winner against the required,
predeclared baseline using a separate fresh holdout dataset:

```sh
decision-programs combine --data development.csv --target label --fit-rows 100 \
  --teachers '[{"rounds":1,"max_depth":2},{"rounds":1,"max_depth":3}]' \
  --meta '[{"rounds":1,"max_depth":2}]' --baseline teacher-prefixes/selected-model.json \
  --training-checkpoint teacher-checkpoint --folds 2 \
  --nested-holdout --holdout-depth 2 --holdout-data fresh-holdout.csv \
  --output confirmed-composition
```

The source checkpoint and baseline must still match the development dataset and
declared teacher prefixes as described above. `--nested-holdout` defaults to depth
one. Supplying `--holdout-depth N` or `--holdout-data FILE` also enables
confirmation. Every enabled invocation requires `--holdout-data`; depth must be a
positive integer no larger than the holdout row count. These options apply only to
`combine` and its `nonlinear` alias; `hpo` and fixed `study` reject them.

The backend divides the fresh holdout rows into N disjoint, balanced, contiguous
blocks in input order, with block sizes differing by at most one row. At each gate
the same frozen inner winner must make **strictly fewer errors** than the same
required baseline. A tie or loss selects the baseline and stops confirmation. The
candidate is retained only when it passes every gate. Holdout labels never fit or
retrain a model, rank another candidate, or trigger a retry. Numerical evaluation
of both models runs on the GPU. Existing FIT/VALID HPO selection and OOF fitting
retain their existing behavior.

This is nested holdout confirmation, not full nested cross-validation or a final
TEST evaluation. Reserve a separate final TEST dataset for evaluating the selected
model once. Freshness and appropriate group, patient, or institution separation
remain the caller's responsibility. Repeated search consumes the holdout: reusing
its labels for later search decisions does not provide independent confirmation.
Increasing depth leaves fewer samples per gate. Passing the gates is neither a
statistical significance test nor a guarantee of improved TEST performance.
The interface rejects shared development/holdout source paths, reused dense input
paths, and copies with identical feature contents; those checks cannot establish
that two datasets have independent provenance.

CSV holdout transport uses the same `--target` and `--header` settings as the
development CSV. Feature names and order must match exactly. Labels reuse the
development class mapping and full class count, including classes absent from the
holdout. A numeric-looking token in a categorical development mapping keeps its
original mapped ID. Unknown categorical labels or numeric IDs outside the
development class range are rejected. For non-CSV development inputs, supply the
original `--label-map INPUT_TRANSPORT.json` when the holdout CSV uses mapped string
labels. The frontend converts input formats on the host without model evaluation.

`--holdout-data HOLDOUT_DESCRIPTOR.json` accepts little-endian FP32 features and
uint32 class IDs already in development order. The dedicated descriptor declares
the role explicitly:

```json
{
  "format": "dense-fp32-u32-holdout-labels-1",
  "role": "HOLDOUT",
  "training_allowed": false,
  "selection_allowed": true,
  "TEST_read": false,
  "features": 4,
  "classes": 3,
  "rows": 40,
  "row_stride": 4,
  "values_path": "/absolute/fresh-holdout.fp32",
  "values_sha256": "SHA256_OF_EXACT_FEATURE_FILE",
  "labels_path": "/absolute/fresh-holdout.u32",
  "labels_sha256": "SHA256_OF_EXACT_LABEL_FILE",
  "preprocessing": "Same input representation as development; class IDs preserve its mapping"
}
```

Replace the SHA placeholders with the actual 64-character hexadecimal hashes.
Feature and class dimensions must equal the development descriptor. No `FIT_rows`
or `VALID_rows` fields are allowed, including zero counts. The CUDA backend
validates file pins, extents and class-ID bounds before evaluation.

The generated reviewable plan contains
`nested_holdout: {"depth": N, "dataset": HOLDOUT_DESCRIPTOR,
"acceptance_rule": "strictly_fewer_errors_at_every_level"}`. Advanced plans use
the same object; no alternative acceptance rule is supported. Use `--dry-run` or
`--save-plan` to inspect and retain it before numerical execution.

For a holdout arranged into contiguous groups, an advanced plan can declare
`nested_holdout.partition_rows` to place gate boundaries between groups. For
example, a depth-two holdout with 40 rows can use
`--set '/nested_holdout/partition_rows=[18,22]'`. Supply one positive count per
level, summing to the holdout row count. The backend preserves the declared row
order and boundaries; arranging independent groups remains the caller's job.
Omitting this setting retains the balanced row blocks described above.

With `--checkpoint`, final saves, explicit checkpoint requests, and stop saves
retain frozen model identities, the gate cursor, and committed results. Resuming
committed decisions does not rescore those gates. A crash before checkpoint
publication may repeat evaluation of the same frozen pair on an uncommitted gate.

### Prediction and scientific interrogation

Generic tools accept canonical CLSGDAG1 models, with an optional CLSG64B1 compact
companion. `predict` returns class IDs. `explain` returns the exact visited shared
nodes, raw threshold words, NaN direction, branches and terminal. The default path
storage limit is 4096 nodes per row; `--max-path-nodes N` changes it. An insufficient
capacity is rejected rather than presenting a partial path as complete.

`export --kind equations` preserves shared subfunctions, exact raw threshold bits,
strict FP32 comparisons and stored NaN directions. `export --kind compact` produces
GPU-packed runtime bytes. Equation export establishes exact representation of the
bound runtime; it does not newly prove source equivalence or human clarity.

Specialized regional `explain/export --format regional` accepts CLSRMDL1 models
under the existing 54-feature rank/category contract. Regional explanations require
the original ten-temperature input schema. Fuzzy scores/gradients describe a
distinct mathematical model and are not native probabilities or calibrated
uncertainty. Generic fuzzy gradients, counterfactuals, causal effects and global
importance remain pending; `capabilities` lists their status.

### RL, checkpoints and profiling

```sh
decision-programs rl --model forest-teacher.json --episodes 32 --output policy-search
decision-programs convert --model trained/model.json --checkpoint converter-checkpoint \
  --output first-attempt
decision-programs resume convert --model trained/model.json \
  --resume converter-checkpoint --output resumed
decision-programs checkpoint --pid LIVE_PUBLIC_COMMAND_PID
decision-programs checkpoint --directory converter-checkpoint
```

RL is specialized to ten numeric features, wilderness4 and soil40 one-hot groups,
and seven classes. Its policy orders equivalent encodings; it does not grant class
authority. `--learning-rate` must lie in `(0,1]`. `rl --resume POLICY.json` is a
**policy-word warm start** into a fresh session, table, schedule and warmup. It is
not optimizer/session checkpoint continuation.

Conversion and experiment checkpoints require unchanged semantic declarations
and a fresh final output. `--checkpoint` enables saving; `--resume` supplies an
existing generation. Whole fixed studies, native HPO and nonlinear OOF workflows
support their implemented stage-boundary recovery. Native fitting recovery covers
completed calls, not an in-flight XGBoost update. A SIGUSR1 request or checkpoint
file inventory is not evidence that a generation committed or resumed successfully.

```sh
decision-programs proofs --check --output proof-reports
decision-programs profile --tool memcheck -- demo --output sanitized-demo
decision-programs profile --tool nsys --output trace -- \
  predict --model compiled --data examples/predict.csv
```

Proof checks require Lean 4.34.1, `leanchecker` and CMake, then compile the maintained
module suite in dependency order and replay the kernel artifacts in a fresh report
directory. Abstract Lean theorems are not a CUDA implementation-refinement proof.

Profiling supports Nsight Systems/Compute, all four Compute Sanitizer modes and
CUDA-GDB. Other installed profiling routes can use an absolute `--tool` executable
and repeated `--tool-arg ARG`. Exactly one tool wraps the command. Availability is
not collection success; inspect exit status and retained reports. Instrumented
timings do not establish an uninstrumented speedup.

## Advanced, reviewable plans

`--dry-run` prints the exact backend argv and generated JSON without numerical
execution. `--save-plan FILE` retains the exact generated plan; CSV artifact paths
remain stable. `--set /json/pointer=JSON_VALUE` exposes all supported backend
settings while their existing validators reject unknown or retired settings.
Strings need JSON quotes, for example:

```sh
decision-programs convert --model trained/model.json --output compiled \
  --set '/split_policy="widest_residual"' --save-plan conversion.json --dry-run
```

The host-only interface checks run without CUDA:

```sh
python3 native/decision_programs_cli_checks.py build/cuda-release/bin/decision-programs
```

They verify command help, installed relocation, automatic hashes, paths containing
shell metacharacters, CSV conversion/mappings, stable saved/checkpoint declarations,
model-directory discovery, existing-output preservation, corrupt-cache refusal,
invalid metadata refusal and RL argument forwarding. Numerical qualification uses
the real CUDA demo and maintained backend tests.
