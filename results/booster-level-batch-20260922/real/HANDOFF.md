# Root-only execution handoff

Environment: `build/benchmark-env/bin/python` (Python 3.12.14). Versions and shared
library hashes are in `environment.json`, exact package list in
`requirements.lock.txt`. The LightGBM 4.7.0 CUDA/SM86 build is installed; its NCCL
shared library search path is set by campaign/smoke runners. No reference libraries
are linked into the custom trainer. Source-build retries are retained.

`training/bench/real_data.cpp` is the standalone `ghb_real_bench` source. Root owns
its CMake target. It consumes the binary format documented in `PROTOCOL.md`, uses
the production CUDA train/predict entry points, and writes native `model.ghb`,
`predictions.f64`, and timing/config/memory `metrics.json`. Syntax checked with
GCC C++23. Four CPU adapter/quality tests passed (`cpu-tests.stderr`).

Fixtures are in `../data/fixtures/{wine,magic,letter,delicious}/{train,validation,test}.ghb`.
All sources, row indices, split target summaries and hashes are preserved. Delicious
contains all 983 target columns and all 500 input features. Its source XML misspells
one label relative to both ARFFs; only the label spelling was reconciled, with no
column movement, omission, or numeric change. The manifest records it explicitly.

Current capability smoke is a GPU workload; run only when no other GPU work is
active. First smoke exposed two adapter/config issues and a constant-class row
prefix, all documented in the protocol and preserved in failed-smoke-sources.
Corrected smoke selects seeded rows and includes every scalar class:

```
build/benchmark-env/bin/python results/booster-level-batch-20260922/real/capability_smoke.py \
  --custom-binary build/booster-level-batch/ghb_real_bench \
  --output results/booster-level-batch-20260922/real/capability-smoke-fixed \
  --implementations xgboost catboost
```

The validation campaign has 80 jobs (four datasets, five implementations, four
configurations). It runs jobs serially with CPU quality evaluation between GPU
processes, saves raw stdout/stderr, parameter/fixture/source/binary hashes and exit
codes, and refuses pre-existing destinations:

```
build/benchmark-env/bin/python results/booster-level-batch-20260922/real/campaign.py \
  --stage validation --custom-binary build/booster-level-batch/ghb_real_bench \
  --run-root results/booster-level-batch-20260922/real/validation
```

Four-config validation must complete before selection. The test campaign chooses
each implementation's configuration solely from validation loss, with training
time only as a tie break, then produces 60 selected test runs (three repetitions):

```
build/benchmark-env/bin/python results/booster-level-batch-20260922/real/campaign.py \
  --stage test --custom-binary build/booster-level-batch/ghb_real_bench \
  --validation-root results/booster-level-batch-20260922/real/validation \
  --run-root results/booster-level-batch-20260922/real/test
```

Both stages accept `--datasets` and `--implementations` to execute bounded subsets
in separate new roots; never select a framework from fewer than all four configs.
LightGBM's Delicious job trains every one of the 983 labels in one serial wrapper
with a reused Dataset. Its actual complete time is reported without extrapolation.

For a separate memory diagnostic, wrap a selected command in
`memory_observation.py --output NEW_DIRECTORY -- COMMAND...`. The command itself
must have a new output-dir. This records timestamped device-wide memory.used
samples, before/after/sample peak, and the observation's exact scope. It is a lower
bound on instantaneous peak and includes driver/context/other process allocations.
Its instrumented timings must never enter rankings. If calling framework.py
directly, export the NCCL library directory in LD_LIBRARY_PATH as in campaign.py.

Limitations carried into every report: native tree/objective/cut methods differ;
CatBoost uses shared symmetric vector leaves; LightGBM's public prediction API is
CPU, explicitly labeled; process RSS/custom owned payload/device-wide samples have
different scopes; MAGIC is published physics simulation data; bounded shallow
tuning is an initial comparison, not saturated predictive accuracy or an NLP model.
