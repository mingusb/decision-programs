# D2 frozen inference and encoding-stage campaign

Prepared before GPU execution on 2026-09-23. Selection and exact contracts are
in `training/FINAL_STATUS_ENCODING_EXPERIMENT.md`. These are evidence helpers,
not production algorithms. No default changes or quality allowances are made.

`run_campaign.py` fixes 28 complete-prediction cases: the exact 25 E cases
(including existing immutable zero/one-tree model files), plus:

| Added case | Rows | Features | Outputs | Dictionary pattern |
| --- | ---: | ---: | ---: | --- |
| wide1025 | 4,096 | 1,025 | 3 | repeating numeric31/category17/empty numeric/empty category/numeric1 |
| large67 | 262,144 | 67 | 3 | same normal pattern |
| skew257 | 8,192 | 257 | 3 | first feature 65,535 categories; other features empty or one entry |

The three synthetic models have independent binary objectives and three
three-node trees, splitting on the first/middle/last feature. They are fixed
validation fixtures, not trained models or evidence of model quality.
Metadata/model bytes and generator provenance are preserved. Inputs use the
existing deterministic inference generator, with each actual FP32 input
fingerprint retained. The 25 existing model/input fingerprints must exactly
match the frozen E protocol. Raw and transformed predictions are checked
against per-tree/per-tile before timing; every measured output is checked too.

The prediction executable and both linked libraries are frozen under
`build/final-status-20260923`, with explicit byte-hash mappings to all 95 files
in `source-snapshot/manifest.json`. The campaign validates and retains all
snapshot identities. Subsequent functional-layout refactoring in the current
working tree belongs to a separate build and is not claimed as the source of
these D2 binaries. Current-source hashes are supplementary stability checks.
The separately compiled encoding helper uses four transitive project headers;
all four must still match that frozen snapshot, and it links those frozen
libraries. Retain its compiler command/log when root builds it.

The four separate encoding cases are normal67/257rows at width1 and width32,
wide1025/257rows at width32, and skew257/4099rows at width32. Each allocates the
exact documented resident-plus-scratch budget for that width, validates all
bins against an explicit independent CPU lower-bound reference, validates
metadata bits/types/offsets and byte accounting, and verifies rejection at one
byte below minimum budget. Every measured result is verified outside timing.

Prediction measures the complete synchronous `predict_gpu` call, exactly as
the production benchmark documents. Encoding measures `encode_quantize`
entry through synchronous return, including its internal scratch release but
excluding destruction of the returned resident allocation. Its independent
CPU reference, verification downloads, stream/context setup and result
destruction are outside timing. This is stage evidence, not an end-to-end
claim; do not pool it with complete-prediction results.

Both comparisons alternate reference/candidate order for at least 15 pairs
after 3 warmups per policy. All raw streams, terminal process receipts,
partial observations, hashes and before/after telemetry are retained. Every
output directory must be new. Any nonzero process exit, timeout, missing
receipt, failed exactness or source/input drift fails the campaign. An absent
completion receipt is not a successful job, even when timing JSON exists.
Stop concurrent builds and CPU/GPU agents before unprofiled ranking. Residual
desktop rendering is an observed condition, not grounds for claiming idle.

Root-only build of the host-side helper (links the already-built CUDA C++23
library; this helper's CPU computations are validation/setup only):

```bash
/usr/bin/c++ -std=c++23 -O2 -Wall -Wextra -Wpedantic \
  -Itraining/include -I/usr/local/cuda/include \
  results/final-status-20260923/encode_bench.cpp \
  build/final-status-20260923/libghb.a \
  build/final-status-20260923/libghb_instrumentation.a \
  -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 \
  -lcudart -lcudadevrt -ldl -lrt -pthread \
  -o results/final-status-20260923/encode_bench
```

After correctness/sanitizer gates and freezing source/builds, root runs:

```bash
python3 results/final-status-20260923/run_campaign.py \
  --output results/final-status-20260923/campaign-observed \
  > results/final-status-20260923/campaign-observed.log 2>&1
python3 results/final-status-20260923/summarize_campaign.py \
  --campaign results/final-status-20260923/campaign-observed \
  --output results/final-status-20260923/performance-summary.json \
  > results/final-status-20260923/performance-summary.log 2>&1
```

Optional `--scope prediction` or `--scope encoding` selects the entire
predeclared subset in its own exclusive directory; it cannot select particular
cases after observing timings. Default `all` runs 32 GPU processes serially.
Summarizer uses only the Python standard library and verifies process receipts,
raw artifact hashes, protocol correspondence, source identity stability,
exactness and medians recomputed from raw pairs. It retains every paired ratio
and reports casewise 95% percentile-bootstrap intervals (20,000 draws,
preselected seed2026092303), without multiplicity/global-win claims.

No workload or build was run while preparing these helpers. Root must compile,
review, execute diagnostics, then measure. The helpers do not add profiler
capture to encoding-stage timing; use the production prediction benchmark's
isolated policy modes for the planned same-call Nsight work-count comparison.
