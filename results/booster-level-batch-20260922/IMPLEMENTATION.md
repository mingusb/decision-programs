# Output-batch construction: implementation and validation

2026-09-22. The new `TreeBuildPolicy::output_batch` path builds independent
output trees one level at a time across a bounded tile. It remains **opt-in**;
the default is `TreeBuildPolicy::per_output`, with the previously promoted root
count/histogram/split policies intact. Original measured counting kernels,
policies and defaults are outside this change.

The pre-implementation selection, contracts and fair experiment were recorded
in [LEVEL_BATCH_EXPERIMENT.md](../../training/LEVEL_BATCH_EXPERIMENT.md).
This document reports implementation correctness and profiling observations.
Uninstrumented performance and held-out quality comparisons belong in the
experiment's main report; passing the checks below does not establish universal
speed, accuracy or arbitrary cross-run model equivalence.

## Production compute contract

The path retains the original independent-tree model architecture and objective
definitions. Quantization, training state, FP64 gradient/Hessian statistics,
uint64 counts, split decisions, stable child numbering, row routing and prediction
remain GPU-resident. Host setup, selector uploads and completed-tile model export
are boundaries of the operation. There is no host active-frontier read between
levels and no CPU training or preprocessing shortcut.

For tile capacity `B`, rows `N`, features `F`, total feature bins `H`, frontier
capacity `C` and per-tree node capacity `P`, retained storage is output-major:

| Buffer | Extent |
|---|---|
| Row assignments | `B × N` signed indices |
| Histograms | `B × C × H` Stats |
| Split candidates / winners | `B × C × F` / `B × C` Splits |
| Frontiers and child maps | Four `B × C` signed-index arrays |
| Scan offsets / block counts | `B × C` / `B × ceil(C/1024)` unsigned arrays |
| Tree nodes / state / packed active counts | `B × P` / `B` / `B` |

The device `OutputBatch` selector carries output beginning/count and derivative
beginning/stride. It supports full, short and empty primitive graph replays;
inactive outputs cannot consume stale active state. Independent regression and
binary objectives use compact row-major derivative tiles. Multiclass retains
its full frozen pre-round derivative snapshot before any output prediction is
updated. The exact multiclass fixture explicitly exercises tile width three and
a short final tile.

Root statistics write directly to each output's first retained histogram slot.
This removes separate root Stats/winner caches and their copies in this path.
Immutable root bin counts are still computed once when the selected count policy
reuses them. Deeper global histograms use the previously measured output-batched
primitive; explicit shared mode remains available subject to its capacity limit.
The output-batch automatic deeper policy currently selects global without
running a new timing tuner.

Split searches batch independent `(output,node,feature)` tasks with device-owned
active counts. Existing arithmetic and tie rules remain, including the owned
block fallback for shapes outside the warp path. For `C <= 1024`, one CTA per
output performs the integer split scan, capacity/status checks and materialization.
Larger frontiers retain separate batched scan, prefix and write stages. Routing
and advance stay separate because routing reads the old active count across
multiple CTAs. Prediction applies each completed independent tree to its output.

Kernels are allocation-free and asynchronous. CUDA C++23 is used. No CUB,
Thrust or external booster algorithm was introduced into production or fallback
paths. CUDA runtime and the existing profiling/instrumentation infrastructure
remain infrastructure.

A separate read-only source review found no issue in the inspected integration
boundaries: allocation accounting matches owned device arrays; the planner
reduces tile width before frontier capacity; graph replay uses the live selector;
multiclass derivatives remain frozen and compact tails use their actual stride;
completed-tile waits protect selector/pinned-buffer reuse; and stream draining
protects allocation lifetimes on exceptions. This review supplements the tests
and traced observations rather than proving the absence of all defects.

## Memory and export accounting

The planner first reduces `B` while retaining the full desired `C`. It reduces
`C` only if even one full-capacity output cannot fit. If an actual frontier then
exceeds its available capacity, the device records an overflow and export raises
an error; trees are not silently truncated.

On this ABI, Stats is 24 bytes, Split 48, Node 32, TreeState 24 and OutputBatch 16.
The reported payloads are checked against independently reconstructed formulas:

- `histogram_bytes = 24 B C H + root_count_bytes`, with `root_count_bytes = 8H`
  for a reuse policy and zero otherwise.
- `tree_state_bytes = B(4N + 32P + 20C + 4 ceil(C/1024) + 28)`.
- Candidate and winner storage is `48 B C (F+1)` bytes.
- `gradient_bytes = 16 N B` for independent outputs and `16 N K` for `K` coupled
  multiclass outputs. Predictions remain a dense `8 N K` payload.
- `device_bytes` also includes resident quantized data, targets, optional weights,
  base scores, initialization/loss workspace, selector/status and optional
  frontier history. History adds `4 B max(1,depth)` bytes when recording stream
  execution. CUDA/recorder internal bookkeeping is excluded from payload counts.

Separate root-cache fields are zero in this path; their storage is not charged
twice. Preparation scratch is released before trainer allocation, so the owned
peak is `max(preparation_peak_bytes, device_bytes)`. Both explicit budgets and
available device memory bound allocation. Tests cover a budget that reduces a
requested tile of 16 to 3 while preserving the full frontier, a one-byte-lower
boundary selecting 2, device-budget limits, and rejection below root capacity.

Completed tiles export in round/output order. Bounded pinned node chunks trade
bytes for fewer waits; compact export first obtains completed states and then
copies exact node extents. Reported pinned export bytes include those node
chunks, all tile states, the selector and optional history. Model export occurs
after tree construction; it is not a CPU split-decision stage.

## Mandatory correctness and sanitizer results

[ctest-contract.stdout](ctest-contract.stdout) records **13/13 passing suites**
(10 GPU suites and 3 CPU suites), run serially. Its
[command receipt](ctest-contract-command.json) records exit zero. The mandatory
full-training contract is detailed in
[BATCH_TRAINING_VALIDATION.md](../../training/BATCH_TRAINING_VALIDATION.md).

Primitive tests cover independent output state, integer child-order preservation,
malformed/nonfinite splits, sticky status, node/frontier overflow, guards,
inactive outputs, invalid assignments, graph selectors, missing/categorical
routing and the larger-frontier fallback. Fixed-histogram split results compare
bitwise against the existing arithmetic, including ties and fallback.

Full-training tests cover all three objectives, depths 0/2/5, bins 17/64,
weighted/missing/categorical features, scalar and 33-output cases, full/short
tiles, both submission modes, compact/bounded export, serialization and memory
limits. Each model's GPU prediction must agree with its exported CPU traversal.
At every round, its recorded GPU objective must agree with an independent
weighted CPU objective of the corresponding exported forest. Emitted frontier
history is reconstructed from exported topology; stage counts and short-tile
operation counts are checked.

Additional mandatory **zero-allowance cross-policy** fixtures use exactly
representable one-round derivative sums:

| Objective | Exact construction |
|---|---|
| Regression | Equal-weight opposite dyadic targets; base score zero |
| Binary | Equal-weight complementary labels; initial `p=1/2`, `h=1/4` before weighting |
| Eight-class softmax | 32 unit-weight rows per class; equal base margins, `p=1/8`, `h=7/32` |

These use depth five, bins 17/64, tile width three, stream/graph execution and
short tails. Every legal encoded tuple is checked, including bin zero. Exact
per-feature statistics make atomic arrival order irrelevant on these fixtures,
so cross-policy whole-bin function equality is a justified correctness test.

**13 Compute Sanitizer runs passed:** memcheck, racecheck and synccheck for each
of root histograms, split search, deeper histograms and batched resident state
(12 runs), plus full trainer memcheck (one run). All command receipts have exit
zero and unchanged executable hashes. Memcheck/synccheck report zero errors;
racecheck reports zero errors and zero warnings. The final trainer check records
7,718,425 checks and 350,800 exhaustive bin tuples in
[memcheck-batch-training.stdout](memcheck-batch-training.stdout).

## Preserved stronger comparison failures

The initial 12/13 test result is retained in
[ctest-initial.stdout](ctest-initial.stdout). That test assumed identical
topology across unordered FP64 training runs. Some differing splits were proved
functionally equivalent over every legal encoded tuple, but a subsequent
multiclass example **failed** that stronger certificate. The diagnostic source,
failed executable snapshots, hashes, receipts and raw stderr were retained.

The unchanged per-output trainer then failed against itself on the same
multiclass fixture. The first requested repeat failed; this receipt is not
evidence of eight completed comparisons. Its training margins were identical,
while the maximum held-out margin difference was
**0.050911730395435617**. The output-batch-versus-baseline diagnostic also failed
on its first requested comparison with that same held-out difference. Both
[baseline-repeat-diagnostic-command.json](baseline-repeat-diagnostic-command.json)
and [batch-targeted-diagnostic-command.json](batch-targeted-diagnostic-command.json)
record exit one with the same unchanged test-binary hash.

The saved [baseline witness](baseline-repeat-failure/failure.json) uses tree 9,
legal bins `[8,0,0]` and thresholds 7 versus 8 at node 3. It reaches leaves
`-0.026312378670106464` and `0.024599351725329198`, a difference about 0.05091173.
The 24 training rows reaching that node contain feature-zero bins
`{1:3, 2:1, 5:2, 6:5, 7:3, 9:3, 10:3, 11:4}`; bin 8 is absent locally.
Thus the thresholds give the same local training partition but different
predictions for a legal unseen combination. Both full models, exact fixtures
and configurations are retained in the two failure directories.

This is **not a small prediction-rounding error**. Small statistic changes can
alter a discrete split on locally empty bins. The final mandatory trainer
memcheck observed two general comparisons outside the rounding bound, with a
maximum held-out raw-margin difference of **0.24877399999999994**, while its
maximum training-margin difference was `1.8804402479588589e-15`. The 0.05091173
witness is therefore not a bound on general prediction differences. The
0.248774 observation is a raw-margin comparison, not an accuracy or loss metric.

The corrected mandatory suite checks the actual unordered implementation
contract and reports these differences as observations. It does not certify
arbitrary runs as identical. `--comparison-diagnostic`, `--targeted-repeat` and
`--targeted-batch` retain the stronger cross-run/all-bin assertions with the
same bounds and evidence dumps. No tolerance was raised, no production tie
canonicalization was added and no failed quality gate was relabeled as passed.
Baseline instability does not excuse a candidate's held-out quality regression.
The strict zero-allowance metric evaluator remains separate and unchanged.

## Nsight Systems evidence

The [audited captures](systems-audit.md) cover 4,096 rows, 16 features, 129 outputs,
three rounds, depth three and 32 bins with output width 16. All represent the
same 387 independent trees. The audit matches **1,442** emitted stage IDs/names
to NVTX ranges and checks full/short tile operation counts.

| Observation | Per-output graph | Output-batch graph | Output-batch stream |
|---|---:|---:|---:|
| Complete boosting kernels including immutable-count setup | 9,817 | 655 | 655 |
| Tree-build scopes | 387 | 27 | 27 |
| Tree graph enqueue calls | 387 | 27 | 0 |
| Completed-tile export scopes / stream waits | 27 / 27 | 27 / 27 | 27 / 27 |
| Tree-scope host waits / D2H / device allocations | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

The complete-operation kernel boundary includes legacy root work outside
per-tree scopes; comparing tree-only totals would omit that work on one side.
Correlation IDs attribute enqueues, kernels and copies even when GPU execution
finishes after the host NVTX range ends. Individual tree scopes also contain
zero synchronous copies and zero device frees. Setup activity before the first
tree is reported separately rather than incorrectly counted as an in-tree wait.

For the output-batch graph capture, deeper histograms account for about 55.69%
of summed tree kernel time, split search 21.86%, root histograms 15.82% and
frontier materialization/routing/advance 4.25%. These are diagnostic shares,
not end-to-end speedup bounds. Profiler durations do not rank uninstrumented
implementations. The audit verifies traced CUDA runtime APIs and correlations;
it does not certify absence of arbitrary CPU work or untraced driver calls.

The implementation is a measured candidate with explicit correctness and
quality boundaries. The new tree-build policy remains opt-in while those
performance and quality results are assessed.
