# Resident learning: first fresh implementation

## Contract and evidence, recorded before kernels

Input is validated feature-major u16 bins plus a fitted Schema; targets/weights
are resident float32, margins and derivative statistics FP64. Objectives are
weighted regression, independent binary/multilabel and coupled multiclass.
All rows count for minimum-leaf constraints even with zero weight. NaN/missing
and category routing are fixed by Schema. Host computation is absent.
Output is compact resident trees, bases, margins and rounds+1 weighted losses.
The caller provisions model/result arrays and one exclusive aligned workspace;
all products, budgets and capacities are checked before effects. Partial output
after any failure is invalid. Round-major construction appends compact node
segments; descriptors are grouped stably by output without moving whole forests.

Use a single GPU coordinator for output slices (width1 handles per-output tree
construction), typed Stats<Order>, common split enumeration with explicitly
different reduction schedules, and two frontier materialization sizes. This
removes duplicate coordinators without normalizing FP64 atomic arrival order.
Gradient planes remain separate for coalesced loads; order2 emits no higher
derivatives. Root counts are invariant and cached with exact u64 accumulation.
Multiclass takes its full derivative snapshot before any per-round margin update.

The selected algorithms reproduce documented measured schedules from the frozen
training/ALGORITHM_DECISIONS.md, DEFAULT_POLICY_DECISION.md, root/deeper/split/
resident implementations and quality audits. Shared histogram and wide split
alternatives remain explicit, not automatic promotions. Original order2 warp
splits retain both zero-contributing reduction levels and the zero scan carry.
Second/higher-order derivative expressions and leaf proposals remain distinct.

Primary sources inspected 2026-09-23:
- https://arxiv.org/abs/2211.11367 (closed-form higher-order boosting proposals;
  not a guarantee of actual loss decrease or superiority on our data).
- https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back
  (large global scans; small frontier scans do not justify its extra state).
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html
  (GPU scheduling/visibility and tail continuation rules).

Memory: tile derivatives cost rows*tile*Order*8 (full outputs for multiclass);
dense histograms cost active_capacity*batch*total_bins*sizeof(Stats<Order>).
Rows, output tiles and frontiers remain bounded by explicit budgets. Small
frontiers fuse prefix and node writes; large frontiers use scan/prefix/write
stages to avoid global spin barriers. Disjoint per-output state enables batching
without shared partitions or vector-leaf model changes. Do not silently switch
to approximate bins, quantized gradients or parent subtraction.

Selected architectural work removal before route implementation: when routing
deactivates a row into its final leaf, add that already materialized FP64 leaf
value to its output margin exactly once with __dadd_rn. At the final split level,
read the selected materialized child leaf. This removes a separate whole-tree
prediction traversal. Independent-output derivatives for the tile and coupled
multiclass derivatives for the round are already frozen; early margin updates
therefore cannot change subsequent splits. Invalid operations expose no model.
The exactness experiment compares final margins to an independent ordered model
traversal, including pruned/terminal nodes, missing routes and cancellation. The
complete-training comparison includes route/export costs; no speedup is presumed.

Before promotion, test independent GPU arithmetic/split/tree fixtures, ordering,
minimum constraints, budgets, zero weights, all objectives and higher orders;
compare frozen predictions and all aggregate/per-output quality metrics. Preserve
unordered-atomic variability and failed zero gates. Run serial complete training
comparisons with three warmups and fifteen alternating paired samples; qualify
device timestamps and report initialization, coordination, calibration and export
cost. Profile separately with Nsight and inspect resources/spills. This choice is
a correctness/performance experiment, not a fastest-code claim.
