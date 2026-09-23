# Independent GPU training checks

This contract precedes the test implementation. The suite is a bounded primitive
and integration check, not the full quality/performance acceptance campaign.
Host code only starts the shared CUDA driver and observes runtime completion.
Fixtures, configuration selection, expected values, comparisons and diagnostics
are computed on the GPU. No old implementation or production learning helper is
used as an oracle.

The caller supplies disjoint global model arrays, feature-major bins, row-major
targets/margins, loss history, guarded workspace and zeroed completion records.
One coordinator submits training. Its tail consumer checks runtime and semantic
completion before reading results or reusing storage, then submits independent
model validation. The next fixture starts only after that validation completes.
Tests intentionally exercise capacity rejection; rejected partial output is not
treated as a valid model. Zero rounds is valid; zero training rows is invalid.
Zero-round cases use a one-byte histogram budget and otherwise incompatible
shared histogram/root-count capacities: absent stages must consume no storage
and impose no launch-resource requirement.

The large-frontier fixture has 2,048 unit-weight rows, eleven binary numeric
features ordered from most to least significant row-index bit, and target equal
to the row index. With L2 zero, unit learning rate and depth twelve, exact dyadic
gains select features in order. The complete balanced tree has 4,095 nodes and
2,048 live singleton nodes at its last visited level, exercising the multi-block
scan/prefix/materialize branch. Base 1023.5, leaf `row-1023.5`, final margin `row`,
initial loss `(2048*2048-1)/24` and final loss zero are independent exact oracles.
Separate fixture inputs add about 160 KiB; the existing model and 4 MiB arena
remain sufficient. This is a correctness workload, not a timing experiment.

GPU unit landmarks directly exercise the production higher-order leaf helper,
which is also used by training. Expected values come from simple quadratic,
cubic and quartic polynomials, not a second copy of the solver. Cases distinguish
Newton retention, strict proposal improvement, clipping, denominator rejection,
signed zero and nonfinite/overflow guards. Discrete selected values are exact;
only non-dyadic diagnostic benefit values use the stated arithmetic tolerance.
An overflowing learning-rate rejection first makes a finite root with unshrunk
leaves -1.5/+1.5, then two descendant splits containing -3/+3 at learning rate
2^1023. Both descendant flags reject overflow; neither may reuse the excluded
prefix ordinal for conflicting child writes. Margins must remain at their zero
base because those rows have not reached a committed final leaf.

Compute Sanitizer initcheck, racecheck and synccheck report unsupported CDP2
execution for the current executor; they are not clean results. Memcheck remains
applicable. Later direct leaf-kernel validation must GPU-initialize fixtures in a
separate host-launched bootstrap, execute an exposed production leaf kernel as a
host-launched top-level grid, then GPU-check the result in another top-level grid.
The host supplies only static launch parameters and runtime sequencing, never
fixture/oracle computation. That executor has not been implemented by this suite.

## Numerical and algorithm choices

Tiny dyadic regression fixtures permit an independent exhaustive one-level
oracle: partition raw row bins for every feature/threshold/missing direction,
sum weighted residuals and counts directly, and compare the regularized Newton
objective. This does no histogram construction or prefix scan. Ties use the
public feature/threshold/missing-direction order. Known balanced binary examples
check clipped order-3/4 leaves without reusing the production higher-order solver.
All successful cases independently traverse exported trees in descriptor order,
starting from the base and adding each selected materialized leaf with
`__dadd_rn`. Its binary64 bits must equal the training margins. This specifically
checks the fused route/update operation, including missing rows, early terminal
nodes, last-level children, multiple rounds and large-base cancellation.

Independent prefix-model loss calculations check every history element and the
weighted base initialization for regression, binary logistic and multiclass.
Regression dyadic split expectations and ordered margin folds are exact. General
log/exp and differently ordered reductions use a documented relative diagnostic
tolerance of 2e-12; this is not a tolerance for model bits or any zero-allowance
quality gate. Structural validation additionally checks grouped tree descriptors,
permuted compact node segments, metadata, all graph constraints and numeric bounds.

The small registry covers scalar and multi-output training, tile tails above 32
outputs, per-output and output-batch builders, batched/unbatched roots and root
splits, per-output/global/shared root counts, global/shared histograms,
block/warp/wide splits, zero-weight row counts, depth/row/Hessian/gain/L2/clipping
constraints, and shape/budget/capacity/nonfinite rejection. The uncached automatic
root case requires a measured record with five nonzero observations for each
candidate and output; multiple rounds exercise reuse of the selected policy.
This verifies calibration execution, not that its choice wins an independent
complete-operation performance experiment. Batched roots retain their default
selection and must not claim an executed calibration.

Serial row scans in the test oracle minimize implementation coupling and scratch
for these tiny cases; they are deliberately not a production GPU algorithm or a
benchmark competitor. Shared fixture storage and one tail chain remove repeated
allocation/transport. The fair production experiment times the complete resident
training operation, including planning, calibration, initialization, all rounds,
losses and completion; it compares each policy on identical inputs and preserves
raw failures and all archived quality metrics. This suite records no speed rank.

Primary contracts: [CUDA dynamic parallelism](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html)
for child/tail visibility, and [CUDA double intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html)
for the explicit round-to-nearest addition. The API is `include/gh/learn.cuh`;
frozen learning schedules are evidence in the external archive, not linked code.

Only an actual successful root-run prints the completed-case count. Source
existence and compile receipts must not be reported as executed coverage.
