# Large-output prediction: remove transfers, then test exact kernel fusion

Recorded 2026-09-23 before implementing any candidate in this document. This is
a design and diagnostic interpretation, not a measured ranking. D2 inference
encoding and E256 model packing are separate experiments. No counting, training,
model, numerical default, or existing prediction policy changes here.

## Selection from the large-output trace

The next architectural candidate is **R0: explicit prepared model, encoded input
and reusable device output**, initially using the existing ordered traversal and
objective transform. Its purpose is to remove repeated setup and the full host
result materialization when the consumer is on the GPU. For callers requiring
host results, separately test **F2: GPU finite-output validation** under the
unchanged vector API and **H1: an explicit owned-overwrite/caller-span API**.
The smallest separate traversal trial is **F1: fuse the existing independent-
logistic sigmoid into ordered traversal**. Test F1 alone against B before
combining it with R0. A shared-memory
output tile, T1, needs store-efficiency evidence before implementation; strided
source addresses alone do not establish the dominant kernel bottleneck.

The reason to prioritize architecture is the complete-call trace, not an assumed
forest-kernel limit. Capture `results/prediction-slab-20260923/nsys-fused-large/`
uses frozen `fused_output` B with 65,536 rows, 16 features, 1,024 outputs, 2,048
trees and 9,984 nodes. Inputs are the benchmark's deterministic model-derived
fixture, not held-out signal-quality data. The 536,870,912-byte output is 512 MiB.

| Observation in this single instrumented call | Value |
| --- | ---: |
| Complete timed `predict_gpu` sample | 384.070373 ms |
| GPU `encode_tile`, one launch | 0.025633 ms |
| GPU `ordered_forest`, one launch | 20.293106 ms |
| GPU `sigmoid_kernel`, one launch | 28.799201 ms |
| Prediction D2H activity, 512 MiB | 69.620656 ms |
| All H2D activities | 24 copies, 4,564,004 bytes, 0.550675 ms |
| All D2H activities | 2 copies, 536,870,916 bytes, 69.623984 ms |
| CUDA allocations / frees | 7 / 7 |
| Allocation / free API durations | 3.886890 / 7.096550 ms |
| Explicit stream synchronization calls | 4; 0.058944 ms combined |
| Capture range start to first prediction CUDA API | 184.928234 ms |
| Final checked stream wait end to cleanup drain start | 52.044416 ms |

The first host gap brackets model/input validation and the 512 MiB result
vector's allocation/value-initialization. The second brackets the host finite
scan after the completed download. These are source-supported brackets, not
exclusive CPU attribution: this capture disabled CPU stack sampling and context
switch tracing. Demand paging, scheduling and allocator behavior can contribute.
Add narrow diagnostic ranges around validation, result construction, finite scan
and packing before assigning those gaps to individual host operations.

The final `cudaMemcpyAsync` API call lasts 133.022820 ms: it starts before the
forest/transform finish and includes waiting for preceding stream work. Do not
add that API duration to the GPU kernel and copy durations. The full capture
range is 456.775381 ms; it extends 72.701706 ms after prediction stream destruction
because the benchmark's exact output comparison and result disposal are outside
its timed call but inside capture. Use the benchmark's timing boundary, not the
NVTX range length, as the complete-call observation. Profiler times do not rank
future candidates or predict their uninstrumented speedups.

The forest uses 30 registers/thread, sigmoid 27; both have 256 threads/block,
65,535 blocks, zero static shared memory and zero recorded local memory/thread.
These are compiler/resource observations, not measured occupancy or proof of a
memory/FP64 bottleneck. NCU should distinguish scatter-store sectors, DRAM/L2
traffic, FP64 math utilization, issue stalls and achieved occupancy.

This 16-feature case has only one encoding tile. D2 cannot remove an intermediate
tile checkpoint here. E256 can remove three model allocations/uploads, but its
model payload is only 368,648 bytes and has zero device padding for this model.
Neither can eliminate the 512 MiB host-return requirement. Keep these mechanisms
and their separately measured results distinct.

## Exact contract, shared by the kernel candidates

Use the same frozen model and input bytes. Do not retrain to compare inference:
training's unordered FP64 histogram additions already allow different trees.
Keep `validate_prediction`, feature encoding, model graph validation and capacity
checks. Dataset values remain row-major FP32; owned encoded bins are feature-major
u16 with the same numeric, categorical, unseen and missing-value behavior.

For every row/output, load the exact FP64 base value, follow the same tree routes,
and add one reached leaf for each tree of that output in original model order.
Preserve each `__dadd_rn`, including zero-valued leaves, signed zero, subnormals
and cancellation. No tree-parallel reduction, changed base placement, FP32 leaf
conversion, approximate exponential, reassociation or skipped additions.
Store the same row-major FP64 matrix at `row*outputs+output`.

Objective behavior is explicit:

- `raw=true`: return ordered margins for every objective.
- Squared error, `raw=false`: same margins; transform is a no-op.
- Independent binary logistic, `raw=false`: preserve the GPU sign branch.
  Nonnegative margin uses `1.0 / (1.0 + exp(-margin))`; negative margin computes
  `e=exp(margin)` then `e / (1.0 + e)`, all in FP64. F1 moves this operation but
  does not substitute a mathematically equivalent formula.
- Multiclass softmax, `raw=false`: initially retain the existing separate kernel.
  Each lane visits outputs `lane, lane+32, ...`; preserve the local max/sum order,
  shuffle-down max/sum offsets 16/8/4/2/1, broadcasts, exp calls and divisions.
  A different reduction tree is a numerical change, even if every tree margin
  is exact. Softmax is coupled across all outputs, unlike independent sigmoid.

The existing public call remains synchronous and returns a `std::vector<double>`.
Its returned values must be finite; preserve the post-transform finite-result
check and CUDA-error propagation. Do not move the finite check to raw margins
and assume that is the same error contract. Invalid model/input, infinity in
features, capacity overflow and allocation failure remain failures; no silent
policy fallback. Empty rows still return an empty vector after validation; empty
forests and outputs without trees still produce exact bases/transforms.

CUDA C++23, SM86, unchanged compiler/math flags are the initial scope. Cross-build
or cross-architecture transcendental bit equality is not inferred from formulas.
Inspect generated code and compare frozen GPU reference bits. Existing CPU
prediction is a useful raw-margin oracle, not an exact oracle for GPU exp/softmax.

## F1: fuse independent sigmoid after the ordered accumulator

Use a separately named/selected specialization of the ordered forest. The raw
and regression paths retain their old work. For transformed binary predictions,
run the existing sigmoid expression on the final FP64 accumulator, then store
that probability once. Remove the following sigmoid launch only in this cell.
Leave the old entry point and kernel callable for a same-build comparison.

There is no new global scratch, atomic or inter-thread synchronization. Per row,
all routing and ordered leaf addition work is unchanged. Logical prediction
array traffic changes from a forest write plus sigmoid read/write (24*N*O bytes)
to one final write (8*N*O). At this shape the reduction is 1 GiB, with one fewer
launch. The exp/division work remains; do not interpret the 28.799201 ms sigmoid
duration as removable time. Output stores are still strided within a warp.
Fusion may increase register live ranges, reduce occupancy, or combine divergent
sigmoid signs with tree traversal divergence. It can lose despite fewer bytes.

Copying the old expression is necessary but not sufficient evidence of exactness.
Keep FP64 intermediates and branch semantics; inspect NVCC code generation for
the same margin addition, exp input and division behavior. If extracting a shared
device helper changes unrelated kernel code, retain that change in provenance
and revalidate it, or keep the frozen reference implementation intact. Do not
use compiler time-trace/debug-device builds to rank runtime performance.

## T1: retain row-coherent traversal, stage coalesced output writes

Current B assigns consecutive lanes to consecutive rows for one output. Bins
for a common feature are adjacent, tree descriptors/nodes can be shared by cache,
and the per-output forest loop is coherent. Its final row-major writes are
8*O bytes apart. For O=1,024, that is 8,192 bytes: a full warp's 32 FP64 stores
touch 32 different 32-byte sectors, versus eight sectors for an aligned contiguous
warp. This is a per-request sector comparison, not a fourfold DRAM-byte or speed
claim; cache behavior and writes from other warps matter.

Compare these options before selecting T1 implementation:

| Candidate | Extra work/storage | Main tradeoff |
| --- | --- | --- |
| Row-major task assignment | No scratch/barrier; different index mapping | Coalesces final stores but adjacent lanes traverse different outputs/trees and may load unrelated feature rows |
| 32-row by 32-output shared tile | 8,448 shared bytes for `double tile[32][33]`, 16*N*O shared bytes, barriers | Retains adjacent-row traversal while transposing only final scalars; adds block-wide waiting across forests |
| Output-major global temporary then transpose | Extra 8*N*O device capacity and 16*N*O logical global traffic, extra launch | Simple coalesced producer, but costs another 512 MiB buffer here; generally unattractive before an in-kernel tile is tested |
| Change public output layout | No required transpose | Different API contract; excluded from exact replacement of the row-major public result |

Bound T1 to a 256-thread block `(32,8)` covering 32 rows and 32 outputs.
Each warp traverses four outputs sequentially for its 32 adjacent rows; every
scalar still visits all its own trees in reference order. Write completed
scalars to the shared tile, synchronize, remap lanes to adjacent outputs of one
row, and issue row-major global stores. A block-uniform grid-stride loop over
tiles needs a second barrier before shared storage is reused. Tail predicates
must not cause some threads to skip a barrier, and no valid store may read an
unwritten shared slot. Flatten tile indices with checked 64-bit arithmetic so
large output counts do not exceed a two-dimensional grid limit.

The 33-element pitch is only a candidate. The familiar FP32 transpose example
does not prove conflict-free FP64 instructions on this device. Inspect shared
wavefronts/bank conflicts and generated 64-bit transactions. A two-plane u32 bit
representation is an alternative exact transport, with extra instructions; do
not add it unless counters justify it. Tree imbalance, shared-memory occupancy,
integer indexing and synchronization can outweigh better stores.

First test T1 with unchanged separate transforms. Only after individual gates
pass, test T1+F1. For softmax T1 merely produces the same row-major margins for
the existing transform; it does not fuse a partial-output softmax. Parallel
traversal of different trees followed by reduction is excluded because it changes
FP64 addition order. Ordered gathering of separately computed leaf values would
need O(rows*trees) scratch and writes and is a separate, less bounded experiment.

## F2 and H1: retain full host results while removing redundant host work

F2 keeps the public `std::vector<double>` API, including its allocation and value
initialization. Add an independent GPU finite-output check after the final
requested transform, with an explicitly initialized integer status word. It
reads each result, flags every NaN/Inf, and otherwise does not write predictions.
Download the same complete output and the status, wait, and reject nonzero status
before returning. The CUDA copy contract and successful checked completion still
apply. Do not return a partially downloaded or unchecked vector after failure.
Retain the original CPU-scan path as the control.

This replaces the source-bracketed 52.044416 ms host scan with a device scan, one
kernel launch, one status clear and a four-byte status transfer; it does not
remove the 512 MiB D2H or result initialization. The new scan logically reads
512 MiB at this shape and might not win. Use the same final-output validity
criterion for raw, sigmoid and softmax, including all NaN payloads and +/-Inf.
An OR flag is exact integer status; it must be freshly cleared each call. Place
its host destination before the exceptional drain in lifetime order. This scan
can later serve R0's checked completion. Fusing it into a producer is a separate
cell that must still check the final requested values, not intermediate logits.

H1 makes allocation semantics explicit instead of manipulating vector internals.
An owned host-result type can hold checked-size, properly aligned FP64 array
storage created with `std::make_unique_for_overwrite<double[]>(count)` and an
extent; the standard specifies array allocation without value initialization
([unique_ptr creation](https://eel.is/c++draft/unique.ptr.create)). No element may
be read or exposed as a result before successful full download and validation.
On failure, destroy or invalidate the result; caller-span contents may be partly
overwritten and are not a successful prediction. The ordinary vector-return
API and its allocation semantics remain unchanged.

A caller-span overload accepts an existing live FP64 array of sufficient checked
capacity. It can reuse that storage for multiple calls, with an explicit owner
lifetime and no alias with inputs or other pending outputs. This is host result
storage, not a CPU inference implementation. Allocation cost, first-touch/page
faults, any pinning and teardown remain in the one-shot boundary. For Q repeated
calls, report setup + all Q complete calls + teardown, including Q=1, alongside
per-call steady state. Overwrite allocation may only defer page commitment into
the D2H path, so avoiding a source-level zero fill is not proof that the entire
184.928234 ms gap disappears. If callers convert the owned result to a vector,
include that extra allocation/copy when reporting their application cost.

## R0: explicit ownership and reuse, without a hidden mutable-model cache

R0 is a new, explicit API boundary, not a claim that a device-only call meets the
old host-vector-return contract. Start with three independently owned resources:

1. A prepared immutable forest snapshot: validate the model once, preserve base,
   node and descriptor bits and stable output grouping, and upload it once.
   It owns its device allocation and a feature-metadata identity. Later edits to
   the public `Model` cannot silently change this snapshot; rebuild explicitly.
2. An encoded batch produced using that exact fitted feature schema and existing
   GPU encoding. It owns the bins/types/offsets and records dimensions/device.
   Preparation is checked and complete before reusable execution. A new batch
   requires new encoding; repeated-input timing must not pretend input changed.
   Feature-count equality alone is insufficient to authorize reusing bins with
   a different model's cuts/categories. The first version accepts owned prepared
   batches, not arbitrary unvalidated borrowed device pointers.
3. A reusable row-major FP64 output allocation, an error-status word and explicit
   completion state. Check element capacity, integer overflow, device binding
   and nonaliasing. The initial version permits one in-flight operation per
   mutable workspace; immutable models may be shared only with distinct outputs
   and correctly ordered prepared inputs. No implicit stream or model cache.

First implement execution using the existing `gpu::predict_forest` and
`gpu::transform`, plus a GPU finite-result check. Preserve the numeric kernels
before testing fusion. The finite check must inspect the final requested output,
OR an integer flag for nonfinite results, and expose failure at an explicit
checked-completion boundary. It does not alter output arithmetic. An asynchronous
enqueue cannot truthfully throw a future execution/status error before completion.
Separate enqueue-time CUDA errors from `wait_checked` errors, and mark results
unvalidated until that boundary succeeds. Do not remove finite checking merely
because valid models normally produce finite values.

The simplest initial checked completion may download only the status word and
wait. That retains a tiny host checkpoint, not the 512 MiB result download/scan.
The extra finite-scan kernel logically reads 8*N*O bytes and needs a launch and
status clear; include these costs. A later independently gated fusion of finite
checking into the final producer can remove that scan. No such fused validation
is assumed free, and the softmax case needs its own final-result treatment.

All owners must outlive enqueued operations. A pending operation retains its
model/input/output resources or the contract requires checked completion before
release, with an exception-safe drain before deallocation. Provide bounded
failure behavior for allocation/launch/copy errors and ordinary validation
rejection; do not promise recovery from a lost device. Device identity is fixed
at preparation and checked on use. Do not introduce `cudaMallocAsync`, graph
updates, cross-stream scratch reuse or thread-pool concurrency by implication.
Graph capture can remain an explicit later experiment using preallocated owners;
the existing allocation-free forest entry point already supports stream capture.

At this shape the prepared forest payload is 368,648 bytes, bins/type/offset
payload 2,097,284 bytes, and output 536,870,912 bytes. Repeated execution can avoid
the seven per-call allocations/frees, model uploads and already-prepared input
uploads/encoding. It also avoids constructing/zeroing a 512 MiB host vector and
scanning it on the CPU. These are host setup and transfer removals, not CPU
training or preprocessing shortcuts. The full output must still exist on device.
Report retained capacity and setup peak; allocation success is never guaranteed
by a prior free-memory snapshot.

Consumers needing all values on the CPU still require the 512 MiB transfer.
Measure an explicit download separately and also report prepare+execute+download
for one-shot equivalence. An optional later `predict_into` host span can amortize
host output allocation/value-initialization but cannot remove that transfer.
Use caller-owned valid storage; writing beyond a vector's size after `reserve`
is not a valid way to remove initialization. Pinned host output is another
separate parameter whose allocation/registration and retained capacity count.
Returning top-k labels, FP32 values, compressed output or a different layout is
a different contract and cannot stand in for the full FP64 result.

## Primary algorithm/source comparison

NVIDIA's current [SIMT memory guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#memory-performance)
describes 32-byte global transactions and shared-memory staging for transpose.
Those mechanisms motivate T1; its FP32 padding example and published hardware
results do not rank this FP64 SM86 forest. The [Best Practices transfer section](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#data-transfer-between-host-and-device)
supports keeping intermediate data on device and accounting for expensive pinned
allocation. R0 applies that architecture rather than hiding transfer costs.

Current [nvForest documentation](https://docs.nvidia.com/nvforest/latest/python_api/)
exposes layout/chunk tuning and device-accessible inputs/outputs. Its pinned
source revision `8d09212ba8f698048f79bd11b9d3ca48e1b151e2` implements row chunks,
shared staging, parallel tree groups and warp aggregation in
[infer_kernel/gpu.cuh](https://github.com/rapidsai/nvforest/blob/8d09212ba8f698048f79bd11b9d3ca48e1b151e2/cpp/include/nvforest/detail/infer_kernel/gpu.cuh#L97-L205).
Its [postprocessor](https://github.com/rapidsai/nvforest/blob/8d09212ba8f698048f79bd11b9d3ca48e1b151e2/cpp/include/nvforest/detail/postprocessor.hpp#L60-L115)
adds bias after aggregation and uses a different sigmoid expression. This is
useful architectural evidence, not an exact replacement for our base-first sums,
sign-stable sigmoid or binned feature semantics. No nvForest/CUB/other NVIDIA
algorithm implementation is introduced into production.

The [FP64 addition intrinsic](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html)
and NVIDIA's [floating-point discussion](https://docs.nvidia.com/cuda/floating-point/index.html)
support explicit rounding/order contracts. Mathematical equivalence does not
permit an observed bit mismatch or quality regression to pass the exact gate.
Tensor cores, quantized leaves, approximate sigmoid and reordered sums are not
applicable shortcuts under this experiment's contract.

## Fair experiment and gates

Freeze source, binary, model and actual feature bytes. R0, F1 and T1 are explicit
cells; hold B's preparation/packing policy fixed when comparing the kernel-only
candidates. Do not mix concurrent D2/E or training changes into an attribution.
Run GPU workloads serially with no concurrent CPU analysis/builds for ranking.
Record native Windows engine telemetry; retain interference and failed samples.

Validate raw and transformed bits against both frozen B and `per_tree`, with
zero differences, before timing. Cover rows 0/1/31/32/33/255/256/257/65,536,
outputs 1/3/31/32/33/65/983/1,024, empty forests, missing output groups, interleaved
trees, shallow/deep unbalanced trees, categorical/unseen/missing inputs and every
numeric boundary already tested. Include +/-0 bases/leaves, FP64 subnormal and
cancellation fixtures, adjacent margins around zero, large finite positive and
negative margins and exp-underflow regions. Compare transform-only candidate
behavior on a bounded independent set of FP64 bit-pattern fixtures; leaf-only
models can produce exact chosen margins without changing the training system.
For softmax retain all output-count tails and dominant/extreme logits.

R0 additionally needs exact preparation identity/binding tests, repeated calls
with changed shapes and prepared batches, zero-capacity/overflow/device-mismatch
rejection, no-alias checks, ordinary fault cleanup/recovery, and lifetime checks
through checked completion. Mutation after snapshot creation must not change the
snapshot. Finite-check controls must detect injected NaN/Inf output fixtures.
T1 requires memcheck/initcheck/synccheck/racecheck on tails and multiple tile-loop
iterations; all buffer slots read must have a producer. Preserve unsupported
tool reports as limitations rather than inventing passes.

For host-return F1/T1 comparisons, time the unchanged complete `predict_gpu`
boundary with all setup, result construction, H2D, quantization, transform, D2H,
finite validation and cleanup included. For R0 report three separate boundaries:
one-shot prepare+checked execute+download+cleanup; checked steady-state resident
execution with declared retained resources; and incremental full-result download.
Include a changed-input workload as well as same-prepared-batch reuse. Report
setup + Q calls + teardown for Q=1/2/10/100, not just per-call resident timing;
do not divide setup away without stating the count. H1/F2 retain a full host
result and require their own complete-call comparison to the vector control.
A GPU-only enqueue timer is not complete latency.

Use three warmups and at least 15 balanced alternating paired samples per case,
including the large shape, actual Delicious validation, small-row and one-output
controls, raw mode and softmax. Retain all raw observations. Require the upper
95% paired-bootstrap interval for median candidate/reference time ratio below
1.0 for a casewise speedup claim; inconclusive cases stay inconclusive. A default
promotion additionally needs all exact gates and no measured regression over
the agreed matrix. Frozen-model bit equality implies unchanged metric inputs;
also retain applicable aggregate/per-label/signal quality checks on actual labeled
data. Synthetic performance fixtures do not establish signal competitiveness.

Use Nsight afterward to confirm removed launches/bytes, distinguish host gaps
from GPU work, and explain resource/stall changes. No profiler time is a ranking.
In particular, fused exp time remains real work, better store addresses do not
prove less DRAM traffic, and a resident result is not a completed host return.

## Source and evidence identity

The trace predates working-tree D2 edits. Its `compiled-provenance.json` points
to `results/prediction-slab-20260923/source-snapshot/manifest.json`; use that
snapshot for the producing `booster.cpp` and quantizer, not their newer working
versions. Stable analyzed numerical paths are `training/src/prediction.cu`,
`training/src/kernels.cu:114`, `:395`, `:399`, `:550` and
`training/include/ghb/prediction.cuh`. Public source behavior is in the captured
`Model::predict_gpu`; benchmark capture/timing boundaries are in captured
`training/bench/prediction.cpp:176-209`.

| Artifact | SHA256 |
| --- | --- |
| Captured binary | `8e51583e2a5afebcc7c2cfe7609d23366f3d30c72b553b5a76d839fa1a9d9a22` |
| `profile.1.sqlite` | `5b53ae3fd53360bb003d21359d5f9f12347fbe317467815c101866942baf1482` |
| `benchmark.json` | `817267f0b8b35579b716c4af9fd8fadaec5d4cde304c899bc84c4c33292837bd` |
| `compiled-provenance.json` | `063e3a6a1d3d0cbc3cc9864f77b5d6ac0978574106d211614649bac65c276ff9` |
| Analyzed `prediction.cu` | `db76cb6383d1ec2393a6b14bd736a6b150d4535798e688661fa6fba2c120dae7` |
| Analyzed `kernels.cu` | `af915c31ed030279db912ab5d8954e98e92103edead8f22c4665fba1b94e3f6d` |
| Analyzed `prediction.cuh` | `3b0013adb424fe32858b30a0ca929cc9d06fdeef2e5a2c509f3fab0482cd56c0` |

Read-only SQLite reproduction: join `CUPTI_ACTIVITY_KIND_KERNEL.demangledName`
and `CUPTI_ACTIVITY_KIND_RUNTIME.nameId` to `StringIds.id`; durations are
`(end-start)/1e6` ms. Group `CUPTI_ACTIVITY_KIND_MEMCPY` by copyKind (1 H2D,
2 D2H), preserving correlation IDs to avoid adding runtime waits to activity
durations. Host gap endpoints in ns are 5,955,469 -> 190,883,703 and
331,225,763 -> 383,270,179. Cleanup ends at 390,029,144; capture ends at
462,730,850. Existing raw artifacts are retained; this review generated no
GPU workload and changed no production source.
