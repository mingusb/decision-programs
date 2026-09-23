# Functional GPU prediction: next bounded implementation

Recorded 2026-09-23 before implementation. This is an independent design review;
no production edit, build, GPU launch or timing was performed for it. Read
`AGENTS.md`, `FUNCTIONAL_DESIGN_AUDIT.md`, `ALGORITHM_DECISIONS.md`, current
prediction/ownership sources and `LARGE_OUTPUT_PREDICTION_EXPERIMENT.md`.
The scalar layout and dead-member changes do not complete the functional migration.

## Decision and measured motivation

Implement **F2, final-output validity as a value reduction plus explicit checked
completion**, first. Use that primitive in a subsequent **R0 owned resident
prediction operation**. This advances the algorithm and the ownership boundary
together: it moves the validity computation to the GPU, specifies when an output
becomes an observable value, and enables reuse without a full host result.
It does not make the existing trainer or prediction traversal functional by itself.
Preserve those numerical kernels as frozen effects while this slice is verified.

T1 now merits a separate coalescing experiment. The newer
[`large-output-memory-selected-counters.json`](../results/prediction-slab-20260923/large-output-memory-selected-counters.json)
supersedes the earlier document's missing-counter condition:

| Ordered-forest observation, 65,536 rows × 1,024 outputs | Value |
| --- | ---: |
| Useful FP64 output | 536,870,912 bytes |
| Global-store requests / sectors | 2,097,152 / 67,108,864 |
| L1 store sectors per request | 32, versus 8 for aligned contiguous FP64 stores |
| L1 store-sector bytes / useful bytes | exactly 4 |
| Global-load sector bytes | 2,644,720,960 |
| Displayed DRAM write total | 2.178017 decimal GB, rounded |

That capture used four kernel replay passes, all-cache flushing, unchanged clocks
and binary `8e51583e2a5afebcc7c2cfe7609d23366f3d30c72b553b5a76d839fa1a9d9a22`.
Its JSON records the source manifest and capture hashes. The model hash was checked
after capture against an earlier identity, not captured contemporaneously.
Chip-global DRAM can include desktop/asynchronous engines. Do not divide counters
by replay passes or infer a fourfold speedup. The exact store amplification is
sufficient motivation for T1; substantial traversal loads remain part of its cost.

R0 retains architectural priority because the complete-call Systems observation
includes a 512 MiB download and substantial host work. The 52.044416 ms gap after
the checked wait brackets the current CPU finite scan; it is not exclusive CPU
attribution or a promised F2 saving. F2 adds a GPU read of that same 512 MiB,
one scan launch, a four-byte status clear/download, and checked completion.
Only complete unprofiled calls can establish whether this substitution wins.

## Relevant approaches and scope of applicability

| Approach | Work, storage and effects | Selection |
| --- | --- | --- |
| Existing CPU final scan | Full host output and O(rows×outputs) host reads; no GPU scan | Unchanged vector-return control |
| Contiguous GPU predicate + integer OR | O(rows×outputs) GPU reads, four-byte status, clear/launch/download; no FP arithmetic reordering | F2 first |
| Two-stage block flags + reduction | Bounded partial array and second launch; avoids a hot status word on densely invalid input | Comparator if invalid-path contention matters; not required for normal finite output |
| Predicate fused into final producer | Removes scan reads/launch; changes producers and has objective-specific final-value locations | Later independent cell, after standalone F2 |
| Prepared forest/batch/output | Retained capacity; removes repeated setup and optional full host materialization | R0 after F2 completion contract |
| T1 shared output tile | 8,448 bytes shared/block, 16×rows×outputs shared bytes, barriers; preserves row-coherent traversal | Separately justified exact trial |
| Row-major traversal assignment | No staging, coalesced writes; changes feature/node request locality and divergent tree work | Useful separate T1 control, not assumed faster |
| Output-major global temporary | Extra full-output allocation, transpose launch and 16×rows×outputs global bytes | Defer; high capacity/traffic cost here |

[CUDA's transfer guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#data-transfer-between-host-and-device)
supports retaining GPU intermediates. Current
[nvForest interfaces](https://docs.nvidia.com/nvforest/latest/python_api/) accept
device arrays and expose chunk/layout tuning. Its pinned
[GPU inference implementation](https://github.com/rapidsai/nvforest/blob/8d09212ba8f698048f79bd11b9d3ca48e1b151e2/cpp/include/nvforest/detail/infer_kernel/gpu.cuh)
is useful architectural evidence for batching/staging; parallel tree aggregation
does not preserve our base-first ordered additions. None is a local ranking or
a production dependency. CUB/Thrust remain excluded from production algorithms.

[Futhark consumption semantics](https://futhark.readthedocs.io/en/latest/language-reference.html#in-place-updates)
give the relevant functional precedent: storage can be reused when the previous
value and aliases are no longer observable. Its
[backend architecture](https://futhark-lang.org/blog/2025-03-04-adding-a-new-backend.html)
separates a value computation from its effectful execution. Apply those principles
in a small domain-specific operation, without a new general expression language.
C++ moves alone do not establish uniqueness.

## F2 contract and functional algorithm

Input is an immutable, completed-on-the-same-stream final row-major FP64 matrix
of checked extent M=rows×outputs, plus an exclusive status allocation on the same
device. Output is the same matrix bits and `bad = OR(map(nonfinite, values))`.
The predicate can classify the binary64 exponent bits; all NaN payloads and both
infinities are bad, while both zeros, subnormals and all finite values are good.
Use a supported bit-preserving operation and verify its emitted code. No FP sum,
conversion, exponential or prediction write is introduced by validation.

The **semantic algorithm is the predicate and associative integer OR**, not a
mutable `found` flag wrapped in a function named `map`. Give the predicate a
stateless callable/static call operator, immutable shape/launch values, and one
small specialized lowering for a contiguous owned reduction. No intermediate
predicate array is materialized. Integer OR permits hierarchical reduction and
arbitrary contribution order exactly; tree-margin addition does not.

Start with 256-thread blocks, adjacent lane reads and a warp vote, followed by an
elected-lane `atomicOr(status, 1)` only for a bad warp. Tail lanes contribute false
and still participate in the vote. An all-finite call performs no status atomics.
The vote, global loads and atomic update are explicitly GPU effects; the pointer
is not a pure array value merely because it is `const`. Bound indexing and launch
coverage from the checked extent/device limits; any grid-stride scheduling is
part of this reduction's executor, not an extra traversal or a new algorithm.
Dense invalid input can contend: retain its observed rejection latency rather
than claiming the successful-path design is universally fastest.

Run validation **after** the requested transform. Raw/regression paths validate
margins, sigmoid validates probabilities, and softmax validates the final coupled
transform. An infinite raw margin can become a finite sigmoid result; checking
the intermediate margin would change the contract. Preserve every existing
tree addition, sigmoid branch, softmax reduction, exception class/message and
successful output bit. Empty rows validate metadata then return empty without
launching a zero-sized grid. Overflow remains rejection, never saturation.

Keep the old CPU-scan route callable. The explicit F2 vector route retains vector
construction, complete prediction download, stream completion and cleanup. Add
the status bytes to reserved capacity; do not silently spend an unchecked budget.
A failed download/CUDA wait cannot become a successful result because status is
zero. CUDA execution errors precede the semantic nonfinite rejection, as in the
current checked-download-then-scan path. New resource failures remain explicit.

## Ownership and effect boundary

Use immutable plan values and a narrowly specified consuming operation, not a
mutable workspace exposed through const methods. The intended states are:

```text
Plan = validate_shape_and_binding(request, resource_metadata)
Ready --enqueue(Plan)--> Pending --wait_checked--> Checked | Failure
Checked --consume_for_next_prediction--> Pending
Checked --download_checked--> HostResult
```

Planning is a fallible value transformation. Allocation, status clearing, kernel
submission, copies, waits and release are effects. A compact compile-time sequence
describes their dependencies; its executor performs them. No heap-backed action
list, virtual dispatch, erased callback, extra CUDA event, hidden stream, hidden
copy or extra wait is justified by functional syntax. Direct kernel arguments
and fixed small aggregates should survive inlining without extra device storage.

For standalone F2, `Pending` keeps the prediction/status owners alive and makes
the output unavailable as a validated result. `wait_checked` can enqueue the
status download into a stack object whose address remains stable through its
wait and exceptional drain. Do not enqueue a copy into a subsequently moved
pending object's inline member. Declare destination storage before the drain;
all device/input owners likewise outlive outstanding work. Failure drains and
invalidates the output before release. A destructor cannot silently turn an
unchecked output into a checked value. Do not promise recovery from device loss.

For the first R0 implementation, move separately prepared forest, batch and
exclusive output/status resources into a sealed operation/session. Let pending
and checked states own that bundle; avoid public raw writable views. This gives
one in-flight prediction and explicit consumption without per-call reference
counting or a new lifetime allocation. A later shared-model interface needs a
real immutable ownership lease, not a borrowed pointer to the public mutable
`Model`. C++ cannot statically prevent every escaped alias: keep unsafe interop
private initially and state/test the remaining foreign-runtime preconditions.

The forest is an immutable snapshot; later Model edits do not affect it. The
batch is encoded on the GPU using that exact snapshot's fitted feature schema.
Use ownership identity for the schema relationship, not only feature count or
an unchecked hash. Dimensions, device ordinal, capacity and stream dependencies
are part of the plan. A different batch requires its own checked GPU encoding.
Output storage may be overwritten only after consuming the prior checked value
and completing all readers. Start with closed checked completion and explicit
download; extend GPU consumer effects with owned dependencies before exporting
arbitrary asynchronous pointers. Retain the existing kernels initially.

This is a bounded functional migration of validity, planning and output lifetime.
Model packing, quantizer orchestration, traversal and training still require
their own migrations; wrapping them as named effects does not finish that work.

## Supported language choices

The completed
[installed feature audit](../results/functional-design-20260923/toolchain-features/REPORT.md)
provides 89 compile probes plus follow-ups, separately covering GCC host, nvcc
host and SM86 device compilation. Use C++23 `std::expected` with small error values
and monadic composition for host plans, and the verified static call operators,
explicit-object callables and folds where they shorten real value transformations.
Device counterparts must use the verified `cuda::std` facilities where needed;
host `std` support is not device support. Current implicit device lambdas need no
new extended-lambda flag. These probes do not establish linked runtime behavior
or absence of generated-code cost.

C++26 `cuda::std::saturating_*` and `inplace_vector` backports are available but
do not improve this fixed reduction/plan contract. Saturation would alter its
overflow semantics; a mutable bounded container adds no useful representation
here. Pack indexing, public device `bind_back`, device `ranges::fold_left` and
`function_ref` are unavailable on the required paths. Do not invent substitutes
or change flags simply to use newer spelling. No O(F²) prefix recomputation,
recursive runtime stack, persistent full-array copy or framework is accepted.

## Fair experiment and completion gate

1. Freeze current source, binaries, model and actual feature bytes. F2 compares
   CPU versus GPU finite checking under identical packing/encoding/forest and
   transform policies. Do not mix D2, E256, F1 or T1 into attribution.
2. Require zero raw/transformed bit differences against frozen B and per-tree
   references, existing CTests, tails/empty forests/missing groups and all
   objectives. Inject every NaN class and ±Inf after final transformation; test
   finite subnormals/±0, bad-then-good reuse, overflow, launch/copy/allocation
   failure and drain/recovery. Include the sigmoid intermediate-Inf case.
3. Inspect device code: untouched numerical sections must remain unchanged;
   added reduction has no unexpected local memory, extra array, FP arithmetic
   or hidden call. Apply memory/init/synchronization diagnostics where relevant.
   Record unsupported diagnostics and failures without reclassifying them.
4. Measure serial unprofiled complete vector calls with three warmups and at
   least 15 balanced paired samples for the large shape, actual Delicious,
   small-row/narrow-output controls, raw, regression and softmax. Use the existing
   upper-95%-interval-below-1 criterion for a casewise speedup claim. Preserve
   raw arrays, identities, telemetry and interference. No build/analysis load
   should overlap rankings.
5. Profile afterward: verify the CPU scan disappears, one GPU scan/status
   sequence appears, full host-result bytes remain, and prediction kernels do
   not change. Measure status/reduction costs and invalid-input contention.
6. Only then integrate R0 and report setup + Q calls + teardown for Q=1/2/10/100,
   checked resident execution, and required download separately and together.
   Include changed-input batches; do not call reuse a changed-input measurement.
   Retained/peak capacity and explicit GPU-consumer lifetimes are part of success.

T1 should remain an independent subsequent cell with unchanged transforms and
preparation. Use the existing 32-row × 32-output tile plan, inspect FP64 shared
bank behavior, and test tail barriers and multiple tile iterations. Its newly
measured motivation does not displace the R0 contract or permit numerical changes.
No policy/default or counting change is selected by this review.

## First integration boundary, selected before edits

While GPU execution is unavailable during the user's game, connect the F2
primitive to an explicit fifth `Model::predict_gpu` argument,
`OutputValidationPolicy::{host, device}`, defaulting to the existing host check.
Reject invalid policy values before the zero-row return. Both forest upload
paths reserve four additional device payload bytes for the device option and
own that status allocation before their existing drain guard. The host status
destination is also declared before that guard, so it cannot expire during an
exceptional pending copy. The default host option has no status allocation.

Use one private completion executor shared by both paths. It submits the
selected final-value check after transformation, the existing full prediction
download, and the optional status download, then performs the existing single
checked wait and rejects nonfinite results with the existing error text. This
is an explicitly effectful bridge into the frozen vector-return interface,
not the later owning `Ready/Pending/Checked` session. Keeping this distinction
visible prevents a helper refactor from being represented as a completed
functional architecture. The reduction predicate and launch plan are immutable
value computations; foreign execution, copying and waiting are effects.

The CPU comparison uses the same finite-value predicate and short-circuit
validation semantics. No numerical kernel, full-result download or default is
removed by this integration. Add the explicit option to the benchmark and
existing prediction checks, and connect a CUDA-free predicate/plan test target.
No unused production entry point or unused generic reduction utility is needed.
Compile/link and CPU-only checks may run now; GPU correctness, sanitizer,
profiling and timing gates remain unexecuted until the user releases the GPU.

## Concrete F2 lowering selected before code

The user is gaming. This implementation slice permits source changes and CPU
compilation/reference checks only; no GPU launch, runtime device query, profiler
or GPU diagnostic is authorized until the GPU is released.

Select one element per thread, 256 threads per block, with a full-cover x-grid.
The immutable host plan rejects an overflowing byte extent, then rejects a grid
larger than 2,147,483,647 blocks, the applicable SM86 x-grid limit. Ceiling division
uses quotient/remainder rather than an overflowing `elements + 255`. The maximum
accepted extent is 549,755,813,632 doubles on a 64-bit host; this is a launch bound,
not a promise of available allocation capacity. The primitive owns no allocation.

Compared with a capped grid-stride reduction, this lowering avoids a mutable
per-thread accumulator and scheduling loop and reads each valid element exactly
once. It can launch more blocks and rejects theoretical extents beyond its grid
coverage; a capped grid-stride path supports larger extents and may amortize block
scheduling better. Compared with two-stage block flags, it has no scratch array
or second launch but can contend on densely invalid inputs. Select the smallest
full-cover lowering for this prototype; timing must determine its performance.

The mathematical computation is `OR(map(nonfinite_bits, input))`, with identity
zero and contributions in {0,1}. A C++23 stateless static-call predicate returns
the contribution from exact binary64 exponent bits. The real warp vote and elected
atomic OR implement the algebra; no unused OR wrapper or general reduction library
is introduced. All threads participate in the full-mask vote, including tail
threads contributing zero. The kernel has no early exit before that collective,
shared memory, block barrier, intermediate array or recursive call. Each bad warp
issues at most one atomic; an all-finite input issues none.

The explicit asynchronous API is `gpu::validate_finite(values, elements, status,
stream)`. Argument/extent/grid rejection precedes every CUDA effect. A valid call
first clears the caller-owned four-byte status to zero; zero elements allow a null
input and require only that clear, without a kernel. Nonzero input and status must
refer to live, sufficiently sized, aligned, nonoverlapping allocations on the
stream's device. The input is the final requested output after transformation,
and preceding producers are ordered on that stream. The function allocates,
downloads and waits for nothing. It reports immediate argument/runtime/launch
status; execution and semantic nonfinite errors require later checked completion.
The caller keeps both allocations alive until that completion, even after a
submission error. Clear/launch/global reads/warp vote/atomic writes are explicitly
effects. The plan and scalar predicate alone are pure value transformations.

Add only the live predicate, checked plan and CUDA submission primitive. The
separate integration may select this prototype explicitly while retaining the
CPU scan as the default. GPU correctness, generated-code diagnostics and complete
operation measurements remain pending; no speed or promotion claim follows from
CPU-only implementation.
