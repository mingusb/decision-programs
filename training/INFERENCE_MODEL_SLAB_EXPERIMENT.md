# Single-allocation model upload for ordered forest inference

Recorded 2026-09-23 before implementation. Candidate E changes only the host
allocation, packing and upload of the four fused-model arrays. Reference B is
the existing explicit `PredictionPolicy::fused_output`; `per_tree` remains a
separate control and the default. No production code, default or algorithm has
been changed by this document. Analysis was read-only apart from this document;
no build, GPU work or timing was performed by the author.

## Evidence and bounded question

The idle prediction matrix in
`results/optimization-20260923/prediction-matrix-idle/` retains two small-model
cases with slower B medians that motivate the experiment:

| Frozen model and rows | Per-tree median | B median |
| --- | ---: | ---: |
| `independent3-rows32.json`, 32 rows | 0.627620 ms | 0.640173 ms |
| `independent3-rows4096.json`, 4,096 rows | 0.690305 ms | 0.746597 ms |

Both use three outputs, three trees and 13 nodes from
`results/profiling-expansion-20260923/model-before/model.ghb`, SHA-256
`25fb9bfd91f85787d70191c93e2dac309b8e5d51619286548ec20c739626f978`.
Their invocation receipts identify benchmark SHA-256
`d3248d75a8759d74d6c0aae0b96f02dadf9966c84f6e4768c18dab243ce998b0`.
The parent reports wins at 65, 983 and 1,024 outputs. Medians identify cases to
investigate; the paired intervals for these two cases include 1, so they do not
establish regressions. They do not establish the cause or predict E's result.

Current `training/src/booster.cpp:Model::predict_gpu` constructs independent
`Device` objects for predictions, base scores, all nodes, tree descriptors and
output offsets. `Device<T>` performs one nonzero `cudaMalloc` in its constructor,
one `cudaFree` in its destructor, and one `cudaMemcpyAsync` per nonzero upload.
For a nonempty fused forest the **model alone** therefore uses four allocations,
four frees and four uploads. In contrast, per-tree uses two model allocations
(base and maximum-tree scratch), one base upload and one upload per tree.
E's three removed allocations are relative to B, not relative to per-tree.

The small model has 24 base bytes, 416 node bytes, 48 descriptor bytes and
32 offset bytes: 520 uploaded model bytes in B. Allocation and API overhead are
plausible targets when payloads are this small, but kernel scheduling, encoding,
validation, packing and system noise remain other explanations.

The separately isolated Delicious B capture
`results/optimization-20260923/nsys-prediction-fused/profile.1.sqlite` observed
seven whole-call allocations/frees, 538 H2D activities, 539 `cudaMemcpyAsync`
calls, 16 `cudaMemcpy2DAsync` calls and 18 kernel launches. E alone predicts
four allocations/frees, 535 H2D activities and 536 `cudaMemcpyAsync` calls for
that same nonempty model, with the other counts unchanged. These are
source-derived expectations, not measured E results. An API upload need not
correspond one-for-one to a physical transfer under every runtime configuration.

## Exact contract

Use the same validated immutable-for-the-call Model and Dataset, CUDA C++23,
current SM86 target, context/device and nonblocking stream. Preserve all
model/input checks, zero-row early return and finite-result checks. Quantization,
numeric/categorical/missing routing, objectives, tree traversal and the GPU
transform are unchanged. Host model packing is setup; no feature encoding or
learning is moved to the CPU.

Nodes and descriptors remain stably grouped by output with the exact same
original per-output tree subsequence. Child indices remain relative to their
tree. The unchanged `gpu::predict_forest` receives typed pointers to slab
regions plus the same counts. Every base and leaf FP64 bit, including signed
zero and subnormal values, is preserved; `__dadd_rn` order remains identical.
No tree, zero addition, validation pass or transform is omitted.

Prediction storage remains a separate writable allocation. The model slab is
read-only while kernels use it and does not overlap predictions or quantizer
storage. Empty forests retain valid base/output-offset regions and pass null
node/descriptor pointers with zero counts. They reduce two model allocations
and uploads to one, not four to one. Zero-row inference still performs no CUDA
work after the existing validation and result construction.

The synchronous host-result API remains synchronous. No cache, pooled allocator,
stream reuse, graph capture or asynchronous public API is introduced in E.
`Model` is publicly mutable, so a hidden persistent slab keyed by its address
would be incorrect. There is no unmeasured fallback or automatic policy switch.

## Layout, checked arithmetic and memory accounting

Use one byte-addressed device allocation and four ordered regions:
base, nodes, descriptors, output offsets. The first bounded variant E256 aligns
each nonempty region start to **256 bytes**, preserving the guaranteed alignment
of B's independent CUDA allocations. This does not promise identical addresses
or cache mapping. Natural `alignof(T)` alignment alone is sufficient for typed
access correctness but changes array-start alignment; treat that more compact
layout as a separately named variant, not an invisible benchmark change.

For each region, check count times `sizeof(T)` before adding it. Compute alignment
padding from the current offset remainder, and use checked addition for both
padding and payload; do not rely on unchecked `(offset + 255) & ~255`. Check the
final slab extent and every region end. Require the involved types to remain
trivially copyable, with their alignments dividing 256. Calculate sizes with
`sizeof`, not hardcoded ABI assumptions; the currently measured ABI has
32-byte Node and 16-byte PredictionTree. Preserve existing u32 per-tree node
counts and u64 packed offsets and their overflow checks.

Skip absent node/descriptor regions rather than making pointers into fictitious
objects. Base and offsets are nonempty for every valid nonzero-row call. No
trailing 256-byte rounding is required. Four nonempty regions add at most
3*255=765 bytes beyond the sum of model payloads. Thus padding is bounded by
the number of arrays, independent of tree size or feature-metadata skew.

For the small measured model, starts are 0, 256, 768 and 1,024; the slab ends at
1,056 bytes. It uploads 536 extra padding bytes relative to B's 520 bytes.
This is deliberately explicit: fewer calls does not mean fewer transferred
bytes. All padding bytes actually uploaded must be initialized. Node object
padding is not model semantics and must not be included in an exactness gate;
all named fields and their floating-point representations are the gate.

Compute the exact slab extent before querying/passing the quantizer budget.
Replace B's reserved inference payload with checked
`prediction_elements*sizeof(double) + slab_extent`. Use that full padded extent
in the same free-memory preflight and quantizer remainder calculation. Report
the padding and any resulting quantizer tile-width change, especially near
tight memory limits; otherwise the supposedly local experiment could silently
change encoding's tile count. Hold the selected quantizer layout constant in
the main performance comparison by supplying ample headroom, and test boundary
behavior separately.

`cudaMemGetInfo` is a snapshot, not a reservation or a guarantee about allocator
bookkeeping/fragmentation. A larger contiguous allocation can fail even when
separate allocations or the preflight would succeed. Propagate host allocation
failure and CUDA allocation errors explicitly; release owned buffers on failure,
and do not quietly fall back. An E-only rejection caused by padding/contiguity
must be reported as a capacity difference. Keep host staging bytes, host helper
arrays, device resident payload and peak device scratch distinct in reports.

## Host packing, lifetimes and errors

Pack directly into the final host upload slab using the existing stable
grouping order. Avoid building a complete packed node vector and then copying
it again into the slab: that adds another full-model host read/write pass and
another live host payload. The current order/prefix helper arrays may remain.
Copy trivially-copyable objects by byte operations into checked regions; do not
dereference misaligned typed pointers inside a byte vector. Only device region
addresses need the 256-byte property if the host staging buffer is accessed
solely as bytes. Typed host staging instead requires its own alignment and
object-lifetime guarantees.

Prefer staging that does not zero the entire model merely to initialize small
alignment gaps; explicitly initialize all gaps, descriptors' reserved fields
and every payload range. If the first implementation uses a zero-initialized
byte vector, retain and charge that full memory pass in the timing boundary.
No staging buffer may include uninitialized inter-array padding in the upload.
Do not allocate a second device slab and copy from it, or add a GPU unpack kernel.

Use ordinary pageable staging initially, consistent with B's current vectors.
Pinning the slab, registering existing storage or using a persistent pinned
allocator are separate candidates with allocation/registration costs included.
An Async-named copy from pageable storage may stage or block; do not assume it
has finished reading its host source when the call returns.

The final host slab and device owner must outlive all queued accesses, including
exceptional paths. Preserve RAII destruction order: the stream exists first;
quantized data, prediction/slab owners and host staging precede a drain guard;
the drain is destroyed before those resources. Complete packing before queuing
its single upload, submit upload then forest/transform/download on the same
stream, and retain the existing checked final wait before inspecting results.
Only free the allocation base, never its typed interior pointers. No owning
`Device<T>` wrapper may independently free an interior view.

The existing drain/destructor waits are cleanup protection and do not replace
checked normal completion. A CUDA error must not become a partially returned
prediction vector. Removing three API calls can change which runtime call
reports an already-pending asynchronous error; identical error-message wording
and failure timing are not promised, but validation and error propagation are.

## Candidate choice versus other approaches

| Candidate | Work removed | Cost and scope |
| --- | --- | --- |
| E256, selected bounded experiment | Three model allocation/free pairs and three model uploads per nonempty B call | At most 765 alignment bytes, one host slab; same kernel, same synchronous API |
| D1 immutable feature metadata packing | Hundreds of tiny quantizer metadata copies on Delicious | Independent encoding setup change; padded metadata can grow drastically under skew |
| D2 final-only quantizer status synchronization | Fifteen intermediate status downloads/waits for the 16-tile Delicious case | Requires immutable upload sources, stream ordering and final cumulative status checks |
| `cudaMallocAsync`/memory pool | Some allocator synchronization/reuse costs | Does not itself merge uploads; new pool residency and stream-ordered lifetime contract |
| Explicit frozen GPU-resident predictor | Repeated model packing, allocation, upload, validation and potentially feature-metadata upload | New construction/use/destruction API, persistent device capacity and concurrency contract |

E has much smaller scope than D, and E's source count reduction is only three
copies regardless of feature count. Do not infer that E beats D because it
targets the small-model cases, or infer that D improves them from its
500-feature call reduction. Benchmark E against unchanged B first. Evaluate
E+D only after each independent experiment passes and preserve all comparisons.

The explicit resident architecture removes more repeated work than any slab
created inside every call. It would construct a validated immutable snapshot
bound to a device/context, copy all model/feature metadata once, then accept
repeated batches. Model mutations require a new snapshot. Mutable per-request
predictions, bins and status need separate workspaces or explicitly serialized
use; concurrent calls must not race on shared scratch. Construction/destruction,
resident memory, cold first call and warm calls must be reported separately.
If offered device-resident features/results, keep that boundary distinct from
the existing host-to-host API. For Q batches compare `setup + sum(call_i) +
teardown`, including Q=1; never compare its warmed kernel-only time to complete
B/E calls. This larger architecture is promising but is not implemented or
silently substituted for the bounded slab question.

NVIDIA's current Best Practices Guide recommends batching transfers and retaining
data on device, and documents CUDA allocation alignment. It supports the
mechanisms above, not a local performance ranking:
https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#data-transfer-between-host-and-device
and the same guide's section 10.2.1.2 on misaligned sequential access.
The installed CUDA13.4 `cuda_runtime_api.h:4734` documents allocation alignment
and allocation failure. CUDA's synchronization reference explains pageable
Async staging/blocking:
https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html
No NVIDIA algorithm implementation is introduced.

## Fair B-versus-E experiment and acceptance gates

Keep B directly callable in the same build as an explicit allocation-policy
control; do not infer B from an old binary with different host flags. Preserve
all kernel source and launch parameters, arithmetic/optimization flags,
quantizer behavior and per-tree default. Compiler time traces must be disabled
in ranking binaries. Compare generated device text/constants to confirm that
the experiment changes host setup only; unchanged bytes are evidence about
code generation, not a proof that cache behavior is identical.

First correctness gates require exact raw and transformed output bits against
B on the same frozen models and input bytes. Reuse the full prediction suite:
all objectives; categorical/unseen/missing inputs and numeric boundaries; row
tails; arbitrary interleaved tree outputs; outputs without trees; zero trees
and zero rows; signed zero, subnormals and cancellation. Compare packed named
fields/counts/order separately, and check every nonempty device address is
aligned and regions neither overlap nor exceed allocation. Include odd output,
tree and node counts that force padding, changed-model calls, repeated calls,
exception cleanup, overflow arithmetic and empty-region handling. CPU layout
tests cover boundary values without attempting impossible allocations.

Run bounded Compute Sanitizer memcheck/initcheck and lifecycle tests through
the root's serial GPU queue. Verify malformed model/input rejection and
explicit budget/host/CUDA failure behavior. A zero-allowance mismatch fails E;
unchanged training does not waive prediction exactness. No retraining is needed
for this inference experiment. Actual held-out predictions and all applicable
saved prediction metrics must remain identical.

When the GPU is idle and the parent authorizes measurement, run the exact two
small-model cases first, plus the same frozen 65-, 983- and 1,024-output
controls already used in B's matrix. Include zero-tree/one-tree controls and
both small and large row batches; do not select only the shape where E wins.
Use three warmups and at least 15 serial paired samples per shape, balanced B/E
order, with same inputs, raw flag and process/context warmup. Preserve every
sample, failure, source/build/model/input hash and Windows/native GPU telemetry.
Agent CPU/build activity must not compete with timing.

Time the **complete synchronous host-to-host call**, including validation,
host packing/zeroing, allocation, quantization, all transfers, kernels,
download, checked completion and cleanup. Exclude model-file loading and
context warmup equally. CPU phase/event diagnostics may explain the result
but cannot replace that boundary. Run Nsight only afterward to verify the
allocation/copy count change and unchanged kernel sequence; profiler timing
does not rank B and E.

Report paired ratios with uncertainty, including E versus B and the separate
per-tree control. A case-level speedup requires the 95% paired-bootstrap
interval's upper bound for E/B below 1.0, with all correctness gates passed.
Do not claim the original small-model concern resolved unless E is also competitive with
the matched per-tree control on those same cases. Any default promotion would
require a separately agreed complete matrix with no measured regression or
capacity surprise. No speedup or default change is authorized by this design
alone.

## Implementation notes, 2026-09-23

The parent selected E256 in
`results/optimization-20260923/MODEL_SLAB_SELECTION.md` before code changes and
preserved the A/B sources and binaries in `candidate-ab-source/`. The explicit
`PredictionPolicy::fused_output_slab` now has its own branch in
`Model::predict_gpu`; the original B/per-tree body remains intact. The CUDA
ordered-forest kernel and its API were not changed.

The implementation computes four checked region extents and the padded device
reservation before invoking the quantizer. It directly copies nodes into one
host slab, writes descriptors individually and reuses one prefix vector as
stable grouping cursors after copying the output offsets. No second packed
forest is constructed. Host staging uses uninitialized 256-aligned byte blocks;
all inter-array gaps are explicitly zeroed and every payload range is filled
before the single upload. Host capacity is rounded to 256 bytes, unlike the
device/upload extent; the unused host suffix is not uploaded. Helper storage is
one `(outputs+1)` u64 vector and one `trees` size_t order vector. Owned host and
device upload storage precedes the drain guard.

`ghb_prediction_bench --policy compare-slab` times B versus E in alternating
pairs; `--policy fused-output-slab` isolates E. The existing default `both`
still compares per-tree versus B. Every mode first computes raw/transformed
per-tree reference predictions, checks the selected policies exactly and keeps
these checks plus warmups outside the optional profile capture. JSON reports
which policies were checked, the padded device/upload extent, rounded host
capacity, alignment gap bytes, auxiliary bytes and all four region offsets.
The timing boundary includes all slab preparation and cleanup.

E is included in the existing public Model objective/forest/row-tail/signed-zero/
subnormal/malformed-input checks; the separate direct-kernel graph cases remain
unchanged. Implementation was reviewed statically without compilation or GPU
work by this agent. Builds, sanitizer execution, exact comparison runs,
generated-code comparison and performance measurements remain the root's
serial validation work; this implementation note claims none of those passed.

Before the next change, the parent authorized extracting only E256's existing
checked region/reservation arithmetic into an internal host-only helper. The
public prediction API otherwise cannot exercise arithmetic overflow or exact
free-budget boundaries without materializing huge models or entering CUDA.
The helper will be used by production E and a standalone CPU test, so boundary
tests exercise the actual arithmetic rather than a duplicate formula. Layout,
policy, kernel, ordering and the B/per-tree bodies must stay unchanged. Tests
will use hand-derived odd/empty/aligned examples and overflow/budget cases;
they will allocate no device memory or giant host arrays. Six pre-extraction
source files and hashes are preserved in
`results/prediction-slab-20260923/pre-layout-extraction/` before edits. Allocation
failure propagation and GPU lifetime/sanitizer checks remain separate work.

The extracted helper is `include/ghb/detail/prediction_layout.hpp`. Production
passes its actual Node/descriptor `sizeof` values to the same template called
by `ghb_prediction_layout_tests` (`ctest -R '^prediction_layout$'`). That target
links neither CUDA nor the booster. It checks fixed odd, empty, aligned and
wide layouts, distinct overflow stages, and strict reservation/remainder
boundaries without constructing huge arrays. The benchmark retains its
independent byte-accounting calculation, allowing its report to be checked
against actual helper results instead of sharing a reporting error. These
tests were added but not built/run by this agent; the root owns execution.
