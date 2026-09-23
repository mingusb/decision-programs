# Resident models, inference, and GHBMODEL v1

All APIs are submitted by one GPU thread. Descriptors and their backing arrays
are global device storage; capacities, products, alignment and arena bounds are
checked before access. Status starts at zero and is consumed only after its tail
completion. Inputs, output, scratch, byte buffers and status are disjoint and
remain alive through completion. Pointer liveness is a caller precondition.
Prediction and export consume a previously successfully validated model whose
descriptors and contents have not changed. Import performs full validation.

## Choice recorded before implementation

The model keeps features and metadata in feature order, feature-major encoded
input, and output-grouped tree descriptors in stable per-output order. Node
segments may be permuted: they must be nonempty, disjoint, and cover the complete
node extent. This lets round-major training append nodes without a second copy.
Feature metadata is finite and strictly increasing; numeric/category bin counts
include missing zero and may reach 65536. Offsets and declared totals must agree.

Choose GPU node-coverage claims, unique-parent claims, and bounded pointer jumping
for structural validation. Tree-local CTA traversal keeps each tree's base/count
explicit and supports arbitrary node numbering/depth. Coverage storage is reused
as parent storage after the coverage check; two u32 arrays cost eight bytes per
node, plus an eight-byte maximum-leaf value per tree and a small control record.
Each valid node has one coverage claim and each nonroot one parent claim. After
ceil(log2(maximum tree size)) pointer-jump rounds every parent must be root zero.
This detects overlaps, holes, duplicate children, disconnected cycles, and
unreachable nodes without a recursive DFS or copying nodes into descriptor order.

A serial DFS is O(nodes) but serializes validation. Independent root walks cost
O(nodes*depth). Sorting segments avoids coverage claims but adds a sorting stage
and does not establish tree reachability. The selected bounded-jump baseline has
O(nodes log maximum-tree-size) traffic and synchronization; shallow forests with
many trees fit its parallelism better than a single giant tree. These tradeoffs
are not a measured ranking. Primary device-effect evidence is NVIDIA's
[CDP2 memory/stream contract](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html)
and [CUDA atomic operations](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html).

Preserve the frozen model bound: absolute base plus each tree's maximum absolute
leaf, in original per-output order, may not exceed DBL_MAX. The archived host
uses x86 extended precision with a 64-bit significand. A positive integer
significand/exponent accumulator emulates those rounded additions on GPU; neither
FP64 accumulation nor an exact real sum has the same boundary acceptance. In
particular DBL_MAX plus a sufficiently tiny leaf remains accepted. Nonfinite
values, invalid leaf fields, child indices, feature indices, missing flags and
thresholds are rejected before prediction is permitted.

Prediction keeps base-first sequential __dadd_rn for each row/output. Sigmoid uses
the sign-stable legacy expression. Softmax keeps lane-strided accumulation and
the 16/8/4/2/1 shuffle tree; final values are checked after transformation. Thus
infinite raw margins that transform to finite sigmoid values are checked at the
correct stage. Prediction validates every supplied bin against its feature domain.

## Byte format and GPU framing

GHBMODEL v1 is little endian: eight magic bytes; u32 version/objective/outputs/
features; u64 tree count; FP64 bases; each feature's u32 type/cut count/category
count and FP32 metadata; then each tree's u32 output/node count and nodes encoded
as three i32 indices, u32 threshold, u32 missing flag, FP64 value (28 bytes).
No native padding, pointer, or size_t is serialized. Reject truncation, trailing
bytes, incompatible counts, unknown tags and payloads beyond one GiB. Preserve
the archived one-GiB decoded-allocation safety accounting (8/output, 56/feature,
32/tree, 4/metadata value, 32/node), independent of the new smaller descriptors.

A GPU coordinator scans variable-length framing, checks all capacities, and
constructs offsets. Bulk payload kernels transfer values between wire bytes and
resident arrays. Import appends nodes in wire order and stable-groups only tree
descriptors; export emits grouped descriptor order, retaining per-output arithmetic
and the byte format, without promising original inter-output byte ordering.
Framing scratch is reused by validation only after ordered payload children finish.
Export writes a separate u64 byte count; Status.required_bytes describes scratch.

## Acceptance and experiment

Independent GPU fixtures cover signed zero, ordered cancellation, missing and
numeric/category routes, softmax landmarks, malformed graphs and permuted node
segments; numerical-bound ties; insufficient/misaligned arenas and capacities;
literal little-endian bytes, codec roundtrips, truncation/trailing bytes, and
malformed serialized tags/counts. Guard output/storage and retain input bytes.
Only root runs GPU tests and sanitizers, serially. Compare complete validation,
prediction, import and export operations separately on shallow forests, deep
trees, large output counts and skewed sizes. Include framing, clearing, payload
work and completion; retain raw unprofiled samples, workspace and Nsight evidence.
No implementation here is called fastest before that evidence exists.
