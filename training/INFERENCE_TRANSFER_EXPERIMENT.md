# Inference metadata transfers and tile synchronization

Recorded 2026-09-23 before implementation. This document proposes separate
candidates D1 and D2 against the explicit fused-output predictor B. It changes
no source, policy or default. Production feature encoding remains on the GPU.
Host packing concerns already fitted model metadata, not feature preprocessing.

The user has confirmed the game is active and requested correctness work for
now. No performance benchmarks or ranking confirmation will run until the GPU
is idle. Existing observations made during gaming remain diagnostic evidence;
they are not discarded or reclassified as uncontended performance results.

## Fresh evidence and scope

The isolated measured-sample capture is
`results/optimization-20260923/nsys-prediction-fused/profile.1.sqlite`, with
command, source/binary identity and settings in its adjacent `manifest.json`.
Prechecks and warmups are outside the capture. The frozen Delicious model has
500 features, 983 outputs and 983 trees; the actual validation input has 2,584
rows. Reading the model confirms **every feature has one FP32 metadata value**.
Read-only SQLite queries find:

| Work within one fused inference call | Observed count |
| --- | ---: |
| `ordered_forest` | 1 |
| `encode_tile` | 16 |
| `sigmoid_kernel` | 1 |
| H2D transfer activities | 538 |
| Four-byte H2D transfers for feature dictionaries | 500 |
| `cudaMemcpyAsync` API calls | 539 |
| `cudaMemcpy2DAsync` API calls | 16 |
| Four-byte status D2H transfers | 16 |
| `cudaStreamSynchronize` API calls | 19 |

The 538 H2D activities consist of 500 dictionaries, 16 size arrays, 16 feature
input tiles, two feature type/bin-offset arrays, and four fused-model arrays.
The reference's previously observed 1,518 H2D activities and 1,001 kernel
launches have already been reduced by B; D must not claim that removed work as
its own gain. These activity counts identify remaining work, not its elapsed
time under uncontended conditions.

The current `encode_quantize` in `training/src/quantize.cu` validates model
metadata, chooses a tile width K<=32 within the device-memory limit, uploads
each feature dictionary separately, uploads K dictionary lengths, launches the
existing encoder, reads cumulative status, and waits after every tile. With
500 features this means T=ceil(500/32)=16 tiles. D1 addresses metadata copies;
D2 addresses unnecessary host checkpoints. Test them independently before
combining them.

## Shared input, numerical, memory and completion contract

Inputs remain immutable for the duration of the call: row-major FP32 Dataset
values and the fitted numerical cuts/categorical dictionaries in Model.
Numeric bins are one plus lower_bound; category bins are one plus the exact
matching dictionary index, otherwise zero. NaNs map to missing bin zero;
positive/negative infinity is rejected. Signed-zero equality, strict dictionary
ordering, missing handling, unseen categories and all u16 bin limits remain
unchanged. No refitting, rounding, sorting, approximate search or CPU feature
encoding is introduced.

Output remains the same owned feature-major u16 bins, u32 histogram offsets,
feature types and copied Feature metadata in QuantizedData. Compare every bin
exactly, every metadata float bitwise, all view fields and allocation accounting.
Both prediction policies retain their existing FP64 tree addition and transform
behavior. D must match B bit for bit on the same frozen model and feature bytes,
for both raw margins and transformed output.

`encode_quantize` remains synchronous at the **whole preparation boundary**:
all its CUDA work and status validation are complete before return, so callers
may discard Dataset afterward. It still rejects stream capture. Device payload
must remain within memory_limit, with overflow-checked sizes and alignment;
this limit excludes host metadata and other caller-owned allocations. Preserve
`resident_bytes` and truthful `peak_bytes`. Low-budget one-feature tiling remains
supported for variants that retain the existing device layout.

Training `fit_quantize` is excluded. It creates metadata on the GPU and needs
different lifetime/size dependencies; removing its waits by analogy would be
incorrect. Histogram kernels, counting defaults, tree training and model
serialization are outside D.

## D1: batch immutable fitted metadata

The first bounded D1 variant, D1-P, retains the existing encoder kernel and
device scratch layout. Build immutable host blocks for all inference tiles.
Each block exactly represents the adjacent `sizes` and `metadata` regions:
K u32 lengths followed by K*M FP32 dictionary slots, where
M=max(1, maximum dictionary length). Initialize unused slots and tail-feature
padding, preserve every live dictionary value bit, and copy one contiguous
block per tile into the existing device_sizes/metadata allocation. Continue
the original per-tile status download and wait for this independent experiment.

The existing `layout(..., fit=false)` already places sizes immediately before
metadata. D1-P therefore needs no scatter/unpack kernel or device representation
change. The encoder receives the same sizes, metadata values and meta_stride.
The host blocks remain immutable and alive until preparation completes, making
the D1-P representation safe to compose with D2 later. No hidden cache of a
publicly mutable Model is introduced.

For this Delicious model, 500 dictionary copies plus 16 size copies become 16
combined copies. Expected activity counts, conditional on unchanged runtime
lowering, are H2D 538->38 and cudaMemcpyAsync 539->39, while 16 encode launches
and 19 explicit stream synchronizations remain. Packed metadata plus lengths
occupy 16*32*(1+1)*4 = 4,096 host bytes and transfer bytes; existing live values
plus lengths transfer 4,000 bytes. The extra 96 bytes are explicit padding.
These are source-derived expectations to verify with Nsight, not measurements
of D1 or a speedup estimate.

D1 alternatives must be compared before generalizing beyond tiny dictionaries:

| Representation | Transfer/host costs | Device and kernel consequences |
| --- | --- | --- |
| D1-P, fixed padded tile blocks | 4*T*K*(M+1) host/transfer bytes, T calls | Existing scratch and encoder unchanged |
| Per-tile maximum stride | Sum over tiles of 4*K*(M_tile+1); same T calls | Existing maximum scratch can hold each tile; per-tile stride/address contract must be tested separately |
| Fully compact resident dictionaries | 4*sum(m_f) values plus O(F) offsets/lengths | Whole dictionary stays resident; encoder needs compact offsets; reserve/account added GPU payload |
| Native per-tile batched copies | Existing live dictionary bytes plus lengths; O(F) pointer/size descriptors | Existing scratch/kernel; runtime batches API submission but need not merge physical transfers |

A single 65,535-entry categorical dictionary among 499 one-entry dictionaries
shows why global padding cannot be an unconditional policy. With K=32, fixed
blocks transfer 128 MiB including lengths, versus 264,136 bytes of actual
dictionary values and 2,000 bytes of live lengths. If the large dictionary is
in one tile, per-tile maximum padding is about 8 MiB plus small other tiles;
compact values avoid that amplification. Conversely, fully resident compact
metadata can use more GPU memory than the tiled reference when most features
have equally large dictionaries. No data-dependent speed ranking follows from
these byte counts alone.

Use explicit policies and a declared host-packing budget for experiments;
overflow or inability to allocate the chosen representation must be reported.
Do not silently replace the low-memory reference or classify a skewed model
as supported by D1-P without accounting for its padded allocation. Keep the
reference callable. Host metadata allocations and temporary packing bytes are
reported separately from the device memory_limit.

CUDA13.4's `cudaMemcpyBatchAsync` is a relevant infrastructure alternative to
packing, particularly for skewed dictionaries. Installed
`cuda_runtime_api.h:6529` and current Runtime documentation agree on its
eight-argument signature; older examples with a fail-index argument do not
match this installation. Each tile's dictionary destinations and size-array
destination are disjoint and can be one batch. The batch executes in stream
order, but its member copies are unordered: **do not batch multiple tiles that
reuse the same device metadata buffer across intervening encoder kernels**.
Begin with stream source ordering and immutable source/descriptor lifetimes;
evaluate other access-order attributes only in a separately specified variant.
An API-count reduction is not proof of fewer DMA transactions or faster copies.

Pinned versus pageable host packing is also an independent parameter. Pinned
storage can improve asynchronous behavior, but allocation/registration cost
must be included in complete-call timings. Do not pin every small model vector
individually, assume WSL pageable copies overlap, or add pinned allocations
outside the timing boundary without a separate reusable-setup API contract.

## D2: one status checkpoint after all inference tiles

D2 can first be tested against unmodified B without D1. Precompute an immutable
F-entry host length array rather than rewriting the current K-entry `sizes`
vector before each asynchronous upload. Source dictionaries, feature types,
histogram offsets and Dataset values already remain immutable throughout the
call. Then enqueue each input upload, metadata/size uploads and encode kernel
in the **same stream**, reusing the existing bounded device scratch. Remove
intermediate status copies/waits and perform one status D2H copy and one checked
stream synchronization after the last tile. Validate cumulative status before
returning or launching prediction.

This order is essential: upload tile j -> encode tile j -> upload tile j+1.
Stream ordering protects input/metadata scratch from overwrite while a prior
encoder reads it. Separate streams, double buffering, overlap and graph capture
are not part of D2. The current status is cleared once and `encode_tile` only
ORs error bits, so a late single read retains any earlier infinity detection.
Valid output bits are unaffected by the checkpoint location.

Every asynchronous host source and the final status destination must outlive
the last dependent operation, including exceptional exits. Put all owned host
arrays/staging before the existing Drain guard in declaration order; the guard
must synchronize before their destructors and before scratch destruction.
No stack-local or reusable tile vector may be overwritten merely because
cudaMemcpyAsync returned. For D1-P+D2, all packed tile blocks remain immutable.
For native batches, conservatively retain the pointer/length/attribute arrays
until completion too. External Dataset/Model lifetime remains the caller's
responsibility for the complete synchronous call.

Preserve host validation and every CUDA enqueue/launch-error check. A normal
invalid-value input still throws std::invalid_argument before returning usable
bins, but detection moves from the first invalid tile to the final checkpoint,
so later tiles may already have run. Early-abort latency is not preserved.
If a separate CUDA failure also occurs, do not promise identical exception
precedence between that failure and an input-status error. Drain safely and
report the encountered failure; do not claim the API is now asynchronous.

D2 alone leaves 538 H2D transfers and 18 kernels in this captured case, but
predicts status D2H 16->1, cudaMemcpyAsync 539->524 and explicit stream waits
19->4 (one quantization completion plus the three existing prediction/lifetime
waits). D1-P+D2 predicts 38 H2D, two D2H activities, 24 cudaMemcpyAsync calls,
16 cudaMemcpy2DAsync calls and four explicit waits. Runtime-internal blocking
may remain, especially with pageable inputs; removing explicit waits does not
establish copy/compute overlap or an equal reduction in CPU stall time.

## Primary approaches and rejected scope expansion

NVIDIA's [Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#data-transfer-between-host-and-device)
recommends reducing transfers and batching small transfers; it also identifies
pinning as a heavyweight operation whose complete cost must be measured.
The [Runtime synchronization contract](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)
allows pageable asynchronous copies to synchronize or stage internally.
The [current batch-copy specification](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html)
defines source access-order attributes and forbids dependent copies within a
batch. These establish applicable mechanisms, not a local ranking.

The larger architectural alternative is an explicitly owned reusable inference
session with uploaded model metadata and device inputs; it can amortize or
remove preparation entirely. That changes ownership and setup-amortization
boundaries and must remain a separate API experiment. Uploading the full input
once and changing the encoder's source stride, GPU transpose/gather, zero-copy
host reads, persistent caches and multi-stream pipelining are likewise separate.
This study first removes demonstrated transfer/checkpoint work without changing
encoding arithmetic or adding a CPU feature stage.

## Exact correctness and lifetime gates

Reuse the independent CPU encoding oracle and all current quantization tests,
then compare B, D1 alone, D2 alone and D1+D2 on fixed inputs. Required gates:

- Zero different u16 bins, metadata float bits, feature offsets/types or view
  bounds. Every reported resident/peak allocation stays within the supplied
  budget; D1-P/D2 retain the existing minimum one-feature device budget.
- Numeric threshold values and adjacent floats, positive/negative zero,
  FP32 subnormals/extremes, multiple NaN payloads, missing/unseen categories,
  empty dictionaries, 65,534 numeric cuts and 65,535 category entries.
- F=1/31/32/33/63/64/65/500, row tails, incomplete final tiles, forced K=1 and
  intermediate K via tight budgets, all-empty dictionaries, one large outlier,
  and mixed numeric/categorical dictionaries with skewed lengths. Test the
  exact minimum device budget and one byte below it.
- Infinity in the first/middle/last tile; invalid metadata/order/type/extent;
  overflow rejection before allocation. Verify failed calls leave no queued
  accesses to freed host or device storage and the stream remains usable after
  ordinary validation rejection. CUDA-fatal-error recovery is not promised.
- Whole-operation completion: immediately release input/model storage after
  successful encode_quantize and consume the returned bins on another stream
  using the existing completed-preparation contract. Test repeated calls with
  changed inputs and independent calls on distinct streams/scratch. Capture
  rejection must remain intact; no source-lifetime assumption may depend on
  pageable transfers accidentally being synchronous.
- Memcheck/initcheck/synccheck/racecheck where applicable, including a pinned
  staging variant that actually permits queued transfers. Normal outputs and
  error paths must pass; no sanitizer suppression hides staging lifetime bugs.
- Frozen-model prediction bits match B for regression, binary and multiclass,
  raw/transformed output, wide outputs and zero-tree/zero-row public prediction
  cases. `encode_quantize` itself still rejects empty dimensions. All existing
  training/count tests and defaults remain unchanged.

## Fair B-versus-D experiment, deferred until the GPU is idle

Freeze one build, model, feature bytes and numerical paths. First compare four
cells: B, B+D1-P, B+D2, B+D1-P+D2. Do not mix parent-leaf sharing, training split
changes or other inference policies into this comparison. If skewed metadata
shows padding problems, compare native batch/compact alternatives as named
additional D1 variants, retaining their extra memory/kernel/setup costs.

Use the saved actual Delicious validation features and model, one <=32-feature
control, one larger-row batch, and a bounded skewed-dictionary fixture. Include
one tight-budget multi-tile case. The primary timing is the complete synchronous
predict_gpu call: validation, host packing, pin allocation if selected, CUDA
allocation, encoding, all uploads, fused traversal/transform, final download
and cleanup. Model/input file loading and context warmup stay outside for every
variant. A separate encode-only measurement may explain the stage but cannot
replace the complete-call ranking. Resident setup-amortized timing is excluded.

Run GPU work serially after the game is idle, with three warmups, at least 15
paired samples, balanced alternating order and complete raw observations.
Record source/compiler/binary/model/input hashes, metadata size distribution,
tile width/count, pageable/pinned choice, host/device peaks, transferred bytes,
all failures and Windows-native GPU engine telemetry. Do not use Nsight or
compiler-time-trace timings to rank variants. A speedup claim for a case needs
the 95% paired-bootstrap interval upper bound for median D/B time ratio below
1.0 and every exactness gate passing. Defaults require no measured regression
across the agreed controls; inconclusive or failed results stay visible.

After uninstrumented measurements, isolated Nsight samples must confirm the
predicted copy/wait reductions and unchanged kernel sequence for D1-P/D2.
Attribute prechecks and warmups separately using the existing benchmark capture
scope. Investigate remaining driver blocking and byte amplification rather
than infer performance from API count alone. Timing confirmation remains
deferred; this document authorizes no benchmark during the active game.
