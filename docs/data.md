# GPU data fitting, encoding and dataset bytes

Preimplementation choice, 2026-09-23. This is fresh source with the archived
quantizer and GHBDS001 reader used as contract evidence, not linked dependencies.
The target is CUDA 13.4 / C++23 / SM86. Nothing here is measured fastest.

## Contracts

All entrypoints execute in one GPU coordinator thread, submit device-null-stream
children and append `finish(Status*)`. The caller zeroes Status first. Input,
output descriptors, backing arrays and workspace are global device allocations,
disjoint where written, and live until tail completion. Outputs are exclusive;
input arrays are immutable. Failed operations may have partial output, which is
not consumable. Shape/extent/capacity checks precede affected accesses. Runtime
submission status and completed semantic Status must both succeed. There is no
allocation, host preprocessing, download or hidden synchronization in these APIs.

`fit_schema` accepts a nonempty row-major FP32 Dataset, optional feature types
(empty means numeric), max_bins in [2,65536], output Schema with supplied Feature,
metadata and offset capacities, feature-major u16 bins, and a supplied workspace.
Targets/weights are irrelevant to feature fitting. Schema Array sizes remain
capacities; columns, metadata_count, per-feature count/begin, offsets, total_bins
and max_feature_bins are computed counts. Metadata is compact and feature ordered.
Types and max_bins are checked before submission. Metadata capacity is checked
against actual cardinalities after sorting, before writing that tile's metadata.

Every NaN is missing bin zero; infinity is an input error. Signed zeros form one
distinct value and fitted metadata contains positive zero. Numeric metadata uses
U distinct finite sorted values, I=min(U,max_bins-1) intervals and cuts at unique
indices floor(k*U/I)-1 for k=1..I-1. U=0 has zero cuts. Numeric histograms always
reserve cut_count+2 bins; categories retain every distinct finite value, require
U<max_bins and reserve U+1 bins. Encoding uses first metadata >= value; numeric
bin is index+1, categorical bin is index+1 only on exact equality, else zero.
`encode` consumes the same schema and encoder; it allows zero rows with valid
nonempty schema. It validates schema metadata, bounds and histogram offsets.
max_feature_bins is accepted as a valid upper bound; fitting emits its exact max.

GHBDS001 is exactly little-endian magic[8], six u32 values (version=1, rows,
columns, target outputs, objective, classes), then row-major feature FP32 bits
and target FP32 bits. Length is exactly 32+4*rows*(columns+outputs). Rows, columns
and outputs are nonzero; multiclass requires one target column and classes>=2.
All features reject infinities but permit NaNs; all targets must be finite;
binary targets are 0/1 and multiclass targets integral in [0,classes).
The format has no weights or feature-type section: export rejects weights rather
than silently discarding them. Decode receives explicit writable value/target
arrays and emits a DatasetRecord view. Byte reads/writes are explicit so input
byte alignment is unrestricted and float payload bits are preserved. Byte
transport and file ownership are outside these GPU computations.

## Algorithm comparison and selection

[Satish, Harris and Garland](https://research.nvidia.com/publication/2009-05_designing-efficient-sorting-algorithms-manycore-gpus)
establish radix sorting with scan and on-chip partitioning as a relevant GPU
approach; their hardware results do not rank today's SM86 choices.
[GPU Gems scan](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
describes hierarchical prefix construction and compaction. We implement our own
integer scans. NVIDIA library algorithms remain benchmark references only.

Choose stable LSD radix4/radix8 over ordered FP32 keys. Tile up to 32 features,
choosing the largest tile fitting the supplied arena, so scratch is O(rows*tile)
rather than O(rows*columns). Radix8 uses four passes and larger count/shared
tables; radix4 uses eight passes and smaller tables. Default remains radix8.
Each 1024-key CTA owns eight contiguous 128-key warp ranges. Scatter computes
stable ranks using warp matches and warp-local counters, then stages digit-sorted
keys in shared storage for coalesced global writes. This is a new lowering;
integer sorting semantics do not require retaining the archived thread mapping.

Count tables are digit-major across row blocks. Hierarchical 256-entry integer
scans produce global digit/block destinations with bounded local planning arrays.
Sorted unique flags are counted/scanned, compacted into the alternate key buffer,
and sampled by distinct rank. A small per-tile GPU descriptor pass packs metadata
and offsets. The shared 32x33 transpose encoder finally reads original rows and
writes feature-major bins. Compact uniques cost one extra key read/write versus
direct sampled extraction; they simplify and share category/numeric extraction.
This tradeoff is explicit and must be timed, not assumed beneficial.

Comparison sort costs O(N log N) comparisons; repeated selection risks repeated
full reads; unordered hashing requires ordering afterward and exact collision
handling. Neither removes required exact distinct ordering. A single-pass
look-back radix implementation can reduce launches/traffic but needs additional
forward-progress and publication proofs; it is not speculative production code.
No O(F^2) metadata rescans or row-wise CPU quantization are introduced.

CDP2 child ordering and tail completion follow the current
[CUDA dynamic-parallelism contract](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html).
Coordinator launch cost and stack/spills are real costs, included in measurement.
There are no parent reads of pending child results; later kernels consume them.

## Fair experiment and acceptance

Compare radix4/radix8 against frozen input/metadata/bin artifacts, then measure
complete fit+encode and already-fitted encode including status/tail completion.
Sweep rows around 32/256/1024 and scan hierarchy boundaries, features around
32, max_bins 2/32/256/65536, repeats/skew, missing-only, signed zeros,
categorical boundaries and small/large arena budgets. Report metadata/bin and
scratch bytes, tile/pass/launch counts, setup/teardown, and separately useful
throughput. Profile only to explain winners; serialize GPU runs.

GPU-only independent tests use explicit distinct-value enumeration/ordering on
small fixtures rather than the radix implementation, and analytic large fixtures.
Check exact metadata/bin bits, both radix policies, tails and multi-tile reuse,
capacity/alignment failures, malformed schemas and dataset bytes, and byte-exact
format roundtrips. Root owns compilation, sanitizer, runtime and benchmark gates.
This document records selection, not completion or measured superiority.

## Compile-only observation

`src/data.cu` and `tests/data_checks.cu` compiled separately with nvcc
`-std=c++23 -O3 -arch=sm_86 -rdc=true -Iinclude -Xptxas=-v --device-c`
(test compilation also adds `-Itests`). Raw logs and objects are in
`/tmp/gh-data-compile.puR5G7` for root to preserve. The first source compile failed
because an indexed namespace constexpr character array was host-only; the final
magic uses a constexpr u64 and explicit byte extraction. That failure remains
in `data.stderr`; successful logs are `data-fixed.stderr` and `checks.stderr`.
This was no device link or GPU execution. Dispatch has real stack/spill costs:
fit reports 320 stack bytes / 256 spill bytes each direction, encode 120/116,
decode 72/72 and dataset export 104/100. Numerical kernels need their own resource
observations; these dispatch figures are not a performance ranking.

The queued GPU suite covers 52 fitting cases across both policies, independent
distinct-value oracles, 65,535-category success/65,536-category rejection,
small-workspace multi-tile operation, fit/query encoder agreement, signed zero,
missing/unseen values, both infinities, guards and capacity failures. It also
checks 16 valid/malformed dataset codec cases and eight zero-row/schema cases.
All runtime and sanitizer results remain pending.
