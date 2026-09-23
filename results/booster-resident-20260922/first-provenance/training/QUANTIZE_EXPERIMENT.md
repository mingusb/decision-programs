# Exact GPU quantization experiment

Decision recorded before implementation, 22 September 2026. Read alongside
[the algorithm audit](ALGORITHM_DECISIONS.md). This selects experimental code,
not a performance winner. Target CUDA C++23 / SM86; no vendor algorithm is linked.

## Contract and selected candidates

Host row-major float32 input becomes resident feature-major uint16 bins.
NaNs map to missing bin zero, infinities fail, and both signed zeros become +0.
Numeric cuts have the existing uniformly spaced **distinct-value** ranks;
categories retain their complete sorted dictionary or reject max_bins overflow.
Only feature metadata/status return to the host; dense bins remain on the GPU.

Implement two owned stable LSD radix parameterizations: 8-bit digits/four
passes and 4-bit digits/eight passes, with 1,024 keys per block. The initial
radix8 default is provisional and must be frozen/confirmed after measurements.
Each pass counts block histograms, scans the digit-major counts, and scatters
through a locally grouped shared-memory tile. Warp match masks group equal
digits, preserving stable rank without a quadratic per-key comparison sort.
Prefix scans are hierarchical parallel scans, not a serial feature scan.
Sorted-run detection and cut/category selection execute on GPU. Block-level
unique prefixes identify exact target ranks; each block compacts only into
shared memory and emits selected cuts directly. No full global distinct vector
is materialized, avoiding a write/read of U values for high-cardinality columns.

This implementation is a conventional count/scan/scatter radix baseline,
**not Onesweep**. Main key-array traffic is approximately 3*N reads/writes per
pass, or 48*N bytes at four passes and 96*N at eight passes, before histogram,
scan, layout, dedup and encoding traffic. The
[Onesweep design](https://arxiv.org/abs/2206.01784) instead estimates 36*N bytes
for four 8-bit passes including its upfront histogram. Implementing and
validating interblock lookback progress is deferred; its theoretical traffic
advantage remains an open contender, not a claimed property of this code.
Likewise exact hash-distinct then sort-U remains a separate contender for low
cardinality. Two radix widths test a real pass-count versus local ranking/
histogram-size tradeoff, but do not exhaust the candidate space.
The workload is a batch of independently segmented columns, often much smaller
than the paper's single 256M-key sort. One launch processes every tile feature;
explicit scans also avoid per-segment lookback initialization/state. Setup cost,
segment size and batching might offset some extra key traffic on small columns;
that is a measurement hypothesis, not evidence that this baseline beats
Onesweep. The installed CUB 3.4.2 sort should be benchmarked in an isolated
reference target to quantify the remaining sorting gap before selection is
considered complete.

## Layout and memory

Bounded feature tiles retain each feature's whole column: row chunks are never
independently quantized. A 32x32 shared-memory transpose with bank padding maps
coalesced row-major input to column-major sortable keys; encoding uses the same
layout pattern and coalesced uint16 output. Input tiles upload with pitched
copies, retaining original values for final encoding. Fit scratch consists of
the float tile, two key buffers, digit histograms, hierarchical scan levels,
unique block counts, metadata slots, sizes and a status word. Inference needs only input tile,
metadata, sizes and status. Tile width is bounded by both a fixed maximum and
the exact allocation-payload budget. CUDA internal bookkeeping and host
Dataset/model vectors are outside that device budget. `peak_bytes` includes
resident output plus every simultaneous preparation allocation.

## Required experiment before promotion

Compare both exact policies, frozen CPU preparation, and installed CUB sorting
only as an isolated reference. Include high-cardinality floats, U=16/256/65535,
rare tails among heavy duplicates, sorted inputs, NaNs, constants/all-missing,
signed zero, category overflow, irregular sizes and enforced tight budgets.
Compare every cut, dictionary entry and bin with an independent CPU reference.
Measure Dataset-to-ready-device-bins plus host model metadata, as well as
separate upload/layout/radix/dedup/encode timings and allocated peak bytes.
Do not omit float upload, metadata synchronization, scratch allocation or the
current CPU baseline's smaller packed-bin upload from end-to-end comparisons.
Run candidates serially, pair orders, freeze the selected policy and confirm on
fresh seeds. Nsight diagnostics explain traffic and occupancy separately from
uninstrumented timing. No fastest-ever claim follows from this implementation.
