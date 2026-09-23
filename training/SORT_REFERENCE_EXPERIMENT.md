# Isolated NVIDIA sorting reference experiment

This diagnostic follows [the algorithm decisions](ALGORITHM_DECISIONS.md). It is
benchmark-only: no trainer linkage, production dependency, or fallback. It tests
an already device-resident, feature-major array of order-preserving unsigned
32-bit keys and produces every feature's complete ascending key sequence.
Duplicates are retained. Host generation and `std::sort` are independent test
references, not preprocessing paths for the trainer.

Before implementation, inspected the installed NVIDIA CUB 3.4.2 headers under
`/usr/local/cuda-13.4/targets/x86_64-linux/include/cccl/`:

- `cub/version.cuh` declares `CUB_VERSION == 300402`.
- `cub/device/device_segmented_radix_sort.cuh` documents the input-preserving
  `SortKeys` overload, contiguous offset aliasing, bit ranges, scratch query, and
  the `INT_MAX` per-segment limit.
- `cub/device/device_radix_sort.cuh` documents the corresponding non-overlapping,
  input-preserving key-only sort and caller-owned scratch.
- `cub/device/dispatch/dispatch_radix_sort.cuh` contains Onesweep dispatch.
  The public API chooses its actual implementation; this experiment does not
  assume a particular dispatch or a published speedup on this GPU.

Compare three complete sort boundaries on identical immutable input:

1. One segmented 32-bit radix sort, one segment per feature. This can share
   submission overhead but its segmented kernel decomposition differs from a
   device-wide sort.
2. One ordinary 32-bit radix sort per column, submitted on the same stream with
   reusable scratch. This accesses contiguous columns and exposes repeated host
   submissions and launch gaps within the measured event interval.
3. GPU pack into `(uint64(feature) << 32) | key`, ordinary global radix sort, and
   GPU strip into the common 32-bit output. The pack and strip are timed. Sorting
   uses bits `[0,32+ceil(log2(features)))`; upper invariant tag bits need no passes.
   It processes wider keys and adds at least 24 bytes per key of pack/strip
   traffic (4+8 read/write, then 8+4), plus the 64-bit sorting traffic and scratch.

The first two always sort all 32 key bits, even for low cardinality. Cardinality
controls a rank domain spread over the full unsigned key range, not a restricted
low-bit optimization. It is an upper bound on observed distinct values; report
the actual per-feature cardinalities. Cardinality zero uses the full uint32
domain. All paths use the same explicitly seeded SplitMix64-generated input.

Allocate/upload once; exclude allocations, scratch queries, input upload, result
download, CPU validation, and event setup from GPU sort timings. Include required
pack/sort/strip kernels and the complete sequence of per-column launches. Use
two complete warmup sweeps, then five event samples per candidate in alternating
forward/reverse candidate order, rotating the initial order by seed. Five rounds
cannot balance all ordering effects; retain execution order and raw samples and
avoid statistical or fastest claims from this diagnostic alone. Validate every
warmup and timed output exactly against independent per-column `std::sort`, then
check that the device input remained unchanged. Validation follows timing and
can affect subsequent cache state equally across candidates.
Poison the shared output before every invocation, outside the event span, so a
missing write cannot inherit a preceding candidate's correct result. The bounded
benchmark requires total keys no greater than `INT_MAX`; ordinary radix sorts use
32-bit item offsets, including for the two requested shapes.

Report each candidate's required scratch and isolated device allocation payload,
plus the actual combined experiment payload with shared scratch. Payload excludes
CUDA/CUB runtime bookkeeping. Report GPU, driver/runtime, CUB version, all shape
and generator parameters, warmup/sample counts, timed boundaries, and raw samples.
Root will run at least 65,536 rows ×32 features and 1,048,576 rows ×8 features
serially. These results are a sorting-only lower boundary: they omit transpose,
float validation/mapping, deduplication, cut selection, bin encoding, transfers,
and training. They cannot establish a full-quantizer or end-to-end speedup.
