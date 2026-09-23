# Resident complete-operation benchmark protocol

Contract and comparison selected before implementation, 2026-09-23. This is a
bounded executable experiment, separate from correctness CTest. Root alone runs
GPU workloads, serially. No timing or superiority is claimed by this document.

The registry has eight cases; each has three warmups per variant followed by
fifteen alternating A/B pairs (36 raw spans). A/A cases exercise the same backend
under both labels. They diagnose drift and scheduling noise; an interval that
excludes one is evidence to investigate, not permission to subtract a correction.

| Case | Resident input | A | B |
| --- | --- | --- | --- |
| 0 | None | Empty completion | Same |
| 1 | 16 Mi u32 IDs, 16,384 bins, u64 output | Shared atomic p15, 48 blocks, u32 local | Same |
| 2 | Same as case 1 | Same as case 1 | Shared atomic p14, 192 blocks, u32 local |
| 3 | 64 Mi IDs, otherwise case 2 | Same as case 1 | Same as case 2 |
| 4 | 8,192 rows, 8 numeric features, 32 regression outputs | Per-output trees and roots | Same |
| 5 | Same as case 4 | Same as case 4 | Output-batched trees and roots, tile 32 |
| 6 | 16 Mi u32 IDs, 16,384 bins, u64 output | Frozen p15/48 atomic kernel | Fresh p15/48 atomic kernel |
| 7 | 64 Mi IDs, otherwise case 6 | Same as case 6 | Same as case 6 |

Mi means 2^20 elements. Both count configurations are explicit existing catalog
policies. p15/48 is the archived A5000 Laptop graph/warm default at case 2's shape;
this experiment executes fresh kernels through a resident CDP stream. It is
neither an old direct-leaf graph measurement nor a new automatic-default result.
At 64 Mi this is an explicit extrapolated candidate, not an archived measured
key. p14 has half as many threads per block and four times as many blocks: this
comparison changes a complete policy/configuration, not just one launch variable.
Both use vector4 loads, one replica and the same shared counting algorithm.
Shared initialization, aggregation, output clearing and completion all count.
Each configuration needs no application scratch. The runtime still sets required
shared-memory function attributes before GPU bootstrap.

Count fixtures are `(13*i+7) mod 16384`. The multiplier is coprime to the bin
count, so every bin receives exactly N/16384 IDs. This ordering is deliberately
specified; it is not random or representative of skew. Inputs are resident and
reused, without a cache flush or an assertion that the full input fits in cache.
Every sample's histogram and source are checked outside its span, including
guards. The checker reads can influence subsequent cache state equally for both
variants; this protocol must not be relabelled a cold-cache experiment.

Pipeline inputs are row-major FP32 values -1/+1 from each of eight row bits and
row-major targets `sign(row bit output%8) * (1+output%3)`. All rows have implicit
unit weight. Each feature's sorted metadata is the single cut -1; its bins are
1/2 in feature-major order. Training uses three rounds, depth at most three,
learning rate 1, l2=0, minimum one row, FP64 order-two statistics, global
histograms and the existing warp32 split schedule. The first round exactly fits
each target with a stump; later rounds add zero leaves. Balanced bases are zero,
all predictions/margins equal the integer targets, initial loss is 145/64 and
later losses are zero. Integer-valued small sums make these assertions exact
without waiving any floating-point gate. This easy fixture measures scheduling
and complete processing, not general learning quality or a deep-tree workload.

Pipeline timing includes GPU schema fitting and bin encoding, all training
planning/initialization/rounds/loss/export-to-resident-model, explicit model
validation, prediction with finite checking, and GHBMODEL byte export. It excludes
fixture creation, reusable-buffer poisoning, independent checks and summaries.
The export's header, feature records, every tree/node field, extent and untouched
suffix are checked independently against the resident result after every sample.
The resulting forest has 96 trees and 160 nodes, hence 5,664 wire bytes. Schema,
bins, analytical predictions, losses and all storage guards are also checked.
The shared 16 MiB arena is reused only after completed stages; all descriptors,
raw samples and backing arrays are device-resident. Host code only performs CUDA
attribute setup, bootstrap launch and synchronization. No CPU fixture, oracle,
selection, summary, clock conversion or training stage is present.

## Timing choice and evidence

[CUDA CDP2 ordering](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html)
requires a tail consumer for child results. Each sample queues its reset child,
then a start-marker child, then an operation-coordinator child on the same
device-null stream. Placing the entire operation behind the start marker includes
its planning and submission work. The end marker is a tail child after the
operation's nested completions. A later tail checker launches the next sample;
there is no parent loop reading pending children or overlapping samples.

The observation source is qualified raw `%globaltimer`, protocol
`globaltimer-cdp-tail-v1`; it is not CUDA-event or host-wall-clock milliseconds.
[PTX documents this special register](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer)
as target-specific tooling infrastructure. The separate observe suite checks
cross-SM bracketing; this executable additionally preserves 36 empty spans and
both same-backend controls. No empty-span subtraction is applied. All raw begin/
end ticks, ordinal, pair, warmup flag and variant are printed by GPU code.
`observe::summarize` computes the geometric mean A-duration/B-duration ratio and
paired log-ratio t interval on GPU; raw spans remain resident and are printed
before summary. A ratio above one means B was shorter in this exact experiment.

Considered alternatives are host-event submission (different launch boundary),
device launchable graphs (current CDP nodes are ineligible), and persistent
execution (changes the production executor). None is silently substituted.
Existing [count evidence](count.md) and [learning decisions](learning.md) select
the algorithms; this harness compares their actual complete operations without
implementing new count/training alternatives. All-zero validation failures abort;
raw observations already emitted and process failure must be retained.

Require the correctness suite and timer qualification before ranking. Preserve
source/binary/runtime identities, raw stdout/stderr, exit status and every failed
run. Keep instrumentation disabled for ranking; profile a separate run to explain
traffic, resources, atomics and launch cost. Stage instrumentation, Nsight or
sanitizer timings cannot rank uninstrumented candidates. No automatic policy or
default changes follow from this bounded experiment. Skew/tails, larger feature/
output counts, deep trees, real-data quality, full archived-pipeline adapters and alternate
executors remain separately declared experiments.

Compile-only verification: `tests/bench_checks.cu` and `tests/bench_driver.cu`
both pass nvcc C++23/O3/sm86/RDC with assertions enabled. Logs, objects and 526
dependency/artifact hashes are retained under `observations/benchmark`. This is
not device linking, runtime correctness, timer qualification or measured ranking.
The dedicated driver initializes count attributes and launches the GPU bootstrap;
the executable belongs outside correctness CTest and must preserve its exit code.

## Frozen kernel comparison, selected before extraction

Cases 6/7 protect the actual archived p15 arithmetic kernel with an isolated
benchmark reference. Fresh-policy comparisons alone cannot establish preservation
of its performance. Extract the definitions of `add`, `clear_histogram_output`,
`warp_add`, `accumulate_shared_tile` and `shared_histogram_loaded` verbatim from
the frozen `src/histogram.cu`; preserve their full-definition and brace-body
SHA-256 hashes, original line ranges and whole-source hash in
`observations/benchmark/frozen-source-identities.json`. A checked-in copy of the
definitions lives in `bench/frozen_count.cu`, under an isolated namespace. No
legacy project header/library or frozen host dispatch is imported. The production
`gh` target has no dependency on this reference; only the manual benchmark links
it. Shared Array/Status/completion infrastructure defines the new boundary.

Instantiate only unsigned input, unsigned-long-long output, unsigned local,
Update::atomic, Partial=false, vector4, 512 threads, 8 items, one replica,
96 KiB policy capacity and 48 blocks. The requested helpers retain all original
template bodies for byte-level provenance, but discarded alternatives produce
no extra executable policy. Actual histogram shared storage is 64 KiB. The
runtime initializes this exact function's opt-in attribute before bootstrap.
Both inputs are exact multiples of the 4,096-item tile and begin 16-byte aligned.
The fixed wrapper checks the allowed sizes, pointer extents/alignment and zero
scratch capacity before effects; the benchmark's common independent GPU checker
validates IDs and exact output/guards. Narrow locals are bounded by the full
64 Mi input size, below UINT32_MAX, even before the tighter CTA ownership proof.

Both new variants explicitly use kernel output clearing: 64 blocks of 256
threads, then the 48-block histogram, then checked shared tail completion.
The first six cases retain their original settings. Fresh generic device
validation/dispatch and the fixed frozen-reference wrapper are included inside
the same operation-coordinator span. Their planning code is not identical, so
this is a complete-operation comparison using archived arithmetic kernels, not
a pure leaf-kernel timing or an archived host/graph performance claim. Both read
the identical resident data and reuse the same guarded buffers and no scratch.
All 36 samples per new case receive exact checks, preserving 288 raw spans total.
No defaults change; performance and generated-resource conclusions wait for
serial runtime evidence and separate profiling.

Frozen whole-source SHA-256:
`1faf482d0f3d50209b9e7d52a3371196e1784e4ba263d0d91da03c42d9538eb6`.
The five brace-body hashes below are independently recorded along with the
full-definition hashes in the receipt; every complete definition is verified
byte-for-byte, including whitespace, against its archived line range.

| Definition | Archived lines | Brace-body SHA-256 |
| --- | --- | --- |
| add | 26–29 | `474af80c6d702371ce9096e06d23009b178eeec6ebe0dae5b94c1fb8615f2e76` |
| clear_histogram_output | 31–37 | `9dd53d77f8f5ad74a260802134f69ed721a53a4a37fb88ffa3f1c95179069703` |
| warp_add | 53–61 | `3ff26bbdc61061f15f1a0fe38944b42e306fb37634d9ef9fab90bb7723d9d9d0` |
| accumulate_shared_tile | 160–221 | `760982d974b20c4dae4f8d8fb455df4bcf273126f41ce0186821cf7a068add29` |
| shared_histogram_loaded | 223–278 | `d6e9b061cbe3ef4f4ec15b60a55487a7f990cc704c57288cb120cdc743cb4453` |

The earlier six-case compile receipt predates this frozen-reference extension.
The final three benchmark/reference translation units compile with nvcc
C++23/O3/sm86/RDC and assertions enabled; `cmake --build build --target benchmark
-j1` also passes compilation, device linking and host linking. The resulting
`build/benchmark`, production library, build settings and source/dependency
identities are included in the 535-file
`observations/benchmark/reference-compile-identities.json` receipt. Raw logs and
objects remain beside it. No GPU execution was performed by the implementing
agent; runtime acceptance and all 288 timed observations remain pending.

The frozen object emits one histogram specialization plus its clear kernel.
The histogram reports 39 registers, zero stack/spills and one barrier resource;
the fixed wrapper reports 64 stack bytes and 60 spill bytes in each direction.
These are compiler resources, not speed evidence, and wrapper costs remain
inside the declared complete-operation span.
