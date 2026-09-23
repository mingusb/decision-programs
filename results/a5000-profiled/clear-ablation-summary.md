# Explicit output-clear kernel: paired ablation

We compare against NVIDIA’s CUB library of optimized GPU routines; its histogram is our reference implementation.

Replacing the output `cudaMemsetAsync` with an explicit CUDA zero kernel improved the complete shared operation in all nine paired measurements. The three-repeat median paired gains are **2.355×** for small8, **5.400×** for smallbyte, and **3.041×** for cachedbyte. Bitplane improves **1.799×** raw and **1.785×** after CUB normalization. Unchanged controls remain near parity.

These are controlled ablations at one data seed, not independent-seed validation or a deployable selection result. Every process uses seed 424242, uniform shuffled input, u32 counters, warm graph mode, protocol 3, 21 sample rounds and batch 32. Each CSV reports the median complete-operation time across its 21 rounds.

Each ratio below is calculated within one old/new process pair, then summarized by the median of three paired ratios. CUB-normalized improvement is `(old_custom/new_custom)/(old_CUB/new_CUB)`. Speedup versus CUB is `CUB/custom` within the same process. Values above 1 favor the new implementation or custom variant, respectively.

| Case / variant | N / bins / input | Runtime-clear median µs | Kernel-clear median µs | Paired old/new | CUB-normalized gain | Old speedup vs CUB | New speedup vs CUB |
|---|---|---:|---:|---:|---:|---:|---:|
| small8 / cub t2 g192 | 1048576 / 8 / u32 | 14.752 | 14.752 | 1.000× | 1.000× | 1.000× | 1.000× |
| small8 / shared t4 g192 | 1048576 / 8 / u32 | 15.072 | 6.368 | 2.355× | 2.355× | 0.979× | 2.305× |
| small8 / shared_partial t2 g96 | 1048576 / 8 / u32 | 8.000 | 8.000 | 1.004× | 1.000× | 1.840× | 1.844× |
| small8 / bitplane t1 g192 | 1048576 / 8 / u32 | 21.504 | 11.968 | 1.799× | 1.785× | 0.685× | 1.233× |
| smallbyte / cub t2 g192 | 4096 / 256 / u8 | 4.192 | 4.192 | 1.000× | 1.000× | 1.000× | 1.000× |
| smallbyte / shared t7 g96 | 4096 / 256 / u8 | 14.048 | 2.880 | 5.400× | 5.400× | 0.271× | 1.456× |
| smallbyte / shared_partial t7 g96 | 4096 / 256 / u8 | 5.600 | 5.600 | 1.000× | 1.000× | 0.749× | 0.749× |
| smallbyte / nvidia_sample256 t2 g192 | 4096 / 256 / u8 | 6.112 | 6.112 | 1.000× | 1.000× | 0.686× | 0.686× |
| cachedbyte / cub t2 g192 | 1048576 / 256 / u8 | 6.560 | 6.560 | 1.000× | 1.000× | 1.000× | 1.000× |
| cachedbyte / shared t7 g96 | 1048576 / 256 / u8 | 16.448 | 5.408 | 3.041× | 3.041× | 0.399× | 1.213× |
| cachedbyte / shared_partial t7 g96 | 1048576 / 256 / u8 | 7.424 | 7.424 | 1.000× | 1.000× | 0.884× | 0.884× |
| cachedbyte / nvidia_sample256 t2 g192 | 1048576 / 256 / u8 | 8.064 | 8.064 | 1.000× | 1.000× | 0.813× | 0.813× |

Time columns are medians of the three process medians; ratio columns are medians of paired ratios and need not equal the ratio of the displayed time columns.

| Changed variant | Paired raw gains, repeats 1/2/3 | Paired CUB-normalized gains, repeats 1/2/3 |
|---|---|---|
| small8 / shared | 2.355×, 2.241×, 2.656× | 2.355×, 2.241×, 2.384× |
| small8 / bitplane | 1.783×, 1.799×, 1.988× | 1.783×, 1.799×, 1.785× |
| smallbyte / shared | 5.467×, 4.573×, 5.400× | 5.467×, 4.573×, 5.400× |
| cachedbyte / shared | 2.905×, 3.041×, 3.417× | 2.905×, 3.041×, 3.417× |

Small8 repeat 3 shows a common timing shift: CUB changes from 14.720 to 13.216 µs and the unchanged shared-partial control from 8.000 to 7.232 µs. CUB normalization helps describe the paired result against that control; neither it nor the before/after telemetry establishes the cause. The ablation does not identify an execution engine or a WSL/driver defect.

All 18 command files have successful exit codes and hashes matching their named executables. The nine runtime-clear metadata corrections now match the archived binary. Paired commands differ only in executable path; workload/policy/protocol fields match exactly. All 72 sample vectors have 21 values and reproduce their CSV medians.

- Runtime-clear binary: `build/profiled-loads/histogram_bench`, SHA256 `5d651b317053a23ed8fb4932066d4020d008655c7634541153c6c262e2702ad4`.
- Kernel-clear binary: `build/histogram_bench`, SHA256 `3485149e183ada05893e59d505bcebc1e86d2a484a4184ec53821216072b43c1`.

The four shared counting-kernel specializations actually used here have byte-identical instruction encodings, identical disassembly text, and identical register/stack/spill counts in the two builds. The complete focused SASS dumps also match byte for byte.

| Counting kernel | Registers, both builds | Encoded SASS bytes, both builds | Stack / spill-store / spill-load bytes |
|---|---:|---:|---|
| u32 / atomic merge / scalar / t128 i8 r4 | 37 | 3072 | 0 / 0 / 0 |
| u32 / partial / scalar / t256 i8 r1 | 32 | 3840 | 0 / 0 / 0 |
| u8 / partial / vector4 / t256 i8 r1 | 32 | 6144 | 0 / 0 / 0 |
| u8 / atomic merge / vector4 / t256 i8 r1 | 32 | 4992 | 0 / 0 / 0 |

`build-clear.log` contains 450 PTXAS function records; 0 report nonzero spill stores or loads. It is an incremental build log, so this count covers the functions recompiled for the clear change. The new typed initializer uses 8 registers for either counter width.

Artifacts: [full paired measurements and validation](clear-ablation-summary.json), [raw CSVs and commands](clear-ablation/), [runtime-clear counting SASS](clear-counting-runtime.sass.txt), [kernel-clear counting SASS](clear-counting-kernel.sass.txt), [exact symbols and prior resources](clear-counting-symbols.json), and [clear build log](build-clear.log). cuobjdump warnings only concern the requested symbols being absent from other embedded translation units; all four intended functions were found.
