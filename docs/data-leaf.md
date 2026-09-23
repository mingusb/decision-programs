# Direct production data-kernel sanitizer route

Contract and algorithm selection recorded before implementation, 2026-09-23.
This bounded harness launches the current production `radix_counts<4/8>`,
`scan_tiles`, `scan_offsets`, `radix_move<4/8>` and `unique_keys<false/true>`
bodies as top-level kernels. In the three-file layout, mode 13 of
`gpu_histogram.cu` imports the actual `gh::data_impl` kernel declarations from
`gpu_histogram.hpp` and links the same `gh_data_leaf` object included in the
production `gh` archive. The leaf object supplies the radix4/radix8 and unique-kernel
specializations; the harness does not include or recompile their definitions.
There is no copied sort, scan or compaction implementation used as the tested
operation.

The fixed shape is 1,025 rows and two feature-major u32 key columns. Two
1,024-key radix blocks exercise one full block and one one-key tail. Radix4 has
32 digit/block cells per feature, requiring one scan block. Radix8 has 512 cells
per feature, requiring two scan blocks, a two-total upper scan and offset add.
All eight radix4 and four radix8 digit passes execute. Each final column has five
256-row unique blocks and a six-cell count/prefix array. UINT32_MAX is the
production missing-key sentinel and is omitted from compact distinct output.
The primitive tests use opaque sortable keys, not a claim of end-to-end float
quantization coverage.

Choose two analytic finite key domains of sizes 251 and 241. GPU fixtures
permute/repeat these keys across the full and tail blocks and insert missing
keys. High key bits distinguish values whose current low digit ties, making
stable ordering observable before the final pass. All domain values remain
present despite missing rows. An independent GPU rank oracle enumerates original
rows and orders by processed low bits followed by original row index. This is
O(rows²) per pass only in this bounded test; production sorting is unchanged.
Separate GPU checks enumerate source digits to verify raw counts and exact
exclusive prefixes. Analytic distinct values and independently counted block
boundaries verify unique counts, scanned prefixes, compaction and untouched
suffixes. Source and destination guards are checked throughout.

The host performs only `cudaGetSymbolAddress`, a compile-time fixed sequence of
constant launches, runtime error checks and final CUDA synchronization. It
does not inspect keys, select cases from input, compute prefixes, evaluate an
oracle, or copy a GPU decision to the host. All fixtures, expected results and
assertions run on GPU. Default-stream top-level ordering supplies dependencies;
there is no CDP call or tail continuation in the executed path. Kernel-local
collectives, layouts, rank updates and shared-memory accesses are exactly the
production bodies. Global backing arrays remain alive for the entire process.

This route is selected because the current sanitizer reports initcheck,
racecheck and synccheck limitations for CDP2. Merely wrapping a copied algorithm
would not test the production body; launching the existing leaf bodies directly
provides a distinct applicable scope. The relevant ordering and participation
contracts are the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
and [Compute Sanitizer's initcheck/racecheck/synccheck documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
(read 2026-09-23). Root must
record actual tool activity and any unsupported result; applicability is not
presumed from source or successful compilation.

The experiment is correctness/sanitizer evidence, not a speed comparison.
Run the executable alone, then the requested sanitizers serially, preserving raw
logs, source/binary identities and failures. A pass certifies only the exercised
production leaf bodies and shapes. It does not certify the CDP coordinator,
feature tiling, metadata selection, full feature fitting, or arbitrary shapes.

## Current build and evidence boundary

`data_leaf_checks` is a separate CMake/CTest target with `RUN_SERIAL` and
`-UNDEBUG` for GPU assertions. The `GH_DATA_LEAF_IMPLEMENTATION` section compiles
once into `gh_data_leaf`: the eight kernels, `block_prefix` and `first_key` have
unchanged bodies. Its fixture section compiles separately from the same CUDA
source. The harness links that object and CUDA runtime, without `gh` or the
coordinator object. Build and run commands are:

```sh
cmake --build build-three --target data_leaf_checks --parallel 1
ctest --test-dir build-three --output-on-failure -R '^data_leaf_checks$' -j 1
```

Only root runs GPU workloads, serially. The initial consolidated harness linked
the full production object. Memcheck passed, but initcheck, racecheck and synccheck
each exited 99 with "CUDA Dynamic Parallelism is not supported by the selected
tool". Raw logs and exit statuses remain in
`observations/consolidation/sanitizers/{data-leaf-*-1.log,exits-1.txt}`. Successful
GPU oracle output did not make those unsupported tool runs pass.

The leaf-object split resolves that observed applicability failure. Root ran
memcheck, initcheck (all address spaces), racecheck and synccheck serially through
the C++ collector; all four passed with both radix GPU receipts and clean error
summaries. Raw output, commands, binary/tool hashes and activity gates are in
`observations/consolidation/leaf-{memcheck,initcheck,racecheck,synccheck}-2/`.
The new ELF symbol dump and coordinator-symbol search are preserved alongside
them. These checks cover the exercised leaf shapes and do not certify the CDP
coordinator. Historical binary correspondence below describes earlier binaries.

## Historical compile and binary audit

Before consolidation, `data_leaf_checks` included `src/data.cu` and was built
with `src/core.cu` without linking `gh`. That Release, C++23, SM86, RDC build passed
on 2026-09-23. `-UNDEBUG` keeps GPU assertions enabled. The first attempted build
preceded CMake regeneration and reported an unknown target; both failure logs
and the subsequent successful configure/build are preserved.

Comparing linked `build/data_checks` with `build/data_leaf_checks` found exact
equality of all eight exercised kernels' 49,920 bytes of instruction/control
words, including padding. Relocation offsets, types, addends and target symbols
also match after replacing each kernel's translation-unit-dependent own symbol
with a common name. The five referenced CUDA arithmetic/collective helpers have
identical instruction words too. Raw SASS, ELF, resource/symbol dumps, comparison
script, per-symbol hashes and source/object/executable hashes are retained in
`observations/data-leaf/compile/`. This establishes the recorded binary correspondence, not
that the same compiled object was linked into both programs.

The supplied CDP2 tooling report recommends one shared leaf object and complete
exclusion of the coordinator object from the harness. That first harness did
not meet that stronger construction: it compiled an inclusion of `data.cu`.
The linked SM86 SASS and ELF contain no `fit_schema`, `finish`, CUDA device-launch
or parameter-buffer functions; the device linker removed their uncalled paths.
Other data kernels, `complete`, and device-runtime copy/fill kernels remain.
The executable has no embedded PTX program. These observations do not establish
which modules a sanitizer accepts or inspects. Root must record actual kernel
activity and unsupported diagnostics before claiming applicable coverage.

At the time of those receipts, the proposed follow-up was a shared leaf object
with coordinator code in a separate, excluded object. The current
`gh_data_leaf` construction now supplies that source/link separation; validation
of the new binary remains a separate step.

Root subsequently ran ordinary execution, memcheck and initcheck with
`--initcheck-address-space all`; all three passed. Logs and exit statuses are
`observations/data-leaf/{run,memcheck,initcheck}-1.*`. These results cover the
top-level path and fixed shapes above; they do not remove the CDP control-plane
gap. Racecheck was still running when that historical receipt was written, and
synccheck was pending. These statements preserve the receipt's original scope
and status; they are not new results for the three-file build.
