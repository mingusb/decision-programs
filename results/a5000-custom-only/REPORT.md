# Production histograms use our kernels only

The user's requirement is explicit: NVIDIA histogram implementations are permitted for benchmarking, never as production implementations or fallbacks. The rebuilt library passes correctness and sanitizer checks. Matched timing and a bounded follow-up are complete, but they do not establish the strict requirement of zero speed loss: a small residual difference remains unresolved.

## Changes

- The public production algorithm enum contains only our kernels. CUB histogram code and NVIDIA's published sample live under `bench/`, in the separate `gh_benchmark_references` target. `GH_BUILD_BENCHMARKS=OFF` excludes those targets entirely.
- Measured custom choices remain in the selector. Other valid shapes use our shared-memory kernels when capacity and overflow bounds allow, otherwise our global-atomic kernel. No input-distribution oracle or per-call device query is added.
- Schema 5 tuning plans always choose one of our implementations. NVIDIA reference results report comparative performance even when faster; they cannot become a selected production plan. Replay rejects older schemas or a reference in the chosen field.
- The host-only policy target allows selector tests without loading CUDA or accessing a GPU.

The counting kernels, their launch templates, the bit-plane source, and the compile-time tuning catalog are unchanged from the preserved fast implementation. See [source preservation evidence](kernel-source-preservation.json). CPU inspection also found matching resource records for all 751 custom device functions and identical instruction encodings for 13 representative kernels, including clearing, shared, partial, global and bit-plane paths. Both binaries already call host validation out of line. [Machine-code evidence](gpu-validation/machine-code/README.md) preserves the exact selection, hashes and disassembly. These checks do not prove unchanged runtime performance.

## Validation status

The following checks completed on the rebuilt implementation:

- 17,036 histogram executions match independent CPU counts, including tails, overwrite behavior, output/workspace canaries and nondefault streams. All 3,170 host-only selector checks and 21 Python autotuner tests pass. [CTest details](gpu-validation/ctest-details.log)
- Compute Sanitizer memcheck, racecheck and synccheck each completed 2,932 histogram executions with zero errors or race hazards. [Memcheck](gpu-validation/memcheck.log), [racecheck](gpu-validation/racecheck.log), [synccheck](gpu-validation/synccheck.log)
- Eight large-count executions reproduce exactly **4,294,967,328**, covering scalar RLE with native/u32 locals, packed RLE with u32 locals, and partial histograms with u32 locals and u64 reduction. All are our kernels. [Large-count log](gpu-validation/large-count.log)
- Ten automatic-selection cases execute and pass CPU verification: measured stream, small-byte, large-byte and opt-in cases; neighboring shapes; a byte tail; global fallback; generic opt-in; cold input; and empty input. [Selected configurations](gpu-validation/automatic-summary.json), [commands and CSVs](gpu-validation/automatic/)
- A schema 5 tuning smoke test chooses a custom shared kernel and successfully replays with metadata verification. This checks integration, not broad performance. [Saved plan](gpu-validation/tuner-smoke/plan.json), [replay log](gpu-validation/tuner-smoke/replay.log)

Static symbol inspection finds no NVIDIA histogram or benchmark-reference symbols in the production libraries. [Linkage audit](production-symbol-audit.json)

## Performance preservation: measured results and unresolved limits

The initial matched old/new experiment completed 13 cases with four adjacent invocation pairs per case, using ABBA/BAAB ordering and explicit matching policies. It does **not** establish zero regressions. For N16M/u32 input/B8192/u64 output, the current binary was slower in all four pairs: median current/old ratio **1.025344** (2.53% slower), ranging from 1.003735 to 1.062809. Other cases also contain individual slow pairs and substantial process variation. [Paired measurements](paired-comparison.md)

The bounded follow-up repeated both ordering rounds for cached-byte, large4096-u64, large8192-u64 and stream4096, adding 32 invocations and 16 pairs. The 8192-bin median ratio was **0.999587** in this repeat, with two slower and two faster current invocations. The initial 2.53% median slowdown did not persist. Keeping both cohorts gives **1.004890** (+0.49%) across eight 8192-bin pairs, six of which were slower. That small remaining difference is unresolved; it is not accepted as a permitted regression. The combined medians for the other repeated cases are 1.000000, 0.998622 and 0.978602, respectively. [Follow-up measurements](followup/paired-comparison.md)

Some invocations of either binary enter a substantially slower timing band. For example, follow-up cached-byte comparisons include +30.37% and -24.58% current/old differences. Clock snapshots and matching instruction encodings do not identify the cause. All original samples, slow pairs and invocation metadata remain preserved; no favorable subset replaces them. A future investigation should control this invocation-level variation before attributing sub-percent differences to source changes or changing a winning kernel. Performance acceptance remains open, and the generic selector has no universal optimality claim.

The comparison uses the exact same explicit custom configurations in both binaries. In particular, the sorted and single-bin cases retain their frozen policy 6 rather than the shape-only automatic policy 11. Thus this experiment does not validate every automatic-selection performance choice. Measurements include complete initialization and counting, use protocol 3 with 21 samples, 32 operations per sample and 200 ms warmup, and run on driver 597.06 with unlocked clocks. No new Nsight capture was needed for these unprofiled preservation measurements; the preceding Nsight findings remain historical evidence.

The rebuilt executables, production archives and schema 5 autotuner are preserved in `build/custom-only`, with SHA256 hashes in its `manifest.json`. The benchmark hash is `d7568b5c93b80fbe39002c82866d08f2c75683b92ae99869d920c9896c481de7`; the old comparison hash is `a91544fe771a19175b05611c51b1b08549314dcee418d624eeba58ab5444ce6d`. Removing reference enum values changes enum ordinals; consumers must rebuild against the current public header.

The [combined investigation](preservation-investigation.md) retains all 136 invocations, 68 pairs and 2,856 raw timing samples, including five pairs at least 5% slower. The follow-up cases were selected after reviewing the first cohort, so the combined summaries are descriptive rather than an independent confirmation test. The exact source snapshot is [archived here](source.tar.gz), with [per-file hashes](source-manifest.json).

## Current comparisons against NVIDIA's benchmarks

A separate stage measured the current custom variant and the applicable NVIDIA references together, on seeds 424242 and 987654 with the same timing protocol. Our custom variant was faster in all 26 case/seed comparisons. Each range below is the faster applicable NVIDIA reference median divided by our median; values above one favor our implementation. These comparisons measure advantage against NVIDIA, not preservation relative to our older binary. [Raw CSVs and invocation records](references/)

| Workload | Speedup range across two seeds |
|---|---:|
| 1M u32, 8 bins | 2.470–2.471× |
| 4K byte inputs, 256 bins | 1.468–1.487× |
| 1M byte inputs, 256 bins | 1.228–1.370× |
| 16M byte inputs, 256 bins | 1.046–1.049× |
| 1M u32, 256 bins, 99% hot | 2.337–2.357× |
| Same shape, sorted hot input | 2.288× |
| Same shape, single-bin input | 2.282–2.296× |
| 16M u32, 4096 bins, u32 counts | 4.148–4.216× |
| 16M u32, 4096 bins, u64 counts | 10.346–10.366× |
| 16M u32, 8192 bins, u64 counts | 20.007–20.085× |
| 16M u32, 16384 bins, u64 counts | 31.555–35.665× |
| 1M u32, 4096 bins, u64 counts, cold input | 5.414–5.416× |
| 1M u32, 4096 bins, u32 counts, stream | 3.332–3.477× |

Here M and K denote powers of two, as specified exactly in the manifest. All rows except the last use graph execution; all except the cold row use warm-cache mode. The unusually broad 16384-bin range and other variation must remain visible; two seeds do not establish a universal or tail-latency speedup. NVIDIA histograms remain confined to the benchmark target.

## Research findings and next steps

Both new user-supplied reports were reviewed. The initialization report supports retaining separate graph and ordinary-stream clearing policies. Its partial-histogram/overwrite-reduction design is already implemented as `shared_partial`; prior search measurements did not make it the winner in the highlighted cases. Future experiments should investigate that existing reducer's cost and graph/stream scheduling with controlled, unprofiled measurements before any replacement of a winning kernel.

The report's external-event graph timing and single-eviction-per-batch recommendations differ from our established internal-event and per-operation eviction protocols. They are diagnostic alternatives to compare, not automatic replacements for the measurement protocol.

The tree-training report describes additional weighted statistics that tree training needs. No weighted histogram or trainer code is being added in this change. Any future extension must build on our kernels in separate specializations, preserve the existing counting path, prove signed weighted-accumulation bounds, and demonstrate its advantage on equivalent work before adoption. Counting speedups alone do not establish weighted-histogram or end-to-end training speedups.
