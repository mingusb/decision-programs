# A5000 profiling and optimization pass

This pass found and removed a substantial initialization bottleneck in repeated CUDA graphs, added full-tile and packed-load policies, and improved the tuning and measurement procedure. It does not establish that every inefficiency has been removed or that this is the fastest histogram on every workload or GPU.

We compare against NVIDIA’s CUB library of optimized GPU routines; its histogram is our reference implementation. The unmodified NVIDIA CUDA histogram sample is an additional reference where applicable.

This report preserves the profiling campaigns and their original binaries. The subsequent [automatic-defaults report](../a5000-defaults/REPORT.md) describes the current API, workload-based selection, and its separate verification. Its new binary is not substituted into the measurements below.

The target remains the NVIDIA RTX A5000 Laptop GPU: SM86, 48 SMs, 4 MiB reported L2, CUDA 13.4, CUB 3.4.2, under WSL2. This report separates the resumed driver 597.06 campaign from the earlier driver 596.71 evidence. Results from the two sessions are not pooled, and differences between sessions do not establish a driver effect.

GPU clocks were not locked, and GPU experiments ran serially. Unprofiled complete-operation timings determine rankings; Nsight measurements diagnose mechanisms. Sizes below use powers of two: 1M means 1,048,576 samples and 16M means 16,777,216 samples.

## Current driver 597.06 results

The September 21 resumed session reruns the archived binaries on the same GPU. The clearing-only comparison reproduces substantial gains: **2.116× for N=1M/u32/B=8**, **4.456× for N=4096/u8/B=256**, and **2.917× for N=1M/u8/B=256**, using u32 output and warm graphs. These are median paired ratios across three invocations, with identical counting configurations and unchanged reference controls. They measure replacing output memset with a clearing kernel; they are not speedups against CUB. See the [current ablation audit](resumed-2026-09-21/clear-ablation-summary.md).

All twelve fresh search/validation plans and their 36 independent confirmation invocations have completed and passed the artifact audit. Eleven frozen custom choices beat the strongest applicable reference in all three confirmation runs; the 16M-byte workload retains the NVIDIA histogram reference. Representative reference-relative gains are **2.46–2.47×** for 1M u32 samples with eight bins, **1.36×** for 1M byte samples with 256 bins, **10.13–10.34×** for 16M u32 samples with 4096 bins and u64 output, and **30.42–30.65×** for the new 16,384-bin/u64 case. The [current-session report](resumed-2026-09-21/REPORT.md) contains all twelve workloads, absolute operation times, separate scalar-control comparisons, and limits.

Most new policy gains over the retained scalar catalog are modest because that catalog shares the new clearing implementation. The cached-byte case improves 1.23×, and the 16,384-bin/u64 case improves 2.71–2.80× over its global-atomic scalar control; the latter also changes algorithm and shared-memory capacity. These are portfolio comparisons, not isolated packed-load gains. The 30× figure above compares against CUB for that specific workload and is not the improvement over our previous implementation.

A subsequent direct-stream ablation found that unconditional kernel clearing can regress this other launch mode: the small-eight-bin case has a reference-normalized runtime/kernel ratio of 0.883–0.908 across three pairs. The byte cases are noisier, and the large case stays near parity. This prompted an explicit clearing policy, retaining runtime clearing unless kernel clearing is requested. That revision completed a separate [12/12-case reconfirmation audit](clear-policy-2026-09-21/final-comparison.md), covering 36 invocations with the earlier choices frozen and graph clearing explicitly selected. All eleven custom choices again beat the reference in every run. The revision is preserved at `build/profiled-clear-policy`, with benchmark SHA256 `5c50596f6c820152e8b3df6c0d9529e11461746b6d4000f89ad1e27639bc74c8`; the earlier numerical table keeps its original binary and results. See the [stream audit](resumed-2026-09-21/stream-ablation-summary.md).

The explicit-clearing revision passes [18,622 histogram executions and sixteen CPU autotuner tests](clear-policy-2026-09-21/ctest-details.log), plus 3,098 executions under each of Compute Sanitizer [memcheck](clear-policy-2026-09-21/memcheck.log), [racecheck](clear-policy-2026-09-21/racecheck.log), and [synccheck](clear-policy-2026-09-21/synccheck.log), with zero errors or race hazards. Fresh Nsight Compute captures from the earlier resumed build show measurable remaining limits: a cache-flushed byte kernel has many cycles without an eligible warp, while the 16,384-bin/u64 policy 14 path reaches 90.60% of peak DRAM throughput despite shared-memory capacity limiting occupancy. These observations guide further experiments; they do not establish that all inefficiencies have been removed.

The [current-driver cache probe](resumed-2026-09-21/cache-eviction-evidence.json) also reproduces the intended eviction effect: moving the 64 MiB eviction pass between reads of a 1 MiB input removes 32,768 L2 read-hit sectors and adds exactly that many misses. This verifies the controlled case, rather than every possible cache state.

## The largest improvement: output initialization

The following mechanism and ablation evidence was collected on September 15–16 with driver 596.71. Current-driver results are reported separately above.

The original nonempty custom atomic and bit-plane operations used `cudaMemsetAsync` before counting. In a warm graph containing twenty byte-histogram operations, Nsight Systems showed a median **211.941 µs in inter-node gaps**, against about 110 µs of active work. Replacing that memset with a small CUDA clearing kernel reduced those gaps to **8.864 µs**. Counting still follows clearing in the same stream, and all initialization remains inside the measured operation. CUB, the NVIDIA sample, shared-partial reduction, and empty-input paths retain their original initialization behavior.

Matched, unprofiled comparisons use the same counting policies and timing protocol on archived before/after binaries. Three invocations per build use seed 424242, 21 samples, and 32 operations per sample; build order alternates. Median paired speedups for shared histograms are **2.36× for N=1M/u32/B=8**, **5.40× for N=4096/u8/B=256**, and **3.04× for N=1M/u8/B=256**. All three cases use u32 output. Reference and shared-partial controls generally remain close to 1×. Reference-normalized ratios account descriptively for a common timing shift in one pair; they do not identify its cause.

For the 1 MiB byte case, `shared:7:96` falls from 15.7–16.5 µs to 4.83–5.41 µs across the three repeats. These are operation times, including clearing. The trace does not identify the physical execution engine used by the memset; no such attribution is necessary to establish the observed gap reduction.

See the [paired ablation audit](clear-ablation-summary.md), [raw ablation data](clear-ablation/), [trace analysis](clear-trace-audit.json), [before trace](timeline-clear-before.nsys-rep), and [after trace](timeline-clear-after.nsys-rep). The before binary is preserved at `build/profiled-loads/histogram_bench` with hashes in [loads-build.json](loads-build.json). The four shared counting specializations used in this ablation have identical encoded instructions and register/stack/spill counts across the two builds.

## Load and resource policies

Policies 0–5 preserve the original scalar counting kernels. Policies 6–13 add full-tile paths, aligned packed loads, wider 512/1024-thread blocks, and additional shared replicas. Full tiles omit per-item bounds checks; misaligned inputs and tails retain scalar handling. Policies 14–15 permit up to 96 KiB dynamic shared memory after explicit `gh::prepare(config)` outside timing and graph capture. This enables a 16,384-bin histogram with 32-bit locals, including 64-bit output.

Compiled SASS confirms that 256-thread/vector4 policies use two 128-bit loads per thread for eight u32 samples, or two 32-bit loads for eight byte samples, on the aligned full-tile path. Sixteen-item policies use four packed loads. None of the 772 function records parsed from the load-policy build log reported spills. Packed loading changes per-thread sample order and therefore RLE opportunities; RLE comparisons do not isolate instruction width alone. Bit-plane vector4 policies use the full-tile path, as CSV metadata explicitly records.

Matched Nsight Compute captures use kernel replay, `--cache-control all`, `--clock-control none`, and one counting kernel after skipping the two standalone correctness launches. Collection settings and workloads match; frequencies are similar but not fixed. Reported average SM frequencies are 1.44 GHz before versus 1.45 GHz after for the byte case, 1.45 GHz for both large-u32 captures, and 1.43 GHz before versus 1.45 GHz after for hot99 RLE. DRAM frequencies match within each comparison at the displayed precision.

| N=16M, `shared:policy:96`, u32 output | Scalar policy 3 | Full-tile policy 8 | Vector policy 9 |
|---|---:|---:|---:|
| u8 input, B=256: kernel duration | 61.22 µs | 55.74 µs | 56.13 µs |
| u8 input: DRAM throughput relative to peak | 78.92% | 87.07% | 86.96% |
| Registers/thread | 52 | 51 | 56 |
| u32 input, B=4096: kernel duration | 190.85 µs | not captured | 191.07 µs |
| u32 input, B=4096: DRAM throughput | 94.25% | not captured | 94.00% |

The large u32 control was already near its profiled DRAM limit and did not improve materially. The hot99 RLE kernel also changed little in this cache-flushed capture: 16.61→16.54 µs. These are diagnostic kernel measurements, separate from warm-graph complete-operation rankings. See the [before audit](before-audit.md), [shared SASS analysis](shared-load-sass.md), and [full-tile byte](loads-byte-full.details.txt), [vector byte](loads-byte-vector.details.txt), [vector large-u32](loads-large-vector.details.txt), and [vector RLE](loads-hot-rle-vector.details.txt) captures.

## Measurement and selection

Timing protocol 3 embeds event-record nodes inside captured graphs. Warm graphs time the entire batch between one event pair; cold graphs contain distinct eviction→start→histogram→end sequences and average the independent intervals. Host submission lies outside graph intervals. Stream timing remains available and can include submission gaps. Nsight Systems exports have zero device timestamps for CUDA events on this stack, so the node traces do not independently reconstruct these event boundaries.

A ladder with thirty invocations per protocol showed that this event-boundary correction alone did **not** solve the observed instability: samples exceeding twice their own row median were 74/1800 before and 108/1800 after. The later output-clear ablation separately established a real operation improvement. See the [timing comparison](timing-comparison.md). Device scheduling and frequency variability are not eliminated; raw samples remain part of the evidence.

The tuner preserves the four fastest custom candidates plus the best candidate from each otherwise missing family. This avoids excluding a stable family when near-duplicate configurations occupy all four slots. Search, validation, and replay use the same configurable batch size. This campaign's schema 3 records timing protocol, effective load policy, shared limit, and warmup setting. The later explicit-clearing revision uses schema 4, which also records the clearing policy. Historical plans retain their matching archived tooling.

The completed current-driver selection campaign uses five search samples and fifteen validation samples, batch 32 throughout, with search seed 12345 and validation seeds 67890/24680. Custom selection requires at least 1.05× the fastest applicable reference on **each** validation seed. CUB and the unmodified NVIDIA sample compete where applicable. Each chosen configuration is then frozen for separate confirmation: two invocations with seed 424242 and one with 987654, 21 samples each. Confirmation also includes the fastest original scalar candidate from that case's search, under the new clearing implementation. That scalar comparison isolates portfolio-policy selection from the earlier clear ablation; it is not a comparison with the entire old implementation.

Final confirmation results are in the [current-session audit](resumed-2026-09-21/final-comparison.md) and [JSON](resumed-2026-09-21/final-comparison.json). The [campaign runner](run_round.py) records invocations, raw CSVs, binary hashes, and before/after telemetry; telemetry snapshots are not a continuous clock trace. The [confirmation analyzer](analyze_round.py) checks exact candidate sets, metadata, hashes, and raw samples. Validation seeds select the winner and are not presented as an independent final test.

## Cache verification

A separate whole-graph probe matches total traffic while comparing E→H→H against H→E→H, where E is the 64 MiB eviction pass and H reads a 1 MiB input. Nsight Compute kernel replay flushes before the graph and preserves cache behavior between its nodes. Moving eviction between the reads changes L2 TEX read hits from 32,768 to 0 and increases misses by exactly 32,768 sectors: one 1 MiB input read. Total traffic includes the same eviction pass in both graphs.

This verifies the capacity-eviction procedure for the controlled input/GPU case. It does not guarantee that every cache line is invalid in every workload. See [cache evidence](cache-eviction-evidence.json), adjacent exact commands, and `histogram_cache_probe`.

## Correctness and limits

The archived final build passed these checks on September 15–16:

- [18,622 histogram executions](final-tests-details.log) against independent CPU counts, including alignment, full-tile boundaries, tails, persistent loops, output/workspace canaries, overwrite behavior, and nondefault streams.
- 3,098 executions under each of Compute Sanitizer [memcheck](memcheck-final.log), [racecheck](racecheck-final.log), and [synccheck](synccheck-final.log): zero errors or race hazards.
- [Ten actual large-count executions](large-count.log), including packed RLE, exactly reproduce 4,294,967,328 in 64-bit output.
- [Cold graph execution](graph-memcheck.log) with a tail and 64 KiB actual shared storage passes memcheck and CPU verification.
- [Fifteen CPU autotuner tests](final-tests-details.log) cover parsing, family preservation, baseline gates, metadata, replay, and batch consistency.

The implementation still accepts direct bin IDs and unsigned counts. Weighted gradient/Hessian accumulation, GBDT integration, sparse output, and architecture-specific SM90+ features are outside this pass. Runtime distribution detection and independently evaluated general dispatch remain open. Performance is scoped to this GPU, contract, and measurement regime. Generic sweeps of the full catalog require the 96 KiB opt-in capacity; `supported()` is structural, while `prepare()` checks device capacity. Other GPUs need validation and potentially a filtered catalog.

The original profiled binary/tool archive is `build/profiled`; [final-build.json](final-build.json) records its hashes and [final-source.tar.gz](final-source.tar.gz) preserves its implementation sources. `build/profiled-clear-policy` preserves the subsequent explicit-clearing revision, and `build/expanded` retains the previous schema 2 binary, tuner, and campaign analyzer. The campaigns keep their original binaries and metadata rather than mixing measurements across revisions. See the [automatic-defaults report](../a5000-defaults/REPORT.md) for the latest implementation and verification.
