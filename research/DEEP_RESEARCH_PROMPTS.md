# GPU histogram Deep Research prompt pack

This pack contains 12 independent research prompts and one synthesis prompt. Each prompt is complete: open its text file and paste the entire contents into a separate ChatGPT Deep Research session. The full text of every prompt is also collected below. No shared preamble or previous chat is required.

Start with **01, 02, 03, 09, and 11** to establish prior art, baselines, hardware, the CUDA C++23 toolchain, and measurement rules. Run **04-08, 10, and 12** in any order. Run **13 last**, attaching the completed reports. Reports 01-12 are independent and can be researched concurrently.

Export each report as a PDF using its prompt filename stem, for example `01-literature-and-problem-map.pdf`. Keep citations, the bibliography, code, and tables in the export. Send the PDFs back here as they finish; there is no need to wait for the whole set. For report 13, attach as much of the full corpus as the interface permits and identify any omissions.

The target is a defensible performance frontier across explicit workloads and GPUs. The prompts require exact contracts, complete cost accounting, versioned sources, and a distinction between published evidence and proposed experiments. They deliberately do not select a GPU, assume all NVIDIA architecture variants are equivalent, or treat C++23 support as timeless.

The evidence-reporting structure follows OpenAI's guidance to connect important claims to original sources and preserve uncertainty: [Synthesize research evidence](https://learn.chatgpt.com/use-cases/synthesize-research-evidence). The technical starting points include [NVIDIA's C++ language support documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html), [CCCL source](https://github.com/NVIDIA/cccl), [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), and the [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html). Each research run must check current versions and follow the literature beyond these seeds.

| Report | Copy-ready prompt | Research outcome |
| --- | --- | --- |
| 01 | [Literature, problem definitions, and the competitive frontier](prompts/01-literature-and-problem-map.txt) | Find the strongest prior art and define which performance claims are comparable. |
| 02 | [CUB/CCCL and production-library source audit](prompts/02-production-library-source-audit.txt) | Understand the actual implementations and establish demanding, fair baselines. |
| 03 | [NVIDIA architecture, atomics, memory, and instructions](prompts/03-hardware-atomics-and-instructions.txt) | Connect histogram costs to hardware mechanisms and measurable limits. |
| 04 | [Dense histograms with small and medium bin counts](prompts/04-dense-histogram-algorithms.txt) | Develop the main kernel families and explain their crossover conditions. |
| 05 | [Skew, input ordering, and adaptive algorithm selection](prompts/05-skew-ordering-and-adaptive-dispatch.txt) | Handle contention robustly and assess whether runtime adaptation pays. |
| 06 | [Large bin counts, sparse outputs, and arbitrary keys](prompts/06-large-bin-sparse-and-arbitrary-keys.txt) | Find alternatives when dense replication, initialization, or random access dominates. |
| 07 | [Asynchronous pipelines, thread-block clusters, and merging](prompts/07-pipelines-clusters-and-merging.txt) | Evaluate modern hardware features against their whole-operation overhead. |
| 08 | [Binning semantics, weighted counts, and structured histograms](prompts/08-binning-weighted-and-structured-histograms.txt) | Cover the important histogram variants without conflating their costs or correctness. |
| 09 | [CUDA C++23 toolchains, code generation, and correctness](prompts/09-cuda-cpp23-toolchain-and-correctness.txt) | Turn research sketches into supported, correct, inspectable CUDA implementations. |
| 10 | [Performance models, microbenchmarks, and profiling](prompts/10-performance-models-and-profiling.txt) | Predict bottlenecks and design experiments that explain them. |
| 11 | [Benchmark design and a defensible fastest claim](prompts/11-benchmark-and-fastest-claim-protocol.txt) | Specify the evidence required to distinguish a real win from a timing artifact. |
| 12 | [Batching, fusion, streaming, and multiple GPUs](prompts/12-batching-fusion-streaming-and-multi-gpu.txt) | Find application-level improvements beyond an isolated histogram kernel. |
| 13 | [Synthesis, novelty checks, and an implementation roadmap](prompts/13-evidence-synthesis-and-experiment-roadmap.txt) | Turn the completed reports into a falsifiable plan for building the implementation. |


**01. Literature, problem definitions, and the competitive frontier**

```text
Report 01: Literature, problem definitions, and the competitive frontier
Suggested PDF filename: 01-literature-and-problem-map.pdf

I am building a state-of-the-art NVIDIA GPU histogram implementation in CUDA C++23. This report will be exported as a PDF and given to the engineer implementing and benchmarking it. Produce an implementation-oriented research report, not a general CUDA tutorial.

No particular GPU or workload has been selected. Investigate a performance frontier across workloads and supported NVIDIA architectures; do not assume one kernel wins everywhere. Prioritize exact integer counting with device-resident input and dense output, while clearly labeling other contracts. Let N mean input items, B output bins, and K observed distinct keys. Cover current, publicly documented and available hardware/toolchains as of your research date, including relevant Ampere, Ada, Hopper, and Blackwell variants; label older results, previews, and unverified availability. Preserve the CUDA C++23 target and verify relevant language, compiler, and library support rather than assuming it.

Use original papers, author artifacts, NVIDIA documentation, and actual implementation source as evidence. Follow promising references beyond these seed sources. Pin software versions and source commits, distinguish stable releases from development branches, and cite exact sections or code locations. Separate documented facts, published measurements, your deductions, and untested hypotheses. Do not invent benchmarks, citations, compilation results, APIs, or hardware throughput figures. If you cannot run CUDA, mark proposed code and experiments untested.

For consequential performance claims, recover GPU, software version, operation semantics, N, B, input/counter types, distribution AND ordering, baseline, and timing boundary; mark missing fields. Do not infer superiority from incomparable experiments. Report negative evidence and uncertainty.

Make the PDF self-contained: state the research date, define notation, use selectable text, keep code lines and tables narrow, and provide a numbered bibliography with full titles, authors/organizations, dates, URLs/DOIs, and versions. Include a claim-to-source table for major conclusions, algorithm sketches where useful, and prioritized experiments with predicted observations and falsification criteria. End with a concise implementation handoff: supported conclusions, open questions, and the next decisions this report enables. Proceed using explicit assumptions without waiting for clarification.

Specific research assignment:

Research the state of the art in exact NVIDIA GPU histograms, from foundational CUDA work through the research date. Search peer-reviewed proceedings, preprints, theses, author repositories, NVIDIA publications, production libraries, and recent source changes. Perform backward and forward citation searches; record the search terms, venues, and coverage gaps. Distinguish historical influence, current reproducible competitiveness, and unsupported claims.

First define the problem space. Separate direct integer bin IDs, evenly spaced numeric bins, arbitrary bin edges, weighted accumulation, independent channel histograms, joint multidimensional histograms, segmented/batched histograms, and dense versus sparse output. Distinguish overwrite from accumulation, exact counts from approximate sketches, and preprocessed inputs from preprocessing charged to this call. Make the main comparison exact unweighted counting; map extensions without treating them as interchangeable.

Build a taxonomy and lineage of direct atomics, privatization/replication, warp aggregation, voting, local sorting, run-length aggregation, hierarchical partial histograms, partitioning/multisplit, hash aggregation, and adaptive algorithms. Search adjacent work on GPU group-by, reduce-by-key, radix-sort digit histograms, and graph degree counting for transferable mechanisms. Explain the transfer cost and semantic differences. Ideas from other hardware or languages are eligible if their CUDA applicability is analyzed.

Create an evidence matrix for the most consequential papers and implementations: contribution, publication/version, supported contract, code/artifact availability and license, hardware, workload, baseline, timing scope, reported result, limitations, and relevance to present GPUs. Select entries by importance and evidence quality rather than meeting a citation quota. Identify conflicting findings and explain whether architecture, skew, ordering, bins, or excluded work could reconcile them.

Deliver a map of workload regimes where there is credible evidence for a leading approach, and mark regions where the winner is unknown. Do not name an overall fastest implementation without comparable evidence, and do not claim public research establishes superiority over unavailable private implementations. Identify the strongest reproducible competitors, useful datasets, missing replications, and a ranked reading list with the exact part worth reading. Finish with the research gaps most likely to yield meaningful improvements.

Starting points, not an exhaustive source list:
- https://github.com/NVIDIA/cccl
- https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
- https://arxiv.org/abs/1701.01189
- https://github.com/owensgroup/GpuMultisplit
- https://arxiv.org/abs/1011.0235
```


**02. CUB/CCCL and production-library source audit**

```text
Report 02: CUB/CCCL and production-library source audit
Suggested PDF filename: 02-production-library-source-audit.pdf

I am building a state-of-the-art NVIDIA GPU histogram implementation in CUDA C++23. This report will be exported as a PDF and given to the engineer implementing and benchmarking it. Produce an implementation-oriented research report, not a general CUDA tutorial.

No particular GPU or workload has been selected. Investigate a performance frontier across workloads and supported NVIDIA architectures; do not assume one kernel wins everywhere. Prioritize exact integer counting with device-resident input and dense output, while clearly labeling other contracts. Let N mean input items, B output bins, and K observed distinct keys. Cover current, publicly documented and available hardware/toolchains as of your research date, including relevant Ampere, Ada, Hopper, and Blackwell variants; label older results, previews, and unverified availability. Preserve the CUDA C++23 target and verify relevant language, compiler, and library support rather than assuming it.

Use original papers, author artifacts, NVIDIA documentation, and actual implementation source as evidence. Follow promising references beyond these seed sources. Pin software versions and source commits, distinguish stable releases from development branches, and cite exact sections or code locations. Separate documented facts, published measurements, your deductions, and untested hypotheses. Do not invent benchmarks, citations, compilation results, APIs, or hardware throughput figures. If you cannot run CUDA, mark proposed code and experiments untested.

For consequential performance claims, recover GPU, software version, operation semantics, N, B, input/counter types, distribution AND ordering, baseline, and timing boundary; mark missing fields. Do not infer superiority from incomparable experiments. Report negative evidence and uncertainty.

Make the PDF self-contained: state the research date, define notation, use selectable text, keep code lines and tables narrow, and provide a numbered bibliography with full titles, authors/organizations, dates, URLs/DOIs, and versions. Include a claim-to-source table for major conclusions, algorithm sketches where useful, and prioritized experiments with predicted observations and falsification criteria. End with a concise implementation handoff: supported conclusions, open questions, and the next decisions this report enables. Proceed using explicit assumptions without waiting for clarification.

Specific research assignment:

Audit the actual source and contracts of the strongest available CUDA histogram libraries. Give deepest attention to CUB in CCCL: DeviceHistogram, BlockHistogram, their dispatch policies, agents, initialization, bin mapping, accumulation, partial storage, and merge paths. Compare the newest stable release with a pinned development commit when there are relevant changes. Do not assume the device-wide implementation uses every algorithm exposed by BlockHistogram.

Trace representative API calls through source files into the kernels and policy choices. Explain the behavior for small versus large N, bins fitting versus exceeding shared memory, input and counter widths, equal-width versus arbitrary-edge binning, channels, strides/regions, and invalid samples. Verify exact endpoint conventions and the relationship between numbers of levels and bins. Record API limits, aliasing restrictions, temporary storage requirements, graph/stream behavior, and any architecture-specific dispatch.

Investigate atomic versus sort-based block histograms, replica counts, work per thread, grid sizing, counter types, overflow handling, and reduction strategy. Identify which parameters are compile-time, runtime, architecture-dependent, or tunable. Explain the evidence for compiler-generated optimizations; do not infer assembly from C++ spelling. Distinguish documented behavior from implementation details that can change.

Review relevant benchmarks, tests, recent releases, performance issues, and pull requests. Follow issue links to their resolution and state whether changes are proposed, merged, released, or superseded. Treat issue measurements as self-reported until their protocol is recoverable. Consider NVIDIA NPP and relevant OpenCV CUDA, framework bincount/histogram, and other maintained implementations as additional baselines. Identify wrappers around the same underlying primitive. If a library is closed source, state what can be learned from its documented API and black-box measurement.

Deliver a source map with immutable links, concise pseudocode for important execution paths, a table of defaults and dispatch conditions, and a baseline integration recipe that preserves equivalent semantics. Include temporary allocation/query costs, initialization, and all launches in clearly defined timing scopes. Propose the most promising weaknesses to investigate, the circumstances in which existing tuning is already strong, and the minimal experiments needed to distinguish a real algorithmic advantage from an unfair configuration comparison.

Seed paths to verify and pin:
- https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_histogram.cuh
- https://github.com/NVIDIA/cccl/blob/main/cub/cub/agent/agent_histogram.cuh
- https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/dispatch/dispatch_histogram.cuh
- https://github.com/NVIDIA/cccl/blob/main/cub/cub/block/block_histogram.cuh
- https://github.com/NVIDIA/cccl/issues/4957
- https://github.com/NVIDIA/cccl/issues/4974
- https://docs.nvidia.com/cuda/npp/index.html
```


**03. NVIDIA architecture, atomics, memory, and instructions**

```text
Report 03: NVIDIA architecture, atomics, memory, and instructions
Suggested PDF filename: 03-hardware-atomics-and-instructions.pdf

I am building a state-of-the-art NVIDIA GPU histogram implementation in CUDA C++23. This report will be exported as a PDF and given to the engineer implementing and benchmarking it. Produce an implementation-oriented research report, not a general CUDA tutorial.

No particular GPU or workload has been selected. Investigate a performance frontier across workloads and supported NVIDIA architectures; do not assume one kernel wins everywhere. Prioritize exact integer counting with device-resident input and dense output, while clearly labeling other contracts. Let N mean input items, B output bins, and K observed distinct keys. Cover current, publicly documented and available hardware/toolchains as of your research date, including relevant Ampere, Ada, Hopper, and Blackwell variants; label older results, previews, and unverified availability. Preserve the CUDA C++23 target and verify relevant language, compiler, and library support rather than assuming it.

Use original papers, author artifacts, NVIDIA documentation, and actual implementation source as evidence. Follow promising references beyond these seed sources. Pin software versions and source commits, distinguish stable releases from development branches, and cite exact sections or code locations. Separate documented facts, published measurements, your deductions, and untested hypotheses. Do not invent benchmarks, citations, compilation results, APIs, or hardware throughput figures. If you cannot run CUDA, mark proposed code and experiments untested.

For consequential performance claims, recover GPU, software version, operation semantics, N, B, input/counter types, distribution AND ordering, baseline, and timing boundary; mark missing fields. Do not infer superiority from incomparable experiments. Report negative evidence and uncertainty.

Make the PDF self-contained: state the research date, define notation, use selectable text, keep code lines and tables narrow, and provide a numbered bibliography with full titles, authors/organizations, dates, URLs/DOIs, and versions. Include a claim-to-source table for major conclusions, algorithm sketches where useful, and prioritized experiments with predicted observations and falsification criteria. End with a concise implementation handoff: supported conclusions, open questions, and the next decisions this report enables. Proceed using explicit assumptions without waiting for clarification.

Specific research assignment:

Research the NVIDIA hardware and instruction behavior that determines histogram performance. Prioritize differences among available Ampere, Ada, Hopper, and Blackwell products. Distinguish compute capabilities and actual SKUs within a marketing family; do not assume all family members expose the same features, resources, or performance. Use older architectures only when they explain a relevant change.

Build a capability matrix covering shared memory capacity and banking, register limits, occupancy constraints, L1/L2 organization, memory bandwidth, local/shared/global/cluster-scoped atomic support, counter types, warp collectives, asynchronous copies, and cluster features. Separate documented limits from microbenchmark observations. Verify each feature's minimum architecture and toolkit requirements.

Analyze same-address atomic contention, different-address conflicts, shared-bank effects, hot cache lines, memory partition effects where supported by evidence, and return-value dependencies. Separate latency of a dependent chain from throughput of independent operations. Compare 32-bit and 64-bit integer updates, relevant floating-point operations, and packed/vector operations only where their actual atomicity semantics are suitable. Explain memory scope and ordering independently of where data resides.

Trace relevant CUDA operations through PTX and, where accessible, SASS. Investigate atomic versus reduction instruction lowering when the old value is unused, automatic warp aggregation and its limits for data-dependent keys, ballots, population counts, shuffles, key matching, and warp reductions. State which behaviors are documented, compiler-dependent, or inferred. Never assume a general warp-aggregation optimization handles arbitrary histograms.

Explain how privatization trades contention against register/shared-memory use, initialization work, merge work, and occupancy. Examine register spills, instruction issue limits, integer address arithmetic, coalescing, alignment, vector loads, cache residency, and small-grid underutilization. Do not claim every histogram is simply memory-bandwidth-bound.

Deliver a capability matrix with exact sources, a bottleneck decision tree, representative instruction sequences or sketches, and isolated microbenchmark designs. Include same-bin versus distinct-bin atomics, address patterns with controlled banking/locality, returned versus unused atomic results, occupancy sweeps, and cache-resident versus streaming input. Specify what each experiment can and cannot establish and how to avoid turning compiler elimination or measurement overhead into a false hardware conclusion.

Starting points:
- https://docs.nvidia.com/cuda/cuda-programming-guide/
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
```


**04. Dense histograms with small and medium bin counts**

```text
Report 04: Dense histograms with small and medium bin counts
Suggested PDF filename: 04-dense-histogram-algorithms.pdf

I am building a state-of-the-art NVIDIA GPU histogram implementation in CUDA C++23. This report will be exported as a PDF and given to the engineer implementing and benchmarking it. Produce an implementation-oriented research report, not a general CUDA tutorial.

No particular GPU or workload has been selected. Investigate a performance frontier across workloads and supported NVIDIA architectures; do not assume one kernel wins everywhere. Prioritize exact integer counting with device-resident input and dense output, while clearly labeling other contracts. Let N mean input items, B output bins, and K observed distinct keys. Cover current, publicly documented and available hardware/toolchains as of your research date, including relevant Ampere, Ada, Hopper, and Blackwell variants; label older results, previews, and unverified availability. Preserve the CUDA C++23 target and verify relevant language, compiler, and library support rather than assuming it.

Use original papers, author artifacts, NVIDIA documentation, and actual implementation source as evidence. Follow promising references beyond these seed sources. Pin software versions and source commits, distinguish stable releases from development branches, and cite exact sections or code locations. Separate documented facts, published measurements, your deductions, and untested hypotheses. Do not invent benchmarks, citations, compilation results, APIs, or hardware throughput figures. If you cannot run CUDA, mark proposed code and experiments untested.

For consequential performance claims, recover GPU, software version, operation semantics, N, B, input/counter types, distribution AND ordering, baseline, and timing boundary; mark missing fields. Do not infer superiority from incomparable experiments. Report negative evidence and uncertainty.

Make the PDF self-contained: state the research date, define notation, use selectable text, keep code lines and tables narrow, and provide a numbered bibliography with full titles, authors/organizations, dates, URLs/DOIs, and versions. Include a claim-to-source table for major conclusions, algorithm sketches where useful, and prioritized experiments with predicted observations and falsification criteria. End with a concise implementation handoff: supported conclusions, open questions, and the next decisions this report enables. Proceed using explicit assumptions without waiting for clarification.

Specific research assignment:

Research exact, unweighted dense histograms for bin counts that range from very small to the practical limits of shared-memory privatization. Use direct valid bin IDs to isolate accumulation first; discuss numeric bin mapping as an added cost. Include byte-valued 256-bin counting as a major case without treating it as the entire problem. Sweep tiny N through streaming datasets much larger than cache, non-power-of-two B, and 32-bit versus 64-bit final counts.

Compare direct global atomics; block-private and warp-private shared histograms; replicated/padded layouts; per-thread register counters when feasible; lane ownership and ballot/popcount counting; duplicate-key aggregation; thread-local run-length counting; local sort-and-count; and hybrid approaches. Include comparisons that show when each technique's overhead outweighs contention reduction. Explain dynamic indexing and possible register spilling rather than assuming a C++ local array resides in registers.

For each serious candidate, show CUDA-style pseudocode for the complete operation: zeroing, load distribution, optional local aggregation, updates, flushing, and final merge. Analyze coalesced versus strided work assignment, items per thread, vectorized loads, alignment and tails, grid-stride loops, persistent grids, and compile-time specialization. Discuss both emitting partial histograms for reduction and directly atomically merging to final output.

Derive parameterized costs in N, B, blocks, replicas, items per thread, local distinct-key counts, and counter width. Include intermediate memory traffic, initialization, barriers, occupancy, and merge work. Give symbolic or explicitly assumed crossover conditions; do not fabricate numeric thresholds. Distinguish an impressive block primitive from a fast complete device-wide histogram.

Address partial-warp masks, barriers under divergence, safe shared-memory reuse, counter overflow within tiles and grid-stride loops, promotion to wider totals, and repeated calls. State exactly when narrow partial counters are safe. Analyze uniform random, single-hot, few-hot, locally correlated, and shuffled inputs.

Deliver a candidate matrix with expected winning and losing regimes, resource formulas, annotated algorithm sketches, and a small prioritized experiment set. Identify which existing CUB primitives can implement a fair prototype and which custom kernels are needed to test a distinct idea. Separate established techniques from proposed combinations whose benefit remains unmeasured.

Starting points:
- https://github.com/NVIDIA/cccl/tree/main/cub/cub
- https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
- https://arxiv.org/abs/1701.01189
```


**05. Skew, input ordering, and adaptive algorithm selection**

```text
Report 05: Skew, input ordering, and adaptive algorithm selection
Suggested PDF filename: 05-skew-ordering-and-adaptive-dispatch.pdf

I am building a state-of-the-art NVIDIA GPU histogram implementation in CUDA C++23. This report will be exported as a PDF and given to the engineer implementing and benchmarking it. Produce an implementation-oriented research report, not a general CUDA tutorial.

No particular GPU or workload has been selected. Investigate a performance frontier across workloads and supported NVIDIA architectures; do not assume one kernel wins everywhere. Prioritize exact integer counting with device-resident input and dense output, while clearly labeling other contracts. Let N mean input items, B output bins, and K observed distinct keys. Cover current, publicly documented and available hardware/toolchains as of your research date, including relevant Ampere, Ada, Hopper, and Blackwell variants; label older results, previews, and unverified availability. Preserve the CUDA C++23 target and verify relevant language, compiler, and library support rather than assuming it.

Use original papers, author artifacts, NVIDIA documentation, and actual implementation source as evidence. Follow promising references beyond these seed sources. Pin software versions and source commits, distinguish stable releases from development branches, and cite exact sections or code locations. Separate documented facts, published measurements, your deductions, and untested hypotheses. Do not invent benchmarks, citations, compilation results, APIs, or hardware throughput figures. If you cannot run CUDA, mark proposed code and experiments untested.

For consequential performance claims, recover GPU, software version, operation semantics, N, B, input/counter types, distribution AND ordering, baseline, and timing boundary; mark missing fields. Do not infer superiority from incomparable experiments. Report negative evidence and uncertainty.

Make the PDF self-contained: state the research date, define notation, use selectable text, keep code lines and tables narrow, and provide a numbered bibliography with full titles, authors/organizations, dates, URLs/DOIs, and versions. Include a claim-to-source table for major conclusions, algorithm sketches where useful, and prioritized experiments with predicted observations and falsification criteria. End with a concise implementation handoff: supported conclusions, open questions, and the next decisions this report enables. Proceed using explicit assumptions without waiting for clarification.

Specific research assignment:

Research how distribution shape and input ordering affect exact CUDA histogram algorithms, and how a practical implementation can select an algorithm without knowing the answer in advance.

Distinguish global bin probabilities from duplicate keys within a warp/block, temporal run lengths, local support size, periodicity, and correlation with thread-to-item mapping. Explain why one scalar such as entropy or maximum-bin probability may be insufficient. Compare uniform random, single-hot, few-hot, Zipf-like, mixtures, clustered regions, long runs, sorted, alternating, and adversarially ordered data. Construct paired cases with identical total counts but different order, and different global distributions with similar local duplication.

Evaluate warp key matching and aggregation, vote-based methods, run-length aggregation, hot-bin replication, local sorting, partitioning, adaptive privatization, and hybrid paths. Charge all detection and preprocessing work. Account for high-entropy inputs on which a contention-avoidance mechanism may become pure overhead and for inputs whose distribution changes over time or across blocks.

Compare static specialization using public metadata, offline autotuning, per-call sampling, per-block adaptation, fused statistics gathering, and reuse of statistics across calls. Sampling may guide an exact algorithm but must never replace counting unsampled data. Analyze sample bias, synchronization, CPU round trips, graph capture, scratch requirements, and the cost of making the decision available to GPU work. Do not assume a host dispatch decision is free when it depends on device data.

Specify a practical tuning search space: block/grid size, items per thread, aggregation method, replica count, safe partial-counter widths, partial layout, and merge strategy. Compare exhaustive, staged, and model-guided search; account for noisy measurements, compilation/code-size costs, cached tuning results, and transfer across devices. Identify parameters that interact and cannot be optimized independently.

Propose a small, interpretable feature set, candidate selector, confidence/fallback policy, and break-even model. Include a static baseline, the complete adaptive pipeline, and a hindsight oracle used only as an analytical upper comparator. Measure selection regret, worst-case slowdown, and adaptation cost as well as average speedup. Explain how to tune without leaking test distributions, dataset identity, or privileged knowledge of actual counts into the selector.

Deliver implementable selector pseudocode, representative generator specifications, a held-out validation plan, failure cases, and experiments that could demonstrate adaptation is unnecessary. Distinguish offline tuning costs from online latency and give a policy for unfamiliar hardware or workloads. Identify which conclusions are supported by histogram research and which are hypotheses transferred from adjacent GPU algorithms.

Starting points:
- https://arxiv.org/abs/1011.0235
- https://github.com/NVIDIA/cccl
- https://arxiv.org/abs/1701.01189
```


**06. Large bin counts, sparse outputs, and arbitrary keys**

```text
Report 06: Large bin counts, sparse outputs, and arbitrary keys
Suggested PDF filename: 06-large-bin-sparse-and-arbitrary-keys.pdf

I am building a state-of-the-art NVIDIA GPU histogram implementation in CUDA C++23. This report will be exported as a PDF and given to the engineer implementing and benchmarking it. Produce an implementation-oriented research report, not a general CUDA tutorial.

No particular GPU or workload has been selected. Investigate a performance frontier across workloads and supported NVIDIA architectures; do not assume one kernel wins everywhere. Prioritize exact integer counting with device-resident input and dense output, while clearly labeling other contracts. Let N mean input items, B output bins, and K observed distinct keys. Cover current, publicly documented and available hardware/toolchains as of your research date, including relevant Ampere, Ada, Hopper, and Blackwell variants; label older results, previews, and unverified availability. Preserve the CUDA C++23 target and verify relevant language, compiler, and library support rather than assuming it.

Use original papers, author artifacts, NVIDIA documentation, and actual implementation source as evidence. Follow promising references beyond these seed sources. Pin software versions and source commits, distinguish stable releases from development branches, and cite exact sections or code locations. Separate documented facts, published measurements, your deductions, and untested hypotheses. Do not invent benchmarks, citations, compilation results, APIs, or hardware throughput figures. If you cannot run CUDA, mark proposed code and experiments untested.

For consequential performance claims, recover GPU, software version, operation semantics, N, B, input/counter types, distribution AND ordering, baseline, and timing boundary; mark missing fields. Do not infer superiority from incomparable experiments. Report negative evidence and uncertainty.

Make the PDF self-contained: state the research date, define notation, use selectable text, keep code lines and tables narrow, and provide a numbered bibliography with full titles, authors/organizations, dates, URLs/DOIs, and versions. Include a claim-to-source table for major conclusions, algorithm sketches where useful, and prioritized experiments with predicted observations and falsification criteria. End with a concise implementation handoff: supported conclusions, open questions, and the next decisions this report enables. Proceed using explicit assumptions without waiting for clarification.

Specific research assignment:

Research exact GPU histograms beyond the capacity or efficiency of small shared-memory histograms: B from thousands through millions or larger where feasible, B much greater than N, huge key domains with small K, and high-cardinality inputs. Use symbolic resource bounds and clearly label illustrative scales rather than assuming all cases fit a device.

Define three separate contracts: a fully materialized B-counter dense output; a compact list of K (key,count) pairs with specified ordering; and sparse updates to an already valid dense output. Distinguish known bounded bin IDs from arbitrary integer labels that require a dictionary or sort. Charge initialization and dense materialization whenever dense output is required. Do not give sparse methods credit for omitting required zeros.

Compare direct global atomics, block/warp sparse aggregation, global and shared hash tables, coarse-to-fine binning, radix partitioning, multisplit, sorting plus run-length encoding or reduce-by-key, tiled bin processing, touched-bin tracking, generation tags, and dense/sparse hybrids. Investigate when multiple input scans can beat random updates, and count partition buffers, sorting traffic, prefix scans, allocation, cleanup, and all merges.

For hash methods, address exact key equality, collision resolution, load factor, capacity estimation, full tables, retries/spill paths, memory ordering when publishing entries, skew, and worst-case behavior. For touched-bin or generation techniques, explain concurrent first-touch initialization, list uniqueness, epoch rollover, and the cost of later dense output. For partitioning, analyze balance, extra passes, and hot partitions. Include 64-bit keys/counters and index/size overflow.

Deliver a regime map in N, B, K, local duplication, skew, counter width, and memory budget, with unsupported boundaries marked as hypotheses. Provide pseudocode and workspace formulas for the most promising alternatives. Identify baseline compositions from current CUB/Thrust or other primary-source implementations, and ensure each composition produces the same result contract before comparing speed.

Finish with a memory-constrained fallback strategy, representative stress inputs, and experiments that establish whether the bottleneck is output initialization, random atomic traffic, insufficient occupancy, partitioning, sorting, or materialization. Include inputs with nearly all keys distinct and cases with very sparse global occupancy but severe local hot spots.

Starting points:
- https://github.com/NVIDIA/cccl
- https://github.com/NVIDIA/cuCollections
- https://arxiv.org/abs/1701.01189
- https://github.com/owensgroup/GpuMultisplit
```


**07. Asynchronous pipelines, thread-block clusters, and merging**

```text
Report 07: Asynchronous pipelines, thread-block clusters, and merging
Suggested PDF filename: 07-pipelines-clusters-and-merging.pdf

I am building a state-of-the-art NVIDIA GPU histogram implementation in CUDA C++23. This report will be exported as a PDF and given to the engineer implementing and benchmarking it. Produce an implementation-oriented research report, not a general CUDA tutorial.

No particular GPU or workload has been selected. Investigate a performance frontier across workloads and supported NVIDIA architectures; do not assume one kernel wins everywhere. Prioritize exact integer counting with device-resident input and dense output, while clearly labeling other contracts. Let N mean input items, B output bins, and K observed distinct keys. Cover current, publicly documented and available hardware/toolchains as of your research date, including relevant Ampere, Ada, Hopper, and Blackwell variants; label older results, previews, and unverified availability. Preserve the CUDA C++23 target and verify relevant language, compiler, and library support rather than assuming it.

Use original papers, author artifacts, NVIDIA documentation, and actual implementation source as evidence. Follow promising references beyond these seed sources. Pin software versions and source commits, distinguish stable releases from development branches, and cite exact sections or code locations. Separate documented facts, published measurements, your deductions, and untested hypotheses. Do not invent benchmarks, citations, compilation results, APIs, or hardware throughput figures. If you cannot run CUDA, mark proposed code and experiments untested.

For consequential performance claims, recover GPU, software version, operation semantics, N, B, input/counter types, distribution AND ordering, baseline, and timing boundary; mark missing fields. Do not infer superiority from incomparable experiments. Report negative evidence and uncertainty.

Make the PDF self-contained: state the research date, define notation, use selectable text, keep code lines and tables narrow, and provide a numbered bibliography with full titles, authors/organizations, dates, URLs/DOIs, and versions. Include a claim-to-source table for major conclusions, algorithm sketches where useful, and prioritized experiments with predicted observations and falsification criteria. End with a concise implementation handoff: supported conclusions, open questions, and the next decisions this report enables. Proceed using explicit assumptions without waiting for clarification.

Specific research assignment:

Investigate whether modern NVIDIA execution and data-movement features can improve complete exact histograms. Treat every proposed benefit as a hypothesis until supported by a comparable measurement. Focus on available hardware and verify each feature's compute-capability, CUDA, launch, and resource requirements.

Study thread-block clusters and distributed shared memory, including NVIDIA's histogram example. Compare local block privatization, cluster-level bin sharding, remote shared-memory atomics, replicated cluster histograms, and direct global atomics. Analyze cluster sizes, resident clusters, scheduling constraints, per-block allocation versus cluster capacity, initialization, remote access locality, barrier costs, and lifetime guarantees before any block exits. Determine which bin-count and skew regimes could justify clusters.

Assess asynchronous global-to-shared copies, cp.async, Tensor Memory Accelerator and asynchronous bulk reduction where applicable, staging buffers, double buffering, warp specialization, and producer/consumer pipelines. Verify supported reduction types and exactness rather than treating a bulk-reduction operation as an arbitrary histogram scatter. Explain when staging adds unnecessary traffic to a one-pass histogram and when reuse, vectorization, or local aggregation could repay the cost. Cover alignment, tails, transaction sizes, barriers, and resource tradeoffs. Investigate newer scheduling features such as Cluster Launch Control only where documented and relevant.

Research the merge as a first-class algorithm: direct final atomic updates, separate reduction kernels, hierarchical reductions, transposed partial layouts, coalesced bin-wise reduction, fused initialization, and narrow partials promoted to wide totals. Derive how partial-histogram layout and grid size change writes, reads, parallelism, and final contention. Include zeroing and small-N launch overhead in comparisons.

Evaluate single-kernel completion schemes, cooperative launches, last-block protocols, persistent execution, and related scan techniques only with explicit proofs of memory visibility and forward progress. Reject implicit grid barriers and protocols that require nonresident blocks to make progress. Explain scratch reset and reentrancy across streams and graph replays.

Deliver a feature/support table, annotated CUDA-style sketches for a few plausible designs, complete traffic/resource models, and experiments comparing each feature-enabled design with a simple optimized counterpart. Require ablations isolating staging, clustering, aggregation, and merge layout. Explain which ideas should be rejected early because their setup, extra movement, or lost occupancy is likely to outweigh any saved atomics.

Starting points:
- https://docs.nvidia.com/cuda/cuda-programming-guide/
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- https://github.com/NVIDIA/cccl
```


**08. Binning semantics, weighted counts, and structured histograms**

```text
Report 08: Binning semantics, weighted counts, and structured histograms
Suggested PDF filename: 08-binning-weighted-and-structured-histograms.pdf

I am building a state-of-the-art NVIDIA GPU histogram implementation in CUDA C++23. This report will be exported as a PDF and given to the engineer implementing and benchmarking it. Produce an implementation-oriented research report, not a general CUDA tutorial.

No particular GPU or workload has been selected. Investigate a performance frontier across workloads and supported NVIDIA architectures; do not assume one kernel wins everywhere. Prioritize exact integer counting with device-resident input and dense output, while clearly labeling other contracts. Let N mean input items, B output bins, and K observed distinct keys. Cover current, publicly documented and available hardware/toolchains as of your research date, including relevant Ampere, Ada, Hopper, and Blackwell variants; label older results, previews, and unverified availability. Preserve the CUDA C++23 target and verify relevant language, compiler, and library support rather than assuming it.

Use original papers, author artifacts, NVIDIA documentation, and actual implementation source as evidence. Follow promising references beyond these seed sources. Pin software versions and source commits, distinguish stable releases from development branches, and cite exact sections or code locations. Separate documented facts, published measurements, your deductions, and untested hypotheses. Do not invent benchmarks, citations, compilation results, APIs, or hardware throughput figures. If you cannot run CUDA, mark proposed code and experiments untested.

For consequential performance claims, recover GPU, software version, operation semantics, N, B, input/counter types, distribution AND ordering, baseline, and timing boundary; mark missing fields. Do not infer superiority from incomparable experiments. Report negative evidence and uncertainty.

Make the PDF self-contained: state the research date, define notation, use selectable text, keep code lines and tables narrow, and provide a numbered bibliography with full titles, authors/organizations, dates, URLs/DOIs, and versions. Include a claim-to-source table for major conclusions, algorithm sketches where useful, and prioritized experiments with predicted observations and falsification criteria. End with a concise implementation handoff: supported conclusions, open questions, and the next decisions this report enables. Proceed using explicit assumptions without waiting for clarification.

Specific research assignment:

Research CUDA histogram variants beyond direct unweighted integer-bin counting. Organize them by mathematical contract and reuse opportunities; identify which require a different algorithm rather than a small extension. Keep performance comparisons within equivalent contracts.

For evenly spaced integer and floating-point bins, derive exact endpoint conventions and safe classification formulas. Investigate division versus reciprocal multiplication, integer overflow, signed inputs, conversion behavior, boundary rounding, fused operations, and fast-math changes. Cover the upper endpoint, values adjacent to boundaries, negative zero, NaN, infinities, and out-of-range handling. For arbitrary edges, compare binary search, search layouts, lookup/coarse indexing, and preprocessing; distinguish fixed reusable edges from per-call edges and count preprocessing accordingly.

For weighted histograms, distinguish integer weights and exact overflow-safe totals from floating-point accumulation with specified tolerances, determinism, or reproducibility. Investigate FP32/FP64 and narrower inputs with wider accumulation, cancellation, dynamic range, order dependence, atomic support, privatization, and reproducible alternatives. Do not label ordinary unordered floating-point addition an exact sum. Explain how restrictions on weights can permit faster methods and state those restrictions explicitly.

Cover independent channel histograms versus joint multidimensional histograms, array-of-structures versus structure-of-arrays layouts, channel masks, strides, pitches, regions, and the product growth of joint bin counts. Study segmented and batched histograms, variable segment lengths, empty segments, and load balancing between many tiny histograms and a few large ones.

Map specialized workloads such as local/sliding-window image histograms, integral histograms, histogram equalization, radix-digit histograms, graph degrees, and machine-learning gradient/Hessian histograms. Explain when additional structure permits incremental updates, reuse, or fusion. Separate approximate quantization or sketches from exact counting and identify what semantic concession produces any speedup.

Deliver a contract matrix, boundary/correctness test vectors, representative API designs, and algorithm choices for the most important variants. Rank their likely value for a reusable CUDA C++23 implementation. Identify optimization ideas that transfer to the core unweighted problem, and those that depend on extra metadata or restricted semantics. Give a plan for measuring classification and accumulation separately as diagnostics while reporting their combined cost for the actual operation.

Starting points:
- https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_histogram.cuh
- https://docs.nvidia.com/cuda/npp/index.html
- https://docs.nvidia.com/cuda/floating-point/index.html
```


**09. CUDA C++23 toolchains, code generation, and correctness**

```text
Report 09: CUDA C++23 toolchains, code generation, and correctness
Suggested PDF filename: 09-cuda-cpp23-toolchain-and-correctness.pdf

I am building a state-of-the-art NVIDIA GPU histogram implementation in CUDA C++23. This report will be exported as a PDF and given to the engineer implementing and benchmarking it. Produce an implementation-oriented research report, not a general CUDA tutorial.

No particular GPU or workload has been selected. Investigate a performance frontier across workloads and supported NVIDIA architectures; do not assume one kernel wins everywhere. Prioritize exact integer counting with device-resident input and dense output, while clearly labeling other contracts. Let N mean input items, B output bins, and K observed distinct keys. Cover current, publicly documented and available hardware/toolchains as of your research date, including relevant Ampere, Ada, Hopper, and Blackwell variants; label older results, previews, and unverified availability. Preserve the CUDA C++23 target and verify relevant language, compiler, and library support rather than assuming it.

Use original papers, author artifacts, NVIDIA documentation, and actual implementation source as evidence. Follow promising references beyond these seed sources. Pin software versions and source commits, distinguish stable releases from development branches, and cite exact sections or code locations. Separate documented facts, published measurements, your deductions, and untested hypotheses. Do not invent benchmarks, citations, compilation results, APIs, or hardware throughput figures. If you cannot run CUDA, mark proposed code and experiments untested.

For consequential performance claims, recover GPU, software version, operation semantics, N, B, input/counter types, distribution AND ordering, baseline, and timing boundary; mark missing fields. Do not infer superiority from incomparable experiments. Report negative evidence and uncertainty.

Make the PDF self-contained: state the research date, define notation, use selectable text, keep code lines and tables narrow, and provide a numbered bibliography with full titles, authors/organizations, dates, URLs/DOIs, and versions. Include a claim-to-source table for major conclusions, algorithm sketches where useful, and prioritized experiments with predicted observations and falsification criteria. End with a concise implementation handoff: supported conclusions, open questions, and the next decisions this report enables. Proceed using explicit assumptions without waiting for clarification.

Specific research assignment:

Establish a precise implementation and correctness foundation for a high-performance histogram library whose requested target is NVIDIA CUDA C++23. Check current released toolchains and versioned documentation rather than repeating historical statements about C++23 support. Distinguish host C++23, CUDA device language support, host standard-library support, libcu++/CCCL support, and build-system support. Identify preview or rolling-documentation features that are not yet in a released toolkit.

Deliver a compatibility matrix for NVCC, supported host compilers, operating systems, CMake CUDA language modes, CCCL, driver requirements, and target architectures. Verify actual flags and minimum versions. If a combination cannot support the requested target, say so and identify a supported path; keep any lower-standard fallback explicitly separate. Supply minimal compilation probes and commands, but mark them unexecuted unless you really compile them with a named environment.

Research zero-overhead policy/template designs for bin count, sample type, counter type, layout, architecture, and algorithm selection. Distinguish compile-time specialization from runtime flexibility and discuss code size, instruction cache, compilation cost, dispatch complexity, and ABI boundaries. Assess concepts, constexpr, device lambdas, cuda::std facilities, and other language features only when they materially help this implementation.

Investigate compiler effects: optimization and architecture flags, native cubins versus PTX JIT, architecture-specific and family-specific targets, feature guards and portability, register allocation/spills, launch bounds, unrolling, vectorization, aliasing/restrict promises, alignment, link-time optimization where relevant, and inline PTX tradeoffs. Explain how to inspect PTX/SASS and resource reports and verify that a source-level optimization changed generated code as intended. Avoid implying that selecting a newer C++ standard alone accelerates kernels.

Give correctness arguments for the proposed histogram patterns: atomic scope/order, barriers and divergent control flow, warp masks under independent thread scheduling, reuse of shared scratch, local-to-global publication, output initialization, temporary counter overflow, integer size arithmetic, empty inputs, and alignment/tail handling. Address concurrent streams, shared workspace, graph replay, and asynchronous object lifetimes. Explain why volatile is not a synchronization substitute and why casual grid-wide spin barriers can deadlock.

Deliver an API/resource-lifetime sketch, meaningful reference/property tests, overflow-bound derivations, and a targeted Compute Sanitizer plan using tools whose current capabilities are verified. Clarify what sanitizers cannot prove. Recommend a minimal initial build and validation configuration with exact versions/flags or clearly marked choices still needing hardware validation.

Starting points:
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html
- https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/
- https://github.com/NVIDIA/cccl
- https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html
- https://cmake.org/cmake/help/latest/prop_tgt/CUDA_STANDARD.html
```


**10. Performance models, microbenchmarks, and profiling**

```text
Report 10: Performance models, microbenchmarks, and profiling
Suggested PDF filename: 10-performance-models-and-profiling.pdf

I am building a state-of-the-art NVIDIA GPU histogram implementation in CUDA C++23. This report will be exported as a PDF and given to the engineer implementing and benchmarking it. Produce an implementation-oriented research report, not a general CUDA tutorial.

No particular GPU or workload has been selected. Investigate a performance frontier across workloads and supported NVIDIA architectures; do not assume one kernel wins everywhere. Prioritize exact integer counting with device-resident input and dense output, while clearly labeling other contracts. Let N mean input items, B output bins, and K observed distinct keys. Cover current, publicly documented and available hardware/toolchains as of your research date, including relevant Ampere, Ada, Hopper, and Blackwell variants; label older results, previews, and unverified availability. Preserve the CUDA C++23 target and verify relevant language, compiler, and library support rather than assuming it.

Use original papers, author artifacts, NVIDIA documentation, and actual implementation source as evidence. Follow promising references beyond these seed sources. Pin software versions and source commits, distinguish stable releases from development branches, and cite exact sections or code locations. Separate documented facts, published measurements, your deductions, and untested hypotheses. Do not invent benchmarks, citations, compilation results, APIs, or hardware throughput figures. If you cannot run CUDA, mark proposed code and experiments untested.

For consequential performance claims, recover GPU, software version, operation semantics, N, B, input/counter types, distribution AND ordering, baseline, and timing boundary; mark missing fields. Do not infer superiority from incomparable experiments. Report negative evidence and uncertainty.

Make the PDF self-contained: state the research date, define notation, use selectable text, keep code lines and tables narrow, and provide a numbered bibliography with full titles, authors/organizations, dates, URLs/DOIs, and versions. Include a claim-to-source table for major conclusions, algorithm sketches where useful, and prioritized experiments with predicted observations and falsification criteria. End with a concise implementation handoff: supported conclusions, open questions, and the next decisions this report enables. Proceed using explicit assumptions without waiting for clarification.

Specific research assignment:

Develop an evidence-grounded performance model for complete exact CUDA histograms and an experimental method for diagnosing why a kernel misses its plausible limit. Cover both streaming and cache-resident inputs, tiny launch-limited workloads, and bin-count/skew regimes where atomic or instruction throughput matters more than input bandwidth.

Define notation and units carefully. Account for input bytes, output materialization, zeroing, partial histograms, merge traffic, extra scans, bin classification, local aggregation, barriers, instruction work, occupancy, and launch/setup time. Distinguish a lower bound on data movement from an achievable throughput estimate. Use measured sustainable bandwidth and atomic rates where available and treat missing values as parameters. A logical input-bytes/time rate is not necessarily measured DRAM bandwidth.

Model contention at thread, warp, block, and device scales. For independent samples with bin probabilities p_b, derive and qualify the expected number of distinct bins in a warp of W active lanes, E[D] = sum_b(1 - (1 - p_b)^W), and relate it to possible warp-aggregation savings. Explain why correlated or sorted input invalidates the independent-sample assumption. Include hottest-bin behavior, replica count, counter width, and merge costs without pretending a simple model captures every scheduling or cache effect.

Propose isolated microbenchmarks for read throughput, shared/global atomics, key matching/voting, integer bin mapping, zeroing, partial reductions, barriers, and launch overhead. Separate dependent latency chains from independent throughput and prevent dead-code elimination. Specify controlled variables, resource use, cache state, warm-up, timing, and observable outputs. Explain how to relate microbench results back to a complete kernel without adding incompatible peak rates.

Design a Nsight Systems/Nsight Compute workflow. Verify metric names and availability for the target architecture and tool version rather than inventing universal counters. Relate observations to competing hypotheses: memory traffic, hot atomics, shared-bank effects, register spills, occupancy, instruction issue, and synchronization. Explain profiler replay, serialization, cache control, instrumentation overhead, and why profiling runs should be separated from final timing.

Deliver symbolic models with assumptions and worked illustrative cases clearly labeled as estimates, a diagnostic decision tree, a metric-to-hypothesis table, and prioritized experiments. Include model-disagreement checks and ablations. State what improvement would be physically plausible under each bottleneck and what result would falsify the current explanation. Do not use a roofline plot alone as proof of optimality.

Starting points:
- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- https://github.com/NVIDIA/nvbench
```


**11. Benchmark design and a defensible fastest claim**

```text
Report 11: Benchmark design and a defensible fastest claim
Suggested PDF filename: 11-benchmark-and-fastest-claim-protocol.pdf

I am building a state-of-the-art NVIDIA GPU histogram implementation in CUDA C++23. This report will be exported as a PDF and given to the engineer implementing and benchmarking it. Produce an implementation-oriented research report, not a general CUDA tutorial.

No particular GPU or workload has been selected. Investigate a performance frontier across workloads and supported NVIDIA architectures; do not assume one kernel wins everywhere. Prioritize exact integer counting with device-resident input and dense output, while clearly labeling other contracts. Let N mean input items, B output bins, and K observed distinct keys. Cover current, publicly documented and available hardware/toolchains as of your research date, including relevant Ampere, Ada, Hopper, and Blackwell variants; label older results, previews, and unverified availability. Preserve the CUDA C++23 target and verify relevant language, compiler, and library support rather than assuming it.

Use original papers, author artifacts, NVIDIA documentation, and actual implementation source as evidence. Follow promising references beyond these seed sources. Pin software versions and source commits, distinguish stable releases from development branches, and cite exact sections or code locations. Separate documented facts, published measurements, your deductions, and untested hypotheses. Do not invent benchmarks, citations, compilation results, APIs, or hardware throughput figures. If you cannot run CUDA, mark proposed code and experiments untested.

For consequential performance claims, recover GPU, software version, operation semantics, N, B, input/counter types, distribution AND ordering, baseline, and timing boundary; mark missing fields. Do not infer superiority from incomparable experiments. Report negative evidence and uncertainty.

Make the PDF self-contained: state the research date, define notation, use selectable text, keep code lines and tables narrow, and provide a numbered bibliography with full titles, authors/organizations, dates, URLs/DOIs, and versions. Include a claim-to-source table for major conclusions, algorithm sketches where useful, and prioritized experiments with predicted observations and falsification criteria. End with a concise implementation handoff: supported conclusions, open questions, and the next decisions this report enables. Proceed using explicit assumptions without waiting for clarification.

Specific research assignment:

Design a reproducible benchmark and correctness protocol for a new exact CUDA histogram implementation intended to challenge the strongest existing implementations. The output should be concrete enough to implement as a benchmark specification and results schema.

Define separate timing scopes: diagnostic accumulation kernel only; complete device-resident histogram including required initialization, all passes and merges, and online selection; host API latency; and transfer-inclusive application time. Define whether temporary allocation, one-time tuning, graph instantiation, cache preparation, output consumption, and compilation/JIT are one-time or per-call work. Show each algorithm under equivalent accounting. A kernel-only win must not be reported as an end-to-end win.

Construct a staged test matrix covering N from empty and tiny inputs to beyond cache; B from very small to large and non-power-of-two sizes; input and counter widths; alignment/tails; valid and invalid samples where supported; warm/cold output; direct IDs and numeric bin mapping; and representative hardware SKUs. Include uniform, single-hot, few-hot, Zipf-like, clustered, sorted, shuffled, periodic, and boundary-heavy data. Pair permutations with identical counts. Include real datasets with provenance plus seeded synthetic generators. Do not create a Cartesian explosion: identify a compact screening suite and a broader held-out confirmation suite.

Specify strong version-pinned baselines, their correct API usage, resource settings, and tuning opportunities. Explain how to avoid comparing a tuned candidate against an accidentally weak baseline. Record all zeroing, scratch reuse, allocation, sampling, preprocessing, and output-conversion costs. Include memory use and failure behavior as well as time.

Define CUDA-event and wall-clock timing appropriately, warm-up/JIT handling, repeated-call output reset, synchronization, graph replay, sample count selection, randomized/interleaved order, and treatment of outliers. Record GPU, clocks/power/thermal state, driver/toolkit/compiler/library commits, flags, ECC/MIG where relevant, and concurrent activity. Address allocator effects, cache residency, input-buffer rotation, profiler interference, timer resolution, and hardware compression of low-entropy data where applicable. Do not assume cache flushing or fixed clocks are available or inherently neutral.

Require correctness against an independent reference, conservation of accepted counts, boundary tests, overflow cases, and deterministic reproducibility of input generation. Distinguish bitwise requirements for integer counts from tolerance/reproducibility contracts for weighted floating point.

Deliver pseudocode for a timing harness, a machine-readable result schema, representative generator specifications, uncertainty reporting, plots/tables, and a publication checklist. Explain paired comparisons, confidence intervals, aggregation of speedups, and reporting losses alongside wins. Define a narrowly supportable fastest claim tied to tested contracts, competitors, hardware, versions, and date; identify exactly what evidence would invalidate it.

Starting points:
- https://github.com/NVIDIA/nvbench
- https://github.com/NVIDIA/cccl
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
```


**12. Batching, fusion, streaming, and multiple GPUs**

```text
Report 12: Batching, fusion, streaming, and multiple GPUs
Suggested PDF filename: 12-batching-fusion-streaming-and-multi-gpu.pdf

I am building a state-of-the-art NVIDIA GPU histogram implementation in CUDA C++23. This report will be exported as a PDF and given to the engineer implementing and benchmarking it. Produce an implementation-oriented research report, not a general CUDA tutorial.

No particular GPU or workload has been selected. Investigate a performance frontier across workloads and supported NVIDIA architectures; do not assume one kernel wins everywhere. Prioritize exact integer counting with device-resident input and dense output, while clearly labeling other contracts. Let N mean input items, B output bins, and K observed distinct keys. Cover current, publicly documented and available hardware/toolchains as of your research date, including relevant Ampere, Ada, Hopper, and Blackwell variants; label older results, previews, and unverified availability. Preserve the CUDA C++23 target and verify relevant language, compiler, and library support rather than assuming it.

Use original papers, author artifacts, NVIDIA documentation, and actual implementation source as evidence. Follow promising references beyond these seed sources. Pin software versions and source commits, distinguish stable releases from development branches, and cite exact sections or code locations. Separate documented facts, published measurements, your deductions, and untested hypotheses. Do not invent benchmarks, citations, compilation results, APIs, or hardware throughput figures. If you cannot run CUDA, mark proposed code and experiments untested.

For consequential performance claims, recover GPU, software version, operation semantics, N, B, input/counter types, distribution AND ordering, baseline, and timing boundary; mark missing fields. Do not infer superiority from incomparable experiments. Report negative evidence and uncertainty.

Make the PDF self-contained: state the research date, define notation, use selectable text, keep code lines and tables narrow, and provide a numbered bibliography with full titles, authors/organizations, dates, URLs/DOIs, and versions. Include a claim-to-source table for major conclusions, algorithm sketches where useful, and prioritized experiments with predicted observations and falsification criteria. End with a concise implementation handoff: supported conclusions, open questions, and the next decisions this report enables. Proceed using explicit assumptions without waiting for clarification.

Specific research assignment:

Research systems-level ways to accelerate histogram workloads in CUDA C++23, while preserving a separate comparison for a standalone device-resident histogram. Identify which improvements reduce overhead or data movement without changing the histogram algorithm, and which require application-specific contracts.

For many small histograms, study one warp/block per histogram, grouped batching, variable segment lengths, occupancy, launch amortization, persistent workers, CUDA Graphs, stream concurrency, and layout transformations. Analyze aggregate throughput and per-request latency separately. Include graph creation/update/replay, scratch initialization, stream-ordered allocation, and small batches where setup cannot be amortized. Compare custom batching with the best applicable existing segmented or library composition.

Investigate fusion with producers and consumers: decoding, filtering, transforms, bin mapping, image processing/equalization, scans/CDFs, and downstream statistics. Explain when avoiding an intermediate buffer is beneficial, and when fusion increases register pressure, loses parallelism, or prevents reuse. Distinguish computing a complete dense histogram from computing only a statistic that does not require that output. Account for materialization if the API requires it.

Study chunked streaming and datasets exceeding device memory, pinned host memory, transfer/compute overlap, double buffering, and memory-capacity limits. Compare already compressed or sorted inputs with paying to compress or sort them; include decompression and exactness constraints. Discuss temporal reuse across calls only when input changes are known and the cost of obtaining that knowledge is charged.

For multiple GPUs, separate initially distributed input from input initially resident on one device or host. Compare input sharding plus histogram reduction, bin sharding and routing, peer access, NCCL or equivalent collectives, NVLink versus PCIe topology, and inter-node communication where relevant. Include 64-bit counts, skew/load imbalance, replication, synchronization, and the output's required location. Do not compare multiple GPUs against one GPU without stating resource differences and reporting scaling efficiency.

Deliver a regime table of application-level opportunities, complete dataflow sketches, latency/throughput and communication models, and a prioritized integration plan. Identify cases in which the standalone fastest kernel is irrelevant because transfer, launching, batching, or downstream work dominates. Provide experiments that separate algorithmic, fusion, and systems gains, with both ideal steady-state and realistic finite-batch accounting.

Starting points:
- https://docs.nvidia.com/cuda/cuda-programming-guide/
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html
- https://docs.nvidia.com/nsight-systems/UserGuide/index.html
```


**13. Synthesis, novelty checks, and an implementation roadmap**

```text
Report 13: Synthesis, novelty checks, and an implementation roadmap
Suggested PDF filename: 13-evidence-synthesis-and-experiment-roadmap.pdf

I am building a state-of-the-art NVIDIA GPU histogram implementation in CUDA C++23. This report will be exported as a PDF and given to the engineer implementing and benchmarking it. Produce an implementation-oriented research report, not a general CUDA tutorial.

No particular GPU or workload has been selected. Investigate a performance frontier across workloads and supported NVIDIA architectures; do not assume one kernel wins everywhere. Prioritize exact integer counting with device-resident input and dense output, while clearly labeling other contracts. Let N mean input items, B output bins, and K observed distinct keys. Cover current, publicly documented and available hardware/toolchains as of your research date, including relevant Ampere, Ada, Hopper, and Blackwell variants; label older results, previews, and unverified availability. Preserve the CUDA C++23 target and verify relevant language, compiler, and library support rather than assuming it.

Use original papers, author artifacts, NVIDIA documentation, and actual implementation source as evidence. Follow promising references beyond these seed sources. Pin software versions and source commits, distinguish stable releases from development branches, and cite exact sections or code locations. Separate documented facts, published measurements, your deductions, and untested hypotheses. Do not invent benchmarks, citations, compilation results, APIs, or hardware throughput figures. If you cannot run CUDA, mark proposed code and experiments untested.

For consequential performance claims, recover GPU, software version, operation semantics, N, B, input/counter types, distribution AND ordering, baseline, and timing boundary; mark missing fields. Do not infer superiority from incomparable experiments. Report negative evidence and uncertainty.

Make the PDF self-contained: state the research date, define notation, use selectable text, keep code lines and tables narrow, and provide a numbered bibliography with full titles, authors/organizations, dates, URLs/DOIs, and versions. Include a claim-to-source table for major conclusions, algorithm sketches where useful, and prioritized experiments with predicted observations and falsification criteria. End with a concise implementation handoff: supported conclusions, open questions, and the next decisions this report enables. Proceed using explicit assumptions without waiting for clarification.

Specific research assignment:

Use the attached GPU histogram research PDFs as a starting corpus and verify consequential claims against original sources on the web. This is the final synthesis report, to run after the independent reports. Inventory every attachment by report ID/title and research date. Identify missing or unreadable reports and state how that limits the synthesis; do not invent their contents. Treat the reports as evidence summaries, not authoritative proof.

Build a unified map of problem contracts, algorithm families, architecture requirements, production baselines, and measurement quality. Resolve contradictions by comparing semantics, hardware, versions, distributions/order, and timing boundaries. Revisit promising claims whose evidence is old, weak, incomparable, or contradicted. Distinguish what is established from what must be measured on our eventual target hardware.

Recommend a small initial kernel portfolio and dispatch strategy for exact unweighted dense counting, with an extension path for large-bin/sparse, weighted, and structured workloads. Since the final GPU and workload priorities are unspecified, give conditional choices and a short list of decisions needed before hardware-specific implementation. Select an initial workload based on technical opportunity and reproducibility rather than claiming a universally optimal target.

Propose and rank roughly 6-10 concrete optimization hypotheses. For each, give the mechanism, applicable regime, closest prior art, expected bottleneck removed, added costs/resource risks, correctness obligations, simplest prototype, strongest baseline, controlled ablation, success criterion, and falsification/abandonment criterion. Any proposed gain must be labeled a model-based estimate unless measured. Search specifically for prior implementations of each proposed combination; absence from your search is not proof of novelty.

Critically assess fashionable features such as clusters, asynchronous staging, persistent execution, aggressive specialization, and tensor/matrix units if proposed. For unconventional reformulations, account for one-hot expansion, data conversion, precision, extra traffic, and result materialization. Include low-complexity improvements and cases where the correct decision is to retain a library baseline.

Deliver a staged implementation plan: pinned CUDA C++23 toolchain; API/correctness contract; baseline and benchmark harness; smallest useful prototypes; profiling and ablations; selector/tuning validation; and final comparison. Specify decision gates and a compact first experiment batch with explicit matrix cells and required measurements. Separate experiments runnable without a GPU, those needing one target GPU, and claims needing multiple architectures.

Finish with an implementation handoff that includes pseudocode/interfaces, resource formulas, supported feature boundaries, a result schema, unresolved questions, and a scoped wording template for a future performance claim. Cite both report page/section and the original source for consequential conclusions. Keep the roadmap actionable even if no speculative hypothesis succeeds.
```
