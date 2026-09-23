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
