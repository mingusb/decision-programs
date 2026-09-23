# Greenfield GPU histogram and booster

Implement the complete accepted hybrid plan: counting, GPU feature fitting,
multi-output learning, inference, formats, instrumentation, tests and measurement.
Use data-oriented CUDA, concise compile-time policies and small pure helpers.
Ordinary loops and local mutation are explicitly authorized. Do not build a
general operation DSL, virtual hierarchy, or functional effect framework.

Before each algorithm, document its I/O, numerical, memory and device contracts,
current primary evidence, chosen algorithm and fair complete-operation experiment.
Compare work, traffic/layout, scratch, atomics, synchronization and launches.
Do not label unmeasured code fastest or waive a failed zero-allowance quality gate.

All production and test computation runs on GPU: planning, validation, fixtures,
oracles, metrics, decisions and benchmark statistics included. Host code performs
CUDA bootstrap/runtime completion and external byte transport only. Compiler,
Git, profiler launch and artifact preservation are development infrastructure.
Data remains resident between stages. Supplied arenas are capacity-checked;
pending readers/writers must complete before their storage is reused.

Write fresh source. Do not include, link or forward production operations into
the old implementation or unfinished functional draft. Frozen evidence lives in
/home/b/gpu_histogram-archive-20260923 and
/home/b/gpu_histogram-cutover-20260923/functional-draft-before-hybrid.
Preserve count policies/defaults and numerical schedules until a separately
measured replacement passes its full correctness/performance/quality gates.
NVIDIA algorithms (including CUB and Thrust) are isolated benchmark references
only. CUDA runtime, fundamental libcu++ facilities, Nsight and NVTX are allowed.

Use C++23 plus useful verified C++26 features/backports. Check actual device
support and generated code. Every abstraction needs exercised callers; retain
experimental policies only with deliberate comparison coverage. No dead code.
Measure formatted source lines and tokens without hiding code in generators.

Only the root agent runs GPU workloads; run them serially. Agents may compile
their owned files without GPU execution, coordinating build resource usage.
Preserve raw observations/failures and distinguish compiled, tested and measured.
Uninstrumented complete-operation timings rank; profiler results explain.
