# Fresh counting backend: contract and selected implementation

Recorded before implementation, 2026-09-23. Preserve the archived counting
algorithm choices, 16 policy tuples and exact measured default keys, while
writing fresh data-oriented kernels. The archive is evidence only; no legacy
source is included, linked or copied into production. New generated work and
performance remain unmeasured. All planning, selection, fixtures and checks are
GPU computations. Host bootstrap performs only CUDA resource setup/submission.

Input is a naturally aligned resident u8 or u32 direct-ID array, IDs in [0,bins).
The caller supplies that ID precondition. Dense u32/u64 output is overwritten,
including zeros and empty input; all input/output/workspace regions are disjoint
and live through checked completion. Bins are 1..INT_MAX-1, with u8 bins<=256.
Input byte extent must fit PTRDIFF_MAX; u32 output requires size<=UINT32_MAX.
The caller initializes Status to zero. Submission and semantic errors are
separate from completion; the device API appends the shared completion tail.

The selected algorithms are global atomic, warp-matched global atomic, shared
atomic, shared per-thread RLE, shared warp matching, shared partials plus ordered
integer reduction, bit-plane ballots, narrow global scratch plus widening,
96-KiB shared prefix plus global overflow, and repeated narrow global windows.
Automatic selects only the preserved custom algorithms. CUB remains a benchmark
reference, never a fallback. [NVIDIA's histogram study](https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/)
motivates shared/two-phase counting and distribution-sensitive atomics; Maxwell
measurements do not rank this SM86 implementation. The local evidence is archived
`src/defaults.cpp`, `src/config.cpp`, histogram/narrow/window/overflow/bitplane
sources and their recorded results. Preserve their measured keys without claiming
that the fresh compiled kernels inherit the old timings.

[NVIDIA's grouped-atomic analysis](https://developer.nvidia.com/blog/voting-and-shuffling-optimize-atomic-operations/)
supports matching equal keys before atomics but also demonstrates sensitivity to
ordering and key distribution. The current [CCCL histogram interface](https://nvidia.github.io/cccl/unstable/cub/api/index.html)
is a reference-comparison candidate. Neither supplies a transferable SM86 winner.
Sorting then run reduction, sparse hashing, and repeated full-input window passes
are not selected defaults: their additional passes/state must repay their cost
against direct dense counting on the actual shape and ordering.

| Approach | Work and memory/effect cost |
| --- | --- |
| Global/warp | One input pass and output clear; per-item or per-distinct-warp-key global atomic |
| Shared variants | One input pass, block initialization/barriers and replica reduction; shared atomics; nonzero block totals update output |
| Shared partial | Blocks*bins*local-width scratch, every partial overwritten, second reduction launch; no output clear |
| Bit-plane | One input pass; predicate ballots, private warp counters and block reduction; capacity is next power of two through 256 |
| Narrow global | bins*4 scratch, clear/count/widen; total size must fit u32 |
| Shared overflow | First 24576 bins in 96 KiB shared, remaining bins use u64 global atomics; no application scratch |
| Global window | min(bins,window_bins)*4 scratch; ceil(bins/window_bins) full input passes, each clear/count/widen |

For u64 output with u32 locals, shared algorithms prove the largest CTA-owned
sample count `(size/stride)*tile + min(size%stride,tile)` fits UINT32_MAX. Global,
warp and window narrow paths instead require the full size<=UINT32_MAX. Replicas
do not relax the CTA bound. All partial/reduction products are checked before
arena use. Empty input needs no scratch and still overwrites output.

Policies 0..5 retain scalar tile ownership; 6/8 use full-tile loading; vector4
policies pack four adjacent samples per thread when fully aligned, otherwise
fall back to scalar loads with exactly the same CTA ownership. Tails are masked.
RLE sample order follows the selected load policy. Bit-plane vector4 is the
preserved alias for full_tile, not a vector-load claim. Policies 14/15 declare 96 KiB
and require explicit CUDA function-attribute bootstrap, even if a specific
histogram uses less. Unsupported capacities fail rather than silently fallback.
Explicit bit-plane calls also check their actual shared allocation against the
supplied device capacity. Invalid bit-plane policies 14/15 have no kernel instance.

Measured default identity is exactly RTX A5000 Laptop/SM86/48SMs. The GPU selector
uses supplied identity/resource metadata, never queries data or cache residency.
Other shapes retain resource-safe archived heuristics, narrow shared locals when
provable and native/global fallback. Stream/graph and warm/cold are declared keys;
changing workload requires resolution again. Explicit policies remain unchanged.

The first runtime executor is CDP2. Algorithms are statically specialized by input,
output/local width and policy, with compact uniform update selection inside the
kernel. This reduces repeated kernel definitions/instantiations; uniform branches
are still real generated work requiring inspection and measurement. CUDA 96 KiB
attribute setup is runtime infrastructure, not a host algorithm selector.

## Graph contract

[CUDA's graph requirements](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
for device-launchable graphs forbid CDP inside their kernel nodes. Consequently
the device count coordinator and completion tail are not legal nodes in a
device-launchable graph. A host-launched graph may contain the CDP bootstrap;
the runtime documents restrictions on updating an initially non-CDP node into
a CDP node. This minimal graph-of-CDP arrangement can exercise replay with changed
resident input, but is a different launch topology from frozen direct-leaf graphs.
No device-graph compatibility or inherited graph performance is implied.
Graph construction/submission is CUDA bootstrap infrastructure; all case/input
updates and correctness decisions remain GPU work. Direct-leaf device graphs
require a separately selected GPU-produced plan and runtime marshalling contract.

## Predeclared acceptance and experiment

GPU checks cover every supported algorithm/type/policy family, all 16 policies,
scalar/full/vector tails, natural-but-vector-misalignment, bins1/8/31/32/33/128/
255/256 and large shared/window domains, empty input, constant/ascending/shuffled
IDs, guarded overwrite/reuse, exact capacity failures, and narrow count bounds.
Independent per-bin GPU reference scans or closed-form distributions never call
the production counter. GPU metadata-only checks test overflowing extents without
dereferencing fictitious allocation capacities. The exact 11 default keys and
near-miss keys are checked on GPU. No test computation occurs on the host.

After root authorizes serial GPU execution, require zero integer differences,
memory/init/synchronization diagnostics, and changed-input graph replay evidence.
Inspect vector loads, match/ballot participation, local/shared resources and the
96KiB attribute route. Preserve failures, unsupported profiler results and all
raw observations. Compile-only success is not runtime correctness.

The compact bit-plane candidate computes masks from runtime capacity and retains
up to eight lane-owned totals. Initial compiler output shows local stack storage
for those arrays. This is a concrete resource difference from the archived
capacity-specialized formulation; inspect local traffic and complete-operation
timing before selecting a replacement formulation or making performance claims.

For ranking, freeze models of input generation and all source/binary/input
identities. Measure complete clear+count+reduce/widen+checked completion, including
CDP overhead, with 3 warmups and 15 alternating pairs; keep sample order and raw
device timestamps with their explicit timing contract. Include exact default
shapes, tails, empty/small calls, skew/ordering controls and shared/window boundary
bins. GPU statistical analysis uses paired ratios and casewise 95% intervals;
inconclusive results remain inconclusive. Profile afterward to explain work and
traffic. Default promotion requires separate correctness/performance evidence.
