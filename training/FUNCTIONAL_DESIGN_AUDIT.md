# Functional C++23 and dead-code audit

Recorded 2026-09-23 before the refactors selected below. This is a bounded
source audit, not proof that the whole repository is pure or contains no dead
code. The requested final state covers the whole implementation; the current
repository does not satisfy it.

## Authoritative scope and current state

Read `AGENTS.md`, root/training CMake targets, current production headers and
sources, including `training/src/batch_training.inc` (included from
`booster.cpp`, not a standalone CMake source), the active D2 design, and current
prediction/quantization tests. Archived result sources and isolated NVIDIA
benchmark references are evidence/reference inputs, not deletion candidates.
No builds or GPU jobs were run for this audit.

Both CMake projects require C++23 for host/device compilation. The current
`build/final-status-20260923/CMakeFiles/ghb.dir/flags.make` has `-std=c++23` for
both; its instrumentation target does too. The frozen count build's
`build/profiling-count/build.ninja` has `-std=gnu++23` for host code and
`-std=c++23` for CUDA. Root CMake does not disable host GNU extensions, while
training CMake does. Language mode alone proves neither purity nor use of every
useful modern feature. NVIDIA's current
[nvcc reference](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#std-c-03-c-11-c-14-c-17-c-20-c-23-std)
explicitly supports C++23.

| Current component | Evidence | Functional status |
|---|---|---|
| Count policy calculations | `src/config.cpp`, `src/defaults.cpp` take configuration/device-property values and return decisions | Mostly value-oriented, with local mutation; measured count code remains frozen |
| Scalar higher-order formulas and split comparisons | `higher_order_math.cuh`, scalar `add/subtract/leaf/benefit/better` in `split_search.cu` | Many deterministic value transformations already exist; floating arithmetic environment and exact expression ordering remain part of their contract |
| Slab layout | `detail/prediction_layout.hpp` | Deterministic checked arithmetic, currently expressed through mutation of a local result/captured lambda |
| Quantization | `quantize.cu` | Mutable ownership, runtime allocation/copy/launch/wait, status atomics, imperative staging; not pure |
| Training/prediction orchestration | `booster.cpp`, `batch_training.inc` | Mutable buffers/model assembly, instrumentation and runtime effects; not pure |
| Device kernels | `prediction.cu`, `kernels.cu`, histograms/resident code | Memory writes, collectives and atomics are effects; `const` input pointers do not make the kernel pure |
| Resource/instrumentation interfaces | `QuantizedData`, `Device`, `Stream`, `Recorder` | RAII improves lifetime safety but constructors/destructors still have effects |
| Explicit CPU references, benchmarks and tools | tests, benchmark targets, Python tools | Validation, clocks, file/process I/O and mutable generators remain; they are not production CPU preprocessing, and they are not a pure-code completion proof |

Calling an imperative routine through a lambda, naming it `map`, adding `const`
to a pointer, or making an owning type move-only does not establish functional
semantics. The public `const Model::predict_gpu` still interacts with CUDA state,
free-memory queries, errors and timing. Weighted FP64 atomic reductions also do
not establish a bitwise deterministic mapping merely because inputs are fixed.

## Relevant primary approaches and their actual limits

[Futhark's ownership/consumption rules](https://futhark.readthedocs.io/en/latest/language-reference.html#in-place-updates)
show how functional array semantics can permit efficient storage reuse: the old
array and all its aliases become unusable after consumption. C++ move semantics
alone do not enforce that property, particularly with exposed raw device views.
Adopting that reasoning requires explicit alias/lifetime invariants, not a new
name for the existing mutable buffers.

[Futhark's backend architecture](https://futhark-lang.org/blog/2025-03-04-adding-a-new-backend.html)
separates value-oriented intermediate programs from imperative code generation
and runtime operations. This supports a design with immutable computation and
typed descriptions of effects, then a narrowly specified execution boundary.
It does not prove that our existing CUDA C++ implementation is pure, and it
does not require changing our implementation language to Futhark.

[Its current scan/scatter fusion analysis](https://futhark-lang.org/blog/2026-03-24-scan-scatter-fusion.html)
also shows why concise functional expressions require performance scrutiny:
fusion can remove intermediate arrays, but register pressure and code generation
still matter, and the author reports cases behind hand-written CUDA. There is
no primary evidence here for a universally fastest functional GPU abstraction.

[CUDA asynchronous execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
requires an actual executor to schedule copies/kernels and preserve ordering
and lifetimes. A pure description of those actions is distinct from executing
them. Future migration should represent dependencies and owned state transitions
as values, with the unavoidable foreign-runtime effects stated explicitly.
The executor must not conceal new host algorithms, copies or waits.

For this project, choose small statically resolved value transformations and
template specialization before a general expression language. Prove elimination
of abstraction cost from emitted code and complete-operation measurements.
Do not materialize intermediate GPU arrays merely to obtain functional-looking
source, or replace an O(F) scan with O(F²) repeated prefix recomputation.

## Selected first functional refactor: scalar slab layout

The source/binary baseline is frozen at
`results/final-status-20260923/source-snapshot/manifest.json`
(SHA-256 `afb0b32201aea66728500c06705462de384df1619f82bf9f75102b66dff91544`).
Before editing, `prediction_layout.hpp` hashes to
`fd46427e01fb12f28e8d3ee00302478221973fa1cea36d713da0fc7b194e0db3`.

Select the fixed four-region slab calculation for the first bounded migration.
It is host setup arithmetic, not CPU training/preprocessing. Inputs are the
four counts and compile-time node/descriptor ABI sizes. Output is the existing
`PredictionSlabLayout`, or the same overflow exception at the same check.
Alignment remains 256; zero-sized regions still report zero offset/bytes and
do not advance the cursor. The layout's public fields and ABI remain unchanged.

Replace the capture that mutates `result.device_bytes` with a small value-return
placement function: `(cursor, count, width) -> (region, next_cursor)`. Bind base,
nodes, descriptors and offsets to successive immutable values, then construct
the final aggregate once. Keep size checks in exactly this order: output-offset
count; four region products/alignment/extents; prediction bytes; reserved bytes;
host block rounding; host bytes. Preserve the existing names in exception text.
The checked public functions remain partial value functions: overflow still
throws through their existing interface. This step does not claim that C++
exception execution or the rest of the program has become a total pure function.

| Alternative | Work/traffic/capacity/synchronization | Selection |
|---|---|---|
| Current captured mutable cursor | Four region calculations, constant scalar storage, no GPU work | Frozen control |
| Immutable placement results and aggregate initialization | Same arithmetic/check order, constant scalar values, no arrays/heap allocation on success, no atomics/launches/waits | Selected; directly expresses value flow without a framework |
| General region list with fold/tuple metaprogramming | Can generalize arbitrary region counts, but adds representation/template complexity for four fixed ABI fields | Defer until a real second use demonstrates value |
| Heap-backed persistent layout or effect graph | Adds allocation/indirection without removing work | Reject for this scalar task |

Acceptance: root builds in fresh C++23 directories; the existing 118 CPU layout
checks must pass, including size/overflow/empty-region/budget behavior. Review
the check order and output ABI, not only successful numbers. Compare generated
device code/constants against the frozen D2 baseline (expected 314 unchanged
sections), and run the existing prediction/quantization correctness gates for
the combined build. This is a source-design change, not a speedup claim. If a
host-code difference could affect complete prediction cost, include it in the
paired unprofiled refactor comparison; profiler timing is not the ranking.

Implementation now uses immutable `Placement` results, const scalar bindings,
C++20 aggregate designated initialization and `constexpr` checked arithmetic
under the C++23 build. There is no reason to add `if consteval`, ranges or a
general fold to this fixed sequence. Successful expressions can be constant
evaluated; overflow retains runtime exceptions and is not a valid constant
expression. A compile-time representative layout assertion is an additional
useful compiler gate. Compiler acceptance remains pending root's fresh build.
Static review confirms the old/new check order and public field order match.
The header changes from 74 to 77 lines: the main calculation is shorter, while
the independently pure placement helper makes the state flow explicit.
Post-edit SHA-256:
`2baf12e950372388698048631243b4d69192fc07cd0e9253401ea5cebc81cdd1`.

## Verified dead state selected for removal

`training/src/quantize.cu` has a private anonymous-namespace `Layout::digits`
member initialized to zero and assigned from the `layout(..., digits, ...)`
parameter. The member is never read. The live `digits` parameter is used to
compute `histogram_length`, and `fit_quantize` uses its local `digits` to choose
layout; radix kernels instead use their live `Bits` template parameter. Those
uses must stay.

Evidence: `rg -n '\bLayout\b|\.digits\b|\bdigits\b' training/src/quantize.cu`
shows the declaration and assignment are the only member references. Inspecting
the complete translation unit confirms every `Layout` consumer (`layout`,
`choose_layout`, `scan`, `sort_keys`, fit and both encoding policies); the type
is neither a kernel argument nor serialized/copied by representation, and its
size/offsets are not externally observed. `training/CMakeLists.txt` compiles
this translation unit into `ghb`; no header exposes `Layout`. No textual
`#include` inserts another body into this translation unit. The candidate is
private dead state, not an unused public API or an untested policy.

Remove only that member and its assignment. Pre-edit `quantize.cu` SHA-256 is
`a6c5f037d8aae554e23f519233cabd2026032b2873ec5a13c738f56d1ae95480`, already in
the frozen snapshot. Device layout payload offsets must remain unchanged: they
are explicitly calculated scalar values, not offsets into this host struct.
Compiler comparison and existing exact quantization/prediction checks are the
gate. No performance improvement is assumed; the optimizer may already remove
the unused store. No other private function is proven dead by this audit.

The selected removal is now implemented: exactly two source lines changed,
with no other quantizer edits. Source line count stays 676 and post-edit
SHA-256 is `462e1332bff2977a58e97d9d16d062adff77c4e3ca189f4a63084ac585d70d36`.
Build/runtime/device-code validation is pending root's checks.

## Next architecture work and incomplete requirements

The two encoding bodies currently duplicate orchestration. A
`template<EncodingPolicy>` implementation with `if constexpr` could remove
duplication and retain two specialized bodies, but an ordinary imperative
template would not meet the user's strengthened all-functional requirement.
Do not present that alone as the next completed functional migration.

Instead design one immutable metadata result containing types, offsets, maximum
bins and policy-appropriate length storage, then a compact statically typed
description of tile uploads/encoding/checkpoints. It must distinguish the
per-tile K-entry scratch lengths from final-status F-entry immutable lengths,
preserve the current allocation count/order and declaration-before-drain
lifetime, and avoid a second F traversal or eager descriptor list. A mutable
capture that fills unrelated result objects is not a pure factory. The
per-tile scratch update is a real effect to model, not hide behind `const`.

Before implementing that migration, establish an explicit representation for
fallible actions, owned resources, stream order and cleanup, with proof that
composition adds no allocations, copies, dynamic dispatch, kernel launches or
waits. Existing D2/error-precedence contracts remain independently observable.
Compare frozen-D2 and refactored complete calls for each policy, not D2 versus
per-tile as a substitute for measuring refactor cost.

Remaining work includes training/prediction orchestration, kernel memory effects,
resource ownership, instrumentation, public mutability, tests/tooling, full
private reachability and generated-code audits. Zero compiler warnings and one
dead-member removal do not prove global dead-code elimination. Historical
source snapshots, alternative benchmarked policies, optional instrumented
targets, and explicit CPU validation are intentionally retained while auditing
their actual reachability. The whole-codebase functional objective stays open.
