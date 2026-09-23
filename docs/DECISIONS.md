# Greenfield contracts and decisions

Target: RTX A5000 Laptop (SM86, 48 SMs), CUDA 13.4.59, C++23. The accepted
design uses direct CUDA control flow, compile-time policies and pure helpers.
All runtime production/test calculation is on GPU; CUDA writes/launches are
effects. The original tree and the unfinished functional draft are external
evidence, never production dependencies.

## Foundation, selected before implementation

Choose plain capacity-bearing device views, one supplied arena and one completion
record. This removes owning host containers, virtual dispatch and recursive effect
wrappers. Checked products/additions precede partitioning/dereference. Arenas
start 16-byte aligned; algorithm inputs and output regions are disjoint, on the
same device, and alive through checked completion. Pointer liveness is a caller
runtime precondition, not discoverable by dereferencing arbitrary pointers.

Each device API is submitted by one coordinator thread. Inputs and supplied
descriptors must remain unchanged through child completion; output and scratch
are exclusive. Callers initialize Status to zero. Child kernels use device-null
stream ordering; result consumers use tail launches. No child receives parent
local/shared addresses. Runtime failures and semantic status must both pass.
Independent operations use separate arenas/status. `finish` appends a completion
tail without pretending that parent submission means child completion.

CDP2 is the first executor because it provides legal GPU-only dynamic scheduling.
Its real launch stack/spills and child/tail overhead are included in observations.
Cooperative persistent execution and device graphs are later separate experiments,
not dormant alternate backends. References: NVIDIA CUDA Programming Guide,
dynamic-parallelism and device-callable-apis appendices (accessed 2026-09-23).

Validation: independent GPU checks for extent overflow, alignment/capacity,
empty ranges, ordered child completion, failure status and guarded reuse. Inspect
PTX/SASS/resource usage, then sanitizer on actual device execution. Measure full
operation spans including coordinator, clearing and tail completion; raw device
timestamp protocol is versioned and cannot be relabelled CUDA-event milliseconds.

## Source and numerical acceptance

Freeze capabilities against the baseline before comparing line counts. Report
formatted nonblank/noncomment lines and tokens for production, tests and tooling
separately, counting registries/generators. Preserve import/export formats, exact
integer/encoding outputs, documented numerical schedules and all quality metrics.
No universal speed claim, hidden CPU fallback, numerical tolerance waiver or
unexercised algorithm instantiation is permitted. Baseline FP64 histogram atomics
are unordered: repeated-baseline variability does not pass a failed quality gate.
