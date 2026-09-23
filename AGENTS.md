# GPU histogram and booster work

The user's objective is the fastest GPU implementation, including large-scale
multi-output learning. Algorithm selection must precede implementation.

Before writing or replacing an algorithm:

1. Define its exact input/output, numerical, memory and device contracts.
2. Analyze the fastest known relevant GPU approaches. Use primary research,
   current implementation sources, and our measured evidence. Check hardware and
   workload applicability; a published win elsewhere is not a local ranking.
3. Compare algorithmic work, memory traffic/layout, scratch capacity, atomics,
   synchronization, launch cost, batching and fusion. Consider removing work or
   changing the larger architecture before optimizing a local kernel.
4. Record the candidate choice and a fair experiment before implementing it.
5. Measure complete operations and end-to-end behavior, use Nsight to explain
   results, and validate correctness plus every applicable quality metric.
   Preserve raw observations and failures. Do not label unmeasured code fastest.

Do not introduce CPU training/preprocessing stages as implementation shortcuts.
The target is GPU-resident binning, training state, statistics, split decisions,
tree construction and prediction. Host setup, transfers, model export and explicit
CPU validation references are distinct from the production compute path. The
current hybrid trainer is a frozen comparison baseline, not the final design.

Keep the measured counting kernels, policies and defaults intact unless a change
is specifically supported by the active task and performance evidence. NVIDIA
algorithm implementations, including CUB, are benchmark references only; do not
add them to production algorithms or fallback paths. CUDA runtime, Nsight and
the explicitly requested NVTX instrumentation are permitted infrastructure.

Use the newest C++ language and functional library features actually supported
by the CUDA toolchain and the relevant host/device compilation path. C++23 is
the installed nvcc 13.4's highest advertised dialect flag, not a ceiling on
individual features: adopt useful supported C++26 features/backports too.
Verify compiler and library support for each relevant compilation path before
adopting a newer feature or dialect.

The requested design is functional throughout, not a mixture of paradigms.
Use immutable values, pure transformations, function composition and concise
template metaprogramming. Describe GPU effects explicitly; do not call CUDA
memory writes, allocation or kernel execution pure, or claim that pure helper
functions alone complete the migration. Prefer small abstractions whose
generated work and performance can be verified. Remove proven dead code while
preserving deliberately exercised comparisons and public contracts.

Run GPU workloads serially when collecting evidence. Do not use
profiler timings to rank uninstrumented implementations. Separate exact arithmetic
or semantic preservation from approximations, altered model architectures and
floating-point reorderings. A failed zero-allowance gate stays failed.

Current analysis and measurements:
- `training/ALGORITHM_DECISIONS.md`
- `results/booster-trainer-20260922/REPORT.md`
- `results/booster-trainer-20260922/quality-audit.md`
