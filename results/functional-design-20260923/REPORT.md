# First functional-design migration

The slab layout now composes immutable scalar placement values and constructs
its result once. The checked helpers and layout function are `constexpr`.
The private unread `Layout::digits` member and assignment were removed from
quantization. No counting kernels, policy defaults or numerical expressions
were changed. Pre-code analysis and the incomplete whole-codebase migration
are recorded in `training/FUNCTIONAL_DESIGN_AUDIT.md`.

Fresh build: `build/functional-design-20260923`, Release, CUDA/host C++23,
SM86, compiler resource diagnostics enabled and device time traces disabled.

- All 20 existing CTests pass serially, including all 118 CPU layout checks.
- A compile-only `static_assert` probe verifies that the layout and budget
  calculation can actually execute at compile time; `constexpr-probe.json`.
- All 314 compiled device code/constant sections are byte-identical to the
  frozen pre-refactor D2 library after unique full-demangled-name mapping;
  `runtime-mapped-sections.json`.

The refactor preserves check order, overflow error text, zero-sized regions,
256-byte alignment and public layout fields. It introduces no heap storage on
the success path, GPU launch, transfer or device scratch. This is a verified
functional-design improvement, not a measured speedup. Host performance
equivalence has not yet been established by paired timing.

The project is not yet entirely functional and this bounded audit does not
prove absence of all dead code. CUDA effects, mutable training/prediction
orchestration and unordered floating-point reductions remain explicit work.
Replacing that work with new names or pure wrappers would not satisfy the goal.

`AGENTS.md` now records the user's requirements for functional design,
concise template composition, dead-code removal and the newest supported
features. C++23 is the highest advertised installed nvcc dialect flag; this
does not exclude individually supported C++26 library backports. The primary
NVIDIA guide explicitly documents such backports in §5.3.6:
https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#cuda-c-standard-library
Separate host/device compile-only feature evidence is in `toolchain-features/`.
