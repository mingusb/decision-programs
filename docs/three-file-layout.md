# Three maintained code files

Selected before implementation, 2026-09-23. The user explicitly requests all
implementation, tests, benchmarks and development tools in one C++ header, one
C++ source and one CUDA source. CMake build metadata, documentation and captured
observations remain separate. The pre-consolidation checkpoint is commit
`1263890`; all eleven GPU suites passed before this change.

This change preserves the algorithms, mathematical schedules, launch geometry,
counting defaults, file formats, capacity checks and GPU-only compute contracts.
It consolidates their source and ports the external development tools to C++.
No character-count saving or performance improvement is assumed from merging
translation units. The 75% character reduction remains an unachieved target.

`gpu_histogram.hpp` holds shared CUDA declarations, types and small inline
helpers, plus the host bootstrap interface. `gpu_histogram.cu` holds production
implementations and individually selected GPU tests/benchmarks. The production
sections compile once into a shared static library, including a dedicated leaf
object built from the same CUDA file. Each executable compiles
only its own fixture section from the same CUDA source; unrelated device
fixtures and benchmark references do not enter production builds.
`gpu_histogram.cpp` holds runtime setup, opaque file transport, completion and
the developer-tool commands. Source analysis and profiler orchestration are
development infrastructure; application fixtures, validation and metrics remain
on GPU. There are no hidden source fragments or embedded Python programs.

Named implementation namespaces retain the independent scopes previously
provided by translation units. Similar helper names are not an assertion that
their contracts are interchangeable. The direct data-kernel harness links the
exact production leaf object and invokes its actual kernel specializations,
excluding the coordinator object. CUDA launch
wrappers remain compiled by nvcc; host compilation does not parse CUDA launch
syntax. Build-mode selection is fixed executable setup.

Verification: retain the original checkpoint, sources/binary hashes and failed
attempts; compile and run all eleven GPU suites serially; exercise the complete
benchmark and frozen prediction imports; rerun applicable sanitizers and compare
kernel resources/code generation where translation-unit changes matter.
Ported development tools must reproduce their receipt/coverage semantics and
clearly distinguish any versioned normalization changes. Source inventories
count every maintained section, helper and macro, with physical totals and
production/test/tool categories reported separately.

The completed consolidation build passed all eleven serial GPU suites and all
288 benchmark correctness checks. The four archived model/dataset pairs (wine,
magic, letter and delicious) reproduce reference prediction bits exactly and
pass the same-engine metric gates. This does not establish retraining quality or
the optional legacy JSON metric-arithmetic gate. The direct production leaf
object passed all four sanitizer modes; the previous unsupported full-object
attempts remain preserved. The C++ collector passed a new Nsight Compute run,
and the disabled marker passed PTX erasure inspection.

Development validation compared 129 C++/CMake files against the original audit
with no differences in legacy metrics or normalized hashes. Collector fixtures
cover all sixteen routes, saved Nsight exports, missing activity, nonzero exits,
timeouts, process-group cleanup and regex parsing. Detailed receipts and failures
are under `observations/consolidation/` and its `tool-ports/` directory. The
source inventory remains an explicit size measurement, not a capability or
performance equivalence claim.
