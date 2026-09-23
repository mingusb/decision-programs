# GPU histogram and booster

CUDA counting, feature fitting and encoding, multi-output boosting, inference,
binary/CSV formats, metrics and optional stage instrumentation. Application
planning, fixtures, validation, oracles, metrics and benchmark statistics run on
GPU. The host handles CUDA bootstrap/completion and external byte transport;
compiler, source-analysis and profiler tools are development infrastructure.

The maintained code is consolidated into three files:

| File | Role |
| --- | --- |
| `gpu_histogram.hpp` | Shared CUDA types, declarations, small helpers and host bootstrap interface |
| `gpu_histogram.cu` | Production implementations and separately selected GPU test, benchmark and quality entries |
| `gpu_histogram.cpp` | Host runtime, opaque file transport and development-tool commands |

Production compiles into `gh`, including a separately compiled `gh_data_leaf`
object shared with the direct data-kernel harness. Each executable selects its
own fixture section from the CUDA source; unrelated fixtures do not enter production.
CMake metadata, documentation and preserved observations remain separate.

## Build and checks

The pinned environment uses Linux, CMake 3.30 or newer, a C++23 host compiler and
CUDA 13.4 with nvcc C++23 support. Development tools also require the development
packages for nlohmann JSON, OpenSSL, SQLite3, ICU and PCRE2's 8-bit library. Source
inventory additionally invokes `clang-format-21`; diagnostic commands require
the selected profiler or sanitizer executable.

From the repository root:

```sh
cmake -S . -B build-three -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-three --parallel 1
cmake --build build-three --target gh_tools --parallel 1
ctest --test-dir build-three --output-on-failure -L gpu -j 1
```

Architecture 86 matches the pinned target. Only the root agent runs GPU
workloads, serially. `benchmark` and `quality` are separate executable targets;
the latter accepts dataset bytes, model bytes, reference prediction bytes and
an optional frozen quality report. `GH_OBSERVE` defaults to `OFF`; enabled stage
markers are diagnostic and affect scheduling.

## Development tools and evidence

```sh
build-three/gh_tools source-size --self-test
build-three/gh_tools source-size --fresh . --output /absolute/new/source-size.json
build-three/gh_tools collect memcheck /absolute/new/receipt -- /absolute/build-three/data_leaf_checks
build-three/gh_tools erasure /absolute/observe_checks.ptx
```

Receipt destinations must be new. Source inventory counts production, tests and
tooling, including all helpers and macros; category markers partition the three
physical source files. Historical Python normalization is reused only from a
frozen receipt after matching file hashes. New or changed Python sources fail
closed. Collector regular expressions use UTF-8 PCRE2; universal compatibility
with Python regex extensions is not claimed.

Consolidation preserves the existing algorithm source and counting defaults;
it does not establish numerical, quality or performance equivalence for the new
binaries. Full quality/performance gates and the requested 75% source-character
reduction are not established. Historical receipts bind their original binaries.
Uninstrumented complete-operation timings rank implementations; profiler results
explain them.

[AGENTS.md](AGENTS.md) defines the repository's implementation and execution
constraints. Current tests are selected from `gpu_histogram.cu`; old `src/`,
`include/`, `tests/` and Python-tool paths in historical receipts refer to the
pre-consolidation checkpoint, not additional maintained implementations.

See [capability status](docs/CAPABILITIES.md), [layout and verification](docs/three-file-layout.md),
[source inventory](docs/source-size.md), [diagnostic collection](docs/observability.md)
and [direct production leaf checks](docs/data-leaf.md).
