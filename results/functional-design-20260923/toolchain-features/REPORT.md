# Installed C++ feature audit — compile only, 2026-09-23

**C++23 is the highest supported nvcc dialect flag here; this does not exclude
individually supported C++26 library features.** NVIDIA explicitly documents
libcu++ host/device backports of C++20, C++23 and C++26 features to C++17 in
[CUDA Programming Guide §5.3.6](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#cuda-c-standard-library).
The installed compiler and headers were inspected and separately instantiated
on host and device paths. No production changes, executable launches, GPU work,
package changes or full rebuild occurred.

## Exact toolchain and scope

- nvcc **13.4.59**, `/usr/local/cuda/bin/nvcc`; GNU C++ **15.2.0** and libstdc++15.
- CCCL/libcu++ **3.4.2**, `CCCL_VERSION=3004002`, under
  `/usr/local/cuda/include/cccl` (resolved paths and 21 header hashes are in
  `installed-headers-and-macros.json`).
- Primary mode: `-std=c++23`, `-O2`; nvcc device target `-arch=sm_86`.
  The normal project has neither extended-lambda nor relaxed-constexpr enabled.
- Host probes compile to objects with GCC and separately with nvcc's `.cu` host
  path. Device probes use runtime-dependent arguments in an emitted kernel and
  compile to PTX, so a host-only success or constant-folded static assertion is
  not being treated as device support. PTX assembly/linking/execution and full
  standard-conformance coverage are outside this audit.
- 89 compile commands, plus a macro preprocessing command and version/help
  queries. Expected negative probes and initial failed attempts are retained.
  Every successful feature probe had empty compiler stderr.

## Results under C++23 and current project flags

“Pass” means the specific operations in the named probe compiled, not every
operation/type combination of that facility. The two host compilers agreed.

| Facility | Host `std::` | Host `cuda::std::` | Device `cuda::std::` |
| --- | --- | --- | --- |
| C++26 `saturating_add/sub/mul/div/cast` | Absent | Pass | Pass |
| Earlier `add_sat/sub_sat/mul_sat/div_sat/saturate_cast` names | Absent | Absent | Absent |
| C++26 `inplace_vector<int,4>`, insertion/access | Header absent | Pass | Pass |
| C++26 `function_ref<int(int)>` | Absent | Absent | Absent |
| C++23 `ranges::fold_left` | Pass | Absent | Absent |
| `views::transform`, array view/indexing | Pass | Pass | Pass |
| C++23 `expected<int,int>::transform/and_then` | Pass | Pass | Pass |
| C++23 `forward_like` | Pass | Pass | Pass |
| C++23 public `bind_back` | Pass | Absent | Absent |

The successful CUDA range/expected probes use unannotated lambdas inside a
`__device__` function; their device execution space is inferred. Initial
explicitly annotated lambdas failed because the normal flags omit
`--extended-lambda`. Those failures are preserved. Separate flag-enabled controls
also passed; enabling that flag is unnecessary for the tested implicit lambdas.

The analogous tested `std::` device calls fail under the current flags; do not
substitute host library names into device code. NVIDIA also
[recommends the device-compatible libcu++ equivalents](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#constexpr-functions)
instead of assuming host standard-library functions are device-callable.

The installed public `<cuda/std/functional>` includes an internal `__bind_back`
implementation, but does **not** expose public `cuda::std::bind_back`. Internal
headers or helper names are not an adoption path. Header presence similarly
would not have proved `function_ref` support.

| Language feature | GCC host C++23 | nvcc host C++23 | nvcc device C++23 |
| --- | --- | --- | --- |
| Explicit object parameter (`this auto`), recursive callable | Pass | Pass | Pass |
| Static call operator | Pass | Pass | Pass |
| `if consteval` | Pass | Pass | Pass |
| Variadic fold expression | Pass | Pass | Pass |
| C++26 pack indexing (`xs...[0]`) | Rejected in strict C++23 | Rejected | Rejected |

The language results agree with NVIDIA's
[C++23 support table](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#c-23-language-features).
GCC's separate C++26 control accepts pack indexing and the earlier saturation
names. Its C++26 controls still reject the newer saturation names, `function_ref`
and `<inplace_vector>`. This installed host library differs from libcu++.
nvcc explicitly rejects `--std=c++26`; its help lists dialects only through
C++23. These controls do not change the project dialect or mix language modes.
[GCC's support table](https://gcc.gnu.org/projects/cxx-status.html) distinguishes
compiler support from library support.

## Relevant C++26 details

Installed libcu++ uses the new `saturating_*` spellings found in the
[current standard draft](https://eel.is/c++draft/numeric.sat), with feature macro
`__cccl_lib_saturation_arithmetic=202603L`. It does not use GCC15's earlier
spellings. The bounded supplementary device probe instantiates all five
operations and uses static assertions for overflow/clamp boundaries; this is
compile-time checking, not GPU runtime validation. Saturating overflow is a
**different contract** from rejecting overflow or computing exact counts. Do
not replace checked histogram/capacity arithmetic with saturation to modernize
its spelling.

The installed [`cuda::std::inplace_vector` implementation](https://github.com/NVIDIA/cccl/blob/v3.4.2/libcudacxx/include/cuda/std/inplace_vector)
also compiles on device for `try_push_back`, including a full-capacity failure
return. Its feature-test macro was absent in the inspected `<cuda/std/version>`;
actual instantiation is stronger evidence than that absent macro. This bounded
mutable container does not itself make a program functional. It is only relevant
if an explicit fixed-capacity construction avoids allocation without worsening
copying, register/local-memory pressure or semantics. No device exception
recovery is claimed from a successful bounded insertion probe.

## Useful adoption choices

1. **Use explicit-object callables, static call operators, constexpr/consteval,
   concepts and ordinary folds where they simplify our own pure transformations.**
   The tested C++23 constructs already work on both paths; recursive syntax still
   needs generated-code inspection and a bound, not an assumed speed benefit.
2. **Use `std::expected` in host setup and `cuda::std::expected` for applicable
   device value/error composition**, with small error types and proven control
   flow. The tested monadic operations work; evaluate object size, copying,
   branches and error semantics before moving a hot path.
3. **Use `forward_like` only for an actual cv/ref-preservation need.** Device
   range views can express lazy scalar transformations, but they are not a
   parallel GPU algorithm and do not automatically improve memory access.
   Host `ranges::fold_left` applies to permitted setup/reference work, never a
   CPU preprocessing/training shortcut. `cuda::std::ranges::fold_left` and
   `cuda::std::bind_back` are unavailable here.
4. **Do not force saturation, inplace_vector, pack indexing or function_ref
   into the design.** The first two are available backports with specific
   semantics; the latter two are unavailable on the required nvcc path. Typed
   statically composed callables remain the useful choice for inlining.

No CUB/Thrust/NVIDIA parallel algorithm was added. Library availability neither
proves purity nor ranks speed. This audit supports the corrected instruction:

> Use the newest supported language/library facilities. C++23 is the highest
> nvcc dialect flag currently supported; individually verified C++26 library
> backports and newer supported features are allowed. Verify host and device
> support separately before adoption.

## Evidence and reproduction

`results.json` preserves all initial commands, exit statuses and stdout/stderr
paths; `followup-results.json` preserves implicit-lambda, explicit-flag, bounded
saturation and failure-returning-container controls. `sources/` contains every
translation unit. `objects/` contains only compile artifacts (objects/PTX), never
executed. `installed-headers-and-macros.json` identifies installed headers and
host feature-test macros. Raw compiler diagnostics, including all negative
probes, are in `logs/`. The capture scripts are `run_probes.py` and
`run_followups.py`; copy them to a fresh evidence directory to reproduce without
overwriting this record. `artifact-manifest.json` seals all files except itself.
