# Optional compiler evidence and device memcheck builds

This is diagnostic infrastructure, not a production algorithm replacement. The
input contract is an existing CMake CUDA C++23 project/build and its exact source,
compiler commands, logs and binary artifacts. Outputs are append-only raw
diagnostics plus a structured report of compilation cost, generated resources and
GPU code size. This work performs no GPU computation and never ranks profiled
timings as production performance.

## Choice recorded before implementation

Use the installed NVIDIA compiler's documented `--resource-usage`,
`--optimization-info=inline`, PTXAS spill/local-memory warnings and
`--fdevice-time-trace=-`. The last spelling gives each object its own trace path
and avoids parallel builds overwriting one trace. These switches provide evidence
without intentionally changing optimization or arithmetic. Do not enable device
LTO, RDC, register caps, fast math or a new language mode. Counting already emits
PTXAS verbose diagnostics; retain its normal flags and all existing defaults.

Expose independent default-OFF `GH_COMPILER_DIAGNOSTICS`,
`GH_COMPILER_TIME_TRACE` and `GH_DEVICE_SANITIZE` options. The last selects the
installed compiler's `--fdevice-sanitize=memcheck`. Its compiler help explicitly
requires running the binary under Compute Sanitizer; these builds are diagnostic
only and are not valid performance candidates. Wrap GPU CTest commands with
Compute Sanitizer when enabled. Check the selected compiler's capabilities and
fail configuration if requested instrumentation is unavailable. A training build
cannot instrument an imported precompiled counting archive: report that limitation
explicitly and build the root counting project separately for that coverage.

The evidence collector reads existing build products and invokes only offline
binary utilities and optional ctadvisor. It must not launch a target, compile it,
or install missing tools. Preserve raw logs, compiler/tool identities, source and
build hashes, per-object time traces and extracted cubins. Represent unavailable
compile wall time or unparsed diagnostics as unavailable rather than zero. Report
trace span separately from summed phase duration because compiler phases may
overlap. Compare resources/code size by exact kernel identity; preserve unmatched
symbols rather than silently normalizing names. Compile Time Advisor is optional
and its absence is a reported capability, not claimed success.

## Verification and gates

CPU tests cover resource extraction, code-section accounting, time-trace units,
missing evidence, comparisons and refusal to overwrite evidence. Configure/build
fresh default, diagnostics and sanitizer directories without running GPU code.
Root owns serial GPU validation. A default build must retain its existing CUDA
flags and instruction bytes; any discrepancy stays visible. Diagnostic builds
must retain optimization/arithmetic flags. Sanitizer builds must show the flag on
every locally compiled CUDA object and run through memcheck only. No kernel,
counting policy or default is promoted or changed by this work.

## Collection recipe

Configure a fresh directory with `CMAKE_EXPORT_COMPILE_COMMANDS=ON`,
`GH_COMPILER_DIAGNOSTICS=ON` and `GH_COMPILER_TIME_TRACE=ON`. Preserve the entire
build stdout/stderr. After the build and source edits finish, collect existing
artifacts, for example:

```sh
python3 tools/compiler_report.py --build-dir build/diagnostics-booster \
  --binary build/diagnostics-booster/libghb.a \
  --build-log results/my-diagnostic-build.log --output results/my-compiler-report
```

The output directory must be new. `--baseline /absolute/path/report.json` adds
exact section/resource comparison. `--ctadvisor /path/to/ctadvisor` overrides
automatic executable discovery. An unavailable advisor is recorded; no package
is installed. This collector never launches the binary or rebuilds source.

Use another fresh build with `GH_DEVICE_SANITIZE=ON` for compile-time memcheck.
Invoke its executables only through `compute-sanitizer --tool memcheck`;
its GPU CTest entries perform that wrapping automatically. Optional validation
targets (`GHB_BUILD_DIAGNOSTIC_VALIDATION`) and hardware calibration
(`GHB_BUILD_HARDWARE_CALIBRATION`) are also default OFF.

The focused CPU parser suite is `python3 tests/test_compiler_report.py`. An offline
smoke on the existing booster archive recovered 9 cubins, 149 executable sections,
1,721,600 code bytes and 149 resource records. This is extraction verification,
not a new performance measurement.

## Observed compiler instrumentation effect

The zero-change check failed for a booster build with device time tracing enabled.
A controlled offline compilation of the unchanged `kernels.cu` with nvcc
13.4.59 isolated the cause: `--fdevice-time-trace=-` alone changed both split
candidate kernels from 48,000 to 60,032 bytes and from 49,664 to 61,312 bytes.
Default compilation, resource/spill warnings alone and inlining remarks alone
each matched all 41 original code/constant sections in that translation unit.
The time-trace-only result matched the combined diagnostic build's changes.

The same controlled default-versus-trace compilation of `initialization.cu`,
`split_search.cu` and `higher_order.cu` reproduced respectively 3, 2 and 3 more
changed executable sections. Higher-order anonymous symbols also changed names;
a unique full-demangled-signature mapping retained every original name and hash
and exposed those three actual instruction changes. Across these four translation
units, all fresh-default code/constant sections matched both historical archives,
and every trace-only section matched the combined diagnostic build. In total,
10 executable sections changed under tracing; renaming was not treated as proof
of instruction equivalence.

Therefore time tracing belongs in a separate compilation-cost build. Keep
`GH_COMPILER_TIME_TRACE=OFF` for runtime diagnostic/performance comparisons;
optional resource diagnostics require their own generated-code verification.
The original failed byte-identity observation remains failed. The compiler's
optimization/arithmetic switches were unchanged, but that did not guarantee
identical generated instructions. No speed or quality conclusion follows from
this compile-only investigation.

Commands, logs, cubins, source identities and isolated comparisons are retained
under `results/profiling-expansion-20260923/compiler-flag-isolation`.
