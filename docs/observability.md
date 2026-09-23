# GPU observations and diagnostic receipts

Contract and selection recorded before implementation, 2026-09-23. Production
stage markers are optional compile-time calls, disabled by default. Enabled
markers enqueue one ordered one-thread child that appends a timestamp, stage,
iteration and begin/end flag into caller-supplied resident storage. Capacity
failure sets the operation Status; storage/counter must remain exclusive and
live through completion. Disabled calls must produce no loads, stores or child
launches; inspect a dedicated exercised disabled kernel in generated PTX/SASS.
These marker kernels perturb scheduling and are diagnostic observations only.
The shared test bootstrap adds an NVTX range from launch through checked CUDA
completion when GH_OBSERVE is enabled. That host range describes the submission
boundary; GPU Stamp records describe the separately instrumented stage order.
No stage GPU duration is inferred from host NVTX timestamps.

The first timing primitive uses the [PTX global timer](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer-globaltimer-lo-globaltimer-hi).
NVIDIA defines a nanosecond timer but warns its behavior is target-specific.
Preserve raw u64 values and report scope `globaltimer-cdp-tail-v1`, never CUDA-event
milliseconds. A start marker is a device-null-stream child before the operation;
its end marker is a tail child after the operation's completion tail. Each sample
has a separate coordinator and next-sample tail continuation, so samples cannot
overlap. Measured spans include intervening launch/clear/reduction/completion work
and marker scheduling. Input creation, reference checks and summary computation
remain outside each span. Qualification checks positive, nonoverlapping intervals,
cross-SM ordered markers, and an empty-operation control; preserve overhead without
subtracting it. Real qualification and all ranking evidence remain pending.

The fixed GPU schedule is three warmups per variant, then fifteen paired samples,
alternating AB/BA between pairs. Keep all 36 raw intervals. GPU summary uses the
geometric mean of A/B duration ratios and a casewise two-sided 95% Student-t
interval on their logarithms (14 degrees of freedom). This is the standard
[paired-observation interval](https://www.itl.nist.gov/div898/handbook/prc/section3/prc312.htm)
applied to log ratios, not a simultaneous multi-case guarantee. Its assumptions
require independent pairs and reasonably behaved log ratios; drift, multimodality
or failed correctness/quality makes promotion inconclusive. A single GPU thread
processes only 15 pairs; a parallel reduction adds synchronization without useful
work reduction at this size. Tests use constant ratios, swapped variants,
scale/translation invariance, invalid intervals and analytically balanced ratios.

## Collection infrastructure

`build/gh_tools collect` launches one external diagnostic tool into a new receipt
directory. The command is implemented in `gpu_histogram.cpp`; it replaces the
former `tools/collect.py` entry. Host process control, hashing, parser validation
of diagnostic activity, and artifact preservation are development infrastructure;
application fixtures, metrics and benchmark statistics remain on GPU. Retain command, selected
environment, binary/tool identities, stdout/stderr, raw reports, exports, timeout
and return status. Never overwrite evidence. Reject inherited competing injectors.
Success requires actual matching tool activity and successful completion, not exit
zero, a banner, or an empty report. Instrumented timings never rank production.

Tool families are Nsight Systems/Compute; memcheck, initcheck, racecheck and
synccheck; installed CUPTI trace/range/continuous-PC samples; NVBit instruction,
memory and graph counters; CUDA-GDB; and offline cuobjdump/nvdisasm/resource dumps.
Vendor binaries are infrastructure; no vendor algorithm enters production.
Missing binaries, denied counters, unsupported modes, zero samples and decoder
failure are failed evidence, preserved explicitly. NVBit graph counts aggregate
unique functions, not graph launch instances. PC sampling is statistical and a
short run can legitimately produce no evidence.

The collector receipt identifies the compiled `gh_tools` executable. User
`--kernel` and `--activity` expressions use UTF-8 PCRE2 with Unicode properties;
compatibility with every Python-specific regular-expression extension is not
claimed. Historical receipts still identify their original Python collector.

[Current CUPTI restrictions](https://docs.nvidia.com/cupti/release-notes/release-notes.html#known-issues)
matter for CDP2: activity tracing records host-launched kernels, not device child
launches; parent metric results cover their call trees. Thus an observed parent
kernel is not per-child attribution. The installed legacy range injection sample
intercepts ordinary host launches and is an explicitly limited comparison, not a
new Range Profiling API implementation or graph profiler. NVBit tools likewise
require evidence of the requested kernel/activity; none is presumed CDP-safe.
[NVBit's upstream distribution](https://github.com/NVlabs/NVBit) supplies the
installed sample injectors. Kernel-node/aggregate graph profiling remain distinct.

Installed evidence paths: CUDA 13.4 under `/usr/local/cuda`; Nsight Compute
2026.3.0 under `/opt/nvidia/nsight-compute`; `nsys` and `cuda-gdb` on PATH;
CUPTI 13.4.58 and NVBit 1.8 under `/home/b/.local/opt/gpu-profiling`.
Explicit `CUPTI_ROOT`/`NVBIT_ROOT` override SDK discovery. Receipt hashes bind the
actual selected binaries; installed version labels alone do not prove compatibility.

Examples (execution is coordinated by root, one command at a time). Substitute
your configured build directory, such as `build-three`, for `build` below.

```sh
cmake --build build --target gh_tools --parallel 1
build/gh_tools collect memcheck observations/observe-memcheck -- /absolute/build/observe_checks
build/gh_tools collect nsys observations/observe-nsys --kernel 'gh::test::observe_suite::run' -- /absolute/build/observe_checks
build/gh_tools collect ncu observations/observe-ncu --kernel 'gh::test::observe_suite::run' -- /absolute/build/observe_checks
build/gh_tools collect cupti-trace observations/observe-cupti -- /absolute/build/observe_checks
build/gh_tools collect nvbit-graph observations/count-nvbit --workload graph -- /absolute/build/count_checks
build/gh_tools collect cuda-gdb observations/observe-debug -- /absolute/build/observe_checks
build/gh_tools collect cuobjdump observations/observe-sass -- /absolute/build/observe_checks
build/gh_tools collect nvdisasm observations/observe-cubin -- /absolute/module.cubin
build/gh_tools erasure /absolute/observe_checks.ptx
```

The same collector supports `initcheck`, `racecheck`, `synccheck`, `cupti-range`,
`cupti-pc`, `nvbit-count` and `nvbit-memory`. `compiler` wraps an explicit absolute
nvcc command containing `-Xptxas=-v`; source and output paths must be absolute.
The target runs in the receipt directory. For `nvbit-graph`, `--limit` bounds
first-seen functions (maximum 100); for Nsight Compute/count it bounds launches.
Sanitizer targets must print a GPU-authored completion message matching
`--activity`; tool startup and an empty error summary alone cannot pass.
`gh_tools erasure` accepts only the exercised `disabled_probe` PTX kernel with its
constant canary store and no loads, calls, atomics, barriers, branches or timer.
The consolidated observation entry is `gh::test::observe_suite::run`; archived
receipts naming `gh::test::run` describe the earlier binary.

## Sanitizer reporting scope

The initial foundation memcheck in `observations/foundation/core-memcheck.log`
reports driver-internal invalid device attribute values 159/160 reached through
`cuLaunchKernel`, while the GPU core checks complete. Do not delete these failures
or infer the driver/tool mismatch's exact cause from those messages alone.
The installed CUDA 13.4 `include/cuda.h` public attribute enum jumps from 158 to
161; neither reported number has a public name in that header. The core target
does not request these attributes explicitly. This establishes an internal
reporting distinction, not proof of a specific driver defect or harmlessness.
[Compute Sanitizer's reporting modes](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#cuda-api-error-checking)
define the current default `extended` as explicit user API errors plus internal
driver logging. The separate `core-memcheck-explicit.log` retains zero memory/API
errors with `--report-api-errors explicit`, which still reports application API
failures. It is a narrower result, not a pass for the original extended run.
The collector defaults to explicit as a recorded application-error scope and
allows a separate extended comparison. It never uses `no`, suppression files,
or filters errors from the raw logs.

Before promotion, root runs the GPU observation suite and each diagnostic serially,
checks actual activity, then qualifies uninstrumented timing. Compare enabled and
disabled codegen/resources and complete-operation overhead. Preserve every failed
tool route; choose an alternative only with a distinct scope and receipt.

Historical compile evidence before three-file consolidation: both observation translation units compile with nvcc C++23,
O3, SM86 and RDC. `observations/observe/erasure.json` records the exercised disabled
probe's PTX as a constant move, canary store and return, with no marker work.
The compiled probe uses 8 registers and no stack or barriers. This proves the
small disabled call path's erasure, not every future caller's entire codegen.
The consolidated translation units require their own code-generation receipts.
Root's initial six-suite integration run passed observation checks, including
GPU interval ordering, independent work checks, multi-SM bounds and empty control.
This is a timing-protocol correctness check, not an idle performance ranking.

## Actual sanitizer coverage

Root's `observations/sanitizer/*-memcheck-1.log` records clean memory checks for
counting, data preparation, model validation/prediction and the initial trainer
suite. These receipts bind those binaries, not later source edits. The trainer's
initcheck, racecheck and synccheck attempts each report CDP unsupported and exit
91. They remain coverage gaps, not passes. The default extended API-error run
described above remains a separate failed observation.

The [current known limitations](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html#known-limitations)
explicitly exclude dynamic parallelism from those three checkers. Memcheck also
does not check device-side CUDA API errors; production/test launch-return checks
remain necessary even when host API reporting is enabled. The
[direct data-kernel harness](data-leaf.md) checks the actual production leaf
kernels with GPU-generated inputs and GPU oracles. Its receipts cover those leaf
executions, not CDP2 coordination, tail ordering or cross-kernel lifetime behavior.
The initial three-file build linked the full production object into that harness:
memcheck passed, but initcheck, racecheck and synccheck each exited 99 with CDP
unsupported. Preserve `observations/consolidation/sanitizers/*-1.log` and
`exits-1.txt`; those runs are not coverage passes. The subsequent shared leaf
object construction requires new binary and sanitizer receipts.
