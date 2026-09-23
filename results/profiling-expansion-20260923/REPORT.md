# Five diagnostic extensions: implementation and evidence

Completed 2026-09-22 local / 2026-09-23 UTC on the RTX A5000 Laptop GPU
(SM86, 48 SMs, 16 GiB, 4 MiB L2), Windows driver 616.92 through WSL.
CUDA compiler 13.4.59, Compute Sanitizer 2026.3, Nsight Compute 2026.3 and
Nsight Systems 2026.3.2. All GPU processes were run serially. The intentional
independent-stream correctness test is the only concurrent work within a process.

All five requested areas are implemented. They produced two concrete findings:
compiler time tracing changes generated instructions, and compact tree export
copied unwritten Node padding. Runtime profiling now uses a separate build
without compiler time tracing; the padding issue is fixed by extending the
existing one-time GPU storage clear to compact exports. No counting policy,
default, production CUDA kernel, model layout or arithmetic formula changed.

## 1. Graph/range profiling and replay/cache controls

`tools/profile_gpu.py` now exposes explicit NCU kernel/application/range/app-range
replay, node/whole-graph profiling, and cache-control settings. Application replay
uses strict matching. Nsight Systems graph/node tracing is independently selected.

Verified captures include:

| Evidence directory | Actual observation |
|---|---|
| `ncu-count-graph-final` | 2 graph workloads with positive CUDA graph IDs and launched CTAs |
| `ncu-count-range` | Native range replay of complete count operations |
| `ncu-application` | Strict application replay with real kernel metrics |
| `runtime-app-range` | Final runtime build: complete train/predict ranges with GPU work |
| `runtime-nsys-graph` | Final runtime build: device graph activity in SQLite |
| `nsys-graph-nodes` | Individual node tracing, 67 kernel records |
| `ncu-atomic-f64` | FP64 hot-bin atomic calibration, counters and memory/occupancy analysis |

The initial `ncu-app-range` attempt failed because `--profile-from-start` is
unsupported for range replay. The runner was corrected and its retry passed.
The initial `ncu-whole-graph` capture contained only standalone preprocessing
kernels. Its original manifest remains intact, but its graph-coverage claim is
**invalidated**: the final evidence gate requires positive graph identity and GPU
work on the same record. `ncu-count-graph-final` satisfies the corrected gate.

Native range replay of the complete booster region (`ncu-stream-range`) failed
with target error 11; the exact cause is unresolved. Native count ranges and
application-range replay of booster regions work. This is workload-specific
support, not a claim that arbitrary CUDA API sequences can be replayed.
Profiler timings were never used to rank implementations.

## 2. Compiler resources, compilation cost and code identity

Default-off CMake options provide resource/spill/local-memory reports, inlining
remarks and per-object compilation traces. `tools/compiler_report.py` retains
raw logs, compiler/tool/source/build hashes, extracted cubins, register/shared/
local-memory resources, code/constant sections, Ninja command durations and
compilation traces. It runs installed NVIDIA Compile Time Advisor 13.4.49 when
traces are present; installation was checked against NVIDIA's archive SHA256.

**The compiler tracing preservation gate failed.** On unchanged source,
`--fdevice-time-trace=-` changes ten booster instruction sections under nvcc
13.4.59. Isolated default/resource/inline/trace compilations identify the trace
flag as the cause. Seven changes have unchanged symbol names; three more occur
among renamed higher-order symbols. Raw observations and explicit unique
demangled mappings are in `compiler-flag-isolation/`. These are separate
compile-cost builds, not runtime profiling/performance baselines.

The final `build/diagnostics-runtime-booster` enables resource diagnostics but
disables compiler time tracing. Its **149 instruction sections and 163 constant
sections are byte-identical** to the prior booster after uniquely mapping
compiler-private symbol renaming. All original/new names and hashes are retained
in `runtime-mapped-sections.json`. Host code and metadata are outside that claim.
The diagnostic counting archive matches all **1,564 instruction/constant
sections** exactly, including all 782 instruction sections, without renaming.

The only production source changes are two host-side initialization conditions
and explanatory comments, recorded in `production-source-audit-final.json`.
All production CUDA sources and counting configuration/default files are intact.

## 3. Expanded correctness instrumentation and the export fix

`GH_DEVICE_SANITIZE=ON` creates a separate compiler-instrumented memcheck build.
GPU CTest commands are wrapped in Compute Sanitizer. The count project is built
separately so imported archives are instrumented too. These binaries are run
only under memcheck. `compiler-flags-audit.json` records configured CUDA C++23
commands and instrumentation coverage; unused test targets need not have been
built to validate the binaries exercised here.

The runner adds allocation padding/leak checks and global/shared/all initcheck
selection, plus unused-memory reporting. Both positive and deliberately faulty
canaries verified collector behavior:

| Canary | Result |
|---|---|
| Compiler memcheck, valid | Zero errors |
| Compiler memcheck, out-of-bounds | Expected failure, 37 errors, process exit 99 |
| Shared-memory uninitialized read | Expected failure, 32 errors, process exit 99 |
| Global uninitialized read/unused storage | Expected failure, 33 errors, process exit 99 |

`canary-fault-proof.json` verifies actual fault messages, not merely failure exit
codes. The canary sources/binaries are isolated evidence, never production code.
Compiler memcheck passed counting, training, numerical stages and lifecycle
cases with padding/leak checking. Racecheck passed the count stream/scratch
stress; synccheck passed numerical graph stages.

Expanded initcheck initially found **13 host-copy errors** in compact tree export.
Each 32-byte Node contains four alignment bytes at offsets 20–23. GPU stores
write every live member but leave these bytes untouched; a bytewise export copies
them too. The existing one-time clear was conditional on bounded export. It is
now unconditional in both trainer setup paths, outside graph replay. This adds
one startup clear for compact export; it adds no production CUDA kernel
implementation or change to live tree arithmetic. CUDA may execute a runtime
memory-set kernel for that clear. Original sources and hashes are in
`node-padding-before/`.

All four final initcheck combinations pass: per-output/batched construction
times compact/bounded export, graph execution with an output-tile tail. The
original failed run remains in `initcheck-all-booster/`.

## 4. Hardware calibration

The optional CUDA C++23 calibration executable covers u32, u64 and FP64 global
atomic additions; uint16/uint32 keys; uniform, single-hot and deterministic
90%-hot patterns; grids of one/four blocks per SM. The grid ratio is not a
measurement of simultaneous block residency. The skew fixture's tail cycles
512 odd bins, not an iid uniform or Zipf distribution.

All **36 final runtime cases x 7 measured repetitions** have exact counter
comparisons. FP64 increments are 0.125 and exactly representable under the
accepted bounds. Each timing includes output clear and accumulation; key
generation and validation copies are outside timing. Synchronized host cost is
separate. This is not the production clear/accumulate/widen operation.

For illustration, four-grid-blocks-per-SM / uint16-key medians (microseconds):

| Atomic | Uniform | Single hot bin | 90%-hot fixture |
|---|---:|---:|---:|
| u32 | 44.032 | 252.928 | 185.312 |
| u64 | 22.528 | 200.704 | 183.296 |
| FP64 | 22.528 | 663.552 | 527.232 |

These are local observations, not algorithm rankings or hardware ceilings.
The matrix starts from a low-clock P8 state in the earlier telemetry; short
warmups, fixed order, cache state and clock/power variation limit cross-cell
comparisons. Validation reads between repetitions can affect cache state.
`runtime-calibration.json` retains every repetition and check. The early
`atomic-first.json` checked only each case's final repetition; this limitation
was corrected before the final measurement and the old file is retained.

SASS confirms discarded-return global RED additions for the three atomic widths;
FP64 uses `RED.E.ADD.F64.RN.STRONG.GPU`. The selected hot FP64 Nsight capture
reports low overall SM/memory throughput despite about 59.5% active-warps
occupancy, consistent with contention being relevant; this is an interpretation,
not proof of a unique bottleneck. Profiler durations do not establish speed.
The calibration executable's 16 code/constant sections also match between the
compile-trace and final runtime builds (`calibration-byte-comparison.json`).

The separately installed NVIDIA `nvbandwidth` reference is pinned to commit
`82fc4e8c6afa0babb8687793678f615b3b8d793e` and is not linked into production.
All 12 local-memory test/size combinations passed verification: copy engine,
SM copy/read/write at 1/4/64 MiB buffers, seven samples each. At 64 MiB its
reported medians were 155.30/155.05/339.97/335.81 GB/s respectively. Copy
rates follow the tool's payload convention; read+write traffic is different.
A two-buffer copy's total working set exceeds its per-buffer size. Raw tool
JSON, commands, telemetry and the initial missing-NVML-header build failure are
retained. NVML headers were installed from a SHA-verified CUDA archive.

## 5. Numerical divergence and lifecycle validation

The optional harness records actual FP64 derivatives/statistics, exact counts,
per-feature GPU candidates/winners, independent long-double candidate scores,
decision margins, leaf values, routing and root-tree prediction. It enumerates
all small-case thresholds/directions; independent scores are explicitly not
called GPU scores. Three outputs use a two-wide tile and one-output tail, with
root/deeper frontiers and three graph repetitions.

The final normal run passed **20,080 gates** across 24 stage snapshots and four
alternating lifecycle workloads. Exact dyadic checks are distinct from
cancellation-sensitive FP64 observations and numerical budgets. Five repeated
cancellation-sensitive snapshots first differed at their histograms; 12
candidate rankings were explicitly ambiguous within the diagnostic budget.
The largest statistic/reference gradient difference was 1.375 in the constructed
large-cancellation fixture, not in ordinary validation or held-out loss.
Those observations are not reclassified as exact preservation.

Independent review found and fixed initially missing winner/leaf/shape gates.
The CPU reference-audit self-test rejects incorrect winners, missing splits and
bad leaf values: **56 checks pass**. Lifecycle cases now include compact and
bounded exports, graph/stream execution, higher-order learning, serialization
and object destruction. Count lifecycle passes **56 exact gates** using independent
graph instances/streams and event-ordered shared scratch with changed-input replay.

Final runtime CTest: **18/18 passed** (14 GPU, 4 CPU). Runner tests: **27 passed**;
compiler evidence tests: **10 passed**. Relevant raw logs are adjacent to this
report. These bounded checks are not full application-quality benchmarks.

## End-to-end preservation observations and limits

Seven alternating-order unprofiled before/after process pairs, after one warmup
pair, exercised 4,096 training rows, 1,024 held-out rows, 16 features, three
outputs, three rounds and depth three with compact batched graph export.
The final `padding-paired-matched/` study uses matching host symbol/frame-pointer
flags and no compiler diagnostic flags. Its 312 mapped device code/constant
sections match the baseline. The earlier `padding-paired/` study is retained
but had different host flags and is not the final overhead comparison.
Training and held-out loss arrays matched exactly in all seven pairs; every
run passed its CPU/GPU prediction tolerance and serialization round trip.

Training medians were 6.923 ms before and 6.920 ms after; the median paired
after/before ratio was 1.000, spanning 0.943–1.160. Total-training medians were
15.306/15.805 ms with median paired ratio 1.039, spanning 0.881–1.233. This
noisy bounded experiment cannot establish zero overhead or a speed improvement.
It did not modify or rerank any measured counting default.

**Exact cross-run model equality failed and remains failed.** A separate small
before/after model comparison had identical topology/metadata but leaf-value
differences up to 6.94e-17. Repeating the unchanged baseline also changed leaf
values (up to 5.55e-17); repeating the final build differed by up to 1.67e-16.
This demonstrates pre-existing run-to-run variation; it does not prove every
difference is caused by the same mechanism or permit relaxing any zero-allowance
gate. In the larger paired run, CPU/GPU maximum-error metadata differed in
two final matched pairs between zero and 1.11e-16 (three in the earlier study),
so exact equality of that field also fails.
All raw models, field comparisons and failed equality results are retained.

No existing strict quality gate was waived. No universal speed, accuracy or
bitwise-determinism claim is made. The intended next use of these additions is
to diagnose and measure candidates under an explicit workload contract.

Usage and design links: [profiling guide](../../training/PROFILING.md),
[runner](../../training/profiling/RUNNER_DESIGN.md),
[compiler](../../training/profiling/COMPILER_DESIGN.md),
[calibration](../../training/profiling/CALIBRATION_DESIGN.md),
[validation](../../training/profiling/VALIDATION_DESIGN.md).
