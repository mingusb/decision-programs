# Profiling and debugging our CUDA code

The diagnostic path covers the custom count kernels and GPU-resident booster.
It must preserve their inputs, outputs, numerical policies and normal build
defaults. Profiling observations explain execution; they are not benchmark
rankings. NVIDIA sample collectors are tooling, not production algorithms.

## Instrumentation choice and validation plan

Use the existing optimized CUDA `-lineinfo` builds and NVTX stage annotations.
Add optional host symbols and a separate device-debug build. Add opt-in outer
capture ranges at benchmark operation boundaries, outside CUDA stream capture.
Only enabled capture ranges synchronize the device or call the CUDA profiler
start/stop API. Keep collector injection outside production libraries.

Most of these tools inspect compiled code or inject instrumentation at runtime;
adding hooks inside counting or training kernels would add unnecessary work.
NVBit's ordinary instruction and memory tools synchronize during launch
callbacks, so captured graphs require its graph-aware instruction counter.
CUPTI's official range-injection example handles ordinary launches; use Nsight
Compute for graph-aware hardware metrics. Continuous CUPTI PC sampling has a
graph-aware injection example.

Validate one tool per process, with GPU work serialized. Require an actual
counter, trace, report or debugger breakpoint, not merely a successful launch.
Check exact count outputs and the booster's prediction/serialization checks.
Compare diagnostic and plain runs of the same deterministic workload for output
and quality preservation. Preserve failed attempts and do not treat instrumented
durations as performance evidence. Record source/binary hashes and tool versions.

The [implementation and verification report](../results/profiling-tools-20260922/REPORT.md)
records 24 successful collector runs on the histogram and booster under WSL
with driver 616.92, plus CUDA-GDB and offline disassembly checks.

## Builds and capture

Both CMake projects accept `GH_PROFILE_HOST_SYMBOLS=ON` for host symbols/frame
pointers with optimized device code. `GH_DEVICE_DEBUG=ON` selects CUDA `-G`
and disables optimization for correctness debugging. Both options default OFF.
Keep these builds in separate directories; never rank device-debug timings.
The CUDA Profiler API headers are a separate toolkit component and must be
installed alongside the CUDA runtime development headers.

```sh
cmake -S training -B build/profiling-booster -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGH_PROFILE_HOST_SYMBOLS=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DGHB_COUNT_BUILD=/home/b/gpu_histogram/build/window-experiment
cmake --build build/profiling-booster -j 2

cmake -S training -B build/profiling-debug -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DGH_DEVICE_DEBUG=ON \
  -DGHB_BUILD_HISTOGRAM_PROBE=OFF
cmake --build build/profiling-debug --target ghb_bench -j 2
```

`support/profiling.hpp` adds `count`, `train`, and `predict` push/pop ranges in
NVTX domain `ghb` to the benchmark drivers. With `GH_PROFILE_CAPTURE=1`, they
also synchronize and call `cudaProfilerStart/Stop` at operation boundaries.
Without that environment setting, the helper makes no CUDA or NVTX calls.
Existing optional per-stage annotations remain available through
`--instrumentation nvtx`. Captured graph execution is represented by outer
host ranges and graph/node tracing, not invented host annotations for replayed
device nodes.

The count probe imports the original count libraries without rebuilding them.
To debug count kernels with `-G`, configure the root CMake project in another
directory with `GH_DEVICE_DEBUG=ON` and use its `histogram_bench` executable.

## Diagnostic runner

`tools/profile_gpu.py` launches exactly one collector per process and records
commands, relevant environment overrides, executable/tool hashes, raw logs,
reports, and success/failure evidence. Every output directory must be new.
Targets run inside their output directory: use absolute target and data paths.
Timeouts terminate owned descendants, including separate process groups and
detached Nsight agents, and preserve partial evidence. PID/start-time identities
and a unique Nsight session identifier prevent unrelated processes being killed.
Nsight CPU symbol resolution defaults off: automatic symbol retrieval delayed
capture completion on this machine. `--nsys-resolve-symbols` explicitly enables
it. NVTX ranges and GPU activity remain collected; graph-level or node-level
detail follows the selected graph trace granularity.

```sh
python3 tools/profile_gpu.py --tool nsys --capture --nsys-graph-trace node \
  --output results/my-booster-timeline -- \
  /home/b/gpu_histogram/build/profiling-booster/ghb_bench \
  --objective binary --rows 128 --test-rows 32 --features 4 \
  --outputs 3 --output-tile 2 --rounds 1 --depth 2 --bins 9 \
  --histogram global --tree-build output-batch --tree-execution graph \
  --instrumentation nvtx
```

Supported launch modes:

| Runner tool | Diagnostic | Execution restriction |
| --- | --- | --- |
| `nsys` | CUDA/NVTX/API timeline, graph or node traces | Stream or graph |
| `ncu` | Kernel counters and source analysis | Stream or graph |
| `memcheck`, `racecheck`, `synccheck`, `initcheck` | Device correctness checks | Stream or graph |
| `nvbit-count` | Dynamic warp instruction counts | Explicit stream |
| `nvbit-graph` | Graph-aware warp instruction counts | Stream or graph |
| `nvbit-memory` | Per-instruction memory addresses | Explicit stream; use tiny inputs |
| `cupti-trace` | CUDA activities and NVTX records | Stream or graph |
| `cupti-range` | Direct CUPTI hardware metrics | Explicit stream |
| `cupti-pc` | Continuous instruction-address/stall sampling | Stream or graph |

For NCU, select kernels with `--kernel '.*global_window_histogram.*'` or a
booster kernel expression, and bound collection with `--launch-count`.
Use `--section SourceCounters --section PmSampling` for source-level stall and
instruction metrics plus time-series hardware samples. Those capabilities use
Nsight Compute's collectors; the standalone CUPTI PM/SASS SDK samples verify
API availability but are not arbitrary-target adapters.
PM sampling is device-wide; WSL does not provide context-switch filtering, so
unrelated Windows GPU activity can contribute even when our runs are serial.

For NVBit, `--instruction-end N` restricts instrumentation to the first N static
instructions, so resulting counts are partial. The graph tool's
`--function-count` bounds first-seen unique functions, not graph launches.
The stock graph example has a finite 100-function capacity. Normal NVBit
instruction/memory tools must not be injected into stream capture.
The runner rejects an observed graph-counter capacity overflow instead of
accepting its partial output as a successful run.

The CUPTI range sample hooks ordinary kernel launches only. Its advertised
`INJECTION_KERNEL_COUNT` setting is not read by the installed source; it uses
fixed internal batches. PC sampling is statistical: very short kernels can
produce zero samples. Preserve such a failed evidence check and use a longer
representative workload. PC decoding currently skips source correlation;
Nsight Compute and offline cubin disassembly provide source mapping separately.

## Source debugging and offline analysis

`cuda-gdb` can stop at GPU kernel entry and display CUDA threads, blocks and
source lines. Use the separate device-debug build to inspect local variables:

```sh
cuda-gdb --args /home/b/gpu_histogram/build/profiling-debug/ghb_bench \
  --rows 128 --test-rows 32 --features 4 --rounds 1 --depth 2 --bins 9
```

In the debugger, use `set cuda break_on_launch application`, `run`,
`info cuda kernels`, and `bt`. Disable the automatic launch breakpoint with
`set cuda break_on_launch none` before continuing to completion.

`cuobjdump --list-elf BINARY` identifies embedded cubins. `nvdisasm -g -gi`
shows source lines, `nvdisasm -plr` shows register live ranges, and
`nvdisasm -cfg` produces control-flow graphs renderable with Graphviz `dot`.
These operations run on the CPU and do not launch GPU workloads.

The reusable offline helper retains all of those artifacts and command results:

```sh
python3 training/profiling/disassemble.py \
  --binary /home/b/gpu_histogram/build/profiling-booster/ghb_bench \
  --elf quantize.sm_86.cubin --output results/my-cubin-analysis
```

The SDK locations are available through
`~/.local/opt/gpu-profiling/env.sh`. Sourcing it only sets `CUPTI_ROOT` and
`NVBIT_ROOT`; it does not inject tools or override global library loading.

## Whole graphs, ranges and cache context

The runner exposes `--replay-mode kernel|application|range|app-range`,
`--graph-profiling node|graph`, and `--cache-control all|none`. Application
replay uses strict matching. Cache control `none` avoids profiler cache flushing;
it does not guarantee reproducible cache state. These remain diagnostic runs.
Range modes require `--capture`; profiler start/stop calls delimit the range.
Nsight rejects `--profile-from-start` for range modes, so the runner omits it.

Use `--replay-mode app-range --capture --cache-control none` for the booster's
complete captured train/predict regions. Native `range` replay worked for the
count operation but failed on the complete booster region on this machine;
the precise cause is unresolved. The failure is preserved. Whole-graph mode
uses kernel replay and aggregates graph work; it can also capture ordinary
kernels, so its evidence gate requires a positive CUDA graph ID and positive
launched CTAs on the same record. A successful preprocessing-kernel capture
alone cannot prove graph coverage.

```sh
python3 tools/profile_gpu.py --tool ncu --capture \
  --graph-profiling graph --cache-control none \
  --metrics sm__ctas_launched.sum --output results/my-whole-count-graph -- \
  /home/b/gpu_histogram/build/diagnostics-runtime-booster/ghb_instrumentation_bench \
  --n 131089 --bins 32768 --repetitions 3 --batch 2 --warmup-ms 0 \
  --instrumentation off --launch graph
```

For Systems, explicitly select `--nsys-graph-trace graph` for graph-level
events or `--nsys-graph-trace node` for individual kernels. Graph proof comes
from device graph activity in its exported SQLite database, not CPU launch
counts. All tool-specific options reject irrelevant combinations. The
[runner contract](profiling/RUNNER_DESIGN.md) records exact restrictions.

## Compiler resources and compilation cost

`GH_COMPILER_DIAGNOSTICS=ON` enables resources, inlining remarks and spill/local
memory warnings without changing the selected optimization/arithmetic flags.
`GH_COMPILER_TIME_TRACE=ON` emits per-object compilation traces. Both default OFF.
**Use time traces only in a separate compile-cost build:** local isolation proved
that nvcc 13.4.59's time-trace flag changes ten booster kernel instruction
sections. Do not assume compiler diagnostics are observationally neutral.

```sh
# Runtime diagnostics: resource reports, no compilation trace.
cmake -S training -B build/diagnostics-runtime-booster -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGH_COMPILER_DIAGNOSTICS=ON \
  -DGH_COMPILER_TIME_TRACE=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DGHB_BUILD_DIAGNOSTIC_VALIDATION=ON -DGHB_BUILD_HARDWARE_CALIBRATION=ON
cmake --build build/diagnostics-runtime-booster -j 2

# Separate compile-cost build; do not rank its runtime performance.
cmake -S training -B build/my-compile-cost -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGH_COMPILER_DIAGNOSTICS=ON \
  -DGH_COMPILER_TIME_TRACE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/my-compile-cost -j 2 > build/my-compile-cost/build.log 2>&1
python3 tools/compiler_report.py --build-dir build/my-compile-cost \
  --binary build/my-compile-cost/libghb.a \
  --build-log build/my-compile-cost/build.log --output results/my-compiler-cost
```

The collector runs offline, saves raw logs/cubins/traces, hashes source and build
inputs, records resources and GPU code size, and runs installed `ctadvisor`.
`--baseline REPORT.json` compares exact identities; renamed/unmatched sections
remain visible. Summed inclusive compiler phases are not build wall time.
Details and the compiler-flag isolation are in
[COMPILER_DESIGN.md](profiling/COMPILER_DESIGN.md).

## Compiler memcheck and shared initialization

`GH_DEVICE_SANITIZE=ON` is a separate default-off build mode using
`--fdevice-sanitize=memcheck`. Run those binaries **only** under matching Compute
Sanitizer memcheck, as required by NVIDIA. GPU CTest entries are automatically
wrapped. A training project cannot instrument imported count archives; build
the root project with the same setting and set `GHB_COUNT_BUILD` to that directory
for full coverage. Ordinary runtime-injected tools use the optimized normal
build, not the compiler-memcheck binary.

The runner accepts memcheck `--padding 128 --leak-check full`; initcheck accepts
`--initcheck-address-space global|shared|all`, `--track-unused-memory`, and
`--unused-memory-threshold 0`. These options are tool-specific. Shared checking
and global checking were verified with deliberate faulty canaries; compiler
memcheck was verified with both valid and out-of-bounds cases. A deliberately
faulty canary must fail; a successful launch is insufficient verification.

Expanded initcheck found four uninitialized alignment bytes in exported Nodes
under compact export. Both trainers now initialize Node storage once before
construction, extending an existing clear used by bounded exports. Live fields
and CUDA kernels are unchanged. The original failed observation is retained;
compact export now incurs that one startup clear.

## Hardware calibration

`GHB_BUILD_HARDWARE_CALIBRATION=ON` adds the standalone
`ghb_hardware_calibration` executable. It measures native global atomic additions
for u32/u64/FP64, both uint16 and uint32 keys, three address patterns and two
grid sizes. Its JSON retains every timing and exact comparison. Timings cover
clear plus accumulation, with synchronized host cost separate; they do not
represent the production count clear/accumulate/widen operation.

```sh
build/diagnostics-runtime-booster/ghb_hardware_calibration \
  --output results/my-atomic-calibration.json
```

For memory rates, the isolated NVIDIA `nvbandwidth` v0.10 reference is installed
under `~/.local/opt/gpu-profiling/nvbandwidth-82fc4e8/`. It is not linked into
production. Device-local copy, SM copy/read/write apply here; TMA and peer tests
do not. Match footprints to queried L2 capacity and retain thermal/clock context.
Observed reference rates are not hardware ceilings. Pattern details, memory
contracts and the predeclared experiment are in
[CALIBRATION_DESIGN.md](profiling/CALIBRATION_DESIGN.md).

## Numerical divergence and lifecycle checks

`GHB_BUILD_DIAGNOSTIC_VALIDATION=ON` builds two optional harnesses:

```sh
build/diagnostics-runtime-booster/ghb_validate_gpu --self-test-reference
build/diagnostics-runtime-booster/ghb_validate_gpu \
  --mode all --output results/my-numerical-lifecycle.json
build/diagnostics-runtime-booster/ghb_validate_count \
  --output results/my-count-lifecycle.json
```

The first captures derivatives, root/deeper histograms, actual GPU feature
candidates and winners, independent candidate scores and margins, leaf values,
routing and prediction. Three outputs use a two-output tile and a one-output
tail. Exact dyadic gates are separate from cancellation-sensitive FP64
observations and explicitly ambiguous rankings. Independent long-double
validation is diagnostic only; it does not replace GPU training or establish
arbitrary-precision truth. A logical first divergence is not its hardware time.

Lifecycle cases alternate shapes, objectives, compact/bounded exports, graph/
stream execution and higher-order learning, then predict, serialize, reload and
destroy. The count harness validates independent graph instances/streams and
event-ordered reuse of nonempty scratch. Concurrent correctness stress occurs
within one process; performance evidence is collected serially. Neither harness
tests hypothetical async-allocator or graph-update APIs absent from our system.
See [VALIDATION_DESIGN.md](profiling/VALIDATION_DESIGN.md) and the
[expansion results](../results/profiling-expansion-20260923/REPORT.md).

## Frozen-model prediction capture

`ghb_prediction_bench` compares the per-tree reference with the explicit
ordered-forest candidate on the same saved model. Optional `CaptureRange`
boundaries exclude exactness prechecks and warmups, so a single-policy capture
contains only that policy's complete prediction call:

```sh
python3 tools/profile_gpu.py --tool nsys --capture --timeout 180 \
  --output results/my-fused-prediction-profile -- \
  /absolute/build/ghb_prediction_bench --model /absolute/model.ghb \
  --features-bin /absolute/row-major-features.f32 --rows 2584 \
  --policy fused-output --pairs 1 --warmup 0
```

The feature file contains exactly rows × model features native FP32 values.
Packing, allocations, quantization, uploads, kernels, download and cleanup are
inside the captured call. Profiler results explain launch/transfer work; their
timings do not rank the implementations. The same benchmark without diagnostic
capture can retain paired complete-call observations when the GPU is idle.
The candidate requires the complete forest on the GPU and remains explicit.
