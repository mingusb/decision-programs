# Explicit diagnostic collection modes

Recorded before implementation, 2026-09-23. Scope: `tools/profile_gpu.py` and
its CPU tests; no CUDA algorithms, production defaults, or historical receipts
change. Existing direct execution, timeout ownership, injection isolation,
artifact hashes, and non-ranking evidence contract remain intact.

## CLI and compatibility contract

- NCU: explicit `--replay-mode kernel|application|range|app-range`,
  `--graph-profiling node|graph`, and `--cache-control all|none`. Defaults remain
  kernel replay, individual nodes, and cache flushing. Application replay uses
  strict matching of name, launch dimensions, context, stream, and order.
- Range modes require `--capture`, our existing CUDA profiler API boundary.
  They use the target's start/stop API pairs directly; `--profile-from-start`
  is used only for kernel/application replay because NCU rejects it for range
  modes. The first app-range attempt exposed this CLI error and is retained in
  `results/profiling-expansion-20260923/ncu-app-range/`.
  They report aggregate ranges, not individual kernels. Kernel-name filtering
  is rejected for aggregate modes. Range replay is rejected for known graph
  benchmark requests because its captured API set excludes graph management;
  app-range supports ranges containing graph launches. Whole-graph profiling
  is a kernel-replay workload, distinct from an app-range containing graphs.
- Whole-graph source/SASS metrics are unsupported. Reject known source sections
  and source metric requests rather than silently collecting an incomplete set.
  Range modes have different metric compatibility; warn/document their lack of
  unit-level source metrics and app-range's JIT SASS limitation. Unsupported
  requested data stays a failed diagnostic, never a performance result.
- Nsight Systems: `--nsys-graph-trace node|graph`; retain the existing omitted
  option/tool default, repeated capture, unique session ownership, and CPU symbol
  resolution disabled. Final installed-help verification corrected the initial
  assumption: Systems 2026.3 uses graph granularity on driver >=11.7. Default
  invocations retain the legacy ordinary-kernel evidence gate; explicit graph
  mode requires graph evidence, and explicit node mode provides node attribution.
- Memcheck only: `--padding N`, `--leak-check full|no`.
- Initcheck only: `--initcheck-address-space global|shared|all`,
  `--track-unused-memory`, and `--unused-memory-threshold 0..100` (requires
  tracking). All are opt-in; current sanitizer defaults stay unchanged.

Installed CPU-only help was inspected: NCU 2026.3, Nsight Systems 2026.3.2,
Compute Sanitizer 13.4. No capability claim substitutes for a GPU receipt.

## Evidence and validation

Keep existing kernel CSV checks for node/kernel collection. For NCU aggregate
collection, automatically request `sm__ctas_launched.sum` and require a real
numeric workload row with a finite positive CTA count; a kernel name is not
required. Whole-graph NCU additionally requests `launch__graph_exec_cuda_id` and
requires a positive executable/source CUDA graph ID on that same workload. The
documented identities are `launch__graph_exec_cuda_id` and
`launch__graph_src_cuda_id`. Positive work from standalone setup kernels is
insufficient, even when `--graph-profiling graph` was requested. Keep all workload
identities and metrics in the export. For Systems
graph-level collection, export SQLite and require positive-duration device graph
activity records; a CPU graph API call or report file alone is insufficient.

CPU tests cover defaults, option forwarding, invalid tool/mode combinations,
source-metric rejection, matching policy, and positive/negative aggregate
evidence. Existing timeout and cleanup tests remain mandatory. Root owns serial
GPU verification using bounded histogram and booster cases. Instrumented times
remain diagnostic only; no ranking or algorithm promotion follows from them.

CPU validation: `python3 -B -W error::ResourceWarning tests/test_profile_gpu.py`
passes all 27 tests. No GPU execution was performed by the runner implementer.

Root's retained GPU receipts exposed and verified the stricter graph gate:
`ncu-whole-graph/export-0.csv` contains only transpose/radix kernels, so the earlier
positive-CTA-only pass is invalid as graph coverage. The corrected parser rejects
it. `ncu-count-graph/export-0.csv` contains two actual graph workloads with positive
graph identities and 704 CTAs each; the corrected parser accepts it. Root also
observed successful app-range replay and counting range replay. Booster stream
range replay failed with generic profiling failure/application code 11; no
specific unsupported API or root cause has been established. Raw failures remain
under `results/profiling-expansion-20260923/`.

## Usage

Each command requires a new evidence directory and an absolute target path.
Replace `/absolute/ghb_bench` with the selected built executable and append the
same bounded fixture arguments to each target command:

```sh
python3 tools/profile_gpu.py --tool ncu --output /tmp/ncu-application-new --capture --replay-mode application --cache-control none --kernel 'ghb::.*split' -- /absolute/ghb_bench --tree-execution stream
python3 tools/profile_gpu.py --tool ncu --output /tmp/ncu-graph-new --capture --graph-profiling graph --metrics sm__ctas_launched.sum -- /absolute/ghb_bench --tree-execution graph
python3 tools/profile_gpu.py --tool ncu --output /tmp/ncu-range-new --capture --replay-mode range --metrics sm__ctas_launched.sum -- /absolute/ghb_bench --tree-execution stream
python3 tools/profile_gpu.py --tool ncu --output /tmp/ncu-app-range-new --capture --replay-mode app-range --metrics sm__ctas_launched.sum -- /absolute/ghb_bench --tree-execution graph
python3 tools/profile_gpu.py --tool nsys --output /tmp/nsys-graph-new --capture --nsys-graph-trace graph -- /absolute/ghb_bench --tree-execution graph
python3 tools/profile_gpu.py --tool memcheck --output /tmp/memcheck-padding-new --padding 128 --leak-check full -- /absolute/ghb_bench --tree-execution graph
python3 tools/profile_gpu.py --tool initcheck --output /tmp/initcheck-all-new --initcheck-address-space all --track-unused-memory --unused-memory-threshold 0 -- /absolute/ghb_bench --tree-execution graph
```

Application modes relaunch the target. Avoid one-time-only target output-directory
creation and other non-repeatable external effects. Aggregate range/graph metrics
do not identify individual kernel costs. Use explicit node profiling separately
for source attribution. Ordinary NCU node mode retains `SourceCounters` and
`PmSampling`; the latter may also be requested for aggregate workloads.

Primary references:
- [NCU replay and metric compatibility](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#compatibility)
- [NCU CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)
- [Systems graph tracing](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [Systems SQLite schema](https://docs.nvidia.com/nsight-systems/AnalysisGuide/)
- [Compute Sanitizer options](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
