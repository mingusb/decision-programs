# Optional profiling and debugging integration

Optional diagnostic integration is implemented and verified on the RTX A5000
Laptop GPU under WSL with Windows driver **616.92**. Restarting WSL restored
driver access after the live update from 597.06. The final serialized matrix
passed **24/24 runs**: 12 collectors on the custom histogram and on the CUDA
booster. CUDA-GDB source debugging and offline machine-code analysis also passed.

See the [usage guide](../../training/PROFILING.md),
[final matrix](driver61692-final-results.json),
[application validation](final-application-validation.json), and
[section-specific evidence](section-specific-proof.json).
These diagnostic runs establish capability on the recorded workloads;
instrumented durations are not performance rankings or a new speed/accuracy claim.

## Implemented

- `support/profiling.hpp`: opt-in count/train/predict capture boundaries in
  `histogram_bench`, the count instrumentation probe, `ghb_bench`, and
  `ghb_real_bench`. `GH_PROFILE_CAPTURE=1` enables CUDA profiler start/stop,
  boundary synchronization and registered NVTX ranges. Disabled capture makes
  no CUDA/NVTX calls.
- Both CMake projects: `GH_PROFILE_HOST_SYMBOLS` adds host symbols while retaining
  optimized device code; `GH_DEVICE_DEBUG` selects separate `-G/-O0` debugging
  code. Both default OFF. CUDA remains C++23.
- `tools/profile_gpu.py`: independent launch adapters for Nsight Systems,
  Nsight Compute, four Compute Sanitizer checks, NVBit instruction/graph/memory
  tools, and CUPTI trace/range/continuous-PC injection tools. It rejects stacked
  injection and unsupported known graph modes, requires actual diagnostic
  records, preserves failures and cleans up owned processes on timeout.
- `training/profiling/disassemble.py`: offline cubin extraction, resource usage,
  source-correlated disassembly, register liveness and control-flow DOT/SVG.

Most collectors inspect binaries or inject their own instrumentation. Production
kernels need no additional instructions for these tools. Nsight Compute provides
arbitrary-target PM sampling and SASS/source metrics; standalone CUPTI PM/SASS
SDK examples validate their APIs but are not generic injection adapters.

## Verified diagnostic coverage

| Collector | Histogram | Booster | Evidence required |
| --- | --- | --- | --- |
| Nsight Systems | Pass | Pass | Kernel timeline, NVTX, graph nodes |
| Nsight Compute | Pass | Pass | Selected kernel, source counters, PM samples |
| Compute Sanitizer memcheck | Pass | Pass | Completed check, no reported errors |
| Compute Sanitizer racecheck | Pass | Pass | Completed check, no reported hazards |
| Compute Sanitizer synccheck | Pass | Pass | Completed check, no reported errors |
| Compute Sanitizer initcheck | Pass | Pass | Completed check, no reported errors |
| NVBit graph instruction counter | Pass | Pass | Dynamic warp counts |
| NVBit instruction counter | Pass | Pass | Dynamic warp counts, explicit stream |
| NVBit memory trace | Pass | Pass | Instruction memory addresses, explicit stream |
| CUPTI activity trace | Pass | Pass | Target activity records |
| CUPTI range profiling | Pass | Pass | Hardware metrics, explicit stream |
| CUPTI continuous PC sampling | Pass | Pass | Decoded instruction/stall samples |

The [plan](matrix-plan.json) records exact inputs. Most count checks use 4,097
elements and 257 bins; most booster checks use 128 training rows, four features,
three outputs, output tiles of two, depth two and a fourth-order binary loss
update. NCU and count PC-sampling cases use larger inputs to obtain useful
records. These bounded cases exercise graph execution and an output-tile tail;
they do not establish complete workload or kernel coverage.

Final Nsight Compute reports contain 21 PM timelines per selected kernel.
SM-active timelines contain 38 positive samples out of 437 for counting and
91 out of 582 for the booster. Source counters report 581,120 and 224,064
executed instructions, with 163 and 855 SASS instruction addresses. PC sampling
reports 2,352 and 5,034 samples, with zero dropped bytes. Metric names, raw values,
source correlations and extraction methods are in
[section-specific-proof.json](section-specific-proof.json).

Nsight Systems records nine histogram graph kernels. Booster training has
61 kernels, including 32 graph nodes, and named stage ranges; the separate
prediction capture has six kernels and its `predict` range.

[CUDA-GDB on driver 616.92](cuda-gdb-driver61692/manifest.json) stopped in the
actual `transpose_keys` kernel, displayed device arguments and local state,
stepped a source line, and continued to successful booster validation.
Earlier optimized/device-debug runs are retained.
[Offline analysis](offline-tool-smoke/manifest.json) passed seven
extraction/disassembly/rendering commands; output-directory reuse was rejected.

## Correctness and preservation

All three builds passed: optimized count profiling, optimized booster profiling,
and device-debug booster. Compile-command checks confirm C++23, optimized line
information for profiling and `-G` for device debugging.

- [18 CTest suites passed](ctest-booster.log), including 14 GPU tests, before
  the driver change; the final collector matrix and debugger run use 616.92.
- [18 diagnostic-runner CPU tests passed](runner-cpu-tests-final.log), including
  timeout cleanup, separate process sessions, detached agents, unrelated-process
  preservation, PID reuse and rejection of incomplete evidence.
- [3,170 default-selection checks passed](count-defaults.log).
- [Capture disabled/enabled checks](capture-validation/manifest.json) produced
  exact count outputs, zero booster CPU/GPU prediction error, successful
  serialization round-trip prediction checks within each run, and identical
  matched-workload training and held-out losses.
- All 24 final collector runs completed application validation. Matched small
  booster cases retain baseline losses; the larger NCU case is recorded separately
  in [application validation](final-application-validation.json).
- The [source audit](source-and-build-audit.json) confirms all 33 recorded
  production files are unchanged. Counting policies and defaults are unchanged.
- [Device instruction comparison](device-code-comparison/DEMANGLED.md) confirms
  identical compiled instruction bytes across 782 counting and 149 booster
  executable sections. It explicitly maps 48 compiler-private anonymous-namespace
  symbol renamings; the original strict-name mismatch is preserved. This excludes
  host code, constants and resource metadata and does not itself prove performance
  equivalence.

Collectors ran serially. A [post-run process audit](post-matrix-process-check.json)
checks recorded PID/start-time identities for leftover collectors or targets.
Commands, target/tool hashes, logs and failures remain in each run directory.

## Tools and practical limits

Installed additions: CUPTI 13.4.58, NVBit 1.8, CUDA-GDB 13.4.49, CUDA Profiler API
headers 13.4.49 and Graphviz 14.1.2. Existing Nsight versions are Compute 2026.3.0
and Systems 2026.3.2; Compute Sanitizer is 2026.3.0. Installation receipts and
earlier SDK sample runs are retained here and under
`~/.local/opt/gpu-profiling/`. [Verification status](verification-status.json)
records the tested configuration separately from installation history.

Ordinary NVBit count/memory tools and the CUPTI range injector require explicit
stream execution. The graph-aware NVBit example has capacity for 100 unique
functions; the runner rejects observed overflow. Matrix NVBit captures use the
static instruction interval `[0,64)`, so counts are explicitly partial.
NVBit working here does not establish compatibility with every driver.

PC sampling may produce no records for very short kernels. PM sampling is
device-wide, and WSL lacks context-switch filtering; serial collection cannot
exclude unrelated Windows GPU activity. Nsight CPU symbol resolution defaults
off because automatic symbol retrieval delayed capture completion here;
`--nsys-resolve-symbols` enables it explicitly. GPU names, NVTX ranges and
graph-node tracing remain available.

## Preserved and resolved failures

- The [initial matrix](matrix-results.json) failed while the live Windows driver
  update left WSL unable to load CUDA. Linux `nvidia-smi` and the plain count
  application also segfaulted. [Driver diagnostics](driver-change/) are retained.
  Restarting WSL resolved this.
- The [first post-restart matrix](driver61692-results.json) passed 23/24 checks;
  booster Nsight Systems timed out during automatic CPU symbol resolution.
  Disabling that option completed both captures in about four seconds.
- That timeout exposed a target escaping process-group cleanup. The launcher now
  tracks owned PID/start-time identities and its unique Nsight session, including
  separate-session descendants and detached agents. Earlier escaped processes
  were cleaned up before the final matrix; the
  [cleanup receipt](nsys-timeout-cleanup.json) is retained. Only the final matrix
  is used as the completed serialized verification.
- Initial CMake generator expressions needed quoting. CUDA-GDB needed a wrapper
  rather than a symlink because its vendor launcher locates sibling binaries.
- An early NVBit attempt injected an external `timeout` process and yielded no
  counters. Direct target injection fixed this; the reusable launcher manages
  timeouts without injecting an intermediate launcher.
- Offline final-report extraction initially exited 139 because the Python
  extractor did not retain an Nsight API timeline owner while using its metric
  proxy. Retaining that owner fixed extraction. Failed attempts and the original
  extractor are preserved; targets and captured reports were unaffected.

No failed check is relabeled as a pass. Final evidence applies to the documented
hardware, tools and workloads, not every possible tool feature or workload.
