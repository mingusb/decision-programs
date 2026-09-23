# GPU histogram experiments

Optional profiling/debugging builds, capture ranges, and tool launchers are
documented in [training/PROFILING.md](training/PROFILING.md).

CUDA C++23 kernels and a measured, workload-specific autotuner. The initial target is the NVIDIA RTX A5000 Laptop GPU (SM86). This is an experimental portfolio, not a universal fastest-histogram claim.

**CUB** is NVIDIA's library of optimized GPU routines. Its histogram and the published NVIDIA histogram sample are **benchmark references only**. Neither is a production algorithm, fallback, or autotuner selection. The `gh` library uses our own kernels for every supported workload.

The [custom-only implementation report](results/a5000-custom-only/REPORT.md) documents the preceding production isolation and performance investigation. The [profiled A5000 report](results/a5000-profiled/REPORT.md) covers the earlier optimization pass, including the output-clear bottleneck found with Nsight Systems. The [expanded](results/a5000-expanded/REPORT.md) and [initial](results/a5000-initial/REPORT.md) reports preserve earlier measurements with their original scope.

The [larger-workload campaign](results/a5000-scaling/README.md) tests 29 uniform u32/u64 workloads, extending separately to 1,073,741,824 inputs and 1,048,576 bins. Selected custom configurations beat NVIDIA's histogram in all 58 fresh confirmation comparisons by 3.70–39.64×. The sweep also exposes a 2.46× latency increase when crossing the current shared-memory catalog boundary at 24,576 bins. Production defaults are unchanged: improvements over the existing defaults were mixed, and all losses are retained in the report.

The subsequent [large-bin implementation report](results/a5000-large-bins/REPORT.md) covers two new owned paths: shared-memory counting with global overflow bins, and global/warp counting into u32 scratch followed by widening to u64 output. Both participate in explicit selection and autotuning. Existing automatic defaults remain unchanged. Correctness and sanitizer checks pass, but old/new runtime checks flagged unresolved slowdowns in two existing configurations; performance preservation is not yet established. Fixed choices from the uniform-input experiments also lose on heavily skewed input.

The [same-process preservation investigation](results/a5000-paired-preservation/REPORT.md) compares the archived old and new implementations using common buffers, stream, and context, with same-backend controls. Across 24 invocations, the aggregate new/old event ratios were 1.008467 for single-valued graph execution and 0.989073 for the stream case. The earlier large stream penalty did not reproduce consistently; substantial control variation leaves small changes and tail behavior unresolved. The [standalone harness](bench/preservation/README.md) changes no production kernel or default.

## Build and verify

Requires CUDA 13.4, a C++23 host compiler, CMake 3.30+, Ninja, and Python 3 for autotuning. The tested stack is nvcc 13.4.59, GCC 15.2, CMake 4.2.3, and bundled CUB 3.4.2. CMake's current NVIDIA dialect table needs the explicit C++23 flag mapping included here.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86-real
cmake --build build -j 3
ctest --test-dir build --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 build/histogram_correctness --sanitizer
build/histogram_correctness --large-count
```

The separate large-count check needs about 4 GiB plus 256 MiB of free device memory and reports a skip if allocation is unavailable. The current [large-count run](results/a5000-custom-only/gpu-validation/large-count.log) completed eight executions with the exact count **4,294,967,328**: two overwrite checks each for scalar shared RLE with native and u32 local counters, packed shared RLE with u32 locals, and shared partial histograms with u32 locals and a u64 final reduction. Every tested implementation is ours. CTest also runs the host-only selector tests and the Python autotuner tests when Python is available.

The large-bin implementation passes 17,516 general correctness executions, 372 focused overflow executions, 3,170 host selector checks, and 24 autotuner tests. Each of memcheck, racecheck and synccheck passes 3,060 general and 156 focused overflow executions with zero errors or hazards. The [preceding implementation report](results/a5000-custom-only/REPORT.md) retains its separate automatic-selection and large-count evidence. To omit the benchmark executables and all NVIDIA histogram reference code, configure with `-DGH_BUILD_BENCHMARKS=OFF`. The host policy tests and Python tests can run without accessing a GPU; runtime correctness, sanitizer checks, benchmarks, and profiling require GPU access. The focused overflow test reports a skip on devices without 96 KiB opt-in shared memory.

The preceding production-isolation campaign has a separate unresolved result: its initial 8192-bin/u64 median current/old ratio of 1.025344 became 0.999587 in the repeat; retaining both cohorts gives 1.004890 across eight pairs. That residual 0.49% difference remains unresolved, so strict zero-loss performance acceptance is still open. Matching resource records for 751 device functions and instruction encodings for 13 selected kernels do not establish equal runtime speed. [Historical measurements and limits](results/a5000-custom-only/REPORT.md)

## Contract

`gh::histogram` consumes device-resident unsigned 8-bit or 32-bit **direct bin IDs** in `[0, bins)` and overwrites every element of a dense unsigned 32-bit or 64-bit output. Invalid IDs are a caller precondition; there is no range-checking pass. Input, output, and workspace must not overlap. Byte inputs currently support at most 256 bins. Empty input clears the output. Unsigned 32-bit counts require `N <= UINT32_MAX`; unsigned 64-bit output supports larger counts within the configuration's address and resource limits.

The API is asynchronous on the supplied CUDA stream. `Config` now defaults to automatic selection. Set the problem dimensions, then call `gh::prepare(config)` on a **mutable** configuration before querying `gh::workspace_bytes` and provisioning device memory. Preparation freezes a concrete algorithm and policy for the current GPU and configures any required opt-in shared memory. Unresolved automatic configurations are rejected by workspace queries and execution. Pointers need the natural alignment of their element types. Workspace returned by `cudaMalloc` is suitable. No allocation, device query, capture-state query, or function-attribute setup occurs inside execution. `gh::supported` checks static configuration limits; preparation and CUDA launch errors are returned separately.

## Measured defaults

Automatic selection uses measured settings for the RTX A5000 Laptop GPU (SM86, 48 SMs), with resource-safe choices from our own kernels for other shapes. It selects by input size, bin count, input/output widths, and declared launch/cache modes. It does not inspect the data or use the benchmark's distribution labels. Outside the measured table, it prefers shared-memory counting when capacity and counter bounds permit, then uses our global-atomic kernel when they do not. Grid size is bounded by the work and device limits. Unmeasured choices are heuristics, not claims of optimal performance; no NVIDIA histogram is used.

```cpp
gh::Config config;                    // algorithm = automatic
config.size = sample_count;
config.bins = bin_count;
config.input_type = gh::InputType::u32;
config.counter_type = gh::CounterType::u64;
config.launch = gh::LaunchMode::graph; // declare intended execution; does not capture a graph
config.cache = gh::CacheMode::warm;
// Check each returned CUDA status in application code.
auto status = gh::prepare(config);    // resolves and freezes the choice before capture
// On success, query workspace_bytes(), allocate, then call histogram().
```

The default context is warm ordinary-stream execution. Automatic graph choices use kernel clearing; automatic stream choices use runtime clearing. Set an explicit `config.algorithm` to bypass automatic selection and retain complete control over tuning, grid, local-counter width, and clearing. For explicit algorithms, `output_clear` retains its runtime default; choose kernel clearing explicitly for graphs. After changing the problem or device, reset `config.algorithm = gh::Algorithm::automatic` and prepare again before resizing workspace. Preparation does not detect changes to a previously frozen explicit configuration.

| Input elements | Bins | Input → output | Execution/cache | Policy / blocks / local counter |
|---:|---:|---|---|---|
| 1,048,576 | 4,096 | u32 → u32 | stream / warm | 10 / 48 / native |
| 1,048,576 | 8 | u32 → u32 | graph / warm | 6 / 192 / native |
| 4,096 | 256 | u8 → u32 | graph / warm | 1 / 96 / native |
| 1,048,576 | 256 | u8 → u32 | graph / warm | 11 / 48 / native |
| 1,048,576 | 256 | u32 → u32 | graph / warm | 11 / 48 / native |
| 16,777,216 | 256 | u8 → u32 | graph / warm | 10 / 96 / native |
| 16,777,216 | 4,096 | u32 → u32 | graph / warm | 11 / 48 / native |
| 16,777,216 | 4,096 | u32 → u64 | graph / warm | 10 / 48 / u32 |
| 16,777,216 | 8,192 or 16,384 | u32 → u64 | graph / warm | 15 / 48 / u32 |
| 1,048,576 | 4,096 | u32 → u64 | graph / cold | 10 / 48 / u32 |

Every measured entry uses shared-memory atomic accumulation. The 16,777,216-byte, 256-bin case uses its best custom validation finalist, even though the old tuner retained NVIDIA's reference under a 5% margin rule. Policies requiring 96 KiB also require sufficient device capacity. The shape shared by several 256-bin distributions uses one fixed policy; it does not switch according to skew or ordering. The [confirmed graph measurements](results/a5000-profiled/clear-policy-2026-09-21/final-comparison.md) and [earlier default-selection evidence](results/a5000-defaults/REPORT.md) preserve their original binaries and policies. Ten current automatic-selection paths have passed GPU execution and CPU-reference verification; generic choices remain heuristics, and performance preservation is under investigation. Retune after substantial workload, hardware, or driver changes.

## Kernel and template policies

The production catalog includes direct global atomics, warp matching, shared-memory atomics, shared RLE, shared warp matching, shared partial histograms plus a reduction, shared-memory counting with global overflow, and register bit-planes for 1–256 bins. Input, output-counter, and supported local-counter widths instantiate distinct kernels. Most policies cap shared memory at 48 KiB per block; policies 14–15 explicitly opt in to 96 KiB. The separate `gh_benchmark_references` target contains NVIDIA's histogram implementations and is linked only into the benchmark executable.

Threads per block, items per thread, and replica count are compile-time parameters. The bounded catalog is defined in [include/gh/histogram.hpp](include/gh/histogram.hpp); it deliberately avoids the full Cartesian product. Small-bin bit-plane capacity is rounded up and specialized at compile time. Grid size remains a runtime parameter. Kernel dispatch and CLI policy bounds derive from the catalog, so adding a policy adds the corresponding template instantiations.

| Policy | Threads | Items/thread | Shared replicas | Loads | Shared limit |
|---|---:|---:|---:|---|---:|
| 0 | 128 | 4 | 1 | scalar | 48 KiB |
| 1 | 256 | 4 | 1 | scalar | 48 KiB |
| 2 | 256 | 8 | 1 | scalar | 48 KiB |
| 3 | 256 | 16 | 1 | scalar | 48 KiB |
| 4 | 128 | 8 | 4 | scalar | 48 KiB |
| 5 | 256 | 8 | 4 | scalar | 48 KiB |
| 6 | 256 | 8 | 1 | full tile | 48 KiB |
| 7 | 256 | 8 | 1 | vector4 | 48 KiB |
| 8 | 256 | 16 | 1 | full tile | 48 KiB |
| 9 | 256 | 16 | 1 | vector4 | 48 KiB |
| 10 | 512 | 8 | 1 | vector4 | 48 KiB |
| 11 | 1024 | 8 | 1 | vector4 | 48 KiB |
| 12 | 256 | 8 | 8 | vector4 | 48 KiB |
| 13 | 512 | 8 | 16 | vector4 | 48 KiB |
| 14 | 256 | 8 | 1 | vector4 | 96 KiB |
| 15 | 512 | 8 | 1 | vector4 | 96 KiB |

Policies 0–5 preserve the original scalar kernels as controls. Full-tile paths remove per-item bounds checks from complete tiles. Shared `vector4` loads four adjacent bytes with one 32-bit load, or four u32 values with a 128-bit load, when aligned; unaligned inputs and tails use scalar loads. This changes per-thread ordering and therefore RLE opportunities. Bit-plane policies map `vector4` to `full_tile`, as recorded in CSV metadata. Bit-plane static shared storage remains limited to 48 KiB. Global and warp-global methods use scalar policies only; opt-in policies 14–15 are shared-only.

Replicas apply to shared atomic policies. Register and global policies ignore them; duplicate configurations are omitted from the search. CUB chooses its own policy, and the NVIDIA sample retains its published fixed policy. Their CSV threads/items/replicas are zero (not applicable); requested tuning/grid values do not describe their actual launches.

### Narrow local counters with u64 output

The four shared-memory families accept `--counter u64 --local-counter u32`. They accumulate in unsigned 32-bit shared counters, then widen to the public unsigned 64-bit output. The partial-histogram variant also stores u32 partials and accumulates its final reduction in u64, halving partial-counter storage relative to native u64 locals.

This policy is accepted only when every block's assigned sample count fits `UINT32_MAX`, even if all samples hit one bin. For `T = threads * items` and `G = blocks`, the maximum is

```text
floor(N / (G*T)) * T + min(N % (G*T), T).
```

The check follows the actual tile assignment, including an incomplete final grid round. It does not assume uniform bin frequencies or use `ceil(N/G)` as a substitute. Boundary tests reject the first unsafe size. `--local-counter native` uses the output width locally. The autotuner considers both widths wherever supported.

Global and warp-global counting also accept `--counter u64 --local-counter u32` with scalar policies 0–5. These paths require **total `N <= UINT32_MAX`**, because all blocks update one shared global scratch histogram. Nonempty calls require `bins * 4` workspace bytes. Execution clears that scratch, counts, and widens every bin to overwrite the u64 output, including zero bins. This halves the randomly updated counter storage relative to native u64 counting; empty calls need no scratch. The per-block shared-counter bound does not apply to global scratch.

### Shared counting with global overflow

`--algorithm shared_overflow` supports u32 input, u64 output, u32 local counters, policies 14/15 and more than 24,576 bins. Each block accumulates the first 24,576 bins in a 96 KiB shared histogram. Keys at or above 24,576 update the u64 output directly; the shared prefix is merged after the scan. It requires no workspace and uses the shared per-block overflow bound above. `prepare()` checks the device capacity and configures only the new kernels, outside capture and timed execution. Full tiles use aligned vector loads, with naturally aligned scalar fallback and tail handling.

This path's performance depends on how many samples land in the prefix versus overflow bins. Fixed settings chosen on uniform inputs need separate evaluation on skewed data; neither this path nor narrowed global counters replace automatic defaults. Example explicit configurations from the uniform warm-graph experiments:

```bash
build/large-bins-overflow/histogram_bench --n 16777216 --bins 24577 --input u32 --counter u64 --algorithm shared_overflow --tuning 15 --blocks 24 --local-counter u32 --launch graph
build/large-bins-overflow/histogram_bench --n 16777216 --bins 1048576 --input u32 --counter u64 --algorithm global --tuning 0 --blocks 48 --local-counter u32 --launch graph
```

### Published NVIDIA baseline

The benchmark-only `--algorithm cub` and `--algorithm nvidia_sample256` options select isolated comparison implementations. These names do not exist in the production `gh::Algorithm` enum. The sample uses unchanged NVIDIA device kernels pinned at `5443602d89ed99aede2e4b7bf329daddeadb320e`. It accepts only u8 input, exactly 256 bins, u32 output, native local counters, `N % 4 == 0`, and four-byte-aligned input. The u32-output size limit also applies. There is no input conversion or padding.

The original policy uses 240 partial-histogram blocks with 192 threads each, followed by 256 merge blocks with 256 threads each. Nonempty execution needs 245,760 workspace bytes. Both kernels, including shared initialization and the complete merge, are timed. The adapter supplies caller-owned workspace, stream handling, error returns, and empty-output clearing. Original source, license, hashes, and precise adaptation notes are in [third_party/nvidia_histogram](third_party/nvidia_histogram/README.md).

## Benchmark and autotune

```bash
build/histogram_bench --n 16777216 --bins 4096 --distribution uniform > default.csv
build/histogram_bench --n 16777216 --bins 4096 --algorithm all > comparison.csv
python3 tools/autotune.py --n 16777216 --bins 4096 --distribution uniform --output results/uniform-4096.json
python3 tools/autotune.py --replay results/uniform-4096.json
build/histogram_bench --n 16777216 --bins 4096 --counter u64 --local-counter u32 --algorithm shared --cache cold --launch graph > cold-graph.csv
```

Use `--input u8|u32`, `--counter u32|u64`, `--distribution uniform|single|two|hot90|hot99`, `--order shuffled|sorted|roundrobin`, `--cache warm|cold`, and `--launch stream|graph`. Defaults are `--algorithm auto`, warm cache and stream launches. Automatic mode emits one row with the concrete selected settings. Use `--algorithm all` for the previous comparison behavior; `--sweep` still searches all supported algorithms and policies. Manual tuning, grid, or local-counter flags require an explicit algorithm or `all`. Ordering transforms preserve the generated multiset. The hot distributions put 90% or 99% of draws in the highest bin and draw remaining samples uniformly, including that bin. Seeds make data generation reproducible within the recorded toolchain.

The tuner searches supported policies at grids of 1, 2, 4, and 8 blocks per SM, including both local-counter widths for shared/global/warp u64 output and valid shared-overflow policies. Fixed reference implementations are included once with native counters during sweeps. Every candidate is checked against independent CPU counts before timing in randomized CUDA-event rounds. The four fastest custom candidates, the best candidate from each additional algorithm/local-counter family, and all applicable references are measured together on two fresh seeds. Search, validation and replay use the same `--batch` (default 32); `--search-samples` and `--validation-samples` default to 3 and 11. The chosen plan is always one of our custom implementations, ranked by median reference-normalized speedup across validation seeds. NVIDIA implementations remain in the measurements to show wins and losses. The 1.05x margin is diagnostic, never permission to select a reference; reference-only searches are rejected.

Schema 5 JSON plans contain the chosen configuration, local width, recorded GPU name/SM, CUDA driver API/runtime/CUB metadata, executable SHA256, measurement protocol, and hashes of raw search and validation CSVs. Workload scope includes cache and launch modes and optional `--warmup-ms` (default 0). Timing protocol 3, effective load policy, and requested output-clearing policy are recorded and checked. Replay checks those recorded fields and the executable hash. `driver_api` is the supported CUDA API version, not the NVIDIA driver release; no GPU UUID or driver-release check is performed, so replay cannot detect every hardware or driver change. The plan is a measured choice for the recorded workload and binary; it does not detect distributions at runtime. Validation measurements select the finalist, so they are not an independent final performance test. Retune after changing kernels or workload scope. Tuning cost is excluded from steady-state call times.

### Timing boundaries

Every measured operation includes output initialization, counting, and all merge/finalization kernels. Input generation, transfers, allocation, CPU validation, graph capture/instantiation, cache eviction, and tuning are excluded. Stream launches may include host submission gaps inside their event intervals; graph mode records timing events inside the graph, excluding the host launch boundary. Protocol 3 supersedes the older external-event graph measurements; compare kernel variants within the same protocol.

Nonempty custom atomic and bit-plane operations support runtime memset or a CUDA clearing kernel before counting. Narrowed global paths apply this choice to scratch initialization and overwrite output during widening; shared-overflow applies it to output initialization. Nsight Systems exposed large gaps in repeated warm graphs containing runtime memset nodes; matched measurements justified kernel clearing there. Stream measurements showed a different tradeoff, so the benchmark's `--clear auto` selects runtime clearing for stream launches and kernel clearing for graphs. Override with `--clear runtime|kernel`, or specify a fifth variant field: `shared:7:96:native:kernel`. Shared-partial kernels overwrite output in their reduction. NVIDIA's implementations and empty-input clearing retain their own initialization paths. The complete operation still includes initialization cost. CSV `clear_policy` records the requested custom policy and is ignored by those internal initialization paths.

| Mode | One recorded timing sample |
|---|---|
| Warm + stream | Warm up, time a batch of complete operations, divide by batch size. |
| Warm + graph | Warm up, then use graph-owned start/end event nodes around the complete batch; divide by batch size. |
| Cold + stream | Evict before every operation, time each operation with its own event pair, average the batch. |
| Cold + graph | One graph contains a batch of eviction → start event → complete operation → end event sequences, with unique event pairs; average their durations. |

Cold mode uses a separate buffer of `max(64 MiB, 8 * reported L2 size)` with `.cg` loads and dependent `.wb` stores before every measured operation. This is an explicit capacity-eviction procedure, not a hardware guarantee that every cache line is invalidated. `eviction_bytes` records the buffer size; zero denotes warm mode. Eviction remains outside the timing interval, so cold results describe operation time after eviction rather than an application's eviction-plus-operation cost. Warm mode reuses resident buffers and makes no claim that the entire input fits in cache.

`histogram_cache_probe --order evict-first|evict-between` compares whole graphs containing E→H→H and H→E→H, with identical total work. On this A5000, moving eviction between the two 1 MiB input reads removed exactly 32,768 L2 read-hit sectors and added the same number of misses. This verifies the procedure for that controlled case. It does not establish a universal cache-state guarantee.

Raw samples are retained; seven or nine samples do not establish a precise tail-latency distribution. `input_gb_s` counts logical input bytes and is not measured DRAM bandwidth. GPU clock frequencies are not locked.

### Workload campaign and archived plans

```bash
python3 tools/campaign.py --output results/my-campaign --sm-count 48
```

[tools/campaign.py](tools/campaign.py) defines a 184-cell matrix covering sizes, bin counts, skew, ordering, input/output widths, and cache/launch comparisons. It runs predeclared representative configurations, preserves every command and CSV, and can resume a matching manifest. This is not a full autotuning search in every cell. Selecting each cell's fastest measured row after the run produces an oracle comparison, not a deployable selector. See the [expanded report](results/a5000-expanded/REPORT.md) for campaign results and limits.

The first campaign's executable and schema 1 tuner are archived under `build/initial`. Current schema 5 tooling rejects older plans and any reference chosen as a production implementation. Schema 4 binaries and tools are archived under `build/profiled-clear-policy` and `build/defaults`. The schema 2 binary, tuner, campaign runner and analyzer are archived under `build/expanded`; use that analyzer for the expanded campaign’s CSV files. Replay them with the matching archived pair, for example:

```bash
python3 build/initial/autotune.py --exe build/initial/histogram_bench --replay results/a5000-initial/u32-4096-uniform.json
```

The executable hash must still match the plan. Those initial measurements used warm resident buffers and stream launches; their scope does not inherit the later cache, graph, local-width, or published-baseline additions.

The focused profiling pass has a separate runner that records the GPU UUID, NVIDIA driver release, system identity, and benchmark hash, and checks them across invocations. Use a fresh directory for each environment:

```bash
python3 results/a5000-profiled/run_round.py tune --output-root results/my-profiled-session
python3 results/a5000-profiled/run_round.py confirm --output-root results/my-profiled-session
python3 results/a5000-profiled/analyze_round.py --output-root results/my-profiled-session
```

This runs twelve bounded searches followed by confirmation on separate seeds. Existing evidence is protected from overwrite. The initial graph-optimized binary and schema 3 tuner are archived in `build/profiled`; the explicit clearing-policy revision uses schema 4 and is archived in `build/profiled-clear-policy`.

## Profiling

The [window-rescan research follow-up](results/a5000-window-rescan/REPORT.md)
adds an explicit `global_window` experiment: a 2 MiB u32 counter buffer and two
input scans improve the tested million-bin uniform workloads by 2.2–2.4× over
our existing narrow kernel. It remains outside automatic selection and generic
autotuning; the report retains losses near the window boundary, the four-scan
alternative, located-skew tests, correctness/sanitizer evidence, and Nsight
profiles. A separate controlled synchronization screen leaves small-regression
measurement precision unresolved. Existing defaults are unchanged.

Use unprofiled runs for rankings. Nsight Systems reveals the full operation and submission gaps; Nsight Compute diagnoses individual kernels and changes execution through instrumentation/replay.

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none -o timeline \
  build/histogram_bench --algorithm shared --n 16777216 --bins 4096 --samples 3 --batch 1
ncu --set basic --clock-control none --cache-control all \
  --kernel-name-base function --kernel-name 'regex:.*shared_histogram.*' \
  --launch-skip 2 --launch-count 1 -o kernel \
  build/histogram_bench --algorithm shared --n 16777216 --bins 4096 --samples 3 --batch 1
```

The kernel filter selects counting and skips its two correctness launches; without it, the first profiled kernel is now output clearing. Nsight kernel durations exclude the other stages of the complete histogram operation.

Further work includes extending the narrow default-selection table across workload regimes and additional GPU architectures. Sparse outputs, weighted counts, floating-point binning, and SM90+ hardware features remain outside this implementation.

Design references: the supplied research PDFs, the installed CUB source, and NVIDIA's [automated tuning infrastructure](https://nvidia.github.io/cccl/unstable/cub/tuning_infra.html).
