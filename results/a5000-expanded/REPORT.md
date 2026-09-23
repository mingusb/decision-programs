# Expanded A5000 histogram results

Safely bounded **u32 block-local counters with u64 output** improve the useful performance range of this CUDA C++23 portfolio. Independent confirmation reached **11.45× CUB at 4,096 bins** and **22.25× at 8,192 bins**, for 16,777,216 uniform shuffled u32 inputs and u64 output. These are workload-specific results on one GPU, not a universal fastest-histogram claim.

The tested device is the NVIDIA RTX A5000 Laptop GPU, SM86, 48 SMs, 16 GiB, with **4 MiB reported L2**. Driver 596.71, nvcc 13.4.59 (CUDA 13.4), CUB 3.4.2, GCC 15.2, Ubuntu/WSL2. Clocks were not locked. Exact versions, device UUID, binary and source hashes are in [environment.json](environment.json).

## Independent confirmation

Each plan searched the supported template catalog on seed 12345 (3 randomized rounds × 2 operations), then validated four custom finalists and all applicable references on seeds 67890 and 24680 (7 × 5). Acceptance required at least 1.05× the fastest reference on **each** validation seed. Selected plans were frozen before the following confirmation on seed 424242 (21 randomized rounds × 20 operations). The confirmation did not change selection. Workload families were chosen after inspecting the broader matrix; this confirms new data seeds, not generalization to unseen workload families.

All rows below use **CUDA Graph execution**. Times are median microseconds per complete device operation, including initialization and all merges. Graph creation, allocation, input transfers, data generation, tuning, and capacity-eviction work are excluded. Warm samples time a graph of 20 complete operations; cold samples average 20 individually timed single-operation graphs, each preceded by eviction. “Warm” means repeated resident buffers, not a guarantee that the input fits in cache. Large u32 inputs occupy 64 MiB.

| Input/output | N | Bins | Distribution/order | Cache | Frozen selection | CUB µs | Selected µs | CUB/selected |
|---|---:|---:|---|---|---|---:|---:|---:|
| u32/u64 | 16,777,216 | 4,096 | uniform/shuffled | warm | [shared t2 g96 local-u32](u64-4096-warm-graph.json) | 2137.34 | 186.68 | 11.45× |
| u32/u64 | 16,777,216 | 8,192 | uniform/shuffled | warm | [shared t3 g48 local-u32](u64-8192-warm-graph.json) | 4199.27 | 188.72 | 22.25× |
| u32/u32 | 16,777,216 | 4,096 | uniform/shuffled | warm | [shared t3 g96 local-native](u32-4096-warm-graph.json) | 955.65 | 187.24 | 5.10× |
| u32/u64 | 1,048,576 | 4,096 | uniform/shuffled | cold | [shared t2 g48 local-u32](u64-4096-cold-graph.json) | 117.20 | 23.14 | 5.06× |
| u32/u32 | 16,777,216 | 4,096 | hot99/sorted | warm | [shared t2 g192 local-native](u32-4096-hot99-sorted-graph.json) | 223.13 | 185.65 | 1.20× |
| u8/u32 | 16,777,216 | 256 | uniform/shuffled | warm | [CUB](u8-256-warm-graph.json) | 50.43 | 50.43 | 1.00× |
| u32/u32 | 1,048,576 | 8 | uniform/shuffled | warm | [CUB](u32-8-warm-graph.json) | 13.31 | 13.31 | 1.00× |

The byte-input reference comparison also measured NVIDIA's published histogram256 sample: **56.27 µs**, versus CUB's **50.43 µs**. The saved byte plan retained CUB. The eight-bin matrix result appeared promising, but the search finalists failed the two-seed gate; that plan also retained CUB. It would be incorrect to promote the fastest matrix row as the deployed choice.

Raw `*.search.csv`, `*.validation.csv`, and `*.confirmation.csv` files are retained. Each plan contains exact search/validation commands and hashes; `*.confirmation.command.json` records the separate confirmation. [confirmation-summary.json](confirmation-summary.json) provides the table data.

## What changed and why it helps

The four shared-memory families now specialize local counter width separately from output width. A deterministic bound on the maximum samples assigned to any CTA rejects configurations that could overflow a u32 local counter, including adversarial all-in-one-bin input. It permits total input sizes above UINT32_MAX when the per-block bound is safe. Partial histograms store u32 values and widen before the final u64 reduction.

Matched confirmation at N=16M/B=4096, shared t2/grid96, u64 output:

- Native u64 locals: **269.67 µs**.
- u32 locals: **186.68 µs**, a **1.44×** speedup with the same policy and grid.
- At N=1M/B=4096 in cold graph mode, the matched grid48 comparison was **43.37 → 23.14 µs**, or **1.87×**.

At 8,192 bins, a single u64 shared histogram would need 64 KiB, exceeding this implementation's conservative 48 KiB per-block cap. u32 locals require 32 KiB and enable this family. The 22.25× CUB result therefore is not a matched native-width ablation.

Nsight Compute profiled the same t2/grid96 kernels at N=16M/B=4096 with application replay. Its instrumented counting-kernel measurements were:

| Local width | Dynamic shared memory | Registers/thread | Achieved occupancy | Peak DRAM utilization | Kernel duration |
|---|---:|---:|---:|---:|---:|
| u64 | 32 KiB | 32 | 33.19% | 60.16% | 298.46 µs |
| u32 | 16 KiB | 32 | 33.22% | 93.54% | 194.56 µs |

The achieved occupancies are essentially equal; the gain should not be attributed solely to an occupancy increase. [Disassembly](local-counter-sass.txt) shows that native u64 shared increments compile into explicit load/add/`ATOMS.CAST.SPIN.64` retry sequences, whereas u32 locals use `ATOMS.POPC.INC.32`. Both preserve 64-bit final global additions. This establishes a different update path; it does not measure the retry contribution to the total speedup. [Native](local-native.details.txt) and [u32](local-u32.details.txt) profile exports and `.ncu-repz` captures are retained. Profiled times diagnose kernels; the unprofiled table ranks complete operations.

## Broader workload matrix

The reproducible [matrix manifest](matrix/manifest.json) defines **184 workloads and 1,700 candidate measurements**, with N=4K/1M/16M, 8–8,192 bins, uniform/hot99/single-bin data, shuffled/sorted ordering, both input/output widths where scoped, stream/graph launches, and explicit warm/cold comparisons. It uses a predeclared representative portfolio, not a full search in every cell. Every candidate passed independent CPU histogram comparison before timing.

The best observed custom row exceeded the fastest applicable reference by at least 1.05× in **140/184 cells**. That is an ex-post oracle statistic, not a dispatch success rate. Across 100 exact-policy/grid native/u32 local-width pairs, u32 was faster in 98, by at least 1.05× in 93; one pair tied and one favored native. The median paired speedup was **1.446×**. Another 40 u32-local rows had no supported native counterpart and were excluded from pairing.

See [matrix-summary.md](matrix-summary.md) and its [JSON](matrix-summary.json) for mode-specific results, losses as well as wins, and all paired comparisons. `tools/analyze_campaign.py` verifies the manifest candidate sets and measurement protocols before summarizing. Equal-weight aggregates describe this selected matrix; they are not application-weighted speedups.

## Cache and graph profiling limits

Cold mode touches a separate **64 MiB** buffer with L2-allocating `.cg` reads and dependent `.wb` stores before every measured operation. Each eviction is outside its timing events. This is a capacity-eviction procedure, not a hardware cache-invalidation guarantee.

Individual-kernel Nsight probes used application replay and `--cache-control none`. At both 4 MiB and 1 MiB input sizes, the warm and cold probes reported all input read sectors missing L2. This did **not** establish an unprofiled warm/cold cache-residency contrast. The reason for cache loss at profiling entry remains unresolved; it must not be presented as a diagnosed WSL defect.

A separate whole-graph probe with 1 MiB input and five repeated operations, using graph/kernel replay with cache clearing before the graph, reported **32,768 L2 read misses and 131,072 hits**: exactly one miss pass and four hit passes. This verifies reuse inside that particular warm graph. It does not isolate or establish the cold eviction procedure's effectiveness. Commands, raw metrics, and summaries are retained as `cache-*` artifacts; the failed application-replay/graph-mode compatibility attempt is also retained rather than treated as data.

[Nsight Systems with graph-node tracing](timeline-cold-graph-nodes.nsys-rep) recorded the cold graph launches and 12 eviction kernels for 2 candidates × 3 rounds × 2 timed repetitions. The [submission-order check](timeline-order-check.json) verified every same-stream eviction → start event → graph → end event sequence. GPU event timestamps were unavailable in this trace, so that check establishes API order rather than independently measuring event boundaries. The eviction kernels took about 394 µs each under tracing, which underscores that excluded eviction work is substantial. Event placement is explicit in `bench/benchmark.cpp`; the reported cold times are not eviction-plus-histogram application costs. Kernel summaries include validation and warmup executions and must not replace complete-operation timings.

## Correctness and reproducibility

- **12,958** normal correctness executions: full-vector CPU comparison, sum checks, output/workspace canaries, tails, repeat calls, nondefault streams, both type widths, local policies, and invalid configurations.
- **8** large-count executions produced exactly **4,294,967,328**, covering native/u32-local shared RLE, u32 partial histograms with u64 reduction, and CUB. This actually exercises a count above UINT32_MAX.
- Compute Sanitizer: **1,698 executions per tool**, with zero memcheck errors, racecheck hazards, and synccheck errors. A separate cold-graph benchmark with CUB, the NVIDIA sample, and shared partials also passed memcheck.
- Eight mode/type smoke configurations verified warm/cold × stream/graph execution and captured-graph overwrite behavior.
- **11** CPU autotuner integration tests passed. Schema 2 replay passed for warm u64, cold u64, and byte-reference plans; malformed/stale metadata rejection is covered by the CPU tests.
- All GPU experiments ran serially. Measurements retain raw samples; clocks and display/WSL scheduling were not controlled. The ratios above are observed medians, not guarantees or precise tail-latency estimates.

Benchmark SHA256: `37f184896e2cb15b25b64e563e34bc77fd2daa722323c15a6cdb307ebe65ed68`.

Current plans replay with `python3 tools/autotune.py --replay <plan.json>`. A matching executable and tuner are also preserved in `build/expanded`; use `python3 build/expanded/autotune.py --exe build/expanded/histogram_bench --replay <plan.json>` after later changes. Plans check GPU name/SM, CUDA driver API/runtime/CUB metadata, eviction extent and binary hash, **not** exact driver release or GPU UUID. The environment artifact records those additional identities separately. Earlier schema-1 plans remain paired with `build/initial`.

The added published baseline is the pinned, unchanged NVIDIA histogram256 device code, with its [source, contract and license](../../third_party/nvidia_histogram/README.md). It is a scoped byte/256-bin/u32 baseline, not a claim to cover every published competitor.

The next useful optimization targets are byte inputs and stable small-workload selection, where CUB still wins or the gate rejects noisy candidates. The large uniform shared kernel already approaches this GPU's DRAM limit. Broader devices, stronger compatible published baselines, and independent workload validation remain necessary before claiming an overall performance frontier; another general research report is not the current bottleneck.
