# A5000 confirmation on driver 597.06

This session checks the profiling changes on Windows NVIDIA driver **597.06**. We compare against NVIDIA’s CUB library of optimized GPU routines; its histogram is our reference implementation. All twelve frozen graph selections passed the confirmation audit: eleven custom choices beat the strongest applicable reference in all three confirmation runs, while the 16M-byte workload retains the NVIDIA histogram reference. Separately, replacing runtime output memset with an explicit CUDA clearing kernel improves warm-graph shared-histogram operations by **2.116×–4.456×** in three controlled cases. A later direct-stream comparison found a regression and prompted an explicit clearing policy, which has also completed correctness, sanitizer, and 12/12-case reconfirmation checks.

The current API subsequently added automatic workload-based defaults. Their behavior and separate verification are documented in the [automatic-defaults report](../../a5000-defaults/REPORT.md). This report keeps its original campaign numbers and binary identity.

The GPU is the NVIDIA RTX A5000 Laptop GPU, SM86, with 48 SMs and 4 MiB reported L2. The software remains CUDA 13.4, CUB 3.4.2, and WSL2. The benchmark binary SHA256 is `3485149e183ada05893e59d505bcebc1e86d2a484a4184ec53821216072b43c1`, matching the archived final build. See [session identity](environment.json) and [archive hashes](../final-build.json). The folder date is local Mountain time; the environment record uses UTC.

GPU experiments run serially with unlocked clocks. Measurements from historical driver 596.71 are kept separate. Neither pooling the sessions nor attributing their differences to the driver is justified by this experiment.

## Complete-operation selection and confirmation

Each of the twelve workloads has a fresh search with seed 12345, five samples, and 32 operations per sample. The tuner retains the four fastest custom configurations plus the fastest representative of otherwise missing families and all applicable references. Validation uses seeds 67890 and 24680, fifteen samples per seed, and the same batch size. A custom configuration must beat the fastest applicable reference by at least 1.05× on each validation seed; otherwise a reference is selected.

The selected configuration is then frozen. Confirmation consists of two separate invocations with seed 424242 and a third with seed 987654, each with 21 samples and batch 32. These seeds are excluded from selection. Every invocation includes the frozen choice, the strongest original scalar candidate from that workload's search, CUB, and the unmodified NVIDIA sample where supported. The strongest reference is determined within each confirmation invocation.

Timing protocol 3 records events inside the graph. Warm graphs bracket the batch; cold graphs place a separate event pair around each histogram after that operation's 64 MiB eviction pass. Timings include histogram initialization and finalization, but exclude cold-cache eviction. Candidate order is randomized, complete outputs are checked against independent CPU counts before timing, and no timing samples are discarded. Unlocked clocks and device scheduling remain possible sources of variation; telemetry snapshots are not a continuous clock trace.

The table shows the range across the three process medians, with two separate comparisons: strongest-reference time divided by chosen time, and scalar-control time divided by chosen time. Ratios above 1 favor the frozen choice. All cases are uniform shuffled data and warm graphs unless specified. Here 4K=4096, 1M=1,048,576, and 16M=16,777,216 samples; arrows indicate input and output types.

| Workload | Frozen choice, algorithm:policy:blocks:local | Operation median, µs | Strongest reference / choice | Original scalar control / choice |
|---|---|---:|---:|---:|
| 1M u32→u32, B=8 | shared:6:192:native | 5.952–6.016 | 2.463–2.473× | 1.016–1.022× |
| 4K u8→u32, B=256 | shared:1:96:native | 2.816 | 1.477× | 1.000–1.011× |
| 1M u8→u32, B=256 | shared:11:48:native | 4.864 | 1.355–1.362× | 1.230× |
| 16M u8→u32, B=256 | NVIDIA reference retained | 52.448–53.184 | 1.000× | 0.978–0.993× |
| 1M u32→u32, B=256, hot99 | shared:11:48:native | 5.472 | 2.345–2.351× | 1.105× |
| 1M u32→u32, B=256, hot99 sorted | shared:6:192:native | 6.080 | 2.295× | 1.032–1.037× |
| 1M u32→u32, B=256, single bin | shared:6:192:native | 6.080 | 2.284× | 1.026× |
| 16M u32→u32, B=4096 | shared:11:48:native | 185.632–188.704 | 4.095–4.174× | 1.000–1.027× |
| 16M u32→u64, B=4096 | shared:10:48:u32 | 185.920–188.384 | 10.133–10.335× | 1.003–1.013× |
| 16M u32→u64, B=8192 | shared:15:48:u32 | 187.072–191.072 | 19.702–20.177× | 1.001–1.038× |
| 16M u32→u64, B=16384 | shared:15:48:u32 | 193.696–195.168 | 30.415–30.647× | 2.708–2.801× |
| 1M u32→u64, B=4096, cold | shared:10:48:u32 | 21.920–21.984 | 5.422–5.482× | 1.048–1.061× |

The scalar controls use the **same kernel-clearing implementation as the choices in this campaign**. Their column measures the selected policy's benefit over the earlier scalar catalog, not the full before/after improvement of the old implementation. Policies may differ in loading, threads, grid, replication, or shared-memory capacity. The 16,384-bin control is global atomic because the old 48 KiB shared-memory policies cannot hold its counters; its 2.708–2.801× improvement is not a pure load-width comparison. Most other large-bin cases already had effective scalar policies, so their new policy gains are small despite large ratios against the NVIDIA histogram reference.

CUB is the strongest applicable reference in these confirmation runs. The sample baseline also competes in supported byte cases. For the 16M-byte case, the scalar control is slightly faster than CUB in confirmation, but its advantage is below the 1.05× selection threshold; confirmation is not used to choose a replacement winner.

The [confirmation audit](final-comparison.md) and [full JSON](final-comparison.json) verify all 36 invocations, candidate sets, workload/protocol metadata, binary and CSV hashes, and raw-sample summaries. Of 756 samples for the frozen choices, one exceeds twice its own row median; the largest excursion is 3.117× in hot99. Across all compared candidates, 10 of 2,394 samples exceed that descriptive threshold. No samples were removed. Two repeats at one seed provide limited repeatability evidence, not a guarantee of tail latency.

## Isolated output-clear improvement

The paired ablation compares archived runtime-clear and explicit-kernel-clear binaries using the same counting configurations, seed 424242, 21 samples, batch 32, and warm graph execution. Three repeats alternate build order. All nine shared pairs improve; unchanged controls remain close to parity.

| Input / bins / sample count, u32 output | Median paired old/new improvement |
|---|---:|
| u32 / 8 / 1,048,576 | 2.116× |
| u8 / 256 / 4,096 | 4.456× |
| u8 / 256 / 1,048,576 | 2.917× |

These ratios isolate the initialization change and are **not** speedups over CUB. The ablation uses one data seed and does not choose a dispatch policy. Its 18 invocations, 72 rows, 1,512 samples, exact command pairs, binary hashes, and GPU/driver identity checks are documented in the [ablation audit](clear-ablation-summary.md) and [machine-readable evidence](clear-ablation-summary.json). The counting specializations used here were previously verified to have byte-identical instructions across the two builds.

## Direct-stream regression and clearing policy

The graph improvement does not carry over uniformly to direct stream launches. A separate 24-process comparison uses the same archived binaries, three pairs per workload, seed 424242, 21 samples, and batch 32. The small-eight-bin shared case favors runtime clearing after normalization to the unchanged CUB reference in every pair:

| Workload | Median paired runtime/kernel ratio | CUB-normalized median | CUB-normalized range |
|---|---:|---:|---:|
| 1M u32 samples, 8 bins | 0.913× | 0.891× | 0.883–0.908× |
| 4K u8 samples, 256 bins | 0.949× | 0.949× | 0.938–0.999× |
| 1M u8 samples, 256 bins | 0.839× | 0.951× | 0.603–1.037× |
| 16M u32 samples, 4096 bins | 0.999× | 0.998× | 0.997–1.021× |

Ratios below 1 favor runtime clearing. The byte comparisons and unchanged controls show process variation, and stream timing includes host submission gaps; these measurements do not isolate a hardware cause. They do show why the graph result cannot justify replacing runtime clearing in every API call.

This finding prompted an explicit output-clearing policy: retain runtime clearing unless kernel clearing is requested. The revised benchmark can compare both policies within the same invocation. That revision completed a separate [12/12-case reconfirmation audit](../clear-policy-2026-09-21/final-comparison.md), with [full evidence in JSON](../clear-policy-2026-09-21/final-comparison.json). All 36 invocations preserve the earlier selected policies and explicitly request kernel clearing for graphs; all eleven custom choices beat the reference in every run. This reconfirmation tests the revised API without conducting another selection search.

The revision is archived at `build/profiled-clear-policy/histogram_bench`, SHA256 `5c50596f6c820152e8b3df6c0d9529e11461746b6d4000f89ad1e27639bc74c8`. Its [session record](../clear-policy-2026-09-21/environment.json) identifies that binary. The twelve-case table above remains attached to the earlier `3485149…` binary; numbers from the two revisions are not pooled. The later automatic selector chooses clearing according to the declared launch mode, as described in the [latest API report](../../a5000-defaults/REPORT.md).

The [stream audit](stream-ablation-summary.md) and [JSON](stream-ablation-summary.json) verify all 24 invocations, 84 rows, and 1,764 samples. Recorded hashes match immutable archived binaries. GPU and driver identity match the session; no samples are discarded, including 62 exceeding twice their own row median.

## Fresh profiling evidence

Two Nsight Compute captures use kernel replay, cache-control all, and unlocked clocks. They diagnose individual counting kernels; their times exclude operation work outside that kernel and do not replace the complete-operation comparison.

| Counting configuration | Workload | Kernel duration | DRAM peak | Registers/thread | Dynamic shared | Achieved occupancy |
|---|---|---:|---:|---:|---:|---:|
| `shared:11:48:native` | 1M u8 samples, 256 bins, u32 output | 5.79 µs | 48.45% | 23 | ≈1 KiB | 63.00% |
| `shared:14:48:u32` | 16M u32 samples, 16,384 bins, u64 output | 197.89 µs | 90.60% | 32 | ≈64 KiB | 17.06% |

Here 1M and 16M denote powers of two. The byte kernel reports 78.76% of scheduler cycles without an eligible warp, leaving latency and scheduling limitations worth investigating. The large-bin capture profiles policy 14, with 256-thread blocks; the frozen choice in the confirmation table uses policy 15, with 512-thread blocks. Policy 14 is limited by shared-memory capacity to one block per SM, yet approaches the DRAM limit. Its occupancy must not be attributed to policy 15, and higher occupancy alone is not an established improvement. Neither capture reports local- or shared-memory spilling requests. Excessive shared wavefronts remain measurable, without proving how much histogram update cost can be removed.

See the [profile audit](profile-summary.md) and [extracted metrics and commands](profile-summary.json). These captures belong to the current session but their command records lack per-profile driver/hash snapshots; the report does not fabricate that missing provenance. Historical profiles from driver 596.71 are not treated as controlled before/after comparisons with these captures.

## Current-driver cache verification

The matched whole-graph probe was rerun with driver 597.06 using the archived kernel-clear probe. It compares eviction→histogram→histogram against histogram→eviction→histogram, with the same total operations, a 1 MiB input, and a 64 MiB eviction buffer. Nsight Compute uses whole-graph kernel replay and flushes caches before each graph, preserving the cache effects between its nodes.

Moving eviction between the input reads changes L2 TEX read hits from 32,768 to zero and increases misses from 2,129,920 to 2,162,688 sectors: exactly 32,768 additional 32-byte sectors, or one 1 MiB input read. This confirms the intended capacity-eviction effect for this controlled case. Whole-graph DRAM traffic also includes the eviction and other work, so its difference is not treated as an exact input-only measurement. One pair of captures does not establish that every cache line is absent for every workload.

See [cache counters](cache-eviction-evidence.json), [evict-first command](cache-evict-first.command.json), and [evict-between command](cache-evict-between.command.json). Both commands succeeded and record driver 597.06 and probe SHA256 `4edff47231c1990af2ffb9e3a8c921b93ee62a10a9affd3d6c71069304031298`.

## Correctness and scope

The original resumed build passed:

- [18,622 histogram executions and fifteen CPU autotuner tests](ctest-details.log), covering independent counts, alignment, tails, persistent loops, canaries, repeated overwrite, and nondefault streams.
- [3,098 Compute Sanitizer memcheck executions](memcheck.log), with zero errors.

The unchanged archived build also previously passed racecheck, synccheck, cold-graph sanitizer checks, and ten actual counts exceeding UINT32_MAX; those are [historical validation results](../REPORT.md), not newly run tests in this session.

The subsequent explicit-clearing revision independently passed [18,622 histogram executions and sixteen CPU autotuner tests](../clear-policy-2026-09-21/ctest-details.log). It also passed 3,098 executions under each of [memcheck](../clear-policy-2026-09-21/memcheck.log), [racecheck](../clear-policy-2026-09-21/racecheck.log), and [synccheck](../clear-policy-2026-09-21/synccheck.log), with zero errors or race hazards. Verification of the later automatic-default path is recorded separately in the [defaults report](../../a5000-defaults/REPORT.md).

The contract is dense output from direct unsigned bin IDs and unsigned counts. Weighted gradient/Hessian accumulation, GBDT integration, runtime distribution detection, general dispatch evaluation, and other GPU architectures remain outside this pass. The evidence supports specific improvements on this GPU and measurement regime; it does not establish that every inefficiency has been removed or that the implementation is universally fastest.
