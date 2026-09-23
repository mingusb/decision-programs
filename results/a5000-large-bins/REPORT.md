# Large-bin histogram implementation and validation

Two new owned CUDA paths improve the measured large-bin workloads: shared-prefix accumulation with global overflow, and narrow global counters followed by widening to u64. In the final five-shape campaign, selected configurations were **1.10–2.44× faster than the frozen native-counter controls**, and **6.81–38.94× faster than the NVIDIA histogram reference**. These are bounded measurements on one GPU, not a universal optimum. Production defaults are unchanged; NVIDIA histogram implementations remain benchmark-only.

**Acceptance limitation:** old/new checks found unresolved slowdowns in two existing configurations. The implementation and correctness checks are complete, but performance preservation is not established. The new configurations also lose on heavily skewed input and must remain workload-specific.

## Final performance measurements

All rows use **16,777,216 uniform shuffled u32 values, u64 output, warm graph execution**, batch 4, and 200 ms requested warmup on an **RTX A5000 Laptop GPU, driver 597.06**. Each comparison below uses the same final executable and benchmark invocation. Times include the operation's clearing, counting, and any widening. Ranges are the minimum and maximum of two fresh confirmation-seed medians, **not confidence intervals**. Ratios above 1 favor the selected configuration.

| Bins | Selected algorithm:policy:grid:local:clear | Selected time (µs) | Native / selected | NVIDIA / selected |
|---:|---|---:|---:|---:|
| 24,577 | `shared_overflow:15:24:u32:kernel` | 194.304–194.560 | 2.411–2.445× | 36.618–36.711× |
| 32,768 | `shared_overflow:15:24:u32:kernel` | 199.168–199.936 | 2.364–2.383× | 38.724–38.940× |
| 65,536 | `shared_overflow:14:48:u32:kernel` | 308.736–308.736 | 1.509–1.509× | 28.486–28.564× |
| 262,144 | `shared_overflow:14:48:u32:kernel` | 433.664–434.176 | 1.102–1.106× | 24.975–25.081× |
| 1,048,576 | `global:0:48:u32:kernel` | 2483.712–2491.904 | 1.814–1.819× | 6.809–6.845× |

![Confirmed large-bin speedups over frozen native controls](plots/native-speedup.png)

[Vector figure](plots/native-speedup.svg), [plot source](plots/plot_large_bins.py), and [plotted data with provenance](plots/plotted-data.json).

Shared overflow won at the first four shapes. At 1,048,576 bins, the independently selected overflow configuration took **4.359–4.366 ms**, losing to narrow global's **2.484–2.492 ms**. Both candidates were retained in confirmation; the losing result is part of the conclusion.

The actual production default at all five shapes remains `global:2:192:native:kernel`. Selected configurations improved over that default by 2.416–2.438×, 2.335–2.379×, 1.509×, 1.102–1.106×, and 1.843–1.851× respectively. These results have **not** been promoted into automatic selection.

The [independent final audit](overflow-evaluation/overflow-analysis.md) verifies all five cases, 30 invocations, 209 candidate measurements, and 2,191 raw samples. Each shape searched 14 overflow candidates: policies 14/15 across grids 24, 48, 96, 192, 384, 768, and 1536. Search used 5 samples; validation used 11 samples on each of two seeds; confirmation used 21 samples on each of two further seeds. Finalists included the top four custom candidates, the best remaining candidate in each algorithm/local-counter family, the default, and NVIDIA. Validation selected the overall winner and best overflow candidate independently using paired reference-normalized medians. Confirmation timings did not influence selection.

Native and narrow comparators were frozen before the overflow search from the earlier narrow study's validation results. The exception is 32,768 bins: its native control came from the scaling study, and its narrow control was an explicit `global:2:192:u32:kernel` configuration. **The 32,768-bin narrow control was not tuned.** The final ratios use fresh final-binary measurements of those configurations, not historical timings. The audit lists every comparator and seed.

## Larger inputs, tails, and other execution modes

The [fixed-policy extension audit](extension-evaluation/extension-analysis.md) covers 15 workloads, 30 invocations, 144 candidate measurements, and 1,584 raw samples. Configurations were frozen before measurement, with two new seeds, 11 samples per seed, batch 4, and 200 ms requested warmup. No NVIDIA reference or retuning was used in this extension campaign.

At **268,435,456 elements**, shared overflow at 24,577 bins took **3.087 ms**, beating the frozen native control by **3.233–3.264×**. Narrow global at 1,048,576 bins took **38.363–38.424 ms**, a **1.878×** gain. The measured uniform tails (`N = 16,777,216 + 17`), cold-cache graphs, and warm direct-stream cases also retained gains over their native controls.

The same frozen uniform choices were **about 9–25× slower than an existing warp-aggregation control** on the six `hot99` cases, where 99% of samples share one bin. Both shuffled and sorted inputs were tested. This is a measured limitation, not an untested hypothetical: the uniform winner must not become a universal default. The extension report preserves all candidates and losses.

## Implemented paths and bounds

- [Narrow global counters](../../src/global_narrow.cu): existing global/warp policies can count into a u32 workspace and then overwrite every u64 output bin with a widening kernel. Nonempty input requires `4 × bins` workspace bytes and **total input count ≤ UINT_MAX**. It supports u8/u32 input, scalar policies 0–5, and either requested clear method. At one million bins, the counter working set is 4 MiB instead of 8 MiB, with an additional widening pass. No allocation or device query occurs in the counting dispatch.
- [Shared overflow](../../src/shared_overflow.cu): for u32 input and u64 output, bins 0–24,575 accumulate in a fixed 96 KiB u32 shared prefix; higher bins accumulate directly into the u64 output. Each block merges the prefix after synchronization. Policies 14/15 use 256/512 threads with eight items per thread, aligned vector loads where possible, scalar alignment fallback, and tail handling. It uses no workspace. The per-block count bound must fit u32; total input may exceed UINT_MAX when that proof holds. `prepare` must enable the shared-memory requirement before capture/timing. The 24,576-bin boundary describes this kernel catalog's 96 KiB prefix, **not a verified physical-device maximum**.

Both paths preserve same-stream ordering and graph capture support. Output clearing and the bounds checks are part of the API contract; callers must provision the reported workspace and prepare the configuration before use.

## Earlier narrow-counter campaign

The [narrow audit](narrow-evaluation/narrow-analysis.md) covers four shapes, 24 invocations, 656 candidate measurements, and 4,040 raw samples. Every shape searched 140 custom configurations: global/warp × policies 0–4 × seven grids × native/u32 counters, plus NVIDIA.

Validation selected native counters at 24,577 and 65,536 bins. Narrow counters won at 262,144 bins by only 1.019–1.022× over the native control, and at 1,048,576 bins by 1.822–1.899×. The small former gain does not establish statistical significance. This preliminary campaign used a different executable from the final overflow campaign; its timings are retained as separate evidence rather than a matched before/after comparison.

## Correctness and sanitizers

The canonical final-build evidence is [validation-final-fixed](validation-final-fixed/). Recorded commands completed successfully. Final target executable hashes match the frozen source manifest.

| Check | Verified result |
|---|---|
| [General correctness](validation-final-fixed/correctness.stdout) | 17,516 histogram executions; independent CPU counts, sum, output/scratch canaries, tails, repeat calls, and nondefault streams |
| [Dedicated overflow correctness](validation-final-fixed/overflow-correctness.stdout) | 50 structural checks, 124 cases, 372 verified executions |
| [CPU selector checks](validation-final-fixed/cpu-defaults.stdout) | 3,170 checks passed |
| [Python tuner tests](validation-final-fixed/cpu-autotuner.log) | 24 tests passed |
| General [memcheck](validation-final-fixed/all-memcheck.stdout), [racecheck](validation-final-fixed/all-racecheck.stdout), [synccheck](validation-final-fixed/all-synccheck.stdout) | 3,060 executions under each tool; zero reported errors/hazards |
| Dedicated overflow [memcheck](validation-final-fixed/overflow-memcheck.stdout), [racecheck](validation-final-fixed/overflow-racecheck.stdout), [synccheck](validation-final-fixed/overflow-synccheck.stdout) | 50 structural checks, 52 cases, 156 executions under each tool; zero reported errors/hazards |

The [CLI and autotuner integration checks](integration-smoke/checks.json) also passed. A real 89-row sweep requested u32 local counters while retaining the NVIDIA reference's supported native width, and enumerated both global/warp counter widths plus shared overflow. A fresh schema-5 plan selected shared overflow and replayed successfully with executable/configuration verification. These short integration timings do not promote defaults or substitute for the performance campaign.

## Nsight findings

Nsight Compute profiles used kernel replay with cache control enabled and clocks unlocked. These are counting-kernel diagnostics, not whole-operation timing experiments. The million-bin comparison uses the same final binary, global policy 0, grid 48, workload, and profile settings; the local-counter mode differs. The boundary comparison changes algorithm and grid. The campaign above supplies the whole-operation performance comparisons.

| Profile | Counting-kernel duration | DRAM throughput (% of peak) | L2 hit rate |
|---|---:|---:|---:|
| 24,577 bins, earlier native global policy 0/grid 192 | 512.42 µs | 35.17% | 88.84% |
| 24,577 bins, final overflow policy 15/grid 24 | 199.14 µs | 90.46% | 6.48% |
| 1,048,576 bins, final native global policy 0/grid 48 | 4.37 ms | 59.96% | 16.56% |
| 1,048,576 bins, final narrow global policy 0/grid 48 | 2.35 ms | 72.47% | 65.86% |

The boundary case now approaches the profiler's peak DRAM-throughput estimate. The million-bin increase in L2 hit rate is consistent with reducing the counter working set from 8 MiB to 4 MiB. Both million-bin profiles report 1.45 GHz SM frequency; achieved occupancy remains low at 7.84% native and 8.29% narrow. The narrow profile still shows substantial memory-latency stalls, so profiling does not show that all inefficiencies have been removed. Full exports: [boundary before](profiles-before/boundary-export.txt), [boundary after](profiles-after/boundary-export.txt), [matched million native](profiles-after/million-native-export.txt), [million narrow](profiles-after/million-export.txt). The earlier unmatched [million-bin warp profile](profiles-before/million-export.txt) is retained separately.

Nsight Systems graph traces retain the actual clear/count/widen sequence. The [million-bin kernel summary](profiles-after/million-timeline-stats.csv) reports approximately 2.400 ms counting, 33 µs widening, and 12 µs clearing per invocation in that diagnostic trace. This uses batch 1 and is separate from the batch-4 ranking campaign. Traces: [boundary](profiles-after/boundary-timeline.nsys-rep), [million](profiles-after/million-timeline.nsys-rep).

## Preservation status

The [static machine-code audit](machine-code-final/README.md) found all **751 existing custom device functions** in the final build, with **zero resource-record differences**, and identical complete instruction/control-word sequences for **13 selected kernels**. The other 738 matching functions received resource comparisons only. The final build adds 24 functions for narrow counting/clearing/widening and shared overflow. This does not inspect host dispatch or prove equal runtime performance. The [production symbol audit](production-symbol-audit.json) found no NVIDIA histogram symbols in `libgh.a` or `libgh_policy.a`.

**Runtime preservation remains unresolved.** The [initial matched old/new campaign](preservation-final/paired-comparison.md) flagged `single` with median current/old 1.159763 (three of four pairs at least 5% slower), and `stream4096` with 1.074403 (all four pairs slower). The three large-bin native controls were near parity, with median ratios 1.004080, 0.998645, and 1.000222.

A [targeted repeat](preservation-followup/paired-comparison.md) retained the original protocol, inputs, and ABBA/BAAB execution order. The [combined audit](preservation-combined.md), which preserves both cohorts, reports median current/old **1.035468 for single-valued input** (five of eight pairs slower, three ties), and **1.149728 for direct-stream execution** (seven of eight pairs slower). Across the full initial campaign and repeat there are 144 invocations, 3,024 raw samples, and 72 adjacent invocation pairs. These are descriptive ratios, not confidence intervals. Unchanged default source and static kernel matches do not explain away the losses.

The subsequent [four matched Nsight Systems traces](preservation-timeline/timeline-analysis.md) localize the stream variability: counting-kernel medians were **6.449–6.490 µs** and device-memset medians **1.438–1.465 µs**, while traced operation medians ranged **22.619–40.031 µs**. The old binary was both the fastest and slowest traced run. Most gaps between this stream's recorded operations occurred after the next operation's host API had returned. These gaps do not prove that the entire GPU was idle, or identify application instructions, Windows/WSL scheduling, driver queues, or profiler overhead as the cause. Instrumented timings do not replace the unprofiled evidence.

A separate bounded CPU-affinity intervention ran 40 invocations in five balanced ABBA/BAAB blocks, with both binaries pinned to CPU 6. Across its 20 pairs, median current/old was **0.966623**, with 12 faster and eight slower pairs; individual ratios ranged **0.274358–2.423081**. Thus fixing CPU affinity did not produce stable timing, and reversing the aggregate comparison does not establish a fix. All original losses and this intervention are retained separately in the [affinity audit](preservation-affinity/analysis.md).

The remaining acceptance work is to obtain stable, attributable old/new stream and single-valued-input measurements before promoting the new build or extending automatic defaults. More unstructured repetition of these noisy timings would not resolve that requirement. No causal code fix is claimed by this implementation pass.

The later [same-process investigation](../a5000-paired-preservation/REPORT.md) completed a frozen 24-invocation comparison with old/old and new/new controls. Its aggregate event ratios were 1.008467 for single-valued graph input and 0.989073 for the stream case. The earlier large stream penalty did not reproduce consistently. Same-backend control variation and retained tails still prevent a zero-loss claim; no production fix or default promotion followed from that diagnostic.

## Reproducibility and limits

The [final JSON audit](overflow-evaluation/overflow-analysis.json) preserves raw samples, summaries, commands, selection records, telemetry, and artifact hashes. Its source checks use the archived source and parser rather than a later working tree. The [final source manifest](final-source-manifest.json) records 29 source files and five binaries.

- Final executable SHA256: `ab480a2e08254cb54a5579125ccef3d1101a776aca434ccd0e2e891ddc642ea1`.
- Final [source archive](final-source.tar.gz) SHA256: `8a302c90ec615b5bb04806ef62792ef46d277084ec65d8cef8bc0fb43b9b593b`.
- Preliminary narrow executable SHA256: `bcf042dee54cd18408e36f5852e756e88348dd5aa3d3083e145a834a41e9f648`.
- Preliminary [source archive](narrow-source.tar.gz) SHA256: `ebb2fb891735cf219dc17656d7ac7f4fc1e767b8da6340533f6b2b99ba83ae68`.

CPU audit scripts: [analyze_narrow.py](analyze_narrow.py), [analyze_overflow.py](analyze_overflow.py), and [analyze_preservation_timeline.py](analyze_preservation_timeline.py). Frozen campaign runners: [run_narrow.py](run_narrow.py), [run_overflow.py](run_overflow.py), [run_extensions.py](run_extensions.py), [run_preservation.py](run_preservation.py), and [run_affinity.py](run_affinity.py). Preservation auditing uses the `analyze` stage; extension and affinity auditing use `--audit`. These audit modes do not access the GPU.

Clocks were unlocked and telemetry was sampled around invocations. Two fresh seeds do not establish statistical significance or performance across other distributions, sizes, modes, or devices. Native/narrow controls were fixed from prior work rather than exhaustively retuned on the final binary. These results support the new explicit configurations on the measured shapes; they do not establish zero regressions, universal superiority, or a reason to change automatic defaults without further evidence.
