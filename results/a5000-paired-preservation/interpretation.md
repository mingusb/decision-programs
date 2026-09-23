# Independent interpretation of the paired comparison

The same-process comparison did not consistently reproduce the earlier slowdown. It also does not establish zero performance loss. No production change is justified by these measurements alone.

The frozen campaign completed all 24 invocations: four real-comparison processes, four old/old controls, and four new/new controls for each of two workloads. Each process contributed 32 balanced ABBA/BAAB quartets; all 3,072 measured positions passed output and canary checks and remain in the results. Ratios below are geometric means of quartet ratios, with equal weight for each of the four processes. Real comparisons use new / old; control comparisons use slot B / slot A of the same backend. A ratio above one means the numerator took longer.

| Workload | Real comparison | Range across four processes | Old/old control range | New/new control range |
|---|---:|---:|---:|---:|
| Single-valued input, graph launch | 1.008467 (+0.847%) | 0.974867–1.041014 | 0.976141–1.049758 | 0.958231–1.001543 |
| Uniform 4,096-bin input, stream launch | 0.989073 (−1.093%) | 0.935825–1.044372 | 0.967309–1.022429 | 0.944123–1.028099 |

For the single-valued case, three of the four real-comparison process medians were exactly 1.000000; the fourth was 0.998682. The overall median of its 128 real-comparison quartet ratios was also 1.000000. The geometric mean retained the effect of longer observations, including quartet ratios from 0.501017 to 2.964809. The corresponding old/old and new/new control quartet ranges were 0.588340–1.721074 and 0.501801–2.018898. Those tails are retained, not discarded. The distinction between a near-one median and a +0.847% geometric mean matters: neither statistic alone demonstrates that the implementation preserves every aspect of latency.

For the stream case, the real-comparison quartet median was 1.000902. Process results changed direction: two favored the new backend and two favored the old backend. The earlier large stream penalty was not consistently reproduced under this protocol. The real-comparison quartet ratios ranged from 0.424646 to 2.169176, while old/old and new/new controls ranged from 0.410785 to 3.043020 and 0.605167 to 1.990508. Substantial variation therefore exists even when the same backend occupies both slots. This identifies a measurement limitation; it does not prove that every difference in the real comparison is noise.

Order remains relevant. The two real-comparison process geometric means for the first order seed, with old in slot A, combined to 1.009538 for single-valued input and 1.006823 for stream launches. With the second order seed and reversed backend slots, they combined to 1.007398 and 0.971636. Slot mapping and pattern seed change together in this bounded design, so these results cannot separate their individual effects. Each order summary has only two processes.

Host measurements add context but do not replace GPU event measurements. The real-comparison host submission ratios were 0.967063 for graph launches and 0.986036 for stream launches. Enqueue-through-synchronization ratios were 0.993305 and 0.994286. Graph submission covers a graph launch; stream submission covers event records and the operation batch. Total host time can include pending untimed warmup work. These timing scopes differ, and their ratios must not be added to the GPU event ratios.

## What this establishes

Both archived backends ran against the same input and output buffers in the same process, GPU context, and stream. Namespace separation retained distinct old/new launch functions; the same-function controls mapped both slots to their declared backend. The repeated changes in direction and control variation weaken the case for a consistent backend-specific regression in these two configurations. They leave small changes and tail behavior unresolved.

The comparison executable changes program layout relative to the original standalone benchmarks. Per-position correctness checks also add an untimed device-to-host copy and synchronization between measurements. GPU clocks remain unlocked. The single-valued input is identical for both data seeds, so those runs are process repetitions rather than additional input distributions. Four real-comparison processes per workload are too few to support a zero-loss claim or a broad statement about all launch modes, distributions, or parameterizations.

Automatic defaults should remain unchanged on the basis of this diagnostic. A reproducible implementation-specific regression would justify a targeted fix; this campaign has not isolated one. Any further preservation experiment should state its acceptable effect size and timing scope in advance and retain the same-backend controls.

## Independent checks

CPU-only checks independently reread every raw CSV, reconstructed the prescribed 64-bit shuffle, checked all 24 workload/seed/comparison combinations and 128 positions per invocation, verified backend bindings and per-process address invariants, checked a single GPU/runtime identity, and checked the successful correctness acknowledgment in every log. Recorded artifact hashes and exact commands matched. Independently recomputed event ratios for every quartet matched all 24 reported process geometric means.

The strict campaign audit separately validates the full frozen manifest, all 55 staged source files, historical archive/manifests, harness sources, build flags and linked libraries, binary hash, raw artifact receipts, and complete schedule. See [analysis.md](analysis.md), [analysis.json](analysis.json), and [manifest.json](manifest.json). No files in the frozen runner, analyzer, harness, or production implementation were changed during this review.
