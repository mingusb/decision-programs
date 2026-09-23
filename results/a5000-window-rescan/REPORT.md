# Window-rescan implementation and research follow-up

The explicit two-pass histogram improves the tested million-bin uniform workloads
by **2.24–2.42× at 16,777,216 inputs** and **2.195–2.205× at 268,435,456 inputs**
over our existing narrow-counter kernel. Both produce dense u64 counts from u32
bin IDs. The complete operation includes scratch clearing, every input scan,
and output conversion. This is a measured improvement on the RTX A5000 Laptop
GPU, not a universal optimum or a zero-regression finding.

The prototype remains explicit. Automatic selection and existing default
parameters are unchanged. The preserved benchmark executables were not rebuilt.
The new production libraries contain no NVIDIA histogram/reference symbols.

## Implementation

[src/global_window.cu](../../src/global_window.cu) processes contiguous ranges of
bin IDs. For each range it clears reused u32 scratch, scans the entire input,
counts IDs in that range with global atomics, then overwrites that range of the
u64 output. Same-stream ordering completes widening before scratch reuse.

The default experimental window contains 524,288 counters (**2 MiB**). At
1,048,576 bins this requires two scans and exactly one successful atomic update
per input element overall. At 16M inputs the second scan adds 64 MiB of logical
input reads; at 256M it adds 1 GiB. These are logical traffic counts, not measured
physical DRAM traffic. Scratch requirement is `4 * min(bins, window_bins)` bytes
for nonempty input. Total `N <= UINT32_MAX` remains mandatory, including for
highly concentrated inputs. Filtering does not relax that worst-case bound.

The path supports scalar compile-time policies 0–5, natural u32 input alignment,
empty input, partial final windows, stream execution, and captured graph replay
after changing input values. It requires explicit u32 input/u64 output/u32 local
counters. It is excluded from automatic selection, generic `all`/`sweep`, and
saved autotuner plans. The experimental CSV adds `window_bins`; ordinary CSV
columns remain unchanged.

```bash
build/window-experiment/histogram_bench \
  --n 16777216 --bins 1048576 --input u32 --counter u64 \
  --launch graph --cache warm --window-bins 524288 \
  --variants global:0:48:u32:kernel,global_window:0:48:u32:kernel \
  --samples 11 --batch 4 --warmup-ms 200
```

## Fixed-policy measurements

All rows use uniform shuffled data, scalar policy 0 (128 threads, four items),
48 blocks, u32 scratch and u64 output. Default clearing is kernel-based for
graphs and runtime memset for direct streams. Each of ten cases ran once on
each of two predeclared seeds, with 11 randomized rounds and four operations per
sample after a requested 200 ms warmup. The existing kernel and candidate are
measured in the same process on the same buffers. No policy was selected from
this matrix. All observations, including losses, are retained.

Ranges are the two process medians or their within-process baseline/candidate
ratios, **not confidence intervals**. The speedup column is calculated within
each invocation, not by dividing unrelated range endpoints.

| Inputs | Bins | Counter window | Cache / launch | Existing narrow time, ms | Window time, ms | Existing / window |
|---:|---:|---:|---|---:|---:|---:|
| 16,777,216 | 1,048,576 | 4 MiB, one scan | warm / graph | 2.580–2.583 | 2.546–2.731 | 0.946–1.013× |
| 16,777,216 | 1,048,576 | 2 MiB, two scans | warm / graph | 2.589–2.615 | 1.080–1.155 | **2.241–2.422×** |
| 16,777,216 | 524,287 | up to 2 MiB, one scan | warm / graph | 0.711–0.716 | 0.722–0.725 | 0.985–0.988× |
| 16,777,216 | 524,288 | 2 MiB, one scan | warm / graph | 0.714–0.715 | 0.718–0.720 | 0.993–0.995× |
| 16,777,216 | 524,289 | 2 MiB, two scans | warm / graph | 0.698–0.699 | 1.058–1.136 | **0.614–0.661×** |
| 16,777,216 | 786,432 | 2 MiB, two scans | warm / graph | 1.393–1.402 | 0.984–0.985 | 1.416–1.423× |
| 16,777,233 | 1,048,576 | 2 MiB, two scans | warm / graph | 2.620–2.655 | 1.081–1.153 | 2.303–2.424× |
| 268,435,456 | 1,048,576 | 2 MiB, two scans | warm / graph | 39.963–40.273 | 18.207–18.264 | **2.195–2.205×** |
| 16,777,216 | 1,048,576 | 2 MiB, two scans | cold / graph | 2.591–2.606 | 1.154–1.255 | 2.077–2.245× |
| 16,777,216 | 1,048,576 | 2 MiB, two scans | warm / stream | 2.508–2.623 | 1.090–1.092 | 2.296–2.407× |

The 524,289-bin case takes **51.3–62.8% longer** because a nearly empty final
window still requires another full input scan. The tested crossover is not an
exact selection threshold. Small one-pass differences also do not demonstrate
zero overhead. The complete [independent audit](independent-audit.md),
[machine-readable audit](independent-audit.json), and [raw analysis](analysis.json)
retain all 20 invocations, 40 rows, 440 samples and 1,760 timed operations.

## Smaller-window follow-up

The two-pass gain justified testing a 262,144-counter window (**1 MiB, four
scans**) on the million-bin/16M-input case. Eight new processes used fresh seeds,
21 rounds, batch four and an existing-narrow comparator in every invocation.
Window-size order was 2/1/1/2 MiB for the first seed and 1/2/2/1 MiB for the
second. This balances a simple order effect while retaining process variation.

The four-scan candidate took 1.570–1.672 ms; the two-scan candidate took
1.081–1.183 ms. After normalizing each process to its measured narrow comparator,
the four-scan/two-scan runtime ratios were **1.538** and **1.427** for the two
seeds. More input scans outweighed any further benefit from smaller counters.
The two-pass configuration remains the better of these tested choices.
See [follow-up evidence](window-sizes/analysis.json) and
[reproducible runner](run_window_sizes.py). Seed and ABBA/BAAB order change
together, so their effects cannot be separated. The
[independent follow-up audit](independent-followup.md) preserves every process
and also covers the skew campaign below. No parameters were promoted.

## Nsight diagnostics

Nsight Compute recorded the existing counting kernel and both counting passes
of the candidate, with kernel replay, cache control enabled and clocks unlocked.
Nsight Systems captured the full graph operation and its clear/count/widen
sequence. [Commands and profiles](profiles/) are separate from the uninstrumented
rankings above.

The initial profiles showed different SM/DRAM clocks. A separately retained
follow-up used a requested two-second warmup and skipped 300 matching counting
launches. **Clocks still differed**, so neither profile pair is a controlled
clock-matched speed comparison. No profiler-derived speedup is claimed.

| Warmed diagnostic | L2 hit rate | Achieved occupancy | SM / DRAM frequency |
|---|---:|---:|---:|
| Existing narrow count | 64.84% | 8.28% | 1.35 / 4.93 GHz |
| Window count, first pass | 78.61% | 8.39% | 1.63 / 5.99 GHz |
| Window count, second pass | 79.19% | 8.38% | 1.63 / 5.99 GHz |

The higher observed cache hit rate is consistent with the smaller working set,
but does not isolate the cause of the speedup. Low achieved occupancy and
remaining memory stalls also leave room for further tuning. The timeline's
median window counting duration was about 542 µs per pass, with about 14.5 µs
widening and 5.8 µs clearing per pass. These summaries include validation and
warmup launches and are not substitutes for the whole-operation campaign.

Exports: [initial narrow](profiles/narrow-export.txt),
[initial window](profiles/window-export.txt),
[warmed narrow](profiles/narrow-warmed-export.txt),
[warmed window](profiles/window-warmed-export.txt),
[narrow timeline summary](profiles/narrow-timeline-stats.csv), and
[window timeline summary](profiles/window-timeline-stats.csv).

## Concentrated-input location experiment

The benchmark now accepts `--distribution hot99@BIN`, moving the forced dominant
value while preserving the historical random background sequence. Original
`hot99` still uses the last bin and produces exactly the same inputs as before.
This is a 99% forced-value mixture plus 1% uniform background, which can also
select that value; it is not exactly 99% measured mass in the dominant bin.

The [32-process skew campaign](../a5000-window-skew/analysis.json) used 1,048,593
inputs, bin counts 24,577 / 32,768 / 1,048,576, distinct valid dominant-bin IDs
24,575 / 24,576 / last, and shuffled/sorted orders. Each process compares the
frozen uniform choice with existing native-u64 and narrow-u32 warp aggregation.
There are 96 candidate rows and 1,056 raw timing samples. No new dominant-value
kernel or selector is introduced by this characterization.

At 24,577 bins, placing the dominant value at 24,575 (inside the shared prefix)
gave the shared-overflow kernel 15.4–16.9 µs, versus 28.2–31.7 µs for native warp
aggregation. Moving it one position to 24,576 gave roughly 680–681 µs versus
28.2–28.7 µs: **23.7–24.2× slower**. At 32,768 bins, concentrated values outside
the prefix likewise strongly favor warp aggregation. Inside-prefix sorted data
at that shape changes the winner between seeds; that variability is retained.
At one million bins, the frozen uniform narrow kernel loses to the native warp
control at every tested location, by 8.0–12.6×.

These results establish location as a necessary benchmark dimension. They do
not establish a universal skew winner or a free runtime method to identify the
dominant value. A future known-value specialization must beat the best existing
skew kernel and include all initialization/finalization costs.

## Synchronization investigation

The [new standalone timing modes](../../bench/preservation/README.md) use four
distinct timing event pairs/graph instances and four pinned output snapshots
in both modes. Position mode synchronizes after each position; quartet mode
defers synchronization **and event reads** until all four positions are queued.
Every output and canary is still checked. Unobserved per-position host completion
times are left blank in quartet mode.

The frozen screen ran 64 processes, 2,048 quartets and 8,192 positions: two
workloads, two same-backend bindings, four batch sizes, two synchronization modes
and two process repetitions per stratum. Workloads and controls are never pooled
as independent repetitions of one effect.

**The change did not establish a general noise fix or batch plateau.** Graph
quartet/batch32 looks promising in this screen, but retains an approximately 15%
same-backend quartet deviation. Direct-stream quartet/batch256 reduces the
selected tail diagnostic in all four matched repetitions, yet roughly 22–27%
90th-percentile symmetric quartet deviation remains. These are screening
observations, not evidence that small regressions or zero loss can be resolved.
No new old/new acceptance campaign was justified by treating those controls as
precise. A scheduling/clock/host-gap diagnostic remains necessary to attribute
the variability. See [full analysis](../a5000-sync-preservation-screen/analysis.md)
and [independent interpretation](../a5000-sync-preservation-screen/interpretation.md).

## Verification and preservation

- Full CTest run: all six targets passed, including 17,516 existing histogram
  executions, 372 dedicated shared-overflow executions, 3,170 CPU selector checks,
  historical input-sequence/located-skew checks, and 24 autotuner tests.
- New focused window suite: 783 structural checks and 628 GPU executions,
  covering boundaries, tails, input alignment, all scalar policies, repeated
  calls, changed-input graph replay, input preservation and buffer canaries.
- Window memcheck, racecheck and synccheck: 340 executions under each tool,
  zero errors or hazards. [Validation records](../a5000-window-validation/).
- Timing harness: both cases and both matched modes passed smoke tests and all
  three sanitizer tools. Seventeen new CPU audit tests and fourteen legacy
  audit tests passed. [Timing validation](../a5000-sync-preservation-validation/).
- Six window-campaign CPU audit tests check raw summaries, missing/duplicate
  rows, incorrect window/scratch/resource metadata and workload/eviction identity.
- Production symbol checks found no NVIDIA reference symbols. Existing
  defaults, narrow-counter, shared-overflow and bit-plane source hashes match
  their prior frozen versions. [Recorded boundaries](../a5000-window-validation/production-boundaries.json).

New benchmark SHA256:
`79bb396fdc50760eed6394c3ca78fe297f79934d9a0df7a45f22b2404d96b374`.
New timing harness SHA256:
`009422daaf686c18abe453629eaeed56641494c0f5cd507348eaf4d7842f2350`.
The earlier standalone and paired benchmark hashes remain unchanged.
The [source archive](../a5000-window-validation/source.tar.gz) and
[source manifest](../a5000-window-validation/source-manifest.json) preserve this
implementation and the documentation snapshot before this results report.

The experimental build adds dispatch/configuration code even though existing
kernel sources/defaults remain unchanged. That does not prove preservation of
all existing runtime paths. The unresolved preservation requirement remains
open; the measured large gains do not waive it. No NVIDIA-reference speedup is
claimed here because these campaigns compare our implementations only.

## Next bounded work

Tune thread policy and grid size around the successful two-pass million-bin
configuration, with separate selection and confirmation data. Keep the bin-count
boundary losses explicit. For concentrated input, investigate a separate
known-dominant-value kernel against the existing warp winner, with a correct
combined finalization and no compulsory sampling in the existing uniform path.
Address the timing environment before using sub-percent differences to accept
changes to existing defaults. Another broad research round is not needed first.

Reproduce/audit the fixed campaign with
`python3 tools/run_window_experiment.py window --output-root results/a5000-window-rescan --audit`,
the located-skew campaign with the same runner's `skew` stage and
`--output-root results/a5000-window-skew --audit`, the size follow-up with
`python3 results/a5000-window-rescan/run_window_sizes.py --audit`, and the
synchronization screen with `python3 tools/run_sync_preservation.py --audit`.
