# Same-process preservation investigation

The controlled comparison **did not consistently reproduce the earlier large
stream slowdown**. It found near-parity typical timings and substantial variation
even when comparing a backend against itself. It did **not** establish zero
performance loss or isolate a code change that should be fixed. Existing
production kernels and automatic defaults were left unchanged.

## Results

Both workloads use 1,048,576 u32 inputs and u32 output counts on the RTX A5000
Laptop GPU. Ratios are new time / old time; above 1 means slower. The aggregate
is the geometric mean of matched quartet ratios, giving equal weight to each of
four processes. Ranges span those four process summaries, not confidence intervals.

| Workload | Aggregate new/old | Process range | Old/old control range | New/new control range |
|---|---:|---:|---:|---:|
| Single-valued input, 256 bins, graph | 1.008467 (**+0.847%**) | 0.974867–1.041014 | 0.976141–1.049758 | 0.958231–1.001543 |
| Uniform input, 4,096 bins, direct stream | 0.989073 (**−1.093%**) | 0.935825–1.044372 | 0.967309–1.022429 | 0.944123–1.028099 |

![All process summaries and same-backend controls](paired-comparison.png)

[SVG figure](paired-comparison.svg), [plot source](plot_results.py), and
[plotted data](plotted-data.json).

The median across real-comparison quartets was **1.000000** for single-valued
input and **1.000902** for stream execution. These medians do not erase the
geometric-mean differences or the retained tails. Real-comparison quartet ratios
ranged from 0.501017–2.964809 and 0.424646–2.169176 respectively. Similar large
swings occurred in the same-backend controls. The four stream processes split
two favoring new and two favoring old.

This weakens the evidence for a consistent implementation-specific penalty in
the two flagged configurations. It does not demonstrate that every difference
is measurement noise, or that small changes and tail latency are preserved.
The [independent interpretation](interpretation.md) explains the order effects,
host-timing scopes, and all retained control results. The earlier
[separate-process results](../a5000-large-bins/preservation-combined.md) remain
valid records of their distinct protocol and have not been replaced.

## What was implemented

The [standalone harness](../../bench/preservation/README.md) validates and stages
the exact old and new source archives, compiles their complete custom libraries
under separate namespaces, and links both into one executable with one dynamic
CUDA runtime. The adapter prepares each configuration once, then calls that
revision's full histogram API. Stream timings therefore retain each revision's
host validation and dispatch.

Both slots share the same device input, output address, nonblocking stream, and
CUDA context. The context is checked after every measured position. Function,
context, and stream addresses are recorded. Old/old and new/new controls bind
both slots to the same function; graph slots have separate captured instances.
No NVIDIA reference histogram is linked into the diagnostic.

The frozen campaign ran **24 invocations, 768 matched quartets, and 3,072 timed
positions**, covering **98,304 timed histogram operations**. Each invocation
used 32 balanced ABBA/BAAB quartets, 32 operations per position, and 200 ms of
alternating warmup. Two execution-order seeds reversed the real comparison's
slot mapping. Every position's final output and guards were checked against
independent CPU counts. All observations were retained.

Graph event nodes sit inside the captured batch. Stream events surround the
complete batch of API submissions. CPU submission and completion durations are
separate diagnostics; they are never added to GPU event times.

## Verification and provenance

- Both workload smoke runs passed output and guard checks.
- Both workloads passed memcheck, racecheck, and synccheck with zero reported
  errors or hazards. [Recorded commands and logs](../a5000-paired-preservation-validation/).
- [Fourteen CPU audit tests](../a5000-paired-preservation-validation/cpu-audit-tests.stdout)
  passed, including rejection of altered commands, bindings, contexts, timing
  values, ordering, and artifact hashes.
- The [independent static audit](../a5000-large-bins/preservation-paired-static/README.md)
  matched all **751 old and 775 new device resource records** to their respective
  preserved binaries. Complete instruction/control words matched for both
  relevant counting kernels and u32 clearing in each backend. Host adapters
  target their intended distinct histogram functions.
- The [strict data audit](analysis.md) and independent review checked every
  declared invocation, raw position, quartet, backend binding, command, receipt,
  source identity, and recorded GPU identity. All 768 event ratios were
  independently recomputed from the raw CSV files.

Executable SHA256:
`731f03397e404783bd29582037b9d800d4f9c6258aaf1e999ae19f681daab9b5`.
The [manifest](manifest.json) records both archived revisions, all 55 staged
source files, harness sources, build configuration, linked libraries, executable,
and complete predetermined schedule. The [analysis JSON](analysis.json) retains
quartet and process summaries and links to original raw evidence.
The nine diagnostic source files are also preserved in a
[source archive](harness-source.tar.gz) with a
[SHA256 manifest](harness-source-manifest.json), separately from the two backend
archives.

## Limits and decision

This is a rebuilt same-process diagnostic, not either original standalone
executable. It changes program layout and CUDA function registration. Each
position also performs an untimed output copy and synchronization, unlike the
historical benchmark. Clocks remain unlocked. The single-valued input is
identical for both data seeds. Slot mapping and execution-order seed change
together, so this campaign cannot distinguish their separate effects.

No reproducible implementation-specific regression was isolated to justify a
targeted production fix. Performance-preservation acceptance remains open;
automatic defaults are not promoted. Any further acceptance work needs a more
stable timing environment and the same-backend controls, rather than treating
another favorable aggregate as a fix. The required zero-loss standard has not
been relaxed.
