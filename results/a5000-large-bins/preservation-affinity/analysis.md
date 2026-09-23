# CPU-affinity preservation diagnostic

All five blocks passed the strict CPU audit: 40 invocations, 840 raw samples, and 20 adjacent pairs. This intervention remains separate from the original unpinned measurements.

Current / old paired-median ratio: **0.966623**, range **0.274358–2.423081**. Current was slower in **8/20** pairs and faster in **12/20**; four slowdowns exceeded 5%. No outcomes were discarded.

| Block | Median ratio | Four paired ratios |
|---|---:|---|
| 1 | 0.996707 | 0.955370, 1.032353, 1.007564, 0.985849 |
| 2 | 0.843142 | 0.818560, 0.274358, 0.924198, 0.867725 |
| 3 | 0.971301 | 0.843529, 1.040917, 0.901685, 2.423081 |
| 4 | 1.057521 | 1.096831, 1.018212, 2.140351, 0.869677 |
| 5 | 0.946328 | 0.889065, 0.914780, 1.055394, 0.977876 |

## All paired outcomes

| Block | Round | Positions | Order | Old median (µs) | Current median (µs) | Ratio |
|---:|---:|---|---|---:|---:|---:|
| 1 | 1 | 1/2 | old→current | 22.944000 | 21.919999 | 0.955370 |
| 1 | 1 | 3/4 | current→old | 21.760000 | 22.464000 | 1.032353 |
| 1 | 2 | 1/2 | current→old | 21.152001 | 21.312000 | 1.007564 |
| 1 | 2 | 3/4 | old→current | 20.352000 | 20.064000 | 0.985849 |
| 2 | 1 | 1/2 | old→current | 23.104001 | 18.912001 | 0.818560 |
| 2 | 1 | 3/4 | current→old | 90.976000 | 24.960000 | 0.274358 |
| 2 | 2 | 1/2 | current→old | 21.952000 | 20.288000 | 0.924198 |
| 2 | 2 | 3/4 | old→current | 42.335998 | 36.736000 | 0.867725 |
| 3 | 1 | 1/2 | old→current | 27.200000 | 22.944000 | 0.843529 |
| 3 | 1 | 3/4 | current→old | 19.552000 | 20.352000 | 1.040917 |
| 3 | 2 | 1/2 | current→old | 22.784000 | 20.544000 | 0.901685 |
| 3 | 2 | 3/4 | old→current | 19.904001 | 48.229001 | 2.423081 |
| 4 | 1 | 1/2 | old→current | 18.176001 | 19.936001 | 1.096831 |
| 4 | 1 | 3/4 | current→old | 19.328000 | 19.680001 | 1.018212 |
| 4 | 2 | 1/2 | current→old | 20.064000 | 42.943999 | 2.140351 |
| 4 | 2 | 3/4 | old→current | 24.800001 | 21.568000 | 0.869677 |
| 5 | 1 | 1/2 | old→current | 20.191999 | 17.952001 | 0.889065 |
| 5 | 1 | 3/4 | current→old | 24.032000 | 21.984000 | 0.914780 |
| 5 | 2 | 1/2 | current→old | 21.952000 | 23.167999 | 1.055394 |
| 5 | 2 | 3/4 | old→current | 21.695999 | 21.215999 | 0.977876 |

Order matters in this cohort: old→current pairs have median ratio 0.922217 (2/10 slower); current→old pairs have median ratio 1.012888 (6/10 slower). This pattern is consistent with substantial temporal variability, but does not identify its cause.

The strongest apparent improvement compares a 90.976 µs old median with 24.960 µs current. The strongest loss compares 19.904 µs old with 48.229 µs current. Pinning the parent to logical CPU 6 did not remove broad, bidirectional variation. The evidence does not support a fixed binary penalty or establish performance preservation.

## Limits

- This is an explicit CPU-affinity intervention and is kept separate from the original unpinned preservation cohorts; their results are not pooled.
- The parent records affinity to WSL logical CPU 6; child processes inherit it. Affinity was not independently sampled in every child or applied system-wide, and does not pin Windows scheduling to a physical core.
- The same two seeds and explicit configuration were repeated five times, with ABBA then BAAB order in each block. They are repetitions rather than fresh input draws.
- The intervention was run later rather than randomized concurrently against unpinned controls. A changed ratio distribution therefore cannot be attributed uniquely to CPU affinity.
- Broad slowdowns remain in both executables despite affinity. Endpoint GPU telemetry does not establish clocks or interference during every timing sample.
- The 21 within-invocation samples are not 21 independent binary-level tests. No formal significance or equivalence claim is made.
- The aggregate ratio below one does not prove a speedup, a fix, or zero regressions. The original unpinned stream preservation result remains unresolved.
- No more GPU measurements are implied by this report; all recorded outcomes and the unresolved acceptance status are retained.

CPU analysis only. Raw CSV, logs, commands, telemetry, and hashes were preserved.
