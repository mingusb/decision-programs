# Nsight trace audit

CPU-only audit of immutable capture artifacts, SQLite integrity, and emitted stage/NVTX correspondence.

| Capture | Matched stages | Tree scopes | Export scopes | Export stream waits | Tree-scope waits / D2H calls |
|---|---:|---:|---:|---:|---:|
| nsys-graph | 27 | 3 | 3 | 6 | 0 / 0 |
| nsys-wide | 1203 | 387 | 387 | 774 | 0 / 0 |
| nsys-16m1 | 0 | unavailable | unavailable | unavailable | unavailable |
| nsys-batched | 843 | 387 | 27 | 27 | 0 / 0 |
| nsys-16m1-optimized | 0 | unavailable | unavailable | unavailable | unavailable |

For the same 387 output trees, completed-tree export stream waits fell from **774 to 27**, across 387 compact exports versus 27 batches. Every old export contains two waits; every batch contains one. These are counts in matched export ranges, not whole-process totals.

Every emitted sample ID and stage name matches exactly one ghb NVTX range. Sample host intervals and NVTX intervals admit one consistent recorder-clock origin per instrumented capture. Each instrumented tree scope contains zero runtime stream/device/event waits and zero runtime calls correlated with device-to-host copies.

The old wide trace uses graph-only activity; the batched trace records graph nodes. Their kernel totals must not be compared as equivalent coverage. NVTX tree ranges bound host submission, so transfer classification follows CUDA correlation IDs, including GPU work completing after a host range ends.

Captures without stage instrumentation retain whole-capture API, kernel, and transfer records, but no training-stage counts are inferred. Counts outside matched ranges in instrumented captures can include setup, inference, validation, and cleanup.

The JSON includes every matched sample, batch count/context, raw transfer attribution, input SHA256, and scope limitations. Nested/export scope durations are not additive. Profiler observations are diagnostic, not a performance ranking.
