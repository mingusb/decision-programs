# Nsight trace audit

CPU-only audit of immutable capture artifacts, SQLite integrity, and emitted stage/NVTX correspondence.

| Capture | Matched stages | Tree scopes | Export scopes | Export stream waits | Tree-scope waits / D2H calls |
|---|---:|---:|---:|---:|---:|
| nsys-base | 870 | 387 | 27 | 27 | 0 / 0 |
| nsys-both | 898 | 387 | 27 | 27 | 0 / 0 |
| nsys-deep-shared | 898 | 387 | 27 | 27 | 0 / 0 |

Every emitted sample ID and stage name matches exactly one ghb NVTX range. Sample host intervals and NVTX intervals admit one consistent recorder-clock origin per instrumented capture. Each instrumented tree scope contains zero runtime stream/device/event waits and zero runtime calls correlated with device-to-host copies.

Both captures in this root/split experiment record graph-node activity. NVTX tree ranges bound host submission, so transfer classification follows CUDA correlation IDs, including GPU work completing after a host range ends.

Captures without stage instrumentation retain whole-capture API, kernel, and transfer records, but no training-stage counts are inferred. Counts outside matched ranges in instrumented captures can include setup, inference, validation, and cleanup.

The JSON includes every matched sample, batch count/context, raw transfer attribution, input SHA256, and scope limitations. Nested/export scope durations are not additive. Profiler observations are diagnostic, not a performance ranking.
