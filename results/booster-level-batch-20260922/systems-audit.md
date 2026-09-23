# Nsight Systems audit of output-tile construction

CPU-only audit of immutable captures, SQLite integrity, runtime enqueue correlations and exact emitted-sample/NVTX correspondence. Profiler durations explain these captures; they do not rank uninstrumented implementations.

| Capture | Tree scopes | Output trees | Boosting kernels including count setup | Export scopes | Export stream waits | Tree waits / D2H / device allocations |
|---|---:|---:|---:|---:|---:|---:|
| nsys-per-output-graph | 387 | 387 | 9817 | 27 | 27 | 0 / 0 / 0 |
| nsys-output-batch-graph | 27 | 387 | 655 | 27 | 27 | 0 / 0 / 0 |
| nsys-output-batch-stream | 27 | 387 | 655 | 27 | 27 | 0 / 0 / 0 |

`nsys-per-output-graph` → `nsys-output-batch-graph`: matched tree scopes 387 → 27; complete-boosting kernel activity records including immutable count setup 9817 → 655; tree graph enqueue calls 387 → 27. Both represent 387 independent output trees. These are observed counts, not timing speedups.

`nsys-output-batch-graph` → `nsys-output-batch-stream`: matched tree scopes 27 → 27; complete-boosting kernel activity records including immutable count setup 655 → 655; tree graph enqueue calls 27 → 0. Both represent 387 independent output trees. These are observed counts, not timing speedups.

Every emitted stage ID/name matches a single ghb NVTX range. Tree scopes cover the exact round/output schedule and retain correct operations counts for full and short tiles. Each tree scope has zero explicit runtime host synchronization, synchronous copies, correlated device-to-host transfer, device allocation and device free calls.

NVTX intervals bound host submission. GPU kernels and copies can finish after a host range ends; all attribution follows CUDA enqueue correlation IDs. Nested scopes are not summed twice in runtime unions. The scope contract checks runtime API records, not arbitrary CPU work or untraced direct driver calls.

## Setup outside tree scopes

The following calls are retained as setup observations. Their presence in a whole-process trace does not imply a wait/allocation inside a tree batch. Counts before the first tree also include preparation, initial objective evaluation and, on the comparison path, initial root work.

| Capture | Before-first-tree waits | Before-first-tree device allocations | Graph-setup waits | Graph-setup device allocations |
|---|---:|---:|---:|---:|
| nsys-per-output-graph | 6 | 29 | 0 | 0 |
| nsys-output-batch-graph | 6 | 28 | 0 | 0 |
| nsys-output-batch-stream | 6 | 29 | 0 | 0 |

## Kernel work inside tree scopes

Families classify actual demangled kernel names, including graph-node records. Graph captures have one outer tree scope; the JSON separately retains direct innermost-NVTX stage attribution for stream captures. Durations are sums of activity-record intervals and are diagnostic only.

| Capture | Kernel family | Records | Summed kernel ms | Share of tree kernel time |
|---|---|---:|---:|---:|
| nsys-per-output-graph | deeper_histogram | 1548 | 21.847857 | 48.27% |
| nsys-per-output-graph | frontier_materialize_route_advance | 5805 | 10.947783 | 24.19% |
| nsys-per-output-graph | split_search | 1548 | 10.202471 | 22.54% |
| nsys-per-output-graph | tree_prediction | 387 | 1.580399 | 3.49% |
| nsys-per-output-graph | tree_initialize | 387 | 0.684819 | 1.51% |
| nsys-output-batch-graph | deeper_histogram | 108 | 8.558954 | 55.69% |
| nsys-output-batch-graph | split_search | 162 | 3.359880 | 21.86% |
| nsys-output-batch-graph | root_histogram | 54 | 2.431883 | 15.82% |
| nsys-output-batch-graph | frontier_materialize_route_advance | 243 | 0.653134 | 4.25% |
| nsys-output-batch-graph | tree_prediction | 27 | 0.315243 | 2.05% |
| nsys-output-batch-graph | tree_initialize | 27 | 0.051044 | 0.33% |
| nsys-output-batch-stream | deeper_histogram | 108 | 7.454249 | 51.81% |
| nsys-output-batch-stream | split_search | 162 | 3.415416 | 23.74% |
| nsys-output-batch-stream | root_histogram | 54 | 2.440801 | 16.97% |
| nsys-output-batch-stream | frontier_materialize_route_advance | 243 | 0.698738 | 4.86% |
| nsys-output-batch-stream | tree_prediction | 27 | 0.319977 | 2.22% |
| nsys-output-batch-stream | tree_initialize | 27 | 0.057666 | 0.40% |

The JSON preserves full kernel names, raw transfer attribution, API counts, every sample/context match, setup ranges and SHA256 for all inputs. Root work outside legacy per-tree scopes remains outside that table; whole-capture groups are also retained. Kernel counts from graph-node and graph-only traces must not be compared; these captures explicitly request graph-node detail.
