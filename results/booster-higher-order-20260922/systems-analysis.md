# Higher-order Nsight Systems diagnostics

Successful profiles-v2 captures cover all 10,336 Delicious training rows, 500 features and 983 labels, one round, depth 3, bins 32, lambda 1, learning rate .1 and unshrunk clipping radius 1. Batching uses width 16 and CUDA Graphs. **These are profiler diagnostics, not speed rankings.**

Captured artifact hashes match their successful audit receipts. SQLite integrity checks pass, inputs remain unchanged, runtime correlations are unique, GPU operations have enqueue correlations, and CSV kernel totals match SQLite. All three captures use the same frozen production binary.

| Order | Captured kernels | Tree graph kernels | Derivative kernels | Tree graphs | State/node exports |
|---|---:|---:|---:|---:|---:|
| 2 | 2899 | 1426 | 62 | 62 | 62 |
| 3 | 2899 | 1426 | 62 | 62 | 62 |
| 4 | 2899 | 1426 | 62 | 62 | 62 |

Each tree graph contains 23 kernels. There are 61 full 16-output batches and one 7-output tail; export byte counts independently match this schedule. Each capture includes 186 split-candidate and 186 winner kernels, 62 root accumulations and 124 deeper accumulations. Launch counts remain equal across orders.

Shares below use summed GPU kernel durations attributed to tree construction, derivatives, invariant root counts and initial/final objective evaluation. Quantization, model export and separate validation prediction are excluded from this denominator. Full-capture values and kernel shapes remain in JSON.

| Family | Order 2 ms / share | Order 3 ms / share | Order 4 ms / share |
|---|---:|---:|---:|
| root_accumulation | 435.113 / 15.572% | 657.576 / 15.060% | 875.819 / 14.665% |
| deeper_accumulation | 1546.848 / 55.361% | 1952.610 / 44.719% | 2396.572 / 40.130% |
| split_candidates | 790.481 / 28.291% | 1732.513 / 39.678% | 2673.519 / 44.767% |
| split_winners | 2.060 / 0.074% | 1.469 / 0.034% | 1.506 / 0.025% |
| root_clear | 0.223 / 0.008% | 0.310 / 0.007% | 0.411 / 0.007% |
| deeper_clear | 0.677 / 0.024% | 1.726 / 0.040% | 2.790 / 0.047% |
| derivatives | 3.172 / 0.114% | 3.069 / 0.070% | 3.369 / 0.056% |
| frontier_materialize_route_advance | 2.492 / 0.089% | 2.436 / 0.056% | 3.659 / 0.061% |
| tree_prediction | 2.857 / 0.102% | 2.722 / 0.062% | 3.452 / 0.058% |
| objective | 9.171 / 0.328% | 10.939 / 0.251% | 9.919 / 0.166% |

Every tree-build **host enqueue scope** has zero host synchronization calls, device allocations/frees, blocking copies and D2H transfers. Each contains one graph launch. Attribution follows graph enqueue correlation: GPU execution finishes after the short host submission range closes.

Completed-tree exports occur outside those scopes. Each of 62 export ranges contains one stream synchronization and two D2H transfers (state and full-capacity nodes), totaling 495,432 bytes per capture. Long host download ranges largely wait for preceding asynchronous tree work; their elapsed times do not measure transfer cost. Separate prediction and quantization transfers remain in whole-capture totals.

| Order | Before-first-tree sync calls | Before-first-tree device allocations | Before-first-tree event creates | Graph-setup sync / allocation |
|---|---:|---:|---:|---:|
| 2 | 37 | 27 | 78672 | 0 / 0 |
| 3 | 37 | 28 | 78672 | 0 / 0 |
| 4 | 37 | 29 | 78672 | 0 / 0 |

Setup is separated from resident training. Instrumentation preallocates timing events before the first tree; these are not per-tree allocation regressions. Graph capture and instantiation have their own initialization scope.

Source-supported interpretation: wider statistics increase histogram fields and traffic (24/32/40-byte cells); higher-order proposals add FP64 arithmetic and safeguards. Candidate scoring and accumulation account for nearly all measured boosting kernel time. With 500 features the current dispatch uses 256-thread candidate blocks even though individual features have very few bins. Parent leaf scoring repeats across all threads before bin predicates; graph grids also include inactive frontier capacity slots. These are mechanisms to investigate with Compute counters, not established causes from Systems alone.

Trace metadata agrees with compiled candidate register counts 72/90/96 and static shared memory 12,480/12,544/12,608 bytes. Those kernels report zero per-thread local memory. Register counts alone do not establish achieved occupancy or a spill bottleneck. Derivative kernels remain a small share of these captures despite adding derivative planes.

Different models can choose different active-row partitions. Order differences therefore combine changed arithmetic, traffic and model-dependent work. No default or quality conclusion follows from these durations. The earlier attempt and its failed Compute root-name filter remain untouched; this analysis uses only successful v2 Systems exports.
