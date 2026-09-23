# GPU-resident scalar-tree experiment

Decision recorded 22 September 2026, before implementation. This is an unmeasured
candidate selected from `ALGORITHM_DECISIONS.md`, not a fastest-known claim.
Target: SM86 RTX A5000 Laptop, CUDA 13.4, CUDA C++23. Existing counting kernels,
policies and defaults remain unchanged. CUDA runtime/graphs are infrastructure;
NVIDIA algorithm libraries are comparisons only, never production fallbacks.

## Contract and selected change

Preserve feature quantization, scalar trees, objective derivatives, missing/category
routing, split tie order, child constraints, leaf clipping and learning-rate rules.
Regression/binary derivatives remain tiled; softmax retains the complete pre-round
snapshot. Host setup/transfers/model export and explicit CPU prediction references
are allowed; dense fitting, validation, base-score computation, training decisions
and GPU inference encoding run on the GPU.

The frozen reference already batches histogram work over nodes. Its avoidable
control dependency is winner download -> stream synchronization -> CPU construction
of child nodes/maps -> upload. This experiment replaces that cycle with device
node storage, a compact frontier, child maps and active/node counts. The host
submits a bounded number of levels and reads status/tree metadata only at completed
tree export. Final-depth child leaf values come from their parent split; no extra
terminal histogram is built. Empty levels return early on the device.

For frontiers up to 1024 entries, a stable one-CTA exclusive scan computes child
IDs in existing frontier order. For larger frontiers, bounded 1024-entry block
scans produce counts, a small device prefix pass scans those block counts, and a
parallel materialization pass writes nodes/maps. This avoids an unsafe inter-block
spin barrier and unbounded persistent worker assumptions. The larger-frontier path
uses O(frontier_capacity) flags/offsets and O(ceil(capacity/1024)) scan metadata.

The device status reports invalid split metadata, nonfinite leaf values, node
capacity overflow or next-frontier overflow. Once failed, active count becomes
zero and later levels/tree prediction are guarded. Failures are surfaced at export,
never silently converted to a shallower tree. Histogram capacity is capped by its
explicit budget, even if worst-case 2^depth is larger. Other persistent payloads
are checked against max_device_bytes; bookkeeping/driver memory is separate.

## Layout, launches and memory

Feature-major u16 bin IDs, uint32 feature offsets, and feature types are retained
from GPU quantization. Derivatives use the existing row-major output tile.
Histograms remain node-major arrays of {double G,double H,uint64 count}. Reusing
the existing histogram/split kernels with a device active-count parameter changes
scheduling, not the arithmetic definition. The clear operation is device guarded.
Shared-memory launch size is a bounded capacity for that level; large capacities
use an explicitly selected supported global policy, never a NVIDIA fallback.

The persistent builder owns a TreeState, Node array, two frontier arrays, child
maps, scan offsets/block counts, and row assignments. Tree capacity is bounded by
rows/min_leaf_rows and depth. Only compact exported trees are retained on host.
The model export boundary includes its copy and synchronization cost in training
wall time; there are no winner/map transfers between levels.

Compare ordinary ordered stream submissions with a graph containing a complete
bounded tree build. Graph replay is annotated as tree_build; recorder events are
outside capture. Graph construction/instantiation cost must be retained separately
from replay. No measured default is claimed. Conditional graphs and persistent
workers remain alternatives for later comparison; CUDA restrictions, occupancy
and empty-level costs make an automatic preference unjustified.

Autotuning cannot restore per-level host waits. Its calibration must be outside
production tree construction, use the actual derivative tile/output, preserve raw
samples, and clearly describe the representative active-frontier configuration.
No count-histogram tuning result is transferred to weighted histograms. Forced
policies remain available for clean comparisons; unsupported forced-shared
capacities must be reported, not silently renamed global.

## Alternatives and evidence

The algorithm review cites current XGBoost batched histograms, parent/child
subtraction, LightGBM device partitions, small scans, and CUDA graphs. Smaller-child
histogram construction, row partitioning, quantized gradients and vector-leaf trees
remain separate experiments. Floating-point subtraction changes rounding;
quantization changes arbitrary gradients; shared vector partitions change model
structure. None is smuggled into this scheduling comparison.

For 32 features x64 bins, current statistics require 48 KiB/768 KiB/3 MiB of dense
histograms for 1/16/64 active nodes per output. Per-feature shared state is
1.5/24/96 KiB, so 64 nodes exceed the current 48 KiB policy. At 1024 outputs,
concurrent independent work must be tiled rather than allocating every histogram
and row assignment simultaneously. Full-gradient vector leaves and reduced-gradient
split scoring remain serious architecture alternatives for high output counts.

## Fair tests before promotion

1. Compare CPU-reference bins/categories/cuts and initialization validation/base
   scores with GPU preparation, including NaNs, signed zero, categorical overflow,
   zero weights, invalid targets, and large output tiles.
2. Test stream and graph trees on numerical/categorical/missing/weighted regression,
   independent binary, multioutput and coupled softmax cases. Cover depth0, early
   empty frontiers, irregular capacities, 1/16/64/1024/>1024 active frontiers,
   histogram/tree capacity overflow, and no input/output beyond allocated bounds.
3. Require model structural validity, CPU/GPU prediction agreement, exact count
   comparisons, recorded floating-point discrepancies and all existing per-output
   quality gates. Preserve a failed zero-allowance gate as failed.
4. Freeze inputs/quantization/derivatives/policies and alternate fresh independent
   stream/graph processes. Measure complete tree and output-tile completion,
   clear+histogram+split+metadata+route, model export, preprocessing and full train
   time; include graph setup. Retain raw rows and source/binary identity.
5. Root alone runs serial GPU tests, Compute Sanitizer and Nsight. Profilers explain
   synchronization/occupancy/traffic; their durations do not rank implementations.
   Include the existing 1024-/4096-output workload and larger row cases bounded by
   actual resident memory. Synthetic output scaling is not evidence of NLP quality.

Graph reuse detail selected before host integration: a persistent device selector
holds derivative stride, local output and global prediction output. A tiny setup
copy selects each tree; the captured kernels read that selector. Consequently at
most two graphs (root global/shared policy) are retained per training call, rather
than one graph per potentially thousands of outputs. The selector copy is setup
outside the tree replay and is included in complete training wall time. This keeps
graph metadata bounded and avoids per-output graph instantiation. The one-tree
export boundary is the initial implementation; batching exports remains future
work, measured separately.

Reporting reduction: GPU loss kernels emit bounded block partials. A single CTA
reduces those partials and divides by the GPU-computed total weight/output count;
only the scalar objective is exported per round. This changes the final reporting
sum from host long-double accumulation to GPU double reduction, so a last-bit loss
comparison is not promised. When detailed stream instrumentation is enabled, an
instrumentation-only device active-count history is exported with each finished
tree and applied to per-level sample metadata; uninstrumented execution omits it.
