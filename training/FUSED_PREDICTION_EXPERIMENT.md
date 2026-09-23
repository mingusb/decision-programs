# Exact ordered forest prediction experiment

Recorded before implementation, 2026-09-23. This is an explicit candidate, not a
measured winner. `PredictionPolicy::per_tree` remains the default and reference.
The counting path, training arithmetic, quantization and objective transforms
are outside this change.

## Contract

`Model::predict_gpu(data, raw, policy)` validates the same immutable-for-the-call
host model and input as before. Inputs are FP32 host features, numerical and
categorical feature metadata, missing NaNs, and independent FP64 leaf-valued
trees. Quantization produces the existing feature-major u16 bins. Output remains
row-major FP64 margins, or the existing GPU sigmoid/softmax transformation.
Rows may be zero; a model may contain zero trees or outputs with no trees.

For every (row, output), initialize an FP64 accumulator from exactly that base
score and add the reached leaf of every tree for that output in the original
model order. Do not omit zero-valued leaves, move the base addition, parallelize
the sum across trees, or change transform arithmetic. This includes signed zero,
subnormal values and cancellation. Each scalar addition uses round-to-nearest
FP64 (`__dadd_rn`), matching the reference's ordinary CUDA FP64 addition. Raw and
transformed outputs must match the reference bit for bit on one frozen model;
retraining independently is an invalid exactness comparison because training
already has unordered FP64 histogram atomics.

The new device entry point allocates nothing, is asynchronous on its caller's
stream, and accepts caller-owned validated node/descriptor/output-offset arrays.
All arrays must remain alive through stream completion and must not alias the
prediction output. Descriptors use u64 node offsets, u32 per-tree node counts
(at most INT32_MAX), and u64 output prefix offsets. Sizes, products and host
packing sums are overflow checked. SM86 CUDA C++23 is the first measurement
target; no special hardware feature newer than the existing path is needed.

The opt-in path requires all forest nodes plus descriptors and output offsets
to fit in the reported free-device-memory budget alongside predictions, bases,
and the existing quantizer. It may reject an otherwise valid model that the
per-tree path can stream through its smaller maximum-tree buffer. Allocation
failure remains an explicit CUDA error; there is no implicit policy switch.
No hidden resident cache is introduced because `Model` is publicly mutable.
Host packing is model setup, not a CPU training or feature-preprocessing stage.

## Candidate selection and work analysis

The current `Model::predict_gpu` uploads each tree from its separate host vector,
launches `add_tree` once per tree, and initializes the complete prediction array
in an earlier kernel. Its inference core logically reads and writes one FP64
prediction per row per tree: 16*N*T bytes, plus the initial 8*N*O-byte write.
This logical count is not a DRAM-traffic measurement.

The candidate packs nodes and tree descriptors stably by output, uploads each
array once, and launches one kernel over (output, row). Adjacent lanes traverse
adjacent rows for the same output, preserving the feature-major bin locality
of the reference. Each thread retains one ordered accumulator in registers and
writes once, reducing core prediction traffic to 8*N*O output bytes plus base
loads. Tree traversal/node/bin work remains. No atomics, shared-memory staging
or cross-thread reduction is needed. The costs are host packing, O(all nodes)
device storage, descriptor reads, longer thread lifetimes and possible cache
or occupancy losses. Wide outputs write the existing strided row-major layout.

NVIDIA's current Best Practices Guide recommends batching small host-device
transfers and retaining device data; it does not establish a local speedup:
https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#data-transfer-between-host-and-device

NVIDIA's FIL redesign studies model layout, node packing and workload-specific
tuning. Its H100 results are not a ranking for this SM86 FP64/bin-based model:
https://developer.nvidia.com/blog/supercharge-tree-based-model-inference-with-forest-inference-library-in-nvidia-cuml/

The CUDA FP64 intrinsic documents explicit round-to-nearest addition:
https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html

Alternatives considered: packing uploads while retaining T launches isolates
transfer overhead but retains repeated prediction traffic; CUDA Graphs amortize
repeat submission but do not remove this traffic and require reusable setup;
parallel tree evaluation plus an ordinary reduction changes addition order;
parallel traversal plus a sequential ordered gather preserves order but adds
O(N*T) scratch and writes. Compact/remapped nodes and persistent uploaded models
are separate future experiments. No NVIDIA algorithm library is introduced.

Existing evidence: the tiny 32-row, 3-tree inference diagnostic in
`results/profiling-expansion-20260923/nsys-graph-nodes/export-1.csv` contains three
tree launches (7.396 us combined), a separate initialization (1.729 us), encoding
(3.329 us) and sigmoid (2.753 us). These profiler times identify work and must
not rank implementations. `compiler-runtime/report.json` reports 17 registers
and zero spills for the old tree kernel. The new kernel's resources must be
measured before making occupancy claims.

## Fair experiment and gates

Use the same saved model and identical feature bytes for both policies. Time
complete `predict_gpu` calls including validation, host packing, allocation,
quantization, all H2D transfers, kernels, final D2H transfer and cleanup. Context
warm-up and model-file loading are outside this boundary and reported as such.
Retain every sample. Alternate policy order, use three warm-ups and at least
15 paired samples per case; measure GPU workloads serially without profilers.
Record device, compiler/build hashes, shape, tree/node counts, packed payload,
raw samples and policy order. No CUDA compiler time-trace flag may be active in
a runtime-ranking build because our NVCC13.4 experiment found codegen changes.

Bound the first study to small/large row batches (32, 4096, 65536), outputs
(1, 3, 65, then 1024 if memory allows), shallow/deeper trees and existing frozen
benchmark models. The benchmark can synthesize deterministic inference features
from each model; this is a performance fixture, not signal-detection evidence.
Also validate actual held-out feature bytes before promoting a policy.

Correctness gate: zero differing FP64 bits for both raw and transformed output;
cover all objectives, empty rows/forests, outputs without trees, arbitrary tree
interleaving, categorical/unseen/missing values, numeric thresholds and tails,
signed zero, subnormal leaves and cancellation. Verify malformed model/policy
rejection and compute-sanitizer memcheck/initcheck on bounded cases. Training
and count defaults must remain unchanged.

Performance gate: report median paired complete-call ratios with uncertainty;
a speedup claim requires the 95% paired-bootstrap interval upper bound below
1.0 on that case. A default change would additionally require no measured
regression across the agreed matrix and exact correctness throughout. A failed
or inconclusive case remains visible. Nsight Systems should then confirm fewer
transfers/launches; Nsight Compute should explain cache/warp/resource behavior,
not supply the ranking times. This experiment alone does not change defaults.
