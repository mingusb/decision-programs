# Prediction bytes and resident pipeline

Contract and algorithm choice recorded before implementation, 2026-09-23.
The frozen `training/bench/real_data.cpp` writes `predictions.f64` directly from
row-major doubles after checking every value is finite. The pinned little-endian
consumer `training/tools/higher_order_campaign.py` reads `<f8` and checks the exact
byte count. This wire format has no header, magic, version, shape or objective.
Rows and output count are external facts; multiclass uses class count as outputs.

The fresh GPU codec preserves exactly `rows*outputs*8` little-endian IEEE binary64
bytes, including negative zero and subnormal values. Successful input/output is
finite; all infinity/NaN encodings fail. Outputs must be positive; zero rows are
valid and require zero payload bytes. Product, pointer range, natural double
alignment and capacities are checked before payload access. Byte buffers need
only byte alignment. Decode requires exact source extent; encode accepts a larger
destination capacity and touches only the active prefix. Status.required_bytes
reports the payload size. Inputs, outputs and status are disjoint live resident
storage; failure may leave a partial output, which must not be consumed.

Candidate comparison: device memcpy plus finite checking requires separate copy
and validation effects/passes; a fused codec reads each value once and writes its
eight bytes while checking exponent bits. Select the fused grid-stride kernel,
specialized only by direction, with 256 threads and bounded grid size. It needs
no workspace, host processing, arithmetic conversion, block barriers or successful
path atomics. Only failures atomically mark Status. Explicit byte assembly handles
unaligned wire storage without relying on unaligned u64 access. Aligned wide-copy
specialization is deferred until complete-operation measurements justify it.
[CUDA's bit-reinterpretation intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__CAST.html)
provide the required binary reinterpretation; they are infrastructure, not a
vendor serialization algorithm. No speed ranking is claimed.

The device-only test program includes literal legacy-layout prediction payloads
and GHBDS001 dataset bytes. Codec checks cover exact bit patterns, unaligned byte
buffers, signed zero, smallest/large finite values, every nonfinite class, short
and trailing payloads, insufficient capacity, overflow, empty rows and guards.
An actual pipeline decodes datasets, fits schema/bins, trains, encodes/decodes the
model, predicts, then encodes/decodes predictions. Independent bounded fixtures
cover a perfect regression stump, two-output opposite stumps, balanced binary
logistic leaves and uniform multiclass bases. Analytical predictions and raw
model margins are checked; model and prediction roundtrips must match bits.
Every stage has a separate [CDP2 tail continuation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html),
which checks completion before reading results or reusing the shared arena.
No model/data/training implementation helper serves as the analytical oracle.

Root alone runs the GPU checks, sanitizers and measurements serially.
Future codec ranking must include
validation, byte conversion and completion, compare aligned/unaligned and tiny/
large payloads with three warmups and fifteen alternating pairs, preserve raw
observations, and profile separately. No benchmark is executed for this task.

## Separate CSV compatibility requirement

The frozen synthetic campaign also consumes decimal CSV. `booster.cpp::save_csv`
exports `targets.csv` with `row_id,target,weight` for one target, or contiguous
`target_0,...` columns before weight for multiple targets. Predictions/baseline
use `row_id,prediction` for scalar output, contiguous `prediction_0,...` for
multiple outputs, and `row_id,p0,p1,...` for multiclass. IDs are decimal 0..N-1,
fields are comma-separated, rows end in LF, and numbers use defaultfloat precision
17. `training/tools/evaluate.py` enforces these headers and matching row order.
This is a distinct export contract, not replaced by the raw binary codec. The
separate [GPU CSV exporter](csv.md) implements decimal export and has its own
compatibility checks. Host numerical text adaptation is not byte transport;
binary-codec checks alone do not establish CSV compatibility.
