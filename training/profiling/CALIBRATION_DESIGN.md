# Hardware calibration contract and experiment

Recorded before implementation, 2026-09-22. This is an optional, standalone
CUDA C++23 diagnostic benchmark. It does not select or replace a production
histogram algorithm. Production policies, arithmetic and defaults stay intact.

The production global-window counting kernel accumulates **u32** global atomics
then widens to u64; root/deeper booster statistics use FP64 and u64 atomics, with
uint16 binned inputs. Consequently calibrate global atomic additions of u32, u64
and FP64 with both uint32 and uint16 keys. Use uniform, single-hot-bin and 90%
hot-bin deterministic patterns and grids of one and four blocks per SM. Return
values are discarded, matching the production update expressions. Native
global atomics, warp aggregation and shared privatization have different work,
scratch, barriers and merge costs: this experiment isolates the native primitive,
not a claim that it is the best histogram algorithm. Retain source/SASS evidence
before interpreting the instruction mechanism; shared-memory POPC claims do not
automatically apply to these global operations.

The uniform pattern cycles a permutation of all 1024 bins. The skew fixture is
approximately 90% bin zero and 10% cycling the 512 odd bins (the deterministic
every-tenth-row selection interacts with the generator). It is not an iid Zipf
or uniform-tail model. Blocks per SM is the grid-size ratio, not measured active
residency. Each measured repetition is checked outside its timing interval;
those D2H reads between repetitions may affect cache state.

Inputs: positive bounded N, 1024 bins, 256-thread blocks, nonaliasing CUDA
allocations; deterministic device-generated keys. Outputs: 1024 counters and raw
timing observations. Integer sums cannot overflow under the accepted N bound.
FP64 adds exactly 0.125, giving an exactly representable reference at this bound.
Host regeneration and comparison are validation only. Key generation and D2H
validation are outside timing; each timed operation includes output clear and
all accumulation work, with a synchronized host boundary measured separately.
This is not the full production clear/accumulate/widen operation. Repetitions
never accumulate previous counts. Both key widths represent identical keys.

Predeclared matrix: 3 atomic types x 2 key widths x 3 address patterns x 2 grids,
N=262144, 2 warmups and 7 measured repetitions. Preserve every repetition and
exact correctness result. CLI filters permit a single small profiler capture.
No tuning winner is selected from this calibration. Timings under profilers or
sanitizers are diagnostics only. Root runs all GPU processes serially; external
thermal/power variation remains a limitation and telemetry is recorded.

For memory reference measurements use NVIDIA's isolated **nvbandwidth** at
commit `82fc4e8c6afa0babb8687793678f615b3b8d793e`, not production linkage. Its
device-local copy-engine copy and SM read/write/copy tests apply to SM86; TMA and
peer-link tests do not. Query L2 capacity and sweep footprints below/near/above
it, preserving raw tool JSON and commands. Those are observed reference rates,
not mathematical hardware ceilings or predictions of end-to-end booster speed.

Primary implementation references: [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/),
[Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/),
[nvbandwidth pinned source](https://github.com/NVIDIA/nvbandwidth/tree/82fc4e8c6afa0babb8687793678f615b3b8d793e).
Local evidence: `src/global_window.cu`, `training/src/root_histogram.cu`,
`training/src/deeper_histogram.cu`; existing measured counting kernels remain
the performance baseline. Calibration adds no NVIDIA algorithm to the system.
