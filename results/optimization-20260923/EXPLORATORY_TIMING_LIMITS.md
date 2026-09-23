# Timing limitations and retained failure

The first split-training campaign and frozen-model prediction campaign were
exploratory measurements. GPU jobs were serial, but CPU-only quality audits
and limited benchmark compilation ran concurrently with parts of the work.
On this laptop, shared system power and host scheduling may affect complete
operation timings. Paired alternation does not prove those effects cancel.
GPU telemetry, raw per-pair observations and hashes are retained.

The full split-operation benchmark stopped progressing. A host debugger
confirmed a wait in cuEventSynchronize; the process was terminated after
more than five minutes. Its incomplete JSON, stderr and both debugger attempts
are retained. The debugger attachment contaminates that unfinished run; it is
not promotion evidence. Stdout buffering prevents determining its exact
stalled cell solely from its final bytes. A progress-logged, bounded replay
is required before diagnosing a particular kernel or graph mode.

After termination, the prediction test again passed all 372,011 exact checks.
NVIDIA reported 100% utilization while its compute-process list was empty.
This observation alone does not identify an external application or establish
a driver cause. The user was asked whether another GPU workload was active.
Confirmation performance measurements are deferred until the load is understood.
Correctness, sanitizer and non-ranking profiling work can continue.

Windows-native GPU Engine counters subsequently recorded process 23040 at
78.75% and 84.74% utilization of its 3D engine; querying that same PID identified
`cod`. Receipts: `windows-gpu-engines.json` and
`windows-busy-gpu-process.json`. These samples establish concurrent graphics
load at their sampling times, not an exact history of every earlier timing
pair. NVIDIA's current WSL guide additionally lists GPU-utilization and active
compute-process queries among NVML limitations:
https://docs.nvidia.com/cuda/wsl-user-guide/
Thus the WSL utilization/process observations alone cannot identify the load.

The casewise timing ratios remain exploratory observations. They do not pass
the idle-device confirmation requirement, irrespective of bootstrap intervals.

The user subsequently confirmed that a game was active and instructed us to
continue correctness work for now. Performance campaigns and timing-based
selection are deferred. The bounded stalled-cell reproductions completed with
all exact pre/post comparisons passing in both stream and graph modes. Their
event/host values are diagnostic output only; this neither turns the interrupted
full benchmark into a pass nor establishes the cause of its stall.

The user later stated that the GPU was fully available again. Native Windows
counters sampled before confirmation reported no engine above5%; the WSL
device had returned to P8/210MHz. Confirmatory campaigns use new `*-idle`
directories, retain earlier evidence, and run without concurrent heavy CPU
audits or compilation. The restrictions above continue to apply to the initial
gaming-time campaigns; confirmation is separately identified rather than
retroactively repairing earlier timings.
