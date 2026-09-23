# Completed-tree export batching experiment

Decision recorded after the first resident campaign and Nsight Systems capture,
before implementing this change. Keep scalar tree arithmetic, GPU preparation,
histogram policies and device frontier construction unchanged.

The first resident implementation waits twice per output tree: download its
status/node count, then download precisely that many nodes. For 129 outputs and
three rounds, Nsight observed 790 stream synchronizations, including the 774
tree-export waits. Export scopes occupied 152.54 ms of a 187.86 ms instrumented
boosting interval, overlapping 107.15 ms of GPU tree-build event intervals.
These durations are not additive and profiling is not a speed ranking. The
uninstrumented 129-output regression is 133.43 ms total for the hybrid baseline
versus 146.25 ms for the first resident graph candidate.

Compare the existing exact-length, two-wait export against bounded asynchronous
export batches. Enqueue each complete tree, its status, and its full bounded
node-capacity buffer to dedicated pinned host slots on the same stream. Reuse
the single device builder only after its queued copy, without a host wait.
Wait once per batch, check all statuses, and retain only actual nodes in the
public model. Derivative tiles remain intact until all associated trees finish.
Multiclass retains the existing full pre-round gradient snapshot.

This trades extra bytes for fewer host dependencies. At depth2, a full tree is
seven 32-byte nodes: copying an extra leaf capacity is negligible compared to
hundreds of WSL stream waits. At large depths it may be expensive. Keep compact
export selectable, expose batch size, cap pinned node staging to 64 MiB by
reducing the effective batch, and fall back to compact export if even one full
tree would exceed that bound. Report the effective batch size (zero means
compact); it is bounded by the independent derivative tile and the requested
export batch, and by output count. Include pinned allocation/setup and every
extra copy in end-to-end timing. Host pinned memory is reported separately
from the GPU payload budget.

Each queued tree needs its own persistent pinned selector and status slot.
Slots may not be overwritten until the batch completes. On exceptions, drain
the stream before destroying any pinned backing. Check statuses before copying
possibly failed model content into host model objects. Copies may include
unused capacity bytes; zero the node buffer once outside training so these
bytes are initialized. Actual tree nodes are initialized by the device builder.

Alternatives: conditional graph/device-side compact-copy lengths are constrained
by runtime APIs; retaining every complete tree on the GPU adds model-sized
device memory; pinned fixed-capacity batches require no additional device node
storage. Compare compact, batch1, batch16 and batch32 on identical scalar and
129/1024/4096-output cases with forced histogram policy, alternating orders.
Measure full training and held-out metrics; run graph and stream correctness,
short tiles, memory caps, sanitizer and Nsight checks. An export optimization
must not alter bins, tree rules or introduce an algorithm-library dependency.
