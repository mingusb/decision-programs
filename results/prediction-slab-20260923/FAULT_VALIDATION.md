# E256 exceptional-path validation

Recorded before writing the diagnostic harness. This is explicit validation,
not a production algorithm or a performance measurement.

Link a small CUDA C++23 host caller against the same built booster archive,
using GNU linker wrappers for cudaMalloc, cudaFree and cudaMemcpyAsync. After
one successful E call records the operation counts, inject one returned error
at each allocation and asynchronous-copy position, separately. Each injected
call must propagate an exception, release all successfully allocated tracked
device buffers, and permit a subsequent successful prediction with the exact
reference output. Also wrap the aligned array allocation used by the host slab,
inject `std::bad_alloc`, and require the same device cleanup and recovery.
Exercise nonempty and empty forests. Run this diagnostic
under padded memcheck with full leak checking.

The wrappers do not exhaust physical GPU memory or poison the CUDA context.
They test handling of synchronous CUDA API error returns; they do not establish
correctness after device loss, asynchronous execution faults, allocator
fragmentation, other host allocation failures, or arbitrary runtime internal errors.
The pure layout test separately checks arithmetic and budget boundaries without
large allocations. No diagnostic timing will be used for ranking.
