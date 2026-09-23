# NVIDIA histogram256 published baseline

This directory vendors the NVIDIA CUDA Samples histogram256 implementation at commit
`5443602d89ed99aede2e4b7bf329daddeadb320e` from the official
[NVIDIA/cuda-samples repository](https://github.com/NVIDIA/cuda-samples/tree/5443602d89ed99aede2e4b7bf329daddeadb320e/cpp/2_Concepts_and_Techniques/histogram).
The upstream files and LICENSE are preserved byte-for-byte. NVIDIA's BSD-style
copyright and redistribution notices apply; see [LICENSE](LICENSE).

| Local file | Upstream path | SHA256 |
|---|---|---|
| `upstream/histogram256.cu` | `cpp/2_Concepts_and_Techniques/histogram/histogram256.cu` | `c09a3e29dd44dfa7f6238fc5d4803266325ef9123c9c2ed1cd44367aefdbf490` |
| `upstream/histogram_common.h` | `cpp/2_Concepts_and_Techniques/histogram/histogram_common.h` | `3b2cc12f357bd80135cab0eb19b404fd3f6ba1a0ee29890602aa4f2e905c76a8` |
| `LICENSE` | `LICENSE` | `b3e40c5bfed1fca5c62d2c1f2208bf51f8d2c910219f94c443f657ace9001be3` |

## Adaptation

`histogram256_kernels.cuh` contains the complete upstream source prefix preceding
the `Host interface to GPU histogram` section. Its only edits are replacing the
host-only `helper_cuda.h` include with a comment and changing the common header
include to `upstream/histogram_common.h`. Both device kernels, helper functions,
and policy constants are unchanged. No CUDA intrinsic compatibility changes were
needed. The adapted header SHA256 is
`1010b8aa30986d46fc392098c6bb29dff244d5e10df06225872beba0e75fa008`.

`bench/sample256.cu` supplies an allocation-free, stream-aware host interface with
CUDA error returns, replacing the sample's process-global workspace allocation
and process-exiting error helpers. It retains the original 240 partial histogram
blocks, 192 threads per partial block, six warp-private shared histograms per
block, and 256 merge blocks with 256 threads each. Requested tuning and grid
values do not alter that published policy.

The supported contract is byte input (`u8`), exactly 256 bins, unsigned 32-bit
output, native local counters, and input size divisible by four and no larger
than `UINT32_MAX` bytes. Input, output, and workspace require four-byte alignment.
The nonempty operation uses 245,760 workspace bytes. It performs all shared
initialization, partial-histogram writes, and the final merge inside the timed
operation. No prior output/workspace clear is required. The empty-input adapter
clears the output directly. There is no conversion, padding, truncation, or data
reordering. The upstream uint input loads are interpreted as four byte samples,
not as one u32 bin ID.

## Why GPU Multisplit is not vendored

The initially considered GPU Multisplit source at
`cd529b6495cdb91237a27acba0e876aec409c6a1` has a restricted
[license](https://github.com/owensgroup/GpuMultisplit/blob/cd529b6495cdb91237a27acba0e876aec409c6a1/LICENSE.md).
Its permission is limited to use:

> by nonprofit educational or research institutions for noncommercial use only

This project has not established that eligibility. No GPU Multisplit source is
vendored or compiled. NVIDIA's sample provides a separately identified published
baseline under its own license; its results must not be labeled GPU Multisplit.

The adapter and sample kernels are benchmark-only. They are excluded from the production `gh` library and from builds configured with `GH_BUILD_BENCHMARKS=OFF`.
