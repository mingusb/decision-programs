# Independent-objective loss layout experiment

Decision recorded before changing the kernel. In the first 16M-row, one-output
preparation diagnostic, Nsight Systems measured 34.31 ms for the initial loss
kernel (66.8% of recorded kernel time). The existing implementation assigns a
warp to each row for all objectives. For one independent output, only one lane
does useful arithmetic and neighboring active lanes read widely separated rows.

The known relevant reduction approach is a coalesced grid-stride transform into
register sums, warp-shuffle/block reduction, then a bounded second reduction.
The existing owned block reduction and final scalar normalization already
implement the latter steps. The problem here is mapping independent elements
to lanes, not choosing an external reduction library. Keep the multiclass
warp-per-row log-sum-exp, where class coupling actually requires row reduction.

For squared error and independent binary cross entropy, compare the existing
warp-per-row baseline with one thread per row/output element. Adjacent threads
read contiguous double predictions and float targets; weights are read by row
and reused through cache across outputs. Input traffic stays at 12 bytes per
element plus weights, partial sums remain bounded by the caller's block count,
and no new allocation, atomics or launch is needed. Use a compile-time single-
output specialization to remove element-to-row division for the sparse case.
Preserve the existing stable binary-loss formula and exact weighting convention.
Floating-point reduction order changes and must be reported and tolerance-tested.

Alternatives include vectorized multi-element loads and fusing loss with the
next round's gradient calculation. Fusion could remove a read pass but changes
when the public per-round objective is available and needs a separate scheduling
experiment. Do not silently omit requested loss reporting or change precision.

Validate against independent long-double objective references, including zero
weights, extreme margins, irregular row/output counts and wide targets. Existing
kernel tests cover independent and multiclass loss with output canaries. Run
sanitizer and compare kernel diagnostics separately from full train time on the
same seeds/shapes; preserve the previous implementation in first-provenance.
