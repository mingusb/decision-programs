# Legacy CSV export

Contract and selection recorded before implementation, 2026-09-23. The frozen
synthetic evaluator consumes row_id plus target/target_N and weight columns, or
prediction/prediction_N/pN columns. Preserve row-major order, LF, decimal row IDs,
C-locale defaultfloat precision 17, including negative zero. Target values and
weights are converted from float to double on GPU without loss before submission.
Inputs must be finite; weights are nonnegative and default to one. Empty rows
still emit a header. Supplied capacities and every byte product are checked.

An independent GPU integer formatter preserves the exact binary64 dyadic value,
rounds its decimal expansion to 17 significant digits with ties to even, trims
trailing zeros, and selects fixed/scientific notation using the rounded exponent.
Parallelism is across values; fixed 25-byte slots avoid atomic append and retain
deterministic order. A hierarchical integer scan compacts the lengths into one
caller-owned byte buffer. Header and row IDs are also GPU-generated.

[Ryu Printf](https://github.com/ulfjack/ryu) provides relevant table-based fixed
precision conversion; its shortest-decimal algorithm alone does not reproduce
precision-17 output. A table-based multiply/shift implementation should do less
arithmetic than exact dyadic expansion. The initial bounded expansion is selected
as a compact compatibility implementation for an external export boundary, not
as a fastest formatter. It uses at most 86 base-10^9 limbs and about 90 small
multiplication passes in the extreme subnormal case; ordinary integers terminate
much earlier. This cost is separate from resident training and must remain in
export-inclusive timings. No host number formatting or printf-based conversion
is used.

The fair replacement experiment is complete CSV generation with identical bytes,
including formatting, temporary slots, scan, compaction and capacity handling.
Use ordinary predictions, signed zeros, powers of ten, subnormals, rounding ties,
and scientific/fixed boundary values. First verify literal independently specified
bytes and decimal rounding boundaries on GPU; then compare full-operation timing
against a table-based implementation when the device is idle. Preserve any failure.
