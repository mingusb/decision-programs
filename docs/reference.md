# Exact GPU import of frozen metric references

Preimplementation decision, 2026-09-23. The four archived real-data `quality.json`
files contain binary64 values emitted by Python JSON, ASCII keys/metadata,
selection metadata and (for multilabel) AP/AUC eligibility counts. The importer
operates on opaque resident bytes; the host transports a fourth file only.
It must not turn a decimal approximation or missing field into a passing gate.

## Decimal contract and algorithm

`parse_decimal` accepts a complete JSON number token: optional minus, JSON's
integer grammar, optional nonempty fraction and exponent. At most 19 significant
digits (counting trailing significant zeros), at most 768 token bytes, and an
effective decimal exponent in [-342,308] are supported. Unsupported ranges and
malformed tokens fail explicitly. All finite binary64 shortest representations
fit this range; the four captured files use at most 17 significant digits.
Underflow rounds to signed zero or a subnormal; finite overflow rounds to signed
infinity, which the metric importer rejects as a nonfinite available metric.
The parser writes output only on success, and preserves negative zero.

[Clinger's exact conversion algorithm](https://scholarsbank.uoregon.edu/server/api/core/bitstreams/ba79372c-1283-483a-9915-24e45d1417dd/content)
provides the relevant exact-integer quotient/remainder approach.
[fast_float](https://github.com/fastfloat/fast_float) is a fast correctly rounded
alternative using cached powers and correction/fallback machinery. For fewer
than 50 scalar metric tokens, choose a small owned fixed-integer converter:
no conversion tables, vendor parser, floating intermediate or fallback path.
This is a bounded correctness-first choice, not a speed claim.

Write the token as sign*M*10^e. Build exact unsigned N=M*5^max(e,0) and
D=5^max(-e,0), retaining the binary factor 2^e separately. Bit lengths and one
shifted comparison yield k=floor(log2(N/D)). Choose quantum q=k+e-52 for normal
numbers, or q=-1074 for subnormals. Shift N or D by e-q. Binary long division
extracts at most 53 quotient bits; twice the exact remainder decides rounding,
with ties resolved by quotient parity. Handle carry into the next exponent,
subnormal-to-normal carry and overflow by constructing binary64 bits directly.

Thirty-six 32-bit limbs provide 1152 bits. Before normalization, N has at most
64+ceil(308*log2(5))=780 bits and D at most ceil(342*log2(5))=795 bits. The
normal scaled numerator/denominator fit within max(bitlen(N),bitlen(D))+54;
the subnormal scale is no larger than normal scaling. Shifted divisor trials
and doubled remainders obey the same bound plus one bit, below 850 bits.
Every shift/multiply checks capacity anyway. There are no unbounded bignums.

## JSON and report contract

The parser follows [JSON's number, whitespace and delimiter syntax](https://www.rfc-editor.org/rfc/rfc8259.html)
for the frozen schema, including ASCII string escapes and `\u00xx` escapes.
Decoded strings are limited to 80 bytes and the complete document to 1 MiB.
Non-ASCII strings, unknown/duplicate keys, arrays/booleans, trailing tokens,
missing required fields and malformed values are rejected, rather than ignored.
The six root keys are reference, fixture_sha256, training_fixture_sha256,
predictions_sha256, metrics and training_mean_baseline. Hash strings are checked
as 64 lowercase hex digits; external provenance verification still authenticates
the file identities. Both metric objects are syntax/schema checked, but only
`metrics` is compared to the frozen prediction computation.

`decode_metric_reference` consumes the completed real-data report on frozen
predictions, plus two caller-capacity reports. It imports the legacy metric
values and copies computed values into the same key order. It independently
requires the task's frozen metric key set, selection metric/value consistency,
clip epsilon and applicable eligibility counts. Null remains unavailable; it
must match the computed availability. Eligibility counts are compared through
the normal metric gate, not silently borrowed from the candidate.

The legacy multilabel evaluator did not emit micro_auc. Only that explicitly
additional metric is absent from this legacy comparison; it remains in the
runner's frozen-versus-new same-engine comparison. Every other applicable
legacy metric is mandatory. This is schema alignment, not a tolerance waiver.

## Integration and independent checks

The optional fourth runner input adds imported-legacy versus recomputed-frozen
metric reports. Record two separate gates: exact arithmetic bits (including
signed zero) and directional zero-allowance quality. Both must pass; a metric
improvement cannot hide arithmetic nonconformance. Prediction bit conformance
and frozen-versus-new quality remain separate gates and continue reporting even
when the legacy arithmetic gate fails.

GPU tests use literal expected bit patterns for .1, minimum subnormal, normal
boundary, DBL_MAX, adjacent values, exact integer halfway cases and signed zero;
malformed and unsupported numbers must leave output unchanged. JSON fixtures
cover all task schemas, duplicate/missing/unknown keys, null availability,
eligibility metadata, metadata consistency and malformed structure. Fixtures,
parsing and comparisons execute only on GPU. Root runs these checks and all four
real files serially; compile success is not a conformance result. A future full
operation measurement includes transport separately and parsing, comparisons,
both metric passes and completion together, with raw observations retained.

## Compile receipt

`src/reference.cu`, `tests/reference_checks.cu` and the extended quality runner
compile with CUDA 13.4, C++23, O3, sm86, RDC and test assertions enabled. CMake's
`reference_checks` and `quality` targets then built and linked serially. Logs and
objects are retained in `/tmp/gh-reference-compile`, including the initial
target-before-CMake-regeneration failure and the subsequent successful build.
The test source contains 42 decimal landmarks, 4096 independent integer-lattice
rounding checks and 27 JSON cases. These are prepared cases, not an execution
receipt. No GPU work was performed by the implementing agent.

Ptxas reports 536-byte stack/100-byte spill traffic for decimal conversion and
1512-byte stack/324-byte spill traffic for JSON import. This bounded scalar
metadata path is correctness-first; no performance ranking is inferred. An
independent source review found no concrete arithmetic or alignment blocker,
which is distinct from the still-required root GPU and frozen-file checks.
