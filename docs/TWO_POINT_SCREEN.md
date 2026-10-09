# Two-point screen before rival proof search

The shared adaptive converter checks two feasible inputs after its common cover
fails to certify a class. If the unchanged native-qualified score rule certifies
different labels, no uniform-class proof can succeed on that region. The
converter skips further rival proof search and continues ordinary construction.
The screen is enabled internally by default; it adds no public conversion option.

The inputs belong to the intersection of the saved witness domain and the current
projected residual domain. Numeric bounds are intersected, missing permissions
are ANDed, and each whole exactly-one group uses the intersection of its masks.
The first input uses finite lower endpoints and first allowed categories; the
second uses finite upper endpoints and last allowed categories. A missing-only
coordinate uses NaN. Identical representatives do not require a second walk.

Both evaluations preserve saved prefix words and the original active-source
addition order, including consumed roots and repeated slots. Each completed
point must separately pass the existing all-channel finite score window and
positive-gap rule. Invalid geometry, scratch overlap, incomplete work,
nonfinite scores, ties, an unqualified gap, or equal labels supply no negative
certificate. Point agreement never proves the region uniform. Before any
remaining rival search, the original whole-region static bounds are restored.

The existing Lean theorem
[`different_certified_parts_exclude_uniform`](../formal/region_proof_synthesis/RegionEnvelope.lean)
provides the mathematical consequence of two nonempty certified subsets with
different labels. No new Lean module is required. Domain membership, ordered
FP32 evaluation, and the native qualification boundary remain implementation
obligations; this is not a formal verification of CUDA.

Both point walks share the existing cover visit allowance. The first also
supplies the ordinary candidate proposal. A second walk can displace later
proof work, so this screen is an optimization with a cost, not additional
acceptance authority. Its failure result does not bypass graph construction.

## Storage and checkpoints

A disposable `PointScreen` record belongs to each private draft child. Its
storage is included in checked scratch estimates. Observation counters live in
`ProofCounters` and reset on process resume. Neither these records nor
`EngineView` are checkpoint payloads. The persistent state, node, status and
control layouts, section schema and source/native identity are unchanged.
Existing checkpoints therefore need no format migration for this screen.

## Validation and measured scope

The promoted source passed a focused CUDA fixture with 559 checks and a
Compute Sanitizer run with zero errors or leaks. The maintained test is
`adaptive_point_screen_checks`; it covers both qualification failures and
witness-domain restrictions, missing values, category bit 64, alias refusal,
source-order rounding and bounded work.

On 2026-10-09, separate untouched-baseline and candidate executables were
compared in baseline/candidate/candidate/baseline order on eight restricted
regions of the 448-tree source. Each process used two measured repetitions per
region and policy with fresh native qualification. Sums of region median times
were:

| Proof effort | Previous default | Two-point screen | Change |
| --- | ---: | ---: | ---: |
| Fixed | ~700 ms | ~630 ms | ~10% lower |
| Dynamic | ~490 ms | ~480 ms | ~2.5% lower |

Every native comparison agreed and all resulting graph sizes were unchanged.
Some individual regions were slower; dynamic construction also created four
additional states in one region. Peak owned GPU storage increased by about 3 KB
in this setup. These measurements support this workload only. They do not
establish a universal speedup or completion of the full 448-tree conversion.
The experiment snapshots and detailed receipts remain private build artifacts.

The diagnostic benchmark supports `--method two_point` for explicit off/on
comparison. Relational and unary comparisons retain this maintained screen in
both arms. The alternative first-point gap-only screen remains private: it
found no refutations in these eight regions and did not improve runtime.

## Maintained build and larger withheld region

The normal CLI, converter backend, regional benchmark and registered GPU test
were rebuilt successfully after adoption. The maintained GPU fixture passed all
559 checks. A fresh eight-region run again produced zero native mismatches and
unchanged final graph sizes.

An additional frozen region, excluded from the eight timing-selection regions,
completed with all approximately one million native signatures agreeing.
With dynamic effort, conversion fell from about 1.2 seconds to 0.9 seconds
(about 20% less time); fixed effort fell from about 2.0 seconds to 1.8 seconds
(about 10% less time). The final graph remained 372 nodes. Dynamic construction
created about 2% more states while source-node visits in cover proofs fell by
about 36%. Additional owned memory remained about 3 KB.

This larger-region comparison used one off/on repetition per policy. It is a
separate scaling observation, not a repeated estimate of whole-model speed.
The full 448-tree conversion remains incomplete.
