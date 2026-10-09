# Proof collection

This directory contains the mathematical arguments supporting the maintained
Decision Programs algorithms. Build and independently replay every module with
`cmake --build BUILD_DIRECTORY --target check-proofs`; set
`DP_LEAN_EXECUTABLE` to Lean 4.34.1 when configuring. The dependency order is in
`cmake/CheckLean.cmake`. The maintained collection registers 28 modules, each
with one canonical source.

## Start with the central argument

A decision program is correct when each terminal has a sound class certificate
and every branch routes to regions whose certificates remain valid. Local
certificates then compose into correctness for the whole program.

Read these entry points first:

| Question | Entry point |
| --- | --- |
| Why does a certified tree return the source class? | `ConverterGuarantees.certified_correct` in `CorrectnessCompletion.lean` |
| Why does the argument survive shared subtrees? | `ConverterSharedDAG.local_contracts_compose` in `SharedDecisionDAG.lean` |
| Why can complete finite-region construction terminate? | `ConverterGuarantees.direct_completion` under its progress premises |
| How is remaining construction bounded? | `ConverterSize.Run.remaining_commits_bound` and `ConverterSize.Run.completed_node_bound` |
| Why can bounded traversal stop early without losing input coverage? | `ConverterApplicabilityCache.frontier_floor_sound` in `ApplicabilityCache.lean`, deriving coverage from explicit work-list transitions |
| Why may a region be closed without enumerating its inputs? | `RegionEnvelope.common_winner_cover` and `rival_specific_covers` |
| When do finite margin intervals select the first maximum, including ties? | `ConverterArithmetic.interval_first_winner_selected` in `OrderedArithmetic.lean`; requires native margin correspondence for direct-class use |
| When is one class across a region impossible? | `RegionEnvelope.different_certified_parts_exclude_uniform`, requiring two nonempty certified subsets with distinct labels |
| When is the requested positive-gap proof impossible? | `RegionEnvelope.opposing_bounds_refute_positive_gap`, requiring nonempty input and opposing score enclosures; no mixed-class conclusion |
| Why can shared contributions establish a class margin despite overlapping score intervals? | `ConverterRelationalMargins.ordered_margin_encloses` and `deferred_margin_encloses` in `RelationalMargins.lean` |
| Why may all contributions on one input coordinate be bounded together? | `ConverterGroupedMargins.exhaustive_axis_floor` and `grouped_margin_encloses` in `GroupedMargins.lean` |
| Why can a smaller score gap certify the qualified native class? | `ConverterNativeClassSeparation.rounded_shift` and `conditional_probability_order` in `NativeClassSeparation.lean` |

Completion and size statements have explicit premises. A finite upper bound does not assert that every model has a
small representation or finishes quickly.

The current research baseline is the default ordered refinement, joint-bound,
and rival-cover path, not static bounds alone. The implementation-to-theorem
map and the limits of current measurements are in
[region proof methods](../docs/REGION_PROOFS.md#characterize-the-default-before-extending-it).
Stronger experimental bounds must demonstrate an advantage over that baseline.

Cache dependency reasoning now reuses `SignatureEnumeration.predicates` and
`same_questions_same_leaf`, retaining the public `ApplicabilityCache` alias and
theorem wrapper. This removes a second source-tree induction while preserving
its guarantees. The consolidated module and five dependencies compiled and
passed independent kernel replay; receipt:
`build/proof-research/cache-signature-consolidation-20261009/result.json`.

The cache applicability argument now uses an arbitrary semantic coordinate type,
rather than twelve fixed coordinates. Its proof needs an independent product
domain, an admitted donor witness, dependence on retained coordinates for both
the reusable program and residual source, and fresh source correspondence.
It does not enumerate the coordinates or assume a particular dataset. Categorical
groups are individual semantic coordinates, not independently free indicator
bits. The changed module and its three dependencies compiled and passed
independent kernel replay with Lean 4.34.1; see
`build/proof-research/generic-cache-kernel-20261009/result.json`.

The bounded-frontier bridge derives exhaustive coverage from the existing
work-list transitions, including conservative absorption when work stops.
It still requires valid subtree bounds, correct aggregation, sound forced
branches and an actual transition trace. It does not assert CUDA refinement.
Its four dependency modules passed standalone compilation and independent
kernel replay with Lean 4.34.1 on 9 October 2026.

## Supporting arguments

- **Input regions:** `DomainPartitions` holds the reusable threshold-rank,
  finite-word, interval, and category-partition facts. `SignatureEnumeration`
  and `MixedRadixCoverage` cover finite enumeration and accounting. The retired
  fixed Forest reference constructor is excluded.
- **Shared storage:** `SharedDecisionDAG`, `OnlineArenaContracts`, and
  `CollectedRoots` cover topology, construction invariants, and reachable-node
  retention. `GraftBounds` covers certified replacement of a pending region.
- **Simplification:** `DynamicAtomRestriction`, `AdjacentPredicateBypass`,
  `LiteralPathBound`, `OrderedBranchLaws`, `TreeFactor`, `ApplicabilityCache`,
  `HardAxisRewrites`, and `HardRewriteExhaustion` cover the corresponding
  restrictions, rewrites, and safe reuse conditions.
- **Score bounds:** `OrderedArithmetic`, `PairedBounds`, `RelationalMargins`,
  `GroupedMargins`, `RegionEnvelope`, and `CoverRefinement` cover ordered arithmetic, compatible
  residual bounds, cross-class differences with rounding error, exhaustive
  covers, and retention of sound bounds after an unsuccessful try.
- **Equations and audit operations:** `DecisionEquations` distinguishes hard
  decision identities from their soft extensions. `ScoreDiagramCongruence`,
  `ScoreAddApplyCongruence`, and `ClassApplyCongruence` cover the maintained
  score-factor, addition, and class audit operations. The latter two currently
  model the optional source-qualified RL pipeline with twelve rank/category
  coordinates and seven score channels; they are not a general-domain theorem
  for the adaptive converter. Their runtime counterparts live in
  `native/rl/class_rank_gpu_score_diagram_factor_audit.cu`,
  `native/rl/class_rank_gpu_score_add_apply_audit.cu`, and
  `native/rl/class_rank_gpu_class_apply.cu`.

The shared frontier arithmetic lives in `CorrectnessCompletion`; size and
graft results reuse it. Benchmark-specific threshold tables, retired converter
algorithms, old serialization byte formulas, and illustrative arithmetic tests
are not part of this collection.

## Relational margin argument

`RelationalMargins` pairs compatible residual leaves across a winner and rival
while retaining both original rounded folds. It proves that the prefix
difference plus lower bounds on leaf differences, minus bounded rounding error,
is a lower bound on the final score difference. An unmatched channel contributes
no additional rounding operation. `deferred_error_identity` and
`deferred_margin_encloses` justify subtracting accumulated error at the end of
the bound calculation without reassociating the source floating-point sum.

The nearest-spacing theorem requires finite neighboring representable values
and supplied spacing bounds. It now admits neighbor gaps up to twice the error
allowance, proving the half-spacing bound in a common exact integer unit fine
enough to represent that half-spacing. The strengthened existing theorem and
its dependencies passed independent kernel replay, without adding a module.
Coverage and directed-arithmetic guarantees remain explicit premises; the
module does not derive IEEE spacing or certify CUDA instructions. Its shared-contribution example establishes a mathematical
proof opportunity, not a measured model speedup. On 9 October 2026, all 26
maintained modules compiled with Lean 4.34.1 and passed independent kernel
replay, including the relational and deferred-error arguments. Focused CUDA
checks also passed, but the selected real-region benchmark showed no construction
reduction and increased runtime; the method remains opt-in. See the measured
comparison in the region-proof documentation.

The arithmetic follows elementary identities such as Metamath's
[order under subtraction](https://us.metamath.org/mpeuni/le2subd.html) and
[finite-sum comparison](https://us.metamath.org/mpeuni/fsumle.html). These are
references for the argument, not imported proofs or a mathematical novelty
claim. See [region proof methods](../docs/REGION_PROOFS.md) for the bounded CUDA
fallback, shared dynamic allocation, unchanged native gate, and comparison
flags including `--relational-bounds` / `relational_bounds: true` (opt-in).

## Grouped unary margin argument

`GroupedMargins` groups all signed residual contributions that depend on the
same numeric coordinate or exactly-one category group within the current
region. The sum of conservative group minima bounds the original signed leaf
sum; unresolved factors can remain singleton groups. The original source
addition order and rounding-error allowances are unchanged.

`exhaustive_axis_floor` explicitly requires region-restricted unary dependence,
coverage of every allowed interval signature or canonical representative by the
finite atoms, and valid bounds at those atoms. Arbitrary numeric measurements
map to their threshold-interval representative; the grouped function must be
constant over that represented interval. `grouped_margin_encloses` additionally requires that the
groups account for the original signed operands exactly once, a conservative
prefix margin, the original ordered rounding-error envelopes, and a directed
final calculation. `unchanged_native_gate` retains the authentic qualified
prediction gate. These premises are not a formal correspondence proof for the
CUDA walker, atom enumerator, or floating-point instructions.

The four-coordinate example has two positive half-contributions and one
negative whole-contribution per coordinate. Disjoint cross-class pairs and a
single-predicate independent cover leave a negative bound, while complete
groups cancel and preserve a strict margin after conservative rounding error.
This is a mathematical opportunity, not a measured workload speedup. The general
unused potential-transfer framework remains outside the maintained collection.

On 9 October 2026, all 27 maintained modules, including `GroupedMargins`,
compiled with Lean 4.34.1 and passed independent kernel replay.

`first_atom_no_improvement` supports an implemented short circuit: once a
computed atom floor is no larger than the independent incumbent, the current
minimum-then-maximum routine cannot improve that incumbent. It is a theorem
about the computed routine, not a claim that no stronger mathematical bound
exists. This addition and its dependencies also passed independent kernel
replay before CUDA implementation. Measured source visits decreased, while
real-region total time remained above the control without grouping.

The computed_group_bounds and optimistic_rival_rejection theorems extend this
to a whole rival comparison. First-atom caps bound the numerical floors
produced by this routine; monotonicity of the same ordered directed fold
propagates those caps through the final margin. A cap below the native threshold
proves this attempt cannot succeed. It does not bound the true model margin or
rule out a different certificate. These additions also passed independent
kernel replay before CUDA implementation. The real-region measurements reduced
source visits but still did not beat the total runtime of the control.

## Native class separation

`NativeClassSeparation` supplies the numerical argument used by the shared
`native/class_native_softprob_contract.hpp` threshold. A computed FP64 gap of
at least 2^-14, with a one-sided subtraction error of at most 2^-48, exceeds
the exactly representable shift gap 3*2^-16. Applying monotonic RN32 rounding
to that representable comparison bounds the shifted rival. The existing
symmetric exponential relative-error allowance 2^-16 and division allowance
2^-20 then establish strictly ordered probabilities with one positive common
denominator. No exact `exp(0)=1` premise is added.

The maintained theorem takes the exponential envelopes, division envelopes,
encoding and rounding properties as explicit premises. The existing runtime
qualification still binds the model, native library, loaded kernel, process,
device and configuration; the shared constant does not establish these facts.
The production host gate and all CUDA region-closing guards use that same
constant. The threshold is 16 times smaller than the former 2^-10 guard;
that factor describes numerical sensitivity, not runtime speedup.

The whole-collection run on 9 October 2026 compiled and independently replayed
all 28 maintained modules with Lean 4.34.1. Its preserved snapshots include the
negative certificates, tie-aware selector, generic cache coordinates, shared
source-question argument and consolidated pair extrema. Receipt:
`build/proof-research/maintained-simplification-full-replay/1d30583dc639d44f/result.json`.
The pair simplification preserves its four public extrema lemmas while deriving
them from two private characterizations instead of four repeated inductions.

After that run, `GroupedMargins.computed_min_le_first` replaced its duplicate
induction with the standard-library `List.foldl_min` theorem. The private name,
public statements and hypotheses are unchanged; the module shrank from 301 to
299 lines. The changed module and its four dependencies subsequently compiled
and passed independent kernel replay:
`build/proof-research/grouped-fold-min-kernel-20261009T131718Z/result.json`.
The full 28-module collection has not been replayed together after this final
proof-body simplification.

A separate 196-line private prefix-interval prototype preserves a nonconstant
program's tie-aware raw-margin winner, and extends this to a positive true gap
and finite score window, under region-wide endpoint certificates and monotone
single-channel folds. Its native corollary restricts the rival exponential
envelope to the derived window [-20,-3*2^-16]. Rounding, division bounds, one
positive common denominator and actual native execution correspondence remain
explicit premises. The prototype and its two maintained
dependencies passed elaboration, independent kernel replay and review:
`build/proof-research/prefix-gap-window-scoped-kernel-20261009/result.json`.
It remains outside this maintained collection; no runtime reuse or performance
benefit is established. See the private-research discussion in
[region proof methods](../docs/REGION_PROOFS.md#single-channel-prefix-interpolation-private-research).

## What kernel checking establishes

This is an abstract mathematical library, not a formal verification of the CUDA
implementation. Lean checks each stated theorem, including its hypotheses. There
is no machine-checked refinement from the C++/CUDA program to these models.
Implementation tests do not supply that missing formal proof. Native
prediction qualification, source/domain binding, numerical implementation tests,
and completed-versus-bounded run reporting remain separate responsibilities.
The production code does not execute these Lean proofs on the GPU. A line count
is neither a proof requirement nor a measure of implementation assurance. The
central correctness argument is the short induction in `certified_correct`; the
other modules establish distinct premises or guarantees for supported operations.

New mathematical experiments should stay outside this maintained collection
until they support an implemented method or an independently useful current
guarantee. Lack of another Lean consumer is not enough to delete an exported
correctness, completion, coverage, or size theorem.
