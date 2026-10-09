# Region proof methods

Decision Programs can avoid expanding a region when sound score enclosures,
together with its existing native prediction gate, establish one output class
throughout that region. These additions extend the maintained CUDA converter;
they do not create a second converter.

## Characterize the default before extending it

The research objective is to explain the maintained implementation, then derive
stronger or cheaper algorithms from that explanation. A stronger mathematical
bound is not, by itself, a faster converter. The ordinary configuration requests
joint bounds and rival covers; relational and unary bounds remain opt-in
(`native/class_study_convert.hpp`, `ConversionOptions`). This default is the
comparison baseline, not static independent bounds alone.

| Maintained operation | Mathematical entry point | Scope |
| --- | --- | --- |
| Restrict residual trees | `ApplicabilityCache.residualize_correct` and `residual_ordered_words` | Preserve selected source operands and their order |
| Refine one subtree with bounded work | `ApplicabilityCache.frontier_floor_sound` | Preserve exhaustive coverage through explored and deferred subtrees |
| Accumulate score intervals | `OrderedArithmetic.ordered_reduction_encloses` | Bound the original ordered additions under explicit rounding premises |
| Bound adjacent residual pairs | `PairedBounds.paired_reduction_encloses`; `CoverRefinement.join_certificates_sound` and `retain_sound` | Exclude incompatible pairs, then retain the conditioned incumbent by interval intersection |
| Close a region through proof-only splits | `RegionEnvelope.common_winner_cover` and `rival_specific_covers` | Cover every allowed input, independently for each rival if useful |
| Establish qualified native probability order | `NativeClassSeparation.rounded_shift` and `conditional_probability_order` | Preserve strict class separation under the existing native error and runtime premises |
| Publish certified decisions | `CorrectnessCompletion.certified_correct` and `SharedDecisionDAG.certified_DAG_correct` | Compose local certificates into source-class preservation |

These are abstract algorithm guarantees. Source/domain validity, numeric
rounding and native qualification, private CUDA storage, and correspondence of
actual operations to the modeled transitions remain explicit obligations.

### What independent score bounds can and cannot decide

For an attained Cartesian product of finite score intervals, a fixed class wins
throughout exactly when its lower endpoint strictly exceeds the upper endpoint
of every earlier class and is at least every later class upper endpoint. The
asymmetry preserves first-index ties. The existing
`OrderedArithmetic.interval_first_winner_selected` supplies sufficiency.
Necessity follows by choosing the winner-lower/rival-upper corner. An actual
model need not attain that corner, even when each marginal interval is exact.

For example, scores `(x + 1/4, x)` at `x` equal to zero or one always have a
quarter-point gap. Their exact individual ranges nevertheless overlap. This
lost relationship differs from actual mixed predictions and from refusal of
the sufficient native numerical guard. A failed interval test establishes none
of those explanations by itself. Only feasible inputs with differing native
labels establish mixedness; a stronger region argument is needed to establish
that correlation loss caused a missed certificate.

The current qualified positive-gap predicate is stronger than raw first-index
argmax. It can be decided using just the largest lower endpoint and largest
competing upper endpoint, as described below. This decision equivalence does
not make the chosen class maximize the signed failed margin, and does not make
the native sufficient predicate complete for all constant-class regions.

### Why a failed pair refinement cannot weaken the incumbent

`paired_reduction_no_looser` compares a complete pair calculation with its own
independent block bounds. It does not state that every budget-limited pair
candidate dominates the preceding conditioned per-tree calculation. The CUDA
pair pass may use static bounds for an unpaired or unvisited residual, so its
raw candidate can be weaker than that incumbent.

The runtime guarantee comes from intersecting two independently sound finite
intervals: retain `max(old lower, candidate lower)` and
`min(old upper, candidate upper)`. Invalid or disjoint numerical results leave
the incumbent unchanged. `CoverRefinement.join_certificates_sound` and
`retain_sound`, with the reversed order for upper bounds, justify that actual
retention step. This source-to-proof mapping concerns
`native/class_conversion/adaptive_effort.cuh`; it does not change its behavior.

### Why a bounded traversal can still certify

`conditioned_subtree_extrema` retains static bounds for every subtree it cannot
explore. On a stack or visit limit, it absorbs the deferred subtree bounds; it
does not discard the deferred cases. The first absorbed bound replaces the
initial root bound, and subsequent bounds enlarge that collected enclosure.
Consequently the input coverage can be complete while the traversal is not.

The maintained frontier proof derives coverage from initialization, absorption,
sound forced decisions, and keeping both branches at an unresolved decision.
Draining the pending list preserves that coverage. Static bounds and sound
aggregation then establish the final enclosure. The proof reuses the existing
tree and execution types, adding no second tree framework. Logical transitions
are not visit counters: draining and zero-budget fallback need not charge a
node visit. The changed module and its dependencies passed standalone Lean
4.34.1 compilation and independent kernel replay on 9 October 2026.

### When per-tree refinement is already exact

For finite numeric inputs, suppose the query region is a nonempty product of
intervals and each source leaf path has a finite feasible input. If every edge
condition on a path individually intersects the query region, the entire path
intersects it. For a coordinate, write the path interval as `[pLo,pHi]` and the
query interval as `[rLo,rHi]`. Path feasibility and the individual intersections
imply

```text
pLo <= pHi, rLo <= rHi, pLo <= rHi, rLo <= pHi,
therefore max(pLo,rLo) <= min(pHi,rHi).
```

Coordinates combine independently. Strict cuts can be expressed as closed
intervals of ordered finite signatures. This is an interval-intersection
argument related to the one-dimensional Helly property; Metamath's
[`iccin`](https://us.metamath.org/mpeuni/iccin.html) records the closed-interval
intersection identity. That real-interval result is a reference, not an import
or a proof of the finite-word encoding.

Under these hypotheses a completed fixed-region traversal already finds
exactly the reachable leaves. Carrying more ancestor restrictions cannot
tighten its completed per-tree extrema. This does not make the ensemble score
bound exact: maxima of different trees can still require contradictory inputs.
Interrupted traversal can also retain a wider static subtree bound.

The hypotheses matter. In an exactly-one group `{a,b,c}`, the path excluding
`a`, then `b`, has the globally valid outcome `c`. Query `{a,b}` meets each
exclusion separately but not both. Missing routes give another counterexample:
with missing values sent left at both tests, the path `x < 1`, then `x >= 0`,
is finite `[0,1)`. Query `[2,3]` plus NaN meets each edge separately but cannot
follow the whole path. Neither case permits the interval conclusion.

Read-only inspection of the qualified 448-tree source found all 40,347 leaf
paths feasible over independent finite numeric coordinates. The eight saved
benchmark regions prohibit NaNs and fix one category in each group, so the
numeric argument applies after those category decisions are forced. This is a
characterization of those regions, not a claim about every allowed full-domain
region or every imported model. A scratch Lean proof of the interval and coordinatewise implications passed
independent kernel replay. Connecting concrete source paths and canonical FP32
keys to the completed walker remains a separate obligation; the scratch proof
is not part of the maintained collection.

### What the existing measurements establish

In the saved eight-region comparison, the default baseline produced a total of
132 output nodes with no native mismatches over about 612,000 signature
evaluations summed across the regions. Two regions closed to a single node
while representing about 272,000 and 289,000 signatures, respectively, with a
recorded successful joint-bound certificate. The counts are not asserted to
be disjoint full-domain coverage. The complete 448-tree domain conversion
remains unfinished.

Some attribution is still missing. Aggregate class pruning includes terminal
point certificates, direct counters exclude nested cover work, and the saved
JSON omits several refinement and cover counters already held in memory.
Before claiming which mechanism supplies the gain, compare static bounds,
conditioned traversal, joint bounds, common covers, and rival covers with
matched fixed allowances. Also replay bounds on identical states; full runs
change their later state populations when an earlier proof succeeds. Evaluate
dynamic scheduling separately. The existing batch-32 regional timings are
controlled comparisons, not peak production-throughput measurements.

## Tighter native class-separation guard

The shared computed-gap requirement is now 2^-14 instead of 2^-10, using the
same qualified symmetric numerical assumptions. This addresses the small-gap
barrier identified by the internal-state audit below, without changing source
precision, prediction semantics, or the set of qualified native executions.
A stronger regional bound may still be needed before this smaller gap can be
established cheaply.

`NativeClassSeparation.lean` proves the following conditional argument in a
common exact-integer scale:

1. For the permitted finite score endpoints in [-10,10], a one-sided RN64
   subtraction allowance of 2^-48 leaves a true gap strictly above 3*2^-16
   whenever the computed gap is at least 2^-14.
2. The latter gap is exactly representable. RN32 monotonicity therefore bounds
   the shifted rival by -3*2^-16. Applying monotonicity directly to an arbitrary
   nonrepresentable true gap would not justify this step.
3. The existing symmetric exponential error allowance 2^-16 and division
   allowance 2^-20, with one positive common denominator, give strict probability
   separation. The decisive exact inequality is
   `65537 * 1048577 * 65536 < 65539 * 65535 * 1048575`.

The exponential envelopes, native error bounds, floating-point encoding,
normality and reviewed instruction dataflow remain explicit premises. No
exact exp(0)=1 premise or global monotonicity of an approximate exponential is
introduced. The source/library/kernel/device/configuration/process checks and
finite score gate remain in force. This is a conditional mathematical theorem
and reviewed implementation, not formal verification of CUDA machine code.

The 58-line maintained proof supports one shared C++ constant used by the host
qualification gate, its receipt, and all seven CUDA numerical comparisons.
Boundary tests cover the representable values immediately below and above the
threshold, the threshold itself, and rejection with qualification disabled.
The half-spacing fixture has moved to `2^-14 + 3*2^-24`; its former full-spacing
bound still fails while the half-spacing bound succeeds. Both focused CUDA
suites pass with the new guard.

The internal residual audit gives candidate margin floors above the new guard
for all 54 saved fixed-policy states, compared with 21 above the old guard.
This describes the audited stronger certificate, not the number of states the
existing CUDA proof routines will close. Runtime measurements must establish
that separately. A 16-times smaller threshold does not imply a 16-times speedup.

### Measured effect of the tighter guard

A matched old/new/new/old process comparison used the same eight frozen regions
of the qualified 448-tree source. Each process independently qualified native
execution and ran three measured conversions per region, method arm and
policy, after warmup. State inspection was disabled for these measurements.
All 512 conversions, including warmups and optional unary arms, completed with
zero native-class mismatches. The complete source domain is still unfinished.

The ordinary default methods produced:

| Proof effort | States, old to new | Sum of regional median times, old to new | Direct plus cover visits, old to new |
| --- | ---: | ---: | ---: |
| Fixed | 716 to 658 | about 0.76 to 0.71 seconds | 415,252 to 406,702 |
| Dynamic | 740 to 678 | about 0.64 to 0.58 seconds | 266,426 to 259,044 |

That is approximately 8% fewer constructed states with either policy, and
about 6% and 8% less measured time, respectively. Both temporal pairing
directions improved the aggregate timing. Some unchanged-work regions varied
substantially in time; one ABBA block does not establish statistical
significance or a speedup on every region. These times sum regional medians,
not setup and exhaustive-comparison wall time.

Native margin and public-prediction fallback rows each fell from 32 to zero in
the default workload summary. Output remained exactly 132 reachable graph
nodes and 2,624 canonical bytes across the eight regions. Owned GPU memory
was unchanged in every region, with a maximum about 12 MB; native XGBoost
allocations are outside that counter. Pair visits are included in direct
visits and are not counted twice. The mathematical improvement saves
construction and native fallback work without changing these finished outputs.

Evidence: `build/native-gap-abba-3g4ugr1z/comparison.json`. The full 28-module
Lean replay is recorded in
`build/proof-research/native-gap-full-replay/e61dbb56b3650498/result.json`.
The production converter and CLI were rebuilt with this guard. Seven integration
checks passed in addition to the two focused CUDA boundary suites.
These local experimental receipts are not required at runtime.

A separate post-measurement inspection found 25 retained fixed-policy
expansions whose finished result is a class leaf, down from 54; the dynamic
count fell from 66 to 35. These state counts can overlap and do not represent
disjoint coverage. The surviving states, concentrated in one saved region,
are the next target for stronger dependency-aware certificates. This inspection
is excluded from comparative timings. Its receipt is
`build/internal-state-inspection-851z6w2w/receipt.json`.

## Research and proof acceptance

The next mathematical target is dependencies across several trees. Investigate
separator-conditioned elimination and bounded-scope cost shifting, deriving
both coverage and the signed-margin bound. Low-width elimination can exploit
an interaction graph whose connected feature groups are small; its complexity
is exponential in elimination width rather than necessarily in the total
number of input axes. Large-width regions still need a conservative fallback.
See [Dechter's elimination framework](https://ics.uci.edu/~dechter/publications/r76A.pdf).

A CPU structural audit of the eight saved regions gives a concrete reason to
pursue this direction: after restricting the original trees, 320--414 of the
448 trees are constant, and only 3--20 retain dependence on several axes.
The conservative feature-interaction graphs have min-fill induced-width upper
bounds of 1 or 2, with connected components of at most four features. This is
an observed elimination order, not a proof of minimum treewidth. The audit
covers all classes together and can overestimate the graph for one rival.
It builds no factor tables and establishes neither stronger certificates on
these regions nor faster CUDA execution. It identifies a small structural
target for the next algorithm instead of presuming low width.

### Exact regional bounds and the first cost comparison

An exact dyadic, CPU-only research calculation now covers all 336 ordered
class-pair comparisons across those eight roots. Five roots have two distinct
source-score classes, with witness ranges and gaps satisfying the existing
qualified-source conditions. The other three have positive worst-case margins
after conservative original-order RN32 error allowances: approximately 0.11,
0.44 and 0.55. All three already close to one node under the saved fixed-effort
baseline. Thus no additional fixed-baseline root closure is established.
This calculation does not reattest the native library or benchmark CUDA.

The small message tables do not guarantee cheap construction. For the single
rival left by complete independent ranges in each constant root:

| Saved root | Baseline source visits, including cover | Scoped factor prefill visits | Estimated streamed factor visits | Message term evaluations |
| --- | ---: | ---: | ---: | ---: |
| 5999 | 2,448 | 10,564 | 31,924 | 3,894 |
| 5018 | 1,820 | 8,601 | 143,679 | 14,987 |
| 6010 | 1,245 | 6,033 | 78,035 | 8,276 |

Baseline refinement counts already include pair visits; only cover visits are
added. The proposed costs omit some setup, use different primitive operations,
and are not GPU timings. They give no support for replacing these successful
baseline calls with naive streamed elimination. Dynamic scheduling previously
split the latter two roots into more states, which identifies an effort-policy
question rather than a missing mathematical certificate.

The finite-message proof is still useful for regions beyond the existing proof
family and for cheaper construction methods. An implementation should first
establish one of those advantages. High-arity source trees can also contribute
conservative conditional tables on selected axes, using existing region bounds
for omitted coordinates; exact low-arity dependence is sufficient but not
necessary. That broader approach needs the same complete cell coverage and
original-source error accounting, plus its own cost justification.

A separate 70-line scratch prototype, ProjectedMessagePrototype, now proves
that conservative per-factor projected tables compose with the maintained
GroupedMargins original-source margin theorem. It imports only the maintained
interface. For three rival factors
`[u != v] + (a + b)/32` on edges `xy`, `yz`, and `zx`, each factor uses four
axes. Projecting onto each edge's two endpoints and bounding the omitted axes
proves a winner margin of 5/16 against a constant score of 2.5. This example
still defeats the adjacent-pair bound after any single-axis split. Lean 4.34.1
compilation, independent kernel replay, and review passed. The implementation
must still supply the actual source-factor inventory, exhaustive projected
cells, and conservative numerical table entries; the theorem does not prove
those runtime obligations or universal dominance over pair bounds.

### Internal-state evidence

An opt-in CUDA harness diagnostic now inspects retained completed states after
conversion and exhaustive native comparison. It records actual expansions
whose resulting graph is a class leaf, together with the saved prefix,
residual roots, projected domain, and first-admitted path witness. Collection
is outside conversion timing and serialized once; these diagnostic runs are
not comparative timing evidence because readback may perturb later trials.

The eight restricted conversions of the 448-tree source retained 54 such
states under fixed proof effort and 66 under dynamic effort, with no state
evictions. These are state counts, not disjoint input regions or additional
certified coverage. Leaf collapse alone does not imply the numerical native
margin condition can hold. All restricted native comparisons passed.

A CPU exact-dyadic audit of the 54 fixed-policy states reconstructed global
source-root offsets and ordered channel prefixes. It checked 127,344 forced
ancestor steps, exact FP32 prefix words, and absence of every widened
coordinate from the remaining subtrees' syntactic support. Those checks
justify transferring witness-domain extrema to the actual saved projected
residual function. No new tree converter or full-grid enumeration was used.

Twenty-one states have sound candidate margin floors between about 0.006 and
0.12 after residual RN32 error allowances, meeting the former 2^-10 native guard's
numerical conditions. All 54 fail even complete independent channel bounds.
The other 33 contain original-order FP32 witnesses with top-two gaps of about
0.00085, 0.00033, or 0.00025, below the former 2^-10 guard. More accurate bounds
alone cannot make those witness gaps exceed that guard. These are research
certificate opportunities under the unchanged qualified-source interpretation,
not certificates already applied by the runtime.

The harness then replayed the existing direct and cover routines on each saved
state at one, four, and sixteen times its original fixed effort budget. All 54
still failed; work was unchanged at 9,363 direct source visits and 27,153 cover
visits per multiplier. Every direct per-tree walk and expected adjacent pair
completed, with zero fallbacks. Cover exposes ordinary fallback counts, which
were also zero, but not nested pair-completion counters. This rules out direct
traversal-budget exhaustion in these cases; it does not establish that arbitrary
increases in effort can never help another input.

A 99-line scratch proof now supplies shared source traversal for projected
cells: a source predicate routes active cells, and an unfinished branch emits
its whole subtree bound. Alternatives for the same factor are minimized, not
summed. Lean compilation, independent kernel replay and review passed. A CPU
structural audit on eight real hard-rival factor sets reproduced all 4,276
lower and upper table cells exactly. Source visits fell from 42,404 with
per-cell root replay to 3,017 with shared traversal, accompanied by 1,104 rank
comparisons and 4,276 table reductions. These operations have different costs;
the counts are not CUDA timings. Scope construction, cached constant factors,
and later elimination work are separate. This provides a concrete lower-cost
construction candidate while preserving the required empirical comparison
against the actual default baseline.

### Candidate selection and deeper covers on the remaining states

After the tighter gap was integrated, the 25 remaining fixed-policy constant
expansions matched previously audited residual payloads. All have one unresolved
rival. An exact CPU capability search found covers with at most two split
levels for every state. Its discovery cost was 466 bound queries, compared with
97 calls in the selected covers; those selected calls alone are not a fair
measure of how an implementation finds a cover.

A practical selector then removed both oracle inputs. One feasible-point
original-order RN32 residual evaluation proposes a class, without certifying
it. At each failed bound, choose the winner-or-rival factor with the widest
conditioned leaf range and split at its first unforced source predicate.
Per-tree ranges and first-unforced metadata can be gathered during the existing
conditioned pass; the proposed method needs no factor tables or scope analysis.
All 25 cases closed within two levels in the CPU audit. Candidate evaluation
used 2,927 visits; the 107 bound calls used 15,848 conditioned visits and 4,758
pair visits, for 23,533 logical visits in total. Root walk/pair counts matched
the existing GPU replay exactly. Metadata storage/selection and region copies
are not timed by these counts.

This also identifies two weaknesses of the current selection policy. Its fixed
predicates closed none of these 25 cases, while choosing the first unforced
rival predicate closed ten. The eventual winner differed from the maximum
complete lower-bound candidate in eight cases. Both class and split choices
therefore matter. The feasible point supplies only a proposal; every rival and
every feasible proof branch must still satisfy the qualified regional bound.

`RegionEnvelope.rival_specific_covers` already supplies the mathematical
composition rule. A CUDA implementation must enforce exhaustive branches,
immutable source prefixes, separate scratch, charged proposal/selection work,
and conservative rejection on visit or stack exhaustion. Two levels describe
this audit, not a required general hard limit. The CPU result establishes
additional capability with a concrete selector; GPU timings remain necessary.
Evidence: `build/proof-research/deeper-cover-selectors-candidate-20261009/`.

The CUDA candidate stays inside the existing cover routine. It preserves the
initial common-cover attempt, then proposes a class and constructs a separate
bounded depth-first proof cover for each unresolved rival. Predicate/side frames
occupy the tail of the existing private stack; each bound walk gets only its
disjoint remaining prefix. A cell is reconstructed from the immutable original
region and saved path. The right branch remains pending until it is certified;
partial work never authorizes a terminal output. The original source, prefix,
residual roots and graph are unchanged during this proof search.

Two optional metadata hooks expose the first unforced predicate and widest
conditioned winner/rival factor. They add no separate source traversal or
per-tree allocation. Zero-width factors are not proposed. Existing whole-class
certificates remain sufficient, and a proved competing class rejects the
candidate immediately. Work is allocated among actually unresolved rivals;
classes already defeated by the static bound no longer dilute another rival's
allowance. Proposal and proof source visits share the existing visit budget.
Region reconstruction is bounded by stack depth and proof visits, and its
runtime contributes to the existing dynamic effort controller.

### Measured deeper-cover capability and cost

The expanded CUDA suite passed 22,644 assertions, including mixed-label
rejection, depth beyond two, numeric missing routes, a 65-member exactly-one
group, and visit/stack exhaustion. Compute Sanitizer reported zero errors.
Sharing one non-inlined device body among correctness fixtures avoided repeated
compiler specialization; the test target then built in about one minute. This
is a test-compilation improvement, not a production-conversion speedup.

A matched old/new/new/old comparison against the tighter-gap one-split baseline
completed all eight restricted real regions with zero native mismatches:

| Default policy | Constructed states, one-split to deeper | Sum of regional median times | Direct plus cover visits |
| --- | ---: | ---: | ---: |
| Fixed | 658 to 608 | about 0.65 to 1.02 seconds | 406,702 to 714,241 |
| Dynamic | 678 to 658 | about 0.53 to 0.54 seconds | 259,044 to 301,625 |

The first implementation therefore supplies stronger certificates, but does
not establish a speed improvement. Both temporal pairings show about 56% more
fixed-policy time; dynamic time is approximately unchanged. Final output
remains 132 reachable nodes and 2,624 canonical bytes, with unchanged peak
owned memory of about 12 MB. These are completed restricted-region conversions
using all 448 source trees, not completion of the entire source domain.
Evidence: `build/deeper-cover-abba-ohxp73m1/comparison.json`.

The separate inspection removed all 25 retained fixed-policy expansions whose
finished output is a class leaf. That does not imply every future constant
region can close immediately. Failed deeper attempts on mixed regions still
cost work. A CPU proposal to reject mixed regions by evaluating the lower and
upper representative inputs found no differing classes in any of the eight
root regions, despite five being mixed. It used about 43,000 source visits and
was not adopted. Its evidence is
`build/proof-research/two-endpoint-screen-20261009/result.json`.

### Reuse unrelated channel bounds during a rival comparison

The next cost refinement retains the static score intervals computed at the
start of an effort call for every class outside the requested winner/rival
pair. When all those retained intervals are finite, ordered, and inside the
native range [-10,10], their conditional subtree walks and pair searches are
unnecessary. Only the two target channels are reset to saved prefixes and
folded again, in their unchanged source order. If a retained interval fails
that premise, or optional whole-class methods need their metadata, the full
existing path is used. The final all-class numeric gate is unchanged.

If the conditioned target bounds already establish the requested gap, the
routine also omits adjacent-pair search. This is a pair fact, not a whole-class
label: the caller still has to certify every rival over every feasible region.
No non-target score bound is reused across a different source or saved state.

This relies on maintained results rather than another Lean module:
`RegionEnvelope.enlarge_candidates_preserves_soundness` justifies inherited
enclosures, `OrderedArithmetic.ordered_reduction_encloses` and
`PairedBounds.paired_reduction_encloses` justify the two ordered channel folds,
and `RegionEnvelope.rival_specific_covers` composes all pair certificates before
`NativeClassSeparation.conditional_probability_order` applies. The actual
storage, finite-budget bookkeeping, and code-to-theorem correspondence remain
implementation obligations. Omitting unrelated walks is a concrete work
reduction; total runtime and changed search behavior require measurement.
The refinement is implemented and undergoing focused tests and timing.

A CPU replay of the same 25 selected covers retained all successes and all
107 proof cells. Restricting the walks to the two requested channels reduced
conditioned visits from 15,848 to 7,762 and pair visits from 4,758 to 2,570.
Stopping already-proved comparisons reduced pair visits further to 2,389.
Including unchanged class-proposal work, that is 23,533 to 13,078 logical
visits, about 44% fewer. This is selected-cover work, not whole-converter GPU
latency. Evidence:
`build/proof-research/target-channel-covers-early-20261009/result.json`.

### Reuse a proof that a region is genuinely mixed

The common-cover routine reports `differing_labels` only after every feasible
side has a certified class and at least two of those classes differ. Feasible
sides are nonempty, so this is evidence of actual class mixing rather than a
failed bound. No exhaustive proof can subsequently establish one common class
on the same region. The portfolio now preserves that result immediately,
omitting class-proposal evaluation and all rival-cover attempts. An
`uncertified_case` still permits the stronger search: insufficient proof power
and genuine mixing are deliberately different outcomes. This shortcut reuses
work already performed and introduces no new numerical assumption.

`RegionEnvelope.different_certified_parts_exclude_uniform` formalizes this
negative result in the existing module. It requires two nonempty contained
subsets, each with its certified label, and distinct labels; a complete cover
or disjointness is unnecessary. The small addition compiled and passed
independent Lean kernel replay with no axioms. Evidence:
`build/proof-research/mixed-certificate-kernel-20261009T113513Z/result.json`.
No new maintained module was introduced.

### Stop failed conjunctions and retain static certificates

A common cover requires every evaluated side to certify a class. Once one
feasible side is uncertified, visiting another side cannot make that particular
cover succeed. The routine now returns ordinary `uncertified_case` immediately,
with visit and case counters describing work actually performed. The remaining
allowance stays available to rival-specific proofs on the original region.
This procedural shortcut does not assert that the input region is mixed.

The portfolio also avoids recomputing the same static rival certificates while
nested calls reuse scratch. For at most 64 classes, one local machine word
records the initial successful comparisons. Static source folds outside the
nested proof routines decrease from C to one for C classes; no new allocation
is needed. Larger class counts retain the general recomputation path and have
no imposed class limit. A set bit is valid only during this call with unchanged
source, saved prefix, residual roots, extrema, winner and qualification. Actual
nested proof/proposal visits remain charged; static preflight folds are not
part of that visit counter. The matched measurements below include these two
cost refinements with the targeted-channel revision.

### Measured cost refinements and larger regions

The targeted-channel and repeated-work refinements passed 33,333 focused CUDA
checks, including cached bit 63 and the generic 65/66-class paths. Compute
Sanitizer reported zero errors. A direct matched comparison against the
2^-14 one-split incumbent gave:

| Default policy, eight original regions | States, incumbent to refined deeper | Sum of regional median times |
| --- | ---: | ---: |
| Dynamic | 678 to 646 | about 0.53 to 0.50 seconds |
| Fixed | 658 to 608 | about 0.66 to 0.69 seconds |

Both temporal pairings improved dynamic time, with an aggregate reduction of
about 6%; fixed effort remained about 4% slower. Output was unchanged at
132 reachable nodes / 2,624 canonical bytes and about 12 MB maximum owned
memory. Compiled draft-kernel resources were 212 registers, an 80-byte stack
and no local allocation, compared with 226 / 80 / zero for the incumbent.
Those resource counts do not independently establish occupancy or causation.
Evidence: `build/cover-cost-vs-incumbent-abba-g0nnamnn/comparison.json`.

The same preserved binaries were then compared on two additional frozen
regions, samples 4883 and 7719, containing 1,016,064 and 3,939,840 signatures.
Every measured conversion completed, and exhaustive native comparisons found
zero mismatches. Dynamic summed regional medians improved from about 1.40 to
1.20 seconds (about 15%); fixed effort worsened from about 1.61 to 2.17 seconds
(about 35%). Dynamic construction rose slightly, 4,526 to 4,546 states, as its
budget policy changed; fixed construction fell from 4,480 to 4,318 states.
This again shows that state count and conversion time are different metrics.

Both variants produced 373 reachable nodes / 6,096 canonical bytes and the
same maximum owned memory, about 26 MB. Sample 7719 closed to one node in both
variants; the material changes arose in 4883. The eight-region and two-region
comparisons are separate matched experiments, not pooled significance claims.
All use the complete source ensemble on restricted regions; full-domain
conversion is unfinished. Evidence:
`build/deeper-cover-abba-qd7zm7d5/comparison.json`.

### Refute an impossible positive-gap proof

`RegionEnvelope.opposing_bounds_refute_positive_gap` proves the next negative
certificate. On a nonempty region, suppose winner scores are at most U and
rival scores are at least L, with U <= L. No positive winner-over-rival gap can
hold throughout that region. The theorem uses exact integer-scaled order and
explicit enclosure/nonempty/positive-gap premises. It compiled and passed
independent kernel replay in the existing module; no new module was added.
Evidence: `build/proof-research/target-gap-refutation-kernel-20261009T114843Z/result.json`.

CUDA checks finite ordered target endpoints and `winner_upper <= rival_lower`
after handling existing whole-class certificates. It then rejects this gap
search before further subdivision. This does not assert that native classes
are mixed: tied scores can select one native class while admitting no positive
gap. Diagnostic codes distinguish common-cover `differing_labels` (genuine
mixed classes), `candidate_refuted` (a certified competing class), and
`target_gap_refuted` (this sufficient gap proof is impossible). The new cutoff
passed 33,339 focused CUDA checks and Compute Sanitizer with zero errors.
The final revision was compared directly against the same one-split incumbent:

| Default policy | Eight original regions | Two larger regions |
| --- | ---: | ---: |
| Dynamic | 0.54 to 0.50 seconds (about 7% less) | 1.43 to 1.20 seconds (about 16% less) |
| Fixed | 0.66 to 0.69 seconds (about 5% more) | 1.61 to 2.15 seconds (about 34% more) |

Both temporal pairings improved dynamic time in both cohorts. All conversions
completed and exhaustive native comparisons found zero mismatches. Output and
maximum owned memory stayed unchanged. Dynamic construction was 678 to 646
states on the eight regions, and 4,522 to 4,560 on the larger two. These are
summed regional median timings, not end-to-end process times or a claim that
this final cutoff alone caused the improvement. The cutoff, deeper covers and
all cost refinements form the measured candidate. Full-domain conversion is
still unfinished. Evidence:
`build/negative-gap-abba-p3e5_s09/comparison.json` and
`build/negative-gap-abba-e8rkf3x4/comparison.json`.

The deeper search retains exhaustive coverage, conservative bounds for
deferred work, and bounded effort without multiplying rivals' partitions.
A relevant search foundation is
[Veritas](https://proceedings.mlr.press/v139/devos21a.html).

The first strict example uses three depth-two trees on binary semantic axes:

```text
winner = 2.5
rival  = [x != y] + [y != z] + [z != x].
```

Any two rival terms can total 2 and the remaining term can reach 1, so pair
bounds permit 3. Fixing any one axis still permits that same relaxed bound in
both branches. But the actual rival never exceeds 2: three binary values
cannot be pairwise different. Eliminating `x` from the negative rival terms
gives the message `min_x(-[x != y] - [x != z]) = -2 + [y != z]`; adding the
remaining `-[y != z]` proves a margin of at least 0.5. This is a small algebraic
example distinguishing multi-feature elimination from adjacent pairs and one
binary proof split, not a GPU speedup or a native-qualified model experiment.

A scratch Lean prototype proves exact binary elimination as an equivalence of
all lower-bound claims, derives this message, and connects the resulting floor
to `GroupedMargins.original_sum_lower_margin`. It also checks every pair
partition on the full cube and all six one-axis slices. Compilation and
independent kernel replay passed, including the five maintained dependencies.
The original ordered-rounding trace remains an explicit premise. This is a
constructive extension of the existing grouped-floor interface: that interface
already accepts arbitrary groups, but does not itself construct these retained
conditional relationships. The prototype now generalizes to nonempty finite,
context-dependent atom lists. Its bounded scan accepts Option-valued atom
evaluators and returns a minimum only after every atom succeeds; evaluator
failure or insufficient atom fuel returns no new bound. Coverage supplies
soundness, while admissibility also supplies exactness. Directed sums and
existing incumbent retention compose with that result. The 261-line
source (about 145 lines of reusable core) and five dependencies passed Lean
4.34.1 compilation and independent kernel replay. Atom fuel is distinct from
runtime primitive-work accounting; it is not a proof of GPU visit counters.
The prototype remains outside the maintained collection until its algorithmic
role is established.

The classical [mini-bucket framework](https://ics.uci.edu/~dechter/publications/r62.pdf)
also offers a bounded-cost fallback when elimination would create an oversized
message. Splitting a bucket replaces a minimum of a sum with a sum of minima,
which is a conservative lower bound. This is a candidate extension of the
existing grouped-floor argument. Increasing permitted message size can preserve
more correlations, but changing the partition or floating-point evaluation
requires explicit incumbent retention; a larger resource allowance alone does
not prove better wall-clock performance.

Recent literature provides additional, conditional directions:

- [Level-wise Optimization and Pruning (ICML 2025)](https://proceedings.mlr.press/v267/devos25a.html)
  compresses ensembles by changing their structure and leaf predictions. Its
  reported predictive-accuracy preservation is not our exact-class guarantee.
  One possible use is a small candidate model accompanied by a separately
  proved regional bound on its difference from the original source. Candidate
  margins must exceed that bound, including original rounding allowances,
  before it can replace any work. This is a research proposal, not an adopted
  exact rewrite or demonstrated speedup.

- [Verifiable Boosted Tree Ensembles (2025)](https://jermp.github.io/assets/pdf/papers/SP2025.pdf)
  derives tractability for a restricted large-spread model class. We can look
  for analogous independence inside a region; arbitrary saved sources do not
  inherit that model restriction.
- [Sensitivity Verification (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/92f79f493ca2d6c0ba04c3af76bb3368-Paper-Conference.pdf)
  suggests pseudo-Boolean certificates with conservative rounding. Applying
  that approach here still requires the native multiclass and ordered-FP32
  correspondence.
- The authors' [OC-space (2026) artifact](https://github.com/ML-KULeuven/OC-space)
  explores reusable output configurations. Its explicit warning about very
  large enumerations reinforces the need to measure representation cost. The
  full paper was unavailable during this review, so no new theorem is adopted
  from it.

Each proposed extension needs (1) a precise rule and its source/domain and
floating-point premises; (2) a Lean derivation using existing infrastructure;
(3) a case where it strictly improves the actual default proof family, or a
cost theorem preserving its result; and (4) measured total runtime, memory and
construction work once GPU experiments are available. Retaining an incumbent
bound guarantees no loss of bound strength, not no loss of runtime.

Metamath, mathlib and current theorem corpora supply candidate identities and
proof techniques. Their assumptions must be translated and checked locally;
the number or recency of cited theorems is not evidence of acceleration. For
example, the selected formalization in OpenAI's
[formula hitting family](https://github.com/openai/math/blob/main/lean/docs/116.md)
does not assert every complexity bound in the associated paper. Algebraic
identity detection must also be connected to our input semantics and original
rounded score computations before it can authorize a converter rewrite.

## Ordered joint bounds

Independent bounds allow extrema from different trees to occur together even
when their split paths contradict one another. The new bounded traversal keeps
only compatible paths of two consecutive residual trees in each score channel.
It checks numeric intervals, missing-value routes, and exactly-one categorical
groups, including groups wider than one machine word.

For incoming accumulator bounds L and U and each feasible leaf pair (a,b), it
bounds the ordered expressions `RN(RN(L+a)+b)` and `RN(RN(U+a)+b)`. It never
replaces the original additions with `RN(L+RN(a+b))`. Other score channels can
be interleaved in source order because their accumulators are independent.

A completed pair traversal supplies an enclosure. Any incomplete traversal uses
the original independent enclosure, never extrema over only visited cases.
The joint result is intersected with the incumbent per-tree enclosure. Its
visits share the effort allocation described below, so the saved incumbent is
the ordinary pass at its allocated budget. Insufficient scratch, invalid
extrema, overflow, or exhausted work cannot produce a certificate from a
partial traversal.

`formal/converter_guarantees/PairedBounds.lean` supplies the abstract ordered
pair-bound argument. The CUDA/domain correspondence is tested, not a claimed
machine-checked refinement of the implementation.

## Correlated margins between classes

Shared leaf contributions cancel in the exact-real difference between classes,
while rounding still needs its own bound. Independent score intervals lose that
correlation. The relational method bounds the difference
between a candidate winner and each rival directly. It pairs the kth active
residual tree in each channel, preserving each channel's source order, and
reuses the joint path traversal to bound compatible leaf differences on the
same input. Numeric, missing-value, and categorical restrictions all apply.

Let `pw` and `pr` be the unchanged, already-rounded channel prefixes, `mk` a
lower bound on each compatible leaf difference, and `Ew` and `Er` upper bounds
on accumulated rounding error in the original channel folds. Then

```text
winner score - rival score >= (pw - pr) + sum(mk) - Ew - Er.
```

Each original RN32 addition retains its place in its channel. An unmatched
residual term contributes its own lower bound, or the negative of the rival's
upper bound; the absent side introduces no synthetic floating-point addition.
The implementation encloses each rounded accumulator and charges half the
outward FP32 spacing at the largest endpoint magnitude. Halving happens in
FP64, retaining the smallest allowance of 2^-150. This relies on nearest-even
addition with gradual underflow. Nonfinite outputs or a nonfinite outward
neighbor reject that error envelope. Error totals use upward FP64 rounding;
differences, sums, and final error subtraction use downward rounding.

The implementation subtracts the total error at the end. This only regroups
the exact-real bookkeeping inequality; it does not reassociate source RN32
operations. `RelationalMargins.lean` proves `deferred_error_identity` and
`deferred_margin_encloses` in the `ConverterRelationalMargins` namespace,
justifying this deferred subtraction in the abstract integer-scaled model.
`ordered_margin_encloses` supplies the underlying ordered-fold argument.

A completed pair walk supplies a compatible difference floor. On budget or
scratch exhaustion, the entire incomplete pair reverts to its static
`minimum(winner tree) - maximum(rival tree)` floor. A partially visited minimum
is never accepted. The method preserves the incumbent score ranges, checks all
channels against the existing finite `[-10,10]` range gate, and requires the
shared `native_softprob_gap::computed_gap_minimum` (now `2^-14`) against every
rival under the same native qualification.
Candidate choice and pairing affect which certificates are found, not the gate.

The elementary arithmetic was informed by Metamath's
[order under subtraction](https://us.metamath.org/mpeuni/le2subd.html),
[comparison of finite sums](https://us.metamath.org/mpeuni/fsumle.html), and
[finite sums of differences](https://us.metamath.org/mpeuni/fsumsub.html).
These are standard order and sum identities, not a novelty claim or a Metamath
library import. The project supplies its own Lean arguments with explicit
coverage, nearest-rounding, spacing, and directed-arithmetic premises.

## Shared bounded proof effort

One effort call owns one visit budget. When unary grouping is enabled and its
scratch is available, it initially reserves half, rounded down. Relational
bounds reserve one third of the remainder when enabled. The remaining
allowance funds ordinary per-tree refinement and same-channel joint bounds;
joint bounds initially receive at most half of that remainder. Unspent
ordinary work can move to the joint pass. If these passes fail, unary grouping
uses its reservation plus unused ordinary/joint work. Relational bounds then
receive all remaining visits if still needed. Successful earlier certificates
return immediately. A zero budget performs no added traversal.

The reported allocations for an attempted call sum to its total budget, and
charged visits cannot exceed that budget. The production ceiling scales the
original source-traversal visit count by the number of enabled passes, up to
four with joint, relational and unary bounds enabled. This is an upper limit
per region: the dynamic tuner or an explicit refinement budget selects the
actual allowance. Exhausted work retains conservative bounds and may leave a
region for later construction. It does not imply completed conversion.

## Separate exhaustive covers for competing classes

To certify winner w, it is sufficient to prove that w beats each competing
class throughout the region. Each comparison may use a different exhaustive
split. There is no need to materialize the Cartesian refinement of all splits.

The implementation first tries the existing common cover with its full
budget, then uses only its unspent budget for independent rival covers. It chooses a candidate from
static lower bounds and a predicate from a rival's residual tree. These choices
only influence which proofs it finds. Acceptance still requires all feasible
sides, every rival, the original floating-point score order, and the unchanged
native margin/range gate. Proof-only splits are not emitted as decision nodes.

`RegionEnvelope.rival_specific_covers` states the core theorem.
The theorem uses independently indexed covers and an explicit native prediction
gate. `CoverRefinement.lean` supplies the retained-bound and complete-cover
lemmas; no additional wrapper module is needed. A partial cover is insufficient.

## Algebraic identity experiment

OpenAI's September 2026
[explicit rational matrix hitting-point construction](https://github.com/openai/math/blob/main/preprints/One-Rational-Matrix-Hitting-Point-for-Noncommutative-Formulas-September-24-2026/build/construction.tex)
includes acyclic path programs. This suggests an identity signature for an
already compact decision DAG. A Boolean guard is represented by a formal
variable, and a decision has arithmetic form `low + guard * (high - low)`.
Terminal codes are exact class IDs or floating-point bit patterns. This does
not justify rewriting the unfinished floating-point sum of an ensemble.

The prototype evaluates truncated integration jets modulo multiple distinct
primes. An integer numerator bound determines how many primes are sufficient;
a single matching modular hash never authorizes equality. Matrix application
uses a linear recurrence instead of a dense matrix product. The difference of
two continuations is represented in the same shared program, without unfolding
it into paths.

The standalone helper additionally checks all abstract Boolean guard assignments
before returning `independently_verified_boolean_equivalence`. This is a bounded
independent acceptance check. `exact_jets_equal` alone is a research result and
does not grant converter/native authority. Treating guards independently is
conservative for correlated feature thresholds, and can miss equivalences.

The OpenAI hitting theorem, the concrete jet denominator/magnitude bound, and
CUDA arithmetic correspondence have not been ported into Lean. The prototype
therefore uses independent exhaustive Boolean verification for acceptance.
Experimental conditional signature lemmas are outside the maintained proof
collection; existing shared-DAG semantics remain in `SharedDecisionDAG.lean`.

Initial compact-fragment measurements found an identity beyond structural
sharing, but modular jets were slower than directly checking the small Boolean
domain. Consequently this experiment is not in the default conversion path.
It is a research starting point, not demonstrated acceleration of large models.

## Use and compatibility

Joint bounds and rival covers are requested by default and run when the existing
native-class gate and proof effort are available. Relational margins are an
explicit experiment: enable `--relational-bounds` or JSON
`relational_bounds: true`. The real-region comparison below did not justify
enabling them by default. `--no-relational-bounds` remains an explicit disable
switch; supplying both switches is an error. Use `--no-joint-bounds`,
`--no-rival-covers`, and `--no-relational-bounds` to compare against the earlier
independent-bound policy (or set all three corresponding JSON options false).
The existing gate retains its qualified source/runtime scope; these algorithms
do not silently widen it to every XGBoost model or hardware configuration.

Checkpoint control/state layouts and identity rules are unchanged by these
strategies. New diagnostic counters are separate and reset when a process
resumes; completed search work and pending regions remain in the checkpoint.
The existing root-path ETA sampler does not model joint or relational search. Its
policy compatibility check declines an ETA for that mismatched regime rather
than reporting a forecast for a different algorithm.

GPU check targets include `adaptive_joint_bounds_checks`, `adaptive_rival_cover_checks`,
`adaptive_relational_bounds_checks`, and `adaptive_identity_signature_checks`.
The relational checks cover compatible differences, original-order rounding,
subnormals, missing routes, categorical constraints, incomplete covers, range
gates, and shared visit allocation. These synthetic fixtures do not qualify an
external native runtime.

On 9 October 2026, all 26 maintained Lean modules compiled with Lean 4.34.1 and
passed independent kernel replay, including the relational and deferred-error
arguments. This checks the abstract theorems under their stated premises; it
does not prove that CUDA implements those models. The focused CUDA test passed
44,322 assertions, including budgets 0 through 200; Compute Sanitizer memcheck
reported zero errors. The joint, rival-cover, complete synthetic-frontier, and
option-parsing regression checks also passed.

## Real-source comparison (9 October 2026)

The new relational rule was compared with the current joint-and-rival baseline
using the unchanged qualified 448-tree Covertype source. Eight selected saved
region proposals were rebuilt from the current source thresholds. Stored labels
and historical authorization fields were ignored; the native gate was qualified
in the current process. Each region fixed the two categorical choices and varied
three to five numeric coordinates. The regions total 612,235 source-threshold signatures, counted separately
within each region; this does not assert disjoint regions. The set is a targeted
case study, not a representative random sample.

All 128 restricted-domain conversions completed, including warmups. Every
source signature was compared with native CUDA predictions outside the timed
conversion, with zero mismatches. This does not complete the full-domain
448-tree conversion. Three alternating measured repetitions followed a warmup.
Times below sum the eight per-region medians; states and graph nodes are also
summed across these eight separate regional graphs. Both policies use batch and
draft widths of 32. These small-frontier measurements isolate proof policy;
they do not measure peak throughput on a large parallel frontier.

| Effort policy | Relational rule | Created states | Reachable graph nodes | Total of median times |
| --- | --- | ---: | ---: | ---: |
| Matched fixed allowance | Off | 716 | 132 | 0.86 s |
| Matched fixed allowance | On | 716 | 132 | 1.0 s |
| Dynamic effort | Off | 740 | 132 | 0.67 s |
| Dynamic effort | On | 754 | 132 | 0.82 s |

The matched comparison took about 18% longer with the new rule, with no net
reduction in intermediate construction or final graph size. Dynamic effort took
about 23% longer and created slightly more states, because the allocation and
ceiling changed. State counts were stable across all three repetitions. Peak
converter-owned GPU allocation was unchanged at about 11 MB; native-library
allocations are excluded. These results are why the rule is opt-in.

Direct relational-prune counters were zero. Those counters exclude attempts
inside proof-only covers, so state counts and end-to-end timings are the outcome
comparison. The stronger mathematical witness remains valid; these real-model
measurements do not demonstrate a useful acceleration from this pairing policy.
The next step is a stronger or better-targeted mathematical bound, followed by
Lean proof and another comparison, not more optimization of an unproductive
pairing policy.

`adaptive_real_region_benchmark` reproduces this kind of comparison with either
training-record neighborhoods or explicit saved region proposals. It requires
explicit source, native-library, and same-process qualification paths. No training
labels are consumed. The pinned external capture helper retains its original
symbol and environment ABI; the current adapter checks that exact binary and
validates any newer environment alias against the same capture directory.

## Grouping all factors of one coordinate

The next mathematical step combines more than two residual trees. After a
region fixes most predicates, many deep trees depend on only one remaining
numeric coordinate or exactly-one group. For a proposed winner and rival,
collect their signed contributions by that coordinate and minimize each entire
group over its source-induced intervals. Unresolved factors retain independent
floors. This enumerates each coordinate separately, not their Cartesian product.

The bound is the exact-prefix difference plus the sum of group floors, less
conservative error envelopes for every original ordered FP32 addition. Grouping
changes only this proof calculation; it does not reorder model evaluation.
`GroupedMargins.lean` supplies the group identity, finite-atom floor composition,
original-order error composition, and unchanged native-gate theorem. Its strict
four-coordinate witness closes a region that disjoint cross-class pairing and
one-predicate independent covers cannot close.

The CUDA integration reuses the existing conservative region traversal to
identify unary factors. Numeric groups cover the finite lower endpoint, every
in-range source threshold, and the missing atom when allowed. Categorical groups
cover every permitted category across all mask words. Incomplete collection or
evaluation restores the entire group's independent bound. The worker's scratch
capacity and traversal allowance are resource bounds, never permission to omit
input cases. Persistent search state and checkpoint layouts are unchanged.

`--unary-bounds` (JSON `unary_bounds: true`) requests the rule. It remains off
by default because the initial completed comparison reduced construction but
increased total time. `adaptive_real_region_benchmark
--method unary` compares it against the same maintained converter with grouping
disabled. Method counters cover direct state proofs only; nested proof-only
cover work is included in aggregate cover visits. State counts and total runtime
are the primary outcome measures. The proof establishes a stronger bound; it does not by itself establish
a real-workload speedup, global minimum tree size, or CUDA refinement.

### Initial unary-group comparison (9 October 2026)

The initial implementation used the same eight restricted regions, current
native qualification, and alternating three-repeat protocol described above.
All 128 conversions completed with zero mismatches on the 612,235 source signatures counted across
the selected regions. These are complete conversions of the selected regions,
not of the full 448-tree domain.

| Effort policy | Unary grouping | Created states | Reachable graph nodes | Total of median times |
| --- | --- | ---: | ---: | ---: |
| Matched fixed allowance | Off | 716 | 132 | 0.75 s |
| Matched fixed allowance | On | 678 | 132 | 1.15 s |
| Dynamic effort | Off | 740 | 132 | 0.61 s |
| Dynamic effort | On | 756 | 132 | 0.91 s |

The fixed comparison reduced constructed states by about 5% but took about
50% longer. Direct grouping counters recorded 17 additional certificates;
these counters exclude nested proof-cover calls. Final graph size did not
change. Peak converter-owned allocation increased from about 11.6 to 12.0 MB,
excluding native-library allocations. The dynamic comparison was also slower
and created slightly more states. A mathematically stronger bound therefore
did not yet provide cheaper conversion.

The initial focused CUDA fixture passed 8,968 assertions. It includes strict
cancellation witnesses, incomplete traversal and scratch fallbacks, missing
values, signed zero, category masks wider than 64 bits, and original-order
rounding cases. All 27 maintained Lean modules, including `GroupedMargins`,
also compiled and passed independent kernel replay. The abstract numerical
premises and CUDA refinement distinction remain unchanged.

The compact-member revision compacts active source IDs once and sorts them by
coordinate, retaining original source order within each group. Complete group
walks then visit contiguous member lists instead of repeatedly scanning all
source trees. This changes metadata work from a possible quadratic scan per
class comparison to `O(T log T + K*T)` before atom/path evaluation, for `T`
source trees and `K` classes. It adds one `T`-entry index buffer per private
draft slot. Finite-budget outcomes can change with group evaluation order;
the following comparison measures those effects. Traversal-visit budgets do
not bound all metadata or threshold-deduplication instructions.

### Compact-member revision

A fresh comparison of the same eight regions completed all 128 conversions
with zero native-class mismatches. The fixed on/off arms created 678/716 states
and took 0.94/0.74 seconds; dynamic arms created 756/740 states and took
0.87/0.61 seconds. Final graphs still total 132 nodes. State and proof-work
counts match the initial grouping implementation on these regions.

Relative to the prior grouping run, the fixed-policy measurement is about 18%
faster and the dynamic measurement about 4% faster. These are separate
sequential runs, not an interleaved old/new comparison. Grouping remains
about 28% slower than its same-run fixed control and 44% slower dynamically,
so it remains opt-in. Its peak owned allocation is about 12.1 MB. The focused
CUDA suite now passes 25,793 checks, including reordered factors, consumed
roots, seven classes, and insufficient ordering storage. Compute Sanitizer
memcheck reports zero errors.

### Proved first-atom shortcut

`GroupedMargins.first_atom_no_improvement` proves a numerical shortcut. The
full group routine takes the minimum of its computed atom floors, then the
maximum with the independent incumbent. If any already-computed floor is at
or below that incumbent, the final result cannot improve it. This does not
claim that the exact mathematical minimum is attained or that a different
proof could not improve the bound.

CUDA probes the first feasible representative before collecting all cuts. A
conclusive probe retains the independent bound. Otherwise it reuses that
computed value in the full scan, adding no source visits to a completed scan.
Insufficient resources keep the full independent fallback. The completion
counter includes proved no-improvement shortcuts as well as exhaustive scans.

The new theorem and its dependencies compiled and passed independent Lean
kernel replay before implementation. The revised focused CUDA fixture passed
25,870 checks and Compute Sanitizer reported zero memory errors. The unchanged
eight-region comparison completed all 128 conversions with zero native-class
mismatches. Fixed on/off arms took 0.94/0.73 seconds and dynamic arms 0.86/0.62
seconds; construction and final graph counts were unchanged from the compact
revision. Unary source visits decreased by about 7% fixed and 13% dynamic,
but the separate sequential runtime differences were only about 1% and 2%.
Those timings do not establish a robust speedup. Memory usage was unchanged.
The rule therefore remains opt-in.

### Proved rejection of an unproductive rival comparison

GroupedMargins.computed_group_bounds and optimistic_rival_rejection bound
the numerical result of this particular routine before a full group scan.
For each group, the maximum of its independent floor and the computed first
atom is an upper bound on the floor the completed scan or conservative fallback
can return. Folding these caps in the same directed order, with the same
prefix and error subtractions, gives an optimistic numerical result. If that
result misses the required native margin, further atom scans cannot make this
routine certify the rival comparison.

The cap is not an upper bound on the true model margin and does not establish
that the region is unclassifiable by another proof. CUDA requires every probe
to succeed before rejecting an attempt. Per-rival caches are cleared, and
their computed first values are reused by any full scans. Insufficient cache
storage disables this shortcut. The new theorems and their dependencies passed
independent Lean kernel replay before implementation.

All 128 conversions again completed with zero native-class mismatches. The
fixed comparison rejected 120 attempts early and reduced direct grouping
visits by about 18%; the dynamic comparison rejected 242 attempts and reduced
those visits by about 39%. Constructed states remained 678/716 fixed and
756/740 dynamic, on/off; final graphs remained 132 nodes in every arm.
Fixed on/off timings were 0.91/0.71 seconds and dynamic timings 0.83/0.58
seconds. Both enabled and control times moved relative to the prior run, so
these measurements do not establish an attributable latency gain. Grouping
remained about 28% slower fixed and 43% slower dynamically, and remains opt-in.

The focused suite passed 25,907 checks and Compute Sanitizer memcheck reported
zero errors. Peak owned allocation was about 12.1 MB enabled versus 11.6 MB
disabled. Static compilation used 226 registers in the draft-preparation kernel,
up from 218 before the whole-rival shortcut, with an 80-byte stack and no
reported local allocation. This is resource evidence, not a measurement of
the dominant GPU bottleneck.

### Sorting only grouped factors

The ordering pass now sorts only active factors with a valid semantic axis,
then appends unclassified factors in original source order. This produces the
same full permutation as the previous total sort, including its none-axis tail.
Sorting work becomes O(T + U log U), where U is the number of active classified
factors among T source trees. Full active-ID capacity checks and the subsequent
folds remain unchanged; even a zero-length unary prefix retains the full tail.

The same comparison completed all 128 conversions with zero native-class
mismatches. Focused tests passed 25,921 checks, including capacity exhaustion
during the tail append; Compute Sanitizer reported zero errors.
Fixed enabled/control times were 0.87/0.72 seconds; dynamic times were
0.75/0.59 seconds. Relative to the previous enabled revision, measured times
fell about 5% and 9%, while the controls rose slightly. These are sequential
revision comparisons, with three alternating measured repeats within each run.
All construction, graph-size, proof-work, memory and register counts were
unchanged. The method remains about 21% slower fixed and 28% slower dynamically
than its same-run control, so it remains opt-in.

### Half-spacing rounding allowance

The existing nearest_spacing_error theorem now admits neighboring gaps up to
twice the supplied error allowance. This proves the half-spacing bound under
the same explicit nearest-selection and finite-neighbor premises. The theorem
was strengthened in place, with no new maintained module. Five relevant
modules compiled and passed independent kernel replay before CUDA integration.

The CUDA error helper halves the outward spacing in FP64 and accumulates
upward. Original FP32 operations and the qualified native margin remain
unchanged. Focused checks include both signs at powers of two, the smaller
spacing below a power of two, signed zero, the least subnormal, cancellation,
and rejection of an infinite outward neighbor or overflowing endpoint.

The original strict fixture used identical 0/1 stumps in two channels, with winner prefix
2^-10 + 3*2^-24. Independent ranges overlap. The former full-spacing penalty
leaves a floor below the required gap; the half-spacing floor is
2^-10 + 2^-24, and the original ordered scores satisfy the gate at every atom.
This demonstrates additional proof power without changing source precision.

The relational suite passed 44,357 checks and the grouping suite 25,921;
both Compute Sanitizer runs reported zero errors. Separate real-region
comparisons for both methods each completed 128 conversions with zero
native-class mismatches across 612,235 regional signature evaluations.
Their state, graph and proof-work counters matched preceding receipts:
the selected real cases did not gain another certificate from this tighter
allowance. Grouping enabled/control times were 0.94/0.76 seconds fixed and
0.82/0.64 seconds dynamic; relational times were 0.92/0.76 and 0.77/0.63.
Control times also changed, so no speedup is attributed to this revision.
Peak owned allocation and static kernel resources were unchanged.

## Historical measurements (8 October 2026)

These measurements predate the relational method and its shared allocation.
They describe the earlier joint, rival-cover, and identity experiments only.

C++23/CUDA on the RTX A5000 Laptop GPU. These are constructed correctness and
cost fixtures, not trained-dataset results or a qualification of the native
prediction gate. Five measured repetitions followed a warmup; conversion time
includes initialization and scheduler allocations, but excludes source setup
and exhaustive result comparison. Small host timings are noisy.

| Fixture and matched effort profile | Strategies | Created states | Expansion counter | Median time |
| --- | --- | ---: | ---: | ---: |
| Three cancelling pairs, twice-original traversal allowance | Earlier / joint | 3 / 1 | 4 / 0 | 2.5 / 1.4 ms |
| Different rival covers, four-times allowance | Earlier / rival covers | 3 / 1 | 2 / 0 | 1.5 / 1.0 ms |
| Tied scores, twice-original allowance | Earlier / joint | 4 / 4 | 6 / 6 | 3.0 / 3.4 ms |

Each of these constant-output fixtures produces one final stored node in both
versions. The gain is avoiding intermediate construction, not shrinking that
already-minimal output. The enlarged matched profiles isolate the methods;
the four-times refinement allowance exceeds the native production refinement
ceiling. Its cover allowance is available to the new cover policy. Actual
initial dynamic defaults showed no state reduction before these tiny cases
finished, and are reported separately in the benchmark executable.

That historical full-frontier run completed 384 conversions across 64 configurations
and compared 86,400 source/graph signatures without a mismatch. Focused joint
checks passed 2,013 assertions. These results establish additional proof
opportunities and their costs; they do not establish an acceleration factor for
a large trained ensemble or eliminate worst-case exponential growth.

The identity prototype found one equivalence beyond structural matching in an
eight-node fixture. GPU kernel time was approximately 0.05 ms versus 0.007 ms for
direct Boolean checking there, and 6 ms versus 0.01 ms in a 14-node fixture.
Allocation/transfer time is excluded from these identity timings. The prototype
therefore remains outside the default runtime path.


## General native correspondence: next proof obligation

The cover mathematics is not specific to a dataset. Native-class pruning also
requires agreement between the imported arithmetic and the native predictor.
A source review of XGBoost 3.4.1 established that its scalar multiclass GPU
path starts from each channel's base score and adds leaf contributions in
stored tree order. This matches the converter's ordered per-channel prefix.

Two previously unchecked structural assumptions are now explicit in both
structural readers, through `class_native_source_contract.hpp`: source tree
weights must be absent, empty, or all exactly one; numeric successors must
satisfy right = left + 1 and advance beyond the current node, matching
native `GetNextNode` and its traversal invariant.
Non-unit weights need their own rounding model and are rejected rather than
silently ignored. These checks do not authorize a new pruning scope.

For `multi:softmax`, native `FindMaxIndex` scans the raw margins and retains
the earlier index on ties. Once imported/native margin equality is established,
finite enclosing intervals certify winner w if U[c] < L[w] for c < w and
U[c] <= L[w] for c > w. `ConverterArithmetic.interval_first_winner_selected`
proves this selector in the existing `OrderedArithmetic.lean` module. It passed
Lean 4.34.1 and independent kernel replay, with no new module. Receipt:
`build/proof-research/tie-aware-selector-kernel-20261009T120327Z/result.json`.
This needs no exponential/division approximation,
positive softprob gap, or softprob-specific score-range restriction.

The remaining correspondence obligations include exact loaded FP32 words,
missing routing, bias and tree order, rounding/subnormal behavior, and binding
the predictor/direct-class instructions actually invoked. Static source or
embedded SASS inspection is not by itself a live-launch certificate. Current
source/runtime authorization remains unchanged while this bridge is developed.
Research evidence and primary-source references:
`build/proof-research/native-correspondence-source-20261009/characterization.md`.


### Production integration and a general-model control

The final deeper-cover revision and shared importer preconditions were rebuilt
into the maintained converter and public CLI. Eight scoped integration tests
passed: both-reader parsing, joint/rival/region GPU checks, batch options, CLI
help/version and CLI transport. Build/test receipts are under
`build/negative-gap-production-4zcfhk0o/final-topology/`; previous executable
artifacts were retained for reproducible comparison.

A fresh public-CLI conversion of the real Wine model (13 features, 3 classes,
3 source trees, direct-class objective) completed with automatic scheduling in
about 0.06 seconds of conversion and 0.4 seconds including setup. It produced
15 nodes, 304 canonical bytes and 272 compact bytes, using about 0.5 MB of
converter-owned GPU memory (excluding native library/runtime allocations).
The graph has exactly the same source identity, predicate words, missing
routes and class terminals as the previous completed graph after node-index
renaming. This is a cross-model regression control, not a matched timing or
new native-qualification claim: source-specific gap pruning was disabled,
and native terminal classification remained active. Evidence:
`build/proof-research/wine-general-control-20261009/conversion/result.json` and
`build/proof-research/wine-general-control-20261009/structural-comparison.json`.

### Shift-invariant score-window research

A private, kernel-replayed prototype shows that the absolute score window can
be replaced by finite valid endpoints with
`b = RN32(min lower - max upper) >= -20` and finite b. Monotonic RN32 then
places every actual shifted score in [-20,0]. Separately, monotonic RN64 fixing
representable h = 3*2^-16 proves that a computed gap >=2^-14 implies true gap
>h, without the previous magnitude-dependent subtraction-error allowance.
The replacement probability lemma restricts its rival exponential premise to
[-20,-h]; it does not assume a global approximation guarantee.

This preserves the reviewed exponential and normal-division assumptions while
admitting some common score offsets outside [-10,10]. It does not cover
arbitrary finite scores whose shifted exponentials reach unreviewed exceptional
paths. The currently qualified 448-tree source already lies inside the old
absolute window, so no new closure or speed benefit is claimed for that source.
The prototype therefore stays outside the maintained collection pending a
useful broader native correspondence certificate. Evidence:
`build/proof-research/ShiftedScoreWindowPrototype.lean`,
`build/proof-research/ShiftedScoreWindowNotes.md`, and
`build/proof-research/shifted-score-window-kernel-20261009T121026Z/result.json`.


### Retaining completed states for the next mathematical experiment

The first diagnostic on larger sample 4883 retained only 542 expanded states:
its 1,024-slot state arena had evicted 3,293 states. No retained expanded state
collapsed to a class leaf, but that could not characterize discarded history.
The benchmark now accepts `--initial-states` (default 1,024 unchanged), records
that capacity and includes the newer negative-certificate rejection codes.
This is diagnostic configurability, not a new converter or production default.

Repeating with 8,192 initial slots retained all 4,317 constructed states and
inspected 2,158 expanded states with zero evictions or inspection truncation.
None eventually collapsed to a class leaf; all 1,016,064 source signatures
still matched native classes. These runs are explicitly excluded from timing
comparisons. Evidence: `build/retained-state-inspection-lofg_c5r/result.json`.

This removes the lost-history concern, but a nonleaf result alone is not a
proof that every retained witness region contains different native classes.
The next characterization should find two feasible differently labelled
native witnesses per expanded region, reusing the already checked root-grid
labels. It will distinguish actual mixed regions from missed constant-region
certificates before investing in a more expensive mathematical bound.

A bounded multi-factor elimination candidate remains private research. It can
retain conditional relationships across several features, but the inspected
larger-region class pairs forecast substantial table-evaluation work, and no
current missed-certificate witness demonstrates a benefit yet. The old small
misses are already resolved by the deeper-cover implementation. See
`build/proof-research/BoundedSeparatorCandidateNotes.md` and
`build/proof-research/bounded-factor-elimination-research-20261009.md`.


### General cache applicability without a fixed feature count

The maintained product-domain reuse argument in `ApplicabilityCache.lean` now
accepts any semantic coordinate type instead of `Fin 12`. This is a
parametric generalization of the existing proof, with unchanged theorem bodies
and no additional maintained module. It neither imposes a model feature limit
nor introduces a different converter.

A target input can use a donor certificate when its retained coordinates lie
inside the donor guard. The proof constructs a donor input using those retained
coordinates and the donor witness on omitted coordinates. Both the proposed
program and the residual source must ignore the omitted coordinates; equality
to the current source is a separate required premise. Product-domain structure
and a nonempty donor remain necessary. Treat each exactly-one category group
as one semantic coordinate; independently varying its indicator bits would
violate that domain model.

The changed module and three dependencies passed standalone compilation and
independent kernel replay in Lean 4.34.1. Independent source review confirmed
that the generalization needs neither axis enumeration nor an additional
choice assumption. Evidence:
`build/proof-research/generic-cache-kernel-20261009/result.json`.
This is a broader mathematical applicability statement, not a newly measured
cache hit rate or a formal CUDA implementation proof.


### Native witness diagnostic: built, device execution pending

The benchmark-only `--witness-lookups N` option reuses the native labels already
computed for the verified root signature grid. It classifies each retained
expanded state's first saved context as mixed (two different native labels),
constant over every admitted signature, or unknown. A per-state lookup allowance
limits additional work; incomplete, missing-value, misaligned or unmapped
contexts remain unknown. The default zero disables this diagnostic and leaves
production behavior unchanged.

Mixed records include both native class indices, reconstructed rows and the
saved region geometry, with membership checked. Constant records retain the
prefix, source positions, residual roots and both saved/projected regions for
subsequent mathematical analysis. Constancy of one saved context is not a
certificate for every incoming context of a shared state. Lookup counts and
sums of local signature counts are separate; overlapping contexts can contain
the same root signature.

The standalone `--witness-mapping-checks` fixture covers numeric rank boundaries,
canonical zero, categorical masks crossing a 64-bit word boundary, exact local
to root offsets, incomplete scans and invalid cases. Independent source review
passed, including fixes for partial mapping-product reporting and attempted
versus actual native-label reads. Source and review receipt:
`build/native-witness-source-review-tf6zqs5o/`.
A subsequent CPU-only build succeeded, using one low-priority build job on two
CPU cores. Help and two argument-validation checks passed with GPU devices
hidden and capture helpers disabled; these command paths return before device
initialization. Evidence: `build/native-witness-cpu-build-kvzkxptr/`.
Device mapping checks and the sample-4883 diagnostic have not run because GPU
work remains paused at the user's request. This supplies build evidence, not
new mixed/constant counts or conversion-speed measurements.

### Checked partitioned-message prototype

`build/proof-research/MiniBucketPrototype.lean` fills the previously missing
partition-inventory argument for the bounded elimination candidate. For a fixed
nonempty atom domain and exact integer-scaled factors, `partition_sandwich`
proves independent minima <= partitioned minima <= the joint minimum. Its
permutation premise preserves factor multiplicities, not merely set membership.
`mini_bucket_elimination` lifts the partitioned bound, together with a remainder
independent of the eliminated atom, to every covered original assignment.

A checked example gives the strict sequence 0 < 1 < 3. This illustrates the
mathematical difference between independent, partially grouped and fully joint
bounds. It is an instance of established min-sum relaxation, not a new algorithm
or a workload speedup. Directed floating-point evaluation and projection can
change numerical bounds; implementation must retain its incumbent and preserve
the original ordered source arithmetic and native-class requirements.

The initial private prototype and six dependencies compiled and passed
independent kernel replay with Lean 4.34.1; a separate source review also passed.
Receipt: `build/proof-research/mini-bucket-kernel-20261009T123412Z/result.json`.
It remains outside the maintained proof collection and default converter until
there is evidence of an advantage over the current deeper-cover baseline.


### Construction reuse is a separate opportunity from class closure

The complete fixed-effort sample-4883 capture has 2,158 retained expanded
states, all with valid nonleaf graph references, but only 372 stored graph
nodes including terminals. Consequently at least 1,786 state-to-output mappings
remain after keeping one representative per referenced node. This conservative
count establishes repeated completed representation; it does not establish
that those constructions could have been avoided cheaply.

The output interner both shares identical node descriptors and bypasses tests
with equal outgoing nodes. The aggregate capture cannot separate these cases.
In particular, the reported `state_reuses` counter means recycled arena slots,
not source-state cache hits. Characterization evidence and derivation:
`build/proof-research/retained-output-sharing-20261009.json`.

Even if every saved context is genuinely mixed, a previously completed
nonconstant decision program might still be reusable under a newly proved
source/domain guard. This needs the existing applicability conditions; matching
output code alone is insufficient. A private source-level hypothesis considers
boxes of starting score prefixes while retaining exact residual inventories
and positions. Its numerical argument must preserve the original rounded
addition order. No broadened cache admission or speedup is claimed here.


### Selective grouping: checked local gain limits

The private mini-bucket prototype now also proves `coarsening_not_weaker`:
merging blocks of existing groups cannot weaken their exact minimum bound when
the factor inventory (with multiplicity) and nonempty atom domain stay the same.
The theorem does not compare arbitrary numerical folds or different downstream
elimination plans.

`cross_probe_merge_gain` supplies a cheaper local screen. For two exact factor
sums A and B, minima mA and mB, and admitted minimizing inputs aA and aB, the
merged gain Delta satisfies
`0 <= Delta <= min(B(aA)-mB, A(aB)-mA)`.
Two cross-evaluations can therefore cap a candidate's improvement without
completing its whole atom sweep. Identical minimizing inputs imply zero local
gain; different saved inputs do not establish positive gain. If the eliminated
axis has only two atoms, a separate screen is not cheaper than its full sweep.

The extended 149-line private prototype compiled and all seven involved
modules passed independent kernel replay. Independent review checked the
explicit membership/attainment premises and source-to-replay correspondence.
Receipt: `build/proof-research/mini-bucket-coarsening-kernel-20261009T124425Z/result.json`.

Cached downward numerical minima do not automatically satisfy exact attainment.
For an implementation, evaluating the planned directed fold at a feasible probe
instead caps the numerical minimum that the full scan could produce. This is
the existing first-atom/optimistic-fold argument; it must retain the same
operation order and keep the incumbent on any failure. A local gain can leave
the final global bound unchanged, so a candidate must complete the affected
downstream recomputation before its resulting certificate is accepted.
The primary-source-based policy, cost ledger and counterexample are recorded in
`build/proof-research/selective-mini-bucket-merges-20261009.md`.


### Maintained proof consolidation and replay

The cache module now reuses the signature module's source-question inventory
and equal-signature leaf argument. Its public alias and agreement theorem remain
available. Pair extrema now derive their four public bounds from two private
characterizations instead of repeating the list inductions. Those two changes
removed 18 source lines and duplicated reasoning without dropping a theorem,
changing hypotheses, or altering any CUDA behavior.

All 28 maintained modules then compiled and passed independent kernel replay
together with Lean 4.34.1, including the new cache import edge and all pair-bound
consumers. Their source snapshots are preserved with the receipt:
`build/proof-research/maintained-simplification-full-replay/1d30583dc639d44f/result.json`.

A subsequent `GroupedMargins` change replaced the private
`computed_min_le_first` induction with `List.foldl_min` and `Int.min_le_left`,
retaining its name and all public statements and hypotheses. This removed two
more lines (301 to 299). The changed module and its four dependencies compiled
and passed independent kernel replay; separate review also passed:
`build/proof-research/grouped-fold-min-kernel-20261009T131718Z/result.json`.
The full 28-module collection has not been replayed together after that final
proof-body change. Both runs were CPU-only and single-threaded; neither asserts
CUDA refinement.


### Prefix-reuse opportunity diagnostic

The restricted-region benchmark now accepts `--prefix-reuse-pairs N` (default
zero). After complete conversion and native comparison, it groups retained
expanded states by exact output-node identity, ordered residual roots, source
positions, projected feature guard and any structural-guard words. Genuine
score-prefix words are omitted from the grouping key to identify candidates
for a separate applicability proof. Fingerprints shortlist; full keys decide.
Native score width zero means all stored channels, not zero score channels.

Aggregate counts cover all retained states. N caps only representative pairs,
with available/reported/omitted counts explicit. Numeric prefix diversity
canonicalizes signed zero for reporting and excludes nonfinite vectors; the
production key is unchanged. Witness contexts are reported separately from the
projected region. Metadata comparison does not evaluate the model on the CPU.
An enabled diagnostic disqualifies the run from comparative timing.

Independent source review passed. The benchmark built with CUDA execution
hidden, and all 16 standalone metadata checks passed, including forced hash
collisions, signed zero, nonfinite inputs and capped examples. Help and invalid
argument handling also passed. Evidence:
`build/prefix-reuse-cpu-build-myapt0uj/cpu-cli-checks.json`.
A subsequent extension adds a separate one-coordinate census: the existing
exact context key plus every other score-prefix word must match. It reports
numeric minimum/maximum endpoints, per-channel membership and distinct-context
counts. Only genuine score channels are tested for finiteness in this new
census; structural suffix words remain opaque and exact. Signed zero has zero
numeric width, forced hash collisions still require full equality, and example
limits do not truncate totals. The earlier all-word summaries are unchanged.
The updated benchmark compiled and all 42 CPU metadata checks passed, with
independent source review. Receipt:
`build/one-coordinate-prefix-cpu-cc17_77y/cpu-cli-checks.json`.
The gather kernel and real-source run await GPU availability; these CPU results
establish neither real candidate counts nor safe cache admission or speedup.


### Single-channel prefix interpolation (private research)

The private Lean prototype now has 196 lines. It proves a potentially
nonconstant program remains valid throughout a single-channel prefix interval
when the same program has tie-aware margin certificates at both endpoints, all
other score channels stay fixed, and the varying channel's ordered score is
monotone. The certificate quantifies over every feature input in the region;
two sampled predictions do not suffice. It reuses the maintained lowest-index
argmax theorem.

The extension preserves the same positive true gap and finite score window
throughout that interval. Its native corollary uses a true gap of at least
2^-14 minus 2^-48 and the score window [-10,10]. Monotone rounding that fixes
-20 derives the rounded rival shift's lower bound; the maintained gap argument
supplies its upper bound -3*2^-16. The exponential envelope is required only on
[-20,-3*2^-16], with the existing symmetric error allowances and one positive
common denominator. Source order, finite encodings, rounding, exponential and
division behavior must still be bound to the actual native execution. The
constant `computed = guard` is an auxiliary proof value; it does not assert a
new runtime subtraction or assume equality of intermediate native labels.

The mathematical direction is connected to classifier equivalence intervals in
[Chan and Darwiche, UAI 2003](https://arxiv.org/pdf/1212.2470), with monotone
rounded folds replacing real additive translations. Mathlib's
[order-connected preimages and intersections](https://leanprover-community.github.io/mathlib4_docs/Mathlib/Order/Interval/Set/OrdConnected.html)
and the local Metamath `pimincfltioc` statement provide related order structure;
they are research references, not imported proofs of the CUDA implementation.

The scoped prototype and its two maintained dependencies, `OrderedArithmetic`
and `NativeClassSeparation`, passed fresh elaboration, independent kernel
replay and review:
`build/proof-research/prefix-gap-window-scoped-kernel-20261009/result.json`.
It stays outside the maintained collection. The theorem supplies a conditional
bridge from positive-gap/window certificates to native separation; endpoint
probability labels alone do not supply those certificates or the numerical
premises. No runtime reuse, candidate hit rate or cost improvement is
established. Two opposite corners of a multi-channel box remain insufficient
even for exact arithmetic. Detailed hypotheses, counterexample and next
experiment: `build/proof-research/prefix-equivalence-research-20261009.md`.


Independent review accepted the interpolation theorem. A separate exact-rational
scalar softmax counterexample rules out an unrestricted unused-class suppression
shortcut: lowering a score for a class absent from a nonconstant program's
outputs can change rounding ties between its used classes. The arithmetic
surrogate includes certified exponential rounding cells and exact division
checks; it is not evidence of a pinned CUDA mismatch or a failure of the
qualified positive-gap theorem. Current output labels do not retain that gap
provenance. Saved audit:
`build/proof-research/unused-class-prefix-audit-20261009/audit.md`.


Two further private results clarify the reuse boundary. A 65-line finite-trace
prototype derives safety at every intermediate addition from safe endpoint
traces, reusing the maintained ordered-enclosure lemma. Identifying the
admitted exact-sum interval and abstract rounder with actual RN32 remains an
explicit premise. Its two modules passed compilation and kernel replay:
`build/proof-research/finite-prefix-trace-kernel-20261009-v2/result.json`.

A 120-line interval-merging prototype proves that two possibly different
programs certified on overlapping prefix intervals agree on the entire shared
feature guard. Either can then serve the connected union. The proof requires
the same score family and positive-gap certificates; a common sampled row is
insufficient. An opposite-variable-order example establishes that identical
program structure is unnecessary. This permits guarded semantic reuse, not
global replacement of every reference to a node outside that guard. The four
modules passed compilation, kernel replay and independent review:
`build/proof-research/prefix-interval-merge-kernel-20261009T132917Z/result.json`.
These are private certificate consequences, not new maintained modules or
measured runtime improvements. The current diagnostic counts same-node
geometry only; it does not claim to measure cross-program interval merging.


### Retaining full-guard evidence through construction (opt-in diagnostic)

The maintained frontier now accepts an optional, caller-owned one-byte-per-state
`gap_evidence::Evidence` buffer. Production callers leave it disabled. The
benchmark enables it with `--prefix-gap-evidence 1` together with
`--prefix-reuse-pairs N`. It observes existing proofs; it does not accept new
proofs, change labels, or admit an interval cache.

A positive record means the completed program has a finite score-window and
positive all-rival gap certificate for its **saved prefix and ordered residual
continuation over the entire projected guard**. It does not mean the original
full forest is certified over the widened feature region. Direct qualified
closures supply this evidence. Expansions compose it only when both outgoing
contexts have it, using the existing normalization/projection correspondence:
source predicates cover the region, original per-channel addition order is
preserved, and residual support only shrinks. Current native/output authority
remains a caller obligation; the diagnostic excludes narrower/composed score
interpretations rather than assuming correspondence from array dimensions.

Native label completions and terminal-label-cache outcomes remain unknown.
Exact-state hits retain their existing evidence. Edge completion consumes the
context evidence before replacing its state reference with a graph-node ID;
equal graph-node IDs alone cannot upgrade an unknown result. Failed node
publication preserves pending evidence for retry. Recycling clears it, and
state growth copies it in the existing staged allocation transaction. The
extra owner is included in memory-budget estimates. Checkpoint formats are
unchanged: all restored evidence starts unknown, including partially resolved
expansions. Unknown does not mean a proof failed or that the region is mixed.

The source bridge is recorded in the private
`build/proof-research/projected-gap-provenance-composition-20261009.md` and uses
existing projection, ordered-arithmetic and generic proof-composition results.
No maintained Lean module was added merely to express a Boolean AND. This is a
reviewed correspondence argument, not a machine-checked refinement of CUDA.

The metadata report separates geometric matches from groups with two qualified
endpoint values. It selects minimum/maximum endpoints only from qualified
contexts, so unknown exterior values cannot enlarge an interval. It reports
both per-channel memberships and distinct retained contexts enclosed by those
endpoints. Counts include the endpoints themselves, do not establish when a
donor became available, and do not measure avoided construction. Example caps
never truncate totals. One-coordinate intervals are not combined into an
unsupported multi-coordinate box. No new per-state disk output is introduced.

The shared lifecycle helper passed 46 CPU bookkeeping checks, and the updated
CUDA benchmark compiled with GPU execution disabled and passed 69 CPU metadata
checks plus help/invalid-option checks. Receipts:
`build/proof-research/gap-evidence-hooks-2ilhkvzu/result.json` and
`build/proof-research/gap-provenance-integration-h1tr9owr/cpu-cli-checks.json`.
The maintained host target `adaptive_gap_evidence_lifecycle_checks` also passed
its 46 checks. The GPU-only target `adaptive_gap_evidence_checks` compiled and
its CPU-only help path passed; its 21 planned scenarios call the actual
scheduler hooks, including staged-growth failure and retry. Receipt:
`build/proof-research/gap-evidence-integration-build-l59980nw/completion.json`.
Independent source review passed the scheduler, report, and both check sources.
CPU checks use synthetic metadata and do not establish the numerical premises
or CUDA hook behavior. GPU lifecycle, memory and real-source checks remain
pending; neither candidate frequency nor speedup is established.

A private lookup design factors the exact residual/guard payload once, then
uses compact per-channel keys and sorted, disjoint certified prefix intervals.
Overlapping intervals for different programs may merge using the previously
checked positive-gap uniqueness result. One predecessor lookup then replaces a
scan of candidate intervals. This is a conditional indexing advantage, not a
measured conversion improvement:
`build/proof-research/certified-prefix-interval-index-20261009.md`.


### Simultaneous unused-score prefix boxes (private research)

A 67-line private Lean result extends qualified reuse beyond one changing
coordinate. Fix every prefix belonging to a class the completed program can
return. Allow each unused class prefix to vary independently between a lower
endpoint and the donor prefix. The donor's full-guard gap certificate supplies
all winner/rival inequalities and upper score bounds; each unused channel needs
only a lower-window certificate over the same feature guard. Monotonicity of
its unchanged ordered score fold then preserves the donor's gap and score
window throughout the entire box. This works for a nonconstant program and
needs no enumeration of classification certificates at every corner.

The proof assumes per-channel dependence, original rounded operand order,
finite encoded arithmetic, a sound output-label support set and the same
projected residual/source interpretation. An abstract Int inequality does not
establish absence of native overflow. Lower-endpoint finite-trace/window
checks and the existing native correspondence remain necessary. The prior
near-tie softmax counterexample remains relevant: a native label alone is not
a qualified donor certificate.

Private source: `build/proof-research/UnusedPrefixBoxPrototype.lean`.
Its four modules passed compilation and independent kernel replay:
`build/proof-research/unused-prefix-box-kernel-20261009-v2/result.json`.
Independent review passed the theorem and the lower-window construction recipe.
The strict two-unused-coordinate example and finite-arithmetic obligations are
recorded in `build/proof-research/unused-prefix-box-20261009.md`.
The maintained proof collection is unchanged. No box cache has been integrated,
and no real-workload hit rate or speedup is claimed.


### Static lower-window check and output-support census

The GPU-only range primitive in
`native/class_conversion/adaptive_prefix_lower_bound.cuh` checks a proposed
lower prefix for one genuine score channel. It visits the exact residual
inventory in original source order, skips consumed roots and other channels,
and adds the admitted static minimum of each selected root with RN32. Every
selected prefix, minimum and intermediate result must remain finite. A final
lower bound at least -10 supplies the current native-window floor. The result
reports inventory visits and additions separately. It proves neither a class
nor cache eligibility by itself, and is not called by default construction.

For a qualified donor with the same residual inventory and projected guard,
static minimum operands enclose the actual leaf operands from below. Pairing
that finite lower trace with the donor's finite actual upper trace derives
finite target execution and the lower score bound. This is an unequal-operand
argument: the earlier same-operand interpolation lemma alone was insufficient.
The private 60-line `StaticLowerPrefixPrototype.lean` closes that obligation
using the maintained one-step enclosure and ordered-fold theorems. Its three
modules passed all six elaboration/kernel stages and independent review:
`build/proof-research/static-lower-prefix-kernel-20261009/result.json`.
Concrete source-array, RN32 and donor/native correspondence remain explicit.
Joint attainability of the different tree minima is unnecessary; loose minima
can only make this sufficient check decline an otherwise usable prefix.

When prefix gap evidence is requested, the benchmark now also computes a
syntactic output-label support mask for each graph node on CUDA. It follows
the interning order, unions successor masks, and supports more than 64 output
classes. The report counts qualified expanded contexts whose programs omit
one or at least two classes, with distinct-node counts reported separately.
Infeasible paths may add extra labels to the masks, making these counts
conservative. A label mask never supplies missing score-gap authority or a
lower-window certificate. Unknown contexts do not become qualified merely
because another context shares their node. This is an optional post-conversion
census, not a production search strategy or timing comparison.

The range-check CUDA fixture and updated benchmark compiled without GPU
execution. All 83 CPU metadata checks and both help paths passed. Evidence:
`build/proof-research/prefix-support-census-l2y7x1v4/cpu-checks.json`.
Independent source review passed. The GPU-only modes
`adaptive_prefix_lower_bound_checks` and
`adaptive_real_region_benchmark --prefix-support-checks` still await execution;
CPU checks do not validate their numerical/kernel behavior. No actual eligible
program counts, cache hits, or speedup have been measured.

### Backward prefix proposals with independent forward acceptance

Private research now has a CUDA proposal generator for the lower endpoint of
an unused-score interval. It traverses the unchanged residual inventory in
reverse, computes each RN32 lower preimage using directed FP64 arithmetic,
and handles an excluded midpoint with one numeric successor. Every accepted
proposal then passes the maintained ORIGINAL-order lower-window check and
must not exceed the finite donor prefix. The inverse never supplies class,
source, guard, or donor authority. This is a private experiment, not another
converter or a change to the default construction strategy.

Two private Lean modules separate the mathematical obligations:

- `RoundingBoundaryPrototype.lean` proves nested-grid directed-ceiling
  composition, uniqueness, and minimality with an optional numeric successor.
  Ceiling/successor existence is explicit. Passing the lower boundary alone
  does not prove that the machine addition is finite.
- `BackwardPrefixThresholdPrototype.lean` proves that exact one-step finite
  preimages compose into a necessary lower cutoff. With the authenticated
  finite donor upper trace, it also derives finite satisfying execution for
  every prefix between cutoff and donor, allowing unequal enclosing operands.
  A failed inverse returns unknown; these results do not assert completeness.

Both modules passed elaboration and independent kernel replay. The binary32
midpoint law, bit/intrinsic correspondence and source bindings remain concrete
premises, rather than being hidden behind abstract integer arithmetic. The
candidate-and-forward-check path does not depend on correctness of the
proposal generator: the successful ordered check already supplies the lower
certificate used by the earlier box theorem. This allows future proposals to
improve search without weakening acceptance.

For optional minimality, check the distinct numeric predecessor. A finite
final value below the floor, or first negative overflow, rules out every
smaller prefix for this static-minimum method. Invalid metadata, unavailable
work, NaN and positive overflow do not establish minimality. The smallest
static-method prefix can still be conservative for the actual region-dependent
score: complementary factors can have individually zero minima while their
sum is always one. This is a missed bound, not evidence of mixed classes.

The experimental CUDA source compiled for fixed SM86 with gradual underflow,
explicit RN additions and no reassociation. Only its CPU help path was run.
Prepared GPU fixtures compare boundary cases against independent numeric-word
bisection, source-order suffix cases, and the existing forward guard. No GPU
numerical execution, eligible-context count, cache admission, or timing gain
is established by compilation. The maintained Lean collection is unchanged.

One reverse inventory pass plus one forward validation pass replaces repeated
whole-fold probes when the proposal succeeds. At fixed binary32 precision both
methods are linear in inventory size; the prospective advantage is fewer
passes, not a different asymptotic order. FP64 setup may cost more than a single
target check, so amortization on actual cache hits must be measured.

Evidence and remaining prerequisites:
`build/proof-research/backward-prefix-threshold-kernel-20261009-v3/result.json`,
`build/proof-research/rounding-boundary-kernel-20261009-v2/result.json`,
`build/proof-research/prefix-inverse-candidate-og4soemi/build-result.json`, and
`build/proof-research/verified-prefix-candidate-admission-20261009.md`.
Derivation: `build/proof-research/static-lower-prefix-and-inverse-20261009.md`.


### One-comparison qualified winner selection

The maintained `qualified_range_label` validates all intervals while selecting
one maximum lower bound, then compares it with the maximum competing upper
bound. Positive qualified separation forces any successful class to be that
unique lower maximum. Monotonic rounded subtraction makes the maximum rival
upper the hardest comparison. The same native gate, full-channel range
validation and RN64 threshold are retained; no source qualification is widened.

Two corollaries in `NativeClassSeparation.lean` reuse the maintained
maximum-selection lemmas: `qualified_winner_is_max_lower` and
`maximum_rival_margin_iff`. They add 40 lines to the existing module, including
comments and axiom reports. Elaboration and independent kernel replay passed.
Concrete finite encodings, the IEEE rounder correspondence, complete inventory
validation and native runtime qualification remain separate premises.

For seven valid classes, worst-case margin comparisons fall from 42 to 1,
with 12 ordinary comparisons for the two maxima. This is not a 42-fold runtime
claim: early successful baselines already use fewer margin tests, and invalid
ranges can stop before any margins. The private CUDA differential fixture passed
3,668 checks; the maintained joint-bound fixture, extended with selector cases,
passed 3,301 checks. Those fixtures do not themselves grant native authority.

Fresh matched baseline/candidate builds completed an A-B-B-A comparison on eight
restricted regions of the 448-tree source, with three internal repetitions,
fixed/dynamic effort and separate optional-unary arms. All 384 conversions
completed with zero native mismatches and identical recorded non-time work,
output size and peak owned memory. Default-arm sums of regional medians were
about 685 to 680 ms (fixed) and 518 to 510 ms (dynamic). Fixed paired timings
were effectively unchanged; dynamic timings provide a modest positive signal,
not a statistical significance claim. These are restricted-region results;
the full-domain 448-tree conversion remains incomplete.

Evidence: `build/proof-research/maintained-linear-selector-6h3i71ig/result.json`,
`build/proof-research/selector-maintained-dl93m1e7/gpu-result.json`,
`build/proof-research/linear-qualified-selector-fhadwats/gpu-result-20261009.json`,
and `build/linear-selector-abba-w6p7fcsm/measurement/summary.json`.

A separate private 57-line result, `StableChannelSelectionPrototype.lean`,
reuses Std's list-fold/filter laws to justify stable per-channel execution and
reverse selection. It does not remove global inventory validation or make
shared error/counter state independent. That scan optimization remains deferred
pending evidence of its value.

### Static compilation evidence for the selector

The exhaustive baseline and one-margin selector were compiled as otherwise
identical, uninstrumented device-only SM86 kernels with runtime `EngineView`
and output-pointer arguments. The strict CUDA 13.4 compilation produced:

| Isolated selector | Registers | Aligned code bytes | Stack / local / spills |
| --- | ---: | ---: | --- |
| Exhaustive baseline | 32 | 2,432 | 0 / 0 / 0 |
| One-margin selector | 14 | 2,048 | 0 / 0 / 0 |

Inspection also exposed a reload of the selected lower bound. Retaining the
already selected value removes one dependent global-load instruction site and
four non-padding instruction sites, with unchanged candidate registers and
aligned code size. The reference kernel's generated instructions remain
identical across these two candidate builds. The maintained selector now uses
that cached value; the numerical and real-region evidence is reported above.

These are isolated code-generation observations. They establish neither
integrated converter occupancy nor latency, memory traffic or runtime speedup.
The baseline's seven static subtraction sites result from unrolling and tails;
they are not its dynamic comparison count. Likewise, the source maximum-scan
comparison count need not equal the generated instruction count. Small class
counts, early baseline success and invalid-range paths require matched runtime
measurements; the completed comparison above supplies that separate evidence.

Evidence: `build/proof-research/linear-selector-static-cached-cuvdebj1/analysis.md`,
its `static-analysis.json` and `comparison.json`, plus the rebuilt private
fixture's `linear-qualified-selector-fhadwats/build-result.json`.
