import RelationalMargins
import CoverRefinement

/-!
Group signed residual factors by a certified unary coordinate, minimize each
whole group, and retain independent floors for unresolved factors. Exact group
bookkeeping is separate from the original ordered rounded source additions.

All quantities are exact integers in a common unit. Source partition identity,
region-restricted unary dependence, exhaustive finite atom coverage, directed
arithmetic, original rounding-error envelopes and the native prediction gate
are explicit premises. This is not a formal CUDA implementation refinement.
-/
namespace ConverterGroupedMargins
open ConverterRelationalMargins ConverterPairedBounds
universe u v w
variable {X : Type u} {C : Type v} {Label : Type w}

/-- Telescope the ORIGINAL selected leaf words and their original ordered
channel updates. No adjusted potential or adjusted leaf enters `actualStep`. -/
theorem original_operand_sum_margin (round : Int → Int) (bs : List MarginBlock) :
    ∀ ps s, ActualTrace round bs ps s →
    s.1 - s.2 + (ps.map Prod.fst).sum - (ps.map Prod.snd).sum -
      (bs.map MarginBlock.winnerError).sum - (bs.map MarginBlock.rivalError).sum ≤
        (actualFold round bs ps s).1 - (actualFold round bs ps s).2 := by
  induction bs with
  | nil =>
    intro ps s trace
    cases ps with
    | nil => simp [actualFold]
    | cons p ps => exact False.elim trace
  | cons b bs ih =>
    intro ps s trace
    cases ps with
    | nil => exact False.elim trace
    | cons p ps =>
      have rest := ih ps (actualStep round b s p) trace.2
      have one := margin_step s.1 s.2 p.1 p.2
        (actualStep round b s p).1 (actualStep round b s p).2
        (s.1 - s.2) (p.1 - p.2) b.winnerError b.rivalError
        (s.1 - s.2 + (p.1 - p.2) - b.winnerError - b.rivalError)
        (Int.le_refl _) (Int.le_refl _) trace.1.2.1.2.1 trace.1.2.2.2.2 (by omega)
      simp only [List.map_cons, List.sum_cons, actualFold]
      omega

/-- Any GLOBAL floor on the original signed operand sum is sufficient. This
is the entry point for grouped-tree floors as well as balanced transfers. -/
theorem original_sum_lower_margin (round : Int → Int) (bs : List MarginBlock)
    (ps : List Pair) (s : Pair) (trace : ActualTrace round bs ps s)
    (initial sumFloor candidate : Int) (prefixBound : initial ≤ s.1 - s.2)
    (sumBound : sumFloor ≤ (ps.map Prod.fst).sum - (ps.map Prod.snd).sum)
    (directed : candidate ≤ initial + sumFloor -
      (bs.map MarginBlock.winnerError).sum - (bs.map MarginBlock.rivalError).sum) :
    candidate ≤ (actualFold round bs ps s).1 - (actualFold round bs ps s).2 := by
  have originalMargin := original_operand_sum_margin round bs ps s trace
  omega

def factorSum (fs : List (X → Int)) (x : X) : Int := (fs.map fun f => f x).sum

def LocalFloors (region : X → Prop) : List Int → List (X → Int) → Prop
  | [], [] => True
  | lower :: rest, f :: fs => (∀ x, region x → lower ≤ f x) ∧ LocalFloors region rest fs
  | _, _ => False

/-- Each factor's oracle may overapproximate its feasible domain, but its
floor must cover every allowed input. Factors need not be independent. -/
theorem local_floors_sum (region : X → Prop) (fs : List (X → Int)) :
    ∀ lows x, LocalFloors region lows fs → region x → lows.sum ≤ factorSum fs x := by
  induction fs with
  | nil =>
    intro lows x cert inside
    cases lows with
    | nil => exact Int.le_refl _
    | cons l ls => exact False.elim cert
  | cons f fs ih =>
    intro lows x cert inside
    cases lows with
    | nil => exact False.elim cert
    | cons l ls =>
      change l + ls.sum ≤ f x + factorSum fs x
      exact Int.add_le_add (cert.1 x inside) (ih ls x cert.2 inside)

/-- Grouping rearranges only the exact signed factor bookkeeping. It does
not group or reorder the rounded source evaluation. -/
theorem grouped_sum_identity (groups : List (List (X → Int))) (x : X) :
    factorSum (groups.map fun fs => fun y => factorSum fs y) x =
      factorSum groups.flatten x := by
  induction groups with
  | nil => rfl
  | cons fs rest ih =>
    simp only [List.map_cons, factorSum, List.sum_cons, List.flatten_cons,
      List.map_append, List.sum_append] at *
    rw [ih]

theorem grouped_floors_sound (region : X → Prop) (groups : List (List (X → Int)))
    (lows : List Int)
    (cert : LocalFloors region lows (groups.map fun fs => fun y => factorSum fs y))
    (x : X) (inside : region x) : lows.sum ≤ factorSum groups.flatten x := by
  have h := local_floors_sum region _ lows x cert inside
  rw [grouped_sum_identity] at h
  exact h

/-- `representative` maps an input to its threshold-interval signature or
canonical sample, NOT its raw numeric measurement. The group must be constant
on each represented interval. Every allowed signature must occur in the finite
atom list, including allowed missing/category atoms. Thus arbitrary fractional
measurements are covered without being individually enumerated. Per-atom bounds
may be conservative. This premise does not certify the CUDA atom enumerator. -/
theorem exhaustive_axis_floor {Atom : Type v}
    (region : X → Prop) (group : List (X → Int))
    (representative : X → Atom) (value : Atom → Int) (atoms : List Atom)
    (unary : ∀ x, region x → factorSum group x = value (representative x))
    (complete : ∀ x, region x → representative x ∈ atoms)
    (floor : Int) (atomBounds : ∀ a ∈ atoms, floor ≤ value a) :
    ∀ x, region x → floor ≤ factorSum group x := by
  intro x inside
  rw [unary x inside]
  exact atomBounds (representative x) (complete x inside)

private theorem computed_min_le_first (values : List Int) (first : Int) :
    values.foldl min first ≤ first := by
  rw [List.foldl_min]
  exact Int.min_le_left _ _

/-- The first computed atom floor belongs to the eventual minimum. If it does
not exceed the independent incumbent, scanning the remaining computed floors
cannot improve this numerical routine's result. These are common-unit integer
encodings of computed values, not an assertion that the exact sum attains its
minimum or that a stronger mathematical bound is impossible. -/
theorem first_atom_no_improvement (independent first : Int) (remaining : List Int)
    (hit : first ≤ independent) :
    max independent (remaining.foldl min first) = independent :=
  Int.max_eq_left (Int.le_trans (computed_min_le_first remaining first) hit)

/-- The interrupted fallback is the independent floor; the complete numerical
floor lies above it and below this first-atom cap. The cap bounds this routine's
computed floor, NOT the true source score. -/
theorem computed_group_bounds (independent first : Int) (remaining : List Int) :
    independent ≤ max independent (remaining.foldl min first) ∧
    max independent (remaining.foldl min first) ≤ max independent first := by
  constructor
  · exact Int.le_max_left _ _
  · exact Int.max_le.mpr ⟨Int.le_max_left _ _,
      Int.le_trans (computed_min_le_first remaining first) (Int.le_max_right _ _)⟩

/-- Apply caps to the SAME ordered numerical fold, prefix and two error
subtractions. Each triple's middle value is a completed floor or independent
fallback, not an actual source operand/score. The monotone rounder premise is
explicit; this theorem neither verifies CUDA rounding nor changes the native
gate. Failure of the optimistic candidate rules out this numerical attempt. -/
theorem optimistic_rival_rejection (round : Int → Int)
    (mono : ConverterArithmetic.MonotoneRound round)
    (terms : List ConverterArithmetic.IntervalTriple) (initial : Int)
    (bounds : ∀ t ∈ terms, ConverterArithmetic.Encloses t)
    (winnerError rivalError threshold : Int) :
    let sums := ConverterArithmetic.orderedReduce round ⟨initial, initial, initial⟩ terms
    round (round (sums.upper - winnerError) - rivalError) < threshold →
      round (round (sums.actual - winnerError) - rivalError) < threshold := by
  dsimp only
  intro rejected
  let sums := ConverterArithmetic.orderedReduce round ⟨initial, initial, initial⟩ terms
  have enclosed : ConverterArithmetic.Encloses sums :=
    ConverterArithmetic.ordered_reduction_encloses round mono terms _
      ⟨Int.le_refl _, Int.le_refl _⟩ bounds
  have afterWinner := mono (sums.actual - winnerError) (sums.upper - winnerError)
    (by have := enclosed.2; omega)
  exact Int.lt_of_le_of_lt
    (mono _ _ (show round (sums.actual - winnerError) - rivalError ≤
      round (sums.upper - winnerError) - rivalError by omega)) rejected
/-- Direct executable grouping API: each original signed factor occurs once
in the flattening, with unresolved factors represented by singleton groups. -/
theorem grouped_margin_encloses (round : Int → Int) (bs : List MarginBlock)
    (ps : List Pair) (s : Pair) (region : X → Prop)
    (groups : List (List (X → Int))) (lows : List Int)
    (cert : LocalFloors region lows (groups.map fun fs => fun y => factorSum fs y))
    (x : X) (inside : region x) (trace : ActualTrace round bs ps s)
    (sourceBinding : factorSum groups.flatten x =
      (ps.map Prod.fst).sum - (ps.map Prod.snd).sum)
    (initial candidate : Int) (prefixBound : initial ≤ s.1 - s.2)
    (directed : candidate ≤ initial + lows.sum -
      (bs.map MarginBlock.winnerError).sum - (bs.map MarginBlock.rivalError).sum) :
    candidate ≤ (actualFold round bs ps s).1 - (actualFold round bs ps s).2 := by
  have lower := grouped_floors_sound region groups lows cert x inside
  apply original_sum_lower_margin round bs ps s trace initial lows.sum candidate prefixBound
  · omega
  · exact directed

/-- Scales to arbitrarily many independent OR dependent coordinates; the
zero-sum fact itself does not require an independence assumption. -/
theorem split_groups_zero_floor (fs : List (X → Int)) :
    LocalFloors (fun _ => True) (List.replicate fs.length 0)
      ((fs.map fun f => [f, f, fun x => -2 * f x]).map
        fun group => fun x => factorSum group x) := by
  induction fs with
  | nil => trivial
  | cons f fs ih =>
    constructor
    · intro x _
      change 0 ≤ factorSum [f, f, fun y => -2 * f y] x
      simp only [factorSum, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
      omega

    · exact ih

private def halfBit (b : Bool) : Int := if b then 1048576 else 0
private def binaryMin (f : Bool → Int) : Int := min (f false) (f true)
private def binaryPairMin (f : Bool → Bool → Int) : Int :=
  min (min (f false false) (f false true)) (min (f true false) (f true true))

/-- Units are 2^-21. These are exact finite minima, not sampled estimates.
Pairing one positive half with a same-coordinate negative whole leaves -1/2;
a different independent coordinate leaves -1. The triple's minimum is zero. -/
theorem split_minima :
    binaryMin (fun b => halfBit b - 2 * halfBit b) = -1048576 ∧
    binaryPairMin (fun a b => halfBit a - 2 * halfBit b) = -2097152 ∧
    binaryMin (fun b => halfBit b + halfBit b - 2 * halfBit b) = 0 := by decide

/-- Every disjoint cross-class pairing leaves a negative bound: there are
four negative whole factors, and each can consume at most one positive half.
An unpaired negative factor is no better than a different-coordinate pair.
This is not a claim about all possible pairwise message-passing algorithms. -/
theorem every_disjoint_pairing_fails (same differentOrUnpaired : Nat)
    (covers : same + differentOrUnpaired = 4) :
    (524288 - (same : Int) * 1048576 - (differentOrUnpaired : Int) * 2097152 : Int)
      < 2048 := by omega

/-- Even resolving one of four predicates leaves a failed independent bound.
This illustrative fixture uses a 2^-10 gap. Four grouped triple floors are zero;
twelve conservative full-ULP allowances leave a strict certified gap. -/
theorem four_coordinate_strict_improvement :
    (524288 - 4 * 2097152 : Int) < 2048 ∧
    (524288 - 3 * 2097152 : Int) < 2048 ∧
    (524288 - 12 : Int) = 524276 ∧
    (2048 : Int) < 524276 := by decide

private def fourAxes : List ((Fin 4 → Bool) → Int) :=
  [fun x => halfBit (x 0), fun x => halfBit (x 1),
   fun x => halfBit (x 2), fun x => halfBit (x 3)]
private def fourGroups := fourAxes.map fun f => [f, f, fun x => -2 * f x]
private def originalWitnessPairs (x : Fin 4 → Bool) : List Pair :=
  [(halfBit (x 0), 2 * halfBit (x 0)), (halfBit (x 0), 2 * halfBit (x 1)),
   (halfBit (x 1), 2 * halfBit (x 2)), (halfBit (x 1), 2 * halfBit (x 3)),
   (halfBit (x 2), 0), (halfBit (x 2), 0), (halfBit (x 3), 0), (halfBit (x 3), 0)]
private def witnessBoth : MarginBlock :=
  { first := (0, 0), rest := [(1048576, 0), (0, 2097152), (1048576, 2097152)],
    winnerError := 1, rivalError := 1 }
private def witnessWinnerOnly : MarginBlock :=
  { first := (0, 0), rest := [(1048576, 0)], rivalActive := false,
    winnerError := 1, rivalError := 0 }
private def originalWitnessBlocks : List MarginBlock :=
  List.replicate 4 witnessBoth ++ List.replicate 4 witnessWinnerOnly

/-- Eight ORIGINAL winner updates and four ORIGINAL rival updates. Grouping
changes no operand or rounded addition. The trace premise supplies the twelve
one-unit error bounds; the other four rival steps are exact no-ops. For finite
FP32 inputs with scores in [0,4.25], one unit is a conservative full ULP, but
that executable spacing correspondence is not asserted by this integer proof. -/
theorem four_coordinate_original_margin (round : Int → Int) (x : Fin 4 → Bool)
    (trace : ActualTrace round originalWitnessBlocks (originalWitnessPairs x) (524288, 0)) :
    524276 ≤
      (actualFold round originalWitnessBlocks (originalWitnessPairs x) (524288, 0)).1 -
      (actualFold round originalWitnessBlocks (originalWitnessPairs x) (524288, 0)).2 := by
  apply grouped_margin_encloses round originalWitnessBlocks (originalWitnessPairs x)
    (524288, 0) (fun _ => True) fourGroups (List.replicate 4 0)
    (split_groups_zero_floor fourAxes) x trivial trace
  · simp [fourGroups, fourAxes, factorSum, originalWitnessPairs]
    omega
  · exact Int.le_refl _
  · decide

/-- Feed the per-rival result above into the SAME region/native-gate theorem.
Range qualification and the authentic native transformation stay in nativeGate;
this composition introduces neither a new authority nor a weakened gap. -/
theorem unchanged_native_gate
    (region : X → Prop) (rival : C → Prop) (actualGap : C → X → Int)
    (floor : C → Int) (threshold : Int) (predict : X → Label) (winner : Label)
    (proved : ∀ c, rival c → CoverRefinement.LowerCert (· ≤ ·) region (actualGap c) (floor c))
    (sufficient : ∀ c, rival c → threshold ≤ floor c)
    (nativeGate : ∀ x, region x → (∀ c, rival c → threshold ≤ actualGap c x) →
      predict x = winner) :
    ∀ x, region x → predict x = winner :=
  CoverRefinement.rival_floors_certify_winner (le := fun a b : Int => a ≤ b)
    (fun {_a _b _c} hab hbc => Int.le_trans hab hbc)
    region rival actualGap floor threshold predict winner proved sufficient nativeGate

#print axioms original_operand_sum_margin
#print axioms original_sum_lower_margin
#print axioms grouped_floors_sound
#print axioms exhaustive_axis_floor
#print axioms first_atom_no_improvement
#print axioms computed_group_bounds
#print axioms optimistic_rival_rejection
#print axioms grouped_margin_encloses
#print axioms split_groups_zero_floor
#print axioms split_minima
#print axioms every_disjoint_pairing_fails
#print axioms four_coordinate_original_margin
#print axioms unchanged_native_gate
end ConverterGroupedMargins
