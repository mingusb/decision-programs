import PairedBounds

/-!
Cross-channel margin bounds for ordered rounded reductions. Compatible leaf
pairs share one input region; membership must cover every actual pair, including
missing-value and categorical paths. Each channel keeps its source operand
order. An absent term is an actual no-op, not an extra floating addition.

Finite values and error bounds are represented in a common exact integer unit.
The spacing, nearest-rounding, directed-arithmetic, and path-coverage premises
are explicit. This file does not prove CUDA instructions, IEEE spacing, or the
native softprob gate. The existing qualified range and native gap are retained.
-/
namespace ConverterRelationalMargins
open ConverterArithmetic ConverterPairedBounds

/-- Nearest selection bounds signed errors by half the corresponding spacing.
Finite neighbors, nearest selection and their spacing bounds are caller premises, not an IEEE or
CUDA verification. Use a common integer unit fine enough for half-spacing:
for binary32, 2^-150 (or finer) also represents half a subnormal spacing. -/
theorem nearest_spacing_error (representable : Int → Prop) (round : Int → Int)
    (nearest : ∀ x v, representable v → distance x (round x) ≤ distance x v)
    (x below above error : Int)
    (hb : representable below) (ha : representable above)
    (lower : below < round x) (upper : round x < above)
    (lowerGap : round x - below ≤ 2 * error) (upperGap : above - round x ≤ 2 * error) :
    x - error ≤ round x ∧ round x ≤ x + error := by
  have nonneg : 0 ≤ error := by omega
  have left := nearest_reversal_boundary x (round x) below lower (nearest x below hb)
  have right := nearest_lower_boundary x above (round x) upper (nearest x above ha)
  constructor <;> omega

/-- One ordered update in each channel, with independent signed error bounds.
The lower-bound computation itself may round downward. -/
theorem margin_step (w r a b w' r' bound difference ew er next : Int)
    (before : bound ≤ w - r) (pair : difference ≤ a - b)
    (winnerError : w + a - ew ≤ w') (rivalError : r' ≤ r + b + er)
    (downward : next ≤ bound + difference - ew - er) :
    next ≤ w' - r' := by omega

/-- Directed addition/subtraction and upward error summation realize the
required downward step without assuming exact double arithmetic. -/
theorem directed_margin_step (addDown subDown addUp : Int → Int → Int)
    (hadd : ∀ x y, addDown x y ≤ x + y)
    (hsub : ∀ x y, subDown x y ≤ x - y)
    (hup : ∀ x y, x + y ≤ addUp x y)
    (bound difference ew er : Int) :
    subDown (addDown bound difference) (addUp ew er) ≤
      bound + difference - ew - er := by
  have := hadd bound difference
  have := hsub (addDown bound difference) (addUp ew er)
  have := hup ew er
  omega

structure MarginBlock where
  first : Pair
  rest : List Pair
  winnerActive : Bool := true
  rivalActive : Bool := true
  winnerError : Int
  rivalError : Int

def candidates (b : MarginBlock) : List Pair := b.first :: b.rest

def differenceLower (b : MarginBlock) : Int :=
  minOver (fun p => p.1 - p.2) b.first b.rest

theorem difference_lower_sound (b : MarginBlock) (p : Pair)
    (member : p ∈ candidates b) : differenceLower b ≤ p.1 - p.2 :=
  min_over_le_member _ _ _ _ member

/-- Safe overapproximations of compatible pairs may be pruned only by a
justification retaining every possible pair. Their minimum cannot decrease. -/
theorem difference_filter_tightens (old new : MarginBlock)
    (subset : ∀ p ∈ candidates new, p ∈ candidates old) :
    differenceLower old ≤ differenceLower new := by
  apply le_min_over
  intro p hp
  exact difference_lower_sound old p (subset p hp)

def channelStep (round : Int → Int) (active : Bool) (acc term : Int) : Int :=
  if active then round (acc + term) else acc

/-- Missing residuals carry zero mathematical contribution and no rounding
operation. Zero errors suffice for this case. -/
theorem inactive_channel (round : Int → Int) (acc : Int) :
    channelStep round false acc 0 = acc := rfl

def channelEncloses (round : Int → Int) (active : Bool) (acc term error : Int) : Prop :=
  (active = false → term = 0) ∧
  acc + term - error ≤ channelStep round active acc term ∧
  channelStep round active acc term ≤ acc + term + error

theorem inactive_encloses (round : Int → Int) (acc : Int) :
    channelEncloses round false acc 0 0 := by simp [channelEncloses, channelStep]

def actualStep (round : Int → Int) (b : MarginBlock) (s p : Pair) : Pair :=
  (channelStep round b.winnerActive s.1 p.1,
   channelStep round b.rivalActive s.2 p.2)

def stepValid (round : Int → Int) (b : MarginBlock) (s p : Pair) : Prop :=
  p ∈ candidates b ∧
  channelEncloses round b.winnerActive s.1 p.1 b.winnerError ∧
  channelEncloses round b.rivalActive s.2 p.2 b.rivalError

def exactBoundStep (bound : Int) (b : MarginBlock) : Int :=
  bound + differenceLower b - b.winnerError - b.rivalError

theorem block_margin_sound (round : Int → Int) (b : MarginBlock)
    (s p : Pair) (bound next : Int) (before : bound ≤ s.1 - s.2)
    (valid : stepValid round b s p) (downward : next ≤ exactBoundStep bound b) :
    next ≤ (actualStep round b s p).1 - (actualStep round b s p).2 :=
  margin_step _ _ _ _ _ _ _ _ _ _ _ before
    (difference_lower_sound b p valid.1) valid.2.1.2.1 valid.2.2.2.2 downward

def ActualTrace (round : Int → Int) : List MarginBlock → List Pair → Pair → Prop
  | [], [], _ => True
  | b :: bs, p :: ps, s =>
      stepValid round b s p ∧ ActualTrace round bs ps (actualStep round b s p)
  | _, _, _ => False

def actualFold (round : Int → Int) : List MarginBlock → List Pair → Pair → Pair
  | b :: bs, p :: ps, s => actualFold round bs ps (actualStep round b s p)
  | _, _, s => s

def boundFold (boundStep : Int → MarginBlock → Int) : List MarginBlock → Int → Int
  | [], bound => bound
  | b :: bs, bound => boundFold boundStep bs (boundStep bound b)

/-- Each input may select different feasible pairs. Only within-pair
compatibility is retained; no independence between blocks is assumed. -/
theorem ordered_margin_encloses (round : Int → Int)
    (boundStep : Int → MarginBlock → Int)
    (downward : ∀ bound b, boundStep bound b ≤ exactBoundStep bound b)
    (bs : List MarginBlock) : ∀ ps s bound,
    ActualTrace round bs ps s → bound ≤ s.1 - s.2 →
    boundFold boundStep bs bound ≤
      (actualFold round bs ps s).1 - (actualFold round bs ps s).2 := by
  induction bs with
  | nil =>
    intro ps s bound trace before
    cases ps with
    | nil => exact before
    | cons p ps => exact False.elim trace
  | cons b bs ih =>
    intro ps s bound trace before
    cases ps with
    | nil => exact False.elim trace
    | cons p ps =>
      exact ih ps (actualStep round b s p) (boundStep bound b) trace.2
        (block_margin_sound round b s p bound (boundStep bound b) before trace.1
          (downward bound b))

/-- The recurrence's exact bookkeeping equals one sum of pair floors minus
both channels' accumulated errors. This equality reassociates exact integers,
not either channel's ordered rounded additions. -/
theorem deferred_error_identity (bs : List MarginBlock) : ∀ bound,
    boundFold exactBoundStep bs bound =
      bound + (bs.map differenceLower).sum -
        (bs.map MarginBlock.winnerError).sum - (bs.map MarginBlock.rivalError).sum := by
  induction bs with
  | nil => intro bound; simp [boundFold]
  | cons b bs ih =>
    intro bound
    rw [boundFold, ih]
    simp only [List.map_cons, List.sum_cons, exactBoundStep]
    omega

/-- The executable implementation may accumulate leaf-difference floors first,
then subtract both upward error totals. Its final directed result must lie below
this exact expression; no intermediate FP64 addition is assumed exact. -/
theorem deferred_margin_encloses (round : Int → Int) (bs : List MarginBlock)
    (ps : List Pair) (s : Pair) (bound final : Int)
    (trace : ActualTrace round bs ps s) (initial : bound ≤ s.1 - s.2)
    (directed : final ≤ bound + (bs.map differenceLower).sum -
      (bs.map MarginBlock.winnerError).sum - (bs.map MarginBlock.rivalError).sum) :
    final ≤ (actualFold round bs ps s).1 - (actualFold round bs ps s).2 := by
  have h := ordered_margin_encloses round exactBoundStep
    (fun _ _ => Int.le_refl _) bs ps s bound trace initial
  rw [deferred_error_identity] at h
  exact Int.le_trans directed h

#print axioms deferred_error_identity
#print axioms deferred_margin_encloses

#print axioms nearest_spacing_error
#print axioms directed_margin_step
#print axioms difference_lower_sound
#print axioms difference_filter_tightens
#print axioms ordered_margin_encloses
/- Eight independent binary stumps share their contribution across two classes:
   W = 1/4 + sum x_i, R = sum x_i. Each x_i is 0 or 1. In units
   2^-20, a full FP32 ULP throughout [0, 8.25] is at most one unit;
   the illustrative 2^-10 gap is 1024 units. The spacing premise above
   and source qualification remain required; this is an algebraic witness.

   A separate-class interval loses all common-mode correlation. Resolving
   one input still leaves seven independent stumps, so neither a single
   common cover nor a single rival cover suffices. Two distinct stumps
   within one class have all four combinations and give no paired tightening.
-/
private def commonModeBlock : MarginBlock :=
  { first := (0, 0), rest := [(1048576, 1048576)],
    winnerError := 1, rivalError := 1 }

theorem independent_common_mode_fails (fixed : Int) (unresolved : Nat)
    (nonempty : 0 < unresolved) :
    (fixed + 262144) - (fixed + (unresolved : Int) * 1048576) < 1024 := by
  omega

/-- A conservative sixteen-ULP allowance leaves more than the qualified
native margin. Every input selects a diagonal pair at each of the eight
steps, so `ordered_margin_encloses` applies whenever its error premises hold. -/
theorem eight_stump_strict_improvement :
    boundFold exactBoundStep (List.replicate 8 commonModeBlock) 262144 = 262128 ∧
    1024 < boundFold exactBoundStep (List.replicate 8 commonModeBlock) 262144 ∧
    (262144 - 8 * 1048576 : Int) < 1024 ∧
    (262144 - 7 * 1048576 : Int) < 1024 := by decide

#print axioms independent_common_mode_fails
#print axioms eight_stump_strict_improvement
end ConverterRelationalMargins