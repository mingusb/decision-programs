import OrderedArithmetic

/-!
Finite joint leaf-pair interval propagation. Each block preserves the original
per-class operand order through TWO rounded additions. No reassociation or
constant pre-summing is permitted. Pair completeness/geometry and the finite
IEEE rounder refinement are explicit caller obligations. These are numeric
bounds, never signed-zero word-identity or native-label authorization.
-/
namespace ConverterPairedBounds
open ConverterArithmetic

abbrev Pair := Int × Int

def pairValue (round : Int → Int) (acc : Int) (p : Pair) : Int :=
  round (round (acc + p.1) + p.2)

theorem pair_value_monotone (round : Int → Int) (mono : MonotoneRound round)
    (x y a a' b b' : Int) (hxy : x ≤ y) (ha : a ≤ a') (hb : b ≤ b') :
    pairValue round x (a,b) ≤ pairValue round y (a',b') := by
  change round (round (x+a)+b) ≤ round (round (y+a')+b')
  have h := mono (x+a) (y+a') (by omega)
  exact mono _ _ (by omega)

def minOver (f : Pair → Int) : Pair → List Pair → Int
  | p, [] => f p
  | p, q :: qs => min (f p) (minOver f q qs)

def maxOver (f : Pair → Int) : Pair → List Pair → Int
  | p, [] => f p
  | p, q :: qs => max (f p) (maxOver f q qs)

theorem min_over_le_member (f : Pair → Int) (ps : List Pair) :
    ∀ p q, q ∈ p :: ps → minOver f p ps ≤ f q := by
  induction ps with
  | nil => intro p q h; simp only [List.mem_cons, List.not_mem_nil, or_false] at h; subst q; exact Int.le_refl _
  | cons r rs ih =>
    intro p q h
    simp only [List.mem_cons] at h
    rcases h with h | h
    · subst q; exact Int.min_le_left _ _
    · exact Int.le_trans (Int.min_le_right _ _) (ih r q (List.mem_cons.mpr h))

theorem le_min_over (f : Pair → Int) (ps : List Pair) :
    ∀ p k, (∀ q ∈ p :: ps, k ≤ f q) → k ≤ minOver f p ps := by
  induction ps with
  | nil => intro p k h; exact h p (by simp)
  | cons r rs ih =>
    intro p k h
    apply Int.le_min.mpr
    exact ⟨h p (by simp), ih r k (by intro q hq; exact h q (List.mem_cons.mpr (Or.inr hq)))⟩

theorem member_le_max_over (f : Pair → Int) (ps : List Pair) :
    ∀ p q, q ∈ p :: ps → f q ≤ maxOver f p ps := by
  induction ps with
  | nil => intro p q h; simp only [List.mem_cons, List.not_mem_nil, or_false] at h; subst q; exact Int.le_refl _
  | cons r rs ih =>
    intro p q h
    simp only [List.mem_cons] at h
    rcases h with h | h
    · subst q; exact Int.le_max_left _ _
    · exact Int.le_trans (ih r q (List.mem_cons.mpr h)) (Int.le_max_right _ _)

theorem max_over_le (f : Pair → Int) (ps : List Pair) :
    ∀ p k, (∀ q ∈ p :: ps, f q ≤ k) → maxOver f p ps ≤ k := by
  induction ps with
  | nil => intro p k h; exact h p (by simp)
  | cons r rs ih =>
    intro p k h
    apply Int.max_le.mpr
    exact ⟨h p (by simp), ih r k (by intro q hq; exact h q (List.mem_cons.mpr (Or.inr hq)))⟩

structure Interval where
  lo : Int
  hi : Int

def Contains (i : Interval) (x : Int) : Prop := i.lo ≤ x ∧ x ≤ i.hi

def Tighter (old new : Interval) : Prop := old.lo ≤ new.lo ∧ new.hi ≤ old.hi

structure Block where
  first : Pair
  rest : List Pair
  independentLow : Pair
  independentHigh : Pair

def Members (b : Block) : List Pair := b.first :: b.rest

def Valid (b : Block) : Prop := ∀ p ∈ Members b,
  b.independentLow.1 ≤ p.1 ∧ p.1 ≤ b.independentHigh.1 ∧
  b.independentLow.2 ≤ p.2 ∧ p.2 ≤ b.independentHigh.2

def pairedStep (round : Int → Int) (i : Interval) (b : Block) : Interval :=
  ⟨minOver (pairValue round i.lo) b.first b.rest,
   maxOver (pairValue round i.hi) b.first b.rest⟩

def independentStep (round : Int → Int) (i : Interval) (b : Block) : Interval :=
  ⟨pairValue round i.lo b.independentLow, pairValue round i.hi b.independentHigh⟩

theorem paired_step_encloses (round : Int → Int) (mono : MonotoneRound round)
    (i : Interval) (x : Int) (b : Block) (p : Pair)
    (inside : Contains i x) (member : p ∈ Members b) :
    Contains (pairedStep round i b) (pairValue round x p) := by
  rcases inside with ⟨hl,hu⟩
  constructor
  · exact Int.le_trans (min_over_le_member _ _ _ _ member)
      (pair_value_monotone round mono _ _ _ _ _ _ hl (Int.le_refl _) (Int.le_refl _))
  · exact Int.le_trans
      (pair_value_monotone round mono _ _ _ _ _ _ hu (Int.le_refl _) (Int.le_refl _))
      (member_le_max_over _ _ _ _ member)

theorem paired_step_no_looser (round : Int → Int) (mono : MonotoneRound round)
    (old new : Interval) (b : Block) (tight : Tighter old new) (valid : Valid b) :
    Tighter (independentStep round old b) (pairedStep round new b) := by
  rcases tight with ⟨hl,hu⟩
  constructor
  · apply le_min_over
    intro p hp
    have h := valid p hp
    exact pair_value_monotone round mono _ _ _ _ _ _ hl h.1 h.2.2.1
  · apply max_over_le
    intro p hp
    have h := valid p hp
    exact pair_value_monotone round mono _ _ _ _ _ _ hu h.2.1 h.2.2.2

/-- Dropping only impossible pairs cannot widen the interval. Both member
lists are nonempty and all actually possible pairs must remain in the new one. -/
theorem filtering_pairs_no_looser (round : Int → Int) (i : Interval)
    (old new : Block) (subset : ∀ p ∈ Members new, p ∈ Members old) :
    Tighter (pairedStep round i old) (pairedStep round i new) := by
  constructor
  · apply le_min_over
    intro p hp
    exact min_over_le_member _ _ _ _ (subset p hp)
  · apply max_over_le
    intro p hp
    exact member_le_max_over _ _ _ _ (subset p hp)

/-- Shrinking a query can remove feasible pairs and tighten its incoming
interval. Both changes preserve numeric containment order. -/
theorem paired_step_tightening (round : Int → Int) (mono : MonotoneRound round)
    (oldI newI : Interval) (oldB newB : Block) (tight : Tighter oldI newI)
    (subset : ∀ p ∈ Members newB, p ∈ Members oldB) :
    Tighter (pairedStep round oldI oldB) (pairedStep round newI newB) := by
  constructor
  · apply le_min_over
    intro p hp
    exact Int.le_trans (min_over_le_member _ _ _ _ (subset p hp))
      (pair_value_monotone round mono _ _ _ _ _ _ tight.1 (Int.le_refl _) (Int.le_refl _))
  · apply max_over_le
    intro p hp
    exact Int.le_trans
      (pair_value_monotone round mono _ _ _ _ _ _ tight.2 (Int.le_refl _) (Int.le_refl _))
      (member_le_max_over _ _ _ _ (subset p hp))
def pairedReduce (round : Int → Int) : Interval → List Block → Interval
  | i, [] => i
  | i, b :: bs => pairedReduce round (pairedStep round i b) bs

def independentReduce (round : Int → Int) : Interval → List Block → Interval
  | i, [] => i
  | i, b :: bs => independentReduce round (independentStep round i b) bs

def actualReduce (round : Int → Int) : Int → List Pair → Int
  | x, [] => x
  | x, p :: ps => actualReduce round (pairValue round x p) ps

def Selections : List Block → List Pair → Prop
  | [], [] => True
  | b :: bs, p :: ps => p ∈ Members b ∧ Selections bs ps
  | _, _ => False

theorem paired_reduction_encloses (round : Int → Int) (mono : MonotoneRound round)
    (bs : List Block) : ∀ ps i x, Selections bs ps → Contains i x →
    Contains (pairedReduce round i bs) (actualReduce round x ps) := by
  induction bs with
  | nil =>
    intro ps i x sel hx
    cases ps with
    | nil => exact hx
    | cons p ps => exact False.elim sel
  | cons b bs ih =>
    intro ps i x sel hx
    cases ps with
    | nil => exact False.elim sel
    | cons p ps =>
      exact ih ps (pairedStep round i b) (pairValue round x p) sel.2
        (paired_step_encloses round mono i x b p hx sel.1)

theorem paired_reduction_no_looser (round : Int → Int) (mono : MonotoneRound round)
    (bs : List Block) : ∀ old new, Tighter old new → (∀ b ∈ bs, Valid b) →
    Tighter (independentReduce round old bs) (pairedReduce round new bs) := by
  induction bs with
  | nil => intro old new h _; exact h
  | cons b bs ih =>
    intro old new h valid
    exact ih _ _ (paired_step_no_looser round mono old new b h (valid b (by simp)))
      (by intro q hq; exact valid q (by simp [hq]))

def Refinements : List Block → List Block → Prop
  | [], [] => True
  | old :: olds, new :: news => (∀ p ∈ Members new, p ∈ Members old) ∧ Refinements olds news
  | _, _ => False

theorem paired_reduction_tightening (round : Int → Int) (mono : MonotoneRound round)
    (olds : List Block) : ∀ news oldI newI,
    Refinements olds news → Tighter oldI newI →
    Tighter (pairedReduce round oldI olds) (pairedReduce round newI news) := by
  induction olds with
  | nil =>
    intro news oldI newI refs tight
    cases news with
    | nil => exact tight
    | cons n ns => exact False.elim refs
  | cons o os ih =>
    intro news oldI newI refs tight
    cases news with
    | nil => exact False.elim refs
    | cons n ns =>
      exact ih ns _ _ refs.2 (paired_step_tightening round mono oldI newI o n tight refs.1)
/-- Flattening pairs leaves the exact ordered operand sequence unchanged. -/
theorem pair_fold_is_original_order (round : Int → Int) (ps : List Pair) :
    ∀ x, actualReduce round x ps =
      wordFold (fun a b => round (a+b)) x (ps.flatMap fun p => [p.1,p.2]) := by
  induction ps with
  | nil => intro x; rfl
  | cons p ps ih => intro x; exact ih (pairValue round x p)

/-- The numeric proof is compatible with the separately proved nearest-rounder
contract; it does not prove that CUDA implements this rounder. -/
theorem nearest_paired_reduction_encloses (representable : Int → Prop) (round : Int → Int)
    (output : ∀ x, representable (round x))
    (nearest : ∀ x v, representable v → distance x (round x) ≤ distance x v)
    (bs : List Block) (ps : List Pair) (i : Interval) (x : Int)
    (sel : Selections bs ps) (inside : Contains i x) :
    Contains (pairedReduce round i bs) (actualReduce round x ps) :=
  paired_reduction_encloses round (nearest_rounding_monotone representable round output nearest)
    bs ps i x sel inside

end ConverterPairedBounds
