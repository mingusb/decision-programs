import CorrectnessCompletion
import DecisionEquations

/-! Exact hard-domain reductions. Integer numeric coordinates are preprocessed
ranks; categorical coordinates are exactly one valid member of each group.
Scope laws concern only their stated box, never all incoming DAG paths by
implication. No sampling, approximate comparison or floating reassociation. -/
namespace ConverterHardAxisRewrites
open ConverterGuarantees
variable {L : Type}

structure Cell where
  numeric : Fin 10 → Int
  wilderness : Fin 4
  soil : Fin 40

inductive Question where
  | numeric : Fin 10 → Int → Question
  | wilderness : Fin 4 → Question
  | soil : Fin 40 → Question

def truth : Question → Cell → Bool
  | .numeric f cut, x => decide (x.numeric f < cut)
  | .wilderness category, x => decide (x.wilderness ≠ category)
  | .soil category, x => decide (x.soil ≠ category)

structure Box where
  lo : Fin 10 → Int
  hi : Fin 10 → Int
  wilderness : Fin 4 → Bool
  soil : Fin 40 → Bool

def Contains (box : Box) (x : Cell) : Prop :=
  (∀ f, box.lo f ≤ x.numeric f ∧ x.numeric f ≤ box.hi f) ∧
  box.wilderness x.wilderness = true ∧ box.soil x.soil = true

/-- Global numeric implication: the left child repeats a weaker test. -/
theorem left_weaker_test (f : Fin 10) (a b : Int) (ordered : a ≤ b)
    (left middle right : Tree Question L) (x : Cell) :
    eval truth (.branch (.numeric f a) (.branch (.numeric f b) left middle) right) x =
    eval truth (.branch (.numeric f a) left right) x := by
  by_cases ha : x.numeric f < a
  · have hb : x.numeric f < b := by omega
    simp [eval, truth, ha, hb]
  · simp [eval, truth, ha]

/-- Global numeric implication: the right child repeats a stronger test. -/
theorem right_stronger_test (f : Fin 10) (a b : Int) (ordered : b ≤ a)
    (left middle right : Tree Question L) (x : Cell) :
    eval truth (.branch (.numeric f a) left (.branch (.numeric f b) middle right)) x =
    eval truth (.branch (.numeric f a) left right) x := by
  by_cases ha : x.numeric f < a
  · simp [eval, truth, ha]
  · have hb : ¬ x.numeric f < b := by omega
    simp [eval, truth, ha, hb]

/-- Threshold absorption retains b and removes the enclosing a question. -/
theorem absorb_smaller_cut (f : Fin 10) (a b : Int) (ordered : b ≤ a)
    (left right : Tree Question L) (x : Cell) :
    eval truth (.branch (.numeric f a) (.branch (.numeric f b) left right) right) x =
    eval truth (.branch (.numeric f b) left right) x := by
  by_cases hb : x.numeric f < b
  · have ha : x.numeric f < a := by omega
    simp [eval, truth, ha, hb]
  · simp [eval, truth, hb]

/-- Complementary threshold absorption retains the larger b question. -/
theorem absorb_larger_cut (f : Fin 10) (a b : Int) (ordered : a ≤ b)
    (left right : Tree Question L) (x : Cell) :
    eval truth (.branch (.numeric f a) left (.branch (.numeric f b) left right)) x =
    eval truth (.branch (.numeric f b) left right) x := by
  by_cases ha : x.numeric f < a
  · have hb : x.numeric f < b := by omega
    simp [eval, truth, ha, hb]
  · simp [eval, truth, ha]

theorem repeated_left_test (q : Question) (left middle right : Tree Question L) (x : Cell) :
    eval truth (.branch q (.branch q left middle) right) x =
    eval truth (.branch q left right) x := by
  cases h : truth q x <;> simp [eval, h]

theorem repeated_right_test (q : Question) (left middle right : Tree Question L) (x : Cell) :
    eval truth (.branch q left (.branch q middle right)) x =
    eval truth (.branch q left right) x := by
  cases h : truth q x <;> simp [eval, h]

/-- Complete rank-box scope, including fractional raw inputs sharing a rank. -/
theorem box_entirely_left (box : Box) (f : Fin 10) (cut : Int)
    (upper : box.hi f < cut) (left right : Tree Question L)
    (x : Cell) (inside : Contains box x) :
    eval truth (.branch (.numeric f cut) left right) x = eval truth left x := by
  have h : x.numeric f < cut := by have := (inside.1 f).2; omega
  simp [eval, truth, h]

theorem box_entirely_right (box : Box) (f : Fin 10) (cut : Int)
    (lower : cut ≤ box.lo f) (left right : Tree Question L)
    (x : Cell) (inside : Contains box x) :
    eval truth (.branch (.numeric f cut) left right) x = eval truth right x := by
  have h : ¬ x.numeric f < cut := by have := (inside.1 f).1; omega
  simp [eval, truth, h]

theorem wilderness_absent (box : Box) (category : Fin 4)
    (absent : box.wilderness category = false) (left right : Tree Question L)
    (x : Cell) (inside : Contains box x) :
    eval truth (.branch (.wilderness category) left right) x = eval truth left x := by
  have h : x.wilderness ≠ category := by
    intro same
    have accepted := inside.2.1
    rw [same, absent] at accepted
    contradiction
  simp [eval, truth, h]

theorem wilderness_only (box : Box) (category : Fin 4)
    (only : ∀ k, box.wilderness k = true → k = category) (left right : Tree Question L)
    (x : Cell) (inside : Contains box x) :
    eval truth (.branch (.wilderness category) left right) x = eval truth right x := by
  have h := only x.wilderness inside.2.1
  simp [eval, truth, h]

theorem soil_absent (box : Box) (category : Fin 40)
    (absent : box.soil category = false) (left right : Tree Question L)
    (x : Cell) (inside : Contains box x) :
    eval truth (.branch (.soil category) left right) x = eval truth left x := by
  have h : x.soil ≠ category := by
    intro same
    have accepted := inside.2.2
    rw [same, absent] at accepted
    contradiction
  simp [eval, truth, h]

theorem soil_only (box : Box) (category : Fin 40)
    (only : ∀ k, box.soil k = true → k = category) (left right : Tree Question L)
    (x : Cell) (inside : Contains box x) :
    eval truth (.branch (.soil category) left right) x = eval truth right x := by
  have h := only x.soil inside.2.2
  simp [eval, truth, h]

/-- A scoped replacement may be used through this exact occurrence. This does
    not grant permission to mutate a shared node reached outside the scope. -/
theorem guarded_occurrence (guard : Cell → Bool) (box : Box)
    (scope : ∀ x, guard x = true → Contains box x)
    (old replacement otherwise : Tree Question L)
    (equivalent : ∀ x, Contains box x → eval truth old x = eval truth replacement x)
    (x : Cell) :
    (if guard x then eval truth old x else eval truth otherwise x) =
    (if guard x then eval truth replacement x else eval truth otherwise x) := by
  cases h : guard x
  · rfl
  · simp only [↓reduceIte]
    exact equivalent x (scope x h)

/-- MDL admission uses the actual chosen codec's final bytes, not node counts. -/
theorem strict_encoded_improvement (oldBytes newBytes budget : Nat)
    (fits : oldBytes ≤ budget) (smaller : newBytes < oldBytes) :
    newBytes < oldBytes ∧ newBytes ≤ budget := by omega

end ConverterHardAxisRewrites

namespace ConverterSoftRewriteAlgebra
open ConverterDecisionEquations

/-- Algebraic equal-child identity over exact rationals. Rounded evaluation
    is a distinct semantic mode and is NOT covered by this theorem. -/
theorem equal_children (g : Rat) (v : Vector) : blend g v v = v := by
  funext k
  change g * v k + (1-g) * v k = v k
  rw [← Rat.add_mul]
  have h : g+(1-g) = 1 := by rw [Rat.add_comm]; exact Rat.sub_add_cancel
  rw [h, Rat.one_mul]

theorem double_complement (g : Rat) : 1-(1-g) = g := by
  apply Rat.add_right_cancel (1-g)
  rw [Rat.sub_add_cancel, Rat.add_comm g]
  exact Rat.sub_add_cancel.symm

/-- Complementing a scalar gate requires swapping both children. -/
theorem complemented_gate (g : Rat) (left right : Vector) :
    blend (1-g) right left = blend g left right := by
  funext k
  simp only [blend, double_complement]
  exact Rat.add_comm _ _

end ConverterSoftRewriteAlgebra
