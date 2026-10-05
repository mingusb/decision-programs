import SharedDecisionDAG
import Init.Data.Rat.Lemmas

/-! Exact algebra for seven-vector equations of a shared decision DAG.
Rational gate values model mathematical coefficients, not rounded runtime
logistic computations. Hard gates preserve classes; soft gates preserve only
convex-mixture properties. No soft argmax or accuracy theorem is claimed. -/
namespace ConverterDecisionEquations
open ConverterGuarantees ConverterSharedDAG
variable {P X : Type}

abbrev Vector := Fin 7 → Rat

def basis (label : Fin 7) : Vector := fun k => if k = label then 1 else 0

def blend (gate : Rat) (left right : Vector) : Vector :=
  fun k => gate * left k + (1-gate) * right k

def totalOn (keys : List (Fin 7)) (vector : Vector) : Rat :=
  (keys.map vector).foldr (· + ·) 0

def total (vector : Vector) : Rat := totalOn (List.finRange 7) vector

def Simplex (vector : Vector) : Prop :=
  (∀ k, 0 ≤ vector k) ∧ total vector = 1

theorem blend_one (left right : Vector) : blend 1 left right = left := by
  funext k
  simp [blend, Rat.sub_self, Rat.add_zero]

theorem blend_zero (left right : Vector) : blend 0 left right = right := by
  funext k
  simp [blend, Rat.sub_eq_add_neg, Rat.zero_add, Rat.add_zero]

theorem totalOn_blend (keys : List (Fin 7)) (gate : Rat) (left right : Vector) :
    totalOn keys (blend gate left right) =
      gate * totalOn keys left + (1-gate) * totalOn keys right := by
  induction keys with
  | nil => simp [totalOn, Rat.zero_add]
  | cons k ks ih =>
    simp only [totalOn, List.map_cons, List.foldr_cons] at *
    rw [ih]
    simp only [blend, Rat.mul_add]
    ac_rfl

theorem total_blend (gate : Rat) (left right : Vector) :
    total (blend gate left right) = gate * total left + (1-gate) * total right :=
  totalOn_blend _ _ _ _

theorem basis_simplex (label : Fin 7) : Simplex (basis label) := by
  constructor
  · intro k
    by_cases h : k = label
    · simp only [basis, h, ↓reduceIte]; decide
    · simp only [basis, h, ↓reduceIte]; exact Rat.le_refl
  · obtain ⟨n,hn⟩ := label
    have cases : n=0 ∨ n=1 ∨ n=2 ∨ n=3 ∨ n=4 ∨ n=5 ∨ n=6 := by omega
    rcases cases with h | h | h | h | h | h | h
    all_goals subst n
    all_goals simp [total, totalOn, basis, List.finRange, Rat.zero_add, Rat.add_zero]

theorem blend_simplex (gate : Rat) (left right : Vector)
    (lower : 0 ≤ gate) (upper : gate ≤ 1)
    (hl : Simplex left) (hr : Simplex right) : Simplex (blend gate left right) := by
  have complement : 0 ≤ 1-gate := (Rat.le_iff_sub_nonneg gate 1).mp upper
  constructor
  · intro k
    exact Rat.add_nonneg (Rat.mul_nonneg lower (hl.1 k)) (Rat.mul_nonneg complement (hr.1 k))
  · rw [total_blend, hl.2, hr.2, Rat.mul_one, Rat.mul_one, Rat.add_comm]
    exact Rat.sub_add_cancel

def equations (graph : DAG P (Fin 7)) (gate : P → X → Rat)
    (root : Fin graph.size) (x : X) : Vector :=
  match graph.node root with
  | .leaf label => basis label
  | .branch q left right => blend (gate q x)
      (equations graph gate (child root left) x)
      (equations graph gate (child root right) x)
termination_by root.val
decreasing_by all_goals exact child_id_decreases _ _

def hardGate (truth : P → X → Bool) (q : P) (x : X) : Rat :=
  if truth q x then 1 else 0

/-- Induction follows actual strictly decreasing child IDs. A shared node has
    one vector equation regardless of the number of incoming references. -/
theorem hard_equations_equal_class (graph : DAG P (Fin 7)) (truth : P → X → Bool) :
    ∀ root x, equations graph (hardGate truth) root x = basis (route graph truth root x) := by
  intro root
  induction h : root.val using Nat.strongRecOn generalizing root with
  | ind n ih =>
    intro x
    rw [equations, route]
    cases words : graph.node root with
    | leaf label => rfl
    | branch q left right =>
      dsimp only
      unfold hardGate
      cases test : truth q x
      · simp only [Bool.false_eq_true, ↓reduceIte, blend_zero]
        exact ih (child root right).val (by have := right.isLt; simp only [child]; omega)
          (child root right) rfl x
      · simp only [↓reduceIte, blend_one]
        exact ih (child root left).val (by have := left.isLt; simp only [child]; omega)
          (child root left) rfl x

/-- Any mathematically bounded scalar gates give a nonnegative unit-mass
    class mixture. This conclusion does not imply the same winning class. -/
theorem bounded_gates_give_simplex (graph : DAG P (Fin 7)) (gate : P → X → Rat)
    (bounded : ∀ q x, 0 ≤ gate q x ∧ gate q x ≤ 1) :
    ∀ root x, Simplex (equations graph gate root x) := by
  intro root
  induction h : root.val using Nat.strongRecOn generalizing root with
  | ind n ih =>
    intro x
    rw [equations]
    cases words : graph.node root with
    | leaf label => exact basis_simplex label
    | branch q left right =>
      apply blend_simplex _ _ _ (bounded q x).1 (bounded q x).2
      · exact ih (child root left).val (by have := left.isLt; simp only [child]; omega)
          (child root left) rfl x
      · exact ih (child root right).val (by have := right.isLt; simp only [child]; omega)
          (child root right) rfl x

end ConverterDecisionEquations
