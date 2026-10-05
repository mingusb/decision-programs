import Std

/- Generic semantic lemmas for literal cofactoring. Atoms can encode complete
   feature/threshold/missing-direction words. Values are uninterpreted: these
   theorems do not prove native floating-point accumulation or CUDA code. -/
namespace DynamicAtomRestriction

inductive Expr (Atom Value : Type) where
  | leaf : Value → Expr Atom Value
  | branch : Atom → Expr Atom Value → Expr Atom Value → Expr Atom Value

def eval (x : Atom → Bool) : Expr Atom Value → Value
  | .leaf v => v
  | .branch a l r => if x a then eval x l else eval x r

def restrict [DecidableEq Atom] (a : Atom) (b : Bool) : Expr Atom Value → Expr Atom Value
  | .leaf v => .leaf v
  | .branch q l r =>
    if q = a then
      if b then restrict a b l else restrict a b r
    else .branch q (restrict a b l) (restrict a b r)

theorem restrict_exact [DecidableEq Atom] (a : Atom) (b : Bool)
    (x : Atom → Bool) (hx : x a = b) (t : Expr Atom Value) :
    eval x (restrict a b t) = eval x t := by
  induction t with
  | leaf v => rfl
  | branch q l r ihl ihr =>
    by_cases h : q = a
    · subst q
      cases b <;> simp [restrict, eval, hx, ihl, ihr]
    · simp [restrict, eval, h, ihl, ihr]

def occurrences [DecidableEq Atom] (a : Atom) : Expr Atom Value → Nat
  | .leaf _ => 0
  | .branch q l r => (if q = a then 1 else 0) + occurrences a l + occurrences a r

theorem selected_atom_removed [DecidableEq Atom] (a : Atom) (b : Bool)
    (t : Expr Atom Value) : occurrences a (restrict a b t) = 0 := by
  induction t with
  | leaf v => rfl
  | branch q l r ihl ihr =>
    by_cases h : q = a
    · cases b <;> simp [restrict, h, ihl, ihr]
    · simp [restrict, occurrences, h, ihl, ihr]

def nodes : Expr Atom Value → Nat
  | .leaf _ => 1
  | .branch _ l r => 1 + nodes l + nodes r

theorem restrict_nodes_le [DecidableEq Atom] (a : Atom) (b : Bool)
    (t : Expr Atom Value) : nodes (restrict a b t) ≤ nodes t := by
  induction t with
  | leaf v => exact Nat.le_refl _
  | branch q l r ihl ihr =>
    by_cases h : q = a
    · cases b <;> simp only [restrict, if_pos h, Bool.false_eq_true, ↓reduceIte, nodes] <;> omega
    · simp only [restrict, if_neg h, nodes]
      omega

theorem root_restriction_strict [DecidableEq Atom] (a : Atom) (b : Bool)
    (l r : Expr Atom Value) :
    nodes (restrict a b (.branch a l r)) < nodes (.branch a l r) := by
  have hl := restrict_nodes_le a b l
  have hr := restrict_nodes_le a b r
  cases b <;> simp only [restrict, if_pos rfl, Bool.false_eq_true, ↓reduceIte, nodes] <;> omega

def potential (xs : List (Expr Atom Value)) : Nat := (xs.map nodes).sum

theorem potential_restriction_le [DecidableEq Atom] (a : Atom) (b : Bool)
    (xs : List (Expr Atom Value)) :
    potential (xs.map (restrict a b)) ≤ potential xs := by
  induction xs with
  | nil => simp [potential]
  | cons t ts ih =>
    have h := restrict_nodes_le a b t
    simp only [potential, List.map_cons, List.sum_cons] at *
    omega

theorem potential_restriction_strict [DecidableEq Atom] (a : Atom) (b : Bool)
    (xs : List (Expr Atom Value)) (l r : Expr Atom Value)
    (member : Expr.branch a l r ∈ xs) :
    potential (xs.map (restrict a b)) < potential xs := by
  induction xs with
  | nil => simp at member
  | cons t ts ih =>
    rcases List.mem_cons.mp member with eq | tail
    · subst t
      have h := root_restriction_strict a b l r
      have rest := potential_restriction_le a b ts
      simp only [potential, List.map_cons, List.sum_cons] at *
      omega
    · have h := restrict_nodes_le a b t
      have rest := ih tail
      simp only [potential, List.map_cons, List.sum_cons] at *
      omega

theorem tuple_restriction_exact [DecidableEq Atom] (a : Atom) (b : Bool)
    (x : Atom → Bool) (hx : x a = b) (xs : List (Expr Atom Value)) :
    (xs.map (restrict a b)).map (eval x) = xs.map (eval x) := by
  induction xs with
  | nil => rfl
  | cons t ts ih => simp [restrict_exact a b x hx t, ih]

theorem any_deterministic_winner_preserved [DecidableEq Atom] (a : Atom) (b : Bool)
    (x : Atom → Bool) (hx : x a = b) (xs : List (Expr Atom Value))
    (winner : List Value → Class) :
    winner ((xs.map (restrict a b)).map (eval x)) = winner (xs.map (eval x)) := by
  rw [tuple_restriction_exact a b x hx xs]

#print axioms restrict_exact
#print axioms selected_atom_removed
#print axioms restrict_nodes_le
#print axioms root_restriction_strict
#print axioms potential_restriction_le
#print axioms potential_restriction_strict
#print axioms tuple_restriction_exact
#print axioms any_deterministic_winner_preserved
end DynamicAtomRestriction
