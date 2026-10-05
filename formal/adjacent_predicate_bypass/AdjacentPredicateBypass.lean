import DynamicAtomRestriction

/- Generic parent-edge rewrite laws. Atoms represent complete predicate words;
   no numerical threshold or missing-value equivalence is assumed. -/
namespace AdjacentPredicateBypass
open DynamicAtomRestriction

def bypass [DecidableEq Atom] (parent : Atom) (truth : Bool) : Expr Atom Value → Expr Atom Value
  | .leaf value => .leaf value
  | .branch atom left right =>
    if atom = parent then
      if truth then bypass parent truth left else bypass parent truth right
    else .branch atom left right

theorem bypass_exact [DecidableEq Atom] (parent : Atom) (truth : Bool)
    (x : Atom → Bool) (known : x parent = truth) (tree : Expr Atom Value) :
    eval x (bypass parent truth tree) = eval x tree := by
  induction tree with
  | leaf value => rfl
  | branch atom left right hl hr =>
    by_cases same : atom = parent
    · subst atom
      cases truth <;> simp [bypass, eval, known, hl, hr]
    · simp [bypass, same]

theorem bypass_size_nonincreasing [DecidableEq Atom] (parent : Atom) (truth : Bool)
    (tree : Expr Atom Value) : nodes (bypass parent truth tree) ≤ nodes tree := by
  induction tree with
  | leaf value => exact Nat.le_refl _
  | branch atom left right hl hr =>
    by_cases same : atom = parent
    · cases truth <;> simp only [bypass, same, Bool.false_eq_true, ↓reduceIte, nodes] <;> omega
    · simp [bypass, same]

def rewriteRoot [DecidableEq Atom] : Expr Atom Value → Expr Atom Value
  | .leaf value => .leaf value
  | .branch atom left right =>
    .branch atom (bypass atom true left) (bypass atom false right)

theorem parent_edge_rewrite_exact [DecidableEq Atom] (x : Atom → Bool)
    (tree : Expr Atom Value) : eval x (rewriteRoot tree) = eval x tree := by
  cases tree with
  | leaf value => rfl
  | branch atom left right =>
    cases known : x atom
    · simpa [rewriteRoot, eval, known] using bypass_exact atom false x known right
    · simpa [rewriteRoot, eval, known] using bypass_exact atom true x known left

theorem parent_edge_size_nonincreasing [DecidableEq Atom] (tree : Expr Atom Value) :
    nodes (rewriteRoot tree) ≤ nodes tree := by
  cases tree with
  | leaf value => exact Nat.le_refl _
  | branch atom left right =>
    have hl := bypass_size_nonincreasing atom true left
    have hr := bypass_size_nonincreasing atom false right
    simp only [rewriteRoot, nodes]
    omega

#print axioms bypass_exact
#print axioms bypass_size_nonincreasing
#print axioms parent_edge_rewrite_exact
#print axioms parent_edge_size_nonincreasing
end AdjacentPredicateBypass
