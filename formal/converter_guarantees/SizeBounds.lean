import CorrectnessCompletion

/-!
Finite-region node-count bounds. Volumes are cardinalities of actual disjoint
feasible sets, not overlapping outer-box volumes. Source class authority and
C++/CUDA implementation refinement remain separate obligations.
-/
namespace ConverterSize
open ConverterGuarantees (mass frontierPotential potential_append split_mass_identity)

def budget (committed : Nat) (frontier : List Nat) : Nat :=
  committed + frontierPotential frontier

theorem split_budget (c : Nat) (pre post : List Nat) (l r : Nat)
    (hl : 0 < l) (hr : 0 < r) :
    budget (c + 1) (pre ++ l :: r :: post) =
      budget c (pre ++ (l + r) :: post) := by
  have hw := split_mass_identity l r hl hr
  simp only [budget, potential_append, frontierPotential]
  omega

theorem leaf_budget_drop (c v : Nat) (pre post : List Nat) (hv : 0 < v) :
    budget (c + 1) (pre ++ post) + 2 * (v - 1) =
      budget c (pre ++ v :: post) := by
  simp only [budget, potential_append, frontierPotential, mass]
  omega

theorem leaf_budget_monotone (c v : Nat) (pre post : List Nat) (hv : 0 < v) :
    budget (c + 1) (pre ++ post) ≤ budget c (pre ++ v :: post) := by
  have h := leaf_budget_drop c v pre post hv
  omega

inductive Step : Nat → List Nat → Nat → List Nat → Prop where
  | leaf (c v : Nat) (pre post : List Nat) (hv : 0 < v) :
      Step c (pre ++ v :: post) (c + 1) (pre ++ post)
  | split (c l r : Nat) (pre post : List Nat) (hl : 0 < l) (hr : 0 < r) :
      Step c (pre ++ (l + r) :: post) (c + 1) (pre ++ l :: r :: post)

theorem Step.budget_monotone {c d : Nat} {f g : List Nat}
    (s : Step c f d g) : budget d g ≤ budget c f := by
  cases s with
  | leaf v pre post hv => exact leaf_budget_monotone c v pre post hv
  | split l r pre post hl hr => exact Nat.le_of_eq (split_budget c pre post l r hl hr)

theorem Step.committed_increment {c d : Nat} {f g : List Nat}
    (s : Step c f d g) : d = c + 1 := by
  cases s <;> rfl

inductive Run : Nat → List Nat → Nat → List Nat → Nat → Prop where
  | nil (c : Nat) (f : List Nat) : Run c f c f 0
  | cons {c d e k : Nat} {f g h : List Nat} :
      Step c f d g → Run d g e h k → Run c f e h (k + 1)

theorem Run.budget_monotone {c d k : Nat} {f g : List Nat}
    (r : Run c f d g k) : budget d g ≤ budget c f := by
  induction r with
  | nil => exact Nat.le_refl _
  | cons s r ih => exact Nat.le_trans ih s.budget_monotone

theorem Run.committed_count {c d k : Nat} {f g : List Nat}
    (r : Run c f d g k) : d = c + k := by
  induction r with
  | nil => omega
  | cons s r ih => have h := s.committed_increment; omega

theorem Run.remaining_commits_bound {c d k : Nat} {f g : List Nat}
    (r : Run c f d g k) : k ≤ frontierPotential f := by
  have hm := r.budget_monotone
  have hc := r.committed_count
  unfold budget at hm
  omega

theorem Run.completed_node_bound {c d k : Nat} {f : List Nat}
    (r : Run c f d [] k) : d ≤ budget c f := by
  simpa [budget, frontierPotential] using r.budget_monotone

theorem initial_grid_node_bound {m d k : Nat}
    (r : Run 0 [m] d [] k) : d ≤ 2 * m - 1 := by
  simpa [budget, frontierPotential, mass] using r.completed_node_bound

#print axioms split_budget
#print axioms leaf_budget_drop
#print axioms leaf_budget_monotone
#print axioms Step.budget_monotone
#print axioms Step.committed_increment
#print axioms Run.budget_monotone
#print axioms Run.committed_count
#print axioms Run.remaining_commits_bound
#print axioms Run.completed_node_bound
#print axioms initial_grid_node_bound
end ConverterSize
