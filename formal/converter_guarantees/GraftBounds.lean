import SizeBounds

/-!
Fully certified subtree replacement inside one pending finite region.
Unlike incomplete oblique children, a completed graft leaves no overlapping
outer boxes on the frontier. Actual leaf disjointness, exact source class
certification, and rank-word predicate refinement are separate obligations.
-/
namespace ConverterGraft
open ConverterSize
open ConverterGuarantees (mass frontierPotential potential_append leaf_mass_positive)

def cells : List Nat → Nat
  | [] => 0
  | v :: rest => v + cells rest

theorem nonempty_leaf_count_bounded_by_cells (parts : List Nat)
    (positive : ∀ v ∈ parts, 0 < v) : parts.length ≤ cells parts := by
  induction parts with
  | nil => simp [cells]
  | cons v rest ih =>
      have hv := positive v (by simp)
      have hr := ih (by intro x hx; exact positive x (by simp [hx]))
      simp only [List.length_cons, cells]
      omega

/-- A nonempty full binary graft with k leaves occupies 2*k-1 nodes. -/
def physical (k : Nat) : Nat := 2*k-1

theorem completed_graft_exact_saving (c v k : Nat) (pre post : List Nat)
    (hk : 0 < k) (hkv : k ≤ v) :
    budget (c + physical k) (pre ++ post) + 2*(v-k) =
      budget c (pre ++ v :: post) := by
  simp only [budget, potential_append, frontierPotential, mass, physical]
  omega

theorem completed_graft_budget_monotone (c v k : Nat) (pre post : List Nat)
    (hk : 0 < k) (hkv : k ≤ v) :
    budget (c + physical k) (pre ++ post) ≤ budget c (pre ++ v :: post) := by
  have h := completed_graft_exact_saving c v k pre post hk hkv
  omega

theorem completed_graft_commits_bounded (c v k : Nat) (pre post : List Nat)
    (hk : 0 < k) (hkv : k ≤ v) :
    physical k + frontierPotential (pre ++ post) ≤ frontierPotential (pre ++ v :: post) := by
  have h := completed_graft_budget_monotone c v k pre post hk hkv
  unfold budget at h
  omega

theorem completed_graft_decreases_pending (v : Nat) (pre post : List Nat)
    (hv : 0 < v) : frontierPotential (pre ++ post) < frontierPotential (pre ++ v :: post) := by
  have hm := leaf_mass_positive v hv
  simp only [potential_append, frontierPotential]
  omega

theorem two_leaf_stump_exact_saving (c v : Nat) (pre post : List Nat)
    (hv : 2 ≤ v) :
    budget (c+3) (pre ++ post) + 2*(v-2) = budget c (pre ++ v :: post) := by
  have h := completed_graft_exact_saving c v 2 pre post (by omega) hv
  simpa [physical] using h

/-- A complete exact partition into positive-cardinality leaves supplies the
    size premise. No sample count or outer-box overlap sum substitutes for it. -/
theorem partition_graft_budget (c v : Nat) (pre post parts : List Nat)
    (positive : ∀ n ∈ parts, 0<n) (nonempty : 0<parts.length)
    (partition : cells parts = v) :
    budget (c + physical parts.length) (pre ++ post) + 2*(v-parts.length) =
      budget c (pre ++ v :: post) := by
  have h := nonempty_leaf_count_bounded_by_cells parts positive
  apply completed_graft_exact_saving c v parts.length pre post nonempty
  omega

#print axioms nonempty_leaf_count_bounded_by_cells
#print axioms completed_graft_exact_saving
#print axioms completed_graft_budget_monotone
#print axioms completed_graft_commits_bounded
#print axioms completed_graft_decreases_pending
#print axioms two_leaf_stump_exact_saving
#print axioms partition_graft_budget
end ConverterGraft
