import Std

/-!
Conditional physical-size and storage bounds. Volumes are cardinalities of
actual disjoint finite feasible sets, not overlapping outer-box volumes.
This file does not establish the C++/CUDA-to-mathematics refinement, source
class authority, a numerical Forest bound, or available filesystem capacity.
-/
namespace ConverterSize

def weight (v : Nat) : Nat := 2 * v - 1

def potential : List Nat → Nat
  | [] => 0
  | v :: vs => weight v + potential vs

def budget (committed : Nat) (frontier : List Nat) : Nat :=
  committed + potential frontier

theorem potential_append (a b : List Nat) :
    potential (a ++ b) = potential a + potential b := by
  induction a with
  | nil => simp [potential]
  | cons v vs ih => simp [potential, ih, Nat.add_assoc]

theorem positive_weight (v : Nat) (hv : 0 < v) : 1 ≤ weight v := by
  unfold weight
  omega

theorem split_weight (l r : Nat) (hl : 0 < l) (hr : 0 < r) :
    weight l + weight r + 1 = weight (l + r) := by
  unfold weight
  omega

theorem split_budget (c : Nat) (pre post : List Nat) (l r : Nat)
    (hl : 0 < l) (hr : 0 < r) :
    budget (c + 1) (pre ++ l :: r :: post) =
      budget c (pre ++ (l + r) :: post) := by
  have hw := split_weight l r hl hr
  simp only [budget, potential_append, potential]
  omega

theorem leaf_budget_drop (c v : Nat) (pre post : List Nat) (hv : 0 < v) :
    budget (c + 1) (pre ++ post) + 2 * (v - 1) =
      budget c (pre ++ v :: post) := by
  simp only [budget, potential_append, potential, weight]
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
    (r : Run c f d g k) : k ≤ potential f := by
  have hm := r.budget_monotone
  have hc := r.committed_count
  unfold budget at hm
  omega

theorem Run.completed_node_bound {c d k : Nat} {f : List Nat}
    (r : Run c f d [] k) : d ≤ budget c f := by
  simpa [budget, potential] using r.budget_monotone

theorem initial_grid_node_bound {m d k : Nat}
    (r : Run 0 [m] d [] k) : d ≤ 2 * m - 1 := by
  simpa [budget, potential, weight] using r.completed_node_bound

-- Exact format extents: CLSDLG01 plus CLSSEA01 and direct 96-byte task table.
def finalBytes (n t p : Nat) : Nat := 384 + 168*n + 24*t + 184*p
-- Conservative reopening peak includes sealed index plus temporary verification indices.
def readerPeakBytes (n t p : Nat) : Nat := 384 + 192*n + 32*t + 192*p

theorem final_bytes_expansion (n t p : Nat) :
    finalBytes n t p =
      (128 + 64*n + 16*t + 176*p) + (256 + 8*(n+t+p)) + 96*n := by
  unfold finalBytes
  omega

theorem reader_peak_expansion (n t p : Nat) :
    readerPeakBytes n t p = finalBytes n t p + 24*n + 8*t + 8*p := by
  unfold readerPeakBytes finalBytes
  omega

theorem final_below_reader_peak (n t p : Nat) :
    finalBytes n t p ≤ readerPeakBytes n t p := by
  unfold finalBytes readerPeakBytes
  omega

theorem reader_peak_monotone {n t p bn bt bp : Nat}
    (hn : n ≤ bn) (ht : t ≤ bt) (hp : p ≤ bp) :
    readerPeakBytes n t p ≤ readerPeakBytes bn bt bp := by
  unfold readerPeakBytes
  omega

theorem conditional_capacity {n t p bn bt bp overhead reserve capacity : Nat}
    (hn : n ≤ bn) (ht : t ≤ bt) (hp : p ≤ bp)
    (fits : readerPeakBytes bn bt bp + overhead + reserve ≤ capacity) :
    readerPeakBytes n t p + overhead + reserve ≤ capacity := by
  have hm := reader_peak_monotone hn ht hp
  omega

theorem axis_conservative_peak {n p : Nat} (hp : p ≤ n) :
    readerPeakBytes n 0 p ≤ 384 + 384*n := by
  unfold readerPeakBytes
  omega

theorem completed_capacity {c d k bp overhead reserve capacity : Nat}
    {f : List Nat} (r : Run c f d [] k)
    (pageBound : bp ≤ budget c f)
    (fits : 384 + 384 * budget c f + overhead + reserve ≤ capacity) :
    readerPeakBytes d 0 bp + overhead + reserve ≤ capacity := by
  have hn := r.completed_node_bound
  unfold readerPeakBytes
  omega

#print axioms potential_append
#print axioms positive_weight
#print axioms split_weight
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
#print axioms final_bytes_expansion
#print axioms reader_peak_expansion
#print axioms final_below_reader_peak
#print axioms reader_peak_monotone
#print axioms conditional_capacity
#print axioms axis_conservative_peak
#print axioms completed_capacity
end ConverterSize
