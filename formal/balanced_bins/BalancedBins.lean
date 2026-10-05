import Std

/- A dataset-independent balanced compiler for an already exact feature-bin
   partition. This models integer bin ranks, not FP32 threshold construction. -/
namespace BalancedBins

inductive Tree where
  | leaf : Nat → Tree
  | split : Nat → Tree → Tree → Tree
  deriving Repr

def eval (index : Nat) : Tree → Nat
  | .leaf value => value
  | .split cut left right => if index < cut then eval index left else eval index right

def depth : Tree → Nat
  | .leaf _ => 0
  | .split _ left right => 1 + max (depth left) (depth right)

def build : Nat → Nat → Nat → Tree
  | 0, base, _ => .leaf base
  | fuel + 1, base, count =>
    if count ≤ 1 then .leaf base
    else
      let left := count / 2
      .split (base + left) (build fuel base left)
        (build fuel (base + left) (count - left))

theorem depth_bounded (fuel base count : Nat) : depth (build fuel base count) ≤ fuel := by
  induction fuel generalizing base count with
  | zero => simp [build, depth]
  | succ fuel ih =>
    by_cases h : count ≤ 1
    · simp [build, h, depth]
    · have hl := ih base (count / 2)
      have hr := ih (base + count / 2) (count - count / 2)
      simp only [build, h, ↓reduceIte, depth]
      omega

theorem balanced_halves_fit (count capacity : Nat) (h : count ≤ 2 * capacity) :
    count / 2 ≤ capacity ∧ count - count / 2 ≤ capacity := by
  omega

theorem exact_bin (fuel base count index : Nat)
    (capacity : count ≤ 2 ^ fuel) (lower : base ≤ index) (upper : index < base + count) :
    eval index (build fuel base count) = index := by
  induction fuel generalizing base count with
  | zero =>
    simp only [Nat.pow_zero] at capacity
    have hi : index = base := by omega
    simp [build, eval, hi]
  | succ fuel ih =>
    by_cases tiny : count ≤ 1
    · have hi : index = base := by omega
      simp [build, tiny, eval, hi]
    · have cap : count ≤ 2 * 2 ^ fuel := by
        simpa [Nat.pow_succ, Nat.mul_comm] using capacity
      obtain ⟨lc, rc⟩ := balanced_halves_fit count (2 ^ fuel) cap
      by_cases side : index < base + count / 2
      · have selected := ih base (count / 2) lc lower side
        simpa [build, tiny, eval, side] using selected
      · have right_lower : base + count / 2 ≤ index := by omega
        have right_upper : index < (base + count / 2) + (count - count / 2) := by omega
        have selected := ih (base + count / 2) (count - count / 2) rc right_lower right_upper
        simpa [build, tiny, eval, side] using selected

theorem arbitrary_payload_exact (fuel base count index : Nat) (value : Nat → Value)
    (capacity : count ≤ 2 ^ fuel) (lower : base ≤ index) (upper : index < base + count) :
    value (eval index (build fuel base count)) = value index := by
  rw [exact_bin fuel base count index capacity lower upper]

def leaves : Tree → Nat
  | .leaf _ => 1
  | .split _ left right => leaves left + leaves right

def nodes : Tree → Nat
  | .leaf _ => 1
  | .split _ left right => 1 + nodes left + nodes right

theorem exact_leaf_count (fuel base count : Nat)
    (capacity : count ≤ 2 ^ fuel) (positive : 0 < count) :
    leaves (build fuel base count) = count := by
  induction fuel generalizing base count with
  | zero =>
    simp only [Nat.pow_zero] at capacity
    have one : count = 1 := by omega
    simp [build, leaves, one]
  | succ fuel ih =>
    by_cases tiny : count ≤ 1
    · have one : count = 1 := by omega
      simp [build, tiny, leaves, one]
    · have cap : count ≤ 2 * 2 ^ fuel := by
        simpa [Nat.pow_succ, Nat.mul_comm] using capacity
      obtain ⟨lc, rc⟩ := balanced_halves_fit count (2 ^ fuel) cap
      have lp : 0 < count / 2 := by omega
      have rp : 0 < count - count / 2 := by omega
      have left := ih base (count / 2) lc lp
      have right := ih (base + count / 2) (count - count / 2) rc rp
      simp only [build, tiny, ↓reduceIte, leaves, left, right]
      omega

theorem binary_nodes (tree : Tree) : nodes tree + 1 = 2 * leaves tree := by
  induction tree with
  | leaf value => simp [nodes, leaves]
  | split cut left right hl hr =>
    simp only [nodes, leaves]
    omega

theorem exact_node_count (fuel base count : Nat)
    (capacity : count ≤ 2 ^ fuel) (positive : 0 < count) :
    nodes (build fuel base count) + 1 = 2 * count := by
  rw [binary_nodes, exact_leaf_count fuel base count capacity positive]

#print axioms depth_bounded
#print axioms balanced_halves_fit
#print axioms exact_bin
#print axioms arbitrary_payload_exact
#print axioms exact_leaf_count
#print axioms binary_nodes
#print axioms exact_node_count
end BalancedBins
