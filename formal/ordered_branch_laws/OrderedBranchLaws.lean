import Std

/- Dataset-independent laws for ordered binary decision construction.
   A Boolean is the exact outcome of a complete predicate, including its
   missing-value route. No floating-point arithmetic identity is assumed. -/
namespace OrderedBranchLaws

def choose (p : Bool) (yes no : Value) : Value := if p then yes else no

theorem equal_children (p : Bool) (value : Value) :
    choose p value value = value := by
  cases p <;> rfl

theorem repeated_then (p : Bool) (a b c : Value) :
    choose p (choose p a b) c = choose p a c := by
  cases p <;> rfl

theorem repeated_else (p : Bool) (a b c : Value) :
    choose p a (choose p b c) = choose p a c := by
  cases p <;> rfl

theorem simultaneous_cofactors (p : Bool) (tt tf ft ff : Value) :
    choose p (choose p tt tf) (choose p ft ff) = choose p tt ff := by
  cases p <;> rfl

theorem shannon_swap (p q : Bool) (tt tf ft ff : Value) :
    choose p (choose q tt tf) (choose q ft ff) =
    choose q (choose p tt ft) (choose p tf ff) := by
  cases p <;> cases q <;> rfl

theorem branch_congruence (p : Bool) {a b c d : Value}
    (yes_equal : a = c) (no_equal : b = d) :
    choose p a b = choose p c d := by
  rw [yes_equal, no_equal]

/- Ranks below count are oriented in decreasing order along a path. An
   implementation with increasing ranks can use count-1-rank. This abstract
   premise must be checked separately on the actual emitted graph. -/
inductive DescendingPath : Nat → List Nat → Prop where
  | nil (bound : Nat) : DescendingPath bound []
  | cons {bound rank : Nat} {tail : List Nat} :
      rank < bound → DescendingPath rank tail →
      DescendingPath bound (rank :: tail)

theorem ordered_path_length {bound : Nat} {path : List Nat}
    (ordered : DescendingPath bound path) : path.length ≤ bound := by
  induction ordered with
  | nil => simp
  | cons less rest ih =>
    simp only [List.length_cons]
    omega

theorem every_rank_below {bound rank : Nat} {path : List Nat}
    (ordered : DescendingPath bound path) (member : rank ∈ path) : rank < bound := by
  induction ordered with
  | nil => simp at member
  | cons less rest ih =>
    simp only [List.mem_cons] at member
    rcases member with same | later
    · subst rank
      exact less
    · exact Nat.lt_trans (ih later) less

theorem no_repeated_rank {bound : Nat} {path : List Nat}
    (ordered : DescendingPath bound path) : path.Nodup := by
  induction ordered with
  | nil => exact List.nodup_nil
  | cons less rest ih =>
    apply List.nodup_cons.mpr
    constructor
    · intro member
      have impossible := every_rank_below rest member
      omega
    · exact ih

#print axioms equal_children
#print axioms repeated_then
#print axioms repeated_else
#print axioms simultaneous_cofactors
#print axioms shannon_swap
#print axioms branch_congruence
#print axioms ordered_path_length
#print axioms every_rank_below
#print axioms no_repeated_rank
end OrderedBranchLaws
