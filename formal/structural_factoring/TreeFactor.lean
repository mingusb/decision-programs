import Lean

/-!
Structural factoring for one physical binary tree.

A predicate symbol is interpreted by an arbitrary total, pure Boolean valuation.
This deliberately leaves floating-point arithmetic and input rejection outside
the theorem. nodeCount counts every occurrence, including duplicated subtrees.
No shared DAG representation is used.
-/

namespace TreeFactor

inductive Tree (Predicate Label : Type) where
  | leaf : Label → Tree Predicate Label
  | branch : Predicate → Tree Predicate Label → Tree Predicate Label → Tree Predicate Label

def run (truth : Predicate → Bool) : Tree Predicate Label → Label
  | .leaf label => label
  | .branch p yes no => match truth p with
    | true => run truth yes
    | false => run truth no

def nodeCount : Tree Predicate Label → Nat
  | .leaf _ => 1
  | .branch _ yes no => nodeCount yes + nodeCount no + 1

variable {Predicate Label Input : Type}

def choose (condition : Bool) (yes no : Label) : Label :=
  match condition with
  | true => yes
  | false => no

theorem choose_same (condition : Bool) (value : Label) :
    choose condition value value = value := by
  cases condition <;> rfl

theorem choose_commonLeft (p q : Bool) (a b c : Label) :
    choose p (choose q a b) (choose q a c) =
    choose q a (choose p b c) := by
  cases p <;> cases q <;> rfl

theorem choose_commonRight (p q : Bool) (a b c : Label) :
    choose p (choose q a c) (choose q b c) =
    choose q (choose p a b) c := by
  cases p <;> cases q <;> rfl

theorem choose_repeatedLeft (p : Bool) (a b c : Label) :
    choose p (choose p a b) c = choose p a c := by
  cases p <;> rfl

theorem choose_repeatedRight (p : Bool) (a b c : Label) :
    choose p a (choose p b c) = choose p a c := by
  cases p <;> rfl

theorem nodeCount_positive (tree : Tree Predicate Label) : 0 < nodeCount tree := by
  cases tree with
  | leaf label => exact Nat.zero_lt_succ 0
  | branch p a b => exact Nat.zero_lt_succ (nodeCount a + nodeCount b)

section LocalIdentities

variable (truth : Predicate → Bool) (p q : Predicate) (a b c : Tree Predicate Label)

theorem identical_run :
    run truth (.branch p a a) = run truth a := by
  exact choose_same (truth p) (run truth a)

theorem identical_count :
    nodeCount (.branch p a a) = nodeCount a + (nodeCount a + 1) := by
  change nodeCount a + nodeCount a + 1 = nodeCount a + (nodeCount a + 1)
  exact Nat.add_assoc _ _ _

/-- Pull the common true branch of q above p, retaining one physical copy of a. -/
theorem commonLeft_run :
    run truth (.branch p (.branch q a b) (.branch q a c)) =
    run truth (.branch q a (.branch p b c)) := by
  exact choose_commonLeft (truth p) (truth q) (run truth a) (run truth b) (run truth c)

theorem commonLeft_count :
    nodeCount (.branch p (.branch q a b) (.branch q a c)) =
    nodeCount (.branch q a (.branch p b c)) + (nodeCount a + 1) := by
  change (nodeCount a + nodeCount b + 1) + (nodeCount a + nodeCount c + 1) + 1 = (nodeCount a + (nodeCount b + nodeCount c + 1) + 1) + (nodeCount a + 1)
  repeat rw [Nat.add_assoc]
  rw [Nat.add_left_comm (nodeCount a) (nodeCount c), Nat.add_left_comm 1 (nodeCount c), Nat.add_left_comm (nodeCount a) 1]

/-- Pull the common false branch of q above p, retaining one physical copy of c. -/
theorem commonRight_run :
    run truth (.branch p (.branch q a c) (.branch q b c)) =
    run truth (.branch q (.branch p a b) c) := by
  exact choose_commonRight (truth p) (truth q) (run truth a) (run truth b) (run truth c)

theorem commonRight_count :
    nodeCount (.branch p (.branch q a c) (.branch q b c)) =
    nodeCount (.branch q (.branch p a b) c) + (nodeCount c + 1) := by
  change (nodeCount a + nodeCount c + 1) + (nodeCount b + nodeCount c + 1) + 1 = ((nodeCount a + nodeCount b + 1) + nodeCount c + 1) + (nodeCount c + 1)
  repeat rw [Nat.add_assoc]
  rw [Nat.add_left_comm 1 (nodeCount b), Nat.add_left_comm (nodeCount c) (nodeCount b), Nat.add_left_comm (nodeCount c) 1, Nat.add_left_comm (nodeCount c) 1]

/-- A second p on p's true path cannot select its false branch b. -/
theorem repeatedLeft_run :
    run truth (.branch p (.branch p a b) c) =
    run truth (.branch p a c) := by
  exact choose_repeatedLeft (truth p) (run truth a) (run truth b) (run truth c)

theorem repeatedLeft_count :
    nodeCount (.branch p (.branch p a b) c) =
    nodeCount (.branch p a c) + (nodeCount b + 1) := by
  change (nodeCount a + nodeCount b + 1) + nodeCount c + 1 = (nodeCount a + nodeCount c + 1) + (nodeCount b + 1)
  repeat rw [Nat.add_assoc]
  rw [Nat.add_left_comm 1 (nodeCount c), Nat.add_left_comm (nodeCount b) (nodeCount c), Nat.add_left_comm (nodeCount b) 1]

/-- A second p on p's false path cannot select its true branch b. -/
theorem repeatedRight_run :
    run truth (.branch p a (.branch p b c)) =
    run truth (.branch p a c) := by
  exact choose_repeatedRight (truth p) (run truth a) (run truth b) (run truth c)

theorem repeatedRight_count :
    nodeCount (.branch p a (.branch p b c)) =
    nodeCount (.branch p a c) + (nodeCount b + 1) := by
  change nodeCount a + (nodeCount b + nodeCount c + 1) + 1 = (nodeCount a + nodeCount c + 1) + (nodeCount b + 1)
  repeat rw [Nat.add_assoc]
  rw [Nat.add_left_comm (nodeCount b) (nodeCount c), Nat.add_left_comm (nodeCount b) 1]

end LocalIdentities

/-- A syntactic derivation of a single rewrite, including its exact saving.
    Context constructors permit the local rewrite anywhere in the physical tree. -/
inductive Step : Tree Predicate Label → Tree Predicate Label → Nat → Prop where
  | identical (p a) :
      Step (.branch p a a) a (nodeCount a + 1)
  | commonLeft (p q a b c) :
      Step (.branch p (.branch q a b) (.branch q a c))
        (.branch q a (.branch p b c)) (nodeCount a + 1)
  | commonRight (p q a b c) :
      Step (.branch p (.branch q a c) (.branch q b c))
        (.branch q (.branch p a b) c) (nodeCount c + 1)
  | repeatedLeft (p a b c) :
      Step (.branch p (.branch p a b) c) (.branch p a c) (nodeCount b + 1)
  | repeatedRight (p a b c) :
      Step (.branch p a (.branch p b c)) (.branch p a c) (nodeCount b + 1)
  | inLeft (p other) (rewrite : Step before after saved) :
      Step (.branch p before other) (.branch p after other) saved
  | inRight (p other) (rewrite : Step before after saved) :
      Step (.branch p other before) (.branch p other after) saved

variable {before after first second final : Tree Predicate Label} {saved laterSaved steps : Nat}

theorem Step.preserves (rewrite : Step before after saved) (truth : Predicate → Bool) :
    run truth before = run truth after := by
  induction rewrite with
  | identical p a => exact identical_run truth p a
  | commonLeft p q a b c => exact commonLeft_run truth p q a b c
  | commonRight p q a b c => exact commonRight_run truth p q a b c
  | repeatedLeft p a b c => exact repeatedLeft_run truth p a b c
  | repeatedRight p a b c => exact repeatedRight_run truth p a b c
  | inLeft p other rewrite ih =>
      exact congrArg (fun value => choose (truth p) value (run truth other)) ih
  | inRight p other rewrite ih =>
      exact congrArg (fun value => choose (truth p) (run truth other) value) ih

theorem add_saving_left {a b saved : Nat} (same : a = b + saved) (other : Nat) :
    a + other + 1 = (b + other + 1) + saved := by
  rw [same]
  repeat rw [Nat.add_assoc]
  rw [Nat.add_left_comm saved other, Nat.add_comm saved 1]

theorem add_saving_right {a b saved : Nat} (same : a = b + saved) (other : Nat) :
    other + a + 1 = (other + b + 1) + saved := by
  rw [same]
  repeat rw [Nat.add_assoc]
  rw [Nat.add_comm saved 1]

theorem add_savings {a b c one two : Nat} (first : a = b + one) (second : b = c + two) :
    a = c + (one + two) := by
  rw [first, second, Nat.add_assoc, Nat.add_comm two one]

theorem Step.exact_saving (rewrite : Step before after saved) :
    nodeCount before = nodeCount after + saved := by
  induction rewrite with
  | identical p a => exact identical_count p a
  | commonLeft p q a b c => exact commonLeft_count p q a b c
  | commonRight p q a b c => exact commonRight_count p q a b c
  | repeatedLeft p a b c => exact repeatedLeft_count p a b c
  | repeatedRight p a b c => exact repeatedRight_count p a b c
  | inLeft p other rewrite ih => exact add_saving_left ih (nodeCount other)
  | inRight p other rewrite ih => exact add_saving_right ih (nodeCount other)

theorem Step.saves_at_least_two (rewrite : Step before after saved) : 2 ≤ saved := by
  induction rewrite with
  | identical p a => exact Nat.succ_le_succ (nodeCount_positive a)
  | commonLeft p q a b c => exact Nat.succ_le_succ (nodeCount_positive a)
  | commonRight p q a b c => exact Nat.succ_le_succ (nodeCount_positive c)
  | repeatedLeft p a b c => exact Nat.succ_le_succ (nodeCount_positive b)
  | repeatedRight p a b c => exact Nat.succ_le_succ (nodeCount_positive b)
  | inLeft p other rewrite ih => exact ih
  | inRight p other rewrite ih => exact ih

theorem Step.strictly_smaller (rewrite : Step before after saved) :
    nodeCount after < nodeCount before := by
  rw [rewrite.exact_saving]
  exact Nat.lt_add_of_pos_right (Nat.lt_of_lt_of_le (Nat.zero_lt_succ 1) rewrite.saves_at_least_two)

/-- Every rewrite sequence retains one tree and adds exact physical savings. -/
inductive Sequence : Tree Predicate Label → Tree Predicate Label → Nat → Nat → Prop where
  | empty (tree) : Sequence tree tree 0 0
  | next {first second final : Tree Predicate Label} {saved laterSaved steps : Nat}
      (rewrite : Step first second saved)
      (tail : Sequence second final laterSaved steps) :
      Sequence first final (saved + laterSaved) (steps + 1)

theorem Sequence.preserves
    (rewrites : Sequence first final saved steps) (truth : Predicate → Bool) :
    run truth first = run truth final := by
  induction rewrites with
  | empty tree => rfl
  | next rewrite tail ih => exact Eq.trans (rewrite.preserves truth) ih

theorem Sequence.exact_saving (rewrites : Sequence first final saved steps) :
    nodeCount first = nodeCount final + saved := by
  induction rewrites with
  | empty tree => rfl
  | next rewrite tail ih => exact add_savings rewrite.exact_saving ih

theorem Sequence.saving_lower_bound (rewrites : Sequence first final saved steps) :
    2 * steps ≤ saved := by
  induction rewrites with
  | empty tree => exact Nat.le_refl 0
  | next rewrite tail ih =>
      calc
        _ = _ := Nat.mul_succ 2 _
        _ ≤ _ := Nat.add_le_add ih rewrite.saves_at_least_two
        _ = _ := Nat.add_comm _ _

/-- Strict savings bound the length of any sequence; repeated application
    cannot run forever while only using these constructors. -/
theorem Sequence.length_bound (rewrites : Sequence first final saved steps) :
    2 * steps + 1 ≤ nodeCount first := by
  have combined := Nat.add_le_add (nodeCount_positive final) rewrites.saving_lower_bound
  rw [Nat.add_comm 1 _, ← rewrites.exact_saving] at combined
  exact combined

/-- This is the executable/refinement boundary: supply any pure total input
    interpretation for each predicate symbol. No numeric algebra is assumed. -/
def runInput (predicate : Predicate → Input → Bool)
    (tree : Tree Predicate Label) (input : Input) : Label :=
  run (fun p => predicate p input) tree

theorem Sequence.preserves_input
    (rewrites : Sequence first final saved steps)
    (predicate : Predicate → Input → Bool) (input : Input) :
    runInput predicate first input = runInput predicate final input :=
  rewrites.preserves (fun p => predicate p input)

-- A concrete two-step example: 13 physical nodes become 7. The first step
-- factors a three-node subtree (saving four); the next removes a repeated
-- predicate with a one-node unreachable branch (saving two).
def exampleA : Tree Bool Nat := .branch true (.leaf 7) (.leaf 8)
def exampleBefore : Tree Bool Nat :=
  .branch false (.branch true exampleA (.branch false (.leaf 2) (.leaf 3)))
    (.branch true exampleA (.leaf 4))
def exampleMiddle : Tree Bool Nat :=
  .branch true exampleA (.branch false (.branch false (.leaf 2) (.leaf 3)) (.leaf 4))
def exampleAfter : Tree Bool Nat :=
  .branch true exampleA (.branch false (.leaf 2) (.leaf 4))

theorem example_derivation : Sequence exampleBefore exampleAfter 6 2 := by
  exact Sequence.next (Step.commonLeft false true exampleA
    (.branch false (.leaf 2) (.leaf 3)) (.leaf 4))
    (Sequence.next (Step.inRight true exampleA
      (Step.repeatedLeft false (.leaf 2) (.leaf 3) (.leaf 4)))
      (Sequence.empty exampleAfter))

theorem example_counts : nodeCount exampleBefore = 13 ∧ nodeCount exampleAfter = 7 :=
  And.intro rfl rfl

end TreeFactor

#print axioms TreeFactor.nodeCount_positive
#print axioms TreeFactor.identical_run
#print axioms TreeFactor.identical_count
#print axioms TreeFactor.commonLeft_run
#print axioms TreeFactor.commonLeft_count
#print axioms TreeFactor.commonRight_run
#print axioms TreeFactor.commonRight_count
#print axioms TreeFactor.repeatedLeft_run
#print axioms TreeFactor.repeatedLeft_count
#print axioms TreeFactor.repeatedRight_run
#print axioms TreeFactor.repeatedRight_count
#print axioms TreeFactor.Step.preserves
#print axioms TreeFactor.Step.exact_saving
#print axioms TreeFactor.Step.saves_at_least_two
#print axioms TreeFactor.Step.strictly_smaller
#print axioms TreeFactor.Sequence.preserves
#print axioms TreeFactor.Sequence.exact_saving
#print axioms TreeFactor.Sequence.saving_lower_bound
#print axioms TreeFactor.Sequence.length_bound
#print axioms TreeFactor.Sequence.preserves_input
#print axioms TreeFactor.example_derivation
#print axioms TreeFactor.example_counts
