import HardAxisRewrites

/-! Termination and catalogue exhaustion for exact hard tree rewrites.
A finite DAG is interpreted by its finite virtual unfolding; implementation
refinement, exact-key interning and complete matcher coverage remain explicit
obligations. Encoded-byte optimality is not claimed. -/
namespace ConverterHardRewriteExhaustion
open ConverterGuarantees ConverterHardAxisRewrites
variable {L : Type}

def internal : Tree Question L → Nat
  | .leaf _ => 0
  | .branch _ a b => internal a + internal b + 1

inductive Step : Tree Question L → Tree Question L → Prop where
  | left_weaker (f : Fin 10) (a b : Int) (ordered : a ≤ b)
      (l m r : Tree Question L) :
      Step (.branch (.numeric f a) (.branch (.numeric f b) l m) r)
        (.branch (.numeric f a) l r)
  | right_stronger (f : Fin 10) (a b : Int) (ordered : b ≤ a)
      (l m r : Tree Question L) :
      Step (.branch (.numeric f a) l (.branch (.numeric f b) m r))
        (.branch (.numeric f a) l r)
  | absorb_smaller (f : Fin 10) (a b : Int) (ordered : b ≤ a)
      (l r : Tree Question L) :
      Step (.branch (.numeric f a) (.branch (.numeric f b) l r) r)
        (.branch (.numeric f b) l r)
  | absorb_larger (f : Fin 10) (a b : Int) (ordered : a ≤ b)
      (l r : Tree Question L) :
      Step (.branch (.numeric f a) l (.branch (.numeric f b) l r))
        (.branch (.numeric f b) l r)
  | repeated_left (q : Question) (l m r : Tree Question L) :
      Step (.branch q (.branch q l m) r) (.branch q l r)
  | repeated_right (q : Question) (l m r : Tree Question L) :
      Step (.branch q l (.branch q m r)) (.branch q l r)
  | equal_children (q : Question) (t : Tree Question L) :
      Step (.branch q t t) t
  | under_left (q : Question) {a b : Tree Question L} (right : Tree Question L) :
      Step a b → Step (.branch q a right) (.branch q b right)
  | under_right (q : Question) (left : Tree Question L) {a b : Tree Question L} :
      Step a b → Step (.branch q left a) (.branch q left b)

theorem step_correct {a b : Tree Question L} (step : Step a b) (x : Cell) :
    eval truth a x = eval truth b x := by
  induction step with
  | left_weaker f a b ordered l m r => exact left_weaker_test f a b ordered l m r x
  | right_stronger f a b ordered l m r => exact right_stronger_test f a b ordered l m r x
  | absorb_smaller f a b ordered l r => exact absorb_smaller_cut f a b ordered l r x
  | absorb_larger f a b ordered l r => exact absorb_larger_cut f a b ordered l r x
  | repeated_left q l m r => exact repeated_left_test q l m r x
  | repeated_right q l m r => exact repeated_right_test q l m r x
  | equal_children q t => simp [eval]
  | under_left q right _ ih => simp only [eval, ih]
  | under_right q left _ ih => simp only [eval, ih]

/-- Every nontrivial occurrence rewrite drops at least one unfolded branch. -/
theorem step_decreases {a b : Tree Question L} (step : Step a b) :
    internal b < internal a := by
  induction step <;> simp_all only [internal] <;> omega

inductive Steps : Nat → Tree Question L → Tree Question L → Prop where
  | refl (t : Tree Question L) : Steps 0 t t
  | next {n : Nat} {a b c : Tree Question L} :
      Step a b → Steps n b c → Steps (n+1) a c

theorem steps_correct {n : Nat} {a b : Tree Question L}
    (path : Steps n a b) (x : Cell) : eval truth a x = eval truth b x := by
  induction path with
  | refl t => rfl
  | next step _ ih => exact (step_correct step x).trans ih

theorem steps_bound {n : Nat} {a b : Tree Question L} (path : Steps n a b) :
    internal b + n ≤ internal a := by
  induction path with
  | refl t => simp
  | next step _ ih => have := step_decreases step; omega

theorem no_infinite_rewrites (start : Tree Question L)
    (trajectory : Nat → Tree Question L) :
    ¬ (∀ n, Steps n start (trajectory n)) := by
  intro all
  have bound := steps_bound (all (internal start + 1))
  omega

def Exhausted (t : Tree Question L) : Prop := ∀ u, ¬ Step t u

/-- This is existence of a catalogue normal form, not a shortest program. -/
theorem reaches_exhausted (t : Tree Question L) :
    ∃ n u, Steps n t u ∧ Exhausted u := by
  induction count : internal t using Nat.strongRecOn generalizing t with
  | ind k ih =>
    classical
    by_cases done : Exhausted t
    · exact ⟨0,t,Steps.refl t,done⟩
    · have hasNext : ∃ u, Step t u := by
        apply Classical.byContradiction
        intro absent
        apply done
        intro u step
        exact absent ⟨u,step⟩
      obtain ⟨u,step⟩ := hasNext
      have smaller : internal u < k := by have := step_decreases step; omega
      obtain ⟨n,v,path,normal⟩ := ih (internal u) smaller u rfl
      exact ⟨n+1,v,Steps.next step path,normal⟩

/-- A complete matcher must inspect the final canonical collected candidate.
    A zero count before reference canonicalization need not meet this premise. -/
theorem complete_matcher_zero (hasNext : Tree Question L → Bool)
    (complete : ∀ t, hasNext t = true ↔ ∃ u, Step t u)
    (t : Tree Question L) (zero : hasNext t = false) : Exhausted t := by
  intro u step
  have positive := (complete t).mpr ⟨u,step⟩
  simp [zero] at positive

/-- Current one-output-per-input staging/collection does not grow physical
    node count. A semantic pass or a pure canonicalization pass then decreases
    this Nat measure. No saturated runtime counter is used by the proof. -/
theorem complete_pass_decreases (oldUnfolded newUnfolded oldPhysical newPhysical : Nat)
    (physical : newPhysical ≤ oldPhysical)
    (progress : newUnfolded < oldUnfolded ∨
      (newUnfolded = oldUnfolded ∧ newPhysical < oldPhysical)) :
    newUnfolded + newPhysical < oldUnfolded + oldPhysical := by
  omega

/-- Exact interning/collection cannot affect the virtual unfolded measure
    when its refinement returns the very same unfolded tree. -/
theorem exact_unfolding_preserves_measure (before after : Tree Question L)
    (same : before = after) : internal before = internal after := by
  rw [same]

/-- Strict-byte selection retains an equivalent incumbent on a tie or loss;
    exploration may still continue from a different equivalent candidate. -/
theorem best_byte_selection (oldBytes candidateBytes : Nat) :
    (if candidateBytes < oldBytes then candidateBytes else oldBytes) ≤ oldBytes := by
  split <;> omega

end ConverterHardRewriteExhaustion
