import CorrectnessCompletion
import DomainPartitions

/- Finite complete source-question signatures. Numerical instruction refinement
and native deterministic transform identity are separate implementation duties. -/
namespace ConverterSignatureEnumeration
open ConverterGuarantees ConverterDomain
variable {P W L X : Type}

def predicates : Tree P W → List P
  | .leaf _ => []
  | .branch p l r => p :: (predicates l ++ predicates r)

def signature (truth : P → X → Bool) : List P → X → List Bool
  | [], _ => []
  | p :: ps, x => truth p x :: signature truth ps x

def Agree (truth : P → X → Bool) (ps : List P) (x y : X) : Prop :=
  ∀ p ∈ ps, truth p x = truth p y

theorem signature_eq_iff (truth : P → X → Bool) (ps : List P) (x y : X) :
    signature truth ps x = signature truth ps y ↔ Agree truth ps x y := by
  induction ps with
  | nil => simp [signature, Agree]
  | cons p ps ih =>
    constructor
    · intro h
      have hs := List.cons.inj h
      intro q hq
      rcases List.mem_cons.mp hq with hq | hq
      · subst q; exact hs.1
      · exact ih.mp hs.2 q hq
    · intro h
      have hp := h p (by simp)
      have ht : Agree truth ps x y := by
        intro q hq
        exact h q (by simp [hq])
      simp only [signature, hp, ih.mpr ht]

theorem same_questions_same_leaf (truth : P → X → Bool) (tree : Tree P W)
    (x y : X) (h : Agree truth (predicates tree) x y) :
    eval truth tree x = eval truth tree y := by
  induction tree with
  | leaf w => rfl
  | branch p l r il ir =>
    have hp := h p (by simp [predicates])
    have hl : Agree truth (predicates l) x y := by
      intro q hq; exact h q (by simp [predicates, hq])
    have hr : Agree truth (predicates r) x y := by
      intro q hq; exact h q (by simp [predicates, hq])
    simp only [eval, hp, il hl, ir hr]

def orderedWords (truth : P → X → Bool) (forest : List (Tree P W)) (x : X) : List W :=
  forest.map fun tree => eval truth tree x

theorem same_signature_same_ordered_words (truth : P → X → Bool)
    (questions : List P) (forest : List (Tree P W))
    (covers : ∀ tree ∈ forest, ∀ p ∈ predicates tree, p ∈ questions)
    (x y : X) (same : signature truth questions x = signature truth questions y) :
    orderedWords truth forest x = orderedWords truth forest y := by
  apply List.map_congr_left
  intro tree ht
  apply same_questions_same_leaf
  intro p hp
  exact (signature_eq_iff truth questions x y).mp same p (covers tree ht p hp)

/-- Native class transformation is an arbitrary deterministic function of the
    COMPLETE ORDERED exact leaf-word vector, with fixed metadata/base words. -/
theorem same_signature_same_native (truth : P → X → Bool)
    (questions : List P) (forest : List (Tree P W))
    (covers : ∀ tree ∈ forest, ∀ p ∈ predicates tree, p ∈ questions)
    (nativeTransform : List W → L) (x y : X)
    (same : signature truth questions x = signature truth questions y) :
    nativeTransform (orderedWords truth forest x) =
      nativeTransform (orderedWords truth forest y) := by
  rw [same_signature_same_ordered_words truth questions forest covers x y same]

def tableTree : List P → (List Bool → L) → Tree P L
  | [], label => .leaf (label [])
  | p :: ps, label => .branch p
      (tableTree ps (fun bs => label (true :: bs)))
      (tableTree ps (fun bs => label (false :: bs)))

theorem table_tree_evaluates_signature (truth : P → X → Bool)
    (questions : List P) (label : List Bool → L) (x : X) :
    eval truth (tableTree questions label) x = label (signature truth questions x) := by
  induction questions generalizing label with
  | nil => rfl
  | cons p ps ih =>
    cases h : truth p x <;> simp [tableTree, eval, signature, h, ih]

theorem table_tree_size (questions : List P) (label : List Bool → L) :
    nodes (tableTree questions label) + 1 = 2 ^ (questions.length + 1) := by
  induction questions generalizing label with
  | nil => rfl
  | cons p ps ih =>
    have hl := ih (fun bs => label (true :: bs))
    have hr := ih (fun bs => label (false :: bs))
    simp only [tableTree, nodes, List.length_cons]
    rw [Nat.pow_succ]
    omega

/-- Coverage chooses an actual valid representative for EACH signature that
    any valid input has. Infeasible signature entries are never consulted. -/
theorem complete_signature_conversion (truth : P → X → Bool)
    (questions : List P) (forest : List (Tree P W))
    (covers : ∀ tree ∈ forest, ∀ p ∈ predicates tree, p ∈ questions)
    (nativeTransform : List W → L) (valid : X → Prop)
    (representative : List Bool → X)
    (coverage : ∀ x, valid x → valid (representative (signature truth questions x)) ∧
      signature truth questions (representative (signature truth questions x)) =
        signature truth questions x) :
    ∀ x, valid x →
      eval truth (tableTree questions (fun bs =>
        nativeTransform (orderedWords truth forest (representative bs)))) x =
      nativeTransform (orderedWords truth forest x) := by
  intro x hx
  rw [table_tree_evaluates_signature]
  exact same_signature_same_native truth questions forest covers nativeTransform
    (representative (signature truth questions x)) x (coverage x hx).2

theorem impossible_left_can_redirect (truth : P → X → Bool)
    (valid : X → Prop) (p : P) (l r : Tree P L)
    (empty : ∀ x, valid x → truth p x = false) :
    ∀ x, valid x → eval truth (.branch p l r) x = eval truth r x := by
  intro x hx; simp [eval, empty x hx]

theorem impossible_right_can_redirect (truth : P → X → Bool)
    (valid : X → Prop) (p : P) (l r : Tree P L)
    (empty : ∀ x, valid x → truth p x = true) :
    ∀ x, valid x → eval truth (.branch p l r) x = eval truth l x := by
  intro x hx; simp [eval, empty x hx]

theorem equal_children_can_redirect (truth : P → X → Bool)
    (p : P) (child : Tree P L) (x : X) :
    eval truth (.branch p child child) x = eval truth child x := by simp [eval]

/-- Integer-key feasibility narrows strict-left and non-strict-right exactly.
    Each representable finite FP32 comparison value has a consecutive key;
    the two zero words share key zero. No real-valued midpoint is required. -/
theorem left_interval_exact (lo hi cut x : Int) :
    (lo ≤ x ∧ x ≤ min hi (cut - 1)) ↔ (lo ≤ x ∧ x ≤ hi) ∧ x < cut := by omega

theorem right_interval_exact (lo hi cut x : Int) :
    (max lo cut ≤ x ∧ x ≤ hi) ↔ (lo ≤ x ∧ x ≤ hi) ∧ ¬ x < cut := by omega

theorem interval_witness_iff (lo hi : Int) :
    (∃ x, lo ≤ x ∧ x ≤ hi) ↔ lo ≤ hi := by
  constructor
  · intro ⟨x,h⟩; omega
  · intro h; exact ⟨lo,Int.le_refl lo,h⟩

theorem canonical_zero_cut_truth (x cut : FiniteWord) :
    decide (orderedKey x < orderedKey (canonicalZero cut)) =
      decide (orderedKey x < orderedKey cut) := by rw [canonical_zero_key]

theorem predicate_replacement_same_truth (truth : P → X → Bool)
    (p q : P) (l r : Tree P L) (same : ∀ x, truth p x = truth q x) :
    ∀ x, eval truth (.branch p l r) x = eval truth (.branch q l r) x := by
  intro x; simp only [eval, same x]

end ConverterSignatureEnumeration
