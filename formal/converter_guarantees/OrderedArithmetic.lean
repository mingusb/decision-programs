import Std

/-!
Ordered source reduction and class-selection lemmas. Int values may represent
fixed-point encodings of finite IEEE values; this file does NOT prove the
machine IEEE rounder implements round or that a native kernel matches it.
Order, bit identity, and deterministic transformation are separate contracts.
In particular +0 and -0 cannot be interchanged in the bit-identity theorem.
-/
namespace ConverterArithmetic

structure IntervalTriple where
  lower : Int
  actual : Int
  upper : Int

def Encloses (x : IntervalTriple) : Prop :=
  x.lower ≤ x.actual ∧ x.actual ≤ x.upper

def step (round : Int → Int) (acc term : IntervalTriple) : IntervalTriple :=
  ⟨round (acc.lower + term.lower), round (acc.actual + term.actual),
   round (acc.upper + term.upper)⟩

def MonotoneRound (round : Int → Int) : Prop :=
  ∀ x y, x ≤ y → round x ≤ round y

theorem rounded_step_encloses (round : Int → Int) (mono : MonotoneRound round)
    (acc term : IntervalTriple) (ha : Encloses acc) (ht : Encloses term) :
    Encloses (step round acc term) := by
  rcases ha with ⟨hal, hau⟩
  rcases ht with ⟨htl, htu⟩
  constructor
  · exact mono _ _ (by omega)
  · exact mono _ _ (by omega)

def orderedReduce (round : Int → Int) : IntervalTriple → List IntervalTriple → IntervalTriple
  | acc, [] => acc
  | acc, t :: rest => orderedReduce round (step round acc t) rest

theorem ordered_reduction_encloses (round : Int → Int) (mono : MonotoneRound round)
    (terms : List IntervalTriple) :
    ∀ acc, Encloses acc → (∀ t ∈ terms, Encloses t) →
    Encloses (orderedReduce round acc terms) := by
  induction terms with
  | nil => intro acc ha _; exact ha
  | cons t rest ih =>
      intro acc ha ht
      apply ih (step round acc t) (rounded_step_encloses round mono acc t ha (ht t (by simp)))
      intro x hx
      exact ht x (by simp [hx])

/-- A deterministic fold preserves exact operand identity, without an
    associativity assumption or algebraic reassociation of FP arithmetic. -/
def wordFold (add : Word → Word → Word) : Word → List Word → Word
  | acc, [] => acc
  | acc, t :: rest => wordFold add (add acc t) rest

theorem exact_operand_sequence_preserves_margin
    (add : Word → Word → Word) (left right : List Word) (bias bias' : Word)
    (hseq : left = right) (hbias : bias = bias') :
    wordFold add bias left = wordFold add bias' right := by
  rw [hseq, hbias]

/-- Identical per-tree words in original channel order imply identical
    transformed class, even if the transform has rounded probability ties. -/
theorem exact_channel_sequences_preserve_native_class
    (add : Word → Word → Word) (bias : Channel → Word)
    (left right : Channel → List Word) (transform : (Channel → Word) → Label)
    (same : ∀ c, left c = right c) :
    transform (fun c => wordFold add (bias c) (left c)) =
    transform (fun c => wordFold add (bias c) (right c)) := by
  have h : (fun c => wordFold add (bias c) (left c)) =
      (fun c => wordFold add (bias c) (right c)) := by
    funext c
    rw [same c]
  rw [h]

/-- Strict score separation is sufficient for unique winner of those scores.
    Applying this to a probability transform requires its own proved contract. -/
theorem interval_strict_winner (lo actual hi : Nat → Int) (winner count : Nat)
    (inside : winner < count)
    (bounds : ∀ c, c < count → lo c ≤ actual c ∧ actual c ≤ hi c)
    (separated : ∀ c, c < count → c ≠ winner → hi c < lo winner) :
    winner < count ∧ ∀ c, c < count → c ≠ winner → actual c < actual winner := by
  constructor
  · exact inside
  · intro c hc hne
    have hc' := bounds c hc
    have hw := bounds winner inside
    have hs := separated c hc hne
    omega

/-- Concrete sequential first-max, matching `>` replacement: ties retain
    the earlier incumbent, not the last equal score. -/
def firstMaxFrom (score : Nat → Int) : Nat → List Nat → Nat
  | incumbent, [] => incumbent
  | incumbent, candidate :: rest =>
      firstMaxFrom score (if score incumbent < score candidate then candidate else incumbent) rest

theorem first_max_member (score : Nat → Int) (rest : List Nat) :
    ∀ incumbent, firstMaxFrom score incumbent rest ∈ incumbent :: rest := by
  induction rest with
  | nil => intro incumbent; simp [firstMaxFrom]
  | cons candidate rest ih =>
      intro incumbent
      simp only [firstMaxFrom]
      split
      · have h := ih candidate
        simp only [List.mem_cons] at h ⊢
        rcases h with h | h
        · exact Or.inr (Or.inl h)
        · exact Or.inr (Or.inr h)
      · have h := ih incumbent
        simp only [List.mem_cons] at h ⊢
        rcases h with h | h
        · exact Or.inl h
        · exact Or.inr (Or.inr h)

theorem first_max_score (score : Nat → Int) (rest : List Nat) :
    ∀ incumbent, ∀ c ∈ incumbent :: rest,
      score c ≤ score (firstMaxFrom score incumbent rest) := by
  induction rest with
  | nil =>
      intro incumbent c hc
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hc
      subst c
      exact Int.le_refl _
  | cons candidate rest ih =>
      intro incumbent c hc
      simp only [firstMaxFrom]
      split
      next hs =>
        have ht := ih candidate candidate (by simp)
        simp only [List.mem_cons] at hc
        rcases hc with hc | hc | hc
        · subst c; omega
        · subst c; exact ht
        · exact ih candidate c (by simp [hc])
      next hs =>
        have ht := ih incumbent incumbent (by simp)
        simp only [List.mem_cons] at hc
        rcases hc with hc | hc | hc
        · subst c; exact ht
        · subst c; omega
        · exact ih incumbent c (by simp [hc])

theorem first_max_keeps_incumbent_on_ties (score : Nat → Int) (rest : List Nat) :
    ∀ incumbent, (∀ c ∈ rest, score c ≤ score incumbent) →
      firstMaxFrom score incumbent rest = incumbent := by
  induction rest with
  | nil => intro incumbent _; rfl
  | cons candidate rest ih =>
      intro incumbent h
      have hs := h candidate (by simp)
      simp only [firstMaxFrom, show ¬score incumbent < score candidate by omega, ↓reduceIte]
      apply ih
      intro c hc
      exact h c (by simp [hc])

theorem strict_winner_selected (score : Nat → Int) (incumbent winner : Nat) (rest : List Nat)
    (member : winner ∈ incumbent :: rest)
    (unique : ∀ c ∈ incumbent :: rest, c ≠ winner → score c < score winner) :
    firstMaxFrom score incumbent rest = winner := by
  have hm := first_max_member score rest incumbent
  have hs := first_max_score score rest incumbent winner member
  by_cases h : firstMaxFrom score incumbent rest = winner
  · exact h
  · have hu := unique _ hm h
    omega

private theorem first_max_append (score : Nat → Int) (left right : List Nat) :
    ∀ incumbent, firstMaxFrom score incumbent (left ++ right) =
      firstMaxFrom score (firstMaxFrom score incumbent left) right := by
  induction left with
  | nil => intro incumbent; rfl
  | cons candidate left ih =>
      intro incumbent
      simp only [List.cons_append, firstMaxFrom]
      exact ih _

/-- Finite ordered margin enclosures certify the lowest-index maximum: earlier
    rivals require strict separation, later rivals may tie. Int encodings exclude
    NaN/infinity; native use additionally requires the source/rounder correspondence.
    The ascending scan starts at class zero, so `inside` also ensures nonemptiness. -/
theorem interval_first_winner_selected (lo actual hi : Nat → Int) (winner count : Nat)
    (inside : winner < count)
    (bounds : ∀ c, c < count → lo c ≤ actual c ∧ actual c ≤ hi c)
    (earlier : ∀ c, c < winner → hi c < lo winner)
    (later : ∀ c, winner < c → c < count → hi c ≤ lo winner) :
    firstMaxFrom actual 0 (List.range count) = winner := by
  induction count generalizing winner with
  | zero => omega
  | succ n ih =>
      rw [List.range_succ, first_max_append]
      by_cases hw : winner < n
      · rw [ih winner hw (fun c hc => bounds c (by omega)) earlier
          (fun c hwc hc => later c hwc (by omega))]
        have hn := bounds n (by omega)
        have hwin := bounds winner inside
        have hl := later n hw (by omega)
        simp [firstMaxFrom, show ¬actual winner < actual n by omega]
      · have heq : winner = n := by omega
        subst winner
        by_cases hn : n = 0
        · subst n; simp [firstMaxFrom]
        · have member := first_max_member actual (List.range n) 0
          have hc : firstMaxFrom actual 0 (List.range n) < n := by
            simp only [List.mem_cons, List.mem_range] at member
            omega
          have hb := bounds _ (by omega : firstMaxFrom actual 0 (List.range n) < n + 1)
          have hwin := bounds n inside
          have he := earlier _ hc
          simp [firstMaxFrom, show actual (firstMaxFrom actual 0 (List.range n)) < actual n by omega]

/-- Exact integer distance; finite binary floating values can be scaled to
    integers by the common smallest-subnormal unit. -/
def distance (x y : Int) : Int := if x < y then y-x else x-y

theorem nearest_reversal_boundary (x a b : Int) (order : b < a)
    (near : distance x a ≤ distance x b) : a+b ≤ 2*x := by
  unfold distance at near
  split at near <;> split at near <;> omega

theorem nearest_lower_boundary (y a b : Int) (order : b < a)
    (near : distance y b ≤ distance y a) : 2*y ≤ a+b := by
  unfold distance at near
  split at near <;> split at near <;> omega

/-- Monotonicity follows from the nearest-representable contract itself.
    The deterministic function handles ties consistently at an identical input;
    no tie-to-even algebra or approximate real addition is needed. -/
theorem nearest_rounding_monotone (representable : Int → Prop) (round : Int → Int)
    (output : ∀ x, representable (round x))
    (nearest : ∀ x v, representable v → distance x (round x) ≤ distance x v) :
    MonotoneRound round := by
  intro x y hxy
  by_cases equal : x = y
  · subst y; exact Int.le_refl _
  · have strict : x < y := by omega
    have nx := nearest x (round y) (output y)
    have ny := nearest y (round x) (output x)
    by_cases ordered : round x ≤ round y
    · exact ordered
    · have reversal : round y < round x := by omega
      have left := nearest_reversal_boundary x (round x) (round y) reversal nx
      have right := nearest_lower_boundary y (round x) (round y) reversal ny
      omega

theorem nearest_ordered_reduction_encloses (representable : Int → Prop) (round : Int → Int)
    (output : ∀ x, representable (round x))
    (nearest : ∀ x v, representable v → distance x (round x) ≤ distance x v)
    (terms : List IntervalTriple) (acc : IntervalTriple)
    (ha : Encloses acc) (ht : ∀ t ∈ terms, Encloses t) :
    Encloses (orderedReduce round acc terms) :=
  ordered_reduction_encloses round (nearest_rounding_monotone representable round output nearest)
    terms acc ha ht

#print axioms nearest_reversal_boundary
#print axioms nearest_lower_boundary
#print axioms nearest_rounding_monotone
#print axioms nearest_ordered_reduction_encloses

#print axioms rounded_step_encloses
#print axioms ordered_reduction_encloses
#print axioms exact_operand_sequence_preserves_margin
#print axioms exact_channel_sequences_preserve_native_class
#print axioms interval_strict_winner
#print axioms first_max_member
#print axioms first_max_score
#print axioms first_max_keeps_incumbent_on_ties
#print axioms interval_first_winner_selected
#print axioms strict_winner_selected
end ConverterArithmetic
