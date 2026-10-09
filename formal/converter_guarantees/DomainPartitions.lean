import Lean

/- General source-threshold and finite-axis partition facts. Feature counts and
   category widths are unrestricted; no dataset-specific box is defined here.
   These mathematical facts do not verify machine floating-point instructions. -/
namespace ConverterDomain

section Rank
variable {α : Type} (le : α → α → Prop) [DecidableRel le]

def rank (cuts : List α) (x : α) : Nat :=
  match cuts with
  | [] => 0
  | t :: ts => (if le t x then 1 else 0) + rank ts x

theorem rank_append (a b : List α) (x : α) :
    rank le (a ++ b) x = rank le a x + rank le b x := by
  induction a with
  | nil => simp [rank]
  | cons t ts ih => simp [rank, ih, Nat.add_assoc]

theorem rank_le_length (cuts : List α) (x : α) :
    rank le cuts x ≤ cuts.length := by
  induction cuts with
  | nil => simp [rank]
  | cons t ts ih =>
    simp only [rank, List.length_cons]
    split <;> omega

theorem rank_all_yes (cuts : List α) (x : α)
    (h : ∀ t ∈ cuts, le t x) : rank le cuts x = cuts.length := by
  induction cuts with
  | nil => rfl
  | cons t ts ih =>
    have ht := h t (by simp)
    have hs : ∀ u ∈ ts, le u x := by
      intro u hu
      exact h u (by simp [hu])
    simp [rank, ht, ih hs, Nat.add_comm]

theorem rank_all_no (cuts : List α) (x : α)
    (h : ∀ t ∈ cuts, ¬le t x) : rank le cuts x = 0 := by
  induction cuts with
  | nil => rfl
  | cons t ts ih =>
    have ht := h t (by simp)
    have hs : ∀ u ∈ ts, ¬le u x := by
      intro u hu
      exact h u (by simp [hu])
    simp [rank, ht, ih hs]

/-- An actual threshold at zero-based index k routes left exactly at rank < k+1.
    Duplicated equal thresholds are permitted; canonical unique tables are a special case. -/
theorem threshold_branch_correspondence
    (trans : ∀ a b c, le a b → le b c → le a c)
    (pre post : List α) (cut x : α)
    (sorted : (pre ++ cut :: post).Pairwise le) :
    rank le (pre ++ cut :: post) x < pre.length + 1 ↔ ¬le cut x := by
  have hp : ∀ t ∈ pre, le t cut := by
    intro t ht
    exact (List.pairwise_append.mp sorted).2.2 t ht cut (by simp)
  have hs : ∀ t ∈ post, le cut t := by
    intro t ht
    exact (List.pairwise_cons.mp (List.pairwise_append.mp sorted).2.1).1 t ht
  by_cases h : le cut x
  · have allpre : ∀ t ∈ pre, le t x := by
      intro t ht
      exact trans t cut x (hp t ht) h
    simp [rank_append, rank, h, rank_all_yes le pre x allpre]
  · have allpost : ∀ t ∈ post, ¬le t x := by
      intro t ht htx
      exact h (trans cut t x (hs t ht) htx)
    have bound := rank_le_length le pre x
    simp [rank_append, rank, h, rank_all_no le post x allpost]
    omega

theorem rank_comparison_congr (cuts : List α) (x y : α)
    (h : ∀ t ∈ cuts, le t x ↔ le t y) :
    rank le cuts x = rank le cuts y := by
  induction cuts with
  | nil => rfl
  | cons t ts ih =>
    have ht := h t (by simp)
    have hs : ∀ u ∈ ts, le u x ↔ le u y := by
      intro u hu
      exact h u (by simp [hu])
    simp only [rank, ih hs]
    by_cases h : le t x
    · simp [h, ht.mp h]
    · have hy : ¬le t y := fun hy => h (ht.mpr hy)
      simp [h, hy]

end Rank

/-- Signless finite IEEE-754 binary32 word: all normal, subnormal and zero
    encodings, excluding infinity and NaN. No integral-input restriction. -/
structure FiniteWord where
  negative : Bool
  magnitude : Fin 2139095040
deriving DecidableEq

/-- Ordered-word specification. The CPU/CUDA comparison implementation must
    refine this specification; that instruction-level fact is not assumed proved. -/
def orderedKey (x : FiniteWord) : Int :=
  if x.negative then -(Int.ofNat x.magnitude.val) else Int.ofNat x.magnitude.val

def wordLE (x y : FiniteWord) : Prop := orderedKey x ≤ orderedKey y
instance : DecidableRel wordLE := fun x y => inferInstanceAs (Decidable (orderedKey x ≤ orderedKey y))

def canonicalZero (x : FiniteWord) : FiniteWord :=
  if x.magnitude.val = 0 then { negative := false, magnitude := x.magnitude } else x

theorem canonical_zero_key (x : FiniteWord) :
    orderedKey (canonicalZero x) = orderedKey x := by
  unfold canonicalZero
  split
  · rename_i h
    simp [orderedKey, h]
  · rfl

theorem signed_zero_equal (negative : Bool) :
    orderedKey { negative := negative, magnitude := ⟨0, by decide⟩ } = 0 := by
  simp [orderedKey]

theorem canonical_zero_compare (x y : FiniteWord) :
    wordLE (canonicalZero x) (canonicalZero y) ↔ wordLE x y := by
  simp [wordLE, canonical_zero_key]

theorem canonical_zero_rank (cuts : List FiniteWord) (x : FiniteWord) :
    rank wordLE cuts (canonicalZero x) = rank wordLE cuts x := by
  apply rank_comparison_congr
  intro t _
  simp [wordLE, canonical_zero_key]

theorem finite_word_threshold_branch (pre post : List FiniteWord) (cut x : FiniteWord)
    (sorted : (pre ++ cut :: post).Pairwise wordLE) :
    rank wordLE (pre ++ cut :: post) x < pre.length + 1 ↔ orderedKey x < orderedKey cut := by
  rw [threshold_branch_correspondence wordLE (fun _ _ _ => Int.le_trans) pre post cut x sorted]
  exact Int.not_le



structure Interval where
  lo : Nat
  hi : Nat
deriving DecidableEq

def Interval.Valid (a : Interval) : Prop := a.lo ≤ a.hi
def Interval.width (a : Interval) : Nat := a.hi - a.lo + 1
def Interval.Mem (a : Interval) (x : Nat) : Prop := a.lo ≤ x ∧ x ≤ a.hi
def Interval.midpoint (a : Interval) : Nat := a.lo + (a.hi - a.lo + 1) / 2
def Interval.left (a : Interval) (cut : Nat) : Interval := ⟨a.lo, cut - 1⟩
def Interval.right (a : Interval) (cut : Nat) : Interval := ⟨cut, a.hi⟩

theorem midpoint_strict (a : Interval) (varying : a.lo < a.hi) :
    a.lo < a.midpoint ∧ a.midpoint ≤ a.hi := by
  simp only [Interval.midpoint]
  omega

theorem interval_split_membership (a : Interval) (cut x : Nat)
    (strict : a.lo < cut ∧ cut ≤ a.hi) :
    ((a.left cut).Mem x ↔ a.Mem x ∧ x < cut) ∧
    ((a.right cut).Mem x ↔ a.Mem x ∧ ¬x < cut) := by
  simp only [Interval.left, Interval.right, Interval.Mem]
  omega

theorem interval_split_cover (a : Interval) (cut x : Nat)
    (strict : a.lo < cut ∧ cut ≤ a.hi) :
    a.Mem x ↔ (a.left cut).Mem x ∨ (a.right cut).Mem x := by
  have h := interval_split_membership a cut x strict
  rw [h.1, h.2]
  by_cases hx : x < cut <;> simp [hx]

theorem interval_split_disjoint (a : Interval) (cut x : Nat)
    (strict : a.lo < cut ∧ cut ≤ a.hi) :
    ¬((a.left cut).Mem x ∧ (a.right cut).Mem x) := by
  have h := interval_split_membership a cut x strict
  rw [h.1, h.2]
  by_cases hx : x < cut <;> simp [hx]

theorem interval_split_valid (a : Interval) (cut : Nat)
    (strict : a.lo < cut ∧ cut ≤ a.hi) :
    (a.left cut).Valid ∧ (a.right cut).Valid := by
  simp only [Interval.left, Interval.right, Interval.Valid]
  omega

theorem interval_split_width (a : Interval) (cut : Nat)
    (strict : a.lo < cut ∧ cut ≤ a.hi) :
    (a.left cut).width + (a.right cut).width = a.width := by
  simp only [Interval.left, Interval.right, Interval.width]
  omega

theorem interval_split_strict_width (a : Interval) (cut : Nat)
    (strict : a.lo < cut ∧ cut ≤ a.hi) :
    0 < (a.left cut).width ∧ (a.left cut).width < a.width ∧
    0 < (a.right cut).width ∧ (a.right cut).width < a.width := by
  simp only [Interval.left, Interval.right, Interval.width]
  omega

def numericVolume (axes : List Interval) : Nat := (axes.map Interval.width).prod

theorem numeric_volume_append (a b : List Interval) :
    numericVolume (a ++ b) = numericVolume a * numericVolume b := by
  simp [numericVolume, List.map_append, List.prod_append]

theorem numeric_volume_positive (axes : List Interval) : 0 < numericVolume axes := by
  induction axes with
  | nil => simp [numericVolume]
  | cons a axes ih =>
    simp only [numericVolume, List.map_cons, List.prod_cons] at *
    exact Nat.mul_pos (by simp [Interval.width]) ih

theorem numeric_split_volume (pre post : List Interval) (a : Interval) (cut : Nat)
    (strict : a.lo < cut ∧ cut ≤ a.hi) :
    numericVolume (pre ++ a.left cut :: post) +
      numericVolume (pre ++ a.right cut :: post) =
      numericVolume (pre ++ a :: post) := by
  rw [numeric_volume_append, numeric_volume_append, numeric_volume_append]
  change numericVolume pre * ((a.left cut).width * numericVolume post) + numericVolume pre * ((a.right cut).width * numericVolume post) = numericVolume pre * (a.width * numericVolume post)
  rw [← Nat.mul_add, ← Nat.add_mul, interval_split_width a cut strict]

def axisCells (a : Interval) : List Nat := List.range' a.lo a.width

def numericCells : List Interval → List (List Nat)
  | [] => [[]]
  | a :: axes => (axisCells a).flatMap fun x => (numericCells axes).map (x :: ·)

theorem flatmap_fixed_length {α β : Type} (xs : List α) (f : α → List β) (k : Nat)
    (h : ∀ x ∈ xs, (f x).length = k) :
    (xs.flatMap f).length = xs.length * k := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
    have hx := h x (by simp)
    have hs : ∀ y ∈ xs, (f y).length = k := by
      intro y hy
      exact h y (by simp [hy])
    simp [hx, ih hs, Nat.add_mul, Nat.add_comm]

theorem numeric_cells_count (axes : List Interval) :
    (numericCells axes).length = numericVolume axes := by
  induction axes with
  | nil => simp [numericCells, numericVolume]
  | cons a axes ih =>
    rw [numericCells, flatmap_fixed_length _ _ (numericVolume axes)]
    · simp [axisCells, numericVolume]
    · intro x _
      simp [ih]

def axesMem : List Interval → List Nat → Prop
  | [], [] => True
  | a :: axes, x :: xs => a.Mem x ∧ axesMem axes xs
  | _, _ => False

theorem numeric_split_cover (pre post : List Interval) (a : Interval) (cut : Nat)
    (strict : a.lo < cut ∧ cut ≤ a.hi) (xs : List Nat) :
    axesMem (pre ++ a :: post) xs ↔
    axesMem (pre ++ a.left cut :: post) xs ∨
    axesMem (pre ++ a.right cut :: post) xs := by
  induction pre generalizing xs with
  | nil =>
    cases xs with
    | nil => simp [axesMem]
    | cons x xs =>
      simp only [List.nil_append, axesMem]
      rw [interval_split_cover a cut x strict]
      simp only [or_and_right]
  | cons p pre ih =>
    cases xs with
    | nil => simp [axesMem]
    | cons x xs =>
      simp only [List.cons_append, axesMem]
      rw [ih xs]
      simp only [and_or_left]

theorem numeric_split_disjoint (pre post : List Interval) (a : Interval) (cut : Nat)
    (strict : a.lo < cut ∧ cut ≤ a.hi) (xs : List Nat) :
    ¬(axesMem (pre ++ a.left cut :: post) xs ∧
      axesMem (pre ++ a.right cut :: post) xs) := by
  induction pre generalizing xs with
  | nil =>
    cases xs with
    | nil => simp [axesMem]
    | cons x xs =>
      simp only [List.nil_append, axesMem]
      intro h
      exact interval_split_disjoint a cut x strict ⟨h.1.1, h.2.1⟩
  | cons p pre ih =>
    cases xs with
    | nil => simp [axesMem]
    | cons x xs =>
      simp only [List.cons_append, axesMem]
      intro h
      exact ih xs ⟨h.1.2, h.2.2⟩

def categorySupport (n mask : Nat) : List Nat :=
  (List.range n).filter mask.testBit

theorem category_support_mem (n mask i : Nat) :
    i ∈ categorySupport n mask ↔ i < n ∧ mask.testBit i = true := by
  simp [categorySupport]

theorem category_support_nodup (n mask : Nat) :
    (categorySupport n mask).Nodup := by
  exact List.Pairwise.filter _ (List.nodup_range (n := n))

theorem axis_cells_mem (a : Interval) (valid : a.Valid) (x : Nat) :
    x ∈ axisCells a ↔ a.Mem x := by
  simp only [axisCells, List.mem_range'_1, Interval.width, Interval.Mem]
  unfold Interval.Valid at valid
  omega

theorem numeric_cells_mem (axes : List Interval) (valid : ∀ a ∈ axes, a.Valid)
    (xs : List Nat) : xs ∈ numericCells axes ↔ axesMem axes xs := by
  induction axes generalizing xs with
  | nil => cases xs <;> simp [numericCells, axesMem]
  | cons a axes ih =>
    have va := valid a (by simp)
    have vs : ∀ b ∈ axes, b.Valid := by
      intro b hb
      exact valid b (by simp [hb])
    cases xs with
    | nil => simp [numericCells, axesMem]
    | cons x xs => simp [numericCells, axesMem, axis_cells_mem a va, ih vs]

theorem product_map_nodup {α β γ : Type} (xs : List α) (ys : List β) (f : α → β → γ)
    (hx : xs.Nodup) (hy : ys.Nodup)
    (inj : ∀ a b c d, f a b = f c d → a = c ∧ b = d) :
    (xs.flatMap fun x => ys.map (f x)).Nodup := by
  apply List.pairwise_flatMap.mpr
  constructor
  · intro x _
    apply List.Pairwise.map (f x) _ hy
    intro y z hyz heq
    exact hyz (inj x y x z heq).2
  · apply List.Pairwise.imp _ hx
    intro x z hxz u hu v hv heq
    rcases List.mem_map.mp hu with ⟨y, _, rfl⟩
    rcases List.mem_map.mp hv with ⟨w, _, rfl⟩
    exact hxz (inj x y z w heq).1

theorem numeric_cells_nodup (axes : List Interval) : (numericCells axes).Nodup := by
  induction axes with
  | nil => simp [numericCells]
  | cons a axes ih =>
    apply product_map_nodup (axisCells a) (numericCells axes) (fun x xs => x :: xs)
    · exact List.nodup_range'
    · exact ih
    · intro x xs y ys h
      exact List.cons.inj h

end ConverterDomain
