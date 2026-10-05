import Lean

/- Concrete rank-map and finite-box progress layer.
   This file specifies finite comparisons; it does not verify machine FP instructions. -/
namespace ConverterRankBox

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

structure Box where
  axes : List Interval
  wilderness : List Nat
  soil : List Nat
deriving DecidableEq

def Box.Valid (b : Box) : Prop :=
  (∀ a ∈ b.axes, a.Valid ∧ a.hi ≤ 16777216) ∧
  b.wilderness ≠ [] ∧ b.soil ≠ [] ∧
  b.wilderness.Nodup ∧ b.soil.Nodup ∧
  (∀ c ∈ b.wilderness, c < 4) ∧ (∀ c ∈ b.soil, c < 40)

def Box.volume (b : Box) : Nat :=
  numericVolume b.axes * b.wilderness.length * b.soil.length

structure Cell where
  ranks : List Nat
  wilderness : Nat
  soil : Nat

def Box.Mem (b : Box) (x : Cell) : Prop :=
  axesMem b.axes x.ranks ∧ x.wilderness ∈ b.wilderness ∧ x.soil ∈ b.soil

def Box.cells (b : Box) : List Cell :=
  (numericCells b.axes).flatMap fun xs =>
    b.wilderness.flatMap fun w =>
      b.soil.map fun s => ⟨xs, w, s⟩

theorem box_cells_count (b : Box) : b.cells.length = b.volume := by
  unfold Box.cells
  rw [flatmap_fixed_length _ _ (b.wilderness.length * b.soil.length)]
  · simp [numeric_cells_count, Box.volume, Nat.mul_assoc]
  · intro xs _
    rw [flatmap_fixed_length _ _ b.soil.length]
    intro w _
    simp

theorem box_volume_positive (b : Box) (valid : b.Valid) : 0 < b.volume := by
  have hw : 0 < b.wilderness.length := by cases h : b.wilderness <;> simp_all [Box.Valid]
  have hs : 0 < b.soil.length := by cases h : b.soil <;> simp_all [Box.Valid]
  exact Nat.mul_pos (Nat.mul_pos (numeric_volume_positive b.axes) hw) hs



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

theorem box_cells_mem (b : Box) (valid : b.Valid) (x : Cell) :
    x ∈ b.cells ↔ b.Mem x := by
  have axesValid : ∀ a ∈ b.axes, a.Valid := by
    intro a ha
    exact (valid.1 a ha).1
  simp only [Box.cells, List.mem_flatMap, List.mem_map, Box.Mem]
  constructor
  · rintro ⟨xs, hx, w, hw, s, hs, heq⟩
    cases heq
    exact ⟨(numeric_cells_mem b.axes axesValid xs).mp hx, hw, hs⟩
  · rintro ⟨hx, hw, hs⟩
    exact ⟨x.ranks, (numeric_cells_mem b.axes axesValid x.ranks).mpr hx,
      x.wilderness, hw, x.soil, hs, by cases x; rfl⟩

theorem box_cells_nodup (b : Box) (valid : b.Valid) : b.cells.Nodup := by
  unfold Box.cells
  apply List.pairwise_flatMap.mpr
  constructor
  · intro xs _
    apply product_map_nodup b.wilderness b.soil (fun w s => Cell.mk xs w s)
      valid.2.2.2.1 valid.2.2.2.2.1
    intro w s w' s' heq
    cases heq
    exact ⟨rfl, rfl⟩
  · apply List.Pairwise.imp _ (numeric_cells_nodup b.axes)
    intro xs ys hxy a ha c hc heq
    rcases List.mem_flatMap.mp ha with ⟨w, _, ha⟩
    rcases List.mem_map.mp ha with ⟨s, _, rfl⟩
    rcases List.mem_flatMap.mp hc with ⟨w', _, hc⟩
    rcases List.mem_map.mp hc with ⟨s', _, rfl⟩
    exact hxy (congrArg Cell.ranks heq)

/-- The actual direct fallback's first-varying numeric-coordinate scan. -/
def firstVarying : List Interval → Option (List Interval × Interval × List Interval)
  | [] => none
  | a :: axes =>
    if a.lo < a.hi then some ([], a, axes)
    else match firstVarying axes with
      | none => none
      | some (pre, chosen, post) => some (a :: pre, chosen, post)

theorem first_varying_sound (axes pre : List Interval) (a : Interval) (post : List Interval)
    (h : firstVarying axes = some (pre, a, post)) :
    axes = pre ++ a :: post ∧ a.lo < a.hi ∧
      (∀ x ∈ pre, ¬x.lo < x.hi) := by
  induction axes generalizing pre with
  | nil => simp [firstVarying] at h
  | cons x xs ih =>
    simp only [firstVarying] at h
    split at h
    · rename_i hx
      cases h
      exact ⟨rfl, hx, by simp⟩
    · rename_i hx
      split at h
      · contradiction
      · rename_i pre' chosen post' hs
        cases h
        have inductionResult := ih pre' hs
        exact ⟨by simp [inductionResult.1], inductionResult.2.1, by
          intro y hy
          rcases List.mem_cons.mp hy with rfl | hy
          · exact hx
          · exact inductionResult.2.2 y hy⟩

theorem first_varying_none (axes : List Interval) :
    firstVarying axes = none ↔ ∀ a ∈ axes, ¬a.lo < a.hi := by
  induction axes with
  | nil => simp [firstVarying]
  | cons a axes ih =>
    simp only [firstVarying]
    by_cases h : a.lo < a.hi
    · simp only [h, ↓reduceIte, Option.some_ne_none, false_iff]
      intro hall
      exact hall a (by simp) h
    · simp only [h, ↓reduceIte]
      cases hv : firstVarying axes with
      | none =>
        have rest := ih.mp hv
        constructor
        · intro _ x hx
          rcases List.mem_cons.mp hx with rfl | hx
          · exact h
          · exact rest x hx
        · intro _
          rfl
      | some result =>
        rcases result with ⟨pre, chosen, post⟩
        constructor
        · intro absurd
          contradiction
        · intro hall
          have rest : ∀ x ∈ axes, ¬x.lo < x.hi := by
            intro x hx
            exact hall x (by simp [hx])
          have hn := ih.mpr rest
          simp [hv] at hn
theorem numeric_volume_atomic (axes : List Interval)
    (valid : ∀ a ∈ axes, a.Valid) (h : firstVarying axes = none) :
    numericVolume axes = 1 := by
  have all := (first_varying_none axes).mp h
  induction axes with
  | nil => rfl
  | cons a axes ih =>
    have ha := valid a (by simp)
    have hn := all a (by simp)
    have w : a.width = 1 := by
      unfold Interval.Valid at ha
      unfold Interval.width
      omega
    have vs : ∀ x ∈ axes, x.Valid := by
      intro x hx
      exact valid x (by simp [hx])
    have ns : ∀ x ∈ axes, ¬x.lo < x.hi := by
      intro x hx
      exact all x (by simp [hx])
    have nr := (first_varying_none axes).mpr ns
    simp only [numericVolume, List.map_cons, List.prod_cons]
    rw [w]
    simpa [numericVolume] using ih vs nr ns



def StrictPartition (parent left right : Box) : Prop :=
  left.Valid ∧ right.Valid ∧
  (∀ x, parent.Mem x ↔ left.Mem x ∨ right.Mem x) ∧
  (∀ x, ¬(left.Mem x ∧ right.Mem x)) ∧
  left.volume + right.volume = parent.volume

theorem numeric_child_valid (pre post : List Interval) (a : Interval) (cut : Nat)
    (w s : List Nat) (v : (Box.mk (pre ++ a :: post) w s).Valid)
    (strict : a.lo < cut ∧ cut ≤ a.hi) :
    (Box.mk (pre ++ a.left cut :: post) w s).Valid ∧
    (Box.mk (pre ++ a.right cut :: post) w s).Valid := by
  have va := v.1 a (by simp)
  have parts := interval_split_valid a cut strict
  constructor
  · refine ⟨?_, v.2⟩
    intro d hd
    simp only [List.mem_append, List.mem_cons] at hd
    rcases hd with hp | (rfl | hs)
    · exact v.1 d (by simp [hp])
    · exact ⟨parts.1, by simp only [Interval.left]; omega⟩
    · exact v.1 d (by simp [hs])
  · refine ⟨?_, v.2⟩
    intro d hd
    simp only [List.mem_append, List.mem_cons] at hd
    rcases hd with hp | (rfl | hs)
    · exact v.1 d (by simp [hp])
    · exact ⟨parts.2, va.2⟩
    · exact v.1 d (by simp [hs])

theorem numeric_box_partition (pre post : List Interval) (a : Interval) (cut : Nat)
    (w s : List Nat) (v : (Box.mk (pre ++ a :: post) w s).Valid)
    (strict : a.lo < cut ∧ cut ≤ a.hi) :
    StrictPartition (Box.mk (pre ++ a :: post) w s)
      (Box.mk (pre ++ a.left cut :: post) w s)
      (Box.mk (pre ++ a.right cut :: post) w s) := by
  have vs := numeric_child_valid pre post a cut w s v strict
  refine ⟨vs.1, vs.2, ?_, ?_, ?_⟩
  · intro x
    simp only [Box.Mem, numeric_split_cover pre post a cut strict x.ranks, or_and_right]
  · intro x h
    exact numeric_split_disjoint pre post a cut strict x.ranks ⟨h.1.1, h.2.1⟩
  · unfold Box.volume
    rw [← Nat.add_mul, ← Nat.add_mul, numeric_split_volume pre post a cut strict]

theorem wilderness_box_partition (axes : List Interval) (c : Nat) (rest s : List Nat)
    (v : (Box.mk axes (c :: rest) s).Valid) (nonempty : rest ≠ []) :
    StrictPartition (Box.mk axes (c :: rest) s)
      (Box.mk axes rest s) (Box.mk axes [c] s) := by
  have nc := (List.nodup_cons.mp v.2.2.2.1).1
  have nr := (List.nodup_cons.mp v.2.2.2.1).2
  have wc := v.2.2.2.2.2.1 c (by simp)
  have wr : ∀ i ∈ rest, i < 4 := by
    intro i hi
    exact v.2.2.2.2.2.1 i (by simp [hi])
  refine ⟨⟨v.1, nonempty, v.2.2.1, nr, v.2.2.2.2.1, wr, v.2.2.2.2.2.2⟩,
    ⟨v.1, by simp, v.2.2.1, by simp, v.2.2.2.2.1, ?_, v.2.2.2.2.2.2⟩, ?_, ?_, ?_⟩
  · intro i hi
    simpa using (List.mem_singleton.mp hi ▸ wc)
  · intro x
    simp only [Box.Mem, List.mem_cons, List.not_mem_nil, or_false]
    constructor
    · rintro ⟨ha, (heq | hr), hs⟩
      · exact Or.inr ⟨ha, heq, hs⟩
      · exact Or.inl ⟨ha, hr, hs⟩
    · rintro (⟨ha, hr, hs⟩ | ⟨ha, heq, hs⟩)
      · exact ⟨ha, Or.inr hr, hs⟩
      · exact ⟨ha, Or.inl heq, hs⟩
  · intro x h
    have hr := h.1.2.1
    have eq := List.mem_singleton.mp h.2.2.1
    exact nc (eq ▸ hr)
  · simp [Box.volume, Nat.mul_add, Nat.add_mul]

theorem soil_box_partition (axes : List Interval) (w : List Nat) (c : Nat) (rest : List Nat)
    (v : (Box.mk axes w (c :: rest)).Valid) (nonempty : rest ≠ []) :
    StrictPartition (Box.mk axes w (c :: rest))
      (Box.mk axes w rest) (Box.mk axes w [c]) := by
  have nc := (List.nodup_cons.mp v.2.2.2.2.1).1
  have nr := (List.nodup_cons.mp v.2.2.2.2.1).2
  have sc := v.2.2.2.2.2.2 c (by simp)
  have sr : ∀ i ∈ rest, i < 40 := by
    intro i hi
    exact v.2.2.2.2.2.2 i (by simp [hi])
  refine ⟨⟨v.1, v.2.1, nonempty, v.2.2.2.1, nr, v.2.2.2.2.2.1, sr⟩,
    ⟨v.1, v.2.1, by simp, v.2.2.2.1, by simp, v.2.2.2.2.2.1, ?_⟩, ?_, ?_, ?_⟩
  · intro i hi
    simpa using (List.mem_singleton.mp hi ▸ sc)
  · intro x
    simp only [Box.Mem, List.mem_cons, List.not_mem_nil, or_false]
    constructor
    · rintro ⟨ha, hw, (heq | hr)⟩
      · exact Or.inr ⟨ha, hw, heq⟩
      · exact Or.inl ⟨ha, hw, hr⟩
    · rintro (⟨ha, hw, hr⟩ | ⟨ha, hw, heq⟩)
      · exact ⟨ha, hw, Or.inr hr⟩
      · exact ⟨ha, hw, Or.inl heq⟩
  · intro x h
    have hr := h.1.2.2
    have eq := List.mem_singleton.mp h.2.2.2
    exact nc (eq ▸ hr)
  · simp [Box.volume, Nat.mul_add]

/-- Direct emitter fallback: numeric dimensions in order, then wilderness,
    then soil. Category supports from categorySupport are in bit-index order. -/
def fallback (b : Box) : Option (Box × Box) :=
  match firstVarying b.axes with
  | some (pre, a, post) =>
    some (⟨pre ++ a.left a.midpoint :: post, b.wilderness, b.soil⟩,
          ⟨pre ++ a.right a.midpoint :: post, b.wilderness, b.soil⟩)
  | none =>
    match b.wilderness with
    | c :: d :: rest => some (⟨b.axes, d :: rest, b.soil⟩, ⟨b.axes, [c], b.soil⟩)
    | _ => match b.soil with
      | c :: d :: rest => some (⟨b.axes, b.wilderness, d :: rest⟩,
                                ⟨b.axes, b.wilderness, [c]⟩)
      | _ => none

theorem fallback_partition (b l r : Box) (v : b.Valid) (h : fallback b = some (l,r)) :
    StrictPartition b l r := by
  unfold fallback at h
  cases hn : firstVarying b.axes with
  | some result =>
    rcases result with ⟨pre, a, post⟩
    simp only [hn] at h
    cases h
    have hs := first_varying_sound b.axes pre a post hn
    have vv : (Box.mk (pre ++ a :: post) b.wilderness b.soil).Valid := by
      simpa [← hs.1] using v
    have out := numeric_box_partition pre post a a.midpoint b.wilderness b.soil vv
      (midpoint_strict a hs.2.1)
    simpa only [← hs.1] using out
  | none =>
    simp only [hn] at h
    cases hw : b.wilderness with
    | nil => exact False.elim (v.2.1 hw)
    | cons c rest =>
      cases rest with
      | cons d rest =>
        simp only [hw] at h
        cases h
        have vv : (Box.mk b.axes (c :: d :: rest) b.soil).Valid := by simpa [← hw] using v
        have out := wilderness_box_partition b.axes c (d::rest) b.soil vv (by simp)
        simpa only [← hw] using out
      | nil =>
        simp only [hw] at h
        cases hs : b.soil with
        | nil => exact False.elim (v.2.2.1 hs)
        | cons c rest =>
          cases rest with
          | nil => simp [hs] at h
          | cons d rest =>
            simp only [hs] at h
            cases h
            have vv : (Box.mk b.axes b.wilderness (c :: d :: rest)).Valid := by
              simpa [← hs] using v
            have out := soil_box_partition b.axes b.wilderness c (d::rest) vv (by simp)
            simpa only [← hs, ← hw] using out

theorem fallback_none_atomic (b : Box) (v : b.Valid) (h : fallback b = none) :
    b.volume = 1 := by
  unfold fallback at h
  cases hn : firstVarying b.axes with
  | some result =>
    rcases result with ⟨pre,a,post⟩
    simp [hn] at h
  | none =>
    simp only [hn] at h
    have av : ∀ a ∈ b.axes, a.Valid := fun a ha => (v.1 a ha).1
    have nv := numeric_volume_atomic b.axes av hn
    cases hw : b.wilderness with
    | nil => exact False.elim (v.2.1 hw)
    | cons c rest =>
      cases rest with
      | cons d rest => simp [hw] at h
      | nil =>
        simp only [hw] at h
        cases hs : b.soil with
        | nil => exact False.elim (v.2.2.1 hs)
        | cons c rest =>
          cases rest with
          | cons d rest => simp [hs] at h
          | nil => simp [Box.volume, nv, hw, hs]

theorem every_nonatomic_box_splits (b : Box) (v : b.Valid) (nonatom : 1 < b.volume) :
    ∃ l r, fallback b = some (l,r) ∧ StrictPartition b l r := by
  cases h : fallback b with
  | none =>
    have atom := fallback_none_atomic b v h
    omega
  | some lr =>
    exact ⟨lr.1, lr.2, rfl, fallback_partition b lr.1 lr.2 v h⟩

theorem partition_strict_volumes (b l r : Box) (part : StrictPartition b l r) :
    0 < l.volume ∧ l.volume < b.volume ∧ 0 < r.volume ∧ r.volume < b.volume := by
  have lp := box_volume_positive l part.1
  have rp := box_volume_positive r part.2.1
  have sum := part.2.2.2.2
  omega

theorem fallback_potential_decreases (b l r : Box) (v : b.Valid)
    (h : fallback b = some (l,r)) :
    (2 * l.volume - 1) + (2 * r.volume - 1) + 1 = 2 * b.volume - 1 := by
  have part := fallback_partition b l r v h
  have sizes := partition_strict_volumes b l r part
  have sum := part.2.2.2.2
  omega



inductive Question where
  | numeric : Nat → Nat → Question
  | wilderness : Nat → Question
  | soil : Nat → Question
deriving DecidableEq

def numericBelow : Nat → Nat → List Nat → Prop
  | 0, cut, x :: _ => x < cut
  | n + 1, cut, _ :: xs => numericBelow n cut xs
  | _, _, [] => False

def Question.Left (q : Question) (x : Cell) : Prop :=
  match q with
  | .numeric feature cut => numericBelow feature cut x.ranks
  | .wilderness bit => x.wilderness ≠ bit
  | .soil bit => x.soil ≠ bit

instance numericBelowDecidable (i c : Nat) (xs : List Nat) : Decidable (numericBelow i c xs) := by
  induction i generalizing xs with
  | zero => cases xs <;> simp only [numericBelow] <;> infer_instance
  | succ i ih =>
    cases xs with
    | nil => exact isFalse (fun h => h)
    | cons x xs => exact ih xs

instance questionLeftDecidable (q : Question) (x : Cell) : Decidable (q.Left x) := by
  cases q <;> simp only [Question.Left] <;> infer_instance

theorem numeric_split_routing (pre post : List Interval) (a : Interval) (cut : Nat)
    (strict : a.lo < cut ∧ cut ≤ a.hi) (xs : List Nat) :
    (axesMem (pre ++ a.left cut :: post) xs ↔
      axesMem (pre ++ a :: post) xs ∧ numericBelow pre.length cut xs) ∧
    (axesMem (pre ++ a.right cut :: post) xs ↔
      axesMem (pre ++ a :: post) xs ∧ ¬numericBelow pre.length cut xs) := by
  induction pre generalizing xs with
  | nil =>
    cases xs with
    | nil => simp [axesMem, numericBelow]
    | cons x xs =>
      simp only [List.nil_append, axesMem, List.length_nil, numericBelow]
      have h := interval_split_membership a cut x strict
      rw [h.1, h.2]
      constructor <;> constructor <;> intro h <;> exact ⟨⟨h.1.1,h.2⟩,h.1.2⟩
  | cons p pre ih =>
    cases xs with
    | nil => simp [axesMem, numericBelow]
    | cons x xs =>
      simp only [List.cons_append, axesMem, List.length_cons, numericBelow]
      rw [(ih xs).1, (ih xs).2]
      constructor <;> exact and_assoc.symm

def fallbackQuestion (b : Box) : Option Question :=
  match firstVarying b.axes with
  | some (pre, a, _) => some (.numeric pre.length a.midpoint)
  | none =>
    match b.wilderness with
    | c :: _ :: _ => some (.wilderness c)
    | _ => match b.soil with
      | c :: _ :: _ => some (.soil c)
      | _ => none

theorem fallback_predicate_routing (b l r : Box) (v : b.Valid)
    (h : fallback b = some (l,r)) :
    ∃ q, fallbackQuestion b = some q ∧
      (∀ x, l.Mem x ↔ b.Mem x ∧ q.Left x) ∧
      (∀ x, r.Mem x ↔ b.Mem x ∧ ¬q.Left x) := by
  unfold fallback at h
  cases hn : firstVarying b.axes with
  | some result =>
    rcases result with ⟨pre,a,post⟩
    simp only [hn] at h
    cases h
    have hs := first_varying_sound b.axes pre a post hn
    refine ⟨.numeric pre.length a.midpoint, by simp [fallbackQuestion,hn], ?_, ?_⟩
    · intro x
      have route := (numeric_split_routing pre post a a.midpoint
        (midpoint_strict a hs.2.1) x.ranks).1
      simp only [Box.Mem, Question.Left, hs.1, route]
      constructor
      · rintro ⟨⟨ha,hq⟩,hw,hs⟩
        exact ⟨⟨ha,hw,hs⟩,hq⟩
      · rintro ⟨⟨ha,hw,hs⟩,hq⟩
        exact ⟨⟨ha,hq⟩,hw,hs⟩
    · intro x
      have route := (numeric_split_routing pre post a a.midpoint
        (midpoint_strict a hs.2.1) x.ranks).2
      simp only [Box.Mem, Question.Left, hs.1, route]
      constructor
      · rintro ⟨⟨ha,hq⟩,hw,hs⟩
        exact ⟨⟨ha,hw,hs⟩,hq⟩
      · rintro ⟨⟨ha,hw,hs⟩,hq⟩
        exact ⟨⟨ha,hq⟩,hw,hs⟩
  | none =>
    simp only [hn] at h
    cases hw : b.wilderness with
    | nil => exact False.elim (v.2.1 hw)
    | cons c rest =>
      cases rest with
      | cons d rest =>
        simp only [hw] at h
        cases h
        have nc : c ∉ d::rest := (List.nodup_cons.mp (hw ▸ v.2.2.2.1)).1
        refine ⟨.wilderness c, by simp [fallbackQuestion,hn,hw], ?_, ?_⟩
        · intro x
          simp only [Box.Mem, hw, Question.Left, List.mem_cons]
          constructor
          · rintro ⟨ha,hr,hs⟩
            refine ⟨⟨ha,Or.inr hr,hs⟩,?_⟩
            intro eq
            exact nc (eq ▸ List.mem_cons.mpr hr)
          · rintro ⟨⟨ha,(heq|hr),hs⟩,ne⟩
            · exact False.elim (ne heq)
            · exact ⟨ha,hr,hs⟩
        · intro x
          simp only [Box.Mem, hw, Question.Left, List.mem_cons, List.not_mem_nil, or_false, Decidable.not_not]
          constructor
          · rintro ⟨ha,heq,hs⟩
            exact ⟨⟨ha,Or.inl heq,hs⟩,heq⟩
          · rintro ⟨⟨ha,_,hs⟩,heq⟩
            exact ⟨ha,heq,hs⟩
      | nil =>
        simp only [hw] at h
        cases hs : b.soil with
        | nil => exact False.elim (v.2.2.1 hs)
        | cons c rest =>
          cases rest with
          | nil => simp [hs] at h
          | cons d rest =>
            simp only [hs] at h
            cases h
            have nc : c ∉ d::rest := (List.nodup_cons.mp (hs ▸ v.2.2.2.2.1)).1
            refine ⟨.soil c, by simp [fallbackQuestion,hn,hw,hs], ?_, ?_⟩
            · intro x
              simp only [Box.Mem, hw, hs, Question.Left, List.mem_cons]
              constructor
              · rintro ⟨ha,hw',hr⟩
                refine ⟨⟨ha,hw',Or.inr hr⟩,?_⟩
                intro eq
                exact nc (eq ▸ List.mem_cons.mpr hr)
              · rintro ⟨⟨ha,hw',(heq|hr)⟩,ne⟩
                · exact False.elim (ne heq)
                · exact ⟨ha,hw',hr⟩
            · intro x
              simp only [Box.Mem, hw, hs, Question.Left, List.mem_cons, List.not_mem_nil, or_false, Decidable.not_not]
              constructor
              · rintro ⟨ha,hw',heq⟩
                exact ⟨⟨ha,hw',Or.inl heq⟩,heq⟩
              · rintro ⟨⟨ha,hw',_⟩,heq⟩
                exact ⟨ha,hw',heq⟩

theorem atomic_box_unique_cell (b : Box) (v : b.Valid) (atom : b.volume = 1) :
    ∃ x, b.Mem x ∧ ∀ y, b.Mem y → y = x := by
  have count := box_cells_count b
  rw [atom] at count
  cases hc : b.cells with
  | nil => simp [hc] at count
  | cons x xs =>
    have tail : xs = [] := by
      simpa [hc] using count
    refine ⟨x, (box_cells_mem b v x).mp ?_, ?_⟩
    · simp [hc,tail]
    · intro y hy
      have mem := (box_cells_mem b v y).mpr hy
      simpa [hc,tail] using mem



theorem category_support_sorted (n mask : Nat) :
    (categorySupport n mask).Pairwise (· < ·) := by
  exact List.Pairwise.filter _ (List.pairwise_lt_range (n := n))

theorem support_head_is_lowest (n mask c : Nat) (rest : List Nat)
    (h : categorySupport n mask = c :: rest) :
    c < n ∧ mask.testBit c = true ∧ ∀ j ∈ rest, c < j := by
  have hc : c ∈ categorySupport n mask := by simp [h]
  have mem := (category_support_mem n mask c).mp hc
  have sorted := category_support_sorted n mask
  rw [h] at sorted
  exact ⟨mem.1,mem.2,(List.pairwise_cons.mp sorted).1⟩

theorem numeric_witness (axes : List Interval) (v : ∀ a ∈ axes, a.Valid) :
    axesMem axes (axes.map Interval.lo) := by
  induction axes with
  | nil => trivial
  | cons a axes ih =>
    have va := v a (by simp)
    have vs : ∀ x ∈ axes, x.Valid := by
      intro x hx
      exact v x (by simp [hx])
    exact ⟨⟨Nat.le_refl _,va⟩,ih vs⟩

def Box.witness (b : Box) : Cell :=
  ⟨b.axes.map Interval.lo,b.wilderness.headD 0,b.soil.headD 0⟩

theorem box_witness_mem (b : Box) (v : b.Valid) : b.Mem b.witness := by
  refine ⟨numeric_witness b.axes (fun a ha => (v.1 a ha).1),?_,?_⟩
  · cases hw : b.wilderness with
    | nil => exact False.elim (v.2.1 hw)
    | cons c rest => simp [Box.witness,hw]
  · cases hs : b.soil with
    | nil => exact False.elim (v.2.2.1 hs)
    | cons c rest => simp [Box.witness,hs]

theorem atomic_witness_unique (b : Box) (v : b.Valid) (atom : b.volume = 1)
    (x : Cell) (hx : b.Mem x) : x = b.witness := by
  rcases atomic_box_unique_cell b v atom with ⟨y,_,unique⟩
  exact (unique x hx).trans (unique b.witness (box_witness_mem b v)).symm

def Box.question (b : Box) : Question :=
  (fallbackQuestion b).getD (.numeric 0 0)

theorem selected_question_routing (b l r : Box) (v : b.Valid)
    (h : fallback b = some (l,r)) :
    (∀ x, l.Mem x ↔ b.Mem x ∧ b.question.Left x) ∧
    (∀ x, r.Mem x ↔ b.Mem x ∧ ¬b.question.Left x) := by
  rcases fallback_predicate_routing b l r v h with ⟨q,hq,hl,hr⟩
  simpa [Box.question,hq] using And.intro hl hr


end ConverterRankBox