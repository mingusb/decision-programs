import SignatureEnumeration

/- Exact finite Cartesian cell enumeration. All arithmetic here is unbounded Nat/Int;
   machine overflow checks and IEEE/CUDA refinement remain implementation duties. -/
namespace ConverterMixedRadix
open ConverterRankBox ConverterGuarantees ConverterSignatureEnumeration
variable {P W L X : Type}

def volume : List Nat → Nat
  | [] => 1
  | b :: bs => b * volume bs

def Within : List Nat → List Nat → Prop
  | [], [] => True
  | b :: bs, d :: ds => d < b ∧ Within bs ds
  | _, _ => False

/-- The final dimension varies fastest, as in the device wire convention. -/
def decode : List Nat → Nat → List Nat
  | [], _ => []
  | _ :: bs, i => i / volume bs :: decode bs (i % volume bs)

def encode : List Nat → List Nat → Nat
  | [], [] => 0
  | _ :: bs, d :: ds => volume bs * d + encode bs ds
  | _, _ => 0

def cells : List Nat → List (List Nat)
  | [] => [[]]
  | b :: bs => (List.range b).flatMap fun d => (cells bs).map (d :: ·)

theorem volume_positive (bs : List Nat) (positive : ∀ b ∈ bs, 0 < b) :
    0 < volume bs := by
  induction bs with
  | nil => simp [volume]
  | cons b bs ih =>
    exact Nat.mul_pos (positive b (by simp)) (ih (by intro x hx; exact positive x (by simp [hx])))

theorem decode_within (bs : List Nat) (positive : ∀ b ∈ bs, 0 < b)
    (i : Nat) (bound : i < volume bs) : Within bs (decode bs i) := by
  induction bs generalizing i with
  | nil => simp [decode, Within]
  | cons b bs ih =>
    have pos : ∀ x ∈ bs, 0 < x := by intro x hx; exact positive x (by simp [hx])
    have tailpos := volume_positive bs pos
    exact ⟨(Nat.div_lt_iff_lt_mul tailpos).mpr bound, ih pos _ (Nat.mod_lt i tailpos)⟩

theorem encode_decode (bs : List Nat) (positive : ∀ b ∈ bs, 0 < b)
    (i : Nat) (bound : i < volume bs) : encode bs (decode bs i) = i := by
  induction bs generalizing i with
  | nil => simp [volume] at bound; simp [encode, decode, bound]
  | cons b bs ih =>
    have pos : ∀ x ∈ bs, 0 < x := by intro x hx; exact positive x (by simp [hx])
    simp only [decode, encode, ih pos _ (Nat.mod_lt i (volume_positive bs pos))]
    exact Nat.div_add_mod i (volume bs)

theorem encode_bound (bs ds : List Nat) (valid : Within bs ds) :
    encode bs ds < volume bs := by
  induction bs generalizing ds with
  | nil => cases ds <;> simp_all [Within, encode, volume]
  | cons b bs ih =>
    cases ds with
    | nil => simp [Within] at valid
    | cons d ds =>
      have ht := ih ds valid.2
      have hm := Nat.mul_le_mul_left (volume bs) (Nat.succ_le_of_lt valid.1)
      simp only [Nat.mul_succ] at hm
      simp only [encode, volume]
      rw [Nat.mul_comm b]
      omega

theorem decode_encode (bs ds : List Nat) (positive : ∀ b ∈ bs, 0 < b)
    (valid : Within bs ds) : decode bs (encode bs ds) = ds := by
  induction bs generalizing ds with
  | nil => cases ds <;> simp_all [Within, decode]
  | cons b bs ih =>
    cases ds with
    | nil => simp [Within] at valid
    | cons d ds =>
      have pos : ∀ x ∈ bs, 0 < x := by intro x hx; exact positive x (by simp [hx])
      have tp := volume_positive bs pos
      have bound := encode_bound bs ds valid.2
      simp only [encode, decode, Nat.mul_add_div tp, Nat.mul_add_mod,
        Nat.div_eq_of_lt bound, Nat.mod_eq_of_lt bound, Nat.add_zero, ih ds pos valid.2]

theorem decode_injective (bs : List Nat) (positive : ∀ b ∈ bs, 0 < b)
    (i j : Nat) (hi : i < volume bs) (hj : j < volume bs)
    (same : decode bs i = decode bs j) : i = j := by
  rw [← encode_decode bs positive i hi, ← encode_decode bs positive j hj, same]

theorem mixed_radix_complete (bs ds : List Nat) (positive : ∀ b ∈ bs, 0 < b)
    (valid : Within bs ds) : ∃ i, i < volume bs ∧ decode bs i = ds := by
  exact ⟨encode bs ds, encode_bound bs ds valid, decode_encode bs ds positive valid⟩

theorem cells_membership (bs ds : List Nat) : ds ∈ cells bs ↔ Within bs ds := by
  induction bs generalizing ds with
  | nil => cases ds <;> simp [cells, Within]
  | cons b bs ih => cases ds <;> simp [cells, Within, ih]

theorem cells_count (bs : List Nat) : (cells bs).length = volume bs := by
  induction bs with
  | nil => simp [cells, volume]
  | cons b bs ih =>
    rw [cells, flatmap_fixed_length _ _ (volume bs)]
    · simp [volume]
    · intro d _; simp [ih]

theorem cells_nodup (bs : List Nat) : (cells bs).Nodup := by
  induction bs with
  | nil => simp [cells]
  | cons b bs ih =>
    exact product_map_nodup (List.range b) (cells bs) (fun x xs => x :: xs)
      List.nodup_range ih (by intro a b c d h; exact List.cons.inj h)

/-- A concrete lower endpoint chosen among the minimum finite key and actual cuts. -/
def endpoint (minimum x : Int) : List Int → Int
  | [] => minimum
  | c :: cs => if c ≤ x then max c (endpoint minimum x cs) else endpoint minimum x cs

theorem endpoint_bounds (minimum x : Int) (cs : List Int) (valid : minimum ≤ x) :
    minimum ≤ endpoint minimum x cs ∧ endpoint minimum x cs ≤ x := by
  induction cs with
  | nil => simp [endpoint, valid]
  | cons c cs ih =>
    simp only [endpoint]
    split
    · rename_i h; rw [Int.max_def]; split <;> omega
    · exact ih

theorem endpoint_is_stored (minimum x : Int) (cs : List Int) :
    endpoint minimum x cs = minimum ∨ endpoint minimum x cs ∈ cs := by
  induction cs with
  | nil => simp [endpoint]
  | cons c cs ih =>
    simp only [endpoint]
    split
    · rw [Int.max_def]; split
      · rcases ih with h | h
        · exact Or.inl h
        · exact Or.inr (by simp [h])
      · exact Or.inr (by simp)
    · rcases ih with h | h
      · exact Or.inl h
      · exact Or.inr (by simp [h])

theorem endpoint_preserves_cut (minimum x : Int) (cs : List Int)
    (valid : minimum ≤ x) (c : Int) (member : c ∈ cs) :
    c ≤ endpoint minimum x cs ↔ c ≤ x := by
  constructor
  · intro h; exact Int.le_trans h (endpoint_bounds minimum x cs valid).2
  · intro h
    induction cs with
    | nil => simp at member
    | cons a cs ih =>
      rcases List.mem_cons.mp member with eq | mem
      · subst a; simp only [endpoint, h, ↓reduceIte]; exact Int.le_max_left _ _
      · have ht := ih mem
        simp only [endpoint]
        split
        · exact Int.le_trans ht (Int.le_max_right _ _)
        · exact ht

theorem endpoint_preserves_signature (minimum x : Int) (cs : List Int)
    (valid : minimum ≤ x) :
    ∀ c ∈ cs, decide (endpoint minimum x cs < c) = decide (x < c) := by
  intro c hc
  have h := endpoint_preserves_cut minimum x cs valid c hc
  by_cases hxc : c ≤ x
  · have hr := h.mpr hxc; simp [Int.not_lt.mpr hxc, Int.not_lt.mpr hr]
  · have hx : x < c := by omega
    have hr : endpoint minimum x cs < c := by omega
    simp [hx, hr]

theorem endpoint_preserves_rank (minimum x : Int) (cs : List Int)
    (valid : minimum ≤ x) : rank (· ≤ ·) cs (endpoint minimum x cs) = rank (· ≤ ·) cs x := by
  exact rank_comparison_congr (· ≤ ·) cs _ _ (endpoint_preserves_cut minimum x cs valid)

/-- At the cut at zero-based position k, its rank is k+1, for strictly sorted keys. -/
theorem sorted_cut_rank (pre post : List Int) (c : Int)
    (sorted : (pre ++ c :: post).Pairwise (· < ·)) :
    rank (· ≤ ·) (pre ++ c :: post) c = pre.length + 1 := by
  have hp : ∀ t ∈ pre, t ≤ c := by
    intro t ht
    have h := (List.pairwise_append.mp sorted).2.2 t ht c (by simp)
    omega
  have hs : ∀ t ∈ post, ¬ t ≤ c := by
    intro t ht
    have h := (List.pairwise_cons.mp (List.pairwise_append.mp sorted).2.1).1 t ht
    omega
  simp [rank_append, rank, rank_all_yes (· ≤ ·) pre c hp, rank_all_no (· ≤ ·) post c hs]

theorem minimum_cut_excludes_rank_zero (minimum x : Int) (cs : List Int)
    (valid : minimum ≤ x) : 0 < rank (· ≤ ·) (minimum :: cs) x := by
  simp only [rank, valid, ↓reduceIte]; omega

theorem minimum_witness_rank_zero (minimum : Int) (cs : List Int)
    (above : ∀ c ∈ cs, minimum < c) : rank (· ≤ ·) cs minimum = 0 := by
  apply rank_all_no (· ≤ ·) cs minimum
  intro c hc; have h := above c hc; omega

/-- Category state representative: tested categories retain their own singleton;
    the OTHER class uses one untested representative. -/
def categoryRep (tested : List Nat) (other x : Nat) : Nat := if x ∈ tested then x else other

def indicatorKey (unit : Int) (category question : Nat) : Int :=
  if category = question then unit else 0

theorem category_rep_valid (tested : List Nat) (other x n : Nat)
    (hx : x < n) (ho : x ∉ tested → other < n) : categoryRep tested other x < n := by
  simp only [categoryRep]; split
  · exact hx
  · rename_i h; exact ho h

theorem category_rep_tested_equality (tested : List Nat) (other x q : Nat)
    (hq : q ∈ tested) (ho : x ∉ tested → other ∉ tested) :
    categoryRep tested other x = q ↔ x = q := by
  unfold categoryRep; split
  · rfl
  · rename_i hx
    have hne : other ≠ q := by intro e; exact ho hx (e ▸ hq)
    have hxn : x ≠ q := by intro e; exact hx (e ▸ hq)
    simp [hne, hxn]

theorem category_question_preserved (tested : List Nat) (other x q : Nat)
    (unit cut : Int) (unitpos : 0 < unit)
    (covered : 0 < cut ∧ cut ≤ unit → q ∈ tested)
    (ho : x ∉ tested → other ∉ tested) :
    decide (indicatorKey unit (categoryRep tested other x) q < cut) =
      decide (indicatorKey unit x q < cut) := by
  by_cases varying : 0 < cut ∧ cut ≤ unit
  · have h := category_rep_tested_equality tested other x q (covered varying) ho
    simp only [indicatorKey]
    by_cases hx : x = q
    · subst x; have hr := h.mpr rfl; simp [hr]
    · have hn : categoryRep tested other x ≠ q := fun he => hx (h.mp he)
      simp [hx, hn]
  · unfold indicatorKey
    split <;> split <;> simp_all <;> omega

theorem tested_category_fixed (tested : List Nat) (other x : Nat) (hx : x ∈ tested) :
    categoryRep tested other x = x := by simp [categoryRep, hx]

theorem other_category_constant (tested : List Nat) (other x y : Nat)
    (hx : x ∉ tested) (hy : y ∉ tested) :
    categoryRep tested other x = categoryRep tested other y := by simp [categoryRep, hx, hy]

/-- Once the concrete numeric/category lemmas provide per-source-question
    agreement, ordered source leaf words and the native class are inherited.
    No CPU probability transform or arithmetic reassociation is introduced. -/
theorem cartesian_representative_native (truth : P → X → Bool)
    (questions : List P) (forest : List (Tree P W))
    (covers : ∀ tree ∈ forest, ∀ p ∈ predicates tree, p ∈ questions)
    (nativeTransform : List W → L) (representative : X → X)
    (same_questions : ∀ x, ∀ p ∈ questions, truth p (representative x) = truth p x) :
    ∀ x, nativeTransform (orderedWords truth forest (representative x)) =
      nativeTransform (orderedWords truth forest x) := by
  intro x
  apply same_signature_same_native truth questions forest covers nativeTransform
  exact (signature_eq_iff truth questions _ _).mpr (same_questions x)


/-- The finite binary32 key interval includes fractions/subnormals and collapses
    only the two encodings of zero, never distinct comparison values. -/
theorem finite_word_key_bounds (x : FiniteWord) :
    -2139095039 ≤ orderedKey x ∧ orderedKey x ≤ 2139095039 := by
  have h := x.magnitude.isLt
  unfold orderedKey
  simp only [Int.ofNat_eq_natCast]
  split <;> omega

theorem finite_key_has_word (k : Int)
    (lo : -2139095039 ≤ k) (hi : k ≤ 2139095039) :
    ∃ x : FiniteWord, orderedKey x = k := by
  by_cases negative : k < 0
  · have h : (-k).toNat < 2139095040 := by omega
    refine ⟨⟨true, ⟨(-k).toNat, h⟩⟩, ?_⟩
    simp only [orderedKey, ↓reduceIte, Int.ofNat_eq_natCast]
    omega
  · have h : k.toNat < 2139095040 := by omega
    refine ⟨⟨false, ⟨k.toNat, h⟩⟩, ?_⟩
    simp only [orderedKey, Bool.false_eq_true, ↓reduceIte, Int.ofNat_eq_natCast]
    omega

structure RawKeys where
  numeric : Fin 10 → Int
  wilderness : Nat
  soil : Nat

def RawKeys.Valid (x : RawKeys) : Prop :=
  (∀ f, -2139095039 ≤ x.numeric f ∧ x.numeric f ≤ 2139095039) ∧
  x.wilderness < 4 ∧ x.soil < 40

inductive Question where
  | numeric : Fin 10 → Int → Question
  | wilderness : Fin 4 → Int → Question
  | soil : Fin 40 → Int → Question

def truth : Question → RawKeys → Bool
  | .numeric f c, x => decide (x.numeric f < c)
  | .wilderness q c, x => decide (indicatorKey 1065353216 x.wilderness q.val < c)
  | .soil q c, x => decide (indicatorKey 1065353216 x.soil q.val < c)

structure Profile where
  cuts : Fin 10 → List Int
  wilderness : List Nat
  soil : List Nat
  otherWild : Nat
  otherSoil : Nat

/-- If OTHER is needed it is a valid untested category. If all categories are
    tested the antecedent is impossible, and no OTHER state is needed. -/
def Profile.Valid (p : Profile) : Prop :=
  (∀ x, x < 4 → x ∉ p.wilderness → p.otherWild < 4 ∧ p.otherWild ∉ p.wilderness) ∧
  (∀ x, x < 40 → x ∉ p.soil → p.otherSoil < 40 ∧ p.otherSoil ∉ p.soil)

def Profile.Covers (p : Profile) : Question → Prop
  | .numeric f c => c ∈ p.cuts f
  | .wilderness q c => 0 < c ∧ c ≤ 1065353216 → q.val ∈ p.wilderness
  | .soil q c => 0 < c ∧ c ≤ 1065353216 → q.val ∈ p.soil

def representative (p : Profile) (x : RawKeys) : RawKeys :=
  ⟨fun f => endpoint (-2139095039) (x.numeric f) (p.cuts f),
   categoryRep p.wilderness p.otherWild x.wilderness,
   categoryRep p.soil p.otherSoil x.soil⟩

theorem representative_valid (p : Profile) (pv : p.Valid) (x : RawKeys)
    (valid : x.Valid) : (representative p x).Valid := by
  constructor
  · intro f
    have h := endpoint_bounds (-2139095039) (x.numeric f) (p.cuts f) (valid.1 f).1
    exact ⟨h.1, Int.le_trans h.2 (valid.1 f).2⟩
  · constructor
    · exact category_rep_valid p.wilderness p.otherWild x.wilderness 4 valid.2.1
        (fun h => (pv.1 _ valid.2.1 h).1)
    · exact category_rep_valid p.soil p.otherSoil x.soil 40 valid.2.2
        (fun h => (pv.2 _ valid.2.2 h).1)

theorem representative_preserves_question (p : Profile) (pv : p.Valid)
    (x : RawKeys) (valid : x.Valid) (q : Question) (covered : p.Covers q) :
    truth q (representative p x) = truth q x := by
  cases q with
  | numeric f c =>
    exact endpoint_preserves_signature (-2139095039) (x.numeric f) (p.cuts f)
      (valid.1 f).1 c covered
  | wilderness q c =>
    exact category_question_preserved p.wilderness p.otherWild x.wilderness q.val
      1065353216 c (by decide) covered (fun h => (pv.1 _ valid.2.1 h).2)
  | soil q c =>
    exact category_question_preserved p.soil p.otherSoil x.soil q.val
      1065353216 c (by decide) covered (fun h => (pv.2 _ valid.2.2 h).2)

theorem concrete_grid_same_ordered_words (p : Profile) (pv : p.Valid)
    (forest : List (Tree Question W))
    (covers : ∀ tree ∈ forest, ∀ q ∈ predicates tree, p.Covers q)
    (x : RawKeys) (valid : x.Valid) :
    orderedWords truth forest (representative p x) = orderedWords truth forest x := by
  apply List.map_congr_left
  intro tree ht
  apply same_questions_same_leaf
  intro q hq
  exact representative_preserves_question p pv x valid q (covers tree ht q hq)

theorem concrete_grid_same_native (p : Profile) (pv : p.Valid)
    (forest : List (Tree Question W))
    (covers : ∀ tree ∈ forest, ∀ q ∈ predicates tree, p.Covers q)
    (nativeTransform : List W → L) (x : RawKeys) (valid : x.Valid) :
    nativeTransform (orderedWords truth forest (representative p x)) =
      nativeTransform (orderedWords truth forest x) := by
  rw [concrete_grid_same_ordered_words p pv forest covers x valid]

/-- A finite audit of all M encoded cells implies whole-source native classes.
    The recipe is a metadata/index correspondence, not an assumed class equality. -/
theorem finite_cartesian_native_audit (p : Profile) (pv : p.Valid)
    (forest : List (Tree Question W))
    (covers : ∀ tree ∈ forest, ∀ q ∈ predicates tree, p.Covers q)
    (nativeTransform : List W → L) (radices : List Nat)
    (positive : ∀ r ∈ radices, 0 < r)
    (coordinates : RawKeys → List Nat) (sample : List Nat → RawKeys)
    (within : ∀ x, x.Valid → Within radices (coordinates x))
    (recipe : ∀ x, x.Valid → sample (coordinates x) = representative p x)
    (table : Nat → L)
    (audited : ∀ i, i < volume radices → table i =
      nativeTransform (orderedWords truth forest (sample (decode radices i)))) :
    ∀ x, x.Valid → table (encode radices (coordinates x)) =
      nativeTransform (orderedWords truth forest x) := by
  intro x hx
  rw [audited _ (encode_bound radices (coordinates x) (within x hx))]
  rw [decode_encode radices (coordinates x) positive (within x hx), recipe x hx]
  exact concrete_grid_same_native p pv forest covers nativeTransform x hx

/-- For the unshared binary fold with exactly one leaf per cell, this identity
    charges occurrences before hash-consing; it is independent of virtual UInt64. -/
theorem full_grid_binary_node_count (tree : Tree P L) (M : Nat)
    (cell_leaves : leaves tree = M) : nodes tree + 1 = 2 * M := by
  rw [physical_node_leaf_identity, cell_leaves]

end ConverterMixedRadix
