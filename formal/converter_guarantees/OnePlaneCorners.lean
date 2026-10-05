import Lean

/- Exact one-sided affine feasibility over independent integer intervals.
   Machine FP64 exactness and native class certification are separate refinements. -/
namespace ConverterOnePlane

structure Coordinate where
  weight : Int
  lo : Int
  hi : Int

def Valid (box : List Coordinate) : Prop := ∀ c ∈ box, c.lo ≤ c.hi

def Inside : List Coordinate → List Int → Prop
  | [], [] => True
  | c :: box, x :: xs => c.lo ≤ x ∧ x ≤ c.hi ∧ Inside box xs
  | _, _ => False

def score : List Coordinate → List Int → Int
  | c :: box, x :: xs => c.weight*x + score box xs
  | _, _ => 0

def lowerCorner (box : List Coordinate) : List Int :=
  box.map fun c => if c.weight < 0 then c.hi else c.lo
def upperCorner (box : List Coordinate) : List Int :=
  box.map fun c => if c.weight < 0 then c.lo else c.hi

theorem lower_corner_inside (box : List Coordinate) (valid : Valid box) :
    Inside box (lowerCorner box) := by
  induction box with
  | nil => trivial
  | cons c box ih =>
    have hc := valid c (by simp)
    have vb : Valid box := by
      intro d hd
      exact valid d (by simp [hd])
    simp only [lowerCorner,List.map_cons,Inside]
    split <;> exact ⟨by omega,by omega,ih vb⟩

theorem upper_corner_inside (box : List Coordinate) (valid : Valid box) :
    Inside box (upperCorner box) := by
  induction box with
  | nil => trivial
  | cons c box ih =>
    have hc := valid c (by simp)
    have vb : Valid box := by
      intro d hd
      exact valid d (by simp [hd])
    simp only [upperCorner,List.map_cons,Inside]
    split <;> exact ⟨by omega,by omega,ih vb⟩

theorem coordinate_extrema (c : Coordinate) (x : Int)
    (inside : c.lo ≤ x ∧ x ≤ c.hi) :
    c.weight*(if c.weight<0 then c.hi else c.lo) ≤ c.weight*x ∧
    c.weight*x ≤ c.weight*(if c.weight<0 then c.lo else c.hi) := by
  by_cases neg : c.weight < 0
  · simp only [neg,↓reduceIte]
    exact ⟨Int.mul_le_mul_of_nonpos_left (by omega) inside.2,
      Int.mul_le_mul_of_nonpos_left (by omega) inside.1⟩
  · simp only [neg,↓reduceIte]
    exact ⟨Int.mul_le_mul_of_nonneg_left inside.1 (by omega),
      Int.mul_le_mul_of_nonneg_left inside.2 (by omega)⟩

theorem exact_corner_bounds (box : List Coordinate) (xs : List Int)
    (inside : Inside box xs) :
    score box (lowerCorner box) ≤ score box xs ∧
    score box xs ≤ score box (upperCorner box) := by
  induction box generalizing xs with
  | nil => cases xs <;> simp_all [Inside,score]
  | cons c box ih =>
    cases xs with
    | nil => exact False.elim inside
    | cons x xs =>
      have term := coordinate_extrema c x ⟨inside.1,inside.2.1⟩
      have tail := ih xs inside.2.2
      simp only [lowerCorner,upperCorner,List.map_cons,score]
      change c.weight*(if c.weight<0 then c.hi else c.lo) + score box (lowerCorner box) ≤
        c.weight*x + score box xs ∧
        c.weight*x + score box xs ≤
        c.weight*(if c.weight<0 then c.lo else c.hi) + score box (upperCorner box)
      omega

/-- No interval-hull approximation: each implication constructs an actual
    integer corner witness. The cut is doubled to represent half-integer cuts. -/
theorem strict_left_feasible_iff_corner (box : List Coordinate) (valid : Valid box) (cut2 : Int) :
    (∃ xs, Inside box xs ∧ 2*score box xs < cut2) ↔
      2*score box (lowerCorner box) < cut2 := by
  constructor
  · rintro ⟨xs,inside,h⟩
    have bounds := exact_corner_bounds box xs inside
    omega
  · intro h
    exact ⟨lowerCorner box,lower_corner_inside box valid,h⟩

theorem right_feasible_iff_corner (box : List Coordinate) (valid : Valid box) (cut2 : Int) :
    (∃ xs, Inside box xs ∧ cut2 ≤ 2*score box xs) ↔
      cut2 ≤ 2*score box (upperCorner box) := by
  constructor
  · rintro ⟨xs,inside,h⟩
    have bounds := exact_corner_bounds box xs inside
    omega
  · intro h
    exact ⟨upperCorner box,upper_corner_inside box valid,h⟩

theorem left_empty_iff_corner (box : List Coordinate) (valid : Valid box) (cut2 : Int) :
    (¬∃ xs, Inside box xs ∧ 2*score box xs < cut2) ↔
      cut2 ≤ 2*score box (lowerCorner box) := by
  rw [strict_left_feasible_iff_corner box valid cut2]
  exact Int.not_lt

theorem right_empty_iff_corner (box : List Coordinate) (valid : Valid box) (cut2 : Int) :
    (¬∃ xs, Inside box xs ∧ cut2 ≤ 2*score box xs) ↔
      2*score box (upperCorner box) < cut2 := by
  rw [right_feasible_iff_corner box valid cut2]
  exact Int.not_le

theorem exact_complement_partition (box : List Coordinate) (xs : List Int) (cut2 : Int) :
    (Inside box xs ↔
      (Inside box xs ∧ 2*score box xs < cut2) ∨
      (Inside box xs ∧ cut2 ≤ 2*score box xs)) ∧
    ¬((Inside box xs ∧ 2*score box xs < cut2) ∧
      (Inside box xs ∧ cut2 ≤ 2*score box xs)) := by
  by_cases h : 2*score box xs < cut2
  · have hn : ¬cut2 ≤ 2*score box xs := by omega
    simp [h,hn]
  · have hn : cut2 ≤ 2*score box xs := by omega
    simp [h,hn]



def SmallCoefficients (box : List Coordinate) : Prop :=
  ∀ c ∈ box, -64 ≤ c.weight ∧ c.weight ≤ 64 ∧ 0 ≤ c.lo ∧ c.hi ≤ 16777216

theorem small_coordinate_product_bound (c : Coordinate) (x : Int)
    (small : -64 ≤ c.weight ∧ c.weight ≤ 64 ∧ 0 ≤ c.lo ∧ c.hi ≤ 16777216)
    (inside : c.lo ≤ x ∧ x ≤ c.hi) :
    -1073741824 ≤ c.weight*x ∧ c.weight*x ≤ 1073741824 := by
  have hx : 0 ≤ x := by omega
  have xmax : x ≤ 16777216 := by omega
  have lo := Int.mul_le_mul_of_nonneg_right small.1 hx
  have hi := Int.mul_le_mul_of_nonneg_right small.2.1 hx
  have lmax := Int.mul_le_mul_of_nonpos_left (a := (-64 : Int)) (by decide) xmax
  have hmax := Int.mul_le_mul_of_nonneg_left (c := (64 : Int)) xmax (by decide)
  omega

theorem integer_score_length_bound (box : List Coordinate) (xs : List Int)
    (small : SmallCoefficients box) (inside : Inside box xs) :
    -1073741824 * (box.length : Int) ≤ score box xs ∧
    score box xs ≤ 1073741824 * (box.length : Int) := by
  induction box generalizing xs with
  | nil => cases xs <;> simp_all [Inside,score]
  | cons c box ih =>
    cases xs with
    | nil => exact False.elim inside
    | cons x xs =>
      have term := small_coordinate_product_bound c x (small c (by simp)) ⟨inside.1,inside.2.1⟩
      have st : SmallCoefficients box := by
        intro d hd
        exact small d (by simp [hd])
      have tail := ih xs st inside.2.2
      simp only [score,List.length_cons,Int.natCast_add,Int.natCast_one]
      omega

theorem ten_term_integer_scores_below_two_pow_34 (box : List Coordinate) (xs : List Int)
    (small : SmallCoefficients box) (inside : Inside box xs) (count : box.length ≤ 10) :
    -17179869184 < score box xs ∧ score box xs < 17179869184 := by
  have bound := integer_score_length_bound box xs small inside
  omega


end ConverterOnePlane