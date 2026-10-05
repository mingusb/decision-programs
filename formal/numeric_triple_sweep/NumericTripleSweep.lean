import Std

/- Inclusive numeric intervals. Tags stand for arbitrary opaque child identities.
   All arithmetic is unbounded Nat; machine arithmetic refinement is external. -/
namespace NumericTripleSweep

structure Arc where
  lo : Nat
  hi : Nat
  tag : Nat
  deriving DecidableEq, Repr

structure Cell where
  left : Arc
  middle : Arc
  right : Arc
  deriving DecidableEq, Repr

def Cell.lo (t : Cell) : Nat := max t.left.lo (max t.middle.lo t.right.lo)
def Cell.hi (t : Cell) : Nat := min t.left.hi (min t.middle.hi t.right.hi)
def Arc.Contains (a : Arc) (v : Nat) : Prop := a.lo ≤ v ∧ v ≤ a.hi
def Cell.Contains (t : Cell) (v : Nat) : Prop := t.lo ≤ v ∧ v ≤ t.hi
abbrev Cell.Nonempty (t : Cell) : Prop := t.lo ≤ t.hi

/- stop is the exclusive final boundary. Stored arc endpoints remain inclusive. -/
inductive Partition : Nat → Nat → List Arc → Prop where
  | nil (stop : Nat) : Partition stop stop []
  | cons {start stop : Nat} {a : Arc} {rest : List Arc}
      (first : a.lo = start) (valid : a.lo ≤ a.hi)
      (tail : Partition (a.hi + 1) stop rest) : Partition start stop (a :: rest)

def Residual (next stop : Nat) : List Arc → Prop
  | [] => next = stop
  | a :: rest => a.lo ≤ next ∧ next ≤ a.hi ∧ Partition (a.hi + 1) stop rest

def advance (a : Arc) (rest : List Arc) (last : Nat) : List Arc :=
  if a.hi = last then rest else a :: rest

def sweepFuel : Nat → List Arc → List Arc → List Arc → Option (List Cell)
  | _, [], [], [] => some []
  | 0, _, _, _ => none
  | fuel + 1, a :: as, b :: bs, c :: cs =>
      let t : Cell := ⟨a, b, c⟩
      if t.Nonempty then
        (sweepFuel fuel (advance a as t.hi) (advance b bs t.hi)
          (advance c cs t.hi)).map (t :: ·)
      else none
  | _, _, _, _ => none

def sweep (as bs cs : List Arc) : Option (List Cell) :=
  sweepFuel (as.length + bs.length + cs.length) as bs cs

def Sound (cells : List Cell) (as bs cs : List Arc) : Prop :=
  ∀ t ∈ cells, t.Nonempty ∧ t.left ∈ as ∧ t.middle ∈ bs ∧ t.right ∈ cs

def Covers (cells : List Cell) (start stop : Nat) : Prop :=
  ∀ v, start ≤ v → v < stop → ∃ t ∈ cells, t.Contains v

theorem cell_ext {t u : Cell} (left : t.left = u.left)
    (middle : t.middle = u.middle) (right : t.right = u.right) : t = u := by
  cases t; cases u
  simp_all

theorem partition_start_le {start stop : Nat} {as : List Arc}
    (p : Partition start stop as) : start ≤ stop := by
  induction p with
  | nil => omega
  | cons first valid tail ih => omega

theorem partition_residual {start stop : Nat} {as : List Arc}
    (p : Partition start stop as) : Residual start stop as := by
  cases p with
  | nil => rfl
  | cons first valid tail => exact ⟨by omega, by omega, tail⟩

theorem residual_head_before_stop {next stop : Nat} {a : Arc} {as : List Arc}
    (r : Residual next stop (a :: as)) : a.hi < stop := by
  have := partition_start_le r.2.2
  omega

theorem residual_empty_iff {next stop : Nat} {as : List Arc}
    (r : Residual next stop as) : as = [] ↔ next = stop := by
  cases as with
  | nil => simp_all [Residual]
  | cons a as =>
    have := residual_head_before_stop r
    have hn := r.2.1
    simp only [List.cons_ne_nil, false_iff]
    omega

theorem cell_contains_iff (t : Cell) (v : Nat) :
    t.Contains v ↔ t.left.Contains v ∧ t.middle.Contains v ∧ t.right.Contains v := by
  simp [Cell.Contains, Arc.Contains, Cell.lo, Cell.hi]
  omega

theorem cell_endpoint_le (t : Cell) :
    t.hi ≤ t.left.hi ∧ t.hi ≤ t.middle.hi ∧ t.hi ≤ t.right.hi := by
  simp [Cell.hi]
  omega

theorem cell_endpoint_attained (t : Cell) :
    t.left.hi = t.hi ∨ t.middle.hi = t.hi ∨ t.right.hi = t.hi := by
  simp only [Cell.hi]
  omega

theorem advance_member {a d : Arc} {as : List Arc} {last : Nat}
    (h : d ∈ advance a as last) : d ∈ a :: as := by
  unfold advance at h
  split at h
  · exact List.mem_cons_of_mem a h
  · exact h

theorem advance_length_le (a : Arc) (as : List Arc) (last : Nat) :
    (advance a as last).length ≤ (a :: as).length := by
  simp only [advance]
  split <;> simp <;> omega

theorem advance_three_decreases (a b c : Arc) (as bs cs : List Arc) :
    (advance a as (Cell.mk a b c).hi).length +
      (advance b bs (Cell.mk a b c).hi).length +
      (advance c cs (Cell.mk a b c).hi).length <
        (a :: as).length + (b :: bs).length + (c :: cs).length := by
  have attained := cell_endpoint_attained (Cell.mk a b c)
  simp only [advance]
  split <;> split <;> split <;> simp_all <;> omega

theorem advance_residual {next stop last : Nat} {a : Arc} {as : List Arc}
    (r : Residual next stop (a :: as)) (lower : next ≤ last) (upper : last ≤ a.hi) :
    Residual (last + 1) stop (advance a as last) := by
  unfold advance
  split
  · rename_i eq
    have tail := partition_residual r.2.2
    simpa [eq] using tail
  · rename_i ne
    exact ⟨by have := r.1; omega, by omega, r.2.2⟩

theorem residual_round_nonempty {next stop : Nat} {a b c : Arc} {as bs cs : List Arc}
    (ra : Residual next stop (a :: as)) (rb : Residual next stop (b :: bs))
    (rc : Residual next stop (c :: cs)) :
    (Cell.mk a b c).Nonempty ∧ (Cell.mk a b c).lo ≤ next ∧
      next ≤ (Cell.mk a b c).hi := by
  have ha := ra.1; have hb := rb.1; have hc := rc.1
  have ha' := ra.2.1; have hb' := rb.2.1; have hc' := rc.2.1
  simp only [Cell.Nonempty, Cell.lo, Cell.hi]
  omega

theorem sweepFuel_success {fuel next stop : Nat} {as bs cs : List Arc}
    (ra : Residual next stop as) (rb : Residual next stop bs) (rc : Residual next stop cs)
    (budget : as.length + bs.length + cs.length ≤ fuel) :
    ∃ cells, sweepFuel fuel as bs cs = some cells ∧
      Sound cells as bs cs ∧ Covers cells next stop := by
  induction fuel generalizing next as bs cs with
  | zero =>
    have ea : as = [] := List.length_eq_zero_iff.mp (by omega)
    have eb : bs = [] := List.length_eq_zero_iff.mp (by omega)
    have ec : cs = [] := List.length_eq_zero_iff.mp (by omega)
    subst as; subst bs; subst cs
    refine ⟨[], rfl, ?_, ?_⟩
    · simp [Sound]
    · intro v hv hs
      simp only [Residual] at ra
      omega
  | succ fuel ih =>
    by_cases empty : as = []
    · have endEq := (residual_empty_iff ra).mp empty
      have eb := (residual_empty_iff rb).mpr endEq
      have ec := (residual_empty_iff rc).mpr endEq
      subst as; subst bs; subst cs
      refine ⟨[], rfl, ?_, ?_⟩
      · simp [Sound]
      · intro v hv hs; omega
    · cases as with
      | nil => contradiction
      | cons a as =>
        have bne : bs ≠ [] := by
          intro eb
          exact empty ((residual_empty_iff ra).mpr ((residual_empty_iff rb).mp eb))
        have cne : cs ≠ [] := by
          intro ec
          exact empty ((residual_empty_iff ra).mpr ((residual_empty_iff rc).mp ec))
        cases bs with
        | nil => contradiction
        | cons b bs =>
          cases cs with
          | nil => contradiction
          | cons c cs =>
            let t : Cell := ⟨a,b,c⟩
            have round : t.Nonempty ∧ t.lo ≤ next ∧ next ≤ t.hi := residual_round_nonempty ra rb rc
            have ends := cell_endpoint_le t
            have rna := advance_residual ra round.2.2 ends.1
            have rnb := advance_residual rb round.2.2 ends.2.1
            have rnc := advance_residual rc round.2.2 ends.2.2
            have smaller : (advance a as t.hi).length + (advance b bs t.hi).length + (advance c cs t.hi).length < (a :: as).length + (b :: bs).length + (c :: cs).length := advance_three_decreases a b c as bs cs
            have nextBudget : (advance a as t.hi).length + (advance b bs t.hi).length +
                (advance c cs t.hi).length ≤ fuel := by omega
            obtain ⟨tail, run, sound, cover⟩ := ih rna rnb rnc nextBudget
            refine ⟨t :: tail, ?_, ?_, ?_⟩
            · simp only [sweepFuel]
              split
              · change Option.map (fun rest => t :: rest) _ = _
                rw [run]
                rfl
              · rename_i no
                exact False.elim (no round.1)
            · intro u mem
              rcases List.mem_cons.mp mem with eq | mem
              · subst u
                exact ⟨round.1, by simp [t], by simp [t], by simp [t]⟩
              · obtain ⟨ne,ma,mb,mc⟩ := sound u mem
                exact ⟨ne,advance_member ma,advance_member mb,advance_member mc⟩
            · intro v hv hs
              by_cases early : v ≤ t.hi
              · refine ⟨t, by simp, ?_⟩
                exact ⟨by have := round.2.1; omega, early⟩
              · obtain ⟨u, mem, hu⟩ := cover v (by omega) hs
                exact ⟨u, List.mem_cons_of_mem t mem, hu⟩


theorem partition_member_bounds {start stop : Nat} {as : List Arc}
    (p : Partition start stop as) {a : Arc} (mem : a ∈ as) :
    start ≤ a.lo ∧ a.hi < stop ∧ a.lo ≤ a.hi := by
  induction p with
  | nil => simp at mem
  | @cons start stop first rest eq valid tail ih =>
    rcases List.mem_cons.mp mem with same | mem
    · subst a
      have finish := partition_start_le tail
      exact ⟨by omega, by omega, valid⟩
    · obtain ⟨lo,hi,validA⟩ := ih mem
      exact ⟨by omega, hi, validA⟩

theorem partition_unique {start stop v : Nat} {as : List Arc}
    (p : Partition start stop as) {a b : Arc} (ma : a ∈ as) (mb : b ∈ as)
    (ha : a.Contains v) (hb : b.Contains v) : a = b := by
  induction p with
  | nil => simp at ma
  | @cons start stop first rest eq valid tail ih =>
    rcases List.mem_cons.mp ma with ea | ma
    · rcases List.mem_cons.mp mb with eb | mb
      · exact ea.trans eb.symm
      · subst a
        have bounds := partition_member_bounds tail mb
        have h1 := ha.2; have h2 := hb.1
        omega
    · rcases List.mem_cons.mp mb with eb | mb
      · subst b
        have bounds := partition_member_bounds tail ma
        have h1 := ha.1; have h2 := hb.2
        omega
      · exact ih ma mb

theorem residual_joint_exhaustion {next stop : Nat} {as bs cs : List Arc}
    (ra : Residual next stop as) (rb : Residual next stop bs)
    (rc : Residual next stop cs) :
    (as = [] ∨ bs = [] ∨ cs = []) ↔ as = [] ∧ bs = [] ∧ cs = [] := by
  constructor
  · intro exhausted
    have final : next = stop := by
      rcases exhausted with ha | hb | hc
      · exact (residual_empty_iff ra).mp ha
      · exact (residual_empty_iff rb).mp hb
      · exact (residual_empty_iff rc).mp hc
    exact ⟨(residual_empty_iff ra).mpr final, (residual_empty_iff rb).mpr final,
      (residual_empty_iff rc).mpr final⟩
  · intro all
    exact Or.inl all.1

theorem advance_every_tie {a : Arc} {as : List Arc} {last : Nat}
    (tie : a.hi = last) : advance a as last = as := by
  simp [advance, tie]

theorem retain_every_untied {a : Arc} {as : List Arc} {last : Nat}
    (untied : a.hi ≠ last) : advance a as last = a :: as := by
  simp [advance, untied]

theorem sweep_partition_success {start stop : Nat} {as bs cs : List Arc}
    (pa : Partition start stop as) (pb : Partition start stop bs)
    (pc : Partition start stop cs) :
    ∃ cells, sweep as bs cs = some cells ∧ Sound cells as bs cs ∧ Covers cells start stop := by
  exact sweepFuel_success (partition_residual pa) (partition_residual pb)
    (partition_residual pc) (Nat.le_refl _)

theorem sound_complete_intersections {start stop : Nat} {as bs cs : List Arc}
    {cells : List Cell} (pa : Partition start stop as) (pb : Partition start stop bs)
    (pc : Partition start stop cs) (sound : Sound cells as bs cs)
    (cover : Covers cells start stop) (t : Cell) :
    t ∈ cells ↔ t.Nonempty ∧ t.left ∈ as ∧ t.middle ∈ bs ∧ t.right ∈ cs := by
  constructor
  · exact sound t
  · rintro ⟨ne,ma,mb,mc⟩
    have atLow : t.Contains t.lo := ⟨Nat.le_refl _,ne⟩
    obtain ⟨ca,cb,cc⟩ := (cell_contains_iff t t.lo).mp atLow
    have bounds := partition_member_bounds pa ma
    have insideLo : start ≤ t.lo := by have := ca.1; omega
    have insideHi : t.lo < stop := by have := ca.2; omega
    obtain ⟨u,mu,cu⟩ := cover t.lo insideLo insideHi
    obtain ⟨_,ua,ub,uc⟩ := sound u mu
    obtain ⟨cuA,cuB,cuC⟩ := (cell_contains_iff u t.lo).mp cu
    have ea := partition_unique pa ua ma cuA ca
    have eb := partition_unique pb ub mb cuB cb
    have ec := partition_unique pc uc mc cuC cc
    have same : u = t := cell_ext ea eb ec
    simpa [same] using mu

/- Exact set equality with all nonempty Cartesian intersections, including tags.
   Every emitted cell is valid and every such intersection is emitted. -/
theorem common_refinement_exact {start stop : Nat} {as bs cs : List Arc}
    (pa : Partition start stop as) (pb : Partition start stop bs)
    (pc : Partition start stop cs) :
    ∃ cells, sweep as bs cs = some cells ∧
      (∀ t, t ∈ cells ↔ t.Nonempty ∧ t.left ∈ as ∧ t.middle ∈ bs ∧ t.right ∈ cs) ∧
      Covers cells start stop := by
  obtain ⟨cells,run,sound,cover⟩ := sweep_partition_success pa pb pc
  exact ⟨cells,run,sound_complete_intersections pa pb pc sound cover,cover⟩

theorem output_point_unique {start stop v : Nat} {as bs cs : List Arc}
    {cells : List Cell} (pa : Partition start stop as) (pb : Partition start stop bs)
    (pc : Partition start stop cs) (sound : Sound cells as bs cs)
    {t u : Cell} (mt : t ∈ cells) (mu : u ∈ cells)
    (ht : t.Contains v) (hu : u.Contains v) : t = u := by
  obtain ⟨_,ta,tb,tc⟩ := sound t mt
  obtain ⟨_,ua,ub,uc⟩ := sound u mu
  obtain ⟨taV,tbV,tcV⟩ := (cell_contains_iff t v).mp ht
  obtain ⟨uaV,ubV,ucV⟩ := (cell_contains_iff u v).mp hu
  exact cell_ext (partition_unique pa ta ua taV uaV)
    (partition_unique pb tb ub tbV ubV) (partition_unique pc tc uc tcV ucV)

theorem inclusive_domain_coverage {low high : Nat} {as bs cs : List Arc}
    (pa : Partition low (high + 1) as) (pb : Partition low (high + 1) bs)
    (pc : Partition low (high + 1) cs) :
    ∃ cells, sweep as bs cs = some cells ∧ Sound cells as bs cs ∧
      ∀ v, low ≤ v → v ≤ high → ∃ t ∈ cells, t.Contains v := by
  obtain ⟨cells,run,sound,cover⟩ := sweep_partition_success pa pb pc
  exact ⟨cells,run,sound,fun v hv hh => cover v hv (by omega)⟩

theorem no_missed_last_point {low high : Nat} {as bs cs : List Arc}
    (nonempty : low ≤ high) (pa : Partition low (high + 1) as)
    (pb : Partition low (high + 1) bs) (pc : Partition low (high + 1) cs) :
    ∃ cells, sweep as bs cs = some cells ∧ ∃ t ∈ cells, t.Contains high := by
  obtain ⟨cells,run,_,cover⟩ := inclusive_domain_coverage pa pb pc
  exact ⟨cells,run,cover high nonempty (Nat.le_refl _)⟩

def identityArc (low high tag : Nat) : Arc := ⟨low,high,tag⟩

theorem identity_partition {low high tag : Nat} (h : low ≤ high) :
    Partition low (high + 1) [identityArc low high tag] := by
  exact Partition.cons rfl h (Partition.nil _)

theorem virtual_identity_operand {low high tag : Nat} {bs cs : List Arc}
    (domain : low ≤ high) (pb : Partition low (high + 1) bs)
    (pc : Partition low (high + 1) cs) :
    ∃ cells, sweep [identityArc low high tag] bs cs = some cells ∧
      (∀ t, t ∈ cells ↔ t.Nonempty ∧ t.left = identityArc low high tag ∧
        t.middle ∈ bs ∧ t.right ∈ cs) ∧ Covers cells low (high + 1) := by
  obtain ⟨cells,run,exact,cover⟩ := common_refinement_exact (identity_partition domain) pb pc
  refine ⟨cells,run,?_,cover⟩
  intro t
  simpa using exact t


theorem nonempty_iff_point (t : Cell) : t.Nonempty ↔ ∃ v, t.Contains v := by
  constructor
  · intro h; exact ⟨t.lo,Nat.le_refl _,h⟩
  · rintro ⟨v,lo,hi⟩; exact Nat.le_trans lo hi

theorem accepted_run_exact {start stop : Nat} {as bs cs : List Arc} {cells : List Cell}
    (pa : Partition start stop as) (pb : Partition start stop bs)
    (pc : Partition start stop cs) (run : sweep as bs cs = some cells) :
    Sound cells as bs cs ∧ Covers cells start stop ∧
      ∀ t, t ∈ cells ↔ t.Nonempty ∧ t.left ∈ as ∧ t.middle ∈ bs ∧ t.right ∈ cs := by
  obtain ⟨other,otherRun,sound,cover⟩ := sweep_partition_success pa pb pc
  have same : other = cells := Option.some.inj (otherRun.symm.trans run)
  subst other
  exact ⟨sound,cover,sound_complete_intersections pa pb pc sound cover⟩

/- A consumer may check only the emitted child-tag triples. This imposes exactly
   the checks imposed by the full nonempty Cartesian intersection enumeration. -/
theorem checked_sweep_iff_cartesian {start stop : Nat} {as bs cs : List Arc}
    {cells : List Cell} (pa : Partition start stop as) (pb : Partition start stop bs)
    (pc : Partition start stop cs) (run : sweep as bs cs = some cells)
    (accepts : Nat → Nat → Nat → Prop) :
    (∀ t ∈ cells, accepts t.left.tag t.middle.tag t.right.tag) ↔
      (∀ a ∈ as, ∀ b ∈ bs, ∀ c ∈ cs, (Cell.mk a b c).Nonempty →
        accepts a.tag b.tag c.tag) := by
  have exact := (accepted_run_exact pa pb pc run).2.2
  constructor
  · intro checked a ma b mb c mc nonempty
    exact checked (Cell.mk a b c) ((exact _).mpr ⟨nonempty,ma,mb,mc⟩)
  · intro checked t mem
    obtain ⟨nonempty,ma,mb,mc⟩ := (exact t).mp mem
    exact checked t.left ma t.middle mb t.right mc nonempty

theorem sweepFuel_output_length {fuel : Nat} {as bs cs : List Arc} {cells : List Cell}
    (run : sweepFuel fuel as bs cs = some cells) : cells.length ≤ fuel := by
  induction fuel generalizing as bs cs cells with
  | zero =>
    cases as <;> cases bs <;> cases cs <;> simp_all [sweepFuel]
  | succ fuel ih =>
    cases as with
    | nil => cases bs <;> cases cs <;> simp_all [sweepFuel]
    | cons a as =>
      cases bs with
      | nil => cases cs <;> simp_all [sweepFuel]
      | cons b bs =>
        cases cs with
        | nil => simp_all [sweepFuel]
        | cons c cs =>
          simp only [sweepFuel] at run
          split at run
          · cases tailRun : sweepFuel fuel (advance a as (Cell.mk a b c).hi)
                (advance b bs (Cell.mk a b c).hi) (advance c cs (Cell.mk a b c).hi) with
            | none => simp [tailRun] at run
            | some tail =>
              simp only [tailRun,Option.map_some,Option.some.injEq] at run
              subst cells
              have bound := ih tailRun
              simp only [List.length_cons]
              omega
          · simp at run

theorem linear_output_bound {as bs cs : List Arc} {cells : List Cell}
    (run : sweep as bs cs = some cells) : cells.length ≤ as.length + bs.length + cs.length :=
  sweepFuel_output_length run

/- Concrete kernel-reduced fixtures, using literal endpoints and opaque tags. -/
theorem singleton_three_way_tie :
    sweep [⟨0,0,1⟩] [⟨0,0,2⟩] [⟨0,0,3⟩] =
      some [⟨⟨0,0,1⟩,⟨0,0,2⟩,⟨0,0,3⟩⟩] := by decide

theorem staggered_two_way_ties_and_last_segment :
    sweep [⟨3,4,10⟩,⟨5,9,11⟩] [⟨3,6,20⟩,⟨7,9,21⟩]
      [⟨3,4,30⟩,⟨5,6,31⟩,⟨7,9,32⟩] = some
      [⟨⟨3,4,10⟩,⟨3,6,20⟩,⟨3,4,30⟩⟩,
       ⟨⟨5,9,11⟩,⟨3,6,20⟩,⟨5,6,31⟩⟩,
       ⟨⟨5,9,11⟩,⟨7,9,21⟩,⟨7,9,32⟩⟩] := by decide

theorem skipped_two_operands_and_singleton_tail :
    sweep [identityArc 0 4 90] [⟨0,0,1⟩,⟨1,3,2⟩,⟨4,4,3⟩]
      [identityArc 0 4 91] = some
      [⟨identityArc 0 4 90,⟨0,0,1⟩,identityArc 0 4 91⟩,
       ⟨identityArc 0 4 90,⟨1,3,2⟩,identityArc 0 4 91⟩,
       ⟨identityArc 0 4 90,⟨4,4,3⟩,identityArc 0 4 91⟩] := by decide

theorem all_identity_at_large_endpoint :
    sweep [identityArc 2147483647 2147483647 1]
      [identityArc 2147483647 2147483647 2] [identityArc 2147483647 2147483647 3] =
      some [⟨identityArc 2147483647 2147483647 1,
        identityArc 2147483647 2147483647 2,identityArc 2147483647 2147483647 3⟩] := by decide

theorem empty_common_domain : sweep [] [] [] = some [] := rfl

theorem mixed_exhaustion_rejected : sweep [⟨0,0,1⟩] [] [⟨0,0,3⟩] = none := by decide

theorem nonoverlap_rejected : sweep [⟨0,0,1⟩] [⟨1,1,2⟩] [⟨0,1,3⟩] = none := by decide

theorem insufficient_fuel_rejected :
    sweepFuel 1 [⟨0,0,1⟩,⟨1,1,2⟩] [⟨0,1,3⟩] [⟨0,1,4⟩] = none := by decide


/- The final successful round exhausts three nonempty heads. All preceding
   rounds consume at least one, giving the standard sum-minus-two bound. -/
theorem sweepFuel_sharp_bound {fuel : Nat} {as bs cs : List Arc} {cells : List Cell}
    (run : sweepFuel fuel as bs cs = some cells) :
    cells = [] ∨ cells.length + 2 ≤ as.length + bs.length + cs.length := by
  induction fuel generalizing as bs cs cells with
  | zero => cases as <;> cases bs <;> cases cs <;> simp_all [sweepFuel]
  | succ fuel ih =>
    cases as with
    | nil => cases bs <;> cases cs <;> simp_all [sweepFuel]
    | cons a as =>
      cases bs with
      | nil => cases cs <;> simp_all [sweepFuel]
      | cons b bs =>
        cases cs with
        | nil => simp_all [sweepFuel]
        | cons c cs =>
          simp only [sweepFuel] at run
          split at run
          · cases tailRun : sweepFuel fuel (advance a as (Cell.mk a b c).hi)
                (advance b bs (Cell.mk a b c).hi) (advance c cs (Cell.mk a b c).hi) with
            | none => simp [tailRun] at run
            | some tail =>
              simp only [tailRun,Option.map_some,Option.some.injEq] at run
              subst cells
              right
              have bound := ih tailRun
              have decrease := advance_three_decreases a b c as bs cs
              simp only [List.length_cons] at decrease
              rcases bound with empty | bound
              · subst tail
                simp
                omega
              · simp only [List.length_cons]
                omega
          · simp at run

theorem sharp_output_bound {as bs cs : List Arc} {cells : List Cell}
    (run : sweep as bs cs = some cells) :
    cells.length ≤ as.length + bs.length + cs.length - 2 := by
  rcases sweepFuel_sharp_bound run with empty | bound
  · simp [empty]
  · omega

end NumericTripleSweep
