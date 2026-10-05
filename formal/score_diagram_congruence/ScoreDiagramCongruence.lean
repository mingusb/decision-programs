import Std

/- Abstract factor-to-score-diagram congruence. Rank and categorical coordinates
   are discrete Nat values; their admissible sets are explicit coordinate predicates.
   Words are uninterpreted exact values, not real arithmetic or native probabilities. -/
namespace ScoreDiagramCongruence

abbrev Point := Nat → Nat
abbrev Box := Nat → Nat → Prop

def Contains (b : Box) (x : Point) : Prop := ∀ d, b d (x d)
def Subbox (b domain : Box) : Prop := ∀ d v, b d v → domain d v

structure Test where
  dimension : Nat
  accepts : Nat → Prop

def Satisfies (x : Point) (ts : List Test) : Prop :=
  ∀ t ∈ ts, t.accepts (x t.dimension)
def LocallyCompatible (b : Box) (ts : List Test) : Prop :=
  ∀ t ∈ ts, ∃ v, b t.dimension v ∧ t.accepts v

def Increasing : Nat → List Test → Prop
  | _, [] => True
  | lower, t :: ts => lower ≤ t.dimension ∧ Increasing (t.dimension + 1) ts

def Restrict (b : Box) (t : Test) : Box :=
  fun d v => b d v ∧ (d = t.dimension → t.accepts v)

theorem restriction_membership (b : Box) (t : Test) (x : Point) :
    Contains (Restrict b t) x ↔ Contains b x ∧ t.accepts (x t.dimension) := by
  constructor
  · intro h
    exact ⟨fun d => (h d).1, (h t.dimension).2 rfl⟩
  · rintro ⟨hb, ht⟩ d
    exact ⟨hb d, by intro eq; simpa [eq] using ht⟩

def RestrictMany : Box → List Test → Box
  | b, [] => b
  | b, t :: ts => RestrictMany (Restrict b t) ts

theorem restriction_induction (b : Box) (ts : List Test) (x : Point) :
    Contains (RestrictMany b ts) x ↔ Contains b x ∧ Satisfies x ts := by
  induction ts generalizing b with
  | nil => simp [RestrictMany, Satisfies]
  | cons t ts ih =>
    rw [RestrictMany, ih, restriction_membership]
    simp [Satisfies, and_assoc]

theorem increasing_lower_bound (lower : Nat) (ts : List Test)
    (h : Increasing lower ts) : ∀ t ∈ ts, lower ≤ t.dimension := by
  induction ts generalizing lower with
  | nil => simp
  | cons t ts ih =>
    intro u hu
    rcases List.mem_cons.mp hu with eq | mem
    · simpa [eq] using h.1
    · have bound := ih (t.dimension + 1) h.2 u mem
      exact Nat.le_trans h.1 (Nat.le_trans (Nat.le_succ _) bound)

/-- A product box lets a current coordinate be changed without changing others. -/
theorem coordinate_restriction_extension (b : Box) (x : Point) (d v : Nat)
    (hx : Contains b x) (hv : b d v) :
    Contains b (fun k => if k = d then v else x k) := by
  intro k
  by_cases hk : k = d
  · simpa [hk] using hv
  · simpa [hk] using hx k

/-- No repeated coordinates: local intersections really do share a global witness. -/
theorem compatible_path_has_witness (b : Box) (ts : List Test) (lower : Nat)
    (nonempty : ∃ x, Contains b x) (ordered : Increasing lower ts)
    (compatible : LocallyCompatible b ts) :
    ∃ x, Contains b x ∧ Satisfies x ts := by
  induction ts generalizing lower with
  | nil =>
    obtain ⟨x, hx⟩ := nonempty
    exact ⟨x, hx, by simp [Satisfies]⟩
  | cons t ts ih =>
    have tailLocal : LocallyCompatible b ts := by
      intro u hu; exact compatible u (List.mem_cons_of_mem t hu)
    obtain ⟨x, hx, hs⟩ := ih (t.dimension + 1) ordered.2 tailLocal
    obtain ⟨v, hv, ht⟩ := compatible t (by simp)
    let y : Point := fun k => if k = t.dimension then v else x k
    refine ⟨y, coordinate_restriction_extension b x t.dimension v hx hv, ?_⟩
    intro u hu
    rcases List.mem_cons.mp hu with eq | mem
    · subst u; simpa [y] using ht
    · have bound := increasing_lower_bound _ _ ordered.2 u mem
      have ne : u.dimension ≠ t.dimension := by omega
      simpa [y, ne] using hs u mem

theorem witness_gives_local_intersections (b : Box) (ts : List Test)
    (x : Point) (hb : Contains b x) (hs : Satisfies x ts) :
    LocallyCompatible b ts := by
  intro t ht
  exact ⟨x t.dimension, hb t.dimension, hs t ht⟩

theorem independent_restrictions_exact (b : Box) (ts : List Test) (lower : Nat)
    (nonempty : ∃ x, Contains b x) (ordered : Increasing lower ts) :
    (∃ x, Contains (RestrictMany b ts) x) ↔ LocallyCompatible b ts := by
  constructor
  · rintro ⟨x, hx⟩
    have h := (restriction_induction b ts x).mp hx
    exact witness_gives_local_intersections b ts x h.1 h.2
  · intro h
    obtain ⟨x, hx, hs⟩ := compatible_path_has_witness b ts lower nonempty ordered h
    exact ⟨x, (restriction_induction b ts x).mpr ⟨hx, hs⟩⟩

structure Arc where
  accepts : Nat → Prop
  child : Nat

inductive Node (Word : Type) where
  | terminal (word : Word)
  | branch (dimension : Nat) (arcs : List Arc)

abbrev Graph (Word : Type) := Nat → Node Word

/-- Global check matches the current CUDA auditor, which checks all stored nodes.
    Values outside a finite stored array can be modeled by terminal nodes. -/
structure WellFormed {Word : Type} (g : Graph Word) (domain : Box) where
  outgoing_cover : ∀ id d arcs, g id = .branch d arcs →
    ∀ v, domain d v → ∃ a ∈ arcs, a.accepts v
  outgoing_disjoint : ∀ id d arcs, g id = .branch d arcs →
    ∀ a ∈ arcs, ∀ z ∈ arcs, ∀ v, a.accepts v → z.accepts v → a = z
  outgoing_domain : ∀ id d arcs, g id = .branch d arcs →
    ∀ a ∈ arcs, ∀ v, a.accepts v → domain d v
  child_earlier : ∀ id d arcs, g id = .branch d arcs →
    ∀ a ∈ arcs, a.child < id
  dimension_increases : ∀ id d arcs, g id = .branch d arcs →
    ∀ a ∈ arcs, ∀ next ds, g a.child = .branch next ds → d < next

inductive Path {Word : Type} (g : Graph Word) : Nat → List Test → Nat → Prop where
  | refl (id : Nat) : Path g id [] id
  | step {id d arcs a ts target} (node : g id = .branch d arcs)
      (member : a ∈ arcs) (tail : Path g a.child ts target) :
      Path g id ({dimension := d, accepts := a.accepts} :: ts) target

inductive Runs {Word : Type} (g : Graph Word) (x : Point) : Nat → Word → Prop where
  | terminal {id word} (node : g id = .terminal word) : Runs g x id word
  | branch {id d arcs a word} (node : g id = .branch d arcs)
      (member : a ∈ arcs) (accepts : a.accepts (x d))
      (tail : Runs g x a.child word) : Runs g x id word

variable {Word : Type} {g : Graph Word} {domain : Box}

theorem path_ids_decrease (wf : WellFormed g domain)
    {id ts target} (path : Path g id ts target) : target ≤ id := by
  induction path with
  | refl => exact Nat.le_refl _
  | step node member tail ih =>
    exact Nat.le_trans ih (Nat.le_of_lt (wf.child_earlier _ _ _ node _ member))

theorem path_dimensions_increase (wf : WellFormed g domain)
    {id ts target} (path : Path g id ts target) (lower : Nat)
    (root_lower : ∀ d arcs, g id = .branch d arcs → lower ≤ d) :
    Increasing lower ts := by
  induction path generalizing lower with
  | refl => trivial
  | @step id d arcs a ts target node member tail ih =>
    exact ⟨root_lower d arcs node, ih (d + 1) (by
      intro next ds hn
      exact wf.dimension_increases id d arcs node a member next ds hn)⟩

theorem execution_coverage (wf : WellFormed g domain) (x : Point)
    (valid : Contains domain x) (id : Nat) : ∃ word, Runs g x id word := by
  induction id using Nat.strongRecOn with
  | ind id ih =>
    cases hn : g id with
    | terminal word => exact ⟨word, .terminal hn⟩
    | branch d arcs =>
      obtain ⟨a, ha, hit⟩ := wf.outgoing_cover id d arcs hn (x d) (valid d)
      obtain ⟨word, run⟩ := ih a.child (wf.child_earlier id d arcs hn a ha)
      exact ⟨word, .branch hn ha hit run⟩

theorem execution_deterministic (wf : WellFormed g domain) {x id first second}
    (a : Runs g x id first) (z : Runs g x id second) : first = second := by
  induction a with
  | terminal hn =>
    cases z with
    | terminal hz => cases hn.symm.trans hz; rfl
    | branch hz => cases hn.symm.trans hz
  | @branch id d arcs arc word hn member hit tail ih =>
    cases z with
    | terminal hz => cases hn.symm.trans hz
    | @branch _ zd zarcs za zw hz zm zh zt =>
      have eq := hn.symm.trans hz
      cases eq
      have same := wf.outgoing_disjoint id d arcs hn arc member za zm (x d) hit zh
      subst za
      exact ih zt

theorem execution_has_path {x id word} (run : Runs g x id word) :
    ∃ ts target, Path g id ts target ∧ Satisfies x ts ∧ g target = .terminal word := by
  induction run with
  | terminal hn => exact ⟨[], _, .refl _, by simp [Satisfies], hn⟩
  | @branch id d arcs a word hn member hit tail ih =>
    obtain ⟨ts, target, path, hs, leaf⟩ := ih
    refine ⟨{dimension := d, accepts := a.accepts} :: ts, target,
      .step hn member path, ?_, leaf⟩
    intro t ht
    rcases List.mem_cons.mp ht with eq | mem
    · subst t; exact hit
    · exact hs t mem

theorem terminal_path_executes {id ts target word x}
    (path : Path g id ts target) (hs : Satisfies x ts)
    (leaf : g target = .terminal word) : Runs g x id word := by
  induction path with
  | refl => exact .terminal leaf
  | @step id d arcs a ts target hn member tail ih =>
    exact .branch hn member (hs {dimension := d, accepts := a.accepts} (by simp))
      (ih (by intro t ht; exact hs t (List.mem_cons_of_mem _ ht)) leaf)

/-- Abstract closure implemented by a factor-by-node mark table. A node is marked
    when some root path has an intersection on every coordinate it tests. -/
def CacheReach (b : Box) (g : Graph Word) (root target : Nat) : Prop :=
  ∃ ts, Path g root ts target ∧ LocallyCompatible b ts

def PointReach (b : Box) (g : Graph Word) (root target : Nat) : Prop :=
  ∃ x ts, Contains b x ∧ Path g root ts target ∧ Satisfies x ts

theorem point_reach_is_cached {b root target} (h : PointReach b g root target) :
    CacheReach b g root target := by
  obtain ⟨x, ts, hx, hp, hs⟩ := h
  exact ⟨ts, hp, witness_gives_local_intersections b ts x hx hs⟩

theorem cached_reach_has_point (wf : WellFormed g domain) {b root target}
    (nonempty : ∃ x, Contains b x) (h : CacheReach b g root target) :
    PointReach b g root target := by
  obtain ⟨ts, hp, hc⟩ := h
  have ordered := path_dimensions_increase wf hp 0 (by intros; omega)
  obtain ⟨x, hx, hs⟩ := compatible_path_has_witness b ts 0 nonempty ordered hc
  exact ⟨x, ts, hx, hp, hs⟩

theorem cached_reach_exact (wf : WellFormed g domain) {b root target}
    (nonempty : ∃ x, Contains b x) :
    CacheReach b g root target ↔ PointReach b g root target := by
  exact ⟨cached_reach_has_point wf nonempty, point_reach_is_cached⟩

/-- The mark-table refinement obligation: seed the class root and propagate every
    factor-intersecting outgoing arc. This abstracts the descending-ID scan. -/
def MarksClosed (b : Box) (g : Graph Word) (marked : Nat → Prop) : Prop :=
  ∀ id d arcs, g id = .branch d arcs → marked id →
    ∀ a ∈ arcs, (∃ v, b d v ∧ a.accepts v) → marked a.child

theorem compatible_path_is_marked {b : Box} {marked : Nat → Prop}
    (closed : MarksClosed b g marked) {id ts target}
    (path : Path g id ts target) (seed : marked id)
    (compatible : LocallyCompatible b ts) : marked target := by
  induction path with
  | refl => exact seed
  | @step id d arcs a ts target hn member tail ih =>
    have hit : ∃ v, b d v ∧ a.accepts v :=
      compatible {dimension := d, accepts := a.accepts} (by simp)
    exact ih (closed id d arcs hn seed a member hit)
      (by intro t ht; exact compatible t (List.mem_cons_of_mem _ ht))

theorem closed_marks_cover_cached_reach {b : Box} {marked : Nat → Prop}
    (closed : MarksClosed b g marked) {root target}
    (seed : marked root) (reach : CacheReach b g root target) : marked target := by
  obtain ⟨ts, path, compatible⟩ := reach
  exact compatible_path_is_marked closed path seed compatible

def TerminalAudit (b : Box) (word : Word) (g : Graph Word) (root : Nat) : Prop :=
  ∀ target actual, CacheReach b g root target → g target = .terminal actual → actual = word

theorem checked_closed_marks_imply_audit {b : Box} {marked : Nat → Prop}
    {word : Word} {root : Nat} (seed : marked root)
    (closed : MarksClosed b g marked)
    (checked : ∀ id actual, marked id → g id = .terminal actual → actual = word) :
    TerminalAudit b word g root := by
  intro id actual reachable leaf
  exact checked id actual (closed_marks_cover_cached_reach closed seed reachable) leaf

theorem factor_terminal_audit_sound {b : Box} {word : Word} {root : Nat}
    (audit : TerminalAudit b word g root) {x : Point} (hx : Contains b x)
    {actual : Word} (run : Runs g x root actual) : actual = word := by
  obtain ⟨ts, target, hp, hs, ht⟩ := execution_has_path run
  exact audit target actual ⟨ts, hp, witness_gives_local_intersections b ts x hx hs⟩ ht

theorem factor_terminal_audit_complete (wf : WellFormed g domain)
    {b : Box} {word : Word} {root : Nat} (nonempty : ∃ x, Contains b x)
    (correct : ∀ x, Contains b x → ∀ actual, Runs g x root actual → actual = word) :
    TerminalAudit b word g root := by
  intro target actual hc ht
  obtain ⟨x, ts, hx, hp, hs⟩ := cached_reach_has_point wf nonempty hc
  exact correct x hx actual (terminal_path_executes hp hs ht)

structure Factor (Word : Type) where
  box : Box
  word : Word

structure FactorPartition (domain : Box) (fs : List (Factor Word)) : Prop where
  nonempty : ∀ f ∈ fs, ∃ x, Contains f.box x
  inside : ∀ f ∈ fs, Subbox f.box domain
  cover : ∀ x, Contains domain x → ∃ f ∈ fs, Contains f.box x
  disjoint : ∀ f ∈ fs, ∀ z ∈ fs, ∀ x, Contains f.box x → Contains z.box x → f = z

def TableValue (fs : List (Factor Word)) (x : Point) (word : Word) : Prop :=
  ∃ f ∈ fs, Contains f.box x ∧ word = f.word

theorem factor_table_single_valued {fs : List (Factor Word)}
    (partition : FactorPartition domain fs) {x first second}
    (hfirst : TableValue fs x first) (hsecond : TableValue fs x second) : first = second := by
  obtain ⟨f, hf, hx, hw⟩ := hfirst
  obtain ⟨z, hz, zx, zw⟩ := hsecond
  have same := partition.disjoint f hf z hz x hx zx
  subst z
  exact hw.trans zw.symm

/-- Main semantic equality: every admissible rank/category point, all exact words.
    One invocation handles one class; the caller binds the matching class root. -/
theorem complete_factor_diagram_congruence (wf : WellFormed g domain)
    {fs : List (Factor Word)} (partition : FactorPartition domain fs) (root : Nat)
    (audits : ∀ f ∈ fs, TerminalAudit f.box f.word g root)
    (x : Point) (valid : Contains domain x) (word : Word) :
    Runs g x root word ↔ TableValue fs x word := by
  constructor
  · intro run
    obtain ⟨f, hf, hx⟩ := partition.cover x valid
    exact ⟨f, hf, hx, factor_terminal_audit_sound (audits f hf) hx run⟩
  · rintro ⟨f, hf, hx, hw⟩
    obtain ⟨actual, run⟩ := execution_coverage wf x valid root
    have same := factor_terminal_audit_sound (audits f hf) hx run
    rw [same, ← hw] at run
    exact run

theorem checked_mark_table_congruence (wf : WellFormed g domain)
    {fs : List (Factor Word)} (partition : FactorPartition domain fs) (root : Nat)
    (marked : Factor Word → Nat → Prop)
    (seed : ∀ f ∈ fs, marked f root)
    (closed : ∀ f ∈ fs, MarksClosed f.box g (marked f))
    (checked : ∀ f ∈ fs, ∀ id actual,
      marked f id → g id = .terminal actual → actual = f.word)
    (x : Point) (valid : Contains domain x) (word : Word) :
    Runs g x root word ↔ TableValue fs x word := by
  apply complete_factor_diagram_congruence wf partition root _ x valid word
  intro f hf
  exact checked_closed_marks_imply_audit (seed f hf) (closed f hf) (checked f hf)

/-- Conditional bridge to any reference function, with its correspondence explicit. -/
theorem reference_congruence (wf : WellFormed g domain)
    {fs : List (Factor Word)} (partition : FactorPartition domain fs) (root : Nat)
    (audits : ∀ f ∈ fs, TerminalAudit f.box f.word g root)
    (reference : Point → Word)
    (correspondence : ∀ f ∈ fs, ∀ x, Contains f.box x → f.word = reference x)
    (x : Point) (valid : Contains domain x) :
    Runs g x root (reference x) ∧ ∀ word, Runs g x root word → word = reference x := by
  obtain ⟨f, hf, hx⟩ := partition.cover x valid
  have ref := correspondence f hf x hx
  have value : TableValue fs x (reference x) := ⟨f, hf, hx, ref.symm⟩
  refine ⟨(complete_factor_diagram_congruence wf partition root audits x valid _).mpr value, ?_⟩
  intro word run
  exact (factor_terminal_audit_sound (audits f hf) hx run).trans ref

/-- Concrete admissible coordinate schema: ten bounded integer ranks, a selected
    wilderness category in 0..3, and a selected soil category in 0..39. Unused
    coordinates are fixed to zero. Categorical subsets remain explicit predicates. -/
def RankCategoryDomain (lower upper : Fin 10 → Nat)
    (wilderness soil : Nat → Prop) : Box := fun d v =>
  if h : d < 10 then lower ⟨d, h⟩ ≤ v ∧ v ≤ upper ⟨d, h⟩
  else if d = 10 then v < 4 ∧ wilderness v
  else if d = 11 then v < 40 ∧ soil v
  else v = 0

theorem rank_category_domain_bounds (lower upper : Fin 10 → Nat)
    (wilderness soil : Nat → Prop) (x : Point)
    (valid : Contains (RankCategoryDomain lower upper wilderness soil) x) :
    (∀ d : Fin 10, lower d ≤ x d ∧ x d ≤ upper d) ∧
    (x 10 < 4 ∧ wilderness (x 10)) ∧ (x 11 < 40 ∧ soil (x 11)) := by
  refine ⟨?_, ?_, ?_⟩
  · intro d; simpa [RankCategoryDomain, d.isLt] using valid d
  · simpa [RankCategoryDomain] using valid 10
  · simpa [RankCategoryDomain] using valid 11

/-- Dropping the no-repeat condition admits a false local-intersection path. -/
theorem repeated_dimension_counterexample :
    ∃ (b : Box) (ts : List Test),
      (∃ x, Contains b x) ∧ LocallyCompatible b ts ∧
      ¬ (∃ x, Contains (RestrictMany b ts) x) := by
  let b : Box := fun _ _ => True
  let zero : Test := {dimension := 0, accepts := fun v => v = 0}
  let one : Test := {dimension := 0, accepts := fun v => v = 1}
  let ts := [zero, one]
  refine ⟨b, ts, ⟨fun _ => 0, by intro d; trivial⟩, ?_, ?_⟩
  · intro t ht
    have cases : t = zero ∨ t = one := by simpa [ts] using ht
    rcases cases with h | h
    · subst t; exact ⟨0, trivial, rfl⟩
    · subst t; exact ⟨1, trivial, rfl⟩
  · rintro ⟨x, hx⟩
    have sat := (restriction_induction b ts x).mp hx
    have hz : x 0 = 0 := sat.2 zero (by simp [ts])
    have ho : x 0 = 1 := sat.2 one (by simp [ts])
    omega

end ScoreDiagramCongruence
