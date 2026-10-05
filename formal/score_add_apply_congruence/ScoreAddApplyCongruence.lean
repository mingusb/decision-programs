import ScoreDiagramCongruence

/- Exact ordered binary Apply over abstract words. No arithmetic laws are assumed. -/
namespace ScoreAddApplyCongruence
open ScoreDiagramCongruence

variable {Word : Type}

def Level (g : Graph Word) (id : Nat) : Nat :=
  match g id with
  | .terminal _ => 12
  | .branch d _ => d

def DimensionBound (g : Graph Word) : Prop :=
  ∀ id d arcs, g id = .branch d arcs → d < 12

def IdentityArc (domain : Box) (id d : Nat) : Arc :=
  { accepts := domain d, child := id }

def Cofactors (g : Graph Word) (domain : Box) (id d : Nat) : List Arc :=
  match g id with
  | .terminal _ => [IdentityArc domain id d]
  | .branch next arcs => if next = d then arcs else [IdentityArc domain id d]

theorem level_le_twelve {g : Graph Word} (bound : DimensionBound g) (id : Nat) :
    Level g id ≤ 12 := by
  cases h : g id with
  | terminal w => simp [Level, h]
  | branch d arcs => simpa [Level, h] using Nat.le_of_lt (bound id d arcs h)

theorem terminal_of_level_twelve {g : Graph Word} (bound : DimensionBound g) {id : Nat}
    (level : Level g id = 12) : ∃ w, g id = .terminal w := by
  cases h : g id with
  | terminal w => exact ⟨w, rfl⟩
  | branch d arcs => have hd := bound id d arcs h; simp [Level, h] at level; omega

theorem cofactor_coverage {g : Graph Word} {domain : Box} (wf : WellFormed g domain)
    (id d v : Nat) (valid : domain d v) :
    ∃ a ∈ Cofactors g domain id d, a.accepts v := by
  cases h : g id with
  | terminal w => exact ⟨IdentityArc domain id d, by simp [Cofactors, h], valid⟩
  | branch next arcs =>
    by_cases eq : next = d
    · subst next
      simpa [Cofactors, h] using wf.outgoing_cover id d arcs h v valid
    · exact ⟨IdentityArc domain id d, by simp [Cofactors, h, eq], valid⟩

theorem cofactor_child_le {g : Graph Word} {domain : Box} (wf : WellFormed g domain)
    {id d : Nat} {a : Arc} (mem : a ∈ Cofactors g domain id d) : a.child ≤ id := by
  cases h : g id with
  | terminal w => simp [Cofactors, h] at mem; subst a; exact Nat.le_refl _
  | branch next arcs =>
    by_cases eq : next = d
    · subst next
      have ma : a ∈ arcs := by simpa [Cofactors, h] using mem
      exact Nat.le_of_lt (wf.child_earlier id d arcs h a ma)
    · simp [Cofactors, h, eq] at mem; subst a; exact Nat.le_refl _

theorem active_cofactor_child_lt {g : Graph Word} {domain : Box}
    (wf : WellFormed g domain) {id d : Nat} {a : Arc}
    (active : Level g id = d) (nonterminal : d < 12)
    (mem : a ∈ Cofactors g domain id d) : a.child < id := by
  cases h : g id with
  | terminal w => simp [Level, h] at active; omega
  | branch next arcs =>
    have eq : next = d := by simpa [Level, h] using active
    subst next
    exact wf.child_earlier id d arcs h a (by simpa [Cofactors, h] using mem)

theorem cofactor_level_increases {g : Graph Word} {domain : Box}
    (wf : WellFormed g domain) {id d : Nat} {a : Arc}
    (lower : d ≤ Level g id) (nonterminal : d < 12)
    (mem : a ∈ Cofactors g domain id d) : d < Level g a.child := by
  cases h : g id with
  | terminal w =>
    simp [Cofactors, h] at mem; subst a; simpa [IdentityArc, Level, h] using nonterminal
  | branch next arcs =>
    by_cases eq : next = d
    · subst next
      have ma : a ∈ arcs := by simpa [Cofactors, h] using mem
      cases hc : g a.child with
      | terminal w => simpa [Level, hc] using nonterminal
      | branch following branches =>
        simpa [Level, hc] using wf.dimension_increases id d arcs h a ma following branches hc
    · simp [Cofactors, h, eq] at mem; subst a
      simp [IdentityArc, Level, h] at lower ⊢; omega

theorem cofactor_runs_iff {g : Graph Word} {domain : Box}
    (wf : WellFormed g domain) {id d : Nat} {a : Arc} {x : Point} {w : Word}
    (mem : a ∈ Cofactors g domain id d) (hit : a.accepts (x d)) :
    Runs g x id w ↔ Runs g x a.child w := by
  cases h : g id with
  | terminal word => simp [Cofactors, h] at mem; subst a; rfl
  | branch next arcs =>
    by_cases eq : next = d
    · subst next
      have ma : a ∈ arcs := by simpa [Cofactors, h] using mem
      constructor
      · intro run
        cases run with
        | terminal ht => cases h.symm.trans ht
        | @branch _ otherDim otherArcs otherArc otherWord hn mb hb tail =>
          cases h.symm.trans hn
          have same := wf.outgoing_disjoint id d arcs h a ma otherArc mb (x d) hit hb
          subst otherArc; exact tail
      · intro run; exact .branch h ma hit run
    · simp [Cofactors, h, eq] at mem; subst a; rfl

structure Triple where
  left : Nat
  right : Nat
  output : Nat

def Weight (t : Triple) : Nat := t.left + t.right + t.output

def MinimumDimension (left right output : Graph Word) (t : Triple) : Nat :=
  min (min (Level left t.left) (Level right t.right)) (Level output t.output)

def ChildTriple (a b c : Arc) : Triple := ⟨a.child, b.child, c.child⟩

def TripleGuard (a b c : Arc) (v : Nat) : Prop :=
  a.accepts v ∧ b.accepts v ∧ c.accepts v

theorem minimum_bounds (left right output : Graph Word) (t : Triple) :
    MinimumDimension left right output t ≤ Level left t.left ∧
    MinimumDimension left right output t ≤ Level right t.right ∧
    MinimumDimension left right output t ≤ Level output t.output := by
  simp only [MinimumDimension]; omega

theorem triple_weight_decreases {left right output : Graph Word} {domain : Box}
    (wl : WellFormed left domain) (wr : WellFormed right domain)
    (wo : WellFormed output domain) (t : Triple) (a b c : Arc)
    (nonterminal : MinimumDimension left right output t < 12)
    (ma : a ∈ Cofactors left domain t.left (MinimumDimension left right output t))
    (mb : b ∈ Cofactors right domain t.right (MinimumDimension left right output t))
    (mc : c ∈ Cofactors output domain t.output (MinimumDimension left right output t)) :
    Weight (ChildTriple a b c) < Weight t := by
  have ha := cofactor_child_le wl ma
  have hb := cofactor_child_le wr mb
  have hc := cofactor_child_le wo mc
  have one : Level left t.left = MinimumDimension left right output t ∨
      Level right t.right = MinimumDimension left right output t ∨
      Level output t.output = MinimumDimension left right output t := by
    simp only [MinimumDimension]; omega
  rcases one with h | h | h
  · have strict := active_cofactor_child_lt wl h nonterminal ma
    simp only [Weight, ChildTriple]; omega
  · have strict := active_cofactor_child_lt wr h nonterminal mb
    simp only [Weight, ChildTriple]; omega
  · have strict := active_cofactor_child_lt wo h nonterminal mc
    simp only [Weight, ChildTriple]; omega

theorem triple_dimension_increases {left right output : Graph Word} {domain : Box}
    (wl : WellFormed left domain) (wr : WellFormed right domain)
    (wo : WellFormed output domain) (t : Triple) (a b c : Arc)
    (nonterminal : MinimumDimension left right output t < 12)
    (ma : a ∈ Cofactors left domain t.left (MinimumDimension left right output t))
    (mb : b ∈ Cofactors right domain t.right (MinimumDimension left right output t))
    (mc : c ∈ Cofactors output domain t.output (MinimumDimension left right output t)) :
    MinimumDimension left right output t < MinimumDimension left right output (ChildTriple a b c) := by
  obtain ⟨hl, hr, ho⟩ := minimum_bounds left right output t
  have ha := cofactor_level_increases wl hl nonterminal ma
  have hb := cofactor_level_increases wr hr nonterminal mb
  have hc := cofactor_level_increases wo ho nonterminal mc
  simp only [MinimumDimension, ChildTriple] at *; omega

/-- One exact aligned cofactor transition, before any memoization. -/
inductive Transition (left right output : Graph Word) (domain : Box) :
    Triple → Test → Triple → Prop where
  | aligned (t : Triple) (a b c : Arc)
      (nonterminal : MinimumDimension left right output t < 12)
      (ma : a ∈ Cofactors left domain t.left (MinimumDimension left right output t))
      (mb : b ∈ Cofactors right domain t.right (MinimumDimension left right output t))
      (mc : c ∈ Cofactors output domain t.output (MinimumDimension left right output t)) :
      Transition left right output domain t
        ⟨MinimumDimension left right output t, TripleGuard a b c⟩ (ChildTriple a b c)

inductive TriplePath (left right output : Graph Word) (domain : Box) :
    Triple → List Test → Triple → Prop where
  | refl (t : Triple) : TriplePath left right output domain t [] t
  | step {t u target test ts}
      (edge : Transition left right output domain t test u)
      (tail : TriplePath left right output domain u ts target) :
      TriplePath left right output domain t (test :: ts) target

theorem triple_path_dimensions_increase {left right output : Graph Word} {domain : Box}
    (wl : WellFormed left domain) (wr : WellFormed right domain)
    (wo : WellFormed output domain) {t target ts}
    (path : TriplePath left right output domain t ts target) (lower : Nat)
    (start : lower ≤ MinimumDimension left right output t) : Increasing lower ts := by
  induction path generalizing lower with
  | refl => trivial
  | @step t u target test ts edge tail ih =>
    cases edge with
    | aligned a b c nonterminal ma mb mc =>
      have later := triple_dimension_increases wl wr wo t a b c nonterminal ma mb mc
      exact ⟨start, ih (MinimumDimension left right output t + 1) later⟩

/-- Strict coordinate progress turns local guard intersections into one real point. -/
theorem cached_triple_trace_has_point {left right output : Graph Word} {domain : Box}
    (wl : WellFormed left domain) (wr : WellFormed right domain)
    (wo : WellFormed output domain) {t target ts}
    (path : TriplePath left right output domain t ts target)
    (nonempty : ∃ x, Contains domain x) (compatible : LocallyCompatible domain ts) :
    ∃ x, Contains domain x ∧ Satisfies x ts := by
  exact compatible_path_has_witness domain ts 0 nonempty
    (triple_path_dimensions_increase wl wr wo path 0 (Nat.zero_le _)) compatible

/-- Every finite sequence of exact guard restrictions has the stated conjunction. -/
theorem triple_guard_restriction_induction (domain : Box) (ts : List Test) (x : Point) :
    Contains (RestrictMany domain ts) x ↔ Contains domain x ∧ Satisfies x ts :=
  restriction_induction domain ts x

/-- Every nonempty aligned guard intersection is inserted, including output-only tests. -/
def AuditClosed (left right output : Graph Word) (domain : Box)
    (marked : Triple → Prop) : Prop :=
  ∀ t, marked t → ∀ a b c,
    let d := MinimumDimension left right output t
    d < 12 → a ∈ Cofactors left domain t.left d →
    b ∈ Cofactors right domain t.right d → c ∈ Cofactors output domain t.output d →
    (∃ v, domain d v ∧ TripleGuard a b c v) → marked (ChildTriple a b c)

/-- The terminal test preserves left/right order and exact word equality. -/
def TerminalAudit (op : Word → Word → Word) (left right output : Graph Word)
    (marked : Triple → Prop) : Prop :=
  ∀ t, marked t → ∀ a b c, left t.left = .terminal a →
    right t.right = .terminal b → output t.output = .terminal c → c = op a b

theorem terminal_run_word {g : Graph Word} {x : Point} {id : Nat} {a b : Word}
    (node : g id = .terminal a) (run : Runs g x id b) : b = a := by
  cases run with
  | terminal h => cases node.symm.trans h; rfl
  | branch h => cases node.symm.trans h

/-- Sparse triple closure and checked terminal words imply the universal relation. -/
theorem checked_triple_congruence {left right output : Graph Word} {domain : Box}
    (wl : WellFormed left domain) (wr : WellFormed right domain)
    (wo : WellFormed output domain) (bl : DimensionBound left)
    (br : DimensionBound right) (bo : DimensionBound output)
    (op : Word → Word → Word) (marked : Triple → Prop)
    (closed : AuditClosed left right output domain marked)
    (checked : TerminalAudit op left right output marked)
    (root : Triple) (seed : marked root) (x : Point) (valid : Contains domain x)
    (a b c : Word) (rl : Runs left x root.left a)
    (rr : Runs right x root.right b) (ro : Runs output x root.output c) : c = op a b := by
  have main : ∀ n, ∀ t : Triple, Weight t = n → marked t →
      ∀ a b c, Runs left x t.left a → Runs right x t.right b →
      Runs output x t.output c → c = op a b := by
    intro n
    induction n using Nat.strongRecOn with
    | ind n ih =>
      intro t mass mark a b c runL runR runO
      let d := MinimumDimension left right output t
      by_cases active : d < 12
      · obtain ⟨al, ml, hl⟩ := cofactor_coverage wl t.left d (x d) (valid d)
        obtain ⟨ar, mr, hr⟩ := cofactor_coverage wr t.right d (x d) (valid d)
        obtain ⟨ao, mo, ho⟩ := cofactor_coverage wo t.output d (x d) (valid d)
        have next := closed t mark al ar ao active ml mr mo ⟨x d, valid d, hl, hr, ho⟩
        have smaller := triple_weight_decreases wl wr wo t al ar ao active ml mr mo
        exact ih (Weight (ChildTriple al ar ao)) (by omega) (ChildTriple al ar ao) rfl next
          a b c ((cofactor_runs_iff wl ml hl).mp runL)
          ((cofactor_runs_iff wr mr hr).mp runR) ((cofactor_runs_iff wo mo ho).mp runO)
      · obtain ⟨hl, hr, ho⟩ := minimum_bounds left right output t
        have ll := level_le_twelve bl t.left
        have lr := level_le_twelve br t.right
        have lo := level_le_twelve bo t.output
        have el : Level left t.left = 12 := by dsimp [d] at active; omega
        have er : Level right t.right = 12 := by dsimp [d] at active; omega
        have eo : Level output t.output = 12 := by dsimp [d] at active; omega
        obtain ⟨av, an⟩ := terminal_of_level_twelve bl el
        obtain ⟨bv, bn⟩ := terminal_of_level_twelve br er
        obtain ⟨cv, cn⟩ := terminal_of_level_twelve bo eo
        have ea := terminal_run_word an runL
        have eb := terminal_run_word bn runR
        have ec := terminal_run_word cn runO
        simpa [ea, eb, ec] using checked t mark av bv cv an bn cn
  exact main (Weight root) root rfl seed a b c rl rr ro

/-- Complete guards supply runs for every admissible point; nothing is sampled. -/
theorem checked_apply_total {left right output : Graph Word} {domain : Box}
    (wl : WellFormed left domain) (wr : WellFormed right domain)
    (wo : WellFormed output domain) (bl : DimensionBound left)
    (br : DimensionBound right) (bo : DimensionBound output)
    (op : Word → Word → Word) (marked : Triple → Prop)
    (closed : AuditClosed left right output domain marked)
    (checked : TerminalAudit op left right output marked)
    (root : Triple) (seed : marked root) (x : Point) (valid : Contains domain x) :
    ∃ a b c, Runs left x root.left a ∧ Runs right x root.right b ∧
      Runs output x root.output c ∧ c = op a b := by
  obtain ⟨a, ha⟩ := execution_coverage wl x valid root.left
  obtain ⟨b, hb⟩ := execution_coverage wr x valid root.right
  obtain ⟨c, hc⟩ := execution_coverage wo x valid root.output
  exact ⟨a,b,c,ha,hb,hc,checked_triple_congruence wl wr wo bl br bo op marked closed checked root seed x valid a b c ha hb hc⟩

/-- Exact relational equality of the output with the specified ordered operation. -/
theorem checked_apply_iff {left right output : Graph Word} {domain : Box}
    (wl : WellFormed left domain) (wr : WellFormed right domain)
    (wo : WellFormed output domain) (bl : DimensionBound left)
    (br : DimensionBound right) (bo : DimensionBound output)
    (op : Word → Word → Word) (marked : Triple → Prop)
    (closed : AuditClosed left right output domain marked)
    (checked : TerminalAudit op left right output marked)
    (root : Triple) (seed : marked root) (x : Point) (valid : Contains domain x)
    (word : Word) : Runs output x root.output word ↔
      ∃ a b, Runs left x root.left a ∧ Runs right x root.right b ∧ word = op a b := by
  constructor
  · intro ho
    obtain ⟨a, ha⟩ := execution_coverage wl x valid root.left
    obtain ⟨b, hb⟩ := execution_coverage wr x valid root.right
    exact ⟨a,b,ha,hb,checked_triple_congruence wl wr wo bl br bo op marked closed checked root seed x valid a b word ha hb ho⟩
  · rintro ⟨a,b,ha,hb,eq⟩
    obtain ⟨c,hc⟩ := execution_coverage wo x valid root.output
    have congr := checked_triple_congruence wl wr wo bl br bo op marked closed checked root seed x valid a b c ha hb hc
    have same : c = word := congr.trans eq.symm
    simpa [same] using hc

/-- Sequential callers inherit only their specified association. -/
theorem ordered_composition (op : Word → Word → Word) (a b c intermediate result : Word)
    (first : intermediate = op a b) (second : result = op intermediate c) :
    result = op (op a b) c := by simpa [first] using second

end ScoreAddApplyCongruence
