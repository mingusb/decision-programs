import ScoreAddApplyCongruence

/- Conditional seven-score class Apply and binary-lowering correctness.
   Exact words and the native row-local transform are uninterpreted. -/
namespace ClassApplyCongruence
open ScoreDiagramCongruence
open ScoreAddApplyCongruence (Level DimensionBound Cofactors cofactor_runs_iff)

abbrev Seven (Word : Type) := Fin 7 → Word
abbrev ExactFP32 := BitVec 32

/-- A complete coordinate partition, with no condition on state ID order. -/
structure Partitions {Value : Type} (g : Graph Value) (domain : Box) : Prop where
  cover : ∀ id d arcs, g id = .branch d arcs →
    ∀ v, domain d v → ∃ a ∈ arcs, a.accepts v
  disjoint : ∀ id d arcs, g id = .branch d arcs →
    ∀ a ∈ arcs, ∀ b ∈ arcs, ∀ v, a.accepts v → b.accepts v → a = b
  inside : ∀ id d arcs, g id = .branch d arcs →
    ∀ a ∈ arcs, ∀ v, a.accepts v → domain d v

/-- OPEN-state reuse is allowed in either ID direction. Only dimensions progress. -/
structure OrderedStates {Value : Type} (g : Graph Value) (domain : Box)
    : Prop extends Partitions g domain where
  bound : DimensionBound g
  later : ∀ id d arcs, g id = .branch d arcs →
    ∀ a ∈ arcs, d < Level g a.child

theorem state_execution_total {Value : Type} {g : Graph Value} {domain : Box}
    (wf : OrderedStates g domain) (x : Point) (valid : Contains domain x) (root : Nat) :
    ∃ value, Runs g x root value := by
  have main : ∀ fuel id, 12 - Level g id = fuel → ∃ value, Runs g x id value := by
    intro fuel
    induction fuel using Nat.strongRecOn with
    | ind fuel ih =>
      intro id size
      cases node : g id with
      | terminal value => exact ⟨value, .terminal node⟩
      | branch d arcs =>
        obtain ⟨a, mem, hit⟩ := wf.cover id d arcs node (x d) (valid d)
        have later := wf.later id d arcs node a mem
        have bounded := ScoreAddApplyCongruence.level_le_twelve wf.bound a.child
        have before := wf.bound id d arcs node
        have own : Level g id = d := by simp [Level, node]
        obtain ⟨value, tail⟩ := ih (12 - Level g a.child) (by omega) a.child rfl
        exact ⟨value, .branch node mem hit tail⟩
  exact main (12 - Level g root) root rfl

theorem partition_execution_unique {Value : Type} {g : Graph Value} {domain : Box}
    (wf : Partitions g domain) {x id a b} (first : Runs g x id a)
    (second : Runs g x id b) : a = b := by
  induction first with
  | terminal hn =>
    cases second with
    | terminal hz => cases hn.symm.trans hz; rfl
    | branch hz => cases hn.symm.trans hz
  | @branch id d arcs arc value hn member hit tail ih =>
    cases second with
    | terminal hz => cases hn.symm.trans hz
    | @branch _ zd zarcs za zw hz zm zh zt =>
      cases hn.symm.trans hz
      have same := wf.disjoint id d arcs hn arc member za zm (x d) hit zh
      subst za
      exact ih zt

/-- The structural facts checked for every complete joint state. Root order is
    the ordered Fin7 operand tuple, never a set or sorted collection of words. -/
structure JointCofactors {Word : Type} (score : Fin 7 → Graph Word)
    (states : Graph (Seven Word)) (operand : Nat → Seven Nat) (domain : Box) : Prop where
  partitions : OrderedStates states domain
  minimum : ∀ id d arcs, states id = .branch d arcs →
    (∀ c, d ≤ Level (score c) (operand id c)) ∧
    (∃ c, Level (score c) (operand id c) = d)
  terminal : ∀ id words, states id = .terminal words →
    ∀ c, score c (operand id c) = .terminal (words c)
  children : ∀ id d arcs, states id = .branch d arcs → ∀ edge ∈ arcs,
    ∀ v, domain d v → edge.accepts v → ∀ c,
      ∃ a ∈ Cofactors (score c) domain (operand id c) d,
        a.accepts v ∧ a.child = operand edge.child c

theorem joint_run_preserves_seven_words {Word : Type} {score : Fin 7 → Graph Word}
    {states : Graph (Seven Word)} {operand : Nat → Seven Nat} {domain : Box}
    (score_wf : ∀ c, WellFormed (score c) domain)
    (checked : JointCofactors score states operand domain)
    {x root words} (valid : Contains domain x) (run : Runs states x root words) :
    ∀ c, Runs (score c) x (operand root c) (words c) := by
  induction run with
  | terminal node => exact fun c => .terminal (checked.terminal _ _ node c)
  | @branch id d arcs edge words node member hit tail ih =>
    intro c
    obtain ⟨a, ma, ha, child⟩ := checked.children id d arcs node edge member
      (x d) (valid d) hit c
    apply (cofactor_runs_iff (score_wf c) ma ha).mpr
    simpa [child] using ih c

/-- Seven total prescribed score functions are bound to the exact ordered roots. -/
def ScoreCorrespondence {Word : Type} (score : Fin 7 → Graph Word)
    (roots : Seven Nat) (domain : Box) (value : Point → Seven Word) : Prop :=
  ∀ x, Contains domain x → ∀ c, Runs (score c) x (roots c) (value x c)

theorem complete_joint_score_function {Word : Type} {score : Fin 7 → Graph Word}
    {states : Graph (Seven Word)} {operand : Nat → Seven Nat} {domain : Box}
    (score_wf : ∀ c, WellFormed (score c) domain)
    (checked : JointCofactors score states operand domain) (root : Nat)
    (value : Point → Seven Word)
    (bound : ScoreCorrespondence score (operand root) domain value)
    (x : Point) (valid : Contains domain x) : Runs states x root (value x) := by
  obtain ⟨words, run⟩ := state_execution_total checked.partitions x valid root
  have same : words = value x := by
    funext c
    exact execution_deterministic (score_wf c)
      (joint_run_preserves_seven_words score_wf checked valid run c) (bound x valid c)
  simpa [same] using run

/-- The explicit trusted native premise. T includes public probability rounding
    and first-argmax; it need not equal raw-score argmax, or be injective. -/
def NativeRowLocal {Word Label : Type} (domain : Box)
    (margins : Point → Seven Word) (nativeClass : Point → Label)
    (T : Seven Word → Label) : Prop :=
  ∀ x, Contains domain x → nativeClass x = T (margins x)

/-- Every terminal query was on a valid point with exactly the seven terminal
    words. Actual GPU/native query provenance is outside this abstract relation. -/
def TerminalQueries {Word Label : Type} (states : Graph (Seven Word)) (domain : Box)
    (margins : Point → Seven Word) (nativeClass : Point → Label)
    (queried : Nat → Label) : Prop :=
  ∀ id words, states id = .terminal words →
    ∃ x, Contains domain x ∧ margins x = words ∧ queried id = nativeClass x

theorem terminal_query_sound {Word Label : Type} {states : Graph (Seven Word)}
    {domain : Box} {margins : Point → Seven Word} {nativeClass : Point → Label}
    {queried : Nat → Label} {T : Seven Word → Label}
    (native : NativeRowLocal domain margins nativeClass T)
    (queries : TerminalQueries states domain margins nativeClass queried)
    {id words} (leaf : states id = .terminal words) : queried id = T words := by
  obtain ⟨x, valid, words_match, query⟩ := queries id words leaf
  exact query.trans ((native x valid).trans (congrArg T words_match))

theorem equal_terminal_tuples_reuse_class {Word Label : Type}
    {states : Graph (Seven Word)} {domain : Box} {margins : Point → Seven Word}
    {nativeClass : Point → Label} {queried : Nat → Label} {T : Seven Word → Label}
    (native : NativeRowLocal domain margins nativeClass T)
    (queries : TerminalQueries states domain margins nativeClass queried)
    {first second words} (a : states first = .terminal words)
    (b : states second = .terminal words) : queried first = queried second := by
  exact (terminal_query_sound native queries a).trans (terminal_query_sound native queries b).symm

/-- Local structural reduction audit: output cofactors point to exactly the child
    results. Identity cofactors include equal-child collapse; ranges can coalesce
    provided every allowed value retains its correct child. -/
structure LocalClassAudit {Word Label : Type} (states : Graph (Seven Word))
    (classes : Graph Label) (result : Nat → Nat) (queried : Nat → Label)
    (domain : Box) : Prop where
  terminal : ∀ id words, states id = .terminal words →
    classes (result id) = .terminal (queried id)
  children : ∀ id d arcs, states id = .branch d arcs → ∀ edge ∈ arcs,
    ∀ v, domain d v → edge.accepts v →
      ∃ a ∈ Cofactors classes domain (result id) d,
        a.accepts v ∧ a.child = result edge.child

theorem checked_class_run {Word Label : Type} {states : Graph (Seven Word)}
    {classes : Graph Label} {result : Nat → Nat} {queried : Nat → Label}
    {domain : Box} {T : Seven Word → Label}
    (class_wf : WellFormed classes domain)
    (checked : LocalClassAudit states classes result queried domain)
    (terminal : ∀ id words, states id = .terminal words → queried id = T words)
    {x root words} (valid : Contains domain x) (run : Runs states x root words) :
    Runs classes x (result root) (T words) := by
  induction run with
  | terminal node =>
    have leaf := checked.terminal _ _ node
    rw [terminal _ _ node] at leaf
    exact .terminal leaf
  | @branch id d arcs edge words node member hit tail ih =>
    obtain ⟨a, ma, ha, child⟩ := checked.children id d arcs node edge member
      (x d) (valid d) hit
    apply (cofactor_runs_iff class_wf ma ha).mpr
    simpa [child] using ih

/-- Conditional universal class semantics, including existence/totality. -/
theorem joint_native_class_iff {Word Label : Type} {score : Fin 7 → Graph Word}
    {states : Graph (Seven Word)} {operand : Nat → Seven Nat}
    {classes : Graph Label} {result : Nat → Nat} {queried : Nat → Label}
    {domain : Box} {margins : Point → Seven Word} {nativeClass : Point → Label}
    {T : Seven Word → Label}
    (score_wf : ∀ c, WellFormed (score c) domain)
    (joint : JointCofactors score states operand domain)
    (class_wf : WellFormed classes domain)
    (classAudit : LocalClassAudit states classes result queried domain)
    (native : NativeRowLocal domain margins nativeClass T)
    (queries : TerminalQueries states domain margins nativeClass queried)
    (root : Nat) (scores : ScoreCorrespondence score (operand root) domain margins)
    (x : Point) (valid : Contains domain x) (label : Label) :
    Runs classes x (result root) label ↔ label = nativeClass x := by
  have run := checked_class_run class_wf classAudit
    (fun _ _ leaf => terminal_query_sound native queries leaf) valid
    (complete_joint_score_function score_wf joint root margins scores x valid)
  rw [← native x valid] at run
  constructor
  · exact fun given => execution_deterministic class_wf given run
  · intro same; simpa [same] using run

/-- Binary tests can repeat one coordinate inside a lowering chain. -/
inductive BinaryNode (Label : Type) where
  | terminal (label : Label)
  | test (dimension : Nat) (accepts : Nat → Bool) (yes no : Nat)

abbrev BinaryGraph (Label : Type) := Nat → BinaryNode Label

def BinaryEarlier {Label : Type} (g : BinaryGraph Label) : Prop :=
  ∀ id d test yes no, g id = .test d test yes no → yes < id ∧ no < id

inductive BinaryRuns {Label : Type} (g : BinaryGraph Label) (x : Point) : Nat → Label → Prop where
  | terminal {id label} (node : g id = .terminal label) : BinaryRuns g x id label
  | yes {id d test left right label} (node : g id = .test d test left right)
      (hit : test (x d) = true) (tail : BinaryRuns g x left label) : BinaryRuns g x id label
  | no {id d test left right label} (node : g id = .test d test left right)
      (miss : test (x d) = false) (tail : BinaryRuns g x right label) : BinaryRuns g x id label

theorem binary_execution_total {Label : Type} {g : BinaryGraph Label}
    (wf : BinaryEarlier g) (x : Point) (root : Nat) : ∃ label, BinaryRuns g x root label := by
  induction root using Nat.strongRecOn with
  | ind id ih =>
    cases node : g id with
    | terminal label => exact ⟨label, .terminal node⟩
    | test d test left right =>
      have earlier := wf id d test left right node
      cases h : test (x d) with
      | false => obtain ⟨label, run⟩ := ih right earlier.2; exact ⟨label, .no node h run⟩
      | true => obtain ⟨label, run⟩ := ih left earlier.1; exact ⟨label, .yes node h run⟩

theorem binary_execution_unique {Label : Type} {g : BinaryGraph Label} {x root a b}
    (first : BinaryRuns g x root a) (second : BinaryRuns g x root b) : a = b := by
  induction first with
  | terminal hn =>
    cases second with
    | terminal hz => cases hn.symm.trans hz; rfl
    | yes hz => cases hn.symm.trans hz
    | no hz => cases hn.symm.trans hz
  | yes hn hit tail ih =>
    cases second with
    | terminal hz => cases hn.symm.trans hz
    | yes hz other rest => cases hn.symm.trans hz; exact ih rest
    | no hz other => cases hn.symm.trans hz; simp_all
  | no hn miss tail ih =>
    cases second with
    | terminal hz => cases hn.symm.trans hz
    | yes hz other => cases hn.symm.trans hz; simp_all
    | no hz other rest => cases hn.symm.trans hz; exact ih rest

/-- A finite same-coordinate lowering route to the translated child root. -/
inductive BlockRoute {Label : Type} (g : BinaryGraph Label) (d value : Nat) : Nat → Nat → Prop where
  | done (id : Nat) : BlockRoute g d value id id
  | yes {id test left right target} (node : g id = .test d test left right)
      (hit : test value = true) (tail : BlockRoute g d value left target) : BlockRoute g d value id target
  | no {id test left right target} (node : g id = .test d test left right)
      (miss : test value = false) (tail : BlockRoute g d value right target) : BlockRoute g d value id target

theorem block_route_preserves_run {Label : Type} {g : BinaryGraph Label} {x d root target label}
    (route : BlockRoute g d (x d) root target) (run : BinaryRuns g x target label) :
    BinaryRuns g x root label := by
  induction route with
  | done => exact run
  | yes node hit tail ih => exact .yes node hit (ih run)
  | no node miss tail ih => exact .no node miss (ih run)

/-- Quantifies over EVERY domain value, not source-equivalence representatives.
    Numeric rank intervals and all allowed 4/40 categories are covered. -/
structure AllValueLowering {Label : Type} (classes : Graph Label)
    (binary : BinaryGraph Label) (lower : Nat → Nat) (domain : Box) : Prop where
  terminal : ∀ id label, classes id = .terminal label →
    binary (lower id) = .terminal label
  route : ∀ id d arcs, classes id = .branch d arcs → ∀ a ∈ arcs,
    ∀ v, domain d v → a.accepts v → BlockRoute binary d v (lower id) (lower a.child)

theorem universal_lowering_preserves_class {Label : Type} {classes : Graph Label}
    {binary : BinaryGraph Label} {lower : Nat → Nat} {domain : Box}
    (audit : AllValueLowering classes binary lower domain) {x root label}
    (valid : Contains domain x) (run : Runs classes x root label) :
    BinaryRuns binary x (lower root) label := by
  induction run with
  | terminal node => exact .terminal (audit.terminal _ _ node)
  | @branch id d arcs a label node member hit tail ih =>
    exact block_route_preserves_run (audit.route id d arcs node a member (x d) (valid d) hit) ih

theorem universal_lowering_iff {Label : Type} {classes : Graph Label}
    {binary : BinaryGraph Label} {lower : Nat → Nat} {domain : Box}
    (wf : WellFormed classes domain) (audit : AllValueLowering classes binary lower domain)
    (x : Point) (valid : Contains domain x) (root : Nat) (label : Label) :
    BinaryRuns binary x (lower root) label ↔ Runs classes x root label := by
  constructor
  · intro run
    obtain ⟨actual, original⟩ := execution_coverage wf x valid root
    have equal := binary_execution_unique run (universal_lowering_preserves_class audit valid original)
    simpa [equal] using original
  · exact universal_lowering_preserves_class audit valid

/-- All-point conditional native class equality of the lowered graph. The codec,
    raw-to-rank map, GPU audits, native transform and identity binding still need
    separate refinement proofs before this could replace an implementation audit. -/
theorem complete_pipeline_native_iff {Word Label : Type} {score : Fin 7 → Graph Word}
    {states : Graph (Seven Word)} {operand : Nat → Seven Nat}
    {classes : Graph Label} {result : Nat → Nat} {queried : Nat → Label}
    {binary : BinaryGraph Label} {lower : Nat → Nat} {domain : Box}
    {margins : Point → Seven Word} {nativeClass : Point → Label} {T : Seven Word → Label}
    (score_wf : ∀ c, WellFormed (score c) domain)
    (joint : JointCofactors score states operand domain)
    (class_wf : WellFormed classes domain)
    (classAudit : LocalClassAudit states classes result queried domain)
    (native : NativeRowLocal domain margins nativeClass T)
    (queries : TerminalQueries states domain margins nativeClass queried)
    (lowering : AllValueLowering classes binary lower domain)
    (root : Nat) (scores : ScoreCorrespondence score (operand root) domain margins)
    (x : Point) (valid : Contains domain x) (label : Label) :
    BinaryRuns binary x (lower (result root)) label ↔ label = nativeClass x := by
  exact (universal_lowering_iff class_wf lowering x valid (result root) label).trans
    (joint_native_class_iff score_wf joint class_wf classAudit native queries root scores x valid label)

theorem complete_pipeline_native_total {Word Label : Type} {score : Fin 7 → Graph Word}
    {states : Graph (Seven Word)} {operand : Nat → Seven Nat}
    {classes : Graph Label} {result : Nat → Nat} {queried : Nat → Label}
    {binary : BinaryGraph Label} {lower : Nat → Nat} {domain : Box}
    {margins : Point → Seven Word} {nativeClass : Point → Label} {T : Seven Word → Label}
    (score_wf : ∀ c, WellFormed (score c) domain)
    (joint : JointCofactors score states operand domain)
    (class_wf : WellFormed classes domain)
    (classAudit : LocalClassAudit states classes result queried domain)
    (native : NativeRowLocal domain margins nativeClass T)
    (queries : TerminalQueries states domain margins nativeClass queried)
    (lowering : AllValueLowering classes binary lower domain)
    (root : Nat) (scores : ScoreCorrespondence score (operand root) domain margins)
    (x : Point) (valid : Contains domain x) :
    BinaryRuns binary x (lower (result root)) (nativeClass x) := by
  exact (complete_pipeline_native_iff score_wf joint class_wf classAudit native queries
    lowering root scores x valid (nativeClass x)).mpr rfl

/-- The same ordered residual tuple has the same seven outputs at every point.
    This justifies key-based reuse semantically, without any cache-hit bound. -/
theorem equal_operand_states_agree {Word : Type} {score : Fin 7 → Graph Word}
    {states : Graph (Seven Word)} {operand : Nat → Seven Nat} {domain : Box}
    (score_wf : ∀ c, WellFormed (score c) domain)
    (joint : JointCofactors score states operand domain)
    {x first second left right} (valid : Contains domain x)
    (same : operand first = operand second)
    (a : Runs states x first left) (b : Runs states x second right) : left = right := by
  funext c
  have ra := joint_run_preserves_seven_words score_wf joint valid a c
  have rb := joint_run_preserves_seven_words score_wf joint valid b c
  rw [same] at ra
  exact execution_deterministic (score_wf c) ra rb

theorem state_path_dimensions_increase {Value : Type} {states : Graph Value} {domain : Box}
    (wf : OrderedStates states domain) {id ts target} (path : Path states id ts target)
    (lower : Nat) (start : lower ≤ Level states id) : Increasing lower ts := by
  induction path generalizing lower with
  | refl => trivial
  | @step id d arcs a ts target node member tail ih =>
    have eq : Level states id = d := by simp [Level, node]
    have bound : lower ≤ d := by omega
    exact ⟨bound, ih (d + 1) (wf.later id d arcs node a member)⟩

theorem cached_joint_path_has_one_point {Value : Type} {states : Graph Value} {domain : Box}
    (wf : OrderedStates states domain) {id ts target} (path : Path states id ts target)
    (nonempty : ∃ x, Contains domain x) (compatible : LocallyCompatible domain ts) :
    ∃ x, Contains domain x ∧ Satisfies x ts := by
  exact compatible_path_has_witness domain ts 0 nonempty
    (state_path_dimensions_increase wf path 0 (Nat.zero_le _)) compatible

theorem prescribed_scores_to_native_margins {Word : Type} {score : Fin 7 → Graph Word}
    {roots : Seven Nat} {domain : Box} {prescribed nativeMargins : Point → Seven Word}
    (scores : ScoreCorrespondence score roots domain prescribed)
    (native_words : ∀ x, Contains domain x → prescribed x = nativeMargins x) :
    ScoreCorrespondence score roots domain nativeMargins := by
  intro x valid c
  have h := scores x valid c
  rw [native_words x valid] at h
  exact h

/-- Concrete category obligations include all allowed values, even values merged
    into an OTHER equivalence class by the source model. -/
theorem wilderness_lowering_all_four {Label : Type} {classes : Graph Label}
    {binary : BinaryGraph Label} {lower : Nat → Nat}
    {lo hi : Fin 10 → Nat} {wilderness soil : Nat → Prop}
    (audit : AllValueLowering classes binary lower (RankCategoryDomain lo hi wilderness soil))
    {id arcs a v} (node : classes id = .branch 10 arcs) (member : a ∈ arcs)
    (valid : v < 4 ∧ wilderness v) (hit : a.accepts v) :
    BlockRoute binary 10 v (lower id) (lower a.child) := by
  exact audit.route id 10 arcs node a member v (by simpa [RankCategoryDomain] using valid) hit

theorem soil_lowering_all_forty {Label : Type} {classes : Graph Label}
    {binary : BinaryGraph Label} {lower : Nat → Nat}
    {lo hi : Fin 10 → Nat} {wilderness soil : Nat → Prop}
    (audit : AllValueLowering classes binary lower (RankCategoryDomain lo hi wilderness soil))
    {id arcs a v} (node : classes id = .branch 11 arcs) (member : a ∈ arcs)
    (valid : v < 40 ∧ soil v) (hit : a.accepts v) :
    BlockRoute binary 11 v (lower id) (lower a.child) := by
  exact audit.route id 11 arcs node a member v (by simpa [RankCategoryDomain] using valid) hit

theorem numeric_lowering_all_ranks {Label : Type} {classes : Graph Label}
    {binary : BinaryGraph Label} {lower : Nat → Nat}
    {lo hi : Fin 10 → Nat} {wilderness soil : Nat → Prop}
    (audit : AllValueLowering classes binary lower (RankCategoryDomain lo hi wilderness soil))
    (d : Fin 10) {id arcs a v} (node : classes id = .branch d.val arcs) (member : a ∈ arcs)
    (valid : lo d ≤ v ∧ v ≤ hi d) (hit : a.accepts v) :
    BlockRoute binary d.val v (lower id) (lower a.child) := by
  exact audit.route id d.val arcs node a member v
    (by simpa [RankCategoryDomain, d.isLt] using valid) hit

/-- A queried witness alone cannot license all-point class reuse if native class
    depends on row information absent from the seven score words. -/
theorem absent_rowlocal_counterexample :
    ∃ (margins : Point → Seven Nat) (nativeClass : Point → Nat) (x y : Point),
      margins x = margins y ∧ nativeClass x ≠ nativeClass y := by
  exact ⟨fun _ _ => 0, fun p => p 0, fun _ => 0, fun _ => 1, rfl, by decide⟩

end ClassApplyCongruence
