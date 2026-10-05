import Lean

/-!
Abstract guarantees for class-tree conversion. Numerical predicates, source
semantics, rank correspondence, native resolution and geometric completeness
are explicit parameters/contracts. No C++/CUDA refinement is asserted here.
-/
namespace ConverterGuarantees

inductive Tree (Predicate Label : Type) where
  | leaf : Label → Tree Predicate Label
  | branch : Predicate → Tree Predicate Label → Tree Predicate Label → Tree Predicate Label

def eval (truth : Predicate → Cell → Bool) : Tree Predicate Label → Cell → Label
  | .leaf c, _ => c
  | .branch p yes no, x => if truth p x then eval truth yes x else eval truth no x

def nodes : Tree Predicate Label → Nat
  | .leaf _ => 1
  | .branch _ a b => nodes a + nodes b + 1

def leaves : Tree Predicate Label → Nat
  | .leaf _ => 1
  | .branch _ a b => leaves a + leaves b

/-- Each branch keeps the complete original ancestor region. -/
def Certified (source : Cell → Label) (truth : Predicate → Cell → Bool)
    (region : Cell → Prop) : Tree Predicate Label → Prop
  | .leaf c => ∀ x, region x → source x = c
  | .branch p yes no =>
      Certified source truth (fun x => region x ∧ truth p x = true) yes ∧
      Certified source truth (fun x => region x ∧ truth p x = false) no

theorem certified_correct (source : Cell → Label) (truth : Predicate → Cell → Bool)
    (tree : Tree Predicate Label) :
    ∀ (region : Cell → Prop), Certified source truth region tree →
      ∀ x, region x → eval truth tree x = source x := by
  induction tree with
  | leaf c =>
      intro region cert x hx
      exact (cert x hx).symm
  | branch p yes no iy ino =>
      intro region cert x hx
      cases hp : truth p x with
      | false =>
          simp only [eval, hp, Bool.false_eq_true, ↓reduceIte]
          exact ino _ cert.2 x ⟨hx, hp⟩
      | true =>
          simp only [eval, hp, ↓reduceIte]
          exact iy _ cert.1 x ⟨hx, hp⟩

/-- Import/rank/native protocol correspondence is a quantified premise. -/
theorem raw_class_correct
    (native : Raw → Label) (source : Cell → Label)
    (encode : Raw → Cell) (domain : Raw → Prop) (region : Cell → Prop)
    (truth : Predicate → Cell → Bool) (tree : Tree Predicate Label)
    (rank_native : ∀ x, domain x → native x = source (encode x))
    (root_cover : ∀ x, domain x → region (encode x))
    (cert : Certified source truth region tree) :
    ∀ x, domain x → eval truth tree (encode x) = native x := by
  intro x hx
  exact (certified_correct source truth tree region cert (encode x) (root_cover x hx)).trans
    (rank_native x hx).symm

theorem physical_node_leaf_identity (tree : Tree Predicate Label) :
    nodes tree + 1 = 2 * leaves tree := by
  induction tree with
  | leaf _ => rfl
  | branch _ a b ia ib => simp only [nodes, leaves]; omega

theorem physical_nodes_positive (tree : Tree Predicate Label) : 0 < nodes tree := by
  cases tree <;> simp [nodes]

/-- Both disagreement regions empty proves identical Boolean routing. -/
theorem disagreement_empty_equal (p q : Cell → Bool) (region : Cell → Prop)
    (a : ∀ x, region x → ¬ (p x = true ∧ q x = false))
    (b : ∀ x, region x → ¬ (p x = false ∧ q x = true)) :
    ∀ x, region x → p x = q x := by
  intro x hx
  cases hp : p x <;> cases hq : q x
  · rfl
  · exact False.elim (b x hx ⟨hp, hq⟩)
  · exact False.elim (a x hx ⟨hp, hq⟩)
  · rfl

/-- The other two empty regions justify complement plus child swap. -/
theorem agreement_empty_complement (p q : Cell → Bool) (region : Cell → Prop)
    (a : ∀ x, region x → ¬ (p x = true ∧ q x = true))
    (b : ∀ x, region x → ¬ (p x = false ∧ q x = false)) :
    ∀ x, region x → p x = !(q x) := by
  intro x hx
  cases hp : p x <;> cases hq : q x
  · exact False.elim (b x hx ⟨hp, hq⟩)
  · rfl
  · rfl
  · exact False.elim (a x hx ⟨hp, hq⟩)

/-- Fitting every known cell makes a wrong-class witness necessarily new. -/
theorem counterexample_is_fresh (source predict : Cell → Label) (known : Cell → Prop)
    (fit : ∀ x, known x → predict x = source x)
    (x : Cell) (wrong : predict x ≠ source x) : ¬ known x := by
  intro hx
  exact wrong (fit x hx)

/-- A finite prefix of actual successful transitions, with its exact length. -/
inductive Executes (step : State → State → Prop) : Nat → State → State → Prop where
  | zero (s) : Executes step 0 s s
  | next {s t u n} : step s t → Executes step n t u → Executes step (n+1) s u

theorem execution_potential_bound
    (potential : State → Nat) (step : State → State → Prop)
    (decreases : ∀ s t, step s t → potential t < potential s)
    (path : Executes step n s t) : n + potential t ≤ potential s := by
  induction path with
  | zero s => omega
  | next h tail ih =>
      have hdrop := decreases _ _ h
      omega

/-- A total finishing-or-progress contract yields a bounded completed path.
    Unknown, cancellation and resource failure do not satisfy this premise. -/
theorem completes_under_total_progress
    (potential : State → Nat) (step : State → State → Prop) (done admissible : State → Prop)
    (progress : ∀ s, admissible s → done s ∨
      ∃ t, step s t ∧ admissible t ∧ potential t < potential s) :
    ∀ s, admissible s → ∃ n t, Executes step n s t ∧ done t ∧ n ≤ potential s := by
  have aux : ∀ k, ∀ s, potential s = k → admissible s →
      ∃ n t, Executes step n s t ∧ done t ∧ n ≤ potential s := by
    intro k
    induction k using Nat.strongRecOn with
    | ind k ih =>
        intro s heq hadmissible
        cases progress s hadmissible with
        | inl hd => exact ⟨0, s, .zero s, hd, Nat.zero_le _⟩
        | inr hp =>
            obtain ⟨t, hst, ht, hlt⟩ := hp
            obtain ⟨n, u, hpath, hd, hn⟩ := ih (potential t) (by omega) t rfl ht
            exact ⟨n+1, u, .next hst hpath, hd, by omega⟩
  intro s hadmissible
  exact aux (potential s) s rfl hadmissible

def mass (v : Nat) : Nat := 2*v-1

def frontierPotential : List Nat → Nat
  | [] => 0
  | v :: rest => mass v + frontierPotential rest

theorem potential_append (a b : List Nat) :
    frontierPotential (a ++ b) = frontierPotential a + frontierPotential b := by
  induction a with
  | nil => simp [frontierPotential]
  | cons v rest ih => simp [frontierPotential, ih, Nat.add_assoc]

theorem split_mass_identity (a b : Nat) (ha : 0 < a) (hb : 0 < b) :
    mass (a+b) = mass a + mass b + 1 := by
  unfold mass
  omega

theorem leaf_mass_positive (v : Nat) (hv : 0 < v) : 0 < mass v := by
  unfold mass
  omega

structure Snapshot where
  committed : Nat
  pendingVolumes : List Nat

def potential (s : Snapshot) := frontierPotential s.pendingVolumes

def upper (s : Snapshot) := s.committed + potential s

/-- FIFO head removal, with exact nonempty child cardinality partition.
    Leaf labels are certified in the separate semantic contract above. -/
inductive DirectStep : Snapshot → Snapshot → Prop where
  | split (c a b : Nat) (rest : List Nat) (ha : 0<a) (hb : 0<b) :
      DirectStep ⟨c, (a+b)::rest⟩ ⟨c+1, rest++[a,b]⟩
  | leaf (c v : Nat) (rest : List Nat) (hv : 0<v) :
      DirectStep ⟨c, v::rest⟩ ⟨c+1, rest⟩

theorem direct_step_decreases (s t : Snapshot) (h : DirectStep s t) :
    potential t < potential s := by
  cases h with
  | split c a b rest ha hb =>
      simp only [potential, frontierPotential, potential_append]
      have hm := split_mass_identity a b ha hb
      omega
  | leaf c v rest hv =>
      simp only [potential, frontierPotential]
      have hm := leaf_mass_positive v hv
      omega

theorem direct_step_commits_one (s t : Snapshot) (h : DirectStep s t) :
    t.committed = s.committed + 1 := by
  cases h <;> rfl

theorem direct_upper_monotone (s t : Snapshot) (h : DirectStep s t) :
    upper t ≤ upper s := by
  have hd := direct_step_decreases s t h
  have hc := direct_step_commits_one s t h
  unfold upper
  omega

theorem direct_split_upper_equal (c a b : Nat) (rest : List Nat)
    (ha : 0<a) (hb : 0<b) :
    upper ⟨c+1, rest++[a,b]⟩ = upper ⟨c, (a+b)::rest⟩ := by
  simp only [upper, potential, frontierPotential, potential_append]
  have hm := split_mass_identity a b ha hb
  omega

theorem direct_leaf_exact_upper_saving (c v : Nat) (rest : List Nat) (hv : 0<v) :
    upper ⟨c, v::rest⟩ = upper ⟨c+1, rest⟩ + 2*(v-1) := by
  simp only [upper, potential, frontierPotential, mass]
  omega

theorem direct_path_commit_count (path : Executes DirectStep n s t) :
    t.committed = s.committed + n := by
  induction path with
  | zero s => omega
  | next h tail ih =>
      have hc := direct_step_commits_one _ _ h
      omega

theorem direct_path_budget (path : Executes DirectStep n s t) :
    n + potential t ≤ potential s :=
  execution_potential_bound potential DirectStep direct_step_decreases path

theorem direct_final_node_upper (path : Executes DirectStep n s t)
    (done : t.pendingVolumes = []) : t.committed ≤ upper s := by
  have hb := direct_path_budget path
  have hc := direct_path_commit_count path
  have hz : potential t = 0 := by
    change frontierPotential t.pendingVolumes = 0
    rw [done]
    rfl
  unfold upper
  omega

/-- Total geometry/native service is explicit, not inferred from finite inputs. -/
theorem direct_completion (admissible : Snapshot → Prop)
    (progress : ∀ s : Snapshot, admissible s → s.pendingVolumes = [] ∨
      ∃ t, DirectStep s t ∧ admissible t) :
    ∀ s, admissible s → ∃ n t,
      Executes DirectStep n s t ∧ t.pendingVolumes = [] ∧ n ≤ potential s := by
  apply completes_under_total_progress potential DirectStep
    (fun s => s.pendingVolumes = []) admissible
  intro s hs
  cases progress s hs with
  | inl h => exact Or.inl h
  | inr h =>
      obtain ⟨t, hst, ht⟩ := h
      exact Or.inr ⟨t, hst, ht, direct_step_decreases s t hst⟩

theorem initial_cell_upper (m : Nat) :
    upper ⟨0, [m]⟩ = 2*m-1 := by
  simp [upper, potential, frontierPotential, mass]

theorem source_decision_count (tree : Tree Predicate Label) :
    nodes tree + leaves tree + 1 = 3*leaves tree := by
  have h := physical_node_leaf_identity tree
  omega

def FreshCellStep (total : Nat) (known next : Nat) : Prop :=
  known < next ∧ next ≤ total

theorem fresh_cell_potential_decreases (total known next : Nat)
    (h : FreshCellStep total known next) : total-next < total-known := by
  unfold FreshCellStep at h
  omega

theorem rejected_candidate_bound (total : Nat)
    (path : Executes (FreshCellStep total) n start finish) :
    n + (total-finish) ≤ total-start := by
  apply execution_potential_bound (fun known => total-known) (FreshCellStep total) _ path
  intro s t h
  exact fresh_cell_potential_decreases total s t h

/-- Acceptance adds at most one proposal after a bounded rejection chain. -/
theorem completed_proposal_bound (total : Nat)
    (path : Executes (FreshCellStep total) n start finish) :
    n+1 ≤ total-start+1 := by
  have h := rejected_candidate_bound total path
  omega

/-- No legal execution prefix can exceed its initial integer budget. -/
theorem no_over_budget_execution
    (potential : State → Nat) (step : State → State → Prop)
    (decreases : ∀ s t, step s t → potential t < potential s)
    (too_long : potential s < n) : ¬ Executes step n s t := by
  intro path
  have hb := execution_potential_bound potential step decreases path
  omega

/-- Composition explicitly includes physical count/serialization refinement.
    The imported/raw/native correspondence remains quantified over the domain. -/
theorem complete_artifact_correct_and_bounded
    (native : Raw → Label) (source : Cell → Label)
    (encode : Raw → Cell) (domain : Raw → Prop) (region : Cell → Prop)
    (truth : Predicate → Cell → Bool) (tree : Tree Predicate Label)
    (rank_native : ∀ x, domain x → native x = source (encode x))
    (root_cover : ∀ x, domain x → region (encode x))
    (cert : Certified source truth region tree)
    (path : Executes DirectStep n s t) (done : t.pendingVolumes = [])
    (count_matches : nodes tree = t.committed) :
    (∀ x, domain x → eval truth tree (encode x) = native x) ∧ nodes tree ≤ upper s := by
  constructor
  · exact raw_class_correct native source encode domain region truth tree
      rank_native root_cover cert
  · rw [count_matches]
    exact direct_final_node_upper path done

end ConverterGuarantees

#print axioms ConverterGuarantees.certified_correct
#print axioms ConverterGuarantees.raw_class_correct
#print axioms ConverterGuarantees.physical_node_leaf_identity
#print axioms ConverterGuarantees.physical_nodes_positive
#print axioms ConverterGuarantees.disagreement_empty_equal
#print axioms ConverterGuarantees.agreement_empty_complement
#print axioms ConverterGuarantees.counterexample_is_fresh
#print axioms ConverterGuarantees.execution_potential_bound
#print axioms ConverterGuarantees.completes_under_total_progress
#print axioms ConverterGuarantees.potential_append
#print axioms ConverterGuarantees.split_mass_identity
#print axioms ConverterGuarantees.leaf_mass_positive
#print axioms ConverterGuarantees.direct_step_decreases
#print axioms ConverterGuarantees.direct_step_commits_one
#print axioms ConverterGuarantees.direct_upper_monotone
#print axioms ConverterGuarantees.direct_split_upper_equal
#print axioms ConverterGuarantees.direct_leaf_exact_upper_saving
#print axioms ConverterGuarantees.direct_path_commit_count
#print axioms ConverterGuarantees.direct_path_budget
#print axioms ConverterGuarantees.direct_final_node_upper
#print axioms ConverterGuarantees.direct_completion
#print axioms ConverterGuarantees.initial_cell_upper
#print axioms ConverterGuarantees.source_decision_count
#print axioms ConverterGuarantees.fresh_cell_potential_decreases
#print axioms ConverterGuarantees.rejected_candidate_bound
#print axioms ConverterGuarantees.completed_proposal_bound
#print axioms ConverterGuarantees.no_over_budget_execution
#print axioms ConverterGuarantees.complete_artifact_correct_and_bounded
