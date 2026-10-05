import CorrectnessCompletion

/-!
Structural sharing for typed binary decision DAGs. Predicates are immutable
word descriptors interpreted by the same pure `truth` function as the tree.
Only the reference graph is verified here; source/native authority remains
in the existing quantified correctness contract. A rootless arena is a cache,
not a completed converted model.
-/
namespace ConverterSharedDAG
open ConverterGuarantees

inductive RawNode (Predicate Label : Type) where
  | leaf : Label → RawNode Predicate Label
  | branch : Predicate → Nat → Nat → RawNode Predicate Label

inductive Node (Predicate Label : Type) (bound : Nat) where
  | leaf : Label → Node Predicate Label bound
  | branch : Predicate → Fin bound → Fin bound → Node Predicate Label bound

structure DAG (Predicate Label : Type) where
  size : Nat
  node : (i : Fin size) → Node Predicate Label i.val

def validEntry (i : Nat) : RawNode Predicate Label → Prop
  | .leaf _ => True
  | .branch _ a b => a < i ∧ b < i

instance (i : Nat) (e : RawNode Predicate Label) : Decidable (validEntry i e) := by
  cases e <;> simp only [validEntry] <;> infer_instance

def WellOrdered (raw : Array (RawNode Predicate Label)) : Prop :=
  ∀ i : Fin raw.size, validEntry i.val raw[i]

instance (raw : Array (RawNode Predicate Label)) : Decidable (WellOrdered raw) :=
  inferInstanceAs (Decidable (∀ i : Fin raw.size, validEntry i.val raw[i]))

def checkTopology (raw : Array (RawNode Predicate Label)) : Bool := decide (WellOrdered raw)

theorem topology_check_iff (raw : Array (RawNode Predicate Label)) :
    checkTopology raw = true ↔ WellOrdered raw := by simp [checkTopology]

def toNode (e : RawNode Predicate Label) (h : validEntry i e) : Node Predicate Label i :=
  match e with
  | .leaf c => .leaf c
  | .branch p a b => .branch p ⟨a,h.1⟩ ⟨b,h.2⟩

def fromChecked (raw : Array (RawNode Predicate Label)) (h : checkTopology raw = true) :
    DAG Predicate Label :=
  ⟨raw.size, fun i => toNode raw[i] ((topology_check_iff raw).mp h i)⟩

def child (i : Fin n) (j : Fin i.val) : Fin n := ⟨j.val, Nat.lt_trans j.isLt i.isLt⟩

theorem child_id_decreases (i : Fin n) (j : Fin i.val) : (child i j).val < i.val := j.isLt

def unfold (g : DAG Predicate Label) (i : Fin g.size) : Tree Predicate Label :=
  match g.node i with
  | .leaf c => .leaf c
  | .branch p a b => .branch p (unfold g (child i a)) (unfold g (child i b))
termination_by i.val
decreasing_by all_goals exact child_id_decreases _ _

def route (g : DAG Predicate Label) (truth : Predicate → Cell → Bool)
    (i : Fin g.size) (x : Cell) : Label :=
  match g.node i with
  | .leaf c => c
  | .branch p a b => if truth p x then route g truth (child i a) x
    else route g truth (child i b) x
termination_by i.val
decreasing_by all_goals exact child_id_decreases _ _

theorem route_eq_unfold (g : DAG Predicate Label) (truth : Predicate → Cell → Bool) :
    ∀ i x, route g truth i x = eval truth (unfold g i) x := by
  intro i
  induction h : i.val using Nat.strongRecOn generalizing i with
  | ind k ih =>
    intro x
    rw [route, unfold]
    cases hn : g.node i with
    | leaf c => rfl
    | branch p a b =>
      simp only [eval]
      split
      · exact ih (child i a).val (by have := a.isLt; simp only [child]; omega) (child i a) rfl x
      · exact ih (child i b).val (by have := b.isLt; simp only [child]; omega) (child i b) rfl x

def occurrences (g : DAG Predicate Label) (i : Fin g.size) : List Nat :=
  match g.node i with
  | .leaf _ => [i.val]
  | .branch _ a b => i.val :: (occurrences g (child i a) ++ occurrences g (child i b))
termination_by i.val
decreasing_by all_goals exact child_id_decreases _ _

theorem occurrences_length (g : DAG Predicate Label) :
    ∀ i, (occurrences g i).length = nodes (unfold g i) := by
  intro i
  induction h : i.val using Nat.strongRecOn generalizing i with
  | ind k ih =>
    rw [occurrences, unfold]
    cases hn : g.node i with
    | leaf c => rfl
    | branch p a b =>
      simp only [List.length_cons, List.length_append, nodes]
      rw [ih (child i a).val (by have := a.isLt; simp only [child]; omega) (child i a) rfl]
      rw [ih (child i b).val (by have := b.isLt; simp only [child]; omega) (child i b) rfl]

def reachable (g : DAG Predicate Label) (i : Fin g.size) : List Nat :=
  (List.range g.size).filter (fun k => (occurrences g i).contains k)

theorem reachable_nodup (g : DAG Predicate Label) (i : Fin g.size) :
    (reachable g i).Nodup := by
  exact (List.nodup_range).filter _

theorem reachable_subset_occurrences (g : DAG Predicate Label) (i : Fin g.size) :
    reachable g i ⊆ occurrences g i := by
  intro k hk
  simp only [reachable, List.mem_filter, List.contains_iff_mem] at hk
  exact hk.2

theorem stored_reachable_le_expanded (g : DAG Predicate Label) (i : Fin g.size) :
    (reachable g i).length ≤ nodes (unfold g i) := by
  rw [← occurrences_length]
  exact (reachable_nodup g i).length_le_of_subset (reachable_subset_occurrences g i)

theorem certified_DAG_correct (g : DAG Predicate Label) (root : Fin g.size)
    (source : Cell → Label) (truth : Predicate → Cell → Bool) (region : Cell → Prop)
    (cert : Certified source truth region (unfold g root)) :
    ∀ x, region x → route g truth root x = source x := by
  intro x hx
  rw [route_eq_unfold]
  exact certified_correct source truth (unfold g root) region cert x hx

/-- Every executed decision follows a strictly decreasing stored ID. -/
def routeSteps (g : DAG Predicate Label) (truth : Predicate → Cell → Bool)
    (i : Fin g.size) (x : Cell) : Nat :=
  match g.node i with
  | .leaf _ => 1
  | .branch p a b => 1 + if truth p x then routeSteps g truth (child i a) x
    else routeSteps g truth (child i b) x
termination_by i.val
decreasing_by all_goals exact child_id_decreases _ _

theorem route_steps_bounded (g : DAG Predicate Label) (truth : Predicate → Cell → Bool) :
    ∀ i x, routeSteps g truth i x ≤ i.val + 1 := by
  intro i
  induction h : i.val using Nat.strongRecOn generalizing i with
  | ind k ih =>
    intro x
    rw [routeSteps]
    cases hn : g.node i with
    | leaf c => change 1 ≤ k + 1; omega
    | branch p a b =>
      change 1 + (if truth p x then routeSteps g truth (child i a) x else routeSteps g truth (child i b) x) ≤ k + 1
      split
      · have hb := ih (child i a).val (by have := a.isLt; simp only [child]; omega) (child i a) rfl x
        have hd := child_id_decreases i a
        omega
      · have hb := ih (child i b).val (by have := b.isLt; simp only [child]; omega) (child i b) rfl x
        have hd := child_id_decreases i b
        omega

inductive Edge (g : DAG Predicate Label) : Fin g.size → Fin g.size → Prop where
  | left (i : Fin g.size) (p : Predicate) (a b : Fin i.val)
      (h : g.node i = .branch p a b) : Edge g i (child i a)
  | right (i : Fin g.size) (p : Predicate) (a b : Fin i.val)
      (h : g.node i = .branch p a b) : Edge g i (child i b)

theorem edge_decreases (g : DAG Predicate Label) {i j : Fin g.size} (h : Edge g i j) : j.val < i.val := by
  cases h <;> exact child_id_decreases _ _

inductive Walk (g : DAG Predicate Label) : Nat → Fin g.size → Fin g.size → Prop where
  | zero (i) : Walk g 0 i i
  | next (h : Edge g i j) (tail : Walk g n j k) : Walk g (n+1) i k

theorem walk_bound (g : DAG Predicate Label) {i j : Fin g.size} (h : Walk g n i j) : n + j.val ≤ i.val := by
  induction h with
  | zero i => omega
  | next e tail ih => have hd := edge_decreases g e; omega

theorem no_positive_cycle (g : DAG Predicate Label) {i : Fin g.size} (h : Walk g n i i) : n = 0 := by
  have hb := walk_bound g h
  omega

/-- One DAG arena may be the cache for several maximal completed donors. -/
def forestOccurrences (g : DAG Predicate Label) (roots : List (Fin g.size)) : List Nat :=
  roots.flatMap (occurrences g)

def expandedForestNodes (g : DAG Predicate Label) (roots : List (Fin g.size)) : Nat :=
  (roots.map (fun i => nodes (unfold g i))).sum

theorem forest_occurrences_length (g : DAG Predicate Label) (roots : List (Fin g.size)) :
    (forestOccurrences g roots).length = expandedForestNodes g roots := by
  induction roots with
  | nil => rfl
  | cons i rest ih =>
    simp only [forestOccurrences, List.flatMap_cons, List.length_append,
      expandedForestNodes, List.map_cons, List.sum_cons] at *
    rw [occurrences_length, ih]

theorem all_reachable_arena_le_forest (g : DAG Predicate Label) (roots : List (Fin g.size))
    (covered : ∀ k, k < g.size → k ∈ forestOccurrences g roots) :
    g.size ≤ expandedForestNodes g roots := by
  have bound : (List.range g.size).length ≤ (forestOccurrences g roots).length :=
    (List.nodup_range).length_le_of_subset (by
      intro k hk
      exact covered k (List.mem_range.mp hk))
  simpa [forest_occurrences_length] using bound

/-- Contract of exact hash-consing. A branch is stored with exact predicate
    and canonical child IDs, or removed only when both children represent the
    same canonical ID. No hash value alone establishes this relation. -/
inductive Represents (g : DAG Predicate Label) : Tree Predicate Label → Fin g.size → Prop where
  | leaf (c : Label) (i : Fin g.size) (h : g.node i = .leaf c) : Represents g (.leaf c) i
  | branch (p : Predicate) (left right : Tree Predicate Label) (i : Fin g.size)
      (a b : Fin i.val) (h : g.node i = .branch p a b)
      (hl : Represents g left (child i a)) (hr : Represents g right (child i b)) :
      Represents g (.branch p left right) i
  | equalChildren (p : Predicate) (left right : Tree Predicate Label) (i : Fin g.size)
      (hl : Represents g left i) (hr : Represents g right i) :
      Represents g (.branch p left right) i

theorem represented_tree_evaluation (g : DAG Predicate Label) (truth : Predicate → Cell → Bool)
    {i : Fin g.size} (h : Represents g tree i) : ∀ x, eval truth tree x = route g truth i x := by
  induction h with
  | leaf c i hn => intro x; rw [eval, route, hn]
  | branch p left right i a b hn hl hr ihl ihr =>
    intro x
    rw [eval, route, hn]
    dsimp only
    split
    · exact ihl x
    · exact ihr x
  | equalChildren p left right i hl hr ihl ihr =>
    intro x
    simp only [eval]
    split
    · exact ihl x
    · exact ihr x

theorem represented_unfolding_not_larger (g : DAG Predicate Label) {i : Fin g.size} (h : Represents g tree i) :
    nodes (unfold g i) ≤ nodes tree := by
  induction h with
  | leaf c i hn => rw [unfold, hn]; exact Nat.le_refl _
  | branch p left right i a b hn hl hr ihl ihr =>
    rw [unfold, hn]
    simp only [nodes]
    omega
  | equalChildren p left right i hl hr ihl ihr =>
    simp only [nodes]
    omega

theorem represented_stored_not_larger (g : DAG Predicate Label) {i : Fin g.size} (h : Represents g tree i) :
    (reachable g i).length ≤ nodes tree := by
  exact Nat.le_trans (stored_reachable_le_expanded g i) (represented_unfolding_not_larger g h)

/-- Sharing preserves an already proved source-class theorem; it does not
    create source authority from cached labels. -/
theorem represented_source_correct (g : DAG Predicate Label) (truth : Predicate → Cell → Bool)
    (source : Cell → Label) (region : Cell → Prop) {i : Fin g.size} (h : Represents g tree i)
    (cert : Certified source truth region tree) :
    ∀ x, region x → route g truth i x = source x := by
  intro x hx
  rw [← represented_tree_evaluation g truth h x]
  exact certified_correct source truth tree region cert x hx

/-- Wire references use exactly the unsigned 64-bit IDs of the disk format.
    Predicate and label words remain unchanged by this decoding function. -/
inductive WordNode (Predicate Label : Type) where
  | leaf : Label → WordNode Predicate Label
  | branch : Predicate → UInt64 → UInt64 → WordNode Predicate Label

def decodeWords : WordNode Predicate Label → RawNode Predicate Label
  | .leaf c => .leaf c
  | .branch p a b => .branch p a.toNat b.toNat

def checkWordTopology (raw : Array (WordNode Predicate Label)) : Bool :=
  checkTopology (raw.map decodeWords)

theorem word_topology_check_sound (raw : Array (WordNode Predicate Label))
    (h : checkWordTopology raw = true) : WellOrdered (raw.map decodeWords) :=
  (topology_check_iff _).mp h

def fromWordChecked (raw : Array (WordNode Predicate Label))
    (h : checkWordTopology raw = true) : DAG Predicate Label :=
  fromChecked (raw.map decodeWords) h

/-- Absence of a root has no returned class, even if the cache contains leaves. -/
def routeRoot (g : DAG Predicate Label) (truth : Predicate → Cell → Bool)
    (root : Option (Fin g.size)) (x : Cell) : Option Label :=
  root.map (fun i => route g truth i x)

theorem rootless_has_no_class (g : DAG Predicate Label) (truth : Predicate → Cell → Bool)
    (x : Cell) : routeRoot g truth none x = none := rfl

/-- A contract may summarize the union of all contexts in which this node is
    shared. Proving this contract is separate from merely storing references. -/
def LocalContracts (g : DAG Predicate Label) (source : Cell → Label)
    (truth : Predicate → Cell → Bool) (region : Fin g.size → Cell → Prop) : Prop :=
  ∀ i, match g.node i with
    | .leaf c => ∀ x, region i x → source x = c
    | .branch p a b => ∀ x, region i x →
        (truth p x = true → region (child i a) x) ∧
        (truth p x = false → region (child i b) x)

theorem local_contracts_compose (g : DAG Predicate Label) (source : Cell → Label)
    (truth : Predicate → Cell → Bool) (region : Fin g.size → Cell → Prop)
    (cert : LocalContracts g source truth region) :
    ∀ i x, region i x → route g truth i x = source x := by
  intro i
  induction h : i.val using Nat.strongRecOn generalizing i with
  | ind k ih =>
    intro x hx
    have hc := cert i
    cases hn : g.node i with
    | leaf c =>
      rw [hn] at hc
      rw [route, hn]
      exact (hc x hx).symm
    | branch p a b =>
      rw [hn] at hc
      rw [route, hn]
      dsimp only
      by_cases ht : truth p x = true
      · simp only [ht, ↓reduceIte]
        exact ih (child i a).val (by have := a.isLt; simp only [child]; omega)
          (child i a) rfl x ((hc x hx).1 ht)
      · have hf : truth p x = false := by cases q : truth p x <;> simp_all
        simp only [ht]
        exact ih (child i b).val (by have := b.isLt; simp only [child]; omega)
          (child i b) rfl x ((hc x hx).2 hf)

/-- Concrete fractional/raw-input correctness inherits the already specified
    rank-to-native correspondence, without modifying arithmetic or tie rules. -/
theorem local_contracts_raw_correct (g : DAG Predicate Label) (root : Fin g.size)
    (native : Raw → Label) (source : Cell → Label) (encode : Raw → Cell)
    (domain : Raw → Prop) (truth : Predicate → Cell → Bool)
    (region : Fin g.size → Cell → Prop)
    (rank_native : ∀ x, domain x → native x = source (encode x))
    (cover : ∀ x, domain x → region root (encode x))
    (cert : LocalContracts g source truth region) :
    ∀ x, domain x → route g truth root (encode x) = native x := by
  intro x hx
  exact (local_contracts_compose g source truth region cert root (encode x) (cover x hx)).trans
    (rank_native x hx).symm

end ConverterSharedDAG