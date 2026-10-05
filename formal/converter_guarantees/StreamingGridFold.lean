import MixedRadixCoverage
import CollectedRoots

/-!
Descending, bounded-batch Cartesian-grid construction. The reference loop has
one suffix accumulator per dimension. Its operator preserves the numeric and
categorical child orientation exactly; batching and exact Arena mappings do
not change the reference expression. The checked claims are mathematical, not
CUDA instruction refinement. All sizes here are Nat, not virtual UInt64.
-/
namespace ConverterStreamingGrid
open ConverterGuarantees ConverterSharedDAG ConverterOnlineArena ConverterCollectedRoots
open ConverterMixedRadix ConverterSignatureEnumeration
variable {P L X A S W : Type}

/-- Numeric questions put the current child on the true/left side. A one-hot
    category question puts that category on the false/right side. -/
def join (category : Bool) (question : P) (current suffix : Tree P L) : Tree P L :=
  if category then .branch question suffix current else .branch question current suffix

def prepend (category : Bool) (item : P × Tree P L)
    (suffix : Option (Tree P L)) : Option (Tree P L) :=
  match suffix with
  | none => some item.2
  | some later => some (join category item.1 item.2 later)

def rightChain (category : Bool) (items : List (P × Tree P L)) : Option (Tree P L) :=
  items.foldr (prepend category) none

def descendingChain (category : Bool) (items : List (P × Tree P L)) : Option (Tree P L) :=
  items.reverse.foldl (fun suffix item => prepend category item suffix) none

theorem numeric_join_eval (truth : P → X → Bool) (q : P) (a b : Tree P L) (x : X) :
    eval truth (join false q a b) x = if truth q x then eval truth a x else eval truth b x := by
  rfl

theorem category_join_eval (truth : P → X → Bool) (q : P) (a b : Tree P L) (x : X) :
    eval truth (join true q a b) x = if truth q x then eval truth b x else eval truth a x := by
  rfl

/-- The last child initializes without emitting its (unused) question. -/
theorem singleton_emits_no_question (category : Bool) (q : P) (t : Tree P L) :
    rightChain category [(q,t)] = some t := by rfl

theorem prepend_completed_suffix (category : Bool) (q : P) (t : Tree P L)
    (tail : List (P × Tree P L)) :
    prepend category (q,t) (rightChain category tail) = rightChain category ((q,t)::tail) := by
  rfl

/-- Reverse consumption is the exact right-associated chain, including every
    original question and orientation. No class equality is assumed. -/
theorem reverse_foldl_eq_foldr (step : A → S → S) (items : List A) (initial : S) :
    items.reverse.foldl (fun acc item => step item acc) initial = items.foldr step initial := by
  induction items with
  | nil => rfl
  | cons a as ih => simp only [List.reverse_cons, List.foldl_append, List.foldl_cons,
      List.foldl_nil, List.foldr_cons, ih]

theorem descending_exact_chain (category : Bool) (items : List (P × Tree P L)) :
    descendingChain category items = rightChain category items := by
  exact reverse_foldl_eq_foldr _ _ _

/-- A completed suffix is sufficient state. Earlier children prepend to it;
    no previously processed child list must be retained. -/
theorem suffix_accumulator_invariant (category : Bool)
    (earlier later : List (P × Tree P L)) :
    earlier.reverse.foldl (fun acc item => prepend category item acc)
      (rightChain category later) = rightChain category (earlier ++ later) := by
  rw [reverse_foldl_eq_foldr]
  simp only [rightChain, List.foldr_append]

/-- Batch boundaries preserve every bit of the abstract accumulator, including
    a boundary in the middle of a dimension group. -/
def runBatches (step : S → A → S) (initial : S) (batches : List (List A)) : S :=
  batches.foldl (fun state batch => batch.foldl step state) initial

theorem batches_equal_flattened (step : S → A → S) (batches : List (List A)) (initial : S) :
    runBatches step initial batches = batches.flatten.foldl step initial := by
  induction batches generalizing initial with
  | nil => rfl
  | cons batch rest ih =>
    simp only [runBatches, List.foldl_cons, List.flatten_cons, List.foldl_append] at *
    exact ih _

theorem batched_children_exact (category : Bool) (items : List (P × Tree P L))
    (batches : List (List (P × Tree P L))) (order : batches.flatten = items.reverse) :
    runBatches (fun acc item => prepend category item acc) none batches = rightChain category items := by
  rw [batches_equal_flattened, order]
  exact descending_exact_chain category items

/-- A recursive Cartesian fold. Child subtrees are completed first; their
    indices are descending within the dimension, exactly the carry contract.
    The question for the final child is deliberately unused. -/
def mergeChild (category : Bool) (item : P × Option (Tree P L))
    (suffix : Option (Tree P L)) : Option (Tree P L) :=
  match item.2 with
  | none => none
  | some t => prepend category (item.1,t) suffix

def referenceGrid (category : Nat → Bool) (question : Nat → Nat → P)
    (label : List Nat → L) : List Nat → Nat → List Nat → Option (Tree P L)
  | [], _, coordsPrefix => some (.leaf (label coordsPrefix))
  | b::bs, dim, coordsPrefix =>
    ((List.range b).map fun j =>
      (question dim j, referenceGrid category question label bs (dim+1) (coordsPrefix ++ [j]))).foldr (mergeChild (category dim)) none

def streamingGrid (category : Nat → Bool) (question : Nat → Nat → P)
    (label : List Nat → L) : List Nat → Nat → List Nat → Option (Tree P L)
  | [], _, coordsPrefix => some (.leaf (label coordsPrefix))
  | b::bs, dim, coordsPrefix =>
    ((List.range b).map fun j =>
      (question dim j, streamingGrid category question label bs (dim+1) (coordsPrefix ++ [j]))).reverse.foldl (fun suffix item => mergeChild (category dim) item suffix) none

/-- Induction on the actual dimension list, not an assumed final root equality.
    Local suffix/carry fidelity is the implementation-refinement premise. -/
theorem streaming_equals_reference (category : Nat → Bool) (question : Nat → Nat → P)
    (label : List Nat → L) (radices : List Nat) (dim : Nat) (coordsPrefix : List Nat) :
    streamingGrid category question label radices dim coordsPrefix =
      referenceGrid category question label radices dim coordsPrefix := by
  induction radices generalizing dim coordsPrefix with
  | nil => rfl
  | cons b bs ih =>
    simp only [streamingGrid, referenceGrid, ih, reverse_foldl_eq_foldr]

/-- Positive radices cannot produce an absent reference root. -/
theorem fold_has_root (category : Bool) (items : List (P × Option (Tree P L)))
    (nonempty : items ≠ []) (complete : ∀ item ∈ items, ∃ t, item.2 = some t) :
    ∃ t, items.foldr (mergeChild category) none = some t := by
  cases items with
  | nil => exact False.elim (nonempty rfl)
  | cons a rest =>
    obtain ⟨t,ht⟩ := complete a (by simp)
    simp only [List.foldr_cons, mergeChild, ht]
    cases h : rest.foldr (mergeChild category) none with
    | none => exact ⟨t,rfl⟩
    | some suffix => exact ⟨join category a.1 t suffix,rfl⟩

theorem positive_grid_has_root (category : Nat → Bool) (question : Nat → Nat → P)
    (label : List Nat → L) (radices : List Nat) (positive : ∀ r ∈ radices, 0 < r)
    (dim : Nat) (coordsPrefix : List Nat) :
    ∃ t, referenceGrid category question label radices dim coordsPrefix = some t := by
  induction radices generalizing dim coordsPrefix with
  | nil => exact ⟨.leaf (label coordsPrefix),rfl⟩
  | cons b bs ih =>
    unfold referenceGrid
    apply fold_has_root
    · intro h
      have lengths := congrArg List.length h
      simp only [List.length_map, List.length_range, List.length_nil] at lengths
      have hb := positive b (by simp)
      omega
    · intro item member
      obtain ⟨j,_,rfl⟩ := List.mem_map.mp member
      exact ih (fun r hr => positive r (by simp [hr])) _ _

theorem positive_stream_has_root (category : Nat → Bool) (question : Nat → Nat → P)
    (label : List Nat → L) (radices : List Nat) (positive : ∀ r ∈ radices, 0 < r)
    (dim : Nat) (coordsPrefix : List Nat) :
    ∃ t, streamingGrid category question label radices dim coordsPrefix = some t := by
  rw [streaming_equals_reference]
  exact positive_grid_has_root category question label radices positive dim coordsPrefix

/-- Last-dimension-fast Cartesian order decomposes into complete child blocks.
    Reversal visits the highest child block first and reverses each block. -/
theorem reversed_child_blocks (blocks : List (List A)) :
    blocks.flatten.reverse = (blocks.reverse.map List.reverse).flatten := by
  induction blocks with
  | nil => rfl
  | cons a as ih =>
    simp only [List.flatten_cons, List.reverse_append, ih, List.reverse_cons,
      List.map_append, List.map_cons, List.map_nil, List.flatten_append,
      List.flatten_cons, List.flatten_nil, List.append_nil]

/-- Explicit descending ordinal order. -/
def descendingOrdinals (total : Nat) : List Nat := (List.range total).reverse

theorem descending_ordinals_length (total : Nat) : (descendingOrdinals total).length = total := by
  simp [descendingOrdinals]

theorem descending_ordinals_membership (total i : Nat) : i ∈ descendingOrdinals total ↔ i < total := by
  simp [descendingOrdinals]

theorem descending_ordinals_nodup (total : Nat) : (descendingOrdinals total).Nodup := by
  have h : (List.range total).Nodup := List.nodup_range
  unfold descendingOrdinals List.Nodup
  rw [List.pairwise_reverse]
  simpa only [List.Nodup, ne_comm] using h

theorem descending_decodes_all_cells (radices ds : List Nat)
    (positive : ∀ r ∈ radices, 0 < r) (valid : Within radices ds) :
    ∃ i ∈ descendingOrdinals (volume radices), decode radices i = ds := by
  refine ⟨encode radices ds, ?_, decode_encode radices ds positive valid⟩
  exact (descending_ordinals_membership _ _).mpr (encode_bound radices ds valid)

theorem descending_decodes_once (radices : List Nat) (positive : ∀ r ∈ radices, 0 < r)
    {i j : Nat} (hi : i ∈ descendingOrdinals (volume radices))
    (hj : j ∈ descendingOrdinals (volume radices))
    (same : decode radices i = decode radices j) : i = j := by
  exact decode_injective radices positive i j
    ((descending_ordinals_membership _ _).mp hi)
    ((descending_ordinals_membership _ _).mp hj) same

/-- Each successful batch consumes a positive bounded suffix of the pending
    ordinal coordsPrefix. Stop/resource failures do not construct this relation. -/
inductive Progress : Nat → Nat → Nat → Prop where
  | done (remaining : Nat) : Progress remaining 0 remaining
  | batch {before take middle steps after : Nat}
      (positive : 0 < take) (bounded : take ≤ before) (remaining : middle = before-take)
      (tail : Progress middle steps after) : Progress before (steps+1) after

theorem progress_counts {before steps after : Nat} (run : Progress before steps after) :
    after + steps ≤ before := by
  induction run with
  | done n => omega
  | batch positive bounded remaining tail ih => omega

theorem successful_batches_bounded {before steps : Nat} (run : Progress before steps 0) :
    steps ≤ before := by have := progress_counts run; omega

theorem positive_batch_strict (remaining take : Nat) (positive : 0 < take)
    (bounded : take ≤ remaining) : remaining-take < remaining := by omega

/-- A concrete bounded batch loop terminates for every finite product when
    capacity is positive and each batch succeeds. No completion is inferred
    from a failed allocation, failed audit, cancellation, or timeout. -/
def batchCount (capacity remaining : Nat) (positive : 0 < capacity) : Nat :=
  if h : remaining = 0 then 0
  else 1 + batchCount capacity (remaining - min capacity remaining) positive
termination_by remaining
decreasing_by
  have hp : 0 < min capacity remaining := Nat.lt_min.mpr ⟨positive, by omega⟩
  omega

theorem batchCount_progress (capacity remaining : Nat) (positive : 0 < capacity) :
    Progress remaining (batchCount capacity remaining positive) 0 := by
  induction remaining using Nat.strongRecOn with
  | ind remaining ih =>
    rw [batchCount]
    split
    · rename_i h; subst remaining; exact Progress.done 0
    · rename_i h
      have hp : 0 < min capacity remaining := Nat.lt_min.mpr ⟨positive, by omega⟩
      have hb : min capacity remaining ≤ remaining := Nat.min_le_right _ _
      have ht : remaining-min capacity remaining < remaining := by omega
      have tail := ih _ ht
      have step := Progress.batch hp hb rfl tail
      simpa only [Nat.add_comm] using step

theorem finite_grid_batch_termination (radices : List Nat) (capacity : Nat) (positive : 0 < capacity) :
    batchCount capacity (volume radices) positive ≤ volume radices := by
  exact successful_batches_bounded (batchCount_progress _ _ _)

/-- Every occupied slot contains one committed root, not one graph node. -/
def handles (slots : List (Option A)) : List A := slots.filterMap id

theorem accumulator_handle_bound (slots : List (Option A)) :
    (handles slots).length ≤ slots.length := by
  induction slots with
  | nil => simp [handles]
  | cons slot rest ih =>
    cases slot <;> simp only [handles, List.filterMap_cons, List.length_cons] at * <;> simp_all <;> omega

theorem twelve_accumulator_handles (slots : List (Option A)) (shape : slots.length = 12) :
    (handles slots).length ≤ 12 := by have := accumulator_handle_bound slots; omega

/-- The completed root exists only after all dimension accumulators are empty.
    This excludes an artificial 12+1 persistent-root count at completion. -/
def liveRoots (slots : List (Option A)) (root : Option A) : List A :=
  handles slots ++ root.toList

theorem live_root_bound (slots : List (Option A)) (root : Option A)
    (exclusive : root ≠ none → handles slots = []) :
    (liveRoots slots root).length ≤ max slots.length 1 := by
  cases root with
  | none =>
    have h := accumulator_handle_bound slots
    simp only [liveRoots, Option.toList_none, List.append_nil]
    exact Nat.le_trans h (Nat.le_max_left _ _)
  | some r =>
    have h := exclusive (by simp)
    simp only [liveRoots, h, List.nil_append, Option.toList_some, List.length_cons, List.length_nil]
    exact Nat.le_max_right _ _

theorem fixed_twelve_live_roots (slots : List (Option A)) (root : Option A)
    (shape : slots.length = 12) (exclusive : root ≠ none → handles slots = []) :
    (liveRoots slots root).length ≤ 12 := by
  have h := live_root_bound slots root exclusive
  simpa only [shape, Nat.max_eq_left (by omega : 1 ≤ 12)] using h

/-- Append preservation follows from exact immutable old-node payloads. -/
theorem append_preserves_each_live_root {old current : DAG P L}
    (ext : Extension old current) (roots : List (Fin old.size)) (truth : P → X → Bool)
    (r : Fin old.size) (_live : r ∈ roots) (x : X) :
    route current truth (liftId ext.size_le r) x = route old truth r x := by
  exact extension_route ext truth r x

/-- The Arena's exact mapping of a staged expression is enough, including
    exact interning and equal-child elimination. -/
theorem staged_mapping_preserves_expression (arena : DAG P L)
    (tree : Tree P L) (root : Fin arena.size) (mapped : Represents arena tree root)
    (truth : P → X → Bool) (x : X) : eval truth tree x = route arena truth root x := by
  exact represented_tree_evaluation arena truth mapped x

/-- Future GC is sound only for a root set containing EVERY current handle.
    The old→new identity is not reused after collection. -/
theorem collect_preserves_each_live_root {old current : DAG P L}
    (roots : List (Fin old.size)) (marked : Fin old.size → Prop)
    (audit : MarkAudit old roots marked) (remap : ExactRenaming old current marked)
    (truth : P → X → Bool) (r : Fin old.size) (live : r ∈ roots) (x : X) :
    route current truth (remap.map ⟨r,audit.root_marked r live⟩) x = route old truth r x := by
  exact remap_route remap truth r (audit.root_marked r live) x

/-- No surplus retained node relative to ALL live roots; this says nothing
    about the number of nodes below even a single live root. -/
theorem live_collection_exact {g : DAG P L} (roots : List (Fin g.size))
    (marked : Fin g.size → Prop) (audit : MarkAudit g roots marked) (i : Fin g.size) :
    marked i ↔ Reached g roots i := marked_iff_reached audit i


/-- A fresh complete cell audit of the final graph, plus membership of EVERY
    emitted question in the source profile, covers every finite valid input.
    Streaming construction and native margin checks do not replace this gate. -/
theorem fresh_final_graph_audit (p : Profile) (pv : p.Valid)
    (forest : List (Tree Question W))
    (source_covers : ∀ tree ∈ forest, ∀ q ∈ predicates tree, p.Covers q)
    (nativeTransform : List W → L) (radices : List Nat)
    (positive : ∀ r ∈ radices, 0 < r)
    (coordinates : RawKeys → List Nat) (sample : List Nat → RawKeys)
    (within : ∀ x, x.Valid → Within radices (coordinates x))
    (recipe : ∀ x, x.Valid → sample (coordinates x) = representative p x)
    (graph : DAG Question L) (root : Fin graph.size)
    (output_covers : ∀ q ∈ predicates (unfold graph root), p.Covers q)
    (audited : ∀ i, i < volume radices →
      route graph truth root (sample (decode radices i)) =
      nativeTransform (orderedWords truth forest (sample (decode radices i)))) :
    ∀ x, x.Valid → route graph truth root x =
      nativeTransform (orderedWords truth forest x) := by
  intro x hx
  have checked := audited (encode radices (coordinates x))
    (encode_bound radices (coordinates x) (within x hx))
  rw [decode_encode radices (coordinates x) positive (within x hx), recipe x hx] at checked
  have same : eval truth (unfold graph root) (representative p x) =
      eval truth (unfold graph root) x := by
    apply same_questions_same_leaf
    intro q hq
    exact representative_preserves_question p pv x hx q (output_covers q hq)
  rw [route_eq_unfold] at checked ⊢
  rw [← same, checked]
  exact concrete_grid_same_native p pv forest source_covers nativeTransform x hx

end ConverterStreamingGrid
