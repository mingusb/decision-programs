import SharedDecisionDAG

/-!
Append-only shared arenas. The premise is equality of every OLD stored node
(including its exact predicate and child IDs), not equality of evaluations.
Predicate semantics and source/native identity are fixed parameters. Hashes,
CUDA validation and journal publication are implementation refinement premises.
-/
namespace ConverterOnlineArena
open ConverterGuarantees ConverterSharedDAG
variable {P L X : Type}

def liftId {a b : Nat} (h : a ≤ b) (i : Fin a) : Fin b :=
  ⟨i.val, Nat.lt_of_lt_of_le i.isLt h⟩

/-- Actual immutable node-prefix equality, with stable old IDs. -/
structure Extension (old current : DAG P L) : Prop where
  size_le : old.size ≤ current.size
  same_node : ∀ i : Fin old.size, current.node (liftId size_le i) = old.node i

@[simp] theorem liftId_value (h : a ≤ b) (i : Fin a) : (liftId h i).val = i.val := rfl

@[simp] theorem lift_child (h : a ≤ b) (i : Fin a) (j : Fin i.val) :
    liftId h (child i j) = child (liftId h i) j := rfl

@[simp] theorem lift_trans (h : a ≤ b) (k : b ≤ c) (i : Fin a) :
    liftId k (liftId h i) = liftId (Nat.le_trans h k) i := rfl

theorem extension_refl (g : DAG P L) : Extension g g := ⟨Nat.le_refl _, fun _ => rfl⟩

theorem extension_trans {a b c : DAG P L} (ab : Extension a b) (bc : Extension b c) :
    Extension a c := by
  refine ⟨Nat.le_trans ab.size_le bc.size_le, ?_⟩
  intro i
  change c.node (liftId bc.size_le (liftId ab.size_le i)) = a.node i
  rw [bc.same_node, ab.same_node]

/-- The exact decoded payload words of each old record are retained. Predicate
    payload P includes the complete ordered terms, not a digest surrogate. -/
structure RawPrefix (old current : Array (RawNode P L)) : Prop where
  size_le : old.size ≤ current.size
  same_entry : ∀ i : Fin old.size, current[liftId size_le i] = old[i]

theorem toNode_equal {i : Nat} {a b : RawNode P L} (same : a = b)
    (ha : validEntry i a) (hb : validEntry i b) : toNode a ha = toNode b hb := by
  subst b
  rfl

theorem checked_raw_prefix_extension (old current : Array (RawNode P L))
    (old_checked : checkTopology old = true) (current_checked : checkTopology current = true)
    (preserved : RawPrefix old current) :
    Extension (fromChecked old old_checked) (fromChecked current current_checked) := by
  refine ⟨preserved.size_le, ?_⟩
  intro i
  apply toNode_equal
  exact preserved.same_entry i

/-- Strong induction follows strictly smaller old child IDs. New suffix nodes
    cannot be reached from any old root because its stored children are fixed. -/
theorem extension_unfold {old current : DAG P L} (ext : Extension old current) :
    ∀ i : Fin old.size, unfold current (liftId ext.size_le i) = unfold old i := by
  intro i
  induction h : i.val using Nat.strongRecOn generalizing i with
  | ind k ih =>
    conv => lhs; rw [unfold, ext.same_node]
    conv => rhs; rw [unfold]
    cases hn : old.node i with
    | leaf c => rfl
    | branch p a b =>
      dsimp only
      rw [← lift_child, ← lift_child]
      rw [ih (child i a).val (by have := a.isLt; simp only [child]; omega) (child i a) rfl]
      rw [ih (child i b).val (by have := b.isLt; simp only [child]; omega) (child i b) rfl]

theorem extension_route {old current : DAG P L} (ext : Extension old current)
    (truth : P → X → Bool) (root : Fin old.size) (x : X) :
    route current truth (liftId ext.size_le root) x = route old truth root x := by
  rw [route_eq_unfold, route_eq_unfold, extension_unfold ext root]

/-- The virtual unfolding uses arbitrary natural-number counts. Correctness
    has no uint64 bound on that count; stored IDs and termination are separate. -/
theorem extension_unfolded_count {old current : DAG P L} (ext : Extension old current)
    (root : Fin old.size) : nodes (unfold current (liftId ext.size_le root)) =
      nodes (unfold old root) := by rw [extension_unfold ext]

theorem old_root_route_bound {old current : DAG P L} (ext : Extension old current)
    (truth : P → X → Bool) (root : Fin old.size) (x : X) :
    routeSteps current truth (liftId ext.size_le root) x ≤ root.val + 1 :=
  route_steps_bounded current truth (liftId ext.size_le root) x

/-- A stored source contract remains valid without a new source evaluation. -/
def Contract (g : DAG P L) (truth : P → X → Bool) (source : X → L)
    (root : Fin g.size) (guard : X → Prop) : Prop :=
  ∀ x, guard x → route g truth root x = source x

theorem extension_contract {old current : DAG P L} (ext : Extension old current)
    (truth : P → X → Bool) (source : X → L) (root : Fin old.size) (guard : X → Prop)
    (cert : Contract old truth source root guard) :
    Contract current truth source (liftId ext.size_le root) guard := by
  intro x hx
  rw [extension_route ext]
  exact cert x hx

theorem extension_guard_reuse {old current : DAG P L} (ext : Extension old current)
    (truth : P → X → Bool) (source : X → L) (root : Fin old.size)
    (guard query : X → Prop) (cert : Contract old truth source root guard)
    (contained : ∀ x, query x → guard x) :
    Contract current truth source (liftId ext.size_le root) query := by
  intro x hx
  exact extension_contract ext truth source root guard cert x (contained x hx)

/-- A finite history of successful checked appends preserves every older root. -/
inductive Extends : DAG P L → DAG P L → Prop where
  | refl (g) : Extends g g
  | step {a b c} : Extends a b → Extension b c → Extends a c

theorem history_extension {a b : DAG P L} (h : Extends a b) : Extension a b := by
  induction h with
  | refl => exact extension_refl _
  | step prior last ih => exact extension_trans ih last

theorem history_contract {old current : DAG P L} (history : Extends old current)
    (truth : P → X → Bool) (source : X → L) (root : Fin old.size) (guard : X → Prop)
    (cert : Contract old truth source root guard) :
    Contract current truth source (liftId (history_extension history).size_le root) guard :=
  extension_contract (history_extension history) truth source root guard cert

/-- Both branch regions are exactly the predicate-restricted parent guard.
    Nonempty sides are an extra progress condition; equivalence uses coverage. -/
structure StrictPartition (truth : P → X → Bool) (p : P)
    (guard left right : X → Prop) : Prop where
  left_exact : ∀ x, left x ↔ guard x ∧ truth p x = true
  right_exact : ∀ x, right x ↔ guard x ∧ truth p x = false
  left_nonempty : ∃ x, left x
  right_nonempty : ∃ x, right x

theorem partition_disjoint {truth : P → X → Bool} {p : P} {guard left right : X → Prop}
    (part : StrictPartition truth p guard left right) :
    ∀ x, ¬ (left x ∧ right x) := by
  intro x both
  have a := (part.left_exact x).mp both.1
  have b := (part.right_exact x).mp both.2
  rw [a.2] at b
  cases b.2

theorem partition_covers {truth : P → X → Bool} {p : P} {guard left right : X → Prop}
    (part : StrictPartition truth p guard left right) :
    ∀ x, guard x ↔ left x ∨ right x := by
  intro x
  constructor
  · intro hx
    cases ht : truth p x
    · exact Or.inr ((part.right_exact x).mpr ⟨hx,ht⟩)
    · exact Or.inl ((part.left_exact x).mpr ⟨hx,ht⟩)
  · intro hx
    cases hx with
    | inl hl => exact ((part.left_exact x).mp hl).1
    | inr hr => exact ((part.right_exact x).mp hr).1

/-- Exact stored parent question plus the two existing child contracts suffices.
    This does not obtain source truth from labels or from containment alone. -/
theorem parent_contract (g : DAG P L) (truth : P → X → Bool) (source : X → L)
    (parent : Fin g.size) (p : P) (a b : Fin parent.val)
    (stored : g.node parent = .branch p a b)
    (guard left right : X → Prop) (part : StrictPartition truth p guard left right)
    (lc : Contract g truth source (child parent a) left)
    (rc : Contract g truth source (child parent b) right) :
    Contract g truth source parent guard := by
  intro x hx
  rw [route, stored]
  dsimp only
  cases ht : truth p x
  · simp only [Bool.false_eq_true, ↓reduceIte]
    exact rc x ((part.right_exact x).mpr ⟨hx,ht⟩)
  · simp only [↓reduceIte]
    exact lc x ((part.left_exact x).mpr ⟨hx,ht⟩)

/-- Growing the arena and composing a parent are independent obligations.
    Only the two old guarded contracts and exact new parent word/ref identity
    are used; neither child source is evaluated again. -/
theorem extended_parent_contract {old current : DAG P L} (ext : Extension old current)
    (truth : P → X → Bool) (source : X → L)
    (leftRoot rightRoot : Fin old.size) (parent : Fin current.size)
    (p : P) (a b : Fin parent.val)
    (stored : current.node parent = .branch p a b)
    (left_id : child parent a = liftId ext.size_le leftRoot)
    (right_id : child parent b = liftId ext.size_le rightRoot)
    (guard left right : X → Prop) (part : StrictPartition truth p guard left right)
    (lc : Contract old truth source leftRoot left)
    (rc : Contract old truth source rightRoot right) :
    Contract current truth source parent guard := by
  apply parent_contract current truth source parent p a b stored guard left right part
  · rw [left_id]
    exact extension_contract ext truth source leftRoot left lc
  · rw [right_id]
    exact extension_contract ext truth source rightRoot right rc

/-- If interning eliminates an equal-child question, both guarded child proofs
    still cover the parent. The resulting stored module can be the old child. -/
theorem equal_child_contract (g : DAG P L) (truth : P → X → Bool) (source : X → L)
    (root : Fin g.size) (p : P) (guard left right : X → Prop)
    (part : StrictPartition truth p guard left right)
    (lc : Contract g truth source root left) (rc : Contract g truth source root right) :
    Contract g truth source root guard := by
  intro x hx
  cases (partition_covers part x).mp hx with
  | inl hl => exact lc x hl
  | inr hr => exact rc x hr

/-- Fractional raw inputs inherit the unchanged qualified rank/native map. -/
theorem extension_raw_contract {Raw : Type} {old current : DAG P L}
    (ext : Extension old current) (truth : P → X → Bool) (source : X → L)
    (native : Raw → L) (encode : Raw → X) (domain : Raw → Prop)
    (root : Fin old.size) (guard : X → Prop) (cert : Contract old truth source root guard)
    (rank_native : ∀ x, domain x → native x = source (encode x))
    (inside : ∀ x, domain x → guard (encode x)) :
    ∀ x, domain x → route current truth (liftId ext.size_le root) (encode x) = native x := by
  intro x hx
  exact (extension_contract ext truth source root guard cert (encode x) (inside x hx)).trans
    (rank_native x hx).symm

end ConverterOnlineArena
