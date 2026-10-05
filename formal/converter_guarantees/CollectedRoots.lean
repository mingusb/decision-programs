import OnlineArenaContracts

/-!
Root-directed collection into a fresh namespace. The collector audit checks
root inclusion, BOTH outgoing edges, and a real higher-ID parent witness for
every retained nonroot. We derive exact reachability; it is not a premise.
Payload/remap equality is then sufficient to derive equal virtual unfolding.
Predicate words include ordered coefficient words, not a hash surrogate.
-/
namespace ConverterCollectedRoots
open ConverterGuarantees ConverterSharedDAG ConverterOnlineArena
variable {P L X : Type}

def Reached (g : DAG P L) (roots : List (Fin g.size)) (i : Fin g.size) : Prop :=
  ∃ r, r ∈ roots ∧ ∃ steps, Walk g steps r i

theorem walk_append_edge {g : DAG P L} {a b c : Fin g.size}
    (walk : Walk g n a b) (edge : Edge g b c) : Walk g (n+1) a c := by
  induction walk with
  | zero i => exact Walk.next edge (Walk.zero c)
  | next e tail ih => exact Walk.next e (ih edge)

theorem root_reached (g : DAG P L) (roots : List (Fin g.size))
    (r : Fin g.size) (h : r ∈ roots) : Reached g roots r :=
  ⟨r,h,0,Walk.zero r⟩

theorem reached_closed {g : DAG P L} {roots : List (Fin g.size)}
    {i j : Fin g.size} (h : Reached g roots i) (edge : Edge g i j) :
    Reached g roots j := by
  obtain ⟨r,hr,n,walk⟩ := h
  exact ⟨r,hr,n+1,walk_append_edge walk edge⟩

structure MarkAudit (g : DAG P L) (roots : List (Fin g.size))
    (marked : Fin g.size → Prop) : Prop where
  root_marked : ∀ r, r ∈ roots → marked r
  closed : ∀ {i j}, marked i → Edge g i j → marked j
  parent : ∀ i, marked i → i ∈ roots ∨ ∃ p, marked p ∧ Edge g p i

theorem walk_stays_closed {g : DAG P L} (keep : Fin g.size → Prop)
    (closed : ∀ {i j}, keep i → Edge g i j → keep j)
    {a b : Fin g.size} (walk : Walk g n a b) (start : keep a) : keep b := by
  induction walk with
  | zero i => exact start
  | next edge tail ih => exact ih (closed start edge)

theorem reached_is_least_closed {g : DAG P L} {roots : List (Fin g.size)}
    (keep : Fin g.size → Prop) (include_roots : ∀ r, r ∈ roots → keep r)
    (closed : ∀ {i j}, keep i → Edge g i j → keep j)
    {i : Fin g.size} (h : Reached g roots i) : keep i := by
  obtain ⟨r,hr,n,walk⟩ := h
  exact walk_stays_closed keep closed walk (include_roots r hr)

/-- Strictly increasing parent witnesses cannot continue indefinitely below
    the finite original arena size. They must terminate at an explicit root. -/
theorem marked_reaches_root {g : DAG P L} {roots : List (Fin g.size)}
    {marked : Fin g.size → Prop} (audit : MarkAudit g roots marked) :
    ∀ i, marked i → Reached g roots i := by
  intro i
  induction h : g.size - i.val using Nat.strongRecOn generalizing i with
  | ind k ih =>
    intro mi
    cases audit.parent i mi with
    | inl root => exact root_reached g roots i root
    | inr more =>
      obtain ⟨parent,mp,edge⟩ := more
      have smaller : g.size - parent.val < k := by
        have hd := edge_decreases g edge
        have hp := parent.isLt
        omega
      exact reached_closed (ih (g.size-parent.val) smaller parent rfl mp) edge

theorem marked_iff_reached {g : DAG P L} {roots : List (Fin g.size)}
    {marked : Fin g.size → Prop} (audit : MarkAudit g roots marked) (i) :
    marked i ↔ Reached g roots i := by
  constructor
  · exact marked_reaches_root audit i
  · exact reached_is_least_closed marked audit.root_marked audit.closed

/-- No retained node is unnecessary relative to this exact root set. This is
    reachability minimality, not minimum representation among equivalent DAGs. -/
theorem marked_minimal {g : DAG P L} {roots : List (Fin g.size)}
    {marked : Fin g.size → Prop} (audit : MarkAudit g roots marked)
    (other : Fin g.size → Prop) (include_roots : ∀ r, r ∈ roots → other r)
    (closed : ∀ {i j}, other i → Edge g i j → other j) :
    ∀ i, marked i → other i := by
  intro i hi
  exact reached_is_least_closed other include_roots closed (marked_reaches_root audit i hi)

theorem empty_roots_retain_nothing {g : DAG P L} {marked : Fin g.size → Prop}
    (audit : MarkAudit g [] marked) : ∀ i, ¬ marked i := by
  intro i hi
  obtain ⟨r,hr,_⟩ := marked_reaches_root audit i hi
  simp at hr

theorem retained_parent_witness_strict {g : DAG P L} {roots : List (Fin g.size)}
    {marked : Fin g.size → Prop} (audit : MarkAudit g roots marked)
    (i : Fin g.size) (hi : marked i) (nonroot : i ∉ roots) :
    ∃ p, marked p ∧ Edge g p i ∧ i.val < p.val := by
  cases audit.parent i hi with
  | inl root => exact False.elim (nonroot root)
  | inr more =>
    obtain ⟨p,hp,edge⟩ := more
    exact ⟨p,hp,edge,edge_decreases g edge⟩

theorem marking_unique {g : DAG P L} {roots : List (Fin g.size)}
    {a b : Fin g.size → Prop} (ha : MarkAudit g roots a) (hb : MarkAudit g roots b) :
    ∀ i, a i ↔ b i := by
  intro i
  exact (marked_iff_reached ha i).trans (marked_iff_reached hb i).symm

/-- Referenced term positions, independent of whether their stored words equal
    another position. Term-owner witnesses establish range membership. -/
def TermsUsed {nodes terms : Nat} (marked : Fin nodes → Prop)
    (references : Fin nodes → Fin terms → Prop) (t : Fin terms) : Prop :=
  ∃ i, marked i ∧ references i t

structure TermMarkAudit {nodes terms : Nat} (marked : Fin nodes → Prop)
    (references : Fin nodes → Fin terms → Prop) (kept : Fin terms → Prop) : Prop where
  cover : ∀ i t, marked i → references i t → kept t
  owner : ∀ t, kept t → ∃ i, marked i ∧ references i t

theorem terms_exact {nodes terms : Nat} {marked : Fin nodes → Prop}
    {references : Fin nodes → Fin terms → Prop} {kept : Fin terms → Prop}
    (audit : TermMarkAudit marked references kept) :
    ∀ t, kept t ↔ TermsUsed marked references t := by
  intro t
  constructor
  · exact audit.owner t
  · intro h
    obtain ⟨i,hi,ref⟩ := h
    exact audit.cover i t hi ref

theorem terms_minimal {nodes terms : Nat} {marked : Fin nodes → Prop}
    {references : Fin nodes → Fin terms → Prop} {kept : Fin terms → Prop}
    (audit : TermMarkAudit marked references kept) (other : Fin terms → Prop)
    (cover : ∀ i t, marked i → references i t → other t) :
    ∀ t, kept t → other t := by
  intro t ht
  obtain ⟨i,hi,ref⟩ := audit.owner t ht
  exact cover i t hi ref

/-- Exact coefficients resolved from the term pool, in evaluation order. -/
structure TermWords where
  feature : UInt32
  weight : UInt32
  deriving DecidableEq
inductive PredicateWords where
  | axis : UInt32 → UInt32 → PredicateWords
  | plane : UInt64 → List TermWords → PredicateWords
  deriving DecidableEq

theorem ordered_terms_from_pointwise {n : Nat} (old current : Fin n → TermWords)
    (same : ∀ k, current k = old k) : List.ofFn current = List.ofFn old := by
  have functions : current = old := by funext k; exact same k
  rw [functions]

theorem plane_payload_from_pointwise {n : Nat} (cut : UInt64)
    (old current : Fin n → TermWords) (same : ∀ k, current k = old k) :
    PredicateWords.plane cut (List.ofFn current) = .plane cut (List.ofFn old) := by
  rw [ordered_terms_from_pointwise old current same]

/-- The typed counterpart of a checked bijective old→new map. Both child
    references and the complete ordered predicate payload are preserved.
    Neither evaluation equality nor unfolding equality is assumed. -/
structure ExactRenaming (old current : DAG P L) (marked : Fin old.size → Prop) where
  closed : ∀ {i j}, marked i → Edge old i j → marked j
  map : {i : Fin old.size // marked i} → Fin current.size
  injective : ∀ i j, map i = map j → i = j
  surjective : ∀ j, ∃ i, map i = j
  leaf : ∀ i hi c, old.node i = .leaf c → current.node (map ⟨i,hi⟩) = .leaf c
  branch : ∀ i hi p a b (words : old.node i = .branch p a b),
    ∃ u v : Fin (map ⟨i,hi⟩).val,
      current.node (map ⟨i,hi⟩) = .branch p u v ∧
      child (map ⟨i,hi⟩) u = map ⟨child i a,closed hi (.left i p a b words)⟩ ∧
      child (map ⟨i,hi⟩) v = map ⟨child i b,closed hi (.right i p a b words)⟩

theorem remap_edge {old current : DAG P L} {marked : Fin old.size → Prop}
    (remap : ExactRenaming old current marked) {i j : Fin old.size}
    (hi : marked i) (hj : marked j) (edge : Edge old i j) :
    Edge current (remap.map ⟨i,hi⟩) (remap.map ⟨j,hj⟩) := by
  cases edge with
  | left p a b words =>
    obtain ⟨u,v,newwords,left,_⟩ := remap.branch i hi p a b words
    rw [← left]
    exact Edge.left _ p u v newwords
  | right p a b words =>
    obtain ⟨u,v,newwords,_,right⟩ := remap.branch i hi p a b words
    rw [← right]
    exact Edge.right _ p u v newwords

theorem remap_walk {old current : DAG P L} {marked : Fin old.size → Prop}
    (remap : ExactRenaming old current marked) {i j : Fin old.size}
    (walk : Walk old n i j) : ∀ hi hj,
    Walk current n (remap.map ⟨i,hi⟩) (remap.map ⟨j,hj⟩) := by
  induction walk with
  | zero i => intro hi hj; exact Walk.zero _
  | next edge tail ih =>
    intro hi hj
    have mid := remap.closed hi edge
    exact Walk.next (remap_edge remap hi mid edge) (ih mid hj)

theorem distinct_kept_ids_stay_distinct {old current : DAG P L}
    {marked : Fin old.size → Prop} (remap : ExactRenaming old current marked)
    (i j : {i : Fin old.size // marked i}) (distinct : i ≠ j) : remap.map i ≠ remap.map j := by
  intro same
  exact distinct (remap.injective i j same)

theorem remap_unfold {old current : DAG P L} {marked : Fin old.size → Prop}
    (remap : ExactRenaming old current marked) :
    ∀ i hi, unfold current (remap.map ⟨i,hi⟩) = unfold old i := by
  intro i
  induction h : i.val using Nat.strongRecOn generalizing i with
  | ind k ih =>
    intro hi
    cases words : old.node i with
    | leaf c =>
      conv => lhs; rw [unfold,remap.leaf i hi c words]
      conv => rhs; rw [unfold,words]
    | branch p a b =>
      obtain ⟨u,v,newwords,left,right⟩ := remap.branch i hi p a b words
      conv => lhs; rw [unfold,newwords]
      conv => rhs; rw [unfold,words]
      dsimp only
      rw [left,right]
      rw [ih (child i a).val (by have := a.isLt; simp only [child]; omega) (child i a) rfl]
      rw [ih (child i b).val (by have := b.isLt; simp only [child]; omega) (child i b) rfl]

theorem remap_route {old current : DAG P L} {marked : Fin old.size → Prop}
    (remap : ExactRenaming old current marked) (truth : P → X → Bool)
    (root : Fin old.size) (retained : marked root) (x : X) :
    route current truth (remap.map ⟨root,retained⟩) x = route old truth root x := by
  rw [route_eq_unfold,route_eq_unfold,remap_unfold remap]

theorem remap_virtual_count {old current : DAG P L} {marked : Fin old.size → Prop}
    (remap : ExactRenaming old current marked) (root : Fin old.size) (h : marked root) :
    nodes (unfold current (remap.map ⟨root,h⟩)) = nodes (unfold old root) := by
  rw [remap_unfold remap]

theorem no_surplus_output_node {old current : DAG P L} {roots : List (Fin old.size)}
    {marked : Fin old.size → Prop} (audit : MarkAudit old roots marked)
    (remap : ExactRenaming old current marked) (j : Fin current.size) :
    ∃ i hi, Reached old roots i ∧ remap.map ⟨i,hi⟩ = j := by
  obtain ⟨⟨i,hi⟩,hmap⟩ := remap.surjective j
  exact ⟨i,hi,marked_reaches_root audit i hi,hmap⟩

theorem every_output_reached_from_mapped_root {old current : DAG P L}
    {roots : List (Fin old.size)} {marked : Fin old.size → Prop}
    (audit : MarkAudit old roots marked) (remap : ExactRenaming old current marked)
    (j : Fin current.size) :
    ∃ (r : Fin old.size) (hr : r ∈ roots) (n : Nat),
      Walk current n (remap.map ⟨r,audit.root_marked r hr⟩) j := by
  obtain ⟨i,hi,reachable,mapped⟩ := no_surplus_output_node audit remap j
  obtain ⟨r,hr,n,walk⟩ := reachable
  refine ⟨r,hr,n,?_⟩
  rw [← mapped]
  exact remap_walk remap walk (audit.root_marked r hr) hi

theorem output_reference_terminates {old current : DAG P L}
    {marked : Fin old.size → Prop} (remap : ExactRenaming old current marked)
    (truth : P → X → Bool) (root : Fin old.size) (h : marked root) (x : X) :
    routeSteps current truth (remap.map ⟨root,h⟩) x ≤ (remap.map ⟨root,h⟩).val+1 :=
  route_steps_bounded current truth (remap.map ⟨root,h⟩) x

theorem collected_source_contract {old current : DAG P L}
    {marked : Fin old.size → Prop} (remap : ExactRenaming old current marked)
    (truth : P → X → Bool) (source : X → L) (root : Fin old.size) (h : marked root)
    (guard : X → Prop) (cert : Contract old truth source root guard) :
    Contract current truth source (remap.map ⟨root,h⟩) guard := by
  intro x hx
  rw [remap_route remap]
  exact cert x hx

theorem collected_raw_contract {Raw : Type} {old current : DAG P L}
    {marked : Fin old.size → Prop} (remap : ExactRenaming old current marked)
    (truth : P → X → Bool) (source : X → L) (native : Raw → L) (encode : Raw → X)
    (root : Fin old.size) (h : marked root) (guard : X → Prop) (domain : Raw → Prop)
    (cert : Contract old truth source root guard)
    (correspondence : ∀ x, domain x → native x = source (encode x))
    (inside : ∀ x, domain x → guard (encode x)) :
    ∀ x, domain x → route current truth (remap.map ⟨root,h⟩) (encode x) = native x := by
  intro x hx
  exact (collected_source_contract remap truth source root h guard cert (encode x) (inside x hx)).trans
    (correspondence x hx).symm

end ConverterCollectedRoots
