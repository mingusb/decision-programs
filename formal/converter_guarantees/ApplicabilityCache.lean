import SharedDecisionDAG

/-!
Residual-source applicability cache. No hash is treated as equality and no
source authority is inferred from cached labels. Exact word vectors and their
original order are retained under a fixed deterministic native transform.
-/
namespace ConverterApplicabilityCache
open ConverterGuarantees ConverterSharedDAG

variable {P W L X V : Type}

def residualize (forced : P → Option Bool) : Tree P W → Tree P W
  | .leaf w => .leaf w
  | .branch p a b => match forced p with
    | some true => residualize forced a
    | some false => residualize forced b
    | none => .branch p (residualize forced a) (residualize forced b)

def ForcedSound (truth : P → X → Bool) (region : X → Prop)
    (forced : P → Option Bool) : Prop :=
  ∀ p b, forced p = some b → ∀ x, region x → truth p x = b

theorem residualize_correct (truth : P → X → Bool) (region : X → Prop)
    (forced : P → Option Bool) (sound : ForcedSound truth region forced)
    (tree : Tree P W) :
    ∀ x, region x → eval truth (residualize forced tree) x = eval truth tree x := by
  induction tree with
  | leaf w => intro x _; rfl
  | branch p a b ia ib =>
    intro x hx
    cases hf : forced p with
    | none =>
      simp only [residualize, hf, eval]
      split
      · exact ia x hx
      · exact ib x hx
    | some v =>
      have ht := sound p v hf x hx
      cases v <;> simp [residualize, hf, eval, ht, ia x hx, ib x hx]

/-- These are leaf WORDS, not real weights; list positions are never permuted. -/
def orderedWords (truth : P → X → Bool) (forest : List (Tree P W)) (x : X) : List W :=
  forest.map (fun tree => eval truth tree x)

def residualForest (forced : P → Option Bool) (forest : List (Tree P W)) : List (Tree P W) :=
  forest.map (residualize forced)

theorem residual_ordered_words (truth : P → X → Bool) (region : X → Prop)
    (forced : P → Option Bool) (sound : ForcedSound truth region forced)
    (forest : List (Tree P W)) (x : X) (hx : region x) :
    orderedWords truth (residualForest forced forest) x = orderedWords truth forest x := by
  induction forest with
  | nil => rfl
  | cons t ts ih =>
    simp only [residualForest, List.map_cons, orderedWords, List.map_map] at *
    rw [residualize_correct truth region forced sound t x hx]
    rw [ih]

def forestClass (truth : P → X → Bool) (transform : List W → L)
    (forest : List (Tree P W)) (x : X) : L := transform (orderedWords truth forest x)

theorem residual_native_class (truth : P → X → Bool) (region : X → Prop)
    (forced : P → Option Bool) (sound : ForcedSound truth region forced)
    (forest : List (Tree P W)) (transform : List W → L) (x : X) (hx : region x) :
    forestClass truth transform (residualForest forced forest) x =
      forestClass truth transform forest x := by
  exact congrArg transform (residual_ordered_words truth region forced sound forest x hx)

/-- A child can residualize its parent's already restricted forest. -/
theorem incremental_residual_correct (truth : P → X → Bool)
    (parent childRegion : X → Prop) (subset : ∀ x, childRegion x → parent x)
    (fp fc : P → Option Bool) (sp : ForcedSound truth parent fp)
    (sc : ForcedSound truth childRegion fc) (forest : List (Tree P W))
    (transform : List W → L) (x : X) (hx : childRegion x) :
    forestClass truth transform (residualForest fc (residualForest fp forest)) x =
      forestClass truth transform forest x := by
  exact (residual_native_class truth childRegion fc sc (residualForest fp forest) transform x hx).trans
    (residual_native_class truth parent fp sp forest transform x (subset x hx))

def DependsOn (view : X → V) (f : X → L) : Prop :=
  ∀ x y, view x = view y → f x = f y

def ProjectionCover (view : X → V) (donor target : X → Prop) : Prop :=
  ∀ x, target x → ∃ y, donor y ∧ view y = view x

/-- A donor-only certificate becomes reusable on a cylindrical guard when
    BOTH residual source and proposed module ignore omitted coordinates. -/
theorem projection_certificate_reuse (view : X → V) (donor target : X → Prop)
    (module residual : X → L) (md : DependsOn view module) (rd : DependsOn view residual)
    (cert : ∀ y, donor y → module y = residual y)
    (cover : ProjectionCover view donor target) :
    ∀ x, target x → module x = residual x := by
  intro x hx
  obtain ⟨y,hy,hview⟩ := cover x hx
  exact (md x y hview.symm).trans ((cert y hy).trans (rd y x hview))

/-- Applicability includes target source equivalence, not only code identity. -/
theorem current_source_reuse (view : X → V) (donor target : X → Prop)
    (module residual source : X → L) (md : DependsOn view module) (rd : DependsOn view residual)
    (cert : ∀ y, donor y → module y = residual y)
    (cover : ProjectionCover view donor target)
    (current : ∀ x, target x → residual x = source x) :
    ∀ x, target x → module x = source x := by
  intro x hx
  exact (projection_certificate_reuse view donor target module residual md rd cert cover x hx).trans
    (current x hx)

/-- Actual residual-code equality, supplied by checked full-key interning. -/
theorem equal_residual_code (truth : P → X → Bool) (transform : List W → L)
    (old current : List (Tree P W)) (same : current = old) (x : X) :
    forestClass truth transform current x = forestClass truth transform old x := by
  rw [same]

/-- Code identity and a donor certificate alone do not authorize another region. -/
theorem donor_only_certificate_insufficient :
    (∀ x : Bool, x = false → (false : Bool) = x) ∧
    ¬ (∀ x : Bool, x = true → (false : Bool) = x) := by
  constructor
  · intro x hx; exact hx.symm
  · intro h; have hf := h true rfl; cases hf

/-- Twelve independent product coordinates model ten rank axes plus exactly
    one wilderness index and one soil index. The latter have finite set domains. -/
abbrev ProductCell := Fin 12 → Nat
abbrev ProductDomain := Fin 12 → Nat → Prop
abbrev ActiveMask := Fin 12 → Bool

def productMem (domain : ProductDomain) (x : ProductCell) : Prop :=
  ∀ i, domain i (x i)

def activeView (active : ActiveMask) (x : ProductCell) : Fin 12 → Nat :=
  fun i => if active i then x i else 0

def activeGuard (domain : ProductDomain) (active : ActiveMask) (x : ProductCell) : Prop :=
  ∀ i, active i = true → domain i (x i)

def splice (active : ActiveMask) (x witness : ProductCell) : ProductCell :=
  fun i => if active i then x i else witness i

theorem splice_in_donor (domain : ProductDomain) (active : ActiveMask)
    (witness x : ProductCell) (nonempty : productMem domain witness)
    (guard : activeGuard domain active x) :
    productMem domain (splice active x witness) := by
  intro i
  cases h : active i
  · simpa [splice,h] using nonempty i
  · simpa [splice,h] using guard i h

theorem splice_same_active (active : ActiveMask) (witness x : ProductCell) :
    activeView active (splice active x witness) = activeView active x := by
  funext i
  cases h : active i <;> simp [activeView,splice,h]

theorem guard_covers_projection (domain : ProductDomain) (active : ActiveMask)
    (witness : ProductCell) (nonempty : productMem domain witness)
    (target : ProductCell → Prop)
    (guard : ∀ x, target x → activeGuard domain active x) :
    ProjectionCover (activeView active) (productMem domain) target := by
  intro x hx
  exact ⟨splice active x witness,
    splice_in_donor domain active witness x nonempty (guard x hx),
    splice_same_active active witness x⟩

/-- Constant-size coordinate inclusion suffices; no donor tree expansion. -/
def guardIncludes (old current : ProductDomain) (active : ActiveMask) : Prop :=
  ∀ i, active i = true → ∀ value, current i value → old i value

theorem coordinate_inclusion_guard (old current : ProductDomain) (active : ActiveMask)
    (inc : guardIncludes old current active) :
    ∀ x, productMem current x → activeGuard old active x := by
  intro x hx i hi
  exact inc i hi (x i) (hx i)

theorem numeric_interval_inclusion (oldLo oldHi newLo newHi value : Nat)
    (inc : oldLo ≤ newLo ∧ newHi ≤ oldHi)
    (inside : newLo ≤ value ∧ value ≤ newHi) :
    oldLo ≤ value ∧ value ≤ oldHi := by omega

theorem category_set_inclusion (old current : List Nat) (inc : current ⊆ old)
    (value : Nat) (inside : value ∈ current) : value ∈ old := inc inside

theorem product_guard_current_source_reuse (old current : ProductDomain)
    (active : ActiveMask) (witness : ProductCell) (nonempty : productMem old witness)
    (module residual source : ProductCell → L)
    (md : DependsOn (activeView active) module)
    (rd : DependsOn (activeView active) residual)
    (cert : ∀ y, productMem old y → module y = residual y)
    (inc : guardIncludes old current active)
    (fresh : ∀ x, productMem current x → residual x = source x) :
    ∀ x, productMem current x → module x = source x := by
  apply current_source_reuse (activeView active) (productMem old) (productMem current)
    module residual source md rd cert
  · exact guard_covers_projection old active witness nonempty (productMem current)
      (coordinate_inclusion_guard old current active inc)
  · exact fresh

/-- The reusable module may be an already checked shared DAG. -/
theorem shared_module_reuse (g : DAG P L) (root : Fin g.size) (truth : P → X → Bool)
    (view : X → V) (donor target : X → Prop) (residual source : X → L)
    (md : DependsOn view (route g truth root)) (rd : DependsOn view residual)
    (cert : ∀ y, donor y → route g truth root y = residual y)
    (cover : ProjectionCover view donor target)
    (fresh : ∀ x, target x → residual x = source x) :
    ∀ x, target x → route g truth root x = source x :=
  current_source_reuse view donor target (route g truth root) residual source md rd cert cover fresh

def predicates : Tree P W → List P
  | .leaf _ => []
  | .branch p a b => p :: (predicates a ++ predicates b)

theorem evaluation_of_predicate_agreement (truth : P → X → Bool)
    (tree : Tree P W) (x y : X)
    (agree : ∀ p ∈ predicates tree, truth p x = truth p y) :
    eval truth tree x = eval truth tree y := by
  induction tree with
  | leaf _ => rfl
  | branch p a b ia ib =>
    have hp := agree p (by simp [predicates])
    have ha : ∀ q ∈ predicates a, truth q x = truth q y := by
      intro q hq; exact agree q (by simp [predicates,hq])
    have hb : ∀ q ∈ predicates b, truth q x = truth q y := by
      intro q hq; exact agree q (by simp [predicates,hq])
    simp only [eval,hp]
    split
    · exact ia ha
    · exact ib hb

theorem tree_dependency_from_syntactic_predicates (truth : P → X → Bool)
    (view : X → V) (tree : Tree P W)
    (deps : ∀ p ∈ predicates tree, DependsOn view (truth p)) :
    DependsOn view (eval truth tree) := by
  intro x y hview
  apply evaluation_of_predicate_agreement truth tree x y
  intro p hp
  exact deps p hp x y hview

theorem forest_dependency_from_tree_dependencies (truth : P → X → Bool)
    (view : X → V) (forest : List (Tree P W)) (transform : List W → L)
    (deps : ∀ tree ∈ forest, DependsOn view (eval truth tree)) :
    DependsOn view (forestClass truth transform forest) := by
  intro x y hview
  apply congrArg transform
  unfold orderedWords
  apply List.map_congr_left
  intro tree ht
  exact deps tree ht x y hview

/-- A freshly proved expanded guard supplies applicability directly. The guard
    can overlap unprocessed regions; donor-region coverage is not required. -/
theorem freshly_expanded_guard_reuse (module source : X → L)
    (expanded target : X → Prop)
    (fresh : ∀ x, expanded x → module x = source x)
    (contained : ∀ x, target x → expanded x) :
    ∀ x, target x → module x = source x := by
  intro x hx
  exact fresh x (contained x hx)

/-- Exact complementary path obligations certify an expanded binary module. -/
theorem expanded_branch_contract (p : X → Bool) (left right source : X → L)
    (guard : X → Prop)
    (hl : ∀ x, guard x → p x = true → left x = source x)
    (hr : ∀ x, guard x → p x = false → right x = source x) :
    ∀ x, guard x → (if p x then left x else right x) = source x := by
  intro x hx
  cases h : p x
  · simpa [h] using hr x hx h
  · simpa [h] using hl x hx h

/-- A fresh expansion certificate for a physical proof tree transfers to its
    exact shared module without rechecking the virtual tree on future hits. -/
theorem expanded_shared_module_reuse (g : DAG P L) (root : Fin g.size)
    (truth : P → X → Bool) (tree : Tree P L) (rep : Represents g tree root)
    (source : X → L) (expanded target : X → Prop)
    (fresh : Certified source truth expanded tree)
    (contained : ∀ x, target x → expanded x) :
    ∀ x, target x → route g truth root x = source x := by
  apply freshly_expanded_guard_reuse (route g truth root) source expanded target
  · exact represented_source_correct g truth source expanded rep fresh
  · exact contained

end ConverterApplicabilityCache