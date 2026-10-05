import Std

/-
Metamath-guided region proof refinement, 2026-10-03.

Source ingredients: infssd (subset infimum monotonicity), supssd (subset
supremum monotonicity), and supremum/upper-bound laws in set.mm. Exact source
hypotheses and representation differences are recorded in the companion docs.
These proofs are independently checked Lean proofs, NOT imported Metamath
proof objects. We require only explicit order laws; no completeness axiom,
empty-family extremum, floating-point reassociation, or real-margin surrogate.

One instantiation must keep the same source, domain, consumed prefix, fixed
winner/rival, and actual rounded margin throughout. CUDA correspondence,
finite score ranges, and the authentic native class gate remain separate
obligations. Incomplete proof searches never supply a new certificate.
-/
namespace CoverRefinement

def LowerCert (le : S → S → Prop) (region : X → Prop)
    (value : X → S) (bound : S) : Prop :=
  ∀ x, region x → le bound (value x)

structure Cover (le : S → S → Prop) (region : X → Prop)
    (part : J → X → Prop) (value : X → S) (bound : J → S) : Prop where
  exhaustive : ∀ x, region x → ∃ j, part j x
  localBound : ∀ j x, region x → part j x → le (bound j) (value x)

/- Only retained/active cases belong to J. Omitting a case requires a proof
   that exhaustiveness still holds; sampling a few cases is insufficient. -/
structure GreatestLowerBound (le : S → S → Prop) (value : J → S)
    (floor : S) : Prop where
  lower : ∀ j, le floor (value j)
  greatest : ∀ b, (∀ j, le b (value j)) → le b floor

theorem exhaustive_cover_floor_sound
    {le : S → S → Prop}
    (trans : ∀ {a b c}, le a b → le b c → le a c)
    {region : X → Prop} {part : J → X → Prop}
    {value : X → S} {bound : J → S} {floor : S}
    (cover : Cover le region part value bound)
    (belowCases : ∀ j, le floor (bound j)) :
    LowerCert le region value floor := by
  intro x hx
  obtain ⟨j, hj⟩ := cover.exhaustive x hx
  exact trans (belowCases j) (cover.localBound j x hx hj)

/- The inequality between aggregate bounds needs BOTH a parent for every
   child and child bounds no weaker than the corresponding parent's bound.
   Refining geometry alone does not establish numeric improvement. This
   inequality is not a soundness certificate: admitting the refined floor
   also requires a locally sound exhaustive Cover. -/
theorem refined_floor_not_weaker
    {le : S → S → Prop}
    (trans : ∀ {a b c}, le a b → le b c → le a c)
    {coarse : J → S} {fine : K → S} {oldFloor newFloor : S}
    (parent : K → J)
    (oldLower : ∀ j, le oldFloor (coarse j))
    (childImproves : ∀ k, le (coarse (parent k)) (fine k))
    (newGlb : GreatestLowerBound le fine newFloor) :
    le oldFloor newFloor := by
  exact newGlb.greatest oldFloor
    (fun k => trans (oldLower (parent k)) (childImproves k))

/- A new exhaustive, locally sound cover gives a sound new certificate.
   Parent inclusion records the intended region refinement explicitly.
   A true subset may inherit its old local bound even when an optional
   tighter computation fails. -/
theorem inherit_parent_bound
    {le : S → S → Prop} {region : X → Prop}
    {coarsePart : J → X → Prop} {finePart : K → X → Prop}
    {value : X → S} {coarseBound : J → S}
    (oldCover : Cover le region coarsePart value coarseBound)
    (parent : K → J)
    (contained : ∀ k x, region x → finePart k x → coarsePart (parent k) x)
    (exhaustive : ∀ x, region x → ∃ k, finePart k x) :
    Cover le region finePart value (fun k => coarseBound (parent k)) := by
  constructor
  · exact exhaustive
  · intro k x hx hk
    exact oldCover.localBound (parent k) x hx (contained k x hx hk)

/- join is maximum in the intended ordered instantiation. Its relevant
   law is explicit, so no total-order or numeric encoding is hidden here. -/
theorem join_certificates_sound
    {le : S → S → Prop} (join : S → S → S)
    (joinLe : ∀ {a b c}, le a c → le b c → le (join a b) c)
    {region : X → Prop} {value : X → S} {a b : S}
    (ha : LowerCert le region value a)
    (hb : LowerCert le region value b) :
    LowerCert le region value (join a b) := by
  intro x hx
  exact joinLe (ha x hx) (hb x hx)

def retain (join : S → S → S) (incumbent : S) : Option S → S
  | none => incumbent
  | some proposed => join incumbent proposed

theorem retain_sound
    {le : S → S → Prop} (join : S → S → S)
    (joinLe : ∀ {a b c}, le a c → le b c → le (join a b) c)
    {region : X → Prop} {value : X → S} {incumbent : S}
    (candidate : Option S)
    (oldSound : LowerCert le region value incumbent)
    (acceptedSound : ∀ b, candidate = some b → LowerCert le region value b) :
    LowerCert le region value (retain join incumbent candidate) := by
  cases candidate with
  | none => exact oldSound
  | some b => exact join_certificates_sound join joinLe oldSound (acceptedSound b rfl)

theorem retain_not_weaker
    {le : S → S → Prop} (refl : ∀ a, le a a)
    (join : S → S → S) (leftLeJoin : ∀ a b, le a (join a b))
    (incumbent : S) (candidate : Option S) :
    le incumbent (retain join incumbent candidate) := by
  cases candidate with
  | none => exact refl incumbent
  | some b => exact leftLeJoin incumbent b

theorem failed_attempt_preserves_bound (join : S → S → S) (incumbent : S) :
    retain join incumbent none = incumbent := rfl

theorem retain_dominates_candidate
    {le : S → S → Prop} (join : S → S → S)
    (rightLeJoin : ∀ a b, le b (join a b)) (incumbent candidate : S) :
    le candidate (retain join incumbent (some candidate)) := by
  exact rightLeJoin incumbent candidate

/- Once each rival has a sufficient certified margin, invoke the existing
   class gate. Each rival can use a different exhaustive cover and a
   different portfolio. Range/native authority is carried by nativeGate. -/
theorem rival_floors_certify_winner
    {le : S → S → Prop}
    (trans : ∀ {a b c}, le a b → le b c → le a c)
    (region : X → Prop) (rival : C → Prop) (gap : C → X → S)
    (floor : C → S) (threshold : S)
    (predict : X → Label) (winner : Label)
    (sound : ∀ c, rival c → LowerCert le region (gap c) (floor c))
    (sufficient : ∀ c, rival c → le threshold (floor c))
    (nativeGate : ∀ x, region x → (∀ c, rival c → le threshold (gap c x)) →
      predict x = winner) :
    ∀ x, region x → predict x = winner := by
  intro x hx
  apply nativeGate x hx
  intro c hc
  exact trans (sufficient c hc) (sound c hc x hx)

/- A failed finer numerical method can always retain a sound parent bound.
   Counterexample: a minimum computed from only examined cases is unsound. -/
theorem partial_cover_minimum_is_not_a_certificate :
    (10 : Int) ≤ (if true then 10 else -1) ∧
    ¬ (∀ p : Bool, (10 : Int) ≤ (if p then 10 else -1)) := by
  constructor
  · decide
  · intro h
    have impossible : ¬ ((10 : Int) ≤ -1) := by decide
    exact impossible (h false)

#print axioms exhaustive_cover_floor_sound
#print axioms refined_floor_not_weaker
#print axioms inherit_parent_bound
#print axioms join_certificates_sound
#print axioms retain_sound
#print axioms retain_not_weaker
#print axioms failed_attempt_preserves_bound
#print axioms retain_dominates_candidate
#print axioms rival_floors_certify_winner
#print axioms partial_cover_minimum_is_not_a_certificate

end CoverRefinement
