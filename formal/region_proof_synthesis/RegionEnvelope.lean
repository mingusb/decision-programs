import Std

/-
Region-proof synthesis, 2026-10-03.

Mathematical ingredients read in the Metamath library:
  https://us.metamath.org/mpeuni/supssd.html
  https://us.metamath.org/mpeuni/infssd.html
  https://us.metamath.org/mpeuni/supub.html
  https://us.metamath.org/mpeuni/supnub.html

These are independent Lean proofs of analogous order/cover consequences,
not an import, translation, or replay of set.mm. No global novelty is claimed.
PairedBounds.lean already proves concrete ordered pair bounds. This file
generalizes the order scaffold and adds competitor-specific proof covers;
it must not be reported as discovering the existing paired-bound result.

All implementation-facing premises are explicit. In particular, this file
does NOT prove CUDA FP32 monotonicity, native softprob qualification, path
enumerator coverage, or correspondence to a running converter. The generic
transfer can represent an ordered rounded fold; associativity is never assumed.
-/
namespace RegionEnvelope

def Encloses (le : S → S → Prop) (lo hi value : S) : Prop :=
  le lo value ∧ le value hi

structure Hull (le : S → S → Prop) (value : V → S)
    (allowed : V → Prop) (lo hi : S) : Prop where
  lower : ∀ v, allowed v → le lo (value v)
  upper : ∀ v, allowed v → le (value v) hi
  greatest_lower : ∀ b, (∀ v, allowed v → le b (value v)) → le b lo
  least_upper : ∀ b, (∀ v, allowed v → le (value v) b) → le hi b

/- Feasible joint leaf sequences are a subset of independent combinations.
   Taking their hull cannot weaken the bound. Hull requires endpoints with
   the stated extremal properties; it does not assert attainment or that the
   family is nonempty. No empty-family endpoint is supplied automatically. -/
theorem restricted_hull_tighter
    {le : S → S → Prop} {value : V → S} {fine coarse : V → Prop}
    {fineLo fineHi coarseLo coarseHi : S}
    (subset : ∀ v, fine v → coarse v)
    (hf : Hull le value fine fineLo fineHi)
    (hc : Hull le value coarse coarseLo coarseHi) :
    le coarseLo fineLo ∧ le fineHi coarseHi := by
  constructor
  · exact hf.greatest_lower coarseLo (fun v hv => hc.lower v (subset v hv))
  · exact hf.least_upper coarseHi (fun v hv => hc.upper v (subset v hv))

/- A joint block is an ordered state transformation, not an unordered sum.
   Evaluate each feasible transformation at BOTH incoming endpoints. -/
theorem ordered_block_encloses
    {le : S → S → Prop}
    (trans : ∀ {a b c}, le a b → le b c → le a c)
    (transfer : V → S → S) (allowed : V → Prop)
    (monotone : ∀ v, allowed v → ∀ {a b}, le a b →
      le (transfer v a) (transfer v b))
    {v : V} {lo hi actual outLo outHi : S}
    (feasible : allowed v) (inside : Encloses le lo hi actual)
    (lower : ∀ w, allowed w → le outLo (transfer w lo))
    (upper : ∀ w, allowed w → le (transfer w hi) outHi) :
    Encloses le outLo outHi (transfer v actual) := by
  constructor
  · exact trans (lower v feasible) (monotone v feasible inside.1)
  · exact trans (monotone v feasible inside.2) (upper v feasible)

/- An overapproximation of feasible sequences is safe. Dropping a feasible
   sequence is not licensed by this theorem. -/
theorem enlarge_candidates_preserves_soundness
    {le : S → S → Prop} {value : V → S} {actual candidates : V → Prop}
    {lo hi : S} (coverage : ∀ v, actual v → candidates v)
    (bounds : ∀ v, candidates v → Encloses le lo hi (value v)) :
    ∀ v, actual v → Encloses le lo hi (value v) := by
  intro v hv
  exact bounds v (coverage v hv)

/- Certify each case BEFORE joining the bounds. The covers may overlap;
   every input must be covered, and all cases must prove one common label. -/
theorem common_winner_cover
    (region : X → Prop) (part : J → X → Prop) (predict : X → Label)
    (winner : Label)
    (covers : ∀ x, region x → ∃ j, part j x)
    (certified : ∀ j x, region x → part j x → predict x = winner) :
    ∀ x, region x → predict x = winner := by
  intro x hx
  obtain ⟨j, hj⟩ := covers x hx
  exact certified j x hx hj

/- Two nonempty certified subsets with different labels exclude a uniform
   class on the parent. They need not cover the parent or be disjoint. -/
theorem different_certified_parts_exclude_uniform
    (region left right : X → Prop) (predict : X → Label) (a b : Label)
    (leftInside : ∀ x, left x → region x) (rightInside : ∀ x, right x → region x)
    (leftNonempty : ∃ x, left x) (rightNonempty : ∃ x, right x)
    (leftLabel : ∀ x, left x → predict x = a)
    (rightLabel : ∀ x, right x → predict x = b) (different : a ≠ b) :
    ¬ ∃ winner, ∀ x, region x → predict x = winner := by
  intro ⟨winner, uniform⟩
  obtain ⟨x, hx⟩ := leftNonempty
  obtain ⟨y, hy⟩ := rightNonempty
  apply different
  calc
    a = predict x := (leftLabel x hx).symm
    _ = winner := uniform x (leftInside x hx)
    _ = predict y := (uniform y (rightInside y hy)).symm
    _ = b := rightLabel y hy

/- Opposing score bounds refute a positive-gap proof on a nonempty cell.
   Scores use a common exact integer scale. This says nothing about native
   class ties or whether the parent has more than one predicted class. -/
theorem opposing_bounds_refute_positive_gap
    (cell : X → Prop) (winner rival : X → Int) (upper lower gap : Int)
    (nonempty : ∃ x, cell x) (positive : 0 < gap) (opposed : upper ≤ lower)
    (winnerBound : ∀ x, cell x → winner x ≤ upper)
    (rivalBound : ∀ x, cell x → lower ≤ rival x) :
    ¬ ∀ x, cell x → gap ≤ winner x - rival x := by
  intro certificate
  obtain ⟨x, hx⟩ := nonempty
  have hw := winnerBound x hx
  have hr := rivalBound x hx
  have hg := certificate x hx
  omega

/- A proven class certificate remains valid on any contained region. -/
theorem certificate_restricts
    (large small : X → Prop) (predict : X → Label) (winner : Label)
    (subset : ∀ x, small x → large x)
    (certificate : ∀ x, large x → predict x = winner) :
    ∀ x, small x → predict x = winner := by
  intro x hx
  exact certificate x (subset x hx)

/- Different rivals may need different proof partitions. Every rival covers
   the entire region independently; there is no need to materialize a common
   Cartesian refinement of those partitions. Missing rivals are forbidden by
   the universal premise. The native gate and range premise are external. -/
theorem rival_specific_covers
    (region : X → Prop) (rival : C → Prop)
    (part : C → J → X → Prop) (beats : X → C → Prop)
    (predict : X → Label) (winner : Label)
    (covers : ∀ c x, rival c → region x → ∃ j, part c j x)
    (localProof : ∀ c j x, rival c → region x → part c j x → beats x c)
    (nativeGate : ∀ x, region x → (∀ c, rival c → beats x c) →
      predict x = winner) :
    ∀ x, region x → predict x = winner := by
  intro x hx
  apply nativeGate x hx
  intro c hc
  obtain ⟨j, hj⟩ := covers c x hc hx
  exact localProof c j x hc hx hj

/- A proof portfolio may select a different sound method in each case.
   This is logical selection of certificates, not averaging their predictions. -/
theorem sound_method_portfolio
    (admissible : X → Prop) (methodProves : M → X → Prop) (goal : X → Prop)
    (sound : ∀ m x, admissible x → methodProves m x → goal x)
    (coverage : ∀ x, admissible x → ∃ m, methodProves m x) :
    ∀ x, admissible x → goal x := by
  intro x hx
  obtain ⟨m, hm⟩ := coverage x hx
  exact sound m x hx hm

#print axioms restricted_hull_tighter
#print axioms ordered_block_encloses
#print axioms enlarge_candidates_preserves_soundness
#print axioms common_winner_cover
#print axioms different_certified_parts_exclude_uniform
#print axioms opposing_bounds_refute_positive_gap
#print axioms certificate_restricts
#print axioms rival_specific_covers
#print axioms sound_method_portfolio

end RegionEnvelope
