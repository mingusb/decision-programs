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

/- Strict-improvement witness in exact small integer arithmetic: two trees
   sharing a predicate cancel. This proves the algebraic example only; it is
   not a GPU/native-runtime test or a general floating-point cancellation law. -/
def leftContribution (p : Bool) : Int := if p then 1 else -1
def rightContribution (p : Bool) : Int := if p then -1 else 1

theorem common_predicate_cancellation (p : Bool) :
    leftContribution p + rightContribution p = 0 := by
  cases p <;> decide

/- Permitting mutually inconsistent predicate outcomes introduces a spurious
   value of two, whereas every feasible same-predicate output is zero. -/
theorem independent_extrema_are_strictly_looser :
    leftContribution true + rightContribution false = 2 ∧
    (∀ p, leftContribution p + rightContribution p = 0) := by
  constructor
  · decide
  · exact common_predicate_cancellation

def bitValue (p : Bool) : Int := if p then 1 else 0
def exampleWinner (a b : Bool) : Int := 2 + bitValue a + bitValue b

/- Every rival can have its own two-case cover. The two predicates are exact
   Boolean outcomes of numeric comparisons; this does not quantize inputs. -/
theorem two_rival_example (a b : Bool) :
    1 ≤ exampleWinner a b - 2 * bitValue a ∧
    1 ≤ exampleWinner a b - 2 * bitValue b := by
  cases a <;> cases b <;> decide

theorem rival_a_conditioned_bound (a b : Bool) :
    1 ≤ 2 - bitValue a ∧
    2 - bitValue a ≤ exampleWinner a b - 2 * bitValue a := by
  cases a <;> cases b <;> decide

theorem rival_b_conditioned_bound (a b : Bool) :
    1 ≤ 2 - bitValue b ∧
    2 - bitValue b ≤ exampleWinner a b - 2 * bitValue b := by
  cases a <;> cases b <;> decide

#print axioms restricted_hull_tighter
#print axioms ordered_block_encloses
#print axioms enlarge_candidates_preserves_soundness
#print axioms common_winner_cover
#print axioms certificate_restricts
#print axioms rival_specific_covers
#print axioms sound_method_portfolio
#print axioms common_predicate_cancellation
#print axioms independent_extrema_are_strictly_looser
#print axioms two_rival_example
#print axioms rival_a_conditioned_bound
#print axioms rival_b_conditioned_bound

end RegionEnvelope
