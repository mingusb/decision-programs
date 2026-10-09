import OrderedArithmetic

/-! Conditional class separation using the qualified symmetric native error
premises, not exact exp(0)=1. Codes use units 2^-150. The real exponential
inequality exp(-h) ≤ 1/(1+h), the native relative-error/normality contracts,
FP32/FP64 encoding, and instruction/runtime binding remain explicit obligations.
The theorem does not establish those native execution premises. -/
namespace ConverterNativeClassSeparation
open ConverterArithmetic

abbrev unit : Int := 1427247692705959881058285969449495136382746624 -- 2^150
abbrev guard : Int := unit / 16384 -- 2^-14
abbrev slack : Int := unit / 281474976710656 -- 2^-48
abbrev shiftGap : Int := 3 * (unit / 65536) -- representable 3*2^-16

/-- Keep a representable gap BEFORE applying RN32 monotonicity; substituting
an arbitrary nonrepresentable true gap into the rounded subtraction is invalid. -/
theorem rounded_shift (gap computed : Int) (rn : Int → Int)
    (computedLower : guard ≤ computed) (roundingError : computed-gap ≤ slack)
    (monotone : ∀ a b, a ≤ b → rn a ≤ rn b)
    (representable : rn (-shiftGap) = -shiftGap) : rn (-gap) ≤ -shiftGap := by
  have positiveGap : shiftGap ≤ gap := by
    simp only [guard, slack, shiftGap, unit] at *
    omega
  have rounded := monotone (-gap) (-shiftGap) (by omega)
  simpa only [representable] using rounded

/-- Both exponential envelopes are symmetric with epsilon=2^-16. For a rival
shift at most -h, h=3*2^-16, the upper coefficient is (65536+1)/(65536+3).
Division uses eta=2^-20 and ONE positive denominator. The displayed integer
inequalities are these real contracts after clearing common scale factors.
The original native score and probability dataflow must supply these premises. -/
theorem conditional_probability_order (gap computed denominator winner rival : Int)
    (rn exponential : Int → Int)
    (computedLower : guard ≤ computed) (roundingError : computed-gap ≤ slack)
    (monotone : ∀ a b, a ≤ b → rn a ≤ rn b)
    (representable : rn (-shiftGap) = -shiftGap)
    (denominatorPositive : 0 < denominator)
    (winnerEnvelope : 65535 * unit ≤ 65536 * exponential 0)
    (rivalEnvelope : ∀ x, x ≤ -shiftGap → 65539 * exponential x ≤ 65537 * unit)
    (winnerDivision : 1048575 * unit * exponential 0 ≤
      1048576 * (winner * denominator))
    (rivalDivision : 1048576 * (rival * denominator) ≤
      1048577 * unit * exponential (rn (-gap))) : rival < winner := by
  have shifted := rounded_shift gap computed rn computedLower roundingError monotone representable
  have rivalBound := rivalEnvelope (rn (-gap)) shifted
  have multiplied : rival * denominator < winner * denominator := by
    simp only [unit] at *
    omega
  exact Int.lt_of_mul_lt_mul_right multiplied (by omega)

/-- The separation step itself fits a small exact-integer calculation. -/
theorem symmetric_separation :
    (65537 * 1048577 * 65536 : Int) < 65539 * 65535 * 1048575 := by decide

/-- A positive rounded margin forces the unique largest lower endpoint.
The caller first validates every finite interval and retains the native gate;
concrete RN64 subtraction must refine this monotone rounder fixing zero. -/
theorem qualified_winner_is_max_lower
    (round : Int → Int) (mono : MonotoneRound round) (fixZero : round 0 = 0)
    (gap : Int) (positive : 0 < gap) (lower upper : Nat → Int)
    (first winner : Nat) (rest : List Nat)
    (bounds : ∀ c ∈ first :: rest, lower c ≤ upper c)
    (member : winner ∈ first :: rest)
    (qualified : ∀ c ∈ first :: rest, c ≠ winner →
      gap ≤ round (lower winner - upper c)) :
    firstMaxFrom lower first rest = winner := by
  apply strict_winner_selected lower first winner rest member
  intro c cMember different
  by_cases strict : lower c < lower winner
  · exact strict
  · have bound := bounds c cMember
    have passes := qualified c cMember different
    have nonpositive := mono (lower winner - upper c) 0 (by omega)
    rw [fixZero] at nonpositive
    omega

/-- For a nonempty list of competing classes, its largest upper endpoint is
exactly the hardest original rounded comparison. The winner is excluded from
this list; zero classes and the vacuous singleton case are handled separately. -/
theorem maximum_rival_margin_iff
    (round : Int → Int) (mono : MonotoneRound round)
    (gap lower : Int) (upper : Nat → Int) (first : Nat) (rest : List Nat) :
    gap ≤ round (lower - upper (firstMaxFrom upper first rest)) ↔
      ∀ c ∈ first :: rest, gap ≤ round (lower - upper c) := by
  constructor
  · intro passes c member
    have bound := first_max_score upper rest first c member
    exact Int.le_trans passes (mono _ _ (by omega))
  · intro allPass
    exact allPass _ (first_max_member upper rest first)

#print axioms qualified_winner_is_max_lower
#print axioms maximum_rival_margin_iff
#print axioms rounded_shift
#print axioms conditional_probability_order
#print axioms symmetric_separation
end ConverterNativeClassSeparation
