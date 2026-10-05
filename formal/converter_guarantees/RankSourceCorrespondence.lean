import RankBoxProgress
import CorrectnessCompletion

/- Source-routing layer over finite binary32 comparison words.
   Import validation and instruction-level IEEE refinement remain explicit. -/
namespace ConverterRankSource
open ConverterRankBox ConverterGuarantees

structure RawInput where
  numeric : Fin 10 → FiniteWord
  wilderness : Fin 4
  soil : Fin 40

structure RankInput where
  numeric : Fin 10 → Nat
  wilderness : Fin 4
  soil : Fin 40

def encode (tables : Fin 10 → List FiniteWord) (x : RawInput) : RankInput :=
  ⟨fun f => rank wordLE (tables f) (x.numeric f),x.wilderness,x.soil⟩

/-- Numeric source thresholds are identified by their exact occurrence in the
    canonical table. Category tables encode the exact one-hot branch truth
    for every legal category, not a learned approximation. -/
inductive SourceQuestion (tables : Fin 10 → List FiniteWord) where
  | numeric (feature : Fin 10) (pre : List FiniteWord) (cut : FiniteWord)
      (post : List FiniteWord) (table : tables feature = pre ++ cut :: post)
  | wilderness (leftCategories : Fin 4 → Bool)
  | soil (leftCategories : Fin 40 → Bool)

def rawTruth {tables : Fin 10 → List FiniteWord}
    (q : SourceQuestion tables) (x : RawInput) : Bool :=
  match q with
  | .numeric f _ cut _ _ => decide (orderedKey (x.numeric f) < orderedKey cut)
  | .wilderness set => set x.wilderness
  | .soil set => set x.soil

def rankTruth {tables : Fin 10 → List FiniteWord}
    (q : SourceQuestion tables) (x : RankInput) : Bool :=
  match q with
  | .numeric f pre _ _ _ => decide (x.numeric f < pre.length + 1)
  | .wilderness set => set x.wilderness
  | .soil set => set x.soil

theorem source_question_correspondence (tables : Fin 10 → List FiniteWord)
    (sorted : ∀ f, (tables f).Pairwise wordLE)
    (q : SourceQuestion tables) (x : RawInput) :
    rawTruth q x = rankTruth q (encode tables x) := by
  cases q with
  | numeric f pre cut post ht =>
    simp only [rawTruth,rankTruth,encode]
    have hs : (pre ++ cut :: post).Pairwise wordLE := ht ▸ sorted f
    apply decide_eq_decide.mpr
    simpa only [ht] using (finite_word_threshold_branch pre post cut (x.numeric f) hs).symm
  | wilderness set => rfl
  | soil set => rfl

/-- Each source tree returns exactly the same stored leaf word after rank encoding.
    The word type may be UInt32; no numerical addition or probability claim occurs here. -/
theorem source_tree_leaf_word {Word : Type} (tables : Fin 10 → List FiniteWord)
    (sorted : ∀ f, (tables f).Pairwise wordLE)
    (tree : Tree (SourceQuestion tables) Word) (x : RawInput) :
    eval rawTruth tree x = eval rankTruth tree (encode tables x) := by
  induction tree with
  | leaf w => rfl
  | branch q l r il ir =>
    simp only [eval,source_question_correspondence tables sorted q x,il,ir]

def rawWords {Word : Type} {tables : Fin 10 → List FiniteWord}
    (forest : List (Tree (SourceQuestion tables) Word)) (x : RawInput) : List Word :=
  forest.map fun tree => eval rawTruth tree x

def rankWords {Word : Type} {tables : Fin 10 → List FiniteWord}
    (forest : List (Tree (SourceQuestion tables) Word)) (x : RankInput) : List Word :=
  forest.map fun tree => eval rankTruth tree x

theorem source_ordered_leaf_words {Word : Type} (tables : Fin 10 → List FiniteWord)
    (sorted : ∀ f, (tables f).Pairwise wordLE)
    (forest : List (Tree (SourceQuestion tables) Word)) (x : RawInput) :
    rawWords forest x = rankWords forest (encode tables x) := by
  apply List.map_congr_left
  intro tree _
  exact source_tree_leaf_word tables sorted tree x

/-- Whatever deterministic ordered FP32/native transform is actually used,
    equality of the complete ordered leaf-word vector preserves its class result.
    This does not replace that transform by real sums or margin argmax. -/
theorem rank_native_class_correspondence {Word Label : Type}
    (tables : Fin 10 → List FiniteWord)
    (sorted : ∀ f, (tables f).Pairwise wordLE)
    (forest : List (Tree (SourceQuestion tables) Word))
    (nativeTransform : List Word → Label) (x : RawInput) :
    nativeTransform (rawWords forest x) =
      nativeTransform (rankWords forest (encode tables x)) := by
  rw [source_ordered_leaf_words tables sorted forest x]

theorem same_rank_same_native_class {Word Label : Type}
    (tables : Fin 10 → List FiniteWord)
    (sorted : ∀ f, (tables f).Pairwise wordLE)
    (forest : List (Tree (SourceQuestion tables) Word))
    (nativeTransform : List Word → Label) (x y : RawInput)
    (same : encode tables x = encode tables y) :
    nativeTransform (rawWords forest x) = nativeTransform (rawWords forest y) := by
  rw [rank_native_class_correspondence tables sorted forest nativeTransform x,
      rank_native_class_correspondence tables sorted forest nativeTransform y,same]

/-- Exact finite one-hot branch table construction for an imported source cut. -/
def oneHotWord (active : Bool) : FiniteWord :=
  if active then ⟨false,⟨1065353216,by decide⟩⟩ else ⟨false,⟨0,by decide⟩⟩

def categoryLeftTable {n : Nat} (indicator : Fin n) (cut : FiniteWord)
    (category : Fin n) : Bool :=
  decide (orderedKey (oneHotWord (decide (category = indicator))) < orderedKey cut)

theorem canonical_zero_tables_rank (cuts : List FiniteWord) (x : FiniteWord) :
    rank wordLE (cuts.map canonicalZero) (canonicalZero x) = rank wordLE cuts x := by
  induction cuts with
  | nil => rfl
  | cons c cs ih =>
    simp only [List.map_cons,rank]
    have hp := canonical_zero_compare c x
    by_cases h : wordLE c x
    · simp [hp.mpr h,h,ih]
    · have hn : ¬wordLE (canonicalZero c) (canonicalZero x) := fun hz => h (hp.mp hz)
      simp [hn,h,ih]



def RankInput.toCell (x : RankInput) : Cell :=
  ⟨List.ofFn x.numeric,x.wilderness.val,x.soil.val⟩

/-- Total decoder; modulo/default branches are unreachable on encoded valid
    raw inputs, as the next theorem proves. -/
def fromCell (x : Cell) : RankInput :=
  ⟨fun f => x.ranks[f.val]?.getD 0,
   ⟨x.wilderness % 4,Nat.mod_lt _ (by decide)⟩,
   ⟨x.soil % 40,Nat.mod_lt _ (by decide)⟩⟩

theorem cell_encoding_roundtrip (x : RankInput) : fromCell x.toCell = x := by
  cases x with
  | mk nums w s =>
    have hn : (fun f : Fin 10 => (List.ofFn nums)[f.val]?.getD 0) = nums := by
      funext f
      simp only [List.getElem?_ofFn, dite_eq_left f.isLt, Option.getD_some]
    have hw : (⟨w.val % 4,Nat.mod_lt _ (by decide)⟩ : Fin 4) = w := by
      apply Fin.ext
      exact Nat.mod_eq_of_lt w.isLt
    have hs : (⟨s.val % 40,Nat.mod_lt _ (by decide)⟩ : Fin 40) = s := by
      apply Fin.ext
      exact Nat.mod_eq_of_lt s.isLt
    simp only [RankInput.toCell,fromCell,hn,hw,hs]

theorem exact_word_vector_cell_factorization {Word : Type}
    (tables : Fin 10 → List FiniteWord)
    (sorted : ∀ f, (tables f).Pairwise wordLE)
    (forest : List (Tree (SourceQuestion tables) Word)) (x : RawInput) :
    rawWords forest x = rankWords forest (fromCell (encode tables x).toCell) := by
  rw [cell_encoding_roundtrip]
  exact source_ordered_leaf_words tables sorted forest x



theorem axes_ofFn_mem {n : Nat} (axes : Fin n → Interval) (values : Fin n → Nat)
    (h : ∀ i, (axes i).Mem (values i)) :
    axesMem (List.ofFn axes) (List.ofFn values) := by
  induction n with
  | zero => simp [axesMem]
  | succ n ih =>
    rw [List.ofFn_succ,List.ofFn_succ]
    exact ⟨h 0,ih _ _ (fun i => h i.succ)⟩

/-- Complete rank/category domain for ten finite numerical Forest features. -/
def rootBox (tables : Fin 10 → List FiniteWord) : Box :=
  ⟨List.ofFn (fun f => ⟨0,(tables f).length⟩),List.range 4,List.range 40⟩

theorem source_root_box_valid (tables : Fin 10 → List FiniteWord)
    (bounded : ∀ f, (tables f).length ≤ 16777216) : (rootBox tables).Valid := by
  refine ⟨?_,by change List.range 4 ≠ []; decide,by change List.range 40 ≠ []; decide,List.nodup_range,List.nodup_range,?_,?_⟩
  · intro a ha
    rcases List.mem_ofFn.mp ha with ⟨f,hf⟩
    rw [← hf]
    exact ⟨Nat.zero_le _,bounded f⟩
  · intro c hc
    exact List.mem_range.mp hc
  · intro c hc
    exact List.mem_range.mp hc

theorem every_finite_raw_input_covered (tables : Fin 10 → List FiniteWord) (x : RawInput) :
    (rootBox tables).Mem (encode tables x).toCell := by
  refine ⟨axes_ofFn_mem _ _ ?_,?_,?_⟩
  · intro f
    exact ⟨Nat.zero_le _,rank_le_length wordLE (tables f) (x.numeric f)⟩
  · exact List.mem_range.mpr x.wilderness.isLt
  · exact List.mem_range.mpr x.soil.isLt


end ConverterRankSource