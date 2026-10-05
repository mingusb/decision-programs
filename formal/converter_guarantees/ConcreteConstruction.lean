import RankBoxProgress
import CorrectnessCompletion
import RankSourceCorrespondence

namespace ConverterConcreteConstruction
open ConverterRankBox ConverterGuarantees

def truth (q : Question) (x : Cell) : Bool := decide (q.Left x)

/-- Executable mathematical reference constructor: finite rank boxes, exact
    fallback questions, and one deterministic source query at each atom.
    This is not a verified translation of the CUDA executable. -/
def completeTree {Label : Type} (source : Cell → Label) (b : Box) (v : b.Valid) :
    Tree Question Label :=
  match h : fallback b with
  | none => .leaf (source b.witness)
  | some (l,r) =>
    .branch b.question
      (completeTree source l (fallback_partition b l r v h).1)
      (completeTree source r (fallback_partition b l r v h).2.1)
termination_by b.volume
decreasing_by
  · exact (partition_strict_volumes b l r (fallback_partition b l r v h)).2.1
  · exact (partition_strict_volumes b l r (fallback_partition b l r v h)).2.2.2

theorem complete_tree_correct_and_bounded {Label : Type} (source : Cell → Label)
    (b : Box) (v : b.Valid) :
    (∀ x, b.Mem x → eval truth (completeTree source b v) x = source x) ∧
    nodes (completeTree source b v) ≤ 2 * b.volume - 1 := by
  generalize hv : b.volume = n
  induction n using Nat.strongRecOn generalizing b with
  | ind n ih =>
    rw [completeTree]
    split
    · rename_i h
      have atom := fallback_none_atomic b v h
      constructor
      · intro x hx
        simp only [eval]
        rw [atomic_witness_unique b v atom x hx]
      · simp only [nodes]
        omega
    · rename_i l r h
      have part := fallback_partition b l r v h
      have sizes := partition_strict_volumes b l r part
      have cl := ih l.volume (by omega) l part.1 rfl
      have cr := ih r.volume (by omega) r part.2.1 rfl
      have route := selected_question_routing b l r v h
      constructor
      · intro x hx
        simp only [eval]
        by_cases hp : b.question.Left x
        · have hl := (route.1 x).mpr ⟨hx,hp⟩
          simpa [truth,hp] using cl.1 x hl
        · have hr := (route.2 x).mpr ⟨hx,hp⟩
          simpa [truth,hp] using cr.1 x hr
      · simp only [nodes]
        have sum := part.2.2.2.2
        omega



open ConverterRankSource

def sourceOnCell {Word Label : Type} {tables : Fin 10 → List FiniteWord}
    (forest : List (Tree (SourceQuestion tables) Word))
    (nativeTransform : List Word → Label) (x : Cell) : Label :=
  nativeTransform (rankWords forest (fromCell x))

/-- A total finite mathematical converter for the whole admitted Forest domain.
    Sorted/bounded table import and exact word-based source representation are
    explicit. There is no assumed region-progress or rank-uniformness premise. -/
def fullForestTree {Word Label : Type} (tables : Fin 10 → List FiniteWord)
    (bounded : ∀ f, (tables f).length ≤ 16777216)
    (forest : List (Tree (SourceQuestion tables) Word))
    (nativeTransform : List Word → Label) : Tree Question Label :=
  completeTree (sourceOnCell forest nativeTransform) (rootBox tables)
    (source_root_box_valid tables bounded)

theorem full_forest_class_equivalence_and_finite_size {Word Label : Type}
    (tables : Fin 10 → List FiniteWord)
    (bounded : ∀ f, (tables f).length ≤ 16777216)
    (sorted : ∀ f, (tables f).Pairwise wordLE)
    (forest : List (Tree (SourceQuestion tables) Word))
    (nativeTransform : List Word → Label) :
    (∀ x : RawInput,
      eval truth (fullForestTree tables bounded forest nativeTransform) (encode tables x).toCell =
      nativeTransform (rawWords forest x)) ∧
    nodes (fullForestTree tables bounded forest nativeTransform) ≤
      2 * (rootBox tables).volume - 1 := by
  have result := complete_tree_correct_and_bounded
    (sourceOnCell forest nativeTransform) (rootBox tables) (source_root_box_valid tables bounded)
  constructor
  · intro x
    have out := result.1 (encode tables x).toCell (every_finite_raw_input_covered tables x)
    change eval truth (completeTree _ _ _) _ = _
    rw [out]
    unfold sourceOnCell
    rw [← exact_word_vector_cell_factorization tables sorted forest x]
  · exact result.2

theorem reference_native_queries_bounded {Word Label : Type}
    (tables : Fin 10 → List FiniteWord)
    (bounded : ∀ f, (tables f).length ≤ 16777216)
    (forest : List (Tree (SourceQuestion tables) Word))
    (nativeTransform : List Word → Label) :
    leaves (fullForestTree tables bounded forest nativeTransform) ≤ (rootBox tables).volume := by
  have bound := (complete_tree_correct_and_bounded (sourceOnCell forest nativeTransform)
    (rootBox tables) (source_root_box_valid tables bounded)).2
  have count := physical_node_leaf_identity (fullForestTree tables bounded forest nativeTransform)
  have pos := box_volume_positive (rootBox tables) (source_root_box_valid tables bounded)
  change nodes (fullForestTree tables bounded forest nativeTransform) ≤ _ at bound
  omega


end ConverterConcreteConstruction