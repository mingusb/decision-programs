import DynamicAtomRestriction

/- Path-length bound for the abstract exact-literal restriction algorithm.
   This does not bound the number of distinct paths or allocated GPU states. -/
namespace LiteralPathBound
open DynamicAtomRestriction

def atoms : Expr Atom Value → List Atom
  | .leaf _ => []
  | .branch a left right => a :: (atoms left ++ atoms right)

def tupleAtoms (xs : List (Expr Atom Value)) : List Atom := xs.flatMap atoms

theorem restriction_no_new_atom [DecidableEq Atom] (selected : Atom) (truth : Bool)
    (tree : Expr Atom Value) : atoms (restrict selected truth tree) ⊆ atoms tree := by
  induction tree with
  | leaf value => simp [restrict, atoms]
  | branch a left right hl hr =>
    intro q member
    by_cases same : a = selected
    · cases truth <;> simp [restrict, same] at member
      · simp only [atoms, List.mem_cons, List.mem_append]
        exact Or.inr (Or.inr (hr member))
      · simp only [atoms, List.mem_cons, List.mem_append]
        exact Or.inr (Or.inl (hl member))
    · simp only [restrict, same, ↓reduceIte, atoms, List.mem_cons, List.mem_append] at member ⊢
      rcases member with root | l | r
      · exact Or.inl root
      · exact Or.inr (Or.inl (hl l))
      · exact Or.inr (Or.inr (hr r))

theorem restricted_atom_absent [DecidableEq Atom] (selected : Atom) (truth : Bool)
    (tree : Expr Atom Value) : selected ∉ atoms (restrict selected truth tree) := by
  induction tree with
  | leaf value => simp [restrict, atoms]
  | branch a left right hl hr =>
    by_cases same : a = selected
    · cases truth <;> simp [restrict, same, hl, hr]
    · simp [restrict, same, atoms, hl, hr, Ne.symm same]

theorem tuple_restriction_no_new_atom [DecidableEq Atom] (selected : Atom) (truth : Bool)
    (xs : List (Expr Atom Value)) :
    tupleAtoms (xs.map (restrict selected truth)) ⊆ tupleAtoms xs := by
  intro q member
  simp only [tupleAtoms, List.mem_flatMap, List.mem_map] at member ⊢
  obtain ⟨restricted, ⟨tree, ht, eq⟩, hq⟩ := member
  subst restricted
  exact ⟨tree, ht, restriction_no_new_atom selected truth tree hq⟩

theorem tuple_restricted_atom_absent [DecidableEq Atom] (selected : Atom) (truth : Bool)
    (xs : List (Expr Atom Value)) :
    selected ∉ tupleAtoms (xs.map (restrict selected truth)) := by
  intro member
  simp only [tupleAtoms, List.mem_flatMap, List.mem_map] at member
  obtain ⟨restricted, ⟨tree, _, eq⟩, hq⟩ := member
  subst restricted
  exact restricted_atom_absent selected truth tree hq

def RootAtom (xs : List (Expr Atom Value)) (a : Atom) : Prop :=
  ∃ left right, Expr.branch a left right ∈ xs

theorem root_is_atom (xs : List (Expr Atom Value)) (a : Atom) (root : RootAtom xs a) :
    a ∈ tupleAtoms xs := by
  obtain ⟨left, right, member⟩ := root
  simp only [tupleAtoms, List.mem_flatMap]
  exact ⟨.branch a left right, member, by simp [atoms]⟩

def ValidPath [DecidableEq Atom] : List (Expr Atom Value) → List (Atom × Bool) → Prop
  | _, [] => True
  | xs, (a, b) :: tail => RootAtom xs a ∧ ValidPath (xs.map (restrict a b)) tail

theorem path_atoms_from_input [DecidableEq Atom] (path : List (Atom × Bool))
    (xs : List (Expr Atom Value)) (valid : ValidPath xs path) :
    path.map Prod.fst ⊆ tupleAtoms xs := by
  induction path generalizing xs with
  | nil => simp
  | cons pair tail ih =>
    obtain ⟨a,b⟩ := pair
    obtain ⟨root, rest⟩ := valid
    intro q member
    simp only [List.map_cons, List.mem_cons] at member
    rcases member with first | later
    · subst q
      exact root_is_atom xs a root
    · exact tuple_restriction_no_new_atom a b xs (ih _ rest later)

theorem no_repeated_path_atom [DecidableEq Atom] (path : List (Atom × Bool))
    (xs : List (Expr Atom Value)) (valid : ValidPath xs path) :
    (path.map Prod.fst).Nodup := by
  induction path generalizing xs with
  | nil => simp
  | cons pair tail ih =>
    obtain ⟨a,b⟩ := pair
    obtain ⟨_, rest⟩ := valid
    simp only [List.map_cons, List.nodup_cons]
    constructor
    · intro repeated
      have member := path_atoms_from_input tail _ rest repeated
      exact tuple_restricted_atom_absent a b xs member
    · exact ih _ rest

theorem path_length_bounded [DecidableEq Atom] (path : List (Atom × Bool))
    (xs : List (Expr Atom Value)) (valid : ValidPath xs path)
    (support : List Atom) (covers : tupleAtoms xs ⊆ support) :
    path.length ≤ support.length := by
  have nodup := no_repeated_path_atom path xs valid
  have subset : path.map Prod.fst ⊆ support := fun _ member =>
    covers (path_atoms_from_input path xs valid member)
  simpa using nodup.length_le_of_subset subset

#print axioms restriction_no_new_atom
#print axioms restricted_atom_absent
#print axioms tuple_restriction_no_new_atom
#print axioms tuple_restricted_atom_absent
#print axioms root_is_atom
#print axioms path_atoms_from_input
#print axioms no_repeated_path_atom
#print axioms path_length_bounded
end LiteralPathBound
