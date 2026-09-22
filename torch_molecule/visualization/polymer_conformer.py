"""Build finite polymer chains and generate UFF-optimized 3D conformers.

The accepted pSMILES representation contains exactly two terminal wildcard
atoms (``*``), each bonded to one atom in the repeat unit. Wildcards may be
atom-mapped as ``[*:1]`` (head) and ``[*:2]`` (tail) to make their direction
explicit. Otherwise, their order in the parsed molecule determines the head
and tail.
"""

from dataclasses import dataclass
from typing import Tuple

from rdkit import Chem
from rdkit.Chem import AllChem


DEFAULT_REPEAT_UNITS = 10


@dataclass(frozen=True)
class PolymerConformer:
    """A finite polymer chain and its UFF-optimized 3D representation."""

    smiles: str
    molecule: Chem.Mol
    mol_block: str
    energy: float
    converged: bool
    repeat_units: int


def _ordered_attachment_atoms(mol: Chem.Mol) -> Tuple[Chem.Atom, Chem.Atom]:
    attachments = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(attachments) != 2:
        raise ValueError("pSMILES must contain exactly two wildcard attachment atoms ('*').")

    if any(atom.GetDegree() != 1 for atom in attachments):
        raise ValueError("Each wildcard attachment atom must have exactly one bond.")

    map_numbers = {atom.GetAtomMapNum() for atom in attachments}
    if map_numbers == {1, 2}:
        attachments.sort(key=lambda atom: atom.GetAtomMapNum())
    else:
        attachments.sort(key=lambda atom: atom.GetIdx())
    return attachments[0], attachments[1]


def _repeat_unit_core(psmiles: str) -> Tuple[Chem.Mol, int, int, Chem.BondType]:
    if not isinstance(psmiles, str) or not psmiles.strip():
        raise ValueError("psmiles must be a non-empty string.")

    mol = Chem.MolFromSmiles(psmiles)
    if mol is None:
        raise ValueError(f"Could not parse pSMILES: {psmiles!r}.")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("pSMILES must describe one connected repeat unit.")

    head, tail = _ordered_attachment_atoms(mol)
    head_neighbor = head.GetNeighbors()[0].GetIdx()
    tail_neighbor = tail.GetNeighbors()[0].GetIdx()
    head_bond_type = mol.GetBondBetweenAtoms(head.GetIdx(), head_neighbor).GetBondType()
    tail_bond_type = mol.GetBondBetweenAtoms(tail.GetIdx(), tail_neighbor).GetBondType()
    if head_bond_type != tail_bond_type:
        raise ValueError("Head and tail attachment bonds must have the same bond type.")

    removed = sorted((head.GetIdx(), tail.GetIdx()))

    def remap(index: int) -> int:
        return index - sum(removed_index < index for removed_index in removed)

    editable = Chem.RWMol(mol)
    for index in reversed(removed):
        editable.RemoveAtom(index)
    core = editable.GetMol()
    Chem.SanitizeMol(core)
    if core.GetNumAtoms() == 0:
        raise ValueError("The repeat unit must contain at least one non-wildcard atom.")

    return core, remap(head_neighbor), remap(tail_neighbor), head_bond_type


def expand_psmiles(psmiles: str, repeat_units: int = DEFAULT_REPEAT_UNITS) -> str:
    """Expand a two-ended pSMILES repeat unit into a finite, H-capped chain.

    Parameters
    ----------
    psmiles:
        Repeat-unit SMILES containing two terminal wildcard atoms.
    repeat_units:
        Number of repeat units in the finite chain. Defaults to 10.

    Returns
    -------
    str
        Canonical SMILES for the finite chain. The two outer termini are
        capped by implicit hydrogens.
    """
    if isinstance(repeat_units, bool) or not isinstance(repeat_units, int):
        raise TypeError("repeat_units must be an integer.")
    if repeat_units < 1:
        raise ValueError("repeat_units must be at least 1.")

    core, head_index, tail_index, bond_type = _repeat_unit_core(psmiles)
    atoms_per_unit = core.GetNumAtoms()
    chain = Chem.RWMol(core)

    for unit_index in range(1, repeat_units):
        chain.InsertMol(core)
        previous_tail = (unit_index - 1) * atoms_per_unit + tail_index
        next_head = unit_index * atoms_per_unit + head_index
        chain.AddBond(previous_tail, next_head, bond_type)

    result = chain.GetMol()
    Chem.SanitizeMol(result)
    return Chem.MolToSmiles(result, canonical=True)


def generate_uff_conformer(
    psmiles: str,
    repeat_units: int = DEFAULT_REPEAT_UNITS,
    *,
    random_seed: int = 42,
    max_iterations: int = 1000,
) -> PolymerConformer:
    """Generate and UFF-optimize one 3D conformer for a finite polymer chain."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    smiles = expand_psmiles(psmiles, repeat_units=repeat_units)
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))

    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = int(random_seed)
    parameters.useRandomCoords = True
    conformer_id = AllChem.EmbedMolecule(molecule, parameters)
    if conformer_id < 0:
        raise RuntimeError("RDKit could not generate a 3D conformer for the polymer chain.")
    if not AllChem.UFFHasAllMoleculeParams(molecule):
        raise ValueError(
            "UFF parameters are unavailable for one or more atoms in the polymer chain."
        )

    status = AllChem.UFFOptimizeMolecule(
        molecule, confId=conformer_id, maxIters=max_iterations
    )
    force_field = AllChem.UFFGetMoleculeForceField(molecule, confId=conformer_id)
    energy = float(force_field.CalcEnergy())
    mol_block = Chem.MolToMolBlock(molecule, confId=conformer_id)

    return PolymerConformer(
        smiles=smiles,
        molecule=molecule,
        mol_block=mol_block,
        energy=energy,
        converged=status == 0,
        repeat_units=repeat_units,
    )


class PolymerConformerGenerator:
    """Configurable generator for finite polymer conformers."""

    def __init__(
        self,
        repeat_units: int = DEFAULT_REPEAT_UNITS,
        random_seed: int = 42,
        max_iterations: int = 1000,
    ) -> None:
        self.repeat_units = repeat_units
        self.random_seed = random_seed
        self.max_iterations = max_iterations

    def expand(self, psmiles: str) -> str:
        """Return the finite-chain SMILES without generating coordinates."""
        return expand_psmiles(psmiles, repeat_units=self.repeat_units)

    def generate(self, psmiles: str) -> PolymerConformer:
        """Return a UFF-optimized conformer and browser-ready mol block."""
        return generate_uff_conformer(
            psmiles,
            repeat_units=self.repeat_units,
            random_seed=self.random_seed,
            max_iterations=self.max_iterations,
        )
