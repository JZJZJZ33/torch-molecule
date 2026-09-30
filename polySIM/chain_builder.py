#!/usr/bin/env python3
"""Expand a two-ended pSMILES repeat unit into a finite H-capped chain.

The repeat unit must contain exactly two terminal, unmapped ``[*]`` atoms. The
first wildcard in the pSMILES is treated as the head and the second as the tail.
For PVDF, the expected input is ``[*]CC(F)(F)[*]``.

Chain assembly uses ``stk.polymer.Linear`` with hydrogen/bromine-capped
building blocks. RDKit is used only to parse pSMILES and interchange molecules
with the downstream MMFF stage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import stk
from rdkit import Chem


def read_psmiles_file(path: str | Path) -> str:
    """Read exactly one non-empty, non-comment pSMILES line from a text file."""
    path = Path(path)
    entries = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(entries) != 1:
        raise ValueError(
            f"Expected exactly one pSMILES entry in {path}, found {len(entries)}"
        )
    if any(character.isspace() for character in entries[0]):
        raise ValueError("The pSMILES entry must not contain whitespace")
    return entries[0]


def _ordered_wildcards(repeat: Chem.Mol) -> tuple[Chem.Atom, Chem.Atom]:
    stars = [atom for atom in repeat.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(stars) != 2:
        raise ValueError("pSMILES must contain exactly two wildcard atoms ('*')")
    if any(star.GetDegree() != 1 for star in stars):
        raise ValueError("Each wildcard must be terminal and have exactly one bond")

    if any(star.GetAtomMapNum() for star in stars):
        raise ValueError("Wildcard atoms must be unmapped; use [*]...[*]")
    # RDKit retains input atom order when parsing SMILES. We deliberately do not
    # canonicalize the repeat unit before using the first wildcard as the head
    # and the second as the tail.
    stars.sort(key=lambda atom: atom.GetIdx())
    return stars[0], stars[1]


def _building_block_smiles(
    psmiles: str,
    replacements: tuple[int, int],
    *,
    force_single_attachment_bonds: bool,
) -> str:
    """Replace ordered wildcards with terminal H/Br atoms for STK."""
    repeat = Chem.MolFromSmiles(psmiles, sanitize=False)
    if repeat is None:
        raise ValueError(f"RDKit could not parse pSMILES: {psmiles!r}")
    editable = Chem.RWMol(repeat)
    head, tail = _ordered_wildcards(editable)
    if editable.GetNumAtoms() == 2:
        raise ValueError("Repeat unit contains no real atoms")
    for wildcard, atomic_number in zip((head, tail), replacements):
        neighbor = wildcard.GetNeighbors()[0]
        bond = editable.GetBondBetweenAtoms(wildcard.GetIdx(), neighbor.GetIdx())
        if force_single_attachment_bonds:
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetBondDir(Chem.BondDir.NONE)
        wildcard.SetAtomicNum(atomic_number)
        wildcard.SetAtomMapNum(0)
    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule)
    return Chem.MolToSmiles(molecule, canonical=True)


def build_chain(
    psmiles: str,
    repeat_units: int,
    *,
    mismatched_bond_policy: str = "strict",
) -> Chem.Mol:
    """Return an H-capped chain containing exactly ``repeat_units`` units."""
    if isinstance(repeat_units, bool) or not isinstance(repeat_units, int):
        raise TypeError("repeat_units must be an integer")
    if repeat_units < 1:
        raise ValueError("repeat_units must be at least 1")
    if not isinstance(psmiles, str) or not psmiles.strip():
        raise ValueError("psmiles must be a non-empty string")

    repeat = Chem.MolFromSmiles(psmiles)
    if repeat is None:
        raise ValueError(f"RDKit could not parse pSMILES: {psmiles!r}")
    if len(Chem.GetMolFrags(repeat)) != 1:
        raise ValueError("pSMILES must describe one connected repeat unit")

    head, tail = _ordered_wildcards(repeat)
    attachment_bonds = tuple(
        repeat.GetBondBetweenAtoms(star.GetIdx(), star.GetNeighbors()[0].GetIdx()).GetBondType()
        for star in (head, tail)
    )
    if mismatched_bond_policy == "strict":
        if attachment_bonds[0] != attachment_bonds[1]:
            raise ValueError("Head and tail wildcard bonds must have the same bond type")
        if attachment_bonds[0] != Chem.BondType.SINGLE:
            raise ValueError(
                "STK Linear creates single inter-unit bonds; use "
                "mismatched_bond_policy='single' to explicitly accept conversion"
            )
    elif mismatched_bond_policy != "single":
        raise ValueError("mismatched_bond_policy must be strict or single")

    force_single = mismatched_bond_policy == "single"
    if repeat_units == 1:
        capped = _building_block_smiles(
            psmiles, (1, 1), force_single_attachment_bonds=force_single
        )
        molecule = stk.BuildingBlock(smiles=capped).to_rdkit_mol()
    else:
        initial = stk.BuildingBlock(
            smiles=_building_block_smiles(
                psmiles, (1, 35), force_single_attachment_bonds=force_single
            ),
            functional_groups=(stk.BromoFactory(),),
        )
        repeating = stk.BuildingBlock(
            smiles=_building_block_smiles(
                psmiles, (35, 35), force_single_attachment_bonds=force_single
            ),
            functional_groups=(stk.BromoFactory(),),
        )
        terminal = stk.BuildingBlock(
            smiles=_building_block_smiles(
                psmiles, (35, 1), force_single_attachment_bonds=force_single
            ),
            functional_groups=(stk.BromoFactory(),),
        )
        sequence = "A" + "B" * (repeat_units - 2) + "C"
        constructed = stk.ConstructedMolecule(
            topology_graph=stk.polymer.Linear(
                building_blocks=(initial, repeating, terminal),
                repeating_unit=sequence,
                num_repeating_units=1,
                orientations=(1,) * repeat_units,
                random_seed=1,
                optimizer=stk.MCHammer(),
            )
        )
        molecule = constructed.to_rdkit_mol()

    molecule = Chem.RemoveHs(molecule)
    Chem.SanitizeMol(molecule)
    # Wildcard-adjacent E/Z markers can become stale when repeat units are
    # connected. Clean and reassign them before 3D embedding; this prevents a
    # native RDKit crash observed in the PI1M compatibility audit.
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule


def save_chain(molecule: Chem.Mol, output: str | Path) -> Path:
    """Save a 2D/topological chain as SDF; hydrogens remain implicit."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    molecule.SetProp("SMILES", Chem.MolToSmiles(molecule, canonical=True))
    writer = Chem.SDWriter(str(output))
    try:
        writer.write(molecule)
    finally:
        writer.close()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--psmiles", help="repeat-unit pSMILES with two terminal * atoms")
    source.add_argument("--psmiles-file", type=Path, help="text file containing one pSMILES")
    parser.add_argument("-n", "--repeat-units", type=int, required=True)
    parser.add_argument(
        "--mismatched-bond-policy",
        choices=("strict", "single"),
        default="strict",
        help="how to join units when wildcard bond orders differ",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    psmiles = args.psmiles or read_psmiles_file(args.psmiles_file)
    molecule = build_chain(
        psmiles,
        args.repeat_units,
        mismatched_bond_policy=args.mismatched_bond_policy,
    )
    save_chain(molecule, args.output)
    print(f"Wrote {args.output}")
    print(f"Canonical SMILES: {Chem.MolToSmiles(molecule, canonical=True)}")


if __name__ == "__main__":
    main()
