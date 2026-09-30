#!/usr/bin/env python3
"""Embed a finite polymer chain in 3D and optimize it with RDKit MMFF."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from xyz_io import write_xyz


@dataclass(frozen=True)
class ConformerResult:
    molecule: Chem.Mol
    conformer_id: int
    energy_kcal_mol: float
    converged: bool
    mmff_variant: str


def read_molecule(path: str | Path) -> Chem.Mol:
    """Read the first molecule from SDF or MOL input."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        molecule = next((mol for mol in supplier if mol is not None), None)
    elif suffix == ".mol":
        molecule = Chem.MolFromMolFile(str(path), removeHs=False)
    else:
        raise ValueError("Input must be an .sdf or .mol file")
    if molecule is None:
        raise ValueError(f"RDKit could not read a molecule from {path}")
    return molecule


def build_mmff_conformer(
    molecule: Chem.Mol,
    *,
    num_conformers: int = 1,
    random_seed: int = 42,
    max_iterations: int = 5000,
    mmff_variant: str = "MMFF94s",
) -> ConformerResult:
    """Embed candidates, MMFF-optimize each, and retain the lowest energy."""
    if num_conformers < 1 or max_iterations < 1:
        raise ValueError("num_conformers and max_iterations must be positive")
    if mmff_variant not in {"MMFF94", "MMFF94s"}:
        raise ValueError("mmff_variant must be MMFF94 or MMFF94s")

    molecule = Chem.AddHs(Chem.Mol(molecule))
    if not AllChem.MMFFHasAllMoleculeParams(molecule):
        raise ValueError(
            "MMFF parameters are unavailable for this chain; choose a compatible "
            "chemistry or explicitly implement a different force-field path"
        )

    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = int(random_seed)
    parameters.useRandomCoords = True
    conformer_ids = list(
        AllChem.EmbedMultipleConfs(
            molecule, numConfs=num_conformers, params=parameters
        )
    )
    if not conformer_ids:
        raise RuntimeError("RDKit failed to embed a 3D conformer")

    properties = AllChem.MMFFGetMoleculeProperties(
        molecule, mmffVariant=mmff_variant
    )
    candidates: list[tuple[float, int, bool]] = []
    for conformer_id in conformer_ids:
        status = AllChem.MMFFOptimizeMolecule(
            molecule,
            mmffVariant=mmff_variant,
            confId=conformer_id,
            maxIters=max_iterations,
        )
        force_field = AllChem.MMFFGetMoleculeForceField(
            molecule, properties, confId=conformer_id
        )
        candidates.append((float(force_field.CalcEnergy()), conformer_id, status == 0))

    energy, best_id, converged = min(candidates, key=lambda item: item[0])
    best = Chem.Mol(molecule)
    best.RemoveAllConformers()
    best.AddConformer(molecule.GetConformer(best_id), assignId=True)
    return ConformerResult(best, 0, energy, converged, mmff_variant)


def save_conformer(result: ConformerResult, output: str | Path) -> Path:
    molecule = result.molecule
    conformer = molecule.GetConformer(result.conformer_id)
    symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    formula = rdMolDescriptors.CalcMolFormula(molecule)
    return write_xyz(
        output,
        symbols,
        conformer.GetPositions(),
        (
            f"formula={formula}; force_field={result.mmff_variant}; "
            f"energy_kcal_mol={result.energy_kcal_mol:.10f}; "
            f"converged={result.converged}"
        ),
    )


def save_conformer_sdf(result: ConformerResult, output: str | Path) -> Path:
    """Write the explicit-H 3D molecule while preserving XYZ atom order."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    molecule = Chem.Mol(result.molecule)
    molecule.SetProp("force_field", result.mmff_variant)
    molecule.SetProp("energy_kcal_mol", f"{result.energy_kcal_mol:.10f}")
    molecule.SetProp("converged", str(result.converged))
    writer = Chem.SDWriter(str(output))
    try:
        writer.write(molecule, confId=result.conformer_id)
    finally:
        writer.close()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="chain SDF/MOL from chain_builder.py")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--sdf-output",
        type=Path,
        help="optional explicit-H 3D SDF with atom order matching the XYZ",
    )
    parser.add_argument("--num-conformers", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--variant", choices=("MMFF94", "MMFF94s"), default="MMFF94s")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_mmff_conformer(
        read_molecule(args.input),
        num_conformers=args.num_conformers,
        random_seed=args.random_seed,
        max_iterations=args.max_iterations,
        mmff_variant=args.variant,
    )
    save_conformer(result, args.output)
    if args.sdf_output:
        save_conformer_sdf(result, args.sdf_output)
    print(f"Wrote {args.output}")
    if args.sdf_output:
        print(f"Wrote {args.sdf_output}")
    print(
        f"{result.mmff_variant} energy: {result.energy_kcal_mol:.10f} kcal/mol; "
        f"converged: {result.converged}"
    )


if __name__ == "__main__":
    main()
