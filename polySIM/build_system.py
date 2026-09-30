#!/usr/bin/env python3
"""Build a finite polymer chain and an approximate multi-chain box.

This main interface reads one pSMILES from a text file, constructs and
MMFF-optimizes one chain, optionally geometry-optimizes it with GFN-FF and
evaluates it with an xTB electronic method, then packs rigid copies into a
periodic cube. The assembled system is not optimized.
"""

from __future__ import annotations

import argparse
import json
import shutil
from importlib.metadata import version
from pathlib import Path

from rdkit.Chem import rdMolDescriptors

try:
    from .chain_builder import build_chain, read_psmiles_file, save_chain
    from .conf_builder import build_mmff_conformer, save_conformer, save_conformer_sdf
    from .system_builder import pack_chains, save_system
    from .xtb_analysis import analyze_results
    from .xtb_calc import run_xtb
    from .xyz_io import read_xyz
except ImportError:  # Direct script execution.
    from chain_builder import build_chain, read_psmiles_file, save_chain
    from conf_builder import build_mmff_conformer, save_conformer, save_conformer_sdf
    from system_builder import pack_chains, save_system
    from xtb_analysis import analyze_results
    from xtb_calc import run_xtb
    from xyz_io import read_xyz


XTB_METHODS = ("none", "gfn0", "gfn1", "gfn2", "gfnff")
XTB_GEOMETRY_METHODS = ("none", "gfnff")
XTB_OPTIMIZER_ENGINES = ("rf", "lbfgs", "inertial")


def build_workflow(
    *,
    psmiles_file: Path,
    repeat_units: int,
    chains: int,
    output_directory: Path,
    mismatched_bond_policy: str = "strict",
    xtb_method: str = "gfn2",
    xtb_geometry_method: str = "gfnff",
    xtb_charge: int = 0,
    xtb_uhf: int = 0,
    xtb_opt_level: str = "normal",
    xtb_optimizer_engine: str = "inertial",
    xtb_threads: int = 14,
    xtb_hessian: bool = False,
    mmff_variant: str = "MMFF94s",
    num_conformers: int = 1,
    density_g_cm3: float = 0.3,
    minimum_distance: float = 2.0,
    random_seed: int = 42,
    max_iterations: int = 5000,
    max_attempts_per_chain: int = 20_000,
    xtb_executable: str = "xtb",
) -> dict:
    """Execute the complete monomer-to-chain-to-system workflow."""
    xtb_method = xtb_method.lower()
    xtb_geometry_method = xtb_geometry_method.lower()
    if xtb_method not in XTB_METHODS:
        raise ValueError(f"xtb_method must be one of {XTB_METHODS}")
    if xtb_geometry_method not in XTB_GEOMETRY_METHODS:
        raise ValueError(
            f"xtb_geometry_method must be one of {XTB_GEOMETRY_METHODS}"
        )
    if xtb_optimizer_engine not in XTB_OPTIMIZER_ENGINES:
        raise ValueError(
            f"xtb_optimizer_engine must be one of {XTB_OPTIMIZER_ENGINES}"
        )

    psmiles_file = psmiles_file.resolve()
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    psmiles = read_psmiles_file(psmiles_file)
    shutil.copy2(psmiles_file, output_directory / "input.psmiles.txt")

    topology_path = output_directory / "chain_topology.sdf"
    mmff_xyz_path = output_directory / "chain_mmff.xyz"
    mmff_sdf_path = output_directory / "chain_mmff_explicit_h.sdf"
    system_path = output_directory / "system.xyz"

    chain = build_chain(
        psmiles,
        repeat_units,
        mismatched_bond_policy=mismatched_bond_policy,
    )
    save_chain(chain, topology_path)
    conformer = build_mmff_conformer(
        chain,
        num_conformers=num_conformers,
        random_seed=random_seed,
        max_iterations=max_iterations,
        mmff_variant=mmff_variant,
    )
    save_conformer(conformer, mmff_xyz_path)
    save_conformer_sdf(conformer, mmff_sdf_path)

    packing_source = mmff_xyz_path
    xtb_result = None
    geometry_result = None
    if xtb_method != "none":
        if xtb_geometry_method == "gfnff":
            packing_source = output_directory / "chain_gfnff_opt.xyz"
            geometry_result = run_xtb(
                mmff_xyz_path,
                packing_source,
                charge=xtb_charge,
                unpaired_electrons=xtb_uhf,
                method="gfnff",
                optimize=True,
                opt_level=xtb_opt_level,
                optimizer_engine=xtb_optimizer_engine,
                threads=xtb_threads,
                hessian=xtb_hessian,
                xtb_executable=xtb_executable,
            )

        if xtb_method == "gfnff":
            if geometry_result is None:
                packing_source = output_directory / "chain_gfnff_opt.xyz"
                geometry_result = run_xtb(
                    mmff_xyz_path,
                    packing_source,
                    charge=xtb_charge,
                    unpaired_electrons=xtb_uhf,
                    method="gfnff",
                    optimize=True,
                    opt_level=xtb_opt_level,
                    optimizer_engine=xtb_optimizer_engine,
                    threads=xtb_threads,
                    hessian=xtb_hessian,
                    xtb_executable=xtb_executable,
                )
            xtb_result = geometry_result
        else:
            electronic_input = packing_source
            electronic_output = output_directory / f"chain_{xtb_method}_sp.xyz"
            xtb_result = run_xtb(
                electronic_input,
                electronic_output,
                charge=xtb_charge,
                unpaired_electrons=xtb_uhf,
                method=xtb_method,
                optimize=False,
                threads=xtb_threads,
                xtb_executable=xtb_executable,
            )
        analyze_results(
            [(xtb_result.optimized_xyz, xtb_result.log)],
            output_directory=output_directory / "xtb_analysis",
        )

    symbols, coordinates, _ = read_xyz(packing_source)
    packed = pack_chains(
        symbols,
        coordinates,
        chains=chains,
        density_g_cm3=density_g_cm3,
        minimum_distance=minimum_distance,
        random_seed=random_seed,
        max_attempts_per_chain=max_attempts_per_chain,
    )
    save_system(packed, system_path)

    manifest = {
        "input": {
            "psmiles": psmiles,
            "psmiles_file": "input.psmiles.txt",
            "connection_rule": "first [*] is head; second [*] is tail",
            "termination": "hydrogen-capped at both outer ends",
            "mismatched_bond_policy": mismatched_bond_policy,
            "repeat_units_per_chain": repeat_units,
            "chains": chains,
        },
        "chain_construction": {
            "engine": "stk.polymer.Linear",
            "stk_version": version("stk"),
            "sequence_rule": "A + B*(N-2) + C; exactly N repeat units",
        },
        "single_chain": {
            "formula": rdMolDescriptors.CalcMolFormula(conformer.molecule),
            "atoms": conformer.molecule.GetNumAtoms(),
            "mmff_variant": mmff_variant,
            "mmff_energy_kcal_mol": conformer.energy_kcal_mol,
            "mmff_converged": conformer.converged,
            "xtb_method": xtb_method,
            "xtb_energy_hartree": xtb_result.energy_hartree if xtb_result else None,
            "xtb_geometry_method": (
                xtb_geometry_method if geometry_result is not None else "none"
            ),
            "xtb_geometry_energy_hartree": (
                geometry_result.energy_hartree if geometry_result else None
            ),
            "xtb_optimizer_engine": (
                xtb_optimizer_engine if geometry_result is not None else None
            ),
            "xtb_threads": xtb_threads if xtb_result else None,
            "xtb_hessian": xtb_hessian if xtb_result else False,
            "packing_source": packing_source.name,
        },
        "system": {
            "optimization_applied": False,
            "total_atoms": len(packed.symbols),
            "target_density_g_cm3": density_g_cm3,
            "minimum_interchain_distance_angstrom": minimum_distance,
            "periodic_cubic_box_angstrom": packed.box_length_angstrom,
        },
        "files": {
            "chain_topology": topology_path.name,
            "chain_mmff_xyz": mmff_xyz_path.name,
            "chain_explicit_h_sdf": mmff_sdf_path.name,
            "system_xyz": system_path.name,
            "system_box_metadata": system_path.with_suffix(".box.json").name,
        },
    }
    if xtb_result:
        manifest["files"]["chain_xtb_xyz"] = xtb_result.optimized_xyz.name
        manifest["files"]["chain_xtb_log"] = xtb_result.log.name
        manifest["files"]["xtb_analysis_json"] = "xtb_analysis/xtb_analysis.json"
        manifest["files"]["xtb_analysis_csv"] = "xtb_analysis/xtb_analysis.csv"
    if geometry_result:
        manifest["files"]["chain_geometry_optimized_xyz"] = packing_source.name
        manifest["files"]["chain_geometry_optimization_log"] = geometry_result.log.name
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psmiles-file", type=Path, required=True)
    parser.add_argument("-n", "--repeat-units", type=int, required=True)
    parser.add_argument("--chains", type=int, required=True)
    parser.add_argument(
        "--mismatched-bond-policy",
        choices=("strict", "single"),
        default="strict",
    )
    parser.add_argument("-o", "--output-directory", type=Path, required=True)
    parser.add_argument("--xtb-method", choices=XTB_METHODS, default="gfn2")
    parser.add_argument(
        "--xtb-geometry-method",
        choices=XTB_GEOMETRY_METHODS,
        default="gfnff",
        help="single-chain geometry stage before the selected xTB method",
    )
    parser.add_argument("--xtb-charge", type=int, default=0)
    parser.add_argument("--xtb-uhf", type=int, default=0)
    parser.add_argument("--xtb-opt-level", choices=("crude", "sloppy", "loose", "normal", "tight", "verytight"), default="normal")
    parser.add_argument(
        "--xtb-optimizer-engine",
        choices=XTB_OPTIMIZER_ENGINES,
        default="inertial",
        help="optimizer for the xTB geometry stage",
    )
    parser.add_argument("--xtb-threads", type=int, default=14)
    parser.add_argument("--xtb-hessian", action="store_true")
    parser.add_argument("--xtb-executable", default="xtb")
    parser.add_argument("--mmff-variant", choices=("MMFF94", "MMFF94s"), default="MMFF94s")
    parser.add_argument("--num-conformers", type=int, default=1)
    parser.add_argument("--density", type=float, default=0.3)
    parser.add_argument("--minimum-distance", type=float, default=2.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--max-attempts-per-chain", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_workflow(
        psmiles_file=args.psmiles_file,
        repeat_units=args.repeat_units,
        chains=args.chains,
        output_directory=args.output_directory,
        mismatched_bond_policy=args.mismatched_bond_policy,
        xtb_method=args.xtb_method,
        xtb_geometry_method=args.xtb_geometry_method,
        xtb_charge=args.xtb_charge,
        xtb_uhf=args.xtb_uhf,
        xtb_opt_level=args.xtb_opt_level,
        xtb_optimizer_engine=args.xtb_optimizer_engine,
        xtb_threads=args.xtb_threads,
        xtb_hessian=args.xtb_hessian,
        mmff_variant=args.mmff_variant,
        num_conformers=args.num_conformers,
        density_g_cm3=args.density,
        minimum_distance=args.minimum_distance,
        random_seed=args.random_seed,
        max_iterations=args.max_iterations,
        max_attempts_per_chain=args.max_attempts_per_chain,
        xtb_executable=args.xtb_executable,
    )
    print(f"Wrote workflow outputs to {args.output_directory.resolve()}")
    print(
        f"Single chain: {manifest['single_chain']['atoms']} atoms; "
        f"system: {manifest['system']['total_atoms']} atoms; "
        f"box: {manifest['system']['periodic_cubic_box_angstrom']:.6f} A"
    )


if __name__ == "__main__":
    main()
