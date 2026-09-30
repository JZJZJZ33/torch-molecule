#!/usr/bin/env python3
"""Pack rigid copies of one chain into a periodic cubic simulation box.

This creates an overlap-free initial geometry by random rigid-body rotations
and translations.  It does not equilibrate the polymer melt; follow it with
force-field minimization and NVT/NPT molecular dynamics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem
from scipy.spatial import cKDTree

try:
    from .xyz_io import read_xyz, write_xyz
except ImportError:  # Direct script execution.
    from xyz_io import read_xyz, write_xyz


AVOGADRO = 6.02214076e23
ANGSTROM3_PER_CM3 = 1.0e24


@dataclass(frozen=True)
class PackedSystem:
    symbols: list[str]
    coordinates: np.ndarray
    box_length_angstrom: float
    chains: int
    atoms_per_chain: int
    density_g_cm3: float


def molecular_weight(symbols: list[str]) -> float:
    """Return molar mass in g/mol from element symbols."""
    table = Chem.GetPeriodicTable()
    weight = 0.0
    for symbol in symbols:
        atomic_number = table.GetAtomicNumber(symbol)
        if atomic_number < 1:
            raise ValueError(f"Unknown element symbol: {symbol}")
        weight += table.GetAtomicWeight(atomic_number)
    return weight


def cubic_box_length(
    symbols: list[str], chains: int, density_g_cm3: float
) -> float:
    """Calculate cubic box length from molecular mass and target density."""
    if chains < 1 or density_g_cm3 <= 0:
        raise ValueError("chains and density_g_cm3 must be positive")
    mass_g = chains * molecular_weight(symbols) / AVOGADRO
    volume_angstrom3 = mass_g / density_g_cm3 * ANGSTROM3_PER_CM3
    return volume_angstrom3 ** (1.0 / 3.0)


def _rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    quaternion = rng.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def pack_chains(
    symbols: list[str],
    chain_coordinates: np.ndarray,
    *,
    chains: int,
    density_g_cm3: float = 0.3,
    minimum_distance: float = 2.0,
    random_seed: int = 42,
    max_attempts_per_chain: int = 20_000,
) -> PackedSystem:
    """Randomly pack rigid chain copies with periodic collision checking."""
    chain_coordinates = np.asarray(chain_coordinates, dtype=float)
    if chain_coordinates.shape != (len(symbols), 3):
        raise ValueError("chain coordinate count does not match symbols")
    if chains < 1 or minimum_distance <= 0 or max_attempts_per_chain < 1:
        raise ValueError("chains, minimum_distance, and max_attempts must be positive")

    box_length = cubic_box_length(symbols, chains, density_g_cm3)
    centered = chain_coordinates - chain_coordinates.mean(axis=0)
    rng = np.random.default_rng(random_seed)
    placed: list[np.ndarray] = []

    for chain_index in range(chains):
        accepted = False
        for _ in range(max_attempts_per_chain):
            rotated = centered @ _rotation_matrix(rng).T
            lower_extent = -rotated.min(axis=0)
            upper_extent = rotated.max(axis=0)
            if np.any(lower_extent + upper_extent >= box_length):
                raise ValueError(
                    f"A chain does not fit wholly inside the {box_length:.3f} A box. "
                    "Use a lower initial density or a more compact chain conformer."
                )
            center = rng.uniform(lower_extent, box_length - upper_extent)
            candidate = rotated + center

            if placed:
                existing = np.vstack(placed)
                tree = cKDTree(existing, boxsize=box_length)
                distances, _ = tree.query(
                    candidate, k=1, distance_upper_bound=minimum_distance
                )
                if np.any(np.isfinite(distances)):
                    continue
            placed.append(candidate)
            accepted = True
            break
        if not accepted:
            raise RuntimeError(
                f"Could not place chain {chain_index + 1}/{chains} after "
                f"{max_attempts_per_chain} attempts. Lower --density, lower "
                "--minimum-distance cautiously, or use Packmol."
            )

    return PackedSystem(
        symbols=symbols * chains,
        coordinates=np.vstack(placed),
        box_length_angstrom=box_length,
        chains=chains,
        atoms_per_chain=len(symbols),
        density_g_cm3=density_g_cm3,
    )


def save_system(system: PackedSystem, output: str | Path) -> Path:
    output = Path(output)
    comment = (
        f"chains={system.chains}; atoms_per_chain={system.atoms_per_chain}; "
        f"periodic_cubic_box_angstrom={system.box_length_angstrom:.10f}; "
        f"target_density_g_cm3={system.density_g_cm3:.10f}; "
        "chain_ranges_are_contiguous=True"
    )
    write_xyz(output, system.symbols, system.coordinates, comment)
    metadata = {
        "coordinate_file": output.name,
        "chains": system.chains,
        "atoms_per_chain": system.atoms_per_chain,
        "total_atoms": len(system.symbols),
        "chain_atom_ranges_zero_based_half_open": [
            [index * system.atoms_per_chain, (index + 1) * system.atoms_per_chain]
            for index in range(system.chains)
        ],
        "periodic_box_angstrom": [system.box_length_angstrom] * 3,
        "target_density_g_cm3": system.density_g_cm3,
    }
    output.with_suffix(".box.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chain_xyz", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--chains", type=int, required=True)
    parser.add_argument(
        "--density",
        type=float,
        default=0.3,
        help="initial packing density in g/cm^3 (default: 0.3)",
    )
    parser.add_argument("--minimum-distance", type=float, default=2.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-attempts-per-chain", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols, coordinates, _ = read_xyz(args.chain_xyz)
    system = pack_chains(
        symbols,
        coordinates,
        chains=args.chains,
        density_g_cm3=args.density,
        minimum_distance=args.minimum_distance,
        random_seed=args.random_seed,
        max_attempts_per_chain=args.max_attempts_per_chain,
    )
    save_system(system, args.output)
    print(f"Wrote {args.output}")
    print(
        f"Periodic cubic box: {system.box_length_angstrom:.6f} A; "
        f"total atoms: {len(system.symbols)}"
    )


if __name__ == "__main__":
    main()
