#!/usr/bin/env python3
"""Analyze, rank, and diversity-filter isolated-chain xTB results."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from xyz_io import read_xyz


ENERGY_PATTERN = re.compile(
    r"total\s+energy\s+(?:\.{2,}|[:|])?\s*(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*Eh",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class XTBAnalysisRecord:
    xyz: str
    log: str
    energy_hartree: float
    relative_energy_kcal_mol: float
    selected: bool
    rmsd_to_closest_selected_angstrom: float | None


def extract_final_energy(log_path: str | Path) -> float:
    """Return the last total energy reported in an xTB log."""
    log_path = Path(log_path)
    matches = ENERGY_PATTERN.findall(log_path.read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"No xTB total energy in {log_path}")
    return float(matches[-1])


def ordered_kabsch_rmsd(
    symbols_a: list[str], coordinates_a: np.ndarray,
    symbols_b: list[str], coordinates_b: np.ndarray,
) -> float:
    """Return aligned RMSD for geometries having identical atom ordering."""
    if symbols_a != symbols_b:
        raise ValueError("Cannot compare geometries with different atom ordering")
    a = np.asarray(coordinates_a, dtype=float)
    b = np.asarray(coordinates_b, dtype=float)
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    difference = a @ rotation - b
    return float(np.sqrt(np.mean(np.sum(difference * difference, axis=1))))


def analyze_results(
    results: list[tuple[Path, Path]],
    *,
    output_directory: Path,
    rmsd_threshold: float = 2.0,
    max_selected: int = 3,
) -> list[XTBAnalysisRecord]:
    """Rank xTB results by energy and select geometrically diverse structures."""
    if rmsd_threshold < 0 or max_selected < 1:
        raise ValueError("rmsd_threshold must be non-negative and max_selected positive")
    loaded = []
    for xyz_path, log_path in results:
        symbols, coordinates, _ = read_xyz(xyz_path)
        loaded.append(
            {
                "xyz": Path(xyz_path).resolve(),
                "log": Path(log_path).resolve(),
                "energy": extract_final_energy(log_path),
                "symbols": symbols,
                "coordinates": coordinates,
            }
        )
    if not loaded:
        raise ValueError("No xTB results were supplied")
    loaded.sort(key=lambda item: item["energy"])
    minimum_energy = loaded[0]["energy"]

    selected_indices: list[int] = []
    closest_rmsd: dict[int, float | None] = {}
    for index, candidate in enumerate(loaded):
        if not selected_indices:
            selected_indices.append(index)
            closest_rmsd[index] = None
            continue
        rmsds = [
            ordered_kabsch_rmsd(
                candidate["symbols"], candidate["coordinates"],
                loaded[accepted]["symbols"], loaded[accepted]["coordinates"],
            )
            for accepted in selected_indices
        ]
        closest_rmsd[index] = min(rmsds)
        if len(selected_indices) < max_selected and min(rmsds) >= rmsd_threshold:
            selected_indices.append(index)

    selected_directory = output_directory / "selected"
    selected_directory.mkdir(parents=True, exist_ok=True)
    records = []
    selected_set = set(selected_indices)
    for index, item in enumerate(loaded):
        record = XTBAnalysisRecord(
            xyz=str(item["xyz"]),
            log=str(item["log"]),
            energy_hartree=item["energy"],
            relative_energy_kcal_mol=(item["energy"] - minimum_energy) * 627.509474,
            selected=index in selected_set,
            rmsd_to_closest_selected_angstrom=closest_rmsd.get(index),
        )
        records.append(record)
        if record.selected:
            label = "lowest" if index == 0 else f"diverse_{len([i for i in selected_indices if i <= index]) - 1}"
            shutil.copy2(item["xyz"], selected_directory / f"{label}.xyz")

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "xtb_analysis.json").write_text(
        json.dumps([asdict(record) for record in records], indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_directory / "xtb_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    return records


def discover_results(input_directory: Path) -> list[tuple[Path, Path]]:
    """Pair `*.xtb.log` files with same-stem XYZ files."""
    results = []
    for log_path in sorted(input_directory.rglob("*.xtb.log")):
        xyz_path = log_path.with_suffix("").with_suffix(".xyz")
        if xyz_path.exists():
            results.append((xyz_path, log_path))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--rmsd-threshold", type=float, default=2.0)
    parser.add_argument("--max-selected", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = discover_results(args.input_directory)
    records = analyze_results(
        results,
        output_directory=args.output_directory,
        rmsd_threshold=args.rmsd_threshold,
        max_selected=args.max_selected,
    )
    print(f"Analyzed {len(records)} xTB result(s)")
    print(f"Selected {sum(record.selected for record in records)} conformer(s)")


if __name__ == "__main__":
    main()
