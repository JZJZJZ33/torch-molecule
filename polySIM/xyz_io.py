"""Minimal XYZ I/O shared by the conformer, xTB, and system builders."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_xyz(path: str | Path) -> tuple[list[str], np.ndarray, str]:
    """Read the first frame of an XYZ file."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid XYZ file (missing header): {path}")
    try:
        atom_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"Invalid XYZ atom count in {path}") from exc
    atom_lines = lines[2 : 2 + atom_count]
    if len(atom_lines) != atom_count:
        raise ValueError(
            f"XYZ declares {atom_count} atoms but contains {len(atom_lines)}: {path}"
        )

    symbols: list[str] = []
    coordinates: list[list[float]] = []
    for line_number, line in enumerate(atom_lines, start=3):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Malformed XYZ record at {path}:{line_number}")
        symbols.append(fields[0])
        try:
            coordinates.append([float(value) for value in fields[1:4]])
        except ValueError as exc:
            raise ValueError(f"Invalid coordinate at {path}:{line_number}") from exc
    return symbols, np.asarray(coordinates, dtype=float), lines[1]


def write_xyz(
    path: str | Path,
    symbols: list[str],
    coordinates: np.ndarray,
    comment: str = "",
) -> Path:
    """Write one XYZ frame with coordinates in Angstrom."""
    path = Path(path)
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.shape != (len(symbols), 3):
        raise ValueError(
            f"Expected coordinate shape {(len(symbols), 3)}, got {coordinates.shape}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(symbols)), comment]
    lines.extend(
        f"{symbol:<2} {x:16.10f} {y:16.10f} {z:16.10f}"
        for symbol, (x, y, z) in zip(symbols, coordinates)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
