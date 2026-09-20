"""Shared helpers for preparing RPPD data for torch-molecule models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem


IDENTIFIER_COLUMNS = {
    "UUID",
    "created_at",
    "updated_at",
    "monomer_ID",
    "smiles_list",
}


def load_rppd(path: str | Path) -> pd.DataFrame:
    """Load an RPPD CSV without silently converting identifier columns."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"RPPD CSV not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if "smiles_list" not in frame.columns:
        raise ValueError("Expected a 'smiles_list' column in the RPPD CSV")
    return frame


def molecule_details(smiles: object) -> tuple[bool, str | None, int | None]:
    """Return validity, canonical SMILES, and heavy-atom count."""
    if not isinstance(smiles, str) or not smiles.strip():
        return False, None, None
    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None:
        return False, None, None
    return True, Chem.MolToSmiles(molecule, canonical=True), molecule.GetNumHeavyAtoms()


def add_molecule_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add canonical structure information used by all preparation scripts."""
    details = frame["smiles_list"].map(molecule_details)
    output = frame.copy()
    if "source_row" not in output.columns:
        output.insert(0, "source_row", np.arange(len(output), dtype=int))
    output["valid_smiles"] = details.map(lambda item: item[0])
    output["canonical_smiles"] = details.map(lambda item: item[1])
    output["heavy_atom_count"] = pd.array(
        details.map(lambda item: item[2]), dtype="Int64"
    )
    return output


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Convert one requested target to numeric values and reject unknown columns."""
    if column not in frame.columns:
        raise ValueError(f"Unknown target column: {column}")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def parse_targets(values: Sequence[str]) -> list[str]:
    """Parse repeated and comma-separated command-line target arguments."""
    targets: list[str] = []
    for value in values:
        targets.extend(part.strip() for part in value.split(",") if part.strip())
    return list(dict.fromkeys(targets))


def aggregate_targets(
    frame: pd.DataFrame,
    targets: Iterable[str],
    method: str,
) -> pd.DataFrame:
    """Aggregate replicate simulations for each canonical polymer structure."""
    targets = list(targets)
    working = frame.copy()
    for target in targets:
        working[target] = numeric_series(working, target)

    grouped = working.groupby("canonical_smiles", sort=True, dropna=False)
    rows: list[dict[str, object]] = []
    reducer = np.nanmedian if method == "median" else np.nanmean

    for canonical_smiles, group in grouped:
        row: dict[str, object] = {
            "smiles_list": canonical_smiles,
            "monomer_ID": first_nonempty(group.get("monomer_ID")),
            "replicate_count": len(group),
            "heavy_atom_count": int(group["heavy_atom_count"].iloc[0]),
        }
        for target in targets:
            values = group[target].dropna().to_numpy(dtype=float)
            row[target] = float(reducer(values)) if len(values) else np.nan
            row[f"{target}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{target}_count"] = int(len(values))
        rows.append(row)
    return pd.DataFrame(rows)


def first_nonempty(series: pd.Series | None) -> object:
    if series is None:
        return None
    values = series.dropna()
    values = values[values.astype(str).str.strip().ne("")]
    return values.iloc[0] if len(values) else None


def iqr_mask(series: pd.Series, multiplier: float) -> pd.Series:
    """Return rows inside Tukey IQR fences; disabled when multiplier is nonpositive."""
    if multiplier <= 0:
        return pd.Series(True, index=series.index)
    q1, q3 = series.quantile([0.25, 0.75])
    spread = q3 - q1
    if not np.isfinite(spread) or spread == 0:
        return pd.Series(True, index=series.index)
    return series.between(q1 - multiplier * spread, q3 + multiplier * spread)


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
