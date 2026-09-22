"""Inspect an RPPD CSV before selecting model targets.

Example:
    python data_process/inspect_rppd.py data/20260920_rppd.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import add_molecule_columns, load_rppd, write_json


def build_report(frame: pd.DataFrame) -> dict[str, object]:
    enriched = add_molecule_columns(frame)
    duplicate_sizes = enriched.groupby("canonical_smiles", dropna=True).size()

    columns: dict[str, object] = {}
    for column in frame.columns:
        present = frame[column].notna() & frame[column].astype(str).str.strip().ne("")
        numeric = pd.to_numeric(frame[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        item: dict[str, object] = {
            "non_missing": int(present.sum()),
            "missing": int((~present).sum()),
            "unique_non_missing": int(frame.loc[present, column].nunique()),
            "numeric_count": int(numeric.notna().sum()),
        }
        if numeric.notna().any():
            item["numeric_min"] = float(numeric.min())
            item["numeric_median"] = float(numeric.median())
            item["numeric_max"] = float(numeric.max())
        columns[column] = item

    heavy = enriched["heavy_atom_count"].dropna().astype(float)
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "valid_smiles_rows": int(enriched["valid_smiles"].sum()),
        "invalid_smiles_rows": int((~enriched["valid_smiles"]).sum()),
        "unique_canonical_smiles": int(enriched["canonical_smiles"].nunique()),
        "duplicate_structure_groups": int((duplicate_sizes > 1).sum()),
        "largest_replicate_group": int(duplicate_sizes.max()) if len(duplicate_sizes) else 0,
        "heavy_atom_count": {
            "min": float(heavy.min()) if len(heavy) else None,
            "median": float(heavy.median()) if len(heavy) else None,
            "p95": float(heavy.quantile(0.95)) if len(heavy) else None,
            "max": float(heavy.max()) if len(heavy) else None,
        },
        "column_summary": columns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="RPPD CSV to inspect")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_process/output/rppd_report.json"),
        help="JSON report path",
    )
    args = parser.parse_args()

    report = build_report(load_rppd(args.input))
    write_json(args.output, report)
    print(f"Rows: {report['rows']}")
    print(f"Valid SMILES: {report['valid_smiles_rows']}")
    print(f"Unique canonical SMILES: {report['unique_canonical_smiles']}")
    print(f"Duplicate structure groups: {report['duplicate_structure_groups']}")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
