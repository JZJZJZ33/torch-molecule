"""Create a lossless, analysis-ready RPPD CSV while retaining replicate rows.

Example:
    python data_process/clean_rppd.py data/20260920_rppd.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import add_molecule_columns, load_rppd, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_process/output/rppd_clean.csv"),
    )
    parser.add_argument(
        "--keep-empty-columns",
        action="store_true",
        help="Retain columns that contain no values",
    )
    parser.add_argument(
        "--keep-invalid-smiles",
        action="store_true",
        help="Retain invalid or empty SMILES rows for auditing",
    )
    args = parser.parse_args()

    original = load_rppd(args.input)
    cleaned = add_molecule_columns(original)
    empty_columns = [column for column in original.columns if original[column].isna().all()]

    if not args.keep_invalid_smiles:
        cleaned = cleaned.loc[cleaned["valid_smiles"]].copy()
    if not args.keep_empty_columns:
        cleaned = cleaned.drop(columns=empty_columns)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output, index=False)
    metadata_path = args.output.with_suffix(".metadata.json")
    write_json(
        metadata_path,
        {
            "input": str(args.input),
            "output": str(args.output),
            "input_rows": int(len(original)),
            "output_rows": int(len(cleaned)),
            "unique_canonical_smiles": int(cleaned["canonical_smiles"].nunique()),
            "removed_invalid_smiles_rows": int((~add_molecule_columns(original)["valid_smiles"]).sum()),
            "removed_empty_columns": [] if args.keep_empty_columns else empty_columns,
            "replicate_rows_preserved": True,
        },
    )
    print(f"Clean data written to {args.output}")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
