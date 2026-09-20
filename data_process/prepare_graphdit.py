"""Prepare deduplicated RPPD structures for Graph-DiT training.

Examples:
    python data_process/prepare_graphdit.py 20260920_rppd.csv
    python data_process/prepare_graphdit.py 20260920_rppd.csv --target density
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import (
    add_molecule_columns,
    aggregate_targets,
    iqr_mask,
    load_rppd,
    parse_targets,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Conditioning property; repeat or comma-separate for multiple properties",
    )
    parser.add_argument("--aggregate", choices=("mean", "median"), default="median")
    parser.add_argument("--max-heavy-atoms", type=int, default=50)
    parser.add_argument(
        "--outlier-iqr",
        type=float,
        default=0.0,
        help="Remove rows outside each target's IQR fences; 0 disables removal",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data_process/output/graphdit")
    )
    args = parser.parse_args()
    targets = parse_targets(args.target)

    frame = add_molecule_columns(load_rppd(args.input))
    frame = frame.loc[
        frame["valid_smiles"] & (frame["heavy_atom_count"] <= args.max_heavy_atoms)
    ].copy()

    if targets:
        prepared = aggregate_targets(frame, targets, args.aggregate)
        prepared = prepared.dropna(subset=targets).copy()
        for target in targets:
            prepared = prepared.loc[iqr_mask(prepared[target], args.outlier_iqr)].copy()
    else:
        prepared = (
            frame.sort_values("source_row")
            .drop_duplicates("canonical_smiles")
            .rename(columns={"canonical_smiles": "prepared_smiles"})
        )
        prepared = prepared[["prepared_smiles", "monomer_ID", "heavy_atom_count"]]
        prepared = prepared.rename(columns={"prepared_smiles": "smiles_list"})
        prepared["replicate_count"] = frame.groupby("canonical_smiles").size().reindex(
            prepared["smiles_list"]
        ).to_numpy()

    transforms: dict[str, dict[str, float]] = {}
    for target in targets:
        mean = float(prepared[target].mean())
        std = float(prepared[target].std(ddof=0))
        if not np.isfinite(std) or std == 0:
            raise ValueError(f"Target {target!r} has zero or invalid variance")
        prepared[f"{target}_raw"] = prepared[target]
        prepared[target] = (prepared[target] - mean) / std
        transforms[target] = {"mean": mean, "std": std}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "train.csv"
    prepared.to_csv(output_path, index=False)
    write_json(
        args.output_dir / "metadata.json",
        {
            "model": "GraphDITMolecularGenerator",
            "input": str(args.input),
            "targets": targets,
            "task_type": ["regression"] * len(targets),
            "aggregation": args.aggregate if targets else None,
            "max_heavy_atoms": args.max_heavy_atoms,
            "outlier_iqr": args.outlier_iqr,
            "unique_structures": int(len(prepared)),
            "target_transforms": transforms,
        },
    )
    print(f"Prepared {len(prepared)} unique polymers in {output_path}")
    print("Mode:", "conditional" if targets else "unconditional")


if __name__ == "__main__":
    main()

