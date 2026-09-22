"""Aggregate RPPD replicates and create leakage-safe GRIN data splits.

Example:
    python data_process/prepare_grin.py data/20260920_rppd.csv --target density
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from common import (
    add_molecule_columns,
    aggregate_targets,
    iqr_mask,
    load_rppd,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--target", required=True, help="Numeric property to predict")
    parser.add_argument("--aggregate", choices=("mean", "median"), default="median")
    parser.add_argument(
        "--outlier-iqr",
        type=float,
        default=0.0,
        help="Remove target outliers outside this many IQRs; 0 disables removal",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--validation-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data_process/output/grin")
    )
    args = parser.parse_args()

    if args.test_size <= 0 or args.validation_size <= 0:
        raise ValueError("test-size and validation-size must be positive")
    if args.test_size + args.validation_size >= 1:
        raise ValueError("test-size + validation-size must be less than 1")

    frame = add_molecule_columns(load_rppd(args.input))
    frame = frame.loc[frame["valid_smiles"]].copy()
    aggregated = aggregate_targets(frame, [args.target], args.aggregate)
    aggregated = aggregated.dropna(subset=[args.target]).copy()
    before_outliers = len(aggregated)
    aggregated = aggregated.loc[iqr_mask(aggregated[args.target], args.outlier_iqr)].copy()

    train_val, test = train_test_split(
        aggregated, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    relative_validation = args.validation_size / (1.0 - args.test_size)
    train, validation = train_test_split(
        train_val,
        test_size=relative_validation,
        random_state=args.seed,
        shuffle=True,
    )

    target_mean = float(train[args.target].mean())
    target_std = float(train[args.target].std(ddof=0))
    if not np.isfinite(target_std) or target_std == 0:
        raise ValueError(f"Target {args.target!r} has zero or invalid training variance")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        output = split.copy()
        output[f"{args.target}_raw"] = output[args.target]
        output[args.target] = (output[args.target] - target_mean) / target_std
        output.to_csv(args.output_dir / f"{name}.csv", index=False)

    write_json(
        args.output_dir / "metadata.json",
        {
            "model": "GRINMolecularPredictor",
            "input": str(args.input),
            "target": args.target,
            "aggregation": args.aggregate,
            "outlier_iqr": args.outlier_iqr,
            "rows_before_outlier_filter": before_outliers,
            "unique_structures": int(len(aggregated)),
            "split_rows": {
                "train": int(len(train)),
                "validation": int(len(validation)),
                "test": int(len(test)),
            },
            "target_transform": {"mean": target_mean, "std": target_std},
            "seed": args.seed,
        },
    )
    print(f"Prepared {len(aggregated)} unique polymers in {args.output_dir}")
    print(f"Split sizes: train={len(train)}, validation={len(validation)}, test={len(test)}")


if __name__ == "__main__":
    main()
