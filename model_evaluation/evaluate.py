"""Evaluate fingerprint baselines for curated RPPD polymer properties.

This workflow is intentionally independent of GRIN training and deployment. It
aggregates replicate simulations by canonical pSMILES, evaluates several models
with five-fold cross-validation, and reports every metric in original units.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

# Joblib reads this setting while scikit-learn is imported. Some macOS systems
# return an empty physical-core count, so use the logical count as a fallback.
if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_process"))
sys.path.insert(0, str(ROOT / "train_grin"))

from common import add_molecule_columns, aggregate_targets, iqr_mask, load_rppd  # noqa: E402
from properties import RPPD_PROPERTIES  # noqa: E402


DEFAULT_INPUT = ROOT / "data_process" / "output" / "rppd_clean.csv"
FALLBACK_INPUT = ROOT / "data" / "20260920_rppd.csv"
DEFAULT_OUTPUT_PARENT = ROOT / "model_evaluation" / "output"
DEFAULT_GRIN_RUN = ROOT / "train_grin" / "output_standardized" / "run_20260921_194205"

LOG_TARGETS = {
    "self-diffusion",
    "compressibility",
    "isentropic_compressibility",
    "bulk_modulus",
    "isentropic_bulk_modulus",
    "volume_expansion",
    "linear_expansion",
    "thermal_conductivity",
    "thermal_diffusivity",
}


def parse_targets(values: list[str]) -> list[str]:
    selected: list[str] = []
    for value in values:
        selected.extend(part.strip() for part in value.split(",") if part.strip())
    return list(dict.fromkeys(selected))


def fingerprint_matrix(smiles: list[str], radius: int, bits: int) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    matrix = np.zeros((len(smiles), bits), dtype=np.float32)
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Invalid canonical pSMILES during fingerprinting: {value!r}")
        fingerprint = generator.GetFingerprint(molecule)
        DataStructs.ConvertToNumpyArray(fingerprint, matrix[index])
    return matrix


def model_factories(seed: int, trees: int) -> dict[str, Callable[[], RegressorMixin]]:
    return {
        "mean": lambda: DummyRegressor(strategy="mean"),
        "ridge": lambda: Ridge(alpha=1.0),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=trees,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        ),
        "extra_trees": lambda: ExtraTreesRegressor(
            n_estimators=trees,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=200,
            l2_regularization=0.1,
            random_state=seed,
        ),
    }


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(observed, predicted)
    return {
        "mae": float(mean_absolute_error(observed, predicted)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(observed, predicted)),
    }


def transformed(values: np.ndarray, transform: str) -> np.ndarray:
    return np.log10(values) if transform == "log10" else values


def inverse_transformed(values: np.ndarray, transform: str) -> np.ndarray:
    return np.power(10.0, values) if transform == "log10" else values


def summarize_folds(folds: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "train_mae",
        "train_rmse",
        "train_r2",
        "test_mae",
        "test_rmse",
        "test_r2",
        "r2_gap",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in folds.groupby(["property", "transform", "model"], sort=False):
        row: dict[str, object] = {
            "property": keys[0],
            "transform": keys[1],
            "model": keys[2],
            "structures": int(group["structures"].iloc[0]),
            "folds": int(len(group)),
        }
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["property", "test_r2_mean"], ascending=[True, False]
    )


def write_best_plot(summary: pd.DataFrame, path: Path) -> None:
    candidates = summary.loc[summary["model"] != "mean"].copy()
    best = candidates.loc[candidates.groupby("property")["test_r2_mean"].idxmax()]
    best = best.sort_values("test_r2_mean")
    labels = [
        f"{row.property} · {row.model} ({row.transform})"
        for row in best.itertuples(index=False)
    ]
    figure_height = max(6.0, 0.42 * len(best))
    fig, ax = plt.subplots(figsize=(11, figure_height))
    ax.barh(labels, best["test_r2_mean"], xerr=best["test_r2_std"], color="#2468A2")
    ax.axvline(0, color="#52606D", linewidth=0.8)
    ax.set_xlabel("Five-fold test R² (mean ± standard deviation)")
    ax.set_title("Best fingerprint baseline by RPPD property")
    ax.grid(axis="x", color="#D9E2EC", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_grin_reference(grin_run: Path, output_path: Path) -> None:
    rows: list[dict[str, object]] = []
    if grin_run.is_dir():
        for metrics_path in sorted(grin_run.glob("*/metrics.json")):
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            test = payload["metrics"]["test"]
            rows.append(
                {
                    "property": payload["target"],
                    "model": "GRIN",
                    "evaluation": "single saved train/validation/test split; not cross-validation",
                    "test_mae": test["mae"],
                    "test_mse": test["mse"],
                    "test_rmse": test["rmse"],
                    "test_r2": test["r2"],
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--property", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aggregate", choices=("mean", "median"), default="median")
    parser.add_argument("--outlier-iqr", type=float, default=3.0)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--fingerprint-bits", type=int, default=1024)
    parser.add_argument("--trees", type=int, default=150)
    parser.add_argument("--grin-run", type=Path, default=DEFAULT_GRIN_RUN)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    if args.folds < 3:
        raise ValueError("At least three folds are required")
    requested = list(RPPD_PROPERTIES) if args.all else parse_targets(args.property)
    if not requested:
        parser.error("choose --all or at least one --property")
    unknown = sorted(set(requested) - set(RPPD_PROPERTIES))
    if unknown:
        raise ValueError(f"Unknown or non-curated properties: {', '.join(unknown)}")

    input_path = args.input
    if input_path is None:
        input_path = DEFAULT_INPUT if DEFAULT_INPUT.is_file() else FALLBACK_INPUT
    input_path = input_path.resolve()
    output_root = args.output_root or (
        DEFAULT_OUTPUT_PARENT / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    )
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    source = add_molecule_columns(load_rppd(input_path))
    source = source.loc[source["valid_smiles"]].copy()
    factories = model_factories(args.seed, args.trees)
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    dataset_rows: list[dict[str, object]] = []

    for position, target in enumerate(requested, start=1):
        aggregated = aggregate_targets(source, [target], args.aggregate)
        aggregated = aggregated.dropna(subset=[target]).copy()
        before_outliers = len(aggregated)
        aggregated = aggregated.loc[iqr_mask(aggregated[target], args.outlier_iqr)].reset_index(drop=True)
        after_outliers = len(aggregated)
        removed_nonpositive = 0
        if target in LOG_TARGETS:
            positive = aggregated[target] > 0
            removed_nonpositive = int((~positive).sum())
            aggregated = aggregated.loc[positive].reset_index(drop=True)
        X = fingerprint_matrix(
            aggregated["smiles_list"].tolist(), args.fingerprint_radius, args.fingerprint_bits
        )
        y = aggregated[target].to_numpy(dtype=float)
        transforms = ["raw"]
        if target in LOG_TARGETS:
            transforms.append("log10")
        dataset_rows.append(
            {
                "property": target,
                "structures_before_outlier_filter": before_outliers,
                "structures": len(aggregated),
                "removed_outliers": before_outliers - after_outliers,
                "removed_nonpositive": removed_nonpositive,
                "target_min": float(np.min(y)),
                "target_median": float(np.median(y)),
                "target_max": float(np.max(y)),
                "transforms": ",".join(transforms),
            }
        )
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        print(f"[{position}/{len(requested)}] {target}: {len(aggregated)} structures")

        for transform in transforms:
            y_fit = transformed(y, transform)
            for model_name, factory in factories.items():
                for fold, (train_index, test_index) in enumerate(splitter.split(X), start=1):
                    model = factory()
                    model.fit(X[train_index], y_fit[train_index])
                    train_prediction = inverse_transformed(model.predict(X[train_index]), transform)
                    test_prediction = inverse_transformed(model.predict(X[test_index]), transform)
                    train_metrics = metrics(y[train_index], train_prediction)
                    test_metrics = metrics(y[test_index], test_prediction)
                    fold_rows.append(
                        {
                            "property": target,
                            "transform": transform,
                            "model": model_name,
                            "fold": fold,
                            "structures": len(aggregated),
                            "train_rows": len(train_index),
                            "test_rows": len(test_index),
                            **{f"train_{key}": value for key, value in train_metrics.items()},
                            **{f"test_{key}": value for key, value in test_metrics.items()},
                            "r2_gap": train_metrics["r2"] - test_metrics["r2"],
                        }
                    )
                    predictions = aggregated.loc[
                        test_index, ["smiles_list", "monomer_ID", "replicate_count"]
                    ].copy()
                    predictions.insert(0, "property", target)
                    predictions.insert(1, "transform", transform)
                    predictions.insert(2, "model", model_name)
                    predictions.insert(3, "fold", fold)
                    predictions["observed"] = y[test_index]
                    predictions["predicted"] = test_prediction
                    predictions["residual"] = test_prediction - y[test_index]
                    prediction_frames.append(predictions)

    folds = pd.DataFrame(fold_rows)
    summary = summarize_folds(folds)
    datasets = pd.DataFrame(dataset_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    folds.to_csv(output_root / "fold_metrics.csv", index=False)
    summary.to_csv(output_root / "summary.csv", index=False)
    datasets.to_csv(output_root / "dataset_summary.csv", index=False)
    predictions.to_csv(output_root / "out_of_fold_predictions.csv", index=False)
    write_best_plot(summary, output_root / "best_model_r2.png")
    write_grin_reference(args.grin_run.resolve(), output_root / "grin_single_split_reference.csv")
    best = summary.loc[summary["model"] != "mean"].copy()
    best = best.loc[best.groupby("property")["test_r2_mean"].idxmax()]
    best.to_csv(output_root / "best_models.csv", index=False)
    config = {
        "input": str(input_path),
        "properties": requested,
        "folds": args.folds,
        "seed": args.seed,
        "aggregation": args.aggregate,
        "outlier_iqr": args.outlier_iqr,
        "fingerprint": {
            "type": "Morgan",
            "radius": args.fingerprint_radius,
            "bits": args.fingerprint_bits,
        },
        "tree_estimators": args.trees,
        "models": list(factories),
        "log10_targets": sorted(LOG_TARGETS),
        "grin_reference": str(args.grin_run.resolve()),
        "note": "GRIN reference uses one saved split and is not directly equivalent to CV means.",
    }
    (output_root / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Evaluation complete: {output_root}")


if __name__ == "__main__":
    main()
