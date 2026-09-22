"""Run replicate-safe five-fold cross-validation for GRIN on RPPD.

The held-out folds are identical to those used by ``evaluate.py``. Within each
fold, the remaining 80% is divided into 70% training and 10% validation data.
Target standardization is fitted on the training partition only.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_process"))
sys.path.insert(0, str(ROOT / "train_grin"))

from common import add_molecule_columns, aggregate_targets, iqr_mask, load_rppd  # noqa: E402
from properties import RPPD_PROPERTIES  # noqa: E402
from torch_molecule import GRINMolecularPredictor  # noqa: E402


DEFAULT_INPUT = ROOT / "data_process" / "output" / "rppd_clean.csv"
FALLBACK_INPUT = ROOT / "data" / "20260920_rppd.csv"
DEFAULT_OUTPUT_PARENT = ROOT / "model_evaluation" / "output"


def parse_targets(values: list[str]) -> list[str]:
    selected: list[str] = []
    for value in values:
        selected.extend(part.strip() for part in value.split(",") if part.strip())
    return list(dict.fromkeys(selected))


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(observed, predicted))
    return {
        "mae": float(mean_absolute_error(observed, predicted)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(observed, predicted)),
    }


def effective_batch_size(requested: int, rows: int) -> int:
    size = min(requested, rows)
    while size > 2 and rows % size == 1:
        size -= 1
    return size


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "train_mae",
        "train_rmse",
        "train_r2",
        "validation_mae",
        "validation_rmse",
        "validation_r2",
        "test_mae",
        "test_rmse",
        "test_r2",
        "r2_gap",
        "epochs_completed",
        "best_epoch_one_based",
    ]
    rows: list[dict[str, object]] = []
    for target, group in folds.groupby("property", sort=False):
        row: dict[str, object] = {
            "property": target,
            "model": "GRIN",
            "structures": int(group["structures"].iloc[0]),
            "folds": int(len(group)),
        }
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--property", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aggregate", choices=("mean", "median"), default="median")
    parser.add_argument("--outlier-iqr", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-layer", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--l1-penalty", type=float, default=1e-3)
    parser.add_argument("--epochs-to-penalize", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--verbose",
        choices=("none", "progress_bar", "print_statement"),
        default="none",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed folds in an existing explicit --output-root",
    )
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
    if args.resume and args.output_root is None:
        parser.error("--resume requires an explicit --output-root")
    output_root = args.output_root or (
        DEFAULT_OUTPUT_PARENT / datetime.now().strftime("grin_cv_%Y%m%d_%H%M%S")
    )
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=args.resume)

    source = add_molecule_columns(load_rppd(input_path))
    source = source.loc[source["valid_smiles"]].copy()
    device = select_device(args.device)
    metrics_path = output_root / "grin_fold_metrics.csv"
    predictions_path = output_root / "grin_out_of_fold_predictions.csv"
    datasets_path = output_root / "grin_dataset_summary.csv"
    if args.resume and metrics_path.is_file():
        fold_rows = pd.read_csv(metrics_path).to_dict("records")
    else:
        fold_rows: list[dict[str, object]] = []
    if args.resume and predictions_path.is_file():
        prediction_frames = [pd.read_csv(predictions_path)]
    else:
        prediction_frames: list[pd.DataFrame] = []
    if args.resume and datasets_path.is_file():
        dataset_rows = pd.read_csv(datasets_path).to_dict("records")
    else:
        dataset_rows: list[dict[str, object]] = []
    completed_folds = {
        (str(row["property"]), int(row["fold"])) for row in fold_rows
    }

    for property_position, target in enumerate(requested, start=1):
        aggregated = aggregate_targets(source, [target], args.aggregate)
        aggregated = aggregated.dropna(subset=[target]).copy()
        before_outliers = len(aggregated)
        aggregated = aggregated.loc[
            iqr_mask(aggregated[target], args.outlier_iqr)
        ].reset_index(drop=True)
        y = aggregated[target].to_numpy(dtype=np.float32)
        smiles = aggregated["smiles_list"].tolist()
        if target not in {str(row["property"]) for row in dataset_rows}:
            dataset_rows.append(
                {
                    "property": target,
                    "structures_before_outlier_filter": before_outliers,
                    "structures": len(aggregated),
                    "removed_outliers": before_outliers - len(aggregated),
                }
            )
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        print(
            f"[{property_position}/{len(requested)}] {target}: "
            f"{len(aggregated)} structures",
            flush=True,
        )

        for fold, (development_index, test_index) in enumerate(
            splitter.split(aggregated), start=1
        ):
            if (target, fold) in completed_folds:
                print(f"  fold {fold}/{args.folds}: already complete", flush=True)
                continue
            train_index, validation_index = train_test_split(
                development_index,
                test_size=0.125,
                random_state=args.seed + fold,
                shuffle=True,
            )
            target_mean = float(np.mean(y[train_index]))
            target_std = float(np.std(y[train_index], ddof=0))
            if not np.isfinite(target_std) or target_std <= 0:
                raise ValueError(f"Zero or invalid training standard deviation for {target}")
            y_standardized = (y - target_mean) / target_std
            fold_seed = args.seed + fold - 1
            set_seed(fold_seed)
            batch_size = effective_batch_size(args.batch_size, len(train_index))
            model = GRINMolecularPredictor(
                num_task=1,
                task_type="regression",
                repetition_augmentation=False,
                num_layer=args.num_layer,
                hidden_size=args.hidden_size,
                batch_size=batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                l1_penalty=args.l1_penalty,
                epochs_to_penalize=args.epochs_to_penalize,
                patience=args.patience,
                evaluate_criterion="mse",
                evaluate_higher_better=False,
                device=device,
                verbose=args.verbose,
            )
            model.fit(
                X_train=[smiles[index] for index in train_index],
                y_train=y_standardized[train_index],
                X_val=[smiles[index] for index in validation_index],
                y_val=y_standardized[validation_index],
            )

            split_metrics: dict[str, dict[str, float]] = {}
            for split_name, indices in (
                ("train", train_index),
                ("validation", validation_index),
                ("test", test_index),
            ):
                prediction_standardized = model.predict(
                    [smiles[index] for index in indices]
                )["prediction"].reshape(-1)
                prediction = prediction_standardized * target_std + target_mean
                observed = y[indices].astype(float)
                split_metrics[split_name] = metrics(observed, prediction)
                if split_name == "test":
                    frame = aggregated.loc[
                        indices, ["smiles_list", "monomer_ID", "replicate_count"]
                    ].copy()
                    frame.insert(0, "property", target)
                    frame.insert(1, "model", "GRIN")
                    frame.insert(2, "fold", fold)
                    frame["observed"] = observed
                    frame["predicted"] = prediction
                    frame["residual"] = prediction - observed
                    prediction_frames.append(frame)

            if args.save_checkpoints:
                checkpoint_dir = output_root / "checkpoints" / target
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                model.save_to_local(str(checkpoint_dir / f"fold_{fold}.pt"))
            fold_rows.append(
                {
                    "property": target,
                    "model": "GRIN",
                    "fold": fold,
                    "seed": fold_seed,
                    "structures": len(aggregated),
                    "train_rows": len(train_index),
                    "validation_rows": len(validation_index),
                    "test_rows": len(test_index),
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "batch_size": batch_size,
                    "epochs_completed": len(model.fitting_loss),
                    "best_epoch_one_based": int(model.fitting_epoch) + 1,
                    **{
                        f"{split_name}_{metric}": value
                        for split_name, values in split_metrics.items()
                        for metric, value in values.items()
                    },
                    "r2_gap": split_metrics["train"]["r2"]
                    - split_metrics["test"]["r2"],
                }
            )
            completed_folds.add((target, fold))
            folds_frame = pd.DataFrame(fold_rows)
            folds_frame.to_csv(metrics_path, index=False)
            pd.concat(prediction_frames, ignore_index=True).to_csv(
                predictions_path, index=False
            )
            summarize(folds_frame).to_csv(
                output_root / "grin_cv_summary.csv", index=False
            )
            pd.DataFrame(dataset_rows).to_csv(datasets_path, index=False)
            print(
                f"  fold {fold}/{args.folds}: "
                f"R²={split_metrics['test']['r2']:.4f}, "
                f"epochs={len(model.fitting_loss)}",
                flush=True,
            )
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()

    config = {
        "input": str(input_path),
        "properties": requested,
        "folds": args.folds,
        "seed": args.seed,
        "aggregation": args.aggregate,
        "outlier_iqr": args.outlier_iqr,
        "device": device,
        "validation_fraction_of_development": 0.125,
        "model_parameters": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "num_layer": args.num_layer,
            "hidden_size": args.hidden_size,
            "learning_rate": args.learning_rate,
            "l1_penalty": args.l1_penalty,
            "epochs_to_penalize": args.epochs_to_penalize,
        },
        "save_checkpoints": args.save_checkpoints,
        "resumed": args.resume,
        "note": "Standardization is fitted independently on each training fold.",
    }
    (output_root / "grin_cv_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"GRIN cross-validation complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
