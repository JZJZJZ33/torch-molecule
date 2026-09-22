"""Train GRIN from prepared splits and produce an evaluation report.

Example:
    python data_process/train_grin.py \
      --data-dir data_process/output/grin \
      --output-dir results/grin_density
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from torch_molecule import GRINMolecularPredictor


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


def load_split(data_dir: Path, name: str, target: str) -> pd.DataFrame:
    path = data_dir / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing prepared split: {path}")
    frame = pd.read_csv(path)
    required = {"smiles_list", target, f"{target}_raw"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def calculate_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(observed, predicted))
    return {
        "mae": float(mean_absolute_error(observed, predicted)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(observed, predicted)),
    }


def save_plots(
    output_dir: Path,
    target: str,
    predictions: dict[str, pd.DataFrame],
    losses: list[float],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plots. Install it with: "
            "python -m pip install matplotlib"
        ) from exc

    colors = {"train": "#4C78A8", "validation": "#F58518", "test": "#54A24B"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    all_values: list[np.ndarray] = []
    for frame in predictions.values():
        all_values.extend([frame["observed"].to_numpy(), frame["predicted"].to_numpy()])
    low = min(float(np.min(values)) for values in all_values)
    high = max(float(np.max(values)) for values in all_values)
    padding = max((high - low) * 0.05, 1e-8)
    limits = (low - padding, high + padding)

    for axis, (split, frame) in zip(axes, predictions.items()):
        split_metrics = calculate_metrics(
            frame["observed"].to_numpy(), frame["predicted"].to_numpy()
        )
        axis.scatter(
            frame["observed"],
            frame["predicted"],
            s=28,
            alpha=0.75,
            color=colors[split],
            edgecolor="white",
            linewidth=0.35,
        )
        axis.plot(limits, limits, "--", color="black", linewidth=1)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(split.capitalize())
        axis.set_xlabel(f"Observed {target}")
        axis.set_ylabel(f"Predicted {target}")
        axis.text(
            0.04,
            0.96,
            f"n = {len(frame)}\nMAE = {split_metrics['mae']:.4g}\n"
            f"RMSE = {split_metrics['rmse']:.4g}\nR² = {split_metrics['r2']:.4f}",
            transform=axis.transAxes,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
    fig.suptitle(f"GRIN predictions for {target}", fontsize=14)
    fig.savefig(output_dir / "prediction_scatter.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.plot(np.arange(1, len(losses) + 1), losses, color="#4C78A8")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training loss")
    axis.set_title("GRIN training loss")
    axis.grid(alpha=0.25)
    fig.savefig(output_dir / "training_loss.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data_process/output/grin")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/grin_density")
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-layer", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--l1-penalty", type=float, default=1e-3)
    parser.add_argument("--epochs-to-penalize", type=int, default=100)
    parser.add_argument(
        "--repetition-augmentation",
        action="store_true",
        help="Enable polymer repetition augmentation (requires CombineMols)",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verbose",
        choices=("none", "progress_bar", "print_statement"),
        default="progress_bar",
    )
    args = parser.parse_args()

    missing_dependencies = [
        package
        for package in ("torch_scatter", "matplotlib")
        if importlib.util.find_spec(package) is None
    ]
    if missing_dependencies:
        raise ImportError(
            "Missing training/report dependencies: "
            + ", ".join(missing_dependencies)
            + ". Install them in the active torch-molecule environment first."
        )

    metadata_path = args.data_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing preparation metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    target = metadata["target"]
    transform = metadata["target_transform"]
    target_mean = float(transform["mean"])
    target_std = float(transform["std"])

    splits = {
        name: load_split(args.data_dir, name, target)
        for name in ("train", "validation", "test")
    }
    device = select_device(args.device)
    set_seed(args.seed)

    print(f"Training GRIN for target: {target}")
    print(f"Device: {device}")
    print("Rows:", ", ".join(f"{name}={len(frame)}" for name, frame in splits.items()))

    effective_batch_size = min(args.batch_size, len(splits["train"]))
    while effective_batch_size > 2 and len(splits["train"]) % effective_batch_size == 1:
        effective_batch_size -= 1
    if effective_batch_size != args.batch_size:
        print(
            f"Batch size adjusted from {args.batch_size} to {effective_batch_size} "
            "to avoid a one-sample BatchNorm batch"
        )

    model = GRINMolecularPredictor(
        num_task=1,
        task_type="regression",
        repetition_augmentation=args.repetition_augmentation,
        num_layer=args.num_layer,
        hidden_size=args.hidden_size,
        batch_size=effective_batch_size,
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
        X_train=splits["train"]["smiles_list"].tolist(),
        y_train=splits["train"][target].to_numpy(dtype=np.float32),
        X_val=splits["validation"]["smiles_list"].tolist(),
        y_val=splits["validation"][target].to_numpy(dtype=np.float32),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "grin_model.pt"
    model.save_to_local(str(checkpoint_path))

    reports: dict[str, dict[str, float]] = {}
    prediction_frames: dict[str, pd.DataFrame] = {}
    for name, frame in splits.items():
        standardized_prediction = model.predict(frame["smiles_list"].tolist())[
            "prediction"
        ].reshape(-1)
        observed_standardized = frame[target].to_numpy(dtype=float)
        # Explicit inverse transform back to the original RPPD property units.
        predicted = standardized_prediction * target_std + target_mean
        observed = frame[f"{target}_raw"].to_numpy(dtype=float)
        result = frame[["smiles_list", "monomer_ID", "replicate_count"]].copy()
        result["observed_standardized"] = observed_standardized
        result["predicted_standardized"] = standardized_prediction
        result["observed"] = observed
        result["predicted"] = predicted
        result["residual"] = predicted - observed
        result.to_csv(args.output_dir / f"predictions_{name}.csv", index=False)
        reports[name] = calculate_metrics(observed, predicted)
        prediction_frames[name] = result

    report = {
        "target": target,
        "target_units": "original RPPD units",
        "device": device,
        "seed": args.seed,
        "best_epoch_one_based": int(model.fitting_epoch) + 1,
        "epochs_completed": len(model.fitting_loss),
        "metrics": reports,
        "model_parameters": {
            "num_layer": args.num_layer,
            "hidden_size": args.hidden_size,
            "batch_size": effective_batch_size,
            "requested_batch_size": args.batch_size,
            "maximum_epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "l1_penalty": args.l1_penalty,
            "epochs_to_penalize": args.epochs_to_penalize,
            "repetition_augmentation": args.repetition_augmentation,
        },
        "target_transform": {"mean": target_mean, "std": target_std},
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "standardization.json").write_text(
        json.dumps(
            {
                "target": target,
                "fitted_on": "training split only",
                "forward_transform": "standardized = (original - mean) / std",
                "inverse_transform": "original = standardized * std + mean",
                "mean": target_mean,
                "std": target_std,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {"epoch": np.arange(1, len(model.fitting_loss) + 1), "training_loss": model.fitting_loss}
    ).to_csv(args.output_dir / "training_history.csv", index=False)
    save_plots(args.output_dir, target, prediction_frames, model.fitting_loss)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metrics: {args.output_dir / 'metrics.json'}")
    print(f"Scatter plot: {args.output_dir / 'prediction_scatter.png'}")
    print("Test metrics:", json.dumps(reports["test"], indent=2))


if __name__ == "__main__":
    main()
