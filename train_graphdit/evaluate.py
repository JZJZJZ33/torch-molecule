"""Evaluate generated repeat units, optionally with existing GRIN predictors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import generation_metrics, inspect_smiles, read_json, write_json


def target_transform(path, target):
    """Accept common saved GRIN scaling layouts; never guess missing values."""
    data = read_json(path)
    candidates = [data]
    for key in ("target_transforms", "standardization", "target_standardization", "target_transform"):
        if isinstance(data.get(key), dict):
            candidates.append(data[key])
    for entry in list(candidates):
        if isinstance(entry.get(target), dict):
            candidates.append(entry[target])
    for entry in candidates:
        for mean_key, std_key in (("mean", "std"), ("target_mean", "target_std"), ("training_mean", "training_std")):
            if mean_key in entry and std_key in entry:
                mean, std = float(entry[mean_key]), float(entry[std_key])
                if np.isfinite([mean, std]).all() and std > 0:
                    return mean, std
    raise ValueError(f"Cannot read target mean/std from {path}; provide a JSON with mean and std")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--grin-model", action="append", default=[], help="PROPERTY=/path/to/grin/property-directory")
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error("Output directory must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(args.samples)
    training = pd.read_csv(args.model_dir / "prepared_data/train.csv")
    flags = pd.DataFrame([inspect_smiles(s) for s in samples.generated_smiles])
    for col in flags:
        samples[col] = flags[col]
    metrics = generation_metrics(samples, training.smiles_list.tolist())
    metrics["conditioning"] = {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for specification in args.grin_model:
        target, directory = specification.split("=", 1)
        directory = Path(directory)
        column = f"requested_{target}"
        if column not in samples:
            raise ValueError(f"No requested values for {target} in the sample file")
        from torch_molecule import GRINMolecularPredictor
        mean, std = target_transform(directory / "standardization.json", target)
        predictor = GRINMolecularPredictor(device="cpu")
        predictor.load_from_local(str(directory / "grin_model.pt"))
        mask = samples.polymer_valid & pd.to_numeric(samples[column], errors="coerce").notna()
        result = {"interpretation": "GRIN-estimated target agreement, not measured properties", "count": int(mask.sum())}
        if mask.any():
            prediction = np.asarray(predictor.predict(samples.loc[mask, "generated_smiles"].tolist())).reshape(-1)
            prediction = prediction * std + mean
            if not np.isfinite(prediction).all():
                raise ValueError(f"GRIN produced nonfinite predictions for {target}")
            samples.loc[mask, f"grin_predicted_{target}"] = prediction
            requested = samples.loc[mask, column].to_numpy(dtype=float)
            residual = prediction - requested
            result.update(mae=float(np.mean(np.abs(residual))), rmse=float(np.sqrt(np.mean(residual ** 2))))
            fig, ax = plt.subplots()
            ax.scatter(requested, prediction, alpha=0.6)
            lo, hi = float(min(requested.min(), prediction.min())), float(max(requested.max(), prediction.max()))
            ax.plot([lo, hi], [lo, hi], "k--")
            ax.set(xlabel=f"Requested {target} (RPPD units)", ylabel=f"GRIN-estimated {target} (RPPD units)")
            fig.tight_layout()
            safe_target = "".join(c if c.isalnum() or c in "-_." else "_" for c in target)
            fig.savefig(args.output_dir / f"conditioning_{safe_target}.png", dpi=160)
            plt.close(fig)
        metrics["conditioning"][target] = result
    sizes = [inspect_smiles(s)["graph_nodes"] for s in training.smiles_list]
    generated_sizes = samples.loc[samples.polymer_valid, "graph_nodes"].dropna()
    fig, ax = plt.subplots()
    ax.hist(sizes, bins=20, density=True, alpha=0.5, label="Training")
    if len(generated_sizes):
        ax.hist(generated_sizes, bins=20, density=True, alpha=0.5, label="Generated polymers")
    ax.set(xlabel="Graph nodes, including attachment points", ylabel="Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "size_distribution.png", dpi=160)
    plt.close(fig)
    samples.to_csv(args.output_dir / "evaluated_samples.csv", index=False)
    write_json(args.output_dir / "metrics.json", metrics)
    print(f"Evaluation saved to {args.output_dir}")


if __name__ == "__main__":
    main()
