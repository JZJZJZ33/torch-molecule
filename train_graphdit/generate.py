"""Generate repeat units from a trained Graph-DiT run."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import device_name, generation_metrics, read_json, sample, write_json
from torch_molecule import GraphDITMolecularGenerator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--number", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    conditions = p.add_mutually_exclusive_group()
    conditions.add_argument("--condition", action="append", default=[], help="Original units, e.g. density=1.2; repeat for joint models")
    conditions.add_argument("--conditions-csv", type=Path, help="One row per request; property columns in original units. Overrides --number.")
    conditions.add_argument("--held-out-conditions", action="store_true", help="Sample target vectors from the held-out test split")
    p.add_argument("--device", choices=["cpu", "auto", "mps", "cuda"], default="cpu")
    p.add_argument("--guide-scale", type=float)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path)
    args = p.parse_args()
    if args.number < 1 or args.batch_size < 1:
        p.error("number and batch size must be positive")
    metadata = read_json(args.model_dir / "standardization.json")
    targets = metadata["targets"]
    number = args.number
    raw = None
    if targets:
        if args.conditions_csv:
            table = pd.read_csv(args.conditions_csv)
            raw = table[targets].to_numpy(dtype=np.float32)
            number = len(raw)
        elif args.held_out_conditions:
            table = pd.read_csv(args.model_dir / "prepared_data/test.csv")
            raw = table[[f"{t}_raw" for t in targets]].sample(n=number, replace=True, random_state=args.seed).to_numpy(dtype=np.float32)
        else:
            values = {}
            for item in args.condition:
                try:
                    key, value = item.split("=", 1)
                    if key.strip() in values:
                        p.error(f"Duplicate condition {key.strip()}")
                    values[key.strip()] = float(value)
                except ValueError:
                    p.error("Conditions must be PROPERTY=NUMBER")
            if set(values) != set(targets):
                p.error(f"Supply exactly these conditions: {targets}, or choose --held-out-conditions")
            raw = np.tile([values[t] for t in targets], (number, 1)).astype(np.float32)
        if not number or not np.isfinite(raw).all():
            p.error("Conditions must contain at least one row and only finite numbers")
    elif args.condition or args.conditions_csv or args.held_out_conditions:
        p.error("An unconditional model does not accept property conditions")
    destination = args.output_dir or args.model_dir / "generation" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    if destination.exists() and any(destination.iterdir()):
        p.error("Output directory is nonempty; choose a fresh directory")
    destination.mkdir(parents=True, exist_ok=True)
    model = GraphDITMolecularGenerator(device=device_name(args.device))
    model.load_from_local(str(args.model_dir / "best_model.pt"))
    model.device = torch.device(device_name(args.device))
    model.model.to(model.device)
    model._setup_diffusion_params({"hyperparameters": {"dataset_info": model.dataset_info,
                                  "timesteps": model.timesteps, "max_node": model.max_node}})
    if args.guide_scale is not None:
        model.guide_scale = args.guide_scale
    samples = sample(model, targets, metadata["target_transforms"], raw, number, args.batch_size, args.seed)
    training = pd.read_csv(args.model_dir / "prepared_data/train.csv")
    samples["seen_in_training"] = samples.canonical_smiles.isin(set(training.smiles_list))
    samples["duplicate_in_batch"] = samples.canonical_smiles.notna() & samples.canonical_smiles.duplicated()
    samples.to_csv(destination / "generated_samples.csv", index=False)
    samples.loc[samples.polymer_valid & ~samples.duplicate_in_batch].to_csv(destination / "valid_unique_polymers.csv", index=False)
    write_json(destination / "metrics.json", generation_metrics(samples, training.smiles_list.tolist()))
    write_json(destination / "generation_config.json", {"model_dir": str(args.model_dir.resolve()),
               "targets": targets, "number": number, "batch_size": args.batch_size,
               "seed": args.seed, "device": args.device, "guide_scale": model.guide_scale,
               "attempt_policy": "One attempt per request; all failures retained"})
    print(f"Saved {number} attempts to {destination}")


if __name__ == "__main__":
    main()
