"""Shared RPPD preparation, sampling, and evaluation helpers."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_name(value):
    if value == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if value == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; choose --device cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; choose --device cpu")
    return value


def names(values):
    return list(dict.fromkeys(x.strip() for v in values for x in v.split(",") if x.strip()))


def inspect_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None
    result = {"valid": mol is not None, "polymer_valid": False,
              "canonical_smiles": None, "rejection_reason": "invalid_smiles",
              "graph_nodes": None, "heavy_atoms": None}
    if mol is None:
        return result
    result.update(canonical_smiles=Chem.MolToSmiles(mol),
                  graph_nodes=sum(a.GetAtomicNum() != 1 for a in mol.GetAtoms()),
                  heavy_atoms=mol.GetNumHeavyAtoms())
    stars = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(Chem.GetMolFrags(mol)) != 1:
        result["rejection_reason"] = "disconnected"
    elif len(stars) != 2:
        result["rejection_reason"] = "attachment_count"
    elif any(a.GetDegree() != 1 for a in stars):
        result["rejection_reason"] = "attachment_degree"
    else:
        result.update(polymer_valid=True, rejection_reason="")
    return result


def load_source(path, targets, max_heavy_atoms, dataset="rppd", smiles_column=None,
                max_structures=None, seed=42):
    if dataset == "pi1m":
        if targets:
            raise ValueError("PI1M is used for unconditional pretraining and has no RPPD targets")
        frame = pd.read_csv(path, low_memory=False)
        candidates = [smiles_column] if smiles_column else [
            "smiles_list", "SMILES", "smiles", "p_smiles", "p-SMILES"
        ]
        column = next((name for name in candidates if name and name in frame.columns), None)
        if column is None:
            raise ValueError(
                f"Cannot find a p-SMILES column in {path}; use --smiles-column. "
                f"Available columns: {list(frame.columns)}"
            )
        source_rows = len(frame)
        inspected = frame[column].map(inspect_smiles)
        prepared = pd.DataFrame({
            "smiles_list": inspected.map(lambda item: item["canonical_smiles"]),
            "heavy_atom_count": inspected.map(lambda item: item["heavy_atoms"]),
            "polymer_valid": inspected.map(lambda item: item["polymer_valid"]),
        })
        valid_unique = int(prepared.smiles_list.dropna().nunique())
        prepared = prepared.loc[
            prepared.polymer_valid
            & prepared.heavy_atom_count.notna()
            & (prepared.heavy_atom_count <= max_heavy_atoms)
        ].drop_duplicates("smiles_list").copy()
        size_unique = len(prepared)
        if max_structures and len(prepared) > max_structures:
            prepared = prepared.sample(n=max_structures, random_state=seed)
        prepared = prepared[["smiles_list", "heavy_atom_count"]]
        prepared["monomer_ID"] = None
        prepared["replicate_count"] = 1
        return prepared, {
            "source_rows": source_rows,
            "valid_unique": valid_unique,
            "within_size_limit": size_unique,
            "polymer_valid_unique": size_unique,
            "selected_structures": len(prepared),
            "complete_labels": len(prepared),
            "smiles_column": column,
        }

    # Load by path to avoid colliding with this module's own name.
    import importlib.util
    spec = importlib.util.spec_from_file_location("rppd_helpers", ROOT / "data_process/common.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    frame = helpers.add_molecule_columns(helpers.load_rppd(path))
    source_rows = len(frame)
    frame = frame.loc[frame.valid_smiles].copy()
    valid_unique = int(frame.canonical_smiles.nunique())
    frame = frame.loc[frame.heavy_atom_count <= max_heavy_atoms].copy()
    size_unique = int(frame.canonical_smiles.nunique())
    # Repeat-unit semantics are part of the dataset contract.
    keep = frame.canonical_smiles.map(lambda s: inspect_smiles(s)["polymer_valid"])
    frame = frame.loc[keep].copy()
    if frame.empty:
        raise ValueError("No valid two-attachment-point repeat units remain")
    prepared = helpers.aggregate_targets(frame, targets, "median")
    before_labels = len(prepared)
    if targets:
        prepared = prepared.dropna(subset=targets).copy()
    return prepared, {"source_rows": source_rows, "valid_unique": valid_unique,
                      "within_size_limit": size_unique, "polymer_valid_unique": before_labels,
                      "complete_labels": len(prepared)}


def prepare(path, targets, destination, args):
    data, counts = load_source(
        path, targets, args.max_heavy_atoms, dataset=args.dataset,
        smiles_column=args.smiles_column, max_structures=args.max_structures,
        seed=args.seed,
    )
    if len(data) < args.minimum_structures:
        raise ValueError(f"Only {len(data)} eligible structures; require {args.minimum_structures}")
    # Stable ordering and a seeded permutation give disjoint structure splits.
    data = data.sort_values("smiles_list").reset_index(drop=True)
    order = np.random.default_rng(args.seed).permutation(len(data))
    n_test = max(1, round(len(data) * args.test_fraction))
    n_val = max(1, round(len(data) * args.validation_fraction))
    splits = {"test": data.iloc[order[:n_test]].copy(),
              "validation": data.iloc[order[n_test:n_test + n_val]].copy(),
              "train": data.iloc[order[n_test + n_val:]].copy()}
    if len(splits["train"]) < 2:
        raise ValueError("Training split must contain at least two structures")
    train = splits["train"]
    fences = {}
    if args.outlier_iqr > 0:
        mask = pd.Series(True, index=train.index)
        for target in targets:
            q1, q3 = train[target].quantile([0.25, 0.75])
            if q3 > q1:
                lo, hi = q1 - args.outlier_iqr * (q3 - q1), q3 + args.outlier_iqr * (q3 - q1)
                fences[target] = [float(lo), float(hi)]
                mask &= train[target].between(lo, hi)
        splits["train"] = train.loc[mask].copy()
    transforms = {}
    for target in targets:
        mean = float(splits["train"][target].mean())
        std = float(splits["train"][target].std(ddof=0))
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"No finite training variance for {target}")
        transforms[target] = {"mean": mean, "std": std, "units": "original RPPD CSV units"}
        for part in splits.values():
            part[f"{target}_raw"] = part[target]
            part[target] = (part[target] - mean) / std
    destination.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(destination / f"{name}.csv", index=False)
    metadata = {"dataset": args.dataset, "targets": targets,
                "target_transforms": transforms, "counts": counts,
                "split_counts": {k: len(v) for k, v in splits.items()},
                "seed": args.seed, "aggregation": "median", "training_outlier_fences": fences,
                "max_heavy_atoms": args.max_heavy_atoms}
    write_json(destination / "metadata.json", metadata)
    write_json(destination.parent / "standardization.json", metadata)
    return splits, metadata


def sample(model, targets, transforms, raw, number, batch_size, seed):
    seed_everything(seed)
    rows = []
    for start in range(0, number, batch_size):
        n = min(batch_size, number - start)
        labels = None
        if targets:
            labels = np.array([[(v - transforms[t]["mean"]) / transforms[t]["std"]
                                for t, v in zip(targets, values)]
                               for values in raw[start:start + n]], dtype=np.float32)
        generated = model.generate(labels=labels, batch_size=n)
        if len(generated) != n:
            raise RuntimeError("Generator returned an unexpected number of samples")
        for i, smiles in enumerate(generated):
            row = {"sample_id": start + i, "generated_smiles": smiles, **inspect_smiles(smiles)}
            for j, target in enumerate(targets):
                row[f"requested_{target}"] = float(raw[start + i][j])
            rows.append(row)
    return pd.DataFrame(rows)


def generation_metrics(samples, training_smiles):
    inspected = [inspect_smiles(s) for s in samples.generated_smiles]
    valid = [r for r in inspected if r["valid"]]
    polymers = [r for r in inspected if r["polymer_valid"]]
    unique = sorted({r["canonical_smiles"] for r in polymers})
    training = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in training_smiles}
    fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    refs = [fingerprint.GetFingerprint(Chem.MolFromSmiles(s)) for s in training]
    similarities = [max(DataStructs.BulkTanimotoSimilarity(
        fingerprint.GetFingerprint(Chem.MolFromSmiles(s)), refs)) for s in unique] if refs else []
    n = len(samples)
    return {"attempts": n, "valid_count": len(valid), "polymer_valid_count": len(polymers),
            "validity": len(valid) / n if n else None,
            "polymer_validity": len(polymers) / n if n else None,
            "unique_polymer_count": len(unique),
            "uniqueness_among_valid_polymers": len(unique) / len(polymers) if polymers else None,
            "novelty_among_unique_polymers": len(set(unique) - training) / len(unique) if unique else None,
            "mean_nearest_training_tanimoto": float(np.mean(similarities)) if similarities else None,
            "mean_generated_graph_nodes": float(np.mean([r["graph_nodes"] for r in polymers])) if polymers else None,
            "mean_training_graph_nodes": float(np.mean([inspect_smiles(s)["graph_nodes"] for s in training])) if training else None}
