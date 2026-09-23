"""Fine-tune the downloaded Llamole Graph-DiT checkpoint on RPPD properties.

The pretrained directory must contain model.pt, config.yaml, and data.meta.json.
RPPD targets are standardized using the training split and assigned to the first
condition slots of the pretrained model; unused condition slots remain masked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import names, prepare, write_json  # noqa: E402
from llamole_graphdit import GraphDiT  # noqa: E402
from train_grin.properties import RPPD_PROPERTIES  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


def resolve_dtype(precision: str, device: torch.device) -> torch.dtype:
    if precision == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if precision == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 requires a CUDA GPU with BF16 support")
        return torch.bfloat16
    if precision == "fp16":
        if device.type != "cuda":
            raise ValueError("FP16 training is supported only on CUDA")
        return torch.float16
    return torch.float32


def validate_bundle(directory: Path) -> tuple[Path, Path, Path]:
    files = tuple(directory / name for name in ("model.pt", "config.yaml", "data.meta.json"))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing pretrained files: " + ", ".join(missing))
    return files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bond_type(bond: Chem.Bond) -> int:
    if bond.GetIsAromatic():
        return 4
    value = bond.GetBondType()
    mapping = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported bond type: {value}")
    return mapping[value]


def frame_to_graphs(
    frame: pd.DataFrame,
    targets: list[str],
    active_atoms: list[str],
    max_nodes: int,
    condition_slots: int,
) -> tuple[list[Data], int]:
    atom_index = {symbol: index for index, symbol in enumerate(active_atoms)}
    graphs: list[Data] = []
    excluded = 0
    for row in frame.itertuples(index=False):
        molecule = Chem.MolFromSmiles(row.smiles_list)
        if molecule is None or molecule.GetNumAtoms() > max_nodes:
            excluded += 1
            continue
        try:
            node_types = [atom_index[atom.GetSymbol()] for atom in molecule.GetAtoms()]
        except KeyError:
            excluded += 1
            continue
        edges: list[tuple[int, int]] = []
        edge_types: list[int] = []
        try:
            for bond in molecule.GetBonds():
                begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                kind = bond_type(bond)
                edges.extend(((begin, end), (end, begin)))
                edge_types.extend((kind, kind))
        except ValueError:
            excluded += 1
            continue
        edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges else torch.empty((2, 0), dtype=torch.long)
        )
        properties = torch.full((1, condition_slots), float("nan"), dtype=torch.float32)
        for slot, target in enumerate(targets):
            properties[0, slot] = float(getattr(row, target))
        graphs.append(
            Data(
                x=torch.tensor(node_types, dtype=torch.long),
                edge_index=edge_index,
                edge_attr=torch.tensor(edge_types, dtype=torch.long),
                y=properties,
            )
        )
    return graphs, excluded


def make_loader(model: GraphDiT, frame: pd.DataFrame, targets: list[str], args, shuffle: bool):
    graphs, excluded = frame_to_graphs(
        frame, targets, model.data_info.active_atoms, model.max_n_nodes, model.ydim
    )
    if not graphs:
        raise ValueError("No RPPD structures fit the pretrained atom vocabulary and 50-node limit")
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        graphs,
        batch_size=args.batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=args.num_workers,
        pin_memory=args.device_resolved.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    return loader, excluded


def configure_trainable_parameters(model: GraphDiT, target_count: int, last_layers: int) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Each selected RPPD target uses one of the existing ten scalar-condition MLPs.
    for slot in range(target_count):
        for parameter in model.denoiser.y_embedder.mlps[slot].parameters():
            parameter.requires_grad = True

    if last_layers:
        for block in model.denoiser.blocks[-last_layers:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in model.denoiser.output_layer.parameters():
            parameter.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable_parameters": trainable, "total_parameters": total,
            "trainable_fraction": trainable / total}


def move_batch(batch, device: torch.device):
    return batch.to(device, non_blocking=device.type == "cuda")


def batch_loss(model: GraphDiT, batch: Data) -> torch.Tensor:
    text = torch.full(
        (batch.num_graphs, model.text_input_size),
        float("nan"),
        dtype=model.model_dtype,
        device=batch.x.device,
    )
    return model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.y,
        text,
        no_label_index=-999.0,
    )


def train_epoch(model, loader, optimizer, scaler, args) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    count = 0
    autocast = args.device_resolved.type == "cuda" and args.dtype_resolved != torch.float32
    for step, batch in enumerate(loader):
        batch = move_batch(batch, args.device_resolved)
        with torch.autocast("cuda", dtype=args.dtype_resolved, enabled=autocast):
            loss = batch_loss(model, batch) / args.gradient_accumulation
        scaler.scale(loss).backward()
        should_step = (step + 1) % args.gradient_accumulation == 0 or step + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.gradient_clip
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total += loss.detach().float().item() * args.gradient_accumulation * batch.num_graphs
        count += batch.num_graphs
    return total / count


@torch.no_grad()
def evaluate(model, loader, args, seed: int) -> float:
    model.eval()
    torch.manual_seed(seed)
    total = 0.0
    count = 0
    autocast = args.device_resolved.type == "cuda" and args.dtype_resolved != torch.float32
    for batch in loader:
        batch = move_batch(batch, args.device_resolved)
        with torch.autocast("cuda", dtype=args.dtype_resolved, enabled=autocast):
            loss = batch_loss(model, batch)
        total += loss.float().item() * batch.num_graphs
        count += batch.num_graphs
    return total / count


def save_weights(model: GraphDiT, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(model.denoiser.state_dict(), temporary)
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrained-dir",
        default=str(HERE / "pretrained/llamole_pretrained_graphdit"),
        help="Directory containing model.pt, config.yaml and data.meta.json",
    )
    parser.add_argument("--input", default=str(ROOT / "data/20260920_rppd.csv"))
    parser.add_argument("--property", action="append", required=True,
                        help="RPPD property; repeat or use a comma-separated list")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--last-layers", type=int, default=2,
                        help="Also tune this many final transformer blocks; use 0 for condition-only")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-heavy-atoms", type=int, default=50)
    parser.add_argument("--minimum-structures", type=int, default=50)
    parser.add_argument("--max-structures", type=int)
    parser.add_argument("--outlier-iqr", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--check-only", action="store_true",
                        help="Validate files, properties and settings without loading/training the model")
    args = parser.parse_args()
    args.dataset = "rppd"
    args.smiles_column = None
    args.targets = names(args.property)
    unknown = sorted(set(args.targets) - set(RPPD_PROPERTIES))
    if unknown:
        parser.error(f"Unknown RPPD properties: {unknown}")
    if len(args.targets) > 10:
        parser.error("The downloaded model supports at most 10 simultaneous conditions")
    if min(args.epochs, args.patience, args.batch_size, args.gradient_accumulation) < 1:
        parser.error("Epoch, patience and batch settings must be positive")
    if not 0 <= args.last_layers <= 28:
        parser.error("--last-layers must be between 0 and 28")
    args.device_resolved = resolve_device(args.device)
    args.dtype_resolved = resolve_dtype(args.precision, args.device_resolved)
    return args


def main() -> None:
    args = parse_args()
    pretrained_dir = Path(args.pretrained_dir).resolve()
    model_path, config_path, metadata_path = validate_bundle(pretrained_dir)
    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"RPPD CSV not found: {input_path}")
    columns = set(pd.read_csv(input_path, nrows=0).columns)
    missing_targets = sorted(set(args.targets) - columns)
    if missing_targets:
        raise ValueError(f"Properties absent from RPPD CSV: {missing_targets}")

    if args.check_only:
        print("Pretrained bundle: OK")
        print(f"RPPD input: {input_path}")
        print(f"Joint conditions: {', '.join(args.targets)}")
        print(f"Training device: {args.device_resolved}; precision: {args.dtype_resolved}")
        return

    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    splits, preparation = prepare(input_path, args.targets, output / "prepared_data", args)
    model = GraphDiT(config_path, metadata_path, args.dtype_resolved)
    state = torch.load(model_path, map_location="cpu", weights_only=True, mmap=True)
    model.denoiser.load_state_dict(state, strict=True)
    del state
    model.to(device=args.device_resolved, dtype=args.dtype_resolved)
    model.denoiser.gradient_checkpointing = True
    parameter_report = configure_trainable_parameters(model, len(args.targets), args.last_layers)

    train_loader, train_excluded = make_loader(model, splits["train"], args.targets, args, True)
    validation_loader, validation_excluded = make_loader(
        model, splits["validation"], args.targets, args, False
    )
    test_loader, test_excluded = make_loader(model, splits["test"], args.targets, args, False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(2, args.patience // 4)
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.device_resolved.type == "cuda" and args.dtype_resolved == torch.float16
    )

    public_args = {
        key: str(value) if isinstance(value, (Path, torch.device, torch.dtype)) else value
        for key, value in vars(args).items()
        if key not in {"property"}
    }
    run_config = {
        "arguments": public_args,
        "pretrained_model_sha256": sha256_file(model_path),
        "condition_slots": {target: slot for slot, target in enumerate(args.targets)},
        **parameter_report,
        "preparation": preparation,
        "excluded_graphs": {
            "train": train_excluded,
            "validation": validation_excluded,
            "test": test_excluded,
        },
    }
    write_json(output / "training_config.json", run_config)
    shutil.copy2(config_path, output / "config.yaml")
    shutil.copy2(metadata_path, output / "data.meta.json")

    best = math.inf
    stale = 0
    history: list[dict] = []
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, args)
        validation_loss = evaluate(model, validation_loader, args, args.seed + 1000)
        scheduler.step(validation_loss)
        improved = validation_loss < best
        if improved:
            best = validation_loss
            stale = 0
            save_weights(model, output / "model.pt")
        else:
            stale += 1
        entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(entry)
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        torch.save(
            {
                "epoch": epoch,
                "best": best,
                "stale": stale,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            output / "last_training_state.pt",
        )
        print(
            f"epoch {epoch + 1}: train={train_loss:.4f}, "
            f"validation={validation_loss:.4f}",
            flush=True,
        )
        if stale >= args.patience:
            break

    best_state = torch.load(output / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    model.denoiser.load_state_dict(best_state, strict=True)
    test_loss = evaluate(model, test_loader, args, args.seed + 2000)
    write_json(
        output / "metrics.json",
        {
            "best_validation_loss": best,
            "test_loss": test_loss,
            "epochs_completed": len(history),
            "properties": args.targets,
        },
    )
    print(f"Finished. Fine-tuned model: {output / 'model.pt'}")


if __name__ == "__main__":
    main()
