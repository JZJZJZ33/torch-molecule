"""Pretrain Graph-DiT unconditionally or fine-tune it on RPPD properties."""

from __future__ import annotations

import argparse
import hashlib
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from common import ROOT, device_name, names, prepare, read_json, seed_everything, write_json
from torch_molecule import GraphDITMolecularGenerator
from torch_molecule.generator.graph_dit.utils import to_dense
from train_grin.properties import RPPD_PROPERTIES


PRESETS = {
    "mac": {
        "epochs": 300, "patience": 40, "batch_size": 8, "num_layer": 3,
        "hidden_size": 128, "num_head": 8, "timesteps": 500,
        "learning_rate": 2e-4, "num_workers": 0, "precision": "fp32",
        "device": "cpu",
    },
    "h20": {
        "epochs": 100, "patience": 15, "batch_size": 128, "num_layer": 6,
        "hidden_size": 512, "num_head": 8, "timesteps": 500,
        "learning_rate": 2e-4, "num_workers": 8, "precision": "bf16",
        "device": "cuda",
    },
}


@torch.no_grad()
def validation_loss(model, loader, seed, precision="fp32"):
    # Use fixed diffusion noise on validation without changing training RNG.
    state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps_state = torch.mps.get_rng_state() if model.device.type == "mps" else None
    torch.manual_seed(seed)
    model.model.eval()
    totals = np.zeros(3)
    count = 0
    try:
        for batch in loader:
            batch = batch.to(model.device)
            autocast_enabled = model.device.type == "cuda" and precision in {"bf16", "fp16"}
            dtype = torch.bfloat16 if precision == "bf16" else torch.float16
            with torch.autocast(device_type=model.device.type, dtype=dtype, enabled=autocast_enabled):
                x = F.one_hot(batch.x, num_classes=118).float()[:, model.dataset_info["active_index"]]
                e = F.one_hot(batch.edge_attr, num_classes=5).float()
                dense, mask = to_dense(x, batch.edge_index, e, batch.batch, model.max_node)
                dense = dense.mask(mask)
                noisy = model.apply_noise(dense.X, dense.E, batch.y, mask)
                loss = model.model.compute_loss(noisy, true_X=dense.X, true_E=dense.E,
                                                lw_X=model.lw_X, lw_E=model.lw_E)
            totals += np.array([v.item() for v in loss]) * batch.num_graphs
            count += batch.num_graphs
    finally:
        torch.set_rng_state(state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
    if not count:
        raise ValueError("Validation has no evaluable structures")
    return (totals / count).tolist()


def make_loader(model, frame, targets, shuffle, num_workers=0):
    y = frame[targets].to_numpy(dtype=np.float32) if targets else None
    x, y = model._validate_inputs(frame.smiles_list.tolist(), y, num_task=len(targets))
    dataset = model._convert_to_pytorch_data(x, y)
    # Held-out structures may exceed the training size/vocabulary. Never silently
    # truncate them or fit vocabulary statistics on held-out data.
    index = torch.as_tensor(model.dataset_info["active_index"])
    active = set(torch.where(index)[0].tolist()) if index.dtype == torch.bool else set(index.tolist())
    kept = [g for g in dataset if g.num_nodes <= model.max_node and set(g.x.tolist()) <= active]
    if not kept:
        raise ValueError("No structures fit the training graph size and atom vocabulary")
    return DataLoader(
        kept, batch_size=model.batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=model.device.type == "cuda",
        persistent_workers=num_workers > 0,
    ), len(dataset) - len(kept)


def transfer_unconditional_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_parameters = checkpoint["hyperparameters"]
    if source_parameters.get("task_type"):
        raise ValueError("--pretrained-model must be an unconditional Graph-DiT checkpoint")
    source_state = checkpoint["model_state_dict"]
    destination_state = model.model.state_dict()
    compatible = {
        name: value for name, value in source_state.items()
        if name in destination_state and destination_state[name].shape == value.shape
    }
    model.model.load_state_dict(compatible, strict=False)
    skipped = sorted(set(source_state) - set(compatible))
    missing = sorted(set(destination_state) - set(compatible))
    if not compatible:
        raise ValueError("The pretrained checkpoint has no parameters compatible with this model")
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "transferred_tensors": len(compatible),
        "source_tensors_skipped": skipped,
        "new_condition_tensors": missing,
    }


def train_epoch(model, loader, optimizer, precision):
    model.model.train()
    losses = []
    autocast_enabled = model.device.type == "cuda" and precision in {"bf16", "fp16"}
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=autocast_enabled and precision == "fp16")
    active_index = model.dataset_info["active_index"]
    for batch in loader:
        batch = batch.to(model.device, non_blocking=model.device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=model.device.type, dtype=dtype, enabled=autocast_enabled):
            x = F.one_hot(batch.x, num_classes=118).float()[:, active_index]
            e = F.one_hot(batch.edge_attr, num_classes=5).float()
            dense, mask = to_dense(x, batch.edge_index, e, batch.batch, model.max_node)
            dense = dense.mask(mask)
            noisy = model.apply_noise(dense.X, dense.E, batch.y, mask)
            loss, _, _ = model.model.compute_loss(
                noisy, true_X=dense.X, true_E=dense.E, lw_X=model.lw_X, lw_E=model.lw_E
            )
        scaler.scale(loss).backward()
        if model.grad_clip_value is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), model.grad_clip_value)
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.detach().float().item())
    return losses


def train_one(args, targets, output):
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    pretrained = None
    if args.pretrained_model:
        pretrained = torch.load(args.pretrained_model, map_location="cpu", weights_only=False)
        source = pretrained["hyperparameters"]
        for key in ("num_layer", "hidden_size", "num_head", "mlp_ratio", "dropout"):
            setattr(args, key, source[key])
        args.timesteps = source["timesteps"]
    config_path = output / "config.json"
    if config_path.exists() and not args.resume:
        if (output / "metrics.json").exists():
            return "skipped_completed"
        raise ValueError(f"Incomplete run at {output}; use --resume or a fresh output directory")
    if args.resume:
        saved = read_json(config_path)
        if saved["targets"] != targets:
            raise ValueError("Resume targets must match the saved model")
        # Architecture, preprocessing and optimizer settings stay fixed on resume.
        for key, value in saved["arguments"].items():
            if key not in {"epochs", "device", "resume", "output_root", "property", "all", "joint", "unconditional", "fail_fast", "list_properties"}:
                setattr(args, key, value)
        splits = {k: pd.read_csv(output / "prepared_data" / f"{k}.csv") for k in ("train", "validation", "test")}
    else:
        splits, _ = prepare(args.input, targets, output / "prepared_data", args)
        write_json(config_path, {"targets": targets,
                   "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                   "arguments": vars(args), "torch_version": torch.__version__})
    model = GraphDITMolecularGenerator(
        task_type=["regression"] * len(targets), num_layer=args.num_layer,
        hidden_size=args.hidden_size, num_head=args.num_head, batch_size=args.batch_size,
        epochs=args.epochs, timesteps=args.timesteps, learning_rate=args.learning_rate,
        drop_condition=0.1 if targets else 0.0, guide_scale=args.guide_scale,
        grad_clip_value=1.0, use_lr_scheduler=True, device=device_name(args.device), verbose="none")
    start, best, stale, history = 0, float("inf"), 0, []
    if args.resume:
        model.load_from_local(str(output / "last_model.pt"))
        model.device = torch.device(device_name(args.device))
        model.model.to(model.device)
        model._setup_diffusion_params({"hyperparameters": {"dataset_info": model.dataset_info,
                                       "timesteps": model.timesteps, "max_node": model.max_node}})
    else:
        if pretrained is not None:
            source = pretrained["hyperparameters"]
            model._setup_diffusion_params({"hyperparameters": {
                "dataset_info": source["dataset_info"],
                "timesteps": source["timesteps"], "max_node": source["max_node"],
            }})
        else:
            model._setup_diffusion_params(splits["train"].smiles_list.tolist())
        model._initialize_model(model.model_class)
        model.model.initialize_parameters()
        if pretrained is not None:
            report = transfer_unconditional_weights(model, args.pretrained_model)
            write_json(output / "transfer_report.json", report)
    train_loader, train_excluded = make_loader(model, splits["train"], targets, True, args.num_workers)
    val_loader, val_excluded = make_loader(model, splits["validation"], targets, False, args.num_workers)
    optimizer, scheduler = model._setup_optimizers()
    if args.resume:
        state = torch.load(output / "last_checkpoint.pt", map_location="cpu", weights_only=False)
        model.model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for entry in optimizer.state.values():
            for key, value in entry.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    entry[key] = value.to(model.device)
        scheduler.load_state_dict(state["scheduler"])
        start, best, stale, history = state["epoch"] + 1, state["best"], state["stale"], state["history"]
        torch.set_rng_state(state["torch_rng"])
        if model.device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if model.device.type == "mps" and state.get("mps_rng") is not None:
            torch.mps.set_rng_state(state["mps_rng"])
    if start >= args.epochs:
        raise ValueError("--epochs must exceed the saved epoch count when resuming")
    for epoch in range(start, args.epochs):
        train_loss = float(np.mean(train_epoch(model, train_loader, optimizer, args.precision)))
        val, atom, bond = validation_loss(model, val_loader, args.seed + 1000, args.precision)
        if not np.isfinite([train_loss, val, atom, bond]).all():
            raise ValueError("Nonfinite training or validation loss")
        scheduler.step(val)
        model.fitting_loss.append(train_loss)
        model.fitting_epoch = epoch
        model.is_fitted_ = True
        improved = val < best
        best, stale = (val, 0) if improved else (best, stale + 1)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val,
                        "validation_atom_loss": atom, "validation_bond_loss": bond,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        if improved:
            model.save_to_local(str(output / "best_model.pt"))
        model.save_to_local(str(output / "last_model.pt"))
        checkpoint = {"model": model.model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "epoch": epoch, "best": best,
                      "stale": stale, "history": history, "torch_rng": torch.get_rng_state(),
                      "cuda_rng": torch.cuda.get_rng_state_all() if model.device.type == "cuda" else None,
                      "mps_rng": torch.mps.get_rng_state() if model.device.type == "mps" else None}
        temporary = output / "last_checkpoint.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(output / "last_checkpoint.pt")
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        print(f"{output.name}: epoch {epoch + 1}, train={train_loss:.4f}, validation={val:.4f}", flush=True)
        if stale >= args.patience:
            break
    # Evaluate held-out loss once using the selected checkpoint.
    model.load_from_local(str(output / "best_model.pt"))
    test_loader, test_excluded = make_loader(model, splits["test"], targets, False, args.num_workers)
    test_loss = validation_loss(model, test_loader, args.seed + 2000, args.precision)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = pd.DataFrame(history)
    fig, ax = plt.subplots()
    ax.plot(h.epoch, h.train_loss, label="Train")
    ax.plot(h.epoch, h.validation_loss, label="Validation")
    ax.set(xlabel="Epoch", ylabel="Denoising loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "training_loss.png", dpi=160)
    plt.close(fig)
    write_json(output / "metrics.json", {"best_validation_loss": best, "test_loss": test_loss[0],
               "test_atom_loss": test_loss[1], "test_bond_loss": test_loss[2], "epochs": len(history),
               "excluded_from_loss_size_or_vocabulary": {"train": train_excluded, "validation": val_excluded, "test": test_excluded}})
    return "completed"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--unconditional", action="store_true")
    selection.add_argument("--property", action="append", default=[])
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--list-properties", action="store_true")
    p.add_argument("--joint", action="store_true", help="Train one model on all selected properties")
    p.add_argument("--preset", choices=sorted(PRESETS), default="mac")
    p.add_argument("--dataset", choices=["rppd", "pi1m"], default="rppd")
    p.add_argument("--smiles-column", help="p-SMILES column for --dataset pi1m")
    p.add_argument("--max-structures", type=int, help="Seeded subset after cleaning; useful for smoke tests")
    p.add_argument("--pretrained-model",
                   help="Unconditional best_model.pt used to initialize conditional training")
    default_input = ROOT / "data_process/output/rppd_clean.csv"
    p.add_argument(
        "--input",
        default=str(
            default_input
            if default_input.exists()
            else ROOT / "data" / "20260920_rppd.csv"
        ),
    )
    p.add_argument("--output-root")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    for flag, default in [("epochs", None), ("patience", None), ("batch-size", None), ("num-layer", None),
                          ("hidden-size", None), ("num-head", None), ("timesteps", None), ("seed", 42),
                          ("max-heavy-atoms", 50), ("minimum-structures", 50)]:
        p.add_argument("--" + flag, type=int, default=default)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--guide-scale", type=float, default=2.0)
    p.add_argument("--outlier-iqr", type=float, default=0.0)
    p.add_argument("--validation-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--device", choices=["cpu", "auto", "mps", "cuda"])
    args = p.parse_args()
    for key, value in PRESETS[args.preset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    resolved_device = device_name(args.device)
    if args.precision != "fp32" and resolved_device != "cuda":
        p.error("bf16/fp16 training is currently supported only with CUDA; use --precision fp32 locally")
    if args.dataset == "pi1m" and not args.unconditional:
        p.error("--dataset pi1m currently supports --unconditional pretraining only")
    if args.pretrained_model and args.unconditional:
        p.error("--pretrained-model is for conditional RPPD fine-tuning")
    if args.pretrained_model and not Path(args.pretrained_model).is_file():
        p.error(f"Pretrained checkpoint not found: {args.pretrained_model}")
    if args.joint and not args.property:
        p.error("--joint requires --property")
    if args.resume and not args.output_root:
        p.error("--resume requires --output-root pointing to the existing run")
    if not (0 < args.validation_fraction < 1 and 0 < args.test_fraction < 1 and args.validation_fraction + args.test_fraction < 1):
        p.error("validation/test fractions must be positive and sum to less than 1")
    if min(args.epochs, args.patience, args.batch_size, args.num_layer, args.hidden_size, args.num_head, args.timesteps) < 1 or args.hidden_size % args.num_head:
        p.error("positive settings required; hidden size must be divisible by head count")
    if args.list_properties:
        from common import load_source
        for target in RPPD_PROPERTIES:
            frame, _ = load_source(args.input, [target], args.max_heavy_atoms)
            print(f"{target:30} {len(frame):5} eligible structures")
        return
    selected = list(RPPD_PROPERTIES) if args.all else names(args.property)
    if args.property and not selected:
        p.error("Select at least one property")
    unknown = set(selected) - set(RPPD_PROPERTIES)
    if unknown:
        p.error(f"Unknown curated properties: {sorted(unknown)}")
    root = Path(args.output_root) if args.output_root else ROOT / "train_graphdit/output" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    groups = [[]] if args.unconditional else ([selected] if args.joint else [[t] for t in selected])
    summary = {}
    for targets in groups:
        label = "joint_" + "__".join(targets) if args.joint else (targets[0] if targets else "unconditional")
        label = re.sub(r"[^A-Za-z0-9._-]", "_", label)
        try:
            # Resume must not alter options for subsequent models.
            summary[label] = train_one(argparse.Namespace(**vars(args)), targets, root / label)
        except Exception as exc:
            summary[label] = {"status": "failed", "error": str(exc)}
            print(f"{label}: FAILED: {exc}", flush=True)
            write_json(root / "run_summary.json", summary)
            if args.fail_fast:
                raise
        write_json(root / "run_summary.json", summary)
    if any(isinstance(v, dict) and v.get("status") == "failed" for v in summary.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
