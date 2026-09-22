"""Prepare, train, and evaluate GRIN models for RPPD properties.

Examples
--------
List curated properties and their usable sample counts::

    python train_grin/train.py --list-properties

Train one property::

    python train_grin/train.py --property density

Train several properties::

    python train_grin/train.py --property density --property tg

Train every available curated property::

    python train_grin/train.py --all
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from properties import RPPD_PROPERTIES


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PREPARE_SCRIPT = REPOSITORY_ROOT / "data_process" / "prepare_grin.py"
TRAIN_SCRIPT = REPOSITORY_ROOT / "data_process" / "train_grin.py"
DEFAULT_INPUT = REPOSITORY_ROOT / "data_process" / "output" / "rppd_clean.csv"
FALLBACK_INPUT = REPOSITORY_ROOT / "data" / "20260920_rppd.csv"
DEFAULT_OUTPUT_PARENT = REPOSITORY_ROOT / "train_grin" / "output_standardized"


def property_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._-")
    if not slug:
        raise ValueError(f"Cannot create an output directory for property {name!r}")
    return slug


def parse_properties(values: list[str]) -> list[str]:
    properties: list[str] = []
    for value in values:
        properties.extend(part.strip() for part in value.split(",") if part.strip())
    return list(dict.fromkeys(properties))


def inspect_properties(frame: pd.DataFrame, properties: list[str]) -> dict[str, dict[str, int]]:
    report: dict[str, dict[str, int]] = {}
    for target in properties:
        if target not in frame.columns:
            report[target] = {"rows": 0, "unique_smiles": 0, "unique_values": 0}
            continue
        numeric = pd.to_numeric(frame[target], errors="coerce")
        usable = frame.loc[numeric.notna() & frame["smiles_list"].notna()].copy()
        report[target] = {
            "rows": int(len(usable)),
            "unique_smiles": int(usable["smiles_list"].nunique()),
            "unique_values": int(numeric.loc[usable.index].nunique()),
        }
    return report


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def write_run_summary(
    output_root: Path,
    input_path: Path,
    requested: list[str],
    completed: list[str],
    skipped: dict[str, str],
    failed: dict[str, str],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "input": str(input_path),
        "requested_properties": requested,
        "completed_properties": completed,
        "skipped_properties": skipped,
        "failed_properties": failed,
    }
    (output_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--property",
        action="append",
        default=[],
        help="Property to train; repeat the option or provide comma-separated names",
    )
    selection.add_argument(
        "--all", action="store_true", help="Train all available curated RPPD properties"
    )
    selection.add_argument(
        "--list-properties", action="store_true", help="Show available targets and exit"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Cleaned RPPD CSV; defaults to rppd_clean.csv, then the raw export",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Run output directory. By default a new timestamped directory is created "
            "under train_grin/output_standardized/."
        ),
    )
    parser.add_argument("--aggregate", choices=("mean", "median"), default="median")
    parser.add_argument(
        "--outlier-iqr",
        type=float,
        default=3.0,
        help="Tukey fence multiplier; use 0 to disable outlier removal",
    )
    parser.add_argument("--minimum-structures", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-layer", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--verbose",
        choices=("none", "progress_bar", "print_statement"),
        default="progress_bar",
        help="Training output style passed to each GRIN model",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetition-augmentation", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain properties whose completed checkpoint already exists",
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop immediately if one property fails"
    )
    args = parser.parse_args()

    input_path = args.input
    if input_path is None:
        input_path = DEFAULT_INPUT if DEFAULT_INPUT.is_file() else FALLBACK_INPUT
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    frame = pd.read_csv(input_path, low_memory=False)
    if "smiles_list" not in frame.columns:
        raise ValueError("Input CSV must contain a smiles_list column")
    availability = inspect_properties(frame, list(RPPD_PROPERTIES))

    if args.list_properties:
        print(f"Input: {input_path}")
        print(f"{'Property':30} {'Rows':>8} {'Structures':>12} {'Values':>8}")
        for target in RPPD_PROPERTIES:
            item = availability[target]
            print(
                f"{target:30} {item['rows']:8d} "
                f"{item['unique_smiles']:12d} {item['unique_values']:8d}"
            )
        return

    selected = list(RPPD_PROPERTIES) if args.all else parse_properties(args.property)
    if not selected:
        parser.error("choose --property NAME, --all, or --list-properties")

    unknown = [target for target in selected if target not in frame.columns]
    if unknown:
        raise ValueError(f"Unknown properties: {', '.join(unknown)}")

    if args.output_root is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        output_root = (DEFAULT_OUTPUT_PARENT / run_name).resolve()
    else:
        output_root = args.output_root.resolve()
    completed: list[str] = []
    skipped: dict[str, str] = {}
    failed: dict[str, str] = {}

    for position, target in enumerate(selected, start=1):
        numeric = pd.to_numeric(frame[target], errors="coerce")
        usable = frame.loc[numeric.notna() & frame["smiles_list"].notna()]
        unique_structures = int(usable["smiles_list"].nunique())
        if unique_structures < args.minimum_structures:
            skipped[target] = (
                f"only {unique_structures} usable unique structures; "
                f"minimum is {args.minimum_structures}"
            )
            print(f"Skipping {target}: {skipped[target]}")
            continue

        property_dir = output_root / property_slug(target)
        checkpoint = property_dir / "grin_model.pt"
        if checkpoint.is_file() and not args.overwrite:
            skipped[target] = "completed checkpoint exists; use --overwrite to retrain"
            print(f"Skipping {target}: {skipped[target]}")
            continue

        print(f"\n=== [{position}/{len(selected)}] Training {target} ===", flush=True)
        prepared_dir = property_dir / "prepared_data"
        prepare_command = [
            sys.executable,
            str(PREPARE_SCRIPT),
            str(input_path),
            "--target",
            target,
            "--aggregate",
            args.aggregate,
            "--outlier-iqr",
            str(args.outlier_iqr),
            "--seed",
            str(args.seed),
            "--output-dir",
            str(prepared_dir),
        ]
        train_command = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--data-dir",
            str(prepared_dir),
            "--output-dir",
            str(property_dir),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--batch-size",
            str(args.batch_size),
            "--num-layer",
            str(args.num_layer),
            "--hidden-size",
            str(args.hidden_size),
            "--learning-rate",
            str(args.learning_rate),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--verbose",
            args.verbose,
        ]
        if args.repetition_augmentation:
            train_command.append("--repetition-augmentation")

        try:
            run(prepare_command)
            run(train_command)
            completed.append(target)
        except subprocess.CalledProcessError as exc:
            failed[target] = f"command exited with status {exc.returncode}"
            print(f"FAILED {target}: {failed[target]}", file=sys.stderr)
            if args.fail_fast:
                write_run_summary(
                    output_root, input_path, selected, completed, skipped, failed
                )
                raise

    write_run_summary(output_root, input_path, selected, completed, skipped, failed)
    print("\nTraining run finished")
    print(f"Completed: {len(completed)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Failed: {len(failed)}")
    print(f"Summary: {output_root / 'run_summary.json'}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
