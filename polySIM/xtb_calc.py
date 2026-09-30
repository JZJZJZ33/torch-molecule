#!/usr/bin/env python3
"""Run an isolated xTB single-point calculation or geometry optimization."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class XTBResult:
    optimized_xyz: Path | None
    log: Path
    energy_hartree: float | None
    thermo_log: Path | None = None


ENERGY_PATTERN = re.compile(
    r"(?:TOTAL ENERGY|total energy)\s+(?:\.{2,}|[:|])?\s*"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*Eh"
)


def _energy_from_json(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("total energy", "total_energy", "energy"):
        if key in data:
            return float(data[key])
    return None


def _energy_from_output(output: str) -> float | None:
    """Return the last energy printed by xTB when JSON is unavailable."""
    matches = ENERGY_PATTERN.findall(output)
    return float(matches[-1]) if matches else None


def run_xtb(
    input_xyz: str | Path,
    output_xyz: str | Path,
    *,
    charge: int = 0,
    unpaired_electrons: int = 0,
    method: str = "gfn2",
    optimize: bool = True,
    opt_level: str = "normal",
    optimizer_engine: str | None = None,
    solvent: str | None = None,
    threads: int = 14,
    hessian: bool = False,
    temperature_kelvin: float = 298.15,
    xtb_executable: str = "xtb",
) -> XTBResult:
    """Run xTB in a temporary directory and copy durable outputs back."""
    input_xyz = Path(input_xyz).resolve()
    output_xyz = Path(output_xyz).resolve()
    if not input_xyz.is_file():
        raise FileNotFoundError(input_xyz)
    executable = shutil.which(xtb_executable)
    if executable is None:
        raise FileNotFoundError(f"xTB executable not found: {xtb_executable}")
    method = method.lower()
    if method not in {"gfn0", "gfn1", "gfn2", "gfnff"}:
        raise ValueError("method must be gfn0, gfn1, gfn2, or gfnff")
    if unpaired_electrons < 0:
        raise ValueError("unpaired_electrons cannot be negative")
    if threads < 1:
        raise ValueError("threads must be positive")
    if optimizer_engine not in {None, "rf", "lbfgs", "inertial"}:
        raise ValueError("optimizer_engine must be rf, lbfgs, inertial, or None")

    output_xyz.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_xyz.with_suffix(".xtb.log")
    with tempfile.TemporaryDirectory(prefix="xtb_calc_") as temporary:
        work = Path(temporary)
        local_input = work / "input.xyz"
        shutil.copy2(input_xyz, local_input)
        command = [executable, local_input.name]
        if method == "gfnff":
            command.append("--gfnff")
        else:
            command.extend(("--gfn", method[-1]))
        command.extend([
            "--chrg",
            str(charge),
            "--uhf",
            str(unpaired_electrons),
            "--json",
        ])
        if optimizer_engine is not None:
            xcontrol = work / "xcontrol.inp"
            xcontrol.write_text(
                f"$opt\n  engine={optimizer_engine}\n$end\n",
                encoding="utf-8",
            )
            command.extend(("--input", xcontrol.name))
        if optimize:
            command.extend(("--opt", opt_level))
        if solvent:
            command.extend(("--alpb", solvent))

        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = str(threads)
        process = subprocess.run(
            command,
            cwd=work,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path.write_text(
            "$ " + " ".join(command) + "\n\n" + process.stdout,
            encoding="utf-8",
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"xTB failed with exit code {process.returncode}; see {log_path}"
            )

        source_xyz = work / ("xtbopt.xyz" if optimize else local_input.name)
        if not source_xyz.exists():
            raise RuntimeError(f"xTB completed but did not produce {source_xyz.name}")
        shutil.copy2(source_xyz, output_xyz)
        energy = _energy_from_json(work / "xtbout.json")
        if energy is None:
            energy = _energy_from_output(process.stdout)
        thermo_log = None
        if hessian and optimize:
            thermo_log = output_xyz.with_suffix(".thermo.log")
            thermo_command = [
                executable,
                source_xyz.name,
                "--hess",
                "--temp",
                str(temperature_kelvin),
                "--chrg",
                str(charge),
                "--uhf",
                str(unpaired_electrons),
            ]
            if method == "gfnff":
                thermo_command.append("--gfnff")
            else:
                thermo_command.extend(("--gfn", method[-1]))
            if solvent:
                thermo_command.extend(("--alpb", solvent))
            thermo_process = subprocess.run(
                thermo_command,
                cwd=work,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            thermo_log.write_text(
                "$ " + " ".join(thermo_command) + "\n\n" + thermo_process.stdout,
                encoding="utf-8",
            )
            if thermo_process.returncode != 0:
                raise RuntimeError(
                    f"xTB Hessian failed with exit code {thermo_process.returncode}; "
                    f"see {thermo_log}"
                )
    return XTBResult(output_xyz, log_path, energy, thermo_log)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--uhf", type=int, default=0, help="number of unpaired electrons")
    parser.add_argument(
        "--method",
        choices=("gfn0", "gfn1", "gfn2", "gfnff"),
        default="gfn2",
    )
    parser.add_argument("--single-point", action="store_true")
    parser.add_argument(
        "--opt-level",
        choices=("crude", "sloppy", "loose", "normal", "tight", "verytight"),
        default="normal",
    )
    parser.add_argument(
        "--optimizer-engine",
        choices=("rf", "lbfgs", "inertial"),
        help="xTB geometry optimizer engine; inertial is robust for long flexible chains",
    )
    parser.add_argument("--solvent", help="xTB ALPB solvent name")
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--hessian", action="store_true")
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument("--xtb-executable", default="xtb")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_xtb(
        args.input,
        args.output,
        charge=args.charge,
        unpaired_electrons=args.uhf,
        method=args.method,
        optimize=not args.single_point,
        opt_level=args.opt_level,
        optimizer_engine=args.optimizer_engine,
        solvent=args.solvent,
        threads=args.threads,
        hessian=args.hessian,
        temperature_kelvin=args.temperature,
        xtb_executable=args.xtb_executable,
    )
    print(f"Wrote {result.optimized_xyz}")
    print(f"Log: {result.log}")
    if result.energy_hartree is not None:
        print(f"xTB energy: {result.energy_hartree:.12f} Eh")


if __name__ == "__main__":
    main()
