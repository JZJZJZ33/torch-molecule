# polySIM

This root-level package converts a two-ended repeat-unit pSMILES into:

1. a hydrogen-terminated finite chain;
2. an RDKit MMFF94/MMFF94s conformer;
3. optionally, a GFN-FF geometry-optimized and xTB-analyzed **single chain**; and
4. an unoptimized approximate periodic box containing rigid chain copies.

Linear-chain assembly uses `stk.polymer.Linear`, following the supplied STK
building-block approach. RDKit remains responsible for pSMILES parsing,
STK/RDKit interchange, and MMFF conformer generation. xTB is never applied to
the assembled multi-chain system.

For `N >= 2`, STK receives an H/Br initial unit (`A`), `N-2` Br/Br middle
units (`B`), and a Br/H terminal unit (`C`). Thus the topology sequence is
`A + B*(N-2) + C`, containing exactly `N` chemical repeat units. For `N=1`,
both wildcard sites are replaced by hydrogen directly.

## pSMILES input

Place exactly one pSMILES on a non-empty, non-comment line. Explicit head and
tail positions are determined by their order in the string:

```text
# pvdf.psmiles.txt
[*]CC(F)(F)[*]
```

An example is provided at `polySIM/examples/pvdf.psmiles.txt`.

The builder does not canonicalize the pSMILES before interpreting its connection
points. RDKit input atom order is retained: the first `[*]` is the head and the
second is the tail. After exactly `N` repeat units are connected, the two outer
valences are hydrogen-capped. For example, `N=10` PVDF gives `C20H22F20`.

By default, the two wildcard bonds must have the same bond order. Some PI1M
records use different head/tail bond orders. For those, explicitly choose
`--mismatched-bond-policy single` to reproduce the supplied STK builder's
effective single-bond linking behavior. The default remains `strict` so
bond-order changes are never silent.

## One-command workflow

From the repository root:

```bash
PYTHONPATH=. python -m polySIM.build_system \
  --psmiles-file polySIM/examples/pvdf.psmiles.txt \
  --repeat-units 50 \
  --chains 30 \
  --xtb-method gfn2 \
  --density 0.3 \
  --minimum-distance 2.0 \
  --output-directory polySIM/output/pvdf_30x50
```

The default is a staged single-chain calculation: GFN-FF geometry optimization
with xTB's inertial optimizer, followed by a GFN2-xTB single-point energy on the
relaxed geometry. Charge 0, UHF 0, no solvent, 14 OpenMP threads, and no Hessian
follow the supplied reference scripts. The inertial optimizer is used because
the default approximate-normal-coordinate optimizers can fail while constructing
a model Hessian for long, flexible chains.

Override the stages with `--xtb-method`, `--xtb-geometry-method`,
`--xtb-optimizer-engine`, `--xtb-charge`, `--xtb-uhf`, `--xtb-threads`,
`--xtb-opt-level`, and `--xtb-hessian`. The Hessian option applies to the
geometry-optimization stage.

Supported `--xtb-method` values:

- `gfn2` (default), `gfn1`, or `gfn0` for the post-optimization electronic
  single point;
- `gfnff` to run only the GFN-FF geometry optimization;
- `none` to pack the converged MMFF chain without xTB.

Set `--xtb-geometry-method none` to omit the GFN-FF relaxation and run the
selected electronic method as a single point on the MMFF geometry. xTB is used
only for one isolated chain. A failure stops the workflow and retains the
`.xtb.log`; it does not silently substitute a result.

## xTB analysis

After xTB, the workflow writes `xtb_analysis/xtb_analysis.json` and `.csv`.
The analysis extracts the final total energy, reports relative energies, ranks
results, and stages selected geometries. `xtb_analysis.py` can also analyze a
directory containing several same-atom-order xTB conformers. It uses ordered
Kabsch RMSD filtering, avoiding unreliable bond inference from XYZ files.

Standalone usage:

```bash
PYTHONPATH=. python -m polySIM.xtb_analysis \
  --input-directory path/to/xtb/results \
  --output-directory path/to/analysis \
  --rmsd-threshold 2.0 \
  --max-selected 3
```

## Output

- `input.psmiles.txt`: copied input for provenance;
- `chain_topology.sdf`: finite-chain topology;
- `chain_mmff.xyz`: MMFF-optimized chain;
- `chain_mmff_explicit_h.sdf`: explicit-H topology matching XYZ atom order;
- `chain_gfnff_opt.xyz` and `.xtb.log` for the geometry optimization;
- `chain_<method>_sp.xyz` and `.xtb.log` for an electronic single point;
- `xtb_analysis/`: xTB energy and conformer-selection reports;
- `system.xyz`: all chains in one approximate box;
- `system.box.json`: box dimensions and atom ranges for every chain;
- `manifest.json`: inputs, methods, convergence, and output inventory.

XYZ does not contain bonds or periodic box vectors. For handover, keep
`system.xyz`, `system.box.json`, `chain_mmff_explicit_h.sdf`, and
`manifest.json` together.

## Independent modules

```text
chain_builder.py  -> finite-chain topology
conf_builder.py   -> MMFF conformer
xtb_calc.py       -> isolated-chain xTB optimization/single point/Hessian
xtb_analysis.py   -> energy ranking and RMSD diversity analysis
system_builder.py -> unoptimized approximate multi-chain box
build_system.py   -> complete orchestration
```

Use `python -m polySIM.<module> --help` for all options.
