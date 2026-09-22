# RPPD preprocessing

These scripts turn a raw RPPD export into reproducible inputs for GRIN and
Graph-DiT. They never modify the source CSV. The cleaned dataset and its metadata
are versioned for server transfer; temporary reports and prepared splits below
`output/` remain excluded from Git because they can be regenerated.

Install the additional table-processing dependency in the torch-molecule environment:

```bash
conda activate torch-molecule
python -m pip install pandas
```

Inspect and clean the source data:

```bash
python data_process/inspect_rppd.py data/20260920_rppd.csv
python data_process/clean_rppd.py data/20260920_rppd.csv
```

Prepare leakage-safe GRIN splits for one regression target:

```bash
python data_process/prepare_grin.py data/20260920_rppd.csv --target density
```

This aggregates repeated simulations by canonical SMILES, splits unique structures,
and standardizes the target using training-set statistics. Raw target values and
replicate statistics are retained in the generated CSVs.

Prepare an unconditional Graph-DiT dataset:

```bash
python data_process/prepare_graphdit.py data/20260920_rppd.csv
```

Prepare a conditional Graph-DiT dataset:

```bash
python data_process/prepare_graphdit.py data/20260920_rppd.csv --target density
```

Multiple conditioning properties may be provided by repeating `--target` or using
a comma-separated value. Graph-DiT targets are standardized and their inverse
transform parameters are saved in `metadata.json`.

Outlier removal is disabled by default. Enable explicit Tukey fences, for example:

```bash
python data_process/prepare_grin.py data/20260920_rppd.csv \
  --target tg --outlier-iqr 3
```

Generated files are placed below `data_process/output/`.

`prepare_grin.py` is also called automatically by the multi-property training
entry point. To train one or more properties and create evaluation artifacts,
use:

```bash
python train_grin/train.py --property density
```

The lower-level `data_process/train_grin.py` remains the single-property trainer
used internally by that command. The training workflow writes a local checkpoint,
MAE/MSE/RMSE/R² metrics, per-split prediction CSVs, a prediction scatter plot,
and a training-loss plot.
