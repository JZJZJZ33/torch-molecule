# Train GRIN across RPPD properties

This directory provides one command for preparing data, training GRIN, saving the
best checkpoint, and producing evaluation artifacts for one or more RPPD material
properties.

The default input is `data_process/output/rppd_clean.csv` when available, otherwise
the original `20260920_rppd.csv` export is used. Source files are never modified.
The first standardized run used by the API is versioned for server deployment.
New timestamped runs remain excluded from Git until explicitly selected for a
future release.

## Commands

List curated properties and usable sample counts:

```bash
python train_grin/train.py --list-properties
```

Train one property:

```bash
python train_grin/train.py --property density
```

Train multiple selected properties:

```bash
python train_grin/train.py --property density --property tg
```

Comma-separated selection is also supported:

```bash
python train_grin/train.py --property density,tg,thermal_conductivity
```

Train all curated properties sequentially:

```bash
python train_grin/train.py --all
```

By default, completed properties are skipped. Use `--overwrite` to retrain them.
Use `--fail-fast` to stop the all-property run at its first failure.

Outlier filtering uses a conservative three-IQR Tukey fence by default. Disable it
with `--outlier-iqr 0`.

## Standardization and inverse transformation

For every property, replicate measurements are aggregated first. The target mean
and standard deviation are then fitted **only on the training split**:

```text
standardized = (original - training_mean) / training_std
```

GRIN is trained on these standardized targets. After prediction, values are
automatically converted back to the original RPPD units:

```text
original = standardized * training_std + training_mean
```

Each property directory contains `standardization.json` with the fitted parameters
and formulas. Prediction CSVs contain both standardized and inverse-transformed
columns. All reported MAE, MSE, RMSE, R² plots, and residuals use original units.

## Output

By default every invocation creates a new timestamped directory, so it cannot
overwrite a previous run:

```text
train_grin/output_standardized/
└── run_20260921_123456/
    ├── run_summary.json
    ├── density/
    │   ├── prepared_data/
    │   │   ├── train.csv
    │   │   ├── validation.csv
    │   │   ├── test.csv
    │   │   └── metadata.json
    │   ├── grin_model.pt
    │   ├── metrics.json
    │   ├── standardization.json
    │   ├── prediction_scatter.png
    │   ├── training_loss.png
    │   ├── training_history.csv
    │   ├── predictions_train.csv
    │   ├── predictions_validation.csv
    │   └── predictions_test.csv
    └── tg/
        └── ...
```

Every metric is reported in the property's original RPPD units. Each property's
`metrics.json` also records the target standardization needed for new predictions.
