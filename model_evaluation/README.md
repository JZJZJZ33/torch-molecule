# RPPD model evaluation

This workflow evaluates simple Morgan-fingerprint regression baselines before
additional GRIN tuning. It does not modify GRIN checkpoints, API code, frontend
code, or existing training outputs.

Each target is aggregated to one row per canonical polymer before splitting, so
replicate simulations cannot leak across folds. Metrics and predictions are
reported in the original RPPD units.

Run the full evaluation:

```bash
conda activate torch-molecule
MPLCONFIGDIR=/private/tmp/torch-molecule-matplotlib \
python model_evaluation/evaluate.py --all
```

Evaluate selected properties:

```bash
python model_evaluation/evaluate.py \
  --property tg --property thermal_conductivity
```

Every invocation creates a new timestamped directory under
`model_evaluation/output/` containing:

- `summary.csv`: cross-validation mean and standard deviation by model;
- `fold_metrics.csv`: train/test metrics for every fold;
- `best_models.csv`: strongest non-mean baseline per property;
- `out_of_fold_predictions.csv`: test-fold predictions for error analysis;
- `dataset_summary.csv`: sample counts, ranges, and outlier removals;
- `best_model_r2.png`: compact comparison figure;
- `grin_single_split_reference.csv`: existing GRIN metrics, clearly marked as a
  single-split reference rather than a cross-validation result;
- `run_config.json`: complete reproducibility settings.

Skewed positive targets are evaluated both in raw space and with a `log10`
training transform. Their predictions and evaluation metrics are converted back
to original units.

Run GRIN on the same five held-out folds:

```bash
python model_evaluation/evaluate_grin_cv.py \
  --all \
  --device cuda \
  --output-root model_evaluation/output/grin_cv_server
```

GRIN standardization is fitted separately on each training fold. Checkpoints
are not retained by default because cross-validation trains 85 models; add
`--save-checkpoints` only when those intermediate models are needed.

The fold metrics and predictions are saved after every completed model. If a
server job is interrupted, continue it without repeating completed folds:

```bash
python model_evaluation/evaluate_grin_cv.py \
  --all \
  --device cuda \
  --output-root model_evaluation/output/grin_cv_server \
  --resume
```

The default training configuration matches the existing GRIN runs: 300 maximum
epochs, patience 40, batch size 32, three GNN layers, hidden size 128, and seed
42. The final directory contains:

- `grin_cv_summary.csv`: five-fold mean and standard deviation per property;
- `grin_fold_metrics.csv`: train, validation, and test metrics for every fold;
- `grin_out_of_fold_predictions.csv`: one held-out prediction per structure;
- `grin_dataset_summary.csv`: structure and outlier counts;
- `grin_cv_config.json`: the complete run configuration.

On an NVIDIA server, install a CUDA-enabled PyTorch build first, followed by the
matching `torch-scatter` wheel. Confirm both before starting the long run:

```bash
python -c "import torch, torch_scatter; print(torch.cuda.is_available(), torch.version.cuda)"
```

The first value must be `True`. If it is `False`, do not start with
`--device cuda`; fix the PyTorch/CUDA installation first.
