# Polymer prediction and generation API

[`app.py`](./app.py) serves polymer-property predictors plus unconditional and
property-guided graph generators from one FastAPI process. Models load lazily on
their first request.

Configure two model locations:

- `GRIN_MODEL_RUN_DIR`: one completed timestamped GRIN run containing property
  subdirectories and `grin_model.pt` checkpoints;
- `GRAPHDIT_MODEL_ROOT`: optional override for the directory containing every
  fine-tuned conditional model. It defaults to
  `./train_graphdit/output/rppd_finetuned`. Discovery is recursive, so single,
  duo, and trio model directories may use any layout below this root;
- `GRAPHDIT_UNCONDITIONAL_MODEL_DIR`: the base-model directory used when the user
  selects **No conditions**. If unset, the API uses
  `train_graphdit/pretrained/llamole_pretrained_graphdit` when it exists.

The generator accepts only the downloaded open-source model format and models
fine-tuned from it. A fine-tuned directory contains `model.pt`, `config.yaml`,
`data.meta.json`, `standardization.json`, and `training_config.json`. The original
unconditional directory contains `model.pt`, `config.yaml`, and `data.meta.json`.
Older `best_model.pt` checkpoints produced by the local training workflow are not
discovered or served.

The API reads `targets` from `standardization.json` and indexes each checkpoint by
its exact property set. Property order does not matter when making a request.
Duplicate checkpoints for the same set cause discovery to fail rather than
selecting one ambiguously.

Start the application with the directory containing the fine-tuned `model.pt`
models and the original model bundle:

```bash
GRIN_MODEL_RUN_DIR="$PWD/train_grin/output_standardized/run_20260921_003803" \
GRAPHDIT_MODEL_ROOT="$PWD/train_graphdit/output/rppd_finetuned" \
GRAPHDIT_UNCONDITIONAL_MODEL_DIR="$PWD/train_graphdit/pretrained/llamole_pretrained_graphdit" \
POLYMER_API_DEVICE=cpu \
python -m uvicorn polymer_api.app:app \
  --host 127.0.0.1 --port 8000 --reload \
  --reload-dir polymer_api --reload-dir grin_frontend
```

When both model directories use their default repository locations, the two
`GRAPHDIT_...` variables may be omitted. Set them only when the server stores the
models elsewhere.

Use `POLYMER_API_DEVICE=cuda` on the H20. `GRIN_API_DEVICE` remains accepted as a
legacy fallback. The predictor is served at `/`, the generator is a separate page
at `/generator`, and interactive API documentation is at `/docs`.

## Discover models

```bash
curl http://localhost:8000/properties
curl http://localhost:8000/generation-models
```

`/generation-models` returns the exact property combination supported by each
checkpoint. For example, selecting density and Tg matches only a checkpoint whose
metadata contains exactly `density` and `tg`.

## Generate without property conditions

An empty `conditions` object selects the configured base generator:

```bash
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "conditions":{},
    "number":16,
    "batch_size":8
  }'
```

## Predict properties

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "smiles":["*CC(*)c1ccccc1"],
    "properties":["density","tg"]
  }'
```

Omit `properties` to run every available GRIN predictor. Results are converted
back to the original RPPD units.

## Generate for one property

This requires a density-only fine-tuned checkpoint below `GRAPHDIT_MODEL_ROOT`:

```bash
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "conditions":{"density":1.2},
    "number":16,
    "batch_size":8
  }'
```

## Generate for a joint property set

```bash
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "conditions":{"tg":420,"density":1.2},
    "number":16,
    "batch_size":8
  }'
```

The request above selects the density + Tg joint model regardless of JSON key
order. If only separate density and Tg checkpoints exist, the request returns 404;
separate models are never combined implicitly.

Responses retain every sampling attempt and report RDKit validity, two-attachment
polymer validity, and rejection reasons. Property values supplied to the endpoint
use original RPPD units and are standardized with the selected model's saved
training statistics.
