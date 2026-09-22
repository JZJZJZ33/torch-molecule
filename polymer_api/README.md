# Polymer prediction and generation API

[`app.py`](./app.py) serves GRIN property predictors and Graph-DiT conditional
generators from one FastAPI process. Models load lazily on their first request.

Configure two model locations:

- `GRIN_MODEL_RUN_DIR`: one completed timestamped GRIN run containing property
  subdirectories and `grin_model.pt` checkpoints;
- `GRAPHDIT_MODEL_ROOT`: a directory containing every available fine-tuned
  conditional Graph-DiT model. Discovery is recursive, so single, duo, and trio
  model directories may use any layout below this root.

Each Graph-DiT model directory must contain `best_model.pt` and
`standardization.json`. The API reads `targets` from that metadata and indexes the
checkpoint by its exact property set. Property order does not matter when making
a request. Duplicate checkpoints for the same set cause startup discovery to fail
rather than selecting one ambiguously.

Start locally with the current joint smoke model:

```bash
GRIN_MODEL_RUN_DIR="$PWD/train_grin/output_standardized/run_20260921_003803" \
GRAPHDIT_MODEL_ROOT="$PWD/train_graphdit/output/graphdit_rppd_density_tg" \
POLYMER_API_DEVICE=cpu \
python -m uvicorn polymer_api.app:app \
  --host 127.0.0.1 --port 8000 --reload \
  --reload-dir polymer_api --reload-dir grin_frontend
```

Use `POLYMER_API_DEVICE=cuda` on the H20. `GRIN_API_DEVICE` remains accepted as a
legacy fallback. The predictor is served at `/`, the conditional generator is a
separate page at `/generator`, and interactive API documentation is at `/docs`.

## Discover models

```bash
curl http://localhost:8000/properties
curl http://localhost:8000/generation-models
```

`/generation-models` returns the exact property combination supported by each
checkpoint. For example, selecting density and Tg matches only a checkpoint whose
metadata contains exactly `density` and `tg`.

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
