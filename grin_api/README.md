# GRIN prediction API

[`app.py`](./app.py) serves every completed property model from one timestamped
`train_grin` run. Models load lazily on their first request, and outputs are
inverse-transformed to original RPPD units using each model's saved
`standardization.json`.

Install the API dependencies in the torch-molecule environment:

```bash
python -m pip install fastapi uvicorn
```

Select a completed training run and start the API:

```bash
GRIN_MODEL_RUN_DIR="$PWD/train_grin/output_standardized/run_20260921_003803" \
GRIN_API_DEVICE=cpu \
python -m uvicorn grin_api.app:app \
  --host 127.0.0.1 --port 8000 --reload \
  --reload-dir grin_api --reload-dir grin_frontend
```

If `GRIN_MODEL_RUN_DIR` is omitted, the newest completed timestamped run is used.
Interactive OpenAPI documentation is available at `/docs` while the server runs.
The browser frontend is served from `/` by the same API process. Its CSS and
JavaScript are embedded into that single HTML response, so no separate static-file
server or `/static` proxy configuration is required.
Open `http://127.0.0.1:8000`; a separate frontend server is not required.

List models:

```bash
curl http://localhost:8000/properties
```

Predict one property:

```bash
curl -X POST http://localhost:8000/predict/density \
  -H 'Content-Type: application/json' \
  -d '{"smiles":["*CC(*)c1ccccc1"]}'
```

Predict selected properties:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "smiles":["*CC(*)c1ccccc1"],
    "properties":["density","tg"]
  }'
```

Omit `properties` from the multi-property endpoint to run every available model.
