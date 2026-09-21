"""FastAPI service for locally trained GRIN property-prediction models.

Set ``GRIN_MODEL_RUN_DIR`` to one timestamped training output directory before
starting the service. If it is unset, the newest run below
``train_grin/output_standardized`` is selected automatically.

Example startup command (not executed by this module)::

    GRIN_MODEL_RUN_DIR=train_grin/output_standardized/run_20260921_123456 \
      uvicorn grin_api.app:app --host 0.0.0.0 --port 8000

All predictions returned by this API are inverse-transformed to the original
RPPD property units.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from torch_molecule import GRINMolecularPredictor


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOT = REPOSITORY_ROOT / "train_grin" / "output_standardized"
FRONTEND_ROOT = REPOSITORY_ROOT / "grin_frontend"
RUN_DIRECTORY_ENV = "GRIN_MODEL_RUN_DIR"
DEVICE_ENV = "GRIN_API_DEVICE"


class PredictionRequest(BaseModel):
    smiles: list[str] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Molecular or polymer SMILES strings to predict",
    )


class MultiPropertyPredictionRequest(PredictionRequest):
    properties: list[str] | None = Field(
        default=None,
        description="Properties to predict; omit to use every available model",
    )


class StructureRequest(BaseModel):
    smiles: str = Field(..., min_length=1, max_length=10000)


class PropertyInfo(BaseModel):
    name: str
    checkpoint: str
    mean: float
    std: float
    units: str = "original RPPD units"


class PropertyPrediction(BaseModel):
    property: str
    units: str
    values: list[float]


class MultiPropertyPrediction(BaseModel):
    smiles: list[str]
    predictions: dict[str, list[float]]
    units: str = "original RPPD units"


@dataclass
class LoadedPropertyModel:
    name: str
    checkpoint: Path
    mean: float
    std: float
    model: GRINMolecularPredictor | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def load(self, device: str) -> GRINMolecularPredictor:
        with self.lock:
            if self.model is None:
                model = GRINMolecularPredictor(device=device, verbose="none")
                model.load_from_local(str(self.checkpoint))
                model.set_params(verbose="none")
                self.model = model
            return self.model

    def predict(self, smiles: list[str], device: str) -> list[float]:
        model = self.load(device)
        with self.lock:
            standardized = model.predict(smiles)["prediction"].reshape(-1)
        original = standardized.astype(float) * self.std + self.mean
        return [float(value) for value in original]


def resolve_run_directory() -> Path:
    configured = os.environ.get(RUN_DIRECTORY_ENV)
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = REPOSITORY_ROOT / path
        return path.resolve()

    candidates = sorted(
        (
            path
            for path in DEFAULT_RUNS_ROOT.glob("run_*")
            if path.is_dir() and any(path.glob("*/grin_model.pt"))
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(
            f"No completed GRIN runs found below {DEFAULT_RUNS_ROOT}. "
            f"Set {RUN_DIRECTORY_ENV} to a completed run directory."
        )
    return candidates[0].resolve()


def load_standardization(property_dir: Path) -> tuple[str, float, float]:
    standardization_path = property_dir / "standardization.json"
    if standardization_path.is_file():
        payload = json.loads(standardization_path.read_text(encoding="utf-8"))
        return str(payload["target"]), float(payload["mean"]), float(payload["std"])

    # Backward compatibility with models created before standardization.json was added.
    metrics_path = property_dir / "metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError(f"Missing standardization metadata in {property_dir}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    transform = payload["target_transform"]
    return str(payload["target"]), float(transform["mean"]), float(transform["std"])


class ModelRegistry:
    def __init__(self) -> None:
        self._run_directory: Path | None = None
        self._models: dict[str, LoadedPropertyModel] | None = None
        self._lock = threading.Lock()

    @property
    def run_directory(self) -> Path:
        self.discover()
        assert self._run_directory is not None
        return self._run_directory

    def discover(self) -> dict[str, LoadedPropertyModel]:
        with self._lock:
            if self._models is not None:
                return self._models

            run_directory = resolve_run_directory()
            if not run_directory.is_dir():
                raise RuntimeError(f"GRIN run directory does not exist: {run_directory}")

            models: dict[str, LoadedPropertyModel] = {}
            for property_dir in sorted(path for path in run_directory.iterdir() if path.is_dir()):
                checkpoint = property_dir / "grin_model.pt"
                if not checkpoint.is_file():
                    continue
                target, mean, std = load_standardization(property_dir)
                if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
                    raise RuntimeError(f"Invalid standardization for {target}: mean={mean}, std={std}")
                if target in models:
                    raise RuntimeError(f"Duplicate model for property {target!r}")
                models[target] = LoadedPropertyModel(
                    name=target,
                    checkpoint=checkpoint.resolve(),
                    mean=mean,
                    std=std,
                    lock=threading.Lock(),
                )

            if not models:
                raise RuntimeError(f"No completed property checkpoints found in {run_directory}")
            self._run_directory = run_directory
            self._models = models
            return models

    def get(self, property_name: str) -> LoadedPropertyModel:
        models = self.discover()
        try:
            return models[property_name]
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": f"No trained model for property {property_name!r}",
                    "available_properties": sorted(models),
                },
            ) from exc


def validate_smiles(smiles: list[str]) -> None:
    invalid = [index for index, value in enumerate(smiles) if Chem.MolFromSmiles(value) is None]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "One or more invalid SMILES strings were supplied",
                "invalid_indices": invalid,
            },
        )


def api_device() -> str:
    return os.environ.get(DEVICE_ENV, "cpu")


registry = ModelRegistry()
app = FastAPI(
    title="GRIN Polymer Property Prediction API",
    version="1.0.0",
    description=(
        "Serve locally trained torch-molecule GRIN models. Predictions are returned "
        "in original RPPD property units. No Hugging Face access is required."
    ),
)


@app.middleware("http")
async def disable_frontend_cache(request, call_next):
    """Always expose current local frontend files while the studio is in development."""
    response = await call_next(request)
    if request.url.path == "/":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.get("/", include_in_schema=False)
def frontend() -> HTMLResponse:
    """Return one self-contained page with no external CSS or JavaScript files."""
    html = (FRONTEND_ROOT / "index.html").read_text(encoding="utf-8")
    styles = (FRONTEND_ROOT / "styles.css").read_text(encoding="utf-8")
    script = (FRONTEND_ROOT / "app.js").read_text(encoding="utf-8")
    html = html.replace(
        '<link rel="stylesheet" href="/static/styles.css" />',
        f"<style>\n{styles}\n</style>",
    )
    html = html.replace('<script src="/static/app.js" defer></script>', "")
    # Inline scripts do not honor ``defer``. Place the application script after
    # the page markup so all queried controls exist before JavaScript executes.
    html = html.replace("</body>", f"<script>\n{script}\n</script>\n</body>")
    return HTMLResponse(html)


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        models = registry.discover()
        return {
            "status": "ok",
            "run_directory": str(registry.run_directory),
            "device": api_device(),
            "property_count": len(models),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/properties", response_model=list[PropertyInfo])
def list_properties() -> list[PropertyInfo]:
    try:
        models = registry.discover()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return [
        PropertyInfo(
            name=name,
            checkpoint=str(item.checkpoint),
            mean=item.mean,
            std=item.std,
        )
        for name, item in sorted(models.items())
    ]


@app.post("/structure", response_class=Response)
def draw_structure(request: StructureRequest) -> Response:
    molecule = Chem.MolFromSmiles(request.smiles)
    if molecule is None:
        raise HTTPException(status_code=422, detail="Invalid pSMILES string")
    try:
        rdDepictor.Compute2DCoords(molecule)
        drawer = rdMolDraw2D.MolDraw2DSVG(720, 420)
        options = drawer.drawOptions()
        options.clearBackground = False
        options.padding = 0.08
        drawer.DrawMolecule(molecule)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RDKit drawing failed: {exc}") from exc
    return Response(content=svg, media_type="image/svg+xml")


@app.post("/predict/{property_name}", response_model=PropertyPrediction)
def predict_property(
    property_name: str,
    request: PredictionRequest,
) -> PropertyPrediction:
    validate_smiles(request.smiles)
    item = registry.get(property_name)
    try:
        values = item.predict(request.smiles, api_device())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return PropertyPrediction(
        property=property_name,
        units="original RPPD units",
        values=values,
    )


@app.post("/predict", response_model=MultiPropertyPrediction)
def predict_multiple(request: MultiPropertyPredictionRequest) -> MultiPropertyPrediction:
    validate_smiles(request.smiles)
    models = registry.discover()
    properties = request.properties or sorted(models)
    unknown = sorted(set(properties) - set(models))
    if unknown:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Some requested properties have no trained model",
                "unknown_properties": unknown,
                "available_properties": sorted(models),
            },
        )

    predictions: dict[str, list[float]] = {}
    try:
        for property_name in properties:
            predictions[property_name] = models[property_name].predict(
                request.smiles, api_device()
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return MultiPropertyPrediction(smiles=request.smiles, predictions=predictions)
