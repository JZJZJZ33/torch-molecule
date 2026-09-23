"""FastAPI service for polyNOVA property prediction and polymer generation.

Set ``GRIN_MODEL_RUN_DIR`` to one timestamped training output directory before
starting the service. If it is unset, the newest run below
``train_grin/output_standardized`` is selected automatically.

Fine-tuned models are read from ``train_graphdit/output/rppd_finetuned`` by
default. Set ``GRAPHDIT_MODEL_ROOT`` only to override that location. Models are
discovered recursively and matched by the exact property set recorded in each
``standardization.json``. Set ``GRAPHDIT_UNCONDITIONAL_MODEL_DIR`` to override the
default original open-source checkpoint bundle.

Example startup command (not executed by this module)::

    GRIN_MODEL_RUN_DIR=train_grin/output_standardized/run_20260921_003803 \
    GRAPHDIT_MODEL_ROOT=train_graphdit/output/rppd_finetuned \
    GRAPHDIT_UNCONDITIONAL_MODEL_DIR=train_graphdit/pretrained/llamole_pretrained_graphdit \
      uvicorn polymer_api.app:app --host 0.0.0.0 --port 8000

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
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from torch_molecule import GRINMolecularPredictor
from torch_molecule.visualization import generate_uff_conformer
from train_graphdit.llamole_graphdit import GraphDiT as LlamoleGraphDiT


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOT = REPOSITORY_ROOT / "train_grin" / "output_standardized"
DEFAULT_GENERATION_ROOT = REPOSITORY_ROOT / "train_graphdit" / "output" / "rppd_finetuned"
FRONTEND_ROOT = REPOSITORY_ROOT / "grin_frontend"
ASSETS_ROOT = REPOSITORY_ROOT / "assets"
PUBLIC_ASSETS = frozenset(
    {
        "repetition-predictor.png",
        "multimodal-predictor.png",
        "conditional-generator.png",
        "generator-components.png",
    }
)
RUN_DIRECTORY_ENV = "GRIN_MODEL_RUN_DIR"
GENERATOR_ROOT_ENV = "GRAPHDIT_MODEL_ROOT"
UNCONDITIONAL_GENERATOR_ENV = "GRAPHDIT_UNCONDITIONAL_MODEL_DIR"
DEVICE_ENV = "POLYMER_API_DEVICE"
LEGACY_DEVICE_ENV = "GRIN_API_DEVICE"
EXCLUDED_PUBLIC_PROPERTIES = frozenset({"r2"})
GENERATION_POOL_MULTIPLIER = 20
MIN_GENERATION_POOL_SIZE = 256
MAX_GENERATION_POOL_SIZE = 5000


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


class StructureResponse(BaseModel):
    repeat_units: int
    expanded_smiles: str
    mol_block: str
    uff_energy: float
    uff_converged: bool
    svg_2d: str


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


class GenerationRequest(BaseModel):
    conditions: dict[str, float] = Field(
        default_factory=dict,
        description="Exact property set and target values; empty selects unconditional generation",
    )
    number: int = Field(default=16, ge=1, le=1000)
    batch_size: int = Field(default=32, ge=1, le=128)
    pool_size: int | None = Field(
        default=None,
        ge=1,
        le=MAX_GENERATION_POOL_SIZE,
        description="Raw candidate count; omit to use the server-selected pool size",
    )


class GenerationModelInfo(BaseModel):
    properties: list[str]
    directory: str
    checkpoint: str


class GeneratedPolymer(BaseModel):
    smiles: str | None
    valid: bool
    polymer_valid: bool
    rejection_reason: str


class GenerationResponse(BaseModel):
    properties: list[str]
    requested: dict[str, float]
    model_directory: str
    requested_count: int
    attempt_count: int
    complete: bool
    samples: list[GeneratedPolymer]


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


def inspect_generated_smiles(smiles: str | None) -> GeneratedPolymer:
    if not smiles:
        return GeneratedPolymer(
            smiles=None, valid=False, polymer_valid=False,
            rejection_reason="invalid_smiles",
        )
    if "." in smiles:
        return GeneratedPolymer(
            smiles=None, valid=True, polymer_valid=False,
            rejection_reason="disconnected",
        )
    try:
        molecule = Chem.MolFromSmiles(smiles)
    except Exception:
        molecule = None
    if molecule is None:
        return GeneratedPolymer(
            smiles=None, valid=False, polymer_valid=False,
            rejection_reason="invalid_smiles",
        )
    canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
    attachment_atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0]
    if "." in canonical_smiles or len(Chem.GetMolFrags(molecule)) != 1:
        reason = "disconnected"
    elif len(attachment_atoms) != 2:
        reason = "attachment_count"
    elif any(atom.GetDegree() != 1 for atom in attachment_atoms):
        reason = "attachment_degree"
    else:
        reason = ""
    return GeneratedPolymer(
        smiles=canonical_smiles if not reason else None,
        valid=True,
        polymer_valid=not reason,
        rejection_reason=reason,
    )


@dataclass
class LoadedGenerationModel:
    properties: tuple[str, ...]
    directory: Path
    checkpoint: Path
    transforms: dict[str, dict[str, float]]
    condition_slots: dict[str, int]
    model: LlamoleGraphDiT | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def load(self, device: str) -> LlamoleGraphDiT:
        with self.lock:
            if self.model is None:
                resolved_device = torch.device(device)
                dtype = torch.bfloat16 if resolved_device.type == "cuda" else torch.float32
                model = LlamoleGraphDiT(
                    self.directory / "config.yaml",
                    self.directory / "data.meta.json",
                    dtype,
                )
                state = torch.load(
                    self.checkpoint, map_location="cpu", weights_only=True, mmap=True
                )
                model.denoiser.load_state_dict(state, strict=True)
                model.to(device=resolved_device, dtype=dtype)
                model.eval()
                self.model = model
            return self.model

    def generate(
        self,
        conditions: dict[str, float],
        number: int,
        batch_size: int,
        pool_size: int,
        device: str,
    ) -> tuple[list[GeneratedPolymer], int]:
        model = self.load(device)
        resolved_device = next(model.parameters()).device
        raw_outputs: list[str | None] = []
        accepted: list[GeneratedPolymer] = []
        accepted_smiles: set[str] = set()

        # Generate the complete raw pool before applying any chemistry filters.
        # Each model call remains bounded so the pool is not loaded onto the GPU
        # all at once.
        with self.lock, torch.inference_mode():
            while len(raw_outputs) < pool_size:
                current = min(batch_size, pool_size - len(raw_outputs))
                properties = torch.full(
                    (current, model.ydim),
                    float("nan"),
                    dtype=model.model_dtype,
                    device=resolved_device,
                )
                for name in self.properties:
                    properties[:, self.condition_slots[name]] = (
                        conditions[name] - self.transforms[name]["mean"]
                    ) / self.transforms[name]["std"]
                text = torch.full(
                    (current, model.text_input_size),
                    float("nan"),
                    dtype=model.model_dtype,
                    device=resolved_device,
                )
                outputs = list(model.generate(properties, text, no_label_index=-999.0))
                if not outputs:
                    break
                raw_outputs.extend(outputs[:current])

        for smiles in raw_outputs:
            inspected = inspect_generated_smiles(smiles)
            if not inspected.polymer_valid or inspected.smiles is None:
                continue
            if "." in inspected.smiles or inspected.smiles in accepted_smiles:
                continue
            accepted_smiles.add(inspected.smiles)
            accepted.append(inspected)
            if len(accepted) == number:
                break
        return accepted, len(raw_outputs)


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
            f"No completed prediction runs found below {DEFAULT_RUNS_ROOT}. "
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


def resolve_generation_root() -> Path | None:
    configured = os.environ.get(GENERATOR_ROOT_ENV)
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = REPOSITORY_ROOT / path
        return path.resolve()
    return DEFAULT_GENERATION_ROOT.resolve() if DEFAULT_GENERATION_ROOT.is_dir() else None


def resolve_unconditional_generation_directory() -> Path | None:
    configured = os.environ.get(UNCONDITIONAL_GENERATOR_ENV)
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = REPOSITORY_ROOT / path
        return path.resolve()
    default = REPOSITORY_ROOT / "train_graphdit/pretrained/llamole_pretrained_graphdit"
    return default.resolve() if default.is_dir() else None


def generation_model_from_directory(
    directory: Path,
    properties: tuple[str, ...],
    transforms: dict[str, dict[str, float]],
) -> LoadedGenerationModel:
    checkpoint = directory / "model.pt"
    if (
        checkpoint.is_file()
        and (directory / "config.yaml").is_file()
        and (directory / "data.meta.json").is_file()
    ):
        slots = {name: index for index, name in enumerate(properties)}
        training_config = directory / "training_config.json"
        if training_config.is_file():
            payload = json.loads(training_config.read_text(encoding="utf-8"))
            configured_slots = payload.get("condition_slots", {})
            slots = {name: int(configured_slots.get(name, slots[name])) for name in properties}
        if len(set(slots.values())) != len(slots) or any(not 0 <= slot < 10 for slot in slots.values()):
            raise RuntimeError(f"Invalid condition-slot mapping in {directory}")
        return LoadedGenerationModel(
            properties=properties,
            directory=directory.resolve(),
            checkpoint=checkpoint.resolve(),
            transforms=transforms,
            condition_slots=slots,
        )
    raise RuntimeError(
        f"Expected model.pt, config.yaml and data.meta.json in generation model directory {directory}"
    )


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
                raise RuntimeError(f"Prediction run directory does not exist: {run_directory}")

            models: dict[str, LoadedPropertyModel] = {}
            for property_dir in sorted(path for path in run_directory.iterdir() if path.is_dir()):
                checkpoint = property_dir / "grin_model.pt"
                if not checkpoint.is_file():
                    continue
                target, mean, std = load_standardization(property_dir)
                if target in EXCLUDED_PUBLIC_PROPERTIES:
                    continue
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


class GenerationModelRegistry:
    def __init__(self) -> None:
        self._root: Path | None = None
        self._models: dict[tuple[str, ...], LoadedGenerationModel] | None = None
        self._lock = threading.Lock()

    @property
    def root(self) -> Path | None:
        self.discover()
        return self._root

    def discover(self) -> dict[tuple[str, ...], LoadedGenerationModel]:
        with self._lock:
            if self._models is not None:
                return self._models
            root = resolve_generation_root()
            if root is not None and not root.is_dir():
                raise RuntimeError(f"Generation model root does not exist: {root}")

            models: dict[tuple[str, ...], LoadedGenerationModel] = {}
            metadata_paths = sorted(root.rglob("standardization.json")) if root else []
            for metadata_path in metadata_paths:
                directory = metadata_path.parent
                if not all(
                    (directory / name).is_file()
                    for name in ("model.pt", "config.yaml", "data.meta.json")
                ):
                    continue
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                properties = tuple(metadata.get("targets", []))
                transforms = metadata.get("target_transforms", {})
                if set(transforms) != set(properties):
                    raise RuntimeError(f"Incomplete target transforms in {metadata_path}")
                for name in properties:
                    mean = float(transforms[name]["mean"])
                    std = float(transforms[name]["std"])
                    if not np.isfinite([mean, std]).all() or std <= 0:
                        raise RuntimeError(
                            f"Invalid generation-model standardization for {name}: mean={mean}, std={std}"
                        )
                key = tuple(sorted(properties))
                if key in models:
                    raise RuntimeError(
                        "Duplicate generation models for property set "
                        f"{list(key)}: {models[key].directory} and {directory}"
                    )
                models[key] = generation_model_from_directory(directory, properties, transforms)

            unconditional = resolve_unconditional_generation_directory()
            if () not in models and unconditional is not None:
                models[()] = generation_model_from_directory(unconditional, (), {})
            self._root = root
            self._models = models
            return models

    def get(self, properties: list[str]) -> LoadedGenerationModel:
        models = self.discover()
        key = tuple(sorted(properties))
        try:
            return models[key]
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": "No generation model matches the exact selected property set",
                    "requested_properties": list(key),
                    "available_property_sets": [list(item) for item in sorted(models)],
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
    return os.environ.get(DEVICE_ENV, os.environ.get(LEGACY_DEVICE_ENV, "cpu"))


registry = ModelRegistry()
generation_registry = GenerationModelRegistry()
app = FastAPI(
    title="polyNOVA API",
    version="2.0.0",
    description=(
        "Serve polymer property predictors and exact-property-set structure generators. "
        "Conditions and predictions use original RPPD property units."
    ),
)


@app.get("/assets/{filename}", include_in_schema=False)
def public_asset(filename: str) -> FileResponse:
    """Serve only the architecture figures used by the public pages."""
    if filename not in PUBLIC_ASSETS:
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(ASSETS_ROOT / filename, media_type="image/png")


@app.middleware("http")
async def disable_frontend_cache(request, call_next):
    """Always expose current local frontend files while the studio is in development."""
    response = await call_next(request)
    if request.url.path in {"/", "/generator", "/zh", "/zh/generator"}:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


def render_frontend(template_name: str, script_name: str) -> HTMLResponse:
    """Return one page with local CSS and JavaScript embedded."""
    html = (FRONTEND_ROOT / template_name).read_text(encoding="utf-8")
    styles = (FRONTEND_ROOT / "styles.css").read_text(encoding="utf-8")
    script = (FRONTEND_ROOT / script_name).read_text(encoding="utf-8")
    for stylesheet_path in ("/static/styles.css", "./static/styles.css", "../static/styles.css"):
        html = html.replace(
            f'<link rel="stylesheet" href="{stylesheet_path}" />',
            f"<style>\n{styles}\n</style>",
        )
    for script_path in (f"/static/{script_name}", f"./static/{script_name}", f"../static/{script_name}"):
        html = html.replace(f'<script src="{script_path}" defer></script>', "")
    # Inline scripts do not honor ``defer``. Place the application script after
    # the page markup so all queried controls exist before JavaScript executes.
    html = html.replace("</body>", f"<script>\n{script}\n</script>\n</body>")
    return HTMLResponse(html)


@app.get("/", include_in_schema=False)
def frontend() -> HTMLResponse:
    """Serve the predictor-only first page."""
    return render_frontend("index.html", "app.js")


@app.get("/generator", include_in_schema=False)
def generator_frontend() -> HTMLResponse:
    """Serve conditional generation on a separate second page."""
    return render_frontend("generator.html", "generator.js")


@app.get("/zh", include_in_schema=False)
def frontend_zh() -> HTMLResponse:
    """Serve the Chinese predictor page."""
    return render_frontend("index.zh.html", "app.js")


@app.get("/zh/generator", include_in_schema=False)
def generator_frontend_zh() -> HTMLResponse:
    """Serve the Chinese conditional-generation page."""
    return render_frontend("generator.zh.html", "generator.js")


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        models = registry.discover()
        generation_models = generation_registry.discover()
        return {
            "status": "ok",
            "prediction_run_directory": str(registry.run_directory),
            "generation_model_root": (
                str(generation_registry.root) if generation_registry.root else None
            ),
            "device": api_device(),
            "property_count": len(models),
            "generation_model_count": len(generation_models),
            "generation_property_sets": [list(key) for key in sorted(generation_models)],
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


@app.get("/generation-models", response_model=list[GenerationModelInfo])
def list_generation_models() -> list[GenerationModelInfo]:
    try:
        models = generation_registry.discover()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return [
        GenerationModelInfo(
            properties=list(item.properties),
            directory=str(item.directory),
            checkpoint=str(item.checkpoint),
        )
        for _, item in sorted(models.items())
    ]


@app.post("/structure", response_model=StructureResponse)
def draw_structure(request: StructureRequest) -> StructureResponse:
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
        raise HTTPException(status_code=500, detail=f"2D structure drawing failed: {exc}") from exc

    try:
        conformer = generate_uff_conformer(request.smiles, repeat_units=10)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"3D conformer generation failed: {exc}") from exc
    return StructureResponse(
        repeat_units=conformer.repeat_units,
        expanded_smiles=conformer.smiles,
        mol_block=conformer.mol_block,
        uff_energy=conformer.energy,
        uff_converged=conformer.converged,
        svg_2d=svg,
    )


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


@app.post("/generate", response_model=GenerationResponse)
def generate_polymers(request: GenerationRequest) -> GenerationResponse:
    properties = list(request.conditions)
    if len(properties) != len(set(properties)):
        raise HTTPException(status_code=422, detail="Condition names must be unique")
    invalid_values = [
        name for name, value in request.conditions.items() if not np.isfinite(value)
    ]
    if invalid_values:
        raise HTTPException(
            status_code=422,
            detail={"message": "Condition values must be finite", "properties": invalid_values},
        )
    pool_size = request.pool_size
    if pool_size is None:
        pool_size = min(
            MAX_GENERATION_POOL_SIZE,
            max(
                MIN_GENERATION_POOL_SIZE,
                request.number * GENERATION_POOL_MULTIPLIER,
                request.batch_size,
            ),
        )
    if pool_size < request.number:
        raise HTTPException(
            status_code=422,
            detail="pool_size must be greater than or equal to number",
        )
    item = generation_registry.get(properties)
    try:
        samples, attempt_count = item.generate(
            request.conditions,
            request.number,
            request.batch_size,
            pool_size,
            api_device(),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc
    return GenerationResponse(
        properties=list(item.properties),
        requested={name: request.conditions[name] for name in item.properties},
        model_directory=str(item.directory),
        requested_count=request.number,
        attempt_count=attempt_count,
        complete=len(samples) == request.number,
        samples=samples,
    )
