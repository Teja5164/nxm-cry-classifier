"""
app.py — FastAPI Backend for Infant Cry Classifier
===================================================
Three-stage pipeline: Cry Gate → OOD Check → 5-class Ensemble.

Run (development):
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

Run (production):
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 2

API Endpoints:
    GET  /health      — health check with gate status
    GET  /info        — model info and class labels
    POST /predict     — single audio file inference (WAV or MP3)
    POST /predict/batch — multiple audio files
"""

import os, sys
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent      # api/
_ROOT = _HERE.parent                         # ML_pipeline/
_INF  = _ROOT / 'inference'
if str(_INF) not in sys.path:
    sys.path.insert(0, str(_INF))

# ── FastAPI ───────────────────────────────────────────────────────────────────
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tempfile, shutil, json

from predict import predict as _predict, predict_batch as _predict_batch, CLASSES
from yamnet_gate import get_gate

app = FastAPI(
    title        = "Infant Cry Classifier API",
    description  = "Three-stage infant cry classifier: Gate → OOD → 5-class ensemble",
    version      = "2.0.0",
    docs_url     = "/docs",
    redoc_url    = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Response models ───────────────────────────────────────────────────────────
class PredictionResult(BaseModel):
    is_cry:        bool
    prediction:    str
    confidence:    float
    probabilities: dict
    reliability:   str
    entropy:       float
    n_models:      int
    device:        str
    dataset:       str
    audio_path:    str
    gate_score:    float
    gate_method:   str
    stage_blocked: Optional[str]
    reason:        str


class HealthResponse(BaseModel):
    status:      str
    version:     str
    classes:     List[str]
    gate_method: str
    gate_ready:  bool


# ── Startup: pre-warm the gate ────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Pre-load the cry gate at startup to avoid cold-start latency."""
    try:
        gate = get_gate()
        print(f"[Startup] Cry gate ready — method: {gate.method}")
    except Exception as e:
        print(f"[Startup] Gate init warning: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, summary="Health check")
@app.get("/",       response_model=HealthResponse, summary="Health check (alias)")
def health():
    gate = get_gate()
    return {
        "status":      "ok",
        "version":     "2.0.0",
        "classes":     CLASSES,
        "gate_method": gate.method,
        "gate_ready":  gate.available,
    }


@app.get("/info", summary="Model and class info")
def info():
    labels_path = _INF / 'labels.json'
    with open(labels_path) as f:
        labels = json.load(f)
    gate = get_gate()
    return {
        "classes":        CLASSES,
        "class_info":     labels.get("descriptions", {}),
        "confusion_pairs": labels.get("confusion_pairs", []),
        "n_models":       4,
        "architecture":   ["BaselineCNN", "CNN+BiLSTM", "CryNet(Transformer)", "SE-ResNet"],
        "dataset":        "dataset1",
        "pipeline":       "3-stage (CryGate → OOD → Ensemble)",
        "gate_method":    gate.method,
    }


@app.post("/predict", response_model=PredictionResult, summary="Classify a .wav file")
async def predict_audio(
    file:    UploadFile = File(..., description="WAV audio file"),
    dataset: str        = Query("dataset1", description="Model set: dataset1 or dataset2"),
    device:  Optional[str] = Query(None, description="Force device: cuda or cpu"),
    yamnet_threshold: float = Query(0.12, description="Cry gate threshold (0–1)"),
):
    """
    Upload a .wav file and get infant cry classification.

    Returns:
    - **is_cry**: whether the audio was identified as a baby cry
    - **prediction**: cry type (belly_pain / burping / discomfort / hungry / tired)
    - **confidence**: ensemble confidence 0–100%
    - **gate_score**: cry gate score (0–1)
    - **stage_blocked**: which stage rejected it (null if accepted)
    """
    allowed = ('.wav', '.mp3', '.ogg', '.flac', '.m4a')
    if not any(file.filename.lower().endswith(ext) for ext in allowed):
        raise HTTPException(status_code=400, detail=f"Unsupported format. Use: {allowed}")

    suffix = Path(file.filename).suffix.lower() or '.wav'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = _predict(
            tmp_path, dataset=dataset, device=device,
            yamnet_threshold=yamnet_threshold,
        )
        result['audio_path'] = file.filename
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Model error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    finally:
        os.unlink(tmp_path)


@app.post("/predict/batch", summary="Classify multiple .wav files")
async def predict_batch_audio(
    files:   List[UploadFile] = File(..., description="Multiple WAV audio files"),
    dataset: str              = Query("dataset1"),
    device:  Optional[str]    = Query(None),
):
    """
    Upload multiple .wav files for batch inference.
    Returns a list of prediction results, one per file.
    Maximum 20 files per request.
    """
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per batch request")

    tmp_paths, names = [], []
    for f in files:
        allowed = ('.wav', '.mp3', '.ogg', '.flac')
        if not any(f.filename.lower().endswith(ext) for ext in allowed):
            raise HTTPException(status_code=400, detail=f"{f.filename}: unsupported format")
        suffix = Path(f.filename).suffix.lower() or '.wav'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            shutil.copyfileobj(f.file, tmp)
            tmp_paths.append(tmp.name)
            names.append(f.filename)

    try:
        results = _predict_batch(tmp_paths, dataset=dataset, device=device)
        for r, name in zip(results, names):
            r['audio_path'] = name
        return results
    finally:
        for p in tmp_paths:
            try: os.unlink(p)
            except: pass
