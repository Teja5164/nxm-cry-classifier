"""
app.py — FastAPI Backend for Infant Cry Classifier
===================================================
Provides REST endpoints for the ML inference pipeline.

Run (development):
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

Run (production):
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 2

API Endpoints:
    GET  /            — health check
    GET  /info        — model info and class labels
    POST /predict     — single audio file inference
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

app = FastAPI(
    title        = "Infant Cry Classifier API",
    description  = "5-class infant cry audio classification (belly_pain, burping, discomfort, hungry, tired)",
    version      = "1.0.0",
    docs_url     = "/docs",
    redoc_url    = "/redoc",
)

# Allow all origins for development — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Response models ───────────────────────────────────────────────────────────
class PredictionResult(BaseModel):
    prediction:    str
    confidence:    float
    probabilities: dict
    reliability:   str
    n_models:      int
    device:        str
    dataset:       str
    audio_path:    str


class HealthResponse(BaseModel):
    status:  str
    version: str
    classes: List[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", response_model=HealthResponse, summary="Health check")
def health():
    return {"status": "ok", "version": "1.0.0", "classes": CLASSES}


@app.get("/info", summary="Model and class info")
def info():
    labels_path = _INF / 'labels.json'
    with open(labels_path) as f:
        labels = json.load(f)
    return {
        "classes":        CLASSES,
        "class_info":     labels.get("class_descriptions", {}),
        "confusion_pairs": labels.get("confusion_pairs", []),
        "n_models":       4,
        "architecture":   ["BaselineCNN", "CNN+BiLSTM", "CryNet(Transformer)", "SE-ResNet"],
        "dataset":        "dataset1",
    }


@app.post("/predict", response_model=PredictionResult, summary="Classify a single .wav file")
async def predict_audio(
    file:    UploadFile = File(..., description="WAV audio file"),
    dataset: str        = Query("dataset1", description="Model set: dataset1 or dataset2"),
    device:  Optional[str] = Query(None, description="Force device: cuda or cpu"),
):
    """
    Upload a .wav file and get infant cry classification.

    - **file**: WAV audio file (any sample rate, mono or stereo, ≥1s)
    - **dataset**: Which trained models to use
    - **device**: Force CPU or CUDA (default: auto-detect GPU)
    """
    if not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = _predict(tmp_path, dataset=dataset, device=device)
        result['audio_path'] = file.filename   # replace temp path with original name
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
    """
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per batch request")

    tmp_paths = []
    names     = []
    for f in files:
        if not f.filename.endswith('.wav'):
            raise HTTPException(status_code=400, detail=f"{f.filename}: only .wav supported")
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
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
            os.unlink(p)
