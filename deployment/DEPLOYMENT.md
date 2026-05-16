# Deployment Guide — Infant Cry Classifier (v2.0)
=================================================

## Pipeline Overview

The system uses a 3-stage pipeline:
1. **Cry Gate** — BinaryCryCNN rejects non-cry audio (98.56% accuracy)
2. **OOD Check** — Entropy + confidence thresholds reject uncertain predictions
3. **5-Class Ensemble** — Calibrated ensemble of 4 models (87.50% accuracy)

## Prerequisites
- Python 3.10+
- PyTorch ≥ 2.0 (CPU or GPU)
- Trained model weights in `models/` (see structure below)
- `training/features/dataset1/norm_stats.json` present
- `models/calibration/` artifacts (run `stage_calibration.py` once after training)

Required model files:
```
models/
├── cry_gate/best_model.pth
├── baseline_cnn/dataset1_augmented/best_model.pth
├── cnn_bilstm/dataset1_augmented/best_model.pth
├── cnn_transformer/dataset1_augmented/best_model.pth
├── se_resnet/dataset1_augmented/best_model.pth
└── calibration/
    ├── temperature.json
    ├── ensemble_weights.json
    └── calibration_report.json
```

---

## 1. Local Development Server

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# Test it
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/predict" -F "file=@cry.wav"
```

---

## 2. Docker (CPU)

```bash
docker build -t cry-classifier -f deployment/Dockerfile .

docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/training/features:/app/training/features:ro \
  cry-classifier
```

## 3. Docker (GPU)

```bash
docker run -p 8000:8000 --gpus all \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/training/features:/app/training/features:ro \
  cry-classifier
```

## 4. Docker Compose

```bash
cd deployment
docker compose up -d
```

---

## 5. Cloud Deployment (Render / Railway / Fly.io)

- Set `KMP_DUPLICATE_LIB_OK=TRUE` as environment variable
- Mount or upload `models/` and `training/features/` separately (large binary files)
- Use `uvicorn api.app:app --host 0.0.0.0 --port $PORT`

---

## 6. Android / iOS Integration

The API is REST+JSON. Recommended: record 5–10s audio at 22050 Hz mono.

```http
POST /predict
Content-Type: multipart/form-data
Body: file=<recorded_wav_or_mp3>
```

**Success Response (cry detected):**
```json
{
  "is_cry":        true,
  "prediction":    "hungry",
  "confidence":    87.3,
  "probabilities": { "hungry": 87.3, "belly_pain": 4.1, "burping": 3.2, "discomfort": 3.0, "tired": 2.4 },
  "reliability":   "HIGH",
  "entropy":       0.847,
  "n_models":      4,
  "gate_score":    0.921,
  "gate_method":   "custom_binary",
  "stage_blocked": null,
  "reason":        "",
  "device":        "cpu",
  "dataset":       "dataset1",
  "audio_path":    "cry.wav"
}
```

**Rejected Response (not a cry):**
```json
{
  "is_cry":        false,
  "prediction":    "not_a_cry",
  "confidence":    0.0,
  "stage_blocked": "cry_gate",
  "reason":        "Gate rejected: score=0.023 < threshold=0.34",
  "gate_score":    0.023,
  "gate_method":   "custom_binary"
}
```

**Rejected Response (OOD / ambiguous):**
```json
{
  "is_cry":        false,
  "prediction":    "not_a_cry",
  "stage_blocked": "ood_check",
  "reason":        "OOD: confidence=38.2% < 42% or entropy=1.42 > 1.35"
}
```

---

## 7. API Endpoint Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` or `/health` | Health check with gate status |
| GET | `/info` | Class info and model details |
| POST | `/predict?dataset=dataset1` | Single audio file inference (WAV, MP3, OGG, FLAC, M4A) |
| POST | `/predict/batch?dataset=dataset1` | Batch inference (max 20 files) |

Interactive docs: http://localhost:8000/docs

---

## 8. Model Weight Distribution

Model `.pth` files are **NOT included in the repository** (~90 MB total).  
Options for deployment:
- Distribute via private release assets on GitHub
- Upload to Google Drive / S3 and download on first launch
- Bundle with app package (if size allows)

See `models/` directory for the expected file structure.
