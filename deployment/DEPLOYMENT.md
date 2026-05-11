# Deployment Guide — Infant Cry Classifier
==========================================

## Prerequisites
- Python 3.10+
- PyTorch ≥ 2.0 (with CUDA 12.4 for GPU)
- Trained model weights in `models/`
- `training/features/dataset1/norm_stats.json` present

---

## 1. Local Development Server

```bash
# Install dependencies
pip install fastapi uvicorn python-multipart

# Start API server
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# Test it
curl http://localhost:8000/
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

The API is REST+JSON. From a mobile app:

```http
POST /predict
Content-Type: multipart/form-data
Body: file=<recorded_wav>

Response:
{
  "prediction": "hungry",
  "confidence": 87.3,
  "probabilities": { "hungry": 87.3, "belly_pain": 6.1, ... },
  "reliability": "HIGH"
}
```

Recommended: record 5–10s WAV at 22050 Hz mono, send to backend.

---

## 7. Model Weight Distribution

Model `.pth` files are **NOT included in the repository** (105 MB total).
Options for deployment:
- Distribute via private release assets on GitHub
- Upload to Google Drive / S3 and download on first launch
- Bundle with app package (if size allows)

See `models/` directory for the expected file structure.

---

## 8. API Endpoint Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/info` | Class info and model details |
| POST | `/predict?dataset=dataset1` | Single WAV inference |
| POST | `/predict/batch?dataset=dataset1` | Batch WAV inference (max 20) |

Interactive docs: http://localhost:8000/docs
