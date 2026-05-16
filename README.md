# Infant Cry Classifier — ML Pipeline

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Ensemble](https://img.shields.io/badge/ensemble%20accuracy-87.50%25-brightgreen)
![Gate](https://img.shields.io/badge/cry%20gate-98.56%25-blue)
![API](https://img.shields.io/badge/API-FastAPI%20v2-teal)

A production-grade 5-class infant cry classification system for a baby care mobile app.  
Classifies audio into: **belly_pain**, **burping**, **discomfort**, **hungry**, **tired**.

> **Note:** Model weights (`.pth`), training features (`.npy`), and audio datasets (`.wav`) are excluded from this repository due to size. See [Setup](#quick-start) for how to obtain or retrain them.

---

## Pipeline Architecture

The production pipeline uses **3 stages** to maximise real-world accuracy and reject non-cry sounds:

```
Audio Input
    │
    ▼
┌─────────────────────────────────┐
│  Stage 1 — Cry Gate             │  Custom BinaryCryCNN (98.56% accuracy)
│  Is this even a baby cry?       │  → Rejects non-cry sounds (music, speech, noise)
└─────────────┬───────────────────┘
              │ is_cry = True
              ▼
┌─────────────────────────────────┐
│  Stage 2 — OOD Check            │  Entropy ≤ 1.35 AND confidence ≥ 42%
│  Is the model confident enough? │  → Rejects unclear / ambiguous audio
└─────────────┬───────────────────┘
              │ accepted
              ▼
┌─────────────────────────────────┐
│  Stage 3 — 5-Class Ensemble     │  4 models + temperature calibration
│  What type of cry is this?      │  + learned ensemble weights
└─────────────────────────────────┘
              │
              ▼
         Prediction
```

---

## Results

### 5-Class Classifier (Dataset1 — 800 samples/class, 4000 total)

| Model | Original | Augmented | Δ |
|---|---|---|---|
| BaselineCNN | 84.67% | 84.12% | -0.55% |
| CNN + BiLSTM | 83.31% | 81.12% | -2.19% |
| CryNet (CNN + Transformer) | 83.14% | **83.38%** | +0.24% |
| SE-ResNet | 83.64% | **87.12%** | **+3.48%** 🔥 |
| **Calibrated Ensemble** | 86.33% | **87.50%** | **+1.17%** |

### Binary Cry Gate

| Metric | Value |
|---|---|
| Validation Accuracy | **98.56%** |
| Threshold (F1-optimized) | 0.34 |
| Trained on | 4000 cry + 2000 ESC-50 non-cry samples |

### Per-Class Accuracy (Production Ensemble)

| Class | Accuracy |
|---|---|
| `discomfort` | 96.2% |
| `tired` | 94.4% |
| `hungry` | 89.4% |
| `burping` | 81.9% |
| `belly_pain` | 75.6% |

---

## Project Structure

```
ML_pipeline/
├── datasets/               # Raw audio (.wav) + label CSVs
│   ├── Dataset1/           # 800 samples/class (4,000 total)
│   ├── Dataset2/           # 1500 samples/class (7,500 total)
│   └── labels/             # clap_labels.csv, final_labels.csv
├── preprocessing/          # Scripts for download → feature extraction
│   ├── scripts/
│   └── clap_model/         # LAION-CLAP model weights
├── training/               # Model training scripts + extracted features
│   ├── scripts/
│   │   ├── stage1_extract_features.py      # Extract mel/mfcc arrays
│   │   ├── master_pipeline_dataset1.py     # Train all 4 classifiers
│   │   ├── stage_binary_cry_gate.py        # Train binary cry gate (Phase 2)
│   │   ├── stage_augmented_retrain.py      # Noise-augmented retraining (Phase 3)
│   │   └── stage_calibration.py           # Temp scaling + ensemble weights (Phase 4)
│   └── features/           # .npy mel/mfcc arrays + norm_stats.json
├── inference/              # Production inference
│   ├── predict.py          # Main inference entry point (3-stage pipeline)
│   ├── yamnet_gate.py      # Binary cry gate (custom CNN + YAMNet fallback)
│   ├── model_loader.py     # Multi-model loader (auto-detects augmented checkpoints)
│   ├── preprocess.py       # Audio preprocessing (resample, mel, normalize)
│   ├── labels.json         # Class labels and descriptions
│   └── stage7_inference.py # CLI wrapper (calls predict.py)
├── models/                 # Trained .pth checkpoints (git-ignored, mount as volume)
│   ├── baseline_cnn/dataset1_augmented/best_model.pth
│   ├── cnn_bilstm/dataset1_augmented/best_model.pth
│   ├── cnn_transformer/dataset1_augmented/best_model.pth
│   ├── se_resnet/dataset1_augmented/best_model.pth
│   ├── cry_gate/best_model.pth             # Binary cry gate (98.56%)
│   └── calibration/                        # Phase 4 calibration artifacts
│       ├── temperature.json
│       ├── ensemble_weights.json
│       └── calibration_report.json
├── api/                    # FastAPI backend (v2.0.0)
│   └── app.py
├── results/                # Metrics, plots, confusion matrices
├── configs/                # Centralized path config (paths.py)
├── deployment/             # Docker + deployment docs
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── DEPLOYMENT.md
├── utils/
├── tests/
├── requirements.txt
├── environment.yml
└── .gitignore
```

---

## Quick Start

### 1. Setup Environment

```bash
# Clone the repo
git clone https://github.com/Teja5164/nxm-cry-classifier.git
cd nxm-cry-classifier

# Create conda environment
conda env create -f environment.yml
conda activate babycare_env
```

Or manually (any Python 3.10+ environment):
```bash
# CPU-only (works everywhere)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# GPU (CUDA 12.4)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Obtain Model Weights

Model weights are not included in the repository (large binary files).  
Place them at the expected paths under `models/` (see structure above),  
or retrain using the [Training Pipeline](#training-pipeline) below.

### 3. Run Inference (CLI)

```bash
# Single file
python inference/stage7_inference.py --audio path/to/cry.wav

# Verbose (shows per-model scores + gate info)
python inference/stage7_inference.py --audio cry.wav --verbose
```

**Example output:**
```
=====================================================
  Infant Cry Classifier v2 — 3-Stage Pipeline
=====================================================
  [Stage 1 PASSED]  gate_score=0.921  method=custom_binary
  [Stage 2 PASSED]  conf=87.3%  entropy=0.847
  PREDICTION : HUNGRY
  CONFIDENCE : 87.3%
  RELIABILITY: HIGH
=====================================================
```

### 4. Run the API Server

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
# Interactive docs: http://localhost:8000/docs
```

```bash
# Test
curl -X POST "http://localhost:8000/predict" -F "file=@cry.wav"
```

---

## Training Pipeline

Run these stages in order to build the full pipeline from scratch:

| Stage | Script | Purpose |
|---|---|---|
| 1 | `preprocessing/scripts/01_download_cry.py` | Download cry audio |
| 2 | `preprocessing/scripts/02_preprocess.py` | Resample + normalize |
| 3 | `preprocessing/scripts/03_extract_features.py` | Extract CSV features |
| 4 | `preprocessing/scripts/04_clap_classify.py` | CLAP zero-shot labeling |
| 5 | `preprocessing/scripts/05_heuristic_classify.py` | Rule-based labeling |
| 6 | `preprocessing/scripts/06a_build_dataset1.py` | Build balanced dataset |
| 7 | `training/scripts/stage1_extract_features.py` | Extract mel/mfcc arrays |
| 8 | `training/scripts/master_pipeline_dataset1.py` | Train 4 classifiers |
| 9 | `training/scripts/stage_binary_cry_gate.py` | Train binary cry gate |
| 10 | `training/scripts/stage_augmented_retrain.py` | Noise-augmented retraining |
| 11 | `training/scripts/stage_calibration.py` | Temperature scaling + ensemble weights |
| 12 | `inference/stage7_inference.py` | Verify production pipeline |

---

## API Response Schema

```json
POST /predict  →  200 OK
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

If rejected by the gate or OOD check:
```json
{
  "is_cry":        false,
  "prediction":    "not_a_cry",
  "stage_blocked": "cry_gate",
  "reason":        "Gate rejected: score=0.023 < threshold=0.34",
  "confidence":    0.0
}
```

---

## Environment

- **Python**: 3.10+  
- **PyTorch**: 2.x (CPU or CUDA)  
- **GPU**: NVIDIA RTX 4050 (CUDA 12.4) — optional, CPU works fine  
- **Conda env**: `babycare_env`

---

## Classes

| Label | Description |
|---|---|
| `belly_pain` | Intense, high-pitched rhythmic cry |
| `burping` | Short, sporadic cry with pauses |
| `discomfort` | Moderate-intensity whining cry |
| `hungry` | Rhythmic, building-intensity cry |
| `tired` | Weak, whimpering, low-energy cry |
