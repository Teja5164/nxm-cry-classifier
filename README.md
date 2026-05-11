# Infant Cry Classifier — ML Pipeline

![Python](https://img.shields.io/badge/python-3.13.5-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.6.0%2Bcu124-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Accuracy](https://img.shields.io/badge/ensemble%20accuracy-86.33%25-brightgreen)

A 5-class infant cry classification system for a baby care mobile app.  
Classifies audio into: **belly_pain**, **burping**, **discomfort**, **hungry**, **tired**.

> **Note:** Model weights (`.pth`), training features (`.npy`), and audio datasets (`.wav`) are excluded from this repository due to size. See [Setup](#quick-start) for how to obtain or retrain them.

---

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Baseline CNN | 82.67% | — |
| CNN + BiLSTM | 85.00% | — |
| CryNet (CNN + Transformer) | 81.83% | — |
| SE-ResNet | 83.67% | — |
| **Ensemble (Dataset1)** | **86.33%** | **86.24%** |

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
│   └── features/           # .npy mel/mfcc arrays + norm_stats.json
├── inference/              # Production inference entry point
│   └── stage7_inference.py
├── models/                 # Trained .pth checkpoints
│   ├── baseline_cnn/
│   ├── cnn_bilstm/
│   ├── cnn_transformer/
│   └── se_resnet/
├── results/                # Metrics, plots, confusion matrices
├── configs/                # Centralized path config (paths.py)
├── deployment/             # API / mobile deployment (future)
├── api/                    # Backend REST API (future)
├── utils/                  # Shared utilities (future)
├── tests/                  # Unit and integration tests (future)
├── requirements.txt
├── environment.yml
└── .gitignore
```

---

## Quick Start

### 1. Setup Environment

```bash
conda env create -f environment.yml
conda activate cry_mlpipeline
```

Or manually:
```bash
conda create -n cry_mlpipeline python=3.13.5
conda activate cry_mlpipeline
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Run Inference

```bash
python inference/stage7_inference.py --audio path/to/cry.wav --dataset dataset1
```

With verbose output:
```bash
python inference/stage7_inference.py --audio cry.wav --dataset dataset1 --verbose
```

**Example output:**
```
=======================================================
Infant Cry Classifier
=======================================================
Audio  : cry.wav
Dataset: dataset1
Device : cuda

Step 3: Loading models...
  Loaded 4 models: ['BaselineCNN', 'CNN+BiLSTM', 'CryNet', 'SE-ResNet']
Step 4: Running ensemble inference...

=======================================================
  PREDICTION : HUNGRY
  CONFIDENCE : 92.4%
=======================================================
Reliability: HIGH confidence
```

---

## Pipeline Stages

| Stage | Script | Purpose |
|---|---|---|
| 1 | `preprocessing/scripts/01_download_cry.py` | Download cry audio |
| 2 | `preprocessing/scripts/02_preprocess.py` | Resample + normalize |
| 3 | `preprocessing/scripts/03_extract_features.py` | Extract CSV features |
| 4 | `preprocessing/scripts/04_clap_classify.py` | CLAP zero-shot labeling |
| 5 | `preprocessing/scripts/05_heuristic_classify.py` | Rule-based labeling |
| 6 | `preprocessing/scripts/06a_build_dataset1.py` | Build balanced dataset |
| 7 | `training/scripts/stage1_extract_features.py` | Extract mel/mfcc arrays |
| 8 | `training/scripts/master_pipeline_dataset1.py` | Train all 4 models |
| 9 | `inference/stage7_inference.py` | Production inference |

---

## Environment

- **Python**: 3.13.5  
- **PyTorch**: 2.6.0+cu124  
- **GPU**: NVIDIA RTX 4050 Laptop GPU (CUDA 12.4)  
- **Conda env**: `cry_mlpipeline`

---

## Classes

| Label | Description |
|---|---|
| `belly_pain` | Intense, high-pitched rhythmic cry |
| `burping` | Short, sporadic cry with pauses |
| `discomfort` | Moderate-intensity whining cry |
| `hungry` | Rhythmic, building-intensity cry |
| `tired` | Weak, whimpering, low-energy cry |
