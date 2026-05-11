"""
Central path configuration for the ML_pipeline project.
PROJECT_ROOT auto-detects from this file's location — no manual edits needed
when the project is moved or cloned on a new machine.
"""
import sys
from pathlib import Path

# ── Project root — resolves automatically from this file's location ───────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Datasets ──────────────────────────────────────────────────────────────────
DATASETS_DIR      = PROJECT_ROOT / 'datasets'
DATASET1_DIR      = DATASETS_DIR / 'Dataset1'
DATASET2_DIR      = DATASETS_DIR / 'Dataset2'
LABELS_DIR        = DATASETS_DIR / 'labels'
RAW_CRY_DIR       = DATASETS_DIR / 'raw_cry'
PREPROCESSED_DIR  = DATASETS_DIR / 'preprocessed'
FEATURES_CSV_DIR  = DATASETS_DIR / 'features'

# ── Preprocessing ─────────────────────────────────────────────────────────────
PREPROCESSING_DIR = PROJECT_ROOT / 'preprocessing'
CLAP_MODEL_DIR    = PREPROCESSING_DIR / 'clap_model'

# ── Training ─────────────────────────────────────────────────────────────────
TRAINING_DIR      = PROJECT_ROOT / 'training'
FEAT_ROOT         = TRAINING_DIR / 'features'      # mel_specs.npy, norm_stats.json
TRAINING_SCRIPTS  = TRAINING_DIR / 'scripts'

# ── Models & Results ─────────────────────────────────────────────────────────
MODELS_DIR        = PROJECT_ROOT / 'models'
RESULTS_DIR       = PROJECT_ROOT / 'results'

# ── Inference ────────────────────────────────────────────────────────────────
INFERENCE_DIR     = PROJECT_ROOT / 'inference'

# ── Python interpreter — always resolves to the active environment ────────────
PYTHON_EXE        = sys.executable
