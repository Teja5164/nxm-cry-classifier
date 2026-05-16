"""
stage_calibration.py — Temperature Scaling + Ensemble Weight Optimization
==========================================================================
Phase 4 of the production pipeline.

What this script does:
  1. Loads all augmented models (falls back to originals if augmented missing)
  2. Runs inference on the full validation set
  3. Learns optimal temperature T via NLL minimization (temperature scaling)
  4. Learns optimal per-model ensemble weights via logistic regression stacking
  5. Saves calibration artifacts:
       - models/calibration/temperature.json      ← T value per model
       - models/calibration/ensemble_weights.json ← per-model weights
       - models/calibration/calibration_report.json

Usage:
    python training/scripts/stage_calibration.py
    python training/scripts/stage_calibration.py --dataset dataset1
    python training/scripts/stage_calibration.py --no-gpu
"""

import os, sys, json, argparse, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_ROOT    = _HERE.parent.parent          # ML_pipeline/
_INF_DIR = _ROOT / 'inference'
sys.path.insert(0, str(_INF_DIR))
sys.path.insert(0, str(_ROOT / 'training' / 'scripts'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
from scipy.special import softmax as scipy_softmax

from model_loader import load_models
from preprocess   import preprocess_audio, extract_mel, normalize_mel, load_norm_stats

# ── Paths ─────────────────────────────────────────────────────────────────────
FEAT_ROOT  = _ROOT / 'training' / 'features'
MODEL_ROOT = _ROOT / 'models'
CALIB_DIR  = MODEL_ROOT / 'calibration'
LABELS_F   = _INF_DIR / 'labels.json'

CLASSES = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_val_files(dataset: str):
    """Load validation split file paths and labels from features directory."""
    split_file = FEAT_ROOT / dataset / 'val_paths.json'
    if not split_file.exists():
        # Fall back: scan dataset folder directly for known splits
        raise FileNotFoundError(
            f"val_paths.json not found at {split_file}\n"
            "Trying alternative: scanning dataset folders..."
        )
    with open(split_file) as f:
        data = json.load(f)
    return data['paths'], data['labels']


def scan_val_from_features(dataset: str):
    """
    Reconstruct the same 80/20 stratified val split used during augmented retraining.
    Uses mel_specs.npy + labels.npy with random_state=42 — matches training exactly.
    """
    from sklearn.model_selection import train_test_split

    feat_dir  = FEAT_ROOT / dataset
    X_all_path = feat_dir / 'mel_specs.npy'
    y_all_path = feat_dir / 'labels.npy'

    if not X_all_path.exists():
        raise FileNotFoundError(f"mel_specs.npy not found in {feat_dir}")

    print(f"  Loading mel_specs.npy from {feat_dir}...")
    X_all = np.load(str(X_all_path))   # (N, 1, 128, 431)
    y_all = np.load(str(y_all_path))   # (N,) int

    # Reproduce the exact same split as stage_augmented_retrain.py
    idx = np.arange(len(X_all))
    _, val_idx = train_test_split(
        idx, test_size=0.2, stratify=y_all, random_state=42
    )
    X_val = X_all[val_idx]   # (800, 1, 128, 431)
    y_val = y_all[val_idx]   # (800,)

    print(f"  Val split: {len(y_val)} samples (20% stratified, seed=42)")
    return X_val, y_val


def get_logits_from_features(models_entries, X: np.ndarray, device, dataset: str):
    """
    Run all models on precomputed feature array X.
    X shape: (N, 1, 128, 431) — already normalized from training pipeline.
    Returns: logits_per_model shape (n_models, N, n_classes)
    """
    all_logits = []
    for entry in models_entries:
        entry.model.eval()
        logits_list = []
        batch_size = 32
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]                          # (B, 1, 128, 431) float32
            t = torch.FloatTensor(batch).to(device)
            with torch.no_grad():
                logits = entry.model(t).cpu().numpy()          # (B, 5)
            logits_list.append(logits)
        all_logits.append(np.concatenate(logits_list, axis=0))  # (N, 5)
        print(f"    {entry.name:15s}: {len(all_logits[-1])} samples processed")

    return np.stack(all_logits, axis=0)  # (n_models, N, 5)


# ─────────────────────────────────────────────────────────────────────────────
# Temperature Scaling
# ─────────────────────────────────────────────────────────────────────────────
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature.clamp(min=0.1, max=10.0)

    def fit(self, logits_np: np.ndarray, labels_np: np.ndarray, verbose=True):
        """
        Optimize temperature T to minimize NLL on validation set.
        logits_np: (N, n_classes) numpy array
        labels_np: (N,) int numpy array
        """
        logits = torch.FloatTensor(logits_np)
        labels = torch.LongTensor(labels_np)

        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=100)
        nll_criterion = nn.CrossEntropyLoss()

        def eval_fn():
            optimizer.zero_grad()
            scaled = self.forward(logits)
            loss   = nll_criterion(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(eval_fn)
        T = float(self.temperature.item())

        # Compute ECE before/after
        probs_before = scipy_softmax(logits_np, axis=1)
        probs_after  = scipy_softmax(logits_np / T, axis=1)
        ece_before   = _compute_ece(probs_before, labels_np)
        ece_after    = _compute_ece(probs_after,  labels_np)

        if verbose:
            print(f"    Temperature T = {T:.4f}")
            print(f"    ECE before:   {ece_before:.4f}")
            print(f"    ECE after:    {ece_after:.4f}")

        return T, ece_before, ece_after


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins=15) -> float:
    """Expected Calibration Error."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies  = (predictions == labels).astype(float)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask   = (confidences >= lo) & (confidences < hi)
        if mask.sum() > 0:
            ece += mask.sum() * abs(accuracies[mask].mean() - confidences[mask].mean())
    return ece / len(labels)


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble Weight Optimization
# ─────────────────────────────────────────────────────────────────────────────
def optimize_ensemble_weights(logits_per_model: np.ndarray, temperatures: list,
                               labels: np.ndarray, model_names: list, verbose=True):
    """
    Learn optimal ensemble weights using stacking (logistic regression meta-learner).

    Returns:
        weights: np.array of shape (n_models,) summing to 1.0
        val_acc_weighted: accuracy with learned weights
        val_acc_uniform:  accuracy with uniform weights (baseline)
    """
    n_models, N, n_classes = logits_per_model.shape

    # Apply per-model temperature scaling
    calibrated_probs = []
    for i, (logits, T) in enumerate(zip(logits_per_model, temperatures)):
        probs = scipy_softmax(logits / T, axis=1)
        calibrated_probs.append(probs)

    # Uniform ensemble baseline
    uniform_probs = np.mean(calibrated_probs, axis=0)
    uniform_acc   = (uniform_probs.argmax(axis=1) == labels).mean() * 100

    # Val-accuracy-weighted baseline
    val_accs = np.array([
        (scipy_softmax(logits_per_model[i] / temperatures[i], axis=1).argmax(axis=1) == labels).mean()
        for i in range(n_models)
    ])
    va_weights = val_accs / val_accs.sum()
    va_probs   = sum(w * p for w, p in zip(va_weights, calibrated_probs))
    va_acc     = (va_probs.argmax(axis=1) == labels).mean() * 100

    # Stacking: grid search over Dirichlet-like weight combinations
    best_acc    = uniform_acc
    best_weights = np.ones(n_models) / n_models

    # Grid search over weight space (efficient for 4 models)
    from itertools import product
    candidates = np.arange(0.0, 1.05, 0.05)
    best_acc_grid = uniform_acc

    # For 4 models, use a smarter search: optimize via scipy
    from scipy.optimize import minimize

    def neg_acc(w):
        w = np.array(w)
        w = np.abs(w) / (np.abs(w).sum() + 1e-9)
        ens = sum(wt * p for wt, p in zip(w, calibrated_probs))
        acc = (ens.argmax(axis=1) == labels).mean()
        return -acc

    # Multiple random starts
    best_res = None
    for _ in range(20):
        w0 = np.random.dirichlet(np.ones(n_models))
        res = minimize(neg_acc, w0, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-5})
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    raw_w     = np.abs(best_res.x)
    opt_weights = raw_w / raw_w.sum()
    opt_probs   = sum(w * p for w, p in zip(opt_weights, calibrated_probs))
    opt_acc     = (opt_probs.argmax(axis=1) == labels).mean() * 100

    if verbose:
        print(f"\n    Ensemble weight optimization:")
        print(f"    Uniform weights:       {uniform_acc:.2f}%")
        print(f"    Val-acc weights:       {va_acc:.2f}%")
        print(f"    Optimized weights:     {opt_acc:.2f}%")
        print(f"\n    Optimal weights:")
        for name, w, va in zip(model_names, opt_weights, val_accs):
            print(f"      {name:15s}: {w:.4f}  (val_acc={va*100:.2f}%)")

    return opt_weights.tolist(), float(opt_acc), float(uniform_acc), float(va_acc)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Phase 4: Calibration + Ensemble Weights')
    parser.add_argument('--dataset',  default='dataset1')
    parser.add_argument('--no-gpu',   action='store_true')
    args = parser.parse_args()

    device = torch.device('cpu' if args.no_gpu else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))

    print("=" * 65)
    print("  Phase 4: Temperature Scaling + Ensemble Weight Optimization")
    print("=" * 65)
    print(f"  Dataset: {args.dataset}  |  Device: {device}")

    CALIB_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n  Loading models...")
    entries = load_models(args.dataset, device=device)
    model_names = [e.name for e in entries]
    print(f"  Loaded {len(entries)} models: {model_names}")

    # ── Load validation features ──────────────────────────────────────────────
    print("\n  Loading validation features...")
    try:
        X_val, y_val = scan_val_from_features(args.dataset)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    # ── Get logits from all models ────────────────────────────────────────────
    print("\n  Running models on validation set...")
    logits_per_model = get_logits_from_features(entries, X_val, device, args.dataset)
    # shape: (n_models, N, 5)

    # ── Temperature scaling per model ─────────────────────────────────────────
    print("\n  Fitting temperature scaling per model...")
    temperatures = []
    temp_report  = {}

    for i, (name, logits) in enumerate(zip(model_names, logits_per_model)):
        print(f"\n  [{i+1}/{len(model_names)}] {name}")
        scaler = TemperatureScaler()
        T, ece_before, ece_after = scaler.fit(logits, y_val)
        temperatures.append(T)
        temp_report[name] = {
            'temperature': round(T, 6),
            'ece_before':  round(float(ece_before), 6),
            'ece_after':   round(float(ece_after), 6),
            'improvement': round(float(ece_before - ece_after), 6),
        }

    # ── Ensemble weight optimization ──────────────────────────────────────────
    print("\n  Optimizing ensemble weights...")
    opt_weights, opt_acc, uniform_acc, va_acc = optimize_ensemble_weights(
        logits_per_model, temperatures, y_val, model_names
    )

    # ── Per-class accuracy with optimized ensemble ────────────────────────────
    from scipy.special import softmax as sp_softmax
    final_probs = sum(
        w * sp_softmax(logits_per_model[i] / temperatures[i], axis=1)
        for i, w in enumerate(opt_weights)
    )
    final_preds = final_probs.argmax(axis=1)
    per_class_acc = {}
    for ci, cls in enumerate(CLASSES):
        mask = y_val == ci
        if mask.sum() > 0:
            per_class_acc[cls] = round(float((final_preds[mask] == ci).mean() * 100), 2)

    # ── Save calibration artifacts ────────────────────────────────────────────
    temp_out = {
        'dataset':      args.dataset,
        'model_names':  model_names,
        'temperatures': {name: temp_report[name]['temperature']
                         for name in model_names},
        'per_model':    temp_report,
    }

    weights_out = {
        'dataset':        args.dataset,
        'model_names':    model_names,
        'weights':        {name: round(w, 6) for name, w in zip(model_names, opt_weights)},
        'weights_list':   [round(w, 6) for w in opt_weights],
        'val_acc_uniform':   round(uniform_acc, 4),
        'val_acc_va_weighted': round(va_acc, 4),
        'val_acc_optimized':  round(opt_acc, 4),
    }

    report = {
        'dataset':           args.dataset,
        'n_models':          len(entries),
        'model_names':       model_names,
        'n_val_samples':     int(len(y_val)),
        'temperature_scaling': temp_out,
        'ensemble_weights':    weights_out,
        'per_class_accuracy':  per_class_acc,
        'overall_val_acc':     round(opt_acc, 4),
    }

    temp_file    = CALIB_DIR / 'temperature.json'
    weights_file = CALIB_DIR / 'ensemble_weights.json'
    report_file  = CALIB_DIR / 'calibration_report.json'

    with open(temp_file,    'w') as f: json.dump(temp_out,    f, indent=2)
    with open(weights_file, 'w') as f: json.dump(weights_out, f, indent=2)
    with open(report_file,  'w') as f: json.dump(report,      f, indent=2)

    print("\n" + "=" * 65)
    print("  CALIBRATION COMPLETE")
    print("=" * 65)
    print(f"\n  Validation Accuracy:")
    print(f"    Uniform weights:    {uniform_acc:.2f}%")
    print(f"    Val-acc weights:    {va_acc:.2f}%")
    print(f"    Optimized weights:  {opt_acc:.2f}%  ← production")
    print(f"\n  Per-class accuracy (optimized ensemble):")
    for cls, acc in per_class_acc.items():
        bar = '█' * int(acc // 5)
        print(f"    {cls:12s}: {acc:5.1f}%  {bar}")
    print(f"\n  Artifacts saved:")
    print(f"    {temp_file}")
    print(f"    {weights_file}")
    print(f"    {report_file}")


if __name__ == '__main__':
    main()
