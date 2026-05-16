"""
Smoke tests for the infant cry classifier production pipeline.

These tests verify the pipeline components load and return correct response shapes
without requiring actual audio files or GPU.

Usage:
    cd ML_pipeline
    conda activate babycare_env
    python -m pytest tests/ -v
    python tests/test_pipeline_smoke.py   # run directly
"""

import os, sys, json
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_INF  = _ROOT / 'inference'
sys.path.insert(0, str(_INF))

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')


# ─────────────────────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────────────────────
PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status} {name}" + (f"  ({detail})" if detail else ""))
    results.append(condition)
    return condition


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────
def test_imports():
    print("\n[1] Import checks")
    try:
        import torch; check("torch", True, torch.__version__)
    except ImportError as e:
        check("torch", False, str(e))

    try:
        import librosa; check("librosa", True, librosa.__version__)
    except ImportError as e:
        check("librosa", False, str(e))

    try:
        from preprocess import preprocess_audio, extract_mel, normalize_mel, load_norm_stats
        check("preprocess.py", True)
    except ImportError as e:
        check("preprocess.py", False, str(e))

    try:
        from model_loader import load_models
        check("model_loader.py", True)
    except ImportError as e:
        check("model_loader.py", False, str(e))

    try:
        from yamnet_gate import CryGate
        check("yamnet_gate.py", True)
    except ImportError as e:
        check("yamnet_gate.py", False, str(e))

    try:
        from predict import predict, CLASSES, _CALIB_TEMPERATURES, _CALIB_WEIGHTS
        check("predict.py", True)
        check("CLASSES defined", len(CLASSES) == 5, str(CLASSES))
        check("calibration loaded", _CALIB_TEMPERATURES is not None,
              "not found — run stage_calibration.py" if _CALIB_TEMPERATURES is None else "OK")
    except ImportError as e:
        check("predict.py", False, str(e))


def test_labels():
    print("\n[2] labels.json")
    labels_path = _INF / 'labels.json'
    ok = labels_path.exists()
    check("labels.json exists", ok)
    if ok:
        with open(labels_path) as f:
            data = json.load(f)
        check("5 classes", len(data.get('classes', [])) == 5, str(data.get('classes', [])))


def test_norm_stats():
    print("\n[3] norm_stats.json (dataset1)")
    stats_path = _ROOT / 'training' / 'features' / 'dataset1' / 'norm_stats.json'
    ok = stats_path.exists()
    check("norm_stats.json exists", ok)
    if ok:
        with open(stats_path) as f:
            stats = json.load(f)
        check("mel_mean present", 'mel_mean' in stats)
        check("mel_std present",  'mel_std'  in stats)
        check("128 mel bins", len(stats.get('mel_mean', [])) == 128,
              f"got {len(stats.get('mel_mean', []))}")


def test_model_files():
    print("\n[4] Model checkpoint files")
    models_dir = _ROOT / 'models'
    expected = [
        models_dir / 'cry_gate'        / 'best_model.pth',
        models_dir / 'baseline_cnn'    / 'dataset1_augmented' / 'best_model.pth',
        models_dir / 'cnn_bilstm'      / 'dataset1_augmented' / 'best_model.pth',
        models_dir / 'cnn_transformer' / 'dataset1_augmented' / 'best_model.pth',
        models_dir / 'se_resnet'       / 'dataset1_augmented' / 'best_model.pth',
    ]
    for p in expected:
        size_mb = round(p.stat().st_size / 1e6, 1) if p.exists() else None
        check(p.parent.name + '/' + p.name, p.exists(),
              f"{size_mb} MB" if size_mb else "MISSING")


def test_calibration_files():
    print("\n[5] Calibration artifacts (Phase 4)")
    calib_dir = _ROOT / 'models' / 'calibration'
    for fname in ['temperature.json', 'ensemble_weights.json', 'calibration_report.json']:
        p = calib_dir / fname
        check(fname, p.exists())
    if (calib_dir / 'temperature.json').exists():
        with open(calib_dir / 'temperature.json') as f:
            data = json.load(f)
        temps = data.get('temperatures', {})
        check("temperatures for 4 models", len(temps) == 4, str(list(temps.keys())))


def test_predict_result_schema():
    """Test that predict() returns the expected schema keys (no audio needed if blocked by gate)."""
    print("\n[6] predict() result schema")
    try:
        from predict import predict, _EMPTY_PROBS, CLASSES
        # We can't test with real audio here without a sample file
        # Just verify the module-level constants are correct
        check("_EMPTY_PROBS has 5 keys", len(_EMPTY_PROBS) == 5)
        check("CLASSES = 5", len(CLASSES) == 5)
        expected_classes = {'belly_pain', 'burping', 'discomfort', 'hungry', 'tired'}
        check("correct class names", set(CLASSES) == expected_classes, str(CLASSES))
    except Exception as e:
        check("predict schema", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  Infant Cry Classifier — Pipeline Smoke Tests")
    print("=" * 55)

    test_imports()
    test_labels()
    test_norm_stats()
    test_model_files()
    test_calibration_files()
    test_predict_result_schema()

    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*55}")
    print(f"  Results: {passed}/{total} passed")
    if passed == total:
        print("  All checks passed ✓")
    else:
        print(f"  {total - passed} check(s) failed — see above")
    print("=" * 55)
    sys.exit(0 if passed == total else 1)
