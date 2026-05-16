"""
predict.py — Production Inference Entrypoint
=============================================
Three-stage pipeline for robust infant cry classification.

Stage 1 — Cry Gate (yamnet_gate.py):
    Blocks all non-cry audio before it reaches the classifier.
    Uses custom binary CNN when trained, falls back to YAMNet.

Stage 2 — OOD Check:
    Entropy + confidence filter to catch borderline ambiguous samples.

Stage 3 — 5-class Ensemble:
    Weighted average of BaselineCNN, CNNBiLSTM, CryNet, SEResNet.

Programmatic usage:
    from inference.predict import predict, predict_batch

    result = predict('baby_cry.wav')
    print(result['prediction'])        # e.g. 'hungry'
    print(result['confidence'])        # e.g. 87.3
    print(result['is_cry'])            # True
    print(result['gate_score'])        # 0.73
    print(result['gate_method'])       # 'custom' or 'yamnet'

CLI usage:
    python predict.py --audio baby_cry.wav
    python predict.py --audio baby_cry.wav --verbose
    python predict.py --audio baby_cry.wav --device cpu
    python predict.py --batch cries_folder/

Output dict schema (cry detected):
    {
      'is_cry':        bool,           # True = passed gate
      'prediction':    str,            # class label e.g. 'hungry'
      'confidence':    float,          # ensemble confidence 0–100
      'probabilities': dict[str,float],# per-class % scores
      'reliability':   str,            # 'HIGH' / 'MEDIUM' / 'LOW'
      'entropy':       float,          # prediction entropy (lower = more certain)
      'n_models':      int,            # models used in ensemble
      'device':        str,
      'dataset':       str,
      'audio_path':    str,
      'gate_score':    float,          # Stage 1 gate score (0–1)
      'gate_method':   str,            # 'custom' | 'yamnet' | 'none'
      'stage_blocked': str | None,     # which stage blocked (None = passed all)
    }

Output dict schema (non-cry blocked):
    {
      'is_cry':        False,
      'prediction':    'not_a_cry',
      'confidence':    0.0,
      'probabilities': {class: 0.0 ...},
      'reliability':   'LOW',
      'stage_blocked': 'cry_gate' | 'ood_check',
      'reason':        str,            # human-readable block reason
      'gate_score':    float,
      'gate_method':   str,
      ...
    }
"""

import os, sys, json, warnings
from pathlib import Path
from typing import List, Optional, Union

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # prevent OpenMP crash with librosa+torch
warnings.filterwarnings('ignore')

# ── Ensure inference/ is importable from any working directory ────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np

# ── Internal imports ──────────────────────────────────────────────────────────
from preprocess import preprocess_audio, extract_mel, normalize_mel, load_norm_stats
from model_loader import load_models, ModelEntry
from yamnet_gate import get_gate, CryGate

# ── Project paths ─────────────────────────────────────────────────────────────
_ROOT      = _HERE.parent               # ML_pipeline/
FEAT_ROOT  = str(_ROOT / 'training' / 'features')
LABELS_F   = str(_HERE / 'labels.json')

# ── Class labels ──────────────────────────────────────────────────────────────
with open(LABELS_F, 'r') as _f:
    _LABEL_DATA = json.load(_f)
CLASSES = _LABEL_DATA['classes']        # ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']


# ─────────────────────────────────────────────────────────────────────────────
# Calibration artifacts (Phase 4)
# ─────────────────────────────────────────────────────────────────────────────

def _load_calibration(dataset: str = 'dataset1'):
    """
    Load temperature scaling and ensemble weights from calibration artifacts.
    Returns (temperatures_dict, weights_dict) or (None, None) if not found.
    """
    calib_dir = _ROOT / 'models' / 'calibration'
    temp_f    = calib_dir / 'temperature.json'
    wt_f      = calib_dir / 'ensemble_weights.json'
    if temp_f.exists() and wt_f.exists():
        with open(temp_f)  as f: temp_data = json.load(f)
        with open(wt_f)    as f: wt_data   = json.load(f)
        temps   = temp_data.get('temperatures', {})   # {name: T}
        weights = wt_data.get('weights', {})           # {name: w}
        return temps, weights
    return None, None

# Cache calibration at import time (None if not yet run)
_CALIB_TEMPERATURES, _CALIB_WEIGHTS = _load_calibration()


# ─────────────────────────────────────────────────────────────────────────────
# Core predict function
# ─────────────────────────────────────────────────────────────────────────────

# OOD thresholds (tune if needed)
CONF_THRESHOLD    = 0.42   # minimum per-class confidence to accept
ENTROPY_THRESHOLD = 1.35   # maximum entropy to accept (5-class max = ln5 ≈ 1.609)

_EMPTY_PROBS = {c: 0.0 for c in ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']}


def predict(
    audio_path: Union[str, Path],
    dataset:    str = 'dataset1',
    device=None,
    feat_root:  Optional[str] = None,
    model_root: Optional[str] = None,
    verbose:    bool = False,
    gate:       Optional[CryGate] = None,
    yamnet_threshold: float = 0.12,
    conf_threshold:   float = CONF_THRESHOLD,
    entropy_threshold: float = ENTROPY_THRESHOLD,
) -> dict:
    """
    Run three-stage infant cry classification on a single .wav file.

    Stage 1: Cry gate (custom binary CNN or YAMNet) — rejects non-cry audio.
    Stage 2: OOD check — rejects low-confidence / high-entropy outputs.
    Stage 3: 5-class ensemble — classifies cry type.

    Args:
        audio_path:        Path to .wav file.
        dataset:           'dataset1' or 'dataset2'.
        device:            'cuda', 'cpu', or None (auto-detect).
        feat_root:         Override for training/features directory.
        model_root:        Override for models directory.
        verbose:           Print per-model scores.
        gate:              Pre-loaded CryGate instance (reuse across calls).
        yamnet_threshold:  YAMNet cry score threshold (0–1, default 0.12).
        conf_threshold:    Minimum max-class confidence for OOD (default 0.42).
        entropy_threshold: Maximum entropy for OOD (default 1.35).

    Returns:
        result dict — see module docstring for full schema.
    """
    import torch
    import torch.nn.functional as F

    audio_path = str(audio_path)
    feats      = feat_root or FEAT_ROOT

    # ── Device setup ──────────────────────────────────────────────────────────
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    _base_result = {
        'is_cry':        False,
        'prediction':    'not_a_cry',
        'confidence':    0.0,
        'probabilities': dict(_EMPTY_PROBS),
        'reliability':   'LOW',
        'entropy':       0.0,
        'n_models':      0,
        'device':        str(device),
        'dataset':       dataset,
        'audio_path':    os.path.abspath(audio_path),
        'gate_score':    0.0,
        'gate_method':   'none',
        'stage_blocked': None,
        'reason':        '',
    }

    # ── Stage 1: Cry Gate ─────────────────────────────────────────────────────
    try:
        _gate = gate or get_gate(device=device, yamnet_threshold=yamnet_threshold)
        gate_result = _gate.is_cry(audio_path)
        _base_result['gate_score']  = gate_result['score']
        _base_result['gate_method'] = gate_result['method']

        if gate_result['is_cry'] is False:
            # Definitively rejected — not a cry
            _base_result['stage_blocked'] = 'cry_gate'
            _base_result['reason']        = gate_result['reason']
            if verbose:
                print(f"  [Stage 1 BLOCKED] {gate_result['reason']}")
            return _base_result

        if verbose and gate_result['is_cry'] is not None:
            print(f"  [Stage 1 PASSED]  gate_score={gate_result['score']:.3f}"
                  f"  method={gate_result['method']}")
    except Exception as e:
        if verbose:
            print(f"  [Stage 1 ERROR]   {e}  → passing through")

    # ── Preprocess audio ─────────────────────────────────────────────────────
    audio    = preprocess_audio(audio_path)
    log_mel  = extract_mel(audio)
    stats    = load_norm_stats(feats, dataset)
    norm_mel = normalize_mel(log_mel, stats)
    x = torch.FloatTensor(norm_mel).unsqueeze(0).unsqueeze(0).to(device)

    # ── Load models & run ensemble ────────────────────────────────────────────
    entries = load_models(dataset, device=device, model_root=model_root)

    all_probs = []
    with torch.no_grad():
        for entry in entries:
            raw_logits = entry.model(x).cpu()
            # Apply temperature scaling if calibration exists
            T = _CALIB_TEMPERATURES.get(entry.name, 1.0) if _CALIB_TEMPERATURES else 1.0
            scaled_logits = raw_logits / T
            probs = F.softmax(scaled_logits, dim=1).numpy()[0]
            all_probs.append((probs, entry.val_acc, entry.name))
            if verbose:
                top_cls = CLASSES[probs.argmax()]
                print(f"  {entry.name:15s} → {top_cls:12s}  ({probs.max()*100:.1f}%)  T={T:.3f}")

    # Use learned ensemble weights if available, else fall back to val-acc weights
    if _CALIB_WEIGHTS:
        raw_w = np.array([_CALIB_WEIGHTS.get(name, 1.0) for _, _, name in all_probs],
                         dtype=np.float64)
        weights = raw_w / raw_w.sum() if raw_w.sum() > 0 else np.ones(len(all_probs)) / len(all_probs)
    else:
        va_arr  = np.array([va for _, va, _ in all_probs])
        weights = va_arr / va_arr.sum() if va_arr.sum() > 0 else \
                  np.ones(len(all_probs)) / len(all_probs)

    ensemble_probs = sum(w * p for (p, _, _), w in zip(all_probs, weights))
    ensemble_probs = ensemble_probs / ensemble_probs.sum()

    # ── Stage 2: OOD check ────────────────────────────────────────────────────
    max_conf = float(ensemble_probs.max())
    entropy  = float(-np.sum(ensemble_probs * np.log(ensemble_probs + 1e-9)))

    if max_conf < conf_threshold or entropy > entropy_threshold:
        _base_result['stage_blocked'] = 'ood_check'
        _base_result['entropy']       = round(entropy, 4)
        _base_result['reason']        = (
            f"OOD: confidence={max_conf*100:.1f}% < {conf_threshold*100:.0f}%"
            f" or entropy={entropy:.3f} > {entropy_threshold:.3f}"
        )
        _base_result['probabilities'] = {
            CLASSES[i]: round(float(ensemble_probs[i] * 100), 2)
            for i in range(len(CLASSES))
        }
        if verbose:
            print(f"  [Stage 2 BLOCKED] {_base_result['reason']}")
        return _base_result

    if verbose:
        print(f"  [Stage 2 PASSED]  conf={max_conf*100:.1f}%  entropy={entropy:.3f}")

    # ── Stage 3: Classification result ───────────────────────────────────────
    pred_idx   = int(ensemble_probs.argmax())
    pred_cls   = CLASSES[pred_idx]
    confidence = float(ensemble_probs[pred_idx] * 100)

    reliability = 'HIGH' if confidence >= 80 else ('MEDIUM' if confidence >= 60 else 'LOW')

    return {
        'is_cry':        True,
        'prediction':    pred_cls,
        'confidence':    round(confidence, 2),
        'probabilities': {CLASSES[i]: round(float(ensemble_probs[i] * 100), 2)
                          for i in range(len(CLASSES))},
        'reliability':   reliability,
        'entropy':       round(entropy, 4),
        'n_models':      len(entries),
        'device':        str(device),
        'dataset':       dataset,
        'audio_path':    os.path.abspath(audio_path),
        'gate_score':    _base_result['gate_score'],
        'gate_method':   _base_result['gate_method'],
        'stage_blocked': None,
        'reason':        '',
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch inference
# ─────────────────────────────────────────────────────────────────────────────
def predict_batch(
    audio_paths: List[Union[str, Path]],
    dataset:    str = 'dataset1',
    device=None,
    feat_root:  Optional[str] = None,
    model_root: Optional[str] = None,
    yamnet_threshold: float = 0.12,
) -> List[dict]:
    """
    Run inference on multiple .wav files. Models and gate are loaded once and reused.

    Returns:
        List of result dicts (same schema as predict()), one per input file.
        Files that fail preprocessing get an 'error' key instead of 'prediction'.
    """
    import torch
    import torch.nn.functional as F

    if isinstance(audio_paths, (str, Path)) and Path(audio_paths).is_dir():
        audio_paths = sorted(Path(audio_paths).glob('**/*.wav'))

    feats  = feat_root or FEAT_ROOT

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    # Load gate and models once — calibration weights applied inside predict()
    _gate   = get_gate(device=device, yamnet_threshold=yamnet_threshold)
    entries = load_models(dataset, device=device, model_root=model_root)
    stats   = load_norm_stats(feats, dataset)

    results = []
    for path in audio_paths:
        path = str(path)
        try:
            result = predict(
                path, dataset=dataset, device=device,
                feat_root=feat_root, model_root=model_root,
                gate=_gate, yamnet_threshold=yamnet_threshold,
            )
            results.append(result)
        except Exception as e:
            results.append({
                'audio_path': os.path.abspath(path),
                'error':      str(e),
                'is_cry':     False,
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────
def _print_result(result: dict, verbose: bool = False):
    """Pretty-print a single prediction result to stdout."""
    if 'error' in result:
        print(f"\n❌  ERROR: {result['error']}")
        print(f"   File: {result['audio_path']}")
        return

    is_cry = result.get('is_cry', True)  # backward compat

    print(f"\n{'='*58}")
    print(f"  Infant Cry Classifier — Three-Stage Pipeline")
    print(f"{'='*58}")
    print(f"  Audio   : {result['audio_path']}")
    print(f"  Gate    : {result.get('gate_method','?')}  score={result.get('gate_score',0):.3f}")
    print(f"  Device  : {result['device']}  |  Models: {result['n_models']}")
    print(f"{'─'*58}")

    if not is_cry:
        blocked = result.get('stage_blocked', 'unknown')
        reason  = result.get('reason', '')
        print(f"  🚫  NOT A BABY CRY")
        print(f"  Blocked at : {blocked}")
        print(f"  Reason     : {reason}")
    else:
        print(f"  ✅  PREDICTION  : {result['prediction'].upper()}")
        print(f"     CONFIDENCE  : {result['confidence']:.1f}%")
        print(f"     RELIABILITY : {result['reliability']}")
        print(f"     ENTROPY     : {result.get('entropy', 'n/a')}")

    print(f"{'='*58}")

    if verbose and is_cry:
        print("\n  Per-class probabilities:")
        sorted_items = sorted(result['probabilities'].items(),
                              key=lambda kv: kv[1], reverse=True)
        for cls, pct in sorted_items:
            bar = '█' * int(pct / 100 * 30)
            print(f"    {cls:12s}: {pct:5.1f}%  {bar}")

    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Infant Cry Classifier — Production Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --audio cry.wav
  python predict.py --audio cry.wav --dataset dataset1 --verbose
  python predict.py --audio cry.wav --device cpu
  python predict.py --batch /path/to/audio/folder --dataset dataset1
  python predict.py --batch /path/to/folder --json > results.json
        """
    )
    parser.add_argument('--audio',   type=str, default=None,
                        help='Path to a single .wav file')
    parser.add_argument('--batch',   type=str, default=None,
                        help='Path to a directory of .wav files (batch inference)')
    parser.add_argument('--dataset', type=str, default='dataset1',
                        choices=['dataset1', 'dataset2'],
                        help='Which trained models to use (default: dataset1)')
    parser.add_argument('--device',  type=str, default=None,
                        choices=['cuda', 'cpu'],
                        help='Force device (default: auto-detect GPU)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show per-model scores and per-class probabilities')
    parser.add_argument('--json',    action='store_true',
                        help='Output results as JSON (for API/piping)')
    args = parser.parse_args()

    if args.audio is None and args.batch is None:
        parser.error("Provide either --audio <file> or --batch <directory>")

    if args.batch:
        # Batch mode
        batch_dir = Path(args.batch)
        if not batch_dir.is_dir():
            print(f"❌  Not a directory: {args.batch}"); sys.exit(1)

        print(f"Running batch inference on: {batch_dir}")
        results = predict_batch(batch_dir, dataset=args.dataset, device=args.device)

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for r in results:
                _print_result(r, verbose=args.verbose)
            ok  = sum(1 for r in results if 'prediction' in r)
            err = len(results) - ok
            print(f"Processed: {ok} OK  |  {err} errors  |  {len(results)} total")

    else:
        # Single file mode
        result = predict(
            args.audio,
            dataset=args.dataset,
            device=args.device,
            verbose=args.verbose,
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            _print_result(result, verbose=args.verbose)
