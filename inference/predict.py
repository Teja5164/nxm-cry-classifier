"""
predict.py — Production Inference Entrypoint
=============================================
Clean, modular inference pipeline for Infant Cry Classification.
Imports preprocessing and model loading from sibling modules.

Programmatic usage (from Python / API):
    from inference.predict import predict, predict_batch

    # Single file
    result = predict('baby_cry.wav', dataset='dataset1')
    print(result['prediction'])   # e.g. 'hungry'
    print(result['confidence'])   # e.g. 87.3

    # Batch inference
    results = predict_batch(['cry1.wav', 'cry2.wav'], dataset='dataset1')

CLI usage:
    python predict.py --audio baby_cry.wav --dataset dataset1
    python predict.py --audio baby_cry.wav --dataset dataset1 --verbose
    python predict.py --audio baby_cry.wav --dataset dataset1 --device cpu
    python predict.py --batch cries_folder/ --dataset dataset1

Output dict schema:
    {
      'prediction':    str,           # class label e.g. 'hungry'
      'confidence':    float,         # ensemble confidence 0–100
      'probabilities': dict[str,float], # per-class % scores
      'reliability':   str,           # 'HIGH' / 'MEDIUM' / 'LOW'
      'n_models':      int,           # number of models used in ensemble
      'device':        str,           # 'cuda' or 'cpu'
      'dataset':       str,           # 'dataset1' or 'dataset2'
      'audio_path':    str,           # resolved input path
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

# ── Project paths ─────────────────────────────────────────────────────────────
_ROOT      = _HERE.parent               # ML_pipeline/
FEAT_ROOT  = str(_ROOT / 'training' / 'features')
LABELS_F   = str(_HERE / 'labels.json')

# ── Class labels ──────────────────────────────────────────────────────────────
with open(LABELS_F, 'r') as _f:
    _LABEL_DATA = json.load(_f)
CLASSES = _LABEL_DATA['classes']        # ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']


# ─────────────────────────────────────────────────────────────────────────────
# Core predict function
# ─────────────────────────────────────────────────────────────────────────────
def predict(
    audio_path: Union[str, Path],
    dataset:    str = 'dataset1',
    device=None,
    feat_root:  Optional[str] = None,
    model_root: Optional[str] = None,
    verbose:    bool = False,
) -> dict:
    """
    Run end-to-end infant cry classification on a single .wav file.

    Args:
        audio_path: Path to .wav file (any length ≥ 1s; longer → center-crop to 10s).
        dataset:    Which trained model set to use: 'dataset1' or 'dataset2'.
        device:     'cuda', 'cpu', or None (auto-detect).
        feat_root:  Override for training/features directory.
        model_root: Override for models directory.
        verbose:    If True, print per-model and per-class scores.

    Returns:
        result dict — see module docstring for full schema.

    Raises:
        FileNotFoundError: If audio file or norm_stats.json are missing.
        RuntimeError:      If no trained models are found.
        ValueError:        If audio file is empty or corrupted.
    """
    import torch

    audio_path = str(audio_path)
    feats      = feat_root or FEAT_ROOT

    # ── Device setup ──────────────────────────────────────────────────────────
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    # ── Step 1: Preprocess audio ──────────────────────────────────────────────
    audio = preprocess_audio(audio_path)

    # ── Step 2: Extract mel spectrogram ───────────────────────────────────────
    log_mel = extract_mel(audio)

    # ── Step 3: Z-score normalize ─────────────────────────────────────────────
    stats    = load_norm_stats(feats, dataset)
    norm_mel = normalize_mel(log_mel, stats)
    x = torch.FloatTensor(norm_mel).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,128,431)

    # ── Step 4: Load models ───────────────────────────────────────────────────
    entries = load_models(dataset, device=device, model_root=model_root)

    # ── Step 5: Per-model inference ───────────────────────────────────────────
    import torch.nn.functional as F

    all_probs = []
    with torch.no_grad():
        for entry in entries:
            logits = entry.model(x)
            probs  = F.softmax(logits, dim=1).cpu().numpy()[0]  # (5,)
            all_probs.append((probs, entry.val_acc, entry.name))

            if verbose:
                top_cls = CLASSES[probs.argmax()]
                print(f"  {entry.name:15s} → {top_cls:12s}  ({probs.max()*100:.1f}%)")

    # ── Step 6: Weighted ensemble (weight = val_acc) ──────────────────────────
    va_arr = np.array([va for _, va, _ in all_probs])
    weights = va_arr / va_arr.sum() if va_arr.sum() > 0 else \
              np.ones(len(all_probs)) / len(all_probs)

    ensemble_probs = sum(w * p for (p, _, _), w in zip(all_probs, weights))
    ensemble_probs = ensemble_probs / ensemble_probs.sum()   # re-normalize

    # ── Step 7: Final result ──────────────────────────────────────────────────
    pred_idx   = int(ensemble_probs.argmax())
    pred_cls   = CLASSES[pred_idx]
    confidence = float(ensemble_probs[pred_idx] * 100)

    if confidence >= 80:
        reliability = 'HIGH'
    elif confidence >= 60:
        reliability = 'MEDIUM'
    else:
        reliability = 'LOW'

    return {
        'prediction':    pred_cls,
        'confidence':    round(confidence, 2),
        'probabilities': {CLASSES[i]: round(float(ensemble_probs[i] * 100), 2)
                          for i in range(len(CLASSES))},
        'reliability':   reliability,
        'n_models':      len(entries),
        'device':        str(device),
        'dataset':       dataset,
        'audio_path':    os.path.abspath(audio_path),
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
) -> List[dict]:
    """
    Run inference on multiple .wav files. Models are loaded once and reused.

    Args:
        audio_paths: List of .wav file paths (or a directory path string).
        dataset:     'dataset1' or 'dataset2'.
        device:      'cuda', 'cpu', or None (auto-detect).
        feat_root:   Override for training/features directory.
        model_root:  Override for models directory.

    Returns:
        List of result dicts (same schema as predict()), one per input file.
        Files that fail preprocessing get an 'error' key instead of 'prediction'.
    """
    import torch
    import torch.nn.functional as F

    # Handle directory input
    if isinstance(audio_paths, (str, Path)) and Path(audio_paths).is_dir():
        audio_paths = sorted(Path(audio_paths).glob('**/*.wav'))

    feats  = feat_root or FEAT_ROOT

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    # Load models and norm_stats once
    entries    = load_models(dataset, device=device, model_root=model_root)
    stats      = load_norm_stats(feats, dataset)
    va_arr     = np.array([e.val_acc for e in entries])
    weights    = va_arr / va_arr.sum() if va_arr.sum() > 0 else \
                 np.ones(len(entries)) / len(entries)

    results = []
    for path in audio_paths:
        path = str(path)
        try:
            audio    = preprocess_audio(path)
            log_mel  = extract_mel(audio)
            norm_mel = normalize_mel(log_mel, stats)
            x = torch.FloatTensor(norm_mel).unsqueeze(0).unsqueeze(0).to(device)

            all_probs = []
            with torch.no_grad():
                for entry in entries:
                    logits = entry.model(x)
                    probs  = F.softmax(logits, dim=1).cpu().numpy()[0]
                    all_probs.append(probs)

            ensemble_probs = sum(w * p for w, p in zip(weights, all_probs))
            ensemble_probs = ensemble_probs / ensemble_probs.sum()

            pred_idx   = int(ensemble_probs.argmax())
            pred_cls   = CLASSES[pred_idx]
            confidence = float(ensemble_probs[pred_idx] * 100)
            reliability = 'HIGH' if confidence >= 80 else ('MEDIUM' if confidence >= 60 else 'LOW')

            results.append({
                'prediction':    pred_cls,
                'confidence':    round(confidence, 2),
                'probabilities': {CLASSES[i]: round(float(ensemble_probs[i] * 100), 2)
                                  for i in range(len(CLASSES))},
                'reliability':   reliability,
                'n_models':      len(entries),
                'device':        str(device),
                'dataset':       dataset,
                'audio_path':    os.path.abspath(path),
            })
        except Exception as e:
            results.append({
                'audio_path': os.path.abspath(path),
                'error':      str(e),
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

    print(f"\n{'='*55}")
    print(f"  Infant Cry Classifier")
    print(f"{'='*55}")
    print(f"  Audio   : {result['audio_path']}")
    print(f"  Dataset : {result['dataset']}  |  Device: {result['device']}"
          f"  |  Models: {result['n_models']}")
    print(f"{'─'*55}")
    print(f"  PREDICTION  : {result['prediction'].upper()}")
    print(f"  CONFIDENCE  : {result['confidence']:.1f}%")
    print(f"  RELIABILITY : {result['reliability']}")
    print(f"{'='*55}")

    if verbose:
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
