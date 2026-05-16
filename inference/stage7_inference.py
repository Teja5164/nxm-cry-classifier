"""
Stage 7: Inference CLI
======================
Command-line interface for the production 3-stage infant cry classifier.

Pipeline:
  Stage 1 → Cry Gate (custom BinaryCryCNN — rejects non-cry audio)
  Stage 2 → OOD check (entropy + confidence thresholds)
  Stage 3 → 5-class calibrated ensemble (BaselineCNN, CNN+BiLSTM, CryNet, SE-ResNet)

Usage:
  python inference/stage7_inference.py --audio path/to/cry.wav
  python inference/stage7_inference.py --audio cry.wav --verbose
  python inference/stage7_inference.py --audio cry.wav --dataset dataset1 --device cpu
"""

import os, sys, argparse, warnings
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
warnings.filterwarnings('ignore')

# ── Ensure inference/ is on the path ─────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Infant Cry Classifier — 3-Stage Pipeline')
parser.add_argument('--audio',   required=True,  help='Path to input audio file (.wav, .mp3, etc.)')
parser.add_argument('--dataset', default='dataset1', choices=['dataset1', 'dataset2'],
                    help='Which trained model set to use (default: dataset1)')
parser.add_argument('--device',  default=None,   choices=['cpu', 'cuda'],
                    help='Force device (default: auto-detect)')
parser.add_argument('--verbose', action='store_true', help='Show per-model scores and gate details')
parser.add_argument('--gate-threshold', type=float, default=0.34,
                    help='Cry gate score threshold (default: 0.34)')
args = parser.parse_args()


def main():
    try:
        from predict import predict
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the ML_pipeline/ root or inference/ directory.")
        sys.exit(1)

    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Infant Cry Classifier v2 — 3-Stage Pipeline")
    print(f"{'='*55}")
    print(f"  Audio  : {args.audio}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Device : {args.device or 'auto'}")
    print()

    result = predict(
        audio_path        = args.audio,
        dataset           = args.dataset,
        device            = args.device,
        verbose           = args.verbose,
        yamnet_threshold  = args.gate_threshold,
    )

    # ── Display result ────────────────────────────────────────────────────────
    if result['stage_blocked']:
        print(f"\n{'='*55}")
        print(f"  BLOCKED by {result['stage_blocked'].upper()}")
        print(f"  Reason : {result['reason']}")
        print(f"{'='*55}\n")
        return

    print(f"\n{'='*55}")
    print(f"  PREDICTION : {result['prediction'].upper()}")
    print(f"  CONFIDENCE : {result['confidence']:.1f}%")
    print(f"  RELIABILITY: {result['reliability']}")
    print(f"{'='*55}")

    if args.verbose:
        print(f"\n  Gate     : score={result['gate_score']:.3f}  method={result['gate_method']}")
        print(f"  Entropy  : {result['entropy']:.4f}")
        print(f"  Models   : {result['n_models']}")
        print(f"\n  Per-class probabilities:")
        sorted_items = sorted(result['probabilities'].items(), key=lambda x: -x[1])
        for cls, pct in sorted_items:
            bar = '█' * int(pct // 5)
            print(f"    {cls:12s}: {pct:5.1f}%  {bar}")
    print()


if __name__ == '__main__':
    main()
