"""
yamnet_gate.py — Two-stage Cry Gate for Infant Cry Classifier
=============================================================
Stage 1a: Custom binary cry detector (if trained model is available)
Stage 1b: YAMNet fallback (Google pre-trained AudioSet classifier)

Both gates answer the same question: "Is this audio a baby cry?"
The custom binary model is preferred when available — it is domain-specific
and outperforms YAMNet on this exact task.

Public API:
    gate = CryGate()
    result = gate.is_cry(audio_path)
    # result: {'is_cry': bool, 'score': float, 'method': str,
    #          'top_class': str, 'reason': str}

    gate.available        → bool (True if at least one gate is usable)
    gate.method           → 'custom' | 'yamnet' | 'none'

Threshold defaults:
    CUSTOM_THRESHOLD  = 0.50   (binary model output probability)
    YAMNET_THRESHOLD  = 0.12   (sum of AudioSet baby-cry class scores)
"""

import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings('ignore')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # suppress TF log spam

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).resolve().parent
_ROOT       = _HERE.parent                                  # ML_pipeline/
_MODEL_ROOT = _ROOT / 'models'
CUSTOM_GATE_PATH = str(_MODEL_ROOT / 'cry_gate' / 'best_model.pth')

# ── Default thresholds ────────────────────────────────────────────────────────
CUSTOM_THRESHOLD  = 0.50
YAMNET_THRESHOLD  = 0.12

# AudioSet class indices for baby/infant cry sounds
YAMNET_CRY_CLASSES = {20: 'Baby cry, infant cry', 21: 'Crying, sobbing', 22: 'Whimper'}


# ─────────────────────────────────────────────────────────────────────────────
# Custom binary cry detector (lightweight CNN)
# Architecture matches stage_binary_cry_gate.py training script
# ─────────────────────────────────────────────────────────────────────────────
def _build_binary_cnn() -> 'torch.nn.Module':
    """Build the lightweight binary cry detector architecture."""
    import torch.nn as nn

    class BinaryCryCNN(nn.Module):
        """
        Lightweight binary CNN: cry vs not-cry.
        Input: (B, 1, 128, 431) log-mel spectrogram (same as 5-class models).
        Output: (B, 1) sigmoid probability of being a cry.
        ~480K parameters.
        """
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1 — 1 → 32
                nn.Conv2d(1, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),          # → (32, 64, 215)

                # Block 2 — 32 → 64
                nn.Conv2d(32, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),          # → (64, 32, 107)

                # Block 3 — 64 → 128
                nn.Conv2d(64, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),          # → (128, 16, 53)

                # Block 4 — 128 → 128
                nn.Conv2d(128, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),  # → (128, 4, 4) = 2048
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, 1),
            )

        def forward(self, x):
            return self.classifier(self.features(x))   # raw logit

    return BinaryCryCNN()


def _load_custom_gate(model_path: str, device):
    """Load the custom binary cry detector from checkpoint."""
    import torch
    if not os.path.exists(model_path):
        return None, None
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model = _build_binary_cnn()
        state = ckpt.get('model_state', ckpt)
        model.load_state_dict(state)
        model.to(device).eval()
        threshold = ckpt.get('threshold', CUSTOM_THRESHOLD)
        return model, threshold
    except Exception as e:
        warnings.warn(f"[CryGate] Custom model load failed: {e}. Falling back to YAMNet.")
        return None, None


def _score_custom(model, audio_path: str, device, threshold: float) -> dict:
    """
    Run custom binary detector on audio_path.
    Returns gate result dict.
    """
    import torch
    import torch.nn.functional as F

    try:
        # Reuse the same preprocessing as the 5-class pipeline
        import sys
        if str(_HERE) not in sys.path:
            sys.path.insert(0, str(_HERE))
        from preprocess import preprocess_audio, extract_mel, load_norm_stats, normalize_mel

        feat_root = str(_ROOT / 'training' / 'features')
        audio    = preprocess_audio(audio_path)
        log_mel  = extract_mel(audio)
        stats    = load_norm_stats(feat_root, 'dataset1')
        norm_mel = normalize_mel(log_mel, stats)
        x = torch.FloatTensor(norm_mel).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            logit = model(x)
            prob  = torch.sigmoid(logit).item()

        is_cry = prob >= threshold
        return {
            'is_cry':     is_cry,
            'score':      round(prob, 4),
            'method':     'custom',
            'top_class':  'Baby cry' if is_cry else 'Not a cry',
            'reason':     f"custom_gate_score={prob:.3f} threshold={threshold:.2f}",
        }
    except Exception as e:
        return {'is_cry': None, 'score': 0.0, 'method': 'custom_failed',
                'top_class': '', 'reason': str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# YAMNet gate (TensorFlow Hub)
# ─────────────────────────────────────────────────────────────────────────────
def _load_yamnet():
    """Attempt to load YAMNet from TF Hub. Returns model or None."""
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
        tf.get_logger().setLevel('ERROR')
        model = hub.load('https://tfhub.dev/google/yamnet/1')
        return model
    except Exception as e:
        warnings.warn(f"[CryGate] YAMNet load failed: {e}. Gate will be disabled.")
        return None


def _score_yamnet(yamnet_model, audio_path: str, threshold: float) -> dict:
    """
    Run YAMNet on audio_path (16kHz required).
    Returns gate result dict.
    """
    try:
        import tensorflow as tf
        import librosa

        audio_16k, _ = librosa.load(audio_path, sr=16000, mono=True)
        audio_tf = tf.constant(audio_16k, dtype=tf.float32)
        scores, _, _ = yamnet_model(audio_tf)
        avg_scores = scores.numpy().mean(axis=0)   # (521,) averaged over patches

        cry_score = float(sum(avg_scores[i] for i in YAMNET_CRY_CLASSES))
        top_idx   = int(avg_scores.argmax())
        top_name  = YAMNET_CRY_CLASSES.get(top_idx, f'class_{top_idx}')

        # Resolve top class name via class map if possible
        try:
            import csv, io, urllib.request
            # Cached class names (don't re-fetch every call — use the known map)
            pass
        except Exception:
            pass

        is_cry = cry_score >= threshold
        return {
            'is_cry':    is_cry,
            'score':     round(cry_score, 4),
            'method':    'yamnet',
            'top_class': top_name,
            'reason':    f"yamnet_cry_score={cry_score:.3f} threshold={threshold:.2f}",
        }
    except Exception as e:
        return {'is_cry': None, 'score': 0.0, 'method': 'yamnet_failed',
                'top_class': '', 'reason': str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Public interface: CryGate
# ─────────────────────────────────────────────────────────────────────────────
class CryGate:
    """
    Two-stage cry gate. Prefers custom binary model when available,
    falls back to YAMNet, falls back gracefully to always-pass if neither works.

    Usage:
        gate = CryGate()                        # lazy-loaded on first call
        result = gate.is_cry('baby.wav')
        if not result['is_cry']:
            print("Not a baby cry:", result['reason'])

    Args:
        device:           torch device string/object or None (auto-detect).
        custom_threshold: Override probability threshold for custom model (0–1).
        yamnet_threshold: Override score threshold for YAMNet gate (0–1).
        custom_path:      Override path to custom binary model checkpoint.
        prefer_yamnet:    Force YAMNet even if custom model is available.
    """

    def __init__(
        self,
        device=None,
        custom_threshold: float = CUSTOM_THRESHOLD,
        yamnet_threshold: float = YAMNET_THRESHOLD,
        custom_path: Optional[str] = None,
        prefer_yamnet: bool = False,
    ):
        import torch
        self._device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self._custom_threshold = custom_threshold
        self._yamnet_threshold = yamnet_threshold
        self._custom_path      = custom_path or CUSTOM_GATE_PATH
        self._prefer_yamnet    = prefer_yamnet

        # Lazy-loaded models
        self._custom_model  = None
        self._custom_thresh = custom_threshold
        self._yamnet_model  = None
        self._initialized   = False

    def _init(self):
        if self._initialized:
            return
        self._initialized = True

        if not self._prefer_yamnet:
            self._custom_model, ct = _load_custom_gate(self._custom_path, self._device)
            if self._custom_model is not None and ct is not None:
                self._custom_thresh = ct

        if self._custom_model is None:
            self._yamnet_model = _load_yamnet()

    @property
    def available(self) -> bool:
        self._init()
        return self._custom_model is not None or self._yamnet_model is not None

    @property
    def method(self) -> str:
        self._init()
        if self._custom_model is not None:
            return 'custom'
        if self._yamnet_model is not None:
            return 'yamnet'
        return 'none'

    def is_cry(self, audio_path: str) -> dict:
        """
        Determine if audio is a baby cry.

        Returns:
            dict with keys:
                is_cry     (bool | None) — None means gate unavailable (always passes)
                score      (float) — gate confidence score 0–1
                method     (str)   — 'custom' | 'yamnet' | 'none'
                top_class  (str)   — top detected audio class
                reason     (str)   — human-readable explanation
        """
        self._init()

        if self._custom_model is not None:
            return _score_custom(
                self._custom_model, audio_path,
                self._device, self._custom_thresh
            )

        if self._yamnet_model is not None:
            return _score_yamnet(
                self._yamnet_model, audio_path, self._yamnet_threshold
            )

        # No gate available — pass through
        return {
            'is_cry':    None,
            'score':     0.0,
            'method':    'none',
            'top_class': 'unknown',
            'reason':    'No cry gate available (neither custom model nor YAMNet loaded)',
        }


# ── Module-level singleton (loaded once per process) ─────────────────────────
_GATE: Optional[CryGate] = None


def get_gate(
    device=None,
    custom_threshold: float = CUSTOM_THRESHOLD,
    yamnet_threshold: float = YAMNET_THRESHOLD,
) -> CryGate:
    """Get (or create) the module-level gate singleton."""
    global _GATE
    if _GATE is None:
        _GATE = CryGate(
            device=device,
            custom_threshold=custom_threshold,
            yamnet_threshold=yamnet_threshold,
        )
    return _GATE


def reset_gate():
    """Force re-initialization of the gate singleton (for testing)."""
    global _GATE
    _GATE = None
