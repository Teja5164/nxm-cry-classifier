"""
preprocess.py — Audio preprocessing for Infant Cry Classifier
==============================================================
Converts a raw .wav file into a normalised mel-spectrogram tensor
ready for model inference. All parameters are identical to those
used during training (stage1_extract_features.py).

Public API:
    preprocess_audio(path)              -> np.ndarray  (N_SAMPLES,)
    extract_mel(audio)                  -> np.ndarray  (128, T)
    normalize_mel(log_mel, norm_stats)  -> np.ndarray  (128, 431)
    load_norm_stats(feat_root, dataset) -> dict
    audio_to_tensor(path, feat_root, dataset, device) -> torch.Tensor (1,1,128,431)
"""

import os, json, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')

# ── Audio constants (MUST match training exactly) ─────────────────────────────
AUDIO_PARAMS = {
    'sr':         22050,
    'duration':   10.0,
    'n_mel':      128,
    'n_fft':      2048,
    'hop_length': 512,
    'fmin':       50,
    'fmax':       8000,
    'target_T':   431,       # time frames after feature extraction
}

SR         = AUDIO_PARAMS['sr']
DURATION   = AUDIO_PARAMS['duration']
N_SAMPLES  = int(SR * DURATION)   # 220500
N_MEL      = AUDIO_PARAMS['n_mel']
N_FFT      = AUDIO_PARAMS['n_fft']
HOP_LEN    = AUDIO_PARAMS['hop_length']
FMIN       = AUDIO_PARAMS['fmin']
FMAX       = AUDIO_PARAMS['fmax']
TARGET_T   = AUDIO_PARAMS['target_T']


def preprocess_audio(path: str) -> np.ndarray:
    """
    Load and preprocess a .wav file into a fixed-length audio array.

    Steps:
      1. Load + resample to 22050 Hz mono
      2. Peak-normalize to [-1, 1]
      3. Clip/pad to exactly 10 seconds (220500 samples)
         - Long files: take center 10s
         - Short files: loop-tile to fill 10s

    Args:
        path: Absolute or relative path to .wav file.

    Returns:
        audio: np.ndarray of shape (220500,) float32.

    Raises:
        FileNotFoundError: If audio file does not exist.
        ValueError: If audio file is empty or unreadable.
    """
    import librosa

    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        audio, _ = librosa.load(path, sr=SR, mono=True)
    except Exception as e:
        raise ValueError(f"Failed to load audio '{path}': {e}") from e

    if len(audio) == 0:
        raise ValueError(f"Audio file is empty: {path}")

    # Peak normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    # Clip/pad to exactly N_SAMPLES
    if len(audio) >= N_SAMPLES:
        start = (len(audio) - N_SAMPLES) // 2
        audio = audio[start:start + N_SAMPLES]
    else:
        n_reps = (N_SAMPLES // len(audio)) + 2
        audio  = np.tile(audio, n_reps)[:N_SAMPLES]

    return audio.astype(np.float32)


def extract_mel(audio: np.ndarray) -> np.ndarray:
    """
    Extract log-mel spectrogram from preprocessed audio.
    Parameters exactly match training (stage1_extract_features.py).

    Args:
        audio: np.ndarray of shape (220500,) float32.

    Returns:
        log_mel: np.ndarray of shape (128, T) float32.
    """
    import librosa

    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR,
        n_fft=N_FFT, hop_length=HOP_LEN,
        n_mels=N_MEL, fmin=FMIN, fmax=FMAX
    )
    log_mel = librosa.power_to_db(mel + 1e-6, ref=np.max)
    return log_mel.astype(np.float32)  # (128, T)


def load_norm_stats(feat_root: str, dataset: str) -> dict:
    """
    Load z-score normalisation statistics saved during training.

    Args:
        feat_root: Path to training/features directory.
        dataset:   'dataset1' or 'dataset2'.

    Returns:
        dict with keys 'mel_mean' (128,) and 'mel_std' (128,).

    Raises:
        FileNotFoundError: If norm_stats.json is missing.
    """
    stats_path = os.path.join(feat_root, dataset, 'norm_stats.json')
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"norm_stats.json not found at: {stats_path}\n"
            "Run training/scripts/stage1_extract_features.py first."
        )
    with open(stats_path, 'r') as f:
        return json.load(f)


def normalize_mel(log_mel: np.ndarray, norm_stats: dict) -> np.ndarray:
    """
    Z-score normalize mel spectrogram using per-frequency-bin statistics.

    Args:
        log_mel:    np.ndarray shape (128, T).
        norm_stats: dict with 'mel_mean' (128,) and 'mel_std' (128,).

    Returns:
        normalized: np.ndarray shape (128, 431) — padded/cropped to TARGET_T.
    """
    mean = np.array(norm_stats['mel_mean'], dtype=np.float32)  # (128,)
    std  = np.array(norm_stats['mel_std'],  dtype=np.float32)  # (128,)
    std  = np.where(std < 1e-8, 1.0, std)

    log_mel = (log_mel - mean[:, None]) / std[:, None]

    # Ensure fixed time dimension
    if log_mel.shape[1] < TARGET_T:
        pad     = TARGET_T - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode='edge')
    elif log_mel.shape[1] > TARGET_T:
        log_mel = log_mel[:, :TARGET_T]

    return log_mel.astype(np.float32)  # (128, 431)


def audio_to_tensor(path: str, feat_root: str, dataset: str, device=None):
    """
    Full preprocessing pipeline: wav file → model-ready tensor.
    Convenience wrapper combining all preprocessing steps.

    Args:
        path:      Path to .wav audio file.
        feat_root: Path to training/features directory.
        dataset:   'dataset1' or 'dataset2'.
        device:    torch.device or None (auto-detect).

    Returns:
        tensor: torch.FloatTensor of shape (1, 1, 128, 431) on device.
        meta:   dict with 'duration_s', 'sr', 'dataset', 'path'.
    """
    import torch

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    audio    = preprocess_audio(path)
    log_mel  = extract_mel(audio)
    stats    = load_norm_stats(feat_root, dataset)
    norm_mel = normalize_mel(log_mel, stats)

    tensor = torch.FloatTensor(norm_mel).unsqueeze(0).unsqueeze(0).to(device)

    meta = {
        'path':       str(path),
        'duration_s': len(audio) / SR,
        'sr':         SR,
        'dataset':    dataset,
        'shape':      list(tensor.shape),
    }
    return tensor, meta
