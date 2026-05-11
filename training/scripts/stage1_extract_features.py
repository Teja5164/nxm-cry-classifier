"""
Stage 1: Feature Extraction Pipeline
=====================================
For each audio file in Dataset1 and Dataset2:
  1. Load & normalize to exactly 10 seconds (pad / center-crop)
  2. Extract Mel Spectrogram  → (128, 431) float32 image
  3. Extract MFCC + Δ + ΔΔ   → (117, 431) float32 image
  4. Extract Log-Mel          → (128, 431) float32 (slight variant for branch diversity)

Saved per-dataset as:
  features/datasetN/mel_specs.npy      shape (N, 1, 128, 431)
  features/datasetN/mfccs.npy          shape (N, 1, 117, 431)
  features/datasetN/labels.npy         shape (N,) int64
  features/datasetN/filenames.npy      shape (N,) object (str)
  features/datasetN/dataset_info.json  metadata

All features are z-score normalised (per-channel, per-feature-type).
Normalisation stats saved → used at inference time too.
"""

from pathlib import Path
import os, json, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
SR          = 22050          # sample rate
DURATION    = 10.0           # fixed clip length (seconds)
N_SAMPLES   = int(SR * DURATION)  # 220500 samples

N_MELS      = 128            # mel spectrogram bins
N_FFT       = 2048           # FFT window
HOP_LENGTH  = 512            # hop → time frames = ceil(N_SAMPLES/HOP_LENGTH)+1 ≈ 431
N_MFCC      = 39             # base MFCCs; + Δ + ΔΔ = 117 total
FMIN        = 50             # min freq (Hz) — infant cry starts ~200Hz but keep margin
FMAX        = 8000           # max freq (Hz) — infant cry energy mostly <4kHz

CLASSES     = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
CLASS2IDX   = {c: i for i, c in enumerate(CLASSES)}

_ROOT = Path(__file__).resolve().parent.parent.parent  # ML_pipeline/
DATASETS = {
    'dataset1': str(_ROOT / 'datasets' / 'Dataset1'),
    'dataset2': str(_ROOT / 'datasets' / 'Dataset2'),
}
OUT_ROOT = str(_ROOT / 'training' / 'features')


# ── Audio Loading & Preprocessing ────────────────────────────────────────────
def load_fixed_length(path, sr=SR, n_samples=N_SAMPLES):
    """Load audio, resample if needed, fix to exactly n_samples."""
    try:
        y, orig_sr = sf.read(path, always_2d=False)
        y = y.astype(np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        # Resample if needed
        if orig_sr != sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        # Normalize amplitude
        peak = np.abs(y).max()
        if peak > 1e-6:
            y = y / peak
        # Fix length: pad or center-crop
        if len(y) < n_samples:
            # Pad with reflection to avoid silence artifacts
            pad_total = n_samples - len(y)
            pad_left  = pad_total // 2
            pad_right = pad_total - pad_left
            y = np.pad(y, (pad_left, pad_right), mode='reflect')
        elif len(y) > n_samples:
            # Center crop → preserves most characteristic cry segment
            start = (len(y) - n_samples) // 2
            y = y[start:start + n_samples]
        return y
    except Exception as e:
        return None


# ── Feature Extraction ────────────────────────────────────────────────────────
def extract_mel(y, sr=SR):
    """128-bin log Mel spectrogram → shape (128, T)."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)   # in dB, range ~[-80, 0]
    return log_mel.astype(np.float32)                 # (128, T)


def extract_mfcc_delta(y, sr=SR):
    """MFCC + Δ + ΔΔ → shape (117, T)."""
    mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH,
                                  fmin=FMIN, fmax=FMAX)
    delta  = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.vstack([mfcc, delta, delta2])       # (117, T)
    return combined.astype(np.float32)


def pad_or_crop_time(feat, target_frames=431):
    """Ensure time axis is exactly target_frames wide."""
    T = feat.shape[1]
    if T < target_frames:
        pad = target_frames - T
        feat = np.pad(feat, ((0,0),(0,pad)), mode='constant')
    elif T > target_frames:
        feat = feat[:, :target_frames]
    return feat


# ── Normalisation (z-score per feature map) ───────────────────────────────────
def compute_norm_stats(arr):
    """arr shape (N, 1, F, T). Returns per-channel (F,) mean & std."""
    # Flatten over N and T, keep F
    data = arr[:, 0, :, :]          # (N, F, T)
    mean = data.mean(axis=(0, 2))   # (F,)
    std  = data.std(axis=(0, 2)).clip(min=1e-8)
    return mean, std


def apply_norm(arr, mean, std):
    """Normalise arr (N,1,F,T) using mean/std (F,)."""
    arr = arr.copy()
    arr[:, 0, :, :] = (arr[:, 0, :, :] - mean[None, :, None]) / std[None, :, None]
    return arr


# ── Worker function (runs in subprocess) ─────────────────────────────────────
def _worker(args):
    fpath, label_idx, fname = args
    try:
        y = load_fixed_length(fpath)
        if y is None:
            return None
        mel  = pad_or_crop_time(extract_mel(y),        431)
        mfcc = pad_or_crop_time(extract_mfcc_delta(y), 431)
        return (mel[np.newaxis], mfcc[np.newaxis], label_idx, fname)
    except Exception:
        return None


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def process_dataset(ds_name, ds_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Processing {ds_name}  →  {out_dir}")
    print(f"{'='*60}")

    # Build task list
    tasks = []
    for cls in CLASSES:
        cls_dir = os.path.join(ds_root, cls)
        files   = sorted([f for f in os.listdir(cls_dir) if f.endswith('.wav')])
        print(f"  [{cls}] {len(files)} files")
        for fname in files:
            tasks.append((os.path.join(cls_dir, fname), CLASS2IDX[cls], fname))

    mel_list, mfcc_list, label_list, fname_list = [], [], [], []
    errors = 0
    N_WORKERS = min(6, os.cpu_count() or 4)
    print(f"\n  Running with {N_WORKERS} parallel workers on {len(tasks)} files ...")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_worker, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(tasks), desc=f"  {ds_name}"):
            result = fut.result()
            if result is None:
                errors += 1
                continue
            mel, mfcc, lbl, fn = result
            mel_list.append(mel)
            mfcc_list.append(mfcc)
            label_list.append(lbl)
            fname_list.append(fn)

    mel_arr  = np.stack(mel_list,  axis=0).astype(np.float32)   # (N, 1, 128, 431)
    mfcc_arr = np.stack(mfcc_list, axis=0).astype(np.float32)   # (N, 1, 117, 431)
    labels   = np.array(label_list, dtype=np.int64)
    fnames   = np.array(fname_list, dtype=object)

    print(f"\n  Raw shapes  → mel:{mel_arr.shape}  mfcc:{mfcc_arr.shape}")
    print(f"  Errors      → {errors}")

    # ── Normalise ──────────────────────────────────────────────────────────
    print("  Computing normalisation stats ...")
    mel_mean,  mel_std  = compute_norm_stats(mel_arr)
    mfcc_mean, mfcc_std = compute_norm_stats(mfcc_arr)

    mel_arr  = apply_norm(mel_arr,  mel_mean,  mel_std)
    mfcc_arr = apply_norm(mfcc_arr, mfcc_mean, mfcc_std)

    # ── Save ───────────────────────────────────────────────────────────────
    print("  Saving arrays ...")
    np.save(os.path.join(out_dir, 'mel_specs.npy'),  mel_arr)
    np.save(os.path.join(out_dir, 'mfccs.npy'),      mfcc_arr)
    np.save(os.path.join(out_dir, 'labels.npy'),      labels)
    np.save(os.path.join(out_dir, 'filenames.npy'),   fnames)

    # Save normalisation stats (needed at inference)
    norm_stats = {
        'mel_mean':  mel_mean.tolist(),
        'mel_std':   mel_std.tolist(),
        'mfcc_mean': mfcc_mean.tolist(),
        'mfcc_std':  mfcc_std.tolist(),
    }
    with open(os.path.join(out_dir, 'norm_stats.json'), 'w') as f:
        json.dump(norm_stats, f)

    # Class distribution check
    info = {
        'dataset': ds_name,
        'total_samples': int(len(labels)),
        'mel_shape': list(mel_arr.shape),
        'mfcc_shape': list(mfcc_arr.shape),
        'class_distribution': {c: int((labels == i).sum()) for i, c in enumerate(CLASSES)},
        'errors': errors,
        'sr': SR,
        'duration_s': DURATION,
        'n_mels': N_MELS,
        'n_mfcc': N_MFCC,
        'hop_length': HOP_LENGTH,
        'n_fft': N_FFT,
        'fmin': FMIN,
        'fmax': FMAX,
    }
    with open(os.path.join(out_dir, 'dataset_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n  {'─'*50}")
    print(f"  {ds_name} Feature Extraction Complete")
    print(f"  {'─'*50}")
    print(f"  Total samples    : {len(labels)}")
    print(f"  mel_specs shape  : {mel_arr.shape}  ({mel_arr.nbytes/1e6:.1f} MB)")
    print(f"  mfccs shape      : {mfcc_arr.shape}  ({mfcc_arr.nbytes/1e6:.1f} MB)")
    print(f"  Class distribution:")
    for i, c in enumerate(CLASSES):
        print(f"    {c:15s}: {(labels==i).sum()}")
    print(f"  Saved to: {out_dir}")
    return info


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    all_info = {}
    for ds_name, ds_root in DATASETS.items():
        out_dir = os.path.join(OUT_ROOT, ds_name)
        info = process_dataset(ds_name, ds_root, out_dir)
        all_info[ds_name] = info

    print("\n" + "="*60)
    print("  STAGE 1 COMPLETE — Feature Extraction Summary")
    print("="*60)
    for ds, info in all_info.items():
        total_mb = sum([
            os.path.getsize(os.path.join(OUT_ROOT, ds, f)) / 1e6
            for f in ['mel_specs.npy', 'mfccs.npy', 'labels.npy']
        ])
        print(f"  {ds}: {info['total_samples']} samples, {total_mb:.0f} MB on disk")
    print("\n  Ready for Stage 2: Model Training")
    print("="*60)
