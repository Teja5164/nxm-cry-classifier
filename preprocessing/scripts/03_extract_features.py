"""
Script 3: Extract acoustic features from preprocessed cry audio.
Features per file:
  - duration, rms_mean, rms_std, rms_slope (energy envelope)
  - zcr_mean, zcr_std
  - f0_mean, f0_std, f0_min, f0_max (fundamental frequency / pitch)
  - spectral_centroid_mean, spectral_rolloff_mean, spectral_bandwidth_mean
  - mfcc_1..13 (mean of each coefficient)
  - periodicity (rhythm regularity via autocorrelation)
Saves to: datasets/features/features.csv
"""
from pathlib import Path
import os, sys, json
import numpy as np
import soundfile as sf
import librosa
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = Path(__file__).resolve().parent.parent.parent  # ML_pipeline/
INPUT_DIR   = str(_ROOT / 'datasets' / 'preprocessed')
FEATURES_F  = str(_ROOT / 'datasets' / 'features' / 'features.csv')
SR          = 22050
N_MFCC      = 13
WORKERS     = 6

os.makedirs(os.path.dirname(FEATURES_F), exist_ok=True)


def compute_periodicity(y, sr, hop_length=512):
    """Compute rhythm periodicity via autocorrelation of RMS envelope."""
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    if len(rms) < 4:
        return 0.0
    rms_norm = rms - rms.mean()
    if rms_norm.std() == 0:
        return 0.0
    rms_norm /= rms_norm.std()
    autocorr = np.correlate(rms_norm, rms_norm, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / (autocorr[0] + 1e-9)
    min_lag = int(0.2 * sr / hop_length)
    max_lag = int(1.5 * sr / hop_length)
    if max_lag > len(autocorr):
        return 0.0
    peak = autocorr[min_lag:max_lag].max()
    return float(np.clip(peak, 0, 1))


def extract_features(fname):
    path = os.path.join(INPUT_DIR, fname)
    try:
        # Use soundfile — faster since already at target SR (no resampling)
        y, sr = sf.read(path, always_2d=False)
        y = y.astype(np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if len(y) == 0:
            return None, fname

        feat = {}
        feat['filename'] = fname
        feat['duration'] = len(y) / SR

        hop = 512
        # Energy / RMS
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        feat['rms_mean']  = float(rms.mean())
        feat['rms_std']   = float(rms.std())
        t = np.arange(len(rms))
        feat['rms_slope'] = float(np.polyfit(t, rms, 1)[0]) if len(t) > 1 else 0.0

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]
        feat['zcr_mean'] = float(zcr.mean())
        feat['zcr_std']  = float(zcr.std())

        # Fundamental frequency — use YIN (much faster than pYIN)
        f0 = librosa.yin(y, fmin=50, fmax=1000, sr=SR, hop_length=hop)
        f0_valid = f0[(f0 > 50) & (f0 < 1000)]
        if len(f0_valid) > 0:
            feat['f0_mean']   = float(np.mean(f0_valid))
            feat['f0_std']    = float(np.std(f0_valid))
            feat['f0_min']    = float(np.min(f0_valid))
            feat['f0_max']    = float(np.max(f0_valid))
            feat['f0_median'] = float(np.median(f0_valid))
        else:
            feat['f0_mean'] = feat['f0_std'] = feat['f0_min'] = feat['f0_max'] = feat['f0_median'] = 0.0

        # Spectral features
        sc    = librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=hop)[0]
        sr_ft = librosa.feature.spectral_rolloff(y=y, sr=SR, hop_length=hop)[0]
        sb    = librosa.feature.spectral_bandwidth(y=y, sr=SR, hop_length=hop)[0]
        feat['spectral_centroid_mean']  = float(sc.mean())
        feat['spectral_rolloff_mean']   = float(sr_ft.mean())
        feat['spectral_bandwidth_mean'] = float(sb.mean())

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=hop)
        for i in range(N_MFCC):
            feat[f'mfcc_{i+1}'] = float(mfccs[i].mean())

        # Periodicity / rhythm
        feat['periodicity'] = compute_periodicity(y, SR, hop_length=hop)

        return feat, None
    except Exception as e:
        return None, {'file': fname, 'error': str(e)}


def extract_all():
    all_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.wav')])

    if os.path.exists(FEATURES_F):
        existing_set = set(pd.read_csv(FEATURES_F)['filename'].tolist())
        files = [f for f in all_files if f not in existing_set]
        print(f"Resuming: {len(files)} new files (skipping {len(existing_set)} done).")
    else:
        files = all_files
        print(f"Extracting features for {len(files)} files with {WORKERS} workers...")

    rows, failed = [], []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(extract_features, f): f for f in files}
        for fut in tqdm(as_completed(futures), total=len(files), unit='file'):
            feat, err = fut.result()
            if feat:
                rows.append(feat)
            if err:
                failed.append(err)

    df_new = pd.DataFrame(rows)
    if os.path.exists(FEATURES_F) and len(existing_set) > 0:
        df_old = pd.read_csv(FEATURES_F)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(FEATURES_F, index=False)
    print(f"\nFeature extraction done. Total rows: {len(df)}, Failed: {len(failed)}")
    if failed:
        with open(str(_ROOT / 'datasets' / 'labels' / 'feature_failures.json'), 'w') as fp:
            json.dump(failed, fp, indent=2)
    print(f"Saved: {FEATURES_F}")


if __name__ == '__main__':
    extract_all()

