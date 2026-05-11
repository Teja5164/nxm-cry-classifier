"""
Script 2: Preprocess all downloaded cry audio files.
  - Resample to 22050 Hz
  - Convert to mono
  - RMS normalize
  - Trim leading/trailing silence
  - Save to: datasets/preprocessed/
"""
from pathlib import Path
import os, sys, json
import numpy as np
import soundfile as sf
import scipy.signal
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = Path(__file__).resolve().parent.parent.parent  # ML_pipeline/
INPUT_DIR  = str(_ROOT / 'datasets' / 'raw_cry')
OUTPUT_DIR = str(_ROOT / 'datasets' / 'preprocessed')
TARGET_SR  = 22050
TOP_DB     = 30
WORKERS    = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)


def preprocess_audio(path):
    # Read with soundfile (fast, no resampling yet)
    y, sr = sf.read(path, always_2d=False)

    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)

    y = y.astype(np.float32)

    # Resample to TARGET_SR using scipy (fast)
    if sr != TARGET_SR:
        num_samples = int(len(y) * TARGET_SR / sr)
        y = scipy.signal.resample(y, num_samples)

    # Trim silence: remove frames where amplitude < threshold
    top_db_linear = 10 ** (-TOP_DB / 20.0)
    frame_len = int(TARGET_SR * 0.025)   # 25ms frames
    hop_len   = int(TARGET_SR * 0.010)   # 10ms hop
    # Simple energy-based trimming
    energy = np.array([
        np.sqrt(np.mean(y[i:i+frame_len]**2))
        for i in range(0, max(1, len(y)-frame_len), hop_len)
    ])
    voiced = energy > top_db_linear
    if voiced.any():
        start = max(0, np.argmax(voiced) * hop_len)
        end   = min(len(y), (len(voiced) - np.argmax(voiced[::-1])) * hop_len)
        y_trimmed = y[start:end]
    else:
        y_trimmed = y

    # Fallback if too short
    if len(y_trimmed) < TARGET_SR * 0.1:
        y_trimmed = y

    # RMS normalize to -20 dBFS
    rms = np.sqrt(np.mean(y_trimmed ** 2))
    if rms > 1e-8:
        target_rms = 10 ** (-20 / 20)
        y_trimmed = y_trimmed * (target_rms / rms)

    y_trimmed = np.clip(y_trimmed, -1.0, 1.0)
    return y_trimmed.astype(np.float32)


def _process_one(fname):
    src = os.path.join(INPUT_DIR, fname)
    dst = os.path.join(OUTPUT_DIR, fname)
    try:
        y = preprocess_audio(src)
        sf.write(dst, y, TARGET_SR, subtype='PCM_16')
        return None
    except Exception as e:
        return {'file': fname, 'error': str(e)}


def process_all():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.wav')]
    already = set(os.listdir(OUTPUT_DIR))
    to_process = [f for f in files if f not in already]

    print(f"Files to preprocess: {len(to_process)} (skipping {len(already)} already done)")
    print(f"Using {WORKERS} parallel workers...")
    failed = []

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_process_one, f): f for f in to_process}
        for fut in tqdm(as_completed(futures), total=len(to_process), unit='file'):
            result = fut.result()
            if result:
                failed.append(result)

    print(f"\nPreprocessing done. Success: {len(to_process)-len(failed)}, Failed: {len(failed)}")
    if failed:
        with open(str(_ROOT / 'datasets' / 'labels' / 'preprocess_failures.json'), 'w') as fp:
            json.dump(failed, fp, indent=2)
    print(f"Total preprocessed files: {len(os.listdir(OUTPUT_DIR))}")


if __name__ == '__main__':
    process_all()

