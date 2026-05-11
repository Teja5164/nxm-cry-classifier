"""
Script 06a: Build Dataset1 — 800 samples/class (4,000 total)
  - Weighted ensemble fusion (CLAP + heuristic)
  - Augmentation for minority classes
  - Stratified 70/15/15 splits
OUTPUT: datasets/Dataset1/
"""

from pathlib import Path
import os, shutil, random, warnings
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent  # ML_pipeline/
CLAP_F       = str(_ROOT / 'datasets' / 'labels' / 'clap_labels.csv')
HEUR_F       = str(_ROOT / 'datasets' / 'labels' / 'heuristic_labels.csv')
FINAL_LABELS = str(_ROOT / 'datasets' / 'labels' / 'final_labels.csv')
PREPROCESSED = str(_ROOT / 'datasets' / 'preprocessed')
OUT_DIR      = str(_ROOT / 'datasets' / 'Dataset1')

CLASSES          = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
SR               = 22050
TARGET_PER_CLASS = 800
MAX_AUG_FACTOR   = 15

CLAP_CLASS_W = np.array([0.55, 0.55, 0.55, 0.28, 0.72])
HEUR_CLASS_W = 1.0 - CLAP_CLASS_W

AUG_TYPES = ['ts_slow', 'ts_slower', 'ts_fast', 'ts_faster',
             'ps_down', 'ps_down2', 'ps_up', 'ps_up2',
             'noise', 'noise_hi', 'vol_up', 'vol_down',
             'ts_ps', 'combined2', 'combined3', 'combined4']


def fuse_labels(clap_df, heur_df):
    clap_cols = [f'clap_prob_{c}'  for c in CLASSES]
    heur_cols = [f'heur_score_{c}' for c in CLASSES]
    df = pd.merge(
        clap_df[['filename', 'clap_label', 'clap_confidence'] + clap_cols],
        heur_df[['filename', 'heuristic_label', 'heuristic_confidence'] + heur_cols],
        on='filename', how='inner'
    )
    clap_mat = df[clap_cols].values.astype(np.float32)
    heur_mat = df[heur_cols].values.astype(np.float32)
    heur_mat = heur_mat / heur_mat.sum(axis=1, keepdims=True).clip(min=1e-8)
    final_scores = clap_mat * CLAP_CLASS_W + heur_mat * HEUR_CLASS_W
    label_idx    = final_scores.argmax(axis=1)
    final_labels = [CLASSES[i] for i in label_idx]
    final_confs  = final_scores.max(axis=1)
    agree = (df['clap_label'].values == df['heuristic_label'].values)
    tier  = np.where(agree, 'HIGH', np.where(final_confs >= 0.48, 'MEDIUM', 'LOW'))
    df['final_label'] = final_labels
    df['final_conf']  = final_confs.round(4)
    df['tier']        = tier
    df['agree']       = agree
    print("\nFused label distribution (all files):")
    print(pd.Series(final_labels).value_counts().to_string())
    return df


def augment(y, aug_type):
    try:
        if   aug_type == 'ts_slow':    return librosa.effects.time_stretch(y, rate=0.85)
        elif aug_type == 'ts_slower':  return librosa.effects.time_stretch(y, rate=0.78)
        elif aug_type == 'ts_fast':    return librosa.effects.time_stretch(y, rate=1.15)
        elif aug_type == 'ts_faster':  return librosa.effects.time_stretch(y, rate=1.22)
        elif aug_type == 'ps_down':    return librosa.effects.pitch_shift(y, sr=SR, n_steps=-2)
        elif aug_type == 'ps_down2':   return librosa.effects.pitch_shift(y, sr=SR, n_steps=-4)
        elif aug_type == 'ps_up':      return librosa.effects.pitch_shift(y, sr=SR, n_steps=2)
        elif aug_type == 'ps_up2':     return librosa.effects.pitch_shift(y, sr=SR, n_steps=4)
        elif aug_type == 'noise':
            return np.clip(y + np.random.randn(len(y)).astype(np.float32)*0.006, -1, 1)
        elif aug_type == 'noise_hi':
            return np.clip(y + np.random.randn(len(y)).astype(np.float32)*0.012, -1, 1)
        elif aug_type == 'vol_up':     return np.clip(y * 1.35, -1, 1)
        elif aug_type == 'vol_down':   return (y * 0.65).astype(np.float32)
        elif aug_type == 'ts_ps':
            y2 = librosa.effects.time_stretch(y, rate=0.90)
            return librosa.effects.pitch_shift(y2, sr=SR, n_steps=1)
        elif aug_type == 'combined2':
            y2 = librosa.effects.time_stretch(y, rate=0.88)
            return np.clip(y2 + np.random.randn(len(y2)).astype(np.float32)*0.005, -1, 1)
        elif aug_type == 'combined3':
            y2 = librosa.effects.pitch_shift(y, sr=SR, n_steps=-3)
            return np.clip(y2 * 1.2, -1, 1)
        elif aug_type == 'combined4':
            y2 = librosa.effects.time_stretch(y, rate=1.10)
            return librosa.effects.pitch_shift(y2, sr=SR, n_steps=2)
    except Exception:
        return y
    return y


def build_dataset(df):
    for cls in CLASSES:
        os.makedirs(os.path.join(OUT_DIR, cls), exist_ok=True)

    all_files = df.sort_values('final_conf', ascending=False)
    print(f"\n── Building Dataset1 (target: {TARGET_PER_CLASS}/class) ──")

    metadata_rows, split_rows = [], []

    for cls in CLASSES:
        cls_df  = all_files[all_files['final_label'] == cls]
        n_avail = len(cls_df)
        out_dir = os.path.join(OUT_DIR, cls)
        print(f"\n  [{cls}] available={n_avail}, target={TARGET_PER_CLASS}")

        if n_avail >= TARGET_PER_CLASS:
            selected = cls_df.head(TARGET_PER_CLASS)
            for _, row in tqdm(selected.iterrows(), total=len(selected),
                               desc=f"  Copying {cls}", leave=False):
                src = os.path.join(PREPROCESSED, row['filename'])
                dst = os.path.join(out_dir, row['filename'])
                if not os.path.exists(dst) and os.path.exists(src):
                    shutil.copy2(src, dst)
                metadata_rows.append({'filename': row['filename'], 'class': cls,
                                      'is_augmented': False, 'aug_type': '',
                                      'confidence': row['final_conf'], 'tier': row['tier']})
            print(f"    → Copied top {TARGET_PER_CLASS} from {n_avail}")
        else:
            needed     = TARGET_PER_CLASS - n_avail
            aug_factor = min(int(np.ceil(needed / max(n_avail, 1))), MAX_AUG_FACTOR)
            print(f"    → {n_avail} originals + {needed} augmented needed (up to {aug_factor}x)")

            orig_list = []
            for _, row in cls_df.iterrows():
                src = os.path.join(PREPROCESSED, row['filename'])
                if os.path.exists(src):
                    dst = os.path.join(out_dir, row['filename'])
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                    orig_list.append(row['filename'])
                    metadata_rows.append({'filename': row['filename'], 'class': cls,
                                          'is_augmented': False, 'aug_type': '',
                                          'confidence': row['final_conf'], 'tier': row['tier']})

            aug_count = 0
            aug_pool  = (list(AUG_TYPES) * (aug_factor + 2))
            file_cycle = (orig_list * (aug_factor + 3))
            random.shuffle(file_cycle)

            pbar = tqdm(total=needed, desc=f"  Augmenting {cls}", leave=False)
            for i, fname in enumerate(file_cycle):
                if aug_count >= needed:
                    break
                aug_type = aug_pool[i % len(aug_pool)]
                src_path = os.path.join(PREPROCESSED, fname)
                if not os.path.exists(src_path):
                    continue
                try:
                    y, _ = sf.read(src_path, always_2d=False)
                    y = y.astype(np.float32)
                    if y.ndim > 1: y = y.mean(axis=1)
                    y_aug  = augment(y, aug_type)
                    stem   = os.path.splitext(fname)[0]
                    aug_fname = f"{stem}_aug_{aug_type}_{aug_count}.wav"
                    out_path  = os.path.join(out_dir, aug_fname)
                    if not os.path.exists(out_path):
                        y_out = y_aug.astype(np.float32)
                        if y_out.ndim > 1: y_out = y_out.mean(axis=1)
                        sf.write(out_path, y_out, SR, subtype='PCM_16')
                    metadata_rows.append({'filename': aug_fname, 'class': cls,
                                          'is_augmented': True, 'aug_type': aug_type,
                                          'confidence': 0.0, 'tier': 'AUGMENTED'})
                    aug_count += 1
                    pbar.update(1)
                except Exception:
                    pass
            pbar.close()
            final_count = len([f for f in os.listdir(out_dir) if f.endswith('.wav')])
            print(f"    → {len(orig_list)} originals + {aug_count} augmented = {final_count} total")

    # Splits
    print("\n── Creating Train/Val/Test Splits (70/15/15) ──")
    for cls in CLASSES:
        cls_files = [f for f in os.listdir(os.path.join(OUT_DIR, cls)) if f.endswith('.wav')]
        random.shuffle(cls_files)
        n = len(cls_files)
        n_train = int(0.70 * n)
        n_val   = int(0.15 * n)
        for i, fname in enumerate(cls_files):
            split = 'train' if i < n_train else ('val' if i < n_train + n_val else 'test')
            split_rows.append({'filename': fname, 'class': cls, 'split': split})

    meta_df  = pd.DataFrame(metadata_rows)
    split_df = pd.DataFrame(split_rows)
    meta_df.to_csv(os.path.join(OUT_DIR, 'metadata.csv'), index=False)
    split_df.to_csv(os.path.join(OUT_DIR, 'splits.csv'), index=False)

    print("\n═══════════════════════════════════════")
    print("       FINAL Dataset1 Summary")
    print("═══════════════════════════════════════")
    total = 0
    for cls in CLASSES:
        count = len([f for f in os.listdir(os.path.join(OUT_DIR, cls)) if f.endswith('.wav')])
        total += count
        cls_split = split_df[split_df['class'] == cls]
        tr = (cls_split['split'] == 'train').sum()
        vl = (cls_split['split'] == 'val').sum()
        ts = (cls_split['split'] == 'test').sum()
        print(f"  {cls:15s}: {count:4d}  (train={tr}, val={vl}, test={ts})")
    print(f"  {'TOTAL':15s}: {total:4d} files")
    print("═══════════════════════════════════════")
    print(f"\n  Output → {OUT_DIR}")


if __name__ == '__main__':
    print("Loading labels ...")
    clap_df = pd.read_csv(CLAP_F)
    heur_df = pd.read_csv(HEUR_F)
    df_fused = fuse_labels(clap_df, heur_df)
    build_dataset(df_fused)
    print("\n✅  Dataset1 complete!")

