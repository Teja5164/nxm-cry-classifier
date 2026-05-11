"""
Script 6: Advanced label fusion + class balancing + augmentation → MainDataset

PIPELINE:
  1. Weighted ensemble fusion of CLAP + heuristic per-class score vectors
     - Per-class CLAP trust: hungry=0.30 (CLAP only predicted 18), tired=0.70
     - Heuristic normalized to unit-sum probability distribution
  2. Confidence tier filtering (HIGH / MEDIUM / DISCARD)
  3. Class balancing:
       - Majority classes: select top-confidence samples (max 800)
       - Minority classes: augment with 8 variation types to reach 800
  4. Train / Val / Test stratified split  (70 / 15 / 15)
  5. Metadata CSV + quality report saved alongside dataset

OUTPUT:
  datasets/{class}/*.wav   (~800 per class, balanced)
  datasets/metadata.csv
  datasets/splits.csv
"""

from pathlib import Path
import os, shutil, random, json, warnings
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
MAIN_DATASET = str(_ROOT / 'datasets')

CLASSES = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
SR = 22050
TARGET_PER_CLASS = 800   # balanced target
MAX_AUG_FACTOR   = 10    # max augmented copies per original file

# ── Per-class CLAP weight (0=trust heuristic, 1=trust CLAP fully) ──────────────
# hungry : CLAP predicted only 18/6586 → heavily distrust CLAP for this class
# tired  : CLAP predicted 3517 → most reliable signal from CLAP
CLAP_CLASS_W = np.array([0.55, 0.55, 0.55, 0.28, 0.72])  # bp, burp, discomf, hungry, tired
HEUR_CLASS_W = 1.0 - CLAP_CLASS_W


# ── 1. Label Fusion ────────────────────────────────────────────────────────────
def fuse_labels(clap_df, heur_df):
    """Weighted ensemble fusion using all per-class score vectors."""
    clap_cols = [f'clap_prob_{c}'  for c in CLASSES]
    heur_cols = [f'heur_score_{c}' for c in CLASSES]

    df = pd.merge(
        clap_df[['filename', 'clap_label', 'clap_confidence'] + clap_cols],
        heur_df[['filename', 'heuristic_label', 'heuristic_confidence'] + heur_cols],
        on='filename', how='inner'
    )

    # Extract score matrices
    clap_mat = df[clap_cols].values.astype(np.float32)   # already softmax → sums to 1
    heur_mat = df[heur_cols].values.astype(np.float32)

    # Normalize heuristic scores row-wise to unit-sum probability
    heur_row_sum = heur_mat.sum(axis=1, keepdims=True).clip(min=1e-8)
    heur_mat = heur_mat / heur_row_sum

    # Weighted ensemble
    final_scores = clap_mat * CLAP_CLASS_W + heur_mat * HEUR_CLASS_W  # shape (N, 5)

    # Final label + confidence
    label_idx    = final_scores.argmax(axis=1)
    final_labels = [CLASSES[i] for i in label_idx]
    final_confs  = final_scores.max(axis=1)

    # Agreement bonus: both methods agree → tier HIGH
    agree = (df['clap_label'].values == df['heuristic_label'].values)
    tier  = np.where(agree, 'HIGH',
            np.where(final_confs >= 0.48, 'MEDIUM', 'DISCARD'))

    df['final_label'] = final_labels
    df['final_conf']  = final_confs.round(4)
    df['tier']        = tier
    df['agree']       = agree

    print("\n── Label Fusion Results ──")
    print(f"  HIGH (both agree)  : {(tier=='HIGH').sum():5d}")
    print(f"  MEDIUM (confident) : {(tier=='MEDIUM').sum():5d}")
    print(f"  DISCARD (low conf) : {(tier=='DISCARD').sum():5d}")
    print("\nFused label distribution:")
    print(pd.Series(final_labels).value_counts().to_string())

    df.to_csv(FINAL_LABELS, index=False)
    print(f"\nFull fusion labels saved → {FINAL_LABELS}")
    return df


# ── 2. Augmentation helpers ────────────────────────────────────────────────────
AUG_TYPES = ['ts_slow', 'ts_fast', 'ps_down', 'ps_up',
             'noise', 'vol_up', 'vol_down', 'ts_ps']

def augment(y, aug_type):
    """Apply one augmentation to waveform y (22050 Hz mono float32)."""
    try:
        if aug_type == 'ts_slow':   # slower → more content
            return librosa.effects.time_stretch(y, rate=0.85)
        elif aug_type == 'ts_fast': # faster → more compact
            return librosa.effects.time_stretch(y, rate=1.15)
        elif aug_type == 'ps_down': # lower pitch
            return librosa.effects.pitch_shift(y, sr=SR, n_steps=-2)
        elif aug_type == 'ps_up':   # higher pitch
            return librosa.effects.pitch_shift(y, sr=SR, n_steps=2)
        elif aug_type == 'noise':   # add background noise (SNR ~25 dB)
            noise = np.random.randn(len(y)).astype(np.float32) * 0.006
            return np.clip(y + noise, -1.0, 1.0)
        elif aug_type == 'vol_up':  # louder
            return np.clip(y * 1.35, -1.0, 1.0)
        elif aug_type == 'vol_down':# softer
            return (y * 0.65).astype(np.float32)
        elif aug_type == 'ts_ps':   # combined stretch + pitch
            y2 = librosa.effects.time_stretch(y, rate=0.90)
            return librosa.effects.pitch_shift(y2, sr=SR, n_steps=1)
    except Exception:
        return y  # fallback: return original on error
    return y


def save_augmented(y_aug, orig_fname, aug_type, out_dir):
    """Save augmented audio; return saved filename."""
    stem = os.path.splitext(orig_fname)[0]
    aug_fname = f"{stem}_aug_{aug_type}.wav"
    out_path  = os.path.join(out_dir, aug_fname)
    y_out = y_aug.astype(np.float32)
    # Ensure mono
    if y_out.ndim > 1:
        y_out = y_out.mean(axis=1)
    sf.write(out_path, y_out, SR, subtype='PCM_16')
    return aug_fname


# ── 3. Build Balanced Dataset ──────────────────────────────────────────────────
def build_dataset(df):
    """Select, augment, and copy files to MainDataset class folders."""
    for cls in CLASSES:
        os.makedirs(os.path.join(MAIN_DATASET, cls), exist_ok=True)

    # Filter out DISCARDs; sort by confidence descending
    keep = df[df['tier'] != 'DISCARD'].sort_values('final_conf', ascending=False)
    print(f"\n── Building Dataset (target: {TARGET_PER_CLASS}/class) ──")

    metadata_rows = []
    split_rows    = []

    for cls in CLASSES:
        cls_df  = keep[keep['final_label'] == cls]
        n_avail = len(cls_df)
        out_dir = os.path.join(MAIN_DATASET, cls)
        print(f"\n  [{cls}] available={n_avail}, target={TARGET_PER_CLASS}")

        # ── Subsample if over-represented ──────────────────────────────────
        if n_avail >= TARGET_PER_CLASS:
            selected = cls_df.head(TARGET_PER_CLASS)  # top-confidence
            for _, row in tqdm(selected.iterrows(), total=len(selected),
                               desc=f"  Copying {cls}", leave=False):
                src = os.path.join(PREPROCESSED, row['filename'])
                dst = os.path.join(out_dir, row['filename'])
                if not os.path.exists(dst) and os.path.exists(src):
                    shutil.copy2(src, dst)
                metadata_rows.append({'filename': row['filename'], 'class': cls,
                                      'is_augmented': False, 'aug_type': '',
                                      'confidence': row['final_conf'], 'tier': row['tier']})
            print(f"    → Copied top {TARGET_PER_CLASS} (undersampled from {n_avail})")

        # ── Augment if under-represented ───────────────────────────────────
        else:
            needed     = TARGET_PER_CLASS - n_avail
            aug_factor = min(int(np.ceil(needed / max(n_avail, 1))), MAX_AUG_FACTOR)
            print(f"    → {n_avail} originals, need {needed} augmented "
                  f"(up to {aug_factor}x per file)")

            # Copy all originals first
            orig_list = []
            for _, row in cls_df.iterrows():
                src = os.path.join(PREPROCESSED, row['filename'])
                dst = os.path.join(out_dir, row['filename'])
                if os.path.exists(src):
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                    orig_list.append(row['filename'])
                    metadata_rows.append({'filename': row['filename'], 'class': cls,
                                          'is_augmented': False, 'aug_type': '',
                                          'confidence': row['final_conf'], 'tier': row['tier']})

            # Augment until we hit target
            aug_count = 0
            aug_pool  = list(AUG_TYPES) * aug_factor  # rotation of aug types
            file_cycle = orig_list * (aug_factor + 1)
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
                    if y.ndim > 1:
                        y = y.mean(axis=1)
                    y_aug = augment(y, aug_type)

                    # Check augmented file doesn't already exist
                    stem = os.path.splitext(fname)[0]
                    aug_fname = f"{stem}_aug_{aug_type}_{aug_count}.wav"
                    out_path  = os.path.join(out_dir, aug_fname)
                    if not os.path.exists(out_path):
                        y_out = y_aug.astype(np.float32)
                        if y_out.ndim > 1:
                            y_out = y_out.mean(axis=1)
                        sf.write(out_path, y_out, SR, subtype='PCM_16')
                    metadata_rows.append({'filename': aug_fname, 'class': cls,
                                          'is_augmented': True, 'aug_type': aug_type,
                                          'confidence': 0.0, 'tier': 'AUGMENTED'})
                    aug_count += 1
                    pbar.update(1)
                except Exception as e:
                    pass  # skip failed augmentations silently
            pbar.close()
            final_count = len([f for f in os.listdir(out_dir) if f.endswith('.wav')])
            print(f"    → {len(orig_list)} originals + {aug_count} augmented = {final_count} total")

    # ── 4. Train / Val / Test Splits ──────────────────────────────────────
    print("\n── Creating Train/Val/Test Splits (70/15/15) ──")
    for cls in CLASSES:
        cls_files = [f for f in os.listdir(os.path.join(MAIN_DATASET, cls))
                     if f.endswith('.wav')]
        random.shuffle(cls_files)
        n = len(cls_files)
        n_train = int(0.70 * n)
        n_val   = int(0.15 * n)
        for i, fname in enumerate(cls_files):
            if i < n_train:
                split = 'train'
            elif i < n_train + n_val:
                split = 'val'
            else:
                split = 'test'
            split_rows.append({'filename': fname, 'class': cls, 'split': split})

    # ── 5. Save metadata & splits ─────────────────────────────────────────
    meta_df  = pd.DataFrame(metadata_rows)
    split_df = pd.DataFrame(split_rows)
    meta_df.to_csv(os.path.join(MAIN_DATASET, 'metadata.csv'), index=False)
    split_df.to_csv(os.path.join(MAIN_DATASET, 'splits.csv'), index=False)

    print("\n═══════════════════════════════════════")
    print("      FINAL MainDataset Summary")
    print("═══════════════════════════════════════")
    total = 0
    for cls in CLASSES:
        cls_dir = os.path.join(MAIN_DATASET, cls)
        count   = len([f for f in os.listdir(cls_dir) if f.endswith('.wav')])
        total  += count
        cls_split = split_df[split_df['class'] == cls]
        tr = (cls_split['split'] == 'train').sum()
        vl = (cls_split['split'] == 'val').sum()
        ts = (cls_split['split'] == 'test').sum()
        print(f"  {cls:15s}: {count:4d} total  (train={tr}, val={vl}, test={ts})")
    print(f"  {'TOTAL':15s}: {total:4d} files")
    print("═══════════════════════════════════════")
    print(f"\n  metadata.csv → {MAIN_DATASET}\\metadata.csv")
    print(f"  splits.csv   → {MAIN_DATASET}\\splits.csv")


# ── Main ───────────────────────────────────────────────────────────────────────
def run():
    print("Loading CLAP labels ...")
    clap_df = pd.read_csv(CLAP_F)
    print(f"  {len(clap_df)} entries, columns: {list(clap_df.columns)}")

    print("Loading heuristic labels ...")
    heur_df = pd.read_csv(HEUR_F)
    print(f"  {len(heur_df)} entries, columns: {list(heur_df.columns)}")

    print("\nStep 1: Fusing labels ...")
    df_fused = fuse_labels(clap_df, heur_df)

    print("\nStep 2-5: Building balanced MainDataset ...")
    build_dataset(df_fused)

    print("\n✅  All done! MainDataset is ready at:", MAIN_DATASET)


if __name__ == '__main__':
    run()



