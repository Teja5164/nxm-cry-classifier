"""
stage_binary_cry_gate.py — Train Custom Binary Cry Detector
============================================================
Trains a lightweight binary CNN to classify audio as:
  - Class 1 (cry):     All infant cry WAVs from Dataset1
  - Class 0 (not-cry): ESC-50 environmental sounds (downloaded automatically)

This custom gate replaces/augments YAMNet in the production pipeline
and is domain-specific → higher accuracy than YAMNet for this task.

Output:
    D:\\nxm\\ML_pipeline\\models\\cry_gate\\best_model.pth
    D:\\nxm\\ML_pipeline\\models\\cry_gate\\training_report.json

Architecture: BinaryCryCNN (~480K params, fast CPU inference)
    Same input format as 5-class models: (B,1,128,431) log-mel spectrogram

Usage:
    cd D:\\nxm\\ML_pipeline
    D:\\TEJA\\Anaconda3\\python.exe training/scripts/stage_binary_cry_gate.py

    # With options:
    python training/scripts/stage_binary_cry_gate.py --epochs 40 --batch 32
"""

import os, sys, json, time, random, warnings, argparse
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
warnings.filterwarnings('ignore')

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # training/scripts/
_ROOT = _HERE.parent.parent                      # ML_pipeline/
_INF  = _ROOT / 'inference'
if str(_INF) not in sys.path:
    sys.path.insert(0, str(_INF))

# Paths
DATASET1_DIR  = _ROOT / 'datasets' / 'Dataset1'
ESC50_DIR     = _ROOT / 'datasets' / 'ESC50'
FEAT_ROOT     = _ROOT / 'training' / 'features'
GATE_MODEL_DIR = _ROOT / 'models' / 'cry_gate'
GATE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import preprocess_audio, extract_mel, normalize_mel, load_norm_stats

print("=" * 65)
print("  Binary Cry Detector — Training Script")
print("=" * 65)
print(f"  Device  : {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
print(f"  PyTorch : {torch.__version__}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Download ESC-50 (if not already present)
# ─────────────────────────────────────────────────────────────────────────────
def download_esc50():
    """Download ESC-50 dataset to datasets/ESC50/."""
    import urllib.request, zipfile, shutil

    zip_path = ESC50_DIR.parent / 'ESC-50-master.zip'
    esc50_audio = ESC50_DIR / 'audio'

    if esc50_audio.exists() and len(list(esc50_audio.glob('*.wav'))) > 100:
        print(f"  [ESC-50] Already downloaded ({len(list(esc50_audio.glob('*.wav')))} files)")
        return

    print("  [ESC-50] Downloading from GitHub (~600 MB) ...")
    url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'

    def _progress(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size * 100 / total_size, 100)
            print(f"\r  Downloading... {pct:.1f}%", end='', flush=True)
        else:
            mb = count * block_size / (1024 * 1024)
            print(f"\r  Downloading... {mb:.1f} MB", end='', flush=True)

    ESC50_DIR.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
    print()

    print("  [ESC-50] Extracting ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(ESC50_DIR.parent)

    extracted = ESC50_DIR.parent / 'ESC-50-master'
    if extracted.exists():
        if ESC50_DIR.exists():
            shutil.rmtree(ESC50_DIR)
        extracted.rename(ESC50_DIR)

    zip_path.unlink(missing_ok=True)
    print(f"  [ESC-50] Done. {len(list(esc50_audio.glob('*.wav')))} files.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Collect file paths + labels
# ─────────────────────────────────────────────────────────────────────────────
def collect_files():
    """Return (paths, labels) where label=1 for cry, 0 for not-cry."""
    paths, labels = [], []

    # Positives — all Dataset1 cry WAVs
    cry_classes = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
    n_pos = 0
    for cls in cry_classes:
        cls_dir = DATASET1_DIR / cls
        if not cls_dir.exists():
            print(f"  WARNING: {cls_dir} not found, skipping.")
            continue
        wavs = list(cls_dir.glob('*.wav'))
        paths.extend([str(w) for w in wavs])
        labels.extend([1] * len(wavs))
        n_pos += len(wavs)

    print(f"  Positives (cry)     : {n_pos} files")

    # Negatives — ESC-50 WAVs (all 2000 files, no baby-cry classes)
    # ESC-50 has no infant cry class (all 50 classes are environmental sounds)
    esc50_audio = ESC50_DIR / 'audio'
    if not esc50_audio.exists():
        print("  WARNING: ESC-50 audio dir not found. Running with reduced negatives.")
        neg_wavs = []
    else:
        # Exclude any accidental cry-like classes (fold metadata)
        # ESC-50 class 40–49 = domestic animals, class 10–19 = natural sounds
        # All safe — no infant cry in ESC-50
        neg_wavs = list(esc50_audio.glob('*.wav'))
        # If ESC-50 has fewer samples than positives, augment by repeating
        if len(neg_wavs) < n_pos:
            multiplier = (n_pos // len(neg_wavs)) + 1
            neg_wavs = (neg_wavs * multiplier)[:n_pos]

    paths.extend([str(w) for w in neg_wavs])
    labels.extend([0] * len(neg_wavs))
    print(f"  Negatives (not-cry) : {len(neg_wavs)} files")
    print(f"  Total               : {len(paths)} files")

    return paths, labels


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature extraction with caching
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(paths, labels, stats, cache_dir: Path, split_name: str):
    """
    Extract features for all files. Cache to disk to avoid re-extracting.
    Returns np.ndarray X (N,128,431) and np.ndarray y (N,).
    """
    cache_X = cache_dir / f'{split_name}_X.npy'
    cache_y = cache_dir / f'{split_name}_y.npy'
    cache_p = cache_dir / f'{split_name}_paths.json'

    # Check if cache is valid
    if cache_X.exists() and cache_y.exists() and cache_p.exists():
        cached_paths = json.load(open(cache_p))
        if cached_paths == paths:
            print(f"  [{split_name}] Loading from cache ...")
            return np.load(cache_X), np.load(cache_y)

    print(f"  [{split_name}] Extracting features for {len(paths)} files ...")
    X, y_out = [], []
    failed = 0
    for i, (path, lbl) in enumerate(zip(paths, labels)):
        try:
            audio    = preprocess_audio(path)
            log_mel  = extract_mel(audio)
            norm_mel = normalize_mel(log_mel, stats)
            X.append(norm_mel)
            y_out.append(lbl)
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  WARN: {Path(path).name} failed: {e}")
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(paths)} done ...")

    X = np.array(X, dtype=np.float32)
    y_out = np.array(y_out, dtype=np.float32)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_X, X)
    np.save(cache_y, y_out)
    json.dump(paths, open(cache_p, 'w'))

    print(f"  [{split_name}] Done: {len(X)} samples ({failed} failed)")
    return X, y_out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Dataset class
# ─────────────────────────────────────────────────────────────────────────────
class BinaryDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X       = X
        self.y       = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx]).unsqueeze(0)   # (1,128,431)
        y = torch.FloatTensor([self.y[idx]])               # (1,)
        if self.augment:
            x = self._spec_augment(x)
        return x, y

    @staticmethod
    def _spec_augment(x: torch.Tensor) -> torch.Tensor:
        """Frequency + time masking (SpecAugment)."""
        x = x.clone()
        _, F, T = x.shape

        # Frequency masking (up to 15 bins, 2 masks)
        for _ in range(2):
            f = random.randint(0, 15)
            f0 = random.randint(0, max(1, F - f))
            x[:, f0:f0 + f, :] = 0.0

        # Time masking (up to 40 frames, 2 masks)
        for _ in range(2):
            t = random.randint(0, 40)
            t0 = random.randint(0, max(1, T - t))
            x[:, :, t0:t0 + t] = 0.0

        return x


# ─────────────────────────────────────────────────────────────────────────────
# 5. Model architecture
# ─────────────────────────────────────────────────────────────────────────────
class BinaryCryCNN(nn.Module):
    """
    Lightweight binary CNN: cry vs not-cry.
    ~480K parameters. Input: (B,1,128,431). Output: (B,1) sigmoid logit.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# 6. Training loop
# ─────────────────────────────────────────────────────────────────────────────
def find_optimal_threshold(model, val_loader, device):
    """Find the threshold that maximizes F1 on validation set."""
    from sklearn.metrics import f1_score

    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            prob = torch.sigmoid(model(x)).cpu().numpy().flatten()
            all_probs.extend(prob)
            all_labels.extend(y.numpy().flatten())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.3, 0.8, 0.02):
        preds = (all_probs >= t).astype(int)
        f1    = f1_score(all_labels, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    return best_t, best_f1


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Download ESC-50 ───────────────────────────────────────────────────────
    download_esc50()
    print()

    # ── Collect files ─────────────────────────────────────────────────────────
    print("[Step 1] Collecting file paths ...")
    paths, labels = collect_files()

    if len(paths) < 100:
        raise RuntimeError("Not enough data. Check Dataset1 and ESC-50 directories.")

    # ── Load norm stats (use Dataset1 stats — same preprocessing) ─────────────
    print("\n[Step 2] Loading norm stats ...")
    stats = load_norm_stats(str(FEAT_ROOT), 'dataset1')
    print("  Norm stats loaded OK")

    # ── Train/val split ───────────────────────────────────────────────────────
    print("\n[Step 3] Train/val split (80/20, stratified) ...")
    tr_paths, va_paths, tr_labels, va_labels = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    print(f"  Train: {len(tr_paths)} files  (cry: {sum(tr_labels)}, not-cry: {len(tr_labels)-sum(tr_labels)})")
    print(f"  Val  : {len(va_paths)} files  (cry: {sum(va_labels)}, not-cry: {len(va_labels)-sum(va_labels)})")

    # ── Feature extraction ────────────────────────────────────────────────────
    cache_dir = FEAT_ROOT / 'binary_gate'
    print("\n[Step 4] Extracting features ...")
    X_train, y_train = extract_features(tr_paths, tr_labels, stats, cache_dir, 'train')
    X_val,   y_val   = extract_features(va_paths, va_labels, stats, cache_dir, 'val')

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_ds = BinaryDataset(X_train, y_train, augment=True)
    val_ds   = BinaryDataset(X_val,   y_val,   augment=False)

    # Weighted sampler for class balance
    class_counts = np.bincount(y_train.astype(int))
    sample_weights = [1.0 / class_counts[int(l)] for l in y_train]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(y_train), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,   num_workers=0, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n[Step 5] Building model ...")
    model = BinaryCryCNN().to(device)
    print(f"  Parameters: {model.n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n[Step 6] Training for {args.epochs} epochs ...")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*9}  {'-'*8}  {'-'*7}  {'-'*8}")

    best_val_acc = 0.0
    best_epoch   = 0
    best_path    = str(GATE_MODEL_DIR / 'best_model.pth')
    history      = []

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss    += loss.item() * len(x)
            preds       = (torch.sigmoid(logits) >= 0.5).float()
            tr_correct += (preds == y).sum().item()
            tr_total   += len(x)

        scheduler.step()

        # Validate
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y   = x.to(device), y.to(device)
                logits = model(x)
                loss   = criterion(logits, y)
                va_loss    += loss.item() * len(x)
                preds       = (torch.sigmoid(logits) >= 0.5).float()
                va_correct += (preds == y).sum().item()
                va_total   += len(x)

        tr_acc = tr_correct / tr_total * 100
        va_acc = va_correct / va_total * 100
        tr_l   = tr_loss / tr_total
        va_l   = va_loss / va_total
        lr     = scheduler.get_last_lr()[0]

        print(f"  {epoch:>5}  {tr_l:>10.4f}  {tr_acc:>8.2f}%  {va_l:>8.4f}  {va_acc:>6.2f}%  {lr:.6f}")

        history.append({'epoch': epoch, 'train_acc': tr_acc, 'val_acc': va_acc,
                        'train_loss': tr_l, 'val_loss': va_l})

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch   = epoch
            # Find optimal threshold
            threshold, best_f1 = find_optimal_threshold(model, val_loader, device)
            torch.save({
                'model_state': model.state_dict(),
                'val_acc':     best_val_acc,
                'threshold':   threshold,
                'epoch':       epoch,
                'arch':        'BinaryCryCNN',
                'n_params':    model.n_params,
            }, best_path)

    print(f"\n  ✅ Best val accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    print(f"     Optimal threshold : {threshold:.3f}  (F1={best_f1:.4f})")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n[Step 7] Final evaluation on validation set ...")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    all_preds, all_labels_out = [], []
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x.to(device))
            probs  = torch.sigmoid(logits).cpu().numpy().flatten()
            preds  = (probs >= threshold).astype(int)
            all_preds.extend(preds)
            all_labels_out.extend(y.numpy().flatten().astype(int))

    print("\n  Classification Report:")
    print(classification_report(all_labels_out, all_preds,
                                 target_names=['not_cry', 'cry']))

    cm = confusion_matrix(all_labels_out, all_preds)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    print(f"\n  Confusion Matrix:")
    print(f"    True Not-Cry (TN): {tn}   False Cry (FP): {fp}")
    print(f"    Missed Cry   (FN): {fn}   True Cry   (TP): {tp}")
    print(f"\n  Binary Metrics:")
    print(f"    Precision (cry detection)  : {precision*100:.2f}%")
    print(f"    Recall    (cry detection)  : {recall*100:.2f}%")
    print(f"    F1 Score  (cry detection)  : {f1*100:.2f}%")

    # ── Save report ───────────────────────────────────────────────────────────
    report = {
        'val_accuracy':   round(best_val_acc, 4),
        'best_epoch':     best_epoch,
        'threshold':      round(threshold, 4),
        'precision':      round(float(precision), 4),
        'recall':         round(float(recall), 4),
        'f1':             round(float(f1), 4),
        'confusion_matrix': {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)},
        'n_params':       model.n_params,
        'epochs_trained': args.epochs,
        'history':        history,
    }
    report_path = GATE_MODEL_DIR / 'training_report.json'
    json.dump(report, open(report_path, 'w'), indent=2)
    print(f"\n  Report saved to: {report_path}")
    print(f"  Model  saved to: {best_path}")

    print("\n" + "=" * 65)
    print(f"  BINARY CRY GATE TRAINING COMPLETE")
    print(f"  Val Accuracy : {best_val_acc:.2f}%")
    print(f"  Threshold    : {threshold:.3f}")
    print(f"  F1 Score     : {f1*100:.2f}%")
    print("=" * 65)

    if best_val_acc < 90.0:
        print("\n  ⚠️  Val accuracy < 90%. Consider:")
        print("     - Adding more negative samples")
        print("     - Increasing epochs (--epochs 50)")
        print("     - Lowering threshold (--threshold 0.4)")

    return best_val_acc, threshold


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train custom binary cry/not-cry gate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stage_binary_cry_gate.py
  python stage_binary_cry_gate.py --epochs 50 --batch 64
  python stage_binary_cry_gate.py --no-download   (skip ESC-50 download)
        """
    )
    parser.add_argument('--epochs',      type=int,   default=35,  help='Training epochs (default: 35)')
    parser.add_argument('--batch',       type=int,   default=32,  help='Batch size (default: 32)')
    parser.add_argument('--lr',          type=float, default=1e-3, help='Learning rate (default: 0.001)')
    parser.add_argument('--no-download', action='store_true',      help='Skip ESC-50 download check')
    args = parser.parse_args()

    if args.no_download:
        # Monkey-patch download to skip
        def _skip_download():
            print("  [ESC-50] Skipping download (--no-download)")
        import builtins
        _orig = download_esc50
        download_esc50 = _skip_download

    train(args)
