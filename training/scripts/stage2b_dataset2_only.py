"""
Stage 2b: Baseline CNN — Dataset2 ONLY (GPU version)
=====================================================
Same architecture as stage2_baseline_cnn.py but:
  - Trains ONLY on Dataset2 (7,500 samples, 1500/class)
  - Automatically uses CUDA GPU if available
  - Larger batch size on GPU for speed (64)
  - Saves to models/baseline_cnn/dataset2/
Run AFTER installing torch+cu124.
"""

from pathlib import Path
import os, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

CLASSES      = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
N_CLASSES    = 5
BATCH_SIZE   = 64        # larger on GPU
MAX_EPOCHS   = 100
LR           = 3e-4
WEIGHT_DECAY = 0.01
PATIENCE     = 15
LABEL_SMOOTH = 0.1

_ROOT = Path(__file__).resolve().parent.parent.parent  # ML_pipeline/
FEAT_ROOT    = str(_ROOT / 'training' / 'features')
MODEL_ROOT   = str(_ROOT / 'models' / 'baseline_cnn')
RESULT_ROOT  = str(_ROOT / 'results' / 'baseline_cnn')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True

# ── SpecAugment ───────────────────────────────────────────────────────────────
class SpecAugment(nn.Module):
    def __init__(self, freq_mask=15, time_mask=40, n_freq_masks=2, n_time_masks=2):
        super().__init__()
        self.freq_mask = freq_mask; self.time_mask = time_mask
        self.n_freq = n_freq_masks; self.n_time = n_time_masks

    def forward(self, x):
        if not self.training: return x
        B, C, F, T = x.shape
        x = x.clone()
        for _ in range(self.n_freq):
            f = random.randint(0, self.freq_mask); f0 = random.randint(0, F - f)
            x[:, :, f0:f0+f, :] = 0.0
        for _ in range(self.n_time):
            t = random.randint(0, self.time_mask); t0 = random.randint(0, T - t)
            x[:, :, :, t0:t0+t] = 0.0
        return x

# ── ResNet Block ──────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.Dropout2d(dropout),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch))
        self.relu = nn.ReLU(True)

    def forward(self, x): return self.relu(self.block(x) + x)

# ── Baseline CNN ──────────────────────────────────────────────────────────────
class BaselineCNN(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.augment = SpecAugment()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True))
        self.block1 = nn.Sequential(ResBlock(32), nn.MaxPool2d(2,2))
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            ResBlock(64), nn.MaxPool2d(2,2))
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            ResBlock(128), nn.MaxPool2d(2,2))
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            ResBlock(256), nn.AdaptiveAvgPool2d((4,4)))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(256*16, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(128, n_classes))

    def forward(self, x):
        x = self.augment(x)
        x = self.stem(x); x = self.block1(x); x = self.block2(x)
        x = self.block3(x); x = self.block4(x)
        return self.head(x)

# ── Label Smoothing Loss ──────────────────────────────────────────────────────
class LabelSmoothingCE(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing; self.cls = classes

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_val  = self.smoothing / (self.cls - 1)
        one_hot = torch.full_like(pred, smooth_val)
        one_hot.scatter_(1, target.unsqueeze(1), confidence)
        log_prob = F.log_softmax(pred, dim=1)
        return -(one_hot * log_prob).sum(dim=1).mean()

# ── Dataset ───────────────────────────────────────────────────────────────────
class MelDataset(Dataset):
    def __init__(self, mel, labels):
        self.mel = torch.FloatTensor(mel)
        self.labels = torch.LongTensor(labels)

    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.mel[i], self.labels[i]

# ── Training helpers ──────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, scheduler, train=True):
    model.train(train)
    total_loss = correct = total = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x); loss = criterion(out, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item(); total += len(y)
    if train and scheduler: scheduler.step()
    return total_loss / total, correct / total

def plot_curves(train_losses, val_losses, train_accs, val_accs, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    ax1.plot(train_losses, label='Train'); ax1.plot(val_losses, label='Val')
    ax1.set_title('Loss'); ax1.legend()
    ax2.plot(train_accs, label='Train'); ax2.plot(val_accs, label='Val')
    ax2.set_title('Accuracy'); ax2.legend()
    plt.tight_layout(); plt.savefig(path, dpi=100); plt.close()

def plot_cm(cm, path):
    fig, ax = plt.subplots(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES,
                cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout(); plt.savefig(path, dpi=100); plt.close()

# ── Main ──────────────────────────────────────────────────────────────────────
def train_dataset(ds_name):
    print(f"\n{'='*60}")
    print(f"Training Baseline CNN on {ds_name}")
    print(f"{'='*60}")

    feat_dir   = os.path.join(FEAT_ROOT, ds_name)
    model_dir  = os.path.join(MODEL_ROOT, ds_name)
    result_dir = os.path.join(RESULT_ROOT, ds_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    print("Loading features...", end=' ', flush=True)
    mel    = np.load(os.path.join(feat_dir, 'mel_specs.npy'))
    labels = np.load(os.path.join(feat_dir, 'labels.npy'))
    print(f"mel={mel.shape} labels={labels.shape}")

    # Split 70/15/15
    idx   = np.arange(len(labels))
    tr_i, te_i = train_test_split(idx, test_size=0.15, stratify=labels, random_state=SEED)
    tr_i, va_i = train_test_split(tr_i, test_size=0.176, stratify=labels[tr_i], random_state=SEED)
    print(f"Split → train={len(tr_i)} val={len(va_i)} test={len(te_i)}")

    # Weighted sampler for class balance
    class_counts = np.bincount(labels[tr_i])
    sample_weights = (1.0 / class_counts)[labels[tr_i]]
    sampler = WeightedRandomSampler(sample_weights, len(tr_i), replacement=True)

    tr_ds = MelDataset(mel[tr_i], labels[tr_i])
    va_ds = MelDataset(mel[va_i], labels[va_i])
    te_ds = MelDataset(mel[te_i], labels[te_i])

    nw = 4 if DEVICE.type == 'cuda' else 2
    tr_ld = DataLoader(tr_ds, BATCH_SIZE, sampler=sampler, num_workers=nw, pin_memory=True)
    va_ld = DataLoader(va_ds, BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=True)
    te_ld = DataLoader(te_ds, BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=True)

    model     = BaselineCNN(N_CLASSES).to(DEVICE)
    criterion = LabelSmoothingCE(N_CLASSES, LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,}")

    best_val_acc = 0; patience_cnt = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for ep in range(1, MAX_EPOCHS+1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, tr_ld, criterion, optimizer, scheduler, train=True)
        va_loss, va_acc = run_epoch(model, va_ld, criterion, None, None, train=False)
        elapsed = time.time() - t0

        train_losses.append(tr_loss); val_losses.append(va_loss)
        train_accs.append(tr_acc); val_accs.append(va_acc)

        print(f"Ep {ep:3d}/{MAX_EPOCHS} | {elapsed:.1f}s | "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}", flush=True)

        if va_acc > best_val_acc:
            best_val_acc = va_acc; patience_cnt = 0
            torch.save({'epoch': ep, 'model_state': model.state_dict(),
                        'val_acc': va_acc, 'optimizer_state': optimizer.state_dict()},
                       os.path.join(model_dir, 'best_model.pth'))
            print(f"  ✓ Saved best model (val_acc={va_acc:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"Early stopping at epoch {ep} (patience={PATIENCE})")
                break

    # Load best and evaluate on test
    ckpt = torch.load(os.path.join(model_dir, 'best_model.pth'), map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for x, y in te_ld:
            x = x.to(DEVICE)
            preds = model(x).argmax(1).cpu().numpy()
            all_preds.extend(preds); all_true.extend(y.numpy())

    from collections import Counter
    pred_dist = Counter(all_preds)
    print("\nPrediction distribution (test):")
    for i, cls in enumerate(CLASSES):
        cnt = pred_dist.get(i, 0)
        print(f"  {cls:15s}: {cnt:4d} ({cnt/len(all_preds)*100:.1f}%)")

    report = classification_report(all_true, all_preds, target_names=CLASSES, digits=4)
    print("\n" + report)

    # Save artifacts
    plot_curves(train_losses, val_losses, train_accs, val_accs,
                os.path.join(result_dir, f'{ds_name}_training_curves.png'))
    cm = confusion_matrix(all_true, all_preds)
    plot_cm(cm, os.path.join(result_dir, f'{ds_name}_confusion_matrix.png'))
    with open(os.path.join(result_dir, f'{ds_name}_classification_report.txt'), 'w') as f:
        f.write(report)

    from sklearn.metrics import accuracy_score, f1_score
    metrics = {
        'dataset': ds_name, 'best_val_acc': float(best_val_acc),
        'test_acc': float(accuracy_score(all_true, all_preds)),
        'macro_f1': float(f1_score(all_true, all_preds, average='macro')),
        'weighted_f1': float(f1_score(all_true, all_preds, average='weighted'))}
    with open(os.path.join(result_dir, f'{ds_name}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ {ds_name} done | test_acc={metrics['test_acc']:.4f} | macro_f1={metrics['macro_f1']:.4f}")
    return metrics

if __name__ == '__main__':
    m = train_dataset('dataset2')
    print(f"\nFinal: test_acc={m['test_acc']:.4f}, macro_f1={m['macro_f1']:.4f}")
