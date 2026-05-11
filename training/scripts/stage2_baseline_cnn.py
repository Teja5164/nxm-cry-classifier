"""
Stage 2: Baseline CNN
======================
Architecture: 4-block 2D CNN with Residual connections on Mel Spectrograms
Input : (batch, 1, 128, 431)  — log-mel spectrogram
Output: 5-class softmax

Anti-overfitting:
  - SpecAugment (time + freq masking) applied on-the-fly during training
  - Dropout (0.3 in conv blocks, 0.5 in FC)
  - BatchNorm after every conv
  - Label Smoothing (0.1)
  - AdamW + weight_decay=0.01
  - CosineAnnealingWarmRestarts scheduler
  - Early stopping (patience=15)

Trains on both Dataset1 (4K) and Dataset2 (7.5K).
Saves best checkpoint + training curves + full classification report.
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

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
CLASSES    = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
N_CLASSES  = 5
BATCH_SIZE = 32
MAX_EPOCHS = 80
LR         = 3e-4
WEIGHT_DECAY = 0.01
PATIENCE   = 15          # early stopping
LABEL_SMOOTH = 0.1

_ROOT = Path(__file__).resolve().parent.parent.parent  # ML_pipeline/
FEAT_ROOT  = str(_ROOT / 'training' / 'features')
MODEL_ROOT = str(_ROOT / 'models' / 'baseline_cnn')
RESULT_ROOT= str(_ROOT / 'results' / 'baseline_cnn')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ── SpecAugment ───────────────────────────────────────────────────────────────
class SpecAugment(nn.Module):
    """Time and frequency masking applied to mel spectrogram during training."""
    def __init__(self, freq_mask=15, time_mask=40, n_freq_masks=2, n_time_masks=2):
        super().__init__()
        self.freq_mask  = freq_mask
        self.time_mask  = time_mask
        self.n_freq     = n_freq_masks
        self.n_time     = n_time_masks

    def forward(self, x):
        # x: (B, 1, F, T)
        if not self.training:
            return x
        B, C, F, T = x.shape
        x = x.clone()
        for _ in range(self.n_freq):
            f  = random.randint(0, self.freq_mask)
            f0 = random.randint(0, F - f)
            x[:, :, f0:f0+f, :] = 0.0
        for _ in range(self.n_time):
            t  = random.randint(0, self.time_mask)
            t0 = random.randint(0, T - t)
            x[:, :, :, t0:t0+t] = 0.0
        return x


# ── Residual Conv Block ───────────────────────────────────────────────────────
class ResBlock(nn.Module):
    """2D Residual block: Conv→BN→ReLU→Conv→BN + skip connection."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.drop  = nn.Dropout2d(0.1)
        self.skip  = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = out + self.skip(x)
        return F.relu(out)


# ── Baseline CNN Model ────────────────────────────────────────────────────────
class BaselineCNN(nn.Module):
    """
    4-stage residual CNN for mel spectrogram classification.
    Input  : (B, 1, 128, 431)
    Output : (B, 5)
    """
    def __init__(self, n_classes=N_CLASSES, dropout=0.5):
        super().__init__()
        self.spec_aug = SpecAugment(freq_mask=15, time_mask=40)

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Dropout2d(0.1)
        )

        # Residual stages
        self.stage1 = ResBlock(32,  64,  stride=2)
        self.stage2 = ResBlock(64,  128, stride=2)
        self.stage3 = ResBlock(128, 256, stride=2)
        self.stage4 = ResBlock(256, 256, stride=1)

        # Global Average Pooling
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)

        # Classifier head
        self.fc1  = nn.Linear(256, 128)
        self.bn_fc= nn.BatchNorm1d(128)
        self.fc2  = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.spec_aug(x)     # SpecAugment (only active in train mode)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).flatten(1)
        x = self.drop(x)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.drop(x)
        return self.fc2(x)       # raw logits; use CrossEntropy


# ── Dataset ───────────────────────────────────────────────────────────────────
class MelDataset(Dataset):
    def __init__(self, mel, labels, augment=False):
        self.mel    = torch.from_numpy(mel).float()
        self.labels = torch.from_numpy(labels).long()
        self.augment= augment

    def __len__(self):  return len(self.labels)

    def __getitem__(self, idx):
        x = self.mel[idx]   # (1, 128, 431)
        y = self.labels[idx]
        if self.augment:
            # MixUp-lite: random gain variation
            gain = random.uniform(0.85, 1.15)
            x = x * gain
        return x, y


# ── Label Smoothing Loss ──────────────────────────────────────────────────────
class LabelSmoothingCE(nn.Module):
    def __init__(self, n_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.n = n_classes

    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            smooth_label = torch.full_like(log_prob, self.smoothing / (self.n - 1))
            smooth_label.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return -(smooth_label * log_prob).sum(dim=-1).mean()


# ── Training utilities ────────────────────────────────────────────────────────
def get_weighted_sampler(labels):
    """Class-balanced sampling to handle any remaining imbalance."""
    counts  = np.bincount(labels, minlength=N_CLASSES).astype(float)
    weights = 1.0 / counts
    sample_w= weights[labels]
    return WeightedRandomSampler(torch.from_numpy(sample_w).float(),
                                  num_samples=len(labels), replacement=True)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0, 0, 0
    all_preds, all_labels  = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)
        total_loss += loss.item() * len(y)
        preds       = logits.argmax(1)
        correct    += (preds == y).sum().item()
        n          += len(y)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss / n, correct / n, np.array(all_preds), np.array(all_labels)


# ── Plot helpers ──────────────────────────────────────────────────────────────
def plot_curves(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'],   label='Val Loss')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history['train_acc'],  label='Train Acc')
    axes[1].plot(history['val_acc'],    label='Val Acc')
    axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Curves saved → {out_path}")


def plot_confusion(cm, labels, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {out_path}")


# ── Main Training Function ────────────────────────────────────────────────────
def train_on_dataset(ds_name):
    feat_dir  = os.path.join(FEAT_ROOT, ds_name)
    model_dir = os.path.join(MODEL_ROOT, ds_name)
    res_dir   = os.path.join(RESULT_ROOT, ds_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(res_dir,   exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Training Baseline CNN on {ds_name}")
    print(f"{'='*60}")

    # Load features
    mel    = np.load(os.path.join(feat_dir, 'mel_specs.npy'))   # (N, 1, 128, 431)
    labels = np.load(os.path.join(feat_dir, 'labels.npy'))      # (N,)
    print(f"  Loaded: mel={mel.shape}, labels={labels.shape}")

    # Stratified split: 70% train, 15% val, 15% test
    idx = np.arange(len(labels))
    idx_tv, idx_test = train_test_split(idx, test_size=0.15,
                                         stratify=labels, random_state=SEED)
    idx_train, idx_val = train_test_split(idx_tv, test_size=0.15/0.85,
                                           stratify=labels[idx_tv], random_state=SEED)
    print(f"  Split: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")

    # Datasets & loaders
    train_ds = MelDataset(mel[idx_train], labels[idx_train], augment=True)
    val_ds   = MelDataset(mel[idx_val],   labels[idx_val],   augment=False)
    test_ds  = MelDataset(mel[idx_test],  labels[idx_test],  augment=False)

    sampler    = get_weighted_sampler(labels[idx_train])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                               num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model     = BaselineCNN(n_classes=N_CLASSES, dropout=0.5).to(DEVICE)
    criterion = LabelSmoothingCE(N_CLASSES, smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")
    print(f"  Starting training (max {MAX_EPOCHS} epochs, patience={PATIENCE}) ...\n")

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    best_path    = os.path.join(model_dir, 'best_model.pth')
    no_improve   = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        vl_loss, vl_acc, _, _ = eval_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step(epoch)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        improved = vl_acc > best_val_acc
        if improved:
            best_val_acc = vl_acc
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_acc': vl_acc, 'optimizer_state': optimizer.state_dict()},
                       best_path)
            no_improve = 0
            tag = '  ← best'
        else:
            no_improve += 1
            tag = f'  (no improve {no_improve}/{PATIENCE})'

        elapsed = time.time() - t0
        print(f"  Ep {epoch:3d}/{MAX_EPOCHS}  "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f}  "
              f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.1f}s{tag}")

        if no_improve >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs)")
            break

    # ── Test Evaluation ──────────────────────────────────────────────────────
    print(f"\n  Loading best model (val_acc={best_val_acc:.4f}) for test evaluation ...")
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    _, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion, DEVICE)

    report = classification_report(test_labels, test_preds,
                                    target_names=CLASSES, digits=4)
    cm     = confusion_matrix(test_labels, test_preds)

    print(f"\n  ── Test Results [{ds_name}] ──")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"\n{report}")

    # ── Prediction distribution check ────────────────────────────────────────
    print("  Prediction distribution (test set):")
    for i, c in enumerate(CLASSES):
        pct = (test_preds == i).sum() / len(test_preds) * 100
        print(f"    {c:15s}: {(test_preds==i).sum():4d}  ({pct:.1f}%)")

    # ── Save results ──────────────────────────────────────────────────────────
    plot_curves(history, os.path.join(res_dir, f'{ds_name}_training_curves.png'))
    plot_confusion(cm, CLASSES,
                   os.path.join(res_dir, f'{ds_name}_confusion_matrix.png'),
                   f'Baseline CNN — {ds_name}')

    with open(os.path.join(res_dir, f'{ds_name}_classification_report.txt'), 'w') as f:
        f.write(f"Dataset: {ds_name}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Best Val Accuracy: {best_val_acc:.4f}\n\n")
        f.write(report)

    metrics = {
        'dataset': ds_name, 'test_accuracy': round(float(test_acc), 4),
        'best_val_accuracy': round(float(best_val_acc), 4),
        'epochs_trained': epoch, 'n_params': n_params,
    }
    with open(os.path.join(res_dir, f'{ds_name}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Results saved → {res_dir}")
    return metrics


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    all_metrics = {}
    for ds in ['dataset1', 'dataset2']:
        m = train_on_dataset(ds)
        all_metrics[ds] = m

    print("\n" + "="*60)
    print("  STAGE 2 COMPLETE — Baseline CNN Summary")
    print("="*60)
    for ds, m in all_metrics.items():
        print(f"  {ds}: test_acc={m['test_accuracy']:.4f}  "
              f"val_acc={m['best_val_accuracy']:.4f}  "
              f"epochs={m['epochs_trained']}")
    print("="*60)
    print("\n  Models saved to:", MODEL_ROOT)
    print("  Results saved to:", RESULT_ROOT)
