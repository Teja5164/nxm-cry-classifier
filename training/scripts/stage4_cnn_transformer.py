"""
Stage 4: CNN + Transformer Attention (CryNet)
=============================================
Architecture:
  - CNN Encoder: same 4-block ResNet-style as Stage 2/3
  - Positional Encoding: learnable 1D positional embeddings
  - Transformer Encoder: 4 layers, 8 heads, self-attention over time
  - [CLS] token aggregation: learned global representation
  - MLP Head: 5-class output

Why Transformer for infant cry:
  - Self-attention learns long-range temporal dependencies globally
  - Multi-head attention captures multiple acoustic patterns simultaneously
  - [CLS] token learns to summarise the entire cry sequence
  - Outperforms BiLSTM when data is sufficient (Dataset2: 7500 samples)

Both datasets trained separately and saved.
"""

from pathlib import Path
import os, json, time, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

CLASSES       = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
N_CLASSES     = 5
BATCH_SIZE    = 32
MAX_EPOCHS    = 100
LR            = 1e-4       # lower LR for transformer stability
WARMUP_EPOCHS = 5
WEIGHT_DECAY  = 0.05       # higher for transformer
PATIENCE      = 15
LABEL_SMOOTH  = 0.1
D_MODEL       = 256        # transformer dimension
N_HEADS       = 8
N_LAYERS      = 4
DIM_FF        = 512
TF_DROPOUT    = 0.2

_ROOT = Path(__file__).resolve().parent.parent.parent  # ML_pipeline/
FEAT_ROOT     = str(_ROOT / 'training' / 'features')
MODEL_ROOT    = str(_ROOT / 'models' / 'cnn_transformer')
RESULT_ROOT   = str(_ROOT / 'results' / 'cnn_transformer')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True

# ── SpecAugment ───────────────────────────────────────────────────────────────
class SpecAugment(nn.Module):
    def __init__(self, freq_mask=20, time_mask=50, n_freq=2, n_time=2):
        super().__init__()
        self.fm = freq_mask; self.tm = time_mask; self.nf = n_freq; self.nt = n_time

    def forward(self, x):
        if not self.training: return x
        B, C, F, T = x.shape
        x = x.clone()
        for _ in range(self.nf):
            f = random.randint(0, self.fm); f0 = random.randint(0, max(1, F - f))
            x[:, :, f0:f0+f, :] = 0.0
        for _ in range(self.nt):
            t = random.randint(0, self.tm); t0 = random.randint(0, max(1, T - t))
            x[:, :, :, t0:t0+t] = 0.0
        return x

# ── CNN Encoder ───────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.Dropout2d(dropout),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch))
        self.relu = nn.ReLU(True)
    def forward(self, x): return self.relu(self.net(x) + x)

class CNNEncoder(nn.Module):
    """
    Input:  (B, 1, 128, 431)
    Output: (B, D_MODEL, T')  — sequence of D_MODEL-dim vectors over time
    """
    def __init__(self, out_dim=D_MODEL):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True))
        self.b1 = nn.Sequential(ResBlock(32), nn.MaxPool2d(2,2))
        self.b2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            ResBlock(64), nn.MaxPool2d(2,2))
        self.b3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            ResBlock(128), nn.MaxPool2d(2,2))
        self.b4 = nn.Sequential(
            nn.Conv2d(128, out_dim, 3, padding=1, bias=False), nn.BatchNorm2d(out_dim), nn.ReLU(True),
            ResBlock(out_dim), nn.MaxPool2d(2,2))
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # collapse freq axis

    def forward(self, x):
        x = self.stem(x); x = self.b1(x); x = self.b2(x); x = self.b3(x); x = self.b4(x)
        x = self.freq_pool(x).squeeze(2)   # (B, D_MODEL, T')
        return x.permute(0, 2, 1)          # (B, T', D_MODEL)

# ── Positional Encoding ───────────────────────────────────────────────────────
class LearnedPosEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model, dropout=0.1):
        super().__init__()
        self.pos_embed = nn.Embedding(max_seq_len + 1, d_model)  # +1 for CLS
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        return self.dropout(x + self.pos_embed(pos))

# ── CryNet: CNN + Transformer ─────────────────────────────────────────────────
class CryNet(nn.Module):
    def __init__(self, n_classes=5, max_seq=64):
        super().__init__()
        self.augment   = SpecAugment()
        self.encoder   = CNNEncoder(out_dim=D_MODEL)
        self.pos_enc   = LearnedPosEncoding(max_seq, D_MODEL, TF_DROPOUT)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder
        tf_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=DIM_FF,
            dropout=TF_DROPOUT, activation='gelu', batch_first=True,
            norm_first=True)   # Pre-LN for more stable training
        self.transformer = nn.TransformerEncoder(tf_layer, num_layers=N_LAYERS,
                                                  norm=nn.LayerNorm(D_MODEL))

        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, 256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, 64),  nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, n_classes))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.augment(x)                               # (B, 1, 128, 431)
        x = self.encoder(x)                               # (B, T', D_MODEL)

        # Prepend CLS token
        cls = self.cls_token.expand(x.size(0), -1, -1)   # (B, 1, D_MODEL)
        x   = torch.cat([cls, x], dim=1)                  # (B, T'+1, D_MODEL)

        # Positional encoding (skip CLS position)
        x[:, 1:, :] = self.pos_enc(x[:, 1:, :])

        # Transformer
        x = self.transformer(x)                           # (B, T'+1, D_MODEL)

        # Use CLS token output for classification
        cls_out = x[:, 0, :]                              # (B, D_MODEL)
        return self.head(cls_out)

# ── Label Smoothing ───────────────────────────────────────────────────────────
class LabelSmoothingCE(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing; self.cls = classes

    def forward(self, pred, target):
        sv = self.smoothing / (self.cls - 1)
        one_hot = torch.full_like(pred, sv)
        one_hot.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return -(one_hot * F.log_softmax(pred, dim=1)).sum(dim=1).mean()

# ── Warmup + Cosine LR ────────────────────────────────────────────────────────
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.opt = optimizer; self.warmup = warmup_epochs
        self.total = total_epochs; self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']; self.ep = 0

    def step(self):
        self.ep += 1
        if self.ep <= self.warmup:
            lr = self.base_lr * self.ep / self.warmup
        else:
            progress = (self.ep - self.warmup) / (self.total - self.warmup)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for g in self.opt.param_groups: g['lr'] = lr

# ── Dataset ───────────────────────────────────────────────────────────────────
class MelDataset(Dataset):
    def __init__(self, mel, labels):
        self.mel = torch.FloatTensor(mel); self.labels = torch.LongTensor(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.mel[i], self.labels[i]

# ── Helpers ───────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, scheduler, train=True):
    model.train(train)
    total_loss = correct = total = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x); loss = criterion(out, y)
            if train:
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item(); total += len(y)
    if train and scheduler: scheduler.step()
    return total_loss / total, correct / total

def save_curves(tl, vl, ta, va, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12,4))
    a1.plot(tl, label='Train'); a1.plot(vl, label='Val'); a1.set_title('Loss'); a1.legend()
    a2.plot(ta, label='Train'); a2.plot(va, label='Val'); a2.set_title('Accuracy'); a2.legend()
    plt.tight_layout(); plt.savefig(path, dpi=100); plt.close()

def save_cm(cm, path):
    fig, ax = plt.subplots(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES,
                cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout(); plt.savefig(path, dpi=100); plt.close()

# ── Train ─────────────────────────────────────────────────────────────────────
def train_dataset(ds_name):
    print(f"\n{'='*60}")
    print(f"Stage 4: CryNet (CNN+Transformer) on {ds_name}")
    print(f"{'='*60}")

    feat_dir   = os.path.join(FEAT_ROOT, ds_name)
    model_dir  = os.path.join(MODEL_ROOT, ds_name)
    result_dir = os.path.join(RESULT_ROOT, ds_name)
    os.makedirs(model_dir, exist_ok=True); os.makedirs(result_dir, exist_ok=True)

    mel    = np.load(os.path.join(feat_dir, 'mel_specs.npy'))
    labels = np.load(os.path.join(feat_dir, 'labels.npy'))
    print(f"Features: mel={mel.shape}")

    idx   = np.arange(len(labels))
    tr_i, te_i = train_test_split(idx, test_size=0.15, stratify=labels, random_state=SEED)
    tr_i, va_i = train_test_split(tr_i, test_size=0.176, stratify=labels[tr_i], random_state=SEED)
    print(f"Split → train={len(tr_i)} val={len(va_i)} test={len(te_i)}")

    cc = np.bincount(labels[tr_i])
    sw = (1.0 / cc)[labels[tr_i]]
    sampler = WeightedRandomSampler(sw, len(tr_i), replacement=True)

    nw = 4 if DEVICE.type == 'cuda' else 2
    pin = DEVICE.type == 'cuda'
    tr_ld = DataLoader(MelDataset(mel[tr_i], labels[tr_i]), BATCH_SIZE, sampler=sampler,
                       num_workers=nw, pin_memory=pin, persistent_workers=True)
    va_ld = DataLoader(MelDataset(mel[va_i], labels[va_i]), BATCH_SIZE, shuffle=False,
                       num_workers=nw, pin_memory=pin, persistent_workers=True)
    te_ld = DataLoader(MelDataset(mel[te_i], labels[te_i]), BATCH_SIZE, shuffle=False,
                       num_workers=nw, pin_memory=pin, persistent_workers=True)

    model     = CryNet(N_CLASSES).to(DEVICE)
    criterion = LabelSmoothingCE(N_CLASSES, LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                                  betas=(0.9, 0.999))
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, MAX_EPOCHS)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    best_val_acc = 0; patience_cnt = 0
    tl, vl, ta, va = [], [], [], []

    for ep in range(1, MAX_EPOCHS+1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, tr_ld, criterion, optimizer, scheduler, True)
        va_loss, va_acc = run_epoch(model, va_ld, criterion, None, None, False)
        elapsed = time.time() - t0
        tl.append(tr_loss); vl.append(va_loss); ta.append(tr_acc); va.append(va_acc)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Ep {ep:3d} | {elapsed:.1f}s | lr={lr_now:.2e} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
              f"va_loss={va_loss:.4f} va_acc={va_acc:.4f}", flush=True)

        if va_acc > best_val_acc:
            best_val_acc = va_acc; patience_cnt = 0
            torch.save({'epoch': ep, 'model_state': model.state_dict(), 'val_acc': va_acc},
                       os.path.join(model_dir, 'best_model.pth'))
            print(f"  ✓ Best saved (val_acc={va_acc:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"Early stop at ep {ep}"); break

    ckpt = torch.load(os.path.join(model_dir, 'best_model.pth'), map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for x, y in te_ld:
            preds.extend(model(x.to(DEVICE)).argmax(1).cpu().numpy())
            true.extend(y.numpy())

    report = classification_report(true, preds, target_names=CLASSES, digits=4)
    print(f"\n{report}")

    save_curves(tl, vl, ta, va, os.path.join(result_dir, f'{ds_name}_training_curves.png'))
    save_cm(confusion_matrix(true, preds), os.path.join(result_dir, f'{ds_name}_confusion_matrix.png'))
    with open(os.path.join(result_dir, f'{ds_name}_classification_report.txt'), 'w') as f: f.write(report)

    metrics = {'dataset': ds_name, 'best_val_acc': float(best_val_acc),
               'test_acc': float(accuracy_score(true, preds)),
               'macro_f1': float(f1_score(true, preds, average='macro')),
               'weighted_f1': float(f1_score(true, preds, average='weighted'))}
    with open(os.path.join(result_dir, f'{ds_name}_metrics.json'), 'w') as f: json.dump(metrics, f, indent=2)
    print(f"\n✅ {ds_name} | test_acc={metrics['test_acc']:.4f} | macro_f1={metrics['macro_f1']:.4f}")
    return metrics

if __name__ == '__main__':
    results = {}
    for ds in ['dataset1', 'dataset2']:
        results[ds] = train_dataset(ds)

    print("\n" + "="*60)
    print("Stage 4 Summary:")
    for ds, m in results.items():
        print(f"  {ds}: test_acc={m['test_acc']:.4f}  macro_f1={m['macro_f1']:.4f}")
