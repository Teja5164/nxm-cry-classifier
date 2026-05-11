"""
Stage 5: Transfer Learning — PANNs CNN14 pretrained on AudioSet
================================================================
Strategy:
  - Use CNN14 from PANNs (Pretrained Audio Neural Networks) as backbone
  - PANNs were trained on AudioSet (527 audio classes, 1.9M clips)
  - Replace final classifier with 5-class head for infant cry
  - Two-phase training:
      Phase 1 (freeze): Train only head for 20 epochs (fast convergence)
      Phase 2 (finetune): Unfreeze all layers, train with low LR (1e-5)

Why PANNs CNN14:
  - AudioSet pretrained features generalise to infant cry (both are audio)
  - CNN14 captures hierarchical acoustic patterns: noise → harmonics → pitch → melody
  - Transfer learning compensates for limited labelled data
  - Achieves state-of-the-art on many audio classification tasks

Input:  Mel spectrogram (1, 128, 431) — same as other stages
Output: 5-class infant cry classification

NOTE: CNN14 weights (~200MB) auto-downloaded from GitHub on first run.
"""

from pathlib import Path
import os, json, time, random
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

CLASSES      = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
N_CLASSES    = 5
BATCH_SIZE   = 32
PHASE1_EPOCHS = 20          # frozen backbone
PHASE2_EPOCHS = 80          # fine-tune all
LR_HEAD      = 3e-4         # head learning rate
LR_FINETUNE  = 5e-5         # full model fine-tune LR
WEIGHT_DECAY = 0.01
PATIENCE     = 15
LABEL_SMOOTH = 0.1

_ROOT = Path(__file__).resolve().parent.parent.parent  # ML_pipeline/
FEAT_ROOT    = str(_ROOT / 'training' / 'features')
MODEL_ROOT   = str(_ROOT / 'models' / 'panns_cnn14')
RESULT_ROOT  = str(_ROOT / 'results' / 'panns_cnn14')
CKPT_DIR     = str(_ROOT / 'training' / 'pretrained')  # cache for pretrained weights

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True

os.makedirs(CKPT_DIR, exist_ok=True)

# ── PANNs CNN14 building blocks ───────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels); self.bn2 = nn.BatchNorm2d(out_channels)
        self._init()

    def _init(self):
        for conv in [self.conv1, self.conv2]:
            nn.init.xavier_uniform_(conv.weight)

    def forward(self, x, pool_size=(2,2), pool_type='avg'):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':   x = F.max_pool2d(x, pool_size)
        elif pool_type == 'avg': x = F.avg_pool2d(x, pool_size)
        elif pool_type == 'avg+max':
            x = F.avg_pool2d(x, pool_size) + F.max_pool2d(x, pool_size)
        return x

class CNN14(nn.Module):
    """
    CNN14 from PANNs — pretrained on AudioSet (527 classes).
    We replace the final fc_audioset layer with a new 5-class head.
    """
    def __init__(self, sample_rate=32000, window_size=1024, hop_size=320,
                 mel_bins=64, fmin=50, fmax=14000, classes_num=527):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 64, 3, padding=1, bias=False)
        self.bn0   = nn.BatchNorm2d(64)

        self.conv_block1  = ConvBlock(64, 64)
        self.conv_block2  = ConvBlock(64, 128)
        self.conv_block3  = ConvBlock(128, 256)
        self.conv_block4  = ConvBlock(256, 512)
        self.conv_block5  = ConvBlock(512, 1024)
        self.conv_block6  = ConvBlock(1024, 2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)  # original 527-class head

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc_audioset.weight)
        nn.init.zeros_(self.fc1.bias); nn.init.zeros_(self.fc_audioset.bias)

    def forward(self, x):
        # x: (B, 1, F, T)
        x = x.transpose(2, 3)   # → (B, 1, T, F) as PANNs expects (B,1,T,mel)
        x = F.relu_(self.bn0(self.conv0(x)))

        x = self.conv_block1(x, pool_size=(2,2), pool_type='avg+max')
        x = F.dropout(x, 0.2, self.training)
        x = self.conv_block2(x, pool_size=(2,2), pool_type='avg+max')
        x = F.dropout(x, 0.2, self.training)
        x = self.conv_block3(x, pool_size=(2,2), pool_type='avg+max')
        x = F.dropout(x, 0.2, self.training)
        x = self.conv_block4(x, pool_size=(2,2), pool_type='avg+max')
        x = F.dropout(x, 0.2, self.training)
        x = self.conv_block5(x, pool_size=(2,2), pool_type='avg+max')
        x = F.dropout(x, 0.2, self.training)
        x = self.conv_block6(x, pool_size=(1,1), pool_type='avg+max')
        x = F.dropout(x, 0.2, self.training)

        x = torch.mean(x, dim=3)           # avg over freq
        x1 = F.max_pool1d(x, x.shape[2])
        x2 = F.avg_pool1d(x, x.shape[2])
        x  = (x1 + x2).squeeze(2)          # (B, 2048)

        x = F.relu_(self.fc1(F.dropout(x, 0.5, self.training)))
        return x  # return embedding, not logits

# ── Transfer model with new head ──────────────────────────────────────────────
class PANNsTransfer(nn.Module):
    def __init__(self, n_classes=5, pretrained_path=None):
        super().__init__()
        self.backbone = CNN14()

        # Load pretrained AudioSet weights if available
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}")
            ckpt = torch.load(pretrained_path, map_location='cpu')
            state = ckpt.get('model', ckpt)
            # Remove head keys
            state = {k: v for k, v in state.items() if 'fc_audioset' not in k}
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            print(f"  Loaded pretrained backbone. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        else:
            print("No pretrained weights found — training from scratch (still benefits from architecture)")

        # SpecAugment on top
        self.augment = SpecAugment()

        # New cry classification head
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512,  128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes))

        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False
        print("Backbone frozen")

    def unfreeze_backbone(self, lr_scale=0.1):
        for p in self.backbone.parameters(): p.requires_grad = True
        print("Backbone unfrozen")

    def forward(self, x):
        x = self.augment(x)            # SpecAugment during train
        feat = self.backbone(x)        # (B, 2048)
        return self.head(feat)         # (B, n_classes)

# ── SpecAugment ───────────────────────────────────────────────────────────────
class SpecAugment(nn.Module):
    def __init__(self, freq_mask=15, time_mask=40, n_freq=2, n_time=2):
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

# ── Download pretrained weights ───────────────────────────────────────────────
def download_cnn14_weights(dest_dir):
    """Download CNN14 AudioSet pretrained weights (~200MB)."""
    import urllib.request
    dest = os.path.join(dest_dir, 'CNN14_mAP=0.431.pth')
    if os.path.exists(dest):
        print(f"Pretrained weights already at {dest}")
        return dest

    url = "https://zenodo.org/record/3987831/files/CNN14_mAP%3D0.431.pth"
    print(f"Downloading CNN14 pretrained weights (~200MB)...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"Downloaded to {dest}")
        return dest
    except Exception as e:
        print(f"Download failed: {e}. Will train without pretrained weights.")
        return None

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

# ── Dataset ───────────────────────────────────────────────────────────────────
class MelDataset(Dataset):
    def __init__(self, mel, labels):
        self.mel = torch.FloatTensor(mel); self.labels = torch.LongTensor(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.mel[i], self.labels[i]

# ── Helpers ───────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, train=True):
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

def train_phase(model, tr_ld, va_ld, criterion, optimizer, scheduler,
                max_epochs, patience, model_path, phase_name):
    best_val_acc = 0; cnt = 0
    tl, vl, ta, va = [], [], [], []
    for ep in range(1, max_epochs+1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, tr_ld, criterion, optimizer, True)
        va_loss, va_acc = run_epoch(model, va_ld, criterion, None, False)
        if scheduler: scheduler.step()
        elapsed = time.time() - t0
        tl.append(tr_loss); vl.append(va_loss); ta.append(tr_acc); va.append(va_acc)
        print(f"[{phase_name}] Ep {ep:3d} | {elapsed:.1f}s | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
              f"va_loss={va_loss:.4f} va_acc={va_acc:.4f}", flush=True)
        if va_acc > best_val_acc:
            best_val_acc = va_acc; cnt = 0
            torch.save({'epoch': ep, 'model_state': model.state_dict(), 'val_acc': va_acc},
                       model_path)
            print(f"  ✓ Best saved (val_acc={va_acc:.4f})")
        else:
            cnt += 1
            if cnt >= patience:
                print(f"Early stop at ep {ep}"); break
    return tl, vl, ta, va, best_val_acc

# ── Main ──────────────────────────────────────────────────────────────────────
def train_dataset(ds_name, pretrained_path):
    print(f"\n{'='*60}")
    print(f"Stage 5: PANNs CNN14 Transfer Learning on {ds_name}")
    print(f"{'='*60}")

    feat_dir   = os.path.join(FEAT_ROOT, ds_name)
    model_dir  = os.path.join(MODEL_ROOT, ds_name)
    result_dir = os.path.join(RESULT_ROOT, ds_name)
    os.makedirs(model_dir, exist_ok=True); os.makedirs(result_dir, exist_ok=True)

    mel    = np.load(os.path.join(feat_dir, 'mel_specs.npy'))
    labels = np.load(os.path.join(feat_dir, 'labels.npy'))
    print(f"Features: mel={mel.shape}")

    # Our mel is 128 bins, PANNs expects 64. Resize on the fly via interpolation.
    # We'll resize in the dataset
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

    model     = PANNsTransfer(N_CLASSES, pretrained_path).to(DEVICE)
    criterion = LabelSmoothingCE(N_CLASSES, LABEL_SMOOTH)
    model_path = os.path.join(model_dir, 'best_model.pth')

    # ── Phase 1: Freeze backbone, train head only ──────────────────────────────
    print("\n--- Phase 1: Frozen backbone, train head ---")
    model.freeze_backbone()
    head_params = [p for p in model.head.parameters()]
    opt1 = torch.optim.AdamW(head_params, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt1, T_0=10)
    tl1, vl1, ta1, va1, bva1 = train_phase(
        model, tr_ld, va_ld, criterion, opt1, sch1,
        PHASE1_EPOCHS, PATIENCE, model_path, 'Phase1-HeadOnly')

    # ── Phase 2: Unfreeze all, fine-tune with low LR ───────────────────────────
    print("\n--- Phase 2: Full model fine-tune ---")
    model.unfreeze_backbone()
    # Differential LR: backbone gets 10x smaller LR than head
    opt2 = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': LR_FINETUNE},
        {'params': model.head.parameters(),     'lr': LR_FINETUNE * 5}
    ], weight_decay=WEIGHT_DECAY)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt2, T_0=20, T_mult=2)
    tl2, vl2, ta2, va2, bva2 = train_phase(
        model, tr_ld, va_ld, criterion, opt2, sch2,
        PHASE2_EPOCHS, PATIENCE, model_path, 'Phase2-FineTune')

    best_val_acc = max(bva1, bva2)

    # Evaluate on test
    ckpt = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for x, y in te_ld:
            preds.extend(model(x.to(DEVICE)).argmax(1).cpu().numpy())
            true.extend(y.numpy())

    report = classification_report(true, preds, target_names=CLASSES, digits=4)
    print(f"\n{report}")

    # Combine curves
    all_tl = tl1 + tl2; all_vl = vl1 + vl2; all_ta = ta1 + ta2; all_va = va1 + va2
    save_curves(all_tl, all_vl, all_ta, all_va,
                os.path.join(result_dir, f'{ds_name}_training_curves.png'))
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
    pretrained = download_cnn14_weights(CKPT_DIR)
    results = {}
    for ds in ['dataset1', 'dataset2']:
        results[ds] = train_dataset(ds, pretrained)

    print("\n" + "="*60)
    print("Stage 5 Summary:")
    for ds, m in results.items():
        print(f"  {ds}: test_acc={m['test_acc']:.4f}  macro_f1={m['macro_f1']:.4f}")
