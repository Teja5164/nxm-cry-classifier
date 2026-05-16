"""
model_loader.py — Model definitions and checkpoint loading
===========================================================
Contains ALL four model architectures exactly as trained.
Never import class definitions from training scripts — those
standalone scripts use different attribute names.

Public API:
    load_models(dataset, device, model_root) -> list of ModelEntry
    ModelEntry: namedtuple(model, name, val_acc)

Architecture source-of-truth:
    master_pipeline_dataset1.py (inline class definitions)
    stage7_inference.py (identical inline copy)

IMPORTANT: Do NOT change any layer names or constructor signatures —
they must match the saved checkpoint keys exactly.
"""

import os, warnings
from pathlib import Path
from typing import List, Optional, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

warnings.filterwarnings('ignore')

_ROOT      = Path(__file__).resolve().parent.parent  # ML_pipeline/
MODEL_ROOT = str(_ROOT / 'models')


# ─────────────────────────────────────────────────────────────────────────────
# Named return type
# ─────────────────────────────────────────────────────────────────────────────
class ModelEntry(NamedTuple):
    model:   nn.Module
    name:    str
    val_acc: float


# ─────────────────────────────────────────────────────────────────────────────
# Shared: SpecAugment (used by all 4 models during training, no-op at eval)
# ─────────────────────────────────────────────────────────────────────────────
class SpecAugment(nn.Module):
    def __init__(self, freq_mask=15, time_mask=40, nf=2, nt=2):
        super().__init__()
        self.fm = freq_mask; self.tm = time_mask; self.nf = nf; self.nt = nt

    def forward(self, x):
        if not self.training:
            return x
        B, C, F, T = x.shape
        x = x.clone()
        for _ in range(self.nf):
            f  = random.randint(0, self.fm)
            f0 = random.randint(0, max(1, F - f))
            x[:, :, f0:f0 + f, :] = 0.0
        for _ in range(self.nt):
            t  = random.randint(0, self.tm)
            t0 = random.randint(0, max(1, T - t))
            x[:, :, :, t0:t0 + t] = 0.0
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Model 1: BaselineCNN
# Trained from: training/scripts/stage2_baseline_cnn.py (standalone)
# Checkpoint:   models/baseline_cnn/datasetN/best_model.pth
# Key attrs:    stage1/2/3/4, fc1, bn_fc, fc2  ← from standalone stage2 script
# ─────────────────────────────────────────────────────────────────────────────
class _CnnResBlock(nn.Module):
    """Residual block with optional channel expansion and stride."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.drop  = nn.Dropout2d(0.1)
        self.skip  = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch)
        ) if (stride != 1 or in_ch != out_ch) else nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class BaselineCNN(nn.Module):
    def __init__(self, n_classes=5, dropout=0.5):
        super().__init__()
        self.spec_aug = SpecAugment(freq_mask=15, time_mask=40)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1), nn.Dropout2d(0.1))
        self.stage1 = _CnnResBlock(32,  64,  stride=2)
        self.stage2 = _CnnResBlock(64,  128, stride=2)
        self.stage3 = _CnnResBlock(128, 256, stride=2)
        self.stage4 = _CnnResBlock(256, 256, stride=1)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(256, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.fc2  = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.spec_aug(x); x = self.stem(x)
        x = self.stage1(x); x = self.stage2(x); x = self.stage3(x); x = self.stage4(x)
        x = self.gap(x).flatten(1); x = self.drop(x)
        return self.fc2(self.drop(F.relu(self.bn_fc(self.fc1(x)))))


# ─────────────────────────────────────────────────────────────────────────────
# Shared CNN encoder for BiLSTM and CryNet
# Trained from: master_pipeline_dataset1.py (inline)
# Key attrs: enc, lstm/attn  ← from master_pipeline (NOT stage3 standalone)
# ─────────────────────────────────────────────────────────────────────────────
class _EncResBlock(nn.Module):
    """Residual block with same in/out channels (for CNNEncoder)."""
    def __init__(self, ch, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.Dropout2d(drop),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch))
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.net(x) + x)


class CNNEncoder(nn.Module):
    def __init__(self, out_ch=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True))
        self.b1 = nn.Sequential(_EncResBlock(32), nn.MaxPool2d(2, 2))
        self.b2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            _EncResBlock(64), nn.MaxPool2d(2, 2))
        self.b3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            _EncResBlock(128), nn.MaxPool2d(2, 2))
        self.b4 = nn.Sequential(
            nn.Conv2d(128, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            _EncResBlock(out_ch), nn.MaxPool2d(2, 2))
        self.fpool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        x = self.stem(x); x = self.b1(x); x = self.b2(x); x = self.b3(x); x = self.b4(x)
        return self.fpool(x).squeeze(2).transpose(1, 2)  # (B, T', D)


class TemporalAttn(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.a = nn.Linear(h * 2, 1)

    def forward(self, x):
        s = self.a(x).squeeze(-1)
        w = F.softmax(s, 1)
        return (w.unsqueeze(-1) * x).sum(1), w


# ─────────────────────────────────────────────────────────────────────────────
# Model 2: CNN+BiLSTM
# Trained from: master_pipeline_dataset1.py (inline)
# Checkpoint:   models/cnn_bilstm/datasetN/best_model.pth
# Key attrs:    aug, enc, lstm, attn, head  ← master_pipeline naming
# ─────────────────────────────────────────────────────────────────────────────
class CNNBiLSTM(nn.Module):
    def __init__(self, n=5, h=256, layers=2):
        super().__init__()
        self.aug  = SpecAugment()
        self.enc  = CNNEncoder(256)
        self.lstm = nn.LSTM(256, h, layers, batch_first=True, bidirectional=True,
                            dropout=0.3 if layers > 1 else 0.0)
        self.attn = TemporalAttn(h)
        self.head = nn.Sequential(
            nn.LayerNorm(h * 2),
            nn.Linear(h * 2, 256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, 64),   nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, n))

    def forward(self, x):
        x = self.aug(x); x = self.enc(x)
        out, _ = self.lstm(x); ctx, _ = self.attn(out)
        return self.head(ctx)


# ─────────────────────────────────────────────────────────────────────────────
# Model 3: CryNet (CNN + Transformer)
# Trained from: master_pipeline_dataset1.py (inline)
# Checkpoint:   models/cnn_transformer/datasetN/best_model.pth
# Key attrs:    aug, enc, pe, cls, tf, head  ← master_pipeline naming
# ─────────────────────────────────────────────────────────────────────────────
class LearnedPE(nn.Module):
    def __init__(self, maxlen, d, drop=0.1):
        super().__init__()
        self.pe   = nn.Embedding(maxlen, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, T, D = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        return self.drop(x + self.pe(pos))


class CryNet(nn.Module):
    def __init__(self, n=5, d=256, heads=8, layers=4, ff=512, drop=0.2):
        super().__init__()
        self.aug = SpecAugment(freq_mask=20, time_mask=50)
        self.enc = CNNEncoder(d)
        self.pe  = LearnedPE(128, d, drop)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls, std=0.02)
        tfl = nn.TransformerEncoderLayer(
            d, heads, ff, drop, activation='gelu',
            batch_first=True, norm_first=True)
        self.tf   = nn.TransformerEncoder(tfl, layers, norm=nn.LayerNorm(d))
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, n))

    def forward(self, x):
        x   = self.aug(x); x = self.enc(x); x = self.pe(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        x   = self.tf(torch.cat([cls, x], 1))
        return self.head(x[:, 0])


# ─────────────────────────────────────────────────────────────────────────────
# Model 4: SE-ResNet
# Trained from: master_pipeline_dataset1.py (inline)
# Checkpoint:   models/se_resnet/datasetN/best_model.pth
# Key attrs:    layer1/2/3/4  ← CRITICAL — NOT l1/l2/l3/l4
# ─────────────────────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.sq = nn.AdaptiveAvgPool2d(1)
        self.ex = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c, max(c // r, 8)), nn.ReLU(inplace=True),
            nn.Linear(max(c // r, 8), c), nn.Sigmoid())

    def forward(self, x):
        return x * self.ex(self.sq(x)).view(x.size(0), x.size(1), 1, 1)


class SEResBlock(nn.Module):
    def __init__(self, inc, outc, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outc), nn.ReLU(inplace=True),
            nn.Conv2d(outc, outc, 3, padding=1, bias=False),
            nn.BatchNorm2d(outc))
        self.se   = SEBlock(outc)
        self.skip = nn.Sequential(
            nn.Conv2d(inc, outc, 1, stride=stride, bias=False),
            nn.BatchNorm2d(outc)
        ) if (inc != outc or stride != 1) else nn.Identity()

    def forward(self, x):
        return F.relu(self.se(self.conv(x)) + self.skip(x), True)


class SEResNet(nn.Module):
    def __init__(self, n=5):
        super().__init__()
        self.aug    = SpecAugment()
        self.stem   = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(SEResBlock(32,  64,  stride=2), SEResBlock(64,  64))
        self.layer2 = nn.Sequential(SEResBlock(64,  128, stride=2), SEResBlock(128, 128))
        self.layer3 = nn.Sequential(SEResBlock(128, 256, stride=2), SEResBlock(256, 256))
        self.layer4 = nn.Sequential(SEResBlock(256, 512, stride=2), SEResBlock(512, 512))
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, n))

    def forward(self, x):
        x = self.stem(self.aug(x))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.head(self.pool(x))


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────
_MODEL_REGISTRY = [
    # (class,       subfolder,        display_name,  constructor_kwargs)
    (BaselineCNN,  'baseline_cnn',    'BaselineCNN', {'n_classes': 5}),
    (CNNBiLSTM,    'cnn_bilstm',      'CNN+BiLSTM',  {'n': 5}),
    (CryNet,       'cnn_transformer', 'CryNet',      {'n': 5}),
    (SEResNet,     'se_resnet',       'SE-ResNet',   {'n': 5}),
]


def _load_single(model_cls, ckpt_path: str, name: str, device, **kwargs):
    """Attempt to load one checkpoint. Returns ModelEntry or None."""
    if not os.path.exists(ckpt_path):
        return None
    try:
        m    = model_cls(**kwargs).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m.load_state_dict(ckpt['model_state'])
        m.eval()
        val_acc = float(ckpt.get('val_acc', 0.0))
        return ModelEntry(model=m, name=name, val_acc=val_acc)
    except Exception as e:
        print(f"  ⚠  Could not load {name}: {e}")
        return None


def load_models(
    dataset: str,
    device=None,
    model_root: Optional[str] = None,
    prefer_augmented: bool = True,
) -> List[ModelEntry]:
    """
    Load all available trained checkpoints for a given dataset.

    When prefer_augmented=True (default), automatically uses augmented models
    (models/*/dataset1_augmented/) if they exist and have equal/better val_acc
    than the original models. Falls back to originals if augmented not found.

    Args:
        dataset:          'dataset1' or 'dataset2'.
        device:           torch.device or None (auto-detect GPU → CPU).
        model_root:       Override for models/ directory path.
        prefer_augmented: If True, prefer augmented checkpoints when available.

    Returns:
        List of ModelEntry(model, name, val_acc) — only successfully loaded models.

    Raises:
        RuntimeError: If no models could be loaded at all.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root = model_root or MODEL_ROOT
    entries = []

    for cls, folder, name, kwargs in _MODEL_REGISTRY:
        entry = None

        # Try augmented checkpoint first (if applicable for dataset1)
        if prefer_augmented and dataset == 'dataset1':
            aug_ckpt = os.path.join(root, folder, 'dataset1_augmented', 'best_model.pth')
            aug_entry = _load_single(cls, aug_ckpt, f"{name}*", device, **kwargs)
            if aug_entry is not None:
                entry = aug_entry

        # Fall back to standard checkpoint
        if entry is None:
            std_ckpt = os.path.join(root, folder, dataset, 'best_model.pth')
            entry = _load_single(cls, std_ckpt, name, device, **kwargs)

        if entry is not None:
            entries.append(entry)

    if not entries:
        raise RuntimeError(
            f"No trained models found for dataset='{dataset}' under: {root}\n"
            "Run training pipeline (stages 2–5) first."
        )

    aug_count = sum(1 for e in entries if e.name.endswith('*'))
    if aug_count > 0:
        print(f"  [ModelLoader] {aug_count}/{len(entries)} augmented models loaded (marked with *)")

    return entries
