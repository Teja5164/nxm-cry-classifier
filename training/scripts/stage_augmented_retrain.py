"""
stage_augmented_retrain.py — Noise-Augmented Retraining (Fix 3)
================================================================
Retrains all 4 models with real-world noise augmentation to close
the gap between test-set accuracy (84–86%) and real-world accuracy (58–60%).

Augmentations applied during training:
  1. Background noise overlay (SNR 5–20 dB) — simulates real environments
  2. Pitch shift ±2 semitones — variation in baby vocal pitch
  3. Time stretch 0.9–1.1× — recording speed variation
  4. Amplitude jitter ±3 dB — microphone gain variation
  5. SpecAugment (freq + time masking) — already in existing training

IMPORTANT:
  - Original models are NOT overwritten
  - Augmented models saved to: models/*/dataset1_augmented/
  - If augmented accuracy > original, update yamnet_gate.py to prefer them

Usage:
    cd D:\\nxm\\ML_pipeline
    D:\\TEJA\\Anaconda3\\python.exe training/scripts/stage_augmented_retrain.py

    # Train only one model to test:
    python training/scripts/stage_augmented_retrain.py --model baseline_cnn
"""

import os, sys, json, random, warnings, argparse
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
warnings.filterwarnings('ignore')

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_INF  = _ROOT / 'inference'
if str(_INF) not in sys.path:
    sys.path.insert(0, str(_INF))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from preprocess import SR, N_MEL, N_FFT, HOP_LEN, FMIN, FMAX, TARGET_T, N_SAMPLES
import librosa

DATASET1_DIR = _ROOT / 'datasets' / 'Dataset1'
ESC50_DIR    = _ROOT / 'datasets' / 'ESC50'
FEAT_ROOT    = _ROOT / 'training' / 'features'
MODEL_ROOT   = _ROOT / 'models'

CLASSES      = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
N_CLASSES    = 5

print("=" * 65)
print("  Noise-Augmented Retraining — All 4 Models")
print("=" * 65)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")


# ─────────────────────────────────────────────────────────────────────────────
# Audio augmentation
# ─────────────────────────────────────────────────────────────────────────────
_noise_cache = []

def _load_noise_pool():
    """Load ESC-50 WAVs into a numpy pool for fast noise overlay."""
    global _noise_cache
    if _noise_cache:
        return

    esc50_audio = ESC50_DIR / 'audio'
    if not esc50_audio.exists():
        print("  [Augment] ESC-50 not found — noise overlay disabled.")
        return

    wavs = list(esc50_audio.glob('*.wav'))[:200]   # use first 200 for speed
    print(f"  [Augment] Loading noise pool ({len(wavs)} clips) ...")
    for wav in wavs:
        try:
            y, _ = librosa.load(str(wav), sr=SR, mono=True)
            if len(y) >= N_SAMPLES:
                _noise_cache.append(y[:N_SAMPLES].astype(np.float32))
            elif len(y) > SR:
                # tile short clips
                n_reps = (N_SAMPLES // len(y)) + 2
                _noise_cache.append(np.tile(y, n_reps)[:N_SAMPLES].astype(np.float32))
        except:
            pass
    print(f"  [Augment] Noise pool ready: {len(_noise_cache)} clips")


def augment_audio(audio: np.ndarray, p_noise=0.5, p_pitch=0.3, p_stretch=0.3) -> np.ndarray:
    """
    Apply random combination of augmentations to a raw audio array.
    All augmentations are applied with probability p.
    """
    audio = audio.copy()

    # 1. Background noise overlay
    if _noise_cache and random.random() < p_noise:
        noise = random.choice(_noise_cache).copy()
        # Random SNR between 5 and 20 dB
        snr_db = random.uniform(5, 20)
        signal_rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
        noise_rms  = np.sqrt(np.mean(noise ** 2)) + 1e-9
        target_noise_rms = signal_rms / (10 ** (snr_db / 20))
        noise = noise * (target_noise_rms / noise_rms)
        audio = audio + noise
        # Re-normalize to [-1, 1]
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

    # 2. Amplitude jitter (±3 dB = factor 0.708 to 1.413)
    if random.random() < 0.5:
        factor = 10 ** (random.uniform(-3, 3) / 20)
        audio  = audio * factor
        audio  = np.clip(audio, -1.0, 1.0)

    # 3. Pitch shift (±2 semitones) — expensive, apply sparingly
    if random.random() < p_pitch:
        n_steps = random.uniform(-2, 2)
        audio   = librosa.effects.pitch_shift(audio, sr=SR, n_steps=n_steps)

    # 4. Time stretch (0.9–1.1×) — expensive, apply sparingly
    if random.random() < p_stretch:
        rate  = random.uniform(0.9, 1.1)
        audio = librosa.effects.time_stretch(audio, rate=rate)
        # Fix length after stretch
        if len(audio) >= N_SAMPLES:
            start = (len(audio) - N_SAMPLES) // 2
            audio = audio[start:start + N_SAMPLES]
        else:
            n_reps = (N_SAMPLES // len(audio)) + 2
            audio  = np.tile(audio, n_reps)[:N_SAMPLES]

    return audio.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class AugmentedCryDataset(Dataset):
    """
    Loads raw WAV files, applies augmentation, then extracts features on-the-fly.
    More memory efficient than pre-extracting all augmented features.
    """
    def __init__(self, file_paths, file_labels, stats, augment=False):
        self.paths   = file_paths
        self.labels  = file_labels
        self.stats   = stats
        self.augment = augment
        self.mean    = np.array(stats['mel_mean'], dtype=np.float32)  # (128,)
        self.std     = np.array(stats['mel_std'],  dtype=np.float32)  # (128,)
        self.std     = np.where(self.std < 1e-8, 1.0, self.std)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path  = str(self.paths[idx])
        label = int(self.labels[idx])

        try:
            audio, _ = librosa.load(path, sr=SR, mono=True)
        except Exception:
            return torch.zeros(1, N_MEL, TARGET_T), label

        # Silence trim + RMS normalize (matches preprocessing pipeline)
        try:
            audio, _ = librosa.effects.trim(audio, top_db=30)
        except:
            pass
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            target_rms = 10 ** (-20 / 20)
            audio = audio * (target_rms / rms)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

        # Fix length
        if len(audio) >= N_SAMPLES:
            start = (len(audio) - N_SAMPLES) // 2
            audio = audio[start:start + N_SAMPLES]
        else:
            n_reps = (N_SAMPLES // len(audio)) + 2
            audio  = np.tile(audio, n_reps)[:N_SAMPLES]

        # Augment (training only)
        if self.augment:
            audio = augment_audio(audio)

        # Feature extraction
        mel    = librosa.feature.melspectrogram(
            y=audio.astype(np.float32), sr=SR,
            n_fft=N_FFT, hop_length=HOP_LEN, n_mels=N_MEL, fmin=FMIN, fmax=FMAX
        )
        log_mel = librosa.power_to_db(mel + 1e-6, ref=np.max).astype(np.float32)

        # Z-score normalize
        log_mel = (log_mel - self.mean[:, None]) / self.std[:, None]

        # Fix time dimension
        if log_mel.shape[1] < TARGET_T:
            pad     = TARGET_T - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode='edge')
        elif log_mel.shape[1] > TARGET_T:
            log_mel = log_mel[:, :TARGET_T]

        # SpecAugment
        if self.augment:
            log_mel = self._spec_augment(log_mel)

        return torch.FloatTensor(log_mel).unsqueeze(0), label

    @staticmethod
    def _spec_augment(x: np.ndarray) -> np.ndarray:
        x = x.copy()
        F, T = x.shape
        for _ in range(2):
            f = random.randint(0, 15); f0 = random.randint(0, max(1, F - f))
            x[f0:f0+f, :] = 0.0
        for _ in range(2):
            t = random.randint(0, 40); t0 = random.randint(0, max(1, T - t))
            x[:, t0:t0+t] = 0.0
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Model architectures (identical to model_loader.py — do not change)
# ─────────────────────────────────────────────────────────────────────────────
from model_loader import (
    BaselineCNN as _BaselineCNN,
    CNNBiLSTM   as _CNNBiLSTM,
    CryNet      as _CryNet,
    SEResNet    as _SEResNet,
)

MODEL_REGISTRY = {
    'baseline_cnn':  (_BaselineCNN, {'n_classes': N_CLASSES}),
    'cnn_bilstm':    (_CNNBiLSTM,   {'n': N_CLASSES}),
    'cnn_transformer': (_CryNet,    {'n': N_CLASSES}),
    'se_resnet':     (_SEResNet,    {'n': N_CLASSES}),
}

# Val accuracies from original training (for checkpoint compatibility)
ORIG_VAL_ACCS = {
    'baseline_cnn':    0.8467,
    'cnn_bilstm':      0.8331,
    'cnn_transformer': 0.8314,
    'se_resnet':       0.8364,
}


# ─────────────────────────────────────────────────────────────────────────────
# Training function (single model)
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model_name: str, train_loader, val_loader, args):
    cls, kwargs = MODEL_REGISTRY[model_name]
    model = cls(**kwargs).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    save_dir  = MODEL_ROOT / model_name / 'dataset1_augmented'
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = str(save_dir / 'best_model.pth')

    best_val_acc = 0.0
    print(f"\n  {'Ep':>4}  {'TrLoss':>8}  {'TrAcc':>7}  {'VaLoss':>8}  {'VaAcc':>7}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*7}")

    for epoch in range(1, args.epochs + 1):
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
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total   += len(x)
        scheduler.step()

        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y   = x.to(device), y.to(device)
                logits = model(x)
                loss   = criterion(logits, y)
                va_loss    += loss.item() * len(x)
                va_correct += (logits.argmax(1) == y).sum().item()
                va_total   += len(x)

        tr_acc = tr_correct / tr_total * 100
        va_acc = va_correct / va_total * 100
        print(f"  {epoch:>4}  {tr_loss/tr_total:>8.4f}  {tr_acc:>6.2f}%  "
              f"{va_loss/va_total:>8.4f}  {va_acc:>6.2f}%")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                'model_state': model.state_dict(),
                'val_acc':     best_val_acc / 100,
                'epoch':       epoch,
                'training':    'augmented',
            }, best_path)

    print(f"\n  ✅ {model_name}: best val acc = {best_val_acc:.2f}%")
    orig_acc = ORIG_VAL_ACCS.get(model_name, 0) * 100
    delta    = best_val_acc - orig_acc
    sym      = '↑' if delta >= 0 else '↓'
    print(f"     Original: {orig_acc:.2f}%  →  Augmented: {best_val_acc:.2f}%  ({sym}{abs(delta):.2f}%)")
    return best_val_acc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    # Collect files
    all_paths, all_labels = [], []
    for i, cls in enumerate(CLASSES):
        cls_dir = DATASET1_DIR / cls
        wavs    = list(cls_dir.glob('*.wav'))
        all_paths.extend(wavs)
        all_labels.extend([i] * len(wavs))
    print(f"\n  Total files: {len(all_paths)} across {N_CLASSES} classes")

    # Load noise pool for augmentation
    _load_noise_pool()

    # Load norm stats
    stats = json.load(open(FEAT_ROOT / 'dataset1' / 'norm_stats.json'))

    # Stratified split
    tr_paths, va_paths, tr_labels, va_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )

    train_ds = AugmentedCryDataset(tr_paths, tr_labels, stats, augment=True)
    val_ds   = AugmentedCryDataset(va_paths, va_labels, stats, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=0, pin_memory=True)

    # Which models to train
    models_to_train = (
        [args.model] if args.model else list(MODEL_REGISTRY.keys())
    )

    results = {}
    for model_name in models_to_train:
        if model_name not in MODEL_REGISTRY:
            print(f"Unknown model: {model_name}. Choose from: {list(MODEL_REGISTRY.keys())}")
            continue
        print(f"\n{'='*65}")
        print(f"  Training: {model_name.upper()}")
        print(f"{'='*65}")
        acc = train_model(model_name, train_loader, val_loader, args)
        results[model_name] = {'augmented_val_acc': acc,
                               'original_val_acc':  ORIG_VAL_ACCS.get(model_name, 0) * 100}

    # Summary
    print(f"\n{'='*65}")
    print(f"  AUGMENTED RETRAINING SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Model':20s}  {'Original':>10}  {'Augmented':>10}  {'Delta':>8}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*8}")
    for name, r in results.items():
        delta = r['augmented_val_acc'] - r['original_val_acc']
        sym   = '↑' if delta >= 0 else '↓'
        print(f"  {name:20s}  {r['original_val_acc']:>9.2f}%  "
              f"{r['augmented_val_acc']:>9.2f}%  {sym}{abs(delta):>6.2f}%")

    # Save summary
    report_path = MODEL_ROOT / 'augmented_training_report.json'
    json.dump(results, open(report_path, 'w'), indent=2)
    print(f"\n  Report saved: {report_path}")
    print(f"\n  ✅ Augmented models saved to: models/*/dataset1_augmented/best_model.pth")
    print(f"  ⚠️  Original models unchanged at: models/*/dataset1/best_model.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Retrain cry classifier with noise augmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stage_augmented_retrain.py
  python stage_augmented_retrain.py --model baseline_cnn --epochs 40
  python stage_augmented_retrain.py --epochs 50 --batch 32
        """
    )
    parser.add_argument('--model',  type=str, default=None,
                        choices=list(MODEL_REGISTRY.keys()) if 'MODEL_REGISTRY' in dir() else None,
                        help='Train only one model (default: all 4)')
    parser.add_argument('--epochs', type=int,   default=40,   help='Epochs per model (default: 40)')
    parser.add_argument('--batch',  type=int,   default=32,   help='Batch size (default: 32)')
    parser.add_argument('--lr',     type=float, default=5e-4, help='Learning rate (default: 0.0005)')
    args = parser.parse_args()

    main(args)
