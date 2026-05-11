"""
Stage 7: Inference Pipeline
============================
Full end-to-end inference:
  - Input:  Any .wav audio file (5–15 seconds; longer → centre 10s crop)
  - Output: Predicted class + confidence + per-class probabilities

Usage:
  python stage7_inference.py --audio path/to/cry.wav --dataset dataset2
  python stage7_inference.py --audio cry.wav           (uses dataset2 by default)

Pipeline:
  1. Load & preprocess audio (resample → 22050Hz, mono, normalize, 10s clip)
  2. Extract mel spectrogram (same params as training)
  3. Z-score normalize (using saved norm_stats.json)
  4. Run through all available models
  5. Temperature-calibrated ensemble → final prediction
  6. Output: class label + confidence score
"""

from pathlib import Path
import os, sys, json, argparse, warnings
import numpy as np
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # prevent OpenMP crash with librosa+torch
warnings.filterwarnings('ignore')

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Infant Cry Classifier Inference')
parser.add_argument('--audio',   required=True,  help='Path to input .wav file')
parser.add_argument('--dataset', default='dataset2', choices=['dataset1', 'dataset2'],
                    help='Which trained models to use (default: dataset2)')
parser.add_argument('--verbose', action='store_true', help='Show per-model and per-class scores')
args = parser.parse_args()

# ── Constants ─────────────────────────────────────────────────────────────────
CLASSES    = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
SR         = 22050
DURATION   = 10.0          # seconds
N_MEL      = 128
N_FFT      = 2048
HOP_LEN    = 512
FMIN       = 50
FMAX       = 8000
N_SAMPLES  = int(SR * DURATION)

_ROOT = Path(__file__).resolve().parent.parent.parent  # ML_pipeline/
MODEL_ROOT = str(_ROOT / 'models')
FEAT_ROOT  = str(_ROOT / 'training' / 'features')

# ── Check dependencies ────────────────────────────────────────────────────────
try:
    import torch, librosa
    import torch.nn.functional as F
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install torch librosa")
    sys.exit(1)

import torch.nn as nn
import random
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Shared SpecAugment ────────────────────────────────────────────────────────
class SpecAugment(nn.Module):
    def __init__(self, freq_mask=15, time_mask=40, nf=2, nt=2):
        super().__init__()
        self.fm=freq_mask; self.tm=time_mask; self.nf=nf; self.nt=nt
    def forward(self, x):
        if not self.training: return x
        B,C,F,T = x.shape; x = x.clone()
        for _ in range(self.nf):
            f=random.randint(0,self.fm); f0=random.randint(0,max(1,F-f)); x[:,:,f0:f0+f,:]=0.0
        for _ in range(self.nt):
            t=random.randint(0,self.tm); t0=random.randint(0,max(1,T-t)); x[:,:,:,t0:t0+t]=0.0
        return x

# ── BaselineCNN ───────────────────────────────────────────────────────────────
class _CnnResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.drop  = nn.Dropout2d(0.1)
        self.skip  = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch)) if (stride != 1 or in_ch != out_ch) else nn.Sequential()
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))); out = self.drop(out)
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
        self.bn_fc= nn.BatchNorm1d(128)
        self.fc2  = nn.Linear(128, n_classes)
    def forward(self, x):
        x = self.spec_aug(x); x = self.stem(x)
        x = self.stage1(x); x = self.stage2(x); x = self.stage3(x); x = self.stage4(x)
        x = self.gap(x).flatten(1); x = self.drop(x)
        return self.fc2(self.drop(F.relu(self.bn_fc(self.fc1(x)))))

# ── CNNEncoder helper classes (for CNN+BiLSTM and CryNet) ────────────────────
class _EncResBlock(nn.Module):
    def __init__(self, ch, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch,ch,3,padding=1,bias=False), nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.Dropout2d(drop), nn.Conv2d(ch,ch,3,padding=1,bias=False), nn.BatchNorm2d(ch))
        self.relu = nn.ReLU(True)
    def forward(self, x): return self.relu(self.net(x)+x)

class CNNEncoder(nn.Module):
    def __init__(self, out_ch=256):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1,32,3,padding=1,bias=False), nn.BatchNorm2d(32), nn.ReLU(True))
        self.b1=nn.Sequential(_EncResBlock(32), nn.MaxPool2d(2,2))
        self.b2=nn.Sequential(nn.Conv2d(32,64,3,padding=1,bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
                              _EncResBlock(64), nn.MaxPool2d(2,2))
        self.b3=nn.Sequential(nn.Conv2d(64,128,3,padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
                              _EncResBlock(128), nn.MaxPool2d(2,2))
        self.b4=nn.Sequential(nn.Conv2d(128,out_ch,3,padding=1,bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True),
                              _EncResBlock(out_ch), nn.MaxPool2d(2,2))
        self.fpool=nn.AdaptiveAvgPool2d((1,None))
    def forward(self, x):
        x=self.stem(x); x=self.b1(x); x=self.b2(x); x=self.b3(x); x=self.b4(x)
        return self.fpool(x).squeeze(2).transpose(1,2)

class TemporalAttn(nn.Module):
    def __init__(self, h): super().__init__(); self.a=nn.Linear(h*2,1)
    def forward(self, x):
        s=self.a(x).squeeze(-1); w=F.softmax(s,1)
        return (w.unsqueeze(-1)*x).sum(1), w

# ── CNN+BiLSTM ────────────────────────────────────────────────────────────────
class CNNBiLSTM(nn.Module):
    def __init__(self, n=5, h=256, layers=2):
        super().__init__()
        self.aug=SpecAugment(); self.enc=CNNEncoder(256)
        self.lstm=nn.LSTM(256,h,layers,batch_first=True,bidirectional=True,
                          dropout=0.3 if layers>1 else 0.0)
        self.attn=TemporalAttn(h)
        self.head=nn.Sequential(nn.LayerNorm(h*2),
            nn.Linear(h*2,256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256,64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64,n))
    def forward(self, x):
        x=self.aug(x); x=self.enc(x)
        out,_=self.lstm(x); ctx,_=self.attn(out)
        return self.head(ctx)

# ── CryNet (CNN + Transformer) ────────────────────────────────────────────────
class LearnedPE(nn.Module):
    def __init__(self, maxlen, d, drop=0.1):
        super().__init__(); self.pe=nn.Embedding(maxlen,d); self.drop=nn.Dropout(drop)
    def forward(self, x):
        B,T,D=x.shape; pos=torch.arange(T,device=x.device).unsqueeze(0).expand(B,-1)
        return self.drop(x+self.pe(pos))

class CryNet(nn.Module):
    def __init__(self, n=5, d=256, heads=8, layers=4, ff=512, drop=0.2):
        super().__init__()
        self.aug=SpecAugment(freq_mask=20, time_mask=50)
        self.enc=CNNEncoder(d)
        self.pe=LearnedPE(128,d,drop)
        self.cls=nn.Parameter(torch.zeros(1,1,d)); nn.init.trunc_normal_(self.cls,std=0.02)
        tfl=nn.TransformerEncoderLayer(d,heads,ff,drop,activation='gelu',batch_first=True,norm_first=True)
        self.tf=nn.TransformerEncoder(tfl,layers,norm=nn.LayerNorm(d))
        self.head=nn.Sequential(nn.LayerNorm(d),
            nn.Linear(d,256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256,64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64,n))
    def forward(self, x):
        x=self.aug(x); x=self.enc(x); x=self.pe(x)
        cls=self.cls.expand(x.size(0),-1,-1)
        x=self.tf(torch.cat([cls,x],1))
        return self.head(x[:,0])

# ── SE-ResNet ─────────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.sq = nn.AdaptiveAvgPool2d(1)
        self.ex = nn.Sequential(nn.Flatten(), nn.Linear(c,max(c//r,8)), nn.ReLU(inplace=True),
                                nn.Linear(max(c//r,8),c), nn.Sigmoid())
    def forward(self, x): return x * self.ex(self.sq(x)).view(x.size(0),x.size(1),1,1)

class SEResBlock(nn.Module):
    def __init__(self, inc, outc, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inc,outc,3,stride=stride,padding=1,bias=False), nn.BatchNorm2d(outc), nn.ReLU(inplace=True),
            nn.Conv2d(outc,outc,3,padding=1,bias=False), nn.BatchNorm2d(outc))
        self.se   = SEBlock(outc)
        self.skip = nn.Sequential(nn.Conv2d(inc,outc,1,stride=stride,bias=False),
                                  nn.BatchNorm2d(outc)) if (inc!=outc or stride!=1) else nn.Identity()
    def forward(self, x): return F.relu(self.se(self.conv(x)) + self.skip(x), True)

class SEResNet(nn.Module):
    def __init__(self, n=5):
        super().__init__()
        self.aug    = SpecAugment()
        self.stem   = nn.Sequential(nn.Conv2d(1,32,3,padding=1,bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(SEResBlock(32,  64,  stride=2), SEResBlock(64,  64))
        self.layer2 = nn.Sequential(SEResBlock(64,  128, stride=2), SEResBlock(128, 128))
        self.layer3 = nn.Sequential(SEResBlock(128, 256, stride=2), SEResBlock(256, 256))
        self.layer4 = nn.Sequential(SEResBlock(256, 512, stride=2), SEResBlock(512, 512))
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(nn.Flatten(), nn.Dropout(0.4),
                                    nn.Linear(512,256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256,n))
    def forward(self, x):
        x = self.stem(self.aug(x))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.head(self.pool(x))

# ── Audio Preprocessing ───────────────────────────────────────────────────────
def preprocess_audio(path):
    """Load, resample, normalize, and clip audio to 10 seconds."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio not found: {path}")

    audio, sr = librosa.load(path, sr=SR, mono=True)

    if len(audio) == 0:
        raise ValueError("Audio file is empty")

    # Peak normalize
    peak = np.max(np.abs(audio))
    if peak > 0: audio = audio / peak

    # Clip to exactly N_SAMPLES
    if len(audio) >= N_SAMPLES:
        # Take centre 10 seconds
        start = (len(audio) - N_SAMPLES) // 2
        audio = audio[start:start + N_SAMPLES]
    else:
        # Pad by repeating (loop padding, better than zero-pad for short cries)
        n_reps = (N_SAMPLES // len(audio)) + 2
        audio  = np.tile(audio, n_reps)[:N_SAMPLES]

    return audio

def extract_mel(audio):
    """Extract log-mel spectrogram matching training features."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LEN,
        n_mels=N_MEL, fmin=FMIN, fmax=FMAX)
    log_mel = librosa.power_to_db(mel + 1e-6, ref=np.max)
    return log_mel  # (128, T)

def normalize_mel(log_mel, norm_stats_path):
    """Z-score normalize using saved training statistics."""
    with open(norm_stats_path) as f:
        stats = json.load(f)
    mean = np.array(stats['mel_mean'])   # (128,)
    std  = np.array(stats['mel_std'])    # (128,)
    std  = np.where(std < 1e-8, 1.0, std)

    # Normalize per frequency bin
    log_mel = (log_mel - mean[:, None]) / std[:, None]

    # Ensure correct time dimension
    target_T = 431
    if log_mel.shape[1] < target_T:
        pad = target_T - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0,0),(0,pad)), mode='edge')
    elif log_mel.shape[1] > target_T:
        log_mel = log_mel[:, :target_T]

    return log_mel  # (128, 431)

def prepare_tensor(log_mel):
    """Convert to (1, 1, 128, 431) tensor."""
    x = torch.FloatTensor(log_mel).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 431)
    return x.to(DEVICE)

# ── Model Loading ─────────────────────────────────────────────────────────────
def try_load_model(model_class, ckpt_path, name, **kwargs):
    try:
        m = model_class(**kwargs).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        m.load_state_dict(ckpt['model_state'])
        m.eval()
        val_acc = ckpt.get('val_acc', 0.0)
        return m, val_acc, name
    except FileNotFoundError:
        return None, 0, name
    except Exception as e:
        print(f"  ⚠ Could not load {name}: {e}")
        return None, 0, name

def load_all_models(ds_name):
    # All model classes are defined inline above — no imports needed
    mroot = MODEL_ROOT
    loaded = []
    for cls, folder, name, kwargs in [
        (BaselineCNN,  'baseline_cnn',    'BaselineCNN',  {'n_classes': 5}),
        (CNNBiLSTM,    'cnn_bilstm',      'CNN+BiLSTM',   {'n': 5}),
        (CryNet,       'cnn_transformer', 'CryNet',       {'n': 5}),
        (SEResNet,     'se_resnet',       'SE-ResNet',    {'n': 5})]:
        ckpt = os.path.join(mroot, folder, ds_name, 'best_model.pth')
        m, va, nm = try_load_model(cls, ckpt, name, **kwargs)
        if m is not None:
            loaded.append((m, va, nm))
    return loaded

# ── Inference ─────────────────────────────────────────────────────────────────
def predict(audio_path, ds_name='dataset2', verbose=False):
    print(f"\n{'='*55}")
    print(f"Infant Cry Classifier")
    print(f"{'='*55}")
    print(f"Audio  : {audio_path}")
    print(f"Dataset: {ds_name}")
    print(f"Device : {DEVICE}")
    print()

    # 1. Preprocess
    print("Step 1: Preprocessing audio...")
    audio = preprocess_audio(audio_path)
    dur   = len(audio) / SR
    print(f"  Duration: {dur:.1f}s | SR: {SR}Hz")

    # 2. Extract mel
    print("Step 2: Extracting mel spectrogram...")
    log_mel = extract_mel(audio)

    # 3. Normalize
    norm_path = os.path.join(FEAT_ROOT, ds_name, 'norm_stats.json')
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"norm_stats.json not found at {norm_path}\n"
                                "Run stage1_extract_features.py first.")
    log_mel = normalize_mel(log_mel, norm_path)
    x = prepare_tensor(log_mel)   # (1, 1, 128, 431)

    # 4. Load models
    print("Step 3: Loading models...")
    models = load_all_models(ds_name)
    if not models:
        raise RuntimeError(f"No trained models found for {ds_name}. Run stages 2–5 first.")
    print(f"  Loaded {len(models)} models: {[n for _,_,n in models]}")

    # 5. Get predictions
    print("Step 4: Running ensemble inference...")
    all_probs = []
    for model, val_acc, name in models:
        with torch.no_grad():
            logits = model(x)
            probs  = F.softmax(logits, dim=1).cpu().numpy()[0]  # (5,)
        all_probs.append((probs, val_acc, name))
        if verbose:
            pred_cls = CLASSES[probs.argmax()]
            print(f"  {name:15s} → {pred_cls:12s} ({probs.max()*100:.1f}%)")

    # 6. Weighted ensemble
    va_arr  = np.array([va for _, va, _ in all_probs])
    if va_arr.sum() > 0:
        weights = va_arr / va_arr.sum()
    else:
        weights = np.ones(len(all_probs)) / len(all_probs)
    ensemble_probs = sum(w * p for (p, _, _), w in zip(all_probs, weights))
    ensemble_probs = ensemble_probs / ensemble_probs.sum()  # renormalise

    # 7. Final prediction
    pred_idx = ensemble_probs.argmax()
    pred_cls = CLASSES[pred_idx]
    confidence = ensemble_probs[pred_idx] * 100

    # ── Output ────────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  PREDICTION : {pred_cls.upper()}")
    print(f"  CONFIDENCE : {confidence:.1f}%")
    print(f"{'='*55}")

    if verbose or confidence < 50:
        print("\nPer-class probabilities:")
        sorted_idx = ensemble_probs.argsort()[::-1]
        for i in sorted_idx:
            bar = '█' * int(ensemble_probs[i] * 30)
            print(f"  {CLASSES[i]:12s}: {ensemble_probs[i]*100:5.1f}%  {bar}")

    # Reliability note
    if confidence >= 80:
        rel = "HIGH confidence"
    elif confidence >= 60:
        rel = "MEDIUM confidence — consider rechecking"
    else:
        rel = "LOW confidence — audio may be unclear or noisy"
    print(f"\nReliability: {rel}")
    print()

    return {
        'prediction': pred_cls,
        'confidence': float(confidence),
        'probabilities': {CLASSES[i]: float(ensemble_probs[i]*100) for i in range(5)},
        'reliability': rel
    }

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    result = predict(args.audio, args.dataset, args.verbose)

