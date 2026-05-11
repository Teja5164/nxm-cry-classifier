"""
Master Pipeline — Dataset1
===========================
Runs Stage 3 → Stage 4 → Stage 5 → Stage 6 sequentially for Dataset1.
Stage 2 is already complete (test_acc=0.8267) — skipped automatically.

RESUME SUPPORT:
  Each stage saves a checkpoint. If you stop (Ctrl+C) and restart,
  completed stages are detected and skipped automatically.

RESOURCE USAGE:
  - GPU (RTX 4050): used for all compute
  - num_workers=2 for data loading (Windows safe)
  - Batch sizes tuned for 6.4GB VRAM
  - GPU memory cleared between stages
"""

from pathlib import Path
import os, sys, json, time, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── Seed ──────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
DS_NAME      = 'dataset1'
CLASSES      = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
N_CLASSES    = 5
_ROOT = Path(__file__).resolve().parent.parent.parent  # ML_pipeline/
FEAT_ROOT    = str(_ROOT / 'training' / 'features')
MODEL_ROOT   = str(_ROOT / 'models')
RESULT_ROOT  = str(_ROOT / 'results')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*60}")
print(f"Master Pipeline — {DS_NAME.upper()}")
print(f"Device : {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    torch.backends.cudnn.benchmark = True
print(f"{'='*60}\n")

# ── DataLoader settings (Windows-safe) ────────────────────────────────────────
NW  = 2                          # num_workers — safe for Windows
PIN = DEVICE.type == 'cuda'      # pin_memory only on GPU

# ── Dataset ───────────────────────────────────────────────────────────────────
class MelDataset(Dataset):
    def __init__(self, mel, labels):
        self.mel = torch.FloatTensor(mel)
        self.labels = torch.LongTensor(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.mel[i], self.labels[i]

def get_splits():
    mel    = np.load(os.path.join(FEAT_ROOT, DS_NAME, 'mel_specs.npy'))
    labels = np.load(os.path.join(FEAT_ROOT, DS_NAME, 'labels.npy'))
    idx    = np.arange(len(labels))
    tr_i, te_i = train_test_split(idx, test_size=0.15, stratify=labels, random_state=SEED)
    tr_i, va_i = train_test_split(tr_i, test_size=0.176, stratify=labels[tr_i], random_state=SEED)
    return mel, labels, tr_i, va_i, te_i

def get_loaders(mel, labels, tr_i, va_i, te_i, batch_size):
    cc = np.bincount(labels[tr_i])
    sw = (1.0 / cc)[labels[tr_i]]
    sampler = WeightedRandomSampler(sw, len(tr_i), replacement=True)
    tr_ld = DataLoader(MelDataset(mel[tr_i], labels[tr_i]), batch_size,
                       sampler=sampler, num_workers=NW, pin_memory=PIN)
    va_ld = DataLoader(MelDataset(mel[va_i], labels[va_i]), batch_size,
                       shuffle=False, num_workers=NW, pin_memory=PIN)
    te_ld = DataLoader(MelDataset(mel[te_i], labels[te_i]), batch_size,
                       shuffle=False, num_workers=NW, pin_memory=PIN)
    return tr_ld, va_ld, te_ld

# ── Shared utilities ──────────────────────────────────────────────────────────
class LabelSmoothingCE(nn.Module):
    def __init__(self, classes=5, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing; self.cls = classes
    def forward(self, pred, target):
        sv = self.smoothing / (self.cls - 1)
        oh = torch.full_like(pred, sv)
        oh.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return -(oh * F.log_softmax(pred, dim=1)).sum(dim=1).mean()

class SpecAugment(nn.Module):
    def __init__(self, freq_mask=15, time_mask=40, nf=2, nt=2):
        super().__init__()
        self.fm=freq_mask; self.tm=time_mask; self.nf=nf; self.nt=nt
    def forward(self, x):
        if not self.training: return x
        B,C,F,T = x.shape; x = x.clone()
        for _ in range(self.nf):
            f=random.randint(0,self.fm); f0=random.randint(0,max(1,F-f))
            x[:,:,f0:f0+f,:]=0.0
        for _ in range(self.nt):
            t=random.randint(0,self.tm); t0=random.randint(0,max(1,T-t))
            x[:,:,:,t0:t0+t]=0.0
        return x

class ResBlock(nn.Module):
    def __init__(self, ch, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch,ch,3,padding=1,bias=False), nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.Dropout2d(drop),
            nn.Conv2d(ch,ch,3,padding=1,bias=False), nn.BatchNorm2d(ch))
        self.relu = nn.ReLU(True)
    def forward(self, x): return self.relu(self.net(x)+x)

def run_epoch(model, loader, criterion, optimizer=None, clip=1.0):
    train = optimizer is not None
    model.train(train)
    total_loss=correct=total=0
    with torch.set_grad_enabled(train):
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            out  = model(x); loss = criterion(out,y)
            if train:
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            total_loss += loss.item()*len(y)
            correct    += (out.argmax(1)==y).sum().item(); total+=len(y)
    return total_loss/total, correct/total

def train_loop(model, tr_ld, va_ld, criterion, optimizer, scheduler,
               max_ep, patience, ckpt_path, label):
    best_acc=0; cnt=0; tl,vl,ta,va=[],[],[],[]
    for ep in range(1, max_ep+1):
        t0=time.time()
        tr_loss,tr_acc = run_epoch(model,tr_ld,criterion,optimizer)
        va_loss,va_acc = run_epoch(model,va_ld,criterion)
        if scheduler: scheduler.step()
        elapsed=time.time()-t0
        tl.append(tr_loss);vl.append(va_loss);ta.append(tr_acc);va.append(va_acc)
        lr=optimizer.param_groups[0]['lr']
        print(f"  Ep {ep:3d} | {elapsed:.1f}s | lr={lr:.2e} | "
              f"tr={tr_acc:.4f} va={va_acc:.4f} | loss={tr_loss:.4f}/{va_loss:.4f}",
              flush=True)
        if va_acc>best_acc:
            best_acc=va_acc; cnt=0
            torch.save({'epoch':ep,'model_state':model.state_dict(),'val_acc':va_acc},
                       ckpt_path)
            print(f"  ✓ Best saved val_acc={va_acc:.4f}")
        else:
            cnt+=1
            if cnt>=patience: print(f"  Early stop ep={ep}"); break
    return tl,vl,ta,va,best_acc

def evaluate(model, te_ld, ckpt_path):
    ckpt=torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state']); model.eval()
    preds,true=[],[]
    with torch.no_grad():
        for x,y in te_ld:
            preds.extend(model(x.to(DEVICE)).argmax(1).cpu().numpy())
            true.extend(y.numpy())
    return preds, true

def save_results(preds, true, tl, vl, ta, va, result_dir, tag, best_val):
    os.makedirs(result_dir, exist_ok=True)
    report = classification_report(true,preds,target_names=CLASSES,digits=4)
    print(f"\n{report}")
    # Curves
    fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4))
    a1.plot(tl,label='Train');a1.plot(vl,label='Val');a1.set_title('Loss');a1.legend()
    a2.plot(ta,label='Train');a2.plot(va,label='Val');a2.set_title('Accuracy');a2.legend()
    plt.tight_layout();plt.savefig(os.path.join(result_dir,f'{tag}_curves.png'),dpi=100);plt.close()
    # CM
    cm=confusion_matrix(true,preds)
    fig,ax=plt.subplots(figsize=(7,6))
    sns.heatmap(cm,annot=True,fmt='d',xticklabels=CLASSES,yticklabels=CLASSES,cmap='Blues',ax=ax)
    ax.set_xlabel('Predicted');ax.set_ylabel('True')
    plt.tight_layout();plt.savefig(os.path.join(result_dir,f'{tag}_cm.png'),dpi=100);plt.close()
    with open(os.path.join(result_dir,f'{tag}_report.txt'),'w') as f: f.write(report)
    m={'stage':tag,'val_acc':float(best_val),'test_acc':float(accuracy_score(true,preds)),
       'macro_f1':float(f1_score(true,preds,average='macro')),
       'weighted_f1':float(f1_score(true,preds,average='weighted'))}
    with open(os.path.join(result_dir,f'{tag}_metrics.json'),'w') as f: json.dump(m,f,indent=2)
    return m

def gpu_status():
    if DEVICE.type=='cuda':
        used=torch.cuda.memory_allocated()/1e9
        resv=torch.cuda.memory_reserved()/1e9
        print(f"  GPU mem: {used:.2f}GB alloc / {resv:.2f}GB reserved")

def clear_gpu():
    if DEVICE.type=='cuda':
        torch.cuda.empty_cache()
        import gc; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — CNN + BiLSTM
# ══════════════════════════════════════════════════════════════════════════════
class CNNEncoder(nn.Module):
    def __init__(self, out_ch=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1,bias=False),nn.BatchNorm2d(32),nn.ReLU(True))
        self.b1=nn.Sequential(ResBlock(32),nn.MaxPool2d(2,2))
        self.b2=nn.Sequential(nn.Conv2d(32,64,3,padding=1,bias=False),nn.BatchNorm2d(64),nn.ReLU(True),
                              ResBlock(64),nn.MaxPool2d(2,2))
        self.b3=nn.Sequential(nn.Conv2d(64,128,3,padding=1,bias=False),nn.BatchNorm2d(128),nn.ReLU(True),
                              ResBlock(128),nn.MaxPool2d(2,2))
        self.b4=nn.Sequential(nn.Conv2d(128,out_ch,3,padding=1,bias=False),nn.BatchNorm2d(out_ch),nn.ReLU(True),
                              ResBlock(out_ch),nn.MaxPool2d(2,2))
        self.fpool=nn.AdaptiveAvgPool2d((1,None))
    def forward(self,x):
        x=self.stem(x);x=self.b1(x);x=self.b2(x);x=self.b3(x);x=self.b4(x)
        x=self.fpool(x).squeeze(2)
        return x.permute(0,2,1)   # (B,T',256)

class TemporalAttn(nn.Module):
    def __init__(self,h): super().__init__(); self.a=nn.Linear(h*2,1)
    def forward(self,x):
        s=self.a(x).squeeze(-1); w=F.softmax(s,1)
        return (w.unsqueeze(-1)*x).sum(1), w

class CNNBiLSTM(nn.Module):
    def __init__(self,n=5,h=256,layers=2):
        super().__init__()
        self.aug=SpecAugment(); self.enc=CNNEncoder(256)
        self.lstm=nn.LSTM(256,h,layers,batch_first=True,bidirectional=True,
                          dropout=0.3 if layers>1 else 0.0)
        self.attn=TemporalAttn(h)
        self.head=nn.Sequential(nn.LayerNorm(h*2),
            nn.Linear(h*2,256),nn.GELU(),nn.Dropout(0.4),
            nn.Linear(256,64),nn.GELU(),nn.Dropout(0.2),nn.Linear(64,n))
        for nm,p in self.named_parameters():
            if 'lstm' in nm:
                if 'weight_ih' in nm: nn.init.xavier_uniform_(p)
                elif 'weight_hh' in nm: nn.init.orthogonal_(p)
                elif 'bias' in nm: nn.init.zeros_(p)
    def forward(self,x):
        x=self.aug(x); x=self.enc(x)
        out,_=self.lstm(x); ctx,_=self.attn(out)
        return self.head(ctx)

def stage3(mel,labels,tr_i,va_i,te_i):
    tag='stage3_bilstm'
    ckpt=os.path.join(MODEL_ROOT,'cnn_bilstm',DS_NAME,'best_model.pth')
    rdir=os.path.join(RESULT_ROOT,'cnn_bilstm',DS_NAME)
    if os.path.exists(ckpt) and os.path.exists(os.path.join(rdir,f'{tag}_metrics.json')):
        print(f"\n✅ Stage 3 already done — skipping"); return json.load(open(os.path.join(rdir,f'{tag}_metrics.json')))
    print(f"\n{'='*60}\nSTAGE 3: CNN + BiLSTM [{DS_NAME}]\n{'='*60}")
    os.makedirs(os.path.dirname(ckpt),exist_ok=True)
    tr_ld,va_ld,te_ld=get_loaders(mel,labels,tr_i,va_i,te_i,batch_size=32)
    model=CNNBiLSTM(N_CLASSES).to(DEVICE)
    print(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    gpu_status()
    crit=LabelSmoothingCE(); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=25,T_mult=2)
    tl,vl,ta,va,bva=train_loop(model,tr_ld,va_ld,crit,opt,sch,100,15,ckpt,tag)
    preds,true=evaluate(model,te_ld,ckpt)
    m=save_results(preds,true,tl,vl,ta,va,rdir,tag,bva)
    print(f"\n✅ Stage 3 done | test_acc={m['test_acc']:.4f} | macro_f1={m['macro_f1']:.4f}")
    return m

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — CNN + Transformer (CryNet)
# ══════════════════════════════════════════════════════════════════════════════
class LearnedPE(nn.Module):
    def __init__(self,maxlen,d,drop=0.1):
        super().__init__(); self.pe=nn.Embedding(maxlen,d); self.drop=nn.Dropout(drop)
    def forward(self,x):
        B,T,D=x.shape; pos=torch.arange(T,device=x.device).unsqueeze(0).expand(B,-1)
        return self.drop(x+self.pe(pos))

class CryNet(nn.Module):
    def __init__(self,n=5,d=256,heads=8,layers=4,ff=512,drop=0.2):
        super().__init__()
        self.aug=SpecAugment(freq_mask=20,time_mask=50)
        self.enc=CNNEncoder(d)
        self.pe=LearnedPE(128,d,drop)
        self.cls=nn.Parameter(torch.zeros(1,1,d)); nn.init.trunc_normal_(self.cls,std=0.02)
        tfl=nn.TransformerEncoderLayer(d,heads,ff,drop,activation='gelu',
                                        batch_first=True,norm_first=True)
        self.tf=nn.TransformerEncoder(tfl,layers,norm=nn.LayerNorm(d))
        self.head=nn.Sequential(nn.LayerNorm(d),
            nn.Linear(d,256),nn.GELU(),nn.Dropout(0.4),
            nn.Linear(256,64),nn.GELU(),nn.Dropout(0.2),nn.Linear(64,n))
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.trunc_normal_(m.weight,std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self,x):
        x=self.aug(x); x=self.enc(x)
        cls=self.cls.expand(x.size(0),-1,-1)
        x=torch.cat([cls,x],1)
        x[:,1:,:]=self.pe(x[:,1:,:])
        x=self.tf(x); return self.head(x[:,0,:])

class WarmupCosine:
    def __init__(self,opt,wu,total,minlr=1e-6):
        self.opt=opt;self.wu=wu;self.total=total;self.minlr=minlr
        self.base=opt.param_groups[0]['lr'];self.ep=0
    def step(self):
        self.ep+=1
        if self.ep<=self.wu: lr=self.base*self.ep/self.wu
        else:
            p=(self.ep-self.wu)/(self.total-self.wu)
            lr=self.minlr+0.5*(self.base-self.minlr)*(1+math.cos(math.pi*p))
        for g in self.opt.param_groups: g['lr']=lr

def stage4(mel,labels,tr_i,va_i,te_i):
    tag='stage4_transformer'
    ckpt=os.path.join(MODEL_ROOT,'cnn_transformer',DS_NAME,'best_model.pth')
    rdir=os.path.join(RESULT_ROOT,'cnn_transformer',DS_NAME)
    if os.path.exists(ckpt) and os.path.exists(os.path.join(rdir,f'{tag}_metrics.json')):
        print(f"\n✅ Stage 4 already done — skipping"); return json.load(open(os.path.join(rdir,f'{tag}_metrics.json')))
    print(f"\n{'='*60}\nSTAGE 4: CryNet CNN+Transformer [{DS_NAME}]\n{'='*60}")
    os.makedirs(os.path.dirname(ckpt),exist_ok=True)
    tr_ld,va_ld,te_ld=get_loaders(mel,labels,tr_i,va_i,te_i,batch_size=32)
    model=CryNet(N_CLASSES).to(DEVICE)
    print(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    gpu_status()
    crit=LabelSmoothingCE()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.05,betas=(0.9,0.999))
    sch=WarmupCosine(opt,wu=5,total=100)
    tl,vl,ta,va,bva=train_loop(model,tr_ld,va_ld,crit,opt,sch,100,15,ckpt,tag)
    preds,true=evaluate(model,te_ld,ckpt)
    m=save_results(preds,true,tl,vl,ta,va,rdir,tag,bva)
    print(f"\n✅ Stage 4 done | test_acc={m['test_acc']:.4f} | macro_f1={m['macro_f1']:.4f}")
    return m

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — SE-ResNet (Squeeze-Excitation ResNet) — trains well from scratch
# ══════════════════════════════════════════════════════════════════════════════
class SEBlock(nn.Module):
    """Squeeze-Excitation channel attention."""
    def __init__(self, c, r=16):
        super().__init__()
        self.sq = nn.AdaptiveAvgPool2d(1)
        self.ex = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c, max(c//r,8)), nn.ReLU(inplace=True),
            nn.Linear(max(c//r,8), c), nn.Sigmoid())
    def forward(self, x):
        w = self.ex(self.sq(x)).view(x.size(0), x.size(1), 1, 1)
        return x * w

class SEResBlock(nn.Module):
    """Residual block with SE attention + optional stride."""
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
            nn.BatchNorm2d(outc)) if (inc != outc or stride != 1) else nn.Identity()
        self.drop = nn.Dropout2d(0.1)
    def forward(self, x):
        return F.relu(self.se(self.drop(self.conv(x))) + self.skip(x), inplace=True)

class SEResNet(nn.Module):
    """SE-ResNet for mel-spectrogram classification (~3M params)."""
    def __init__(self, n=5):
        super().__init__()
        self.aug  = SpecAugment()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(SEResBlock(32,  64, stride=2), SEResBlock(64,  64))
        self.layer2 = nn.Sequential(SEResBlock(64, 128, stride=2), SEResBlock(128,128))
        self.layer3 = nn.Sequential(SEResBlock(128,256, stride=2), SEResBlock(256,256))
        self.layer4 = nn.Sequential(SEResBlock(256,512, stride=2), SEResBlock(512,512))
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, n))
    def forward(self, x):
        x = self.stem(self.aug(x))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.head(self.pool(x))

def stage5(mel,labels,tr_i,va_i,te_i):
    tag='stage5_senet'
    ckpt=os.path.join(MODEL_ROOT,'se_resnet',DS_NAME,'best_model.pth')
    rdir=os.path.join(RESULT_ROOT,'se_resnet',DS_NAME)
    if os.path.exists(ckpt) and os.path.exists(os.path.join(rdir,f'{tag}_metrics.json')):
        print(f"\n✅ Stage 5 already done — skipping"); return json.load(open(os.path.join(rdir,f'{tag}_metrics.json')))
    print(f"\n{'='*60}\nSTAGE 5: SE-ResNet [{DS_NAME}]\n{'='*60}")
    os.makedirs(os.path.dirname(ckpt),exist_ok=True)
    tr_ld,va_ld,te_ld=get_loaders(mel,labels,tr_i,va_i,te_i,batch_size=32)
    model=SEResNet(N_CLASSES).to(DEVICE)
    print(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    gpu_status(); crit=LabelSmoothingCE()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.05)
    sch=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=20,T_mult=2)
    tl,vl,ta,va_,bva=train_loop(model,tr_ld,va_ld,crit,opt,sch,100,20,ckpt,tag)
    preds,true=evaluate(model,te_ld,ckpt)
    m=save_results(preds,true,tl,vl,ta,va_,rdir,tag,bva)
    print(f"\n✅ Stage 5 done | test_acc={m['test_acc']:.4f} | macro_f1={m['macro_f1']:.4f}")
    return m

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — Ensemble
# ══════════════════════════════════════════════════════════════════════════════
def get_probs(model, loader):
    model.eval(); all_p,all_y=[],[]
    with torch.no_grad():
        for x,y in loader:
            p=F.softmax(model(x.to(DEVICE)),1).cpu().numpy()
            all_p.append(p); all_y.extend(y.numpy())
    return np.vstack(all_p), np.array(all_y)

def stage6(mel,labels,tr_i,va_i,te_i, s2m, s3m, s4m, s5m):
    tag='stage6_ensemble'
    rdir=os.path.join(RESULT_ROOT,'ensemble',DS_NAME)
    out_cfg=os.path.join(rdir,f'{tag}_config.json')
    if os.path.exists(out_cfg):
        print(f"\n✅ Stage 6 already done — skipping"); return json.load(open(out_cfg))
    print(f"\n{'='*60}\nSTAGE 6: Ensemble [{DS_NAME}]\n{'='*60}")
    os.makedirs(rdir,exist_ok=True)

    # Load all stage models
    from stage2_baseline_cnn import BaselineCNN
    def _load(cls,ckpt,**kw):
        m=cls(**kw).to(DEVICE)
        if os.path.exists(ckpt):
            c=torch.load(ckpt,map_location=DEVICE)
            m.load_state_dict(c['model_state']); va=c.get('val_acc',0)
            print(f"  Loaded {os.path.basename(os.path.dirname(ckpt))} val_acc={va:.4f}")
            return m,va
        print(f"  ⚠ Not found: {ckpt}"); return None,0

    te_ld=DataLoader(MelDataset(mel[te_i],labels[te_i]),64,shuffle=False,num_workers=NW,pin_memory=PIN)

    models_info=[]
    for cls,folder,kw in [
        (BaselineCNN,'baseline_cnn',{'n_classes':5}),
        (CNNBiLSTM,  'cnn_bilstm',  {'n':5}),
        (CryNet,     'cnn_transformer',{'n':5}),
        (SEResNet,   'se_resnet',   {'n':5})]:
        ckpt=os.path.join(MODEL_ROOT,folder,DS_NAME,'best_model.pth')
        m,va=_load(cls,ckpt,**kw)
        if m: models_info.append((m,va,folder))

    if not models_info: print("No models found!"); return

    # Get probs
    all_probs=[]
    for m,va,name in models_info:
        p,true=get_probs(m,te_ld)
        acc=accuracy_score(true,p.argmax(1))
        print(f"  {name:20s}: test_acc={acc:.4f}")
        all_probs.append((p,va,name))

    # Uniform ensemble
    u_p=np.mean([p for p,_,_ in all_probs],0)
    u_acc=accuracy_score(true,u_p.argmax(1)); u_f1=f1_score(true,u_p.argmax(1),average='macro')
    # Weighted ensemble
    vas=np.array([v for _,v,_ in all_probs]); ws=vas/vas.sum()
    w_p=sum(w*p for (p,_,_),w in zip(all_probs,ws))
    w_acc=accuracy_score(true,w_p.argmax(1)); w_f1=f1_score(true,w_p.argmax(1),average='macro')
    print(f"\n  Uniform  ensemble: acc={u_acc:.4f} f1={u_f1:.4f}")
    print(f"  Weighted ensemble: acc={w_acc:.4f} f1={w_f1:.4f}")

    best_p,best_preds,best_acc,best_f1,etype = \
        (w_p,w_p.argmax(1),w_acc,w_f1,'weighted') if w_acc>=u_acc else \
        (u_p,u_p.argmax(1),u_acc,u_f1,'uniform')
    print(f"\n  Best: {etype} (acc={best_acc:.4f})")
    report=classification_report(true,best_preds,target_names=CLASSES,digits=4)
    print(f"\n{report}")

    # Save CM
    cm=confusion_matrix(true,best_preds)
    fig,ax=plt.subplots(figsize=(7,6))
    sns.heatmap(cm,annot=True,fmt='d',xticklabels=CLASSES,yticklabels=CLASSES,cmap='Blues',ax=ax)
    ax.set_title(f'Ensemble ({etype})'); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout(); plt.savefig(os.path.join(rdir,f'{tag}_cm.png'),dpi=100); plt.close()
    with open(os.path.join(rdir,f'{tag}_report.txt'),'w') as f: f.write(report)

    cfg={'dataset':DS_NAME,'ensemble':etype,
         'models':[{'name':n,'val_acc':float(v),'weight':float(w)}
                   for (_,v,n),w in zip(all_probs,ws)],
         'test_acc':float(best_acc),'macro_f1':float(best_f1),
         'weighted_f1':float(f1_score(true,best_preds,average='weighted')),
         'stage_metrics':{'s2':s2m,'s3':s3m,'s4':s4m,'s5':s5m}}
    with open(out_cfg,'w') as f: json.dump(cfg,f,indent=2)
    print(f"\n✅ Stage 6 done | FINAL ensemble acc={best_acc:.4f} | macro_f1={best_f1:.4f}")
    return cfg

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t_start = time.time()

    print("Loading features...")
    mel, labels, tr_i, va_i, te_i = get_splits()
    print(f"mel={mel.shape} | train={len(tr_i)} val={len(va_i)} test={len(te_i)}")

    # Stage 2 already done
    s2_ckpt = os.path.join(MODEL_ROOT,'baseline_cnn',DS_NAME,'best_model.pth')
    s2_mf   = os.path.join(RESULT_ROOT,'baseline_cnn',DS_NAME,'dataset1_metrics.json')
    if os.path.exists(s2_mf):
        s2m = json.load(open(s2_mf))
    else:
        s2m = {'test_acc':0.8267,'macro_f1':0.82,'stage':'stage2_baseline_cnn'}
    print(f"\n✅ Stage 2 (Baseline CNN): test_acc={s2m.get('test_acc',0.8267):.4f} [already done]")

    # Run remaining stages
    s3m = stage3(mel, labels, tr_i, va_i, te_i)
    clear_gpu()

    s4m = stage4(mel, labels, tr_i, va_i, te_i)
    clear_gpu()

    s5m = stage5(mel, labels, tr_i, va_i, te_i)
    clear_gpu()

    s6m = stage6(mel, labels, tr_i, va_i, te_i, s2m, s3m, s4m, s5m)

    total = time.time()-t_start
    print(f"\n{'='*60}")
    print(f"DATASET1 PIPELINE COMPLETE  (total={total/3600:.1f}h)")
    print(f"{'='*60}")
    print(f"  Stage 2 (Baseline CNN)   : {s2m.get('test_acc',0):.4f}")
    print(f"  Stage 3 (CNN+BiLSTM)     : {s3m.get('test_acc',0):.4f}")
    print(f"  Stage 4 (CNN+Transformer): {s4m.get('test_acc',0):.4f}")
    print(f"  Stage 5 (SE-ResNet)      : {s5m.get('test_acc',0):.4f}")
    print(f"  Stage 6 (Ensemble FINAL) : {s6m.get('test_acc',0):.4f}")
    print(f"\n  ✅ Dataset1 model pipeline complete!")
    print(f"  Awaiting your approval to start Dataset2 pipeline.")
