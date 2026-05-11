"""
Script 4: CLAP zero-shot classification of cry audio into 5 classes.
Uses laion/larger_clap_general via HuggingFace Transformers.
Saves to: datasets/labels/clap_labels.csv
"""
from pathlib import Path
import os, sys, json, gc
import numpy as np
import soundfile as sf
import scipy.signal
import pandas as pd
import torch
from transformers import ClapModel, ClapProcessor
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent.parent  # ML_pipeline/
INPUT_DIR  = str(_ROOT / 'datasets' / 'preprocessed')
OUTPUT_F   = str(_ROOT / 'datasets' / 'labels' / 'clap_labels.csv')
MODEL_NAME = str(_ROOT / 'preprocessing' / 'clap_model')  # locally downloaded model
CLAP_SR    = 48000   # CLAP requires 48 kHz
SOURCE_SR  = 22050   # preprocessed files are at 22050 Hz
BATCH_SIZE = 8
SAVE_EVERY = 100     # save CSV every N batches (crash-safe)
CLASSES    = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']

# Carefully crafted prompts describing ACOUSTIC properties of each cry type
TEXT_PROMPTS = {
    'belly_pain': [
        "infant screaming loudly in severe pain, very high pitched intense and frantic crying with irregular bursts",
        "baby crying in agony from stomach cramp, high frequency sharp sudden crying sounds"
    ],
    'burping': [
        "baby burping after feeding, short low frequency guttural single vocalization sound",
        "infant producing a brief burp or belch, short guttural low pitched sound under one second"
    ],
    'discomfort': [
        "baby crying continuously from mild discomfort or irritation, steady medium pitched persistent crying",
        "infant fussing and whimpering with medium intensity, sustained continuous crying from discomfort"
    ],
    'hungry': [
        "hungry baby crying rhythmically and repeatedly, regular periodic pattern of crying with brief pauses",
        "infant crying insistently for food, repetitive rhythmic crying with consistent tempo"
    ],
    'tired': [
        "sleepy tired infant crying softly, low energy decreasing whiny nasal sounds fading away",
        "baby crying from exhaustion and tiredness, gentle decreasing soft whining sounds"
    ]
}


def load_model():
    print(f"Loading CLAP model from local path: {MODEL_NAME}")
    model = ClapModel.from_pretrained(MODEL_NAME, local_files_only=True)
    processor = ClapProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
    model.eval()
    print("Model loaded successfully.")
    return model, processor


def _extract_features(out):
    """Extract feature tensor from model output (handles tensor or pooled output)."""
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, 'pooler_output') and out.pooler_output is not None:
        return out.pooler_output
    return out.last_hidden_state[:, 0, :]  # CLS token fallback


def get_text_embeddings(model, processor):
    """Pre-compute text embeddings for all classes (amortized cost)."""
    all_class_text_embs = {}
    for cls, prompts in TEXT_PROMPTS.items():
        inputs = processor(text=prompts, return_tensors='pt', padding=True)
        with torch.no_grad():
            out = model.get_text_features(**inputs)
        text_embs = _extract_features(out)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
        all_class_text_embs[cls] = text_embs.mean(dim=0)  # average over prompts
    return all_class_text_embs


def classify_batch(audio_batch, model, processor, text_embs):
    """Classify a batch of audio arrays. Returns (labels, confidences)."""
    inputs = processor(
        audio=audio_batch,
        return_tensors='pt',
        padding=True,
        sampling_rate=CLAP_SR
    )
    with torch.no_grad():
        out = model.get_audio_features(**inputs)
    audio_feats = _extract_features(out)
    audio_feats = audio_feats / audio_feats.norm(dim=-1, keepdim=True)

    # Cosine similarity against each class
    class_names = list(text_embs.keys())
    text_matrix = torch.stack([text_embs[c] for c in class_names])  # (num_classes, dim)
    sims = (audio_feats @ text_matrix.T)  # (batch, num_classes)
    probs = torch.softmax(sims * 10, dim=-1)  # temperature scaling

    labels = [class_names[p.argmax().item()] for p in probs]
    confs  = [float(p.max().item()) for p in probs]
    all_probs = probs.numpy().tolist()
    return labels, confs, all_probs, class_names


def classify_all():
    device = 'cpu'
    model, processor = load_model()
    text_embs = get_text_embeddings(model, processor)
    print("Text embeddings computed for all 5 classes.")

    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.wav')])

    if os.path.exists(OUTPUT_F):
        done = set(pd.read_csv(OUTPUT_F)['filename'].tolist())
        files = [f for f in files if f not in done]
        print(f"Classifying {len(files)} new files (skipping {len(done)} already done).")
    else:
        print(f"Classifying {len(files)} files...")

    rows   = []
    failed = []

    for i in tqdm(range(0, len(files), BATCH_SIZE), unit='batch'):
        batch_files = files[i:i+BATCH_SIZE]
        audio_batch = []
        valid_files = []

        for fname in batch_files:
            try:
                y, sr = sf.read(os.path.join(INPUT_DIR, fname), always_2d=False)
                y = y.astype(np.float32)
                if y.ndim > 1:
                    y = y.mean(axis=1)
                # Resample from 22050 → 48000 with scipy (faster than librosa)
                if sr != CLAP_SR:
                    num_samples = int(len(y) * CLAP_SR / sr)
                    y = scipy.signal.resample(y, num_samples)
                audio_batch.append(y)
                valid_files.append(fname)
            except Exception as e:
                failed.append({'file': fname, 'error': str(e)})

        if not audio_batch:
            continue

        try:
            labels, confs, all_probs, class_names = classify_batch(
                audio_batch, model, processor, text_embs
            )
            for fname, label, conf, probs in zip(valid_files, labels, confs, all_probs):
                row = {'filename': fname, 'clap_label': label, 'clap_confidence': conf}
                for cls, p in zip(class_names, probs):
                    row[f'clap_prob_{cls}'] = p
                rows.append(row)
        except Exception as e:
            for fname in valid_files:
                failed.append({'file': fname, 'error': str(e)})

        # Periodic save for crash safety
        batch_idx = i // BATCH_SIZE
        if rows and batch_idx % SAVE_EVERY == 0:
            df_partial = pd.DataFrame(rows)
            if os.path.exists(OUTPUT_F):
                df_old = pd.read_csv(OUTPUT_F)
                df_save = pd.concat([df_old, df_partial], ignore_index=True)
            else:
                df_save = df_partial
            df_save.to_csv(OUTPUT_F, index=False)
            rows = []  # clear buffer after saving

    # Final save for remaining rows
    if rows:
        df_partial = pd.DataFrame(rows)
        if os.path.exists(OUTPUT_F):
            df_old = pd.read_csv(OUTPUT_F)
            df = pd.concat([df_old, df_partial], ignore_index=True)
        else:
            df = df_partial
        df.to_csv(OUTPUT_F, index=False)
    else:
        df = pd.read_csv(OUTPUT_F) if os.path.exists(OUTPUT_F) else pd.DataFrame()
    print(f"\nCLAP classification done. Total: {len(df)}, Failed: {len(failed)}")
    print("\nClass distribution from CLAP:")
    print(df['clap_label'].value_counts())
    if failed:
        import json
        with open(str(_ROOT / 'datasets' / 'labels' / 'clap_failures.json'), 'w') as fp:
            json.dump(failed, fp, indent=2)


if __name__ == '__main__':
    classify_all()

