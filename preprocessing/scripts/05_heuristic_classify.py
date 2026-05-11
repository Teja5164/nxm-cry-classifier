"""
Script 5: Acoustic heuristic rule-based classification.
Uses features from features.csv to classify cry type based on
scientifically-documented acoustic signatures of infant cry types.
Saves to: datasets/labels/heuristic_labels.csv
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent  # ML_pipeline/
FEATURES_F  = str(_ROOT / 'datasets' / 'features' / 'features.csv')
OUTPUT_F    = str(_ROOT / 'datasets' / 'labels' / 'heuristic_labels.csv')


def heuristic_classify(row):
    """
    Rule-based classification using acoustic features.

    Acoustic profiles (based on infant cry research):
      burping   : very short (<1.5s), low pitch, low ZCR, single event
      belly_pain: high pitch (>500 Hz), high pitch variability (std>80), explosive
      hungry    : medium pitch, high periodicity (>0.55), regular rhythm
      tired     : lower pitch, negative energy slope, lower energy
      discomfort: default / sustained medium pitch, medium energy
    """
    dur         = row['duration']
    f0_mean     = row['f0_mean']
    f0_std      = row['f0_std']
    rms_mean    = row['rms_mean']
    rms_slope   = row['rms_slope']
    zcr_mean    = row['zcr_mean']
    periodicity = row['periodicity']
    sc_mean     = row['spectral_centroid_mean']

    scores = {
        'belly_pain':  0.0,
        'burping':     0.0,
        'discomfort':  0.0,
        'hungry':      0.0,
        'tired':       0.0,
    }

    # --- BURPING ---
    # Very short, low pitch, low ZCR, low spectral centroid
    if dur < 1.5:
        scores['burping'] += 3.0
    elif dur < 2.5:
        scores['burping'] += 1.0
    if f0_mean < 350 and f0_mean > 0:
        scores['burping'] += 2.0
    if zcr_mean < 0.04:
        scores['burping'] += 1.5
    if sc_mean < 1500:
        scores['burping'] += 1.0

    # --- BELLY PAIN ---
    # High pitch, high pitch variability, high energy, irregular
    if f0_mean > 550:
        scores['belly_pain'] += 3.0
    elif f0_mean > 480:
        scores['belly_pain'] += 1.5
    if f0_std > 100:
        scores['belly_pain'] += 2.5
    elif f0_std > 70:
        scores['belly_pain'] += 1.0
    if rms_mean > 0.08:
        scores['belly_pain'] += 1.0
    if periodicity < 0.35:
        scores['belly_pain'] += 0.5  # irregular pattern

    # --- HUNGRY ---
    # Medium pitch, highly rhythmic, regular pattern, persistent
    if periodicity > 0.60:
        scores['hungry'] += 3.5
    elif periodicity > 0.45:
        scores['hungry'] += 1.5
    if 350 <= f0_mean <= 520:
        scores['hungry'] += 1.5
    if dur > 1.5:
        scores['hungry'] += 0.5
    if rms_mean > 0.04:
        scores['hungry'] += 0.5

    # --- TIRED ---
    # Decreasing energy, lower-medium pitch, low-medium energy
    if rms_slope < -0.0002:
        scores['tired'] += 3.0
    elif rms_slope < -0.0001:
        scores['tired'] += 1.5
    if f0_mean < 430 and f0_mean > 0:
        scores['tired'] += 1.5
    if rms_mean < 0.05:
        scores['tired'] += 1.0
    if zcr_mean < 0.07:
        scores['tired'] += 0.5

    # --- DISCOMFORT ---
    # Sustained, medium pitch, medium energy, medium ZCR
    if 400 <= f0_mean <= 560:
        scores['discomfort'] += 1.5
    if 0.04 <= rms_mean <= 0.09:
        scores['discomfort'] += 1.0
    if 0.05 <= zcr_mean <= 0.12:
        scores['discomfort'] += 1.0
    if dur > 1.8:
        scores['discomfort'] += 0.5
    if 0.30 <= periodicity <= 0.60:
        scores['discomfort'] += 0.5

    predicted = max(scores, key=scores.get)
    top_score  = scores[predicted]
    total      = sum(scores.values()) + 1e-9
    confidence = top_score / total

    return predicted, confidence, scores


def classify_all():
    df = pd.read_csv(FEATURES_F)
    print(f"Classifying {len(df)} files with acoustic heuristics...")

    labels, confs, all_scores = [], [], []

    for _, row in df.iterrows():
        label, conf, scores = heuristic_classify(row)
        labels.append(label)
        confs.append(conf)
        all_scores.append(scores)

    df_out = pd.DataFrame({
        'filename':             df['filename'],
        'heuristic_label':      labels,
        'heuristic_confidence': confs,
    })
    for cls in ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']:
        df_out[f'heur_score_{cls}'] = [s[cls] for s in all_scores]

    df_out.to_csv(OUTPUT_F, index=False)

    print(f"\nHeuristic classification done. Results saved to: {OUTPUT_F}")
    print("\nClass distribution:")
    print(df_out['heuristic_label'].value_counts())
    print(f"\nAvg heuristic confidence: {df_out['heuristic_confidence'].mean():.3f}")
    return df_out


if __name__ == '__main__':
    classify_all()

