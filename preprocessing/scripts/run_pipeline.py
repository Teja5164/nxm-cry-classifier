"""
Master Pipeline Runner
Runs all 6 scripts in sequence for the full cry classification pipeline.

Pipeline:
  1. Download cry samples from Google Drive
  2. Preprocess audio (resample, normalize, trim)
  3. Extract acoustic features
  4. CLAP zero-shot classification
  5. Heuristic rule-based classification
  6. Combine labels + organize MainDataset

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-download  (if already downloaded)
    python run_pipeline.py --start-from 3   (start from step N)
"""
import sys, os, subprocess, time, argparse
from pathlib import Path

PYTHON      = sys.executable  # always uses the active environment's Python
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT       = Path(__file__).resolve().parent.parent.parent  # ML_pipeline/

STEPS = [
    (1, '01_download_cry.py',       'Downloading cry samples from Google Drive'),
    (2, '02_preprocess.py',         'Preprocessing audio files'),
    (3, '03_extract_features.py',   'Extracting acoustic features'),
    (4, '04_clap_classify.py',      'CLAP zero-shot classification (downloads ~1.5GB model on first run)'),
    (5, '05_heuristic_classify.py', 'Acoustic heuristic classification'),
    (6, '06_combine_and_organize.py','Combining labels and organizing MainDataset'),
]


def run_step(step_num, script, description):
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {description}")
    print(f"{'='*60}")
    t0 = time.time()
    script_path = os.path.join(SCRIPTS_DIR, script)
    result = subprocess.run([PYTHON, script_path], capture_output=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n❌ Step {step_num} FAILED (exit code {result.returncode})")
        print(f"   Script: {script_path}")
        sys.exit(1)
    print(f"\n✅ Step {step_num} completed in {elapsed/60:.1f} minutes")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip step 1 (download) if files already present')
    parser.add_argument('--start-from', type=int, default=1,
                        help='Start from this step number (1-6)')
    args = parser.parse_args()

    start = args.start_from
    if args.skip_download and start == 1:
        start = 2

    print("\n🚀 Infant Cry Classification Pipeline")
    print(f"   Starting from step: {start}")
    print(f"   Python: {PYTHON}")

    t_total = time.time()
    for step_num, script, desc in STEPS:
        if step_num < start:
            print(f"  [SKIP] Step {step_num}: {desc}")
            continue
        run_step(step_num, script, desc)

    total_min = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"🎉 PIPELINE COMPLETE in {total_min:.1f} minutes")
    print(f"   MainDataset: {_ROOT / 'datasets'}")
    print(f"   Final labels: {_ROOT / 'datasets' / 'labels' / 'final_labels.csv'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
