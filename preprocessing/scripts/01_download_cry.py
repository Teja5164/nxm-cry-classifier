"""
Script 1: Download cry samples from Google Drive (icsd_processed only)
Saves files to: datasets/raw_cry/
"""
from pathlib import Path
import sys, os, json

import gdown
from tqdm import tqdm

DRIVE_URL   = 'https://drive.google.com/drive/folders/1DWTHdv_KQmDXmZ5LtztwqnBvIEjF7v2A'
_ROOT = Path(__file__).resolve().parent.parent.parent  # ML_pipeline/
OUTPUT_DIR  = str(_ROOT / 'datasets' / 'raw_cry')
MANIFEST_F  = str(_ROOT / 'datasets' / 'labels' / 'manifest_cry.json')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MANIFEST_F), exist_ok=True)


def get_cry_manifest():
    print("Fetching folder manifest from Google Drive (this may take 2-3 minutes)...")
    files = gdown.download_folder(url=DRIVE_URL, skip_download=True, quiet=True)
    cry_files = [
        {'id': f.id, 'path': f.path}
        for f in files
        if ('icsd_processed' in f.path and
            '\\cry\\' in f.path and
            f.path.endswith('.wav'))
    ]
    with open(MANIFEST_F, 'w') as fp:
        json.dump(cry_files, fp, indent=2)
    print(f"Found {len(cry_files)} cry files from icsd_processed.")
    return cry_files


def _download_one(item):
    fname = os.path.basename(item['path'])
    dest  = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return None  # already done
    url = f"https://drive.google.com/uc?id={item['id']}"
    try:
        gdown.download(url, dest, quiet=True)
        return None
    except Exception as e:
        return {'file': fname, 'error': str(e)}


def download_files(cry_files):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    already = {f for f in os.listdir(OUTPUT_DIR)
               if os.path.getsize(os.path.join(OUTPUT_DIR, f)) > 0}
    to_download = [f for f in cry_files if os.path.basename(f['path']) not in already]
    print(f"Downloading {len(to_download)} files ({len(already)} already present) with 20 threads...")

    failed = []
    WORKERS = 20
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_download_one, item): item for item in to_download}
        for fut in tqdm(as_completed(futures), total=len(to_download), unit='file'):
            result = fut.result()
            if result:
                failed.append(result)

    print(f"\nDownload complete. Success: {len(to_download)-len(failed)}, Failed: {len(failed)}")
    if failed:
        with open(str(_ROOT / 'datasets' / 'labels' / 'download_failures.json'), 'w') as fp:
            json.dump(failed, fp, indent=2)
    total = len(os.listdir(OUTPUT_DIR))
    print(f"Total files in raw_cry: {total}")
    return total


if __name__ == '__main__':
    if os.path.exists(MANIFEST_F):
        print(f"Using cached manifest: {MANIFEST_F}")
        with open(MANIFEST_F) as fp:
            cry_files = json.load(fp)
        print(f"Manifest has {len(cry_files)} entries.")
    else:
        cry_files = get_cry_manifest()

    download_files(cry_files)
    print("Done! Raw cry files saved to:", OUTPUT_DIR)

