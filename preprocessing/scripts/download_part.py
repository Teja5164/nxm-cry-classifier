"""
Part-based downloader: downloads one part at a time.
Usage: python download_part.py <part_number>   (1-6)
"""
from pathlib import Path
import sys, os, json, time, random, threading
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = Path(__file__).resolve().parent.parent.parent  # ML_pipeline/
OUTPUT_DIR   = str(_ROOT / 'datasets' / 'raw_cry')
LABELS_DIR   = str(_ROOT / 'datasets' / 'labels')
os.makedirs(OUTPUT_DIR, exist_ok=True)

WORKERS     = 8
RATE_LIMIT  = threading.Semaphore(8)
MAX_RETRIES = 4

def _download_one(item):
    fname = os.path.basename(item['path'])
    dest  = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return None

    url = f"https://drive.google.com/uc?id={item['id']}&export=download&confirm=t"
    for attempt in range(MAX_RETRIES):
        with RATE_LIMIT:
            time.sleep(random.uniform(0.05, 0.3))
            try:
                r = requests.get(url, allow_redirects=True, timeout=45, stream=True)
                r.raise_for_status()
                with open(dest, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
                if os.path.exists(dest) and os.path.getsize(dest) > 0:
                    return None
            except Exception:
                pass
        if attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt + random.random())

    return {'file': fname, 'error': f'Failed after {MAX_RETRIES} attempts'}


def download_part(part_num):
    part_file = os.path.join(LABELS_DIR, f'manifest_part{part_num}.json')
    if not os.path.exists(part_file):
        print(f"ERROR: {part_file} not found.")
        sys.exit(1)

    with open(part_file) as fp:
        items = json.load(fp)

    already = {f for f in os.listdir(OUTPUT_DIR)
               if os.path.getsize(os.path.join(OUTPUT_DIR, f)) > 0}
    to_download = [f for f in items if os.path.basename(f['path']) not in already]

    print(f"\n=== PART {part_num}/6 ===")
    print(f"Part total  : {len(items)} files")
    print(f"Already done: {len(already)} files in raw_cry (across all parts)")
    print(f"To download : {len(to_download)} files with {WORKERS} workers (requests, rate-limited)\n")

    if not to_download:
        print("Nothing to download for this part — all files already present.")
        return

    failed = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_download_one, item): item for item in to_download}
        for fut in tqdm(as_completed(futures), total=len(to_download), unit='file',
                        desc=f'Part {part_num}'):
            result = fut.result()
            if result:
                failed.append(result)

    success = len(to_download) - len(failed)
    print(f"\nPart {part_num} done: {success} downloaded, {len(failed)} failed.")
    if failed:
        fail_file = os.path.join(LABELS_DIR, f'failures_part{part_num}.json')
        with open(fail_file, 'w') as fp:
            json.dump(failed, fp, indent=2)
        print(f"Failures logged to: {fail_file}")

    total = len([f for f in os.listdir(OUTPUT_DIR)
                 if os.path.getsize(os.path.join(OUTPUT_DIR, f)) > 0])
    print(f"Total files now in raw_cry: {total} / 6586")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python download_part.py <part_number>  (1-6)")
        sys.exit(1)
    part_num = int(sys.argv[1])
    if not 1 <= part_num <= 6:
        print("Part number must be 1-6")
        sys.exit(1)
    download_part(part_num)
