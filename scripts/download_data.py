"""
Download TREC-COVID Round 1 topics, qrels, and CORD-19 full dataset.

Downloads:
  - topics-rnd1.xml       to data/topics/
  - qrels-rnd1.txt        to data/qrels/
  - metadata.csv          to data/corpus/  (CORD-19 April 10, 2020 release)
  - Full-text tarballs     to data/corpus/  (4 subsets, extracted to JSON)
"""

import sys
import tarfile
import urllib.request
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import DATA_DIR

S3_BASE = "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-04-10"
CORPUS_DIR = DATA_DIR / "corpus"

DOWNLOADS = [
    {
        "url": "https://ir.nist.gov/covidSubmit/data/topics-rnd1.xml",
        "dest": DATA_DIR / "topics" / "topics-rnd1.xml",
        "description": "TREC-COVID Round 1 topics (30 topics)",
    },
    {
        "url": "https://ir.nist.gov/covidSubmit/data/qrels-rnd1.txt",
        "dest": DATA_DIR / "qrels" / "qrels-rnd1.txt",
        "description": "TREC-COVID Round 1 qrels",
    },
    {
        "url": f"{S3_BASE}/metadata.csv",
        "dest": CORPUS_DIR / "metadata.csv",
        "description": "CORD-19 metadata (April 10, 2020)",
    },
]

# Full-text subsets (each is a tarball containing JSON files)
FULLTEXT_SUBSETS = [
    "comm_use_subset",
    "custom_license",
    "noncomm_use_subset",
    "biorxiv_medrxiv",
]


def download_file(url: str, dest: Path, description: str) -> None:
    """Download a single file with progress feedback."""
    if dest.exists():
        print(f"  [skip] {description} -- already exists at {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [downloading] {description}")
    print(f"    {url}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(dest, "wb") as out:
            shutil.copyfileobj(response, out)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  [done] {size_mb:.1f} MB saved to {dest}")
    except Exception as e:
        if dest.exists():
            dest.unlink()
        print(f"  [error] Failed to download {description}: {e}")
        raise


def download_and_extract_fulltext() -> None:
    """Download and extract all full-text subset tarballs."""
    for subset in FULLTEXT_SUBSETS:
        extract_dir = CORPUS_DIR / subset
        tarball_path = CORPUS_DIR / f"{subset}.tar.gz"

        # Skip if already extracted
        if extract_dir.exists() and any(extract_dir.rglob("*.json")):
            json_count = len(list(extract_dir.rglob("*.json")))
            print(f"  [skip] {subset} -- already extracted ({json_count} JSON files)")
            continue

        # Download tarball
        url = f"{S3_BASE}/{subset}.tar.gz"
        download_file(url, tarball_path, f"CORD-19 full text ({subset})")

        # Extract
        print(f"  [extracting] {subset}...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(path=CORPUS_DIR, filter="data")

        json_count = len(list(extract_dir.rglob("*.json")))
        print(f"  [done] Extracted {json_count} JSON files to {extract_dir}")

        # Remove tarball to save space
        tarball_path.unlink()
        print(f"  [cleanup] Removed {tarball_path}")
        print()


def main():
    print("Downloading TREC-COVID Round 1 data...\n")

    for item in DOWNLOADS:
        download_file(item["url"], item["dest"], item["description"])
        print()

    print("Downloading CORD-19 full-text subsets...\n")
    download_and_extract_fulltext()

    print("All downloads complete.")
    print(f"  Topics:   {DOWNLOADS[0]['dest']}")
    print(f"  Qrels:    {DOWNLOADS[1]['dest']}")
    print(f"  Metadata: {DOWNLOADS[2]['dest']}")
    print(f"  Full text: {CORPUS_DIR}/<subset>/")


if __name__ == "__main__":
    main()
