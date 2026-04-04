"""
Build a 200-document sample dataset for reproducibility verification.

Copies the first 200 documents from data/corpus_jsonl/docs.jsonl into
sample_data/docs.jsonl, and copies the topics and qrels files unchanged.

Markers can verify the full pipeline using this lightweight subset
without downloading the full 945 MB corpus.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import PROJECT_ROOT, TOPICS_FILE, QRELS_FILE

FULL_JSONL = PROJECT_ROOT / "data" / "corpus_jsonl" / "docs.jsonl"
SAMPLE_DIR = PROJECT_ROOT / "sample_data"
SAMPLE_JSONL = SAMPLE_DIR / "docs.jsonl"
SAMPLE_SIZE = 200


def build_sample_jsonl(source: Path, dest: Path, n: int) -> int:
    """Copy the first n lines from source JSONL to dest JSONL."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(source, encoding="utf-8") as src, open(dest, "w", encoding="utf-8") as out:
        for line in src:
            if count >= n:
                break
            out.write(line)
            count += 1
    return count


def copy_benchmark_files(sample_dir: Path) -> None:
    """Copy topics and qrels into the sample directory."""
    topics_dest = sample_dir / "topics" / TOPICS_FILE.name
    qrels_dest = sample_dir / "qrels" / QRELS_FILE.name

    topics_dest.parent.mkdir(parents=True, exist_ok=True)
    qrels_dest.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(TOPICS_FILE, topics_dest)
    shutil.copy2(QRELS_FILE, qrels_dest)

    print(f"Copied topics  -> {topics_dest}")
    print(f"Copied qrels   -> {qrels_dest}")


def main():
    if not FULL_JSONL.exists():
        print(f"Error: full corpus JSONL not found at {FULL_JSONL}")
        print("Run scripts/build_index.py first to generate docs.jsonl.")
        sys.exit(1)

    if SAMPLE_JSONL.exists():
        print(f"Sample JSONL already exists at {SAMPLE_JSONL}. Skipping.")
    else:
        print(f"Building {SAMPLE_SIZE}-document sample from {FULL_JSONL}...")
        written = build_sample_jsonl(FULL_JSONL, SAMPLE_JSONL, SAMPLE_SIZE)
        print(f"Wrote {written} documents to {SAMPLE_JSONL}")

    copy_benchmark_files(SAMPLE_DIR)
    print(f"\nSample data ready in {SAMPLE_DIR}/")
    print("To run with sample data, point INDEX_DIR and CORPUS paths to sample_data/.")


if __name__ == "__main__":
    main()
