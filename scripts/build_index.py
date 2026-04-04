"""
Build the Pyserini/Lucene index from CORD-19 documents.

Two steps:
  1. Convert metadata CSV to JSONL (with preprocessing)
  2. Run Pyserini indexer with positions, doc vectors, and raw text stored

Flags --storePositions --storeDocvectors are required for RM3 feedback.
Flag --storeRaw is required for accessing raw document text at rerank time.
"""

import os
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import INDEX_DIR
from index import build_jsonl, JSONL_DIR


def run_pyserini_index(input_dir: Path, index_dir: Path) -> None:
    """Invoke Pyserini's Lucene indexer via command line."""
    index_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(input_dir),
        "--index", str(index_dir),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "4",
        "--fields", "title", "abstract", "body",
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]

    print(f"\nRunning Pyserini indexer...")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {index_dir}")
    print(f"  Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, check=True)

    if result.returncode == 0:
        print(f"\nIndex built successfully at {index_dir}")
    else:
        print(f"\nIndexing failed with return code {result.returncode}")
        sys.exit(1)


def main():
    print("=" * 60)
    print("Step 1: Converting CORD-19 metadata to JSONL")
    print("=" * 60)

    if (JSONL_DIR / "docs.jsonl").exists():
        print(f"  [skip] JSONL already exists at {JSONL_DIR / 'docs.jsonl'}")
        print("  Delete it to regenerate.\n")
    else:
        build_jsonl()
        print()

    print("=" * 60)
    print("Step 2: Building Pyserini/Lucene index")
    print("=" * 60)

    # Check if index already exists
    if (INDEX_DIR / "segments.gen").exists() or any(INDEX_DIR.glob("segments_*")):
        print(f"  [skip] Index already exists at {INDEX_DIR}")
        print("  Delete the index directory to rebuild.")
        return

    run_pyserini_index(JSONL_DIR, INDEX_DIR)


if __name__ == "__main__":
    main()
