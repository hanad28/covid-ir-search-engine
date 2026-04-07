"""
Convert CORD-19 metadata and full-text JSON into Pyserini-compatible documents.

Each document has separate fields (title, abstract, body) for BM25F
fielded retrieval, plus a combined 'contents' field for standard BM25.

Text is stored raw. Lucene's DefaultEnglishAnalyzer (Porter stemmer +
stopword removal) handles normalisation at index time.
"""

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import CORPUS_METADATA, DATA_DIR, CORPUS_DIR, PROJECT_ROOT

JSONL_DIR = DATA_DIR / "corpus_jsonl"

FULLTEXT_SUBSETS = [
    "comm_use_subset",
    "custom_license",
    "noncomm_use_subset",
    "biorxiv_medrxiv",
]


def _build_sha_to_path_map(corpus_dir: Path = CORPUS_DIR) -> dict[str, Path]:
    """
    Build a mapping from paper SHA to its JSON file path.

    Only stores file paths (not content) to avoid memory issues.
    """
    sha_to_path = {}

    for subset in FULLTEXT_SUBSETS:
        subset_dir = corpus_dir / subset
        if not subset_dir.exists():
            print(f"  [warn] Subset not found: {subset_dir}")
            continue

        json_files = list(subset_dir.rglob("*.json"))
        print(f"  Indexed {len(json_files):,} files from {subset}")

        for json_path in json_files:
            # Filename without extension is the SHA
            sha = json_path.stem
            sha_to_path[sha] = json_path

    print(f"  Total: {len(sha_to_path):,} full-text files mapped")
    return sha_to_path


def _clean_text(text: str) -> str:
    """Remove control characters that can corrupt JSON output."""
    return "".join(ch for ch in text if ch == "\n" or ch == "\t" or (ord(ch) >= 32))


def _extract_body_text(json_path: Path) -> str:
    """Read a single JSON file and extract concatenated body text."""
    try:
        with open(json_path, encoding="utf-8") as f:
            doc = json.load(f)
        body_parts = doc.get("body_text", [])
        if body_parts:
            text = " ".join(p.get("text", "") for p in body_parts).strip()
            return _clean_text(text)
    except (json.JSONDecodeError, KeyError, OSError):
        pass
    return ""


def build_jsonl(
    metadata_path: Path = CORPUS_METADATA,
    output_dir: Path = JSONL_DIR,
) -> int:
    """
    Read metadata CSV and full-text JSON, write Pyserini documents.

    Each JSON doc has:
      - id:       cord_uid (unique document identifier)
      - contents: title + abstract + body (for BM25 baseline)
      - title:    title field (for BM25F)
      - abstract: abstract field (for BM25F)
      - body:     body text field (for BM25F)

    Text is stored raw; Lucene's analyzer (Porter stemmer + stopword
    removal) handles tokenisation and normalisation at index time.

    Returns the number of documents written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Mapping full-text JSON files...")
    sha_to_path = _build_sha_to_path_map()

    df = pd.read_csv(
        metadata_path,
        usecols=["cord_uid", "sha", "title", "abstract"],
        dtype=str,
    )
    df = df.fillna("")
    df = df.drop_duplicates(subset="cord_uid", keep="first")
    df = df[~((df["title"] == "") & (df["abstract"] == ""))]

    doc_count = 0
    body_found = 0
    output_path = output_dir / "docs.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building JSONL"):
            title = _clean_text(row["title"].strip())
            abstract = _clean_text(row["abstract"].strip())

            # Look up body text by SHA, read file on demand
            sha = row["sha"].strip()
            body = ""
            if sha and sha in sha_to_path:
                body = _extract_body_text(sha_to_path[sha])
                if body:
                    body_found += 1

            contents = f"{title} {abstract} {body}".strip()

            if not contents:
                continue

            doc = {
                "id": row["cord_uid"],
                "contents": contents,
                "title": title,
                "abstract": abstract,
                "body": body,
            }

            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            doc_count += 1

    print(f"Wrote {doc_count:,} documents to {output_path}")
    print(f"  {body_found:,} documents have body text ({body_found*100/doc_count:.1f}%)")
    return doc_count
