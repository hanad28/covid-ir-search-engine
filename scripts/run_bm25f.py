"""
Run B: BM25F field-aware retrieval.

Retrieves documents using BM25 with Lucene field boosting
(title > abstract > body) to simulate BM25F scoring.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import RUNS_DIR, RUN_NAMES
from topics import get_queries
from retrieve import search_bm25f, save_trec_run


def main():
    run_name = RUN_NAMES["B"]
    print(f"Run B: {run_name}")
    print("=" * 40)

    queries = get_queries(preprocess_text=False)
    print(f"Loaded {len(queries)} queries")

    results = search_bm25f(queries)
    print(f"Retrieved results for {len(results)} topics")

    output_path = RUNS_DIR / f"{run_name}.txt"
    save_trec_run(results, output_path, run_name)


if __name__ == "__main__":
    main()
