"""
Run C: BM25F + RM3 pseudo-relevance feedback.

Retrieves documents using BM25F field boosting, then expands
queries using RM3 (top-k feedback docs) and re-retrieves.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import RUNS_DIR, RUN_NAMES
from topics import get_queries
from retrieve import search_bm25f_rm3, save_trec_run


def main():
    run_name = RUN_NAMES["C"]
    print(f"Run C: {run_name}")
    print("=" * 40)

    queries = get_queries(preprocess_text=False)
    print(f"Loaded {len(queries)} queries")

    results = search_bm25f_rm3(queries)
    print(f"Retrieved results for {len(results)} topics")

    output_path = RUNS_DIR / f"{run_name}.txt"
    save_trec_run(results, output_path, run_name)


if __name__ == "__main__":
    main()
