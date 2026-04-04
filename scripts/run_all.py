"""
Run the full ablation ladder (Runs A-C) and evaluate.

Run D (ColBERTv2) must be executed separately on Colab with GPU.
Run E (RRF) requires Run D output, so it is also excluded here.
After running Run D on Colab, run scripts/run_rrf.py then scripts/evaluate.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import RUNS_DIR, RUN_NAMES
from topics import get_queries
from retrieve import search_bm25, search_bm25f, search_bm25f_rm3, save_trec_run
from evaluate import evaluate_all_runs, save_results_csv


def main():
    print("COVID-19 IR Search Engine - Full Pipeline")
    print("=" * 50)

    queries = get_queries(preprocess_text=False)
    print(f"Loaded {len(queries)} queries\n")

    # Run A: BM25 baseline
    print("Run A: BM25 baseline")
    print("-" * 30)
    results_a = search_bm25(queries)
    save_trec_run(results_a, RUNS_DIR / f"{RUN_NAMES['A']}.txt", RUN_NAMES["A"])
    print()

    # Run B: BM25F
    print("Run B: BM25F fielded retrieval")
    print("-" * 30)
    results_b = search_bm25f(queries)
    save_trec_run(results_b, RUNS_DIR / f"{RUN_NAMES['B']}.txt", RUN_NAMES["B"])
    print()

    # Run C: BM25F + RM3
    print("Run C: BM25F + RM3")
    print("-" * 30)
    results_c = search_bm25f_rm3(queries)
    save_trec_run(results_c, RUNS_DIR / f"{RUN_NAMES['C']}.txt", RUN_NAMES["C"])
    print()

    # Evaluate available runs
    print("Evaluation")
    print("-" * 30)
    all_results = evaluate_all_runs()
    if all_results:
        print()
        save_results_csv(all_results)

    print("\nRuns A-C complete.")
    print("Next steps:")
    print("  1. Run D (ColBERTv2) on Google Colab with GPU")
    print("  2. Copy the Run D output to runs/")
    print("  3. Run scripts/run_rrf.py for Run E")
    print("  4. Run scripts/evaluate.py for final evaluation")


if __name__ == "__main__":
    main()
