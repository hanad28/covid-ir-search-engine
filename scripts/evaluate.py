"""
Evaluate all available TREC runfiles against official qrels.

Prints a summary table and saves results to CSV.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluate import evaluate_all_runs, save_results_csv


def main():
    print("Evaluating runs against TREC-COVID Round 1 qrels")
    print("=" * 50)

    all_results = evaluate_all_runs()

    if all_results:
        print()
        save_results_csv(all_results)
    else:
        print("\nNo runs found to evaluate. Run the retrieval scripts first.")


if __name__ == "__main__":
    main()
