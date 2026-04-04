"""
Evaluate TREC runfiles against official qrels using ir_measures.

Computes nDCG@10, MAP, Recall@100, and Recall@1000 for each run.
"""

import csv
from pathlib import Path

import ir_measures
from ir_measures import nDCG, AP, R

from config import QRELS_FILE, RUNS_DIR, RESULTS_DIR, RUN_NAMES, EVAL_METRICS


# Map config metric names to ir_measures objects
METRIC_MAP = {
    "nDCG@10": nDCG @ 10,
    "AP": AP,
    "R@100": R @ 100,
    "R@1000": R @ 1000,
}


def load_qrels(path: Path = QRELS_FILE) -> list:
    """Load qrels file into ir_measures format (materialised as list)."""
    return list(ir_measures.read_trec_qrels(str(path)))


def load_run(path: Path) -> list:
    """Load a TREC runfile into ir_measures format (materialised as list)."""
    return list(ir_measures.read_trec_run(str(path)))


def evaluate_run(
    run_path: Path,
    qrels: list = None,
    metrics: list[str] = EVAL_METRICS,
) -> dict[str, float]:
    """
    Evaluate a single run against qrels.

    Returns {metric_name: score} averaged across all topics.
    """
    if qrels is None:
        qrels = load_qrels()

    run = load_run(run_path)
    measure_objects = [METRIC_MAP[m] for m in metrics]

    results = ir_measures.calc_aggregate(measure_objects, qrels, run)

    return {str(k): round(v, 4) for k, v in results.items()}


def evaluate_all_runs(
    runs_dir: Path = RUNS_DIR,
    run_names: dict = RUN_NAMES,
) -> dict[str, dict[str, float]]:
    """Evaluate all runs and return nested dict of results."""
    qrels = load_qrels()
    all_results = {}

    for run_key, run_name in sorted(run_names.items()):
        run_path = runs_dir / f"{run_name}.txt"
        if not run_path.exists():
            print(f"  [skip] Run {run_key} ({run_name}) not found at {run_path}")
            continue

        print(f"  Evaluating Run {run_key}: {run_name}")
        scores = evaluate_run(run_path, qrels)
        all_results[run_name] = scores

        for metric, value in scores.items():
            print(f"    {metric}: {value:.4f}")

    return all_results


def save_results_csv(
    all_results: dict[str, dict[str, float]],
    output_path: Path = RESULTS_DIR / "evaluation.csv",
) -> None:
    """Save evaluation results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not all_results:
        print("No results to save.")
        return

    metrics = list(next(iter(all_results.values())).keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run"] + metrics)
        for run_name, scores in all_results.items():
            writer.writerow([run_name] + [scores[m] for m in metrics])

    print(f"Saved evaluation results to {output_path}")
