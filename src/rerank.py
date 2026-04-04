"""
ColBERTv2 neural reranking (Run D).

Reranks the top-N candidates from Run C using ColBERTv2 late interaction.
Intended to run on Google Colab with GPU. This module provides the
reranking logic; the Colab notebook handles model loading and execution.
"""

import json
from pathlib import Path

from config import RERANK_DEPTH
from retrieve import save_trec_run


def get_rerank_candidates(
    run_path: Path,
    depth: int = RERANK_DEPTH,
) -> dict[str, list[str]]:
    """
    Extract top-N document IDs per topic from a TREC runfile.

    Returns {topic_id: [doc_id, ...]} truncated to depth.
    """
    candidates = {}
    with open(run_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            topic_id, _, doc_id, rank, _, _ = parts
            if int(rank) > depth:
                continue
            if topic_id not in candidates:
                candidates[topic_id] = []
            candidates[topic_id].append(doc_id)
    return candidates


def save_reranked_results(
    results: dict[str, list[tuple[str, float]]],
    output_path: Path,
    run_name: str,
) -> None:
    """Write reranked results to TREC runfile format."""
    save_trec_run(results, output_path, run_name)
