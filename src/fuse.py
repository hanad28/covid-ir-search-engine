"""
Reciprocal Rank Fusion (RRF) for combining multiple ranked lists.

Used in Run E to fuse Run C (BM25F+RM3) and Run D (ColBERTv2 reranked).
RRF score = sum over runs of 1 / (k + rank), where k is a damping constant.
"""

from pathlib import Path

from config import RRF_K, RETRIEVAL_DEPTH


def load_trec_run(path: Path) -> dict[str, list[tuple[str, float]]]:
    """
    Load a TREC runfile into {topic_id: [(doc_id, score), ...]}.

    Results are ordered by the rank in the file.
    """
    results = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            topic_id, _, doc_id, rank, score, _ = parts
            if topic_id not in results:
                results[topic_id] = []
            results[topic_id].append((doc_id, float(score)))
    return results


def reciprocal_rank_fusion(
    runs: list[dict[str, list[tuple[str, float]]]],
    k: int = RRF_K,
    depth: int = RETRIEVAL_DEPTH,
) -> dict[str, list[tuple[str, float]]]:
    """
    Fuse multiple ranked lists using RRF.

    For each document, RRF score = sum of 1/(k + rank_i) across all
    input runs where the document appears. Documents not in a run
    receive no contribution from that run.
    """
    # Collect all topic IDs across runs
    all_topics = set()
    for run in runs:
        all_topics.update(run.keys())

    fused = {}
    for topic_id in sorted(all_topics, key=int):
        doc_scores = {}

        for run in runs:
            if topic_id not in run:
                continue
            for rank, (doc_id, _) in enumerate(run[topic_id], start=1):
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                doc_scores[doc_id] += 1.0 / (k + rank)

        # Sort by RRF score descending, truncate to depth
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:depth]
        fused[topic_id] = ranked

    return fused
