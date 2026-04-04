"""
Run D: ColBERTv2 reranking of Run C candidates.

This script is intended to run on Google Colab with GPU access.
It reranks the top-100 candidates from Run C (BM25F+RM3) using
ColBERTv2 late interaction scoring on title + abstract text.

Prerequisites (install on Colab):
  pip install colbert-ai torch
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import RUNS_DIR, RUN_NAMES, COLBERT_MODEL, RERANK_DEPTH, INDEX_DIR
from rerank import get_rerank_candidates, save_reranked_results
from topics import get_queries

# ColBERTv2 imports (available on Colab)
try:
    import torch
    from colbert import Searcher
    from colbert.infra import ColBERTConfig, Run, RunConfig
except ImportError:
    print("ColBERTv2 dependencies not found.")
    print("This script is designed to run on Google Colab with GPU.")
    print("Install: pip install colbert-ai torch")
    sys.exit(1)


def load_document_text(doc_id: str, searcher_lucene) -> str:
    """Load title + abstract for a document from the Lucene index."""
    raw = searcher_lucene.doc(doc_id).raw()
    doc = json.loads(raw)
    title = doc.get("title", "")
    abstract = doc.get("abstract", "")
    return f"{title} {abstract}".strip()


def rerank_with_colbert(
    queries: dict[str, str],
    candidates: dict[str, list[str]],
    model_name: str = COLBERT_MODEL,
) -> dict[str, list[tuple[str, float]]]:
    """
    Rerank candidates using ColBERTv2 MaxSim scoring.

    For each query-document pair, computes the ColBERT late interaction
    score and re-sorts documents by this score.
    """
    from colbert.modeling.checkpoint import Checkpoint

    checkpoint = Checkpoint(model_name, colbert_config=ColBERTConfig())

    # Load Lucene searcher for document text
    from pyserini.search.lucene import LuceneSearcher
    lucene_searcher = LuceneSearcher(str(INDEX_DIR))

    results = {}
    for topic_id, query_text in queries.items():
        if topic_id not in candidates:
            continue

        doc_ids = candidates[topic_id]
        doc_texts = [load_document_text(did, lucene_searcher) for did in doc_ids]

        # Score all candidate documents against the query
        scores = checkpoint.queryFromText(
            [query_text],
            bsize=32,
            to_cpu=True,
        )

        doc_encodings = checkpoint.docFromText(
            doc_texts,
            bsize=32,
            to_cpu=True,
        )

        # Compute MaxSim scores
        query_emb = scores[0]  # (query_len, dim)
        doc_scores = []
        for i, doc_emb in enumerate(doc_encodings):
            # MaxSim: for each query token, find max similarity across doc tokens
            sim = torch.matmul(query_emb, doc_emb.T)
            max_sim = sim.max(dim=1).values.sum().item()
            doc_scores.append((doc_ids[i], max_sim))

        # Sort by score descending
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        results[topic_id] = doc_scores

    lucene_searcher.close()
    return results


def main():
    run_name = RUN_NAMES["D"]
    print(f"Run D: {run_name}")
    print("=" * 40)

    run_c_path = RUNS_DIR / f"{RUN_NAMES['C']}.txt"
    if not run_c_path.exists():
        print(f"Error: Run C not found at {run_c_path}")
        print("Run scripts/run_rm3.py first.")
        sys.exit(1)

    queries = get_queries(preprocess_text=False)
    print(f"Loaded {len(queries)} queries (raw text for neural model)")

    candidates = get_rerank_candidates(run_c_path, depth=RERANK_DEPTH)
    print(f"Loaded top-{RERANK_DEPTH} candidates from Run C")

    print(f"Reranking with {COLBERT_MODEL}...")
    results = rerank_with_colbert(queries, candidates)
    print(f"Reranked {len(results)} topics")

    output_path = RUNS_DIR / f"{run_name}.txt"
    save_reranked_results(results, output_path, run_name)


if __name__ == "__main__":
    main()
