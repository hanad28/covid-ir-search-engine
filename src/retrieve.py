"""
Retrieval functions for BM25 (Run A), BM25F (Run B), and BM25F+RM3 (Run C).

All functions return results as a dict: {topic_id: [(doc_id, score), ...]}.
"""

import re
from pathlib import Path

from pyserini.search.lucene import LuceneSearcher

from config import (
    INDEX_DIR,
    RETRIEVAL_DEPTH,
    BM25_K1,
    BM25_B,
    BM25F_WEIGHTS,
    BM25F_K1,
    BM25F_B,
    RM3_FB_DOCS,
    RM3_FB_TERMS,
    RM3_ORIGINAL_WEIGHT,
)


def _init_searcher(index_dir: Path = INDEX_DIR) -> LuceneSearcher:
    """Load the Lucene index."""
    searcher = LuceneSearcher(str(index_dir))
    return searcher


def search_bm25(
    queries: dict[str, str],
    index_dir: Path = INDEX_DIR,
    k1: float = BM25_K1,
    b: float = BM25_B,
    depth: int = RETRIEVAL_DEPTH,
) -> dict[str, list[tuple[str, float]]]:
    """
    Run A: Standard BM25 over the combined contents field.
    """
    searcher = _init_searcher(index_dir)
    searcher.set_bm25(k1, b)

    results = {}
    for topic_id, query_text in queries.items():
        hits = searcher.search(query_text, k=depth)
        results[topic_id] = [(hit.docid, hit.score) for hit in hits]

    searcher.close()
    return results


def _build_boosted_query(query_text: str, field_weights: dict[str, float]) -> str:
    """
    Build a Lucene boosted query string for BM25F simulation.

    Searches each field with its configured weight boost.
    Example: "title:(covid origin)^2.0 abstract:(covid origin)^1.5"
    """
    parts = []
    for field, weight in field_weights.items():
        parts.append(f"{field}:({query_text})^{weight}")
    return " ".join(parts)


def search_bm25f(
    queries: dict[str, str],
    index_dir: Path = INDEX_DIR,
    field_weights: dict[str, float] = BM25F_WEIGHTS,
    k1: float = BM25F_K1,
    b: float = BM25F_B,
    depth: int = RETRIEVAL_DEPTH,
) -> dict[str, list[tuple[str, float]]]:
    """
    Run B: BM25F using Lucene field boosting.

    Simulates BM25F by searching title, abstract, and body fields
    with different boost weights. Higher weight = more influence.
    """
    searcher = _init_searcher(index_dir)
    searcher.set_bm25(k1, b)

    results = {}
    for topic_id, query_text in queries.items():
        boosted_query = _build_boosted_query(query_text, field_weights)
        hits = searcher.search(boosted_query, k=depth)
        results[topic_id] = [(hit.docid, hit.score) for hit in hits]

    searcher.close()
    return results


def search_bm25f_rm3(
    queries: dict[str, str],
    index_dir: Path = INDEX_DIR,
    field_weights: dict[str, float] = BM25F_WEIGHTS,
    k1: float = BM25F_K1,
    b: float = BM25F_B,
    fb_docs: int = RM3_FB_DOCS,
    fb_terms: int = RM3_FB_TERMS,
    original_weight: float = RM3_ORIGINAL_WEIGHT,
    depth: int = RETRIEVAL_DEPTH,
) -> dict[str, list[tuple[str, float]]]:
    """
    Run C: BM25F + RM3 pseudo-relevance feedback.

    First retrieves with BM25F, then expands the query using RM3
    (top feedback docs), and re-retrieves with the expanded query.
    """
    searcher = _init_searcher(index_dir)
    searcher.set_bm25(k1, b)
    searcher.set_rm3(fb_docs, fb_terms, original_weight)

    results = {}
    for topic_id, query_text in queries.items():
        boosted_query = _build_boosted_query(query_text, field_weights)
        hits = searcher.search(boosted_query, k=depth)
        results[topic_id] = [(hit.docid, hit.score) for hit in hits]

    searcher.close()
    return results


def get_rm3_expansion_terms(
    query_text: str,
    index_dir: Path = INDEX_DIR,
    field_weights: dict[str, float] = BM25F_WEIGHTS,
    k1: float = BM25F_K1,
    b: float = BM25F_B,
    fb_docs: int = RM3_FB_DOCS,
    fb_terms: int = RM3_FB_TERMS,
    original_weight: float = RM3_ORIGINAL_WEIGHT,
) -> list[tuple[str, float]]:
    """
    Return the terms RM3 added for a single query, with their weights.

    Performs a BM25F+RM3 search and parses the expanded query string
    to extract terms that were injected beyond the original query tokens.
    Returns a list of (term, weight) tuples sorted by weight descending.
    """
    searcher = _init_searcher(index_dir)
    searcher.set_bm25(k1, b)
    searcher.set_rm3(fb_docs, fb_terms, original_weight)

    boosted_query = _build_boosted_query(query_text, field_weights)
    searcher.search(boosted_query, k=fb_docs)

    # Pyserini exposes the RM3-expanded query string after a search.
    # Not available in all Pyserini versions — return empty list if unsupported.
    try:
        expanded_query = searcher.get_feedback_query()
    except AttributeError:
        searcher.close()
        return []
    searcher.close()

    if not expanded_query:
        return []

    # Parse the Lucene query string into (term, weight) pairs.
    # RM3 produces terms like: term^weight or field:term^weight
    original_tokens = set(query_text.lower().split())
    expansion_terms = []

    for match in re.finditer(r"(?:\w+:)?(\w+)\^([\d.]+)", expanded_query):
        term = match.group(1).lower()
        weight = float(match.group(2))
        if term not in original_tokens and len(term) > 1:
            expansion_terms.append((term, weight))

    # Sort by weight descending, deduplicate
    seen = set()
    unique_terms = []
    for term, weight in sorted(expansion_terms, key=lambda x: x[1], reverse=True):
        if term not in seen:
            seen.add(term)
            unique_terms.append((term, weight))

    return unique_terms


def save_trec_run(
    results: dict[str, list[tuple[str, float]]],
    output_path: Path,
    run_name: str,
) -> None:
    """
    Write results to a TREC-format runfile.

    Format: topic_id Q0 doc_id rank score run_name
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for topic_id, doc_scores in sorted(results.items(), key=lambda x: int(x[0])):
            for rank, (doc_id, score) in enumerate(doc_scores, start=1):
                f.write(f"{topic_id} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")

    print(f"Saved {sum(len(v) for v in results.values()):,} results to {output_path}")
