"""
Centralised configuration for the COVID-19 IR search engine.
All hyperparameters, paths, and constants live here.
"""

import os
from pathlib import Path

# Java home (required by Pyserini/Lucene)
# Auto-detect if not already set in environment
if "JAVA_HOME" not in os.environ:
    _corretto_path = Path("C:/Program Files/Amazon Corretto/jdk11.0.30_7")
    if _corretto_path.exists():
        os.environ["JAVA_HOME"] = str(_corretto_path)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directory paths
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
TOPICS_FILE = DATA_DIR / "topics" / "topics-rnd1.xml"
QRELS_FILE = DATA_DIR / "qrels" / "qrels-rnd1.txt"
CORPUS_METADATA = DATA_DIR / "corpus" / "metadata.csv"

INDEX_DIR = PROJECT_ROOT / "index"
RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Topic field selection
# Only query + question fields; narrative excluded to avoid drift
TOPIC_FIELDS = ["query", "question"]

# BM25 baseline (Run A)
BM25_K1 = 0.9
BM25_B = 0.4

# BM25F field weights (Run B)
# Higher weight = more influence on ranking
BM25F_WEIGHTS = {
    "title": 2.0,
    "abstract": 1.5,
    "body": 0.5,
}
BM25F_K1 = 0.9
BM25F_B = 0.4

# RM3 pseudo-relevance feedback (Run C)
RM3_FB_DOCS = 10       # Number of feedback documents
RM3_FB_TERMS = 10      # Number of expansion terms
RM3_ORIGINAL_WEIGHT = 0.5  # Interpolation weight for original query

# ColBERTv2 reranking (Run D)
RERANK_DEPTH = 100     # Rerank top-N candidates from Run C
COLBERT_MODEL = "colbert-ir/colbertv2.0"

# RRF fusion (Run E)
RRF_K = 60             # Damping constant

# Retrieval depth
RETRIEVAL_DEPTH = 1000  # Number of documents retrieved per query

# Evaluation metrics
EVAL_METRICS = ["nDCG@10", "MAP", "R@100", "R@1000"]

# Run names (used in TREC runfile output)
RUN_NAMES = {
    "A": "bm25_baseline",
    "B": "bm25f_fielded",
    "C": "bm25f_rm3",
    "D": "bm25f_rm3_colbert",
    "E": "rrf_fusion",
}

# Reproducibility
RANDOM_SEED = 42
