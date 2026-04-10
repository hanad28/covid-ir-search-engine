# COVID-19 Biomedical IR Search Engine

Multi-stage information retrieval pipeline for COVID-19 scientific literature, evaluated on the TREC-COVID Round 1 benchmark over the CORD-19 corpus. 

The system implements a five-run ablation ladder: BM25 baseline, BM25F field-aware retrieval, BM25F+RM3 query expansion, ColBERTv2 neural reranking, and RRF rank fusion. A Streamlit GUI provides a live search interface, corpus inspection, and evaluation results.

---

## Requirements

- Python 3.10 or later
- Java 11 or later (required by Pyserini/Lucene — [Amazon Corretto 11](https://aws.amazon.com/corretto/) recommended; set `JAVA_HOME` manually if not auto-detected)
- A Google Colab account with T4 GPU runtime (for Run D only)

---

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Data

The full CORD-19 corpus and TREC-COVID benchmark files are required. To download them:

```bash
python scripts/download_data.py
```

This downloads and extracts the CORD-19 April 2020 release (~4 GB) and the Round 1 topics and qrels into `data/`.

A 200-document sample for quick reproducibility checks is available in `sample_data/`. See the [Sample data](#sample-data) section below.

---

## Running the full pipeline

### Step 1: Build the index

Converts the corpus to Pyserini JSONL format and builds a Lucene inverted index. Takes approximately 20-40 minutes.

```bash
python scripts/build_index.py
```

### Step 2: Run A, B, C (local)

Runs BM25 baseline, BM25F fielded retrieval, and BM25F+RM3 query expansion, then evaluates all three.

```bash
python scripts/run_all.py
```

Run files are saved to `runs/` and evaluation scores to `results/evaluation.csv`.

### Step 3: Run D — ColBERTv2 reranking (GPU required)

A CUDA-capable GPU is required for this step. Open `notebooks/run_colbert_colab.ipynb` on a machine with a GPU (or Google Colab with GPU runtime):

1. Set runtime to GPU: Runtime > Change runtime type > T4 GPU
2. Clone the repository and install dependencies (Steps 1–3 in the notebook)
3. Upload `runs/bm25f_rm3.txt` and `data/corpus_jsonl/docs.jsonl` when prompted (Step 4)
4. Build the index and run reranking (Steps 5–6) — this produces `runs/bm25f_rm3_colbert.txt`
5. Download `runs/bm25f_rm3_colbert.txt` and place it in your local `runs/` folder

### Step 4: Run E — RRF fusion (local)

```bash
python scripts/run_rrf.py
```

### Step 5: Final evaluation

```bash
python scripts/evaluate.py
```

Results are saved to `results/evaluation.csv`.

### Step 6: Launch the search GUI

```bash
streamlit run app.py
```

Opens a browser-based search interface at `http://localhost:8501`.

---

## Sample data

A 200-document subset is provided in `sample_data/` for markers to verify the implementation without downloading the full 3 GB corpus.

To run the full pipeline against the sample, make two small edits:

**1. Update topic and qrels paths in `src/config.py`** (lines 22–23):

```python
TOPICS_FILE = PROJECT_ROOT / "sample_data" / "topics" / "topics-rnd1.xml"
QRELS_FILE  = PROJECT_ROOT / "sample_data" / "qrels" / "qrels-rnd1.txt"
```

**2. Update the JSONL path in `src/index.py`** (line 19):

Change:
```python
JSONL_DIR = DATA_DIR / "corpus_jsonl"
```
to:
```python
JSONL_DIR = PROJECT_ROOT / "sample_data"
```

The JSONL file (`sample_data/docs.jsonl`) is already provided, so the JSONL build step is skipped automatically.

Then run:

```bash
python scripts/build_index.py   # builds Lucene index on 200 docs
python scripts/run_all.py       # runs all 5 retrieval methods
python scripts/evaluate.py      # outputs nDCG@10, MAP, R@100, R@1000
```

---

## Project structure

```
covid-ir-search-engine/
  app.py                        Streamlit search GUI
  requirements.txt              Python dependencies
  src/
    config.py                   All hyperparameters and paths
    preprocess.py               Text preprocessing (lowercasing, lemmatisation)
    index.py                    CORD-19 JSONL builder
    retrieve.py                 BM25, BM25F, BM25F+RM3, RM3 expansion inspector
    rerank.py                   ColBERTv2 candidate handling
    fuse.py                     Reciprocal Rank Fusion
    evaluate.py                 Evaluation via ir_measures
    topics.py                   TREC-COVID topic parser
  scripts/
    download_data.py            Download CORD-19 and TREC-COVID data
    build_index.py              Build Pyserini/Lucene index
    build_sample.py             Generate 200-document sample dataset
    run_bm25.py                 Run A: BM25
    run_bm25f.py                Run B: BM25F
    run_rm3.py                  Run C: BM25F+RM3
    run_colbert.py              Run D logic (used by Colab notebook)
    run_rrf.py                  Run E: RRF fusion
    run_all.py                  Runs A-C in sequence + evaluation
    evaluate.py                 Evaluate all available runs
  notebooks/
    run_colbert_colab.ipynb     Google Colab notebook for Run D
  data/
    corpus/                     Raw CORD-19 JSON files
    corpus_jsonl/               Processed docs.jsonl
    topics/                     TREC-COVID Round 1 topics
    qrels/                      TREC-COVID Round 1 qrels
  index/                        Pyserini/Lucene index
  runs/                         TREC runfiles (one per run)
  results/                      evaluation.csv
  sample_data/                  200-document reproducibility subset
```

---

## Evaluation metrics

| Metric | Description |
|--------|-------------|
| nDCG@10 | Normalised Discounted Cumulative Gain at rank 10 — measures early precision |
| MAP | Mean Average Precision — overall ranking quality across all topics |
| R@100 | Recall at 100 — candidate quality for neural reranking input |
| R@1000 | Recall at 1000 — full candidate generation recall |

---

## Ablation runs

| Run | File | Description |
|-----|------|-------------|
| A | `bm25_baseline.txt` | BM25 baseline |
| B | `bm25f_fielded.txt` | BM25F: title^2.0, abstract^1.5, body^0.5 |
| C | `bm25f_rm3.txt` | BM25F + RM3 pseudo-relevance feedback |
| D | `bm25f_rm3_colbert.txt` | Run C candidates reranked by ColBERTv2 |
| E | `rrf_fusion.txt` | RRF fusion of Run C + Run D (k=60) |

---

## Authors

Hanad Ali | Abdallah Ramadan | Kieran Cooke | Jamaldeen Adesope

ECS736P Information Retrieval - MSc Data Science and AI, Queen Mary University of London
