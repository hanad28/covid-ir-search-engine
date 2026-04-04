"""
Streamlit search GUI for the COVID-19 IR search engine.

Three tabs:
  Search          -- live query interface with query pipeline display and scored results
  Corpus & Index  -- corpus statistics with field length distribution charts
  Evaluation      -- ablation results table, interactive Plotly chart, per-topic breakdown
"""

import json
import random
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import ir_measures
from ir_measures import nDCG

from config import INDEX_DIR, RUNS_DIR, RESULTS_DIR, RUN_NAMES, RANDOM_SEED, QRELS_FILE
from preprocess import preprocess
from retrieve import (
    search_bm25,
    search_bm25f,
    search_bm25f_rm3,
    get_rm3_expansion_terms,
)
from fuse import load_trec_run


# ---------------------------------------------------------------------------
# Page configuration and global styles
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="COVID-19 IR Search Engine",
    page_icon=None,
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .metric-card {
        background: #1e2130;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .metric-card .label {
        font-size: 0.78rem;
        color: #888;
        margin-bottom: 0.2rem;
    }
    .metric-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f0f0f0;
    }
    .score-bar-bg {
        background: #2a2d3e;
        border-radius: 4px;
        height: 6px;
        width: 100%;
        margin-top: 4px;
    }
    .score-bar-fill {
        background: #4c9be8;
        border-radius: 4px;
        height: 6px;
    }
    mark {
        background-color: #f0c040;
        color: #111;
        border-radius: 2px;
        padding: 0 2px;
    }
    .run-desc {
        font-size: 0.78rem;
        color: #aaa;
        margin-top: 0.2rem;
        margin-bottom: 0.8rem;
        line-height: 1.4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUN_FRIENDLY_NAMES = {
    "bm25_baseline": "BM25 Baseline",
    "bm25f_fielded": "BM25F",
    "bm25f_rm3": "BM25F + RM3",
    "bm25f_rm3_colbert": "ColBERTv2",
    "rrf_fusion": "RRF Fusion",
}

RUN_DESCRIPTIONS = {
    "BM25F + RM3": "BM25F field-weighted retrieval with RM3 pseudo-relevance feedback query expansion. Recommended default.",
    "BM25": "Standard BM25 over the combined contents field. Strong bag-of-words baseline.",
    "BM25F": "BM25F with field boosts: title x2.0, abstract x1.5, body x0.5. Weights important fields higher.",
    "ColBERTv2": "Neural reranker using late interaction. Cannot run live — shows BM25F+RM3 candidates.",
    "RRF Fusion": "Reciprocal Rank Fusion of BM25F+RM3 and ColBERTv2. Combines lexical and neural signals.",
}


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading search index...")
def load_searcher():
    """Load the Lucene index once and reuse across all queries."""
    if not INDEX_DIR.exists() or not any(INDEX_DIR.iterdir()):
        return None
    from pyserini.search.lucene import LuceneSearcher
    return LuceneSearcher(str(INDEX_DIR))


@st.cache_data(show_spinner=False)
def load_evaluation_results():
    """Load evaluation.csv into a pandas DataFrame."""
    csv_path = RESULTS_DIR / "evaluation.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


@st.cache_data(show_spinner=False)
def load_per_topic_results(run_name: str):
    """
    Compute per-topic nDCG@10 for a given run using ir_measures.

    Returns a DataFrame with columns: Topic, nDCG@10.
    """
    run_path = RUNS_DIR / f"{run_name}.txt"
    if not run_path.exists():
        return None

    qrels = list(ir_measures.read_trec_qrels(str(QRELS_FILE)))
    run = list(ir_measures.read_trec_run(str(run_path)))
    results = ir_measures.iter_calc([nDCG @ 10], qrels, run)

    rows = [{"Topic": r.query_id, "nDCG@10": round(r.value, 4)} for r in results]
    df = pd.DataFrame(rows).sort_values("Topic", key=lambda col: col.astype(int))
    return df.reset_index(drop=True)


@st.cache_data(show_spinner="Sampling corpus for field length statistics...")
def sample_field_lengths(num_docs: int = 200):
    """
    Sample documents from the index and return word counts per field.

    Returns a dict with keys 'title', 'abstract', 'body', each a list of int.
    """
    searcher = load_searcher()
    if searcher is None:
        return {}

    random.seed(RANDOM_SEED)
    total = searcher.num_docs
    sample_ids = [searcher.doc(i).docid() for i in range(min(num_docs, total))]
    random.shuffle(sample_ids)

    lengths = {"title": [], "abstract": [], "body": []}
    for doc_id in sample_ids:
        try:
            raw = searcher.doc(doc_id).raw()
            doc = json.loads(raw)
            for field in lengths:
                lengths[field].append(len(doc.get(field, "").split()))
        except Exception:
            continue

    return lengths


def fetch_document_metadata(doc_id: str, searcher) -> dict:
    """Retrieve title and abstract for a document from the Lucene index."""
    try:
        raw = searcher.doc(doc_id).raw()
        doc = json.loads(raw)
        return {
            "title": doc.get("title", "").strip() or "[No title]",
            "abstract": doc.get("abstract", "").strip() or "[No abstract]",
        }
    except Exception:
        return {"title": "[Title unavailable]", "abstract": "[Abstract unavailable]"}


def highlight_query_terms(text: str, query_tokens: set[str]) -> str:
    """Wrap query tokens found in text with HTML <mark> tags."""
    for token in sorted(query_tokens, key=len, reverse=True):
        if len(token) < 2:
            continue
        text = re.sub(
            rf"(?i)({re.escape(token)})",
            r"<mark>\1</mark>",
            text,
        )
    return text


def score_bar_html(score: float, max_score: float) -> str:
    """Return an HTML score bar scaled relative to max_score."""
    pct = int((score / max_score) * 100) if max_score > 0 else 0
    return (
        f'<div class="score-bar-bg">'
        f'<div class="score-bar-fill" style="width:{pct}%;"></div>'
        f"</div>"
    )


def run_search(
    query_text: str,
    run_label: str,
    top_k: int,
    searcher,
) -> tuple[list[tuple[str, float]], str]:
    """
    Execute the selected retrieval run for a single query.

    Returns (ranked_results, preprocessed_query_text).
    """
    processed = preprocess(query_text)
    queries = {"q": processed}

    if run_label == "BM25":
        results = search_bm25(queries)
    elif run_label == "BM25F":
        results = search_bm25f(queries)
    elif run_label in ("BM25F + RM3", "ColBERTv2", "RRF Fusion"):
        if run_label == "ColBERTv2":
            st.info(
                "ColBERTv2 reranking requires a GPU and pre-computed candidates. "
                "Showing BM25F+RM3 results as the retrieval stage."
            )
        elif run_label == "RRF Fusion":
            rrf_path = RUNS_DIR / f"{RUN_NAMES['E']}.txt"
            if not rrf_path.exists():
                st.error("RRF run file not found. Run scripts/run_rrf.py first.")
                return [], processed
            st.info("RRF results are pre-computed per TREC topic. Showing BM25F+RM3 for live queries.")
        results = search_bm25f_rm3(queries)
    else:
        results = search_bm25f_rm3(queries)

    hits = results.get("q", [])[:top_k]
    return hits, processed


# ---------------------------------------------------------------------------
# Tab: Search
# ---------------------------------------------------------------------------

def render_search_tab(searcher):
    st.subheader("Search COVID-19 Literature")

    with st.sidebar:
        st.header("Search settings")
        run_label = st.selectbox(
            "Retrieval run",
            options=["BM25F + RM3", "BM25", "BM25F", "ColBERTv2", "RRF Fusion"],
            index=0,
        )
        st.markdown(
            f'<div class="run-desc">{RUN_DESCRIPTIONS[run_label]}</div>',
            unsafe_allow_html=True,
        )
        top_k = st.slider("Number of results", min_value=5, max_value=20, value=10)

    query_input = st.text_input(
        label="Enter your query",
        placeholder="e.g. what is the origin of COVID-19",
    )
    search_clicked = st.button("Search", type="primary")

    if not search_clicked or not query_input.strip():
        st.info("Enter a query above and click Search.")
        return

    if searcher is None:
        st.error("Search index not found. Run `python scripts/build_index.py` first.")
        return

    with st.spinner("Searching..."):
        hits, processed_query = run_search(query_input.strip(), run_label, top_k, searcher)

    if not hits:
        st.warning("No results returned.")
        return

    query_tokens = set(processed_query.lower().split())

    # Query pipeline display
    with st.expander("Query processing pipeline", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Raw query**")
            st.code(query_input.strip())
        with col2:
            st.markdown("**After preprocessing**")
            st.code(processed_query)

        if run_label in ("BM25F + RM3", "ColBERTv2", "RRF Fusion"):
            with st.spinner("Fetching RM3 expansion terms..."):
                expansion_terms = get_rm3_expansion_terms(processed_query)
            if expansion_terms:
                st.markdown("**RM3 expansion terms**")
                term_display = "  ".join(
                    f"`{term}` ({weight:.3f})" for term, weight in expansion_terms[:10]
                )
                st.markdown(term_display)
            else:
                st.markdown("_RM3 expansion terms not available in this Pyserini version._")

    st.markdown(
        f"**{len(hits)} results** using {run_label} "
        f"| Query: _{query_input.strip()}_"
    )

    max_score = hits[0][1] if hits else 1.0

    for rank, (doc_id, score) in enumerate(hits, start=1):
        meta = fetch_document_metadata(doc_id, searcher)
        title = meta["title"]
        abstract = meta["abstract"]
        snippet = abstract[:400] + "..." if len(abstract) > 400 else abstract
        highlighted_snippet = highlight_query_terms(snippet, query_tokens)

        with st.expander(f"{rank}. {title}", expanded=(rank <= 3)):
            col_left, col_right = st.columns([4, 1])
            with col_left:
                st.markdown(
                    f"**Abstract:** {highlighted_snippet}",
                    unsafe_allow_html=True,
                )
            with col_right:
                st.markdown(f"**Rank:** {rank}")
                st.markdown(f"**Score:** {score:.4f}")
                st.markdown(score_bar_html(score, max_score), unsafe_allow_html=True)
                st.markdown(f"**Doc ID:** `{doc_id}`")


# ---------------------------------------------------------------------------
# Tab: Corpus & Index
# ---------------------------------------------------------------------------

def render_corpus_tab(searcher):
    st.subheader("Corpus Statistics and Index Overview")

    if searcher is None:
        st.error("Search index not found. Run `python scripts/build_index.py` first.")
        return

    total_docs = searcher.num_docs

    # Top metric cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div class="metric-card">'
            '<div class="label">Total indexed documents</div>'
            f'<div class="value">{total_docs:,}</div>'
            "</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card">'
            '<div class="label">Corpus</div>'
            '<div class="value">CORD-19 Apr 2020</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    # Field length statistics
    st.markdown("### Field length distribution")
    st.caption("Estimated from a 200-document sample.")

    lengths = sample_field_lengths(200)
    if not lengths:
        st.warning("Could not load field length statistics.")
        return

    def avg(lst):
        return round(sum(lst) / len(lst)) if lst else 0

    # Average word count cards
    c1, c2, c3 = st.columns(3)
    cards = [
        ("Avg title length", avg(lengths["title"]), "words"),
        ("Avg abstract length", avg(lengths["abstract"]), "words"),
        ("Avg body length", avg(lengths["body"]), "words"),
    ]
    for col, (label, value, unit) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label">{label}</div>'
                f'<div class="value">{value:,} <span style="font-size:0.9rem;color:#aaa">{unit}</span></div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    # Distribution chart
    field_labels = ["Title", "Abstract", "Body"]
    field_keys = ["title", "abstract", "body"]
    field_colours = ["#4c9be8", "#2ecc71", "#e67e22"]

    fig = go.Figure()
    for label, key, colour in zip(field_labels, field_keys, field_colours):
        fig.add_trace(go.Box(
            y=lengths[key],
            name=label,
            marker_color=colour,
            boxmean=True,
        ))

    fig.update_layout(
        title="Word count distribution per field (200-doc sample)",
        yaxis_title="Word count",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#f0f0f0",
        showlegend=False,
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Sample documents
    st.markdown("### Sample indexed documents")
    st.markdown(
        "These documents illustrate the three-field structure "
        "(title, abstract, body) used by the BM25F index."
    )

    random.seed(RANDOM_SEED)
    sample_ids = [searcher.doc(i).docid() for i in range(min(200, total_docs))]
    random.shuffle(sample_ids)

    for doc_id in sample_ids[:5]:
        try:
            raw = searcher.doc(doc_id).raw()
            doc = json.loads(raw)
            title = doc.get("title", "[No title]").strip()
            abstract = doc.get("abstract", "[No abstract]").strip()
            body_preview = doc.get("body", "")[:200].strip()

            with st.expander(f"Doc ID: {doc_id}  |  {title[:80]}"):
                st.markdown(f"**Title:** {title}")
                st.markdown(
                    f"**Abstract:** {abstract[:400]}{'...' if len(abstract) > 400 else ''}"
                )
                st.markdown(f"**Body (first 200 chars):** {body_preview}...")
                st.markdown("**Index fields:** `title`, `abstract`, `body`, `contents`")
        except Exception:
            continue


# ---------------------------------------------------------------------------
# Tab: Evaluation
# ---------------------------------------------------------------------------

def render_evaluation_tab():
    st.subheader("Ablation Study Results")

    df = load_evaluation_results()
    if df is None:
        st.warning(
            "No evaluation results found. "
            "Run `python scripts/evaluate.py` after completing all runs."
        )
        return

    # Rename and reorder columns
    df = df.rename(columns={"run": "Run", "AP": "MAP"})
    df["Run"] = df["Run"].map(lambda r: RUN_FRIENDLY_NAMES.get(r, r))
    ordered_cols = ["Run", "nDCG@10", "MAP", "R@100", "R@1000"]
    df = df[[c for c in ordered_cols if c in df.columns]]

    metric_cols = [c for c in ordered_cols if c != "Run"]

    # Colour-coded summary table: green cell for best in each metric column
    st.markdown("### Summary: all runs")

    def highlight_best(col):
        is_best = col == col.max()
        return [
            "background-color: #1a472a; color: #2ecc71; font-weight: bold" if v else ""
            for v in is_best
        ]

    styled = df.style.apply(highlight_best, subset=metric_cols).format(
        {c: "{:.4f}" for c in metric_cols}
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Interactive Plotly bar chart
    st.markdown("### nDCG@10 across ablation runs")

    ndcg_scores = df["nDCG@10"].tolist()
    run_labels = df["Run"].tolist()
    best_idx = ndcg_scores.index(max(ndcg_scores))
    bar_colours = ["#2ecc71" if i == best_idx else "#4c9be8" for i in range(len(ndcg_scores))]

    fig = go.Figure(data=[
        go.Bar(
            x=run_labels,
            y=ndcg_scores,
            marker_color=bar_colours,
            text=[f"{s:.4f}" for s in ndcg_scores],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>nDCG@10: %{y:.4f}<extra></extra>",
        )
    ])
    fig.update_layout(
        title="Ablation ladder: nDCG@10 per run",
        yaxis_title="nDCG@10",
        yaxis_range=[0, max(ndcg_scores) * 1.25],
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#f0f0f0",
        height=400,
        margin=dict(l=40, r=20, t=50, b=80),
        xaxis_tickangle=-15,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-topic breakdown
    st.markdown("### Per-topic nDCG@10 breakdown")

    available_run_names = list(RUN_FRIENDLY_NAMES.values())
    raw_run_names = list(RUN_FRIENDLY_NAMES.keys())
    selected_display = st.selectbox("Select run", options=available_run_names, index=0)
    selected_run = raw_run_names[available_run_names.index(selected_display)]

    per_topic_df = load_per_topic_results(selected_run)
    if per_topic_df is None:
        st.info(f"Run file for {selected_run} not found.")
        return

    # Per-topic bar chart
    fig2 = go.Figure(data=[
        go.Bar(
            x=per_topic_df["Topic"].astype(str).tolist(),
            y=per_topic_df["nDCG@10"].tolist(),
            marker_color="#4c9be8",
            hovertemplate="Topic %{x}<br>nDCG@10: %{y:.4f}<extra></extra>",
        )
    ])
    fig2.update_layout(
        title=f"Per-topic nDCG@10 — {selected_display}",
        xaxis_title="Topic",
        yaxis_title="nDCG@10",
        yaxis_range=[0, 1.1],
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#f0f0f0",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Full table and best/worst topics
    st.dataframe(per_topic_df, use_container_width=True, hide_index=True,
                 height=35 * len(per_topic_df) + 38)

    low = per_topic_df.nsmallest(3, "nDCG@10")
    high = per_topic_df.nlargest(3, "nDCG@10")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**3 lowest-scoring topics**")
        st.dataframe(low, use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**3 highest-scoring topics**")
        st.dataframe(high, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("COVID-19 Biomedical Search Engine")
    st.caption(
        "Multi-stage IR pipeline: BM25 / BM25F / BM25F+RM3 / ColBERTv2 / RRF "
        "| TREC-COVID Round 1 | CORD-19 corpus"
    )

    searcher = load_searcher()

    tab_search, tab_corpus, tab_eval = st.tabs(
        ["Search", "Corpus & Index", "Evaluation"]
    )

    with tab_search:
        render_search_tab(searcher)

    with tab_corpus:
        render_corpus_tab(searcher)

    with tab_eval:
        render_evaluation_tab()


if __name__ == "__main__":
    main()
