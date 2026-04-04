"""
Text preprocessing pipeline for documents and queries.

Applies the same normalisation to both documents and queries
to ensure consistent term matching at retrieval time.
"""

import re
import spacy

# Load spaCy model once at import time
_nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def clean_text(text: str) -> str:
    """Lowercase, strip HTML/XML artifacts, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)       # strip tags
    text = re.sub(r"[^a-z0-9\s\-]", " ", text) # keep alphanumeric, hyphens
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatise(text: str) -> str:
    """Lemmatise with spaCy, removing stopwords and short tokens."""
    doc = _nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and len(token.lemma_) > 1
    ]
    return " ".join(tokens)


def preprocess(text: str) -> str:
    """Full preprocessing pipeline: clean then lemmatise."""
    if not text or not text.strip():
        return ""
    return lemmatise(clean_text(text))
