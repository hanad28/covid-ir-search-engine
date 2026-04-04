"""
Parse TREC-COVID topics XML into structured topic dicts.

Each topic has: number, query, question, narrative.
Query formulation concatenates the fields specified in config.TOPIC_FIELDS.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

from config import TOPICS_FILE, TOPIC_FIELDS
from preprocess import preprocess


def parse_topics(path: Path = TOPICS_FILE) -> list[dict]:
    """Parse topics XML and return list of topic dicts."""
    tree = ET.parse(path)
    root = tree.getroot()

    topics = []
    for topic_el in root.findall("topic"):
        topic = {
            "number": topic_el.attrib["number"],
            "query": topic_el.findtext("query", default="").strip(),
            "question": topic_el.findtext("question", default="").strip(),
            "narrative": topic_el.findtext("narrative", default="").strip(),
        }
        topics.append(topic)

    return topics


def formulate_query(topic: dict, fields: list[str] = TOPIC_FIELDS) -> str:
    """Concatenate selected topic fields into a single query string."""
    parts = [topic[f] for f in fields if topic.get(f)]
    return " ".join(parts)


def get_queries(preprocess_text: bool = True) -> dict[str, str]:
    """
    Return {topic_number: query_string} for all topics.

    If preprocess_text is True, applies the same preprocessing
    pipeline used on documents.
    """
    topics = parse_topics()
    queries = {}
    for topic in topics:
        raw = formulate_query(topic)
        queries[topic["number"]] = preprocess(raw) if preprocess_text else raw
    return queries
