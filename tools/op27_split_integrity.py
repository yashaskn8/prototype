"""Pure, read-only Op27 train/development split integrity gate.

Called before loading training weights. This is structural isolation; cannot
certify human truth, paraphrase independence or safe candidate authorization.
"""
from __future__ import annotations

import re


def _norm(value):
    return " ".join(re.findall(r"\w+", str(value or "").casefold()))


def _values(row, plural, singular):
    value = row.get(plural)
    if value is None:
        value = row.get(singular)
    if value is None:
        return set()
    if not isinstance(value, list):
        value = [value]
    return {str(x).strip() for x in value if x is not None and str(x).strip()}


def _inventory(rows, role):
    if not rows:
        raise ValueError(f"{role}: dataset is empty")
    result = {"questions": set(), "groups": set(), "docs": set(),
              "positive_texts": set(), "negative_texts": set(),
              "positive_ids": set(), "negative_ids": set(), "qids": set()}
    for i, row in enumerate(rows, 1):
        key = str(row.get("query_id") or "").strip()
        tenant = str(row.get("tenant_id") or "").strip()
        query = _norm(row.get("query_text") or row.get("query"))
        if not key or not query or not tenant:
            raise ValueError(f"{role} row {i}: missing query_id, tenant or query")
        if key in result["qids"]:
            raise ValueError(f"{role}: duplicate query ID {key}")
        result["qids"].add(key)
        result["questions"].add((tenant, query))
        group = row.get("source_group")
        if group:
            result["groups"].add((tenant, _norm(group)))
        docs = _values(row, "positive_document_ids", "positive_document_id")
        docs |= _values(row, "negative_document_ids", "negative_document_id")
        if not docs:
            raise ValueError(f"{role}/{key}: source document identity missing")
        result["docs"].update((tenant, doc) for doc in docs)
        result["positive_ids"].update(
            (tenant, cid) for cid in _values(row, "positive_chunk_ids", "positive_chunk_id"))
        result["negative_ids"].update(
            (tenant, cid) for cid in _values(row, "negative_chunk_ids", "negative_chunk_id"))
        pos = row.get("positive_text") or ""
        neg = row.get("negative_text") or ""
        if pos:
            result["positive_texts"].add((tenant, _norm(pos)))
        if neg:
            result["negative_texts"].add((tenant, _norm(neg)))
    return result


def enforce_train_dev_separation(train_rows, dev_rows):
    """Raise on known source/query/positive/negative overlap; return counts otherwise."""
    train = _inventory(train_rows, "train")
    dev = _inventory(dev_rows, "dev")
    comparisons = {
        "same_query": ("questions", "questions"),
        "same_source_group": ("groups", "groups"),
        "same_source_document": ("docs", "docs"),
        "same_positive_chunk": ("positive_ids", "positive_ids"),
        "training_negative_matches_dev_positive": ("negative_ids", "positive_ids"),
        "dev_negative_matches_training_positive": ("positive_ids", "negative_ids"),
        "same_positive_passage": ("positive_texts", "positive_texts"),
        "training_negative_matches_dev_positive_text": ("negative_texts", "positive_texts"),
        "dev_negative_matches_training_positive_text": ("positive_texts", "negative_texts"),
    }
    collisions = {}
    for label, (a, b) in comparisons.items():
        common = train[a] & dev[b]
        if common:
            collisions[label] = sorted(common)[:5]
    if collisions:
        raise ValueError(f"TRAINING_BLOCKED: train/dev contamination: {collisions}")
    # Report does not claim negatives are factual; it only checks structural leakage.
    return {
        "train_queries": len(train["questions"]),
        "dev_queries": len(dev["questions"]),
        "train_documents_including_negatives": len(train["docs"]),
        "dev_documents_including_negatives": len(dev["docs"]),
        "structural_overlap_count": 0,
        "human_truth_verified": False,
    }
