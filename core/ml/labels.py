"""Relevance-label workflow: candidate export for HUMAN review, validation, splitting.

No label produced here is trusted until a human sets review_status="approved".

Label file (JSONL, stored under private_training_data/, git-ignored), one row per query:
{
  "query_id": "q-0001",
  "query_text": "Milk analyzer powers on but does not draw a sample",
  "tenant_id": 1,
  "positive_chunk_ids": [123],
  "negative_chunk_ids": [456],          # must be reviewed as genuinely irrelevant
  "source_group": "milk-analyzer-manual", # used for leakage-safe splitting
  "review_status": "approved",          # candidate | approved | rejected
  "reviewer": "name"
}
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List

APPROVED = "approved"


def load_jsonl(path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def export_review_candidates(queries: Iterable[dict], profile_key: str, out_path, top_k: int = 5) -> int:
    """For each {query_id, query_text, tenant_id}, write top-k candidate chunks for a reviewer.

    Output rows have review_status="candidate" and are NOT usable for training or
    evaluation until a human edits positive/negative IDs and sets status to approved.
    """
    from core.models import KnowledgeChunk
    from .embeddings import get_profile, load_model, get_collection, encode_queries
    p = get_profile(profile_key)
    model, coll = load_model(p), get_collection(p, create=False)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for q in queries:
            emb = encode_queries(p, model, [q["query_text"]])[0].tolist()
            res = coll.query(query_embeddings=[emb], n_results=top_k,
                             where={"tenant_id": int(q["tenant_id"])}, include=["metadatas", "distances"])
            ids = [m["chunk_id"] for m in res["metadatas"][0]]
            texts = dict(KnowledgeChunk.objects.filter(id__in=ids).values_list("id", "text"))
            f.write(json.dumps({
                **q, "review_status": "candidate", "positive_chunk_ids": [], "negative_chunk_ids": [],
                "candidates": [{"chunk_id": cid, "document_id": m["document_id"],
                                "similarity": round(1 - d, 4), "preview": (texts.get(cid) or "")[:400]}
                               for cid, m, d in zip(ids, res["metadatas"][0], res["distances"][0])],
            }, ensure_ascii=False) + "\n")
            n += 1
    return n


def validate_labels(rows: List[dict]) -> dict:
    """Return {'valid': [...], 'rejected': [(query_id, reason)]}. Only approved rows can be valid."""
    from core.models import KnowledgeChunk
    all_ids = {i for r in rows for i in r.get("positive_chunk_ids", []) + r.get("negative_chunk_ids", [])}
    chunks = {c.id: c for c in KnowledgeChunk.objects.filter(id__in=all_ids).select_related("document")}
    valid, rejected, seen = [], [], set()
    for r in rows:
        qid = r.get("query_id")
        pos, neg = r.get("positive_chunk_ids") or [], r.get("negative_chunk_ids") or []
        reason = None
        if r.get("review_status") != APPROVED:
            reason = "not human-approved"
        elif not (r.get("query_text") or "").strip():
            reason = "empty query"
        elif not pos:
            reason = "no positive"
        elif set(pos) & set(neg):
            reason = "contradictory: chunk is both positive and negative"
        elif qid in seen:
            reason = "duplicate query_id"
        elif not (r.get("reviewer") or "").strip():
            reason = "missing reviewer identity"
        else:
            for cid in pos + neg:
                c = chunks.get(cid)
                if c is None:
                    reason = f"chunk {cid} missing/deleted"; break
                if c.tenant_id != r.get("tenant_id"):
                    reason = f"chunk {cid} cross-tenant"; break
                if c.is_quarantined or not c.document.is_rag_enabled:
                    reason = f"chunk {cid} quarantined or RAG-disabled"; break
                if c.document.is_confidential and not r.get("allow_confidential", False):
                    reason = f"chunk {cid} is confidential and not permitted for general training"; break
        if reason:
            rejected.append((qid, reason))
        else:
            seen.add(qid)
            valid.append(r)
    return {"valid": valid, "rejected": rejected}


def resolve_provenance_verified_examples(rows: List[dict], require_approved: bool = True) -> dict:
    """Resolve database chunk text for verified training rows. Never trusts caller-supplied text.

    Enforces:
    - Provenance verification: resolves positive and negative texts directly from DB chunks.
    - Checksum integrity: verifies SHA-256 matches DB chunk text if source_content_hash is supplied.
    - Version consistency: verifies document version if recorded.
    - Returns {'verified_examples': [...], 'rejected': [(qid, reason)]}.
    """
    from core.models import KnowledgeChunk
    val_res = validate_labels(rows) if require_approved else {"valid": rows, "rejected": []}
    valid_rows = val_res["valid"]
    rejected = list(val_res["rejected"])

    all_ids = {i for r in valid_rows for i in r.get("positive_chunk_ids", []) + r.get("negative_chunk_ids", [])}
    chunks = {c.id: c for c in KnowledgeChunk.objects.filter(id__in=all_ids).select_related("document")}

    verified_examples = []
    for r in valid_rows:
        qid = r.get("query_id")
        pos_id = r["positive_chunk_ids"][0]
        pos_chunk = chunks.get(pos_id)
        if not pos_chunk:
            rejected.append((qid, f"positive chunk {pos_id} not found in database"))
            continue

        # Content hash check
        computed_hash = hashlib.sha256(pos_chunk.text.encode("utf-8")).hexdigest()
        expected_hash = r.get("source_content_hash")
        if expected_hash and computed_hash != expected_hash:
            rejected.append((qid, f"positive chunk {pos_id} text has changed since approval (hash mismatch)"))
            continue

        # Document version check
        expected_ver = r.get("document_version")
        if expected_ver and pos_chunk.document.index_version != expected_ver:
            rejected.append((qid, f"document version mismatch (expected {expected_ver}, got {pos_chunk.document.index_version})"))
            continue

        # Resolve negative text if present
        neg_text = None
        if r.get("negative_chunk_ids"):
            neg_id = r["negative_chunk_ids"][0]
            neg_chunk = chunks.get(neg_id)
            if neg_chunk:
                neg_text = neg_chunk.text

        verified_examples.append({
            "query_id": qid,
            "query_text": r["query_text"],
            "tenant_id": r["tenant_id"],
            "positive_chunk_id": pos_id,
            "positive_document_id": pos_chunk.document_id,
            "positive_text": pos_chunk.text,
            "negative_text": neg_text,
            "source_group": r.get("source_group") or str(pos_chunk.document_id),
            "reviewer": r.get("reviewer"),
            "reviewed_at": r.get("reviewed_at"),
            "source_content_hash": computed_hash,
            "dataset_version": r.get("dataset_version", "1.0"),
            "review_status": APPROVED if require_approved else r.get("review_status", "candidate"),
        })

    return {"verified_examples": verified_examples, "rejected": rejected}


def approve_label_candidate(candidate: dict, reviewer: str, positive_chunk_id: int,
                            negative_chunk_ids: list | None = None, source_group: str | None = None) -> dict:
    """Helper for human reviewers to convert a candidate into an approved, provenance-stamped row."""
    from django.utils import timezone
    from core.models import KnowledgeChunk
    chunk = KnowledgeChunk.objects.select_related("document").get(id=positive_chunk_id)
    content_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()

    return {
        "query_id": candidate["query_id"],
        "query_text": candidate["query_text"],
        "tenant_id": candidate["tenant_id"],
        "positive_chunk_ids": [positive_chunk_id],
        "negative_chunk_ids": negative_chunk_ids or [],
        "source_group": source_group or f"doc-{chunk.document_id}",
        "review_status": APPROVED,
        "reviewer": reviewer.strip(),
        "reviewed_at": timezone.now().isoformat(),
        "source_content_hash": content_hash,
        "document_version": chunk.document.index_version,
        "dataset_version": "1.0",
    }


def split_by_group(rows: List[dict], val=0.15, test=0.15):
    """Deterministic split by source_group so related passages never straddle splits."""
    out = {"train": [], "validation": [], "test": []}
    for r in rows:
        g = r.get("source_group") or str(r.get("positive_chunk_ids", [0])[0] if isinstance(r.get("positive_chunk_ids"), list) and r.get("positive_chunk_ids") else "default")
        h = int(hashlib.sha256(g.encode()).hexdigest(), 16) % 1000 / 1000
        out["test" if h < test else "validation" if h < test + val else "train"].append(r)
    return out
