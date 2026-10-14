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
        else:
            for cid in pos + neg:
                c = chunks.get(cid)
                if c is None:
                    reason = f"chunk {cid} missing/deleted"; break
                if c.tenant_id != r.get("tenant_id"):
                    reason = f"chunk {cid} cross-tenant"; break
                if c.is_quarantined or not c.document.is_rag_enabled:
                    reason = f"chunk {cid} quarantined or RAG-disabled"; break
        if reason:
            rejected.append((qid, reason))
        else:
            seen.add(qid)
            valid.append(r)
    return {"valid": valid, "rejected": rejected}


def split_by_group(rows: List[dict], val=0.15, test=0.15):
    """Deterministic split by source_group so related passages never straddle splits."""
    out = {"train": [], "validation": [], "test": []}
    for r in rows:
        g = r.get("source_group") or str(r["positive_chunk_ids"][0])
        h = int(hashlib.sha256(g.encode()).hexdigest(), 16) % 1000 / 1000
        out["test" if h < test else "validation" if h < test + val else "train"].append(r)
    return out
