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


def is_authorized_reviewer(reviewer: str, tenant_id: int | None = None) -> bool:
    """Validate that reviewer identity is authenticated and authorized, not a bot or arbitrary string."""
    if not reviewer or not isinstance(reviewer, str):
        return False
    reviewer = reviewer.strip()
    if len(reviewer) < 3:
        return False
    # Reject automated/synthetic/placeholder reviewer strings
    disallowed = {"auto", "ai", "bot", "synthetic", "test", "dummy", "unknown", "system", "anonymous", "none", "placeholder", "self-approved"}
    rev_lower = reviewer.lower()
    if rev_lower in disallowed or any(d in rev_lower for d in disallowed):
        return False
    # Check against authorized settings if configured
    from django.conf import settings
    authorized = getattr(settings, "SERVY_AUTHORIZED_TRAINING_REVIEWERS", None)
    if authorized is not None:
        return reviewer in authorized or any(a.lower() == rev_lower for a in authorized)
    # Fail closed to authenticated staff user if allowlist not explicitly defined
    try:
        from django.contrib.auth.models import User
        if User.objects.filter(username__iexact=reviewer, is_staff=True).exists():
            return True
        for u in User.objects.filter(is_staff=True):
            full = f"{u.first_name} {u.last_name}".strip()
            if full and full.lower() == rev_lower:
                return True
    except Exception:
        pass
    return False


def validate_labels(rows: List[dict], require_approved: bool = True) -> dict:
    """Return {'valid': [...], 'rejected': [(query_id, reason)]}. Enforces full data and provenance integrity."""
    from core.models import KnowledgeChunk, SourceProvenanceRecord
    from django.utils.dateparse import parse_datetime
    def _get_pos(row):
        p = row.get("positive_chunk_ids")
        if p is None and row.get("positive_chunk_id") is not None:
            return [row.get("positive_chunk_id")]
        return p or []

    def _get_neg(row):
        n = row.get("negative_chunk_ids")
        if n is None and row.get("negative_chunk_id") is not None:
            return [row.get("negative_chunk_id")]
        return n or []

    all_ids = {i for r in rows for i in _get_pos(r) + _get_neg(r)}
    chunks = {c.id: c for c in KnowledgeChunk.objects.filter(id__in=all_ids).select_related("document")}
    
    doc_ids = {c.document_id for c in chunks.values() if c.document_id}
    # Explicit tenant-scoped, type-scoped provenance lookup
    prov_records = list(SourceProvenanceRecord.objects.filter(
        local_object_type="KnowledgeDocument",
        local_object_id__in=doc_ids,
    ))
    prov_map = {(p.tenant_id, p.local_object_id): p for p in prov_records}

    valid, rejected, seen = [], [], set()
    for r in rows:
        qid = r.get("query_id")
        pos, neg = _get_pos(r), _get_neg(r)
        reviewer = r.get("reviewer")
        reviewed_at = r.get("reviewed_at")
        reason = None
        if not (r.get("query_text") or "").strip():
            reason = "empty query"
        elif not pos:
            reason = "no positive"
        elif set(pos) & set(neg):
            reason = "contradictory: chunk is both positive and negative"
        elif qid in seen:
            reason = "duplicate query_id"
        elif require_approved and r.get("review_status") != APPROVED:
            reason = "not human-approved"
        elif require_approved and (not reviewer or not is_authorized_reviewer(reviewer, r.get("tenant_id"))):
            reason = "invalid or unauthorized reviewer identity"
        elif require_approved and (not reviewed_at or not parse_datetime(str(reviewed_at))):
            reason = "missing or invalid approval timestamp"
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
                # Check document provenance verification status
                prov = prov_map.get((c.tenant_id, c.document_id))
                if prov is not None:
                    if prov.verification_status not in ("VERIFIED", "") or prov.revoked_at is not None:
                        reason = f"chunk {cid} document has unverified/revoked provenance ({prov.verification_status})"
                        break
                    if r.get("source_org") and prov.source_org != str(r.get("source_org")):
                        reason = f"chunk {cid} document source organization mismatch ({prov.source_org} != {r.get('source_org')})"
                        break
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
    val_res = validate_labels(rows, require_approved=require_approved)
    valid_rows = val_res["valid"]
    rejected = list(val_res["rejected"])

    all_ids = {i for r in valid_rows for i in r.get("positive_chunk_ids", []) + r.get("negative_chunk_ids", [])}
    chunks = {c.id: c for c in KnowledgeChunk.objects.filter(id__in=all_ids).select_related("document")}

    verified_examples = []
    for r in valid_rows:
        qid = r.get("query_id")
        raw_pos = r.get("positive_chunk_ids")
        if raw_pos is None and r.get("positive_chunk_id") is not None:
            raw_pos = [r.get("positive_chunk_id")]
        pos_ids = raw_pos or []
        if not pos_ids:
            rejected.append((qid, "no positive chunk IDs"))
            continue

        pos_chunks = [chunks.get(pid) for pid in pos_ids if chunks.get(pid)]
        if len(pos_chunks) != len(pos_ids):
            missing_pids = [pid for pid in pos_ids if pid not in chunks]
            rejected.append((qid, f"positive chunk(s) {missing_pids} not found in database"))
            continue

        primary_pos = pos_chunks[0]
        # Content hash check on primary positive chunk
        computed_hash = hashlib.sha256(primary_pos.text.encode("utf-8")).hexdigest()
        expected_hash = r.get("source_content_hash")
        if expected_hash and computed_hash != expected_hash:
            rejected.append((qid, f"positive chunk {primary_pos.id} text has changed since approval (hash mismatch)"))
            continue

        # Document version check
        expected_ver = r.get("document_version")
        if expected_ver and primary_pos.document.index_version != expected_ver:
            rejected.append((qid, f"document version mismatch (expected {expected_ver}, got {primary_pos.document.index_version})"))
            continue

        # Resolve all negative chunks if present
        raw_neg = r.get("negative_chunk_ids")
        if raw_neg is None and r.get("negative_chunk_id") is not None:
            raw_neg = [r.get("negative_chunk_id")]
        neg_ids = raw_neg or []
        neg_chunks = [chunks.get(nid) for nid in neg_ids if chunks.get(nid)]
        
        pos_texts = [pc.text for pc in pos_chunks]
        neg_texts = [nc.text for nc in neg_chunks]

        verified_examples.append({
            "query_id": qid,
            "query_text": r["query_text"],
            "tenant_id": r["tenant_id"],
            "positive_chunk_id": primary_pos.id,
            "positive_chunk_ids": pos_ids,
            "positive_texts": pos_texts,
            "negative_chunk_ids": [nc.id for nc in neg_chunks],
            "negative_texts": neg_texts,
            "positive_document_id": primary_pos.document_id,
            "positive_text": pos_texts[0],
            "negative_text": neg_texts[0] if neg_texts else None,
            "source_group": r.get("source_group") or str(primary_pos.document_id),
            "reviewer": r.get("reviewer"),
            "reviewed_at": r.get("reviewed_at"),
            "source_content_hash": computed_hash,
            "dataset_version": r.get("dataset_version", "1.0"),
            "review_status": APPROVED if require_approved else r.get("review_status", "candidate"),
        })

    return {"verified_examples": verified_examples, "rejected": rejected}


def expand_training_triplets(verified_examples: List[dict]) -> List[dict]:
    """Expands multi-positive and multi-negative examples into individual training triplets.
    Ensures every reviewed hard negative and valid positive passage is preserved during training.
    """
    triplets = []
    for ex in verified_examples:
        pos_texts = ex.get("positive_texts") or ([ex.get("positive_text")] if ex.get("positive_text") else [])
        neg_texts = ex.get("negative_texts") or ([ex.get("negative_text")] if ex.get("negative_text") else [])
        qid = ex.get("query_id")
        qtext = ex.get("query_text")
        tenant_id = ex.get("tenant_id")
        source_group = ex.get("source_group")

        for p_idx, ptext in enumerate(pos_texts):
            if neg_texts:
                for n_idx, ntext in enumerate(neg_texts):
                    triplets.append({
                        "query_id": f"{qid}-p{p_idx}-n{n_idx}",
                        "query_text": qtext,
                        "positive_text": ptext,
                        "negative_text": ntext,
                        "tenant_id": tenant_id,
                        "source_group": source_group,
                    })
            else:
                triplets.append({
                    "query_id": f"{qid}-p{p_idx}",
                    "query_text": qtext,
                    "positive_text": ptext,
                    "negative_text": None,
                    "tenant_id": tenant_id,
                    "source_group": source_group,
                })
    return triplets


def approve_label_candidate(candidate: dict, reviewer: str, positive_chunk_id: int,
                            negative_chunk_ids: list | None = None, source_group: str | None = None) -> dict:
    if not is_authorized_reviewer(reviewer, candidate.get("tenant_id")):
        raise ValueError(f"Reviewer '{reviewer}' is not an authorized human reviewer.")
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
