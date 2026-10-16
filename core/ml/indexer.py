"""Parallel, versioned, idempotent dense indexing for a non-active embedding profile.

Usage:
    python manage.py shell -c "from core.ml.indexer import build_index; print(build_index('bge-small'))"

Safety:
- Writes ONLY to the profile's own stamped collection.
- Refuses to write to the legacy MiniLM collection (servy_dense_kb).
- Uses the same eligibility rules as production indexing: RAG-enabled,
  INDEXED, non-quarantined chunks with non-empty text.
- Chunk IDs and metadata preserve tenant/document/chunk identity.
- Re-running skips chunks whose content hash is unchanged (resumable).
"""
from __future__ import annotations

import hashlib
import time

from .embeddings import get_profile, load_model, get_collection, passage_text, EmbeddingConfigError

INDEX_VERSION = "v1"


def _eligible_chunks(profile=None):
    from django.conf import settings
    from core.models import KnowledgeChunk
    qs = (KnowledgeChunk.objects
            .filter(is_quarantined=False, document__is_rag_enabled=True, document__index_status="INDEXED")
            .exclude(text="")
            .select_related("document", "document__domain", "document__category",
                            "document__product", "document__asset")
            .order_by("document_id", "chunk_index"))
    if profile is not None:
        real_tenants = getattr(settings, "STARLLY_REAL_TENANTS", ["skydream"])
        if profile.key.startswith("company"):
            qs = qs.filter(tenant__slug__in=real_tenants)
        else:
            qs = qs.exclude(tenant__slug__in=real_tenants)
    return qs


def _chunk_key(c) -> str:
    return f"t{c.tenant_id}_doc_{c.document_id}_v{c.document.index_version}_c{c.chunk_index}"


def build_index(profile_key: str, batch_size: int = 16, limit=None) -> dict:
    profile = get_profile(profile_key)
    if profile.collection == "servy_dense_kb":
        raise EmbeddingConfigError("Refusing to rebuild the active legacy MiniLM collection.")
    model = load_model(profile)
    coll = get_collection(profile)

    existing = {}
    got = coll.get(include=["metadatas"])
    for cid, meta in zip(got["ids"], got["metadatas"]):
        existing[cid] = (meta or {}).get("content_sha256")

    is_full_reconciliation = (limit is None)
    chunks = list(_eligible_chunks(profile)[:limit] if limit else _eligible_chunks(profile))

    wanted_ids = set()
    todo = []
    for c in chunks:
        text = passage_text(profile, c)
        if not text.strip():
            continue
        cid = _chunk_key(c)
        wanted_ids.add(cid)
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if existing.get(cid) != h:
            todo.append((cid, c, text, h))

    t0 = time.time()
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        emb = model.encode([b[2] for b in batch], normalize_embeddings=True, batch_size=batch_size)
        d = [b[1].document for b in batch]
        coll.upsert(
            ids=[b[0] for b in batch],
            embeddings=emb.tolist(),
            documents=[b[1].text for b in batch],
            metadatas=[{
                "tenant_id": int(b[1].tenant_id),
                "document_id": int(b[1].document_id),
                "chunk_id": int(b[1].id),
                "chunk_index": int(b[1].chunk_index),
                "document_version": int(dd.index_version or 0),
                "is_confidential": bool(dd.is_confidential),
                "customer_id": int(dd.customer_id or 0),
                "product_id": int(dd.product_id or 0),
                "asset_id": int(dd.asset_id or 0),
                "source_kind": "knowledge_document",
                "embedding_model": profile.model_id,
                "embedding_revision": profile.revision or "",
                "index_version": INDEX_VERSION,
                "content_sha256": b[3],
            } for b, dd in zip(batch, d)],
        )
    secs = time.time() - t0

    # CRITICAL: Stale cleanup must ONLY execute during a validated full reconciliation.
    # Partial indexing runs (e.g. limit=10) must NEVER prune vectors outside their explicit scope.
    stale = []
    if is_full_reconciliation:
        stale = [cid for cid in existing if cid not in wanted_ids]
        if stale:
            coll.delete(ids=stale)

    return {
        "profile": profile.key, "collection": profile.collection,
        "eligible_chunks": len(wanted_ids), "embedded_now": len(todo),
        "skipped_unchanged": len(wanted_ids) - len(todo), "removed_stale": len(stale),
        "is_full_reconciliation": is_full_reconciliation,
        "collection_count": coll.count(), "embed_seconds": round(secs, 2),
        "chunks_per_sec": round(len(todo) / secs, 2) if secs > 0 and todo else None,
    }


def index_single_document_bge(document, profile_key: str = "bge-small") -> int:
    """Safely index or refresh vectors for a single KnowledgeDocument.

    Preserves all other documents and removes only outdated vectors for this specific document.
    """
    profile = get_profile(profile_key)
    if profile.collection == "servy_dense_kb":
        raise EmbeddingConfigError("Refusing to mutate the legacy MiniLM collection directly.")
    if not document.is_rag_enabled or document.index_status != "INDEXED":
        return remove_document_bge(document.id, profile_key=profile_key)

    model = load_model(profile)
    coll = get_collection(profile)

    # Fetch existing vectors for this document only
    got = coll.get(where={"document_id": int(document.id)}, include=["metadatas"])
    existing_cids = set(got["ids"])

    chunks = list(document.chunks.filter(is_quarantined=False).exclude(text="").order_by("chunk_index"))
    if not chunks:
        return remove_document_bge(document.id, profile_key=profile_key)

    wanted_cids = set()
    todo = []
    for c in chunks:
        text = passage_text(profile, c)
        if not text.strip():
            continue
        cid = _chunk_key(c)
        wanted_cids.add(cid)
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        todo.append((cid, c, text, h))

    if todo:
        emb = model.encode([b[2] for b in todo], normalize_embeddings=True, batch_size=len(todo))
        coll.upsert(
            ids=[b[0] for b in todo],
            embeddings=emb.tolist(),
            documents=[b[1].text for b in todo],
            metadatas=[{
                "tenant_id": int(document.tenant_id),
                "document_id": int(document.id),
                "chunk_id": int(b[1].id),
                "chunk_index": int(b[1].chunk_index),
                "document_version": int(document.index_version or 0),
                "is_confidential": bool(document.is_confidential),
                "customer_id": int(document.customer_id or 0),
                "product_id": int(document.product_id or 0),
                "asset_id": int(document.asset_id or 0),
                "source_kind": "knowledge_document",
                "embedding_model": profile.model_id,
                "embedding_revision": profile.revision or "",
                "index_version": INDEX_VERSION,
                "content_sha256": b[3],
            } for b in todo],
        )

    # Clean up stale chunks for THIS document only
    doc_stale = [cid for cid in existing_cids if cid not in wanted_cids]
    if doc_stale:
        coll.delete(ids=doc_stale)

    return len(todo)


def remove_document_bge(document_id: int, profile_key: str = "bge-small") -> int:
    """Purge all chunk vectors for a single document from the BGE collection."""
    profile = get_profile(profile_key)
    if profile.collection == "servy_dense_kb":
        raise EmbeddingConfigError("Refusing to mutate the legacy MiniLM collection directly.")
    coll = get_collection(profile, create=False)
    got = coll.get(where={"document_id": int(document_id)})
    if got and got["ids"]:
        coll.delete(ids=got["ids"])
        return len(got["ids"])
    return 0


def reconcile_index(profile_key: str = "bge-small", dry_run: bool = True) -> dict:
    """Full reconciliation between database chunks and Chroma vectors with dry-run support."""
    from core.models import KnowledgeChunk
    profile = get_profile(profile_key)
    coll = get_collection(profile, create=False)
    got = coll.get(include=["metadatas"])

    expected = {_chunk_key(c) for c in _eligible_chunks(profile) if passage_text(profile, c).strip()}
    present = set(got["ids"])
    missing = expected - present
    orphans = present - expected

    report = {
        "profile": profile.key,
        "collection": profile.collection,
        "dry_run": dry_run,
        "expected_count": len(expected),
        "present_count": len(present),
        "missing_count": len(missing),
        "orphan_count": len(orphans),
        "is_consistent": (expected == present),
    }

    if not dry_run:
        if orphans:
            coll.delete(ids=list(orphans))
        if missing:
            build_index(profile_key, limit=None)
        report["reconciled"] = True
        report["final_count"] = coll.count()

    return report


def verify_index(profile_key: str) -> dict:
    """Completeness check: every eligible chunk present, no foreign/orphan rows."""
    from core.models import KnowledgeChunk
    profile = get_profile(profile_key)
    coll = get_collection(profile, create=False)
    got = coll.get(include=["metadatas"])
    expected = {_chunk_key(c) for c in _eligible_chunks(profile) if passage_text(profile, c).strip()}

    present = set(got["ids"])
    wrong_model = [i for i, m in zip(got["ids"], got["metadatas"]) if m.get("embedding_model") != profile.model_id]
    chunk_ids = [m["chunk_id"] for m in got["metadatas"]]
    real = set(KnowledgeChunk.objects.filter(id__in=chunk_ids).values_list("id", flat=True))
    return {
        "expected": len(expected), "present": len(present),
        "missing": len(expected - present), "orphans": len(present - expected),
        "wrong_model_rows": len(wrong_model),
        "unresolvable_chunk_ids": len(set(chunk_ids) - real),
        "complete": expected == present and not wrong_model and set(chunk_ids) <= real,
    }
