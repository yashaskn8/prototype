"""Reconstruct a disposable customer-scoped BGE index from verified documents.

This is NOT recovery of the original frozen Chroma image. It is an explicit
separate demo retrieval mode that trusts the independently frozen source
documents / DB record hashes and recomputes vectors locally. It never reads
the drifted Chroma index or modifies the original DB, index or manifests.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile

MAX_APPROVED_CHUNKS = 3000
BATCH_SIZE = 16
_RUNTIME = None


def authoritative_chunks(asset_scope, asset_lookup):
    """Select and independently reauthorize each approved customer source."""
    from core.services.customer_answer import source_chunk
    from core.services.retriever import _knowledge_candidates
    from core.services.customer_isolation import is_verified_real_document
    from rest_framework.exceptions import NotFound
    gathered = {}
    per_asset = {}
    for asset_id, customer_id in sorted(asset_scope.items()):
        tenant, customer, asset = asset_lookup(asset_id)
        if (asset.id != asset_id or customer.id != customer_id
                or asset.customer_id != customer_id or asset.tenant_id != tenant.id):
            raise RuntimeError("Rebuild scope differs from frozen customer/asset binding.")
        valid = 0
        for candidate in _knowledge_candidates(
            tenant, asset, customer, False, backend="dense"
        ):
            try:
                permitted = source_chunk(
                    tenant=tenant, customer=customer, asset=asset,
                    document_id=candidate.document_id, chunk_id=candidate.id,
                )
            except NotFound:
                continue
            doc = permitted.document
            if (permitted.tenant_id != tenant.id
                    or permitted.document_id != candidate.document_id
                    or doc.disable_sharing or doc.is_confidential
                    or not doc.is_rag_enabled or doc.index_status != "INDEXED"
                    or not is_verified_real_document(doc)
                    or permitted.is_quarantined or not permitted.text.strip()):
                raise RuntimeError("Unverified or unauthorized source in rebuild.")
            existing = gathered.get(permitted.id)
            if existing is not None and (
                existing.document_id != permitted.document_id
                or existing.text != permitted.text
                or existing.heading != permitted.heading
            ):
                raise RuntimeError("Conflicting identity for an approved source chunk.")
            gathered[permitted.id] = permitted
            valid += 1
            if len(gathered) > MAX_APPROVED_CHUNKS:
                raise RuntimeError("Approved source limit exceeded; refuse incomplete rebuilding.")
        per_asset[asset_id] = valid
    # The original demo requires verified documentation for asset 60.
    if not per_asset.get(60) or not gathered:
        raise RuntimeError("The demo's required approved source passages are unavailable.")
    return [gathered[key] for key in sorted(gathered)], per_asset


def rebuild_isolated_bge_index(private, original_index_manifest, asset_scope, asset_lookup):
    """Create and validate a fresh local BGE collection without opening old Chroma.

    Caller must verify: SELECT-only MySQL identity, original attachment hashes,
    frozen source database counts and document content, BGE/V5 model identity,
    and the exact pre-existing original index manifest used only as a
    read-only drift sentinel. No source-index hash is ever accepted here.
    """
    from django.conf import settings
    from core.services import chroma_store
    from core.ml.embeddings import (
        get_profile, get_collection, load_model, encode_passages, passage_text,
    )
    from core.ml.indexer import (
        _chunk_key, _expected_vector_metadata, _profile_fingerprint,
    )
    from tools.servy_500_benchmark import file_manifest

    global _RUNTIME
    if _RUNTIME is not None or chroma_store._chroma_client is not None:
        raise RuntimeError("An index is already open; isolated reconstruction must run first.")
    root = Path(private).resolve(strict=True)
    production = Path(settings.CHROMA_PERSIST_DIR).resolve(strict=True)
    if not root.is_dir() or root == production or root.is_relative_to(production):
        raise RuntimeError("The rebuild destination must be a separate private directory.")
    if file_manifest(production) != original_index_manifest:
        raise RuntimeError("Original index changed before source verification.")
    chunks, per_asset = authoritative_chunks(asset_scope, asset_lookup)

    profile = get_profile("company-bge-small")
    if (not profile.promotion_eligible or profile.dim != 384
            or profile.collection != "servy_company_bge_small_v1"):
        raise RuntimeError("Unexpected local BGE model profile.")
    model = load_model(profile, allow_download=False)
    if model is None:
        raise RuntimeError("Verified local embedding model is unavailable.")
    fp = _profile_fingerprint(profile)
    instance = tempfile.TemporaryDirectory(
        prefix="servy_source_rebuilt_", dir=str(root), ignore_cleanup_errors=True
    )
    target = (Path(instance.name) / "index").resolve()
    target.mkdir()
    if target.is_relative_to(production) or production.is_relative_to(target):
        instance.cleanup()
        raise RuntimeError("Rebuild destination overlaps original Chroma.")
    original_setting = settings.CHROMA_PERSIST_DIR
    try:
        settings.CHROMA_PERSIST_DIR = target
        collection = get_collection(profile, create=True)
        ids = set()
        expected_rows = {}
        for offset in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[offset:offset + BATCH_SIZE]
            passages = [passage_text(profile, c) for c in batch]
            if any(not text.strip() for text in passages):
                raise RuntimeError("Empty source passage cannot be embedded.")
            chunk_ids = [_chunk_key(c) for c in batch]
            if len(chunk_ids) != len(set(chunk_ids)) or ids.intersection(chunk_ids):
                raise RuntimeError("Conflicting BGE vector IDs in authorized source.")
            meta = [
                _expected_vector_metadata(
                    c, profile, hashlib.sha256(text.encode("utf-8")).hexdigest(), fp
                )
                for c, text in zip(batch, passages)
            ]
            vectors = encode_passages(profile, model, passages).tolist()
            if len(vectors) != len(batch) or any(
                len(vec) != profile.dim or not all(math.isfinite(float(v)) for v in vec)
                for vec in vectors
            ):
                raise RuntimeError("Invalid local BGE model output.")
            collection.upsert(
                ids=chunk_ids, embeddings=vectors,
                metadatas=meta, documents=[c.text for c in batch],
            )
            ids.update(chunk_ids)
            expected_rows.update({
                key: (attributes, c.text, vec)
                for key, attributes, c, vec in zip(chunk_ids, meta, batch, vectors)
            })

        # Read back every generated vector and identity. No external Chroma
        # data was read, and no unapproved document can enter this collection.
        stored = collection.get(include=["metadatas", "documents", "embeddings"])
        if set(stored["ids"]) != ids or len(stored["ids"]) != len(ids):
            raise RuntimeError("The isolated BGE collection has unexpected or missing vector IDs.")
        for key, meta, doc, vec in zip(
            stored["ids"], stored["metadatas"], stored["documents"], stored["embeddings"]
        ):
            expected_meta, expected_doc, expected_vec = expected_rows[key]
            if any(meta.get(field) != value for field, value in expected_meta.items()):
                raise RuntimeError("The reconstructed vector provenance does not match the approved source.")
            if doc != expected_doc or len(vec) != profile.dim:
                raise RuntimeError("The reconstructed passage differs from the approved source.")
            if any(abs(float(a) - float(b)) > 0.00001 for a, b in zip(vec, expected_vec)):
                raise RuntimeError("BGE vectors changed when persisted to the isolated index.")
        if collection.count() != len(ids):
            raise RuntimeError("Isolated BGE collection count changed.")
        if file_manifest(production) != original_index_manifest:
            raise RuntimeError("Original Chroma bytes changed during independent reconstruction.")
    except BaseException:
        settings.CHROMA_PERSIST_DIR = original_setting
        instance.cleanup()
        raise

    _RUNTIME = instance
    return {"mode": "rebuilt-from-frozen-authorized-documents",
            "verified_chunks": len(ids), "approved_assets": per_asset}
