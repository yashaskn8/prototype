import logging
import threading
from typing import List, Dict, Any, Optional
from django.conf import settings
from django.utils import timezone
from django.db import transaction

logger = logging.getLogger("servy.rag.chroma")

_chroma_client = None
_sentence_model = None
_model_lock = threading.Lock()


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        persist_dir = str(getattr(settings, "CHROMA_PERSIST_DIR", settings.BASE_DIR / "chroma_db"))
        _chroma_client = chromadb.PersistentClient(path=persist_dir)
    return _chroma_client


def get_sentence_transformer():
    global _sentence_model
    if _sentence_model is None:
        with _model_lock:
            if _sentence_model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    try:
                        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
                    except Exception:
                        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
                    logger.info("Loaded SentenceTransformer('all-MiniLM-L6-v2') successfully")
                except Exception as exc:
                    logger.warning("Could not load SentenceTransformer: %s", exc)
                    _sentence_model = None
    return _sentence_model


def get_dense_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name="servy_dense_kb",
        metadata={"hnsw:space": "cosine"}
    )


def get_company_dense_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name="servy_company_dense_kb",
        metadata={"hnsw:space": "cosine"}
    )


def is_real_company_tenant(tenant):
    real_tenants = getattr(settings, "STARLLY_REAL_TENANTS", ["skydream"])
    slug = getattr(tenant, "slug", str(tenant) if isinstance(tenant, str) else "")
    return slug in real_tenants


def get_sparse_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name="servy_sparse_fallback",
        metadata={"hnsw:space": "cosine"}
    )


def delete_document_vectors(document_id: Any, allow_company_purge: bool = False):
    """Purge chunk vectors for a given document.
    Zero-deletion contract: Real company vectors are strictly protected from deletion.
    Purge overrides are unconditionally rejected for real company data.
    """
    from core.models import KnowledgeDocument
    doc_id = document_id.id if hasattr(document_id, "id") else int(document_id)
    doc = KnowledgeDocument.objects.filter(id=doc_id).select_related("tenant").first()
    if doc and is_real_company_tenant(doc.tenant):
        logger.warning("Zero-deletion contract: Refusing to purge real company vectors for document %s", doc_id)
        raise RuntimeError(f"Zero-deletion violation: Refusing to purge real company vectors for document {doc_id}")

    try:
        dense_coll = get_dense_collection()
        dense_coll.delete(where={"document_id": doc_id})
    except Exception as exc:
        logger.warning("Error deleting document %s from dense collection: %s", doc_id, exc)

    try:
        sparse_coll = get_sparse_collection()
        sparse_coll.delete(where={"document_id": document_id})
    except Exception as exc:
        logger.warning("Error deleting document %s from sparse collection: %s", document_id, exc)

    try:
        from core.ml.indexer import remove_document_bge
        remove_document_bge(document_id)
    except Exception as exc:
        logger.debug("Error removing BGE vectors for doc %s: %s", document_id, exc)


def get_document_retrieval_readiness(document_id: Any) -> Dict[str, Any]:
    """Truthfully report readiness across retrieval backends without modifying existing vectors."""
    from core.models import KnowledgeDocument
    doc_id = document_id.id if hasattr(document_id, "id") else int(document_id)
    doc = KnowledgeDocument.objects.filter(id=doc_id).select_related("tenant", "asset").first()
    if not doc:
        return {"status": "NOT_FOUND", "customer_serving_eligible": False}

    chunks_count = doc.chunks.count()
    has_chunks = chunks_count > 0
    parsed = bool(doc.content_text and has_chunks)
    sparse_ready = has_chunks and all(c.embedding for c in doc.chunks.all())

    minilm_ready = False
    try:
        coll = get_company_dense_collection() if is_real_company_tenant(doc.tenant) else get_dense_collection()
        res = coll.get(where={"document_id": doc.id}, limit=1)
        minilm_ready = bool(res and res.get("ids"))
    except Exception:
        minilm_ready = False

    bge_ready = False
    try:
        from core.ml.embeddings import get_profile, get_collection
        bge_profile = get_profile("company-bge-small" if is_real_company_tenant(doc.tenant) else "bge-small")
        bge_coll = get_collection(bge_profile, create=False)
        bge_res = bge_coll.get(where={"document_id": doc.id}, limit=1)
        bge_ready = bool(bge_res and bge_res.get("ids"))
    except Exception:
        bge_ready = False

    from core.services.customer_isolation import is_verified_real_document
    customer_eligible = is_verified_real_document(doc) if is_real_company_tenant(doc.tenant) else (doc.is_rag_enabled and not doc.is_confidential)

    return {
        "document_id": doc.id,
        "title": doc.title,
        "parsed_successfully": parsed,
        "sparse_index_ready": sparse_ready,
        "minilm_vectors_ready": minilm_ready,
        "bge_vectors_ready": bge_ready,
        "customer_serving_eligible": customer_eligible,
        "index_status": doc.index_status,
        "index_error": doc.index_error,
    }


def sync_document_to_chroma(document_id: int):
    """Sync an indexed KnowledgeDocument from SQLite to ChromaDB.
    Runs non-destructively: never deletes existing vectors.
    """
    from core.models import KnowledgeDocument
    try:
        doc = KnowledgeDocument.objects.select_related(
            "tenant", "product", "asset", "category", "domain", "customer"
        ).prefetch_related("chunks").get(id=document_id)
    except KnowledgeDocument.DoesNotExist:
        return False

    if not doc.is_rag_enabled:
        KnowledgeDocument.objects.filter(id=document_id).update(
            index_status="NOT_INDEXED",
            index_error="RAG is disabled for this document"
        )
        return False

    chunks = list(doc.chunks.filter(is_quarantined=False).order_by("chunk_index"))
    if not chunks:
        KnowledgeDocument.objects.filter(id=document_id).update(
            index_status="NOT_INDEXED",
            index_error="No valid chunks to index"
        )
        return False

    model = get_sentence_transformer()
    if model is None:
        KnowledgeDocument.objects.filter(id=document_id).update(
            index_status="FAILED",
            indexed_at=timezone.now(),
            index_error="Dense index failed: SentenceTransformer model unavailable. Dense vectors not indexed."
        )
        return False

    try:
        # Stable chunk ID format: doc_{id}_v{version}_c{idx}
        version = doc.index_version
        ids = [f"doc_{doc.id}_v{version}_c{c.chunk_index}" for c in chunks]

        # Build composite metadata-aware text for embedding
        metadata_prefix = " ".join(filter(None, [
            doc.title, doc.description, doc.tags, doc.doc_type,
            doc.domain.name if doc.domain else "",
            doc.category.name if doc.category else "",
            doc.product.name if doc.product else "",
            doc.asset.name if doc.asset else "",
            doc.asset.model_number if doc.asset else "",
        ]))
        texts_to_embed = [
            f"{metadata_prefix} {c.heading}\n{c.text}".strip() for c in chunks
        ]

        embeddings = model.encode(texts_to_embed, normalize_embeddings=True).tolist()

        metadatas = [{
            "document_id": int(doc.id),
            "tenant_id": int(doc.tenant_id),
            "chunk_index": int(c.chunk_index),
            "is_confidential": bool(doc.is_confidential),
            "customer_id": int(doc.customer_id) if doc.customer_id else 0,
            "product_id": int(doc.product_id) if doc.product_id else 0,
            "asset_id": int(doc.asset_id) if doc.asset_id else 0,
            "title": str(doc.title)[:200],
            "heading": str(c.heading)[:200],
            "doc_type": str(doc.doc_type),
        } for c in chunks]

        chunk_texts = [c.text for c in chunks]

        is_real = is_real_company_tenant(doc.tenant)
        if is_real:
            dense_coll = get_company_dense_collection()
        else:
            dense_coll = get_dense_collection()

        # Non-destructive vector addition: do NOT delete previous vectors
        existing = dense_coll.get(ids=ids)
        existing_ids = set(existing.get("ids") or [])
        new_indices = [i for i, cid in enumerate(ids) if cid not in existing_ids]

        if new_indices:
            dense_coll.add(
                ids=[ids[i] for i in new_indices],
                embeddings=[embeddings[i] for i in new_indices],
                documents=[chunk_texts[i] for i in new_indices],
                metadatas=[metadatas[i] for i in new_indices]
            )

        KnowledgeDocument.objects.filter(id=document_id).update(
            index_status="INDEXED",
            indexed_at=timezone.now(),
            index_error=""
        )
        logger.info("Successfully indexed doc %s (%s chunks) to Chroma dense collection (is_real=%s)", doc.id, len(chunks), is_real)

        if getattr(settings, "SERVY_USE_EXPERIMENTAL_RAG", False):
            try:
                from core.ml.indexer import index_single_document_bge
                bge_profile = "company-bge-small" if is_real else "bge-small"
                index_single_document_bge(doc, profile_key=bge_profile)
            except Exception as bge_exc:
                logger.warning("BGE experimental indexing skipped/failed for doc %s: %s", doc.id, bge_exc)


        return True

    except Exception as exc:
        logger.error("Failed to index doc %s to Chroma: %s", doc.id, exc, exc_info=True)
        # IMPORTANT: Do NOT set index_status to FAILED here.  The SQLite
        # sparse embeddings were created successfully by index_document() and
        # must remain discoverable by the fallback retriever.  Only record
        # the Chroma-specific failure in index_error.
        KnowledgeDocument.objects.filter(id=document_id).update(
            index_status="INDEXED",
            index_error=f"Chroma sync failed: {str(exc)[:480]}; sparse index active"
        )
        return False


def queue_chroma_sync(document_id: int):
    """Queue synchronization to Chroma upon SQLite transaction commit."""
    transaction.on_commit(lambda: sync_document_to_chroma(document_id))


def queue_chroma_deletion(document_id: int):
    """Queue deletion from Chroma upon SQLite transaction commit."""
    transaction.on_commit(lambda: delete_document_vectors(document_id))
