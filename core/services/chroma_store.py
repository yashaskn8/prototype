import logging
from typing import List, Dict, Any, Optional
from django.conf import settings
from django.utils import timezone
from django.db import transaction

logger = logging.getLogger("servy.rag.chroma")

_chroma_client = None


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        persist_dir = str(getattr(settings, "CHROMA_PERSIST_DIR", settings.BASE_DIR / "chroma_db"))
        _chroma_client = chromadb.PersistentClient(path=persist_dir)
    return _chroma_client


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
        sparse_coll = get_sparse_collection()
        sparse_coll.delete(where={"document_id": doc_id})
    except Exception as exc:
        logger.warning("Error deleting document %s from sparse collection: %s", doc_id, exc)

    try:
        from core.ml.indexer import remove_document_bge
        remove_document_bge(doc_id)
    except Exception as exc:
        logger.debug("Error removing BGE vectors for doc %s: %s", doc_id, exc)


def get_document_retrieval_readiness(document_id: Any) -> Dict[str, Any]:
    """Truthfully report readiness across retrieval backends without modifying existing vectors."""
    from core.models import KnowledgeDocument
    doc_id = document_id.id if hasattr(document_id, "id") else int(document_id)
    doc = KnowledgeDocument.objects.filter(id=doc_id).select_related("tenant", "asset").first()
    if not doc:
        return {
            "status": "NOT_FOUND",
            "parsed_successfully": False,
            "sparse_index_ready": False,
            "bge_vectors_ready": False,
            "source_provenance_status": "UNKNOWN",
            "customer_serving_eligible": False,
        }

    chunks_count = doc.chunks.count()
    has_chunks = chunks_count > 0
    parsed = bool(doc.content_text and has_chunks)
    sparse_ready = has_chunks and all(bool(c.embedding) for c in doc.chunks.all())

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
    if is_real_company_tenant(doc.tenant):
        source_verified = is_verified_real_document(doc)
        source_provenance_status = "VERIFIED" if source_verified else "UNVERIFIED"
        customer_eligible = source_verified
    else:
        source_provenance_status = "SYNTHETIC_OR_DEMO"
        customer_eligible = bool(doc.is_rag_enabled and not doc.is_confidential)

    parse_status = "PARSED" if parsed else "UNPARSED"
    sparse_index_status = "INDEXED" if sparse_ready else "NOT_INDEXED"
    bge_index_status = "INDEXED" if bge_ready else "NOT_INDEXED"

    return {
        "document_id": doc.id,
        "title": doc.title,
        "parsed_successfully": parsed,
        "parse_status": parse_status,
        "sparse_index_ready": sparse_ready,
        "sparse_index_status": sparse_index_status,
        "bge_vectors_ready": bge_ready,
        "bge_index_status": bge_index_status,
        "source_provenance_status": source_provenance_status,
        "customer_serving_eligible": customer_eligible,
        "index_status": doc.index_status,
        "index_error": doc.index_error,
    }


def sync_document_to_chroma(document_id: int):
    """Sync an indexed KnowledgeDocument from the authoritative relational database (MySQL) to ChromaDB.
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

    is_real = is_real_company_tenant(doc.tenant)
    profile_key = "company-bge-small" if is_real else "bge-small"
    try:
        # Dense vectors come only from the approved BGE profile, written to its
        # own stamped collection with section chunk identity and metadata
        # preserved. The model is loaded offline; there is no download fallback.
        from core.ml.indexer import index_single_document_bge
        doc.index_status = "INDEXED"   # in memory: request vectors for these chunks
        index_single_document_bge(doc, profile_key=profile_key)
        KnowledgeDocument.objects.filter(id=document_id).update(
            index_status="INDEXED",
            indexed_at=timezone.now(),
            index_error=""
        )
        logger.info("Indexed doc %s (%s chunks) to BGE profile %s", doc.id, len(chunks), profile_key)
        return True
    except Exception as exc:
        logger.error("Failed to index doc %s to Chroma: %s", doc.id, exc, exc_info=True)
        # Truthful reporting: dense index failed. Sparse index remains in the
        # relational database and is discoverable by the sparse fallback.
        KnowledgeDocument.objects.filter(id=document_id).update(
            index_status="FAILED",
            index_error=f"Chroma sync failed: {str(exc)[:480]}; sparse index may be active"
        )
        return False


def queue_chroma_sync(document_id: int):
    """Queue synchronization to Chroma upon relational database (MySQL) transaction commit."""
    transaction.on_commit(lambda: sync_document_to_chroma(document_id))


def queue_chroma_deletion(document_id: int):
    """Queue deletion from Chroma upon relational database (MySQL) transaction commit."""
    transaction.on_commit(lambda: delete_document_vectors(document_id))
