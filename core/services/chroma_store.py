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


def get_sparse_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name="servy_sparse_fallback",
        metadata={"hnsw:space": "cosine"}
    )


def delete_document_vectors(document_id: int):
    """Purge all chunk vectors for a given document from both Chroma collections."""
    try:
        dense_coll = get_dense_collection()
        dense_coll.delete(where={"document_id": document_id})
    except Exception as exc:
        logger.warning("Error deleting document %s from dense collection: %s", document_id, exc)

    try:
        sparse_coll = get_sparse_collection()
        sparse_coll.delete(where={"document_id": document_id})
    except Exception as exc:
        logger.warning("Error deleting document %s from sparse collection: %s", document_id, exc)


def sync_document_to_chroma(document_id: int):
    """Sync an indexed KnowledgeDocument from SQLite to ChromaDB.
    Runs after SQLite transaction commit.
    """
    from core.models import KnowledgeDocument
    try:
        doc = KnowledgeDocument.objects.select_related(
            "tenant", "product", "asset", "category", "domain", "customer"
        ).prefetch_related("chunks").get(id=document_id)
    except KnowledgeDocument.DoesNotExist:
        delete_document_vectors(document_id)
        return False

    if not doc.is_rag_enabled:
        delete_document_vectors(document_id)
        KnowledgeDocument.objects.filter(id=document_id).update(
            index_status="NOT_INDEXED",
            index_error="RAG is disabled for this document"
        )
        return False

    chunks = list(doc.chunks.filter(is_quarantined=False).order_by("chunk_index"))
    if not chunks:
        delete_document_vectors(document_id)
        KnowledgeDocument.objects.filter(id=document_id).update(
            index_status="NOT_INDEXED",
            index_error="No valid chunks to index"
        )
        return False

    model = get_sentence_transformer()
    if model is None:
        # Sentence Transformer unavailable; record state and let fallback retriever handle it
        KnowledgeDocument.objects.filter(id=document_id).update(
            index_status="INDEXED",
            indexed_at=timezone.now(),
            index_error="Indexed with sparse fallback only (SentenceTransformer unavailable)"
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
        
        dense_coll = get_dense_collection()
        # Clean previous chunks for this document
        dense_coll.delete(where={"document_id": doc.id})
        
        dense_coll.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=metadatas
        )

        KnowledgeDocument.objects.filter(id=document_id).update(
            index_status="INDEXED",
            indexed_at=timezone.now(),
            index_error=""
        )
        logger.info("Successfully indexed doc %s (%s chunks) to Chroma dense collection", doc.id, len(chunks))
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
