import hashlib
import re

from .content_safety import detect_document_prompt_injection
from .document_loader import extract_document_text
from .chunking import chunk_markdownish
from .vectors import embed_text, VECTOR_MODEL
from core.models import KnowledgeChunk, ServiceResolutionIndex, ServiceCall


def _doc_embedding_text(document, heading, body):
    metadata = " ".join(filter(None, [
        document.title, document.description, document.tags, document.doc_type,
        document.domain.name if document.domain else "",
        document.category.name if document.category else "",
        document.product.name if document.product else "",
        document.asset.name if document.asset else "",
        document.asset.model_number if document.asset else "",
        heading,
    ]))
    return f"{metadata}\n{body}".strip()


def index_document(document):
    text = extract_document_text(document)
    document.chunks.all().delete()
    document.checksum_sha256 = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest() if text else ""
    document.index_status = "INDEXING"
    document.index_version = (document.index_version or 0) + 1
    document.save(update_fields=["checksum_sha256", "index_status", "index_version"])
    if not document.is_rag_enabled or not text.strip():
        document.index_status = "NOT_INDEXED"
        document.save(update_fields=["index_status"])
        from .chroma_store import queue_chroma_sync
        queue_chroma_sync(document.id)
        return 0

    chunks = chunk_markdownish(text)
    objects = []
    for i, (heading, body) in enumerate(chunks):
        flags = detect_document_prompt_injection(body)
        objects.append(KnowledgeChunk(
            tenant=document.tenant,
            document=document,
            chunk_index=i,
            heading=heading,
            text=body,
            embedding=embed_text(_doc_embedding_text(document, heading, body)),
            content_embedding=embed_text(body),
            embedding_model=VECTOR_MODEL,
            safety_flags=flags,
            is_quarantined=bool(flags),
        ))
    KnowledgeChunk.objects.bulk_create(objects)
    from .chroma_store import queue_chroma_sync
    queue_chroma_sync(document.id)
    return len(objects)


def _redact_resolution_text(text, call):
    text = text or ""
    # Remove direct identifiers before resolved cases become reusable RAG memory.
    # This permits staff to learn from similar cases without exposing another
    # customer's contact details.
    literals = [
        getattr(call.customer, "name", "") if call.customer_id else "",
        call.contact_name, call.contact_phone, call.contact_email,
        getattr(call.customer, "contact_name", "") if call.customer_id else "",
        getattr(call.customer, "phone", "") if call.customer_id else "",
        getattr(call.customer, "email", "") if call.customer_id else "",
    ]
    for value in sorted({str(v).strip() for v in literals if str(v or "").strip()}, key=len, reverse=True):
        text = re.sub(re.escape(value), "[REDACTED]", text, flags=re.I)
    text = re.sub(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "[REDACTED_EMAIL]", text, flags=re.I)
    text = re.sub(r"(?<![A-Za-z0-9])\+?\d[\d\s().-]{7,}\d(?![A-Za-z0-9])", "[REDACTED_PHONE]", text)
    return text


def resolution_text_for_call(call):
    updates = "\n".join(u.note for u in call.updates.all()[:30])
    raw = "\n".join(filter(None, [
        f"Complaint type: {call.complaint_type}",
        f"Complaint: {call.complaint_text}",
        f"Product: {call.asset.product.name if call.asset and call.asset.product else ''}",
        f"Asset model: {call.asset.model_number if call.asset else ''}",
        f"Technician notes: {call.technician_notes}",
        f"Resolution: {call.resolution_text}",
        f"Updates: {updates}",
    ])).strip()
    return _redact_resolution_text(raw, call)


def index_service_resolution(call):
    if call.status != "closed" or not call.resolution_text.strip():
        ServiceResolutionIndex.objects.filter(service_call=call).delete()
        return None
    text = resolution_text_for_call(call)
    obj, _ = ServiceResolutionIndex.objects.update_or_create(
        tenant=call.tenant,
        service_call=call,
        defaults={"text": text, "embedding": embed_text(text)},
    )
    return obj


def reindex_all_resolutions(tenant):
    count = 0
    for call in ServiceCall.objects.filter(tenant=tenant, status="closed").exclude(resolution_text=""):
        if index_service_resolution(call):
            count += 1
    return count
