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


def index_document(document, *, authorized: bool = False):
    from django.conf import settings
    # Protect verified/company documents from unauthorized destructive chunk deletion
    real_tenants = getattr(settings, "STARLLY_REAL_TENANTS", ["skydream"])
    is_protected_tenant = bool(document.tenant and document.tenant.slug in real_tenants)
    has_verified_provenance = bool(
        hasattr(document, "source_provenance")
        and getattr(document.source_provenance, "verification_status", None) == "VERIFIED"
    )
    if (is_protected_tenant or has_verified_provenance) and not authorized:
        if document.chunks.exists():
            raise PermissionError(
                f"Unauthorized reindex operation on protected document {document.id} ({document.title}). "
                "Historical vector identity and knowledge chunks are protected."
            )

    from django.db import transaction
    from django.utils import timezone as _tz
    from .chroma_store import queue_chroma_sync

    # Stage 1 (no destructive effects): extract, chunk, and embed. Any failure
    # here propagates with existing chunks untouched.
    text = extract_document_text(document)
    new_checksum = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest() if text else ""
    index_enabled = bool(document.is_rag_enabled and text.strip())
    objects = []
    if index_enabled:
        for i, (heading, body) in enumerate(chunk_markdownish(text)):
            flags = detect_document_prompt_injection(body)
            full_text = f"## {heading}\n{body}" if heading and heading not in body else body
            content_embedding_text = f"{heading}\n{body}" if heading and heading not in body else body
            objects.append(KnowledgeChunk(
                tenant=document.tenant,
                document=document,
                chunk_index=i,
                heading=heading,
                text=full_text,
                embedding=embed_text(_doc_embedding_text(document, heading, full_text)),
                content_embedding=embed_text(content_embedding_text),
                embedding_model=VECTOR_MODEL,
                safety_flags=flags,
                is_quarantined=bool(flags),
            ))

    # Stage 2: atomic swap. Either all old chunks are replaced or none are.
    with transaction.atomic():
        document.chunks.all().delete()
        if objects:
            KnowledgeChunk.objects.bulk_create(objects)
        document.checksum_sha256 = new_checksum
        document.index_version = (document.index_version or 0) + 1
        if index_enabled:
            # INDEXED immediately so the sparse/fallback retriever can find
            # these chunks even before Chroma sync completes.
            document.index_status = "INDEXED"
            document.indexed_at = _tz.now()
            document.save(update_fields=["checksum_sha256", "index_status", "index_version", "indexed_at"])
        else:
            document.index_status = "NOT_INDEXED"
            document.save(update_fields=["checksum_sha256", "index_status", "index_version"])
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
