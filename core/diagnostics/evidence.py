"""
Durable Evidence Anchoring, Hierarchy Scoping, and Chunk Quarantine Invalidation.

Invariants:
  - If a KnowledgeDocument changes such that its checksum or version drifts from the EvidenceAnchor,
    the dependent SAFE_ACTION is invalidated immediately.
  - Quarantined or prompt-injected documents or KnowledgeChunks may NEVER authorize customer actions.
  - Confidential documents may NEVER authorize customer actions.
  - RAG-disabled documents may NEVER authorize customer actions.
  - Documents MUST strictly match the asset knowledge hierarchy (Asset -> Product -> Category -> Domain -> Fleet General).
"""

from typing import Any, Dict, Optional, Tuple
from core.models import KnowledgeDocument, KnowledgeChunk
from core.services.content_safety import detect_document_prompt_injection
from core.services.knowledge_scope import get_applicable_knowledge_queryset


def verify_evidence_anchor(
    evidence_anchor: Dict[str, Any],
    tenant_id: int,
    customer=None,
    asset=None
) -> Tuple[bool, str, Optional[KnowledgeDocument]]:
    """Verify an evidence anchor against the active database state.

    Returns (is_valid, failure_reason_code, document_instance).
    """
    if not evidence_anchor or not isinstance(evidence_anchor, dict):
        return False, "INVALID_ANCHOR_PAYLOAD", None

    doc_id = evidence_anchor.get("document_id")
    expected_checksum = evidence_anchor.get("checksum_sha256")
    expected_version = evidence_anchor.get("version")

    if not doc_id:
        return False, "MISSING_DOCUMENT_ID", None

    doc = KnowledgeDocument.objects.filter(
        id=doc_id, tenant_id=tenant_id
    ).select_related(
        "product", "product__category", "product__category__domain", "asset", "customer"
    ).first()

    if not doc:
        return False, "DOCUMENT_NOT_FOUND", None

    # Customer confidentiality gate
    if doc.is_confidential:
        return False, "DOCUMENT_CONFIDENTIAL", doc

    # RAG enabled gate
    if not doc.is_rag_enabled:
        return False, "DOCUMENT_RAG_DISABLED", doc

    # Drift Detection 1: Checksum verification
    if expected_checksum and doc.checksum_sha256 != expected_checksum:
        return False, f"DOCUMENT_DRIFT_CHECKSUM: expected {expected_checksum}, found {doc.checksum_sha256}", doc

    # Drift Detection 2: Version verification
    if expected_version and doc.version != expected_version:
        return False, f"DOCUMENT_DRIFT_VERSION: expected {expected_version}, found {doc.version}", doc

    # Customer isolation scope
    if doc.customer_id and customer and doc.customer_id != customer.id:
        return False, "DOCUMENT_CROSS_CUSTOMER", doc

    # Asset Hierarchy Applicability Check (Critical Defect 4)
    if asset:
        ctx = {
            "tenant": doc.tenant,
            "role": "customer",
            "customer": customer,
            "can_view_confidential": False,
        }
        applicable_qs = get_applicable_knowledge_queryset(asset, ctx)
        if not applicable_qs.filter(id=doc.id).exists():
            return False, "DOCUMENT_SCOPE_INAPPLICABLE", doc
    else:
        if doc.asset_id:
            return False, "DOCUMENT_SCOPE_ASSET_SPECIFIC", doc

    # Security Gate: Content safety / quarantine on whole doc (Critical Defect 5)
    if doc.content_text:
        injections = detect_document_prompt_injection(doc.content_text)
        if injections:
            return False, f"DOCUMENT_QUARANTINED: prompt injection flags {injections}", doc

    # Security Gate: Chunk-level Quarantine and Injection Check (Critical Defect 5)
    excerpt = (evidence_anchor.get("excerpt") or evidence_anchor.get("procedure_text") or "").strip()
    chunks_qs = doc.chunks.all()

    if chunks_qs.exists():
        if excerpt:
            # Check chunks containing the excerpt
            matching_chunks = chunks_qs.filter(text__icontains=excerpt[:80])
            if not matching_chunks.exists():
                return False, "EVIDENCE_EXCERPT_NOT_FOUND", doc
            if matching_chunks.filter(is_quarantined=True).exists():
                return False, "EVIDENCE_CHUNK_QUARANTINED", doc
            for chunk in matching_chunks:
                injections = detect_document_prompt_injection(chunk.text)
                if injections or (chunk.safety_flags and "injection" in str(chunk.safety_flags).lower()):
                    return False, f"EVIDENCE_CHUNK_INJECTION: chunk #{chunk.chunk_index}", doc
        else:
            # If no specific excerpt, any quarantined chunk in the doc blocks authority
            if chunks_qs.filter(is_quarantined=True).exists():
                return False, "EVIDENCE_CHUNK_QUARANTINED", doc
            for chunk in chunks_qs:
                injections = detect_document_prompt_injection(chunk.text)
                if injections or (chunk.safety_flags and "injection" in str(chunk.safety_flags).lower()):
                    return False, f"EVIDENCE_CHUNK_INJECTION: chunk #{chunk.chunk_index}", doc

    return True, "VALID", doc
