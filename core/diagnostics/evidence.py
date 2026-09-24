"""
Durable Evidence Anchoring and Document Drift Invalidation.

Invariants:
  - If a KnowledgeDocument changes such that its checksum or version drifts from the EvidenceAnchor,
    the dependent SAFE_ACTION is invalidated immediately.
  - Quarantined or prompt-injected documents may NEVER authorize customer actions.
  - Confidential documents may NEVER authorize customer actions.
  - RAG-disabled documents may NEVER authorize customer actions.
"""

from typing import Any, Dict, Optional, Tuple
from core.models import KnowledgeDocument
from core.services.content_safety import detect_document_prompt_injection


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

    doc = KnowledgeDocument.objects.filter(id=doc_id, tenant_id=tenant_id).first()
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

    # Security Gate: Content safety / quarantine
    if doc.content_text:
        injections = detect_document_prompt_injection(doc.content_text)
        if injections:
            return False, f"DOCUMENT_QUARANTINED: prompt injection flags {injections}", doc

    # Customer isolation scope
    if doc.customer_id and customer and doc.customer_id != customer.id:
        return False, "DOCUMENT_CROSS_CUSTOMER", doc

    # Asset isolation scope
    if doc.asset_id and asset and doc.asset_id != asset.id:
        return False, "DOCUMENT_CROSS_ASSET", doc

    return True, "VALID", doc
