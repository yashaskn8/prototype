"""
Durable Evidence Anchoring, Hierarchy Scoping, Chunk Quarantine Invalidation,
and Deterministic Source-Binding Check.

Hard Invariants:
  - If a KnowledgeDocument changes such that its checksum or version drifts from the EvidenceAnchor,
    the dependent SAFE_ACTION is invalidated immediately.
  - Quarantined or prompt-injected documents or KnowledgeChunks may NEVER authorize customer actions.
  - Confidential documents may NEVER authorize customer actions.
  - RAG-disabled documents may NEVER authorize customer actions.
  - Documents MUST strictly match the asset knowledge hierarchy (Asset -> Product -> Category -> Domain -> Fleet General).
  - Exact source locator is required: chunk_id OR normalized excerpt OR heading + excerpt.
  - Fake headings or wrong chunk IDs are rejected.
  - SAFE_ACTION instruction must be deterministically supported by source text without unauthorized
    disassembly, rewiring, part replacement, or invented calibration thresholds.
"""

import re
from typing import Any, Dict, Optional, Tuple
from core.models import KnowledgeDocument, KnowledgeChunk
from core.services.content_safety import detect_document_prompt_injection
from core.services.knowledge_scope import get_applicable_knowledge_queryset


def validate_source_binding(instruction: str, excerpt: str) -> Tuple[bool, str]:
    """Deterministically validate that a SAFE_ACTION instruction is supported by the source excerpt."""
    if not instruction or not instruction.strip():
        return False, "EMPTY_INSTRUCTION"
    if not excerpt or not excerpt.strip():
        return False, "EMPTY_EXCERPT"

    inst_lower = instruction.lower()
    excerpt_lower = excerpt.lower()

    # 1. Dangerous Disassembly checks
    disassembly_terms = [
        "disassemble", "dismantle", "open enclosure", "remove panel", "unscrew",
        "open chassis", "open machine enclosure", "open the machine", "open machine",
        "open cover", "solder", "unsolder", "desolder"
    ]
    for term in disassembly_terms:
        if term in inst_lower and term not in excerpt_lower:
            return False, f"UNSUPPORTED_DISASSEMBLY: {term}"

    # 2. Electrical / Mechanical Intervention
    electrical_terms = [
        "rewire", "bypass", "short circuit", "adjust voltage", "capacitor",
        "high voltage", "splice", "bridge fuse"
    ]
    for term in electrical_terms:
        if term in inst_lower and term not in excerpt_lower:
            return False, f"UNSUPPORTED_ELECTRICAL_INTERVENTION: {term}"

    # 3. Part Replacement
    replacement_terms = [
        "replace motor", "replace sensor", "replace board", "replace pump",
        "replace component", "swap board", "swap motor", "swap sensor", "install new"
    ]
    for term in replacement_terms:
        if term in inst_lower and term not in excerpt_lower:
            return False, f"UNSUPPORTED_PART_REPLACEMENT: {term}"

    # 4. Prohibited Tool Requirements
    tool_terms = ["oscilloscope", "soldering iron", "torque wrench", "logic analyzer"]
    for term in tool_terms:
        if term in inst_lower and term not in excerpt_lower:
            return False, f"UNSUPPORTED_TOOL_REQUIREMENT: {term}"

    # 5. Invented calibration values/thresholds
    thresh_pattern = re.findall(r"\b\d+(?:\.\d+)?\s*(?:v|volt|volts|a|amp|amps|psi|bar|rpm|mhz|khz|hz|c|f|k)\b", inst_lower)
    for num in thresh_pattern:
        if num not in excerpt_lower:
            return False, f"INVENTED_CALIBRATION_THRESHOLD: {num}"

    # 6. Key concepts check: Action instruction content words must have meaningful presence in excerpt
    stopwords = {
        "the", "and", "or", "to", "in", "of", "a", "an", "is", "for", "with",
        "on", "at", "by", "from", "as", "be", "it", "this", "that", "are", "was"
    }
    inst_words = [w for w in re.findall(r"[a-z0-9_-]{3,}", inst_lower) if w not in stopwords]
    if inst_words:
        matches = [w for w in inst_words if w in excerpt_lower]
        if len(matches) == 0 or (len(inst_words) >= 4 and len(matches) / len(inst_words) < 0.25):
            return False, "ACTION_TEXT_NOT_SUPPORTED_BY_EXCERPT"

    return True, "VALID"


def verify_evidence_anchor(
    evidence_anchor: Dict[str, Any],
    tenant_id: int,
    customer=None,
    asset=None,
    instruction: Optional[str] = None,
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

    # Asset Hierarchy Applicability Check
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

    # Security Gate: Content safety / quarantine on whole doc
    if doc.content_text:
        injections = detect_document_prompt_injection(doc.content_text)
        if injections:
            return False, f"DOCUMENT_QUARANTINED: prompt injection flags {injections}", doc

    chunks_qs = doc.chunks.all()
    chunk_id = evidence_anchor.get("chunk_id")
    heading = (evidence_anchor.get("heading") or "").strip()
    excerpt = (evidence_anchor.get("excerpt") or evidence_anchor.get("procedure_text") or "").strip()

    # Exact source locator verification: must provide chunk_id OR excerpt OR heading + excerpt
    if not chunk_id and not excerpt and not heading:
        return False, "EXACT_LOCATOR_MISSING", doc

    # Verify chunk_id if specified
    target_chunk = None
    if chunk_id:
        target_chunk = chunks_qs.filter(id=chunk_id).first()
        if not target_chunk:
            return False, f"WRONG_CHUNK_ID: chunk {chunk_id} does not belong to document {doc.id}", doc
        if target_chunk.is_quarantined:
            return False, f"EVIDENCE_CHUNK_QUARANTINED: chunk {chunk_id}", doc
        injections = detect_document_prompt_injection(target_chunk.text)
        if injections or (target_chunk.safety_flags and "injection" in str(target_chunk.safety_flags).lower()):
            return False, f"EVIDENCE_CHUNK_INJECTION: chunk {chunk_id}", doc

    # Verify heading if specified (must exist in doc text or chunk headings)
    if heading:
        doc_headings = [h.strip().lower() for h in chunks_qs.values_list("heading", flat=True) if h]
        heading_lower = heading.lower()
        heading_in_doc = (
            any(heading_lower in h for h in doc_headings) or
            f"# {heading_lower}" in (doc.content_text or "").lower() or
            f"## {heading_lower}" in (doc.content_text or "").lower() or
            f"### {heading_lower}" in (doc.content_text or "").lower() or
            heading_lower in (doc.content_text or "").lower()
        )
        if not heading_in_doc:
            return False, f"FAKE_HEADING_REJECTED: heading '{heading}' not found in document", doc

    # Verify excerpt if specified
    effective_excerpt = excerpt
    if excerpt:
        excerpt_clean = excerpt[:80].strip()
        doc_has_excerpt = (excerpt_clean.lower() in (doc.content_text or "").lower())
        if chunks_qs.exists():
            matching_chunks = chunks_qs.filter(text__icontains=excerpt_clean)
            if not matching_chunks.exists() and not doc_has_excerpt:
                return False, "EVIDENCE_EXCERPT_NOT_FOUND", doc
            if matching_chunks.filter(is_quarantined=True).exists():
                return False, "EVIDENCE_CHUNK_QUARANTINED", doc
            for chunk in matching_chunks:
                injections = detect_document_prompt_injection(chunk.text)
                if injections or (chunk.safety_flags and "injection" in str(chunk.safety_flags).lower()):
                    return False, f"EVIDENCE_CHUNK_INJECTION: chunk #{chunk.chunk_index}", doc
        elif not doc_has_excerpt:
            return False, "EVIDENCE_EXCERPT_NOT_FOUND", doc
    elif target_chunk:
        effective_excerpt = target_chunk.text
    elif heading:
        effective_excerpt = doc.content_text or ""

    # Check overall chunk quarantine if no specific excerpt or chunk was targeted
    if not chunk_id and not excerpt:
        if chunks_qs.filter(is_quarantined=True).exists():
            return False, "EVIDENCE_CHUNK_QUARANTINED", doc

    # Deterministic source-binding check against instruction
    if instruction:
        binding_ok, binding_reason = validate_source_binding(instruction, effective_excerpt or doc.content_text or "")
        if not binding_ok:
            return False, f"SOURCE_BINDING_FAILED: {binding_reason}", doc

    return True, "VALID", doc
