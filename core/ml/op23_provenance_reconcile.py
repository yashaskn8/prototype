"""Read-only proposal for authoritative MySQL provenance reconciliation."""
from __future__ import annotations

from core.ml.op23_source_admission import sha256


def reconcile_document(source: dict, asset, document, source_text: str, existing=None, *, expected_org: str) -> dict:
    """Return a proposed mapping or a conflict. Never saves ORM objects."""
    reasons = []
    if source.get("source_verification") != "VERIFIED_ORIGINAL_COMPANY_SOURCE":
        reasons.append("source_not_byte_verified")
    if str(source.get("source_organisation")) != str(expected_org):
        reasons.append("source_organisation_mismatch")
    if source.get("tenant_id") != asset.tenant_id or document.tenant_id != asset.tenant_id:
        reasons.append("tenant_mismatch")
    if source.get("asset_code") != asset.asset_code or source.get("external_asset_id") is None:
        reasons.append("asset_identity_mismatch")
    if (source.get("equipment_model") or "") != (asset.model_number or ""):
        reasons.append("equipment_model_mismatch")
    if document.asset_id != asset.id or document.checksum_sha256 != source.get("raw_sha256"):
        reasons.append("document_asset_or_raw_hash_mismatch")
    if sha256((document.content_text or "").encode()) != source.get("text_sha256"):
        reasons.append("document_extracted_text_mismatch")
    if sha256(source_text.encode()) != source.get("text_sha256"):
        reasons.append("source_text_file_mismatch")
    chunk_links = []
    for chunk in document.chunks.all():
        present = bool(chunk.text.strip() and chunk.text.strip() in source_text)
        chunk_links.append({"chunk_id": chunk.id, "source_text_contains_chunk": present})
        if not present:
            reasons.append("chunk_outside_verified_source")
    if not chunk_links:
        reasons.append("document_has_no_chunks")
    external_id = str(source.get("external_attachment_id"))
    if existing is not None and (
        existing.tenant_id != asset.tenant_id or existing.source_org != str(expected_org)
        or existing.external_object_type != "document" or existing.external_id != external_id
        or existing.local_object_type != "KnowledgeDocument" or existing.local_object_id != document.id
        or existing.raw_attachment_sha256 != source.get("raw_sha256")
    ):
        reasons.append("existing_provenance_conflicts")
    proposal = {"tenant_id": asset.tenant_id, "source_system": "starlly",
                "source_org": str(expected_org), "external_object_type": "document",
                "external_id": external_id, "local_object_type": "KnowledgeDocument",
                "local_object_id": document.id, "asset_code": asset.asset_code,
                "raw_attachment_sha256": source.get("raw_sha256"),
                "extracted_text_sha256": source.get("text_sha256"),
                "document_version": document.version, "external_asset_id": source.get("external_asset_id")}
    status = "CONFLICT" if reasons else ("EXISTING_MATCH" if existing else "WOULD_INSERT_PENDING_AUTHORIZED_WRITE")
    return {"status": status, "reasons": sorted(set(reasons)), "proposed_mapping": proposal,
            "chunks": chunk_links}
