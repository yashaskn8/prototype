"""Required asset / customer / manual bindings for the three-asset customer demo.

One place states which nine documents the demo needs and decides, per asset,
whether they are actually usable. A document counts only when every check
passes: the database record alone is never enough. The function reads; it
never writes, repairs, substitutes or re-indexes anything, and its issue codes
contain no file paths or document text.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

REQUIRED_MANUALS = {
    59: {"code": "SKY-AST-00001", "name": "Split AC Unit 2T", "customer_id": 14,
         "documents": {168: "troubleshooting", 169: "maintenance", 170: "installation"}},
    60: {"code": "SKY-AST-00002", "name": "VRF Outdoor Module", "customer_id": 13,
         "documents": {171: "troubleshooting", 172: "maintenance", 173: "installation"}},
    61: {"code": "SKY-AST-00003", "name": "Ductable AC 5T", "customer_id": 14,
         "documents": {174: "troubleshooting", 175: "maintenance", 176: "installation"}},
}
REQUIRED_DOCUMENT_COUNT = sum(len(row["documents"]) for row in REQUIRED_MANUALS.values())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _registry_by_document():
    from core.services.customer_isolation import VERIFIED_REAL_PROVENANCE_REGISTRY
    out = {}
    for checksum, entry in VERIFIED_REAL_PROVENANCE_REGISTRY["documents"].items():
        out[entry.get("local_doc_id")] = {**entry, "registered_checksum": checksum}
    return out


def document_issues(doc, *, asset_id, customer_id, doc_type, registry_entry, media_root,
                    searchable_chunks, is_verified_document):
    """Every reason one required manual cannot be used. Empty list means ready."""
    issues = []
    if doc is None:
        return ["DOCUMENT_RECORD_MISSING"]
    if getattr(doc, "asset_id", None) != asset_id:
        issues.append("NOT_BOUND_TO_REQUIRED_ASSET")
    if getattr(doc, "customer_id", None) not in (None, customer_id):
        issues.append("BOUND_TO_ANOTHER_CUSTOMER")
    if (not getattr(doc, "is_rag_enabled", False) or getattr(doc, "is_confidential", False)
            or getattr(doc, "disable_sharing", False)):
        issues.append("NOT_ELIGIBLE_FOR_CUSTOMER_RAG")
    if not is_verified_document:
        issues.append("PROVENANCE_NOT_VERIFIED")
    if registry_entry is None:
        issues.append("NOT_IN_PROVENANCE_REGISTRY")
    else:
        if registry_entry.get("local_asset_id") != asset_id:
            issues.append("REGISTRY_ASSET_MISMATCH")
        if registry_entry.get("doc_type") != doc_type:
            issues.append("DOCUMENT_TYPE_MISMATCH")
    recorded = str(getattr(doc, "checksum_sha256", "") or "").casefold()
    file_field = getattr(doc, "file", None)
    if not file_field or not recorded:
        issues.append("ATTACHMENT_OR_CHECKSUM_MISSING")
    else:
        try:
            path = Path(file_field.path).resolve()
            inside = path.is_relative_to(Path(media_root).resolve()) and path.is_file()
        except (OSError, ValueError):
            inside = False
        if not inside:
            issues.append("ATTACHMENT_FILE_UNAVAILABLE")
        else:
            digest = _sha256_file(path).casefold()
            if digest != recorded:
                issues.append("ATTACHMENT_CHECKSUM_MISMATCH")
            if registry_entry is not None and digest != str(
                    registry_entry.get("registered_checksum", "")).casefold():
                issues.append("REGISTRY_CHECKSUM_MISMATCH")
    if searchable_chunks < 1:
        issues.append("NO_SEARCHABLE_CHUNKS")
    return issues


def manual_readiness(tenant):
    """Readiness of each required asset, judged from all three of its manuals."""
    from django.conf import settings
    from core.models import Asset, KnowledgeChunk, KnowledgeDocument
    from core.services.customer_isolation import (
        is_verified_real_asset, is_verified_real_customer, is_verified_real_document,
    )
    registry = _registry_by_document()
    report = {}
    for asset_id, required in REQUIRED_MANUALS.items():
        asset_issues = []
        asset = Asset.objects.select_related("tenant", "customer", "product", "site").filter(
            pk=asset_id, tenant=tenant).first()
        if asset is None:
            asset_issues.append("ASSET_RECORD_MISSING")
        else:
            if asset.customer_id != required["customer_id"]:
                asset_issues.append("ASSET_CUSTOMER_BINDING_MISMATCH")
            if asset.asset_code != required["code"]:
                asset_issues.append("ASSET_CODE_MISMATCH")
            if not is_verified_real_asset(asset) or not asset.customer \
                    or not is_verified_real_customer(asset.customer):
                asset_issues.append("ASSET_OR_CUSTOMER_NOT_VERIFIED")
        documents = []
        for doc_id, doc_type in required["documents"].items():
            doc = KnowledgeDocument.objects.select_related(
                "tenant", "asset", "customer", "product", "category", "domain",
            ).filter(pk=doc_id, tenant=tenant).first()
            chunks = 0
            if doc is not None:
                chunks = sum(
                    1 for text in KnowledgeChunk.objects.filter(
                        document=doc, tenant=tenant, is_quarantined=False,
                    ).values_list("text", flat=True)
                    if isinstance(text, str) and text.strip())
            issues = document_issues(
                doc, asset_id=asset_id, customer_id=required["customer_id"],
                doc_type=doc_type, registry_entry=registry.get(doc_id),
                media_root=settings.MEDIA_ROOT, searchable_chunks=chunks,
                is_verified_document=bool(doc is not None and is_verified_real_document(doc)),
            )
            documents.append({
                "id": doc_id, "doc_type": doc_type,
                "title": getattr(doc, "title", "") if doc is not None else "",
                "searchable_chunks": chunks, "ready": not issues, "issues": issues,
            })
        report[asset_id] = {
            "asset_id": asset_id, "name": required["name"],
            "customer_id": required["customer_id"],
            "ready": not asset_issues and all(d["ready"] for d in documents),
            "asset_issues": asset_issues, "documents": documents,
            "incomplete_documents": [d["id"] for d in documents if not d["ready"]],
        }
    return report


def readiness_failures(report):
    """Human-readable, path-free reasons the three-asset demo is not ready."""
    lines = []
    for asset_id in REQUIRED_MANUALS:
        row = (report or {}).get(asset_id)
        if row is None:
            lines.append(f"asset {asset_id}: readiness was not evaluated")
            continue
        for issue in row["asset_issues"]:
            lines.append(f"asset {asset_id}: {issue}")
        for document in row["documents"]:
            if not document["ready"]:
                lines.append(f"asset {asset_id} document {document['id']} "
                             f"({document['doc_type']}): {', '.join(document['issues'])}")
    return lines
