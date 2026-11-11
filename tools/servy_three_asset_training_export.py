"""Export the nine approved manuals to a private SME-review workspace.

This is a read-only company database export. It performs no model training,
company-record writes, index mutation, cloud upload, or git write. Source text
and the review template are saved only under the current Windows user's
LOCALAPPDATA/Servy/private_evaluations directory with restrictive ACLs.

The CSV is intentionally a review template, not a training dataset. Every
proposed query remains pending until an authorized reviewer confirms the
question, positive passage, hard negatives, answerability, and split.
"""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = {
    59: {"code": "SKY-AST-00001", "name": "Split AC Unit 2T", "customer_id": 14},
    60: {"code": "SKY-AST-00002", "name": "VRF Outdoor Module", "customer_id": 13},
    61: {"code": "SKY-AST-00003", "name": "Ductable AC 5T", "customer_id": 14},
}
DOCUMENTS = {
    168: (59, "troubleshooting"),
    169: (59, "maintenance"),
    170: (59, "installation"),
    171: (60, "troubleshooting"),
    172: (60, "maintenance"),
    173: (60, "installation"),
    174: (61, "troubleshooting"),
    175: (61, "maintenance"),
    176: (61, "installation"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export():
    from tools.servy_100_readonly_eval import configure_process, verify_readonly_identity
    configure_process()
    database_identity = verify_readonly_identity()

    import django
    django.setup()

    from django.conf import settings
    from core.models import Asset, KnowledgeDocument, KnowledgeChunk, Tenant
    from core.services.customer_isolation import (
        VERIFIED_REAL_PROVENANCE_REGISTRY,
        is_verified_real_asset,
        is_verified_real_customer,
        is_verified_real_document,
    )
    from tools.servy_500_benchmark import private_folder, read_only_database

    tenant = Tenant.objects.get(slug="skydream", is_active=True)
    if tenant.id != 9:
        raise RuntimeError("Approved tenant identity changed.")

    expected_registry = VERIFIED_REAL_PROVENANCE_REGISTRY["documents"]
    registered = {}
    for digest, entry in expected_registry.items():
        if entry.get("local_asset_id") in ASSETS:
            registered[int(entry["local_doc_id"])] = {
                **entry, "registered_checksum": digest,
            }
    if set(registered) != set(DOCUMENTS):
        raise RuntimeError("The provenance registry no longer contains exactly the approved nine documents.")

    media_root = Path(settings.MEDIA_ROOT).resolve()
    corpus_rows = []
    review_rows = []
    doc_manifest = {}
    counts_before = None
    counts_after = None

    with read_only_database():
        from tools.servy_500_benchmark import operational_counts
        counts_before = operational_counts()

        for asset_id, asset_info in ASSETS.items():
            asset = Asset.objects.select_related(
                "tenant", "customer", "product", "site"
            ).get(pk=asset_id, tenant=tenant)
            if (
                asset.asset_code != asset_info["code"]
                or asset.name != asset_info["name"]
                or asset.customer_id != asset_info["customer_id"]
                or not is_verified_real_asset(asset)
                or not is_verified_real_customer(asset.customer)
            ):
                raise RuntimeError(f"Asset/customer provenance failed for {asset_info['code']}.")

        for document_id, (asset_id, doc_type) in DOCUMENTS.items():
            asset_info = ASSETS[asset_id]
            doc = KnowledgeDocument.objects.select_related(
                "tenant", "asset", "customer", "product", "category", "domain"
            ).get(pk=document_id, tenant=tenant)
            if (
                doc.asset_id != asset_id
                or doc.customer_id not in (None, asset_info["customer_id"])
                or not doc.file
                or not doc.checksum_sha256
                or not doc.is_rag_enabled
                or doc.is_confidential
                or doc.disable_sharing
                or not is_verified_real_document(doc)
            ):
                raise RuntimeError(f"Document provenance/visibility failed for document {document_id}.")

            attachment = Path(doc.file.path).resolve()
            if not attachment.is_relative_to(media_root) or not attachment.is_file():
                raise RuntimeError(f"Document {document_id} is outside the approved media root.")
            attachment_hash = sha256_file(attachment)
            if (
                attachment_hash.casefold() != doc.checksum_sha256.casefold()
                or attachment_hash.casefold() != registered[document_id]["registered_checksum"].casefold()
            ):
                raise RuntimeError(f"Attachment checksum failed for document {document_id}.")

            chunks = list(
                KnowledgeChunk.objects.filter(
                    document=doc,
                    tenant=tenant,
                    is_quarantined=False,
                ).order_by("chunk_index", "id")
            )
            if not chunks:
                raise RuntimeError(f"Approved document {document_id} has no eligible chunks.")

            source_family = f"skydream-asset-{asset_id}-doc-{document_id}"
            doc_manifest[str(document_id)] = {
                "asset_id": asset_id,
                "asset_code": asset_info["code"],
                "asset_name": asset_info["name"],
                "document_type": doc_type,
                "title": doc.title,
                "index_status": doc.index_status,
                "index_version": doc.index_version,
                "attachment_sha256": attachment_hash,
                "chunk_count": len(chunks),
                "source_family": source_family,
            }
            for chunk in chunks:
                text = (chunk.text or "").strip()
                if not text:
                    continue
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                row = {
                    "tenant_id": tenant.id,
                    "asset_id": asset_id,
                    "asset_code": asset_info["code"],
                    "equipment_family": asset_info["name"],
                    "model_number": asset.model_number or "",
                    "document_id": document_id,
                    "document_type": doc_type,
                    "document_title": doc.title,
                    "chunk_id": chunk.id,
                    "chunk_index": chunk.chunk_index,
                    "heading": chunk.heading or "",
                    "passage": text,
                    "passage_sha256": text_hash,
                    "attachment_sha256": attachment_hash,
                    "source_family": source_family,
                    "is_confidential": False,
                    "review_status": "pending_sme_review",
                }
                corpus_rows.append(row)
                review_rows.append({
                    "query_id": f"REVIEW-A{asset_id}-D{document_id}-C{chunk.id}",
                    "tenant_id": tenant.id,
                    "equipment_family": asset_info["name"],
                    "model_number": asset.model_number or "",
                    "query_type": "",
                    "query_text": "",
                    "positive_chunk_ids": json.dumps([chunk.id]),
                    "positive_doc_ids": json.dumps([document_id]),
                    "hard_negative_chunk_ids": "",
                    "source_family": source_family,
                    "expected_abstain": "",
                    "review_status": "pending_sme_review",
                    "reviewer": "",
                    "split": "unassigned",
                    "document_type": doc_type,
                    "document_title": doc.title,
                    "heading": chunk.heading or "",
                    "source_passage_sha256": text_hash,
                    "source_attachment_sha256": attachment_hash,
                    "sme_notes": "",
                })

        counts_after = operational_counts()

    if counts_before != counts_after:
        raise RuntimeError("Operational record counts changed during the read-only export.")
    if len(doc_manifest) != 9 or len({row["chunk_id"] for row in corpus_rows}) != len(corpus_rows):
        raise RuntimeError("Nine-document export coverage or chunk identity is incomplete.")

    base = Path(os.environ["LOCALAPPDATA"]) / "Servy" / "private_evaluations"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    from tools.servy_500_benchmark import private_folder
    out_dir = private_folder(base / f"servy_three_asset_sme_review_{stamp}")

    corpus_path = out_dir / "approved_source_chunks.jsonl"
    with corpus_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in corpus_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    review_path = out_dir / "sme_review_template.csv"
    fieldnames = list(review_rows[0].keys()) if review_rows else []
    with review_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)

    manifest = {
        "export_version": "three-asset-nine-approved-documents-v1",
        "database_identity": database_identity,
        "approved_assets": ASSETS,
        "approved_documents": doc_manifest,
        "document_count": len(doc_manifest),
        "chunk_count": len(corpus_rows),
        "operational_counts_before": counts_before,
        "operational_counts_after": counts_after,
        "operational_counts_unchanged": counts_before == counts_after,
        "corpus_file_sha256": sha256_file(corpus_path),
        "review_template_sha256": sha256_file(review_path),
        "training_status": "NOT_TRAINING_INPUT_PENDING_SME_REVIEW",
        "review_requirements": [
            "Enter a real natural-language query for the selected equipment.",
            "Confirm the positive chunk truly answers that query.",
            "Review hard negatives and exclude any valid alternate positive.",
            "Set query_type and expected_abstain.",
            "Provide authorized reviewer identity and review reference.",
            "Assign source-family-disjoint train/dev/independent-test splits.",
        ],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "output_directory": str(out_dir),
        "documents": len(doc_manifest),
        "chunks_exported": len(corpus_rows),
        "assets": len(ASSETS),
        "operational_counts_unchanged": counts_before == counts_after,
        "training_status": manifest["training_status"],
        "corpus_sha256": manifest["corpus_file_sha256"],
        "review_template": str(review_path),
    }, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    export()
