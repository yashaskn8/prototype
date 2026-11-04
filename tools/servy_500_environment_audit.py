"""Aggregate-only, read-only audit before resuming the Servy 500 baseline."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.servy_500_benchmark import (AUTHORIZED_ASSETS, load_json, private_folder,
                                       read_only_database, serving, write_json)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit(previous, current, external_preview=False):
    from django.conf import settings
    from core.models import Tenant, Asset, KnowledgeDocument, KnowledgeChunk, SourceProvenanceRecord
    from core.services.customer_isolation import is_verified_real_document
    from tools.servy_500_benchmark import operational_counts

    old_freeze = load_json(previous / "freeze.json")
    old_integrity = load_json(previous / "run_integrity.json")
    old_results = load_json(previous / "servy_100_actual_answers.json")
    new_freeze = load_json(current / "freeze.json")
    new_results = load_json(current / "servy_500_baseline_actual_test_results.json")
    old_questions = load_json(previous / "servy_100_hard_questions.json")
    new_questions = load_json(current / "servy_500_corrected_questions_v1.json")["questions"]
    old_ids = {r["id"] for r in old_questions}
    new_ids = {r["id"] for r in new_questions}
    old_sources = load_json(previous / "authorized_source_snapshot.json")
    new_sources = load_json(current / "authorized_source_snapshot.json")
    tenant = Tenant.objects.get(slug="skydream")
    doc_rows = list(KnowledgeDocument.objects.filter(tenant=tenant).select_related(
        "tenant", "asset", "product", "category", "domain", "customer"))
    asset_audit = {}
    for asset_id in sorted(AUTHORIZED_ASSETS):
        asset = Asset.objects.filter(tenant=tenant, id=asset_id).first()
        direct = [d for d in doc_rows if d.asset_id == asset_id]
        verified = [d for d in direct if is_verified_real_document(d)]
        asset_audit[str(asset_id)] = {
            "exists": bool(asset), "customer_bound": bool(asset and asset.customer_id),
            "customer_id": asset.customer_id if asset else None,
            "site_bound": bool(asset and asset.site_id),
            "verified_asset_provenance_records": SourceProvenanceRecord.objects.filter(
                tenant=tenant, local_object_type="Asset", local_object_id=asset_id,
                verification_status="VERIFIED").count(),
            "direct_document_rows": len(direct),
            "direct_chunk_rows": sum(KnowledgeChunk.objects.filter(document=d).count() for d in direct),
            "direct_verified_documents": len(verified),
            "direct_document_index_statuses": dict(Counter(d.index_status for d in direct)),
            "direct_document_visibility": {"confidential": sum(d.is_confidential for d in direct),
                                            "sharing_disabled": sum(d.disable_sharing for d in direct),
                                            "rag_disabled": sum(not d.is_rag_enabled for d in direct)},
            "eligible_documents": new_freeze["scope"][str(asset_id)].get("document_count", 0),
            "eligible_chunks": new_freeze["scope"][str(asset_id)].get("chunk_count", 0),
            "binding_status": new_freeze["scope"][str(asset_id)]["status"],
        }
    index_path = Path(settings.CHROMA_PERSIST_DIR)
    index_files = [p for p in index_path.rglob("*") if p.is_file()] if index_path.exists() else []
    index_info = {"path": str(index_path), "file_count": len(index_files),
                  "total_bytes": sum(p.stat().st_size for p in index_files),
                  "latest_mtime_epoch": max((p.stat().st_mtime for p in index_files), default=None),
                  "files_modified_after_previous_freeze": sum(p.stat().st_mtime > old_freeze["created_at"] for p in index_files),
                  "current_manifest_sha256": {str(p.relative_to(index_path)): digest(p) for p in index_files}}
    records = SourceProvenanceRecord.objects.filter(tenant=tenant)
    ingestion = {
        "provenance_statuses": dict(Counter(records.values_list("verification_status", flat=True))),
        "provenance_object_types": dict(Counter(records.values_list("local_object_type", flat=True))),
        "documents_updated_after_previous_freeze": sum(d.updated_at.timestamp() > old_freeze["created_at"] for d in doc_rows),
        "documents_indexed_after_previous_freeze": sum(bool(d.indexed_at and d.indexed_at.timestamp() > old_freeze["created_at"]) for d in doc_rows),
    }
    media_checked = 0
    media_missing = 0
    media_checksum_mismatch = 0
    for doc in doc_rows:
        if not doc.file:
            continue
        media_checked += 1
        path = Path(doc.file.path)
        if not path.is_file():
            media_missing += 1
        elif doc.checksum_sha256 and digest(path).lower() != doc.checksum_sha256.lower():
            media_checksum_mismatch += 1
    ingestion["local_company_documents"] = len(doc_rows)
    ingestion["media_files_checked"] = media_checked
    ingestion["media_files_missing"] = media_missing
    ingestion["media_checksum_mismatch"] = media_checksum_mismatch
    source_60_old = [(r["document_id"], r["chunk_id"], r["sha256"]) for r in old_sources]
    source_60_new = [(r["document_id"], r["chunk_id"], r["sha256"]) for r in new_sources.get("60", [])]
    allowed_chunks = {key: {(r["document_id"], r["chunk_id"]) for r in rows} for key, rows in new_sources.items()}
    capture = []
    for row in new_results:
        raw = row.get("raw") or {}
        formatted = row.get("formatted") or {}
        refs = raw.get("references") or []
        citations = formatted.get("citations") or []
        allowed = allowed_chunks.get(str(row["asset_id"]), set())
        capture.append({"id": row["id"], "status": row["status"], "classification": row["classification"],
                        "raw_engine": raw.get("engine"), "retrieval_backend": raw.get("retrieval_backend"),
                        "reranker_applied": raw.get("reranker_applied"), "reranker_model": raw.get("reranker_model"),
                        "raw_references": len(refs), "formatted_citations": len(citations),
                        "raw_refs_all_in_scope": all((r.get("document_id"), r.get("chunk_id")) in allowed for r in refs),
                        "formatted_response_type": formatted.get("response_type"),
                        "has_raw_answer": bool(raw.get("answer")), "has_formatted_answer": bool(formatted.get("answer"))})
    result = {
        "folder_separation": {"distinct_paths": previous.resolve() != current.resolve(),
                              "old_result_count": len(old_results), "new_result_count": len(new_results),
                              "question_id_overlap": len(old_ids & new_ids),
                              "new_result_duplicate_ids": len(new_results) - len({r["id"] for r in new_results}),
                              "old_run_was_read_only": old_integrity.get("operational_counts_unchanged"),
                              "old_run_counts_equal_current": old_integrity.get("after") == operational_counts(),
                              "asset_60_source_chunk_hashes_equal": source_60_old == source_60_new,
                              "old_snapshot_sha256_valid": old_freeze["sha256"].get("authorized_source_snapshot.json") == digest(previous / "authorized_source_snapshot.json"),
                              "new_snapshot_sha256_valid": new_freeze["input_sha256"].get("authorized_source_snapshot.json") == digest(current / "authorized_source_snapshot.json")},
        "assets": asset_audit, "ingestion": ingestion, "vector_index": index_info,
        "serving": serving(), "sample_capture": capture,
        "limits": ["No earlier full vector-index manifest exists for byte-for-byte historical comparison.",
                   "Document timestamps and one prior source snapshot cannot prove every company file stayed unchanged.",
                   "Mechanical source-scope and response-shape checks do not establish semantic answer/citation correctness."],
    }
    if external_preview:
        # Metadata only. The existing key is tenant-locked; no attachment URLs are followed.
        from core.services.starlly_api import StarllyApiClient
        try:
            client = StarllyApiClient(tenant=tenant)
            remote_assets = client.fetch_assets()["assets"]
            remote_installations = client.fetch_site_assets()["site_assets"]
            by_code = {r.get("code"): r for r in remote_assets}
            result["external_preview"] = {str(asset_id): {
                "asset_in_api": bool(remote := by_code.get(Asset.objects.get(id=asset_id, tenant=tenant).asset_code)),
                "attachment_count": len(remote.get("attachments") or []) if remote else 0,
                "installed_site_asset_rows": sum(str(r.get("asset")) == str(remote.get("id")) for r in remote_installations) if remote else 0,
            } for asset_id in sorted(AUTHORIZED_ASSETS)}
            result["external_preview_status"] = "READ_ONLY_METADATA_SUCCEEDED"
        except Exception as exc:
            result["external_preview_status"] = type(exc).__name__
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-dir", type=Path, required=True)
    parser.add_argument("--current-dir", type=Path, required=True)
    parser.add_argument("--external-preview", action="store_true")
    args = parser.parse_args()
    previous = private_folder(args.previous_dir)
    current = private_folder(args.current_dir)
    import django
    django.setup()
    with read_only_database():
        report = audit(previous, current, args.external_preview)
    write_json(current / "environment_audit.json", report)
    safe = {key: value for key, value in report.items() if key != "vector_index"}
    safe["vector_index"] = {key: value for key, value in report["vector_index"].items() if key != "current_manifest_sha256"}
    print(json.dumps(safe, indent=2), flush=True)


if __name__ == "__main__":
    main()
