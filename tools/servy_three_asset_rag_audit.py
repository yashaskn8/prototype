"""Read-only live retrieval audit for all three approved assets / nine manuals.

Usage (Windows, in the authorized local company environment):
    .\.venv\Scripts\python.exe -B tools\servy_three_asset_rag_audit.py

The script uses the existing DPAPI-protected SELECT-only evaluation identity,
the frozen public 64-query regression fixture, exact nine-document provenance,
and a private disposable report folder. It does not train, reindex, mutate
company records, or copy source text into Git.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")

ASSET_SCOPE = {
    59: {"code": "SKY-AST-00001", "name": "Split AC Unit 2T", "customer_id": 14},
    60: {"code": "SKY-AST-00002", "name": "VRF Outdoor Module", "customer_id": 13},
    61: {"code": "SKY-AST-00003", "name": "Ductable AC 5T", "customer_id": 14},
}
ASSET_NAME_TO_ID = {row["name"]: asset_id for asset_id, row in ASSET_SCOPE.items()}
DOCUMENTS_BY_ASSET = {
    59: {168: "troubleshooting", 169: "maintenance", 170: "installation"},
    60: {171: "troubleshooting", 172: "maintenance", 173: "installation"},
    61: {174: "troubleshooting", 175: "maintenance", 176: "installation"},
}
REQUIRED_DOC_TYPES = {"troubleshooting", "maintenance", "installation"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_scope(tenant):
    from django.conf import settings
    from core.models import Asset, KnowledgeDocument, KnowledgeChunk
    from core.services.customer_isolation import (
        VERIFIED_REAL_PROVENANCE_REGISTRY,
        is_verified_real_asset,
        is_verified_real_customer,
        is_verified_real_document,
    )

    registry = VERIFIED_REAL_PROVENANCE_REGISTRY["documents"]
    expected_by_asset = {}
    for checksum, entry in registry.items():
        asset_id = entry.get("local_asset_id")
        if asset_id in DOCUMENTS_BY_ASSET:
            expected_by_asset.setdefault(asset_id, {})[entry["local_doc_id"]] = {
                **entry, "registered_checksum": checksum,
            }

    if set(expected_by_asset) != set(ASSET_SCOPE):
        raise RuntimeError("The approved nine-document registry no longer covers all three expected assets.")
    for asset_id, expected in DOCUMENTS_BY_ASSET.items():
        actual = expected_by_asset[asset_id]
        if set(actual) != set(expected):
            raise RuntimeError(f"Approved document IDs changed for asset {asset_id}.")
        if {entry["doc_type"] for entry in actual.values()} != REQUIRED_DOC_TYPES:
            raise RuntimeError(f"Asset {asset_id} no longer has one document of each approved type.")

    media_root = Path(settings.MEDIA_ROOT).resolve()
    assets = {}
    documents = {}
    chunk_counts = {}
    for asset_id, expected in ASSET_SCOPE.items():
        asset = Asset.objects.select_related(
            "tenant", "customer", "product", "site"
        ).get(pk=asset_id, tenant=tenant)
        if (asset.asset_code != expected["code"] or asset.name != expected["name"]
                or asset.customer_id != expected["customer_id"]
                or not is_verified_real_asset(asset)
                or not is_verified_real_customer(asset.customer)):
            raise RuntimeError(f"Approved asset binding failed for asset {asset_id}.")
        assets[asset_id] = asset

        for doc_id, doc_type in DOCUMENTS_BY_ASSET[asset_id].items():
            doc = KnowledgeDocument.objects.select_related(
                "tenant", "asset", "customer", "product", "category", "domain"
            ).get(pk=doc_id, tenant=tenant)
            if (doc.asset_id != asset_id
                    or doc.customer_id not in (None, expected["customer_id"])
                    or not doc.file or not doc.checksum_sha256
                    or not doc.is_rag_enabled or doc.is_confidential
                    or doc.disable_sharing or not is_verified_real_document(doc)):
                raise RuntimeError(f"Approved document eligibility failed for document {doc_id}.")
            file_path = Path(doc.file.path).resolve()
            if not file_path.is_relative_to(media_root) or not file_path.is_file():
                raise RuntimeError(f"Approved attachment path failed for document {doc_id}.")
            digest = _sha256_file(file_path)
            if digest.casefold() != doc.checksum_sha256.casefold():
                raise RuntimeError(f"Approved attachment checksum failed for document {doc_id}.")
            registered = expected_by_asset[asset_id][doc_id]["registered_checksum"]
            if digest.casefold() != registered.casefold():
                raise RuntimeError(f"Registry checksum mismatch for document {doc_id}.")

            chunks = list(KnowledgeChunk.objects.filter(
                document=doc, tenant=tenant, is_quarantined=False,
            ).order_by("chunk_index", "id"))
            if not chunks:
                raise RuntimeError(f"Approved document {doc_id} has no eligible chunks.")
            documents[doc_id] = {
                "asset_id": asset_id,
                "asset_code": expected["code"],
                "doc_type": doc_type,
                "title": doc.title,
                "index_status": doc.index_status,
                "index_version": doc.index_version,
                "attachment_sha256": digest,
                "chunk_count": len(chunks),
            }
            chunk_counts[doc_id] = len(chunks)

    return assets, documents, chunk_counts


def _build_cases(benchmark):
    cases = [
        row for row in benchmark["cases"]
        if row.get("equipment") in ASSET_NAME_TO_ID and row.get("positive_chunk_ids")
    ]
    counts = Counter(row["equipment"] for row in cases)
    expected = {
        "Split AC Unit 2T": 11,
        "VRF Outdoor Module": 8,
        "Ductable AC 5T": 9,
    }
    if counts != expected:
        raise RuntimeError(
            "Frozen three-asset regression fixture changed; refusing silent population substitution."
        )
    if any(int(row.get("tenant_id", 0)) != 9 for row in cases):
        raise RuntimeError("Selected regression queries do not share the approved Skydream tenant.")
    return cases


def run_audit():
    # Match the customer demo's extractive answer path and keep the audit fully
    # local/offline. This measures retrieval, not LLM paraphrase quality.
    os.environ["SERVY_LLM_PROVIDER"] = "extractive"
    os.environ["SERVY_ALLOW_REMOTE_LLM"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from tools.servy_100_readonly_eval import configure_process, verify_readonly_identity
    configure_process()
    identity = verify_readonly_identity()

    import django
    django.setup()

    from django.conf import settings
    from core.models import Tenant, KnowledgeChunk
    from core.ml.benchmark_registry import load_and_verify_benchmark
    from core.services.rag import ask_rag
    from core.services.retriever import get_serving_config
    from tools.servy_500_benchmark import (
        operational_counts, private_folder, read_only_database,
    )

    if settings.SERVY_CUSTOMER_DATA_MODE != "verified_real_only":
        raise RuntimeError("Live audit requires verified_real_only serving mode.")
    if settings.SERVY_ALLOW_REMOTE_LLM:
        raise RuntimeError("Live audit refuses remote model/API access.")
    config = get_serving_config()
    benchmark = load_and_verify_benchmark("servy-ci-regression-fixture-v1")
    cases = _build_cases(benchmark)

    tenant = Tenant.objects.get(slug="skydream", is_active=True)
    if tenant.id != 9:
        raise RuntimeError("Skydream tenant identity changed.")

    before = None
    after = None
    assets = {}
    documents = {}
    chunk_counts = {}
    query_rows = []
    with read_only_database():
        before = operational_counts()
        assets, documents, chunk_counts = _load_scope(tenant)

        all_target_doc_ids = set(documents)
        positive_ids = {
            int(cid)
            for row in cases
            for cid in row.get("positive_chunk_ids", [])
        }
        positive_chunks = {
            chunk.id: chunk for chunk in KnowledgeChunk.objects.filter(
                id__in=positive_ids
            ).select_related("document")
        }
        for row in cases:
            asset_id = ASSET_NAME_TO_ID[row["equipment"]]
            asset = assets[asset_id]
            expected_doc_ids = set(DOCUMENTS_BY_ASSET[asset_id])
            positives = {int(cid) for cid in row["positive_chunk_ids"]}
            if positives - set(positive_chunks):
                raise RuntimeError(f"{row['query_id']}: a labeled positive chunk is missing.")
            for cid in positives:
                chunk = positive_chunks[cid]
                if chunk.document_id not in expected_doc_ids:
                    raise RuntimeError(
                        f"{row['query_id']}: positive chunk is not bound to its registered asset manuals."
                    )

            result = ask_rag(
                tenant=tenant,
                customer=asset.customer,
                asset=asset,
                question=row["query_text"],
                original_question=row["query_text"],
                include_confidential=False,
                service_call=None,
                staff_mode=False,
                top_k=10,
            )
            retrieved = [
                item for item in (result.get("retrieved") or [])
                if getattr(item, "source_kind", "knowledge_document") == "knowledge_document"
            ]
            ranked_ids = [int(item.chunk_id) for item in retrieved if item.chunk_id is not None]
            ranked_doc_ids = [
                int(item.document_id) for item in retrieved if item.document_id is not None
            ]
            expected_positive_document_ids = sorted({
                int(positive_chunks[chunk_id].document_id) for chunk_id in positives
            })
            if any(doc_id not in expected_doc_ids for doc_id in expected_positive_document_ids):
                raise RuntimeError(
                    f"{row['query_id']}: expected positive document is outside its asset's nine manuals."
                )
            hit_ranks = [
                index + 1 for index, chunk_id in enumerate(ranked_ids)
                if chunk_id in positives
            ]
            first_relevant = min(hit_ranks) if hit_ranks else None
            document_hit_ranks = [
                index + 1 for index, document_id in enumerate(ranked_doc_ids)
                if document_id in expected_positive_document_ids
            ]
            first_relevant_document_rank = (
                min(document_hit_ranks) if document_hit_ranks else None
            )
            cross_asset_documents = sorted({
                doc_id for doc_id in ranked_doc_ids
                if doc_id in all_target_doc_ids and doc_id not in expected_doc_ids
            })
            referenced_chunk_ids = {
                int(ref["chunk_id"])
                for ref in (result.get("references") or [])
                if ref.get("chunk_id") is not None
            }
            query_rows.append({
                "query_id": row["query_id"],
                "equipment": row["equipment"],
                "category": row.get("category"),
                "query_sha256": hashlib.sha256(row["query_text"].encode("utf-8")).hexdigest(),
                "expected_positive_chunk_ids": sorted(positives),
                "expected_positive_document_ids": expected_positive_document_ids,
                "ranked_chunk_ids": ranked_ids[:10],
                "ranked_document_ids": ranked_doc_ids[:10],
                "first_relevant_rank": first_relevant,
                "first_relevant_document_rank": first_relevant_document_rank,
                "document_hit_at_1": first_relevant_document_rank == 1,
                "document_hit_at_5": (
                    first_relevant_document_rank is not None
                    and first_relevant_document_rank <= 5
                ),
                "document_hit_at_10": (
                    first_relevant_document_rank is not None
                    and first_relevant_document_rank <= 10
                ),
                "hit_at_1": first_relevant == 1,
                "hit_at_5": first_relevant is not None and first_relevant <= 5,
                "hit_at_10": first_relevant is not None and first_relevant <= 10,
                "cited_positive": bool(referenced_chunk_ids & positives),
                "cross_asset_document_ids": cross_asset_documents,
                "retrieval_backend": result.get("retrieval_backend"),
                "reranker_applied": bool(result.get("reranker_applied")),
                "reranker_model": result.get("reranker_model"),
                "engine": result.get("engine"),
            })
        after = operational_counts()

    if before != after:
        raise RuntimeError("Operational record counts changed during the read-only audit.")

    by_asset = {}
    for asset_name in ASSET_NAME_TO_ID:
        rows = [item for item in query_rows if item["equipment"] == asset_name]
        denom = len(rows)
        by_asset[asset_name] = {
            "queries": denom,
            "hit_at_1": sum(item["hit_at_1"] for item in rows),
            "hit_at_5": sum(item["hit_at_5"] for item in rows),
            "hit_at_10": sum(item["hit_at_10"] for item in rows),
            "cited_positive": sum(item["cited_positive"] for item in rows),
            "cross_asset_leaks": sum(bool(item["cross_asset_document_ids"]) for item in rows),
        }

    by_document = {}
    for document_id, metadata in documents.items():
        rows = [
            item for item in query_rows
            if document_id in item["expected_positive_document_ids"]
        ]
        denom = len(rows)
        by_document[str(document_id)] = {
            **metadata,
            "queries": denom,
            "coverage_status": "EVALUATED" if denom else "NOT_EVALUATED_BY_FROZEN_FIXTURE",
            "chunk_hit_at_1": sum(item["hit_at_1"] for item in rows) if denom else None,
            "chunk_hit_at_5": sum(item["hit_at_5"] for item in rows) if denom else None,
            "document_hit_at_1": sum(item["document_hit_at_1"] for item in rows) if denom else None,
            "document_hit_at_5": sum(item["document_hit_at_5"] for item in rows) if denom else None,
            "citation_coverage": sum(item["cited_positive"] for item in rows) if denom else None,
        }

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(os.environ["LOCALAPPDATA"]) / "Servy" / "private_evaluations"
    output_dir = private_folder(base / f"servy_three_asset_rag_{timestamp}")
    report = {
        "audit_version": "three-asset-nine-document-v1",
        "benchmark": {
            "dataset_id": benchmark["dataset_id"],
            "dataset_role": benchmark["dataset_role"],
            "sha256": benchmark["sha256_digest"],
            "population_warning": (
                "Sanitized software regression fixture, not an independent production holdout "
                "or a subject-matter-approved accuracy estimate."
            ),
        },
        "database_identity": identity,
        "serving_config": {
            "embedding_profile": config.embedding_profile,
            "retrieval_mode": config.retrieval_mode,
            "reranker_enabled": config.reranker_enabled,
            "top_k": config.top_k,
        },
        "approved_assets": {
            str(asset_id): {
                "asset_code": ASSET_SCOPE[asset_id]["code"],
                "name": ASSET_SCOPE[asset_id]["name"],
                "customer_id": asset.customer_id,
                "approved_document_ids": sorted(DOCUMENTS_BY_ASSET[asset_id]),
            }
            for asset_id, asset in assets.items()
        },
        "approved_documents": documents,
        "chunk_counts": chunk_counts,
        "operational_counts_before": before,
        "operational_counts_after": after,
        "operational_counts_unchanged": before == after,
        "metrics_by_asset": by_asset,
        "metrics_by_document": by_document,
        "total_queries": len(query_rows),
        "total_hit_at_1": sum(item["hit_at_1"] for item in query_rows),
        "total_hit_at_5": sum(item["hit_at_5"] for item in query_rows),
        "total_hit_at_10": sum(item["hit_at_10"] for item in query_rows),
        "total_cross_asset_leaks": sum(bool(item["cross_asset_document_ids"]) for item in query_rows),
        "queries": query_rows,
        "training": {
            "status": "NOT_RUN",
            "reason": (
                "The existing company fine-tuning labels were recorded as weakly supervised "
                "and unreviewed. This audit produces retrieval diagnostics only; it does not "
                "convert heading-derived or benchmark-derived matches into training labels."
            ),
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "audit": report["audit_version"],
        "report_path": str(report_path),
        "benchmark_role": report["benchmark"]["dataset_role"],
        "serving_config": report["serving_config"],
        "metrics_by_asset": by_asset,
        "metrics_by_document": by_document,
        "total_queries": len(query_rows),
        "total_hit_at_1": report["total_hit_at_1"],
        "total_hit_at_5": report["total_hit_at_5"],
        "total_cross_asset_leaks": report["total_cross_asset_leaks"],
        "operational_counts_unchanged": report["operational_counts_unchanged"],
        "training": report["training"]["status"],
    }, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run_audit()
