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


def _norm(text):
    import re
    text = re.sub(r"^\s*(?:[-*•]+|\[\s?\]|\d+[.)])\s*", "", str(text or ""), flags=re.M)
    return " ".join(text.split()).casefold()


def extraction_coverage(tenant, documents):
    """Compare each original source file with its searchable chunks.

    Reports, per document, how many non-empty source lines appear in the
    indexed chunk text and which do not. Nothing is assumed complete: a
    document whose source cannot be read as text is flagged, not passed.
    """
    from django.conf import settings
    from core.models import KnowledgeChunk, KnowledgeDocument
    out = {}
    for doc_id in documents:
        doc = KnowledgeDocument.objects.get(pk=doc_id, tenant=tenant)
        path = Path(doc.file.path).resolve()
        row = {"file_type": path.suffix.lower() or "none", "warnings": []}
        chunk_text = _norm(" ".join(KnowledgeChunk.objects.filter(
            document=doc, tenant=tenant, is_quarantined=False,
        ).order_by("chunk_index", "id").values_list("text", flat=True)))
        headings = [h for h in KnowledgeChunk.objects.filter(
            document=doc, tenant=tenant, is_quarantined=False,
        ).values_list("heading", flat=True) if h]
        row["chunk_headings"] = sorted(set(headings))
        if row["file_type"] not in {".txt", ".md"}:
            # A PDF or scan cannot be compared line by line here; say so.
            row["warnings"].append("SOURCE_NOT_PLAIN_TEXT_COVERAGE_NOT_VERIFIED")
            row["source_lines"] = row["lines_in_chunks"] = None
            out[str(doc_id)] = row
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        lines = [line for line in raw.splitlines() if _norm(line) and len(_norm(line)) >= 4]
        missing = [line.strip()[:80] for line in lines if _norm(line) not in chunk_text]
        row["source_lines"] = len(lines)
        row["lines_in_chunks"] = len(lines) - len(missing)
        row["lines_not_searchable"] = missing[:12]
        if "\ufffd" in raw:
            row["warnings"].append("SOURCE_HAS_UNDECODABLE_CHARACTERS")
        if missing:
            row["warnings"].append(f"{len(missing)}_SOURCE_LINES_NOT_IN_SEARCHABLE_CHUNKS")
        out[str(doc_id)] = row
    return out


def _open_source_in_process(url):
    """Open a signed citation exactly as the demo's source route does."""
    from types import SimpleNamespace
    from urllib.parse import parse_qs, urlsplit
    from django.core import signing
    from core.services.customer_answer import SALT, open_citation
    from tools import servy_real_demo as demo
    try:
        token = parse_qs(urlsplit(url).query).get("citation", [""])[0]
        payload = signing.loads(token, salt=SALT, max_age=3600)
        asset_id = payload.get("asset")
        if (demo.ASSET_SCOPE.get(asset_id) != payload.get("customer")
                or payload.get("user") != demo.DEMO_USER_ID):
            return 404, {}
        tenant, customer, _asset = demo._asset(asset_id)
        ctx = {"user": SimpleNamespace(id=demo.DEMO_USER_ID), "tenant": tenant,
               "customer": customer, "can_view_confidential": False}
        return 200, open_citation(token, ctx)
    except Exception:
        return 404, {}


def evaluate_manual_grounded(tenant, assets, documents):
    """Retrieval stage and full customer path for manual-grounded questions.

    For each answerable single question: the positive passages are the
    indexed chunks of the named manual that contain the expected phrase, the
    pre-reranking candidates and final ranking come from ``ask_rag``, and the
    customer answer comes from the demo's own ``_answer`` path. Failures are
    attributed to candidate recall, reranking or answer construction.
    """
    import uuid
    from types import SimpleNamespace
    from core.models import KnowledgeChunk
    from core.services.rag import ask_rag
    from tools import servy_real_demo as demo
    from tools import servy_three_asset_customer_eval as evaluation

    sources = {}
    chunks_by_doc = {}
    for doc_id in documents:
        rows = list(KnowledgeChunk.objects.filter(
            document_id=doc_id, tenant=tenant, is_quarantined=False,
        ).order_by("chunk_index", "id"))
        chunks_by_doc[doc_id] = rows
        sources[doc_id] = " ".join(f"{row.heading or ''} {row.text or ''}" for row in rows)
    opener = SimpleNamespace(source=_open_source_in_process)
    all_docs = set(documents)
    out = []
    for case_id, asset_id, manual, turns in evaluation.CASES:
        asset = assets[asset_id]
        expected_docs = set(DOCUMENTS_BY_ASSET[asset_id])
        continuation, thread_id = None, "thread-" + uuid.uuid4().hex
        for index, expected in enumerate(turns, start=1):
            row = {"case": case_id, "turn": index, "asset_id": asset_id, "manual": manual}
            if index == 1 and expected["kind"] == evaluation.ANSWER:
                positives = {
                    chunk.id for chunk in chunks_by_doc[expected["doc"]]
                    if any(_norm(phrase) in _norm(chunk.text) for phrase in expected["include"])}
                result = ask_rag(
                    tenant=tenant, customer=asset.customer, asset=asset,
                    question=expected["question"], original_question=expected["question"],
                    include_confidential=False, service_call=None, staff_mode=False, top_k=10)
                retrieved = [item for item in (result.get("retrieved") or [])
                             if getattr(item, "source_kind", "knowledge_document") == "knowledge_document"]
                ranked = [int(i.chunk_id) for i in retrieved if i.chunk_id is not None]
                ranked_docs = [int(i.document_id) for i in retrieved if i.document_id is not None]
                candidates = [int(c) for c in result.get("pre_rerank_candidate_chunk_ids") or []]
                hits = [n + 1 for n, cid in enumerate(ranked) if cid in positives]
                pre_hits = [n + 1 for n, cid in enumerate(candidates) if cid in positives]
                first = min(hits) if hits else None
                row["retrieval"] = {
                    "positive_chunk_ids": sorted(positives),
                    "ground_truth_passage_indexed": bool(positives),
                    "pre_rerank_candidate_chunk_ids": candidates,
                    "pre_rerank_first_positive_rank": min(pre_hits) if pre_hits else None,
                    "candidate_recall": bool(pre_hits),
                    "final_ranked_chunk_ids": ranked[:10],
                    "final_first_positive_rank": first,
                    "hit_at_1": first == 1, "hit_at_5": first is not None and first <= 5,
                    "retrieval_backend": result.get("retrieval_backend"),
                    "reranker_applied": bool(result.get("reranker_applied")),
                    "reranker_model": result.get("reranker_model"),
                    "cited_positive": bool({int(r["chunk_id"]) for r in result.get("references") or []
                                            if r.get("chunk_id") is not None} & positives),
                    "cross_asset_document_ids": sorted({
                        d for d in ranked_docs if d in all_docs and d not in expected_docs}),
                }
            try:
                data = demo._answer(asset_id, expected["question"], continuation=continuation,
                                    thread_id=thread_id)
                status = 200
            except ValueError:
                data, status = {}, 400
            answer = evaluation.evaluate_turn(expected, status, data, asset_id=asset_id,
                                              sources=sources, client=opener)
            continuation = data.get("continuation") or continuation
            row["answer"] = answer
            retrieval = row.get("retrieval")
            if answer["passed"]:
                row["failure_stage"] = None
            elif retrieval and not retrieval["ground_truth_passage_indexed"]:
                row["failure_stage"] = "GROUND_TRUTH_PASSAGE_NOT_INDEXED"
            elif retrieval and not retrieval["candidate_recall"]:
                row["failure_stage"] = "CANDIDATE_RECALL_FAILURE"
            elif retrieval and not retrieval["hit_at_5"]:
                row["failure_stage"] = "RERANKING_FAILURE"
            else:
                row["failure_stage"] = "ANSWER_CONSTRUCTION_FAILURE"
            out.append(row)
    return out


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
        activate_index_copy, file_manifest, load_json, operational_counts,
        prepare_index_copy, private_folder, read_only_database, serving,
    )

    if settings.SERVY_CUSTOMER_DATA_MODE != "verified_real_only":
        raise RuntimeError("Live audit requires verified_real_only serving mode.")
    if settings.SERVY_ALLOW_REMOTE_LLM:
        raise RuntimeError("Live audit refuses remote model/API access.")
    # Evaluated configuration only: BGE-small, hybrid, V5 enabled, pool five.
    serving_configuration = serving()
    config = get_serving_config()

    # Chroma rewrites files when opened, so no query touches the original
    # index. Copy it byte-for-byte into this run's private folder, verify the
    # original and copied manifests, and point this process at the copy.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(os.environ["LOCALAPPDATA"]) / "Servy" / "private_evaluations"
    output_dir = private_folder(base / f"servy_three_asset_rag_{timestamp}")
    original_index = Path(settings.CHROMA_PERSIST_DIR).resolve()
    prepare_index_copy(output_dir)
    index_manifest_before = load_json(output_dir / "production_index_manifest.json")
    if file_manifest(output_dir / "chroma_index_copy") != index_manifest_before:
        raise RuntimeError("Private index copy differs from the original manifest; no queries run.")
    activate_index_copy(output_dir)

    # No result is recorded unless the stack demonstrably runs: local BGE
    # dimensions, both collections present, and real SHA-pinned V5 inference.
    from core.management.commands.check_rag_serving import verify_local_serving_artifacts
    verified_serving = verify_local_serving_artifacts()
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
    reranker_gaps = []
    with read_only_database():
        before = operational_counts()
        assets, documents, chunk_counts = _load_scope(tenant)
        coverage = extraction_coverage(tenant, documents)

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
            candidate_ids = [int(c) for c in result.get("pre_rerank_candidate_chunk_ids") or []]
            if config.reranker_enabled and not result.get("reranker_applied"):
                reranker_gaps.append(row["query_id"])
            query_rows.append({
                "pre_rerank_candidate_chunk_ids": candidate_ids,
                "candidate_recall": bool(set(candidate_ids) & positives),
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
        manual_rows = evaluate_manual_grounded(tenant, assets, documents)
        after = operational_counts()

    if before != after:
        raise RuntimeError("Operational record counts changed during the read-only audit.")
    index_manifest_after = file_manifest(original_index)
    if index_manifest_after != index_manifest_before:
        raise RuntimeError("Original Chroma index bytes changed during the read-only audit.")

    def _stage_counts(rows):
        return dict(Counter(str(r["failure_stage"] or "PASS") for r in rows))

    retrieval_rows = [r for r in manual_rows if r.get("retrieval")]
    manual_by_document = {}
    for document_id in documents:
        rows = [r for r in manual_rows if r["manual"] == document_id]
        staged = [r for r in rows if r.get("retrieval")]
        manual_by_document[str(document_id)] = {
            "turns": len(rows),
            "customer_answer_passed": sum(r["answer"]["passed"] for r in rows),
            "retrieval_questions": len(staged),
            "candidate_recall": sum(r["retrieval"]["candidate_recall"] for r in staged),
            "final_hit_at_1": sum(r["retrieval"]["hit_at_1"] for r in staged),
            "final_hit_at_5": sum(r["retrieval"]["hit_at_5"] for r in staged),
            "citation_coverage": sum(r["retrieval"]["cited_positive"] for r in staged),
            "failure_stages": _stage_counts(rows),
        }
    manual_by_asset = {}
    for asset_id in ASSET_SCOPE:
        rows = [r for r in manual_rows if r["asset_id"] == asset_id]
        staged = [r for r in rows if r.get("retrieval")]
        manual_by_asset[str(asset_id)] = {
            "turns": len(rows),
            "customer_answer_passed": sum(r["answer"]["passed"] for r in rows),
            "retrieval_questions": len(staged),
            "candidate_recall": sum(r["retrieval"]["candidate_recall"] for r in staged),
            "final_hit_at_1": sum(r["retrieval"]["hit_at_1"] for r in staged),
            "final_hit_at_5": sum(r["retrieval"]["hit_at_5"] for r in staged),
            "cross_asset_leaks": sum(
                bool(r["retrieval"]["cross_asset_document_ids"]) for r in staged)
                + sum(not r["answer"]["checks"]["no_cross_asset_citation"] for r in rows),
            "restricted_content_turns": sum(
                not r["answer"]["checks"]["no_restricted_or_procedural_content"] for r in rows),
            "claims_not_verbatim": sum(
                not r["answer"]["checks"]["claims_verbatim_in_cited_manual"] for r in rows),
            "citation_open_failures": sum(
                r["answer"]["checks"].get("citation_opens_to_the_claim") is False for r in rows),
            "failure_stages": _stage_counts(rows),
        }

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

    report = {
        "audit_version": "three-asset-nine-document-v2",
        "index_under_test": (
            "byte-identical private copy of the original Chroma index "
            "(the customer demo may serve from a separately rebuilt isolated index)"),
        "serving_verification": {
            "configuration": serving_configuration,
            "artifacts": verified_serving,
            "fixture_queries_where_v5_did_not_apply": reranker_gaps,
        },
        "integrity": {
            "original_index_manifest_unchanged": index_manifest_after == index_manifest_before,
            "private_copy_matched_original_before_queries": True,
            "operational_counts_unchanged": before == after,
        },
        "extraction_coverage": coverage,
        "manual_grounded": {
            "note": ("Expected passages are located in the indexed chunks by the manual's own "
                     "wording. This is a small, hand-built question set, not an independent "
                     "holdout or an accuracy estimate."),
            "turns": len(manual_rows),
            "customer_answer_passed": sum(r["answer"]["passed"] for r in manual_rows),
            "retrieval_questions": len(retrieval_rows),
            "candidate_recall": sum(r["retrieval"]["candidate_recall"] for r in retrieval_rows),
            "final_hit_at_1": sum(r["retrieval"]["hit_at_1"] for r in retrieval_rows),
            "final_hit_at_5": sum(r["retrieval"]["hit_at_5"] for r in retrieval_rows),
            "failure_stages": _stage_counts(manual_rows),
            "by_asset": manual_by_asset,
            "by_document": manual_by_document,
            "rows": manual_rows,
        },
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
        "index_under_test": report["index_under_test"],
        "serving_verification": report["serving_verification"],
        "integrity": report["integrity"],
        "extraction_coverage": {
            doc: {k: v for k, v in row.items() if k != "chunk_headings"}
            for doc, row in coverage.items()},
        "manual_grounded": {k: v for k, v in report["manual_grounded"].items() if k != "rows"},
        "manual_grounded_failures": [
            {"case": r["case"], "turn": r["turn"], "stage": r["failure_stage"],
             "question": r["answer"]["question"], "failed_checks": r["answer"]["failed_checks"],
             "response_type": r["answer"]["response_type"]}
            for r in manual_rows if not r["answer"]["passed"]],
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
