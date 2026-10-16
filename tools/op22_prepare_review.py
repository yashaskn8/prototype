"""Build private, unapproved source-backed review queues without rewriting labels."""
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SERVY_DB_ENGINE", "sqlite3")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
import django
django.setup()

from core.ml.benchmark_manifest import build_manifest, resolve_query_asset
from core.models import Asset, KnowledgeChunk, KnowledgeDocument, Tenant
from core.services.retriever import _knowledge_candidates
from core.services.vectors import compute_bm25_scores

PRIVATE = ROOT / "private_training_data"
OUT = PRIVATE / "op22_review"
OUT.mkdir(parents=True, exist_ok=True)
snapshot = PRIVATE / "op22_starlly_snapshot"
source_audit = json.loads((PRIVATE / "op22_provenance_audit.json").read_text(encoding="utf-8"))
source_docs = {r["document_id"]: r for r in source_audit["document_links"]}
complaints = json.loads((snapshot / "complaints.json").read_text(encoding="utf-8"))
external_assets = json.loads((snapshot / "assets.json").read_text(encoding="utf-8"))
asset_by_external = {r["id"]: r for r in external_assets}
local_assets = {a.asset_code: a for a in Asset.objects.filter(tenant_id=9)}
chunks = {c.id: c for c in KnowledgeChunk.objects.select_related("document").all()}

suspect = {
    "EVAL-ERR-01", "EVAL-TRAP-02", "EVAL-TRAP-03", "EVAL-TRAP-04", "EVAL-TRAP-06",
    "EVAL-AMB-03", "EVAL-AMB-04", "EVAL-TECH-01", "EVAL-TECH-02", "EVAL-XTLK-01", "EVAL-XTLK-02",
}
possible_alternatives = {
    "EVAL-ERR-04": 129, "EVAL-SYM-04": 108, "EVAL-SYM-06": 117,
    "EVAL-MSP-02": 107, "EVAL-MLT-01": 154, "EVAL-TECH-03": 220,
}

historical = []
for case in build_manifest("servy-real-holdout-v1")["cases"]:
    original = [chunks[cid] for cid in case["positive_chunk_ids"] if cid in chunks]
    source = source_docs.get(original[0].document_id) if original else None
    asset = resolve_query_asset(case["tenant_id"], case["equipment"])
    historical.append({
        "query_id": case["query_id"], "question_text": case["query_text"],
        "dataset_role": "HISTORICAL_DIAGNOSTIC_NOT_INDEPENDENT_FINAL_TEST",
        "tenant_id": case["tenant_id"], "asset_id": asset.id if asset else None,
        "expected_equipment": case["equipment"],
        "source_organisation": "273" if case["tenant_id"] == 9 else None,
        "source_document_id": original[0].document_id if original else None,
        "source_attachment_id": source["external_attachment_id"] if source else None,
        "source_sha256": source["raw_sha256"] if source else (original[0].document.checksum_sha256 if original else None),
        "source_version": original[0].document.index_version if original else None,
        "source_verification": source["classification"] if source else "NOT_CERTIFIED_BY_ORG_273_SNAPSHOT",
        "original_labeled_chunk_ids": case["positive_chunk_ids"],
        "original_labeled_sections": [c.heading for c in original],
        "original_labeled_passages": [c.text for c in original],
        "required_technical_fact_or_procedure": case["query_text"],
        "candidate_alternative_chunk_ids": [possible_alternatives[case["query_id"]]] if case["query_id"] in possible_alternatives else [],
        "candidate_alternative_passages": [chunks[possible_alternatives[case["query_id"]]].text]
                                         if case["query_id"] in possible_alternatives else [],
        "valid_alternative_chunk_ids": [],
        "candidate_answerability": "SUSPECTED_UNANSWERABLE_PENDING_REVIEW" if case["query_id"] in suspect else "PENDING_REVIEW",
        "support_status": "PENDING_REVIEW",
        "review_status": "PENDING_REVIEW", "reviewer_identity": None, "reviewed_at": None,
        "approval_evidence": None,
        "review_flags": (["POSSIBLE_MISSING_TECHNICAL_SUPPORT"] if case["query_id"] in suspect else [])
                        + (["POSSIBLE_ALTERNATIVE_VALID_PASSAGE"] if case["query_id"] in possible_alternatives else []),
    })

company_cases = []
tenant = Tenant.objects.get(id=9)
for complaint in complaints:
    ext_asset = asset_by_external.get(complaint.get("asset"))
    asset = local_assets.get(ext_asset["code"]) if ext_asset else None
    authorized = list(_knowledge_candidates(
        tenant=tenant, asset=asset, customer=None,
        include_confidential=False, staff_mode=False,
    )) if asset else []
    verified = [c for c in authorized if source_docs.get(c.document_id, {}).get("classification") == "VERIFIED_ORIGINAL_COMPANY_SOURCE"]
    scores = compute_bm25_scores(complaint.get("description") or "", [f"{c.heading}\n{c.text}" for c in verified])
    suggested = [verified[i].id for i in sorted(range(len(verified)), key=lambda i: (-scores[i], verified[i].id))[:5]]
    candidate_passages = []
    for chunk_id in suggested:
        chunk = chunks[chunk_id]
        source = source_docs[chunk.document_id]
        candidate_passages.append({
            "chunk_id": chunk_id, "document_id": chunk.document_id,
            "external_attachment_id": source["external_attachment_id"],
            "source_sha256": source["raw_sha256"],
            "heading": chunk.heading, "passage_text": chunk.text,
            "evidence_status": "CANDIDATE_PENDING_TECHNICAL_REVIEW",
        })
    company_cases.append({
        "query_id": f"OP22-COMPLAINT-{complaint['id']}",
        "question_text": complaint.get("description") or "",
        "dataset_role": "AUTHENTIC_COMPLAINT_CANDIDATE_NOT_GROUND_TRUTH",
        "source_system": "starlly", "source_organisation": "273",
        "source_snapshot_sha256": source_audit["snapshot_sha256"],
        "source_complaint_id": complaint["id"], "tenant_id": tenant.id,
        "external_asset_id": complaint.get("asset"), "asset_id": asset.id if asset else None,
        "expected_equipment": asset.name if asset else None,
        "candidate_passage_chunk_ids": suggested,
        "candidate_evidence_passages": candidate_passages,
        "valid_positive_chunk_ids": [], "valid_alternative_chunk_ids": [],
        "candidate_answerability": "PENDING_REVIEW" if asset else "NEEDS_ASSET_CLARIFICATION",
        "support_status": "PENDING_REVIEW", "review_status": "PENDING_REVIEW",
        "reviewer_identity": None, "reviewed_at": None, "approval_evidence": None,
        "review_flags": ["GENERIC_COMPLAINT"] + ([] if asset else ["NO_ASSET_IDENTITY"]),
    })

def write_jsonl(name, rows):
    with (OUT / name).open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")

write_jsonl("historical_label_review.jsonl", historical)
write_jsonl("authentic_complaint_candidates.jsonl", company_cases)
summary = {
    "historical_cases": len(historical), "historical_suspect_support": len(suspect),
    "historical_possible_alternatives": len(possible_alternatives),
    "historical_source_verification": dict(Counter(x["source_verification"] for x in historical)),
    "authentic_complaint_candidates": len(company_cases),
    "complaints_with_asset": sum(x["asset_id"] is not None for x in company_cases),
    "complaints_needing_asset_clarification": sum(x["asset_id"] is None for x in company_cases),
    "approved_judgments": 0,
    "independent_source_separated_test_cases": 0,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
exclusion_path = OUT / "prior_source_exclusion.json"
if not exclusion_path.exists():
    prior_hashes = sorted({value for value in KnowledgeDocument.objects.values_list("checksum_sha256", flat=True)
                           if value and len(value) == 64})
    with exclusion_path.open("x", encoding="utf-8") as target:
        target.write(json.dumps({
            "purpose": "Immutable baseline of sources present before a new independent test is collected",
            "source_snapshot_sha256": source_audit["snapshot_sha256"],
            "excluded_raw_source_sha256": prior_hashes,
        }, indent=2))
print(json.dumps(summary, indent=2))
