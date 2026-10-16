"""Strict-source complaint retrieval diagnostics; never creates ground truth."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SERVY_DB_ENGINE", "sqlite3")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
import django
django.setup()

import numpy as np
from core.ml.embeddings import get_profile, load_model, passage_text
from core.ml.op22_technical_evidence import parse_technical_query, rank_authorized_candidates, verify_answer_evidence
from core.models import Asset, Tenant
from core.services.retriever import _knowledge_candidates
from core.services.vectors import compute_bm25_scores

PRIVATE = ROOT / "private_training_data"
OUT = Path(os.environ.get("SERVY_EXPERIMENT_OUTPUT_DIR", str(PRIVATE / "op22_review"))).resolve()
if not OUT.is_relative_to(PRIVATE.resolve()):
    raise ValueError("Company diagnostics must remain private")
OUT.mkdir(parents=True, exist_ok=True)
audit = json.loads((PRIVATE / "op22_provenance_audit.json").read_text(encoding="utf-8"))
verified_ids = {r["document_id"] for r in audit["document_links"] if r["classification"] == "VERIFIED_ORIGINAL_COMPANY_SOURCE"}
cases = [json.loads(line) for line in (PRIVATE / "op22_review" / "authentic_complaint_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
tenant = Tenant.objects.get(id=audit["source_tenant"])
profile = get_profile("minilm")
model = load_model(profile)
results = []
for case in cases:
    if case["asset_id"] is None:
        results.append({"query_id": case["query_id"], "decision": "NEEDS_ASSET_CLARIFICATION",
                        "candidate_chunk_ids": [], "review_status": "PENDING_REVIEW"})
        continue
    asset = Asset.objects.get(id=case["asset_id"])
    scoped = list(_knowledge_candidates(tenant, asset, None, False))
    authorized = [c for c in scoped if c.document_id in verified_ids and c.document.asset_id == asset.id]
    query = parse_technical_query(case["question_text"])
    if not authorized:
        results.append({"query_id": case["query_id"], "decision": "NO_SUFFICIENT_EVIDENCE_FOUND",
                        "candidate_chunk_ids": [], "review_status": "PENDING_REVIEW"})
        continue
    vectors = np.asarray(model.encode([passage_text(profile, c) for c in authorized], normalize_embeddings=True))
    query_vector = np.asarray(model.encode([profile.query_prefix + query.text], normalize_embeddings=True))[0]
    dense_scores = vectors @ query_vector
    lexical_scores = compute_bm25_scores(query.text, [f"{c.heading}\n{c.text}" for c in authorized])
    dense_order = sorted(range(len(authorized)), key=lambda i: (-float(dense_scores[i]), authorized[i].id))
    lexical_order = sorted(range(len(authorized)), key=lambda i: (-float(lexical_scores[i]), authorized[i].id))
    ranked = rank_authorized_candidates(
        query, authorized,
        {authorized[i].id: rank + 1 for rank, i in enumerate(dense_order)},
        {authorized[i].id: rank + 1 for rank, i in enumerate(lexical_order)},
        verified_document_ids=verified_ids, asset=asset,
    )
    verdict = verify_answer_evidence(query, ranked, index_complete=True,
                                     source_universe_complete=True,
                                     expected_candidate_count=len(authorized),
                                     searched_candidate_count=len(ranked))
    results.append({
        "query_id": case["query_id"], "asset_id": asset.id,
        "source_verified_candidate_count": len(authorized),
        "minilm_top_5": [authorized[i].id for i in dense_order[:5]],
        "evidence_ranked_top_5": [r["chunk_id"] for r in ranked[:5]],
        "machine_evidence_decisions_top_5": [r["assessment"].decision for r in ranked[:5]],
        "machine_verifier_decision": verdict["decision"],
        "answer_coverage": verdict["answer_coverage"],
        "review_status": "PENDING_REVIEW", "accuracy": "NOT_MEASURED_NO_APPROVED_RELEVANCE_OR_ANSWER_LABELS",
    })

(OUT / "company_complaint_diagnostic.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
