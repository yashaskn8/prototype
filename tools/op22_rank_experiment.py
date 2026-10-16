"""Fixed-rule evidence ranker versus fair baselines; no training or holdout tuning."""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SERVY_DB_ENGINE", "sqlite3")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
import django
django.setup()

import numpy as np
from core.ml.embeddings import get_profile, load_model, passage_text
from core.ml.indexer import _eligible_chunks
from core.ml.op22_technical_evidence import parse_technical_query, rank_authorized_candidates, verify_answer_evidence
from core.models import Asset, KnowledgeChunk, Tenant
from core.services.retriever import _knowledge_candidates

PRIVATE = ROOT / "private_training_data"
OUT = PRIVATE / "op22_review"
historical = [json.loads(line) for line in (PRIVATE / "op21_accuracy_audit" / "per_query.jsonl").read_text(encoding="utf-8").splitlines()]
validation = [json.loads(line) for line in (PRIVATE / "company_val_weakly_supervised.jsonl").read_text(encoding="utf-8").splitlines()]
dev_old = {r["query_id"]: r for r in json.loads((PRIVATE / "op21_accuracy_audit" / "dev_experiments.json").read_text(encoding="utf-8"))["per_query"]}
source_audit = json.loads((PRIVATE / "op22_provenance_audit.json").read_text(encoding="utf-8"))
verified_company_docs = {d["document_id"] for d in source_audit["document_links"] if d["classification"] == "VERIFIED_ORIGINAL_COMPANY_SOURCE"}
all_chunks = list(_eligible_chunks())
chunk_by_id = {c.id: c for c in all_chunks}
tenants = {t.id: t for t in Tenant.objects.all()}


def score_rows(rows):
    evaluated = [r for r in rows if r.get("positive_chunk_ids")]
    n = len(evaluated)
    if not n:
        return {"n": 0}
    results = {}
    for key in ("fair_minilm", "fair_base_bge", "fair_op20_bge_ft", "evidence_ranked"):
        positions = [next((i + 1 for i, cid in enumerate(r[key]) if cid in r["positive_chunk_ids"]), None) for r in evaluated]
        results[key] = {
            "n": n,
            "hit_at_1": round(sum(p == 1 for p in positions) / n, 4),
            "hit_at_5": round(sum(p is not None and p <= 5 for p in positions) / n, 4),
            "recall_at_10": round(sum(p is not None and p <= 10 for p in positions) / n, 4),
            "recall_at_20": round(sum(p is not None and p <= 20 for p in positions) / n, 4),
            "mrr_at_10": round(sum(1/p for p in positions if p and p <= 10) / n, 4),
            "ndcg_at_10": round(sum(1/np.log2(p + 1) for p in positions if p and p <= 10) / n, 4),
        }
    return results


dev = []
for row in validation:
    pos = row["positive_chunk_ids"][0]
    source_doc = chunk_by_id[pos].document
    asset = source_doc.asset or Asset.objects.filter(tenant_id=row["tenant_id"], product_id=source_doc.product_id).order_by("id").first()
    allowed = list(_knowledge_candidates(tenants[row["tenant_id"]], asset, None, False))
    allowed_ids = {c.id for c in allowed} & set(chunk_by_id)
    if pos not in allowed_ids:
        raise RuntimeError(f"Development positive outside authorized corpus: {row['query_id']}")
    dev.append({
        "query_id": row["query_id"], "query_text": row["query_text"],
        "tenant_id": row["tenant_id"], "asset": asset, "asset_id": asset.id,
        "positive_chunk_ids": row["positive_chunk_ids"], "category": "weak_validation",
        "lexical": dev_old[row["query_id"]]["lexical"],
        "fair_op20_bge_ft": dev_old[row["query_id"]]["dense"],
        "authorized_ids": allowed_ids,
    })

for name, profile_key in (("fair_minilm", "minilm"), ("fair_base_bge", "bge-small")):
    profile = get_profile(profile_key)
    model = load_model(profile)
    cv = np.asarray(model.encode([passage_text(profile, c) for c in all_chunks], normalize_embeddings=True, batch_size=8))
    qv = np.asarray(model.encode([profile.query_prefix + r["query_text"] for r in dev], normalize_embeddings=True, batch_size=8))
    scores = qv @ cv.T
    for idx, row in enumerate(dev):
        indices = [i for i, c in enumerate(all_chunks) if c.id in row["authorized_ids"]]
        row[name] = [all_chunks[i].id for i in sorted(indices, key=lambda i: (-float(scores[idx, i]), all_chunks[i].id))]


def evaluate_row(row, *, strict_source):
    asset = row.get("asset")
    if asset is None and row.get("asset_id"):
        asset = Asset.objects.get(id=row["asset_id"])
    authorized = list(_knowledge_candidates(tenants[row["tenant_id"]], asset, None, False))
    if strict_source:
        verified = verified_company_docs
    else:
        # Explicitly uncertified legacy research mode for mixed-tenant comparison.
        verified = {c.document_id for c in authorized}
    query = parse_technical_query(row["query_text"])
    dense = {cid: i + 1 for i, cid in enumerate(row["fair_minilm"])}
    lexical = {cid: i + 1 for i, cid in enumerate(row["lexical"])}
    started = time.perf_counter()
    ranked = rank_authorized_candidates(query, authorized, dense, lexical, verified_document_ids=verified, asset=asset)
    latency_ms = (time.perf_counter() - started) * 1000
    verification = verify_answer_evidence(query, ranked)
    return {
        "evidence_ranked": [r["chunk_id"] for r in ranked],
        "evidence_top_5": [{"chunk_id": r["chunk_id"], "score": r["score"],
                            "decision": r["assessment"].decision,
                            "reasons": r["assessment"].reasons} for r in ranked[:5]],
        "verifier": verification, "feature_rank_latency_ms": round(latency_ms, 3),
    }


dev_results = []
for row in dev:
    result = {k: v for k, v in row.items() if k not in {"asset", "authorized_ids"}}
    result.update(evaluate_row(row, strict_source=False))
    dev_results.append(result)

hist_results = []
for row in historical:
    result = {k: row[k] for k in ("query_id", "query_text", "tenant_id", "asset_id", "category", "equipment", "role", "positive_chunk_ids", "fair_minilm", "fair_base_bge", "fair_op20_bge_ft", "lexical")}
    result.update(evaluate_row(row, strict_source=False))
    hist_results.append(result)

relevance = [r for r in hist_results if r["role"] == "RELEVANCE_EVALUATION"]
per_category = defaultdict(list)
per_equipment = defaultdict(list)
for row in relevance:
    per_category[row["category"]].append(row)
    per_equipment[row["equipment"]].append(row)

dev_metrics = score_rows(dev_results)
historical_metrics = score_rows(relevance)
summary = {
    "ranker_version": "op22_fixed_evidence_v1",
    "source_mode": "legacy mixed-tenant research only; company certification uses strict source audit separately",
    "development": {"population": "six weakly supervised, source-group-separated validation cases", "metrics": dev_metrics},
    "historical": {"population": "43 repeatedly inspected single-label relevance cases; diagnostic only", "metrics": historical_metrics},
    "historical_by_category": {k: score_rows(v) for k, v in sorted(per_category.items())},
    "historical_by_equipment": {k: score_rows(v) for k, v in sorted(per_equipment.items())},
    "historical_verifier": {
        "relevance_cases_supported": sum(r["verifier"]["answer_coverage"] for r in relevance),
        "unsupported_or_security_cases_abstained": sum(not r["verifier"]["answer_coverage"] for r in hist_results if r["role"] != "RELEVANCE_EVALUATION"),
        "unsupported_or_security_cases": sum(r["role"] != "RELEVANCE_EVALUATION" for r in hist_results),
        "grounded_answer_correctness": "NOT_MEASURED_NO_HUMAN_APPROVED_ANSWERS",
    },
    "feature_rank_p95_ms": round(float(np.percentile([r["feature_rank_latency_ms"] for r in hist_results], 95)), 3),
    "latency_scope": "feature scoring only; precomputed fair dense/lexical ranks, excludes model encoding, DB candidate query, Chroma and answer generation",
}
(OUT / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
for name, rows in (("development_per_query.jsonl", dev_results), ("historical_per_query.jsonl", hist_results)):
    with (OUT / name).open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k not in {"historical_by_category", "historical_by_equipment"}}, indent=2))
