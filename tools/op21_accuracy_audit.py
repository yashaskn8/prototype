"""Read-only Operation 21 retrieval audit over one SQLite snapshot.

Private query diagnostics are written under private_training_data (gitignored).
This script never trains, modifies source records, or changes Chroma collections.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SERVY_DB_ENGINE", "sqlite3")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")

import django

django.setup()

import numpy as np
from core.ml.benchmark_manifest import ROLE_RELEVANCE, build_manifest, resolve_query_asset
from core.ml.benchmark_registry import load_and_verify_benchmark
from core.ml.embeddings import EmbeddingProfile, BGE_QUERY_PREFIX, get_profile, load_model, passage_text
from core.ml.indexer import _eligible_chunks
from core.models import KnowledgeChunk, Tenant
from core.services.retriever import _knowledge_candidates
from core.services.vectors import compute_bm25_scores, compute_char_ngram_scores


OUT = ROOT / "private_training_data" / "op21_accuracy_audit"
OUT.mkdir(parents=True, exist_ok=True)


def metrics(rows, key):
    rel = [r for r in rows if r["role"] == ROLE_RELEVANCE]
    positions = []
    for row in rel:
        positives = set(row["positive_chunk_ids"])
        ranking = row[key]
        positions.append(next((i + 1 for i, x in enumerate(ranking) if x in positives), None))
    n = len(positions)
    return {
        "n": n,
        "hit_at_1": round(sum(p == 1 for p in positions) / n, 4),
        "hit_at_5": round(sum(p is not None and p <= 5 for p in positions) / n, 4),
        "recall_at_10": round(sum(p is not None and p <= 10 for p in positions) / n, 4),
        "recall_at_50": round(sum(p is not None and p <= 50 for p in positions) / n, 4),
        "mrr_at_10": round(sum(1 / p for p in positions if p and p <= 10) / n, 4),
        "ndcg_at_10": round(sum(1 / math.log2(p + 1) for p in positions if p and p <= 10) / n, 4),
    }


def main():
    verified = load_and_verify_benchmark("servy-real-holdout-v1")
    manifest = build_manifest("servy-real-holdout-v1")
    cases = manifest["cases"]
    chunks = list(_eligible_chunks())
    by_id = {c.id: c for c in chunks}
    all_chunks = {c.id: c for c in KnowledgeChunk.objects.select_related("document").all()}
    tenants = {t.id: t for t in Tenant.objects.all()}
    print(f"Benchmark={verified['sha256_digest']} cases={len(cases)} eligible={len(chunks)}", flush=True)

    records = []
    for case in cases:
        asset = resolve_query_asset(case["tenant_id"], case["equipment"])
        authorized = _knowledge_candidates(
            tenant=tenants[case["tenant_id"]], asset=asset, customer=None,
            include_confidential=False, staff_mode=False,
        )
        authorized_ids = {c.id for c in authorized}
        eligible_authorized_ids = authorized_ids & set(by_id)
        positives = set(case["positive_chunk_ids"])
        records.append({
            "query_id": case["query_id"], "query_text": case["query_text"],
            "tenant_id": case["tenant_id"], "equipment": case["equipment"],
            "asset_id": asset.id if asset else None, "category": case["category"],
            "role": case["primary_evaluation_role"],
            "positive_chunk_ids": sorted(positives),
            "positive_in_database": sorted(positives & set(all_chunks)),
            "positive_in_eligible_corpus": sorted(positives & set(by_id)),
            "positive_authorized": sorted(positives & authorized_ids),
            "positive_eligible_authorized": sorted(positives & eligible_authorized_ids),
            "authorized_candidate_count": len(authorized_ids),
            "eligible_authorized_candidate_count": len(eligible_authorized_ids),
            "_authorized_ids": eligible_authorized_ids,
        })

    profiles = {
        "fair_minilm": get_profile("minilm"),
        "fair_base_bge": get_profile("bge-small"),
        "fair_op20_bge_ft": EmbeddingProfile(
            key="op21-op20-bge-ft", model_id=str(ROOT / "ml_models" / "checkpoints" / "op20_company_bge_ft" / "final"),
            revision=None, dim=384, collection="unused_op21_in_memory",
            query_prefix=BGE_QUERY_PREFIX, content_first=True,
            local_path=str(ROOT / "ml_models" / "checkpoints" / "op20_company_bge_ft" / "final"),
        ),
    }
    timings = {}
    for name, profile in profiles.items():
        print(f"Encoding {name}...", flush=True)
        t0 = time.perf_counter()
        model = load_model(profile)
        texts = [passage_text(profile, c) for c in chunks]
        questions = [profile.query_prefix + r["query_text"] for r in records]
        cv = np.asarray(model.encode(texts, normalize_embeddings=True, batch_size=8), dtype=np.float32)
        qv = np.asarray(model.encode(questions, normalize_embeddings=True, batch_size=8), dtype=np.float32)
        scores = qv @ cv.T
        for qi, row in enumerate(records):
            indices = [i for i, c in enumerate(chunks) if c.id in row["_authorized_ids"]]
            ranked = sorted(indices, key=lambda i: (-float(scores[qi, i]), chunks[i].id))
            row[name] = [chunks[i].id for i in ranked]
            row[name + "_top_10"] = [
                {"chunk_id": chunks[i].id, "cosine_similarity": round(float(scores[qi, i]), 5)}
                for i in ranked[:10]
            ]
        timings[name] = round(time.perf_counter() - t0, 2)
        del model, cv, qv, scores

    print("Scoring lexical and fixed RRF...", flush=True)
    for row in records:
        allowed = [c for c in chunks if c.id in row["_authorized_ids"]]
        passages = [f"{c.heading}\n{c.text}" for c in allowed]
        bm25 = compute_bm25_scores(row["query_text"], passages)
        char = compute_char_ngram_scores(row["query_text"], passages)
        lexical = [(max(float(bm25[i]), 0.5 * float(char[i])), c.id) for i, c in enumerate(allowed)]
        lexical.sort(key=lambda item: (-item[0], item[1]))
        row["bm25"] = [allowed[i].id for i in sorted(range(len(allowed)), key=lambda i: (-bm25[i], allowed[i].id))]
        row["lexical"] = [cid for _, cid in lexical]
        row["lexical_top_10"] = [{"chunk_id": cid, "score": round(score, 5)} for score, cid in lexical[:10]]
        dense = row["fair_op20_bge_ft"][:50]
        sparse = row["lexical"][:50]
        fused = defaultdict(float)
        for rank, cid in enumerate(dense, 1):
            fused[cid] += 0.55 / (60 + rank)
        for rank, cid in enumerate(sparse, 1):
            fused[cid] += 0.45 / (60 + rank)
        row["fixed_rrf"] = sorted(fused, key=lambda cid: (-fused[cid], cid))
        row["fixed_rrf_top_10"] = [
            {"chunk_id": cid, "rrf_score": round(fused[cid], 7)} for cid in row["fixed_rrf"][:10]
        ]

        positives = set(row["positive_chunk_ids"])
        if row["role"] != ROLE_RELEVANCE:
            row["failure_mechanism"] = "abstention_or_security_case; retrieval hit metrics excluded"
        elif not row["positive_in_database"]:
            row["failure_mechanism"] = "labeled_positive_missing_from_database"
        elif not row["positive_in_eligible_corpus"]:
            row["failure_mechanism"] = "labeled_positive_not_index_eligible"
        elif not row["positive_authorized"]:
            row["failure_mechanism"] = "labeled_positive_excluded_by_authorization_or_applicability"
        elif positives & set(row["fair_op20_bge_ft"][:1]):
            row["failure_mechanism"] = "labeled_hit_at_1"
        elif positives & set(row["fair_op20_bge_ft"][:10]):
            row["failure_mechanism"] = "ranking_error_positive_in_dense_top_10"
        elif positives & set(row["fair_op20_bge_ft"][:50]):
            row["failure_mechanism"] = "ranking_error_positive_in_dense_11_to_50"
        elif positives & set(row["lexical"][:50]):
            row["failure_mechanism"] = "dense_miss_lexical_candidate_available"
        else:
            row["failure_mechanism"] = "candidate_miss_at_depth_50"
        row.pop("_authorized_ids")

    rel = [r for r in records if r["role"] == ROLE_RELEVANCE]
    counts = Counter(r["failure_mechanism"] for r in rel)
    coverage = {
        "database_positive": sum(bool(r["positive_in_database"]) for r in rel),
        "index_eligible_positive": sum(bool(r["positive_in_eligible_corpus"]) for r in rel),
        "authorized_positive": sum(bool(r["positive_authorized"]) for r in rel),
        "eligible_authorized_positive": sum(bool(r["positive_eligible_authorized"]) for r in rel),
        "dense_top_50_positive": sum(bool(set(r["positive_chunk_ids"]) & set(r["fair_op20_bge_ft"][:50])) for r in rel),
        "lexical_top_50_positive": sum(bool(set(r["positive_chunk_ids"]) & set(r["lexical"][:50])) for r in rel),
        "union_top_50_positive": sum(bool(set(r["positive_chunk_ids"]) & (set(r["fair_op20_bge_ft"][:50]) | set(r["lexical"][:50]))) for r in rel),
    }
    summary = {
        "benchmark_id": verified["dataset_id"], "benchmark_sha256": verified["sha256_digest"],
        "dataset_role": verified["dataset_role"], "n_cases": len(records), "n_relevance": len(rel),
        "eligible_chunk_count": len(chunks), "eligible_documents": len({c.document_id for c in chunks}),
        "model_encoding_wall_seconds": timings, "stage_coverage_counts": coverage,
        "failure_mechanism_counts": counts,
        "metrics": {k: metrics(records, k) for k in ["fair_minilm", "fair_base_bge", "fair_op20_bge_ft", "bm25", "lexical", "fixed_rrf"]},
        "by_category": {
            category: {k: metrics(rows, k) for k in ["fair_minilm", "fair_base_bge", "fair_op20_bge_ft", "lexical", "fixed_rrf"]}
            for category, rows in sorted({r["category"]: [x for x in rel if x["category"] == r["category"]] for r in rel}.items())
        },
        "by_equipment": {
            equipment: {k: metrics(rows, k) for k in ["fair_minilm", "fair_base_bge", "fair_op20_bge_ft", "lexical", "fixed_rrf"]}
            for equipment, rows in sorted({r["equipment"]: [x for x in rel if x["equipment"] == r["equipment"]] for r in rel}.items())
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=list), encoding="utf-8")
    with (OUT / "per_query.jsonl").open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    print(json.dumps({"coverage": coverage, "mechanisms": counts, "metrics": summary["metrics"], "seconds": timings}, indent=2), flush=True)


if __name__ == "__main__":
    main()
