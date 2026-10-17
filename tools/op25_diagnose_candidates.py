"""Diagnose Candidate Recall Bottleneck for the 6 queries outside top-K."""
from __future__ import annotations

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
from core.models import Tenant, Asset, KnowledgeChunk, KnowledgeDocument
from core.ml.benchmark_registry import load_and_verify_benchmark
from core.ml.benchmark_manifest import resolve_query_asset
from core.services.retriever import (
    _knowledge_candidates, _dense_score_candidates, _sparse_score_candidates,
    normalize_domain_terms, retrieve
)
from core.ml.embeddings import get_profile, load_model, passage_text
from core.services.vectors import compute_bm25_scores

verified = load_and_verify_benchmark("servy-real-holdout-v1")
cases = {c["query_id"]: c for c in verified["cases"]}

target_qids = ["EVAL-TRAP-02", "EVAL-TRAP-06", "EVAL-AMB-01", "EVAL-AMB-03", "EVAL-TECH-03", "EVAL-XTLK-02"]

profile = get_profile("minilm")
model = load_model(profile)

findings = []

for qid in target_qids:
    case = cases[qid]
    q = case["query_text"]
    pos_ids = case.get("positive_chunk_ids", [])
    tid = case["tenant_id"]
    tenant = Tenant.objects.get(id=tid)
    asset = resolve_query_asset(tid, case.get("equipment", ""))
    gold_id = pos_ids[0] if pos_ids else None
    
    print("=" * 70)
    print(f"Query: {qid} | Equipment: {case.get('equipment')} | Gold chunk: {gold_id}")
    print(f"Query text: {q}")
    
    gold_chunk = KnowledgeChunk.objects.filter(id=gold_id).first()
    if not gold_chunk:
        print("  GOLD CHUNK NOT FOUND IN DB!")
        continue
    
    doc = gold_chunk.document
    print(f"  Gold Doc: ID={doc.id} | Title='{doc.title}' | Asset ID={doc.asset_id} | Tenant ID={doc.tenant_id}")
    print(f"  Resolved Query Asset: ID={getattr(asset, 'id', None)} | Name='{getattr(asset, 'name', None)}'")
    print(f"  Gold Chunk Heading: '{gold_chunk.heading}'")
    print(f"  Gold Chunk Snippet: {repr(gold_chunk.text[:120])}")
    
    # 1. Check candidate scoping
    scoped_candidates = list(_knowledge_candidates(tenant, asset, None, False))
    cand_ids = [c.id for c in scoped_candidates]
    in_scoped = gold_id in cand_ids
    print(f"  In _knowledge_candidates: {in_scoped} (pool size: {len(scoped_candidates)})")
    
    if not in_scoped:
        print(f"  [DISCOVERY FAILURE] Gold chunk {gold_id} was filtered out before scoring!")
        print(f"  Candidate filter check:")
        print(f"    tenant match: {doc.tenant_id == tenant.id}")
        print(f"    asset match: doc.asset_id={doc.asset_id} vs asset.id={getattr(asset, 'id', None)}")
        print(f"    doc status: doc.is_active={doc.is_active if hasattr(doc, 'is_active') else 'N/A'}")
        findings.append({
            "qid": qid,
            "gold_id": gold_id,
            "failure_point": "CANDIDATE_SCOPING",
            "details": f"doc.asset_id={doc.asset_id} != asset.id={getattr(asset, 'id', None)}",
        })
        continue
    
    # 2. Check Dense Scoring
    norm_q = normalize_domain_terms(q)
    q_vec = np.asarray(model.encode([norm_q], normalize_embeddings=True))[0]
    cand_texts = [passage_text(profile, c) for c in scoped_candidates]
    cand_vecs = np.asarray(model.encode(cand_texts, normalize_embeddings=True))
    dense_scores = cand_vecs @ q_vec
    
    gold_idx = cand_ids.index(gold_id)
    gold_dense_score = float(dense_scores[gold_idx])
    dense_rank = int(np.sum(dense_scores > gold_dense_score) + 1)
    
    # 3. Check BM25 / Sparse Scoring
    bm25_scores = np.asarray(compute_bm25_scores(norm_q, [f"{c.heading}\n{c.text}" for c in scoped_candidates]))
    gold_bm25_score = float(bm25_scores[gold_idx])
    sparse_rank = int(np.sum(bm25_scores > gold_bm25_score) + 1)
    
    print(f"  Dense:  score={gold_dense_score:.4f}, rank={dense_rank}/{len(scoped_candidates)}")
    print(f"  Sparse: score={gold_bm25_score:.4f}, rank={sparse_rank}/{len(scoped_candidates)}")
    
    # 4. Check Retriever's internal pipeline
    # Let's see what retrieve(top_k=20) yields
    results = retrieve(
        tenant=tenant, question=q, asset=asset,
        customer=None, include_confidential=False, top_k=20,
        min_score=0.01, service_call=None,
        include_past_resolutions=False, staff_mode=False,
    )
    retrieved_chunk_ids = [r.chunk_id for r in results if r.source_kind == "knowledge_document"]
    print(f"  Retrieved top 20 IDs: {retrieved_chunk_ids}")
    
    rank_in_retrieved = None
    if gold_id in retrieved_chunk_ids:
        rank_in_retrieved = retrieved_chunk_ids.index(gold_id) + 1
    print(f"  Rank in retrieve(top_k=20): {rank_in_retrieved}")
    
    findings.append({
        "qid": qid,
        "gold_id": gold_id,
        "dense_score": gold_dense_score,
        "dense_rank": dense_rank,
        "sparse_score": gold_bm25_score,
        "sparse_rank": sparse_rank,
        "rank_in_retrieved": rank_in_retrieved,
    })

print("\n" + "=" * 70)
print("Summary of 6 outside-top-K queries:")
print(json.dumps(findings, indent=2))
