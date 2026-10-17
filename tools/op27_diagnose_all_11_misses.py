"""Op27 Phase 6 — Deep Inspection of All 11 Missing Queries.

Analyzes the exact candidate ranking, headings, text content, and ranking
decisions for all 11 queries missed in Exp 1.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
os.environ.setdefault("SERVY_DB_ENGINE", "mysql")

import django
django.setup()

from core.models import Tenant, KnowledgeChunk
from core.ml.benchmark_manifest import resolve_query_asset
from core.services.retriever import retrieve

holdout_path = BASE_DIR / "private_training_data" / "frozen_independent_holdout_benchmark.jsonl"
rows = [json.loads(l) for l in holdout_path.read_text(encoding="utf-8").strip().split("\n") if l.strip()]

target_qids = {
    "EVAL-TRAP-02", "EVAL-TRAP-03", "EVAL-TRAP-06",
    "EVAL-MSP-02", "EVAL-MLT-01", "EVAL-MLT-04", "EVAL-MLT-05",
    "EVAL-AMB-03", "EVAL-TECH-03", "EVAL-XTLK-01", "EVAL-XTLK-02",
}

selected = [r for r in rows if r.get("query_id") in target_qids]
tenants = {t.id: t for t in Tenant.objects.all()}

print(f"Analyzing {len(selected)} target queries...")

for r in selected:
    qid = r["query_id"]
    q = r["query_text"]
    tid = int(r["tenant_id"])
    pos_ids = set(r.get("positive_chunk_ids", []))
    cat = r.get("category", "")
    equip = r.get("equipment", "")

    tenant = tenants[tid]
    asset = resolve_query_asset(tid, equip)

    results = retrieve(tenant=tenant, question=q, asset=asset, top_k=10, staff_mode=False)

    ranked_cids = [res.chunk_id for res in results]
    hits = [i + 1 for i, cid in enumerate(ranked_cids) if cid in pos_ids]
    target_pos = KnowledgeChunk.objects.filter(id__in=pos_ids).first()

    print("\n" + "=" * 80)
    print(f"QUERY [{qid}] Category: {cat}")
    print(f"Query text: '{q}'")
    print(f"Equipment: '{equip}' -> Asset: {asset}")
    print(f"Target Positive Chunk: {list(pos_ids)} (Found at rank: {hits[0] if hits else 'NOT IN TOP 10'})")
    if target_pos:
        print(f"  Target Heading: '{target_pos.heading}'")
        print(f"  Target Text: {repr(target_pos.text[:150])}")

    print("Top 3 Retrieved Candidates:")
    for i, res in enumerate(results[:3], 1):
        is_target = res.chunk_id in pos_ids
        print(f"  Rank {i}: Chunk {res.chunk_id} {'[TARGET!]' if is_target else ''}")
        print(f"    Title: '{res.title}'")
        print(f"    Heading: '{res.heading}'")
        print(f"    Score: {res.score:.4f} (dense={res.dense_score}, sparse={res.sparse_score})")
        print(f"    Snippet: {repr(res.text[:120])}")
