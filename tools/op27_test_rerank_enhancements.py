"""Op27 Phase 6 — Test Enhanced Technical Entity & Intent Reranking.

Tests reranker enhancements on the holdout benchmark:
1. Expanded generic heading detection: adds generic checklist, pre-installation, verification, safety warnings.
2. Technical entity phrase matching: dynamic 2-gram technical matching.
3. Expanded swap_window from 5 to 10 for high-confidence matches.
4. Symptom specificity ordering: specific physical defects (ice, foam, belt, heater) override broad performance symptoms (poor cooling, noise).
5. Typo-robust domain normalization for misspelled queries.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
os.environ.setdefault("SERVY_DB_ENGINE", "mysql")
os.environ.setdefault("WANDB_DISABLED", "true")

import django
django.setup()

from core.models import Tenant, KnowledgeChunk
from core.ml.benchmark_manifest import resolve_query_asset
from core.services.retriever import retrieve
import core.ml.conditional_rerank as cr_module


# Specific physical symptoms vs broad operational symptoms
PHYSICAL_DEFECT_SYMPTOMS = frozenset({"ice", "leak", "overheating", "noise", "pressure"})
BROAD_OPERATIONAL_SYMPTOMS = frozenset({"not_cooling", "weak_airflow", "not_starting"})


def evaluate_reranker():
    print("=" * 80)
    print("EVALUATING PRODUCTION RETRIEVE WITH ENHANCED CONDITIONAL RERANKING")
    print("=" * 80)

    path = BASE_DIR / "private_training_data" / "frozen_independent_holdout_benchmark.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
    relevance_rows = [r for r in rows if r.get("positive_chunk_ids") and len(r["positive_chunk_ids"]) > 0]

    tenants = {t.id: t for t in Tenant.objects.all()}

    r1 = 0
    r3 = 0
    r5 = 0
    hits = []
    misses = []

    for r in relevance_rows:
        qid = r["query_id"]
        qtext = r["query_text"]
        cat = r.get("category", "")
        tid = int(r["tenant_id"])
        tenant = tenants[tid]
        asset = resolve_query_asset(tid, r.get("equipment", ""))
        pos = set(r.get("positive_chunk_ids", []))

        results = retrieve(tenant=tenant, question=qtext, asset=asset, top_k=20, staff_mode=False)
        ranked = [res.chunk_id for res in results if res.chunk_id]
        hit_indices = [i for i, c in enumerate(ranked) if c in pos]

        h1 = any(h < 1 for h in hit_indices)
        h3 = any(h < 3 for h in hit_indices)
        h5 = any(h < 5 for h in hit_indices)

        if h1: r1 += 1
        if h3: r3 += 1
        if h5: r5 += 1

        if h1:
            hits.append((qid, qtext, ranked[0]))
        else:
            rank_str = f"rank {hit_indices[0]+1}" if hit_indices else "NOT IN TOP 20"
            misses.append((qid, qtext, cat, ranked[0] if ranked else None, rank_str))

    n = len(relevance_rows)
    print(f"\nRESULTS ACROSS N={n} RELEVANCE QUERIES:")
    print(f"  Hit@1: {r1}/{n} ({r1/n:.4f})")
    print(f"  Hit@3: {r3}/{n} ({r3/n:.4f})")
    print(f"  Hit@5: {r5}/{n} ({r5/n:.4f})")

    print(f"\nRemaining Misses ({len(misses)}):")
    for qid, qtext, cat, top1, rank_str in misses:
        print(f"  [{cat}] {qid}: '{qtext[:50]}' -> Top1: {top1}, Target: {rank_str}")


if __name__ == "__main__":
    evaluate_reranker()

