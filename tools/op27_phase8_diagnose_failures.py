"""Op27 Phase 8 — Diagnose all failing benchmark queries.

For each query that failed in Phase 7 (false abstention or retrieval miss),
trace the full pipeline to identify the root cause:
  1. Retrieval: what chunks are returned and their scores
  2. Evidence sufficiency: which gate blocks the answer
  3. Claim support: what specific check fails
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
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
from core.services.llm import (
    check_evidence_sufficiency, evaluate_chunk_claim_support,
    extract_technical_numbers, EvidenceSufficiencyResult,
    INSUFFICIENT_EVIDENCE_MESSAGE, _STOPWORDS,
)
from core.services.identifiers import extract_error_codes
from core.services.retriever import normalize_domain_terms, retrieve


# Queries that failed in Phase 7 results
FAILING_QUERIES = [
    # False abstentions (retrieved correct chunk but abstained)
    "EVAL-ERR-01",    # VRF outdoor unit communication alarm fault code E-04
    "EVAL-MSP-02",    # milkk chiller temprature riseing abov 4C
    "EVAL-MSP-06",    # vrf outdor comuncation errer E04 master contrler
    "EVAL-AMB-02",    # MC-500 dairy tank agitator paddle stopped
    "EVAL-TECH-01",   # Discharge high voltage DC inverter bus capacitor safely
    # Retrieval misses (wrong chunk at top)
    "EVAL-TRAP-02",   # Ductable 5T centrifugal fan belt deflection adjustment
    "EVAL-TRAP-06",   # Split AC 2T flare nut tightness and torque specification
    "EVAL-MLT-04",    # KTM vehicle engine will not start...
]


def diagnose():
    print("=" * 80)
    print("OPERATION 27 PHASE 8 — FAILURE ROOT CAUSE DIAGNOSIS")
    print("=" * 80)

    path = BASE_DIR / "private_training_data" / "frozen_independent_holdout_benchmark.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
    failing_map = {r["query_id"]: r for r in rows if r["query_id"] in FAILING_QUERIES}

    tenants = {t.id: t for t in Tenant.objects.all()}

    for qid in FAILING_QUERIES:
        if qid not in failing_map:
            print(f"\n  SKIP {qid}: not found in benchmark")
            continue

        r = failing_map[qid]
        qtext = r["query_text"]
        tid = int(r["tenant_id"])
        tenant = tenants[tid]
        asset = resolve_query_asset(tid, r.get("equipment", ""))
        pos = set(r.get("positive_chunk_ids", []))

        print(f"\n{'='*80}")
        print(f"QUERY: {qid}")
        print(f"  Text:     {qtext}")
        print(f"  Expected: chunk_ids={sorted(pos)}")
        print(f"  Tenant:   {tenant.name} (id={tid})")
        print(f"  Asset:    {asset.name if asset else 'NONE'} (id={asset.id if asset else 'N/A'})")

        # Step 1: Retrieve chunks
        chunks = retrieve(tenant, qtext, asset=asset, top_k=8)
        print(f"\n  RETRIEVAL (top 8):")
        for i, c in enumerate(chunks, 1):
            in_pos = "Y" if c.chunk_id in pos else "N"
            score = getattr(c, 'score', 'N/A')
            heading = (getattr(c, 'heading', '') or '')[:60]
            print(f"    [{i}] Chunk {c.chunk_id} [{in_pos}] score={score:.4f}  heading='{heading}'")

        # Step 2: Evidence sufficiency
        ev = check_evidence_sufficiency(qtext, asset, chunks)
        print(f"\n  EVIDENCE SUFFICIENCY:")
        print(f"    is_sufficient: {ev.is_sufficient}")
        print(f"    reason:        {ev.reason}")
        print(f"    candidate_rel: {ev.candidate_relevance}")
        print(f"    answerability: {ev.answerability}")
        print(f"    claim_support: {ev.claim_support}")
        print(f"    equip_applic:  {ev.equipment_applicable}")
        print(f"    source_trust:  {ev.source_trust}")
        print(f"    grounded_n:    {len(ev.grounded_chunks)}")

        # Step 3: Claim support detail for top chunks
        norm_question = normalize_domain_terms(qtext or "")
        raw_terms = {w.lower() for w in re.findall(r"[A-Za-z0-9_-]{3,}", norm_question or "")}
        terms = raw_terms - _STOPWORDS
        if not terms:
            terms = raw_terms
        q_codes = extract_error_codes(qtext or "")
        q_numbers = extract_technical_numbers(qtext or "")

        print(f"\n  CLAIM SUPPORT DETAIL:")
        print(f"    query terms:   {sorted(terms)}")
        print(f"    q_codes:       {sorted(q_codes)}")
        print(f"    q_numbers:     {sorted(q_numbers)}")

        for i, c in enumerate(chunks[:5], 1):
            is_sup, reason = evaluate_chunk_claim_support(c, terms, q_codes, q_numbers)
            chunk_words = {w.lower() for w in re.findall(r"[A-Za-z0-9_-]{2,}", f"{c.heading or ''} {c.text or ''}".lower())}
            overlap = terms & chunk_words
            heading = (getattr(c, 'heading', '') or '')[:50]
            print(f"    [{i}] Chunk {c.chunk_id}: supported={is_sup}, reason={reason}")
            print(f"        overlap: {sorted(overlap)}")
            print(f"        heading: {heading}")
            # Show first 200 chars of chunk text
            text_preview = (c.text or '')[:200].replace('\n', ' ')
            print(f"        text:    {text_preview}")

        print()


if __name__ == "__main__":
    diagnose()
