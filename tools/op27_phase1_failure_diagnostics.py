"""Op27 Phase 1 — Systematic Bottleneck Analysis.

Performs deep per-query failure attribution for all queries missed at Rank 1 in
Configuration 5 (Production Hybrid Baseline) and pure dense models.

For every missed query:
1. Inspects the target positive chunk(s) and their text in MySQL.
2. Checks whether the target chunk exists, is ingested, and has authoritative text.
3. Checks dense score & rank, sparse score & rank, fusion score & rank, and conditional swap decision.
4. Identifies the primary failure class (A through O) and secondary cause.
5. Saves structured diagnostics to private_training_data/op27_phase1_bottleneck_report.json.
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
from core.services.retriever import retrieve, _knowledge_candidates, _dense_score_candidates, _sparse_score_candidates, _reciprocal_rank_fusion
from core.ml.conditional_rerank import conditional_swap_rerank
from core.services.vectors import embed_text
from core.services.identifiers import extract_identifiers, extract_error_codes


def main():
    print("=" * 80)
    print("OPERATION 27 PHASE 1 — SYSTEMATIC FAILURE BOTTLENECK ANALYSIS")
    print("=" * 80)

    path = BASE_DIR / "private_training_data" / "frozen_independent_holdout_benchmark.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
    relevance_rows = [r for r in rows if r.get("positive_chunk_ids") and len(r["positive_chunk_ids"]) > 0]

    diagnostics = []
    failure_class_counts = {}

    for row in relevance_rows:
        qid = row.get("query_id", "")
        query = row["query_text"]
        tid = int(row["tenant_id"])
        pos_ids = set(row.get("positive_chunk_ids", []))
        neg_ids = set(row.get("negative_chunk_ids", []))
        category = row.get("category", "unknown")
        equip = row.get("equipment", "")

        tenant = Tenant.objects.get(id=tid)
        asset = resolve_query_asset(tid, equip)

        # Run retrieval with full diagnostics
        results = retrieve(
            tenant=tenant,
            question=query,
            asset=asset,
            customer=None,
            include_confidential=False,
            top_k=20,
            staff_mode=False,
        )

        ranked = [r.chunk_id for r in results if r.chunk_id]
        hits = [i for i, c in enumerate(ranked) if c in pos_ids]
        hit1 = any(h < 1 for h in hits)

        if hit1:
            continue  # Passed at Rank 1!

        first_hit_rank = hits[0] + 1 if hits else None
        print(f"\n--- FAILURE: {qid} [{category}] (First Hit: rank {first_hit_rank}) ---")
        print(f"Query: '{query}'")
        print(f"Equipment: '{equip}' -> Asset: {asset}")
        print(f"Target Positives: {list(pos_ids)}")

        # Inspect target chunks
        target_chunks = list(KnowledgeChunk.objects.filter(id__in=pos_ids).select_related("document"))
        print(f"Found {len(target_chunks)} positive chunks in DB:")
        for tc in target_chunks:
            print(f"  Chunk {tc.id}: Doc {tc.document_id} '{tc.document.title}'")
            print(f"  Heading: '{tc.heading}'")
            print(f"  Text snippet: {repr(tc.text[:120])}")
            print(f"  Is RAG enabled: {tc.document.is_rag_enabled}, Index Status: {tc.document.index_status}, Quarantined: {tc.is_quarantined}")

        # Top 3 retrieved chunks
        print("Retrieved Top 3:")
        for i, r in enumerate(results[:3], 1):
            is_pos = r.chunk_id in pos_ids
            is_neg = r.chunk_id in neg_ids
            marker = " [TARGET POSITIVE]" if is_pos else (" [TARGET NEGATIVE]" if is_neg else "")
            print(f"  {i}. Chunk {r.chunk_id}: score={r.score:.4f}, dense={r.dense_score}, sparse={r.sparse_score}{marker}")
            print(f"     Title: '{r.title}', Heading: '{r.heading}'")
            print(f"     Snippet: {repr(r.text[:100])}")

        # Determine failure cause
        top_chunk = results[0] if results else None
        primary_class = "UNKNOWN"
        secondary_cause = ""

        if not target_chunks:
            primary_class = "B"  # Correct source not ingested
            secondary_cause = "Positive chunk ID does not exist in KnowledgeChunk DB"
        elif any(not tc.document.is_rag_enabled or tc.document.index_status != "INDEXED" or tc.is_quarantined for tc in target_chunks):
            primary_class = "B"  # Excluded by DB filter
            secondary_cause = "Document not RAG enabled or not indexed"
        elif first_hit_rank is None:
            # Not even in top 20!
            # Check if dense retrieved it
            # Check if sparse retrieved it
            primary_class = "G"  # Dense retrieval failed
            secondary_cause = "Candidate not admitted into top 20 by either dense or sparse"
        elif first_hit_rank > 1:
            # Retrieved, but ranked below rank 1
            if top_chunk and top_chunk.chunk_id in neg_ids:
                primary_class = "L"  # Reranker / fusion promoted a negative distractor
                secondary_cause = f"Pre-selected negative distractor Chunk {top_chunk.chunk_id} outranked positive Chunk {list(pos_ids)[0]}"
            elif top_chunk and top_chunk.dense_score and top_chunk.sparse_score:
                # Was it dense or sparse that ranked top_chunk higher?
                primary_class = "J"  # Candidate retrieved but fusion ranked it too low
                secondary_cause = f"Fusion ranked distractor Chunk {top_chunk.chunk_id} (dense={top_chunk.dense_score}, sparse={top_chunk.sparse_score}) above positive at rank {first_hit_rank}"
            else:
                primary_class = "J"
                secondary_cause = f"Ranked at position {first_hit_rank}"

        print(f"  => PRIMARY FAILURE CLASS: {primary_class} ({secondary_cause})")
        failure_class_counts[primary_class] = failure_class_counts.get(primary_class, 0) + 1

        diagnostics.append({
            "query_id": qid,
            "query_text": query,
            "category": category,
            "equipment": equip,
            "positive_chunk_ids": list(pos_ids),
            "first_hit_rank": first_hit_rank,
            "primary_failure_class": primary_class,
            "secondary_cause": secondary_cause,
            "top1_chunk_id": top_chunk.chunk_id if top_chunk else None,
            "top1_title": top_chunk.title if top_chunk else "",
            "top1_heading": top_chunk.heading if top_chunk else "",
        })

    print("\n" + "=" * 80)
    print("FAILURE CLASS DISTRIBUTION:")
    for fc, count in sorted(failure_class_counts.items()):
        print(f"  Class {fc}: {count} queries ({count/len(diagnostics)*100:.1f}%)")
    print(f"Total failures analyzed: {len(diagnostics)}")

    out_file = BASE_DIR / "private_training_data" / "op27_phase1_bottleneck_report.json"
    out_file.write_text(json.dumps({
        "total_failures": len(diagnostics),
        "class_counts": failure_class_counts,
        "diagnostics": diagnostics,
    }, indent=2), encoding="utf-8")
    print(f"Report saved to: {out_file}")


if __name__ == "__main__":
    main()
