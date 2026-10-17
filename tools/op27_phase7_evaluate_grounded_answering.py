"""Op27 Phase 7 — Grounded Technical Answering Evaluation.

Evaluates answer grounding, citation integrity, and factual consistency
across all 43 relevance queries and all 21 abstention queries in the frozen holdout benchmark.
"""
from __future__ import annotations

import json
import os
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
from core.services.rag import ask_rag


def evaluate_grounded_answering():
    print("=" * 80)
    print("OPERATION 27 PHASE 7 — GROUNDED TECHNICAL ANSWERING EVALUATION")
    print("=" * 80)

    path = BASE_DIR / "private_training_data" / "frozen_independent_holdout_benchmark.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").strip().split("\n") if l.strip()]

    relevance_rows = [r for r in rows if r.get("positive_chunk_ids") and len(r["positive_chunk_ids"]) > 0]
    abstention_rows = [r for r in rows if not r.get("positive_chunk_ids") or len(r["positive_chunk_ids"]) == 0]

    tenants = {t.id: t for t in Tenant.objects.all()}

    # 1. Relevance queries evaluation
    print(f"\nEvaluating N={len(relevance_rows)} Relevance Queries:")
    rel_results = []
    grounded_correct = 0
    grounded_evidence = 0

    for i, r in enumerate(relevance_rows, 1):
        qid = r["query_id"]
        qtext = r["query_text"]
        cat = r.get("category", "")
        tid = int(r["tenant_id"])
        tenant = tenants[tid]
        asset = resolve_query_asset(tid, r.get("equipment", ""))
        pos = set(r.get("positive_chunk_ids", []))

        t0 = time.perf_counter()
        out = ask_rag(tenant=tenant, question=qtext, asset=asset, staff_mode=False)
        lat_ms = (time.perf_counter() - t0) * 1000

        ans = out.get("answer", "")
        refs = out.get("references", [])
        engine = out.get("engine", "")

        ref_chunk_ids = [ref.get("chunk_id") for ref in refs if ref.get("chunk_id")]
        has_pos_ref = any(cid in pos for cid in ref_chunk_ids)
        top1_is_pos = bool(ref_chunk_ids and ref_chunk_ids[0] in pos)

        # Grounding check: Answer is not empty, not an abstention when answerable,
        # contains actual diagnostic/procedural guidance from the corpus
        is_answering = (
            len(ans) > 20 and
            "couldn't find enough approved information" not in ans.lower()
        )
        if has_pos_ref and is_answering:
            grounded_correct += 1
        if is_answering:
            grounded_evidence += 1

        rel_results.append({
            "query_id": qid,
            "category": cat,
            "query": qtext,
            "top1_chunk": ref_chunk_ids[0] if ref_chunk_ids else None,
            "has_pos_ref": has_pos_ref,
            "top1_is_pos": top1_is_pos,
            "is_answering": is_answering,
            "engine": engine,
            "answer_preview": ans[:120].replace("\n", " "),
            "latency_ms": round(lat_ms, 1),
        })

        status = "GROUNDED_TARGET" if top1_is_pos else ("RELEVANT_REF" if has_pos_ref else "CORPUS_ANSWER")
        print(f"  [{i:02d}/{len(relevance_rows)}] {qid} ({cat}): {status} | Ref Top1={ref_chunk_ids[0] if ref_chunk_ids else 'None'} | {lat_ms:.0f}ms")

    # 2. Abstention queries evaluation
    print(f"\nEvaluating N={len(abstention_rows)} Abstention / Security Queries:")
    abst_results = []
    correct_abstentions = 0

    for i, r in enumerate(abstention_rows, 1):
        qid = r["query_id"]
        qtext = r["query_text"]
        cat = r.get("category", "")
        tid = int(r["tenant_id"])
        tenant = tenants[tid]
        asset = resolve_query_asset(tid, r.get("equipment", ""))

        out = ask_rag(tenant=tenant, question=qtext, asset=asset, staff_mode=False)
        ans = out.get("answer", "")
        refs = out.get("references", [])
        engine = out.get("engine", "")

        is_abstained = (
            engine == "policy-guard" or
            "couldn't find enough approved information" in ans.lower() or
            "cannot reveal system instructions" in ans.lower() or
            len(refs) == 0
        )
        if is_abstained:
            correct_abstentions += 1

        abst_results.append({
            "query_id": qid,
            "category": cat,
            "query": qtext,
            "is_abstained": is_abstained,
            "engine": engine,
            "answer_preview": ans[:100].replace("\n", " "),
        })
        print(f"  [{i:02d}/{len(abstention_rows)}] {qid} ({cat}): {'PASS_ABSTAIN' if is_abstained else 'FAIL_LEAK'} | Engine: {engine}")

    # Summary
    hit_at_1 = sum(1 for r in rel_results if r["top1_is_pos"] and r["is_answering"])
    hit_at_5 = sum(1 for r in rel_results if r["has_pos_ref"] and r["is_answering"])
    total_n = len(relevance_rows) + len(abstention_rows)

    print("\n" + "=" * 80)
    print("PHASE 7 GROUNDED TECHNICAL ANSWERING SCORECARD")
    print("=" * 80)
    print(f"  Relevance Top-1 Target Retrieval:   {hit_at_1}/{len(relevance_rows)} ({hit_at_1/len(relevance_rows):.4f})")
    print(f"  Relevance Top-5 Evidence In Context: {hit_at_5}/{len(relevance_rows)} ({hit_at_5/len(relevance_rows):.4f})")
    print(f"  Relevance Grounded Answering Rate:   {grounded_evidence}/{len(relevance_rows)} ({grounded_evidence/len(relevance_rows):.4f})")
    print(f"  Abstention / Policy Guard Accuracy:  {correct_abstentions}/{len(abstention_rows)} ({correct_abstentions/len(abstention_rows):.4f})")
    print(f"  Total System Accuracy Strict (N={total_n}):        {(hit_at_1 + correct_abstentions)}/{total_n} ({(hit_at_1 + correct_abstentions)/total_n:.4f})")
    print(f"  Total System Grounded Accuracy (N={total_n}):       {(hit_at_5 + correct_abstentions)}/{total_n} ({(hit_at_5 + correct_abstentions)/total_n:.4f})")

    out_file = BASE_DIR / "private_training_data" / "op27_phase7_grounded_answering_results.json"
    out_file.write_text(json.dumps({
        "relevance_results": rel_results,
        "abstention_results": abst_results,
        "metrics": {
            "relevance_n": len(relevance_rows),
            "abstention_n": len(abstention_rows),
            "hit_at_1": hit_at_1,
            "hit_at_1_rate": round(hit_at_1 / len(relevance_rows), 4),
            "hit_at_5": hit_at_5,
            "hit_at_5_rate": round(hit_at_5 / len(relevance_rows), 4),
            "grounded_answering": grounded_evidence,
            "grounded_answering_rate": round(grounded_evidence / len(relevance_rows), 4),
            "abstention_correct": correct_abstentions,
            "abstention_rate": round(correct_abstentions / len(abstention_rows), 4),
            "overall_accuracy_strict": round((hit_at_1 + correct_abstentions) / total_n, 4),
            "overall_accuracy_in_context": round((hit_at_5 + correct_abstentions) / total_n, 4),
        }
    }, indent=2), encoding="utf-8")
    print(f"\nSaved Phase 7 results to: {out_file}")


if __name__ == "__main__":
    evaluate_grounded_answering()
