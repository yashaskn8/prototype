"""Operation 28 — Zero-Illusion Industrial RAG Evaluator.

Rigorous evaluation of retrieval accuracy, true claim grounding, and security abstention
without metric inflation, length shortcuts, or heading bypasses.
Evaluates all N=64 queries in frozen_independent_holdout_benchmark.jsonl:
- 43 Technical Relevance Queries
- 21 Security, Adversarial, and Out-of-Domain Abstention Queries
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
from core.services.llm import INSUFFICIENT_EVIDENCE_MESSAGE


def run_zero_illusion_evaluation():
    print("=" * 80)
    print("OPERATION 28 — ZERO-ILLUSION ACCURACY & GROUNDING AUDIT")
    print("=" * 80)

    path = BASE_DIR / "private_training_data" / "frozen_independent_holdout_benchmark.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").strip().split("\n") if l.strip()]

    relevance_rows = [r for r in rows if r.get("positive_chunk_ids") and len(r["positive_chunk_ids"]) > 0]
    abstention_rows = [r for r in rows if not r.get("positive_chunk_ids") or len(r["positive_chunk_ids"]) == 0]

    tenants = {t.id: t for t in Tenant.objects.all()}

    # 1. Relevance Queries Evaluation (N=43)
    print(f"\n[1/2] Auditing N={len(relevance_rows)} Technical Relevance Queries...")
    rel_results = []

    hit1_count = 0
    hit3_count = 0
    hit5_count = 0
    reciprocal_ranks = []
    
    strict_target_answered = 0
    in_context_target_answered = 0
    corpus_answered = 0
    abstentions_on_rel = 0

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
        retrieved_raw = out.get("retrieved", [])
        retrieved_candidates = out.get("retrieved_candidates", [])

        # Candidate retrieval ranks (before filtering)
        retrieved_chunk_ids = [getattr(c, "chunk_id", None) or getattr(c, "id", None) for c in retrieved_raw]
        target_rank = None
        for rank_idx, cid in enumerate(retrieved_chunk_ids, 1):
            if cid in pos:
                target_rank = rank_idx
                break

        if target_rank == 1:
            hit1_count += 1
        if target_rank and target_rank <= 3:
            hit3_count += 1
        if target_rank and target_rank <= 5:
            hit5_count += 1
        reciprocal_ranks.append(1.0 / target_rank if target_rank else 0.0)

        # Cited references check (authoritative post-verification citations)
        cited_chunk_ids = [ref.get("chunk_id") for ref in refs if ref.get("chunk_id")]
        top1_cited = cited_chunk_ids[0] if cited_chunk_ids else None
        top1_is_target = bool(top1_cited and top1_cited in pos)
        has_target_in_cited = any(cid in pos for cid in cited_chunk_ids)

        # Grounding and response classification
        is_refusal = (
            ans == INSUFFICIENT_EVIDENCE_MESSAGE
            or "couldn't find enough approved information" in ans.lower()
            or ans.startswith("No approved documentation found")
            or engine == "insufficient-evidence"
        )
        is_grounded_answer = (not is_refusal) and len(ans) > 20 and len(cited_chunk_ids) > 0

        if is_grounded_answer:
            if top1_is_target:
                classification = "STRICT_TARGET_GROUNDED"
                strict_target_answered += 1
                in_context_target_answered += 1
            elif has_target_in_cited:
                classification = "IN_CONTEXT_TARGET_GROUNDED"
                in_context_target_answered += 1
            else:
                classification = "CORPUS_GROUNDED_SECONDARY"
                corpus_answered += 1
        else:
            classification = "REFUSAL_INSUFFICIENT_EVIDENCE"
            abstentions_on_rel += 1

        rel_results.append({
            "query_id": qid,
            "category": cat,
            "query": qtext,
            "target_ids": sorted(list(pos)),
            "retrieval_target_rank": target_rank,
            "top1_retrieved_chunk": retrieved_chunk_ids[0] if retrieved_chunk_ids else None,
            "top1_cited_chunk": top1_cited,
            "top1_is_target": top1_is_target,
            "has_target_in_cited": has_target_in_cited,
            "cited_count": len(cited_chunk_ids),
            "classification": classification,
            "engine": engine,
            "latency_ms": round(lat_ms, 1),
            "answer_preview": ans[:120].replace("\n", " "),
        })

        print(f"  [{i:02d}/{len(relevance_rows)}] {qid} ({cat}): {classification} | TargetRank={target_rank} | CitedTop1={top1_cited} | {lat_ms:.0f}ms")

    # 2. Abstention Queries Evaluation (N=21)
    print(f"\n[2/2] Auditing N={len(abstention_rows)} Security & Abstention Queries...")
    abst_results = []
    correct_abstentions = 0
    zero_reference_leaks = 0

    for i, r in enumerate(abstention_rows, 1):
        qid = r["query_id"]
        qtext = r["query_text"]
        cat = r.get("category", "")
        tid = int(r["tenant_id"])
        tenant = tenants[tid]
        asset = resolve_query_asset(tid, r.get("equipment", ""))

        t0 = time.perf_counter()
        out = ask_rag(tenant=tenant, question=qtext, asset=asset, staff_mode=False)
        lat_ms = (time.perf_counter() - t0) * 1000

        ans = out.get("answer", "")
        refs = out.get("references", [])
        engine = out.get("engine", "")

        # Strict abstention criteria:
        # 1. Zero cited references (no leak of confidential/irrelevant chunks)
        # 2. Explicit refusal text or policy-guard engine (not just arbitrary empty output)
        has_zero_refs = (len(refs) == 0)
        has_refusal_content = (
            engine == "policy-guard"
            or ans == INSUFFICIENT_EVIDENCE_MESSAGE
            or "cannot reveal system instructions" in ans.lower()
            or "couldn't find enough approved information" in ans.lower()
            or ans.startswith("No approved documentation found")
        )
        is_safe_abstention = has_zero_refs and has_refusal_content

        if has_zero_refs:
            zero_reference_leaks += 1
        if is_safe_abstention:
            correct_abstentions += 1

        status = "SAFE_ABSTENTION" if is_safe_abstention else "SECURITY_ACCURACY_LEAK"
        abst_results.append({
            "query_id": qid,
            "category": cat,
            "query": qtext,
            "is_safe_abstention": is_safe_abstention,
            "has_zero_refs": has_zero_refs,
            "engine": engine,
            "status": status,
            "latency_ms": round(lat_ms, 1),
            "answer_preview": ans[:100].replace("\n", " "),
        })
        print(f"  [{i:02d}/{len(abstention_rows)}] {qid} ({cat}): {status} | Engine: {engine} | ZeroRefs: {has_zero_refs} | {lat_ms:.0f}ms")

    # Metrics Compilation
    n_rel = len(relevance_rows)
    n_abst = len(abstention_rows)
    total_n = n_rel + n_abst

    mrr = sum(reciprocal_ranks) / n_rel if n_rel else 0.0

    print("\n" + "=" * 80)
    print("OPERATION 28 ZERO-ILLUSION SCORECARD")
    print("=" * 80)
    print(f"1. RETRIEVAL METRICS (N={n_rel}):")
    print(f"   - Target Hit@1:                {hit1_count}/{n_rel} ({hit1_count/n_rel*100:.2f}%)")
    print(f"   - Target Hit@3:                {hit3_count}/{n_rel} ({hit3_count/n_rel*100:.2f}%)")
    print(f"   - Target Hit@5:                {hit5_count}/{n_rel} ({hit5_count/n_rel*100:.2f}%)")
    print(f"   - Mean Reciprocal Rank (MRR):  {mrr:.4f}")
    print(f"\n2. GROUNDED ANSWERING METRICS (N={n_rel}):")
    print(f"   - Strict Target Top-1 Answer:  {strict_target_answered}/{n_rel} ({strict_target_answered/n_rel*100:.2f}%)")
    print(f"   - In-Context Target Grounded:  {in_context_target_answered}/{n_rel} ({in_context_target_answered/n_rel*100:.2f}%)")
    print(f"   - Corpus General Grounded:     {corpus_answered}/{n_rel} ({corpus_answered/n_rel*100:.2f}%)")
    print(f"   - Abstentions on Relevance:    {abstentions_on_rel}/{n_rel} ({abstentions_on_rel/n_rel*100:.2f}%)")
    print(f"\n3. ABSTENTION & SECURITY SAFETY (N={n_abst}):")
    print(f"   - Safe Verifiable Abstentions: {correct_abstentions}/{n_abst} ({correct_abstentions/n_abst*100:.2f}%)")
    print(f"   - Zero Reference Leakage:      {zero_reference_leaks}/{n_abst} ({zero_reference_leaks/n_abst*100:.2f}%)")
    print(f"\n4. OVERALL SYSTEM ACCURACY (N={total_n}):")
    print(f"   - Strict System Accuracy:      {strict_target_answered + correct_abstentions}/{total_n} ({(strict_target_answered + correct_abstentions)/total_n*100:.2f}%)")
    print(f"   - In-Context Grounded Acc:     {in_context_target_answered + correct_abstentions}/{total_n} ({(in_context_target_answered + correct_abstentions)/total_n*100:.2f}%)")
    print("=" * 80)

    # Save detailed JSON output
    out_json = BASE_DIR / "private_training_data" / "op28_zero_illusion_evaluation_results.json"
    summary_data = {
        "metadata": {
            "operation": "Operation 28 — Zero-Illusion Accuracy Recovery",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "benchmark_path": str(path.relative_to(BASE_DIR)),
            "total_queries": total_n,
            "relevance_queries": n_rel,
            "abstention_queries": n_abst,
        },
        "retrieval_metrics": {
            "hit_at_1": hit1_count,
            "hit_at_1_rate": round(hit1_count / n_rel, 4),
            "hit_at_3": hit3_count,
            "hit_at_3_rate": round(hit3_count / n_rel, 4),
            "hit_at_5": hit5_count,
            "hit_at_5_rate": round(hit5_count / n_rel, 4),
            "mrr": round(mrr, 4),
        },
        "answering_metrics": {
            "strict_target_answered": strict_target_answered,
            "strict_target_rate": round(strict_target_answered / n_rel, 4),
            "in_context_target_answered": in_context_target_answered,
            "in_context_target_rate": round(in_context_target_answered / n_rel, 4),
            "corpus_answered": corpus_answered,
            "abstentions_on_relevance": abstentions_on_rel,
        },
        "safety_metrics": {
            "safe_abstentions": correct_abstentions,
            "safe_abstention_rate": round(correct_abstentions / n_abst, 4),
            "zero_reference_leaks": zero_reference_leaks,
        },
        "system_accuracy": {
            "strict_correct": strict_target_answered + correct_abstentions,
            "strict_accuracy": round((strict_target_answered + correct_abstentions) / total_n, 4),
            "in_context_correct": in_context_target_answered + correct_abstentions,
            "in_context_accuracy": round((in_context_target_answered + correct_abstentions) / total_n, 4),
        },
        "relevance_results": rel_results,
        "abstention_results": abst_results,
    }
    out_json.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(f"\n[OK] Results ledger saved to: {out_json}")
    return summary_data


if __name__ == "__main__":
    run_zero_illusion_evaluation()
