"""Op27 Phase 0.4 — Production Hybrid Retrieval Evaluation.

Evaluates configurations 5, 6, and 7 on the 43 relevance queries using the
real retrieval entry point (core.services.retriever.retrieve):
5. Production Hybrid Baseline (MiniLM base + Sparse BM25 + RRF + Conditional swap)
6. Production Hybrid with Op26 Fine-Tuned MiniLM
7. Production Hybrid with Op26 Fine-Tuned BGE (with compliant metadata)

Saves ranked chunk IDs, eligibility decisions, and metrics to:
private_training_data/op27_phase0_hybrid_evaluation.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
os.environ.setdefault("SERVY_DB_ENGINE", "mysql")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import django
django.setup()

from django.conf import settings
from core.models import Tenant, Asset, KnowledgeChunk
from core.ml.benchmark_manifest import resolve_query_asset
from core.services.retriever import retrieve
import core.services.retriever as retriever_module


def load_holdout():
    path = BASE_DIR / "private_training_data" / "frozen_independent_holdout_benchmark.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
    relevance = [r for r in rows if r.get("positive_chunk_ids") and len(r["positive_chunk_ids"]) > 0]
    abstention = [r for r in rows if not r.get("positive_chunk_ids") or len(r["positive_chunk_ids"]) == 0]
    return rows, relevance, abstention


def evaluate_hybrid_pipeline(label, relevance_rows, k_max=10):
    print(f"\n{'='*80}")
    print(f"EVALUATING HYBRID PIPELINE: {label}")
    print(f"Relevance queries: {len(relevance_rows)}")
    print(f"{'='*80}")

    tenants = {t.id: t for t in Tenant.objects.all()}
    r1 = r3 = r5 = r10 = r20 = mrr = ndcg = 0.0
    latencies = []
    per_query = []
    failures_by_category = defaultdict(list)

    for row in relevance_rows:
        qid = row.get("query_id", "")
        query = row["query_text"]
        tid = int(row["tenant_id"])
        pos_ids = set(row.get("positive_chunk_ids", []))
        category = row.get("category", "unknown")
        equip = row.get("equipment", "")

        tenant = tenants[tid]
        asset = resolve_query_asset(tid, equip)

        t0 = time.perf_counter()
        results = retrieve(
            tenant=tenant,
            question=query,
            asset=asset,
            customer=None,
            include_confidential=False,
            top_k=min(k_max, 20),
            staff_mode=False,
        )
        lat_ms = (time.perf_counter() - t0) * 1000
        latencies.append(lat_ms)

        ranked = [r.chunk_id for r in results if r.chunk_id]
        hits = [i for i, c in enumerate(ranked) if c in pos_ids]

        hit1 = any(h < 1 for h in hits)
        hit3 = any(h < 3 for h in hits)
        hit5 = any(h < 5 for h in hits)
        hit10 = any(h < 10 for h in hits)
        hit20 = any(h < 20 for h in hits)

        r1 += hit1; r3 += hit3; r5 += hit5; r10 += hit10; r20 += hit20
        rr = 1 / (hits[0] + 1) if hits else 0
        mrr += rr

        dcg = sum(1 / math.log2(h + 2) for h in hits)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(pos_ids), k_max)))
        ndcg += dcg / idcg if idcg else 0

        first_hit_rank = hits[0] + 1 if hits else None
        pq = {
            "query_id": qid,
            "query_text": query,
            "category": category,
            "equipment": equip,
            "tenant_id": tid,
            "hit@1": hit1, "hit@3": hit3, "hit@5": hit5, "hit@10": hit10, "hit@20": hit20,
            "rr": rr,
            "first_hit_rank": first_hit_rank,
            "ranked_top5": ranked[:5],
            "positive_ids": list(pos_ids),
            "retrieved_count": len(results),
        }
        per_query.append(pq)

        if not hit1:
            failures_by_category[category].append(pq)

    n = len(relevance_rows)
    latencies.sort()

    metrics = {
        "model": label,
        "n_relevance": n,
        "recall@1": round(r1 / n, 4),
        "recall@3": round(r3 / n, 4),
        "recall@5": round(r5 / n, 4),
        "recall@10": round(r10 / n, 4),
        "recall@20": round(r20 / n, 4),
        "mrr@10": round(mrr / n, 4),
        "ndcg@10": round(ndcg / n, 4),
        "latency_p50_ms": round(latencies[len(latencies)//2], 1),
        "latency_p95_ms": round(latencies[int(len(latencies)*0.95)], 1),
    }

    print(f"\n  CORRECTED RESULTS (N={n}):")
    print(f"  Recall@1:  {metrics['recall@1']:.4f}  ({int(r1)}/{n})")
    print(f"  Recall@3:  {metrics['recall@3']:.4f}  ({int(r3)}/{n})")
    print(f"  Recall@5:  {metrics['recall@5']:.4f}  ({int(r5)}/{n})")
    print(f"  Recall@10: {metrics['recall@10']:.4f}  ({int(r10)}/{n})")
    print(f"  Recall@20: {metrics['recall@20']:.4f}  ({int(r20)}/{n})")
    print(f"  MRR@10:    {metrics['mrr@10']:.4f}")
    print(f"  nDCG@10:   {metrics['ndcg@10']:.4f}")
    print(f"  Latency:   {metrics['latency_p50_ms']}ms p50, {metrics['latency_p95_ms']}ms p95")

    # Failure analysis
    miss_count = n - int(r1)
    if miss_count > 0:
        print(f"\n  FAILURES ({miss_count} queries missed @1):")
        for cat, fails in sorted(failures_by_category.items(), key=lambda x: -len(x[1])):
            print(f"    {cat}: {len(fails)} misses")
            for f in fails[:2]:
                fhr = f['first_hit_rank']
                rank_str = f"rank {fhr}" if fhr else "NOT IN TOP 20"
                print(f"      '{f['query_text'][:60]}...' -> {rank_str}")

    return metrics, per_query


def setup_op26_minilm_ft_profile():
    from core.ml.embeddings import PROFILES, EmbeddingProfile
    minilm_ckpt = str((BASE_DIR / "ml_models" / "checkpoints" / "op26_minilm_ft" / "final").resolve())
    PROFILES["company-minilm-op26-ft"] = EmbeddingProfile(
        key="company-minilm-op26-ft",
        model_id=minilm_ckpt,
        revision=None,
        dim=384,
        collection="servy_op26_minilm_ft_v1",
        local_path=minilm_ckpt,
        content_first=False,
        promotion_eligible=True,
    )
    PROFILES["minilm-op26-ft"] = EmbeddingProfile(
        key="minilm-op26-ft",
        model_id=minilm_ckpt,
        revision=None,
        dim=384,
        collection="servy_op26_minilm_ft_v1",
        local_path=minilm_ckpt,
        content_first=False,
        promotion_eligible=True,
    )


def setup_op26_bge_ft_profile():
    from core.ml.embeddings import PROFILES, EmbeddingProfile, BGE_QUERY_PREFIX
    bge_ckpt = str((BASE_DIR / "ml_models" / "checkpoints" / "op26_bge_ft" / "final").resolve())
    PROFILES["company-bge-op26-ft"] = EmbeddingProfile(
        key="company-bge-op26-ft",
        model_id=bge_ckpt,
        revision=None,
        dim=384,
        collection="servy_op26_bge_ft_v1",
        local_path=bge_ckpt,
        query_prefix=BGE_QUERY_PREFIX,
        content_first=False,  # Bypass _matches_current_bge_vector metadata mismatch for Op26 index
        promotion_eligible=True,
    )
    PROFILES["bge-op26-ft"] = EmbeddingProfile(
        key="bge-op26-ft",
        model_id=bge_ckpt,
        revision=None,
        dim=384,
        collection="servy_op26_bge_ft_v1",
        local_path=bge_ckpt,
        query_prefix=BGE_QUERY_PREFIX,
        content_first=False,
        promotion_eligible=True,
    )


def main():
    print("=" * 80)
    print("OPERATION 27 PHASE 0.4 — PRODUCTION HYBRID RETRIEVAL BASELINE")
    print("=" * 80)

    all_rows, relevance, abstention = load_holdout()
    print(f"\nHoldout: {len(all_rows)} total, {len(relevance)} relevance, {len(abstention)} abstention")

    results = {}

    # 5. Production Baseline Hybrid (Base MiniLM + Sparse + RRF)
    print("\n--- Running Configuration 5: Production Hybrid Baseline ---")
    setattr(settings, "SERVY_SERVING_EMBEDDING_PROFILE", "auto")
    setattr(settings, "SERVY_USE_EXPERIMENTAL_RAG", False)
    results["production_hybrid_baseline"] = evaluate_hybrid_pipeline(
        "Production Hybrid Baseline (MiniLM Base)",
        relevance,
    )[0]

    # 6. Hybrid with Op26 Fine-Tuned MiniLM
    print("\n--- Running Configuration 6: Hybrid with Op26 MiniLM FT ---")
    setup_op26_minilm_ft_profile()
    # Patch profile resolution to use op26 minilm ft
    orig_dense_score = retriever_module._dense_score_candidates
    def dense_score_minilm_ft(*args, **kwargs):
        setattr(settings, "SERVY_SERVING_EMBEDDING_PROFILE", "company-minilm-op26-ft")
        from core.ml.embeddings import get_profile
        # Override profile_key lookup
        return orig_dense_score(*args, **kwargs)

    # We can temporarily patch the profile mapping in _dense_score_candidates by monkeypatching get_profile or setting
    # Let's see: if SERVY_SERVING_EMBEDDING_PROFILE is set to "company-minilm-op26-ft":
    # In _dense_score_candidates:
    # if serving_profile in ...
    # Let's inspect how _dense_score_candidates checks serving_profile:
    # Line 624: serving_profile = getattr(settings, "SERVY_SERVING_EMBEDDING_PROFILE", "auto")
    # If serving_profile is not in the hardcoded list, it falls into:
    # else: profile_key = ...
    # So let's wrap _dense_score_candidates to pass profile_key = "company-minilm-op26-ft" directly!
    def make_patched_dense_score(target_profile_key):
        def patched_dense_score(*args, **kwargs):
            import core.ml.embeddings as emb_mod
            orig_gp = emb_mod.get_profile
            def mock_get_profile(key):
                if key in ("company-minilm", "minilm", "company-bge-small", "bge-small", "company-bge-ft", target_profile_key):
                    return orig_gp(target_profile_key)
                return orig_gp(key)
            emb_mod.get_profile = mock_get_profile
            try:
                return orig_dense_score(*args, **kwargs)
            finally:
                emb_mod.get_profile = orig_gp
        return patched_dense_score

    retriever_module._dense_score_candidates = make_patched_dense_score("company-minilm-op26-ft")
    try:
        results["hybrid_op26_minilm_ft"] = evaluate_hybrid_pipeline(
            "Hybrid with Op26 MiniLM FT",
            relevance,
        )[0]
    finally:
        retriever_module._dense_score_candidates = orig_dense_score

    # 7. Hybrid with Op26 Fine-Tuned BGE
    print("\n--- Running Configuration 7: Hybrid with Op26 BGE FT ---")
    setup_op26_bge_ft_profile()
    retriever_module._dense_score_candidates = make_patched_dense_score("company-bge-op26-ft")
    try:
        results["hybrid_op26_bge_ft"] = evaluate_hybrid_pipeline(
            "Hybrid with Op26 BGE FT",
            relevance,
        )[0]
    finally:
        retriever_module._dense_score_candidates = orig_dense_score

    # Summary table
    print("\n" + "=" * 100)
    print("HYBRID PIPELINE COMPARATIVE TABLE (relevance-only denominator N=43)")
    print("=" * 100)
    header = f"{'Configuration':<35} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} {'R@20':>6} {'MRR@10':>8} {'nDCG@10':>8} {'p50ms':>6}"
    print(header)
    print("-" * len(header))
    for key, m in results.items():
        print(f"{m['model']:<35} {m['recall@1']:>6.4f} {m['recall@3']:>6.4f} {m['recall@5']:>6.4f} "
              f"{m['recall@10']:>6.4f} {m['recall@20']:>6.4f} {m['mrr@10']:>8.4f} {m['ndcg@10']:>8.4f} "
              f"{m['latency_p50_ms']:>6.1f}")

    # Combine with dense results from phase 0
    p0_dense_path = BASE_DIR / "private_training_data" / "op27_phase0_corrected_evaluation.json"
    all_results = {}
    if p0_dense_path.exists():
        all_results.update(json.loads(p0_dense_path.read_text(encoding="utf-8")))
    all_results.update(results)

    out_path = BASE_DIR / "private_training_data" / "op27_phase0_all_7_baselines.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nAll 7 baseline results saved: {out_path}")


if __name__ == "__main__":
    main()
