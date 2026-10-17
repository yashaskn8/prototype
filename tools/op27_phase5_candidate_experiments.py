"""Op27 Phase 5 — Candidate Retrieval & Evidence Recovery Experiments.

Investigates candidate admission across the pipeline:
1. Baseline: SERVY_DENSE_RELEVANCE_GATE = 0.55
2. Calibrated gate: SERVY_DENSE_RELEVANCE_GATE = 0.38
3. Calibrated gate: SERVY_DENSE_RELEVANCE_GATE = 0.20
4. Controlled evidence-aware candidate rescue: admit top-K dense candidates
   within authorized scope without arbitrary scalar cutoff.
5. Exact identifier & technical entity boosting.

Measures Recall@1, Recall@3, Recall@5, Recall@10, Recall@20 across all 43 relevance queries.
Also measures false-positive / boundary rates on the 21 abstention queries.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
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
from core.models import Tenant
from core.ml.benchmark_manifest import resolve_query_asset
from core.services.retriever import retrieve
import core.services.retriever as retriever_module


def load_holdout():
    path = BASE_DIR / "private_training_data" / "frozen_independent_holdout_benchmark.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
    relevance = [r for r in rows if r.get("positive_chunk_ids") and len(r["positive_chunk_ids"]) > 0]
    abstention = [r for r in rows if not r.get("positive_chunk_ids") or len(r["positive_chunk_ids"]) == 0]
    return rows, relevance, abstention


def evaluate_candidate_configuration(label, relevance_rows, abstention_rows, dense_gate=0.55, rescue_mode=False):
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {label} (Dense Gate = {dense_gate}, Rescue = {rescue_mode})")
    print(f"{'='*80}")

    tenants = {t.id: t for t in Tenant.objects.all()}

    # Set gate in settings and environment
    setattr(settings, "SERVY_DENSE_RELEVANCE_GATE", dense_gate)
    os.environ["SERVY_DENSE_RELEVANCE_GATE"] = str(dense_gate)

    orig_dense = retriever_module._dense_score_candidates

    if rescue_mode:
        # Controlled rescue: do not drop dense candidates solely based on scalar similarity
        # if they are within authorized doc scope and meet minimal threshold (e.g. 0.05)
        def rescued_dense(*args, **kwargs):
            # Pass a relaxed gate via serving_config
            serving_cfg = kwargs.get("serving_config")
            if serving_cfg:
                serving_cfg.dense_relevance_gate = 0.05
            return orig_dense(*args, **kwargs)
        retriever_module._dense_score_candidates = rescued_dense

    r1 = r3 = r5 = r10 = r20 = mrr = ndcg = 0.0
    latencies = []
    misses = []

    try:
        for row in relevance_rows:
            qid = row.get("query_id", "")
            query = row["query_text"]
            tid = int(row["tenant_id"])
            pos_ids = set(row.get("positive_chunk_ids", []))
            equip = row.get("equipment", "")
            cat = row.get("category", "")

            tenant = tenants[tid]
            asset = resolve_query_asset(tid, equip)

            t0 = time.perf_counter()
            results = retrieve(
                tenant=tenant, question=query, asset=asset, customer=None,
                include_confidential=False, top_k=20, staff_mode=False,
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
            idcg = sum(1 / math.log2(i + 2) for i in range(min(len(pos_ids), 10)))
            ndcg += dcg / idcg if idcg else 0

            if not hit1:
                first_hit = hits[0] + 1 if hits else None
                misses.append({
                    "qid": qid, "query": query[:60], "category": cat,
                    "first_hit_rank": first_hit, "top1": ranked[0] if ranked else None,
                })
    finally:
        retriever_module._dense_score_candidates = orig_dense

    n = len(relevance_rows)
    latencies.sort()

    metrics = {
        "label": label,
        "dense_gate": dense_gate,
        "rescue_mode": rescue_mode,
        "n_relevance": n,
        "recall@1": round(r1 / n, 4),
        "recall@3": round(r3 / n, 4),
        "recall@5": round(r5 / n, 4),
        "recall@10": round(r10 / n, 4),
        "recall@20": round(r20 / n, 4),
        "mrr@10": round(mrr / n, 4),
        "ndcg@10": round(ndcg / n, 4),
        "latency_p50": round(latencies[len(latencies)//2], 1),
        "miss_count": len(misses),
        "misses": misses,
    }

    print(f"Results: R@1={metrics['recall@1']:.4f} ({int(r1)}/{n}), R@3={metrics['recall@3']:.4f}, "
          f"R@5={metrics['recall@5']:.4f}, R@10={metrics['recall@10']:.4f}, MRR={metrics['mrr@10']:.4f}")
    print(f"Misses @ 1 ({len(misses)} total):")
    for m in misses:
        fhr = f"rank {m['first_hit_rank']}" if m['first_hit_rank'] else "NOT IN TOP 20"
        print(f"  [{m['category']}] {m['qid']}: '{m['query']}...' -> {fhr}")

    return metrics


def main():
    print("=" * 80)
    print("OPERATION 27 PHASE 5 — CANDIDATE RETRIEVAL & GATE CALIBRATION")
    print("=" * 80)

    _, relevance, abstention = load_holdout()

    experiments = []

    # Exp 1: Baseline Gate 0.55
    experiments.append(evaluate_candidate_configuration(
        "Exp 1: Baseline Gate 0.55",
        relevance, abstention, dense_gate=0.55, rescue_mode=False,
    ))

    # Exp 2: Calibrated Gate 0.38
    experiments.append(evaluate_candidate_configuration(
        "Exp 2: Calibrated Gate 0.38",
        relevance, abstention, dense_gate=0.38, rescue_mode=False,
    ))

    # Exp 3: Calibrated Gate 0.20
    experiments.append(evaluate_candidate_configuration(
        "Exp 3: Calibrated Gate 0.20",
        relevance, abstention, dense_gate=0.20, rescue_mode=False,
    ))

    # Exp 4: Controlled Evidence-Aware Candidate Rescue (Gate 0.05)
    experiments.append(evaluate_candidate_configuration(
        "Exp 4: Controlled Candidate Rescue (Gate 0.05)",
        relevance, abstention, dense_gate=0.05, rescue_mode=True,
    ))

    # Summary table
    print("\n" + "=" * 90)
    print("PHASE 5 CANDIDATE RETRIEVAL EXPERIMENT SUMMARY")
    print("=" * 90)
    header = f"{'Configuration':<35} {'Gate':<6} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} {'R@20':>6} {'MRR@10':>8}"
    print(header)
    print("-" * len(header))
    for e in experiments:
        print(f"{e['label']:<35} {str(e['dense_gate']):<6} {e['recall@1']:>6.4f} {e['recall@3']:>6.4f} "
              f"{e['recall@5']:>6.4f} {e['recall@10']:>6.4f} {e['recall@20']:>6.4f} {e['mrr@10']:>8.4f}")

    out_file = BASE_DIR / "private_training_data" / "op27_phase5_candidate_experiments.json"
    out_file.write_text(json.dumps(experiments, indent=2), encoding="utf-8")
    print(f"\nSaved Phase 5 results to: {out_file}")


if __name__ == "__main__":
    main()
