"""Comprehensive Evaluation Metrics for Servy Retrieval and Grounded RAG.

Accurately distinguishes and computes:
- Hit@K: 1 if at least one relevant chunk is in top-K, else 0.
- Recall@K: Fraction of total relevant chunks present in top-K (len(hits_in_k) / len(relevant_ids)).
- Precision@K: Fraction of top-K results that are relevant (len(hits_in_k) / K).
- MRR@K: Mean Reciprocal Rank (1 / (rank of first relevant item + 1)).
- nDCG@K: Normalized Discounted Cumulative Gain against ideal DCG.
- Abstention Accuracy: 1 if out-of-domain/unsupported query correctly returned 0 references, else 0.
- Latency percentiles: p50 and p95 retrieval latency in milliseconds.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Set, Any


def hit_at_k(ranked_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """1.0 if any relevant chunk is within top-k, else 0.0."""
    top = set(ranked_ids[:k])
    return 1.0 if bool(top & relevant_ids) else 0.0


def recall_at_k(ranked_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """Fraction of all relevant chunks retrieved in top-k."""
    if not relevant_ids:
        return 0.0
    top = set(ranked_ids[:k])
    return float(len(top & relevant_ids) / len(relevant_ids))


def precision_at_k(ranked_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """Fraction of top-k retrieved chunks that are relevant."""
    if k <= 0:
        return 0.0
    top = set(ranked_ids[:k])
    return float(len(top & relevant_ids) / k)


def mrr_at_k(ranked_ids: List[int], relevant_ids: Set[int], k: int = 10) -> float:
    """Reciprocal rank of the first relevant chunk in top-k, or 0.0."""
    for rank, cid in enumerate(ranked_ids[:k]):
        if cid in relevant_ids:
            return 1.0 / (rank + 1.0)
    return 0.0


def ndcg_at_k(ranked_ids: List[int], relevant_ids: Set[int], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    if not relevant_ids:
        return 0.0
    dcg = 0.0
    for rank, cid in enumerate(ranked_ids[:k]):
        if cid in relevant_ids:
            dcg += 1.0 / math.log2(rank + 2.0)
    idcg = sum(1.0 / math.log2(i + 2.0) for i in range(min(len(relevant_ids), k)))
    return (dcg / idcg) if idcg > 0 else 0.0


def compute_metrics(
    eval_records: List[Dict[str, Any]],
    k_values: tuple = (1, 5, 10),
) -> Dict[str, Any]:
    """Compute aggregate retrieval metrics over an evaluation dataset.

    Each record in eval_records must have:
      - 'ranked_chunk_ids': List[int]
      - 'relevant_chunk_ids': Set[int] or List[int]
      - optional 'latency_ms': float
      - optional 'is_unsupported': bool
    """
    n = len(eval_records)
    if n == 0:
        return {"n_queries": 0, "status": "NO_EVALUATION_RECORDS"}

    hits_k = {k: 0.0 for k in k_values}
    recalls_k = {k: 0.0 for k in k_values}
    precisions_k = {k: 0.0 for k in k_values}
    mrr_total = 0.0
    ndcg_total = 0.0
    latencies = []
    abstention_correct = 0
    abstention_total = 0

    for rec in eval_records:
        ranked = rec.get("ranked_chunk_ids", [])
        rel = set(rec.get("relevant_chunk_ids", []))

        if rec.get("latency_ms") is not None:
            latencies.append(rec["latency_ms"])

        if rec.get("is_unsupported", False):
            abstention_total += 1
            if len(ranked) == 0:
                abstention_correct += 1
            continue

        for k in k_values:
            hits_k[k] += hit_at_k(ranked, rel, k)
            recalls_k[k] += recall_at_k(ranked, rel, k)
            precisions_k[k] += precision_at_k(ranked, rel, k)

        mrr_total += mrr_at_k(ranked, rel, 10)
        ndcg_total += ndcg_at_k(ranked, rel, 10)

    evaluated_queries = max(n - abstention_total, 1)
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else None
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else None

    result = {
        "n_queries_evaluated": evaluated_queries,
        "n_abstention_queries": abstention_total,
        "mrr@10": round(mrr_total / evaluated_queries, 4),
        "ndcg@10": round(ndcg_total / evaluated_queries, 4),
        "latency_ms_p50": round(p50, 2) if p50 else None,
        "latency_ms_p95": round(p95, 2) if p95 else None,
    }
    for k in k_values:
        result[f"hit@{k}"] = round(hits_k[k] / evaluated_queries, 4)
        result[f"recall@{k}"] = round(recalls_k[k] / evaluated_queries, 4)
        result[f"precision@{k}"] = round(precisions_k[k] / evaluated_queries, 4)

    if abstention_total > 0:
        result["abstention_accuracy"] = round(abstention_correct / abstention_total, 4)

    return result


def check_chunk_boundaries(
    chunk_meta: Dict[str, Any],
    query_tenant_id: int,
    customer: Any = None,
    asset: Any = None,
) -> Dict[str, Any]:
    """Inspect a single retrieved chunk against all 5 retrieval-level boundaries."""
    tenant_id = chunk_meta.get("tenant_id")
    customer_id = chunk_meta.get("customer_id")
    asset_id = chunk_meta.get("asset_id")
    product_id = chunk_meta.get("product_id")
    model_number = chunk_meta.get("model_number")

    unauthorized_tenant = (tenant_id is not None and tenant_id != query_tenant_id)

    unauthorized_customer = False
    if customer:
        cust_id = getattr(customer, "id", customer)
        if customer_id is not None and customer_id != cust_id:
            unauthorized_customer = True
    elif customer_id is not None:
        unauthorized_customer = True

    wrong_asset = False
    if asset:
        a_id = getattr(asset, "id", None)
        if asset_id is not None and asset_id != a_id:
            wrong_asset = True

    wrong_product = False
    if asset:
        p_id = getattr(asset, "product_id", None)
        if p_id is not None and product_id is not None and product_id != p_id:
            wrong_product = True

    wrong_model = False
    if asset and getattr(asset, "model_number", None):
        a_model = str(asset.model_number).lower().strip()
        if model_number and str(model_number).lower().strip() != a_model:
            wrong_model = True

    if (asset_id is not None and asset and asset_id == getattr(asset, "id", None)) or (
        product_id is not None and asset and product_id == getattr(asset, "product_id", None)
    ):
        applicability = "CORRECTLY_SCOPED"
    elif asset_id is None and product_id is None:
        applicability = "GENERAL_LEGITIMATELY_APPLICABLE"
    elif wrong_asset or wrong_product or wrong_model:
        applicability = "INCORRECTLY_SCOPED"
    else:
        applicability = "UNKNOWN_APPLICABILITY"

    return {
        "unauthorized_tenant": unauthorized_tenant,
        "unauthorized_customer": unauthorized_customer,
        "wrong_product": wrong_product,
        "wrong_model": wrong_model,
        "wrong_asset": wrong_asset,
        "applicability": applicability,
    }

