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
from typing import Dict, List, Set, Any, Literal, TypedDict


class AnswerOutcomeClassification(TypedDict):
    """Stable result contract for answer-quality benchmark records."""

    outcome: Literal["EVIDENCE_SUPPORTED", "INSUFFICIENT_EVIDENCE", "OTHER"]
    has_references: bool
    observed_abstain: bool


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


def candidate_recall_at_k(candidate_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """Fraction of relevant chunks present in the pre-rerank candidate pool at K."""
    return recall_at_k(candidate_ids, relevant_ids, k)


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
      - optional 'candidate_chunk_ids': pre-rerank candidate chunk IDs
      - optional 'latency_ms': float
      - optional 'is_unsupported': bool
      - optional 'expected_abstain' and 'observed_abstain': answer-policy labels

    Expected unsupported records are excluded from every retrieval denominator.
    Candidate recall is computed only for supported records that provide
    ``candidate_chunk_ids``; it is intentionally independent of the post-rerank
    ``ranked_chunk_ids`` list.
    """
    n = len(eval_records)
    if n == 0:
        result = {
            "n_queries": 0,
            "n_total_queries": 0,
            "n_queries_evaluated": 0,
            "n_supported_queries": 0,
            "n_unsupported_queries": 0,
            "n_abstention_queries": 0,
            "retrieval_denominator": 0,
            "n_candidate_recall_queries": 0,
            "candidate_recall_denominator": 0,
            "n_candidate_recall_missing": 0,
            "n_candidate_recall_not_evaluated": 0,
            "n_correct_abstentions": 0,
            "abstention_denominator": 0,
            "n_expected_abstentions": 0,
            "abstention_decision_correct": 0,
            "abstention_decision_denominator": 0,
            "n_false_abstentions": 0,
            "false_abstention_denominator": 0,
            "mrr@10": None,
            "ndcg@10": None,
            "latency_ms_p50": None,
            "latency_ms_p95": None,
            "abstention_accuracy": None,
            "abstention_decision_accuracy": None,
            "false_abstention_rate": None,
            "status": "NO_EVALUATION_RECORDS",
        }
        for k in k_values:
            result[f"hit@{k}"] = None
            result[f"recall@{k}"] = None
            result[f"precision@{k}"] = None
            result[f"candidate_recall@{k}"] = None
        return result

    hits_k = {k: 0.0 for k in k_values}
    recalls_k = {k: 0.0 for k in k_values}
    precisions_k = {k: 0.0 for k in k_values}
    candidate_recalls_k = {k: 0.0 for k in k_values}
    mrr_total = 0.0
    ndcg_total = 0.0
    latencies = []
    abstention_correct = 0
    abstention_total = 0
    abstention_scored_total = 0
    explicit_abstention_correct = 0
    explicit_abstention_total = 0
    false_abstention_count = 0
    answerable_abstention_total = 0
    supported_total = 0
    candidate_recall_total = 0
    candidate_recall_missing = 0
    candidate_recall_not_evaluated = 0

    for rec in eval_records:
        ranked = rec.get("ranked_chunk_ids", [])
        rel = set(rec.get("relevant_chunk_ids", []))

        if rec.get("latency_ms") is not None:
            latencies.append(rec["latency_ms"])

        # New benchmark records distinguish the expected answerability label
        # from the observed response. Keep is_unsupported for older artifacts.
        expected_abstain = bool(rec.get("expected_abstain", rec.get("is_unsupported", False)))
        observed_abstain = rec.get("observed_abstain")
        if observed_abstain is None:
            answer_outcome = rec.get("answer_outcome")
            if isinstance(answer_outcome, dict):
                observed_abstain = answer_outcome.get("observed_abstain")
            if observed_abstain is None and expected_abstain and "expected_abstain" not in rec:
                # Legacy records treated empty retrieval as an abstention.
                observed_abstain = len(ranked) == 0

        if "expected_abstain" in rec and observed_abstain is not None:
            explicit_abstention_total += 1
            explicit_abstention_correct += int(expected_abstain == bool(observed_abstain))
            if not expected_abstain:
                answerable_abstention_total += 1
                false_abstention_count += int(bool(observed_abstain))

        if expected_abstain:
            abstention_total += 1
            if observed_abstain is not None:
                abstention_scored_total += 1
                abstention_correct += int(bool(observed_abstain))
            continue

        supported_total += 1

        for k in k_values:
            hits_k[k] += hit_at_k(ranked, rel, k)
            recalls_k[k] += recall_at_k(ranked, rel, k)
            precisions_k[k] += precision_at_k(ranked, rel, k)

        candidate_ids = rec.get("candidate_chunk_ids")
        candidate_evaluated = rec.get("candidate_recall_evaluated", True)
        if candidate_evaluated is False:
            candidate_recall_not_evaluated += 1
        elif "candidate_chunk_ids" not in rec or candidate_ids is None:
            candidate_recall_missing += 1
        else:
            candidate_recall_total += 1
            for k in k_values:
                candidate_recalls_k[k] += candidate_recall_at_k(candidate_ids, rel, k)

        mrr_total += mrr_at_k(ranked, rel, 10)
        ndcg_total += ndcg_at_k(ranked, rel, 10)

    evaluated_queries = supported_total
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else None
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else None

    result = {
        "n_total_queries": n,
        "n_queries_evaluated": evaluated_queries,
        "n_supported_queries": supported_total,
        "n_unsupported_queries": abstention_total,
        "n_abstention_queries": abstention_total,
        "retrieval_denominator": evaluated_queries,
        "n_candidate_recall_queries": candidate_recall_total,
        "candidate_recall_denominator": candidate_recall_total,
        "n_candidate_recall_missing": candidate_recall_missing,
        "n_candidate_recall_not_evaluated": candidate_recall_not_evaluated,
        "n_correct_abstentions": abstention_correct,
        "abstention_denominator": abstention_scored_total,
        "n_expected_abstentions": abstention_total,
        "abstention_decision_correct": explicit_abstention_correct,
        "abstention_decision_denominator": explicit_abstention_total,
        "n_false_abstentions": false_abstention_count,
        "false_abstention_denominator": answerable_abstention_total,
        "mrr@10": round(mrr_total / evaluated_queries, 4) if evaluated_queries else None,
        "ndcg@10": round(ndcg_total / evaluated_queries, 4) if evaluated_queries else None,
        "latency_ms_p50": round(p50, 2) if p50 is not None else None,
        "latency_ms_p95": round(p95, 2) if p95 is not None else None,
        "status": "OK" if evaluated_queries else "NO_SUPPORTED_QUERIES",
    }
    if not evaluated_queries:
        result["mrr@10"] = None
        result["ndcg@10"] = None
    if abstention_scored_total > 0:
        result["abstention_accuracy"] = round(abstention_correct / abstention_scored_total, 4)
    else:
        result["abstention_accuracy"] = None
    result["abstention_decision_accuracy"] = (
        round(explicit_abstention_correct / explicit_abstention_total, 4)
        if explicit_abstention_total else None
    )
    result["false_abstention_rate"] = (
        round(false_abstention_count / answerable_abstention_total, 4)
        if answerable_abstention_total else None
    )

    for k in k_values:
        if evaluated_queries:
            result[f"hit@{k}"] = round(hits_k[k] / evaluated_queries, 4)
            result[f"recall@{k}"] = round(recalls_k[k] / evaluated_queries, 4)
            result[f"precision@{k}"] = round(precisions_k[k] / evaluated_queries, 4)
        else:
            result[f"hit@{k}"] = None
            result[f"recall@{k}"] = None
            result[f"precision@{k}"] = None

        if candidate_recall_total:
            result[f"candidate_recall@{k}"] = round(candidate_recalls_k[k] / candidate_recall_total, 4)
        else:
            result[f"candidate_recall@{k}"] = None

    return result


def classify_answer_outcome(rag_result: Dict[str, Any]) -> AnswerOutcomeClassification:
    """Classify a serving result for answer-quality evaluation.

    The returned dictionary is a stable evaluator contract. A policy guard is
    always ``OTHER``. The explicit ``insufficient-evidence`` engine takes
    precedence over answer wording, but an insufficient-evidence response with
    references is ``OTHER``. ``observed_abstain`` is true only for
    insufficient-evidence without references/citations.
    """
    if not isinstance(rag_result, dict):
        rag_result = {}

    references = []
    for key in ("references", "citations", "cited_passages"):
        value = rag_result.get(key)
        if isinstance(value, (list, tuple)):
            references.extend(value)
    has_references = bool(references)

    engine = str(rag_result.get("engine") or "").strip().casefold()
    if engine == "policy-guard":
        outcome = "OTHER"
    else:
        mode = str(
            rag_result.get("response_mode")
            or rag_result.get("answer_mode")
            or rag_result.get("response_type")
            or ""
        ).strip().upper()
        reason = str(rag_result.get("reason_code") or "").strip().upper()
        answer = str(rag_result.get("answer") or "").strip().casefold()
        is_insufficient = (
            engine == "insufficient-evidence"
            or mode in {"INSUFFICIENT_EVIDENCE", "ABSTAIN"}
            or reason in {"NO_AUTHORIZED_ORIGINAL_DOCUMENTATION", "NO_AUTHORIZED_TEST_DOCUMENTATION"}
            or answer.startswith("i couldn't find enough approved information")
        )
        if is_insufficient and not has_references:
            outcome = "INSUFFICIENT_EVIDENCE"
        elif is_insufficient:
            outcome = "OTHER"
        elif has_references:
            outcome = "EVIDENCE_SUPPORTED"
        else:
            outcome = "OTHER"

    return {
        "outcome": outcome,
        "has_references": has_references,
        "observed_abstain": outcome == "INSUFFICIENT_EVIDENCE",
    }


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

