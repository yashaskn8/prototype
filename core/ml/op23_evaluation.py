"""Metrics for externally approved cases; keeps each dataset role separate."""
from __future__ import annotations

import math
from collections import defaultdict


class EvaluationError(ValueError):
    pass


def evaluate_reviewed(cases: list[dict], predictions: list[dict]) -> dict:
    by_id = {p["query_id"]: p for p in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {c["query_id"] for c in cases}:
        raise EvaluationError("Predictions must match reviewed case IDs exactly once")
    roles = defaultdict(list)
    for case in cases:
        if case.get("review_status") != "APPROVED":
            raise EvaluationError("Unapproved case in reviewed evaluation")
        roles[case["dataset_role"]].append((case, by_id[case["query_id"]]))
    return {role: _role_metrics(items) for role, items in roles.items()}


def _role_metrics(items: list[tuple[dict, dict]]) -> dict:
    answerable, unanswerable, clarification = [], [], []
    latencies = []
    tenant_leak_observations = []
    for case, prediction in items:
        if prediction.get("tenant_id") != case["tenant_id"]:
            raise EvaluationError("Prediction tenant differs from reviewed tenant")
        if "end_to_end_latency_ms" in prediction:
            latency = prediction["end_to_end_latency_ms"]
            if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
                raise EvaluationError("Invalid end-to-end latency")
            latencies.append(latency)
        if "retrieved_tenant_ids" in prediction:
            observed = prediction["retrieved_tenant_ids"]
            if not isinstance(observed, list):
                raise EvaluationError("Invalid retrieved tenant IDs")
            tenant_leak_observations.append(any(t != case["tenant_id"] for t in observed))
        status = case["answerability"]
        if status == "ANSWERABLE":
            positives = set(case["valid_positive_passage_ids"])
            ranked_ids = prediction.get("ranked_passage_ids") or []
            ranks = [rank for rank, pid in enumerate(ranked_ids, 1) if pid in positives]
            dcg = sum(1 / math.log2(rank + 1) for rank in ranks if rank <= 10)
            ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(positives), 10) + 1))
            answerable.append({
                "hit1": bool(1 in ranks), "hit5": any(r <= 5 for r in ranks),
                "hit20": any(r <= 20 for r in ranks),
                "candidate": bool(positives & set(prediction.get("candidate_passage_ids") or ranked_ids)),
                "mrr10": 1 / min(ranks) if ranks and min(ranks) <= 10 else 0,
                "ndcg10": dcg / ideal if ideal else 0,
                "asset": prediction.get("external_asset_id") == case.get("external_asset_id"),
                "source": prediction.get("selected_source_id") == case.get("source_id"),
                "answered": bool(prediction.get("answered")),
                "answer_evidence": prediction.get("selected_passage_id") in positives,
            })
        elif status == "UNANSWERABLE_FROM_APPROVED_SOURCES":
            unanswerable.append(not prediction.get("answered"))
        elif status == "NEEDS_ASSET_CLARIFICATION":
            clarification.append(prediction.get("decision") == "EQUIPMENT_CLARIFICATION_REQUIRED"
                                 and not prediction.get("answered"))
        else:
            raise EvaluationError("Unknown reviewed answerability")
    count = len(answerable)
    rate = lambda key: sum(row[key] for row in answerable) / count if count else None
    ordered_latencies = sorted(latencies)
    p95 = ordered_latencies[math.ceil(0.95 * len(ordered_latencies)) - 1] if len(latencies) == len(items) else None
    return {
        "case_count": len(items), "answerable_count": count,
        "passage_hit_at_1": rate("hit1"), "passage_hit_at_5": rate("hit5"),
        "passage_hit_at_20": rate("hit20"), "answer_present_in_candidate_set": rate("candidate"),
        "mrr_at_10": rate("mrr10"), "ndcg_at_10": rate("ndcg10"),
        "correct_asset_selection": rate("asset"), "primary_source_selection": rate("source"),
        "answer_coverage": rate("answered"),
        "answer_bearing_evidence_rate_among_answered": (
            sum(row["answer_evidence"] for row in answerable if row["answered"])
            / sum(row["answered"] for row in answerable) if any(row["answered"] for row in answerable) else None),
        "unanswerable_count": len(unanswerable),
        "unanswerable_abstention_rate": sum(unanswerable) / len(unanswerable) if unanswerable else None,
        "clarification_count": len(clarification),
        "correct_clarification_rate": sum(clarification) / len(clarification) if clarification else None,
        "grounded_technical_answer_correctness": None,
        "unsafe_action_rate": None,
        "cross_tenant_leakage_rate": (sum(tenant_leak_observations) / len(tenant_leak_observations)
                                       if len(tenant_leak_observations) == len(items) else None),
        "p95_end_to_end_latency_ms": p95,
        "grounding_note": "Requires separate authenticated human answer judgments; passage selection is not answer correctness",
    }
