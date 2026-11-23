"""Per-turn diagnostics built from what actually executed.

Counts come only from the execution events the pipeline emitted during this
turn; a stage that did not run is reported as not run, never as zero work.
Only counts, statuses, titles, headings and citation ids are included; no
source text, prompts or record contents.
"""
from __future__ import annotations

from typing import Any, Dict, List


def _metrics(events, stage):
    return [e.get("metrics") or {} for e in events if e.get("stage") == stage]


def _statuses(events, stage):
    return [e.get("status") for e in events if e.get("stage") == stage]


def summarize_turn(events: List[dict], contract: Dict[str, Any]) -> Dict[str, Any]:
    fusion = _metrics(events, "rank_fusion_summary")
    retrieval = {
        "passes": len(fusion),
        "dense_candidates": sum(int(m.get("dense_count") or 0) for m in fusion) if fusion else None,
        "sparse_candidates": sum(int(m.get("sparse_count") or 0) for m in fusion) if fusion else None,
        "retrieved": sum(int(m.get("retrieved_count") or 0) for m in fusion) if fusion else None,
        "backends": sorted({m.get("retrieval_backend") for m in fusion if m.get("retrieval_backend")}),
        "reranker_applied": any(m.get("reranker_applied") for m in
                                _metrics(events, "retrieval_summary") + _metrics(events, "cause_retrieval_summary")),
    }
    presentation = contract.get("presentation") or {}
    response = presentation.get("response") or {}
    citations = {c.get("id"): c for c in presentation.get("citations") or [] if isinstance(c, dict)}
    cited_ids = []
    for segment in response.get("segments") or []:
        for cid in segment.get("citation_ids") or []:
            if cid in citations and cid not in cited_ids:
                cited_ids.append(cid)
    if not response:
        for section in presentation.get("sections") or []:
            for claim in section.get("claims") or []:
                for cid in claim.get("evidence_ids") or []:
                    if cid in citations and cid not in cited_ids:
                        cited_ids.append(cid)
    cited = [{"id": cid, "title": str(citations[cid].get("title") or "")[:160],
              "heading": str(citations[cid].get("heading") or "")[:160]} for cid in cited_ids]

    rejections = []
    response_type = contract.get("response_type") or presentation.get("response_type")
    if response_type == "INSUFFICIENT_EVIDENCE":
        rejections.append(contract.get("reason_code") or "no_validated_claim_for_question")
    for item in response.get("explanations") or []:
        if item.get("status") != "explained":
            rejections.append("cause_without_explaining_evidence")
    if retrieval["passes"] and not retrieval["retrieved"]:
        rejections.append("no_authorized_candidates_retrieved")

    qwen_outcome = _metrics(events, "customer_qwen_outcome")
    qwen_stage = _statuses(events, "customer_qwen_generation")
    if qwen_outcome:
        last = qwen_outcome[-1]
        qwen = {"ran": "completed" in qwen_stage or "failed" in qwen_stage,
                "status": last.get("generation_status") or "unknown",
                "eligible_claims": last.get("candidate_count"),
                "accepted_claims": last.get("supported_count")}
    else:
        qwen = {"ran": False, "status": "skipped" if "skipped" in qwen_stage else "not_run",
                "eligible_claims": None, "accepted_claims": None}
    return {"retrieval": retrieval, "cited_evidence": cited, "rejection_reasons": rejections,
            "qwen": qwen, "response_type": response_type}


def _last_status(events, stage):
    statuses = [(e.get("status"), e.get("error_type")) for e in events if e.get("stage") == stage]
    return statuses[-1] if statuses else (None, None)


def build_outcome(events: List[dict], contract: Dict[str, Any], deadline=None) -> Dict[str, Any]:
    """Typed per-request outcome (extends the contract; existing fields are unchanged).

    status: success | degraded | insufficient_evidence | safety_escalation | conversational
    response_mode: grounded_generated | grounded_extractive | no_validated_evidence | safety_response
    Every value reflects what executed in this request.
    """
    presentation = contract.get("presentation") or {}
    response_type = contract.get("response_type") or presentation.get("response_type")
    diagnostics = contract.get("diagnostics") or summarize_turn(events, contract)
    retrieval = diagnostics.get("retrieval") or {}

    dense_events = [e for e in events if e.get("stage") == "dense_retrieval"]
    dense_error = next((e.get("error_type") for e in dense_events if e.get("error_type")), None)
    dense_status = "failed" if dense_error else (dense_events[-1].get("status") if dense_events else None)
    if not retrieval.get("passes"):
        retrieval_status = "not_run"
    elif dense_status == "completed":
        retrieval_status = "hybrid"
    else:
        retrieval_status = "sparse_only_dense_timeout" if dense_error in ("StageTimeout", "StageBusy") \
            else "sparse_only_dense_unavailable"

    v5_outcome, v5_error = _last_status(events, "v5_reranking_outcome")
    v5_stage, v5_stage_error = _last_status(events, "v5_reranking")
    reranker_applied = bool(retrieval.get("reranker_applied"))
    if reranker_applied:
        reranker_status = "applied"
    elif v5_error == "StageTimeout":
        reranker_status = "timeout"
    elif v5_error == "StageBusy":
        reranker_status = "busy"
    elif v5_stage is None:
        reranker_status = "not_run"
    else:
        reranker_status = "failed" if v5_stage_error or v5_stage == "failed" else "not_applied"

    qwen = diagnostics.get("qwen") or {}
    generation_status = str(qwen.get("status") or "not_run")
    generation_applied = bool(qwen.get("ran") and (qwen.get("accepted_claims") or 0) > 0
                              and generation_status in ("applied", "applied_partial"))

    citations = presentation.get("citations") or []
    if response_type in ("ESCALATION_REQUIRED",) or (presentation.get("hazard") or {}).get("severity") == "immediate":
        status, mode, reason = "safety_escalation", "safety_response", None
    elif response_type == "DIAGNOSTIC_GUIDANCE" and citations:
        degraded_reasons = []
        if retrieval_status.startswith("sparse_only") and retrieval.get("passes"):
            degraded_reasons.append(retrieval_status)
        if reranker_status in ("timeout", "busy", "failed"):
            degraded_reasons.append(f"reranker_{reranker_status}")
        if not generation_applied and generation_status not in ("disabled", "not_run", "skipped",
                                                                "no_eligible_claims"):
            degraded_reasons.append(f"generation_{generation_status}")
        mode = "grounded_generated" if generation_applied else "grounded_extractive"
        status = "degraded" if degraded_reasons else "success"
        reason = ",".join(degraded_reasons) or None
    elif response_type in ("GREETING", "CLARIFICATION", "TECHNICIAN_HANDOFF"):
        status, mode, reason = "conversational", "no_technical_claims", None
    else:
        status, mode = "insufficient_evidence", "no_validated_evidence"
        reason = contract.get("reason_code") or "no_validated_claim_for_question"

    total_ms = round(deadline.elapsed() * 1000, 1) if deadline is not None else None
    return {
        "status": status,
        "response_mode": mode,
        "deadline_seconds": deadline.total if deadline is not None else None,
        "deadline_exceeded": bool(deadline is not None and deadline.expired()),
        "stage_timeouts": sorted(deadline.timeouts) if deadline is not None else [],
        "total_duration_ms": total_ms,
        "retrieval_backend": contract.get("retrieval_backend") or ",".join(retrieval.get("backends") or []) or None,
        "retrieval_status": retrieval_status,
        "reranker_applied": reranker_applied,
        "reranker_model": contract.get("reranker_model") if reranker_applied else None,
        "reranker_status": reranker_status,
        "generation_status": generation_status,
        "generation_applied": generation_applied,
        "fallback_reason": reason,
    }
