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
