"""One immutable plan for the shared technical customer serving path.

The original user text stays separate from the resolved answer request and
retrieval query. Direct safety, authorization and conversation-only routes
remain authoritative. No model output can create a source or approve an action.
"""
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class TechnicalTurnPlan:
    request: str
    answer_question: str
    retrieval_question: str
    evidence_question: str
    intent: str
    cause_reference: Optional[dict] = None


def plan_technical_turn(request, context, asset=None):
    from .conversation_context import (
        contextualize_customer_query, canonical_issue_question, resolve_cause_reference,
        topic_shift_query, ISSUE_CHECKS_PREFIX, ISSUE_MATCH_PREFIX, extract_symptoms_from_text,
    )
    from .customer_answer import customer_evidence_question
    reference = resolve_cause_reference(request, context)
    answer = (topic_shift_query(request) if context.is_followup and not reference else None) or request
    retrieval = contextualize_customer_query(request, context, asset=asset)
    canonical = None if reference else canonical_issue_question(request, context)
    intent = "technical"
    if canonical:
        answer = canonical
        intent = ("next_step" if canonical.startswith(ISSUE_CHECKS_PREFIX) else
                  "match_causes" if canonical.startswith(ISSUE_MATCH_PREFIX) else "causes")
        # Only affirmative observations from this incident enter the query.
        observations = [s for s in getattr(context, "active_observations", [])
                        if extract_symptoms_from_text(s)] if intent == "match_causes" else []
        retrieval = "; ".join(p for p in (
            getattr(asset, "name", ""), context.active_issue,
            " ".join(observations[-3:]),
            "customer-safe checks and observations" if intent == "next_step" else "possible causes",
        ) if p)[:450]
    if reference:
        intent = "cause_detail"
        selected = reference["selected"]
        retrieval = f"{selected['heading']}: {selected['source_text']}".strip(": ")
    return TechnicalTurnPlan(request, answer, retrieval or request,
                             customer_evidence_question(answer, context, reference),
                             intent, reference)


def final_response_is_valid(contract, plan):
    """Independently check the displayed response's claim binding and scoped intent.

    This is a structural/intent gate, not a semantic entailment model. Every
    technical segment must correspond to an already validated claim. Relevance
    for a cause match is additionally restricted to the named symptom.
    """
    if contract.get("response_type") != "DIAGNOSTIC_GUIDANCE":
        return True
    from .customer_response_composer import _claim_body, _norm
    from .term_matching import symptom_concepts, heading_states_symptom, question_states_heading
    presentation = contract.get("presentation") or {}
    response = presentation.get("response") or {}
    segments = response.get("segments")
    if not isinstance(segments, list) or not segments:
        # Legacy document fact contracts can be rendered from their sections.
        # Never allow that fallback for a resolved intent that requires filtering.
        return plan.intent not in {"match_causes", "next_step", "cause_detail"}
    citations = {c.get("id") for c in presentation.get("citations", []) if isinstance(c, dict)}
    claims = presentation.get("supporting_claims") or []
    by_id = {}
    for claim in claims:
        for cid in claim.get("evidence_ids") or []:
            by_id.setdefault(cid, []).append(claim)
    for segment in segments:
        ids = segment.get("citation_ids") or []
        kind = segment.get("kind")
        if kind not in {"claim", "list_item"}:
            if ids or kind not in {"text", "question"}:
                return False
            continue
        if len(ids) != 1 or ids[0] not in citations:
            return False
        candidates = by_id.get(ids[0], [])
        text = _norm(segment.get("text"))
        bound = [c for c in candidates if text in {
            _norm(_claim_body(c)), _norm(c.get("source_text")), _norm(c.get("text"))
        }]
        if not bound:
            return False
        if plan.intent == "next_step" and any(c.get("kind") == "possible_cause" for c in bound):
            return False
        if plan.intent == "match_causes" and any(c.get("kind") == "possible_cause" for c in bound):
            concepts = symptom_concepts(plan.evidence_question) - {"generic_failure"}
            if concepts and not any(
                heading_states_symptom(c.get("source_heading") or "", concepts)
                or question_states_heading(plan.evidence_question, c.get("source_heading") or "")
                for c in bound):
                return False
    return True
