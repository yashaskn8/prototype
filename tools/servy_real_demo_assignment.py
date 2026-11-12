"""Evidence-sufficiency decisions and issue escalation for Servy's customer RAG.

This module never retrieves, generates technical content, or writes a ticket.
It classifies an already validated answer contract into one of three evidence
states, resolves "the second cause" style references against the signed list
of causes from the previous answer, and prepares a customer-visible issue
description for the existing local ticket workflow. The customer still
confirms and submits that ticket through the existing form.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path

SUFFICIENT = "SUFFICIENT"
PARTIAL = "PARTIAL"
INSUFFICIENT = "INSUFFICIENT"

AWAITING_CONFIRMATION = "AWAITING_CUSTOMER_CONFIRMATION"
ALREADY_ASSIGNED = "ALREADY_ASSIGNED"

MAX_CAUSES = 10
MAX_CAUSE_CHARS = 240
MAX_DESCRIPTION_CHARS = 3500
CAUSE_KINDS = frozenset({"possible_cause"})
CAUSE_DETAIL_KINDS = frozenset({"possible_cause", "information"})
LIMITATION_TITLES = frozenset({"Evidence limitation", "What I could not verify"})
# Canonical wording understood by the existing explanation-follow-up path.
EXPLAIN_CAUSES_QUESTION = "Explain those causes in more detail"

_ORDINALS = {
    "first": 0, "1st": 0, "one": 0, "second": 1, "2nd": 1, "two": 1,
    "third": 2, "3rd": 2, "three": 2, "fourth": 3, "4th": 3, "four": 3,
    "fifth": 4, "5th": 4, "five": 4, "sixth": 5, "6th": 5,
    "seventh": 6, "7th": 6, "eighth": 7, "8th": 7,
}
_SUBJECT = r"(?:causes?|reasons?|possibilit(?:y|ies)|issues?|points?|items?|ones?|options?)"
_ORDINAL_REFERENCE = re.compile(
    r"\b(?:the\s+)?(" + "|".join(_ORDINALS) + r"|last|final)\s+"
    r"(?:documented\s+|possible\s+|listed\s+)*" + _SUBJECT + r"\b"
    r"|\b" + _SUBJECT + r"\s+(?:number\s+|no\.?\s*|#\s*)?(\d{1,2})\b"
    r"|#\s*(\d{1,2})\b",
    re.I,
)
_STOP = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "may", "might", "can",
    "could", "when", "into", "are", "was", "has", "have", "not", "its", "unit",
    "equipment", "system", "possible", "documented", "cause", "causes", "issue",
    "problem", "condition", "affect", "affects", "reduced", "poor",
})
_DURATION = re.compile(
    r"\b(?:since\s+\w+(?:\s+\w+)?|yesterday|today|last (?:night|week|month)|"
    r"(?:for|about|around)?\s*\d+\s*(?:minutes?|hours?|days?|weeks?|months?)|"
    r"every\s+\w+(?:\s+\w+)?|each time|once|twice|daily|weekly|hourly|"
    r"sometimes|often|occasionally|intermittent(?:ly)?|constant(?:ly)?|"
    r"continuous(?:ly)?|frequently|all the time)\b", re.I,
)
_CODE = re.compile(r"(?<![A-Z0-9])(?:ERR|E|F)[- ]?\d{1,3}(?![A-Z0-9])", re.I)


def _clean(text, limit):
    return " ".join(str(text or "").split())[:limit]


def _norm(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold()).strip()


def _tokens(text):
    return {t for t in _norm(text).split() if len(t) >= 4 and t not in _STOP}


def cause_reference(question, last_causes):
    """Resolve "the second cause" against the previous answer's cause list.

    Returns None when the question has no ordinal reference or no cause list
    was shown; otherwise {"index", "cause"} where cause is None if the
    requested position does not exist in the previous answer.
    """
    causes = [c for c in (last_causes or []) if isinstance(c, str) and c.strip()]
    if not causes or not isinstance(question, str):
        return None
    match = _ORDINAL_REFERENCE.search(" ".join(question.split()))
    if not match:
        return None
    word, number = match.group(1), match.group(2) or match.group(3)
    if word:
        word = word.casefold()
        index = len(causes) - 1 if word in {"last", "final"} else _ORDINALS[word]
    else:
        index = int(number) - 1
    if index < 0 or index >= len(causes):
        return {"index": index, "cause": None, "available": len(causes)}
    return {"index": index, "cause": causes[index], "available": len(causes)}


def listed_causes(presentation):
    """Ordered source sentences of the causes shown in a validated answer."""
    out = []
    for claim in (presentation or {}).get("supporting_claims", []):
        if (isinstance(claim, dict) and claim.get("kind") in CAUSE_KINDS
                and isinstance(claim.get("source_text"), str)
                and claim["source_text"].strip()):
            text = _clean(claim["source_text"], MAX_CAUSE_CHARS)
            if text not in out:
                out.append(text)
        if len(out) >= MAX_CAUSES:
            break
    return out


def valid_cause_state(value):
    return (isinstance(value, list) and len(value) <= MAX_CAUSES
            and all(isinstance(item, str) and item.strip()
                    and len(item) <= MAX_CAUSE_CHARS for item in value))


def _related(cause, source_text):
    cause_tokens, text_tokens = _tokens(cause), _tokens(source_text)
    shared = cause_tokens & text_tokens
    return len(shared) >= (2 if len(cause_tokens) >= 3 else 1)


def focus_on_cause(contract, cause, shown_before, fingerprint):
    """Restrict a validated explanation answer to one previously listed cause.

    Keeps the exact cited cause sentence plus additional cited sentences that
    share its technical terms and were not shown before. Nothing is reworded
    or generated. Returns (contract, has_new_detail).
    """
    presentation = dict(contract.get("presentation") or {})
    citations = {c.get("id"): c for c in presentation.get("citations", [])
                 if isinstance(c, dict)}
    shown = set(shown_before or ())
    target, extra = [], []
    for claim in presentation.get("supporting_claims", []):
        if not isinstance(claim, dict) or claim.get("kind") not in CAUSE_DETAIL_KINDS:
            continue
        evidence = claim.get("evidence_ids") or []
        citation = citations.get(evidence[0]) if len(evidence) == 1 else None
        if citation is None:
            continue
        source = claim.get("source_text", "")
        if _norm(source) == _norm(cause):
            target.append(claim)
        elif _related(cause, source) and fingerprint(claim, citation) not in shown:
            extra.append(claim)
    target, extra = target[:1], extra[:8]
    kept = target + extra
    if not kept:
        message = (
            "The approved internal documentation does not contain any further "
            "explanation of that cause for this equipment. I won't add technical "
            "detail that the documents do not support."
        )
        return {
            **contract, "response_type": "INSUFFICIENT_EVIDENCE",
            "response_mode": "INSUFFICIENT_EVIDENCE",
            "reason_code": "NO_RELEVANT_APPROVED_EVIDENCE",
            "summary": message, "answer": message, "citations": [],
            "references": [], "customer_safe_actions": [], "can_resolve": False,
            "presentation": {
                **presentation, "view_mode": "focused",
                "response_type": "INSUFFICIENT_EVIDENCE", "sections": [],
                "citations": [], "supporting_claims": [],
                "customer_permitted_observations": [],
                "customer_authorized_actions": [], "message": message,
                "follow_up_question": None,
            },
        }, False
    used = {claim["evidence_ids"][0] for claim in kept}
    kept_citations = [c for c in presentation.get("citations", [])
                      if isinstance(c, dict) and c.get("id") in used]
    sections = [{"number": 1, "title": "Documented possible causes", "claims": kept}]
    if extra:
        message = (
            "Here is what the approved documentation states about that cause. "
            "It is a documented possibility, not a confirmed diagnosis."
        )
    else:
        message = (
            "The approved documentation states this cause only as cited below. "
            "It is a documented possibility, not a confirmed diagnosis."
        )
        sections.append({"number": 2, "title": "Evidence limitation", "claims": [{
            "text": ("The approved documents for this equipment contain no further "
                     "explanation of this cause. I won't add technical detail they "
                     "do not support."),
            "kind": "uncertainty", "evidence_ids": [],
        }]})
    presentation.update({
        "view_mode": "focused", "sections": sections, "citations": kept_citations,
        "supporting_claims": kept, "customer_permitted_observations": [],
        "customer_authorized_actions": [], "message": message,
        "follow_up_question": None,
    })
    return {
        **contract, "presentation": presentation, "summary": message,
        "answer": message + "\n" + "\n".join(c.get("text", "") for c in kept),
        "citations": kept_citations, "references": kept_citations,
        "customer_safe_actions": [], "can_resolve": False,
    }, bool(extra)


def code_conflict(presentation, question):
    """A requested fault code described differently by two approved documents."""
    asked = {_norm(code).replace(" ", "") for code in _CODE.findall(str(question or ""))}
    if not asked or not isinstance(presentation, dict):
        return None
    citations = {c.get("id"): c for c in presentation.get("citations", [])
                 if isinstance(c, dict)}
    by_code = {}
    for claim in presentation.get("supporting_claims", []):
        if not isinstance(claim, dict) or claim.get("kind") not in {
                "possible_cause", "information", "document_fact"}:
            continue
        evidence = claim.get("evidence_ids") or []
        citation = citations.get(evidence[0]) if len(evidence) == 1 else None
        if citation is None:
            continue
        source = claim.get("source_text", "")
        for code in _CODE.findall(source):
            key = _norm(code).replace(" ", "")
            if key in asked:
                by_code.setdefault(key, {}).setdefault(
                    citation.get("document_id"), set()).add(_norm(source))
    for key, documents in by_code.items():
        statements = list(documents.values())
        if len(statements) > 1 and any(
                a.isdisjoint(b) for i, a in enumerate(statements)
                for b in statements[i + 1:]):
            return key.upper()
    return None


def add_limitation(contract, text):
    """Append one uncited limitation section to a focused, validated answer."""
    presentation = dict(contract.get("presentation") or {})
    sections = [dict(s) for s in presentation.get("sections", []) if isinstance(s, dict)]
    if len(sections) >= 6:
        return contract
    sections.append({"number": len(sections) + 1, "title": "Evidence limitation",
                     "claims": [{"text": text[:900], "kind": "uncertainty",
                                 "evidence_ids": []}]})
    presentation["sections"] = sections
    return {**contract, "presentation": presentation}


def limitation_texts(presentation):
    out = []
    for section in (presentation or {}).get("sections", []):
        if isinstance(section, dict) and section.get("title") in LIMITATION_TITLES:
            for claim in section.get("claims", []):
                if isinstance(claim, dict) and isinstance(claim.get("text"), str):
                    out.append(_clean(claim["text"], 400))
    return out


def evidence_state(response_type, reason_code, presentation):
    """Classify a technical answer. None means the turn is not a technical answer."""
    if response_type == "DIAGNOSTIC_GUIDANCE":
        return PARTIAL if limitation_texts(presentation) else SUFFICIENT
    if response_type == "INSUFFICIENT_EVIDENCE" and reason_code != "OUT_OF_SCOPE_QUERY":
        return INSUFFICIENT
    return None


def issue_key(thread_id, original_complaint):
    """Stable identity of one unresolved complaint inside one chat thread."""
    if not isinstance(thread_id, str) or not thread_id:
        return ""
    digest = hashlib.sha256(_norm(original_complaint).encode("utf-8")).hexdigest()[:16]
    return thread_id + ":" + digest


def find_open_ticket(tenant_id, customer_id, asset_id, conversation_key):
    """Read-only lookup of an unresolved ticket already raised for this issue.

    Never creates the store and never writes. Any failure means "unknown",
    which callers treat as no existing ticket.
    """
    if not conversation_key:
        return None
    try:
        from django.conf import settings
        path = Path(settings.SERVY_LOCAL_TICKET_DB_PATH).expanduser().resolve()
        if not path.is_file():
            return None
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5)
        try:
            row = connection.execute(
                "SELECT id, status FROM local_tickets WHERE tenant_id = ? "
                "AND customer_id = ? AND asset_id = ? AND source_conversation_id = ? "
                "AND status != 'closed' ORDER BY created_at DESC LIMIT 1",
                (tenant_id, customer_id, asset_id, conversation_key),
            ).fetchone()
        finally:
            connection.close()
    except Exception:
        return None
    if not row:
        return None
    return {"id": str(row[0]), "status": str(row[1])}


def build_issue_description(*, asset, context, history, question, state,
                            presentation, selected_cause=None, conflict=None):
    """Customer-visible escalation summary built only from known information."""
    lines = ["Servy escalation: approved internal documentation could not "
             + ("fully " if state == PARTIAL else "") + "answer a customer technical question."]
    name = _clean(getattr(asset, "name", ""), 120)
    model = _clean(getattr(asset, "model_number", ""), 60)
    code = _clean(getattr(asset, "asset_code", ""), 60)
    lines.append("Asset: " + (name or "Selected equipment")
                 + (f" | Model: {model}" if model else "")
                 + (f" | Asset code: {code}" if code else ""))
    complaint = _clean(getattr(context, "original_complaint", ""), 400)
    if complaint:
        lines.append("Original complaint (customer-reported): " + complaint)
    symptoms = [_clean(s, 80) for s in getattr(context, "reported_symptoms", []) or []][:8]
    if symptoms:
        lines.append("Customer-reported symptoms (not verified): " + "; ".join(symptoms))
    turns = [t for t in list(history or []) + [question] if isinstance(t, str)]
    timing = list(dict.fromkeys(
        _clean(m.group(0), 40).casefold() for t in turns for m in _DURATION.finditer(t)))[:6]
    if timing:
        lines.append("Reported duration/frequency: " + "; ".join(timing))
    followups = [_clean(t, 220) for t in turns[-5:] if _norm(t) != _norm(complaint)]
    if followups:
        lines.append("Customer follow-ups: " + " | ".join(followups))
    established, references = [], []
    citations = {c.get("id"): c for c in (presentation or {}).get("citations", [])
                 if isinstance(c, dict)}
    for claim in (presentation or {}).get("supporting_claims", [])[:8]:
        if not isinstance(claim, dict) or not isinstance(claim.get("source_text"), str):
            continue
        evidence = claim.get("evidence_ids") or []
        citation = citations.get(evidence[0]) if len(evidence) == 1 else None
        if citation is None:
            continue
        title = _clean(citation.get("title", ""), 120)
        heading = _clean(citation.get("heading", ""), 80)
        reference = title + (f" — {heading}" if heading else "")
        established.append("- " + _clean(claim["source_text"], 260)
                           + (f" [{reference}]" if reference else ""))
        if reference and reference not in references:
            references.append(reference)
    lines.append("Documentation establishes:" if established
                 else "Documentation establishes: no relevant approved passage was found.")
    lines.extend(established)
    gaps = limitation_texts(presentation)
    if conflict:
        gaps.insert(0, f"Approved documents describe {conflict} differently.")
    if not gaps:
        gaps = ["No approved passage for this equipment answers the question."]
    lines.append("Documentation cannot establish: " + " ".join(dict.fromkeys(gaps)))
    unanswered = _clean(question, 400)
    if selected_cause:
        unanswered += " (refers to documented cause: " + _clean(selected_cause, 240) + ")"
    lines.append("Unanswered question: " + unanswered)
    if references:
        lines.append("Internal references shown to the customer: " + "; ".join(references[:6]))
    lines.append("Reason for escalation: "
                 + ("conflicting internal documentation" if conflict
                    else "partial internal evidence" if state == PARTIAL
                    else "no sufficient internal evidence"))
    return "\n".join(lines)[:MAX_DESCRIPTION_CHARS]


def issue_assignment(*, state, description, question, existing_ticket, conflict=None):
    """Response block telling the client to invoke the existing ticket workflow."""
    if state not in {PARTIAL, INSUFFICIENT}:
        return None
    reason = ("CONFLICTING_INTERNAL_DOCUMENTS" if conflict
              else "PARTIAL_INTERNAL_EVIDENCE" if state == PARTIAL
              else "NO_SUFFICIENT_INTERNAL_EVIDENCE")
    if existing_ticket:
        status = ALREADY_ASSIGNED
        notice = (
            "This unresolved issue is already recorded as ticket "
            + existing_ticket["id"] + " (status: " + existing_ticket["status"]
            + "). No duplicate ticket was created."
        )
    else:
        status = AWAITING_CONFIRMATION
        notice = (
            "The approved internal documentation cannot "
            + ("fully " if state == PARTIAL else "")
            + "answer this. I have prepared an issue for the service team. "
            "Nothing has been submitted yet: review the ticket form and submit it "
            "to assign the issue."
        )
    return {
        "required": True, "evidence_state": state, "reason": reason,
        "status": status, "workflow": "local-ticket",
        "unanswered_question": _clean(question, 400),
        "description": description, "existing_ticket": existing_ticket,
        "notice": notice,
    }
