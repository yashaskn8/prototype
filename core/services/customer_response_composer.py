"""Conversational response composition from already-validated evidence.

The answer builder produces authorized, cited, source-validated claims grouped in
four fixed report sections. This module turns those claims into a natural
technical-support reply without adding facts:

* Every technical sentence is a claim the builder already validated (its text is
  either the exact source wording or a Qwen paraphrase that passed the source
  validator) and carries its own citation.
* Connective text (list lead-ins, limitation, closing question) is fixed wording
  chosen by intent. (A Qwen-written acknowledgement and question was evaluated and
  rejected: no measured quality gain, 2.5x latency; see the quality report.)
* The structure follows the customer's intent: a cause list for a reported
  symptom, a direct statement for a fact or schedule, a single cause for an
  ordinal follow-up, and a clarifying question for an ambiguous message.

The fixed ``sections`` stay in the contract for the ticket builder, evaluations
and older clients; ``presentation["response"]`` is the composed reply.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

ORDINAL_WORDS = ("first", "second", "third", "fourth", "fifth", "sixth")
LIMITATION = "These are documented possibilities, not a confirmed diagnosis of your unit."
MAX_SEGMENT_CHARS = 600


def _norm(text: str) -> str:
    return " ".join(str(text or "").split()).casefold()


def _section_label(heading: str) -> str:
    """'3. SYMPTOM: POOR COOLING' -> 'poor cooling' (the manual's own section name)."""
    label = re.sub(r"^\s*\d+[.)]\s*", "", str(heading or ""))
    label = re.sub(r"^(?:symptom|fault)\s*:\s*", "", label, flags=re.I)
    return label.strip().lower()


def _claim_body(claim: Dict[str, Any]) -> str:
    """The claim's validated wording without the builder's 'Possible ... cause:' prefix."""
    text = str(claim.get("text") or "").strip()
    text = re.sub(r"^(?:Under .+?:\s*)?Possible (?:documented )?cause:\s*", "", text, flags=re.I)
    return text[:1].upper() + text[1:] if text else text


_STOP = frozenset("""what which when where why how who is are does do did can could should would will may must
the a an about and or of for to in on with this that these those it its my our your any there their be been
have has had get got i we you me us unit equipment system machine ac module please tell more also then
instead now so just again else also still""".split())


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _terms(text) -> set:
    return {_stem(w) for w in re.findall(r"[a-z]{3,}", _norm(text)) if w not in _STOP}


def _subject_terms(question: str) -> set:
    """The thing a short follow-up asks about ('What about the filters?' -> {'filter'})."""
    terms = _terms(question)
    return terms if 0 < len(terms) <= 3 else set()


def _previous_claim_sources(context) -> set:
    contract = getattr(context, "last_contract", None) or {}
    return {_norm(c.get("source_text")) for c in (contract.get("presentation") or {}).get("supporting_claims") or []
            if isinstance(c, dict) and c.get("source_text")}


def _previous_texts(context) -> set:
    contract = getattr(context, "last_contract", None) or {}
    response = (contract.get("presentation") or {}).get("response") or {}
    return {_norm(s.get("text")) for s in response.get("segments") or [] if isinstance(s, dict)}


def _segment(kind: str, text: str, citation_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    return {"kind": kind, "text": str(text)[:MAX_SEGMENT_CHARS], "citation_ids": list(citation_ids or [])}


def _cited_claims(presentation: Dict[str, Any], kinds: set) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for section in presentation.get("sections") or []:
        for claim in section.get("claims") or []:
            if (isinstance(claim, dict) and claim.get("kind") in kinds and claim.get("evidence_ids")
                    and _norm(claim.get("text")) not in seen):
                seen.add(_norm(claim.get("text")))
                out.append(claim)
    return out


def compose_response(contract: Dict[str, Any], question: str, context=None,
                     cause_reference: Optional[Dict[str, Any]] = None,
                     language: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """Return {"text", "segments", "style", "composer"} for a supported answer, else None.

    ``language`` may carry validated Qwen wording for "opening" and "question";
    anything missing falls back to fixed wording.
    """
    presentation = contract.get("presentation") or {}
    if presentation.get("response_type") != "DIAGNOSTIC_GUIDANCE":
        return None
    language = language or {}
    equipment = str((presentation.get("equipment") or {}).get("name") or "this unit")
    causes = _cited_claims(presentation, {"possible_cause"})
    facts = _cited_claims(presentation, {"information", "document_fact"})
    checks = _cited_claims(presentation, {"observation", "customer_procedure"})
    if not (causes or facts or checks):
        return None
    previous = _previous_texts(context)
    segments: List[Dict[str, Any]] = []
    followup_subject = None

    if cause_reference and causes:
        # Only the cause the customer pointed at, plus checks from its own section.
        selected = cause_reference.get("selected") or {}
        target = _norm(selected.get("source_text"))
        chosen = [c for c in causes if _norm(c.get("source_text")) == target]
        if chosen:
            heading = chosen[0].get("source_heading")
            causes = chosen
            facts = []
            checks = [c for c in checks if heading and _norm(c.get("source_heading")) == _norm(heading)]
    elif getattr(context, "is_followup", False):
        subject = _subject_terms(question)
        if subject:
            about = lambda c: bool(subject & _terms(c.get("source_text") or c.get("text")))  # noqa: E731
            if any(about(c) for c in causes + facts + checks):
                causes, facts, checks = ([c for c in group if about(c)] for group in (causes, facts, checks))
                followup_subject = " ".join(sorted(subject))
        if not followup_subject:
            # Do not repeat what the previous reply already listed, unless nothing new remains.
            shown = _previous_claim_sources(context)
            fresh = [[c for c in group if _norm(c.get("source_text")) not in shown] for group in (causes, facts, checks)]
            if any(fresh):
                causes, facts, checks = fresh

    if not (causes or facts or checks):
        return None
    subject_lead = (f"Here is what the {equipment} manual says about {followup_subject}:"
                    if followup_subject else None)

    if cause_reference and causes:
        style = "cause_detail"
        position = int(cause_reference.get("position", 0))
        ordinal = ORDINAL_WORDS[position] if 0 <= position < len(ORDINAL_WORDS) else "selected"
        segments.append(_segment("text", language.get("opening")
                                 or f"Here is the {ordinal} possible cause from the list:"))
        for claim in causes:
            segments.append(_segment("claim", _claim_body(claim), claim.get("evidence_ids")))
    elif causes:
        style = "symptom_causes"
        labels = sorted({_section_label(c.get("source_heading")) for c in causes if c.get("source_heading")})
        lead = (f"For {labels[0]} on the {equipment}, the manual lists "
                f"{'this possible cause' if len(causes) == 1 else 'these possible causes'}:"
                if len(labels) == 1 else
                f"The {equipment} manual lists these possible causes:")
        if language.get("opening"):
            segments.append(_segment("text", language["opening"]))
        segments.append(_segment("text", subject_lead or lead))
        for claim in causes:
            segments.append(_segment("list_item", _claim_body(claim), claim.get("evidence_ids")))
    else:
        style = "documented_facts"
        if language.get("opening"):
            segments.append(_segment("text", language["opening"]))
        if subject_lead:
            segments.append(_segment("text", subject_lead))
        for claim in facts:
            segments.append(_segment("claim", _claim_body(claim), claim.get("evidence_ids")))

    if causes and not cause_reference and facts:
        segments.append(_segment("text", "The manual also notes:"))
        for claim in facts:
            segments.append(_segment("claim", _claim_body(claim), claim.get("evidence_ids")))
    # Role distinction: only customer-approved procedures are presented as checks;
    # observations are things to notice and report, never instructions to act.
    procedures = [c for c in checks if c.get("kind") == "customer_procedure"]
    observations = [c for c in checks if c.get("kind") != "customer_procedure"]
    if procedures:
        segments.append(_segment("text", "Checks the manual approves for customers:"))
        for claim in procedures:
            segments.append(_segment("list_item", _claim_body(claim), claim.get("evidence_ids")))
    if observations:
        segments.append(_segment("text", "Details worth noting for the technician:"))
        for claim in observations:
            segments.append(_segment("list_item", _claim_body(claim), claim.get("evidence_ids")))
    if causes and _norm(LIMITATION) not in previous:
        segments.append(_segment("text", LIMITATION))
    question_text = language.get("question")
    if not question_text and style == "symptom_causes" and len(causes) > 1:
        question_text = "Do any of these match what you are seeing?"
    if question_text:
        segments.append(_segment("question", question_text))

    text = render_text(segments, presentation.get("citations") or [])
    return {"text": text, "segments": segments, "style": style,
            "composer": "qwen" if language else "deterministic"}


def render_text(segments: List[Dict[str, Any]], citations: List[Dict[str, Any]]) -> str:
    """Plain-text rendering (stored message, ticket transcript) with [n] citation markers."""
    number = {c.get("id"): i + 1 for i, c in enumerate(citations) if isinstance(c, dict)}
    lines = []
    for seg in segments:
        marks = "".join(f" [{number[c]}]" for c in seg.get("citation_ids") or [] if c in number)
        body = seg["text"].rstrip()
        if seg["kind"] == "list_item":
            lines.append(f"- {body}{marks}")
        else:
            lines.append(f"{body}{marks}")
    return "\n".join(lines)


def clarification_for_ambiguous(question: str, asset, tenant) -> Optional[Dict[str, Any]]:
    """For a message with no identifiable symptom, offer the asset's documented symptom sections.

    Only manual section titles from documents the customer may already see are
    used; no symptom, cause or instruction is invented.
    """
    from core.models import KnowledgeChunk
    from .conversation_context import is_self_contained_question
    words = [w for w in re.findall(r"[a-z]{3,}", _norm(question))
             if w not in {"the", "and", "this", "that", "unit", "not", "working", "properly", "work", "its",
                          "isnt", "doesnt", "broken", "problem", "issue", "wrong", "right", "help", "with", "has"}]
    if is_self_contained_question(question) or len(words) > 1 or asset is None:
        return None
    headings = (KnowledgeChunk.objects
                .filter(tenant=tenant, document__asset=asset, document__is_rag_enabled=True,
                        document__is_confidential=False, is_quarantined=False,
                        heading__iregex=r"symptom\s*:")
                .order_by("document_id", "chunk_index").values_list("heading", flat=True))
    labels = []
    for heading in headings:
        label = _section_label(heading)
        if label and label not in labels:
            labels.append(label)
    if len(labels) < 2:
        return None
    options = ", ".join(labels[:-1]) + f", or {labels[-1]}"
    return {"text": f"Could you tell me a little more about what is happening? Which of these is closest: {options}?",
            "options": labels}
