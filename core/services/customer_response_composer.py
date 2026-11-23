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

ORDINAL_WORDS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
                 "ninth", "tenth", "eleventh", "twelfth")
LIMITATION = "These are possibilities from the approved manuals, not a confirmed diagnosis of your unit."
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
            key = (_norm(claim.get("text")), _norm(claim.get("source_heading"))) if isinstance(claim, dict) else None
            if (isinstance(claim, dict) and claim.get("kind") in kinds and claim.get("evidence_ids")
                    and key not in seen):
                seen.add(key)
                out.append(claim)
    return out


def _displayed(claim: Dict[str, Any]) -> Dict[str, Any]:
    """The reference record of a claim shown to the customer (for "the first cause")."""
    return {"kind": claim.get("kind"), "source_text": claim.get("source_text"),
            "source_heading": claim.get("source_heading"), "evidence_ids": list(claim.get("evidence_ids") or [])}


def _label_title(label: str) -> str:
    return label[:1].upper() + label[1:] if label else label


def _acknowledge(answer: str) -> str:
    said = " ".join(str(answer or "").split()).rstrip(" .!")
    said = said.lower() if said.isupper() else said[:1].lower() + said[1:]
    return f"Thanks, I've noted that: {said}." if said else "Thanks, I've noted that."


def compose_reply_acknowledgement(context, equipment: str) -> Optional[Dict[str, Any]]:
    """A reply to our own question with nothing further documented for the current topic.

    Records the customer's answer and says plainly that the approved manuals add
    nothing more specific, instead of switching to an unrelated symptom.
    """
    if not getattr(context, "answers_pending_question", False):
        return None
    answers = getattr(context, "resolved_answers", None) or []
    topics = getattr(context, "active_topics", None) or []
    topic = topics[0] if len(topics) == 1 else "this issue"
    segments = [
        _segment("text", _acknowledge(answers[-1]["answer"] if answers else "")),
        _segment("text", f"The approved {equipment} manuals don't describe {topic} in any more detail than "
                         f"what I've already shared, so I can't narrow it down further from them."),
        _segment("question", "If it keeps happening, would you like a technician to take a look? "
                             "I can prepare a service request with what you've told me."),
    ]
    return {"text": render_text(segments, []), "segments": segments, "style": "reply_acknowledged",
            "composer": "deterministic", "displayed_claims": []}


def compose_response(contract: Dict[str, Any], question: str, context=None,
                     cause_reference: Optional[Dict[str, Any]] = None,
                     language: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """Return {"text", "segments", "style", "composer", "displayed_claims"} for a supported answer, else None.

    ``language`` may carry validated Qwen wording for "opening" and "question";
    anything missing falls back to fixed wording. ``displayed_claims`` is the exact
    ordered list of claims shown, which later ordinal references resolve against.
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
    replying = bool(getattr(context, "answers_pending_question", False)) and not cause_reference

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
    elif replying:
        # A short answer to our own question stays on the current topic: only
        # claims from the manual sections last discussed, and only ones not shown
        # already. Evidence from other sections (poor cooling after a noise
        # question) is not promoted just because retrieval returned it.
        topics = set(getattr(context, "active_topics", None) or [])
        shown = _previous_claim_sources(context)
        on_topic = (lambda c: _section_label(c.get("source_heading")) in topics) if topics else (lambda c: True)
        causes, facts, checks = ([c for c in group if on_topic(c) and _norm(c.get("source_text")) not in shown]
                                 for group in (causes, facts, checks))
        if not (causes or facts):
            return compose_reply_acknowledgement(context, equipment)
    elif getattr(context, "is_followup", False):
        # A broad equipment-failure question is not a component follow-up.
        # Filtering on its generic words ("causes", "not work") can silently
        # reduce an authorized multi-section overview to one matching fault.
        from .customer_answer import is_generic_cause_overview
        from .conversation_context import is_explain_all_request
        original = getattr(context, "original_complaint", "")
        broad_request = (is_generic_cause_overview(question) or is_explain_all_request(question)
                         or bool(original and is_generic_cause_overview(f"{question} {original}")))
        subject = set() if broad_request else _subject_terms(question)
        if subject:
            about = lambda c: bool(subject & _terms(c.get("source_text") or c.get("text")))  # noqa: E731
            if any(about(c) for c in causes + facts + checks):
                causes, facts, checks = ([c for c in group if about(c)] for group in (causes, facts, checks))
                followup_subject = " ".join(sorted(subject))
        if not followup_subject and not broad_request:
            # An explicit request for the causes (or for all of them) is answered
            # with the complete authorized list, including ones shown before.
            # Otherwise do not repeat what the previous reply already listed,
            # unless nothing new remains.
            shown = _previous_claim_sources(context)
            fresh = [[c for c in group if _norm(c.get("source_text")) not in shown] for group in (causes, facts, checks)]
            # A leftover note is not a new answer: when every cause and fact was
            # already listed ("What else could it be?"), keep them rather than
            # replying with only an observation.
            if fresh[0] or fresh[1]:
                causes, facts, checks = fresh

    if not (causes or facts or checks):
        return None
    displayed: List[Dict[str, Any]] = []
    if replying:
        answers = getattr(context, "resolved_answers", None) or []
        segments.append(_segment("text", _acknowledge(answers[-1]["answer"] if answers else "")))
    subject_lead = f"Here's what I found about the {followup_subject}:" if followup_subject else None

    if cause_reference and causes:
        style = "cause_detail"
        position = int(cause_reference.get("position", 0))
        ordinal = ORDINAL_WORDS[position] if 0 <= position < len(ORDINAL_WORDS) else "selected"
        segments.append(_segment("text", language.get("opening") or f"About the {ordinal} possible cause:"))
        for claim in causes:
            segments.append(_segment("claim", _claim_body(claim), claim.get("evidence_ids")))
            displayed.append(_displayed(claim))
    elif causes:
        style = "symptom_causes"
        labels = []
        for claim in causes:
            label = _section_label(claim.get("source_heading"))
            if label not in labels:
                labels.append(label)
        if language.get("opening"):
            segments.append(_segment("text", language["opening"]))
        if subject_lead:
            segments.append(_segment("text", subject_lead))
        elif len(labels) == 1 and labels[0]:
            segments.append(_segment("text", (
                f"One possible cause of {labels[0]} on the {equipment} is:" if len(causes) == 1 else
                f"Here are the possible causes of {labels[0]} on the {equipment}:")))
        else:
            segments.append(_segment("text", (
                f"A few different problems can cause this on the {equipment}. "
                "Here are the possibilities, grouped by what you would notice:")))
        grouped = len(labels) > 1 and not subject_lead
        for label in labels:
            group = [c for c in causes if _section_label(c.get("source_heading")) == label]
            if grouped and label:
                segments.append(_segment("text", f"{_label_title(label)}:"))
            for claim in group:
                segments.append(_segment("list_item", _claim_body(claim), claim.get("evidence_ids")))
                displayed.append(_displayed(claim))
    else:
        style = "documented_facts"
        if language.get("opening"):
            segments.append(_segment("text", language["opening"]))
        if subject_lead:
            segments.append(_segment("text", subject_lead))
        for claim in facts:
            segments.append(_segment("claim", _claim_body(claim), claim.get("evidence_ids")))
            displayed.append(_displayed(claim))

    if causes and not cause_reference and facts:
        segments.append(_segment("text", "Also worth knowing:"))
        for claim in facts:
            segments.append(_segment("claim", _claim_body(claim), claim.get("evidence_ids")))
            displayed.append(_displayed(claim))
    # Role distinction: only customer-approved procedures are presented as checks;
    # observations are things to notice and report, never instructions to act.
    procedures = [c for c in checks if c.get("kind") == "customer_procedure"]
    observations = [c for c in checks if c.get("kind") != "customer_procedure"]
    if procedures:
        segments.append(_segment("text", "Things you can safely check yourself:"))
        for claim in procedures:
            segments.append(_segment("list_item", _claim_body(claim), claim.get("evidence_ids")))
            displayed.append(_displayed(claim))
    if observations:
        segments.append(_segment("text", "Useful details to note for a technician:"))
        for claim in observations:
            segments.append(_segment("list_item", _claim_body(claim), claim.get("evidence_ids")))
            displayed.append(_displayed(claim))
    # Every multi-cause list carries the limitation; a repeated single-cause
    # detail does not restate it.
    last_response = ((getattr(context, "last_contract", None) or {}).get("presentation") or {}).get("response") or {}
    previous_causes = {_norm(c.get("source_text")) for c in last_response.get("displayed_claims") or []
                       if isinstance(c, dict) and c.get("kind") == "possible_cause"}
    new_list = {_norm(c.get("source_text")) for c in causes} != previous_causes
    if causes and (_norm(LIMITATION) not in previous or (len(causes) > 1 and new_list)):
        segments.append(_segment("text", LIMITATION))
    from .customer_answer import MAX_SECTION_CLAIMS
    if style == "symptom_causes" and len(causes) >= MAX_SECTION_CLAIMS:
        # Lists are capped; say so instead of implying the list is complete.
        segments.append(_segment("text", f"I've listed the first {len(causes)} documented possibilities; the manuals "
                                         "may describe more. Tell me the specific symptom to narrow it down."))
    question_text = language.get("question")
    if not question_text and style == "symptom_causes" and len(causes) > 1:
        question_text = "Do any of these match what you are seeing?"
    if question_text:
        segments.append(_segment("question", question_text))

    text = render_text(segments, presentation.get("citations") or [])
    return {"text": text, "segments": segments, "style": style,
            "composer": "qwen" if language else "deterministic", "displayed_claims": displayed}


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


def clarification_for_ambiguous(question: str, asset, tenant, customer=None) -> Optional[Dict[str, Any]]:
    """For a message with no identifiable symptom, offer the asset's documented symptom sections.

    Only manual section titles from documents the customer may already see are
    used; no symptom, cause or instruction is invented.
    """
    from rest_framework.exceptions import APIException
    from .conversation_context import is_self_contained_question
    from .customer_answer import source_chunk
    from .retriever import _knowledge_candidates
    words = [w for w in re.findall(r"[a-z]{3,}", _norm(question))
             if w not in {"the", "and", "this", "that", "unit", "not", "working", "properly", "work", "its",
                          "isnt", "doesnt", "broken", "problem", "issue", "wrong", "right", "help", "with", "has"}]
    if is_self_contained_question(question) or len(words) > 1 or asset is None:
        return None
    customer = customer or getattr(asset, "customer", None)
    if customer is None:
        return None
    # Section titles are content too: only chunks that pass the same per-chunk
    # authorization gate as cited evidence may contribute one.
    candidates = sorted(
        (c for c in _knowledge_candidates(tenant, asset, customer, False, allow_failed=True, backend="sparse")
         if c.document.asset_id == asset.id and re.search(r"symptom\s*:", c.heading or "", re.I)),
        key=lambda c: (c.document_id, c.chunk_index))[:200]
    labels = []
    for chunk in candidates:
        try:
            source_chunk(tenant=tenant, customer=customer, asset=asset,
                         document_id=chunk.document_id, chunk_id=chunk.id)
        except APIException:
            continue
        heading = chunk.heading
        label = _section_label(heading)
        if label and label not in labels:
            labels.append(label)
    if len(labels) < 2:
        return None
    options = ", ".join(labels[:-1]) + f", or {labels[-1]}"
    return {"text": f"Could you tell me a little more about what is happening? Which of these is closest: {options}?",
            "options": labels}
