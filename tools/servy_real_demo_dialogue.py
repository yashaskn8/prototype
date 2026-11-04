"""Intent-aware, evidence-bound dialogue for the read-only company RAG demo.

No model can author citations, approved procedures, diagnoses or new claims.
This layer only chooses among already authorized exact source-backed claims
from the normal customer-answer contract. Its optional conversational language
never asserts a new technical fact.
"""
import hashlib
import re

TOPICS = frozenset({"causes", "observations", "next", "meaning", "documents", "general"})
FACT_KINDS = frozenset({"possible_cause", "information", "document_fact", "observation", "customer_procedure"})
HEADING = {
    "causes": "Documented possible causes",
    "observations": "Safe observations from the manual",
    "next": "Approved customer checks",
    "meaning": "What the manual says",
    "documents": "What the document says",
    "general": "What the manual supports",
}
LEAD = {
    "causes": "The approved manual lists these possibilities. It does not confirm which one is causing your problem.",
    "observations": "These are the observations the approved documentation supports; they are not permission to open or repair equipment.",
    "next": "Here is what the approved documentation permits you to observe or do. It does not authorize other procedures.",
    "meaning": "This is what the approved manual states about the reported code or condition.",
    "documents": "Here is the relevant information in the approved document.",
    "general": "Here is what I could verify in the approved documentation.",
}
NO_EVIDENCE = {
    "causes": "I couldn't verify a documented cause for that specific symptom from the available approved passages.",
    "observations": "I couldn't verify a customer-safe check for that specific issue. I won't suggest an unapproved procedure.",
    "next": "I couldn't verify an approved customer check for this issue. Please ask an authorized technician for model-specific help.",
    "meaning": "I couldn't verify the meaning of that code or condition from the approved passages for this equipment.",
    "documents": "I couldn't find a relevant, source-backed statement in the approved documentation for that question.",
    "general": "I couldn't verify a relevant answer in the approved documentation. Please share the specific symptom or displayed code.",
}
SHORT_TURNS = frozenset({"hi", "hello", "hey", "good morning", "good evening",
                         "thanks", "thank you", "thank you very much", "okay", "ok"})


def _words(text):
    return (text or "").strip().casefold().strip(" .!?,")


def conversational_turn(text):
    normalized = _words(text)
    if normalized in SHORT_TURNS:
        return ("Hi — what problem are you seeing with this equipment?"
                if normalized not in {"thanks", "thank you", "thank you very much", "ok", "okay"}
                else "You're welcome. If the issue is still happening, tell me what changed or what the display shows.")
    return None


def conversation_recap(question, history):
    """Summarize only authenticated customer reports; never invent results."""
    q = _words(question)
    if not history or not re.search(
        r"\b(?:summari[sz]e|recap|what (?:have|did) i (?:told|tell|say)|"
        r"what do you (?:know|remember) about (?:my|the) (?:problem|issue)|"
        r"what have we discussed)\b", q
    ):
        return None
    customer_reports = []
    for message in [history[0], *history[-3:]]:
        clean = " ".join(str(message or "").split())[:190]
        if clean and clean not in customer_reports:
            customer_reports.append(clean)
    if not customer_reports:
        return None
    return (
        "Here's what you've reported so far: "
        + " | ".join(customer_reports[:4])
        + ". These are your reports, not a verified diagnosis. "
          "Tell me what has changed, or ask about a specific documented cause."
    )


def followup_clarification(question, history):
    """Never infer which earlier suggestion an ambiguous reply refers to."""
    if not history:
        return None
    q = _words(question)
    if re.fullmatch(
        r"(?:i (?:have |already |have already )?(?:checked|tried|did|done) "
        r"(?:it|that|this)|(?:already )?done (?:that|it)|i checked)",
        q,
    ):
        return "Which specific check did you complete, and what did you observe? I don't want to assume it is resolved."
    if q in {"yes", "yes it is", "yes there is", "yep", "yeah"}:
        return "What exact code, warning, or message is visible on the display?"
    return None


def question_topic(question, last_topic=None):
    """Deterministic routing; no extra intents are invented from documents."""
    q = _words(question)
    if re.search(r"\b(?:[A-Z]{1,3}[- ]?\d{1,3})\b", question, re.I) and re.search(
        r"\b(?:showing|displays?|see|saw|code|alarm|fault)\b", q
    ) and not re.search(r"\b(?:cause|why|reason|fix|repair|check)\b", q):
        return "meaning"
    if last_topic in TOPICS and (
        q in {"no", "nope", "no error", "no error code", "no warning"}
        or re.search(
            r"\b(?:no (?:displayed )?(?:error|fault|alarm|code|warning)|"
            r"nothing (?:is )?(?:shown|displayed)|not showing a code)\b", q,
        )
    ):
        return "next"
    if re.search(r"\b(?:what does|what is|what's)\b.{0,90}\b(?:mean|indicate|signify)\b"
                 r"|\b(?:meaning|indicates?|denotes?|definition)\b"
                 r"|\b(?:explain|identify)\s+(?:the\s+)?(?:error|fault|alarm)\s+code\b", q):
        return "meaning"
    if re.search(r"\b(?:causes?|reasons?|why|what (?:could|might) be wrong|diagnos\w*|root cause)\b", q):
        return "causes"
    if re.search(r"\b(?:what (?:should|can) i do(?:\s+next)?|what next|next step|"
                 r"what now|how (?:can|do) i (?:fix|resolve)|how to (?:fix|resolve)|"
                 r"what else can i do)\b", q):
        return "next"
    if last_topic in TOPICS and re.search(
        r"\b(?:already checked|i checked|i tried|did that|still not|"
        r"didn't help|did not help|not resolved|not fixed|same issue)\b", q
    ) and not re.search(r"\b(?:why|cause|reason)\b", q):
        return "next"
    if re.search(r"\b(?:safe(?:ly)?|check|inspect|observe|observation|look at|"
                 r"report|visible|display|indicator|notice|readout)\b", q):
        return "observations"
    if re.search(r"\b(?:manual|document|documentation|checklist|"
                 r"source|according to|listed|template|log|record)\b", q):
        return "documents"
    if (last_topic in TOPICS and
        (len(q.split()) <= 5 or re.search(r"\b(?:more|another|other|still|that|then|it)\b", q))):
        return last_topic
    if re.search(r"\b(?:fault|alarm|error code|e-\d+|f-\d+|not cooling|"
                 r"not working|weak airflow|leak|rattl|noise)\b", q):
        return "causes"
    return "general"


def wants_additional(question):
    return bool(re.search(r"\b(?:what else|anything else|other|another|"
                          r"additional|more causes|more reasons|any more|"
                          r"tell me more|give me more)\b", _words(question)))


def _error_codes(text):
    """Fault identifiers must agree across user question and cited source."""
    return {
        re.sub(r"[^A-Z0-9]", "", match.group(0).upper())
        for match in re.finditer(r"\b(?:E|F|ERR)[- ]?\d{1,3}\b", text or "", re.I)
    }


def fact_fingerprint(claim, citation):
    """Stable across turns even when response-local citation IDs change."""
    value = "|".join([
        str(citation.get("document_id", "")),
        str(citation.get("chunk_id", "")),
        str(claim.get("kind", "")),
        str(claim.get("source_text", claim.get("text", ""))).strip().casefold(),
    ])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _explicit_intent(question):
    """Lock explicit customer requests so a model cannot change their meaning."""
    return bool(re.search(
        r"\b(?:why|causes?|reasons?|root cause|diagnos\w*|"
        r"what (?:can|should) i (?:check|do)|what next|next steps?|"
        r"safe(?:ly)?|inspect|observe|observations?|"
        r"what does|meaning|indicate|code|fault|alarm|"
        r"manual|document|source|according to|repair|fix|resolve)\b",
        _words(question),
    ))


def _match_kinds(topic):
    if topic == "causes":
        return frozenset({"possible_cause", "information"})
    if topic == "meaning":
        return frozenset({"information", "document_fact"})
    if topic == "documents":
        return frozenset({"document_fact", "information", "possible_cause"})
    if topic in {"next", "observations"}:
        return frozenset({"observation", "customer_procedure"})
    return FACT_KINDS


def _follow_up(topic, question):
    # Only questions about *user-observable reports*, never a procedure.
    if topic == "causes":
        if re.search(r"\b(?:error|fault|alarm|code)\b", question, re.I):
            return "When does that displayed fault appear?"
        return "Is the display showing an error code or warning?"
    if topic in {"observations", "next"}:
        return "What can you see on the external display or indicator?"
    if topic == "general":
        return "What exact symptom or displayed code are you seeing?"
    return None


def focus_answer(contract, question, *, last_topic=None, previous_facts=(), plan=None):
    """Keep only direct, valid, signed evidence matching the user's intent.

    The caller passes a trusted structured contract from build_customer_answer.
    Revalidate referenced claims and retain only used citations. Never show
    uncertainty boilerplate as if it came from a manual.
    """
    if contract.get("response_type") != "DIAGNOSTIC_GUIDANCE":
        # Rewrite ONLY the known generic no-evidence fallback. Preserve
        # policy refusals, safety escalations, revoked-document explanations
        # and exact-spec guards verbatim.
        presentation = contract.get("presentation") or {}
        if (contract.get("response_type") == "INSUFFICIENT_EVIDENCE"
            and not contract.get("reason_code")
            and presentation.get("message") ==
                "There is insufficient evidence to establish a safe answer for this equipment and question. Ask an authorized technician for help."):
            topic = question_topic(question, last_topic)
            message = NO_EVIDENCE[topic]
            return {
                **contract, "summary": message, "answer": message,
                "presentation": {**presentation, "message": message},
            }, {"topic": topic, "facts": list(previous_facts)[:24],
                "suggestions": []}
        return contract, {"topic": last_topic, "facts": list(previous_facts)[:24],
                          "suggestions": []}
    original = contract.get("presentation") or {}
    topic = question_topic(question, last_topic)
    # Only resolve ambiguous follow-ups with model assistance. Explicit
    # customer requests remain authoritative over an open-weight model.
    if (not _explicit_intent(question)
            and isinstance(plan, dict)
            and plan.get("intent") in TOPICS):
        topic = plan["intent"]
    allowed = _match_kinds(topic)
    citations_by_id = {
        c.get("id"): c for c in original.get("citations", [])
        if isinstance(c, dict) and isinstance(c.get("id"), str)
    }
    collected = []
    shown = set(previous_facts)
    seen = set()
    novelty = wants_additional(question) and topic == last_topic
    for claim in original.get("supporting_claims", []):
        if not isinstance(claim, dict) or claim.get("kind") not in allowed:
            continue
        if (not isinstance(claim.get("text"), str)
            or not claim["text"].strip()
            or not isinstance(claim.get("source_text"), str)):
            continue
        evidence = claim.get("evidence_ids")
        if (not isinstance(evidence, list) or len(evidence) != 1
            or evidence[0] not in citations_by_id):
            continue
        citation = citations_by_id[evidence[0]]
        asked_codes = _error_codes(question)
        if asked_codes and topic in {"causes", "meaning", "documents"}:
            # A fact about E-05 cannot answer what E-04 means. For a
            # code-specific question, require the same code in the exact
            # approved source sentence, not merely the retrieved chunk.
            if not asked_codes.issubset(_error_codes(claim["source_text"])):
                continue
        if (not isinstance(citation.get("document_id"), int)
            or not isinstance(citation.get("chunk_id"), int)
            or not isinstance(citation.get("url"), str)
            or not citation["url"].startswith("/demo-api/source/?citation=")):
            continue
        fp = fact_fingerprint(claim, citation)
        if fp in seen or (novelty and fp in shown):
            continue
        seen.add(fp)
        collected.append((claim, citation, fp))
        if len(collected) >= (4 if topic == "causes" else 3):
            break

    # A local model may rank/deselect permitted source IDs but can never add
    # a claim absent from the server-authorized candidate collection.
    if isinstance(plan, dict):
        chosen = plan.get("claim_ids")
        if isinstance(chosen, list) and chosen:
            rank = {id_: index for index, id_ in enumerate(chosen) if isinstance(id_, str)}
            planned = [row for row in collected if row[0].get("id") in rank]
            if planned:
                collected = sorted(planned, key=lambda row: rank[row[0]["id"]])
    if not collected:
        message = (("I don't have another distinct, documented answer for that question. "
                    "Please share a different symptom or displayed fault code.")
                   if novelty else NO_EVIDENCE[topic])
        revised = {**contract,
                   "response_type": "INSUFFICIENT_EVIDENCE",
                   "response_mode": "INSUFFICIENT_EVIDENCE",
                   "summary": message, "answer": message,
                   "presentation": {
                       **original,
                       "view_mode": "focused",
                       "response_type": "INSUFFICIENT_EVIDENCE",
                       "sections": [], "citations": [], "supporting_claims": [],
                       "customer_permitted_observations": [],
                       "customer_authorized_actions": [],
                       "message": message, "follow_up_question": None,
                   },
                   "citations": [], "references": [], "customer_safe_actions": [],
                   "can_resolve": False, "reason_code": "NO_RELEVANT_APPROVED_EVIDENCE"}
        return revised, {"topic": topic, "facts": list(previous_facts)[:24],
                         "suggestions": []}

    follow_up = _follow_up(topic, question)
    style = "direct"
    if isinstance(plan, dict):
        from tools.servy_real_demo_local_planner import FOLLOW_UPS, STYLES
        if plan.get("follow_up") in FOLLOW_UPS[topic]:
            follow_up = plan["follow_up"]
        if plan.get("style") in STYLES:
            style = plan["style"]
    selected_claims = [row[0] for row in collected]
    selected_citations = [row[1] for row in collected]
    next_facts = list(dict.fromkeys([*previous_facts, *(row[2] for row in collected)]))[-24:]
    # Variation is limited to nontechnical connective language. No source
    # facts or customer-approved procedures can be generated by a model.
    lead = LEAD[topic]
    if style == "friendly":
        lead = {
            "causes": "I found these possibilities in your approved manual, but I can't confirm which one applies yet.",
            "observations": "I can help with these documented observations without asking you to open or repair anything.",
            "next": "Here are the customer-permitted observations or checks supported by your approved manual.",
            "meaning": "Here's what I could verify about that code or condition in your approved manual.",
            "documents": "I found these source-backed details in the approved document.",
            "general": "Based on the approved manual, here's the information I can support.",
        }[topic]
    elif style == "brief":
        lead = {
            "causes": "Documented possibilities, not a confirmed diagnosis:",
            "observations": "Documented customer-safe observations:",
            "next": "Documented customer-permitted checks:",
            "meaning": "The approved manual states:",
            "documents": "The approved document states:",
            "general": "Verified manual information:",
        }[topic]
    new_presentation = {
        **original,
        "view_mode": "focused",
        "sections": [{"number": 1, "title": HEADING[topic], "claims": selected_claims}],
        "citations": selected_citations,
        "supporting_claims": selected_claims,
        "customer_permitted_observations": [
            c for c in selected_claims if c["kind"] == "observation"
        ],
        "customer_authorized_actions": [
            c for c in selected_claims if c["kind"] == "customer_procedure"
        ],
        "message": lead,
        "follow_up_question": follow_up,
    }
    response = {**contract, "presentation": new_presentation,
                "answer": lead + "\n" + "\n".join(c["text"] for c in selected_claims),
                "summary": lead,
                "citations": selected_citations, "references": selected_citations,
                "customer_safe_actions": [
                    a for a in contract.get("customer_safe_actions", [])
                    if a.get("action_id") in {c.get("id") for c in selected_claims}
                ],
                "can_resolve": bool(topic in {"next", "observations"}
                                    and any(c.get("kind") in {"observation", "customer_procedure"}
                                            for c in selected_claims)
                                    and contract.get("can_resolve"))}
    suggestions = {
        "causes": ["What can I safely observe?", "What should I do next?"],
        "observations": ["What documented causes could explain this?", "What should I do next?"],
        "next": ["What possible causes are documented?"],
        "meaning": ["What possible causes are documented?", "What should I report?"],
        "documents": ["What possible causes are documented?"],
        "general": ["What possible causes are documented?", "What can I safely observe?"],
    }[topic]
    return response, {"topic": topic, "facts": next_facts, "suggestions": suggestions}
