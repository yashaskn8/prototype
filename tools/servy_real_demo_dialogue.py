"""Intent-aware, evidence-bound dialogue for the read-only company RAG demo.

No model can author citations, approved procedures, diagnoses or new claims.
This layer only chooses among already authorized exact source-backed claims
from the normal customer-answer contract. Its optional conversational language
never asserts a new technical fact.
"""
import hashlib
import re

from core.services.content_safety import detect_document_prompt_injection
from core.services.query_decomposition import (
    ask_text, negated_text, relevance_text, retrieval_focus,
)

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
    "causes": "I couldn't verify a documented cause for that symptom in the approved passages. What are you actually observing: does the equipment fail to start, start but perform poorly, stop intermittently, or display a fault code? Please share the exact code if one is shown.",
    "observations": "I couldn't verify a customer-safe check for that specific issue. I won't suggest an unapproved procedure.",
    "next": "I couldn't verify an approved customer check for this issue. Please ask an authorized technician for model-specific help.",
    "meaning": "I couldn't verify the meaning of that code or condition from the approved passages for this equipment.",
    "documents": "I couldn't find a relevant, source-backed statement in the approved documentation for that question.",
    "general": "I couldn't verify a relevant answer in the approved documentation. Please share the specific symptom or displayed code.",
}
_HANDOFF_RECIPIENT = re.compile(
    r"\b(?:technician|service team|service staff|service department|"
    r"service desk|support)\b", re.I,
)
_HANDOFF_REQUEST = re.compile(
    r"\b(?:details?|information|summari[sz](?:e|es|ed|ing)?|summary|"
    r"tell|give|share|report|prepare|provide|include|pass on)\b", re.I,
)
_HANDOFF_PROCEDURE_REQUEST = re.compile(
    r"\b(?:how (?:do|can|should) i|how to|can i|may i|should i)\b"
    r".{0,100}\b(?:open|remove|replace|repair|fix|reset|restart|adjust|"
    r"measure|probe|wire|bypass|charge|clean|test|operate|run)\b|"
    r"\b(?:tell|ask|instruct|explain)\b.{0,80}\b(?:how to|to)\s+"
    r"(?:open|remove|replace|repair|fix|reset|restart|adjust|measure|"
    r"probe|wire|bypass|charge|clean|test|operate|run)\b", re.I,
)
_HANDOFF_TECHNICAL_REQUEST = re.compile(
    r"\b(?:why|cause|causes|reason|diagnos(?:e|is|tic)|troubleshoot(?:ing)?|"
    r"fix(?:es|ing)?|repair(?:ing)?|solve|solution|resolve|recommend(?:ation)?|"
    r"what should i do|what can i do|how (?:do|can|should) i|"
    r"is it safe to|is it normal to)\b", re.I,
)
_START_STOP_SYMPTOM = (
    r"\b(?:start(?:s|ed|ing)?\b.{0,60}\b"
    r"(?:stop(?:s|ped|ping)?|shuts?\s+(?:down|off))|"
    r"(?:stop(?:s|ped|ping)?|shuts?\s+(?:down|off))\b.{0,35}\b"
    r"after\s+(?:it\s+)?start(?:s|ed|ing)?)\b"
)
_HANDOFF_PERSISTENCE = re.compile(
    r"\b(?:issue|problem|symptom)\b.{0,30}\b(?:still happening|"
    r"continues?|persists?|recurs?)\b|\b(?:still happening|"
    r"still occurs|still stops?|problem persists|issue continues)\b", re.I,
)
_HANDOFF_CODE = re.compile(r"\b(?:E|F|ERR)[- ]?\d{1,3}\b", re.I)
_HANDOFF_NO_CODE = re.compile(
    r"\b(?:no (?:displayed )?(?:fault |error |alarm )?code|"
    r"(?:not|isn't|doesn't|does not) (?:show|display|have) "
    r"(?:any )?(?:displayed )?(?:fault |error |alarm )?code|"
    r"(?:no|not|isn't|doesn't|does not) (?:any )?(?:fault |error |alarm )?code "
    r"(?:is )?(?:shown|displayed|on|present)|"
    r"nothing (?:is )?(?:shown|displayed) on the display)\b", re.I,
)
_HANDOFF_TIME = re.compile(
    r"\b(?:since|yesterday|today|last (?:night|week|month)|"
    r"every|each time|once|twice|daily|weekly|hourly|"
    r"\d+\s*(?:minutes?|hours?|days?|weeks?|months?)|"
    r"sometimes|often|occasionally|intermittently|constantly|frequently)\b|"
    r"\b(?:issue|problem|symptom)\s+(?:first )?(?:started|began)\b", re.I,
)
_HANDOFF_OPERATING_STATE = re.compile(
    r"\b(?:starts?|running|stopped|shuts?\s+down|trips?|"
    r"while (?:cooling|heating|running)|during operation)\b", re.I,
)
_HANDOFF_VISIBLE = re.compile(
    r"\b(?:visible|see|saw|smoke|sparks?|burning smell|leak\w*|dripping|"
    r"noise|rattl\w*|display shows|indicator shows|residue|damage)\b", re.I,
)
_HANDOFF_RECENT_CHANGE = re.compile(
    r"\b(?:recent(?:ly)?|service|maintenance|repair|installation|"
    r"installed|replaced|changed|power outage)\b", re.I,
)
_HANDOFF_TRIED = re.compile(
    r"\b(?:i|we)\s+(?:have\s+)?(?:already\s+)?"
    r"(?:tried|checked|cleaned|restarted|reset|inspected|replaced|changed|called)\b",
    re.I,
)
_HANDOFF_SAFETY = re.compile(
    r"\b(?:smoke|sparks?|burning smell|electric(?:al)? shock|"
    r"burn(?:ing)?|unsafe|safety concern)\b", re.I,
)
_HANDOFF_QUESTION_START = re.compile(
    r"^(?:if|whether|what if|what (?:should|can|could|do|does|is|are)\s+(?:i|you)|"
    r"how (?:should|can|could|do|does)\s+(?:i|you)|"
    r"(?:can|could|should|may)\s+(?:i|you)|"
    r"is there|are there|does anyone|would it)\b", re.I,
)
_HANDOFF_CLAUSE_SPLIT = re.compile(
    r"(?<=[.!?])\s+|\s+[—–]\s+|,\s*(?=(?:what|how|can i|could i|"
    r"should i|can you|could you|please)\b)", re.I,
)


def _handoff_has_symptom(text):
    return bool(TECHNICAL_SYMPTOMS.search(text))


def is_technician_handoff_request(question):
    """Recognize nontechnical requests to prepare information for service."""
    q = " ".join(str(question or "").split())
    return bool(
        q and _HANDOFF_RECIPIENT.search(q) and _HANDOFF_REQUEST.search(q)
        and not _HANDOFF_PROCEDURE_REQUEST.search(q)
        and not _HANDOFF_TECHNICAL_REQUEST.search(q)
    )


def build_technician_handoff(question, history=()):
    """Build a source-free handoff checklist from customer-authored reports.

    The result contains no technical diagnosis or procedure. Each reported
    detail remains explicitly attributable to the customer.
    """
    turns = [*(history or []), question]
    clauses = []
    for turn in turns:
        normalized = " ".join(str(turn or "").split())
        for raw_clause in _HANDOFF_CLAUSE_SPLIT.split(normalized):
            clause = raw_clause.strip(" \t\r\n,;:-")
            if not clause or _HANDOFF_QUESTION_START.search(clause):
                continue
            if not any(pattern.search(clause) for pattern in (
                TECHNICAL_SYMPTOMS,
                _HANDOFF_PERSISTENCE, _HANDOFF_CODE,
                _HANDOFF_NO_CODE, _HANDOFF_TIME, _HANDOFF_OPERATING_STATE,
                _HANDOFF_VISIBLE, _HANDOFF_RECENT_CHANGE, _HANDOFF_TRIED,
                _HANDOFF_SAFETY,
            )):
                continue
            if clause not in clauses:
                clauses.append(clause[:260])

    reported_details = [
        {"text": clause, "provenance": "customer-reported"}
        for clause in clauses[:6]
    ]
    has_symptom = any(_handoff_has_symptom(clause) for clause in clauses)
    has_code = any(_HANDOFF_CODE.search(clause) or _HANDOFF_NO_CODE.search(clause)
                   for clause in clauses)
    has_timing = any(_HANDOFF_TIME.search(clause) for clause in clauses)
    has_operating_state = any(_HANDOFF_OPERATING_STATE.search(clause)
                              for clause in clauses)
    has_visible_sign = any(_HANDOFF_VISIBLE.search(clause) for clause in clauses)
    has_recent_change = any(_HANDOFF_RECENT_CHANGE.search(clause)
                            for clause in clauses)
    has_tried_action = any(_HANDOFF_TRIED.search(clause) for clause in clauses)
    has_safety_concern = any(_HANDOFF_SAFETY.search(clause) for clause in clauses)

    checklist = []
    if not has_symptom:
        checklist.append("The exact symptom: what the equipment does or does not do.")
    if not has_timing:
        checklist.append("When it began and how often it occurs.")
    if not has_operating_state:
        checklist.append("What the equipment was doing when it occurred.")
    if not has_code:
        checklist.append("Any displayed code, copied exactly, or that no code was shown.")
    if not has_visible_sign:
        checklist.append("Any visible signs noticed.")
    if not has_recent_change:
        checklist.append("Any recent changes, installation, or service.")
    if not has_tried_action:
        checklist.append("What you already tried and what happened.")
    if not has_safety_concern:
        checklist.append("Any safety concern.")

    if not has_symptom:
        follow_up = "What exact symptom is the equipment showing or experiencing?"
    elif not has_code:
        follow_up = "Is any code displayed? If so, what exact characters appear?"
    elif not has_timing:
        follow_up = "When did this start, and how often does it happen?"
    elif not has_tried_action:
        follow_up = "What have you already tried, and what happened?"
    else:
        follow_up = None
    return {
        "reported_details": reported_details,
        "checklist": checklist,
        "follow_up_question": follow_up,
        "reported_details_label": "Customer-reported so far (not independently verified)",
        "checklist_label": "Details to share with the technician, if known",
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



def vague_problem_clarification(question, history):
    """Ask one concrete symptom question when no specific problem is known.

    Never infer a failure from a generic 'not working' report. If an earlier
    customer turn already names a symptom or fault code, let the regular
    evidence-backed follow-up resolver use that state instead.
    """
    q = _words(question)
    if not re.fullmatch(
        r"(?:it(?: is|'s)? |the (?:unit|machine|equipment|ac|system) (?:is )?)?"
        r"(?:not working|doesn'?t work|won'?t work|is broken|isn'?t functioning)"
        r"(?: properly| at all)?", q,
    ):
        return None
    substantive = re.compile(
        r"\b(?:e[- ]?\d{1,3}|f[- ]?\d{1,3}|"
        r"cool(?:ing)?|heat(?:ing)?|noise|noisy|leak\w*|drip\w*|"
        r"overheat\w*|compressor|fan|filter|display|airflow|"
        r"starts? then stops?|trips?|shuts? down)\b", re.I,
    )
    if any(substantive.search(str(turn)) for turn in (history or [])):
        return None
    return "What happens when you try to use it—for example, does it fail to start, run without the expected result, or show a code?"


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
        return "What are you confirming? Please describe what you observed."
    return None




# A narrow, domain-specific demo must not turn unrelated questions into
# technical answers merely because retrieval still remembers an AC symptom.
PERSON_TOPIC = re.compile(
    r"\b(?:who(?:'s| is| was| are)|whose|whom|"
    r"gender|sex of|pronouns? of|age of|birthday of|"
    r"nationality of|religion of|ethnicity of|"
    r"biography of|personal life of|"
    r"tell me about (?:a person|someone|[a-z]+ (?:kn|patel|kumar)))\b",
    re.I,
)
UNRELATED_TOPIC = re.compile(
    r"\b(?:weather|stock price|sports score|"
    r"capital of|president of|prime minister of|"
    r"tell (?:me )?(?:a|another) joke|write (?:a|me a) poem|"
    r"movie review|celebrity|horoscope|"
    r"solve (?:this )?math|python programming)\b",
    re.I,
)
TECHNICAL_SYMPTOMS = re.compile(
    r"\b(?:overheat(?:ing|ed|s)?|hot to (?:the )?touch|too hot|"
    r"not cool(?:ing)?|poor cooling|weak cooling|no cooling|"
    r"not heat(?:ing)?|no heating|"
    r"leak(?:ing|age)?|dripping|"
    r"noise|noisy|vibration|vibrating|"
    r"not starting|not turning on|not working|"
    r"low airflow|weak airflow|" + _START_STOP_SYMPTOM + r"|"
    r"e[- ]?\d{1,3}|f[- ]?\d{1,3})\b", re.I,
)
SPECIFIC_EVIDENCE = {
    "overheating": (
        re.compile(r"\b(?:overheat(?:ing|ed|s)?|hot to (?:the )?touch|too hot)\b", re.I),
        re.compile(r"\b(?:overheat(?:ing|ed|s)?|hot to (?:the )?touch|too hot|excessive heat|high temperature)\b", re.I),
    ),
    "noise or vibration": (
        re.compile(r"\b(?:nois(?:e|y)|rattl\w*|vibrat\w*|buzz\w*|hum\w*)\b", re.I),
        re.compile(r"\b(?:nois(?:e|y)|rattl\w*|vibrat\w*|buzz\w*|hum\w*)\b", re.I),
    ),
    "water leakage": (
        re.compile(r"\b(?:leak\w*|drip\w*|water escap\w*|water collect\w*)\b", re.I),
        re.compile(r"\b(?:leak\w*|drip\w*|water|drain\w*|condensat\w*)\b", re.I),
    ),
    "poor cooling or airflow": (
        re.compile(r"\b(?:not cool(?:ing)?|poor cooling|weak cooling|no cooling|weak airflow|low airflow)\b", re.I),
        re.compile(r"\b(?:cool(?:ing)?|air ?flow|capacity)\b", re.I),
    ),
    "starting and stopping": (
        re.compile(_START_STOP_SYMPTOM, re.I),
        re.compile(_START_STOP_SYMPTOM, re.I),
    ),
    "low-pressure or capacity fault": (
        re.compile(r"\b(?:low[ -]?pressure|capacity[ -]?fault)\b", re.I),
        re.compile(r"\b(?:low[ -]?pressure|capacity[ -]?fault)\b", re.I),
    ),
    "high-pressure or heat-exchange fault": (
        re.compile(r"\b(?:high[ -]?pressure|heat[ -]?exchange)\b", re.I),
        re.compile(r"\b(?:high[ -]?pressure|heat[ -]?exchange)\b", re.I),
    ),
    "fan or temperature-sensor fault": (
        re.compile(r"\b(?:fan(?:\s+or\s+)?temperature[- ]sensor|"
                   r"temperature[- ]sensor|fan)\b", re.I),
        re.compile(r"\b(?:fan|temperature[- ]sensor)\b", re.I),
    ),
}


def out_of_scope_response(question):
    """Only explicitly non-equipment domains are rejected before RAG.

    Unknown *technical* questions must still go through evidence checks and
    can abstain normally. A user's name is never inferred from cooling text.
    """
    q = _words(question)
    # Questions about the equipment's manufacturer or authorized service
    # contact are still equipment questions. They must be sourced or refused
    # for lack of evidence, not misclassified as personal biography.
    equipment_contact = bool(re.search(
        r"\bwho (?:is|was|are) (?:the |an? )?"
        r"(?:manufacturer|maker|oem|installer|service provider|technician)\b", q
    ))
    if PERSON_TOPIC.search(q) and not equipment_contact:
        return (
            "This demo answers equipment questions from approved company manuals. "
            "I can't verify information about a person's identity or personal "
            "attributes from those manuals, so I won't guess. "
            "You can ask about this unit, its symptoms, or a documented fault code."
        )
    if UNRELATED_TOPIC.search(q):
        return (
            "This demo is limited to equipment support using approved company manuals. "
            "I don't have verified source information for that unrelated question. "
            "Ask about the selected equipment, symptoms, or manual."
        )
    return None


# Short component-level questions often refer to the active reported symptom.
# They are not independent diagnoses or reasons to fetch any other asset.
_COMPONENT_ELLIPSIS = re.compile(
    r"^(?:what (?:about|if)|how about|and|"
    r"could (?:it|that|this) be|is (?:it|that|this) (?:the|a)?|"
    r"maybe (?:the|a)?|does (?:the|a))\s+", re.I
)
_COMPONENT_NOUN = re.compile(
    r"\b(?:compressor|fan|coil|filter|refrigerant|thermostat|sensor|"
    r"condenser|outdoor unit|drain|remote|airflow|vent|electrical supply)\b",
    re.I
)
_REPORTABLE_DETAIL_SUBJECT = re.compile(
    r"\b(?:code|operating mode|mode|affected indoor units?|indoor units?|"
    r"trend(?: data| history)?|ambient (?:conditions?|temperature)|"
    r"alarm history|sensor readings?|display|indicator|readout)\b", re.I,
)
_REPORTING_INTENT = re.compile(
    r"\b(?:record|note|report|mention|include|share|log|document|"
    r"pass along|relevant|useful|capture)\b", re.I,
)


def is_component_followup(question):
    """True only for bounded elliptical references to a named component.

    A short unrelated question or an independent symptom does not inherit
    the earlier equipment issue. This function makes no technical claims.
    """
    q = " ".join(str(question or "").strip().split())
    if (not q or len(q) > 180 or out_of_scope_response(q)
            or TECHNICAL_SYMPTOMS.search(q)):
        return False
    return bool(_COMPONENT_ELLIPSIS.match(q) and _COMPONENT_NOUN.search(q))


def is_reportable_detail_request(question):
    """Recognize questions about documented observation/report fields."""
    q = _words(question)
    return bool(_REPORTABLE_DETAIL_SUBJECT.search(q)
                and _REPORTING_INTENT.search(q))


def is_contextual_followup(question):
    """Reuse earlier symptoms only for genuine elliptical equipment questions.

    Short length alone is NEVER sufficient. An explicit new symptom is a new
    search, even if the message contains words like 'it' or 'this'.
    """
    q = _words(question)
    if not q or out_of_scope_response(q):
        return False
    if TECHNICAL_SYMPTOMS.search(q):
        return False
    if is_technician_handoff_request(q):
        return True
    if is_reportable_detail_request(q):
        return True
    if is_component_followup(q):
        return True
    if is_explanation_followup(q):
        return True
    if re.fullmatch(
        r"(?:what (?:next|now|else)|what should i do(?: next)?|"
        r"any(?:thing)? (?:else|more)|tell me more|"
        r"other causes|more causes|"
        r"what (?:are )?(?:the )?(?:possible|documented|other|more)? ?causes?"
        r"(?: (?:are )?documented)?|"
        r"what can i (?:safely )?(?:check|observe|report)|"
        r"what does (?:it|that|this) mean|"
        r"why is that|why does it keep happening|why does this keep happening|"
        r"why does it happen again|how about that|"
        r"i (?:already )?(?:checked|tried) (?:that|it)|"
        r"no(?:pe| error(?: code)?| warning)?|yes|yeah|"
        r"same issue|still not fixed)\??",
        q,
    ):
        return True
    # A mere pronoun ("that", "these", "it") cannot turn an unrelated
    # question into another equipment retrieval. Require a known continuation
    # phrase, not a short-word heuristic.
    if re.search(
        r"\b(?:what else|what other|another cause|other reason|what next|"
        r"same issue|still not fixed|already checked|already tried|"
        r"that did not help|that didn't help)\b", q
    ):
        return True
    return False


def _specific_symptom_supported(question, source_context):
    """Allow a passage to support only the symptom it actually covers.

    A multi-symptom question may have evidence for one part and none for
    another. Keep the matching claim; the response separately states which
    requested symptoms remain unsupported.
    """
    q = _words(question)
    requested = [
        evidence for query, evidence in SPECIFIC_EVIDENCE.values()
        if query.search(q)
    ]
    return not requested or any(
        evidence.search(source_context or "") for evidence in requested
    )


def _requested_topics(question, primary):
    """Keep independently answerable intents in a mixed customer question."""
    q = _words(question)
    topics = [primary]
    asks_causes = bool(re.search(
        r"\b(?:causes?|reasons?|why|what (?:could|might) be wrong|"
        r"diagnos\w*|root cause)\b|"
        r"\b(?:could|might|may)\b.{0,100}\b(?:cause|explain|account for|"
        r"lead to|result in)\b", q
    ))
    asks_observations = bool(re.search(
        r"\b(?:what|which)\b.{0,80}\b(?:safely\s+)?"
        r"(?:check|inspect|observe|report|notice|note|record)\b|"
        r"\b(?:safe|customer-safe)\s+(?:checks?|observations?)\b", q
    ))
    asks_next = bool(re.search(
        r"\b(?:what should i do|what next|next step|how do i (?:fix|resolve))\b", q
    ))
    for requested, condition in (
        ("causes", asks_causes),
        ("observations", asks_observations),
        ("next", asks_next),
    ):
        if condition and requested not in topics:
            topics.append(requested)
    return topics


def requested_topics(question):
    """Independently answerable intents in the customer's question."""
    return _requested_topics(question, question_topic(question))


def focused_causal_query(question):
    """One focused retrieval query for a mixed question, or None.

    Keeps the customer's relevant reported condition together with the
    substantive request. Drops only incidental negatives ("the filter is
    clean", "no warning is showing"), completed checks, and content-free
    request tails; a relevant context clause (for example a blocked or cramped
    location) is never discarded.
    """
    if out_of_scope_response(question) is not None:
        return None
    return retrieval_focus(question)


# --- clause-aware relevance -------------------------------------------------
# Customer-reported negatives must neither dominate retrieval nor become a
# subject every answer claim has to mention. These helpers only read the
# customer's words; they add no technical statement.
_HYPOTHESIS = re.compile(
    r"\b(?:could|might|may|can)\b\s+(?P<h>.{3,140}?)\s+"
    r"(?:cause|explain|account for|lead to|result in)\b", re.I)
_LOCATION_WORDS = frozenset({
    "outdoor", "indoor", "unit", "external", "internal", "system", "machine",
    "equipment", "split", "this", "that", "these", "those", "manual", "according",
    "troubleshooting", "poor", "weak", "cooling", "with", "from",
})
_OBSTRUCTION_FAMILY = re.compile(
    r"^(?:block|obstruct|restrict|clearance|clog|blockage)", re.I)
_OBSTRUCTION_SUPPORT = (
    r"(?:block\w*|obstruct\w*|restrict\w*|clearance|clog\w*|"
    r"insufficient (?:space|clearance)|too close)"
)


def _hypothesis(ask):
    """Return (support_regex, hypothesis_text) for 'Could X explain Y?'."""
    match = _HYPOTHESIS.search(ask or "")
    if not match:
        return None, ""
    text = match.group("h")
    supports = []
    for token in re.findall(r"[a-z]{4,}", text.casefold()):
        if token in _LOCATION_WORDS:
            continue
        # Match each asked word's own stem family only. "Blocked" must not be
        # widened to "restricted": that would admit unrelated causes.
        if re.match(r"(?:block|obstruct|clog)", token):
            supports.append(r"(?:block|obstruct|clog)\w*")
        else:
            supports.append(re.escape(token[:5]) + r"\w*")
    if not supports:
        return None, text
    return re.compile(r"\b(?:" + "|".join(dict.fromkeys(supports)) + r")", re.I), text


class _FocusPlan:
    """What the current question genuinely asks about."""
    def __init__(self, question):
        self.relevant = relevance_text(question)
        ask = ask_text(question)
        self.hypothesis, hypothesis_text = _hypothesis(ask)
        # With an explicit hypothesis, the hypothesis defines the subject; a
        # reported context clause only helps retrieval and is not a hard
        # requirement on every claim.
        required_source = (
            re.sub(re.escape(hypothesis_text), " ", ask, flags=re.I)
            if self.hypothesis is not None else self.relevant
        )
        self.required = [pair for pair in _FOCUSED_SUBJECTS
                         if re.search(pair[0], required_source, re.I)]
        wanted = [pair for pair in _FOCUSED_SUBJECTS
                  if re.search(pair[0], self.relevant, re.I)]
        self.wanted_support = [pair[1] for pair in wanted]
        negated = negated_text(question)
        self.negated_support = [
            supported for asked, supported in _FOCUSED_SUBJECTS
            if re.search(asked, negated, re.I) and not re.search(asked, self.relevant, re.I)
        ]

    def accepts(self, claim, citation):
        source = f"{claim.get('source_text', '')} {citation.get('heading') or ''}"
        claim_only = str(claim.get("source_text", ""))
        if not all(re.search(supported, source, re.I) for _asked, supported in self.required):
            return False
        # The hypothesis ("could X explain Y?") constrains only cause claims; a
        # separately requested safe observation is judged on its own evidence.
        if (self.hypothesis is not None
                and claim.get("kind") in {"possible_cause", "information"}
                and not self.hypothesis.search(claim_only)):
            return False
        # A component the customer reported fine/already checked, and did not
        # ask about, is not offered as an unrelated answer.
        if any(re.search(supported, claim_only, re.I) for supported in self.negated_support):
            asked_about = any(re.search(supported, claim_only, re.I)
                              for supported in self.wanted_support)
            if not asked_about and not (
                    self.hypothesis is not None and self.hypothesis.search(claim_only)):
                return False
        return True

    def relevance(self, claim):
        """Count distinctive request words in the claim's own sentence."""
        words = {w[:5] for w in re.findall(r"[a-z]{4,}", self.relevant.casefold())
                 if w not in _GENERAL_WORDS and w not in _LOCATION_WORDS}
        claim_words = {w[:5] for w in re.findall(r"[a-z]{4,}",
                                                 str(claim.get("source_text", "")).casefold())}
        return len(words & claim_words)


def _claim_topic(claim, primary):
    if claim.get("kind") == "possible_cause":
        return "causes"
    if claim.get("kind") == "observation":
        return "observations"
    if claim.get("kind") == "customer_procedure":
        return "next"
    return primary


_GENERAL_WORDS = frozenset({
    "what", "who", "where", "when", "why", "which", "how",
    "this", "that", "these", "those", "they", "them",
    "does", "did", "can", "could", "should", "would", "will",
    "with", "from", "into", "about", "there", "have",
    "the", "and", "are", "was", "were", "for", "its",
    "our", "your", "you", "not", "some", "any", "all",
    "unit", "system", "machine", "equipment", "manual",
    "please", "tell", "explain", "information", "give",
    "know", "mean", "means", "support", "answer",
    "also", "more", "than", "say", "says",
})


_FOCUSED_SUBJECTS = (
    (r"\bfilter\w*\b", r"\bfilter\w*\b"),
    (r"\bcoil\w*\b", r"\bcoil\w*\b"),
    (r"\bfan\b", r"\bfan\w*\b"),
    (r"\bcompressor\w*\b", r"\bcompressor\w*\b"),
    (r"\brefrigerant\b", r"\brefrigerant\b"),
    (r"\bthermostat\b", r"\bthermostat\b"),
    (r"\bsensor\b", r"\bsensor\w*\b"),
    (r"\bdrain\w*\b", r"\bdrain\w*\b"),
    (r"\b(?:airflow|air flow)\b", r"\b(?:airflow|air flow)\b"),
    (r"\b(?:outdoor|external)\s+unit\b", r"\b(?:outdoor|external|condenser)\b"),
    (r"\b(?:remote|controller)\b", r"\b(?:remote|controller)\b"),
    (r"\b(?:settings?|setpoint)\b", r"\b(?:settings?|setpoint)\b"),
    (r"\b(?:heating|doesn.t heat|not warm)\b", r"\b(?:heat\w*|warm\w*)\b"),
    (r"\b(?:power supply|power failure|no power)\b",
     r"\b(?:power|supply|electrical)\b"),
    (r"\b(?:low[ -]?pressure|capacity[ -]?fault)\b",
     r"\b(?:low[ -]?pressure|capacity[ -]?fault)\b"),
    (r"\b(?:high[ -]?pressure|heat[ -]?exchange)\b",
     r"\b(?:high[ -]?pressure|heat[ -]?exchange)\b"),
    (r"\b(?:error |fault |alarm )?code\b", r"\bcode\b"),
    (r"\b(?:operating|operation|running)\s+mode\b|\bmode\b",
     r"\b(?:operating\s+)?mode\b"),
    (r"\b(?:affected\s+)?indoor\s+units?\b|\baffected\s+units?\b",
     r"\bindoor\s+units?\b"),
    (r"\btrend(?:\s+data|\s+history|s)?\b",
     r"\btrend(?:\s+data|\s+history|s)?\b"),
    (r"\bambient\s+(?:conditions?|temperature)\b",
     r"\b(?:ambient|outdoor)\s+temperature\b|\bambient\s+conditions?\b"),
    (r"\balarm\s+history\b", r"\balarm\s+history\b"),
    (r"\b(?:display|displayed|indicator|readout)\b",
     r"\b(?:display(?:ed)?|indicator|readout|code)\b"),
    (r"\bsensor\s+readings?\b|\b(?:record|note|report)\s+readings?\b",
     r"\b(?:sensor|readings?)\b"),
    (r"\b(?:fan(?:\s+or\s+)?temperature[- ]sensor|"
     r"temperature[- ]sensor|fan)\b",
     r"\b(?:fan|temperature[- ]sensor)\b"),
    (r"\b(?:not starting|doesn.t start|won.t start|no power)\b",
     r"\b(?:start\w*|power|supply)\b"),
    (_START_STOP_SYMPTOM, _START_STOP_SYMPTOM),
)

_OBSERVATION_DETAIL_INTENT = re.compile(
    r"\b(?:record|report|note|mention|include|share|log|document|"
    r"pass along|relevant|useful|capture|details?|information|give|tell)\b",
    re.I,
)
_OBSERVATION_FACETS = (
    re.compile(r"\b(?:fault|error|alarm)?\s*code\b", re.I),
    re.compile(r"\b(?:operating|operation|running)?\s*mode\b", re.I),
    re.compile(r"\b(?:affected\s+)?indoor\s+units?\b|\baffected\s+units?\b", re.I),
    re.compile(r"\btrend(?:\s+(?:data|history))?\b", re.I),
    re.compile(r"\bambient\s+(?:conditions?|temperature)\b", re.I),
    re.compile(r"\balarm\s+(?:history|log)\b", re.I),
    re.compile(r"\b(?:sensor\s+)?readings?\b", re.I),
    re.compile(r"\b(?:display|indicator|readout)\b", re.I),
    re.compile(r"\b(?:manufacturer(?:'s)?|specified)\s+ranges?\b", re.I),
)


def _focused_observation_text(source_text, question):
    """Show only the requested item from a simple cited observation list.

    The source sentence and citation remain attached unchanged. This narrows
    display text only; it does not split evidence, add technical content, or
    turn a documentation sentence into an equipment procedure.
    """
    if not _OBSERVATION_DETAIL_INTENT.search(str(question or "")):
        return source_text
    match = re.fullmatch(
        r"\s*(?P<verb>record|report|note|log|document|capture)\s+"
        r"(?P<items>[^.!?;:]+)[.!?]?\s*",
        str(source_text or ""), re.I,
    )
    if not match or re.search(
        r"\b(?:compare|only|specified|manufacturer|range)\b",
        match.group("items"), re.I,
    ):
        return source_text
    items = [part.strip() for part in re.split(
        r",\s*(?:and\s+)?|\s+and\s+", match.group("items"), flags=re.I
    ) if part.strip()]
    if len(items) < 2:
        return source_text
    asked = [pattern for pattern in _OBSERVATION_FACETS
             if pattern.search(str(question or ""))]
    if not asked:
        return source_text
    selected = [item for item in items if any(pattern.search(item) for pattern in asked)]
    if not selected or len(selected) == len(items):
        return source_text
    return f"{match.group('verb').capitalize()} {' and '.join(selected)}."


def _direct_subject_match(question, claim, citation):
    """An authorized citation must mention the subject asked about now.

    Subjects come from what the customer is asking (and reporting as a
    problem), never from components they say are fine or already checked.
    """
    return _FocusPlan(question).accepts(claim, citation)


def _direct_general_overlap(question, claim, citation):
    """A general question requires its own content overlap, not old history."""
    if is_contextual_followup(question):
        return True
    tokens = {w for w in re.findall(r"[a-z0-9]{3,}", _words(relevance_text(question)))
              if w not in _GENERAL_WORDS}
    if not tokens:
        return False
    # The document title can name the equipment but is not evidence of what
    # the manual says about a new topic. Match the actual cited source/heading.
    source = f"{claim['source_text']} {citation.get('heading') or ''}".casefold()
    words = set(re.findall(r"[a-z0-9]{3,}", source))
    return bool(tokens & words)


_EXPLANATION_FOLLOWUP = re.compile(
    r"\b(?:explain|elaborate|expand on|break down|go into detail on|"
    r"go into more detail on)\b.{0,100}\b(?:each|both|those|these|them|"
    r"the above|the listed|all)\b.{0,45}\b(?:issues?|problems?|causes?|"
    r"reasons?|possibilities|factors?|items?)\b|"
    r"\b(?:explain|elaborate|expand|break down)\b.{0,80}\b"
    r"(?:each|those|these|them)\b|"
    r"\b(?:more detail|in detail|in depth)\b.{0,60}\b"
    r"(?:above|those|these|issues?|causes?|possibilities)\b",
    re.I,
)


def is_explanation_followup(question):
    """Recognize a request to expand on a previously cited equipment answer."""
    return bool(_EXPLANATION_FOLLOWUP.search(" ".join(str(question or "").split())))


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
    if re.search(
        r"\b(?:causes?|reasons?|why|what (?:could|might) be wrong|diagnos\w*|root cause)\b|"
        r"\b(?:could|might|may)\b.{0,100}\b(?:cause|explain|account for|lead to|result in)\b",
        q,
    ):
        return "causes"
    if is_reportable_detail_request(q):
        return "observations"
    if re.search(
        r"\b(?:how (?:do|can|should) i\b|how to\b|"
        r"what settings? should i (?:use|select)|"
        r"(?:can i|may i)\s+(?!safely\s+(?:observe|check|report)\b|(?:observe|check|report)\b)"
        r"|is it safe to\s+(?!(?:observe|check|report)\b))", q
    ):
        return "next"
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
                 r"note|record|report|mention|include|share|log|document|"
                 r"visible|display|indicator|notice|readout)\b", q):
        return "observations"
    if re.search(r"\b(?:manual|document|documentation|checklist|"
                 r"source|according to|listed|template|log|record)\b", q):
        return "documents"
    # Preserve context only for genuine referential follow-ups.
    # "Who is Yashas?" must never inherit "causes" merely for being short.
    if last_topic in {"causes", "documents", "observations"} and is_explanation_followup(question):
        return last_topic
    if last_topic in TOPICS and is_component_followup(question):
        # The component question inherits the current *type* of question,
        # not arbitrary prior answer text. This helps "what about the fan?"
        # remain about causes after a causes conversation.
        return last_topic if last_topic in {"causes", "documents", "observations"} else "general"
    if last_topic in TOPICS and is_contextual_followup(question):
        return last_topic
    if (re.search(_START_STOP_SYMPTOM, q, re.I)
            or re.search(r"\b(?:fault|alarm|error code|e-\d+|f-\d+|not cooling|"
                         r"not working|weak airflow|leak|rattl|noise)\b", q)):
        return "causes"
    return "general"


def expand_display_observation_query(question, retrieval_query=None):
    """Map display/indicator wording to the manual's documented code terminology.

    This changes retrieval terms only. It does not supply a code value or add
    any customer-facing technical claim.
    """
    query = " ".join(str(retrieval_query or question or "").split())
    q = _words(question)
    if (question_topic(question) == "observations"
            and re.search(r"\b(?:display|indicator|readout)\b", q)
            and re.search(r"\b(?:see|shown|showing|record|note|report|observe)\b", q)
            and not re.search(r"\b(?:error|fault|alarm)\s+code\b|\bcode\b", q)):
        return query + " displayed fault code"
    return query


def wants_additional(question):
    """Detect an explicit request for more of the current answer's facts.

    Do not treat words such as "additional" inside a multi-part request
    (for example, "causes and additional diagnostic checks") as a request to
    suppress every fact already shown. "Other" and "additional" must qualify
    causes/reasons; broad conversational phrases remain supported.
    """
    return bool(re.search(
        r"\b(?:what else|anything else|"
        r"other (?:documented )?(?:possible )?causes?|"
        r"another (?:documented )?(?:possible )?cause|"
        r"additional (?:documented )?(?:possible )?causes?|"
        r"more causes?|more reasons?|any more(?: causes?| reasons?)?|"
        r"tell me more|give me more)\b",
        _words(question),
    ))


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
        r"what is|what's|explain|define|how does|"
        r"what (?:can|should) i (?:check|do|note|record|log)|what next|next steps?|"
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
    # A general fact question cannot be answered with a cause-list item
    # or an unapproved action just because a word overlaps.
    return frozenset({"information", "document_fact"})


def _already_reported_code_absence(text):
    return bool(re.search(
        r"\b(?:no|without|not showing|does(?:n'?t| not) show)"
        r"(?:\s+(?:error|fault|alarm|displayed))?"
        r"\s+(?:codes?|warnings?|alarm)\b"
        r"|\bnothing (?:on|in) (?:the )?display\b", text, re.I
    ))


def _repeat_or_answered_followup(candidate, report):
    """A fixed follow-up is not useful when the customer already answered it."""
    if not candidate:
        return False
    if _already_reported_code_absence(report) and re.search(
        r"\b(?:code|warning|alarm)\b", candidate, re.I
    ):
        return True
    if _error_codes(report) and re.search(
        r"\b(?:is there|showing|display showing)\b", candidate, re.I
    ):
        return True
    return False


def _follow_up(topic, question, history=()):
    """Choose one safe, contextual observation; do not repeat known questions.

    The customer's reports are unverified, but can prevent us asking the
    same thing repeatedly. These questions never authorize physical work.
    """
    reports = " ".join(
        [str(t)[:220] for t in list(history or [])[-6:] if isinstance(t, str)]
        + [question or ""]
    )
    if topic == "causes":
        if _already_reported_code_absence(reports):
            return "Does this issue happen constantly, or only at certain times?"
        if _error_codes(reports):
            return "When does that displayed code appear?"
        return "Is a fault code or warning visible on the display?"
    if topic in {"observations", "next"}:
        if _already_reported_code_absence(reports):
            return "When did you first notice this symptom?"
        return "What can you see on the external display or indicator?"
    if topic == "meaning":
        return "Does the reported code appear consistently or intermittently?"
    if topic == "general":
        return "Which specific symptom or displayed code should I focus on?"
    return None

def focus_answer(contract, question, *, last_topic=None, previous_facts=(), plan=None,
                 history=()):
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
            preserve_topic = (
                bool(previous_facts) and is_contextual_followup(question)
                and last_topic in TOPICS
            )
            return {
                **contract, "summary": message, "answer": message,
                "presentation": {**presentation, "message": message},
            }, {"topic": last_topic if preserve_topic else topic,
                "facts": list(previous_facts)[:24], "suggestions": []}
        return contract, {"topic": last_topic, "facts": list(previous_facts)[:24],
                          "suggestions": []}
    original = contract.get("presentation") or {}
    off_topic = out_of_scope_response(question)
    topic = question_topic(question, last_topic)
    # Only resolve ambiguous follow-ups with model assistance. Explicit
    # customer requests remain authoritative over an open-weight model.
    if (not _explicit_intent(question)
            and isinstance(plan, dict)
            and plan.get("intent") in TOPICS):
        topic = plan["intent"]
    topics = _requested_topics(question, topic)
    focus = _FocusPlan(question)
    allowed = (frozenset().union(*(_match_kinds(item) for item in topics))
               if not off_topic else frozenset())
    citations_by_id = {
        c.get("id"): c for c in original.get("citations", [])
        if isinstance(c, dict) and isinstance(c.get("id"), str)
    }
    collected = []
    shown = set(previous_facts)
    seen = set()
    novelty = wants_additional(question) and topic == last_topic == "causes"
    for claim in original.get("supporting_claims", []):
        if not isinstance(claim, dict) or claim.get("kind") not in allowed:
            continue
        if (not isinstance(claim.get("text"), str)
            or not claim["text"].strip()
            or not isinstance(claim.get("source_text"), str)
            or not claim["source_text"].strip()):
            continue
        # A claim is a *view* of an original authorized source sentence, not
        # free-form model output. Reject any altered technical wording.
        if claim["kind"] == "possible_cause" and (
            claim["text"] != "Possible documented cause: " + claim["source_text"]
        ):
            continue
        if claim["kind"] in {"information", "observation", "customer_procedure"} and (
            claim["text"] != claim["source_text"]
        ):
            continue
        # Ingestion already scans sources; this is a second fail-closed
        # barrier for a stale/corrupt contract or poisoned retrieved passage.
        if detect_document_prompt_injection(claim["source_text"]):
            continue
        evidence = claim.get("evidence_ids")
        if (not isinstance(evidence, list) or len(evidence) != 1
            or evidence[0] not in citations_by_id):
            continue
        citation = citations_by_id[evidence[0]]
        if claim["kind"] == "customer_procedure":
            # Never promote a plausible-looking source sentence to a
            # customer action without the underlying authorization record.
            authorized_presentation = any(
                isinstance(item, dict)
                and item.get("id") == claim.get("id")
                and item.get("source_text") == claim["source_text"]
                and item.get("evidence_ids") == evidence
                for item in original.get("customer_authorized_actions", [])
            )
            authorized_contract = any(
                isinstance(item, dict)
                and item.get("action_id") == claim.get("id")
                and item.get("safety_classification") == "CUSTOMER_APPROVED"
                and item.get("document_id") == citation.get("document_id")
                and item.get("chunk_id") == citation.get("chunk_id")
                for item in contract.get("customer_safe_actions", [])
            )
            if not (authorized_presentation and authorized_contract):
                continue
        if (claim.get("kind") == "observation"
                and _already_reported_code_absence(" ".join([*history[-6:], question]))
                and re.search(r"\b(?:code|warning|alarm)\b",
                              claim["source_text"], re.I)):
            continue
        # An authorized citation alone does not establish relevance to
        # the CURRENT user question.
        source_context = (
            str(claim["source_text"]) + " " + str(citation.get("heading") or "")
        )
        if (not _specific_symptom_supported(focus.relevant, source_context)
                or not focus.accepts(claim, citation)):
            continue
        if topic == "general" and not _direct_general_overlap(
            question, claim, citation
        ):
            continue
        asked_codes = _error_codes(question)
        claim_codes = _error_codes(claim["source_text"])
        if (claim_codes and not asked_codes
                and not re.search(r"\b(?:fault|error|alarm)\s+codes?\b", question, re.I)):
            continue
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
        # An explicit request for additional causes may hide previously
        # shown cause facts, but it must not hide already-approved observations
        # or checks in the same multi-part question.
        if fp in seen or (
            novelty and claim["kind"] in {"possible_cause", "information"} and fp in shown
        ):
            continue
        seen.add(fp)
        collected.append((claim, citation, fp))

    # For a specific hypothesis ("could X explain Y?") rank the claim that
    # states X first; ties keep the authorized source order. Never filters.
    claim_limit = sum(4 if item == "causes" else 3 for item in topics)
    if focus.hypothesis is not None:
        collected.sort(key=lambda row: -focus.relevance(row[0]))
    collected = collected[:claim_limit]

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
        if off_topic:
            message = off_topic
        elif novelty:
            message = ("I don't have another distinct, documented answer for that question. "
                       "Please share a different symptom or displayed fault code.")
        else:
            message = NO_EVIDENCE[topic]
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
                   "can_resolve": False, "reason_code":
                       "OUT_OF_SCOPE_QUERY" if off_topic else "NO_RELEVANT_APPROVED_EVIDENCE"}
        # A failed contextual follow-up must not overwrite the topic of the
        # last successful cited answer. Otherwise "explain each issue" after
        # an unanswered observation request loses the earlier cause list.
        preserve_topic = (
            off_topic is not None
            or (bool(previous_facts) and is_contextual_followup(question))
        )
        return revised, {"topic": last_topic if preserve_topic and last_topic in TOPICS else topic,
                         "facts": list(previous_facts)[:24], "suggestions": []}

    follow_up = _follow_up(topic, question, history)
    style = "direct"
    if isinstance(plan, dict):
        from tools.servy_real_demo_local_planner import FOLLOW_UPS, STYLES
        if (plan.get("follow_up") in FOLLOW_UPS[topic]
                and not _repeat_or_answered_followup(
                    plan["follow_up"], " ".join([*history[-6:], question])
                )):
            follow_up = plan["follow_up"]
        if plan.get("style") in STYLES:
            style = plan["style"]
    selected_claims = []
    for source_claim, _citation, _fingerprint in collected:
        claim = dict(source_claim)
        if claim.get("kind") == "observation":
            claim["text"] = _focused_observation_text(
                claim["source_text"], question,
            )
        selected_claims.append(claim)
    selected_citations = [row[1] for row in collected]
    next_facts = list(dict.fromkeys([*previous_facts, *(row[2] for row in collected)]))[-24:]
    # Variation is limited to nontechnical connective language. No source
    # facts or customer-approved procedures can be generated by a model.
    lead = LEAD[topic]
    # Vary only connective language. No model is allowed to invent a cause.
    if topic == "causes" and previous_facts and wants_additional(question):
        lead = "Here are additional documented possibilities, if any. None confirms a diagnosis."
    elif topic == "causes" and _already_reported_code_absence(
        " ".join([*history[-6:], question])
    ):
        lead = "With no code reported, these are still only documented possibilities—not a confirmed diagnosis."
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
    if len(topics) > 1:
        lead = ("I found supporting evidence for part of your question. "
                "The sections separate documented claims from parts I could not verify.")

    requested_symptoms = [
        label for label, (query_pattern, _evidence_pattern) in SPECIFIC_EVIDENCE.items()
        if query_pattern.search(_words(focus.relevant))
    ]
    supported_symptoms = set()
    for claim, citation, _fingerprint in collected:
        source_context = f"{claim['source_text']} {citation.get('heading') or ''}"
        supported_symptoms.update(
            label for label, (query_pattern, evidence_pattern) in SPECIFIC_EVIDENCE.items()
            if query_pattern.search(_words(focus.relevant)) and evidence_pattern.search(source_context)
        )
    unsupported_symptoms = [label for label in requested_symptoms
                            if label not in supported_symptoms]
    symptom_limit = None
    if supported_symptoms and unsupported_symptoms:
        symptom_limit = (
            "The retrieved citations support " + ", ".join(sorted(supported_symptoms))
            + " only; they do not establish a cause for "
            + ", ".join(unsupported_symptoms) + "."
        )

    detail_limit = None
    if topic == "causes" and is_explanation_followup(question):
        has_new_detail = any(row[2] not in shown for row in collected)
        if not has_new_detail:
            detail_limit = (
                "The approved passages retrieved for this asset provide only the "
                "cited high-level categories; they do not contain a separately "
                "citable explanation of each category. I won't add technical details "
                "that the approved manuals do not support."
            )

    sections = []
    if len(topics) > 1:
        for section_topic in topics:
            section_claims = [claim for claim in selected_claims
                              if _claim_topic(claim, topic) == section_topic]
            if section_claims:
                sections.append({"number": len(sections) + 1,
                                 "title": HEADING[section_topic],
                                 "claims": section_claims})
            else:
                sections.append({"number": len(sections) + 1,
                                 "title": "What I could not verify",
                                 "claims": [{"text": NO_EVIDENCE[section_topic],
                                             "kind": "uncertainty",
                                             "evidence_ids": []}]})
    else:
        sections.append({"number": 1, "title": HEADING[topic],
                         "claims": selected_claims})
    if symptom_limit:
        sections.append({"number": len(sections) + 1,
                         "title": "Evidence limitation",
                         "claims": [{"text": symptom_limit, "kind": "uncertainty",
                                     "evidence_ids": []}]})
        lead = (lead + " " + symptom_limit).strip()
    if detail_limit:
        sections.append({"number": len(sections) + 1,
                         "title": "Evidence limitation",
                         "claims": [{"text": detail_limit, "kind": "uncertainty",
                                     "evidence_ids": []}]})
        lead = (lead + " " + detail_limit).strip()
    new_presentation = {
        **original,
        "view_mode": "focused",
        "sections": sections,
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
        "can_resolve": bool(set(topics) & {"next", "observations"}
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
