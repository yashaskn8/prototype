"""
Conversation Context and Query Contextualization Service for Servy RAG.

Provides attributable multi-turn context extraction and retrieval query
rewriting for Ask Servy:
- Bounded recent-turn window preserving attributable diagnostic state.
- Explicit distinction between reported symptoms, suggested checks,
  confirmed completed checks, and technician advice.
- Query contextualization that incorporates equipment identity, active issue,
  prior observations, and completed steps into a targeted retrieval query.
- Strict anti-hallucination guarantees: never invents completed checks,
  causes, measurements, or fault codes.
- Safe deterministic rule-based contextualization with optional local Ollama enhancement.
"""

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Common equipment symptom and component indicators
SYMPTOM_INDICATORS = [
    "not cooling", "does not cool", "doesn't cool",
    "poor cooling", "weak cooling", "insufficient cooling",
    "not heating", "poor heating", "weak heating",
    "weak airflow", "low airflow", "no airflow", "restricted airflow",
    "water leaking", "water leak", "dripping", "condensate leak",
    "making noise", "abnormal noise", "rattling", "grinding", "vibration",
    "unusual smell", "burning smell", "odor",
    "not turning on", "power issue", "tripping", "trips breaker",
    "blinking light", "error code", "fault code", "warning light",
    "suction failure", "fails to draw", "draw failure", "sample error",
    "frozen", "icing", "frost on coil", "ice buildup",
    "noise", "noisy", "leak", "leaking", "smell", "stopped working", "not working",
]
# "no water leak", "there is no noise now", "not showing an error code": a negated
# observation is recorded as such and never counted as a reported symptom.
_NEGATION_BEFORE = re.compile(
    r"\b(?:no|not|never|none|without|isn['’]?t|aren['’]?t|doesn['’]?t|don['’]?t|didn['’]?t|"
    r"hasn['’]?t|haven['’]?t|wasn['’]?t|nothing)\b(?:\s+[a-z'’]+){0,3}\s*$", re.I)

# A completed check needs a named component, not a following clause such as
# "cleaned but cooling" or a running-state report about the equipment.
_COMPONENT_WORD = r"(?!(?:but|and|or|was|is|still|after|before|because|so|then)\b)[a-z0-9_-]+"
_COMPONENT = rf"({_COMPONENT_WORD}(?:\s+{_COMPONENT_WORD})?)"

# Words indicating that the user confirmed performing a check
COMPLETION_PATTERNS = [
    re.compile(r"\b(?:i\s+)?(?:already\s+)?(?:checked|inspected|cleaned|verified|confirmed|did|done|tested)\s+(?:that|this|it|already)\b", re.I),
    re.compile(r"\b(?:checked\s+that\s+already|already\s+checked|i\s+checked\s+it|already\s+done|done\s+that|i\s+did\s+that)\b", re.I),
    re.compile(rf"\b(?:the\s+)?{_COMPONENT}\s+(?:was|were|has been|have been)\s+(?:cleaned|washed|replaced|checked|inspected)\b", re.I),
    re.compile(rf"\b(?:i\s+have\s+)?(?:already\s+)?(?:cleaned|washed|replaced)\s+(?:the\s+)?{_COMPONENT}\b", re.I),
    re.compile(rf"\b(?:i\s+have\s+)?(?:already\s+)?(?:checked|inspected)\s+(?:the\s+)?{_COMPONENT}\b", re.I),
]

# Patterns indicating the user is asking what to do next
# A negative statement must never imply that a customer has completed a check.
_NEGATED_COMPLETION = re.compile(
    r"\b(?:haven['’]?t|hasn['’]?t|hadn['’]?t|didn['’]?t|don['’]?t|"
    r"have\s+not|has\s+not|did\s+not|do\s+not|not|never|"
    r"cannot|can['’]?t)\b(?:\s+\w+){0,5}\s+"
    r"(?:check(?:ed)?|inspect(?:ed)?|clean(?:ed)?|wash(?:ed)?|"
    r"verify|verified|confirm(?:ed)?|do(?:ne)?|test(?:ed)?|replac(?:e|ed))\b",
    re.I,
)


def _confirmed_checks(text: str, suggested_actions: List[str]) -> List[str]:
    """Keep only affirmative and attributable customer-reported completions."""
    # "What is inspected during routine maintenance?" asks; it reports nothing done.
    # Judged per sentence: "I checked that already. What next?" still reports a check.
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(str(text or "").split()))
    text = " ".join(s for s in sentences if s and not _is_question(s))
    if not text or _NEGATED_COMPLETION.search(text):
        return []
    for pattern in COMPLETION_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        if match.lastindex and match.group(1):
            component = _clean_phrase(match.group(1))
            return [component] if component else []
        # Pronouns cannot identify one action if several were offered.
        return [suggested_actions[0]] if len(suggested_actions) == 1 else []
    return []


def completed_action_matches(claim_text: str, completed_checks: List[str]) -> bool:
    """Require multiword component agreement; one shared word is not enough."""
    ignored = {
        "the", "and", "for", "with", "that", "this", "unit", "equipment",
        "check", "checked", "inspect", "inspected", "verify", "verified",
        "observe", "observed", "visible", "visually", "report", "reported",
        "note", "record", "initial", "customer", "already", "dust",
    }
    def words(value):
        return set(re.findall(r"[a-z0-9]{3,}", str(value).casefold())) - ignored
    candidate = words(claim_text)
    if len(candidate) < 2:
        return False
    for completed in completed_checks or []:
        done = words(completed)
        if len(done) >= 2 and len(done & candidate) >= 2:
            if len(done & candidate) / min(len(done), len(candidate)) >= 0.8:
                return True
    return False


NEXT_STEP_PATTERNS = [
    re.compile(r"\b(?:what\s+next|what\s+should\s+i\s+do(?:\s+next)?|what\s+now|what\s+else|what\s+to\s+do|what\s+is\s+next)\b", re.I),
    re.compile(r"\b(?:and\s+then|next\s+step|now\s+what|what's\s+next)\b", re.I),
    re.compile(r"\b(?:still\s+not\s+working|problem\s+persists|issue\s+continues|didn't\s+help|did\s+not\s+work|same\s+issue)\b", re.I),
]


_QUESTION_START = re.compile(
    r"^\s*(?:what|which|when|where|why|how|who|is|are|does|do|did|can|could|should|will|would|may|must)\b", re.I)
# References to the selected equipment itself are not references to the earlier topic.
_EQUIPMENT_REFERENCE = re.compile(
    r"\b(?:this|that|the|my|our)\s+(?:unit|equipment|ac|a/c|system|machine|module|model|device|appliance)\b", re.I)
_BACK_REFERENCE = re.compile(
    r"\b(?:it|that|this|they|them|its|those|these|already|still|again|else|also|instead)\b"
    r"|\b(?:what|how)\s+about\b|\bthe\s+(?:same|other|first|second|previous|last)\b", re.I)
_CONTENT_STOPWORDS = frozenset(
    "what which when where why how who is are does do did can could should will would may must the a an "
    "for in on of to and or with my our your any there their be been being have has had get i we you me "
    "unit equipment system machine".split())


def is_self_contained_question(text: str) -> bool:
    """A complete new question that names its own subject and does not refer back."""
    value = " ".join(str(text or "").split())
    if not _QUESTION_START.match(value):
        return False
    without_equipment = _EQUIPMENT_REFERENCE.sub(" ", value)
    if _BACK_REFERENCE.search(without_equipment):
        return False
    content = [w for w in re.findall(r"[a-z]{3,}", without_equipment.lower()) if w not in _CONTENT_STOPWORDS]
    return len(content) >= 3


_GREETING = re.compile(
    r"^\s*(?:(?:hi|hii+|hello|hey|hiya|greetings|good\s+(?:morning|afternoon|evening|day)|"
    r"thanks?|thank\s+you|thank\s+you\s+very\s+much|thanks\s+a\s+lot|ok(?:ay)?|cool|great|"
    r"bye|goodbye|see\s+you)(?:\s+(?:there|servy|again|so\s+much|very\s+much))?[\s!.,]*)+$",
    re.I)


def is_greeting(text: str) -> bool:
    """A message that is only a greeting, thanks or sign-off, with no request."""
    value = " ".join(str(text or "").split())
    return bool(value) and len(value) <= 60 and bool(_GREETING.match(value))


def is_supported_answer(contract: Dict[str, Any]) -> bool:
    """A stored assistant answer that showed cited, validated guidance."""
    presentation = (contract or {}).get("presentation") or {}
    return (presentation.get("response_type") == "DIAGNOSTIC_GUIDANCE"
            and bool(presentation.get("citations"))
            and bool(presentation.get("supporting_claims")))


# "Explain each issue", "explain them all", "explain the causes mentioned above",
# "please explain these issues": a request to explain every previously listed cause.
_EXPLAIN_VERB = r"(?:explain|elaborate(?:\s+on)?|describe|detail|break\s+down|tell\s+me\s+(?:more\s+)?about|go\s+through)"
_ALL_REFERENCE = (
    r"(?:each|every|all|all\s+of|each\s+of|both|these|those|the\s+above|them|them\s+all|all\s+of\s+them|"
    r"each\s+one|one\s+by\s+one)")
_ITEM_WORDS = r"(?:issues?|causes?|reasons?|problems?|possibilities|points?|items?|ones?|them|of\s+them|of\s+these|of\s+those)"
_EXPLAIN_ALL = re.compile(
    rf"\b{_EXPLAIN_VERB}\s+(?:the\s+)?{_ALL_REFERENCE}\b(?:\s+(?:the\s+)?(?:listed\s+|possible\s+|above\s+)?{_ITEM_WORDS})?"
    rf"|\b{_EXPLAIN_VERB}\s+(?:the\s+)?(?:listed|possible|above|previous|mentioned)\s+{_ITEM_WORDS}"
    rf"|\b{_EXPLAIN_VERB}\s+(?:the\s+)?{_ITEM_WORDS}\s+(?:mentioned|listed|given|shown)\s+(?:above|earlier|before)"
    rf"|\b{_ALL_REFERENCE}\s+{_ITEM_WORDS}\s*,?\s*(?:explained|in\s+detail)\b",
    re.I)


def is_explain_all_request(text: str) -> bool:
    return bool(_EXPLAIN_ALL.search(" ".join(str(text or "").split())))


_LIST_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)]|\(\d+\))\s+(?P<body>.+?)\s*$")
_CITATION_MARK = re.compile(r"\s*\[\d+\]")


def _norm_line(text: str) -> str:
    return " ".join(_CITATION_MARK.sub("", str(text or "")).split()).strip(" .").casefold()


def split_pasted_history(text: str, previous_texts: List[str]) -> Tuple[str, List[str]]:
    """Separate the customer's own request from text pasted from an earlier answer.

    Returns ``(request, quoted_items)``. A line is treated as pasted history when
    it is a list item or it repeats a line of a previous assistant reply; the
    remaining lines are the actual request. Without pasted content the message
    is returned unchanged with no quoted items.
    """
    value = str(text or "")
    lines = [line for line in value.splitlines() if line.strip()]
    if len(lines) < 2:
        return value.strip(), []
    previous = set()
    for block in previous_texts or []:
        for line in str(block or "").splitlines():
            body = _LIST_LINE.match(line)
            normalized = _norm_line(body.group("body") if body else line)
            if normalized:
                previous.add(normalized)
    request, quoted = [], []
    for line in lines:
        body = _LIST_LINE.match(line)
        normalized = _norm_line(body.group("body") if body else line)
        if body:
            quoted.append(_CITATION_MARK.sub("", body.group("body")).strip())
        elif normalized in previous or line.lstrip().startswith(">"):
            continue
        else:
            request.append(line.strip())
    if not quoted and len(request) == len(lines):
        return value.strip(), []
    return " ".join(request).strip(), quoted


@dataclass
class ConversationContext:
    conversation_uuid: str
    asset_id: Optional[int] = None
    asset_name: str = ""
    asset_model: str = ""
    asset_code: str = ""
    product_name: str = ""
    original_complaint: str = ""
    reported_symptoms: List[str] = field(default_factory=list)
    customer_observations: List[str] = field(default_factory=list)
    suggested_checks: List[str] = field(default_factory=list)
    completed_checks: List[str] = field(default_factory=list)
    pending_question: Optional[str] = None
    # The current message is a short reply to ``pending_question`` ("Only at
    # startup." after "When does the noise occur?"). It is resolved against that
    # question and the active issue, never against the first complaint.
    answers_pending_question: bool = False
    # The latest substantive issue the customer raised (a new symptom replaces an
    # earlier one); replies and clarifications refine it rather than replace it.
    active_issue: str = ""
    # Manual section labels of the claims last shown to the customer
    # ("noise or vibration"): the subject a short reply refers to.
    active_topics: List[str] = field(default_factory=list)
    negative_observations: List[str] = field(default_factory=list)
    # (question, answer) pairs: what the customer told us in reply to our questions.
    resolved_answers: List[Dict[str, str]] = field(default_factory=list)
    # Earlier issues replaced by a newer problem or a correction (oldest first).
    issue_history: List[str] = field(default_factory=list)
    # Problems the customer withdrew ("the water is from a window, not the AC").
    retracted_issues: List[str] = field(default_factory=list)
    # The problem withdrawn by the current message, if any.
    just_retracted: str = ""
    turn_count: int = 0
    is_followup: bool = False
    last_assistant_answer: str = ""
    last_contract: Dict[str, Any] = field(default_factory=dict)
    # The most recent assistant answer that was citation-backed guidance. A later
    # greeting, refusal or clarification does not erase what the customer was shown.
    last_supported_contract: Dict[str, Any] = field(default_factory=dict)
    # The most recent supported answer that listed two or more causes: "explain
    # each" and "the third cause" refer to that list even after a single-cause reply.
    last_cause_list_contract: Dict[str, Any] = field(default_factory=dict)
    history_turns: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def symptoms(self) -> List[str]:
        return self.reported_symptoms

    def format_summary(self) -> str:
        parts = []
        if self.original_complaint:
            parts.append(f"Initial complaint: {self.original_complaint}")
        if self.reported_symptoms:
            parts.append(f"Symptoms: {', '.join(self.reported_symptoms)}")
        if self.active_issue and self.active_issue != self.original_complaint:
            parts.append(f"Current issue: {self.active_issue}")
        if self.customer_observations:
            parts.append(f"Customer observations: {', '.join(self.customer_observations)}")
        if self.negative_observations:
            parts.append(f"Reported absent: {', '.join(self.negative_observations)}")
        for pair in self.resolved_answers[-3:]:
            parts.append(f"Asked: {pair['question']} Answer: {pair['answer']}")
        if self.completed_checks:
            parts.append(f"Completed checks: {', '.join(self.completed_checks)}")
        return "; ".join(parts)


def _clean_phrase(text: str) -> str:
    cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", (text or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _indicator_hits(text: str):
    """(indicator, negated) for each symptom indicator named in the text."""
    lower = (text or "").lower()
    hits = []
    for indicator in SYMPTOM_INDICATORS:
        for match in re.finditer(r"\b" + re.escape(indicator) + r"\b", lower):
            # Only the words of the same clause before the indicator decide negation.
            clause = re.split(r"[,;.!?]|\bbut\b|\band\b", lower[:match.start()])[-1]
            hits.append((indicator, bool(_NEGATION_BEFORE.search(clause))))
    return hits


def extract_negated_observations(text: str) -> List[str]:
    """Symptoms the customer explicitly says are absent ("no water leak is visible")."""
    found = list(dict.fromkeys(ind for ind, negated in _indicator_hits(text) if negated))
    # "water leaking" also matches "leaking": keep the most specific phrase only.
    return [ind for ind in found if not any(ind != other and ind in other for other in found)]


# Symptom concepts shared with evidence matching (term_matching.SYMPTOM_CONCEPTS),
# so "lost connection", "some rooms are warmer" or "isn't working" are recognised
# here exactly as they are when matching manual sections.
_CONCEPT_LABELS = {"poor_cooling": "poor cooling", "uneven_temperature": "uneven room temperature",
                   "water_leakage": "water leak", "communication": "communication fault",
                   "generic_failure": "not working", "noise_vibration": "noise", "weak_airflow": "weak airflow",
                   "low_capacity": "low capacity", "compressor_fault": "compressor fault", "fan_fault": "fan fault",
                   "high_pressure": "high pressure", "low_pressure": "low pressure", "no_start": "not starting"}
_EXTRA_SYMPTOMS = {
    "weak airflow": re.compile(r"\bair\s*flow\b[^.?!,;]{0,60}\b(?:weak|low|reduced|poor)\b"
                               r"|\b(?:weak|low|reduced|poor)\s+air\s*flow\b", re.I),
    "noise": re.compile(r"\b(?:rattl\w*|squeal\w*|clank\w*|buzz\w*)\b", re.I),
}


def extract_symptoms_from_text(text: str) -> List[str]:
    """Identify attributable, affirmatively reported symptom phrases from text."""
    if not text:
        return []
    found = []
    negated_words = [w for phrase in extract_negated_observations(text) for w in phrase.split()]
    for indicator, negated in _indicator_hits(text):
        if not negated and indicator not in found:
            found.append(indicator)
    from .term_matching import symptom_concepts
    for concept in sorted(symptom_concepts(text)):
        label = _CONCEPT_LABELS.get(concept)
        # "There's no water leaking" names the concept but negates it.
        # Negated when a negated phrase shares a word stem with the label
        # ("No water is leaking" negates "water leak").
        negated = any(word[:4] == other[:4] for word in label.split() if len(word) > 3 for other in negated_words)
        if label and label not in found and not negated:
            found.append(label)
    for label, pattern in _EXTRA_SYMPTOMS.items():
        match = pattern.search(text)
        if match and label not in found:
            clause = re.split(r"[,;.!?]|\bbut\b", text[:match.start()].lower())[-1]
            if not _NEGATION_BEFORE.search(clause):
                found.append(label)
    # Also extract error code patterns like E-04, F-07
    codes = re.findall(r"\b[A-Z]{1,3}[- ]?\d{1,3}\b", text)
    for code in codes:
        found.append(f"code {code}")
    return list(dict.fromkeys(found))


def extract_suggested_actions(contract: Dict[str, Any], answer_text: str = "") -> List[str]:
    """Extract recommended checks or observations from structured contract or answer."""
    actions = []
    if not contract and not answer_text:
        return actions

    # 1. From presentation sections
    presentation = contract.get("presentation") or {}
    if isinstance(presentation, dict):
        obs = presentation.get("customer_permitted_observations") or []
        for o in obs:
            if isinstance(o, dict) and o.get("source_text"):
                actions.append(o["source_text"])
            elif isinstance(o, dict) and o.get("text"):
                actions.append(o["text"])
        acts = presentation.get("customer_authorized_actions") or []
        for a in acts:
            if isinstance(a, dict) and a.get("source_text"):
                actions.append(a["source_text"])
            elif isinstance(a, dict) and a.get("text"):
                actions.append(a["text"])

    # 2. From standard diagnostic contract fields
    if not actions:
        safe_actions = contract.get("customer_safe_actions") or []
        for a in safe_actions:
            if isinstance(a, dict):
                act_str = a.get("action") or a.get("action_text") or a.get("text")
                if act_str:
                    actions.append(act_str)
            elif isinstance(a, str):
                actions.append(a)
        check_first = contract.get("check_first") or []
        for c in check_first:
            if isinstance(c, str):
                actions.append(c)

    # 3. Fallback to answer text parsing if structured contract lacked actions
    if not actions and answer_text:
        # Match lines under CHECK FIRST or STEP-BY-STEP
        cf_match = re.search(r"CHECK FIRST\s*\n(.*?)(?=\n\n|\n[A-Z]|$)", answer_text, re.S)
        if cf_match:
            for line in cf_match.group(1).splitlines():
                clean = re.sub(r"^[-*•\d\.\)]\s*", "", line).strip()
                if clean:
                    actions.append(clean)

    # Clean action strings (strip leading bullets, numbers, "check", etc.)
    cleaned = []
    for a in actions:
        c = re.sub(r"^[-*•\d\.\)]\s*", "", str(a)).strip()
        if c and len(c) > 5 and c not in cleaned:
            cleaned.append(c)
    return cleaned


def _topic_label(heading: str) -> str:
    """'6. SYMPTOM: NOISE OR VIBRATION' -> 'noise or vibration'."""
    label = re.sub(r"^\s*\d+[.)]\s*", "", str(heading or ""))
    label = re.sub(r"^(?:symptom|fault)\s*:\s*", "", label, flags=re.I)
    return " ".join(label.split()).lower()


def asked_question(contract: Dict[str, Any]) -> Optional[str]:
    """The question an assistant reply put to the customer, if it asked one.

    Sources, in order: an explicit clarification question, the composed reply's
    closing question, and a manual observation of the "note whether ..." kind that
    the reply showed (the customer may answer it directly: "Only at startup.").
    """
    contract = contract or {}
    presentation = contract.get("presentation") or {}
    question = contract.get("clarification_question") or presentation.get("follow_up_question")
    if not question:
        segments = (presentation.get("response") or {}).get("segments") or []
        asked = [s.get("text") for s in segments
                 if isinstance(s, dict) and s.get("kind") == "question" and s.get("text")]
        observed = [s.get("text") for s in segments
                    if isinstance(s, dict) and s.get("kind") == "list_item"
                    and re.search(r"\b(?:whether|when|where|if)\b", str(s.get("text") or ""), re.I)
                    and re.match(r"\s*(?:record|note|observe|report|check)\b", str(s.get("text") or ""), re.I)]
        question = " ".join(asked + observed[:2]) or None
    if not question and contract.get("response_type") == "CLARIFICATION":
        question = contract.get("summary")
    return " ".join(str(question).split()) if question else None


def shown_topics(contract: Dict[str, Any]) -> List[str]:
    """Manual section labels of the claims a reply showed the customer."""
    presentation = (contract or {}).get("presentation") or {}
    displayed = (presentation.get("response") or {}).get("displayed_claims")
    claims = displayed if isinstance(displayed, list) else presentation.get("supporting_claims") or []
    labels = []
    for claim in claims:
        label = _topic_label(claim.get("source_heading") if isinstance(claim, dict) else "")
        if label and label not in labels:
            labels.append(label)
    return labels


_FAULT_MENTION = re.compile(r"\b(?:fault|error|alarm|code|trip(?:s|ping)?|alert)\b", re.I)
_CODE = re.compile(r"\b[A-Z]{1,3}[- ]?\d{1,3}\b")
# "Actually, I may have read that code incorrectly": doubt about an earlier
# observation, not a new problem.
_UNCERTAIN_OBSERVATION = re.compile(
    r"\b(?:may|might|could)\s+have\s+(?:read|seen|heard|misread)\b|\bmisread\b"
    r"|\bread\s+(?:it|that|the\s+code|the\s+display)\s+(?:wrong|incorrectly)\b"
    r"|\bnot\s+sure\s+(?:about|of)\s+(?:the|that)\s+(?:code|reading|number)\b", re.I)


# Chat shorthand that hides a question or an intent from the classifiers
# ("whats likely causing it", "did u log a ticket", "what will the tech see").
# Used for classification only; the customer's words are stored unchanged.
_SHORT = r"(?<![\w'-]){}(?![\w'-])"
_CHAT_SHORTHAND = tuple((re.compile(_SHORT.format(word), re.I), full) for word, full in (
    ("u", "you"), ("ur", "your"), ("pls", "please"), ("plz", "please"),
    ("techs?", "technician"), ("engineers?", "technician"),
    ("whats", "what's"), ("hows", "how's"), ("wheres", "where's"), ("whos", "who's"), ("thats", "that's"),
    ("theres", "there's"), ("ive", "I've"), ("im", "I'm"), ("dont", "don't"), ("cant", "can't"),
    ("wont", "won't"), ("isnt", "isn't"), ("arent", "aren't"), ("wasnt", "wasn't"), ("hasnt", "hasn't"),
    ("havent", "haven't"), ("didnt", "didn't"), ("doesnt", "doesn't"),
    ("wat", "what"), ("wht", "what"), ("nxt", "next"), ("tiket|tikt|ticet|tkt", "ticket"), ("creat", "create"),
))
# "list the possible causes", "explain it simpler please", "create a call for me":
# a request even without a question mark; it never reports a symptom.
_REQUEST_START = re.compile(
    r"^\W*(?:please\s+|kindly\s+)?(?:list|tell|give|show|explain|describe|summari[sz]e|recap|create|make|"
    r"raise|book|log|open|register|lodge|arrange|send|suggest|recommend)\b", re.I)
# "so whats the main issue then", "ok and is it safe": a discourse word before the question.
_LEADING_FILLER = re.compile(
    r"^\W*(?:(?:so|ok(?:ay)?|and|but|well|then|right|alright|hmm+|um+|uh+|also|now|oh)\b[\s,.-]*)+", re.I)


def normalize_chat_text(text: str) -> str:
    value = " ".join(str(text or "").split())
    for pattern, full in _CHAT_SHORTHAND:
        value = pattern.sub(full, value)
    return value


def _is_question(value: str) -> bool:
    value = _LEADING_FILLER.sub("", normalize_chat_text(value))
    if value.rstrip().endswith("?"):
        return True
    if _REQUEST_START.match(value) or (re.search(r"\bplease\W*$", value, re.I)
                                       and not extract_symptoms_from_text(value)):
        return True
    # "can't get it to start" is a statement, although "can" opens questions.
    return bool(_QUESTION_START.match(value)) and not re.match(r"\s*can['\u2019]t\b", value, re.I)


def _code_symptom(symptom: str) -> bool:
    # Reported codes and generic "not working/running" wording ("the fan is not
    # running") are details of an existing issue, not a new symptom.
    return symptom.startswith("code ") or symptom == "not working"


def is_problem_report(text: str) -> bool:
    """A statement reporting a problem (not a question, not only negated symptoms)."""
    value = " ".join(str(text or "").split())
    return bool(value) and not _is_question(value) and bool(extract_symptoms_from_text(value))


def is_observation_statement(text: str, ctx: "ConversationContext") -> bool:
    """A short statement adding detail to the current issue rather than a new one.

    "The noise stops after a few seconds.", "There's no error code showing.",
    "The airflow seems normal, though.", "The outdoor module now shows E-99."
    """
    value = " ".join(str(text or "").split())
    if (not value or _is_question(value) or len(value.split()) > 25
            or not (ctx.active_issue or ctx.active_topics)):
        return False
    if is_explain_all_request(value) or _ORDINAL_REFERENCE.search(value) or _BACK_TO_LAST_CAUSE.search(value):
        return False
    if _UNCERTAIN_OBSERVATION.search(value):
        return True
    shifted = topic_shift_query(value)
    if shifted and _names_problem(shifted):
        return False
    known = f"{ctx.active_issue} {' '.join(ctx.active_topics)}".lower()
    new = [s for s in extract_symptoms_from_text(value)
           if not _code_symptom(s) and s not in known and s.split()[0] not in known]
    return not new


def is_reply_to_question(text: str, pending: Optional[str], active_issue: str = "") -> bool:
    """A short statement answering the question the assistant just asked.

    "Only at startup.", "Yes, the fan is running.", "No error code." are replies;
    a question, an explicit topic change, a request to explain listed causes or a
    newly reported symptom is not, so a self-contained new request can still
    change the subject.
    """
    value = " ".join(str(text or "").split())
    if not pending or not value or is_greeting(value) or len(value.split()) > 12:
        return False
    if _is_question(value):
        return False
    shifted = topic_shift_query(value)
    if is_explain_all_request(value) or (shifted and _names_problem(shifted)) or _ORDINAL_REFERENCE.search(value):
        return False
    known = f"{pending} {active_issue}".lower()
    new_symptoms = [s for s in extract_symptoms_from_text(value)
                    if not _code_symptom(s) and s not in known and s.split()[0] not in known]
    return not new_symptoms


def generic_symptom(symptom: str) -> bool:
    """"not working", "stopped working", "broken": a failure without a named symptom."""
    from .term_matching import symptom_concepts
    return symptom == "not working" or symptom_concepts(symptom) == {"generic_failure"}


def _specific_symptoms(text: str) -> List[str]:
    """Named symptoms only: not generic failures and not reported codes."""
    return [s for s in extract_symptoms_from_text(text or "") if not _code_symptom(s) and not generic_symptom(s)]


def _names_problem(text: str) -> bool:
    """True when the text names a problem of its own (a symptom or a fault)."""
    return bool([s for s in extract_symptoms_from_text(text) if not _code_symptom(s)]
                or _FAULT_MENTION.search(text))


# "the water is from a window, not the AC": the customer withdraws a reported
# problem as not coming from the equipment.
_RETRACTION = re.compile(
    r"\b(?:not|isn'?t|is\s+not|wasn'?t|was\s+not)\s+(?:coming\s+)?(?:from|caused\s+by)\s+(?:the|my|this|our)\s+"
    r"(?:ac|a/c|aircon|air\s*con|unit|system|equipment|module|machine)\b"
    r"|\bnot\s+(?:the|my|this|our)\s+(?:ac|a/c|aircon|air\s*con|unit|system|equipment|module|machine)\b", re.I)


def _track_customer_turn(ctx: "ConversationContext", text: str) -> None:
    """Update the active issue, negative observations and answered questions."""
    value = " ".join(str(text or "").split())
    ctx.just_retracted = ""
    if not value or is_greeting(value):
        ctx.answers_pending_question = False
        return
    if (is_summary_request(value) or conversation_intent(value)) and not extract_symptoms_from_text(value):
        # "sum up what I've said", "did u log a ticket?": requests about the
        # conversation or the service request are not details of the problem.
        ctx.answers_pending_question = False
        return
    if ctx.active_issue and _RETRACTION.search(value) and not _is_question(value):
        # The latest problem is withdrawn; the one before it is current again.
        ctx.answers_pending_question = False
        ctx.retracted_issues.append(ctx.active_issue)
        ctx.just_retracted = ctx.active_issue
        ctx.active_issue = ctx.issue_history.pop() if ctx.issue_history else ""
        ctx.active_topics = []
        return
    for negated in extract_negated_observations(value):
        if negated not in ctx.negative_observations:
            ctx.negative_observations.append(negated)
    for symptom in extract_symptoms_from_text(value):
        if symptom in ctx.negative_observations:
            ctx.negative_observations.remove(symptom)
    # The first reported problem (e.g. the answer to "please describe the symptom")
    # becomes the current issue; it is not an acknowledgement-only reply.
    if not ctx.active_issue and is_problem_report(value):
        ctx.answers_pending_question = False
        _set_issue(ctx, value)
        return
    # "its not working" then (after "which of these is closest?") "poor cooling": the
    # specific symptom replaces the generic issue instead of being a mere detail.
    if (is_problem_report(value) and _specific_symptoms(value)
            and not _specific_symptoms(ctx.active_issue)):
        ctx.answers_pending_question = False
        shifted = topic_shift_query(value)
        _set_issue(ctx, shifted if shifted and _names_problem(shifted) else value)
        return
    reply = (is_reply_to_question(value, ctx.pending_question, ctx.active_issue)
             or is_observation_statement(value, ctx))
    ctx.answers_pending_question = reply
    if reply:
        ctx.resolved_answers.append({"question": ctx.pending_question or "", "answer": value})
        return
    shifted = topic_shift_query(value)
    if shifted and not _names_problem(shifted):
        # "correction: only one indoor unit is affected" refines the current issue.
        shifted = None
    if shifted and _is_question(value) and not (extract_symptoms_from_text(value) or _FAULT_MENTION.search(value)):
        # "Does that correction change the possible causes?" asks about the
        # conversation; only a question naming a problem changes the subject.
        shifted = None
    if shifted or is_problem_report(value):
        # A newly reported problem or a correction replaces the current issue (the
        # earlier one is kept in issue_history); questions never do.
        _set_issue(ctx, shifted or value)


# "ok forget that, now there's a clicking sound": the set-aside words are not
# part of the new problem.
_SET_ASIDE_PREFIX = re.compile(
    r"^\W*(?:(?:ok(?:ay)?|so|well|wait|actually|sorry|no|right|anyway|hmm+)\W+)*"
    r"(?:forget\s+(?:about\s+)?(?:that|it)|never\s*mind(?:\s+(?:that|it))?|scratch\s+that|ignore\s+that)"
    r"\W+(?:(?:now|but|also|and|instead)\W+)*", re.I)


def _set_issue(ctx: "ConversationContext", value: str) -> None:
    issue = (_LEADING_FILLER.sub("", _SET_ASIDE_PREFIX.sub("", value)).rstrip(" .!")
             or value.rstrip(" .!"))
    if ctx.active_issue and ctx.active_issue != issue:
        ctx.issue_history.append(ctx.active_issue)
    ctx.active_issue = issue
    ctx.active_topics = []


MAX_STATE_MESSAGES = 400
_ISSUE_DEPENDENT = re.compile(
    r"\b(?:causes?|causing|reasons?|possibilit(?:y|ies)|observe|check|look\s+for|do\s+next|next\s+steps?|"
    r"fix|wrong|problem|issue|symptoms?|match|explain|why|happening|record|note)\b", re.I)

_SUMMARY_REQUEST = re.compile(
    r"\b(?:summari[sz]e|summary|recap|sum\s+up)\b"
    r"|\bwhat\s+(?:have|did)\s+i\s+(?:told|tell|said|say|mention(?:ed)?)\b"
    r"|\beverything\s+(?:i(?:'ve|\s+have)?|we(?:'ve|\s+have)?)\s+(?:told|said|discussed|mentioned)\b", re.I)


def is_summary_request(text: str) -> bool:
    """A request to recap the conversation itself; it needs no manual evidence."""
    return bool(_SUMMARY_REQUEST.search(" ".join(str(text or "").split())))


def conversation_summary_text(ctx: "ConversationContext", ticket_ref: Optional[str] = None) -> str:
    """Summarize only what the customer said; adds no technical facts."""
    equipment = ctx.asset_name or "your equipment"
    lines = [f"Here's what you've told me about the {equipment} so far:"]
    original = (ctx.original_complaint or "").strip()
    if original and not is_greeting(original):
        lines.append(f'- You first reported: "{original}"')
    for earlier in ctx.issue_history:
        if earlier.strip() != original.rstrip(" .!") and earlier.strip():
            lines.append(f'- You also mentioned: "{earlier.strip()}"')
    if ctx.active_issue and ctx.active_issue.strip() != original.rstrip(" .!"):
        lines.append(f'- The current problem, as you last described it: "{ctx.active_issue.strip()}"')
    for withdrawn in ctx.retracted_issues:
        lines.append(f'- You said this isn\'t coming from the {equipment}: "{withdrawn}"')
    details = [pair["answer"].rstrip(" .") for pair in ctx.resolved_answers[-8:]]
    if details:
        lines.append("- Details you added: " + "; ".join(f'"{d}"' for d in details))
    if ctx.negative_observations:
        lines.append(f"- You said there is no: {', '.join(ctx.negative_observations)}")
    if ctx.completed_checks:
        lines.append(f"- You told me you already checked: {', '.join(ctx.completed_checks)}")
    if len(lines) == 1:
        lines = [f"You haven't described a problem with the {equipment} yet. What is it doing?"]
    lines.append(f"A service request is open ({ticket_ref})." if ticket_ref
                 else "No service request has been created.")
    return "\n".join(lines)


# Customers say ticket, call, service call, complaint, request or visit for the same thing.
_TICKET_WORDS = (r"(?:(?:service\s+|support\s+|repair\s+)?(?:ticket|call)|service\s+request|complain(?:t|ts)?|request|"
                 r"(?:technician\s+|service\s+|engineer\s+)?visit)")
_TICKET_STATUS = re.compile(
    r"\b(?:did|have|has)\s+(?:you|it|anyone)\s+(?:actually\s+|already\s+)?"
    r"(?:create|raise|open|log|file|submit|book|ma(?:k|d)e)\w*\s+(?:a|an|the|my)?\s*" + _TICKET_WORDS
    + r"|\b(?:status|progress|number|reference)\s+of\s+(?:my|the)\s+" + _TICKET_WORDS
    + r"|\b(?:what(?:'s|\s+is)|where(?:'s|\s+is))\s+(?:my|the)\s+" + _TICKET_WORDS
    + r"(?:\s+(?:number|status|reference))?\b"
    + r"|\bis\s+(?:there|my)\s+(?:a\s+|an\s+)?(?:open\s+)?" + _TICKET_WORDS + r"\b", re.I)
_TICKET_CONTENTS = re.compile(
    r"\bwhat\b[^?.!]{0,40}\b(?:in|included\s+in|on|sent\s+with|shared\s+in|added\s+to)\s+"
    r"(?:the|my|a)\s+" + _TICKET_WORDS
    + r"|\bwhat\b[^?.!]{0,30}\b(?:will|would|does)\s+the\s+(?:ticket|service\s+request|technician)\s+"
    r"(?:say|include|contain|see|get|receive|know|be\s+told)\b", re.I)
_TICKET_HELP = re.compile(
    r"\b(?:create|raise|open|log|file|submit|book|make|arrange|schedule|register|lodge|place|generate|initiate|"
    r"put\s+in)\s+(?:a|an|the|my|one)?\s*(?:new\s+)?" + _TICKET_WORDS
    + r"|\bhow\s+(?:do|can|should)\s+i\s+(?:create|raise|open|get|book|request|log)\s+(?:a|an)?\s*"
    r"(?:ticket|service\s+request|service\s+call|technician)\b"
    # A visit request: "can I get someone to come out", "send a technician".
    r"|\b(?:get|send|book|arrange|need)\s+(?:someone|somebody|an?\s+technician|a\s+person)\b"
    r"(?:\s+(?:to\s+)?(?:come|visit|look|check|out|over|round))?"
    r"|\b(?:can|could|will|would)\s+(?:someone|somebody|a\s+technician)\s+(?:come|visit|look)\b", re.I)
_CERTAINTY = re.compile(
    r"\b(?:definitely|for\s+sure|certainly|100\s*%|guaranteed)\b"
    r"|\bare\s+you\s+(?:sure|certain|confident)\b"
    r"|\bhow\s+(?:sure|certain|confident)\s+are\s+you\b"
    r"|\bis\s+(?:this|that|it)\s+(?:confirmed|the\s+(?:actual|real|exact|definite)\s+(?:cause|problem|fault|issue))\b"
    r"|\bwhat\b[^?.!]{0,30}\bconfirmed\b|\bconfirmed\s+(?:and|or|vs\.?|versus)\s+what\b"
    r"|\bonly\s+(?:a\s+)?possibilit(?:y|ies)\b"
    r"|\b(?:is|are|has|have)\s+(?:any|all|some|anything|any\s+part)\b[^?.!]{0,20}\bconfirmed\b",
    re.I)
_ASSESSMENT = re.compile(
    r"\bwhat\s+do\s+you\s+think\s+(?:the\s+|is\s+the\s+)?(?:main\s+|real\s+|actual\s+|likely\s+)?"
    r"(?:problem|issue|cause|fault)\b"
    r"|\bwhat(?:'s|\s+is)\s+(?:the\s+)?(?:main|most\s+likely|likely|real)\s+(?:problem|issue|cause|fault)\b"
    r"|\bwhich\s+(?:one|cause)\s+is\s+(?:it|most\s+likely)\b", re.I)
_RELATION = re.compile(
    r"\b(?:related|connected|linked)\s+(?:to|with)\b|\bthe\s+same\s+(?:as|problem|issue|thing|fault)\b"
    r"|\bcaused\s+by\s+the\s+same\b", re.I)


def conversation_intent(text: str) -> Optional[str]:
    """Requests about the conversation or the service request, answered without manuals.

    Returns "ticket_status", "ticket_contents", "ticket_help", "certainty",
    "assessment", "relation" or None.
    Status is checked before help, so "Did you create a ticket?" is not a request to
    create one. A message that also reports a symptom is left to normal handling.
    """
    value = normalize_chat_text(text)
    if not value or len(value) > 300:
        return None
    if _TICKET_STATUS.search(value):
        return "ticket_status"
    if _TICKET_CONTENTS.search(value):
        return "ticket_contents"
    if _TICKET_HELP.search(value) and not extract_symptoms_from_text(value):
        return "ticket_help"
    is_question = _is_question(value)
    if _CERTAINTY.search(value) and is_question:
        return "certainty"
    if _ASSESSMENT.search(value) and is_question:
        return "assessment"
    # "Is this related to the problem I mentioned earlier?": about the customer's
    # own problems (a question naming a new fault type is answered from evidence).
    # "Is the clicking related to the cooling issue?" names the other problem.
    if (_RELATION.search(value) and is_question
            and re.search(r"\b(?:earlier|before|first|previous|other)\s+(?:problem|issue)\b"
                          r"|\b(?:problem|issue)\s+i\s+(?:mentioned|described|reported)\b"
                          r"|\b(?:the|my|that)\s+(?:[a-z]+\s+){1,2}(?:problem|issue)\b", value, re.I)):
        return "relation"
    return None


def retraction_reply(ctx: "ConversationContext") -> str:
    """Acknowledge a withdrawn problem and say which problem is current again."""
    equipment = ctx.asset_name or "the equipment"
    text = (f'Thanks for clarifying. I\'ve set aside "{ctx.just_retracted}", since you\'ve said it isn\'t '
            f"coming from the {equipment}.")
    if ctx.active_issue:
        text += f' We\'re back to the problem you reported before: "{ctx.active_issue}".'
    return text


def conversation_intent_reply(intent: str, ctx: "ConversationContext", ticket: Optional[Dict[str, str]] = None) -> str:
    """Truthful replies built from saved conversation state and the linked ticket only."""
    equipment = ctx.asset_name or "your equipment"
    issue = (ctx.active_issue or ctx.original_complaint or "").strip()
    if intent == "ticket_status":
        if ticket:
            return (f"Yes. Service request {ticket['ref']} is open for this conversation "
                    f"(status: {ticket['status']}).")
        return ("No, I haven't created a service request for this conversation. "
                "One is created only when you confirm it, and I can prepare it whenever you're ready.")
    if intent == "ticket_help":
        if ticket:
            return (f"A service request is already open for this conversation ({ticket['ref']}, "
                    f"status: {ticket['status']}), so I won't create another one.")
        return ("I can prepare a service request for a technician with what you've told me"
                + (f' ("{issue}")' if issue else "")
                + ". It's created only after you confirm it with the service request option.")
    if intent == "ticket_contents":
        items = [f"the {equipment} details"]
        if issue:
            items.append(f'the problem you described ("{issue}")')
        items += ["what you've told me in this chat", "the possibilities we discussed, marked as unconfirmed"]
        return ("The service request includes " + ", ".join(items[:-1]) + ", and " + items[-1]
                + ". It doesn't include a confirmed diagnosis, because none has been made.")
    listed = listed_causes(ctx.last_cause_list_contract or ctx.last_supported_contract or {})
    reported = [pair["answer"].rstrip(" .") for pair in ctx.resolved_answers[-4:]]
    if intent == "relation":
        earlier = next((i for i in reversed(ctx.issue_history) if i.strip()), ctx.original_complaint or "")
        if earlier and issue and earlier.strip() != issue:
            return (f'I can\'t confirm a link between "{issue}" and what you mentioned earlier ("{earlier}"). '
                    "Nothing in the information I have connects them, so a technician would need to check both.")
        return ("I can't confirm a link between them from the information I have. "
                "A technician can check whether they share a cause.")
    if intent == "assessment":
        parts = [f'From what you\'ve told me, the main problem is: "{issue}".' if issue
                 else "You haven't described the problem in detail yet."]
        if reported:
            parts.append("You've also told me: " + "; ".join(f'"{r}"' for r in reported) + ".")
        if listed:
            parts.append("I can't say which cause is responsible. The causes I listed are possibilities, "
                         "and only a technician's inspection can confirm one.")
        else:
            parts.append("I can't identify the cause from the information I have; a technician's inspection is needed.")
        return " ".join(parts)
    # certainty
    if reported:
        facts = "What you've reported yourself: " + "; ".join(f'"{r}"' for r in reported) + ". "
    else:
        facts = ""
    if listed:
        return ("No, nothing has been confirmed. " + facts + "The causes I mentioned are possibilities for this "
                f"kind of problem, not a diagnosis of your {equipment}. Only a technician's inspection can confirm "
                "which one, if any, applies.")
    return (f"No, nothing has been confirmed. {facts}I can't confirm a diagnosis of your {equipment} from a "
            "conversation; a technician's inspection is needed to establish the actual cause.")


def extract_conversation_context(
    conv: Any,
    current_input: str,
    asset: Optional[Any] = None,
    max_history_turns: int = 10,
    max_turns: Optional[int] = None,
) -> ConversationContext:
    """
    Load recent conversation turns and extract structured diagnostic state.
    
    Guarantees:
    - Bounded window of recent messages (last max_history_turns messages).
    - Preserves distinction between original reported complaint, symptoms,
      observations, assistant-suggested checks, and customer-completed checks.
    - Does not fabricate or speculate.
    """
    if max_turns is not None:
        max_history_turns = max_turns * 2

    clean_input = (current_input or "").strip()
    target_asset = asset or getattr(conv, "asset", None)

    ctx = ConversationContext(
        conversation_uuid=str(getattr(conv, "conversation_uuid", "") or ""),
        asset_id=getattr(target_asset, "id", None),
        asset_name=getattr(target_asset, "name", "") or "equipment",
        asset_model=getattr(target_asset, "model_number", "") or "",
        asset_code=getattr(target_asset, "asset_code", "") or "",
        product_name=(target_asset.product.name if target_asset and getattr(target_asset, "product", None) else ""),
    )

    if not conv:
        ctx.original_complaint = clean_input
        ctx.reported_symptoms = extract_symptoms_from_text(clean_input)
        return ctx

    # Limit the database query rather than loading an unbounded transcript.
    ctx.turn_count = conv.messages.filter(sender_role="customer").count()
    # The complaint is the first substantive customer message; a greeting is not one.
    early = [m for m in conv.messages.filter(sender_role="customer").order_by("created_at", "id")[:20]
             if not is_greeting(m.message_text)]
    first_customer_msg = next((m for m in early if is_problem_report(m.message_text)),
                              early[0] if early else None)
    if not first_customer_msg:
        ctx.original_complaint = "" if is_greeting(clean_input) else clean_input
        ctx.reported_symptoms = extract_symptoms_from_text(clean_input)
        return ctx
    ctx.original_complaint = first_customer_msg.message_text.strip()
    if not is_problem_report(ctx.original_complaint) and _is_question(ctx.original_complaint):
        # A question ("what is inspected during routine maintenance?") is not a
        # complaint; it must not become the issue or the "symptoms" of later turns.
        ctx.original_complaint = ""
    for sym in extract_symptoms_from_text(ctx.original_complaint):
        if sym not in ctx.reported_symptoms:
            ctx.reported_symptoms.append(sym)
    # Durable diagnostic state is rebuilt from the whole saved conversation (bounded),
    # so corrections, observations and answers survive long conversations; only the
    # wording context (history_turns) uses the short recent window.
    all_messages = list(conv.messages.order_by("-created_at", "-id")[:MAX_STATE_MESSAGES])
    all_messages.reverse()
    recent_messages = all_messages
    ctx.history_turns = [{"role": m.sender_role, "text": m.message_text}
                         for m in all_messages[-max(1, max_history_turns):]]

    # Track suggested actions by turn to associate customer confirmations
    last_assistant_suggested_actions: List[str] = []

    # The current issue is established by the first problem report, in order.
    ctx.active_issue = ""
    for msg in recent_messages:
        text = msg.message_text.strip()
        if msg.sender_role == "customer":
            _track_customer_turn(ctx, text)
            # Extract symptoms mentioned in customer message
            for sym in extract_symptoms_from_text(text):
                if sym not in ctx.reported_symptoms:
                    ctx.reported_symptoms.append(sym)

            # Check if customer stated an observation
            if any(k in text.lower() for k in ("weak", "slow", "loud", "noise", "hot", "cold", "leak", "stopped", "blinking")):
                clean_obs = _clean_phrase(text)
                if clean_obs and clean_obs not in ctx.customer_observations:
                    ctx.customer_observations.append(clean_obs)

            for completed in _confirmed_checks(text, last_assistant_suggested_actions):
                if completed not in ctx.completed_checks:
                    ctx.completed_checks.append(completed)

        elif msg.sender_role == "assistant":
            ctx.last_assistant_answer = text
            contract = msg.structured_contract or {}
            ctx.last_contract = contract
            if is_supported_answer(contract):
                ctx.last_supported_contract = contract
                if len(listed_causes(contract)) >= 2:
                    ctx.last_cause_list_contract = contract

            # Extract suggested checks from assistant's structured contract
            suggested = extract_suggested_actions(contract, text)
            if suggested:
                last_assistant_suggested_actions = suggested
                for s in suggested:
                    if s not in ctx.suggested_checks:
                        ctx.suggested_checks.append(s)

            # The question this reply asked the customer, if any. Only the most
            # recent assistant message counts: an answered or ignored question is
            # not pending forever (that would force every later turn into
            # follow-up mode, including independent questions).
            if contract.get("response_type") != "GREETING":
                ctx.pending_question = asked_question(contract)
                topics = shown_topics(contract)
                if topics:
                    ctx.active_topics = topics

    # Now evaluate current_input in relation to conversation state
    current_lower = clean_input.lower()

    # This message has not been persisted yet; classify it explicitly.
    for completed in _confirmed_checks(clean_input, last_assistant_suggested_actions):
        if completed not in ctx.completed_checks:
            ctx.completed_checks.append(completed)

    # Did current_input add new symptoms?
    for sym in extract_symptoms_from_text(clean_input):
        if sym not in ctx.reported_symptoms:
            ctx.reported_symptoms.append(sym)
    _track_customer_turn(ctx, clean_input)

    # Determine if current_input is a follow-up query
    is_followup = False
    if ctx.turn_count > 0:
        # If it's a short question or contains follow-up indicators
        has_next_pattern = any(p.search(clean_input) for p in NEXT_STEP_PATTERNS)
        has_pronoun = bool(re.search(r"\b(?:it|that|this|they|them|its|already|now|still)\b", current_lower))
        is_short = len(clean_input.split()) <= 12
        lacks_explicit_equipment = target_asset and target_asset.name.lower() not in current_lower
        is_clarif_reply = ctx.answers_pending_question

        # A complete new question ("How many years will the blower motor
        # last?") is not a follow-up merely because it is short or omits the
        # asset name: merging the earlier symptoms into it would answer the
        # previous topic instead of this question.
        if is_clarif_reply:
            is_followup = True
        elif (ctx.active_issue and _ISSUE_DEPENDENT.search(clean_input)
              and not extract_symptoms_from_text(clean_input) and not names_own_subject(clean_input)):
            # "What could be causing this?", "What can I safely observe?": no subject
            # of their own, so they are about the current issue.
            is_followup = True
        elif is_self_contained_question(clean_input) and not has_next_pattern:
            is_followup = False
        elif has_next_pattern or (is_short and (has_pronoun or lacks_explicit_equipment)) or is_clarif_reply:
            is_followup = True

    if not ctx.last_supported_contract or not ctx.last_cause_list_contract:
        # The supported answer or cause list may be older than the recent-turn window.
        for older in conv.messages.filter(sender_role="assistant").order_by("-created_at", "-id")[:30]:
            contract = older.structured_contract or {}
            if not is_supported_answer(contract):
                continue
            ctx.last_supported_contract = ctx.last_supported_contract or contract
            if not ctx.last_cause_list_contract and len(listed_causes(contract)) >= 2:
                ctx.last_cause_list_contract = contract
            if ctx.last_cause_list_contract:
                break

    ctx.is_followup = is_followup
    return ctx


_TOPIC_SHIFT = re.compile(
    r"\b(?:i meant|i mean|sorry|actually|correction|instead|rather than|not that one|"
    r"different (?:issue|problem|fault|symptom)|clarify|explained it (?:incorrectly|wrong)|"
    r"said it wrong|got it wrong|i was wrong|let me correct)\b", re.I)
# A sentence that only announces a correction ("Wait, I think I explained it
# incorrectly.", "I should clarify something:") carries no subject of its own.
_CORRECTION_PREFACE = re.compile(
    r"^\W*(?:(?:wait|sorry|ok(?:ay)?|hmm|so|well)\W+)*(?:i\s+think\s+)?(?:i\s+)?"
    r"(?:explained|said|described|told\s+you|put)\s+(?:it|that|this)\s+(?:incorrectly|wrong(?:ly)?|badly)\W*$"
    r"|^\W*(?:i\s+(?:should|need\s+to|want\s+to)|let\s+me)\s+(?:clarify|correct)"
    r"(?:\s+(?:something|that|myself|this|one\s+thing))?\W*$"
    r"|^\W*(?:correction|to\s+clarify)\W*$", re.I)
_SHIFT_PREFIX = re.compile(r"^\s*(?:(?:sorry|actually|correction|no)\s*[,:-]?\s*)*(?:i\s+meant?\s*(?:that\s*)?)?", re.I)
_NEGATED_TAIL = re.compile(
    r"[,;]?\s*(?:and\s+)?not\s+(?:the\s+|a\s+|an\s+)?[a-z/-]+(?:\s+[a-z/-]+)?\s*[.!?]?\s*$", re.I)
_INSTEAD = re.compile(r"\s*\b(?:instead|rather)\b\s*", re.I)


def topic_shift_query(text: str) -> Optional[str]:
    """For a correction or explicit topic change, the new subject alone; otherwise None.

    "Sorry, I meant ice is forming on the coil, not water." -> "ice is forming on the coil"
    "What about a communication fault instead?"             -> "What about a communication fault?"
    The customer's own words are not altered elsewhere; this is only the retrieval
    and claim-selection wording, and it never adds terms.
    """
    value = " ".join(str(text or "").split())
    if not _TOPIC_SHIFT.search(value):
        return None
    sentences = [x for x in re.split(r"(?<=[.!?:])\s+", value) if x.strip()]
    if len(sentences) > 1:
        value = " ".join(x for x in sentences if not _CORRECTION_PREFACE.match(x)) or value
    cleaned = _NEGATED_TAIL.sub("", value)
    cleaned = _SHIFT_PREFIX.sub("", cleaned)
    cleaned = _INSTEAD.sub(" ", cleaned).strip(" ,;")
    cleaned = re.sub(r"\s+([?.!])", r"\1", cleaned)
    words = [w for w in re.findall(r"[a-z]{3,}", cleaned.lower()) if w not in _CONTENT_STOPWORDS]
    return cleaned if len(words) >= 2 else None


_MAINTENANCE_TIMING = re.compile(
    r"\bhow\s+(?:often|frequently)\b"
    r"|\b(?:service|maintenance|servicing|inspection)\s+(?:intervals?|schedules?|frequency)\b"
    r"|\bwhen\s+(?:is|should|does|do)\b[^?.!]{0,30}\b(?:serviced|service|maintenance|maintained)\b", re.I)


def contextualize_customer_query(
    current_input: str,
    context: ConversationContext,
    asset: Optional[Any] = None,
) -> str:
    """
    Rewrite a customer's follow-up message into a focused, domain-rich retrieval query.

    Rules:
    - Never invent completed checks, fault codes, causes, or measurements.
    - If Turn 1 or fully self-contained question, preserve current_input.
    - If a follow-up ("What next?", "Airflow is weak", "I checked that already"):
      Synthesize asset context + original complaint + verified symptoms +
      completed checks indicator + current intent.
    - Returns a concise, retrieval-optimized query string.
    """
    clean_input = (current_input or "").strip()
    if not clean_input:
        return ""

    # A mixed message ("filter is clean, outdoor unit is near a wall, could
    # blocked clearance explain poor cooling? what can I observe?") is searched
    # by its substantive report + request, not by incidental negative clauses.
    # The unmodified message is still used for intent detection below.
    from .query_decomposition import retrieval_focus
    query_input = retrieval_focus(clean_input) or clean_input

    if not context.is_followup or context.turn_count == 0:
        return query_input

    # A correction or explicit change of topic replaces the earlier subject.
    shifted = topic_shift_query(clean_input)
    if shifted:
        return shifted[:450]

    # "How often does it need a service?" asks about maintenance, not the current
    # fault; adding the fault would pull retrieval toward that fault's section.
    if _MAINTENANCE_TIMING.search(clean_input) and not [
            s for s in extract_symptoms_from_text(clean_input) if not _code_symptom(s)]:
        return query_input

    target_asset = asset or (context.asset_name if context else None)
    asset_str = context.asset_name or (target_asset.name if hasattr(target_asset, "name") else "equipment")
    model_str = context.asset_model or (target_asset.model_number if hasattr(target_asset, "model_number") else "")
    equipment_header = f"{asset_str} {model_str}".strip()

    issue = context.active_issue or context.original_complaint

    # A short reply to the question just asked is resolved against that question
    # and the current issue ("noise" + "only at startup"), not the first complaint
    # or symptoms the customer raised earlier and has since moved on from.
    if context.answers_pending_question:
        topics = context.active_topics if 0 < len(context.active_topics) <= 2 else []
        parts = [equipment_header, *topics]
        if issue and not is_greeting(issue):
            parts.append(issue)
        parts.append(clean_input)
        return "; ".join(p for p in parts if p)[:450]

    # Determine current user intent
    is_asking_next = any(p.search(clean_input) for p in NEXT_STEP_PATTERNS)

    # Gather known symptoms: the current issue's, not every symptom ever named,
    # and never one the customer has said is absent.
    issue_symptoms = extract_symptoms_from_text(issue)
    symptoms = [s for s in (issue_symptoms or context.reported_symptoms)
                if s not in context.negative_observations]
    for sym in extract_symptoms_from_text(clean_input):
        if sym not in symptoms:
            symptoms.append(sym)

    # No issue and no symptom yet: add nothing ("symptoms: troubleshooting" only
    # diluted retrieval for a self-contained question).
    symptoms_str = ", ".join(symptoms) if symptoms else issue

    # Gather completed checks
    completed_str = ""
    if context.completed_checks:
        clean_completed = [_clean_phrase(c) for c in context.completed_checks[:2]]
        completed_str = f"customer completed initial check of {', '.join(clean_completed)}"

    # Nothing from the conversation to add: retrieval is already scoped to this asset,
    # and its name alone pulls the documents' purpose/overview sections forward.
    if not issue and not symptoms_str and not completed_str:
        return query_input

    # Construct contextualized query
    parts = [equipment_header]

    if issue and issue.lower() not in symptoms_str.lower():
        parts.append(f"issue: {issue}")

    if symptoms_str:
        parts.append(f"symptoms: {symptoms_str}")

    if completed_str:
        parts.append(completed_str)

    # The customer's own words always lead, so truncation never removes the current
    # request; "what next?" adds the intent rather than replacing what was said.
    current = query_input
    if is_asking_next:
        current = f"{query_input}; next troubleshooting steps and technical causes"
    rewritten = "; ".join(p for p in [current] + parts if p)
    # Bound query length
    return rewritten[:450]



# ---------------------------------------------------------------------------
# Ordinal references to a previously listed cause ("Explain the second cause.")
# ---------------------------------------------------------------------------
_ORDINALS = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2,
             "fourth": 3, "4th": 3, "fifth": 4, "5th": 4, "sixth": 5, "6th": 5,
             "seventh": 6, "7th": 6, "eighth": 7, "8th": 7, "ninth": 8, "9th": 8, "tenth": 9, "10th": 9,
             "eleventh": 10, "11th": 10, "twelfth": 11, "12th": 11, "last": -1}
_CAUSE_WORDS = r"(?:cause|causes|reason|reasons|possibility|option|item|one|point)"
_ORDINAL_REFERENCE = re.compile(
    r"\b(?P<ordinal>" + "|".join(_ORDINALS) + r")\s+(?:(?:possible|documented|listed|probable)\s+)?"
    + _CAUSE_WORDS + r"\b"
    r"|\b(?:cause|reason|item|option)\s*(?:number\s*|#\s*|no\.?\s*)?(?P<number>[1-9]|1[0-9]|20)\b",
    re.I,
)


def listed_causes(contract: Dict[str, Any]) -> List[Dict[str, str]]:
    """The possible causes in the order the customer saw them in a previous answer."""
    presentation = (contract or {}).get("presentation") or {}
    citations = {c.get("id"): c for c in presentation.get("citations") or [] if isinstance(c, dict)}
    # The composed reply records exactly which claims it displayed, in order:
    # "the first cause" means the first one the customer saw, not the first
    # validated claim (the reply may have omitted or reordered some).
    displayed = (presentation.get("response") or {}).get("displayed_claims")
    if isinstance(displayed, list):
        ordered = list(displayed)
    else:
        ordered = []
        for section in presentation.get("sections") or []:
            ordered.extend(section.get("claims") or [])
        ordered.extend(presentation.get("supporting_claims") or [])
    causes, seen = [], set()
    for claim in ordered:
        if not isinstance(claim, dict) or claim.get("kind") != "possible_cause":
            continue
        source = " ".join(str(claim.get("source_text") or "").split())
        key = (source.casefold(), " ".join(str(claim.get("source_heading") or "").split()).casefold())
        if not source or key in seen:
            continue
        seen.add(key)
        evidence = citations.get((claim.get("evidence_ids") or [None])[0]) or {}
        causes.append({"source_text": source, "heading": str(claim.get("source_heading") or ""),
                       "document_id": evidence.get("document_id"), "chunk_id": evidence.get("chunk_id")})
    return causes


_CAUSE_REQUEST = re.compile(
    r"\b(?:causes?|causing|reasons?|possibilit(?:y|ies)|what\s+(?:could|might|may)\s+be\s+(?:wrong|happening)"
    r"|what(?:'s|\s+is)\s+wrong|match(?:es)?\s+(?:these|my|the|everything)"
    r"|what(?:'s|\s+is)\s+(?:likely\s+|probably\s+)?(?:going\s+on|happening)"
    r"|what\s+(?:could|might|may)\s+be\s+going\s+on)\b", re.I)


def is_issue_cause_request(text: str, context: "ConversationContext") -> bool:
    """A request for causes that names no symptom of its own: it is about the current issue.

    "What could be causing this?", "Which possible causes best match everything I've
    reported?", "Does that correction change the possible causes?"
    """
    value = normalize_chat_text(text)
    return bool(context and context.active_issue and _CAUSE_REQUEST.search(value)
                and not _ORDINAL_REFERENCE.search(value) and not names_own_subject(value)
                and not [s for s in extract_symptoms_from_text(value) if not _code_symptom(s)])


_REFERS_BACK = re.compile(r"\b(?:this|that|these|those|it|everything|what\s+i(?:'ve|\s+have)?\s+(?:told|said|reported))\b", re.I)


def names_own_subject(value: str) -> bool:
    """"Causes of a high-pressure fault on a VRF outdoor module" names its own subject."""
    # "this model" / "the unit" name the selected equipment, not an earlier topic;
    # "for the technician" names who the answer is for, not a subject.
    value = _ROLE_REFERENCE.sub(" ", value)
    if _REFERS_BACK.search(_EQUIPMENT_REFERENCE.sub(" ", value)):
        return False
    return bool(_FAULT_MENTION.search(value)
                or re.search(r"\b(?:of|for|with)\s+(?:an?|the|my)\s+[a-z]", value, re.I))


# "What can I safely observe?", "What should I do next?", "How do I know which one
# it is?", "What should I record before the technician arrives?": requests for
# customer-safe checks about the current issue.
_ROLE_REFERENCE = re.compile(
    r"\b(?:for|with|to|from)\s+(?:the|a|an|my|our)\s+(?:technician|tech|engineer|installer|service\s+(?:team|person))s?\b",
    re.I)
_CHECK_REQUEST = re.compile(
    r"\b(?:observe|look\s+(?:at|for)|inspect|record|note\s+down|write\s+down|keep\s+track)\b"
    r"|\bwhat\s+(?:should|can|do)\s+i\s+do(?:\s+next|\s+now)?\b|\bnext\s+steps?\b"
    # "what to do next?", "what now", "now what?", "how do I proceed", "what do you suggest"
    r"|\bwhat\s+(?:to|shall\s+i|should\s+we|do\s+we|can\s+we)\s+do\b|\bwhat\s+(?:now|next)\b|\bnow\s+what\b"
    r"|\bhow\s+(?:do|should|can)\s+i\s+proceed\b|\bwhat\s+(?:do|would|can)\s+you\s+(?:suggest|recommend|advise)\b"
    r"|\bwhat\s+should\s+be\s+done\b"
    r"|\bhow\s+(?:do|can|would)\s+i\s+(?:know|tell|find\s+out|check)\b"
    # "What can I safely check myself?", "is there anything I can check?"
    r"|\b(?:what|anything|something)\s+(?:(?:can|could|should)\s+i|i\s+(?:can|could|should))\s+(?:safely\s+)?check\b"
    r"|\bsafely\s+check\b|\bcheck\s+(?:it\s+)?myself\b", re.I)
_VAGUE_TROUBLE = re.compile(
    r"\b(?:acting\s+up|playing\s+up|(?:has|having|got)\s+(?:a\s+|some\s+)?(?:problems?|issues?|trouble)|"
    r"something(?:'s|\s+is)\s+wrong|not\s+(?:right|ok(?:ay)?|good)|faulty|broken|giving\s+trouble)\b", re.I)
_MORE_CAUSES = re.compile(r"\b(?:other|else|more|additional|any\s+further)\b", re.I)


def is_issue_check_request(text: str, context: "ConversationContext") -> bool:
    value = normalize_chat_text(text)
    return bool(context and context.active_issue and _CHECK_REQUEST.search(value)
                and _is_question(value)
                and not names_own_subject(value)
                and not [s for s in extract_symptoms_from_text(value) if not _code_symptom(s)])


ISSUE_CAUSES_PREFIX = "What are the possible causes of this:"
ISSUE_MORE_CAUSES_PREFIX = "What other possible causes are there for this:"
ISSUE_CHECKS_PREFIX = "What can I safely check or observe for this:"


def issue_subject(context: "ConversationContext") -> str:
    """What the conversation is about now: the customer's issue, or, when that names no
    specific symptom ("its not working"), the manual section they were just shown."""
    issue = context.active_issue or ""
    if not _specific_symptoms(issue) and context.active_topics:
        return " / ".join(context.active_topics[:2])
    return issue


def canonical_issue_question(text: str, context: "ConversationContext") -> Optional[str]:
    """The current-issue form of a request that names no subject of its own."""
    if is_issue_cause_request(text, context):
        prefix = ISSUE_MORE_CAUSES_PREFIX if _MORE_CAUSES.search(text) else ISSUE_CAUSES_PREFIX
        return f"{prefix} {context.active_issue}?"
    if is_issue_check_request(text, context):
        return f"{ISSUE_CHECKS_PREFIX} {issue_subject(context)}?"
    return None


def is_vague_problem(text: str, context: "ConversationContext") -> bool:
    """A problem statement that names no symptom ("its not working", "my AC is acting up")
    while no specific problem is known yet: clarify before retrieving anything."""
    from .customer_diagnostic import VAGUE_PATTERNS
    value = normalize_chat_text(text)
    if not value or _is_question(value) or re.search(r"\b[A-Z]{1,3}[- ]?\d{1,3}\b", value):
        return False
    if (_specific_symptoms(context.active_issue) or _specific_symptoms(value)
            or re.search(r"\b[A-Z]{1,3}[- ]?\d{1,3}\b", context.active_issue or "")):
        return False
    lowered = value.lower().strip(" .!")
    return bool(any(re.match(p, lowered) for p in VAGUE_PATTERNS)
                or any(generic_symptom(s) for s in extract_symptoms_from_text(value))
                or _VAGUE_TROUBLE.search(value))


def ordinal_beyond_list(text: str, context: "ConversationContext") -> Optional[List[Dict[str, str]]]:
    """The displayed causes when "the second one" points past the end of that list."""
    match = _ORDINAL_REFERENCE.search(str(text or ""))
    if not match or not context:
        return None
    causes = listed_causes(context.last_cause_list_contract or context.last_supported_contract
                           or context.last_contract)
    if not causes:
        return None
    index = (_ORDINALS[match.group("ordinal").casefold()] if match.group("ordinal")
             else int(match.group("number")) - 1)
    return causes if index >= len(causes) else None


# "Can you explain that in simpler words?", "Why might that cause poor cooling?":
# about the single cause just discussed.
_BACK_TO_LAST_CAUSE = re.compile(
    r"\b(?:simpler|simply|plain(?:er)?\s+(?:words|english|terms|language)|simple\s+terms|in\s+other\s+words"
    r"|easier\s+to\s+understand|rephrase)\b"
    r"|\bwhy\s+(?:might|would|could|does|did|can|is)\s+(?:that|this|it)\b", re.I)


def resolve_cause_reference(text: str, context: "ConversationContext") -> Optional[Dict[str, Any]]:
    """Resolve "the second cause" against the causes listed in the previous answer.

    Returns the selected cause and the other listed causes, or None when the
    message has no positional reference, the previous answer listed no causes,
    or the position does not exist (never guessed).
    """
    if not context or not context.is_followup:
        return None
    match = _ORDINAL_REFERENCE.search(str(text or ""))
    if not match:
        # "explain that more simply" / "why might that ..." after a single-cause reply.
        mode = _BACK_TO_LAST_CAUSE.search(str(text or ""))
        last = listed_causes(context.last_contract)
        if not last:
            # A one-cause answer is never a "cause list"; fall back to it as well.
            last = (listed_causes(context.last_cause_list_contract)
                    or listed_causes(context.last_supported_contract))
        if not mode or len(last) != 1:
            return None
        full = listed_causes(context.last_cause_list_contract or context.last_contract)
        keys = [(c["source_text"].casefold(), c["heading"].casefold()) for c in full]
        key = (last[0]["source_text"].casefold(), last[0]["heading"].casefold())
        position = keys.index(key) if key in keys else 0
        return {"selected": last[0], "position": position,
                "others": [c["source_text"] for c in full if c is not full[position]] if full else [],
                "mode": "simpler" if "why" not in mode.group(0).casefold() else "why"}
    causes = listed_causes(context.last_cause_list_contract or context.last_supported_contract
                           or context.last_contract)
    if not causes:
        return None
    index =(_ORDINALS[match.group("ordinal").casefold()] if match.group("ordinal")
             else int(match.group("number")) - 1)
    if index >= len(causes):
        return None
    selected = causes[index]
    return {"selected": selected, "position": index if index >= 0 else len(causes) - 1,
            "others": [c["source_text"] for c in causes if c is not selected]}


def focus_presentation_on_cause(contract: Dict[str, Any], reference: Dict[str, Any]) -> None:
    """Drop the *other* previously listed causes from a follow-up answer about one cause."""
    others = {" ".join(o.split()).casefold() for o in reference.get("others") or []}
    if not others:
        return
    presentation = contract.get("presentation") or {}

    def keep(claim):
        return not (isinstance(claim, dict) and claim.get("kind") == "possible_cause"
                    and " ".join(str(claim.get("source_text") or "").split()).casefold() in others)

    presentation["supporting_claims"] = [c for c in presentation.get("supporting_claims") or [] if keep(c)]
    for section in presentation.get("sections") or []:
        if isinstance(section, dict) and isinstance(section.get("claims"), list):
            section["claims"] = [c for c in section["claims"] if keep(c)]
    # The stored message text was assembled before focusing; rebuild it from the
    # focused sections so the saved answer never lists the other causes.
    sections = [s for s in presentation.get("sections") or [] if isinstance(s, dict)]
    if sections and contract.get("answer"):
        contract["answer"] = "\n\n".join(
            f"{s.get('number')}. {s.get('title')}\n" + " ".join(str(c.get("text") or "") for c in s.get("claims") or [])
            for s in sections)


# ---------------------------------------------------------------------------
# Conversation titles from the customer's first real question (no model call)
# ---------------------------------------------------------------------------
_DEFAULT_TITLE = re.compile(r"^(?:Inquiry for .*|Support Inquiry)$")
_TITLE_LEAD = re.compile(
    r"^(?:(?:hi|hello|hey|please|so|ok)[,!.\s]+)*"
    r"(?:(?:what|which)\s+(?:are|is|could\s+be|might\s+be|can\s+be)\s+(?:the\s+)?(?:possible\s+|likely\s+|common\s+)?"
    r"(?:causes?|reasons?)\s+(?:of\s+|for\s+|why\s+)?"
    r"|what\s+(?:can\s+|could\s+|might\s+)?causes?\s+"
    r"|what\s+(?:does|do|is|are)\s+"
    r"|why\s+(?:is|are|does|do|did|would|has|have)\s+"
    r"|how\s+(?:often\s+)?(?:do|should|can|could)\s+(?:i|we)\s+"
    r"|(?:can|could)\s+you\s+(?:tell\s+me\s+|explain\s+)?"
    r"|(?:tell\s+me\s+about|explain)\s+)", re.I)
_TITLE_FILLER = re.compile(r"\b(?:my|our|the|a|an|showing|show|shows|this)\b", re.I)
_TITLE_PHRASES = ((re.compile(r"\bleaking water\b", re.I), "water leakage"),
                  (re.compile(r"\bleaking\b", re.I), "leakage"))
_ACRONYMS = {"ac": "AC", "vrf": "VRF", "hvac": "HVAC", "pcb": "PCB", "led": "LED"}


def is_default_title(title: str) -> bool:
    return bool(_DEFAULT_TITLE.match(str(title or "").strip()))


def conversation_title(question: str, max_chars: int = 60) -> str:
    """Short title from the customer's own words: drop question openers and filler.

    Only words from the question are kept (plus "leaking" -> "leakage"); fault
    codes keep their case; long digit runs (phone-like numbers) and e-mail
    addresses are removed. Returns "" when nothing meaningful remains.
    """
    text = " ".join(str(question or "").split())
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\+?\d[\d\s-]{6,}\d", " ", text)
    text = re.sub(r"[?!.]+$", "", text).strip()
    text = _TITLE_LEAD.sub("", text)
    text = re.sub(r"\s+mean$", "", text, flags=re.I)
    for pattern, replacement in _TITLE_PHRASES:
        text = pattern.sub(replacement, text)
    text = _TITLE_FILLER.sub(" ", text)
    words = [_ACRONYMS.get(w.casefold(), w) for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9/-]*", text)]
    title = ""
    for word in words:
        if len(title) + len(word) + 1 > max_chars:
            break
        title = f"{title} {word}".strip()
    return title[:1].upper() + title[1:] if title else ""
