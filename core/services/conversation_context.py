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
]

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
    if _NEGATED_COMPLETION.search(text):
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
    turn_count: int = 0
    is_followup: bool = False
    last_assistant_answer: str = ""
    last_contract: Dict[str, Any] = field(default_factory=dict)
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
        if self.customer_observations:
            parts.append(f"Customer observations: {', '.join(self.customer_observations)}")
        if self.completed_checks:
            parts.append(f"Completed checks: {', '.join(self.completed_checks)}")
        return "; ".join(parts)


def _clean_phrase(text: str) -> str:
    cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", (text or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def extract_symptoms_from_text(text: str) -> List[str]:
    """Identify attributable symptom phrases from text."""
    if not text:
        return []
    lower = text.lower()
    found = []
    for indicator in SYMPTOM_INDICATORS:
        if indicator in lower:
            found.append(indicator)
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
    first_customer_msg = conv.messages.filter(sender_role="customer").order_by("created_at", "id").first()
    if not first_customer_msg:
        ctx.original_complaint = clean_input
        ctx.reported_symptoms = extract_symptoms_from_text(clean_input)
        return ctx
    ctx.original_complaint = first_customer_msg.message_text.strip()
    for sym in extract_symptoms_from_text(ctx.original_complaint):
        if sym not in ctx.reported_symptoms:
            ctx.reported_symptoms.append(sym)
    recent_messages = list(conv.messages.order_by("-created_at", "-id")[:max(1, max_history_turns)])
    recent_messages.reverse()
    ctx.history_turns = [{"role": m.sender_role, "text": m.message_text} for m in recent_messages]

    # Track suggested actions by turn to associate customer confirmations
    last_assistant_suggested_actions: List[str] = []

    for msg in recent_messages:
        text = msg.message_text.strip()
        if msg.sender_role == "customer":
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

            # Extract suggested checks from assistant's structured contract
            suggested = extract_suggested_actions(contract, text)
            if suggested:
                last_assistant_suggested_actions = suggested
                for s in suggested:
                    if s not in ctx.suggested_checks:
                        ctx.suggested_checks.append(s)

            # Extract pending question if clarification was requested
            if contract.get("response_type") == "CLARIFICATION":
                ctx.pending_question = contract.get("clarification_question") or contract.get("summary")
            elif "clarification_question" in contract and contract["clarification_question"]:
                ctx.pending_question = contract["clarification_question"]

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

    # Determine if current_input is a follow-up query
    is_followup = False
    if ctx.turn_count > 0:
        # If it's a short question or contains follow-up indicators
        has_next_pattern = any(p.search(clean_input) for p in NEXT_STEP_PATTERNS)
        has_pronoun = bool(re.search(r"\b(?:it|that|this|they|them|its|already|now|still)\b", current_lower))
        is_short = len(clean_input.split()) <= 12
        lacks_explicit_equipment = target_asset and target_asset.name.lower() not in current_lower
        is_clarif_reply = bool(ctx.pending_question)

        # A complete new question ("How many years will the blower motor
        # last?") is not a follow-up merely because it is short or omits the
        # asset name: merging the earlier symptoms into it would answer the
        # previous topic instead of this question.
        if is_self_contained_question(clean_input) and not has_next_pattern:
            is_followup = False
        elif has_next_pattern or (is_short and (has_pronoun or lacks_explicit_equipment)) or is_clarif_reply:
            is_followup = True

    ctx.is_followup = is_followup
    return ctx


_TOPIC_SHIFT = re.compile(
    r"\b(?:i meant|i mean|sorry|actually|correction|instead|rather than|not that one|"
    r"different (?:issue|problem|fault|symptom))\b", re.I)
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
    cleaned = _NEGATED_TAIL.sub("", value)
    cleaned = _SHIFT_PREFIX.sub("", cleaned)
    cleaned = _INSTEAD.sub(" ", cleaned).strip(" ,;")
    cleaned = re.sub(r"\s+([?.!])", r"\1", cleaned)
    words = [w for w in re.findall(r"[a-z]{3,}", cleaned.lower()) if w not in _CONTENT_STOPWORDS]
    return cleaned if len(words) >= 2 else None


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

    target_asset = asset or (context.asset_name if context else None)
    asset_str = context.asset_name or (target_asset.name if hasattr(target_asset, "name") else "equipment")
    model_str = context.asset_model or (target_asset.model_number if hasattr(target_asset, "model_number") else "")
    equipment_header = f"{asset_str} {model_str}".strip()

    # Determine current user intent
    is_asking_next = any(p.search(clean_input) for p in NEXT_STEP_PATTERNS)

    # Gather known symptoms
    symptoms = list(context.reported_symptoms)
    for sym in extract_symptoms_from_text(clean_input):
        if sym not in symptoms:
            symptoms.append(sym)

    symptoms_str = ", ".join(symptoms) if symptoms else (context.original_complaint or "troubleshooting")

    # Gather completed checks
    completed_str = ""
    if context.completed_checks:
        clean_completed = [_clean_phrase(c) for c in context.completed_checks[:2]]
        completed_str = f"customer completed initial check of {', '.join(clean_completed)}"

    # Construct contextualized query
    parts = [equipment_header]

    if context.original_complaint and context.original_complaint.lower() not in symptoms_str.lower():
        parts.append(f"issue: {context.original_complaint}")

    parts.append(f"symptoms: {symptoms_str}")

    if completed_str:
        parts.append(completed_str)

    if is_asking_next:
        parts.append("next troubleshooting steps and technical causes")
    else:
        # Include user's current specific remark/question
        parts.append(query_input)

    rewritten = "; ".join(p for p in parts if p)
    # Bound query length
    return rewritten[:450]



# ---------------------------------------------------------------------------
# Ordinal references to a previously listed cause ("Explain the second cause.")
# ---------------------------------------------------------------------------
_ORDINALS = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2,
             "fourth": 3, "4th": 3, "fifth": 4, "5th": 4, "sixth": 5, "6th": 5, "last": -1}
_CAUSE_WORDS = r"(?:cause|causes|reason|reasons|possibility|option|item|one|point)"
_ORDINAL_REFERENCE = re.compile(
    r"\b(?P<ordinal>" + "|".join(_ORDINALS) + r")\s+(?:(?:possible|documented|listed|probable)\s+)?"
    + _CAUSE_WORDS + r"\b"
    r"|\b(?:cause|reason|item|option)\s*(?:number\s*|#\s*|no\.?\s*)?(?P<number>[1-6])\b",
    re.I,
)


def listed_causes(contract: Dict[str, Any]) -> List[Dict[str, str]]:
    """The possible causes in the order the customer saw them in a previous answer."""
    presentation = (contract or {}).get("presentation") or {}
    ordered = []
    for section in presentation.get("sections") or []:
        ordered.extend(section.get("claims") or [])
    ordered.extend(presentation.get("supporting_claims") or [])
    causes, seen = [], set()
    for claim in ordered:
        if not isinstance(claim, dict) or claim.get("kind") != "possible_cause":
            continue
        source = " ".join(str(claim.get("source_text") or "").split())
        if not source or source.casefold() in seen:
            continue
        seen.add(source.casefold())
        causes.append({"source_text": source, "heading": str(claim.get("source_heading") or "")})
    return causes


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
        return None
    causes = listed_causes(context.last_contract)
    if not causes:
        return None
    index = (_ORDINALS[match.group("ordinal").casefold()] if match.group("ordinal")
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
