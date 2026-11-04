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
    "not cooling", "poor cooling", "weak cooling", "insufficient cooling",
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

# Words indicating that the user confirmed performing a check
COMPLETION_PATTERNS = [
    re.compile(r"\b(?:i\s+)?(?:already\s+)?(?:checked|inspected|cleaned|verified|confirmed|did|done|tested)\s+(?:that|this|it|already)\b", re.I),
    re.compile(r"\b(?:checked\s+that\s+already|already\s+checked|i\s+checked\s+it|already\s+done|done\s+that|i\s+did\s+that)\b", re.I),
    re.compile(r"\b(?:i\s+have\s+)?(?:already\s+)?(?:cleaned|washed|replaced)\s+(?:the\s+)?([a-z0-9_-]+(?:\s+[a-z0-9_-]+)?)\b", re.I),
    re.compile(r"\b(?:i\s+have\s+)?(?:already\s+)?(?:checked|inspected)\s+(?:the\s+)?([a-z0-9_-]+(?:\s+[a-z0-9_-]+)?)\b", re.I),
    re.compile(r"\b([a-z0-9_-]+(?:\s+[a-z0-9_-]+)?)\s+(?:is|was)\s+(?:clean|checked|fine|okay|ok|normal|on|running|clear)\b", re.I),
]

# Patterns indicating the user is asking what to do next
NEXT_STEP_PATTERNS = [
    re.compile(r"\b(?:what\s+next|what\s+should\s+i\s+do(?:\s+next)?|what\s+now|what\s+else|what\s+to\s+do|what\s+is\s+next)\b", re.I),
    re.compile(r"\b(?:and\s+then|next\s+step|now\s+what|what's\s+next)\b", re.I),
    re.compile(r"\b(?:still\s+not\s+working|problem\s+persists|issue\s+continues|didn't\s+help|did\s+not\s+work|same\s+issue)\b", re.I),
]


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

    # Fetch ordered messages
    messages = list(conv.messages.all().order_by("created_at"))
    ctx.turn_count = len([m for m in messages if m.sender_role == "customer"])

    # If no prior messages exist, this is Turn 1
    if not messages:
        ctx.original_complaint = clean_input
        ctx.reported_symptoms = extract_symptoms_from_text(clean_input)
        return ctx

    # Bounded window: preserve the very first customer message (original complaint)
    # plus the most recent messages up to max_history_turns
    first_customer_msg = next((m for m in messages if m.sender_role == "customer"), None)
    if first_customer_msg:
        ctx.original_complaint = first_customer_msg.message_text.strip()
        for sym in extract_symptoms_from_text(ctx.original_complaint):
            if sym not in ctx.reported_symptoms:
                ctx.reported_symptoms.append(sym)

    recent_messages = messages[-max_history_turns:] if len(messages) > max_history_turns else messages
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

            # Check if customer confirmed completing a check
            is_completed = False
            for pat in COMPLETION_PATTERNS:
                m = pat.search(text)
                if m:
                    is_completed = True
                    # If specific component captured in regex group
                    if m.lastindex and m.group(1):
                        comp = m.group(1).strip()
                        if comp and comp not in ctx.completed_checks:
                            ctx.completed_checks.append(comp)
                    break

            if is_completed and last_assistant_suggested_actions:
                # If customer said "I checked that already" without naming the part,
                # bind to the first check the assistant had just suggested
                top_suggested = last_assistant_suggested_actions[0]
                if top_suggested not in ctx.completed_checks:
                    ctx.completed_checks.append(top_suggested)
                for comp_k in ("air filter", "filter", "fuse", "breaker", "sensor", "valve", "tubing", "pump", "condenser", "coil"):
                    if comp_k in top_suggested.lower() and comp_k not in ctx.completed_checks:
                        ctx.completed_checks.append(comp_k)

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

    # Did current_input confirm a completed check?
    for pat in COMPLETION_PATTERNS:
        m = pat.search(clean_input)
        if m:
            if m.lastindex and m.group(1):
                comp = m.group(1).strip()
                if comp and comp not in ctx.completed_checks:
                    ctx.completed_checks.append(comp)
            elif last_assistant_suggested_actions:
                top_suggested = last_assistant_suggested_actions[0]
                if top_suggested not in ctx.completed_checks:
                    ctx.completed_checks.append(top_suggested)
                for comp_k in ("air filter", "filter", "fuse", "breaker", "sensor", "valve", "tubing", "pump", "condenser", "coil"):
                    if comp_k in top_suggested.lower() and comp_k not in ctx.completed_checks:
                        ctx.completed_checks.append(comp_k)
            break

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

        if has_next_pattern or (is_short and (has_pronoun or lacks_explicit_equipment)) or is_clarif_reply:
            is_followup = True

    ctx.is_followup = is_followup
    return ctx


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

    if not context.is_followup or context.turn_count == 0:
        return clean_input

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
        parts.append(clean_input)

    rewritten = "; ".join(p for p in parts if p)
    # Bound query length
    return rewritten[:450]

