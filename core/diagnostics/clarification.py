"""
Clarification Question Safety Validator and Deterministic Helper for Servy Red-Team.

Hard Invariants:
  - Clarification questions must act ONLY as a constrained intake helper.
  - Allowed topics: symptom category, when issue started, error code, observable state,
    whether an approved prior step was already completed.
  - MUST NEVER suggest or contain procedural repairs or unsafe instructions
    ("replace", "open", "remove panel", "rewire", "adjust voltage", "calibrate",
     "disassemble", "reset firmware", "bypass").
  - Strictly validated against schema:
      {
          "question": str,
          "answer_type": "BOOLEAN" | "SINGLE_CHOICE" | "SHORT_TEXT" | "NUMBER" | "IDENTIFIER",
          "options": List[str]
      }
  - If safety validation fails: use deterministic fallback clarification.
"""

from typing import Any, Dict, List, Tuple
from core.diagnostics.schemas import ALLOWED_RESPONSE_SCHEMAS

DISALLOWED_CLARIFICATION_KEYWORDS = [
    "replace",
    "open",
    "remove panel",
    "rewire",
    "adjust voltage",
    "calibrate",
    "disassemble",
    "dismantle",
    "reset firmware",
    "bypass",
    "solder",
    "unscrew",
    "short circuit",
]

DEFAULT_FALLBACK_CLARIFICATION: Dict[str, Any] = {
    "question": "Can you specify any error code displayed on screen or describe when this symptom started?",
    "answer_type": "SINGLE_CHOICE",
    "options": [
        "Error code displayed on screen",
        "Issue started immediately following a cleaning cycle",
        "Issue began during standard sample run",
        "Intermittent sensor or reading fluctuations",
    ],
}


def validate_clarification_payload(payload: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate a candidate clarification question payload against strict schema and safety bounds."""
    if not isinstance(payload, dict):
        return False, "Payload must be a dictionary."

    question = payload.get("question")
    if not question or not isinstance(question, str) or not question.strip():
        return False, "Field 'question' must be a non-empty string."

    answer_type = payload.get("answer_type")
    if answer_type not in ALLOWED_RESPONSE_SCHEMAS:
        return False, f"Invalid 'answer_type' '{answer_type}'. Permitted: {sorted(ALLOWED_RESPONSE_SCHEMAS)}."

    options = payload.get("options")
    if options is not None:
        if not isinstance(options, list) or not all(isinstance(o, str) for o in options):
            return False, "Field 'options' must be a list of strings."

    # Procedural Keyword / Repair Suggestion Check
    question_lower = question.lower()
    for kw in DISALLOWED_CLARIFICATION_KEYWORDS:
        if kw in question_lower:
            return False, f"Clarification question contains prohibited procedural/repair instruction: '{kw}'."

    if options:
        for opt in options:
            opt_lower = opt.lower()
            for kw in DISALLOWED_CLARIFICATION_KEYWORDS:
                if kw in opt_lower:
                    return False, f"Clarification option contains prohibited procedural/repair instruction: '{kw}'."

    return True, "VALID"


def get_safe_clarification(candidate_payload: Any = None) -> Dict[str, Any]:
    """Return verified clarification payload or safe deterministic fallback."""
    if candidate_payload and isinstance(candidate_payload, dict):
        is_valid, _ = validate_clarification_payload(candidate_payload)
        if is_valid:
            return {
                "question": candidate_payload["question"].strip(),
                "answer_type": candidate_payload["answer_type"],
                "options": candidate_payload.get("options", []),
            }
    return dict(DEFAULT_FALLBACK_CLARIFICATION)
