"""
Schemas, constants, and validation primitives for Servy Zero-Repeat Diagnostic Recovery.

Runtime Action Classes:
  - OBSERVE
  - SAFE_ACTION
  - VERIFY
  - ESCALATE
No other runtime action types are permitted.

Safety Classes:
  - GREEN: normal customer-safe operator action.
  - AMBER: allowed only when explicitly approved as customer/operator procedure.
  - RED: customer must never receive repair procedure; escalate immediately.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Core Constants
# ---------------------------------------------------------------------------

NODE_OBSERVE = "OBSERVE"
NODE_SAFE_ACTION = "SAFE_ACTION"
NODE_VERIFY = "VERIFY"
NODE_ESCALATE = "ESCALATE"

ALLOWED_NODE_TYPES = {NODE_OBSERVE, NODE_SAFE_ACTION, NODE_VERIFY, NODE_ESCALATE}

SAFETY_GREEN = "GREEN"
SAFETY_AMBER = "AMBER"
SAFETY_RED = "RED"

ALLOWED_SAFETY_CLASSES = {SAFETY_GREEN, SAFETY_AMBER, SAFETY_RED}

RESPONSE_BOOLEAN = "BOOLEAN"
RESPONSE_SINGLE_CHOICE = "SINGLE_CHOICE"
RESPONSE_NUMBER = "NUMBER"
RESPONSE_SHORT_TEXT = "SHORT_TEXT"
RESPONSE_IDENTIFIER = "IDENTIFIER"
RESPONSE_CONFIRMATION = "CONFIRMATION"

ALLOWED_RESPONSE_SCHEMAS = {
    RESPONSE_BOOLEAN,
    RESPONSE_SINGLE_CHOICE,
    RESPONSE_NUMBER,
    RESPONSE_SHORT_TEXT,
    RESPONSE_IDENTIFIER,
    RESPONSE_CONFIRMATION,
}

SOURCE_SYSTEM = "SYSTEM_FACT"
SOURCE_CUSTOMER_ASSERTED = "CUSTOMER_ASSERTED"
SOURCE_CUSTOMER_SELECTED = "CUSTOMER_SELECTED"
SOURCE_CUSTOMER_VERIFIED = "CUSTOMER_VERIFIED"
SOURCE_SYSTEM_DERIVED = "SYSTEM_DERIVED"
SOURCE_VERIFIED_OBSERVATION = "VERIFIED_OBSERVATION"
SOURCE_TECHNICIAN_RECORDED = "TECHNICIAN_RECORDED"
SOURCE_DERIVED = "DERIVED"
SOURCE_UNVERIFIED_CANDIDATE = "UNVERIFIED_CANDIDATE"

ALLOWED_FACT_SOURCES = {
    SOURCE_SYSTEM,
    SOURCE_CUSTOMER_ASSERTED,
    SOURCE_CUSTOMER_SELECTED,
    SOURCE_CUSTOMER_VERIFIED,
    SOURCE_SYSTEM_DERIVED,
    SOURCE_VERIFIED_OBSERVATION,
    SOURCE_TECHNICIAN_RECORDED,
    SOURCE_DERIVED,
    SOURCE_UNVERIFIED_CANDIDATE,
}

AUTHORITATIVE_FACT_SOURCES = {
    SOURCE_SYSTEM,
    SOURCE_CUSTOMER_ASSERTED,
    SOURCE_CUSTOMER_SELECTED,
    SOURCE_CUSTOMER_VERIFIED,
    SOURCE_SYSTEM_DERIVED,
    SOURCE_VERIFIED_OBSERVATION,
    SOURCE_TECHNICIAN_RECORDED,
    SOURCE_DERIVED,
}

# Event Types
EVENT_SESSION_STARTED = "SESSION_STARTED"
EVENT_COMPLAINT_RECORDED = "COMPLAINT_RECORDED"
EVENT_PLAYBOOK_SELECTED = "PLAYBOOK_SELECTED"
EVENT_OBSERVATION_REQUESTED = "OBSERVATION_REQUESTED"
EVENT_OBSERVATION_RECORDED = "OBSERVATION_RECORDED"
EVENT_SAFE_ACTION_PRESENTED = "SAFE_ACTION_PRESENTED"
EVENT_SAFE_ACTION_CONFIRMED = "SAFE_ACTION_CONFIRMED"
EVENT_VERIFICATION_REQUESTED = "VERIFICATION_REQUESTED"
EVENT_VERIFICATION_RECORDED = "VERIFICATION_RECORDED"
EVENT_CONTRADICTION_FOUND = "CONTRADICTION_FOUND"
EVENT_CONTRADICTION_RESOLVED = "CONTRADICTION_RESOLVED"
EVENT_CLARIFICATION_RECORDED = "CLARIFICATION_RECORDED"
EVENT_SESSION_RESOLVED = "SESSION_RESOLVED"
EVENT_SESSION_ESCALATED = "SESSION_ESCALATED"

PROGRESSION_EVENT_TYPES = {
    EVENT_OBSERVATION_RECORDED,
    EVENT_SAFE_ACTION_CONFIRMED,
    EVENT_VERIFICATION_RECORDED,
    EVENT_CONTRADICTION_RESOLVED,
    EVENT_CLARIFICATION_RECORDED,
}

MAX_SESSION_STEPS = 25
MAX_EXECUTION_DEPTH = 15
MAX_ACTIVE_SESSIONS_PER_CUSTOMER = 5


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticFact:
    key: str
    value: Any
    source: str
    verified: bool = False
    updated_at: Optional[str] = None
    node_id: Optional[str] = None


@dataclass
class ContradictionRecord:
    fact_key: str
    earlier_value: Any
    new_value: Any
    reason: str
    resolved: bool = False
    resolution_note: str = ""


@dataclass
class CompletedActionRecord:
    node_id: str
    instruction: str
    evidence_anchor: Dict[str, Any]
    confirmed_at: str
    safety_class: str


@dataclass
class DiagnosticState:
    session_id: str
    status: str
    current_node_id: str
    facts: Dict[str, DiagnosticFact] = field(default_factory=dict)
    contradictions: List[ContradictionRecord] = field(default_factory=list)
    completed_actions: List[CompletedActionRecord] = field(default_factory=list)
    evidence_completeness: float = 0.0
    escalation_reason: Optional[str] = None
    resolution_summary: Optional[str] = None
    step_count: int = 0
