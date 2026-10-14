"""
Customer Diagnostic Service for Servy RAG.

Provides input normalization, ambiguity/clarification detection, industrial safety boundaries,
structured diagnostic response contracts, and reconciliation with deterministic DiagnosticSessions.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from django.db import transaction
from django.utils import timezone

from core.models import (
    Asset,
    Customer,
    CustomerInteraction,
    DiagnosticEvent,
    DiagnosticSession,
    Site,
)
from core.diagnostics.schemas import (
    EVENT_CLARIFICATION_RECORDED,
    EVENT_COMPLAINT_RECORDED,
    EVENT_OBSERVATION_RECORDED,
    EVENT_OBSERVATION_REQUESTED,
    EVENT_SAFE_ACTION_CONFIRMED,
    EVENT_SAFE_ACTION_PRESENTED,
    EVENT_SESSION_ESCALATED,
    EVENT_SESSION_RESOLVED,
    EVENT_SESSION_STARTED,
)


# ---------------------------------------------------------------------------
# Ambiguity and Vague Query Heuristics
# ---------------------------------------------------------------------------

VAGUE_PATTERNS = [
    r"^(my\s+)?(machine|asset|equipment|device|analyzer|unit)\s+(is\s+)?(not\s+working|broken|down|faulty|dead|having\s+issues?|stopped)\.?$",
    r"^(it\s+)?(is\s+)?(not\s+working|broken|down|stopped|failed)\.?$",
    r"^(not\s+working|broken|machine\s+down|help|error|issue|problem)\.?$",
    r"^(my\s+)?(milk\s+analyzer|analyzer)\s+(is\s+)?(not\s+working|broken|down)\.?$",
    r"^(the\s+)?equipment\s+is\s+not\s+working\.?$",
]

HAZARD_PATTERNS = [
    r"\b(smoke|smoking|fumes)\b",
    r"\b(spark|sparks|sparking)\b",
    r"\b(fire|flames?|burning\s+smell|burnt|charred)\b",
    r"\b(electric\s+shock|shocked|electrocution)\b",
    r"\b(explosion|exploded)\b",
    r"\b(chemical\s+spill|acid\s+leak|hazardous\s+leak)\b",
]

TECHNICIAN_ONLY_KEYWORDS = [
    "energized",
    "live circuit",
    "live wire",
    "high voltage",
    "415v",
    "440v",
    "internal wiring",
    "open cabinet while running",
    "open electrical cabinet",
    "bypass safety",
    "defeat interlock",
    "disable safety",
    "bridge fuse",
    "refrigerant recovery",
    "refrigerant charging",
    "disassemble motor",
    "replace control board",
    "replace pcb",
    "tamper with calibration potentiometer",
    "pressurized vessel overhaul",
    "internal transformer",
]

OBSERVATION_KEYWORDS = [
    "check",
    "confirm",
    "ensure",
    "inspect",
    "verify",
    "examine",
    "look at",
    "observe",
    "read",
]


def check_for_safety_hazard(complaint: str) -> Tuple[bool, str]:
    """Identify immediate safety-critical symptoms that prohibit customer DIY troubleshooting."""
    text = (complaint or "").lower().strip()
    for pat in HAZARD_PATTERNS:
        if re.search(pat, text):
            return True, (
                "IMMEDIATE SAFETY RISK DETECTED: Symptoms indicate a hazardous electrical, thermal, or chemical condition. "
                "Disconnect main power immediately if it is safe to do so. Do not operate or open the equipment. "
                "An urgent technician service call should be dispatched."
            )
    return False, ""


def check_for_vague_complaint(complaint: str) -> Tuple[bool, str, List[str]]:
    """Determine if a customer complaint is too underspecified to retrieve accurate technical procedures."""
    text = (complaint or "").lower().strip()

    # Direct regex match for known vague phrases
    for pat in VAGUE_PATTERNS:
        if re.match(pat, text):
            return True, (
                "To provide the correct troubleshooting steps, could you specify what you are observing?"
            ), [
                "Does not power on at all",
                "Powers on, but does not draw a sample",
                "Displays an error code on the screen",
                "Readings are abnormal or fluctuating",
                "Fluid leakage or unusual noise",
            ]

    # Length / word count heuristic: fewer than 3 content words without an error code
    words = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text)
    has_error_code = bool(re.search(r"\b(err|error|code|e-?\d+|f-?\d+)\b", text))
    if len(words) < 3 and not has_error_code:
        return True, (
            "Could you provide a few more details about the issue with your equipment?"
        ), [
            "Does not power on",
            "Fails during sample cycle",
            "Specific error code displayed",
            "Calibration or measurement error",
        ]

    return False, "", []


# ---------------------------------------------------------------------------
# Industrial Safety Classification
# ---------------------------------------------------------------------------

def classify_action_safety(step_text: str) -> Tuple[str, str]:
    """
    Classify a troubleshooting action according to industrial safety boundaries.
    Returns (safety_level, reason_or_warning)
    safety_level is one of:
      - 'TECHNICIAN_ONLY': Dangerous, must not be presented as a customer DIY action
      - 'SAFE_OBSERVATION': Non-invasive check (visual, switch status, display reading)
      - 'CUSTOMER_SAFE': Permissible operator action (cleaning, power-cycling, flushing)
    """
    lower = step_text.lower()

    # Check for technician-only hazardous instructions
    for kw in TECHNICIAN_ONLY_KEYWORDS:
        if kw in lower:
            return "TECHNICIAN_ONLY", f"Procedure involves hazardous work ({kw}). Certified technician required."

    # Check for non-invasive observations
    if any(lower.startswith(k) or f" {k} " in lower for k in OBSERVATION_KEYWORDS):
        # Non-invasive unless it touches internal electricals
        if any(w in lower for w in ["cabinet", "panel", "enclosure", "chassis"]) and any(w in lower for w in ["open", "remove", "disassemble"]):
            return "TECHNICIAN_ONLY", "Requires opening equipment chassis/enclosure."
        return "SAFE_OBSERVATION", "Approved non-invasive visual/status check."

    # General operator action
    return "CUSTOMER_SAFE", "Approved operator procedure."


# ---------------------------------------------------------------------------
# Structured Diagnostic Response Contract
# ---------------------------------------------------------------------------

def build_structured_diagnostic_contract(
    *,
    interaction: CustomerInteraction,
    session: Optional[DiagnosticSession],
    raw_answer: str,
    retrieved_references: List[Dict[str, Any]],
    clarification_question: Optional[str] = None,
    clarification_options: Optional[List[str]] = None,
    is_hazard: bool = False,
    hazard_warning: str = "",
) -> Dict[str, Any]:
    """
    Constructs an authoritative, validated structured customer diagnostic contract.
    Ensures:
      - Clean separation of customer-safe vs technician-only steps
      - Explicit citations tied to verified retrieved documents
      - Safe fallback when evidence is insufficient
      - Backward compatibility with legacy frontend keys (answer, references)
    """
    asset = interaction.asset
    asset_dict = {
        "id": asset.id,
        "name": asset.name,
        "asset_code": asset.asset_code,
        "model_number": asset.model_number or "Not recorded",
    } if asset else {}

    # Extract clean references and validate against actual retrieved documents
    validated_citations = []
    seen_citation_keys = set()
    for ref in (retrieved_references or []):
        doc_id = ref.get("document_id")
        chunk_id = ref.get("chunk_id")
        title = ref.get("title") or ref.get("reference") or "Approved Technical Reference"
        heading = ref.get("heading") or ""
        doc_type = ref.get("doc_type") or "Technical Document"
        key = (doc_id, chunk_id, title, heading)
        if key not in seen_citation_keys:
            seen_citation_keys.add(key)
            validated_citations.append({
                "document_id": doc_id,
                "chunk_id": chunk_id,
                "title": title,
                "heading": heading,
                "reference": ref.get("reference") or title,
                "doc_type": doc_type,
            })

    if is_hazard:
        return {
            "response_type": "ESCALATION_REQUIRED",
            "interaction_id": interaction.id,
            "session_id": str(session.session_id) if session else None,
            "session_version": session.version if session else 1,
            "session_status": session.status if session else "ESCALATED",
            "asset": asset_dict,
            "summary": "Immediate Safety Risk Detected",
            "clarification_question": None,
            "clarification_options": None,
            "check_first": [],
            "customer_safe_actions": [],
            "technician_only_actions": [{
                "action_id": "hazard_stop",
                "action": "Immediate onsite technician service required.",
                "warning": hazard_warning,
            }],
            "expected_result": "Equipment is safely de-energized pending technician dispatch.",
            "stop_conditions": [hazard_warning],
            "citations": validated_citations,
            "can_resolve": False,
            "can_escalate": True,
            "answer": f"DIAGNOSTIC SUMMARY\n{hazard_warning}\n\nSTOP AND ESCALATE IF\n- {hazard_warning}",
            "references": retrieved_references,
            "engine": "safety-boundary",
        }

    if clarification_question:
        return {
            "response_type": "CLARIFICATION",
            "interaction_id": interaction.id,
            "session_id": str(session.session_id) if session else None,
            "session_version": session.version if session else 1,
            "session_status": session.status if session else "WAITING_INPUT",
            "asset": asset_dict,
            "summary": "Please clarify your equipment symptom so we can provide accurate technical steps.",
            "clarification_question": clarification_question,
            "clarification_options": clarification_options or [],
            "check_first": [],
            "customer_safe_actions": [],
            "technician_only_actions": [],
            "expected_result": "Clarification enables retrieval of exact approved documentation.",
            "stop_conditions": ["If you are unsure or unable to inspect the equipment, request a service technician."],
            "citations": validated_citations,
            "can_resolve": False,
            "can_escalate": True,
            "answer": f"DIAGNOSTIC SUMMARY\n{clarification_question}",
            "references": retrieved_references,
            "engine": "clarification-controller",
        }

    # Parse sections from raw answer text
    summary = ""
    check_first = []
    customer_safe_actions = []
    technician_only_actions = []
    expected_result = ""
    stop_conditions = []

    # Parse sections from the standard 6-header text or fallback
    lines = (raw_answer or "").split("\n")
    current_sec = None
    step_num = 1

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        header_match = re.match(
            r"^(DIAGNOSTIC SUMMARY|CHECK FIRST|STEP-BY-STEP TROUBLESHOOTING|EXPECTED RESULT|STOP AND ESCALATE IF|VERIFIED REFERENCES)\s*$",
            stripped,
            re.IGNORECASE
        )
        if header_match:
            current_sec = header_match.group(1).upper()
            continue

        if current_sec == "DIAGNOSTIC SUMMARY":
            if not summary:
                summary = stripped
            else:
                summary += " " + stripped

        elif current_sec == "CHECK FIRST":
            clean_check = re.sub(r"^[-*•\d\.\)]\s*", "", stripped)
            safety_level, reason = classify_action_safety(clean_check)
            if safety_level == "TECHNICIAN_ONLY":
                technician_only_actions.append({
                    "action_id": f"tech_check_{len(technician_only_actions) + 1}",
                    "action": clean_check,
                    "warning": reason,
                })
            else:
                check_first.append(clean_check)

        elif current_sec == "STEP-BY-STEP TROUBLESHOOTING":
            step_match = re.match(r"^Step\s*\d*\s*[-:]?\s*(.+)", stripped, re.IGNORECASE)
            action_text = step_match.group(1).strip() if step_match else stripped
            if action_text.lower().startswith("why:"):
                # Attach reason to last customer action if present
                if customer_safe_actions:
                    customer_safe_actions[-1]["why"] = action_text[4:].strip()
                continue

            clean_action = re.sub(r"^[-*•\d\.\)]\s*", "", action_text)
            safety_level, reason = classify_action_safety(clean_action)
            if safety_level == "TECHNICIAN_ONLY":
                technician_only_actions.append({
                    "action_id": f"tech_step_{len(technician_only_actions) + 1}",
                    "action": clean_action,
                    "warning": reason,
                })
            else:
                citation_label = validated_citations[0]["reference"] if validated_citations else "Approved Technical Manual"
                customer_safe_actions.append({
                    "action_id": f"step_{step_num}",
                    "step_number": step_num,
                    "action": clean_action,
                    "why": "",
                    "safety_classification": safety_level,
                    "citation_ref": citation_label,
                })
                step_num += 1

        elif current_sec == "EXPECTED RESULT":
            clean_res = re.sub(r"^[-*•]\s*", "", stripped)
            if not expected_result:
                expected_result = clean_res
            else:
                expected_result += " " + clean_res

        elif current_sec == "STOP AND ESCALATE IF":
            clean_stop = re.sub(r"^[-*•]\s*", "", stripped)
            stop_conditions.append(clean_stop)

    # If only technician actions were detected and no customer-safe actions:
    if technician_only_actions and not customer_safe_actions:
        response_type = "ESCALATION_REQUIRED"
        stop_conditions.insert(0, "All documented technical steps require certified technician tools or training.")
    else:
        response_type = "DIAGNOSTIC_GUIDANCE"

    if not summary:
        summary = f"Guidance based on approved documentation for {asset.name if asset else 'equipment'}."
    if not expected_result:
        expected_result = "Equipment returns to normal operating parameters."
    if not stop_conditions:
        stop_conditions = ["If issue persists after performing safe initial checks, escalate to service technician."]

    return {
        "response_type": response_type,
        "interaction_id": interaction.id,
        "session_id": str(session.session_id) if session else None,
        "session_version": session.version if session else 1,
        "session_status": session.status if session else "ACTIVE",
        "asset": asset_dict,
        "summary": summary,
        "clarification_question": None,
        "clarification_options": None,
        "check_first": check_first,
        "customer_safe_actions": customer_safe_actions,
        "technician_only_actions": technician_only_actions,
        "expected_result": expected_result,
        "stop_conditions": stop_conditions,
        "citations": validated_citations,
        "can_resolve": True,
        "can_escalate": True,
        "answer": raw_answer,
        "references": retrieved_references,
        "engine": "unified-rag-zero-repeat",
    }


# ---------------------------------------------------------------------------
# Persistent DiagnosticSession Reconciliation
# ---------------------------------------------------------------------------

def get_or_create_customer_session(
    *,
    tenant,
    customer: Customer,
    site: Site,
    asset: Asset,
    complaint: str,
    user=None,
) -> DiagnosticSession:
    """
    Finds an active DiagnosticSession for the given asset and customer, or creates
    a new persistent session and records initial event ledger entries.
    """
    active_session = DiagnosticSession.objects.filter(
        tenant=tenant,
        customer=customer,
        asset=asset,
        status__in=["ACTIVE", "WAITING_INPUT", "WAITING_VERIFY", "NO_PLAYBOOK_AVAILABLE"],
    ).order_by("-updated_at").first()

    if active_session:
        # Append updated complaint event if significantly new
        last_seq = active_session.events.count()
        DiagnosticEvent.objects.create(
            tenant=tenant,
            session=active_session,
            seq_num=last_seq + 1,
            event_type=EVENT_COMPLAINT_RECORDED,
            actor_role="customer",
            payload={"complaint": complaint},
        )
        active_session.complaint = complaint
        active_session.version += 1
        active_session.save(update_fields=["complaint", "version", "updated_at"])
        return active_session

    with transaction.atomic():
        session = DiagnosticSession.objects.create(
            tenant=tenant,
            customer=customer,
            site=site,
            asset=asset,
            complaint=complaint,
            status="ACTIVE",
            current_node_id="intake",
            version=1,
            created_by=user,
        )

        DiagnosticEvent.objects.create(
            tenant=tenant,
            session=session,
            seq_num=1,
            event_type=EVENT_SESSION_STARTED,
            actor_role="customer",
            payload={"start_node_id": "intake", "asset_id": asset.id, "channel": "customer_support"},
        )

        DiagnosticEvent.objects.create(
            tenant=tenant,
            session=session,
            seq_num=2,
            event_type=EVENT_COMPLAINT_RECORDED,
            actor_role="customer",
            payload={"complaint": complaint},
        )

        return session


def record_customer_action_presentation(
    session: DiagnosticSession,
    actions: List[Dict[str, Any]],
    tenant,
) -> None:
    """Record that customer-safe actions were presented to the user."""
    if not session or not actions:
        return
    last_seq = session.events.count()
    DiagnosticEvent.objects.create(
        tenant=tenant,
        session=session,
        seq_num=last_seq + 1,
        event_type=EVENT_SAFE_ACTION_PRESENTED,
        actor_role="system",
        payload={"presented_actions": actions, "count": len(actions)},
    )


def resolve_customer_session(
    *,
    tenant,
    customer: Customer,
    session: DiagnosticSession,
    interaction: CustomerInteraction,
    verified: bool = False,
) -> None:
    """
    Transition a customer session to RESOLVED status with explicit audit event,
    preserving the distinction between customer-reported success and verified repair.
    """
    with transaction.atomic():
        interaction.resolved = True
        interaction.save(update_fields=["resolved"])

        if session:
            last_seq = session.events.count()
            DiagnosticEvent.objects.create(
                tenant=tenant,
                session=session,
                seq_num=last_seq + 1,
                event_type=EVENT_SESSION_RESOLVED,
                actor_role="customer",
                payload={
                    "resolution_type": "VERIFIED_RESOLVED" if verified else "CUSTOMER_REPORTED",
                    "verified": verified,
                    "interaction_id": interaction.id,
                },
            )
            session.status = "RESOLVED"
            session.version += 1
            session.save(update_fields=["status", "version", "updated_at"])
