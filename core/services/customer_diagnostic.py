"""
Customer Diagnostic Service for Servy RAG.

Provides fail-closed industrial safety classification, input normalization,
equipment-aware ambiguity/clarification detection, strict action-to-evidence grounding,
leak-free customer response contracts, and atomic DiagnosticSession reconciliation.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from django.db import transaction
from django.db.models import Max
from django.utils import timezone

from core.models import (
    Asset,
    Customer,
    CustomerInteraction,
    DiagnosticEvent,
    DiagnosticSession,
    KnowledgeChunk,
    KnowledgeDocument,
    Site,
)
from core.diagnostics.evidence import validate_source_binding
from core.diagnostics.orchestrator import TerminalSessionMutationError
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

PROHIBITED_KEYWORDS = [
    "bypass safety",
    "defeat interlock",
    "disable safety",
    "bridge fuse",
    "short circuit safety",
    "defeat ground",
    "tamper with calibration",
    "tamper with potentiometer",
    "override limit",
]

TECHNICIAN_ONLY_KEYWORDS = [
    "energized",
    "live circuit",
    "live wire",
    "high voltage",
    "415v",
    "440v",
    "220v internal",
    "internal wiring",
    "open cabinet while running",
    "open electrical cabinet",
    "open cabinet",
    "open electrical",
    "open chassis",
    "open enclosure",
    "open housing",
    "open cover",
    "remove panel",
    "remove cover",
    "unscrew cover",
    "disassemble motor",
    "replace control board",
    "replace pcb",
    "pressurized vessel overhaul",
    "internal transformer",
    "refrigerant recovery",
    "refrigerant charging",
    "freon",
    "compressor teardown",
    "disassemble",
    "dismantle",
    "solder",
    "desolder",
    "motor winding",
    "probe terminal",
    "probe live",
    "strip wire",
    "logic analyzer",
    "oscilloscope",
    "hydraulic cylinder overhaul",
    "pump assembly manipulation",
    "manipulate pump assembly",
    "disassemble pump",
    "pressure relief valve adjustment",
    "adjust pressure relief valve",
    # Unapproved hazardous chemicals
    "sodium hydroxide",
    "caustic soda",
    "nitric acid",
    "hydrochloric acid",
    "sulfuric acid",
    "toxic chemical",
    "concentrated chemical",
    "chemical cleaning",
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
    "monitor",
]

APPROVED_OPERATOR_ACTIONS = [
    "clean exterior",
    "wipe exterior",
    "wipe surface",
    "dry cloth",
    "rinse sample tube",
    "rinse external",
    "flush external",
    "distilled water",
    "clean water",
    "power cycle",
    "turn off power switch",
    "restart the unit",
    "unplug power cord",
    "reconnect power cord",
    "reconnect tubing",
    "seat container",
    "load sample",
    "press reset button",
    "run automated cleaning cycle",
    "peristaltic pump roller carriage for blockage",
    "sample intake tubing is securely attached",
    "wash reservoir contains sufficient",
]


def check_for_safety_hazard(text_input: str) -> Tuple[bool, str]:
    """Identify immediate safety-critical symptoms that prohibit customer DIY troubleshooting."""
    text = (text_input or "").lower().strip()
    for pat in HAZARD_PATTERNS:
        if re.search(pat, text):
            return True, (
                "IMMEDIATE SAFETY RISK DETECTED: Symptoms indicate a hazardous electrical, thermal, or chemical condition. "
                "Disconnect main power immediately if it is safe to do so. Do not operate or open the equipment. "
                "An urgent technician service call should be dispatched."
            )
    return False, ""


def check_for_vague_complaint(complaint: str, asset: Optional[Asset] = None) -> Tuple[bool, str, List[str]]:
    """Determine if a customer complaint is too underspecified to retrieve accurate technical procedures.

    Provides domain-aware options based on equipment context (e.g. dairy analyzers vs lifts vs cooling).
    """
    text = (complaint or "").lower().strip()

    is_vague = False
    for pat in VAGUE_PATTERNS:
        if re.match(pat, text):
            is_vague = True
            break

    if not is_vague:
        words = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text)
        has_error_code = bool(re.search(r"\b(err|error|code|e-?\d+|f-?\d+)\b", text))
        if len(words) < 3 and not has_error_code:
            is_vague = True

    if not is_vague:
        return False, "", []

    # Domain-aware clarification options
    asset_name = (((asset.name or "") + " " + (asset.product.name if asset.product else "")) if asset else "").lower()

    if any(k in asset_name for k in ["analyzer", "milk", "lacto", "dairy"]):
        clarif_opts = [
            "Does not power on at all",
            "Powers on, but does not draw a sample",
            "Displays an error code on the screen",
            "Readings are abnormal or fluctuating",
            "Fluid leakage or unusual noise",
        ]
    elif any(k in asset_name for k in ["refrigerat", "chiller", "cooling", "freezer"]):
        clarif_opts = [
            "Does not power on at all",
            "Compressor runs but does not cool",
            "Frost or ice buildup on lines",
            "Temperature alarm or error code displayed",
            "Unusual vibration or noise",
        ]
    elif any(k in asset_name for k in ["lift", "hydraulic", "press", "crane", "hoist"]):
        clarif_opts = [
            "Does not power on / motor inactive",
            "Hydraulic pressure loss or slow movement",
            "Fluid leak or puddle beneath unit",
            "Unusual grinding, squealing, or knocking noise",
            "Control panel error code",
        ]
    else:
        clarif_opts = [
            "Does not power on at all",
            "Powers on, but operation halts unexpectedly",
            "Error code or fault message displayed",
            "Unusual noise, vibration, or leakage",
            "Performance degradation or calibration fault",
        ]

    return True, "To provide the correct troubleshooting steps, could you specify what you are observing?", clarif_opts


# ---------------------------------------------------------------------------
# Industrial Safety Classification (Fail-Closed)
# ---------------------------------------------------------------------------

def classify_action_safety(step_text: str) -> Tuple[str, str]:
    """
    Classify a troubleshooting action according to industrial safety boundaries.
    Returns (safety_level, reason_or_warning)
    safety_level is strictly one of:
      - 'PROHIBITED': Violates basic safety principles (interlock defeat, tampering)
      - 'TECHNICIAN_ONLY': Dangerous or invasive, certified technician required
      - 'INFORMATIONAL_OBSERVATION': Non-invasive external check (visual, status, display)
      - 'CUSTOMER_APPROVED': Authorized customer procedure
      - 'UNKNOWN_REQUIRES_REVIEW': Fail-closed default for unverified actions
    """
    lower = (step_text or "").lower()

    # 1. Prohibited check
    for kw in PROHIBITED_KEYWORDS:
        if kw in lower:
            return "PROHIBITED", f"Prohibited safety violation ({kw}). Defeating interlocks or safety circuits is strictly forbidden."

    # 2. Technician-only check
    for kw in TECHNICIAN_ONLY_KEYWORDS:
        if kw in lower:
            return "TECHNICIAN_ONLY", f"Procedure involves hazardous work ({kw}). Certified technician required."

    # 3. Observation keywords check
    is_obs = any(lower.startswith(k) or f" {k} " in lower for k in OBSERVATION_KEYWORDS)
    if is_obs:
        # Check if observation requires invasive internal access
        if any(w in lower for w in ["cabinet", "panel", "enclosure", "chassis", "housing", "internal", "wiring", "terminal"]):
            if any(w in lower for w in ["open", "remove", "disassemble", "probe", "touch", "strip"]):
                return "TECHNICIAN_ONLY", "Requires opening equipment chassis/enclosure or accessing internal components."
        return "INFORMATIONAL_OBSERVATION", "Approved non-invasive visual/status check."

    # 4. Check for recognized safe operator actions
    for safe_act in APPROVED_OPERATOR_ACTIONS:
        if safe_act in lower:
            return "CUSTOMER_APPROVED", "Approved operator procedure."

    # 5. Fail-Closed Default: Unknown actions require technician review
    return "UNKNOWN_REQUIRES_REVIEW", "Action is not an authorized customer-safe procedure. Certified technician review required."


# ---------------------------------------------------------------------------
# Strict Action Grounding Verification
# ---------------------------------------------------------------------------

def find_grounding_reference(
    action_text: str,
    retrieved_references: List[Dict[str, Any]],
    tenant_id: int,
    asset: Optional[Asset] = None,
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Verifies that action_text is deterministically supported by an applicable, authorized,
    non-quarantined chunk in the retrieved references.
    """
    if not retrieved_references or not action_text:
        return False, None, "NO_REFERENCES_AVAILABLE"

    for ref in retrieved_references:
        doc_id = ref.get("document_id")
        chunk_id = ref.get("chunk_id")
        excerpt = (
            ref.get("excerpt")
            or ref.get("chunk_text")
            or ref.get("text")
            or ref.get("content")
            or ""
        )

        # Re-verify document active state in DB if doc_id is available
        if doc_id:
            doc = KnowledgeDocument.objects.filter(id=doc_id, tenant_id=tenant_id).select_related("product").first()
            if not doc:
                continue
            if not doc.is_rag_enabled or doc.is_confidential:
                continue
            if asset:
                if doc.asset_id and doc.asset_id != asset.id:
                    continue
                if doc.product_id and asset.product_id and doc.product_id != asset.product_id:
                    continue

        if chunk_id:
            chunk = KnowledgeChunk.objects.filter(id=chunk_id, document_id=doc_id).first()
            if chunk:
                if chunk.is_quarantined:
                    continue
                if not excerpt:
                    excerpt = chunk.text

        # If excerpt still empty, fallback to document content if doc available
        if not excerpt and doc_id:
            doc = KnowledgeDocument.objects.filter(id=doc_id, tenant_id=tenant_id).first()
            if doc:
                excerpt = doc.content_text

        is_bound, reason = validate_source_binding(action_text, excerpt)
        if is_bound:
            return True, ref, "VALID"

    return False, None, "ACTION_NOT_GROUNDED_IN_CHUNKS"


# ---------------------------------------------------------------------------
# Customer Payload Sanitization (Zero Procedure Leakage)
# ---------------------------------------------------------------------------

RESTRICTED_SNIPPETS = [
    "open electrical cabinet",
    "defeat interlock",
    "bridge fuse",
    "live circuit",
    "live wire",
    "415v",
    "440v",
    "open cabinet while running",
    "disassemble motor",
    "replace control board",
    "tamper with calibration",
    "pressurized vessel overhaul",
    "internal transformer",
    "refrigerant recovery",
    "refrigerant charging",
    "sodium hydroxide",
    "caustic soda",
    "nitric acid",
    "hydrochloric acid",
]


def sanitize_customer_payload(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively inspects and sanitizes all customer-facing strings.
    Replaces any restricted or technician-only instructions with safe technician advice.
    """
    def _contains_restricted(text: str) -> bool:
        lower = text.lower()
        return any(snip in lower for snip in RESTRICTED_SNIPPETS)

    def _sanitize_string(s: str) -> str:
        if _contains_restricted(s):
            return "This procedure involves restricted technical operations. Please request an authorized service technician."
        return s

    def _walk_and_clean(val: Any) -> Any:
        if isinstance(val, str):
            return _sanitize_string(val)
        elif isinstance(val, list):
            return [_walk_and_clean(item) for item in val]
        elif isinstance(val, dict):
            return {k: _walk_and_clean(v) for k, v in val.items()}
        return val

    # Clean the technician_only_actions to ensure NO raw dangerous procedural text is exposed
    if "technician_only_actions" in contract:
        clean_tech = []
        for act in contract["technician_only_actions"]:
            clean_tech.append({
                "action_id": act.get("action_id", "tech_procedure"),
                "action": "Authorized service technician procedure required.",
                "warning": "This procedure requires an authorized service technician. Please request service.",
                "requires_technician": True,
            })
        contract["technician_only_actions"] = clean_tech

    return _walk_and_clean(contract)


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
      - Fail-closed safety decisions
      - No technician-only actions or detailed instructions exposed to customers
      - Grounding verification per action against valid source chunks
      - Truthful insufficient evidence handling (no generic invented guidance)
    """
    asset = interaction.asset
    asset_dict = {
        "id": asset.id,
        "name": asset.name,
        "asset_code": asset.asset_code,
        "model_number": asset.model_number or "Not recorded",
    } if asset else {}

    tenant_id = interaction.tenant_id

    # Filter and validate citations
    validated_citations = []
    seen_citation_keys = set()
    for ref in (retrieved_references or []):
        doc_id = ref.get("document_id")
        chunk_id = ref.get("chunk_id")
        title = ref.get("title") or ref.get("reference") or "Approved Technical Reference"
        heading = ref.get("heading") or ""
        doc_type = ref.get("doc_type") or "Technical Document"

        # Check doc validity
        if doc_id:
            doc = KnowledgeDocument.objects.filter(id=doc_id, tenant_id=tenant_id).first()
            if not doc or not doc.is_rag_enabled or doc.is_confidential:
                continue

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
        res = {
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
                "action": "Authorized service technician procedure required.",
                "warning": "This procedure requires an authorized service technician. Please request service.",
                "requires_technician": True,
            }],
            "expected_result": "",
            "stop_conditions": [hazard_warning],
            "citations": validated_citations,
            "can_resolve": False,
            "can_escalate": True,
            "answer": f"DIAGNOSTIC SUMMARY\n{hazard_warning}\n\nSTOP AND ESCALATE IF\n- {hazard_warning}",
            "references": [],
            "engine": "safety-boundary",
        }
        return sanitize_customer_payload(res)

    if clarification_question:
        res = {
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
            "expected_result": "",
            "stop_conditions": ["If you are unsure or unable to inspect the equipment, request a service technician."],
            "citations": validated_citations,
            "can_resolve": False,
            "can_escalate": True,
            "answer": f"DIAGNOSTIC SUMMARY\n{clarification_question}",
            "references": retrieved_references,
            "engine": "clarification-controller",
        }
        return sanitize_customer_payload(res)

    # Check if raw_answer is a policy-guard response
    raw_lower = (raw_answer or "").lower()
    if "cannot reveal system instructions" in raw_lower or (raw_answer or "").startswith("I can help with product"):
        return {
            "response_type": "POLICY_VIOLATION",
            "interaction_id": interaction.id,
            "session_id": str(session.session_id) if session else None,
            "session_version": session.version if session else 1,
            "session_status": session.status if session else "ACTIVE",
            "asset": asset_dict,
            "summary": raw_answer,
            "clarification_question": None,
            "clarification_options": None,
            "check_first": [],
            "customer_safe_actions": [],
            "technician_only_actions": [],
            "expected_result": "",
            "stop_conditions": ["Input violated security or system instructions policy."],
            "citations": [],
            "can_resolve": False,
            "can_escalate": True,
            "answer": raw_answer,
            "references": [],
            "engine": "policy-guard",
        }

    # Parse sections from raw answer text
    summary = ""
    check_first = []
    customer_safe_actions = []
    technician_only_actions = []
    expected_result = ""
    stop_conditions = []

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
            re.IGNORECASE,
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
            if safety_level in ("TECHNICIAN_ONLY", "PROHIBITED"):
                technician_only_actions.append({
                    "action_id": f"tech_check_{len(technician_only_actions) + 1}",
                    "action": "Authorized service technician procedure required.",
                    "warning": reason,
                })
            elif safety_level in ("CUSTOMER_APPROVED", "INFORMATIONAL_OBSERVATION"):
                # Non-invasive check
                check_first.append(clean_check)
                is_grounded, matched_ref, ground_reason = find_grounding_reference(
                    clean_check,
                    retrieved_references,
                    tenant_id,
                    asset,
                )
                if is_grounded and matched_ref:
                    citation_label = matched_ref.get("reference") or matched_ref.get("title") or "Approved Technical Manual"
                    doc_id = matched_ref.get("document_id")
                    chunk_id = matched_ref.get("chunk_id")
                    customer_safe_actions.append({
                        "action_id": f"step_{step_num}",
                        "step_number": step_num,
                        "action": clean_check,
                        "why": "",
                        "safety_classification": safety_level,
                        "citation_ref": citation_label,
                        "document_id": doc_id,
                        "chunk_id": chunk_id,
                    })
                    step_num += 1

        elif current_sec == "STEP-BY-STEP TROUBLESHOOTING":
            step_match = re.match(r"^Step\s*\d*\s*[-:]?\s*(.+)", stripped, re.IGNORECASE)
            action_text = step_match.group(1).strip() if step_match else stripped
            if action_text.lower().startswith("why:"):
                if customer_safe_actions:
                    customer_safe_actions[-1]["why"] = action_text[4:].strip()
                continue

            clean_action = re.sub(r"^[-*•\d\.\)]\s*", "", action_text)
            safety_level, reason = classify_action_safety(clean_action)

            if safety_level in ("TECHNICIAN_ONLY", "PROHIBITED"):
                technician_only_actions.append({
                    "action_id": f"tech_step_{len(technician_only_actions) + 1}",
                    "action": "Authorized service technician procedure required.",
                    "warning": reason,
                })
            elif safety_level in ("CUSTOMER_APPROVED", "INFORMATIONAL_OBSERVATION"):
                # Strict action-to-evidence grounding check
                is_grounded, matched_ref, ground_reason = find_grounding_reference(
                    clean_action,
                    retrieved_references,
                    tenant_id,
                    asset,
                )
                if is_grounded and matched_ref:
                    citation_label = matched_ref.get("reference") or matched_ref.get("title") or "Approved Technical Manual"
                    doc_id = matched_ref.get("document_id")
                    chunk_id = matched_ref.get("chunk_id")
                    customer_safe_actions.append({
                        "action_id": f"step_{step_num}",
                        "step_number": step_num,
                        "action": clean_action,
                        "why": "",
                        "safety_classification": safety_level,
                        "citation_ref": citation_label,
                        "document_id": doc_id,
                        "chunk_id": chunk_id,
                    })
                    step_num += 1
                else:
                    # Ungrounded action fails closed
                    pass
            else:
                # UNKNOWN_REQUIRES_REVIEW fails closed (not presented as customer-safe)
                pass

        elif current_sec == "EXPECTED RESULT":
            clean_res = re.sub(r"^[-*•]\s*", "", stripped)
            if not expected_result:
                expected_result = clean_res
            else:
                expected_result += " " + clean_res

        elif current_sec == "STOP AND ESCALATE IF":
            clean_stop = re.sub(r"^[-*•]\s*", "", stripped)
            stop_conditions.append(clean_stop)

    # Truthful Response Type Determination
    if not customer_safe_actions and not check_first:
        if technician_only_actions:
            response_type = "ESCALATION_REQUIRED"
            summary = "All documented technical procedures require certified technician tools or training."
            stop_conditions = ["All documented technical steps require certified technician tools or training. Please request service."]
            answer_text = "DIAGNOSTIC SUMMARY\nAll documented technical steps require certified technician tools or training.\n\nSTOP AND ESCALATE IF\n- Please request an authorized service technician."
        else:
            response_type = "INSUFFICIENT_EVIDENCE"
            summary = "No approved customer troubleshooting procedures could be verified for this issue."
            stop_conditions = ["No manufacturer-approved customer self-service steps are available. Please request an authorized service technician."]
            answer_text = "INSUFFICIENT EVIDENCE: No approved customer diagnostic procedures exist in the Knowledge Base for this issue on this equipment. Please escalate to a certified service technician."
        expected_result = ""
        can_resolve = False
    elif technician_only_actions and not customer_safe_actions:
        response_type = "ESCALATION_REQUIRED"
        summary = "All documented technical steps require certified technician tools or training."
        stop_conditions = ["All documented technical steps require certified technician tools or training. Please request service."]
        answer_text = "DIAGNOSTIC SUMMARY\nAll documented technical steps require certified technician tools or training.\n\nSTOP AND ESCALATE IF\n- Please request an authorized service technician."
        expected_result = ""
        can_resolve = False
    else:
        response_type = "DIAGNOSTIC_GUIDANCE"
        can_resolve = True
        if not summary:
            summary = f"Guidance based on approved documentation for {asset.name if asset else 'equipment'}."
        if not stop_conditions:
            stop_conditions = ["If issue persists after performing safe initial checks, escalate to service technician."]

        # Rebuild customer-safe answer text without any raw technician procedures
        safe_steps_text = "\n".join(f"{act['step_number']}. {act['action']}" for act in customer_safe_actions)
        answer_parts = [f"DIAGNOSTIC SUMMARY\n{summary}"]
        if check_first:
            answer_parts.append("CHECK FIRST\n" + "\n".join(f"- {c}" for c in check_first))
        if safe_steps_text:
            answer_parts.append(f"STEP-BY-STEP TROUBLESHOOTING\n{safe_steps_text}")
        if expected_result:
            answer_parts.append(f"EXPECTED RESULT\n{expected_result}")
        if stop_conditions:
            answer_parts.append("STOP AND ESCALATE IF\n" + "\n".join(f"- {s}" for s in stop_conditions))
        answer_text = "\n\n".join(answer_parts)

    res = {
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
        "can_resolve": can_resolve,
        "can_escalate": True,
        "answer": answer_text,
        "references": retrieved_references,
        "engine": "unified-rag-zero-repeat",
    }
    return sanitize_customer_payload(res)


# ---------------------------------------------------------------------------
# Persistent DiagnosticSession Reconciliation (Atomic & Row-Locked)
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
    """Finds an active DiagnosticSession or creates a new persistent session with atomic ledger."""
    with transaction.atomic():
        active_session = (
            DiagnosticSession.objects.select_for_update()
            .filter(
                tenant=tenant,
                customer=customer,
                asset=asset,
                status__in=["ACTIVE", "WAITING_INPUT", "WAITING_VERIFY", "NO_PLAYBOOK_AVAILABLE"],
            )
            .order_by("-updated_at")
            .first()
        )

        if active_session:
            last_seq = active_session.events.aggregate(m=Max("seq_num"))["m"] or 0
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
    """Record presented customer-safe actions under atomic row lock."""
    if not session or not actions:
        return
    with transaction.atomic():
        locked_session = DiagnosticSession.objects.select_for_update().filter(id=session.id).first()
        if not locked_session or locked_session.status in ("RESOLVED", "ESCALATED", "ABANDONED"):
            return
        last_seq = locked_session.events.aggregate(m=Max("seq_num"))["m"] or 0
        DiagnosticEvent.objects.create(
            tenant=tenant,
            session=locked_session,
            seq_num=last_seq + 1,
            event_type=EVENT_SAFE_ACTION_PRESENTED,
            actor_role="system",
            payload={"presented_actions": actions, "count": len(actions)},
        )
        locked_session.version += 1
        locked_session.save(update_fields=["version", "updated_at"])


def resolve_customer_session(
    *,
    tenant,
    customer: Customer,
    session: Optional[DiagnosticSession],
    interaction: CustomerInteraction,
    verified: bool = False,
) -> None:
    """Transition a customer session to RESOLVED status with atomic audit event."""
    with transaction.atomic():
        if session:
            locked_session = DiagnosticSession.objects.select_for_update().filter(id=session.id).first()
            if not locked_session:
                raise ValueError("DiagnosticSession not found.")
            if locked_session.status in ("RESOLVED", "ESCALATED", "ABANDONED"):
                raise TerminalSessionMutationError(f"Cannot resolve session in terminal status '{locked_session.status}'.")

            last_seq = locked_session.events.aggregate(m=Max("seq_num"))["m"] or 0
            DiagnosticEvent.objects.create(
                tenant=tenant,
                session=locked_session,
                seq_num=last_seq + 1,
                event_type=EVENT_SESSION_RESOLVED,
                actor_role="customer",
                payload={
                    "resolution_type": "VERIFIED_RESOLVED" if verified else "CUSTOMER_REPORTED",
                    "verified": verified,
                    "interaction_id": interaction.id,
                },
            )
            locked_session.status = "RESOLVED"
            locked_session.version += 1
            locked_session.save(update_fields=["status", "version", "updated_at"])

        interaction.resolved = True
        interaction.save(update_fields=["resolved"])
