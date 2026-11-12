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
    "interlock bypass",
    "bypass interlock",
    "override interlock",
    "disable interlock",
    "jumper interlock",
    "bridge interlock",
    "force interlock",
    "jumper wire",
    "tape door switch",
    "tape switch",
    "wire around",
    "unsafe reset",
    "factory reset bypass",
    "ignore previous safety",
    "disregard safety",
    "override safety policy",
]

TECHNICIAN_ONLY_KEYWORDS = [
    "energized",
    "live circuit",
    "live wire",
    "high voltage",
    "415v",
    "440v",
    "480v",
    "three-phase",
    "3-phase",
    "three phase",
    "phase voltage",
    "voltage measurement",
    "measure voltage",
    "multimeter on live",
    "multimeter",
    "220v internal",
    "internal wiring",
    "open cabinet while running",
    "open electrical cabinet",
    "open cabinet",
    "open electrical",
    "electrical cabinet",
    "cabinet access",
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
    "refrigerant servicing",
    "reclaim refrigerant",
    "charge refrigerant",
    "refrigerant",
    "freon",
    "r-410a",
    "r-32",
    "r-134a",
    "r-22",
    "manifold gauge",
    "vacuum pump",
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
    "bleed hydraulic",
    "pump assembly manipulation",
    "manipulate pump assembly",
    "disassemble pump",
    "pressure relief valve adjustment",
    "adjust pressure relief valve",
    "pressure relief valve",
    "pressurized component",
    "high pressure line",
    "capacitor discharge",
    # Unapproved hazardous chemicals & washes
    "sodium hydroxide",
    "caustic soda",
    "nitric acid",
    "hydrochloric acid",
    "sulfuric acid",
    "toxic chemical",
    "concentrated chemical",
    "chemical cleaning",
    "chemical wash",
    "chemical washing",
    "acid wash",
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
                "IMMEDIATE SAFETY RISK DETECTED: The reported condition may indicate immediate danger. "
                "Move away from the equipment and keep others clear. Do not touch the equipment, controls, "
                "wiring or electrical switches. Do not open, operate, reset or attempt to disconnect power. "
                "Follow the site's emergency procedure and notify the site safety lead and an authorized "
                "service technician. Contact emergency responders if anyone is injured or there is immediate danger."
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

# Versioned Authoritative Customer Procedure Registry
# Separates RAG candidate action discovery from deterministic procedure authorization.
VERSIONED_APPROVED_PROCEDURES: Dict[str, Dict[str, Any]] = {
    # Split AC Unit 2T (SKY-AST-00001) - Unverified pending authenticated OEM signoff
    "PROC-SKY-AST-00001-AIR-FILTER-CLEAN": {
        "action_id": "PROC-SKY-AST-00001-AIR-FILTER-CLEAN",
        "equipment_model": "Split AC Unit 2T",
        "asset_code": "SKY-AST-00001",
        "document_sha256": "9e5d2d9e772dc71a849fa5cd1b7512b60f0f417f6832e0cdc77a879ab301e8d1",
        "document_version": "1.0",
        "procedure_keywords": [
            "wipe dust filter", "clean exterior and wipe dust filter with clean water",
            "clean filters and confirm airflow is unobstructed", "clean exterior and wipe dust filter"
        ],
        "permitted_audience": "customer",
        "preconditions": ["unit_powered_off", "no_water_hazard"],
        "reviewer_id": "UNVERIFIED",
        "approval_evidence": "PENDING_AUTHENTICATED_OEM_APPROVAL",
        "approval_status": "UNVERIFIED",
        "approval_timestamp": None,
        "revoked": False,
        "safety_policy": "non-invasive-external-filter-clean",
    },
    "PROC-SKY-AST-00001-POWER-RESET": {
        "action_id": "PROC-SKY-AST-00001-POWER-RESET",
        "equipment_model": "Split AC Unit 2T",
        "asset_code": "SKY-AST-00001",
        "document_sha256": "9e5d2d9e772dc71a849fa5cd1b7512b60f0f417f6832e0cdc77a879ab301e8d1",
        "document_version": "1.0",
        "procedure_keywords": [
            "restart the unit using the main power switch", "restart using power switch"
        ],
        "permitted_audience": "customer",
        "preconditions": ["no_burning_smell", "no_tripping_breaker"],
        "reviewer_id": "UNVERIFIED",
        "approval_evidence": "PENDING_AUTHENTICATED_OEM_APPROVAL",
        "approval_status": "UNVERIFIED",
        "approval_timestamp": None,
        "revoked": False,
        "safety_policy": "external-main-switch-reset",
    },
    # General / Demo Asset Procedures (strictly for isolated demo/test fixtures)
    "PROC-DEMO-SAMPLE-TUBE-CLEAN": {
        "action_id": "PROC-DEMO-SAMPLE-TUBE-CLEAN",
        "equipment_model": "Milk Analyzer",
        "asset_code": "*",
        "document_sha256": "*",
        "document_version": "1.0",
        "procedure_keywords": [
            "clean sample intake tube using 5% sodium hydroxide", "clean sample intake tube",
            "rinse sample tube", "distilled water", "clean water",
            "run automated cleaning cycle", "wash reservoir contains sufficient"
        ],
        "permitted_audience": "customer",
        "preconditions": ["wash_reservoir_accessible"],
        "reviewer_id": "DEMO_TEST_ONLY",
        "approval_evidence": "DEMO_TEST_FIXTURE",
        "approval_status": "TEST_ONLY",
        "approval_timestamp": None,
        "revoked": False,
        "safety_policy": "external-rinse-non-chemical",
    },
}


def classify_action_safety(step_text: str, asset: Optional[Asset] = None) -> Tuple[str, str]:
    """
    Classify a troubleshooting action according to industrial safety boundaries.
    Enforces deterministic procedure authorization. Fails closed.
    Returns (safety_level, reason_or_warning).
    """
    lower = (step_text or "").lower()

    # 1. Prohibited and Interlock Tamper check
    if "interlock" in lower:
        return "PROHIBITED", "Defeating, bypassing, or servicing safety interlocks is strictly prohibited for customer operations."

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

    # 4. Check Database-backed ProcedureApproval Model with all 10 industrial safety gates
    asset_code = getattr(asset, "asset_code", "") or ""
    asset_model = getattr(asset, "model_number", "") or ""
    asset_mfr = getattr(asset, "manufacturer", "") or ""
    if not asset_mfr and hasattr(asset, "product") and asset.product:
        asset_mfr = getattr(asset.product, "manufacturer", "") or ""
    if isinstance(asset, dict):
        asset_code = asset.get("asset_code", "") or ""
        asset_model = asset.get("model_number", "") or ""
        asset_mfr = asset.get("manufacturer", "") or ""

    if asset:
        try:
            from core.models import ProcedureApproval
            from core.services.safety_authority import validate_procedure_safety_approval
            tenant = getattr(asset, "tenant", None)
            if tenant:
                approvals = ProcedureApproval.objects.filter(
                    tenant=tenant,
                    approval_status="APPROVED",
                    permitted_audience="customer",
                    revoked_at__isnull=True,
                ).select_related("document", "source_chunk")
                for app in approvals:
                    is_valid, classification, reason = validate_procedure_safety_approval(app, lower, asset=asset, tenant=tenant)
                    if is_valid:
                        return classification, reason
                    elif classification in ("TECHNICIAN_ONLY", "PROHIBITED"):
                        return classification, reason
        except Exception:
            pass

    # 5. Deterministic Versioned Procedure Check
    matched_proc = None
    for proc_id, proc in VERSIONED_APPROVED_PROCEDURES.items():
        if proc.get("revoked", False):
            continue
        if proc.get("asset_code") != "*" and asset_code and proc.get("asset_code") != asset_code:
            continue
        if any(kw in lower for kw in proc.get("procedure_keywords", [])):
            matched_proc = proc
            break

    if matched_proc:
        # Check if approval status is UNVERIFIED (per Ironclad 5.0 contract)
        if matched_proc.get("approval_status") == "UNVERIFIED":
            return "UNKNOWN_REQUIRES_REVIEW", f"Procedure '{matched_proc['action_id']}' lacks verified manufacturer sign-off. Certified technician required."

        if matched_proc.get("approval_status") == "TEST_ONLY":
            from .customer_isolation import is_test_environment
            if not is_test_environment():
                return "UNKNOWN_REQUIRES_REVIEW", f"Procedure '{matched_proc['action_id']}' is not authorized for customer production use."

        if matched_proc.get("permitted_audience") not in ("customer", "operator"):
            return "TECHNICIAN_ONLY", f"Procedure '{matched_proc['action_id']}' is certified for technician use only."

        # Model-inapplicable or undocumented assets fail closed
        if asset and not KnowledgeDocument.objects.filter(
            tenant=asset.tenant,
            asset=asset,
            is_rag_enabled=True,
            is_confidential=False,
        ).exists() and not KnowledgeDocument.objects.filter(
            tenant=asset.tenant,
            product=asset.product,
            asset__isnull=True,
            is_rag_enabled=True,
            is_confidential=False,
        ).exists():
            return "UNKNOWN_REQUIRES_REVIEW", f"Equipment '{asset.name}' has no authorized technical manual. Customer repair procedure cannot be approved."

        return "CUSTOMER_APPROVED", f"Authorized customer procedure ({matched_proc['action_id']}) pending deterministic manual grounding."

    # 6. Fail-Closed Default: Unapproved actions require technician review
    return "UNKNOWN_REQUIRES_REVIEW", "Action is not an authorized customer-safe procedure. Certified technician review required."


# ---------------------------------------------------------------------------
# Strict Action Grounding Verification
# ---------------------------------------------------------------------------

def find_grounding_reference(
    action_text: str,
    retrieved_references: List[Dict[str, Any]],
    tenant_id: int,
    asset: Optional[Asset] = None,
    customer: Optional[Customer] = None,
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Verifies that action_text is deterministically supported by an applicable, authorized,
    persisted, non-quarantined chunk loaded directly from the database.
    Caller-provided excerpts and whole-document fallbacks are strictly rejected.
    """
    if not retrieved_references or not action_text:
        return False, None, "NO_REFERENCES_AVAILABLE"

    for ref in retrieved_references:
        raw_doc_id = ref.get("document_id")
        raw_chunk_id = ref.get("chunk_id")

        if not raw_doc_id or not raw_chunk_id:
            continue

        try:
            doc_id = int(raw_doc_id)
            chunk_id = int(raw_chunk_id)
        except (ValueError, TypeError):
            continue

        # Load chunk directly from database with strict relational bindings
        chunk = KnowledgeChunk.objects.filter(
            id=chunk_id,
            document_id=doc_id,
            tenant_id=tenant_id,
        ).select_related("document", "document__asset", "document__product").first()
        if not chunk:
            continue

        if chunk.is_quarantined:
            continue

        doc = chunk.document
        if not doc:
            continue

        # Document operational gates
        if not doc.is_rag_enabled or doc.is_confidential or getattr(doc, "disable_sharing", False):
            continue
        if doc.index_status not in ("INDEXED", "INDEXING"):
            if doc.index_status == "FAILED":
                # Sparse fallback consistency: check if chunk has verified text or embedding content
                if not (chunk.text or chunk.content_embedding or chunk.embedding):
                    continue
            else:
                continue

        # Customer isolation
        if doc.customer_id and customer and doc.customer_id != customer.id:
            continue

        # Asset and equipment hierarchy applicability (canonical 5-level hierarchy)
        from core.services.knowledge_scope import is_document_applicable_to_asset
        if not is_document_applicable_to_asset(doc, asset):
            continue

        # Real-only provenance verification
        from .customer_isolation import get_customer_data_mode, is_verified_real_document
        if get_customer_data_mode() == "verified_real_only":
            if not is_verified_real_document(doc, allow_failed=True):
                continue

        # Check prompt injection on the persisted chunk text
        from core.services.content_safety import detect_document_prompt_injection
        if chunk.text:
            injections = detect_document_prompt_injection(chunk.text)
            if injections or (chunk.safety_flags and "injection" in str(chunk.safety_flags).lower()):
                continue

        # MUST validate deterministically against persisted chunk text (NOT caller-provided excerpt)
        is_bound, reason = validate_source_binding(action_text, chunk.text)
        if is_bound:
            verified_ref = dict(ref)
            verified_ref["document_id"] = doc.id
            verified_ref["chunk_id"] = chunk.id
            verified_ref["heading"] = chunk.heading or ""
            verified_ref["text"] = chunk.text
            verified_ref["document_version"] = str(getattr(doc, "index_version", "1.0") or "1.0")
            return True, verified_ref, "VALID"

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

    if "presentation" not in contract:
        from .customer_answer import exception_presentation
        contract["presentation"] = exception_presentation(contract)
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
    conversation_context: Optional[Any] = None,
    original_question: Optional[str] = None,
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

    # Exception responses carry no sources. Ordinary citations are constructed
    # only from persisted, scoped chunks by customer_answer below.
    # Reconstructing a contract must not erase a previously reported hazard
    # merely because the caller did not supply the optional hazard flag.
    if not is_hazard:
        is_hazard, derived_warning = check_for_safety_hazard(interaction.question or "")
        if is_hazard and not hazard_warning:
            hazard_warning = derived_warning

    if is_hazard:
        if not hazard_warning:
            hazard_warning = "Immediate safety risk detected. Stop equipment operation and request an authorized service technician."
        res = {
            "response_mode": "IMMEDIATE_SAFETY_ESCALATION",
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
            "citations": [],
            "can_resolve": False,
            "can_escalate": True,
            "answer": f"DIAGNOSTIC SUMMARY\n{hazard_warning}\n\nSTOP AND ESCALATE IF\n- {hazard_warning}",
            "references": [],
            "engine": "safety-boundary",
        }
        return sanitize_customer_payload(res)

    if clarification_question:
        res = {
            "response_mode": "CLARIFICATION_REQUIRED",
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
            "citations": [],
            "can_resolve": False,
            "can_escalate": True,
            "answer": f"DIAGNOSTIC SUMMARY\n{clarification_question}",
            "references": [],
            "engine": "clarification-controller",
        }
        return sanitize_customer_payload(res)

    # Check if raw_answer is a policy-guard response
    raw_lower = (raw_answer or "").lower()
    if "cannot reveal system instructions" in raw_lower or (raw_answer or "").startswith("I can help with product"):
        res = {
            "response_mode": "POLICY_BLOCKED",
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
        return sanitize_customer_payload(res)

    # Ordinary customer answers come from validated persisted passages. The
    # generator's free-form text is not an authority for claims or procedures.
    from .customer_answer import build_customer_answer
    return build_customer_answer(
        interaction,
        session,
        retrieved_references,
        conversation_context=conversation_context,
        original_question=original_question,
    )


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
    session_id: Optional[str] = None,
    force_new: bool = False,
) -> DiagnosticSession:
    """Finds a specific or active DiagnosticSession, or creates a new persistent session with atomic ledger."""
    with transaction.atomic():
        active_session = None

        if session_id:
            # Explicit session lookup with strict customer and tenant isolation
            active_session = (
                DiagnosticSession.objects.select_for_update()
                .filter(
                    tenant=tenant,
                    customer=customer,
                    session_id=session_id,
                )
                .first()
            )
            if not active_session:
                from rest_framework.exceptions import PermissionDenied
                raise PermissionDenied("Diagnostic session not found or does not belong to authorized customer.")

        elif not force_new:
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
            if not active_session.complaint:
                active_session.complaint = complaint
                event_type = EVENT_COMPLAINT_RECORDED
                payload = {"complaint": complaint}
                update_fields = ["complaint", "version", "updated_at"]
            else:
                event_type = EVENT_OBSERVATION_RECORDED
                payload = {"observation": complaint}
                update_fields = ["version", "updated_at"]

            DiagnosticEvent.objects.create(
                tenant=tenant,
                session=active_session,
                seq_num=last_seq + 1,
                event_type=event_type,
                actor_role="customer",
                payload=payload,
            )
            active_session.version += 1
            active_session.save(update_fields=update_fields)
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
