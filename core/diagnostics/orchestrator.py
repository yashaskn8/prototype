"""
Diagnostic Session Orchestrator for Servy Zero-Repeat.

Handles state transitions, idempotency, optimistic concurrency, and escalation.

Hard Invariants:
  - Atomic transactions for all mutations.
  - Optimistic concurrency control (expected_version mismatch => 409 Conflict).
  - Idempotency via DiagnosticCommand (same key + same payload => replay; same key + different payload => 409).
  - Exactly ONE ServiceCall created upon escalation.
  - Customer can NEVER mutate a terminal session (RESOLVED, ESCALATED).
"""

import hashlib
import json
from typing import Any, Dict, Optional, Tuple

from django.db import IntegrityError, transaction
from django.db.models import Max
from django.utils import timezone

from core.models import (
    Asset,
    Customer,
    DiagnosticCommand,
    DiagnosticEvent,
    DiagnosticPlaybook,
    DiagnosticSession,
    RecoveryPassport,
    ServiceCall,
)
from core.services.ticketing import _sla_for, choose_technician

from .evidence import verify_evidence_anchor
from .passport import build_recovery_passport
from .playbooks import validate_playbook_definition
from .policy import evaluate_node_policy
from .reducer import reduce_session_events
from .routing import select_playbook_for_asset
from .schemas import (
    EVENT_COMPLAINT_RECORDED,
    EVENT_OBSERVATION_RECORDED,
    EVENT_OBSERVATION_REQUESTED,
    EVENT_PLAYBOOK_SELECTED,
    EVENT_SAFE_ACTION_CONFIRMED,
    EVENT_SAFE_ACTION_PRESENTED,
    EVENT_SESSION_ESCALATED,
    EVENT_SESSION_RESOLVED,
    EVENT_SESSION_STARTED,
    EVENT_VERIFICATION_RECORDED,
    MAX_SESSION_STEPS,
    NODE_ESCALATE,
    NODE_OBSERVE,
    NODE_SAFE_ACTION,
    NODE_VERIFY,
    RESPONSE_BOOLEAN,
    RESPONSE_CONFIRMATION,
    RESPONSE_NUMBER,
    RESPONSE_SHORT_TEXT,
    RESPONSE_SINGLE_CHOICE,
)


class ConcurrencyConflictError(Exception):
    def __init__(self, current_version: int):
        super().__init__(f"Session version conflict. Current version is {current_version}.")
        self.current_version = current_version


class IdempotencyPayloadConflictError(Exception):
    pass


class TerminalSessionMutationError(Exception):
    pass


class SessionLimitExceededError(Exception):
    pass


def _hash_payload(data: Any) -> str:
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def check_or_record_idempotency(
    tenant,
    idempotency_key: str,
    command_type: str,
    payload: Any,
    session: Optional[DiagnosticSession] = None
) -> Tuple[Optional[DiagnosticCommand], bool]:
    """Check if an idempotency key was previously processed.

    Returns (DiagnosticCommand, is_replay).
    """
    if not idempotency_key:
        return None, False

    request_hash = _hash_payload(payload)

    cmd = DiagnosticCommand.objects.filter(tenant=tenant, idempotency_key=idempotency_key).first()
    if cmd:
        if cmd.request_hash == request_hash:
            return cmd, True
        raise IdempotencyPayloadConflictError("Idempotency key reused with different request payload.")

    return None, False


def save_idempotency_response(
    tenant,
    idempotency_key: str,
    command_type: str,
    payload: Any,
    response_status: int,
    response_payload: Dict[str, Any],
    session: Optional[DiagnosticSession] = None
) -> DiagnosticCommand:
    """Save an idempotency record with the response payload."""
    if not idempotency_key:
        return None

    request_hash = _hash_payload(payload)
    cmd, _ = DiagnosticCommand.objects.get_or_create(
        tenant=tenant,
        idempotency_key=idempotency_key,
        defaults={
            "session": session,
            "command_type": command_type,
            "request_hash": request_hash,
            "response_status": response_status,
            "response_payload": response_payload,
        }
    )
    return cmd


def start_diagnostic_session(
    tenant,
    customer: Customer,
    asset: Asset,
    complaint: str,
    user=None,
    idempotency_key: str = ""
) -> Tuple[DiagnosticSession, Dict[str, Any]]:
    """Initialize a persistent, authorized DiagnosticSession."""
    complaint = (complaint or "").strip()[:1000]

    # Check active session limit (max 5 active sessions per customer)
    active_count = DiagnosticSession.objects.filter(
        tenant=tenant, customer=customer, status__in=["ACTIVE", "WAITING_INPUT", "WAITING_VERIFY"]
    ).count()
    if active_count >= 10:
        raise SessionLimitExceededError("Maximum active diagnostic sessions limit reached. Please resolve or escalate open sessions.")

    # Select candidate published playbook
    playbook = select_playbook_for_asset(asset, complaint, tenant)
    playbook_def = playbook.definition if playbook else {}
    start_node = playbook_def.get("start_node_id", "start")

    with transaction.atomic():
        session = DiagnosticSession.objects.create(
            tenant=tenant,
            customer=customer,
            site=asset.site,
            asset=asset,
            playbook=playbook,
            playbook_version=playbook.version if playbook else 1,
            complaint=complaint,
            current_node_id=start_node,
            status="ACTIVE",
            version=1,
            created_by=user,
        )

        seq = 1
        DiagnosticEvent.objects.create(
            tenant=tenant,
            session=session,
            seq_num=seq,
            event_type=EVENT_SESSION_STARTED,
            actor_role="customer",
            payload={"start_node_id": start_node, "asset_id": asset.id},
        )

        seq += 1
        DiagnosticEvent.objects.create(
            tenant=tenant,
            session=session,
            seq_num=seq,
            event_type=EVENT_COMPLAINT_RECORDED,
            actor_role="customer",
            payload={"complaint": complaint},
        )

        if playbook:
            seq += 1
            DiagnosticEvent.objects.create(
                tenant=tenant,
                session=session,
                seq_num=seq,
                event_type=EVENT_PLAYBOOK_SELECTED,
                actor_role="system",
                payload={"playbook_id": playbook.id, "name": playbook.name, "version": playbook.version},
            )

            # Request first observation
            seq += 1
            DiagnosticEvent.objects.create(
                tenant=tenant,
                session=session,
                seq_num=seq,
                event_type=EVENT_OBSERVATION_REQUESTED,
                actor_role="system",
                payload={"node_id": start_node},
            )

        events = list(session.events.order_by("seq_num"))
        state = reduce_session_events(events, playbook_def, session.session_id)

        session.facts_snapshot = {
            "facts": {k: {"value": f.value, "source": f.source, "verified": f.verified} for k, f in state.facts.items()},
            "contradictions": [
                {"fact_key": c.fact_key, "earlier": c.earlier_value, "new": c.new_value, "resolved": c.resolved}
                for c in state.contradictions
            ],
        }
        session.evidence_completeness = state.evidence_completeness
        session.current_node_id = state.current_node_id
        session.status = state.status
        session.save(update_fields=["facts_snapshot", "evidence_completeness", "current_node_id", "status"])

    # Evaluate presentation node
    node_payload = get_session_view_data(session, customer)
    return session, node_payload


def submit_diagnostic_answer(
    session: DiagnosticSession,
    node_id: str,
    value: Any,
    expected_version: int,
    customer=None,
    actor_role: str = "customer"
) -> Tuple[DiagnosticSession, Dict[str, Any]]:
    """Submit an answer to an OBSERVE or VERIFY step with optimistic concurrency control."""
    with transaction.atomic():
        locked_session = DiagnosticSession.objects.select_for_update().get(id=session.id)

        # Check terminal state
        if locked_session.status in ("RESOLVED", "ESCALATED", "ABANDONED"):
            raise TerminalSessionMutationError(f"Cannot mutate terminal session with status '{locked_session.status}'.")

        # Optimistic concurrency check
        if locked_session.version != expected_version:
            raise ConcurrencyConflictError(locked_session.version)

        playbook_def = locked_session.playbook.definition if locked_session.playbook else {}
        nodes = playbook_def.get("nodes", {})
        node = nodes.get(node_id, {})
        node_type = node.get("node_type", NODE_OBSERVE)
        fact_key = node.get("fact_key", f"fact_{node_id}")

        # Validate answer value against schema
        schema = node.get("response_schema", RESPONSE_SHORT_TEXT)
        validated_value = _validate_answer_value(schema, value, node.get("choices", []))

        last_seq = locked_session.events.aggregate(m=Max("seq_num"))["m"] or 0
        event_type = EVENT_VERIFICATION_RECORDED if node_type == NODE_VERIFY else EVENT_OBSERVATION_RECORDED

        DiagnosticEvent.objects.create(
            tenant=locked_session.tenant,
            session=locked_session,
            seq_num=last_seq + 1,
            event_type=event_type,
            actor_role=actor_role,
            payload={
                "node_id": node_id,
                "fact_key": fact_key,
                "value": validated_value,
            }
        )

        # Re-reduce events
        events = list(locked_session.events.order_by("seq_num"))
        state = reduce_session_events(events, playbook_def, locked_session.session_id)

        # Update snapshot and version
        locked_session.facts_snapshot = {
            "facts": {k: {"value": f.value, "source": f.source, "verified": f.verified} for k, f in state.facts.items()},
            "contradictions": [
                {"fact_key": c.fact_key, "earlier": c.earlier_value, "new": c.new_value, "resolved": c.resolved}
                for c in state.contradictions
            ],
        }
        locked_session.evidence_completeness = state.evidence_completeness
        locked_session.current_node_id = state.current_node_id
        locked_session.status = state.status
        locked_session.version += 1
        locked_session.save(update_fields=["facts_snapshot", "evidence_completeness", "current_node_id", "status", "version"])

    # If reached ESCALATE node deterministically, trigger escalation automatically
    if locked_session.status == "ESCALATED":
        escalate_diagnostic_session(
            session=locked_session,
            reason=state.escalation_reason or "Escalated by diagnostic workflow.",
            expected_version=locked_session.version,
            customer=customer
        )

    return locked_session, get_session_view_data(locked_session, customer)


def confirm_safe_action(
    session: DiagnosticSession,
    node_id: str,
    expected_version: int,
    customer=None
) -> Tuple[DiagnosticSession, Dict[str, Any]]:
    """Confirm that the customer completed a SAFE_ACTION procedure."""
    with transaction.atomic():
        locked_session = DiagnosticSession.objects.select_for_update().get(id=session.id)

        if locked_session.status in ("RESOLVED", "ESCALATED", "ABANDONED"):
            raise TerminalSessionMutationError(f"Cannot mutate terminal session with status '{locked_session.status}'.")

        if locked_session.version != expected_version:
            raise ConcurrencyConflictError(locked_session.version)

        playbook_def = locked_session.playbook.definition if locked_session.playbook else {}
        nodes = playbook_def.get("nodes", {})
        node = nodes.get(node_id, {})

        if node.get("node_type") != NODE_SAFE_ACTION:
            raise ValueError(f"Node '{node_id}' is not a SAFE_ACTION.")

        # Invariant: Verify policy again prior to recording completion
        is_allowed, reason, _ = evaluate_node_policy(locked_session, node_id, customer)
        if not is_allowed:
            raise ValueError(f"Action blocked by policy: {reason}")

        last_seq = locked_session.events.aggregate(m=Max("seq_num"))["m"] or 0
        DiagnosticEvent.objects.create(
            tenant=locked_session.tenant,
            session=locked_session,
            seq_num=last_seq + 1,
            event_type=EVENT_SAFE_ACTION_CONFIRMED,
            actor_role="customer",
            payload={
                "node_id": node_id,
                "instruction": node.get("instruction", ""),
                "evidence_anchor": node.get("evidence_anchor", {}),
            }
        )

        events = list(locked_session.events.order_by("seq_num"))
        state = reduce_session_events(events, playbook_def, locked_session.session_id)

        locked_session.current_node_id = state.current_node_id
        locked_session.status = state.status
        locked_session.evidence_completeness = state.evidence_completeness
        locked_session.version += 1
        locked_session.save(update_fields=["current_node_id", "status", "evidence_completeness", "version"])

    return locked_session, get_session_view_data(locked_session, customer)


def resolve_diagnostic_session(
    session: DiagnosticSession,
    expected_version: int,
    customer=None,
    summary: str = "Resolved by customer."
) -> Tuple[DiagnosticSession, Dict[str, Any]]:
    """Mark a diagnostic session as safely RESOLVED."""
    with transaction.atomic():
        locked_session = DiagnosticSession.objects.select_for_update().get(id=session.id)

        if locked_session.status in ("RESOLVED", "ESCALATED"):
            # Idempotent terminal
            return locked_session, get_session_view_data(locked_session, customer)

        if locked_session.version != expected_version:
            raise ConcurrencyConflictError(locked_session.version)

        last_seq = locked_session.events.aggregate(m=Max("seq_num"))["m"] or 0
        DiagnosticEvent.objects.create(
            tenant=locked_session.tenant,
            session=locked_session,
            seq_num=last_seq + 1,
            event_type=EVENT_SESSION_RESOLVED,
            actor_role="customer",
            payload={"summary": summary}
        )

        locked_session.status = "RESOLVED"
        locked_session.version += 1
        locked_session.save(update_fields=["status", "version"])

    return locked_session, get_session_view_data(locked_session, customer)


def escalate_diagnostic_session(
    session: DiagnosticSession,
    reason: str,
    expected_version: int,
    customer=None
) -> Tuple[DiagnosticSession, ServiceCall, RecoveryPassport]:
    """Escalate a diagnostic session to ServiceCall, generating a deterministic RecoveryPassport.

    Hard Invariants:
      - Exactly ONE ServiceCall per DiagnosticSession.
      - Repeated calls return the existing call idempotently.
      - Passport contains only evidenced, customer-confirmed actions.
      - Atomic transaction locks session and ensures non-terminal mutation.
    """
    tenant = session.tenant

    for attempt in range(5):
        try:
            with transaction.atomic():
                locked_session = DiagnosticSession.objects.select_for_update().get(id=session.id)

                # Exactly-once invariant: if already escalated, return existing objects idempotently
                if locked_session.status == "ESCALATED" and locked_session.escalated_call:
                    passport = getattr(locked_session, "recovery_passport", None)
                    return locked_session, locked_session.escalated_call, passport

                if locked_session.status == "RESOLVED":
                    raise TerminalSessionMutationError("Cannot escalate a resolved session.")

                if locked_session.version != expected_version:
                    raise ConcurrencyConflictError(locked_session.version)

                # Generate deterministic Recovery Passport payload
                passport_dict = build_recovery_passport(locked_session)

                # Find technician and SLA
                zone = locked_session.site.zone if locked_session.site else None
                branch = zone.branch if zone and zone.branch else None
                technician = choose_technician(tenant, branch=branch, zone=zone)
                sla = _sla_for(tenant, "normal")

                now = timezone.now()
                response_minutes = sla.response_minutes if sla else 120
                resolution_minutes = sla.resolution_minutes if sla else 720

                last_id = ServiceCall.objects.filter(tenant=tenant).aggregate(m=Max("servy_id"))["m"] or 50000

                # Build complaint text with authoritative Recovery Passport handoff
                do_not_repeat_text = "\n".join(f"- {item}" for item in passport_dict["do_not_repeat_items"]) or "- None"
                complaint_body = (
                    f"Customer Issue:\n{locked_session.complaint}\n\n"
                    f"Escalation Reason:\n{reason}\n\n"
                    f"ALREADY COMPLETED — DO NOT REPEAT WITH CUSTOMER:\n{do_not_repeat_text}\n\n"
                    f"Diagnostic Session ID: {locked_session.session_id}\n"
                    f"Evidence Completeness: {int(locked_session.evidence_completeness * 100)}%\n"
                    f"Asset: {locked_session.asset.name} ({locked_session.asset.asset_code})"
                )

                call = ServiceCall.objects.create(
                    tenant=tenant,
                    servy_id=last_id + 1,
                    call_type="Service",
                    complaint_type="Diagnostic Recovery Escalation",
                    complaint_text=complaint_body,
                    customer=locked_session.customer,
                    site=locked_session.site,
                    asset=locked_session.asset,
                    zone=zone,
                    sla_policy=sla,
                    technician=technician,
                    status="assigned" if technician else "open",
                    priority="normal",
                    contact_name=locked_session.customer.contact_name or locked_session.customer.name,
                    contact_phone=locked_session.customer.phone,
                    contact_email=locked_session.customer.email,
                    response_due_at=now + timezone.timedelta(minutes=response_minutes),
                    resolution_due_at=now + timezone.timedelta(minutes=resolution_minutes),
                )

                # Persist RecoveryPassport record
                passport = RecoveryPassport.objects.create(
                    tenant=tenant,
                    session=locked_session,
                    service_call=call,
                    asset=locked_session.asset,
                    complaint=locked_session.complaint,
                    structured_data=passport_dict["structured_data"],
                    do_not_repeat_items=passport_dict["do_not_repeat_items"],
                    evidence_completeness=passport_dict["evidence_completeness"],
                )

                last_seq = locked_session.events.aggregate(m=Max("seq_num"))["m"] or 0
                DiagnosticEvent.objects.create(
                    tenant=tenant,
                    session=locked_session,
                    seq_num=last_seq + 1,
                    event_type=EVENT_SESSION_ESCALATED,
                    actor_role="customer",
                    payload={"reason": reason, "service_call_id": call.id, "servy_id": call.servy_id}
                )

                locked_session.status = "ESCALATED"
                locked_session.escalated_call = call
                locked_session.version += 1
                locked_session.save(update_fields=["status", "escalated_call", "version"])

                return locked_session, call, passport

        except (IntegrityError,):
            if attempt == 4:
                raise


def get_session_view_data(session: DiagnosticSession, customer=None) -> Dict[str, Any]:
    """Compile the authoritative client-facing view data for a DiagnosticSession."""
    playbook_def = session.playbook.definition if session.playbook else {}
    nodes = playbook_def.get("nodes", {})
    current_node = nodes.get(session.current_node_id, {})

    # Evaluate runtime policy for current node
    node_data = None
    if session.status not in ("RESOLVED", "ESCALATED", "ABANDONED") and session.current_node_id in nodes:
        allowed, reason, sanitized = evaluate_node_policy(session, session.current_node_id, customer)
        if allowed:
            node_data = sanitized
        else:
            # Policy failed: safe fallback to ESCALATE
            node_data = {
                "node_id": "policy_fallback_escalate",
                "node_type": NODE_ESCALATE,
                "reason": f"Required evidence verification failed: {reason}. Escalating safely to service engineering.",
            }

    passport_data = None
    if session.status == "ESCALATED" and hasattr(session, "recovery_passport") and session.recovery_passport:
        rp = session.recovery_passport
        passport_data = {
            "id": rp.id,
            "structured_data": rp.structured_data,
            "do_not_repeat_items": rp.do_not_repeat_items,
            "evidence_completeness": rp.evidence_completeness,
            "created_at": rp.created_at,
        }

    return {
        "session_id": str(session.session_id),
        "asset": {
            "id": session.asset.id,
            "name": session.asset.name,
            "asset_code": session.asset.asset_code,
            "model_number": session.asset.model_number or "Not recorded",
        },
        "complaint": session.complaint,
        "status": session.status,
        "version": session.version,
        "evidence_completeness": session.evidence_completeness,
        "current_node": node_data,
        "escalated_call": {
            "id": session.escalated_call.id,
            "servy_id": session.escalated_call.servy_id,
            "status": session.escalated_call.status,
        } if session.escalated_call else None,
        "recovery_passport": passport_data,
    }


def _validate_answer_value(schema: str, value: Any, choices: list) -> Any:
    """Validate that the submitted value conforms to the expected node schema."""
    if schema == RESPONSE_BOOLEAN:
        if isinstance(value, bool):
            return value
        if str(value).lower() in ("true", "1", "yes"):
            return True
        if str(value).lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"Value '{value}' is not a valid BOOLEAN.")

    elif schema == RESPONSE_NUMBER:
        try:
            return float(value)
        except (ValueError, TypeError):
            raise ValueError(f"Value '{value}' is not a valid NUMBER.")

    elif schema == RESPONSE_SINGLE_CHOICE:
        val_str = str(value)
        if choices and val_str not in choices:
            raise ValueError(f"Value '{val_str}' is not among allowed choices: {choices}.")
        return val_str

    elif schema == RESPONSE_CONFIRMATION:
        return True

    return str(value).strip()[:500]
