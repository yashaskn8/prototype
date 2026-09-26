"""
Diagnostic Session Orchestrator for Servy Zero-Repeat.

Handles state transitions, atomic idempotency lifecycle, optimistic concurrency, and escalation.

Hard Invariants:
  - Atomic transactions for all mutations.
  - Optimistic concurrency control (expected_version must match locked DB record).
  - Current-node enforcement (client cannot skip ahead, replay old nodes, or execute out-of-order).
  - Presentation requirement (SAFE_ACTION confirmation strictly requires prior SAFE_ACTION_PRESENTED in sequence).
  - Atomic escalation inside the same transaction that sets ESCALATED status.
  - Exactly ONE ServiceCall per DiagnosticSession, repeated escalations are idempotent.
  - Immutable DiagnosticEvent append-only ledger.
  - Customer can NEVER mutate a terminal session (RESOLVED, ESCALATED, ABANDONED).
  - Resolution safety: manual resolve requires a verified resolution state and no blocking contradictions.
  - Atomic idempotency reservation (IN_PROGRESS -> COMPLETED / FAILED) before business logic.
"""

import hashlib
import json
import re
import time
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

from django.core.exceptions import ValidationError
from django.core.serializers.json import DjangoJSONEncoder
from django.db import IntegrityError, OperationalError, transaction
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
from core.services.ticketing import (
    _sla_for,
    choose_technician,
    create_service_call_record,
)
from .evidence import verify_evidence_anchor
from .passport import build_recovery_passport
from .policy import evaluate_node_policy
from .reducer import reduce_session_events
from .routing import select_playbook_for_asset
from .schemas import (
    EVENT_CLARIFICATION_RECORDED,
    EVENT_COMPLAINT_RECORDED,
    EVENT_CONTRADICTION_RESOLVED,
    EVENT_OBSERVATION_RECORDED,
    EVENT_OBSERVATION_REQUESTED,
    EVENT_PLAYBOOK_SELECTED,
    EVENT_SAFE_ACTION_CONFIRMED,
    EVENT_SAFE_ACTION_PRESENTED,
    EVENT_SESSION_ESCALATED,
    EVENT_SESSION_RESOLVED,
    EVENT_SESSION_STARTED,
    EVENT_VERIFICATION_RECORDED,
    MAX_ACTIVE_SESSIONS_PER_CUSTOMER,
    MAX_SESSION_STEPS,
    NODE_ESCALATE,
    NODE_OBSERVE,
    NODE_SAFE_ACTION,
    NODE_VERIFY,
    PROGRESSION_EVENT_TYPES,
    RESPONSE_BOOLEAN,
    RESPONSE_CONFIRMATION,
    RESPONSE_NUMBER,
    RESPONSE_SHORT_TEXT,
    RESPONSE_SINGLE_CHOICE,
)


class ConcurrencyConflictError(Exception):
    def __init__(self, current_version: int = 0, detail: str = ""):
        msg = detail or f"Session version conflict. Current version is {current_version}."
        super().__init__(msg)
        self.current_version = current_version


class IdempotencyPayloadConflictError(Exception):
    pass


class TerminalSessionMutationError(Exception):
    pass


class SessionLimitExceededError(Exception):
    pass


class CurrentNodeMismatchError(Exception):
    """Raised when caller tries to submit for a node that is not the current node."""
    def __init__(self, expected_node: str, actual_node: str):
        super().__init__(
            f"Node mismatch: submitted for '{expected_node}' but session is at '{actual_node}'. "
            f"The client may interpret language, but the server decides the current legal transition."
        )
        self.expected_node = expected_node
        self.actual_node = actual_node


class PresentationRequiredError(Exception):
    """Raised when SAFE_ACTION_CONFIRMED is attempted without a prior SAFE_ACTION_PRESENTED."""
    pass


def _hash_payload(data: Any) -> str:
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _generate_presentation_token(session_id: Any, node_id: str, session_version: int, seq_num: int) -> str:
    raw = f"{session_id}:{node_id}:{session_version}:{seq_num}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# CRITICAL FIX 1: Atomic Idempotency Lifecycle Reservation
# ---------------------------------------------------------------------------

def reserve_idempotency_key(
    tenant,
    idempotency_key: str,
    command_type: str,
    payload: Any,
    session: Optional[DiagnosticSession] = None
) -> Tuple[Optional[DiagnosticCommand], bool]:
    """Atomically reserve an idempotency key before business logic execution.

    Returns:
        (DiagnosticCommand, is_replay)
        - If is_replay is True: command already completed; caller should return stored response.
        - If is_replay is False: reservation secured; caller must proceed with business mutation
          and call complete_idempotency() on success or fail_idempotency() on error.
    """
    if not idempotency_key:
        return None, False

    request_hash = _hash_payload(payload)

    for attempt in range(3):
        try:
            with transaction.atomic():
                cmd = DiagnosticCommand.objects.create(
                    tenant=tenant,
                    idempotency_key=idempotency_key,
                    command_type=command_type,
                    request_hash=request_hash,
                    session=session,
                    state=DiagnosticCommand.STATE_IN_PROGRESS,
                    response_status=200,
                    response_payload={},
                )
                return cmd, False
        except (IntegrityError, OperationalError):
            # Key already exists or concurrent insert
            cmd = DiagnosticCommand.objects.filter(tenant=tenant, idempotency_key=idempotency_key).first()
            if not cmd:
                time.sleep(0.05)
                continue

            # Command type conflict verification
            if cmd.command_type != command_type:
                raise IdempotencyPayloadConflictError(
                    f"Idempotency key reused for different command type '{cmd.command_type}' vs '{command_type}'."
                )

            # Payload conflict verification
            if cmd.request_hash != request_hash:
                raise IdempotencyPayloadConflictError("Idempotency key reused with different request payload.")

            if cmd.state == DiagnosticCommand.STATE_COMPLETED:
                return cmd, True

            if cmd.state == DiagnosticCommand.STATE_IN_PROGRESS:
                # Stale recovery check: if process died / timeout > 60s, permit recovery
                if cmd.updated_at and (timezone.now() - cmd.updated_at) > timedelta(seconds=60):
                    with transaction.atomic():
                        cmd = DiagnosticCommand.objects.select_for_update().get(id=cmd.id)
                        cmd.state = DiagnosticCommand.STATE_IN_PROGRESS
                        cmd.updated_at = timezone.now()
                        cmd.save(update_fields=["state", "updated_at"])
                        return cmd, False

                # Concurrent active execution
                raise ConcurrencyConflictError(
                    detail="A request with this idempotency key is currently in progress. Please retry shortly."
                )

            if cmd.state == DiagnosticCommand.STATE_FAILED:
                # Safe retry of failed command: transition to IN_PROGRESS
                with transaction.atomic():
                    cmd = DiagnosticCommand.objects.select_for_update().get(id=cmd.id)
                    cmd.state = DiagnosticCommand.STATE_IN_PROGRESS
                    cmd.request_hash = request_hash
                    cmd.save(update_fields=["state", "request_hash", "updated_at"])
                    return cmd, False

    raise ConcurrencyConflictError(detail="Could not acquire idempotency lock. Please retry.")


def _json_safe(value: Any) -> Any:
    """Normalize a Python structure to strictly JSON-compliant types (handling datetime, UUID, Decimal, etc.)."""
    if value is None:
        return None
    return json.loads(json.dumps(value, cls=DjangoJSONEncoder))


def complete_idempotency(
    cmd: Optional[DiagnosticCommand],
    response_status: int,
    response_payload: Dict[str, Any]
) -> None:
    """Mark an idempotency reservation as COMPLETED with authoritative response data."""
    if not cmd:
        return
    safe_payload = _json_safe(response_payload)
    DiagnosticCommand.objects.filter(id=cmd.id).update(
        state=DiagnosticCommand.STATE_COMPLETED,
        response_status=response_status,
        response_payload=safe_payload,
        updated_at=timezone.now(),
    )


def fail_idempotency(
    cmd: Optional[DiagnosticCommand],
    error_detail: str = "",
    response_status: int = 400
) -> None:
    """Mark an idempotency reservation as FAILED so subsequent retries can re-execute cleanly."""
    if not cmd:
        return
    safe_payload = _json_safe({"detail": error_detail or "Command execution failed."})
    DiagnosticCommand.objects.filter(id=cmd.id).update(
        state=DiagnosticCommand.STATE_FAILED,
        response_status=response_status,
        response_payload=safe_payload,
        updated_at=timezone.now(),
    )


def check_or_record_idempotency(
    tenant,
    idempotency_key: str,
    command_type: str,
    payload: Any,
    session: Optional[DiagnosticSession] = None
) -> Tuple[Optional[DiagnosticCommand], bool]:
    """Backwards-compatible wrapper around atomic reservation."""
    return reserve_idempotency_key(tenant, idempotency_key, command_type, payload, session)


def save_idempotency_response(
    tenant,
    idempotency_key: str,
    command_type: str,
    payload: Any,
    response_status: int,
    response_payload: Dict[str, Any],
    session: Optional[DiagnosticSession] = None
) -> Optional[DiagnosticCommand]:
    """Backwards-compatible wrapper to complete an idempotency record."""
    if not idempotency_key:
        return None
    request_hash = _hash_payload(payload)
    safe_payload = _json_safe(response_payload)
    cmd = DiagnosticCommand.objects.filter(tenant=tenant, idempotency_key=idempotency_key).first()
    if cmd:
        complete_idempotency(cmd, response_status, safe_payload)
        return cmd
    else:
        return DiagnosticCommand.objects.create(
            tenant=tenant,
            idempotency_key=idempotency_key,
            command_type=command_type,
            request_hash=request_hash,
            session=session,
            state=DiagnosticCommand.STATE_COMPLETED,
            response_status=response_status,
            response_payload=safe_payload,
        )


# ---------------------------------------------------------------------------
# HIGH FIX 7 & 8: Shared Mutation Guard & Step Counting
# ---------------------------------------------------------------------------

def _validate_mutable_session(locked_session: DiagnosticSession, expected_version: int) -> None:
    """Shared mutation guard enforcing session active state, expiration, optimistic lock, and max progression steps."""
    # Expiration check (7 days inactivity)
    if locked_session.updated_at and (timezone.now() - locked_session.updated_at) > timedelta(days=7):
        locked_session.status = "ABANDONED"
        locked_session.save(update_fields=["status"])
        raise TerminalSessionMutationError("Diagnostic session has expired due to inactivity.")

    # Terminal state check
    if locked_session.status in ("RESOLVED", "ESCALATED", "ABANDONED"):
        raise TerminalSessionMutationError(f"Cannot mutate terminal session with status '{locked_session.status}'.")

    # Optimistic concurrency check
    if locked_session.version != expected_version:
        raise ConcurrencyConflictError(locked_session.version)

    # Max progression steps (counts actual customer diagnostic progression, not raw audit events)
    progression_count = locked_session.events.filter(event_type__in=PROGRESSION_EVENT_TYPES).count()
    if progression_count >= MAX_SESSION_STEPS:
        raise ValueError(f"Maximum session diagnostic steps ({MAX_SESSION_STEPS}) reached. Please escalate to service.")


# ---------------------------------------------------------------------------
# Core Helpers
# ---------------------------------------------------------------------------

def _append_event(locked_session, event_type: str, actor_role: str, payload: dict) -> DiagnosticEvent:
    """Locked helper: append an immutable event to the ledger under existing row lock."""
    last_seq = locked_session.events.aggregate(m=Max("seq_num"))["m"] or 0
    return DiagnosticEvent.objects.create(
        tenant=locked_session.tenant,
        session=locked_session,
        seq_num=last_seq + 1,
        event_type=event_type,
        actor_role=actor_role,
        payload=payload,
    )


def _build_facts_snapshot(state):
    """Build a serializable facts snapshot from the reducer state."""
    return {
        "facts": {k: {"value": f.value, "source": f.source, "verified": f.verified} for k, f in state.facts.items()},
        "contradictions": [
            {"fact_key": c.fact_key, "earlier": c.earlier_value, "new": c.new_value, "resolved": c.resolved}
            for c in state.contradictions
        ],
    }


def _execute_escalation_atomically(
    locked_session,
    reason: str,
    customer=None,
    customer_note: str = ""
) -> Tuple[ServiceCall, RecoveryPassport]:
    """Execute escalation inside the SAME transaction that detected ESCALATED status."""
    tenant = locked_session.tenant

    # Exactly-once: if already escalated, return existing
    if locked_session.escalated_call_id:
        passport = getattr(locked_session, "recovery_passport", None)
        return locked_session.escalated_call, passport

    passport_dict = build_recovery_passport(locked_session)

    zone = locked_session.site.zone if locked_session.site else None
    branch = zone.branch if zone and zone.branch else None
    technician = choose_technician(tenant, branch=branch, zone=zone)
    sla = _sla_for(tenant, "normal")

    do_not_repeat_text = "\n".join(f"- {item}" for item in passport_dict["do_not_repeat_items"]) or "- None"

    # Sanitize customer note to prevent spoofing structural escalation headers or prompt injections
    clean_note = (customer_note or "").strip()[:500]
    if clean_note:
        clean_note = re.sub(r"</?retrieved_source\b[^>]*>", "", clean_note)
        clean_note = re.sub(r"UNTRUSTED_CONTENT_(?:START|END)", "", clean_note)
        clean_note = re.sub(
            r"(?im)^\s*(?:system diagnostic escalation reason|already completed|diagnostic session id|evidence completeness|customer issue)\s*:",
            "[Customer Note]:",
            clean_note
        )

    # HIGH FIX 19: Clearly distinguish customer comment from system escalation reason
    complaint_body = (
        f"Customer Issue:\n{locked_session.complaint}\n\n"
        f"Customer Comment / Requested Reason:\n{clean_note or 'None provided'}\n\n"
        f"System Diagnostic Escalation Reason:\n{reason}\n\n"
        f"ALREADY COMPLETED — DO NOT REPEAT WITH CUSTOMER:\n{do_not_repeat_text}\n\n"
        f"Diagnostic Session ID: {locked_session.session_id}\n"
        f"Evidence Completeness: {int(locked_session.evidence_completeness * 100)}%\n"
        f"Asset: {locked_session.asset.name} ({locked_session.asset.asset_code})"
    )

    call = create_service_call_record(
        tenant=tenant,
        complaint_type="Diagnostic Recovery Escalation",
        complaint_text=complaint_body,
        customer=locked_session.customer,
        site=locked_session.site,
        asset=locked_session.asset,
        zone=zone,
        priority="normal",
        sla=sla,
        technician=technician,
        contact_name=locked_session.customer.contact_name or locked_session.customer.name,
        contact_phone=locked_session.customer.phone,
        contact_email=locked_session.customer.email,
    )

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

    _append_event(
        locked_session, EVENT_SESSION_ESCALATED, "system",
        {"reason": reason, "service_call_id": call.id, "servy_id": call.servy_id, "customer_note": customer_note}
    )

    locked_session.escalated_call = call
    return call, passport


# ---------------------------------------------------------------------------
# Public Orchestrator Operations
# ---------------------------------------------------------------------------

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

    # Explicit service-layer tenant and customer relationship validation
    if customer.tenant_id != tenant.id:
        raise ValidationError("Customer does not belong to the active tenant.")
    if asset.tenant_id != tenant.id or asset.customer_id != customer.id:
        raise ValidationError("Asset does not belong to the selected customer/tenant.")

    # Check active session limit
    active_count = DiagnosticSession.objects.filter(
        tenant=tenant, customer=customer, status__in=["ACTIVE", "WAITING_INPUT", "WAITING_VERIFY"]
    ).count()
    if active_count >= MAX_ACTIVE_SESSIONS_PER_CUSTOMER:
        raise SessionLimitExceededError("Maximum active diagnostic sessions limit reached. Please resolve or escalate open sessions.")

    # Select candidate published playbook with positive threshold + hybrid RAG
    playbook = select_playbook_for_asset(asset, complaint, tenant, customer=customer)

    # Safe intake state if no playbook available
    if not playbook:
        status_init = "NO_PLAYBOOK_AVAILABLE"
        start_node = ""
        playbook_def = {}
    else:
        status_init = "ACTIVE"
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
            status=status_init,
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
            payload={"start_node_id": start_node, "asset_id": asset.id, "has_playbook": bool(playbook)},
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

            session.facts_snapshot = _build_facts_snapshot(state)
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

        # Enforce shared mutation invariants
        _validate_mutable_session(locked_session, expected_version)

        # Reject node mutations on NO_PLAYBOOK_AVAILABLE state
        if locked_session.status == "NO_PLAYBOOK_AVAILABLE":
            raise ValueError("Cannot submit node answers for a session in NO_PLAYBOOK_AVAILABLE state. Please clarify or escalate.")

        # Enforce current node — caller cannot skip ahead or replay old nodes
        if node_id != locked_session.current_node_id:
            raise CurrentNodeMismatchError(node_id, locked_session.current_node_id)

        # Playbook version pinning guarantee
        if locked_session.playbook and locked_session.playbook.version != locked_session.playbook_version:
            raise ValueError("Playbook version pinning mismatch. Historical session cannot execute on altered version.")

        playbook_def = locked_session.playbook.definition if locked_session.playbook else {}
        nodes = playbook_def.get("nodes", {})
        node = nodes.get(node_id, {})
        node_type = node.get("node_type", NODE_OBSERVE)
        fact_key = node.get("fact_key", f"fact_{node_id}")

        # Validate answer value against schema
        schema = node.get("response_schema", RESPONSE_SHORT_TEXT)
        validated_value = _validate_answer_value(schema, value, node.get("choices", []))

        event_type = EVENT_VERIFICATION_RECORDED if node_type == NODE_VERIFY else EVENT_OBSERVATION_RECORDED

        _append_event(locked_session, event_type, actor_role, {
            "node_id": node_id,
            "fact_key": fact_key,
            "value": validated_value,
        })

        # Re-reduce events
        events = list(locked_session.events.order_by("seq_num"))
        state = reduce_session_events(events, playbook_def, locked_session.session_id)

        # Update snapshot and version
        locked_session.facts_snapshot = _build_facts_snapshot(state)
        locked_session.evidence_completeness = state.evidence_completeness
        locked_session.current_node_id = state.current_node_id
        locked_session.status = state.status
        locked_session.version += 1

        # Atomic escalation check
        if locked_session.status == "ESCALATED":
            call, passport = _execute_escalation_atomically(
                locked_session,
                reason=state.escalation_reason or "Escalated by diagnostic workflow.",
                customer=customer
            )

        # Emit SAFE_ACTION_PRESENTED with presentation token if new current node is SAFE_ACTION
        new_node = nodes.get(state.current_node_id, {})
        if new_node.get("node_type") == NODE_SAFE_ACTION and locked_session.status not in ("RESOLVED", "ESCALATED"):
            last_seq = locked_session.events.aggregate(m=Max("seq_num"))["m"] or 0
            token = _generate_presentation_token(locked_session.session_id, state.current_node_id, locked_session.version, last_seq + 1)
            _append_event(locked_session, EVENT_SAFE_ACTION_PRESENTED, "system", {
                "node_id": state.current_node_id,
                "instruction": new_node.get("instruction", ""),
                "presentation_token": token,
            })

        locked_session.save(update_fields=["facts_snapshot", "evidence_completeness", "current_node_id", "status", "version", "escalated_call"])

    return locked_session, get_session_view_data(locked_session, customer)


def confirm_safe_action(
    session: DiagnosticSession,
    node_id: str,
    expected_version: int,
    presentation_token: Optional[str] = None,
    customer=None
) -> Tuple[DiagnosticSession, Dict[str, Any]]:
    """Confirm that the customer completed a SAFE_ACTION procedure."""
    with transaction.atomic():
        locked_session = DiagnosticSession.objects.select_for_update().get(id=session.id)

        _validate_mutable_session(locked_session, expected_version)

        if locked_session.status == "NO_PLAYBOOK_AVAILABLE":
            raise ValueError("Cannot confirm SAFE_ACTION on a session in NO_PLAYBOOK_AVAILABLE state.")

        # Current node match
        if node_id != locked_session.current_node_id:
            raise CurrentNodeMismatchError(node_id, locked_session.current_node_id)

        # Version pinning
        if locked_session.playbook and locked_session.playbook.version != locked_session.playbook_version:
            raise ValueError("Playbook version pinning mismatch.")

        playbook_def = locked_session.playbook.definition if locked_session.playbook else {}
        nodes = playbook_def.get("nodes", {})
        node = nodes.get(node_id, {})

        if node.get("node_type") != NODE_SAFE_ACTION:
            raise ValueError(f"Node '{node_id}' is not a SAFE_ACTION.")

        # Presentation binding: Find latest presentation event for this node
        presentation_event = locked_session.events.filter(
            event_type=EVENT_SAFE_ACTION_PRESENTED,
            payload__node_id=node_id,
        ).order_by("-seq_num").first()

        if not presentation_event:
            raise PresentationRequiredError(
                f"Cannot confirm SAFE_ACTION '{node_id}' without prior presentation. "
                f"The action must be presented to the customer before it can be confirmed."
            )

        # Prevent double-confirmation after the latest presentation
        already_confirmed = locked_session.events.filter(
            event_type=EVENT_SAFE_ACTION_CONFIRMED,
            payload__node_id=node_id,
            seq_num__gt=presentation_event.seq_num
        ).exists()
        if already_confirmed:
            raise ValueError(f"SAFE_ACTION '{node_id}' has already been confirmed.")

        # Verify presentation token if provided or present on the presentation event
        stored_token = presentation_event.payload.get("presentation_token")
        if presentation_token and stored_token and presentation_token != stored_token:
            raise PresentationRequiredError("Presentation token does not match the active presentation instance.")

        # Runtime policy check
        is_allowed, reason, _ = evaluate_node_policy(locked_session, node_id, customer)
        if not is_allowed:
            raise ValueError(f"Action blocked by policy: {reason}")

        _append_event(locked_session, EVENT_SAFE_ACTION_CONFIRMED, "customer", {
            "node_id": node_id,
            "instruction": node.get("instruction", ""),
            "evidence_anchor": node.get("evidence_anchor", {}),
            "presentation_token": stored_token or "",
        })

        events = list(locked_session.events.order_by("seq_num"))
        state = reduce_session_events(events, playbook_def, locked_session.session_id)

        locked_session.current_node_id = state.current_node_id
        locked_session.status = state.status
        locked_session.evidence_completeness = state.evidence_completeness
        locked_session.version += 1

        if locked_session.status == "ESCALATED":
            _execute_escalation_atomically(
                locked_session,
                reason=state.escalation_reason or "Escalated after action confirmation.",
                customer=customer
            )

        new_node = nodes.get(state.current_node_id, {})
        if new_node.get("node_type") == NODE_SAFE_ACTION and locked_session.status not in ("RESOLVED", "ESCALATED"):
            last_seq = locked_session.events.aggregate(m=Max("seq_num"))["m"] or 0
            token = _generate_presentation_token(locked_session.session_id, state.current_node_id, locked_session.version, last_seq + 1)
            _append_event(locked_session, EVENT_SAFE_ACTION_PRESENTED, "system", {
                "node_id": state.current_node_id,
                "instruction": new_node.get("instruction", ""),
                "presentation_token": token,
            })

        locked_session.save(update_fields=["facts_snapshot", "evidence_completeness", "current_node_id", "status", "version", "escalated_call"])

    return locked_session, get_session_view_data(locked_session, customer)


def resolve_diagnostic_session(
    session: DiagnosticSession,
    expected_version: int,
    customer=None,
    summary: str = ""
) -> Tuple[DiagnosticSession, Dict[str, Any]]:
    """Mark a diagnostic session as safely RESOLVED.

    A session may become RESOLVED only if the CURRENT deterministic path reaches a verified resolution outcome.
    Historical verifications on diverged or previous branches cannot authorize resolution.
    """
    with transaction.atomic():
        locked_session = DiagnosticSession.objects.select_for_update().get(id=session.id)

        if locked_session.status in ("RESOLVED", "ESCALATED"):
            # Idempotent terminal
            return locked_session, get_session_view_data(locked_session, customer)

        _validate_mutable_session(locked_session, expected_version)

        if locked_session.status == "NO_PLAYBOOK_AVAILABLE":
            raise ValueError("Cannot resolve a session in NO_PLAYBOOK_AVAILABLE state.")

        # Check for unresolved contradictions
        facts_snapshot = locked_session.facts_snapshot or {}
        contradictions = facts_snapshot.get("contradictions", [])
        if any(not c.get("resolved") for c in contradictions):
            raise ValueError("Cannot resolve session with unresolved contradictions.")

        # Authoritative resolution check based strictly on current reduced deterministic state
        playbook_def = locked_session.playbook.definition if locked_session.playbook else {}
        nodes = playbook_def.get("nodes", {})
        events = list(locked_session.events.order_by("seq_num"))
        state = reduce_session_events(events, playbook_def, locked_session.session_id)

        # Only allow resolution if reduced state is RESOLVED, or current node explicitly has is_terminal resolution
        current_node = nodes.get(locked_session.current_node_id, {})
        is_path_resolved = (state.status == "RESOLVED") or (current_node.get("is_terminal") and current_node.get("node_type") != NODE_ESCALATE)

        if not is_path_resolved:
            raise ValueError(
                "Cannot mark session as RESOLVED: the active diagnostic path has not reached a verified resolution state. "
                "To exit an incomplete session, use ABANDON."
            )

        resolution_text = current_node.get("instruction") or "Diagnostic issue resolved through verified procedure."
        clean_summary = (summary or "").strip()[:500]
        if clean_summary:
            clean_summary = re.sub(r"</?retrieved_source\b[^>]*>", "", clean_summary)
            clean_summary = re.sub(r"UNTRUSTED_CONTENT_(?:START|END)", "", clean_summary)
            clean_summary = re.sub(r"(?im)^\s*(?:system|status|resolved|summary)\s*:\s*", "", clean_summary).strip()

        _append_event(locked_session, EVENT_SESSION_RESOLVED, "customer", {
            "summary": resolution_text,
            "customer_note": clean_summary
        })

        locked_session.status = "RESOLVED"
        locked_session.version += 1
        locked_session.save(update_fields=["status", "version"])

    return locked_session, get_session_view_data(locked_session, customer)


def clarify_diagnostic_session(
    session: DiagnosticSession,
    clarification: str,
    expected_version: int,
    customer=None
) -> Tuple[DiagnosticSession, Dict[str, Any]]:
    """Clarify symptom for a session in NO_PLAYBOOK_AVAILABLE state, safely rerunning playbook routing."""
    clarification = (clarification or "").strip()[:1000]
    # Sanitize clarification against prompt injection, delimiter echoing, and header spoofing
    clarification = re.sub(r"</?retrieved_source\b[^>]*>", "", clarification)
    clarification = re.sub(r"UNTRUSTED_CONTENT_(?:START|END)", "", clarification)
    clarification = re.sub(r"(?im)^\s*(?:system|playbook|routing|instruction)\s*:\s*", "", clarification).strip()
    if not clarification:
        raise ValueError("Clarification text cannot be empty.")

    with transaction.atomic():
        locked_session = DiagnosticSession.objects.select_for_update().get(id=session.id)
        _validate_mutable_session(locked_session, expected_version)

        if locked_session.status != "NO_PLAYBOOK_AVAILABLE":
            raise ValueError("Clarification is only available for sessions without an active playbook.")

        # Append immutable clarification event
        _append_event(
            locked_session,
            EVENT_CLARIFICATION_RECORDED,
            "customer",
            {"clarification": clarification}
        )

        # Combine original complaint + clarification safely
        combined_query = f"{locked_session.complaint}\nClarification: {clarification}".strip()

        # Re-run playbook routing
        matched_playbook = select_playbook_for_asset(
            locked_session.asset,
            combined_query,
            locked_session.tenant,
            customer=locked_session.customer
        )

        if matched_playbook:
            locked_session.playbook = matched_playbook
            locked_session.playbook_version = matched_playbook.version
            playbook_def = matched_playbook.definition or {}
            start_node = playbook_def.get("start_node_id", "start")
            locked_session.current_node_id = start_node
            locked_session.status = "ACTIVE"

            _append_event(
                locked_session,
                EVENT_PLAYBOOK_SELECTED,
                "system",
                {"playbook_id": matched_playbook.id, "name": matched_playbook.name, "version": matched_playbook.version}
            )
            _append_event(
                locked_session,
                EVENT_OBSERVATION_REQUESTED,
                "system",
                {"node_id": start_node}
            )

            events = list(locked_session.events.order_by("seq_num"))
            state = reduce_session_events(events, playbook_def, locked_session.session_id)
            locked_session.facts_snapshot = _build_facts_snapshot(state)
            locked_session.evidence_completeness = state.evidence_completeness
            locked_session.current_node_id = state.current_node_id
            locked_session.status = state.status
        else:
            locked_session.status = "NO_PLAYBOOK_AVAILABLE"
            locked_session.current_node_id = ""

        locked_session.version += 1
        locked_session.save(update_fields=["playbook", "playbook_version", "facts_snapshot", "evidence_completeness", "current_node_id", "status", "version"])

    return locked_session, get_session_view_data(locked_session, customer)


def resolve_contradiction(
    session: DiagnosticSession,
    fact_key: str,
    value: Any,
    expected_version: int,
    customer=None
) -> Tuple[DiagnosticSession, Dict[str, Any]]:
    """Explicitly resolve a contradictory fact by authoritative customer clarification."""
    with transaction.atomic():
        locked_session = DiagnosticSession.objects.select_for_update().get(id=session.id)
        _validate_mutable_session(locked_session, expected_version)

        facts_snapshot = locked_session.facts_snapshot or {}
        contradictions = facts_snapshot.get("contradictions", [])
        matching_c = next((c for c in contradictions if c.get("fact_key") == fact_key and not c.get("resolved")), None)

        if not matching_c:
            raise ValueError(f"No unresolved contradiction found for fact key '{fact_key}'.")

        # Record contradiction resolved event
        _append_event(
            locked_session,
            EVENT_CONTRADICTION_RESOLVED,
            "customer",
            {
                "fact_key": fact_key,
                "value": value,
                "earlier_value": matching_c.get("earlier"),
                "new_value": matching_c.get("new"),
                "resolution_note": f"Authoritative customer clarification: set '{fact_key}' to '{value}'.",
            }
        )

        # Re-reduce events to update authoritative facts & state
        playbook_def = locked_session.playbook.definition if locked_session.playbook else {}
        events = list(locked_session.events.order_by("seq_num"))
        state = reduce_session_events(events, playbook_def, locked_session.session_id)

        locked_session.facts_snapshot = _build_facts_snapshot(state)
        locked_session.evidence_completeness = state.evidence_completeness
        locked_session.version += 1
        locked_session.save(update_fields=["facts_snapshot", "evidence_completeness", "version"])

    return locked_session, get_session_view_data(locked_session, customer)


def escalate_diagnostic_session(
    session: DiagnosticSession,
    reason: str,
    expected_version: int,
    customer=None
) -> Tuple[DiagnosticSession, ServiceCall, RecoveryPassport]:
    """Escalate a diagnostic session to ServiceCall, generating a deterministic RecoveryPassport."""
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

                call, passport = _execute_escalation_atomically(
                    locked_session,
                    reason="Customer requested escalation to technician.",
                    customer=customer,
                    customer_note=reason
                )

                locked_session.status = "ESCALATED"
                locked_session.version += 1
                locked_session.save(update_fields=["status", "escalated_call", "version"])

                return locked_session, call, passport

        except (IntegrityError, OperationalError):
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def get_session_view_data(session: DiagnosticSession, customer=None) -> Dict[str, Any]:
    """Compile the authoritative client-facing view data for a DiagnosticSession."""
    playbook_def = session.playbook.definition if session.playbook else {}
    nodes = playbook_def.get("nodes", {})

    node_data = None
    blocked_reason = None
    can_escalate = True
    can_ask_quick_ai = True
    can_clarify = False

    # Extract contradictions
    facts_snapshot = session.facts_snapshot or {}
    contradictions = facts_snapshot.get("contradictions", [])
    unresolved_contradictions = [
        {"fact_key": c["fact_key"], "earlier_value": c.get("earlier"), "new_value": c.get("new")}
        for c in contradictions if not c.get("resolved")
    ]
    can_resolve_contradiction = len(unresolved_contradictions) > 0

    if session.status == "NO_PLAYBOOK_AVAILABLE":
        can_clarify = True
        node_data = None
        blocked_reason = "No confident diagnostic playbook matches the reported symptom. You may clarify symptoms, query Quick AI, or request an engineer service call."

    elif session.status not in ("RESOLVED", "ESCALATED", "ABANDONED") and session.current_node_id in nodes:
        allowed, reason, sanitized = evaluate_node_policy(session, session.current_node_id, customer)
        if allowed:
            node_data = sanitized
            # If current node is SAFE_ACTION, attach active presentation_token
            if sanitized.get("node_type") == NODE_SAFE_ACTION:
                pres_event = session.events.filter(
                    event_type=EVENT_SAFE_ACTION_PRESENTED,
                    payload__node_id=session.current_node_id
                ).order_by("-seq_num").first()
                if pres_event and pres_event.payload.get("presentation_token"):
                    node_data["presentation_token"] = pres_event.payload["presentation_token"]
        else:
            node_data = None
            blocked_reason = f"Diagnostic action blocked by safety policy ({reason}). Escalation available."

    passport_data = None
    rp = getattr(session, "recovery_passport", None)
    if session.status == "ESCALATED" and rp:
        passport_data = {
            "id": rp.id,
            "structured_data": rp.structured_data,
            "do_not_repeat_items": rp.do_not_repeat_items,
            "evidence_completeness": rp.evidence_completeness,
            "created_at": rp.created_at.isoformat() if getattr(rp, "created_at", None) else None,
        }

    view_data = {
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
        "current_node": node_data if session.status not in ("RESOLVED", "ESCALATED", "ABANDONED") else None,
        "can_escalate": can_escalate,
        "can_ask_quick_ai": can_ask_quick_ai,
        "can_clarify": can_clarify,
        "can_resolve_contradiction": can_resolve_contradiction,
        "unresolved_contradictions": unresolved_contradictions,
        "blocked_reason": blocked_reason,
        "escalated_call": {
            "id": session.escalated_call.id,
            "servy_id": session.escalated_call.servy_id,
            "status": session.escalated_call.status,
        } if getattr(session, "escalated_call", None) else None,
        "recovery_passport": passport_data,
    }
    return _json_safe(view_data)


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
