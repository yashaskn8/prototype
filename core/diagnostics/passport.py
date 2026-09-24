"""
Deterministic Recovery Passport Generator for Servy Zero-Repeat.

Zero LLM inside the authoritative passport.
Constructed purely from the immutable DiagnosticEvent ledger and deterministic facts.

DEFECT 12 FIX: Verify EVENT_SAFE_ACTION_PRESENTED preceded EVENT_SAFE_ACTION_CONFIRMED
in the ledger before adding to do_not_repeat_items.

DEFECT 14 FIX: Separate system_escalation_reason from customer_requested_reason.
"""

from typing import Any, Dict, List
from core.models import DiagnosticSession, RecoveryPassport
from .schemas import (
    EVENT_SAFE_ACTION_CONFIRMED,
    EVENT_SAFE_ACTION_PRESENTED,
    EVENT_SESSION_ESCALATED,
    EVENT_VERIFICATION_RECORDED,
    SOURCE_CUSTOMER_VERIFIED,
)


def build_recovery_passport(session: DiagnosticSession) -> Dict[str, Any]:
    """Build a deterministic RecoveryPassport structure from a DiagnosticSession."""
    asset = session.asset
    events = list(session.events.order_by("seq_num"))
    facts_snapshot = session.facts_snapshot or {}
    facts = facts_snapshot.get("facts", {})

    # Extract verified observations
    verified_observations = []
    unresolved_observations = []
    for k, f in facts.items():
        if k == "complaint":
            continue
        val = f.get("value")
        source = f.get("source")
        if f.get("verified") or source == SOURCE_CUSTOMER_VERIFIED:
            verified_observations.append({
                "fact_key": k,
                "value": val,
                "source": source,
                "updated_at": f.get("updated_at"),
            })
        else:
            unresolved_observations.append({
                "fact_key": k,
                "value": val,
                "source": source,
            })

    # DEFECT 12 FIX: Build a set of node_ids that were PRESENTED before checking CONFIRMED
    presented_node_ids = set()
    for ev in events:
        if ev.event_type == EVENT_SAFE_ACTION_PRESENTED:
            p = ev.payload or {}
            node_id = p.get("node_id")
            if node_id:
                presented_node_ids.add(node_id)

    # Extract completed actions (strictly from EVENT_SAFE_ACTION_CONFIRMED events)
    # DEFECT 12 FIX: Only include actions where presentation preceded confirmation
    completed_actions = []
    do_not_repeat = []
    for ev in events:
        if ev.event_type == EVENT_SAFE_ACTION_CONFIRMED:
            p = ev.payload or {}
            node_id = p.get("node_id", "")
            instruction = p.get("instruction", "")

            # DEFECT 12: Only trust confirmed actions that had prior presentation
            if node_id not in presented_node_ids:
                # Defensive: action was confirmed without presentation — do NOT add to passport
                continue

            completed_actions.append({
                "node_id": node_id,
                "instruction": instruction,
                "confirmed_at": str(ev.created_at),
                "evidence_anchor": p.get("evidence_anchor", {}),
            })
            if instruction:
                do_not_repeat.append(f"Action '{instruction[:80]}' verified completed by customer.")

    # Extract verifications
    verifications = []
    for ev in events:
        if ev.event_type == EVENT_VERIFICATION_RECORDED:
            p = ev.payload or {}
            verifications.append({
                "fact_key": p.get("fact_key"),
                "value": p.get("value"),
                "recorded_at": str(ev.created_at),
            })
            if p.get("value") is False or str(p.get("value")).lower() == "false":
                do_not_repeat.append(
                    f"Check '{p.get('fact_key')}' failed to resolve the issue after action."
                )

    contradictions = facts_snapshot.get("contradictions", [])

    # DEFECT 14 FIX: Extract escalation reasons separated by source
    system_escalation_reason = None
    customer_requested_reason = None
    for ev in events:
        if ev.event_type == EVENT_SESSION_ESCALATED:
            p = ev.payload or {}
            reason = p.get("reason", "")
            actor = getattr(ev, "actor_role", "system")
            if actor == "customer":
                customer_requested_reason = reason
            else:
                system_escalation_reason = reason

    structured_data = {
        "passport_version": "1.0",
        "asset": {
            "id": asset.id,
            "name": asset.name,
            "asset_code": asset.asset_code,
            "model_number": asset.model_number or "Not recorded",
            "serial_number": asset.serial_number or "Not recorded",
            "product_name": asset.product.name if asset.product else "Not recorded",
        },
        "customer": {
            "id": session.customer.id,
            "name": session.customer.name,
        },
        "complaint": session.complaint,
        "verified_observations": verified_observations,
        "unresolved_observations": unresolved_observations,
        "completed_actions": completed_actions,
        "verifications": verifications,
        "contradictions": contradictions,
        "unresolved_node": session.current_node_id,
        "evidence_completeness": session.evidence_completeness,
        "total_steps": len(events),
        "system_escalation_reason": system_escalation_reason,
        "customer_requested_reason": customer_requested_reason,
    }

    return {
        "structured_data": structured_data,
        "do_not_repeat_items": do_not_repeat,
        "evidence_completeness": session.evidence_completeness,
    }
