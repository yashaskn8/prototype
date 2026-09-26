"""
Deterministic Recovery Passport Generator for Servy Zero-Repeat.

Zero LLM inside the authoritative passport.
Constructed purely from the immutable DiagnosticEvent ledger and deterministic facts.

Hard Invariants:
  - Strict chronological replay: SAFE_ACTION_CONFIRMED is valid ONLY IF a matching
    SAFE_ACTION_PRESENTED occurred strictly prior in the ordered event sequence (CRITICAL FIX 6).
  - Explicit separation between authoritative System Escalation Reason and non-authoritative
    Customer Requested Reason (HIGH FIX 19).
"""

from typing import Any, Dict, List
from core.models import DiagnosticSession, RecoveryPassport, KnowledgeDocument
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

    # CRITICAL FIX 6: Strictly single-pass chronological event replay
    # A confirmed action is ONLY valid if its presentation occurred earlier in the sequence.
    seen_presented = set()
    completed_actions = []
    do_not_repeat = []
    do_not_repeat_provenance = []
    verifications = []
    system_escalation_reason = None
    customer_requested_reason = None

    for ev in events:
        event_type = ev.event_type
        p = ev.payload or {}

        if event_type == EVENT_SAFE_ACTION_PRESENTED:
            node_id = p.get("node_id")
            if node_id:
                seen_presented.add(node_id)

        elif event_type == EVENT_SAFE_ACTION_CONFIRMED:
            node_id = p.get("node_id", "")
            instruction = p.get("instruction", "")

            # Causality check: must have been presented strictly prior in this event stream
            if node_id and node_id in seen_presented:
                anchor = p.get("evidence_anchor", {})
                evidence_status = "NO_ANCHOR"
                if anchor and isinstance(anchor, dict):
                    doc_id = anchor.get("document_id")
                    anchor_checksum = anchor.get("checksum_sha256")
                    if doc_id:
                        doc = KnowledgeDocument.objects.filter(id=doc_id, tenant_id=session.tenant_id).first()
                        if not doc:
                            evidence_status = "DOCUMENT_REMOVED"
                        elif anchor_checksum and doc.checksum_sha256 != anchor_checksum:
                            evidence_status = "DOCUMENT_MODIFIED_SINCE_CONFIRMATION"
                        else:
                            evidence_status = "VERIFIED_FRESH"

                completed_actions.append({
                    "event_id": getattr(ev, "id", None),
                    "seq_num": getattr(ev, "seq_num", None),
                    "node_id": node_id,
                    "instruction": instruction,
                    "confirmed_at": str(ev.created_at),
                    "evidence_anchor": anchor,
                    "evidence_status": evidence_status,
                    "provenance": "DIAGNOSTIC_EVENT_LEDGER",
                })
                if instruction:
                    item_text = f"Action '{instruction[:80]}' verified completed by customer."
                    do_not_repeat.append(item_text)
                    do_not_repeat_provenance.append({
                        "item": item_text,
                        "event_id": getattr(ev, "id", None),
                        "seq_num": getattr(ev, "seq_num", None),
                        "provenance": "DIAGNOSTIC_EVENT_LEDGER",
                    })

        elif event_type == EVENT_VERIFICATION_RECORDED:
            verifications.append({
                "event_id": getattr(ev, "id", None),
                "seq_num": getattr(ev, "seq_num", None),
                "fact_key": p.get("fact_key"),
                "value": p.get("value"),
                "recorded_at": str(ev.created_at),
                "provenance": "DIAGNOSTIC_EVENT_LEDGER",
            })
            if p.get("value") is False or str(p.get("value")).lower() == "false":
                item_text = f"Check '{p.get('fact_key')}' failed to resolve the issue after action."
                do_not_repeat.append(item_text)
                do_not_repeat_provenance.append({
                    "item": item_text,
                    "event_id": getattr(ev, "id", None),
                    "seq_num": getattr(ev, "seq_num", None),
                    "provenance": "DIAGNOSTIC_EVENT_LEDGER",
                })

        elif event_type == EVENT_SESSION_ESCALATED:
            reason = p.get("reason", "")
            actor = getattr(ev, "actor_role", "system")
            if actor == "customer":
                customer_requested_reason = reason
            else:
                system_escalation_reason = reason

    contradictions = facts_snapshot.get("contradictions", [])

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
        "do_not_repeat_provenance": do_not_repeat_provenance,
    }

    return {
        "structured_data": structured_data,
        "do_not_repeat_items": do_not_repeat,
        "evidence_completeness": session.evidence_completeness,
    }
