"""
Runtime Policy Gateway for Servy Zero-Repeat Diagnostic Recovery.

Enforces zero-tolerance invariants before any diagnostic node is presented to the customer.
If validation fails:
  - SAFE_ACTION is blocked.
  - Falls back to OBSERVE or ESCALATE.
"""

from typing import Any, Dict, Optional, Tuple
from core.models import Asset, DiagnosticPlaybook, DiagnosticSession
from .evidence import verify_evidence_anchor
from .schemas import (
    NODE_ESCALATE,
    NODE_OBSERVE,
    NODE_SAFE_ACTION,
    NODE_VERIFY,
    SAFETY_RED,
)


def evaluate_node_policy(
    session: DiagnosticSession,
    node_id: str,
    customer=None
) -> Tuple[bool, str, Dict[str, Any]]:
    """Evaluate whether the given node is authorized and safe to present to the customer.

    Returns (is_allowed, reason, sanitized_node_definition).
    """
    # Invariant 1: Session non-terminal
    if session.status in ("RESOLVED", "ESCALATED", "ABANDONED"):
        return False, f"SESSION_TERMINAL_{session.status}", {}

    # Invariant 2: Tenant isolation
    if not session.tenant or not session.tenant.is_active:
        return False, "INACTIVE_OR_INVALID_TENANT", {}

    # Invariant 3: Customer isolation
    if customer and session.customer_id != customer.id:
        return False, "CROSS_CUSTOMER_VIOLATION", {}

    # Invariant 4: Asset customer match
    asset = session.asset
    if not asset or (customer and asset.customer_id != customer.id):
        return False, "ASSET_CUSTOMER_MISMATCH", {}

    # Invariant 5: Playbook published status and version pinning
    playbook = session.playbook
    if not playbook or playbook.status != "PUBLISHED":
        return False, "PLAYBOOK_NOT_PUBLISHED", {}
    if session.playbook_version and playbook.version != session.playbook_version:
        return False, "PLAYBOOK_VERSION_MISMATCH", {}

    definition = playbook.definition or {}
    nodes = definition.get("nodes", {})

    if node_id not in nodes:
        return False, f"NODE_NOT_IN_PLAYBOOK: {node_id}", {}

    node = nodes[node_id]
    node_type = node.get("node_type")

    # Invariant 6: Node type check
    if node_type not in (NODE_OBSERVE, NODE_SAFE_ACTION, NODE_VERIFY, NODE_ESCALATE):
        return False, f"ILLEGAL_NODE_TYPE: {node_type}", {}

    # Invariant 7: Contradiction block for SAFE_ACTION
    facts_snapshot = session.facts_snapshot or {}
    contradictions = facts_snapshot.get("contradictions", [])
    unresolved_contradictions = [c for c in contradictions if not c.get("resolved")]
    if node_type == NODE_SAFE_ACTION and unresolved_contradictions:
        return False, "UNRESOLVED_CONTRADICTIONS_BLOCK_ACTION", {}

    # Invariant 8: SAFE_ACTION Strict Safety & Evidence verification
    if node_type == NODE_SAFE_ACTION:
        safety_class = node.get("safety_class")
        if safety_class == SAFETY_RED:
            return False, "RED_SAFETY_CLASS_FORBIDDEN", {}

        # CRITICAL FIX 4: ALL customer repair actions strictly require valid evidence anchor
        evidence_anchor = node.get("evidence_anchor")
        if not evidence_anchor:
            return False, "NO_EVIDENCE_NO_REPAIR_INSTRUCTION", {}

        valid_anchor, anchor_reason, doc = verify_evidence_anchor(
            evidence_anchor=evidence_anchor,
            tenant_id=session.tenant_id,
            customer=customer,
            asset=asset
        )
        if not valid_anchor:
            return False, f"EVIDENCE_INVALID_{anchor_reason}", {}

    # Sanitize node for customer presentation (no internal chunk IDs or raw scores)
    sanitized = {
        "node_id": node_id,
        "node_type": node_type,
        "question": node.get("question", ""),
        "instruction": node.get("instruction", ""),
        "response_schema": node.get("response_schema", ""),
        "fact_key": node.get("fact_key", ""),
        "choices": node.get("choices", []),
        "safety_class": node.get("safety_class", "GREEN"),
        "is_terminal": node.get("is_terminal", False),
    }

    # Add safe source citation for UI
    if node_type == NODE_SAFE_ACTION and node.get("evidence_anchor"):
        anchor = node["evidence_anchor"]
        sanitized["source_citation"] = {
            "heading": anchor.get("heading", "Official Troubleshooting Procedure"),
            "document_title": anchor.get("title") or "Approved Operator Manual",
            "version": anchor.get("version", "1.0"),
        }

    return True, "ALLOWED", sanitized
