"""
Diagnostic Playbook publication validator and lifecycle manager.

Hard Invariants:
  - Exactly 4 runtime node types allowed: OBSERVE, SAFE_ACTION, VERIFY, ESCALATE.
  - SAFE_ACTION must have an evidence_anchor with document_id, checksum_sha256, version.
  - SAFE_ACTION must have safety_class in {'GREEN', 'AMBER'}. RED actions are NEVER allowed in published playbooks.
  - Start node must exist and be reachable.
  - Non-terminal nodes must have valid transitions leading to existing nodes.
  - Unbounded loops or execution depth exceeding MAX_EXECUTION_DEPTH are rejected.
  - Dangling transitions or cross-tenant references fail validation.
  - Must have an escalation exit.
"""

from typing import Any, Dict, List, Tuple
from django.utils import timezone
from django.core.exceptions import ValidationError

from core.models import KnowledgeDocument
from .evidence import verify_evidence_anchor
from .schemas import (
    ALLOWED_NODE_TYPES,
    ALLOWED_RESPONSE_SCHEMAS,
    ALLOWED_SAFETY_CLASSES,
    MAX_EXECUTION_DEPTH,
    NODE_ESCALATE,
    NODE_OBSERVE,
    NODE_SAFE_ACTION,
    NODE_VERIFY,
    SAFETY_RED,
)


def validate_playbook_definition(definition: Dict[str, Any], tenant_id: int) -> Tuple[bool, List[str]]:
    """Validate a playbook definition against all structural and security invariants.

    Returns (is_valid, list_of_error_strings).
    """
    errors: List[str] = []

    if not isinstance(definition, dict):
        return False, ["Playbook definition must be a JSON object."]

    start_node_id = definition.get("start_node_id")
    if not start_node_id or not isinstance(start_node_id, str):
        errors.append("Playbook must specify a non-empty 'start_node_id'.")

    nodes = definition.get("nodes")
    if not nodes or not isinstance(nodes, dict):
        errors.append("Playbook must specify a non-empty 'nodes' dictionary.")
        return False, errors

    if start_node_id not in nodes:
        errors.append(f"Start node '{start_node_id}' does not exist in 'nodes'.")
    else:
        # HIGH FIX 21: Start node must be OBSERVE for safe diagnostic intake
        start_node_def = nodes.get(start_node_id, {})
        if start_node_def.get("node_type") != NODE_OBSERVE:
            errors.append(
                f"Start node '{start_node_id}' must be of type OBSERVE for customer safety and intake clarity. "
                f"Got '{start_node_def.get('node_type')}'."
            )

    has_escalation_node = False

    for node_id, node in nodes.items():
        if not isinstance(node, dict):
            errors.append(f"Node '{node_id}' definition must be an object.")
            continue

        node_type = node.get("node_type")
        if node_type not in ALLOWED_NODE_TYPES:
            errors.append(f"Node '{node_id}' has illegal node_type '{node_type}'. Permitted: {sorted(ALLOWED_NODE_TYPES)}.")
            continue

        if node_type == NODE_ESCALATE:
            has_escalation_node = True

        # Validation for OBSERVE and VERIFY
        if node_type in (NODE_OBSERVE, NODE_VERIFY):
            question = node.get("question")
            if not question or not isinstance(question, str):
                errors.append(f"Node '{node_id}' ({node_type}) must contain a non-empty 'question'.")

            response_schema = node.get("response_schema")
            if response_schema not in ALLOWED_RESPONSE_SCHEMAS:
                errors.append(
                    f"Node '{node_id}' has invalid response_schema '{response_schema}'. Permitted: {sorted(ALLOWED_RESPONSE_SCHEMAS)}."
                )

            fact_key = node.get("fact_key")
            if not fact_key or not isinstance(fact_key, str):
                errors.append(f"Node '{node_id}' must specify a non-empty 'fact_key'.")

        # Validation for SAFE_ACTION
        elif node_type == NODE_SAFE_ACTION:
            instruction = node.get("instruction")
            if not instruction or not isinstance(instruction, str):
                errors.append(f"SAFE_ACTION node '{node_id}' must specify a non-empty 'instruction'.")

            safety_class = node.get("safety_class")
            if safety_class not in ALLOWED_SAFETY_CLASSES:
                errors.append(
                    f"SAFE_ACTION node '{node_id}' has invalid safety_class '{safety_class}'. Permitted: {sorted(ALLOWED_SAFETY_CLASSES)}."
                )
            elif safety_class == SAFETY_RED:
                errors.append(
                    f"SAFE_ACTION node '{node_id}' has RED safety classification. RED actions must NEVER be presented to customers and cannot be published."
                )

            # CRITICAL FIX 4 & RED-TEAM AREA 1 & 2: ALL SAFE_ACTION nodes strictly require a valid,
            # exact evidence_anchor that deterministically supports the instruction.
            evidence_anchor = node.get("evidence_anchor")
            if not evidence_anchor or not isinstance(evidence_anchor, dict):
                errors.append(
                    f"SAFE_ACTION node '{node_id}' must include a valid 'evidence_anchor'. "
                    f"All customer repair instructions strictly require authoritative grounding evidence."
                )
            else:
                is_valid_anchor, anchor_reason, doc = verify_evidence_anchor(
                    evidence_anchor=evidence_anchor,
                    tenant_id=tenant_id,
                    instruction=instruction,
                )
                if not is_valid_anchor:
                    errors.append(
                        f"SAFE_ACTION node '{node_id}' evidence anchor is invalid: {anchor_reason}."
                    )

        # Transitions validation
        is_terminal = node.get("is_terminal", False) or node_type == NODE_ESCALATE
        if not is_terminal:
            next_node = node.get("next_node")
            next_verification_node = node.get("next_verification_node")
            transitions = node.get("transitions", [])

            targets = []
            if next_node:
                targets.append(next_node)
            if next_verification_node:
                targets.append(next_verification_node)
            if transitions and isinstance(transitions, list):
                for t in transitions:
                    if isinstance(t, dict) and t.get("next_node"):
                        targets.append(t["next_node"])

            if not targets:
                errors.append(f"Non-terminal node '{node_id}' has no outgoing transitions or next_node.")

            for target in targets:
                if target not in nodes:
                    errors.append(f"Node '{node_id}' has dangling transition to non-existent node '{target}'.")

    if not has_escalation_node:
        errors.append("Playbook must include at least one terminal ESCALATE node.")

    # Graph Traversal: Reachability and Maximum Depth
    visited = set()
    depth_limit_exceeded = False

    def check_depth(current_id: str, depth: int, path: set) -> None:
        nonlocal depth_limit_exceeded
        if depth > MAX_EXECUTION_DEPTH:
            depth_limit_exceeded = True
            return
        if current_id in path:
            # DEFECT 8 FIX: Cycle detected — reject publication instead of silently returning
            cycle_path = list(path) + [current_id]
            errors.append(f"Playbook contains a cycle: {' -> '.join(cycle_path)}. Cycles are not permitted.")
            return

        visited.add(current_id)
        current_node = nodes.get(current_id)
        if not current_node:
            return

        targets = []
        if current_node.get("next_node"):
            targets.append(current_node["next_node"])
        if current_node.get("next_verification_node"):
            targets.append(current_node["next_verification_node"])
        for t in current_node.get("transitions", []):
            if isinstance(t, dict) and t.get("next_node"):
                targets.append(t["next_node"])

        new_path = path | {current_id}
        for target in targets:
            if target in nodes:
                check_depth(target, depth + 1, new_path)

    if start_node_id in nodes:
        check_depth(start_node_id, 1, set())

    if depth_limit_exceeded:
        errors.append(f"Playbook execution graph exceeds maximum execution depth of {MAX_EXECUTION_DEPTH} steps.")

    unreachable_nodes = set(nodes.keys()) - visited
    if unreachable_nodes:
        errors.append(f"Playbook contains unreachable nodes: {sorted(unreachable_nodes)}.")

    return len(errors) == 0, errors


def publish_playbook(playbook, user=None) -> None:
    """Publish a DiagnosticPlaybook after enforcing all validation rules."""
    is_valid, errors = validate_playbook_definition(playbook.definition, playbook.tenant_id)
    if not is_valid:
        raise ValidationError({"definition": errors})

    playbook.status = "PUBLISHED"
    playbook.published_at = timezone.now()
    playbook.save(update_fields=["status", "published_at", "updated_at"])
