"""
Pure Deterministic Event Reducer for Servy Zero-Repeat Diagnostic Recovery.

Hard Invariant:
  Same sequence of ordered DiagnosticEvents + same Playbook Definition => exactly same DiagnosticState.
  ZERO LLM inference inside the reducer.
"""

from typing import Any, Dict, List, Optional
from .schemas import (
    CompletedActionRecord,
    ContradictionRecord,
    DiagnosticFact,
    DiagnosticState,
    EVENT_COMPLAINT_RECORDED,
    EVENT_CONTRADICTION_FOUND,
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
    EVENT_VERIFICATION_REQUESTED,
    NODE_ESCALATE,
    NODE_OBSERVE,
    NODE_SAFE_ACTION,
    NODE_VERIFY,
    SOURCE_CUSTOMER_ASSERTED,
    SOURCE_CUSTOMER_VERIFIED,
)


def _evaluate_transition(transitions: List[Dict[str, Any]], value: Any, default_next: Optional[str] = None) -> Optional[str]:
    """Deterministically evaluate transitions based on an observed fact value."""
    if not transitions:
        return default_next

    for t in transitions:
        condition = t.get("condition", {})
        if "equals" in condition:
            if condition["equals"] == value:
                return t.get("next_node")
        elif "in" in condition:
            if value in condition["in"]:
                return t.get("next_node")
        elif "not_equals" in condition:
            if condition["not_equals"] != value:
                return t.get("next_node")

    # If no explicit transition matched, look for default
    for t in transitions:
        if t.get("default"):
            return t.get("next_node")

    return default_next


def reduce_session_events(
    events: List[Any],
    playbook_definition: Dict[str, Any],
    session_id: str = ""
) -> DiagnosticState:
    """Reduce an ordered list of DiagnosticEvents into the authoritative DiagnosticState."""
    nodes = playbook_definition.get("nodes", {})
    start_node_id = playbook_definition.get("start_node_id", "start")
    required_facts = playbook_definition.get("required_facts", [])

    state = DiagnosticState(
        session_id=str(session_id),
        status="ACTIVE",
        current_node_id=start_node_id,
        facts={},
        contradictions=[],
        completed_actions=[],
        evidence_completeness=0.0,
        escalation_reason=None,
        resolution_summary=None,
        step_count=0,
    )

    for event in events:
        state.step_count += 1
        event_type = getattr(event, "event_type", None) or event.get("event_type")
        payload = getattr(event, "payload", None) or event.get("payload", {})
        created_at = getattr(event, "created_at", None)
        timestamp_str = str(created_at) if created_at else ""

        if event_type == EVENT_SESSION_STARTED:
            state.status = "ACTIVE"
            initial_node = payload.get("start_node_id") or start_node_id
            if initial_node in nodes:
                state.current_node_id = initial_node

        elif event_type == EVENT_COMPLAINT_RECORDED:
            complaint_text = payload.get("complaint", "")
            state.facts["complaint"] = DiagnosticFact(
                key="complaint",
                value=complaint_text,
                source=SOURCE_CUSTOMER_ASSERTED,
                verified=False,
                updated_at=timestamp_str,
            )

        elif event_type == EVENT_OBSERVATION_REQUESTED:
            node_id = payload.get("node_id")
            if node_id and node_id in nodes:
                state.current_node_id = node_id
            state.status = "WAITING_INPUT"

        elif event_type == EVENT_OBSERVATION_RECORDED:
            fact_key = payload.get("fact_key")
            value = payload.get("value")
            node_id = payload.get("node_id") or state.current_node_id

            if fact_key:
                # Contradiction detection: check if previously asserted fact conflicts
                if fact_key in state.facts:
                    earlier_fact = state.facts[fact_key]
                    if earlier_fact.value != value:
                        state.contradictions.append(
                            ContradictionRecord(
                                fact_key=fact_key,
                                earlier_value=earlier_fact.value,
                                new_value=value,
                                reason=f"Customer previously answered '{earlier_fact.value}', but updated to '{value}'.",
                                resolved=False,
                            )
                        )

                state.facts[fact_key] = DiagnosticFact(
                    key=fact_key,
                    value=value,
                    source=SOURCE_CUSTOMER_ASSERTED,
                    verified=False,  # DEFECT 24 FIX: Observations are assertions, not verified. Only VERIFY steps set verified=True.
                    updated_at=timestamp_str,
                    node_id=node_id,
                )

            # Determine next node via transitions
            current_node = nodes.get(node_id, {})
            transitions = current_node.get("transitions", [])
            default_next = current_node.get("next_node")
            next_node = _evaluate_transition(transitions, value, default_next)

            if next_node and next_node in nodes:
                state.current_node_id = next_node
                next_node_def = nodes[next_node]
                if next_node_def.get("node_type") == NODE_ESCALATE:
                    state.status = "ESCALATED"
                    state.escalation_reason = next_node_def.get("reason", "Automatic escalation from diagnostic path.")
                elif next_node_def.get("is_terminal") and next_node_def.get("node_type") != NODE_ESCALATE:
                    state.status = "RESOLVED"
                    state.resolution_summary = next_node_def.get("instruction", "Issue resolved.")
                else:
                    state.status = "ACTIVE"

        elif event_type == EVENT_SAFE_ACTION_PRESENTED:
            node_id = payload.get("node_id")
            if node_id and node_id in nodes:
                state.current_node_id = node_id
            state.status = "ACTIVE"

        elif event_type == EVENT_SAFE_ACTION_CONFIRMED:
            node_id = payload.get("node_id") or state.current_node_id
            action_node = nodes.get(node_id, {})

            state.completed_actions.append(
                CompletedActionRecord(
                    node_id=node_id,
                    instruction=action_node.get("instruction", payload.get("instruction", "")),
                    evidence_anchor=action_node.get("evidence_anchor", payload.get("evidence_anchor", {})),
                    confirmed_at=timestamp_str,
                    safety_class=action_node.get("safety_class", "GREEN"),
                )
            )

            # Move to next verification node
            next_verify = action_node.get("next_verification_node") or action_node.get("next_node")
            if next_verify and next_verify in nodes:
                state.current_node_id = next_verify
                state.status = "WAITING_VERIFY"
            elif action_node.get("is_terminal"):
                state.status = "RESOLVED"
                state.resolution_summary = action_node.get("instruction", "Issue resolved.")

        elif event_type == EVENT_VERIFICATION_REQUESTED:
            node_id = payload.get("node_id")
            if node_id and node_id in nodes:
                state.current_node_id = node_id
            state.status = "WAITING_VERIFY"

        elif event_type == EVENT_VERIFICATION_RECORDED:
            fact_key = payload.get("fact_key")
            value = payload.get("value")
            node_id = payload.get("node_id") or state.current_node_id

            if fact_key:
                state.facts[fact_key] = DiagnosticFact(
                    key=fact_key,
                    value=value,
                    source=SOURCE_CUSTOMER_VERIFIED,
                    verified=True,
                    updated_at=timestamp_str,
                    node_id=node_id,
                )

            current_node = nodes.get(node_id, {})
            transitions = current_node.get("transitions", [])
            default_next = current_node.get("next_node")
            next_node = _evaluate_transition(transitions, value, default_next)

            if next_node and next_node in nodes:
                state.current_node_id = next_node
                next_node_def = nodes[next_node]
                if next_node_def.get("node_type") == NODE_ESCALATE:
                    state.status = "ESCALATED"
                    state.escalation_reason = next_node_def.get("reason", "Verification unsuccessful; escalated.")
                elif next_node_def.get("is_terminal"):
                    state.status = "RESOLVED"
                    state.resolution_summary = next_node_def.get("instruction", "Verification confirmed resolution.")
                else:
                    state.status = "ACTIVE"

        elif event_type == EVENT_CONTRADICTION_RESOLVED:
            fact_key = payload.get("fact_key")
            for c in state.contradictions:
                if c.fact_key == fact_key:
                    c.resolved = True
                    c.resolution_note = payload.get("resolution_note", "Resolved by explicit customer clarification.")

        elif event_type == EVENT_SESSION_RESOLVED:
            state.status = "RESOLVED"
            state.resolution_summary = payload.get("summary", "Resolved by customer.")

        elif event_type == EVENT_SESSION_ESCALATED:
            state.status = "ESCALATED"
            state.escalation_reason = payload.get("reason", "Escalated to engineering service.")

    # Calculate deterministic evidence completeness (HIGH FIX 12)
    active_node = nodes.get(state.current_node_id, {})
    branch_required = active_node.get("required_facts_for_handoff") or active_node.get("required_facts")
    if branch_required and isinstance(branch_required, list) and len(branch_required) > 0:
        captured_count = sum(1 for k in branch_required if k in state.facts and state.facts[k].value is not None)
        state.evidence_completeness = round(min(1.0, captured_count / len(branch_required)), 2)
    elif required_facts and isinstance(required_facts, list) and len(required_facts) > 0:
        captured_count = sum(1 for k in required_facts if k in state.facts and state.facts[k].value is not None)
        state.evidence_completeness = round(min(1.0, captured_count / len(required_facts)), 2)
    else:
        # Default metric based on actual diagnostic facts and completed actions
        captured = len([k for k, f in state.facts.items() if k != "complaint" and f.value is not None]) + len(state.completed_actions)
        total_steps = max(1.0, float(min(5, len(nodes) or 3)))
        state.evidence_completeness = round(min(1.0, captured / total_steps), 2)

    return state
