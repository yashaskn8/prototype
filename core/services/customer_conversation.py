"""
Customer Conversation and Diagnostic Orchestration Service.

Owns the customer-facing turn lifecycle:
- Authorization context & verified-real serving policy
- Incident association: durable 1-to-1 conversation-to-session identity (no silent session reuse)
- Idempotency & deduplication (prevents duplicate transcript entries on retry or double-click)
- Hazard detection & atomic escalation
- Grounded RAG retrieval & customer-safe structured contracts
- Message persistence: guarantees exactly ONE customer message and ONE assistant message per turn
"""

import hashlib
import logging
import re
from typing import Any, Dict, Optional
from django.core.exceptions import PermissionDenied, ValidationError
from django.db import transaction
from django.db.models import Max
from django.utils import timezone

from core.models import (
    Asset,
    Customer,
    CustomerConversation,
    CustomerInteraction,
    CustomerMessage,
    CustomerTurnIdempotency,
    DiagnosticEvent,
    DiagnosticSession,
    ServiceCall,
    Site,
    Tenant,
)
from core.diagnostics.schemas import (
    EVENT_CLARIFICATION_RECORDED,
    EVENT_OBSERVATION_RECORDED,
    EVENT_SESSION_ESCALATED,
    EVENT_SESSION_STARTED,
)
from core.services.customer_isolation import validate_customer_serving_policy
from core.services.customer_diagnostic import (
    check_for_safety_hazard,
    check_for_vague_complaint,
    build_structured_diagnostic_contract,
    record_customer_action_presentation,
)

logger = logging.getLogger(__name__)


def _compose_conversational_reply(contract, turn_text, conv_context, cause_reference, asset, tenant):
    """Add a natural, cited reply built only from the already-validated claims.

    The fixed report sections stay in the contract; the composed reply becomes
    the stored message text. Any failure keeps the existing answer unchanged.
    """
    from core.services.customer_response_composer import (
        clarification_for_ambiguous, compose_reply_acknowledgement, compose_response)
    try:
        presentation = contract.get("presentation") or {}
        response = compose_response(contract, turn_text, conv_context, cause_reference)
        if response:
            presentation["response"] = response
            contract["answer"] = response["text"]
            return
        # A short answer to our own question ("Only at startup.") with no further
        # documented detail is acknowledged on the current topic, never answered
        # with an unrelated section or a generic "which symptom?" prompt.
        equipment = str((presentation.get("equipment") or {}).get("name") or getattr(asset, "name", "") or "this unit")
        acknowledgement = compose_reply_acknowledgement(conv_context, equipment)
        if acknowledgement and presentation.get("response_type") in ("DIAGNOSTIC_GUIDANCE", "INSUFFICIENT_EVIDENCE"):
            presentation["response"] = acknowledgement
            if presentation.get("response_type") != "DIAGNOSTIC_GUIDANCE":
                presentation["message"] = acknowledgement["text"]
            contract["answer"] = acknowledgement["text"]
            return
        if presentation.get("response_type") == "INSUFFICIENT_EVIDENCE":
            clarification = clarification_for_ambiguous(
                turn_text, asset, tenant, customer=getattr(asset, "customer", None))
            if clarification:
                contract["response_type"] = presentation["response_type"] = "CLARIFICATION"
                contract["clarification_question"] = clarification["text"]
                presentation["follow_up_question"] = clarification["text"]
                presentation["message"] = "I need a little more detail to help."
                contract["answer"] = f"{presentation['message']} {clarification['text']}"
    except Exception:
        logger.exception("Conversational composition failed; keeping the structured answer")


def process_customer_diagnostic_turn(**kwargs) -> Dict[str, Any]:
    """Run one turn while recording what executed, for truthful per-turn diagnostics."""
    from core.services.rag_execution_log import capture_execution_events
    from core.services.rag_admission import rag_request
    from core.services.request_deadline import request_deadline
    with rag_request(), request_deadline(), capture_execution_events():
        result = _process_customer_diagnostic_turn(**kwargs)
    _title_from_first_question(result.get("conversation"), result.get("user_message", {}).get("message_text"))
    return result


def _title_from_first_question(conv, question):
    """Replace the default "Inquiry for ..." title once, from the first real question."""
    from core.services.conversation_context import conversation_title, is_default_title, is_greeting
    if conv is None or not is_default_title(conv.title) or is_greeting(question):
        return
    title = conversation_title(question)
    if title:
        CustomerConversation.objects.filter(pk=conv.pk, title=conv.title).update(title=title[:80])
        conv.title = title[:80]


def _attach_diagnostics(contract):
    from core.services.rag_execution_log import captured_events
    from core.services.turn_diagnostics import summarize_turn
    from core.services.request_deadline import current_deadline
    from core.services.turn_diagnostics import build_outcome
    try:
        events = captured_events()
        contract["diagnostics"] = summarize_turn(events, contract)
        contract["outcome"] = build_outcome(events, contract, current_deadline())
        from core.services.rag_metrics import record
        record(contract, events, asset_id=(contract.get("asset") or {}).get("id")
               or ((contract.get("presentation") or {}).get("equipment") or {}).get("id"))
    except Exception:
        logger.exception("Turn diagnostics could not be summarized")


def _greeting_contract(asset, session, has_previous_answer):
    name = getattr(asset, "name", None) or "your equipment"
    message = (
        f"Hello again. You can ask me to explain any of the causes above, or describe a new symptom of the {name}."
        if has_previous_answer else
        f"Hello! Tell me what the {name} is doing, such as a symptom, alarm or error code, "
        "and I'll help you work out what's going on."
    )
    return {
        "response_mode": "GREETING", "response_type": "GREETING",
        "session_id": str(session.session_id) if session else None,
        "session_status": session.status if session else "ACTIVE",
        "summary": message, "answer": message, "citations": [], "references": [],
        "customer_safe_actions": [], "check_first": [], "technician_only_actions": [],
        "clarification_question": None, "can_resolve": False, "can_escalate": True,
        "engine": "conversation-greeting",
        "presentation": {"schema_version": "customer-answer-v1", "response_type": "GREETING",
                         "sections": [], "citations": [], "supporting_claims": [], "message": message,
                         "hazard": {"severity": "none", "message": ""}},
    }


def _summary_contract(session, message, response_type="CONVERSATION_SUMMARY", reason_code=None):
    """A conversation-only reply (recap, ticket questions, certainty); no technical claims."""
    return {
        "response_mode": response_type, "response_type": response_type, "reason_code": reason_code,
        "session_id": str(session.session_id) if session else None,
        "session_status": session.status if session else "ACTIVE",
        "summary": message, "answer": message, "citations": [], "references": [],
        "customer_safe_actions": [], "check_first": [], "technician_only_actions": [],
        "clarification_question": None, "can_resolve": False, "can_escalate": True,
        "engine": "conversation-intent",
        "presentation": {"schema_version": "customer-answer-v1", "response_type": response_type,
                         "sections": [], "citations": [], "supporting_claims": [], "message": message,
                         "hazard": {"severity": "none", "message": ""}},
    }


_TECHNICIAN_DECISION = re.compile(
    r"\b(?:should|can|could|may|do|must)\s+(?:i|we)\s+"
    r"(?:call|contact|request|ask for|arrange)\s+"
    r"(?:an?\s+)?(?:authorized\s+)?"
    r"(?:service\s+)?(?:technician|engineer|service\s+team|support)\b"
    r"|\b(?:do|would)\s+(?:i|we)\s+need\s+(?:to\s+)?"
    r"(?:call|contact|request)\s+(?:an?\s+)?(?:technician|service\s+team|support)\b"
    r"|\b(?:i|we)\s+should\s+(?:call|contact|get|request|ask\s+for)\s+(?:an?\s+)?(?:authorized\s+)?"
    r"(?:service\s+)?(?:technician|engineer|service\s+team)\b",
    re.I,
)


def _technician_decision_contract(asset, session, ticketed=False):
    """Answer a service-handoff decision without misrepresenting manual evidence."""
    message = (
        "Yes. Contacting an authorized technician is appropriate while the cause "
        "has not been established. Tell them what is happening, when it started, "
        "and any alarm or code shown. Do not open the equipment or attempt internal "
        "electrical or refrigerant repairs yourself. This is not a diagnosis."
    )
    # Advice only: say truthfully whether a service request exists.
    message += (" A service request is already open for this conversation." if ticketed else
                " No service request has been created yet; I can prepare one if you would like.")
    return {
        "response_mode": "TECHNICIAN_HANDOFF",
        "response_type": "TECHNICIAN_HANDOFF",
        "session_id": str(session.session_id) if session else None,
        "session_status": session.status if session else "ACTIVE",
        "asset": {"id": asset.id, "name": asset.name, "model_number": asset.model_number}
                 if asset else {},
        "summary": message, "message": message, "answer": message,
        "citations": [], "references": [], "presentation": None,
        "customer_safe_actions": [], "check_first": [], "technician_only_actions": [],
        "clarification_question": None, "can_resolve": False, "can_escalate": True,
        "engine": "customer-technician-decision",
        "reason_code": "TECHNICIAN_CONTACT_RECOMMENDED",
    }


def _explanation_contract(interaction, session, asset, built):
    from core.services.cause_explanations import explanation_response
    equipment = {"id": asset.id, "name": asset.name, "model_number": asset.model_number} if asset else {}
    presentation = {
        "schema_version": "customer-answer-v1", "response_type": "DIAGNOSTIC_GUIDANCE",
        "equipment": equipment, "sections": built["sections"], "citations": built["citations"],
        "hazard": {"severity": "none", "message": ""}, "supporting_claims": built["supporting_claims"],
        "customer_permitted_observations": [], "customer_authorized_actions": [], "message": "",
        "technician_only_warnings": ["Physical procedures require separate customer approval or an authorized technician."],
        "known_limitations": ["No physical inspection or confirmed diagnosis."],
        "authorized_user_actions": {"can_resolve": False, "can_escalate": True},
    }
    if built.get("multi_section"):
        presentation["view_mode"] = "focused"
    response = explanation_response(equipment.get("name") or "equipment", built)
    presentation["response"] = response
    return {
        "presentation": presentation, "response_mode": "ANSWER_WITH_VERIFIED_EVIDENCE",
        "response_type": "DIAGNOSTIC_GUIDANCE", "interaction_id": interaction.id,
        "session_id": str(session.session_id) if session else None,
        "session_status": session.status if session else "ACTIVE",
        "session_version": session.version if session else 1, "asset": equipment,
        "summary": "Each previously listed cause with its documented explanation.",
        "answer": response["text"], "citations": built["citations"], "references": built["citations"],
        "customer_safe_actions": [], "check_first": [], "technician_only_actions": [],
        "stop_conditions": [], "expected_result": "", "clarification_question": None,
        "can_resolve": False, "can_escalate": True, "engine": "source-structured-cause-explanations",
        "reason_code": None,
    }


def _persist_direct_reply(*, tenant, customer, user, site, asset, conv, session, turn_text,
                          idem_record, contract, source_refs=None):
    """Store one customer and one assistant message for a turn answered without ask_rag."""
    with transaction.atomic():
        user_msg = CustomerMessage.objects.create(
            tenant=tenant, conversation=conv, sender_role="customer", message_text=turn_text)
        contract["conversation_uuid"] = str(conv.conversation_uuid)
        _attach_diagnostics(contract)
        asst_msg = CustomerMessage.objects.create(
            tenant=tenant, conversation=conv, sender_role="assistant", message_text=contract["answer"],
            structured_contract=contract, source_refs=source_refs or [])
        conv.updated_at = timezone.now()
        conv.save(update_fields=["updated_at"])
        if idem_record:
            idem_record.processing_status = "COMPLETED"
            idem_record.user_message = user_msg
            idem_record.assistant_message = asst_msg
            idem_record.response_payload = {"contract": contract}
            idem_record.updated_at = timezone.now()
            idem_record.save(update_fields=["processing_status", "user_message", "assistant_message",
                                            "response_payload", "updated_at"])
    return {
        "user_message": {"id": user_msg.id, "sender_role": "customer", "message_text": user_msg.message_text,
                         "created_at": user_msg.created_at},
        "assistant_message": {"id": asst_msg.id, "sender_role": "assistant", "message_text": asst_msg.message_text,
                              "structured_contract": contract, "created_at": asst_msg.created_at},
        "contract": contract,
        "conversation": conv,
    }


def _process_customer_diagnostic_turn(
    *,
    tenant: Tenant,
    customer: Customer,
    user: Any,
    question: str,
    site: Optional[Site] = None,
    asset: Optional[Asset] = None,
    clarification_response: Optional[str] = None,
    user_observation: Optional[str] = None,
    conversation_uuid: Optional[str] = None,
    session_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a single, atomic customer diagnostic interaction turn.

    Guarantees:
    - Enforces verified-real serving policy on the asset.
    - Conversation-to-session identity is durably preserved; foreign or mismatched identities are rejected.
    - Durable idempotency: exact-duplicate requests or repeated submissions with the same idempotency key
      return the existing response without generating duplicate transcript records.
    - Exactly ONE customer message and ONE assistant message persist per completed turn.
    - Atomic transaction boundaries: all turn mutations commit together or roll back cleanly.
    """
    from core.services.rag_execution_log import execution_stage, emit_execution_event

    clean_question = (question or "").strip()
    clean_clarification = (clarification_response or "").strip()
    clean_obs = (user_observation or "").strip()

    # Prioritize clarification response or observation as the turn input if provided
    if clean_clarification:
        turn_text = clean_clarification
    elif clean_obs:
        turn_text = clean_obs
    else:
        turn_text = clean_question

    if not turn_text:
        raise ValidationError("Message text or observation is required.")

    # 1. Enforce Asset & Site Scope and Serving Policy
    if asset:
        if site and asset.site_id and asset.site_id != site.id:
            raise ValidationError("Asset does not belong to selected site.")
        if not site and asset.site:
            site = asset.site

        # Enforce asset-level real data isolation
        validate_customer_serving_policy(tenant=tenant, customer=customer, asset=asset)

    # 2. Conversation & Session Identity Resolution (P0-4)
    conv: Optional[CustomerConversation] = None
    if conversation_uuid:
        try:
            conv = CustomerConversation.objects.select_related(
                "customer", "site", "asset", "current_diagnostic_session"
            ).get(tenant=tenant, conversation_uuid=conversation_uuid)
        except CustomerConversation.DoesNotExist:
            raise ValidationError("Conversation not found or access denied.")

        if conv.customer_id != customer.id:
            raise PermissionDenied("Customer does not own this conversation.")

        if conv.status in ("RESOLVED", "CLOSED"):
            raise ValidationError("This conversation is closed. Start a new conversation for new issues.")

        # Bind asset and site from conversation if not supplied
        if not asset and conv.asset:
            asset = conv.asset
            validate_customer_serving_policy(tenant=tenant, customer=customer, asset=asset)
        elif asset and conv.asset_id and conv.asset_id != asset.id:
            raise ValidationError("Supplied asset does not match conversation asset.")

        if not site and conv.site:
            site = conv.site
        elif site and conv.site_id and conv.site_id != site.id:
            raise ValidationError("Supplied site does not match conversation site.")

        session = conv.current_diagnostic_session
        if session_id:
            if session and str(session.session_id) != str(session_id):
                raise ValidationError("Provided session ID does not match this conversation incident.")
            elif not session:
                session = DiagnosticSession.objects.filter(
                    tenant=tenant, customer=customer, session_id=session_id
                ).first()
                if not session:
                    raise ValidationError("Diagnostic session not found or access denied.")
                conv.current_diagnostic_session = session
                conv.save(update_fields=["current_diagnostic_session", "updated_at"])
    else:
        # If conversation_uuid was not passed in the request, check for an existing active
        # in-progress conversation on the same asset (e.g. multi-turn query API tests)
        conv = None
        if asset:
            if session_id:
                conv = (
                    CustomerConversation.objects.filter(
                        tenant=tenant, customer=customer, asset=asset, current_diagnostic_session__session_id=session_id
                    )
                    .order_by("-updated_at")
                    .first()
                )
            if not conv:
                conv = (
                    CustomerConversation.objects.filter(
                        tenant=tenant, customer=customer, asset=asset, status__in=["ACTIVE", "WAITING_INPUT", "ESCALATED"]
                    )
                    .order_by("-updated_at")
                    .first()
                )

        if conv:
            session = conv.current_diagnostic_session
        else:
            # Create a new conversation and dedicated diagnostic session for this independent incident
            conv = CustomerConversation.objects.create(
                tenant=tenant,
                customer=customer,
                site=site,
                asset=asset,
                created_by=user,
                title=(clean_question or (f"Inquiry for {asset.name}" if asset else "Support Inquiry"))[:80],
                status="ACTIVE",
            )
            session = None

    # Ensure a dedicated DiagnosticSession exists for this conversation (no silent reuse of foreign sessions)
    if not session and asset:
        session = DiagnosticSession.objects.create(
            tenant=tenant,
            customer=customer,
            site=site,
            asset=asset,
            complaint=clean_question or clean_clarification or "Customer conversation started",
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
            payload={"start_node_id": "intake", "asset_id": asset.id, "channel": "conversation"},
        )
        conv.current_diagnostic_session = session
        conv.save(update_fields=["current_diagnostic_session", "updated_at"])

    # 3. Durable Idempotency Enforcement (P0-2)
    payload_raw = f"{clean_question}###{clean_clarification}###{clean_obs}###{asset.id if asset else ''}###{site.id if site else ''}"
    canonical_fingerprint = hashlib.sha256(payload_raw.encode("utf-8")).hexdigest()
    idem_record = None

    if idempotency_key:
        clean_key = str(idempotency_key).strip()
        with transaction.atomic():
            idem, created = CustomerTurnIdempotency.objects.select_for_update().get_or_create(
                tenant=tenant,
                customer=customer,
                idempotency_key=clean_key,
                defaults={
                    "conversation": conv,
                    "payload_fingerprint": canonical_fingerprint,
                    "processing_status": "PENDING",
                },
            )
            if not created:
                if idem.payload_fingerprint != canonical_fingerprint:
                    raise ValidationError("Idempotency key collision with different request payload.")

                if idem.processing_status == "COMPLETED":
                    contract = (
                        idem.response_payload.get("contract")
                        or (idem.assistant_message.structured_contract if idem.assistant_message else {})
                    )
                    return {
                        "user_message": {
                            "id": idem.user_message.id if idem.user_message else None,
                            "sender_role": "customer",
                            "message_text": idem.user_message.message_text if idem.user_message else turn_text,
                            "created_at": idem.user_message.created_at if idem.user_message else idem.created_at,
                        },
                        "assistant_message": {
                            "id": idem.assistant_message.id if idem.assistant_message else None,
                            "sender_role": "assistant",
                            "message_text": idem.assistant_message.message_text if idem.assistant_message else (contract.get("raw_answer") or ""),
                            "structured_contract": contract,
                            "created_at": idem.assistant_message.created_at if idem.assistant_message else idem.updated_at,
                        },
                        "contract": contract,
                        "conversation": conv or idem.conversation,
                        "idempotent_replay": True,
                    }
                elif idem.processing_status == "PENDING":
                    age = (timezone.now() - idem.updated_at).total_seconds()
                    if age < 30:
                        raise ValidationError("Request with this idempotency key is currently being processed.")
                    idem.conversation = conv
                    idem.updated_at = timezone.now()
                    idem.save(update_fields=["conversation", "updated_at"])
                elif idem.processing_status == "FAILED":
                    idem.processing_status = "PENDING"
                    idem.conversation = conv
                    idem.error_message = ""
                    idem.updated_at = timezone.now()
                    idem.save(update_fields=["processing_status", "conversation", "error_message", "updated_at"])
            idem_record = idem

    try:
        # Extract Attributable Multi-turn Context
        from core.services.conversation_context import extract_conversation_context, contextualize_customer_query
        with execution_stage(
            "customer_conversation_context",
            "Reconstructing this customer's authorized conversation context",
        ):
            conv_context = extract_conversation_context(
                conv, turn_text, asset=asset
            )
        emit_execution_event(
            "customer_conversation_context_summary",
            "Conversation context summary",
            "info",
            metrics={
                "candidate_count": len(
                    getattr(conv_context, "reported_symptoms", None) or []
                ),
            },
        )

        # 4. Safety Hazard Evaluation (P0-1)
        all_inputs = [clean_question, clean_clarification, clean_obs]
        if session and session.complaint:
            all_inputs.append(session.complaint)
        if conv_context.reported_symptoms:
            all_inputs.extend(conv_context.reported_symptoms)
        combined_text = " ".join(t for t in all_inputs if t)

        import sys
        _api_views = sys.modules.get("core.api_views")
        _hazard_checker = getattr(_api_views, "check_for_safety_hazard", check_for_safety_hazard) if _api_views else check_for_safety_hazard

        with execution_stage(
            "customer_safety_screening",
            "Checking current and previously reported safety conditions",
        ):
            is_hazard, hazard_warning = _hazard_checker(combined_text)
            if not is_hazard:
                for inp in all_inputs:
                    if inp:
                        is_h, h_w = _hazard_checker(inp)
                        if is_h:
                            is_hazard, hazard_warning = is_h, h_w
                            break

            if not is_hazard and session:
                facts = session.facts_snapshot or {}
                if facts.get("hazard"):
                    is_hazard = True
                    # Do not replay an older persisted warning that may contain
                    # unsafe generic power-isolation advice from an earlier version.
                    hazard_warning = (
                        "IMMEDIATE SAFETY RISK DETECTED: A previously reported safety concern still requires escalation. "
                        "Move away from the equipment and keep others clear. Do not touch the equipment, controls, "
                        "wiring or electrical switches. Do not open, operate, reset or attempt to disconnect power. "
                        "Follow the site's emergency procedure and notify the site safety lead and an authorized "
                        "service technician. Contact emergency responders if anyone is injured or there is immediate danger."
                    )
        emit_execution_event(
            "customer_safety_outcome",
            (
                "Safety condition detected; RAG retrieval will not run"
                if is_hazard else
                "No safety escalation condition detected"
            ),
            "info",
        )


        if is_hazard:
            emit_execution_event(
                "customer_rag_execution",
                "Manual retrieval skipped because safety escalation takes precedence",
                "skipped",
            )
            emit_execution_event(
                "customer_evidence_validation",
                "Evidence validation and citation assembly skipped for safety response",
                "skipped",
            )
            emit_execution_event(
                "customer_qwen_generation",
                "Local Qwen generation skipped for safety escalation",
                "skipped",
            )
            with transaction.atomic():
                user_msg = CustomerMessage.objects.create(
                    tenant=tenant,
                    conversation=conv,
                    sender_role="customer",
                    message_text=turn_text,
                )
                if session:
                    locked_session = DiagnosticSession.objects.select_for_update().get(id=session.id)
                    facts = locked_session.facts_snapshot or {}
                    facts["hazard"] = True
                    facts["hazard_warning"] = hazard_warning
                    locked_session.facts_snapshot = facts
                    locked_session.status = "ESCALATED"
                    locked_session.version += 1
                    locked_session.save(update_fields=["status", "version", "facts_snapshot", "updated_at"])

                    last_seq = locked_session.events.aggregate(m=Max("seq_num"))["m"] or 0
                    DiagnosticEvent.objects.create(
                        tenant=tenant,
                        session=locked_session,
                        seq_num=last_seq + 1,
                        event_type=EVENT_OBSERVATION_RECORDED,
                        actor_role="customer",
                        payload={"observation": turn_text, "hazard": True},
                    )
                    DiagnosticEvent.objects.create(
                        tenant=tenant,
                        session=locked_session,
                        seq_num=last_seq + 2,
                        event_type=EVENT_SESSION_ESCALATED,
                        actor_role="system",
                        payload={"reason": "Safety hazard detected in customer input", "warning": hazard_warning},
                    )

                conv.status = "ESCALATED"
                conv.updated_at = timezone.now()
                conv.save(update_fields=["status", "updated_at"])

                interaction = CustomerInteraction.objects.create(
                    tenant=tenant,
                    created_by=user,
                    customer=customer,
                    site=site,
                    asset=asset,
                    conversation=conv,
                    question=combined_text or clean_question,
                    answer=hazard_warning,
                    source_refs=[],
                )

                contract = build_structured_diagnostic_contract(
                    interaction=interaction,
                    session=session,
                    raw_answer=hazard_warning,
                    retrieved_references=[],
                    is_hazard=True,
                    hazard_warning=hazard_warning,
                )
                contract["conversation_uuid"] = str(conv.conversation_uuid)

                _attach_diagnostics(contract)
                asst_msg = CustomerMessage.objects.create(
                    tenant=tenant,
                    conversation=conv,
                    sender_role="assistant",
                    message_text=hazard_warning,
                    structured_contract=contract,
                    source_refs=[],
                )

                if idem_record:
                    idem_record.processing_status = "COMPLETED"
                    idem_record.user_message = user_msg
                    idem_record.assistant_message = asst_msg
                    idem_record.response_payload = {"contract": contract}
                    idem_record.updated_at = timezone.now()
                    idem_record.save(update_fields=["processing_status", "user_message", "assistant_message", "response_payload", "updated_at"])

            return {
                "user_message": {
                    "id": user_msg.id,
                    "sender_role": user_msg.sender_role,
                    "message_text": user_msg.message_text,
                    "created_at": user_msg.created_at,
                },
                "assistant_message": {
                    "id": asst_msg.id,
                    "sender_role": asst_msg.sender_role,
                    "message_text": asst_msg.message_text,
                    "structured_contract": contract,
                    "created_at": asst_msg.created_at,
                },
                "contract": contract,
                "conversation": conv,
            }

        # Direct service-handoff questions are decisions, not new fault-diagnosis
        # requests. Do not route them through old cause retrieval or invent citations.
        if _TECHNICIAN_DECISION.search(turn_text):
            from core.services.rag_execution_log import emit_execution_event
            for stage in ("customer_rag_execution", "customer_evidence_validation",
                          "customer_qwen_generation"):
                emit_execution_event(stage, "Technician decision answered without manual claims", "skipped")
            return _persist_direct_reply(
                tenant=tenant, customer=customer, user=user, site=site, asset=asset,
                conv=conv, session=session, turn_text=turn_text, idem_record=idem_record,
                contract=_technician_decision_contract(asset, session, ticketed=bool(getattr(conv, "linked_service_call_id", None))),
            )

        # 4b. Conversational routing that needs no new retrieval for the original question.
        from core.services.cause_explanations import (
            causes_for_request, explain_causes, previous_answer_texts)
        from core.services.conversation_context import (
            is_explain_all_request, is_greeting, split_pasted_history)
        request_text, quoted_items = split_pasted_history(turn_text, previous_answer_texts(conv_context))
        if is_greeting(request_text or turn_text):
            for stage, label in (("customer_rag_execution", "Manual retrieval skipped: greeting only"),
                                 ("customer_evidence_validation", "No technical claim to validate for a greeting"),
                                 ("customer_qwen_generation", "Local Qwen generation skipped for a greeting")):
                emit_execution_event(stage, label, "skipped")
            return _persist_direct_reply(
                tenant=tenant, customer=customer, user=user, site=site, asset=asset, conv=conv,
                session=session, turn_text=turn_text, idem_record=idem_record,
                contract=_greeting_contract(asset, session, bool(conv_context.last_supported_contract)))

        # A recap of the conversation itself needs no manual evidence.
        from core.services.conversation_context import conversation_summary_text, is_summary_request
        from core.services.customer_answer import _question_guard as _summary_guard
        if (conv_context.turn_count > 0 and is_summary_request(request_text or turn_text)
                and not _summary_guard(request_text or turn_text)):
            for stage, label in (("customer_rag_execution", "Manual retrieval skipped: conversation summary"),
                                 ("customer_evidence_validation", "No technical claim in a conversation summary"),
                                 ("customer_qwen_generation", "Local Qwen generation skipped for a summary")):
                emit_execution_event(stage, label, "skipped")
            linked = getattr(conv, "linked_service_call", None)
            return _persist_direct_reply(
                tenant=tenant, customer=customer, user=user, site=site, asset=asset, conv=conv,
                session=session, turn_text=turn_text, idem_record=idem_record,
                contract=_summary_contract(session, conversation_summary_text(
                    conv_context, f"SRV-{linked.servy_id}" if linked else None)))

        # Questions about the service request or about certainty are answered from
        # saved conversation state and the linked ticket only, never from manuals.
        from core.services.conversation_context import conversation_intent, conversation_intent_reply
        intent = conversation_intent(request_text or turn_text)
        guard = _summary_guard(request_text or turn_text) if intent else None
        # A certainty question is what the "unsupported" guard describes; the reply
        # below is itself a refusal to confirm, so no retrieval is needed. Safety
        # guards always keep precedence.
        if intent and (not guard or (intent == "certainty" and guard == "unsupported")):
            for stage, label in (("customer_rag_execution", f"Manual retrieval skipped: {intent} question"),
                                 ("customer_evidence_validation", "No technical claim in this reply"),
                                 ("customer_qwen_generation", "Local Qwen generation skipped")):
                emit_execution_event(stage, label, "skipped")
            linked = getattr(conv, "linked_service_call", None)
            ticket = ({"ref": f"SRV-{linked.servy_id}", "status": linked.status} if linked else None)
            return _persist_direct_reply(
                tenant=tenant, customer=customer, user=user, site=site, asset=asset, conv=conv,
                session=session, turn_text=turn_text, idem_record=idem_record,
                contract=_summary_contract(session, conversation_intent_reply(intent, conv_context, ticket),
                                           response_type="CONVERSATION_REPLY", reason_code=intent.upper()))

        from core.services.conversation_context import ordinal_beyond_list
        from core.services.customer_answer import asks_about_other_equipment
        beyond = ordinal_beyond_list(request_text or turn_text, conv_context) if conv_context.turn_count else None
        other_equipment = asks_about_other_equipment(request_text or turn_text, asset)
        if beyond or other_equipment:
            name = getattr(asset, "name", None) or "this equipment"
            if beyond:
                label = beyond[0]["heading"]
                label = re.sub(r"^\s*\d+[.)]\s*(?:symptom|fault)?\s*:?\s*", "", label, flags=re.I).strip().lower()
                listed_text = "; ".join(c["source_text"].rstrip(".") for c in beyond)
                message = (f"I only listed {len(beyond)} possible cause{'s' if len(beyond) != 1 else ''}"
                           + (f" for {label}" if label else "") + f": {listed_text}. "
                           "Would you like me to go through it in more detail?" if len(beyond) == 1 else
                           f"I only listed {len(beyond)} possible causes" + (f" for {label}" if label else "")
                           + f": {listed_text}.")
                reason = "ORDINAL_BEYOND_LIST"
            else:
                message = (f"The information I have is for the {name} only, and it doesn't cover other types of "
                           "equipment, so I can't apply their fault codes or instructions to this unit. "
                           f"If the {name} shows a code, note it exactly as displayed for the technician.")
                reason = "OTHER_EQUIPMENT"
            for stage, label_text in (("customer_rag_execution", "Manual retrieval skipped: conversation-only reply"),
                                      ("customer_evidence_validation", "No technical claim in this reply"),
                                      ("customer_qwen_generation", "Local Qwen generation skipped")):
                emit_execution_event(stage, label_text, "skipped")
            return _persist_direct_reply(
                tenant=tenant, customer=customer, user=user, site=site, asset=asset, conv=conv,
                session=session, turn_text=turn_text, idem_record=idem_record,
                contract=_summary_contract(session, message, response_type="CONVERSATION_REPLY",
                                           reason_code=reason))

        from core.services.customer_answer import _CAUSE_EXPLANATION_REQUEST, _question_guard
        from core.services.conversation_context import resolve_cause_reference
        from core.services.content_safety import query_is_control_abuse
        wants_all_explained = conv_context.turn_count > 0 and (
            is_explain_all_request(request_text)
            or (quoted_items and _CAUSE_EXPLANATION_REQUEST.search(request_text or "")))
        selected_reference = resolve_cause_reference(request_text, conv_context)
        wants_selected_explained = bool(
            selected_reference and not selected_reference.get("mode") and re.search(
                r"\b(?:explain|elaborate|expand|describe|tell me more|more detail|in detail)\b",
                request_text, re.I,
            )
        )
        listed = (
            [selected_reference["selected"]] if wants_selected_explained
            else causes_for_request(conv_context, quoted_items) if wants_all_explained
            else []
        )
        # Explanation retrieval must not bypass a safety or policy guard, nor the
        # original-document integrity gate (stored causes are re-gated every turn).
        if query_is_control_abuse(request_text) or _question_guard(request_text):
            listed = []
        if listed:
            from core.services.rag import original_documentation_permits
            if not original_documentation_permits(tenant=tenant, customer=customer, asset=asset):
                listed = []
        if listed:
            with transaction.atomic():
                interaction = CustomerInteraction.objects.create(
                    tenant=tenant, created_by=user, customer=customer, site=site, asset=asset,
                    conversation=conv, question=request_text or turn_text, answer="", source_refs=[])
                try:
                    with execution_stage("customer_rag_execution",
                                         "Retrieving evidence separately for each listed cause"):
                        built = explain_causes(interaction=interaction, causes=listed)
                except ValueError:
                    built = None
                if built is not None:
                    emit_execution_event("customer_qwen_generation",
                                         "Exact cited source wording used for per-cause explanations",
                                         "skipped")
                    contract = _explanation_contract(interaction, session, asset, built)
                    interaction.answer = contract["answer"]
                    interaction.source_refs = built["citations"]
                    interaction.save(update_fields=["answer", "source_refs"])
                    return _persist_direct_reply(
                        tenant=tenant, customer=customer, user=user, site=site, asset=asset, conv=conv,
                        session=session, turn_text=turn_text, idem_record=idem_record,
                        contract=contract, source_refs=built["citations"])
                # No listed cause is citable any more: answer the message normally.
                interaction.delete()

        # 5. Ambiguity & Clarification Evaluation
        # Only evaluate vague complaint at start of issue when no symptoms/history are established
        is_vague, clarif_q, clarif_opts = False, None, None
        with execution_stage(
            "customer_clarification_screening",
            "Checking whether more symptom detail is needed",
        ):
            if not conv_context.is_followup and conv_context.turn_count == 0:
                is_vague, clarif_q, clarif_opts = check_for_vague_complaint(
                    clean_question, asset=asset
                )
        if is_vague and not clean_clarification:
            emit_execution_event(
                "customer_clarification_outcome",
                "Additional symptom detail is required before retrieval",
                "info",
            )
        else:
            emit_execution_event(
                "customer_clarification_outcome",
                "Customer request is ready for evidence retrieval",
                "info",
            )
        if is_vague and not clean_clarification:
            emit_execution_event(
                "customer_rag_execution",
                "Manual retrieval skipped until the customer clarifies the symptom",
                "skipped",
            )
            emit_execution_event(
                "customer_evidence_validation",
                "Evidence validation and citation assembly skipped for clarification response",
                "skipped",
            )
            emit_execution_event(
                "customer_qwen_generation",
                "Local Qwen generation skipped until the customer clarifies the symptom",
                "skipped",
            )
            with transaction.atomic():
                user_msg = CustomerMessage.objects.create(
                    tenant=tenant,
                    conversation=conv,
                    sender_role="customer",
                    message_text=turn_text,
                )
                interaction = CustomerInteraction.objects.create(
                    tenant=tenant,
                    created_by=user,
                    customer=customer,
                    site=site,
                    asset=asset,
                    conversation=conv,
                    question=clean_question,
                    answer=clarif_q,
                    source_refs=[],
                )
                contract = build_structured_diagnostic_contract(
                    interaction=interaction,
                    session=session,
                    raw_answer=clarif_q,
                    retrieved_references=[],
                    clarification_question=clarif_q,
                    clarification_options=clarif_opts,
                )
                contract["conversation_uuid"] = str(conv.conversation_uuid)

                _attach_diagnostics(contract)
                asst_msg = CustomerMessage.objects.create(
                    tenant=tenant,
                    conversation=conv,
                    sender_role="assistant",
                    message_text=clarif_q,
                    structured_contract=contract,
                    source_refs=[],
                )
                conv.updated_at = timezone.now()
                conv.save(update_fields=["updated_at"])

                if idem_record:
                    idem_record.processing_status = "COMPLETED"
                    idem_record.user_message = user_msg
                    idem_record.assistant_message = asst_msg
                    idem_record.response_payload = {"contract": contract}
                    idem_record.updated_at = timezone.now()
                    idem_record.save(update_fields=["processing_status", "user_message", "assistant_message", "response_payload", "updated_at"])

            return {
                "user_message": {
                    "id": user_msg.id,
                    "sender_role": user_msg.sender_role,
                    "message_text": user_msg.message_text,
                    "created_at": user_msg.created_at,
                },
                "assistant_message": {
                    "id": asst_msg.id,
                    "sender_role": asst_msg.sender_role,
                    "message_text": asst_msg.message_text,
                    "structured_contract": contract,
                    "created_at": asst_msg.created_at,
                },
                "contract": contract,
                "conversation": conv,
            }

        # 6. Grounded RAG Retrieval & Safe Response Construction
        from core.services.rag import ask_rag

        # Pasted earlier answers are context, not the request: guards, intent and
        # retrieval use the customer's own words; the stored message stays verbatim.
        routed_text = request_text if (quoted_items and request_text) else turn_text
        retrieval_query = contextualize_customer_query(routed_text, conv_context, asset=asset)
        # "Explain the second cause." refers to the previous answer's listed causes:
        # search for that one cause in its own manual section.
        from core.services.conversation_context import focus_presentation_on_cause, resolve_cause_reference
        cause_reference = resolve_cause_reference(routed_text, conv_context)
        # A correction or explicit topic change ("Sorry, I meant ice, not water")
        # is answered for the new subject; the stored message keeps the exact words.
        from core.services.conversation_context import topic_shift_query
        answer_question = (topic_shift_query(routed_text) if conv_context.is_followup and not cause_reference
                           else None) or routed_text
        # A cause request without its own subject is about the current issue: answer
        # it as "what are the possible causes of <issue>?" (retrieval and claims).
        from core.services.conversation_context import canonical_issue_question
        canonical = None if cause_reference else canonical_issue_question(routed_text, conv_context)
        if canonical:
            answer_question = canonical
            retrieval_query = "; ".join(p for p in (
                getattr(asset, "name", ""), " ".join(conv_context.active_topics[:2]), conv_context.active_issue,
                "customer-safe checks and observations" if "check or observe" in canonical else "possible causes")
                if p)[:450]
        if cause_reference:
            selected = cause_reference["selected"]
            retrieval_query = f"{selected['heading']}: {selected['source_text']}".strip(": ")
        if not retrieval_query:
            combined_query = f"{clean_question} (User clarification: {clean_clarification})" if clean_clarification else clean_question
            retrieval_query = combined_query or turn_text

        with execution_stage(
            "customer_rag_execution",
            "Retrieving approved customer-visible manual evidence",
        ):
            rag_res = ask_rag(
                tenant=tenant,
                question=retrieval_query,
                original_question=answer_question,
                conversation_context=conv_context,
                asset=asset,
                customer=customer,
                include_confidential=False,
                staff_mode=False,
            )

        with transaction.atomic():
            user_msg = CustomerMessage.objects.create(
                tenant=tenant,
                conversation=conv,
                sender_role="customer",
                message_text=turn_text,
            )
            # Record clarification answer if provided
            if clean_clarification and session:
                locked_session = DiagnosticSession.objects.select_for_update().get(id=session.id)
                last_seq = locked_session.events.aggregate(m=Max("seq_num"))["m"] or 0
                DiagnosticEvent.objects.create(
                    tenant=tenant,
                    session=locked_session,
                    seq_num=last_seq + 1,
                    event_type=EVENT_CLARIFICATION_RECORDED,
                    actor_role="customer",
                    payload={"question": clean_question, "clarification": clean_clarification},
                )
                locked_session.version += 1
                locked_session.save(update_fields=["version", "updated_at"])

            interaction = CustomerInteraction.objects.create(
                tenant=tenant,
                created_by=user,
                customer=customer,
                site=site,
                asset=asset,
                conversation=conv,
                question=retrieval_query,
                answer=rag_res.get("answer", ""),
                source_refs=rag_res.get("references", []),
            )

            with execution_stage(
                "customer_evidence_validation",
                "Validating manual claims and constructing citation-backed answer",
            ):
                from core.services.rag import NO_AUTHORIZED_ORIGINAL_DOCUMENTATION
                contract = build_structured_diagnostic_contract(
                    interaction=interaction,
                    session=session,
                    raw_answer=rag_res.get("answer", ""),
                    retrieved_references=rag_res.get("references", []),
                    conversation_context=conv_context,
                    original_question=answer_question,
                    evidence_blocked=rag_res.get("reason_code") == NO_AUTHORIZED_ORIGINAL_DOCUMENTATION,
                )
            if cause_reference:
                focus_presentation_on_cause(contract, cause_reference)
            _compose_conversational_reply(contract, answer_question, conv_context, cause_reference, asset, tenant)
            from core.services.followup_suggestions import suggest
            asked = [t.get("text") or "" for t in conv_context.history_turns if t.get("role") == "customer"]
            contract["suggested_follow_ups"] = suggest(contract, asked + [turn_text])
            presentation = contract.get("presentation") or {}
            with execution_stage(
                "customer_citation_assembly",
                "Assembling source references for the customer answer",
            ):
                assembled_citations = (
                    presentation.get("citations", [])
                    or contract.get("citations", [])
                    or rag_res.get("references", [])
                    or []
                )
                supported_claims = presentation.get("supporting_claims", []) or []
            emit_execution_event(
                "customer_evidence_summary",
                "Customer answer evidence summary",
                "info",
                metrics={
                    "supported_count": len(supported_claims),
                    "citation_count": len(assembled_citations),
                },
            )
            if session:
                record_customer_action_presentation(session, contract.get("customer_safe_actions", []), tenant)

            contract["conversation_uuid"] = str(conv.conversation_uuid)

            # Enrich past resolution references with full ServiceCall details for backward compatibility
            past_res_call_ids = [
                r.get("service_call_id")
                for r in rag_res.get("references", [])
                if r.get("source_kind") == "past_resolution" and r.get("service_call_id")
            ]
            sc_map = {
                sc.id: sc
                for sc in ServiceCall.objects.filter(tenant=tenant, id__in=past_res_call_ids)
            }
            past_resolutions = []
            for r in rag_res.get("references", []):
                if r.get("source_kind") == "past_resolution":
                    sc = sc_map.get(r.get("service_call_id"))
                    servy_id = sc.servy_id if sc else r.get("service_call_id")
                    complaint_type = sc.complaint_type if sc and sc.complaint_type else (r.get("heading") or "Past Service Call")
                    resolution_text = (sc.resolution_text or sc.technician_notes or "Resolved according to standard technical procedures.") if sc else "Resolved according to standard technical procedures."
                    past_resolutions.append({
                        "service_call_id": r.get("service_call_id"),
                        "servy_id": servy_id,
                        "complaint_type": complaint_type,
                        "resolution_text": resolution_text,
                        "title": r.get("title"),
                        "reference": r.get("reference"),
                        "doc_type": r.get("doc_type"),
                        "score": r.get("score"),
                    })

            contract["past_resolutions"] = past_resolutions
            contract["retrieval_backend"] = rag_res.get("retrieval_backend", "dense_chroma")
            contract["reranker_applied"] = rag_res.get("reranker_applied", False)
            contract["reranker_model"] = rag_res.get("reranker_model")
            _attach_diagnostics(contract)

            asst_msg = CustomerMessage.objects.create(
                tenant=tenant,
                conversation=conv,
                sender_role="assistant",
                message_text=contract["answer"],
                structured_contract=contract,
                source_refs=rag_res.get("references", []),
            )

            conv.updated_at = timezone.now()
            conv.save(update_fields=["updated_at"])

            if idem_record:
                idem_record.processing_status = "COMPLETED"
                idem_record.user_message = user_msg
                idem_record.assistant_message = asst_msg
                idem_record.response_payload = {"contract": contract}
                idem_record.updated_at = timezone.now()
                idem_record.save(update_fields=["processing_status", "user_message", "assistant_message", "response_payload", "updated_at"])

        return {
            "user_message": {
                "id": user_msg.id,
                "sender_role": user_msg.sender_role,
                "message_text": user_msg.message_text,
                "created_at": user_msg.created_at,
            },
            "assistant_message": {
                "id": asst_msg.id,
                "sender_role": asst_msg.sender_role,
                "message_text": asst_msg.message_text,
                "structured_contract": contract,
                "created_at": asst_msg.created_at,
            },
            "contract": contract,
            "conversation": conv,
        }
    except Exception as exc:
        if idem_record:
            try:
                with transaction.atomic():
                    rec = CustomerTurnIdempotency.objects.select_for_update().get(id=idem_record.id)
                    rec.processing_status = "FAILED"
                    rec.error_message = str(exc)
                    rec.updated_at = timezone.now()
                    rec.save(update_fields=["processing_status", "error_message", "updated_at"])
            except Exception:
                pass
        raise

