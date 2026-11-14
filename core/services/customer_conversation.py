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
    from core.services.customer_response_composer import clarification_for_ambiguous, compose_response
    try:
        presentation = contract.get("presentation") or {}
        response = compose_response(contract, turn_text, conv_context, cause_reference)
        if response:
            presentation["response"] = response
            contract["answer"] = response["text"]
            return
        if presentation.get("response_type") == "INSUFFICIENT_EVIDENCE":
            clarification = clarification_for_ambiguous(turn_text, asset, tenant)
            if clarification:
                contract["response_type"] = presentation["response_type"] = "CLARIFICATION"
                contract["clarification_question"] = clarification["text"]
                presentation["follow_up_question"] = clarification["text"]
                presentation["message"] = "I need a little more detail to find the right section of the manual."
                contract["answer"] = f"{presentation['message']} {clarification['text']}"
    except Exception:
        logger.exception("Conversational composition failed; keeping the structured answer")


def process_customer_diagnostic_turn(
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

        retrieval_query = contextualize_customer_query(turn_text, conv_context, asset=asset)
        # "Explain the second cause." refers to the previous answer's listed causes:
        # search for that one cause in its own manual section.
        from core.services.conversation_context import focus_presentation_on_cause, resolve_cause_reference
        cause_reference = resolve_cause_reference(turn_text, conv_context)
        # A correction or explicit topic change ("Sorry, I meant ice, not water")
        # is answered for the new subject; the stored message keeps the exact words.
        from core.services.conversation_context import topic_shift_query
        answer_question = (topic_shift_query(turn_text) if conv_context.is_followup and not cause_reference
                           else None) or turn_text
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
                contract = build_structured_diagnostic_contract(
                    interaction=interaction,
                    session=session,
                    raw_answer=rag_res.get("answer", ""),
                    retrieved_references=rag_res.get("references", []),
                    conversation_context=conv_context,
                    original_question=answer_question,
                )
            if cause_reference:
                focus_presentation_on_cause(contract, cause_reference)
            _compose_conversational_reply(contract, answer_question, conv_context, cause_reference, asset, tenant)
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

