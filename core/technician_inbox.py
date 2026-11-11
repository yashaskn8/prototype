"""Authenticated technician inbox APIs backed by the existing ServiceCall table."""

from django.db import transaction
from django.utils import timezone
from rest_framework import permissions, status
from rest_framework.exceptions import NotFound, PermissionDenied
from rest_framework.response import Response
from rest_framework.views import APIView

from core.models import CallUpdate, CustomerConversation, KnowledgeChunk, RecoveryPassport
from core.security import TECH_ROLES, authorized_knowledge_queryset, require_api_context
from core.services.audit import audit_event
from core.services.ticket_handoff import (
    authorized_service_calls,
    customer_ticket_state,
    customer_ticket_states,
)
from core.services.retriever import is_authorized_candidate


def _authorized_context(request, *, require_technician=False):
    ctx = require_api_context(request)
    if ctx["role"] not in TECH_ROLES:
        raise PermissionDenied("Technician ticket access is not available to this account.")
    if ctx["role"] == "technician" or require_technician:
        profile = ctx.get("staff_profile")
        if (not profile or not profile.is_active or profile.role != "technician"
                or profile.tenant_id != ctx["tenant"].id or profile.user_id != request.user.id):
            raise PermissionDenied("An active technician profile is required.")
    return ctx


def _ticket_queryset(ctx):
    return authorized_service_calls(ctx)


def _ticket_summary(call, ctx, handoff_data=None):
    handoff_data = handoff_data or customer_ticket_state(call)
    state = handoff_data["handoff_state"]
    return {
        "id": call.id,
        "servy_id": call.servy_id,
        "ticket_ref": f"SRV-{call.servy_id}",
        "status": call.status,
        "handoff_state": state,
        "priority": call.priority,
        "complaint_type": call.complaint_type,
        "complaint_text": call.complaint_text,
        "created_at": call.created_at,
        "updated_at": call.updated_at,
        "acknowledged_at": handoff_data["acknowledged_at"],
        "technician_acknowledged": handoff_data["technician_acknowledged"],
        "technician_assigned": bool(call.technician_id),
        "technician_name": call.technician.full_name if call.technician_id else None,
        "customer": {"id": call.customer_id, "name": call.customer.name} if call.customer_id else None,
        "tenant": {"id": call.tenant_id, "name": call.tenant.name},
        "contact": {
            "name": call.contact_name or None,
            "phone": call.contact_phone or None,
            "email": call.contact_email or None,
        },
        "site": {"id": call.site_id, "name": call.site.name} if call.site_id else None,
        "asset": {
            "id": call.asset_id,
            "name": call.asset.name,
            "asset_code": call.asset.asset_code,
            "model_number": call.asset.model_number or None,
            "product_name": call.asset.product.name if call.asset.product_id else None,
        } if call.asset_id else None,
        "is_new": not handoff_data["technician_acknowledged"] and call.status in {"open", "assigned"},
        "can_acknowledge": (
            ctx["role"] == "technician"
            and call.status in {"open", "assigned"}
            and (not call.technician_id or call.technician_id == ctx["staff_profile"].id)
        ),
    }


def _verified_references(messages, call, ctx):
    if not call.asset_id:
        return []
    allowed_docs = authorized_knowledge_queryset(ctx)
    refs = {}
    for message in messages:
        for ref in message.source_refs if isinstance(message.source_refs, list) else []:
            if not isinstance(ref, dict):
                continue
            try:
                document_id = int(ref.get("document_id"))
                chunk_id = int(ref.get("chunk_id"))
            except (TypeError, ValueError, OverflowError):
                continue
            chunk = KnowledgeChunk.objects.select_related("document").filter(
                pk=chunk_id,
                document_id=document_id,
                tenant_id=call.tenant_id,
                document__in=allowed_docs,
            ).first()
            if not chunk or chunk.document.disable_sharing or not is_authorized_candidate(
                chunk,
                tenant=ctx["tenant"],
                customer=call.customer,
                asset=call.asset,
                include_confidential=bool(ctx.get("can_view_confidential")),
                staff_mode=True,
                backend="sparse",
                allow_failed=True,
            ):
                continue
            refs[(document_id, chunk_id)] = {
                "document_id": document_id,
                "chunk_id": chunk_id,
                "title": chunk.document.title,
                "heading": chunk.heading or "",
            }
    return list(refs.values())


def _conversation_data(call, ctx):
    conversations = CustomerConversation.objects.filter(
        tenant_id=call.tenant_id,
        customer_id=call.customer_id,
        asset_id=call.asset_id,
        linked_service_call_id=call.id,
    ).order_by("created_at", "id")
    messages = []
    for conversation in conversations:
        for message in conversation.messages.filter(sender_role__in=("customer", "assistant")).order_by("created_at", "id"):
            messages.append(message)

    transcript = [
        {
            "id": message.id,
            "sender_role": message.sender_role,
            "message_text": message.message_text,
            "created_at": message.created_at,
            "confirmation": "reported by customer" if message.sender_role == "customer" else "assistant guidance; completion not confirmed",
        }
        for message in messages
    ]

    outstanding = []
    if messages and messages[-1].sender_role == "assistant":
        contract = messages[-1].structured_contract if isinstance(messages[-1].structured_contract, dict) else {}
        presentation = contract.get("presentation") if isinstance(contract.get("presentation"), dict) else {}
        question = contract.get("clarification_question") or presentation.get("clarification_question")
        response_type = str(contract.get("response_type") or presentation.get("response_type") or "").upper()
        if question and (response_type in {"CLARIFICATION", "NEEDS_CLARIFICATION"} or contract.get("clarification_question")):
            outstanding.append(str(question)[:500])

    return transcript, outstanding, _verified_references(messages, call, ctx)


class TechnicianInboxView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        ctx = _authorized_context(request)
        calls = list(_ticket_queryset(ctx).order_by("-created_at", "-id")[:200])
        handoff_data = customer_ticket_states(calls)
        return Response({
            "tickets": [
                _ticket_summary(call, ctx, handoff_data.get(call.id))
                for call in calls
            ],
            "count": len(calls),
            "polled_at": timezone.now(),
        })


class TechnicianInboxDetailView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk):
        ctx = _authorized_context(request)
        call = _ticket_queryset(ctx).filter(pk=pk).first()
        if not call:
            raise NotFound("Ticket not found.")
        transcript, outstanding, references = _conversation_data(call, ctx)
        passport = RecoveryPassport.objects.filter(
            tenant_id=call.tenant_id,
            service_call_id=call.id,
            asset_id=call.asset_id,
        ).first()
        confirmed_steps = passport.do_not_repeat_items if passport and isinstance(passport.do_not_repeat_items, list) else []
        ticket = _ticket_summary(call, ctx)
        ticket.update({
            "conversation": transcript,
            "confirmed_troubleshooting": confirmed_steps,
            "supporting_documents": references,
            "outstanding_information": outstanding,
        })
        return Response({"ticket": ticket})


class TechnicianTicketAcknowledgeView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, pk):
        ctx = _authorized_context(request, require_technician=True)
        profile = ctx["staff_profile"]

        with transaction.atomic():
            call = _ticket_queryset(ctx).select_for_update().filter(pk=pk).first()
            if not call:
                raise NotFound("Ticket not found.")
            if call.status not in {"open", "assigned", "in_progress"}:
                return Response({"detail": "This ticket can no longer be acknowledged."}, status=status.HTTP_409_CONFLICT)
            if call.technician_id and call.technician_id != profile.id:
                return Response({"detail": "This ticket is assigned to another technician."}, status=status.HTTP_409_CONFLICT)

            existing = CallUpdate.objects.filter(
                tenant_id=call.tenant_id,
                service_call_id=call.id,
                author_id=profile.id,
                status="acknowledged",
            ).order_by("created_at", "id").first()
            if existing:
                return Response({"ticket": _ticket_summary(call, ctx), "detail": "Ticket was already acknowledged."})

            changed_fields = []
            if not call.technician_id:
                call.technician = profile
                call.status = "assigned"
                changed_fields.extend(("technician", "status"))
            if changed_fields:
                call.save(update_fields=(*changed_fields, "updated_at"))

            event = CallUpdate.objects.create(
                tenant_id=call.tenant_id,
                service_call=call,
                author=profile,
                status="acknowledged",
                note="Technician acknowledged the incoming service request.",
            )
            audit_event(
                request,
                "technician_ticket_acknowledged",
                tenant=call.tenant,
                object_type="ServiceCall",
                object_id=call.id,
                metadata={"servy_id": call.servy_id, "acknowledgement_id": event.id},
            )
            return Response({
                "ticket": _ticket_summary(call, ctx),
                "detail": "Ticket acknowledged.",
            }, status=status.HTTP_200_OK)
