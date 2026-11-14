"""Access-code gateways that serve the official Servy APIs from servy_dev.

The Customer RAG demo (8002) and the Technician demo (8003) keep their existing
access codes: the same private files, the same "Demo <code>" / "Technician <code>"
headers, loopback-only and constant-time comparison. A verified code is mapped on
the server to one dedicated synthetic user in servy_dev (dev_customer or
dev_technician); every ownership, tenant and asset check is the official one.
The access code is never a customer or tenant identifier and is never logged.
"""
import secrets
import threading
from pathlib import Path

from django.conf import settings
from django.contrib.auth.models import User
from django.core.exceptions import ImproperlyConfigured
from django.db import transaction
from django.http import FileResponse
from rest_framework import permissions, status
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import APIException, AuthenticationFailed, NotFound
from rest_framework.response import Response
from rest_framework.views import APIView

from core.api_views import CustomerConversationTicketView
from core.dev_database import DevDatabaseIsolationError, verify_isolated_database
from core.models import Asset, CallUpdate, CustomerConversation, KnowledgeDocument, ServiceCall
from core.services.call_attachments import (
    AttachmentError, authorized_attachment, serialize_attachment, store_attachments, validate_attachments)
from core.technician_inbox import TechnicianInboxDetailView, _authorized_context, _ticket_queryset

_isolation_lock = threading.Lock()
_isolation_verified = False


class GatewayUnavailable(APIException):
    status_code = 503
    default_detail = "The development database failed its isolation check; the gateway is unavailable."


def ensure_isolation():
    """Verify once per process that the live connection is the isolated dev database."""
    global _isolation_verified
    if _isolation_verified:
        return
    with _isolation_lock:
        if _isolation_verified:
            return
        try:
            verify_isolated_database(settings.SERVY_DEMO_EXPECTED_DATABASE, settings.SERVY_DEMO_EXPECTED_USER)
        except DevDatabaseIsolationError as exc:
            raise GatewayUnavailable() from exc
        _isolation_verified = True


def _read_access_code(path_setting):
    path = Path(str(getattr(settings, path_setting, "") or ""))
    if not path.is_file():
        raise ImproperlyConfigured(f"{path_setting} does not point to the demo access-code file.")
    code = path.read_text(encoding="ascii").strip()
    if len(code) < 40:
        raise ImproperlyConfigured("The demo access-code file is invalid.")
    return code


def _loopback(request):
    host = request.get_host().split(":", 1)[0]
    return request.META.get("REMOTE_ADDR") in ("127.0.0.1", "::1") and host in ("127.0.0.1", "localhost")


class _AccessCodeAuthentication(BaseAuthentication):
    scheme = ""
    username = ""
    path_setting = ""
    missing_detail = ""

    def authenticate(self, request):
        ensure_isolation()
        supplied = request.META.get("HTTP_AUTHORIZATION", "")
        if not _loopback(request):
            raise AuthenticationFailed("This demo accepts loopback requests only.")
        if not supplied:
            return None
        if not secrets.compare_digest(supplied, f"{self.scheme} {_read_access_code(self.path_setting)}"):
            raise AuthenticationFailed(self.missing_detail)
        user = User.objects.filter(username=self.username, is_active=True).first()
        if user is None:
            raise GatewayUnavailable("The synthetic development identity is missing; run seed_servy_dev.")
        return user, None

    def authenticate_header(self, request):
        return self.scheme


class CustomerAccessCodeAuthentication(_AccessCodeAuthentication):
    scheme = "Demo"
    username = "dev_customer"
    path_setting = "SERVY_DEMO_ACCESS_FILE"
    missing_detail = "Demo identity required"


class TechnicianAccessCodeAuthentication(_AccessCodeAuthentication):
    scheme = "Technician"
    username = "dev_technician"
    path_setting = "SERVY_TECHNICIAN_ACCESS_FILE"
    missing_detail = "Technician access code required or invalid."


def bind(view_class, authentication_class):
    """The official view, authenticated by the demo's access code instead of a session."""
    return view_class.as_view(authentication_classes=[authentication_class])


# --------------------------------------------------------------------------- customer
class CustomerInventoryView(APIView):
    """The synthetic customer's assets and their approved manuals (existing contract)."""
    authentication_classes = [CustomerAccessCodeAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        from core.security import require_api_context
        ctx = require_api_context(request)
        assets = Asset.objects.filter(tenant=ctx["tenant"], customer=ctx["customer"]).select_related(
            "site", "product").order_by("id")
        rows = []
        for asset in assets:
            documents = KnowledgeDocument.objects.filter(
                tenant=ctx["tenant"], asset=asset, is_rag_enabled=True, is_confidential=False).order_by("id")
            manuals = [{"id": d.id, "title": d.title, "doc_type": d.doc_type,
                        "ready": d.index_status == "INDEXED" and d.chunks.exists()} for d in documents]
            rows.append({
                # Field names follow the existing Customer RAG inventory contract.
                "id": asset.id, "name": asset.name, "code": asset.asset_code, "asset_code": asset.asset_code,
                "model": asset.model_number, "model_number": asset.model_number,
                "product": asset.product.name if asset.product_id else "",
                "site": asset.site.name if asset.site_id else "",
                "customer_name": ctx["customer"].name, "customer_id": ctx["customer"].id,
                "documents": [{"id": m["id"], "title": m["title"], "doc_type": m["doc_type"]} for m in manuals],
                "required_manuals": manuals,
                "manuals_ready": bool(manuals) and all(m["ready"] for m in manuals),
                "can_query": bool(manuals) and all(m["ready"] for m in manuals),
            })
        return Response({"assets": rows, "customer": {"name": ctx["customer"].name}})


TICKET_FIELDS = ("call_type", "complaint_type", "severity", "priority", "machine_status")


class CustomerConversationTicketGatewayView(CustomerConversationTicketView):
    """Official preview/creation, plus multipart attachments (0-10) stored in servy_dev."""
    authentication_classes = [CustomerAccessCodeAuthentication]

    def post(self, request, conversation_uuid):
        is_multipart = (request.content_type or "").startswith("multipart/form-data")
        try:
            items = validate_attachments(request.FILES.getlist("attachments")) if is_multipart else []
        except AttachmentError as exc:
            return Response({"detail": exc.detail, "code": "attachment_invalid"}, status=status.HTTP_400_BAD_REQUEST)
        if is_multipart:
            unexpected = set(request.FILES) - {"attachments"}
            if unexpected:
                return Response({"detail": "Unexpected upload field."}, status=status.HTTP_400_BAD_REQUEST)
        confirm = request.data.get("confirm_ticket_creation")
        details = [f"{name.replace('_', ' ').title()}: {' '.join(str(request.data.get(name)).split())[:120]}"
                   for name in TICKET_FIELDS if request.data.get(name)]
        note = " | ".join(details)
        # Hand the official view plain JSON-equivalent data: explicit boolean
        # confirmation (multipart sends text) and the customer's form details.
        request._full_data = {"confirm_ticket_creation": confirm is True or confirm == "true", "note": note}
        existed = CustomerConversation.objects.filter(
            conversation_uuid=conversation_uuid, linked_service_call__isnull=False).exists()
        response = super().post(request, conversation_uuid)
        if response.status_code != 200 or not isinstance(response.data, dict) or not response.data.get("ticket"):
            return response
        ticket = dict(response.data["ticket"])
        # The official view reports whether this request created the call; a
        # concurrent confirmation that lost the race is a duplicate, not a creation.
        existed = existed or response.data.get("created") is False
        call = ServiceCall.objects.get(pk=ticket["id"])
        if items and not existed:
            try:
                store_attachments(call, items, uploaded_by=request.user)
            except Exception:
                return Response({"mode": "SERVY_DEV_MYSQL", "ticket": ticket, "duplicate": False,
                                 "detail": f"Service Call {ticket.get('ticket_ref')} was created, but its attachments "
                                           "could not be saved."}, status=status.HTTP_207_MULTI_STATUS)
        ticket["attachments"] = [serialize_attachment(a) for a in call.attachments.all()]
        return Response({"mode": "SERVY_DEV_MYSQL", "ticket": ticket, "duplicate": existed,
                         "detail": response.data.get("detail")}, status=status.HTTP_200_OK if existed else status.HTTP_201_CREATED)


def _file_response(attachment):
    response = FileResponse(attachment.file.open("rb"), as_attachment=True, filename=attachment.original_name,
                            content_type=attachment.content_type)
    response["X-Content-Type-Options"] = "nosniff"
    response["Cache-Control"] = "no-store"
    return response


class CustomerAttachmentView(APIView):
    authentication_classes = [CustomerAccessCodeAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, call_id, attachment_id):
        from core.security import require_api_context
        ctx = require_api_context(request)
        mine = ServiceCall.objects.filter(tenant=ctx["tenant"], customer=ctx["customer"])
        attachment = authorized_attachment(mine, call_id, attachment_id)
        if attachment is None:
            raise NotFound("Attachment not found.")
        return _file_response(attachment)


# --------------------------------------------------------------------------- technician
class TechnicianTicketDetailGatewayView(TechnicianInboxDetailView):
    """Official ticket detail plus the full description, attachments and update history."""
    authentication_classes = [TechnicianAccessCodeAuthentication]

    def get(self, request, pk):
        response = super().get(request, pk)
        ctx = _authorized_context(request)
        call = _ticket_queryset(ctx).filter(pk=pk).first()
        ticket = dict(response.data["ticket"])
        ticket["description"] = call.technician_notes
        ticket["attachments"] = [serialize_attachment(a) for a in call.attachments.all()]
        ticket["updates"] = [{"id": u.id, "status": u.status, "note": u.note, "created_at": u.created_at,
                              "author": u.author.full_name if u.author_id else None}
                             for u in CallUpdate.objects.filter(service_call=call, tenant_id=call.tenant_id)
                             .select_related("author").order_by("created_at", "id")]
        return Response({"ticket": ticket})


class TechnicianAttachmentView(APIView):
    authentication_classes = [TechnicianAccessCodeAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk, attachment_id):
        ctx = _authorized_context(request)
        attachment = authorized_attachment(_ticket_queryset(ctx), pk, attachment_id)
        if attachment is None:
            raise NotFound("Attachment not found.")
        return _file_response(attachment)
