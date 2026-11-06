"""Local-only ticket workspace; never writes to Servy's ServiceCall tables."""

import ipaddress
import json
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from django.conf import settings
from django.core.exceptions import PermissionDenied
from django.http import FileResponse, Http404
from rest_framework import permissions, status
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from core.models import Asset, CustomerConversation, ServiceCall, Site
from core.local_ticket_options import COMPLAINT_TYPES, MACHINE_STATUSES, SEVERITIES, TICKET_TYPES
from core.local_ticket_store import (
    MAX_ATTACHMENTS, MAX_ATTACHMENT_BYTES, MAX_DESCRIPTION_CHARS,
    list_tickets as _list_tickets, open_local_store as _open_local_store,
    serialize_ticket as _serialize_ticket,
)
from core.security import ADMIN_ROLES, CUSTOMER_ROLES, TECH_ROLES, require_api_context
from core.services.customer_isolation import (
    get_customer_data_mode,
    is_synthetic_asset,
    validate_customer_serving_policy,
)

LOCAL_TICKET_STATUSES = {"open", "assigned", "in_progress", "marked_for_closure", "closed"}


def _ensure_loopback_debug_request(request):
    """Fail closed outside a DEBUG server reached over loopback."""
    if not settings.DEBUG:
        raise Http404
    try:
        address = ipaddress.ip_address(request.META.get("REMOTE_ADDR", ""))
    except ValueError as exc:
        raise Http404 from exc
    if not address.is_loopback:
        raise Http404


def _transcript_for(conversation):
    parts = []
    for message in conversation.messages.all().order_by("created_at", "id"):
        if message.sender_role not in ("customer", "assistant"):
            continue
        text = (message.message_text or "").strip()
        if text:
            speaker = "Customer" if message.sender_role == "customer" else "Servy"
            parts.append(f"{speaker}:\n{text}")
    return "\n\n".join(parts)


def _authorize_ticket_list(context):
    role = context["role"]
    if role in CUSTOMER_ROLES:
        try:
            validate_customer_serving_policy(tenant=context["tenant"], customer=context["customer"])
        except PermissionDenied:
            raise
        return context["customer"].id
    if role in (TECH_ROLES | ADMIN_ROLES | {"superuser"}):
        return None
    raise PermissionDenied("This account cannot access the local ticket workspace.")


class LocalTicketWorkspaceView(APIView):
    """Authenticated customer and technician API backed by a separate local SQLite file."""

    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [JSONParser, FormParser, MultiPartParser]

    def get(self, request):
        _ensure_loopback_debug_request(request)
        context = require_api_context(request)
        customer_id = _authorize_ticket_list(context)
        connection = _open_local_store()
        try:
            rows = _list_tickets(connection, context["tenant"].id, customer_id)
            return Response({"tickets": [_serialize_ticket(connection, row, include_description=False) for row in rows]})
        finally:
            connection.close()

    def post(self, request):
        _ensure_loopback_debug_request(request)
        context = require_api_context(request)
        if context["role"] not in CUSTOMER_ROLES or not context["customer"]:
            raise PermissionDenied("Only an authenticated customer may raise a customer ticket.")

        customer = context["customer"]
        tenant = context["tenant"]
        try:
            validate_customer_serving_policy(tenant=tenant, customer=customer)
        except PermissionDenied:
            raise

        try:
            submitted = json.loads(request.data.get("ticket", "{}")) if hasattr(request.data, "get") else {}
        except (TypeError, json.JSONDecodeError):
            return Response({"detail": "Ticket details are not valid JSON."}, status=status.HTTP_400_BAD_REQUEST)
        if not isinstance(submitted, dict):
            return Response({"detail": "Ticket details must be an object."}, status=status.HTTP_400_BAD_REQUEST)

        site_id = submitted.get("siteId")
        asset_id = submitted.get("assetId")
        if any(not str(value or "").isdigit() for value in (site_id, asset_id)):
            return Response({"detail": "Select a valid facility and equipment item."}, status=status.HTTP_400_BAD_REQUEST)

        site = Site.objects.filter(
            tenant=tenant, customer=customer, id=site_id,
        ).first()
        asset = Asset.objects.filter(
            tenant=tenant, customer=customer, site=site, id=asset_id, status="active",
        ).first() if site else None
        if not site or not asset:
            return Response({"detail": "Select an active equipment item from your authorized facility."}, status=status.HTTP_400_BAD_REQUEST)
        if get_customer_data_mode() == "verified_real_only" and is_synthetic_asset(asset):
            raise PermissionDenied("This equipment is not eligible for the verified customer workspace.")

        call_type = str(submitted.get("callType", "")).strip()
        complaint_type = str(submitted.get("complaintType", "")).strip()
        description = str(submitted.get("complaintText", "")).strip()
        contact_name = str(submitted.get("contactName", "")).strip()
        severity = str(submitted.get("severity", "")).strip()
        priority = str(submitted.get("priority", "normal")).strip().lower()
        machine_status = str(submitted.get("machineStatus", "")).strip()
        if call_type not in TICKET_TYPES:
            return Response({"detail": "Choose a supported ticket type."}, status=status.HTTP_400_BAD_REQUEST)
        if complaint_type not in COMPLAINT_TYPES:
            return Response({"detail": "Choose a supported complaint type."}, status=status.HTTP_400_BAD_REQUEST)
        if not description:
            return Response({"detail": "Description is required."}, status=status.HTTP_400_BAD_REQUEST)
        if len(description) > MAX_DESCRIPTION_CHARS:
            return Response({"detail": "The description is too long to save."}, status=status.HTTP_400_BAD_REQUEST)
        if len(contact_name) > 150 or severity not in SEVERITIES or priority not in {value for value, _ in ServiceCall.PRIORITIES} or machine_status not in MACHINE_STATUSES:
            return Response({"detail": "One or more ticket fields are invalid."}, status=status.HTTP_400_BAD_REQUEST)

        conversation_id = str(submitted.get("sourceConversationId") or "").strip()
        if conversation_id:
            try:
                conversation_id = str(uuid.UUID(conversation_id))
            except ValueError:
                return Response({"detail": "The source conversation ID is invalid."}, status=status.HTTP_400_BAD_REQUEST)
            conversation = CustomerConversation.objects.filter(
                tenant=tenant, customer=customer, asset=asset, conversation_uuid=conversation_id,
            ).first()
            if not conversation:
                return Response({"detail": "The source conversation is not available for this customer and equipment."}, status=status.HTTP_400_BAD_REQUEST)
            transcript = _transcript_for(conversation)
            if transcript and transcript not in description:
                description = f"{transcript}\n\nCustomer additions:\n{description}"

        attachments = request.FILES.getlist("attachments")
        if len(attachments) > MAX_ATTACHMENTS:
            return Response({"detail": f"Attach no more than {MAX_ATTACHMENTS} files."}, status=status.HTTP_400_BAD_REQUEST)
        if any(file.size > MAX_ATTACHMENT_BYTES for file in attachments):
            return Response({"detail": "Each attachment must be 25 MB or smaller."}, status=status.HTTP_400_BAD_REQUEST)

        now = datetime.now(timezone.utc).isoformat()
        ticket_id = f"LOCAL-{now[:10].replace('-', '')}-{uuid.uuid4().hex[:8].upper()}"
        record = {
            "id": ticket_id,
            "tenant_id": tenant.id,
            "customer_id": customer.id,
            "customer_name": customer.name,
            "site_id": site.id,
            "site_name": site.name,
            "asset_id": asset.id,
            "asset_name": asset.name,
            "asset_code": asset.asset_code,
            "call_type": call_type,
            "complaint_type": complaint_type,
            "complaint_text": description,
            "contact_name": contact_name,
            "severity": severity,
            "priority": priority,
            "machine_status": machine_status,
            "status": "open",
            "technician_notes": "",
            "source_conversation_id": conversation_id,
            "created_by": request.user.get_username(),
            "created_at": now,
            "updated_at": now,
        }
        connection = _open_local_store()
        try:
            with connection:
                columns = ", ".join(record.keys())
                placeholders = ", ".join("?" for _ in record)
                connection.execute(
                    f"INSERT INTO local_tickets ({columns}) VALUES ({placeholders})",
                    tuple(record.values()),
                )
                for uploaded in attachments:
                    attachment_id = uuid.uuid4().hex
                    safe_name = Path(uploaded.name.replace("\\", "/")).name[:255]
                    connection.execute(
                        "INSERT INTO local_ticket_attachments (id, ticket_id, tenant_id, name, content_type, size, content) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (attachment_id, ticket_id, tenant.id, safe_name, uploaded.content_type or "application/octet-stream", uploaded.size, uploaded.read()),
                    )
            saved = connection.execute("SELECT * FROM local_tickets WHERE id = ?", (ticket_id,)).fetchone()
            return Response({"ticket": _serialize_ticket(connection, saved)}, status=status.HTTP_201_CREATED)
        finally:
            connection.close()


class LocalTicketWorkspaceDetailView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [JSONParser]

    def _ticket_and_context(self, request, ticket_id):
        _ensure_loopback_debug_request(request)
        context = require_api_context(request)
        customer_id = _authorize_ticket_list(context)
        connection = _open_local_store()
        row = connection.execute(
            "SELECT * FROM local_tickets WHERE id = ? AND tenant_id = ?",
            (ticket_id, context["tenant"].id),
        ).fetchone()
        if not row or (customer_id is not None and int(row["customer_id"]) != int(customer_id)):
            connection.close()
            raise Http404
        return connection, context, row

    def get(self, request, ticket_id):
        connection, _, row = self._ticket_and_context(request, ticket_id)
        try:
            return Response({"ticket": _serialize_ticket(connection, row)})
        finally:
            connection.close()

    def patch(self, request, ticket_id):
        connection, context, row = self._ticket_and_context(request, ticket_id)
        try:
            if context["role"] not in (TECH_ROLES | ADMIN_ROLES | {"superuser"}):
                raise PermissionDenied("Only technician roles can update a local ticket.")
            status_value = request.data.get("status", row["status"])
            notes = request.data.get("technicianNotes", row["technician_notes"])
            if status_value not in LOCAL_TICKET_STATUSES or not isinstance(notes, str) or len(notes) > 8000:
                return Response({"detail": "Ticket status or technician notes are invalid."}, status=status.HTTP_400_BAD_REQUEST)
            now = datetime.now(timezone.utc).isoformat()
            with connection:
                connection.execute(
                    "UPDATE local_tickets SET status = ?, technician_notes = ?, updated_at = ? WHERE id = ? AND tenant_id = ?",
                    (status_value, notes, now, ticket_id, context["tenant"].id),
                )
            updated = connection.execute("SELECT * FROM local_tickets WHERE id = ?", (ticket_id,)).fetchone()
            return Response({"ticket": _serialize_ticket(connection, updated)})
        finally:
            connection.close()


class LocalTicketAttachmentView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, ticket_id, attachment_id):
        connection, _, ticket = LocalTicketWorkspaceDetailView()._ticket_and_context(request, ticket_id)
        try:
            row = connection.execute(
                "SELECT * FROM local_ticket_attachments WHERE id = ? AND ticket_id = ? AND tenant_id = ?",
                (attachment_id, ticket_id, ticket["tenant_id"]),
            ).fetchone()
            if not row:
                raise Http404
            response = FileResponse(
                BytesIO(row["content"]),
                as_attachment=True,
                filename=Path(row["name"].replace("\\", "/")).name,
                content_type=row["content_type"],
            )
            response["Content-Length"] = str(row["size"])
            return response
        finally:
            connection.close()
