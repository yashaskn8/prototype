"""Local-only ticket workspace; never writes to Servy's ServiceCall tables."""

import ipaddress
import json
import sqlite3
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
from core.security import ADMIN_ROLES, CUSTOMER_ROLES, TECH_ROLES, require_api_context
from core.services.customer_isolation import (
    get_customer_data_mode,
    is_synthetic_asset,
    validate_customer_serving_policy,
)

MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024
MAX_ATTACHMENTS = 10
MAX_DESCRIPTION_CHARS = 1_000_000
TICKET_TYPES = {
    "Breakdown Service", "Installation / Demo", "Service",
    "Preventive Maintenance (PM)", "AMC",
}
SEVERITIES = {"", "Low", "Medium", "High", "Critical"}
MACHINE_STATUSES = {"", "Running", "Intermittent", "Stopped", "Unknown"}
COMPLAINT_TYPES = {
    "Cooling / performance", "Electrical", "Noise / vibration", "Leakage",
    "Installation / commissioning", "Preventive maintenance", "Other",
}
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


def _open_local_store():
    path = Path(settings.SERVY_LOCAL_TICKET_DB_PATH).expanduser().resolve()
    for database in settings.DATABASES.values():
        if "sqlite" in str(database.get("ENGINE", "")).lower():
            configured_database = Path(database["NAME"]).expanduser().resolve()
            if configured_database == path:
                raise RuntimeError("The local ticket workspace cannot share the Servy application database file.")
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS local_tickets (
            id TEXT PRIMARY KEY,
            tenant_id INTEGER NOT NULL,
            customer_id INTEGER NOT NULL,
            customer_name TEXT NOT NULL,
            site_id INTEGER NOT NULL,
            site_name TEXT NOT NULL,
            asset_id INTEGER NOT NULL,
            asset_name TEXT NOT NULL,
            asset_code TEXT NOT NULL,
            call_type TEXT NOT NULL,
            complaint_type TEXT NOT NULL,
            complaint_text TEXT NOT NULL,
            contact_name TEXT NOT NULL,
            severity TEXT NOT NULL,
            priority TEXT NOT NULL,
            machine_status TEXT NOT NULL,
            status TEXT NOT NULL,
            technician_notes TEXT NOT NULL,
            source_conversation_id TEXT NOT NULL,
            created_by TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS local_tickets_tenant_created
            ON local_tickets (tenant_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS local_tickets_customer_created
            ON local_tickets (tenant_id, customer_id, created_at DESC);
        CREATE TABLE IF NOT EXISTS local_ticket_attachments (
            id TEXT PRIMARY KEY,
            ticket_id TEXT NOT NULL,
            tenant_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            content_type TEXT NOT NULL,
            size INTEGER NOT NULL,
            content BLOB NOT NULL,
            FOREIGN KEY(ticket_id) REFERENCES local_tickets(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS local_ticket_attachments_ticket
            ON local_ticket_attachments (ticket_id);
    """)
    return connection


def _attachments_for(connection, ticket_id):
    return [
        {
            "id": row["id"], "name": row["name"], "type": row["content_type"], "size": row["size"],
        }
        for row in connection.execute(
            "SELECT id, name, content_type, size FROM local_ticket_attachments WHERE ticket_id = ? ORDER BY rowid",
            (ticket_id,),
        )
    ]


def _serialize_ticket(connection, row, include_description=True):
    serialized = {
        "id": row["id"],
        "tenantId": str(row["tenant_id"]),
        "customerId": str(row["customer_id"]),
        "customerName": row["customer_name"],
        "siteId": str(row["site_id"]),
        "siteName": row["site_name"],
        "assetId": str(row["asset_id"]),
        "assetName": row["asset_name"],
        "assetCode": row["asset_code"],
        "callType": row["call_type"],
        "complaintType": row["complaint_type"],
        "contactName": row["contact_name"],
        "severity": row["severity"],
        "priority": row["priority"],
        "machineStatus": row["machine_status"],
        "status": row["status"],
        "technicianNotes": row["technician_notes"],
        "sourceConversationId": row["source_conversation_id"],
        "createdBy": row["created_by"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
        "attachments": _attachments_for(connection, row["id"]),
    }
    if include_description:
        serialized["complaintText"] = row["complaint_text"]
    return serialized


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
            sql = "SELECT * FROM local_tickets WHERE tenant_id = ?"
            params = [context["tenant"].id]
            if customer_id is not None:
                sql += " AND customer_id = ?"
                params.append(customer_id)
            sql += " ORDER BY created_at DESC"
            rows = connection.execute(sql, params).fetchall()
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
