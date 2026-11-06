"""SQLite persistence helpers shared by the local ticket API and RAG demo."""

import sqlite3
from pathlib import Path

MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024
MAX_ATTACHMENTS = 10
MAX_DESCRIPTION_CHARS = 1_000_000


def open_local_store():
    """Open the separate SQLite ticket store, rejecting the app SQLite file."""
    from django.conf import settings

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


def serialize_ticket(connection, row, include_description=True):
    attachments = [
        {"id": item["id"], "name": item["name"], "type": item["content_type"], "size": item["size"]}
        for item in connection.execute(
            "SELECT id, name, content_type, size FROM local_ticket_attachments "
            "WHERE ticket_id = ? ORDER BY rowid", (row["id"],),
        )
    ]
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
        "attachments": attachments,
    }
    if include_description:
        serialized["complaintText"] = row["complaint_text"]
    return serialized


def list_tickets(connection, tenant_id, customer_id=None):
    """Read only tickets in the caller's tenant and optional customer scope."""
    sql = "SELECT * FROM local_tickets WHERE tenant_id = ?"
    params = [tenant_id]
    if customer_id is not None:
        sql += " AND customer_id = ?"
        params.append(customer_id)
    sql += " ORDER BY created_at DESC"
    return connection.execute(sql, params).fetchall()
