"""Loopback-only, code-gated technician view for authorized RAG demo tickets.

This service reads and updates only the separate local SQLite ticket workspace.
It never connects to MySQL or modifies Servy operational ServiceCall records.
"""
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
from urllib.parse import quote, unquote, urlsplit

PRIVATE = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Servy" / "private_evaluations" / "servy_technician_demo_20261111_01"
ACCESS_FILE = PRIVATE / "technician_access.txt"
DB_PATH = Path(os.environ.get(
    "SERVY_LOCAL_TICKET_DB_PATH",
    str(Path.home() / "AppData" / "Local" / "Servy" / "local-ticket-workspace.sqlite3"),
)).expanduser().resolve()
PORT = 8003
HOST = "127.0.0.1"
DEMO_CREATOR = "servy-real-demo"
ALLOWED_STATUSES = {"open", "assigned", "in_progress", "marked_for_closure", "closed"}
MAX_BODY = 16000


def _open_store(db_path=DB_PATH):
    """Open an existing dedicated ticket DB without ever creating a new file."""
    path = Path(db_path).expanduser().resolve()
    if path.name.lower() != "local-ticket-workspace.sqlite3":
        raise RuntimeError("Technician demo is not pointed at the dedicated local ticket workspace.")
    if not path.is_file():
        raise FileNotFoundError("The customer RAG has not created its local ticket workspace yet.")
    uri = "file:" + quote(str(path).replace("\\", "/"), safe="/:") + "?mode=rw"
    connection = sqlite3.connect(uri, uri=True, timeout=10)
    connection.row_factory = sqlite3.Row
    tables = {row["name"] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    if not {"local_tickets", "local_ticket_attachments"}.issubset(tables):
        connection.close()
        raise RuntimeError("The existing local ticket workspace schema is incomplete.")
    required = {"id", "tenant_id", "customer_name", "site_name", "asset_name",
                "complaint_text", "status", "technician_notes", "created_by",
                "created_at", "updated_at"}
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(local_tickets)")}
    if not required.issubset(columns):
        connection.close()
        raise RuntimeError("The local ticket workspace does not match the expected schema.")
    return connection


def _serialize(connection, row, include_description=True):
    attachments = [
        {"id": item["id"], "name": item["name"], "type": item["content_type"], "size": item["size"]}
        for item in connection.execute(
            "SELECT id, name, content_type, size FROM local_ticket_attachments "
            "WHERE ticket_id = ? AND tenant_id = ? ORDER BY rowid",
            (row["id"], row["tenant_id"]),
        )
    ]
    value = {
        "id": row["id"],
        "tenantId": str(row["tenant_id"]),
        "customerName": row["customer_name"],
        "siteName": row["site_name"],
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
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
        "attachments": attachments,
    }
    if include_description:
        value["complaintText"] = row["complaint_text"]
    return value


def list_tickets(db_path=DB_PATH):
    connection = _open_store(db_path)
    try:
        rows = connection.execute(
            "SELECT * FROM local_tickets WHERE created_by = ? ORDER BY created_at DESC",
            (DEMO_CREATOR,),
        ).fetchall()
        return [_serialize(connection, row, include_description=False) for row in rows]
    finally:
        connection.close()


def get_ticket(ticket_id, db_path=DB_PATH):
    if not re.fullmatch(r"LOCAL-[0-9]{8}-[A-F0-9]{8}", str(ticket_id or "")):
        raise LookupError("Ticket not found")
    connection = _open_store(db_path)
    try:
        row = connection.execute(
            "SELECT * FROM local_tickets WHERE id = ? AND created_by = ?",
            (ticket_id, DEMO_CREATOR),
        ).fetchone()
        if row is None:
            raise LookupError("Ticket not found")
        return _serialize(connection, row, include_description=True)
    finally:
        connection.close()


def update_ticket(ticket_id, changes, db_path=DB_PATH):
    if not re.fullmatch(r"LOCAL-[0-9]{8}-[A-F0-9]{8}", str(ticket_id or "")):
        raise LookupError("Ticket not found")
    if not isinstance(changes, dict) or not changes or set(changes) - {"status", "technicianNotes"}:
        raise ValueError("Only ticket status and technician notes can be changed.")
    updates = {}
    if "status" in changes:
        status_value = changes["status"]
        if status_value not in ALLOWED_STATUSES:
            raise ValueError("Ticket status is invalid.")
        updates["status"] = status_value
    if "technicianNotes" in changes:
        notes = changes["technicianNotes"]
        if not isinstance(notes, str) or len(notes) > 8000:
            raise ValueError("Technician notes must be text of 8,000 characters or less.")
        updates["technician_notes"] = notes
    updates["updated_at"] = datetime.now(timezone.utc).isoformat()

    connection = _open_store(db_path)
    try:
        existing = connection.execute(
            "SELECT id FROM local_tickets WHERE id = ? AND created_by = ?",
            (ticket_id, DEMO_CREATOR),
        ).fetchone()
        if existing is None:
            raise LookupError("Ticket not found")
        assignments = ", ".join(f"{column} = ?" for column in updates)
        with connection:
            connection.execute(
                f"UPDATE local_tickets SET {assignments} WHERE id = ? AND created_by = ?",
                (*updates.values(), ticket_id, DEMO_CREATOR),
            )
        row = connection.execute("SELECT * FROM local_tickets WHERE id = ?", (ticket_id,)).fetchone()
        return _serialize(connection, row, include_description=True)
    finally:
        connection.close()


def read_attachment(ticket_id, attachment_id, db_path=DB_PATH):
    if not re.fullmatch(r"LOCAL-[0-9]{8}-[A-F0-9]{8}", str(ticket_id or "")):
        raise LookupError("Attachment not found")
    if not re.fullmatch(r"[A-Fa-f0-9]{32}", str(attachment_id or "")):
        raise LookupError("Attachment not found")
    connection = _open_store(db_path)
    try:
        row = connection.execute(
            "SELECT a.name, a.content_type, a.content, a.size "
            "FROM local_ticket_attachments a "
            "JOIN local_tickets t ON t.id = a.ticket_id AND t.tenant_id = a.tenant_id "
            "WHERE a.id = ? AND a.ticket_id = ? AND t.created_by = ?",
            (attachment_id, ticket_id, DEMO_CREATOR),
        ).fetchone()
        if row is None:
            raise LookupError("Attachment not found")
        return row["name"], row["content_type"], bytes(row["content"]), int(row["size"])
    finally:
        connection.close()


def _ensure_access_code():
    PRIVATE.mkdir(parents=True, exist_ok=True)
    if ACCESS_FILE.exists():
        token = ACCESS_FILE.read_text(encoding="ascii").strip()
        if len(token) < 40:
            raise RuntimeError("The existing technician access-code file is invalid; it was not overwritten.")
        return token
    token = secrets.token_urlsafe(40)
    temporary = ACCESS_FILE.with_suffix(".tmp")
    temporary.write_text(token + "\n", encoding="ascii")
    os.replace(temporary, ACCESS_FILE)
    return token


def _loopback_request(handler):
    remote = handler.client_address[0]
    host = handler.headers.get("Host", "").split(":", 1)[0].strip("[]").lower()
    return remote in ("127.0.0.1", "::1") and host in ("127.0.0.1", "localhost", "::1")


class Handler(BaseHTTPRequestHandler):
    token = ""

    def log_message(self, *_args):
        # Do not log access codes, ticket contents, attachment names, or URLs.
        pass

    def _send_json(self, status, value):
        raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store, private")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(raw)
        route = urlsplit(self.path).path
        safe_route = "tickets" if route.startswith("/tech-api/tickets/") else "other"
        print(f"{self.command} {safe_route} {status}", flush=True)

    def _authorized(self):
        supplied = self.headers.get("Authorization", "")
        return _loopback_request(self) and secrets.compare_digest(supplied, "Technician " + self.token)

    def do_GET(self):
        if not self._authorized():
            return self._send_json(401, {"detail": "Technician access code required or invalid."})
        path = urlsplit(self.path).path
        if path == "/tech-api/tickets/":
            try:
                return self._send_json(200, {"tickets": list_tickets()})
            except Exception:
                return self._send_json(503, {"detail": "The local technician ticket workspace is unavailable."})

        match = re.fullmatch(r"/tech-api/tickets/([^/]+)/", path)
        if match:
            try:
                ticket_id = unquote(match.group(1))
                return self._send_json(200, {"ticket": get_ticket(ticket_id)})
            except LookupError:
                return self._send_json(404, {"detail": "Ticket not found."})
            except Exception:
                return self._send_json(503, {"detail": "The local technician ticket could not be opened."})

        attachment = re.fullmatch(r"/tech-api/tickets/([^/]+)/attachments/([^/]+)/", path)
        if attachment:
            try:
                name, content_type, content, size = read_attachment(
                    unquote(attachment.group(1)), unquote(attachment.group(2))
                )
                safe_name = Path(name.replace("\\", "/")).name.replace('"', "")
                self.send_response(200)
                self.send_header("Content-Type", content_type or "application/octet-stream")
                self.send_header("Content-Length", str(size))
                self.send_header("Content-Disposition", f'attachment; filename="{safe_name}"')
                self.send_header("Cache-Control", "no-store, private")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(content)
                print("GET ticket-attachment 200", flush=True)
                return
            except LookupError:
                return self._send_json(404, {"detail": "Attachment not found."})
            except Exception:
                return self._send_json(503, {"detail": "The ticket attachment is unavailable."})
        return self._send_json(404, {"detail": "Not found."})

    def do_PATCH(self):
        if not self._authorized():
            return self._send_json(401, {"detail": "Technician access code required or invalid."})
        match = re.fullmatch(r"/tech-api/tickets/([^/]+)/", urlsplit(self.path).path)
        if not match:
            return self._send_json(404, {"detail": "Not found."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > MAX_BODY:
                return self._send_json(400, {"detail": "Invalid request size."})
            changes = json.loads(self.rfile.read(length).decode("utf-8"))
            ticket = update_ticket(unquote(match.group(1)), changes)
            return self._send_json(200, {"ticket": ticket})
        except LookupError:
            return self._send_json(404, {"detail": "Ticket not found."})
        except (ValueError, TypeError, json.JSONDecodeError):
            return self._send_json(400, {"detail": "Ticket update is invalid."})
        except Exception:
            return self._send_json(503, {"detail": "Ticket update could not be saved."})


def main():
    if DB_PATH.name.lower() != "local-ticket-workspace.sqlite3" or not DB_PATH.is_file():
        raise RuntimeError(
            "The authorized customer RAG must create the separate local ticket workspace first. "
            "This technician launcher does not initialize or create ticket storage."
        )
    # Validate schema before publishing the listener, without changing a row.
    connection = _open_store()
    connection.close()
    Handler.token = _ensure_access_code()
    server = HTTPServer((HOST, PORT), Handler)
    print(json.dumps({
        "ready": True,
        "mode": "local-rag-technician-workspace",
        "ticket_count": len(list_tickets()),
        "database": "separate-local-sqlite-ticket-workspace",
        "access_file": str(ACCESS_FILE),
        "url": "http://127.0.0.1:8003/",
    }), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
