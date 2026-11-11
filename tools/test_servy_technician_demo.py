"""Isolation and update tests for the separate code-gated technician demo."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import servy_technician_demo as tech


class TechnicianDemoWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "local-ticket-workspace.sqlite3"
        self.ticket_id = "LOCAL-20261111-ABCDEF12"
        self.attachment_id = "a" * 32
        connection = sqlite3.connect(self.db)
        connection.executescript("""
            CREATE TABLE local_tickets (
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
            CREATE TABLE local_ticket_attachments (
                id TEXT PRIMARY KEY,
                ticket_id TEXT NOT NULL,
                tenant_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                content_type TEXT NOT NULL,
                size INTEGER NOT NULL,
                content BLOB NOT NULL
            );
        """)
        columns = (
            "id, tenant_id, customer_id, customer_name, site_id, site_name, "
            "asset_id, asset_name, asset_code, call_type, complaint_type, "
            "complaint_text, contact_name, severity, priority, machine_status, "
            "status, technician_notes, source_conversation_id, created_by, "
            "created_at, updated_at"
        )
        values = (
            self.ticket_id, 7, 14, "Demo Customer", 4, "Demo Facility",
            60, "VRF Outdoor Unit", "VRF-060", "Service", "Cooling / performance",
            "Customer: Unit is not cooling.\nServy: Please keep the unit stopped pending inspection.",
            "Demo Contact", "High", "high", "Intermittent", "open", "",
            "", tech.DEMO_CREATOR, "2026-11-11T10:00:00+00:00",
            "2026-11-11T10:00:00+00:00",
        )
        placeholders = ", ".join("?" for _ in values)
        connection.execute(f"INSERT INTO local_tickets ({columns}) VALUES ({placeholders})", values)
        # An ordinary local app ticket must never leak into the code-gated RAG queue.
        other_values = list(values)
        other_values[0] = "LOCAL-20261111-ABCDEF13"
        other_values[19] = "ordinary-customer-account"
        connection.execute(f"INSERT INTO local_tickets ({columns}) VALUES ({placeholders})", other_values)
        connection.execute(
            "INSERT INTO local_ticket_attachments "
            "(id, ticket_id, tenant_id, name, content_type, size, content) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (self.attachment_id, self.ticket_id, 7, "report.txt", "text/plain", 6, b"report"),
        )
        connection.commit()
        connection.close()

    def tearDown(self):
        self.temp.cleanup()

    def test_queue_includes_only_authorized_rag_demo_tickets(self):
        tickets = tech.list_tickets(self.db)
        self.assertEqual([item["id"] for item in tickets], [self.ticket_id])
        self.assertEqual(tickets[0]["assetName"], "VRF Outdoor Unit")
        self.assertEqual(tickets[0]["attachments"][0]["name"], "report.txt")
        self.assertNotIn("complaintText", tickets[0])

    def test_detail_contains_full_report_but_other_creator_is_hidden(self):
        ticket = tech.get_ticket(self.ticket_id, self.db)
        self.assertIn("Customer: Unit is not cooling.", ticket["complaintText"])
        with self.assertRaises(LookupError):
            tech.get_ticket("LOCAL-20261111-ABCDEF13", self.db)

    def test_technician_can_change_only_status_and_internal_notes(self):
        updated = tech.update_ticket(
            self.ticket_id,
            {"status": "in_progress", "technicianNotes": "Inspection scheduled."},
            self.db,
        )
        self.assertEqual(updated["status"], "in_progress")
        self.assertEqual(updated["technicianNotes"], "Inspection scheduled.")
        with self.assertRaises(ValueError):
            tech.update_ticket(self.ticket_id, {"tenant_id": 999}, self.db)
        with self.assertRaises(ValueError):
            tech.update_ticket(self.ticket_id, {"status": "deleted"}, self.db)

    def test_attachment_is_scoped_to_rag_ticket(self):
        name, content_type, body, size = tech.read_attachment(
            self.ticket_id, self.attachment_id, self.db,
        )
        self.assertEqual((name, content_type, body, size),
                         ("report.txt", "text/plain", b"report", 6))
        with self.assertRaises(LookupError):
            tech.read_attachment(
                "LOCAL-20261111-ABCDEF13", self.attachment_id, self.db,
            )

    def test_queue_can_start_before_first_ticket_without_creating_database(self):
        missing = Path(self.temp.name) / "missing-local-ticket-workspace.sqlite3"
        self.assertEqual(tech.list_tickets(missing), [])
        self.assertFalse(missing.exists(), "Technician queue must not initialize customer ticket storage")

    def test_main_starts_empty_queue_without_creating_workspace(self):
        missing = Path(self.temp.name) / "local-ticket-workspace.sqlite3"
        with patch.object(tech, "DB_PATH", missing), \
                patch.object(tech, "_ensure_access_code", return_value="test-only-code"), \
                patch.object(tech, "HTTPServer") as server:
            server.return_value.serve_forever.side_effect = RuntimeError("test server stop")
            with self.assertRaisesRegex(RuntimeError, "test server stop"):
                tech.main()
        self.assertFalse(missing.exists(), "Technician startup must not create customer ticket storage")

    def test_workspace_path_must_keep_expected_name(self):
        wrong = Path(self.temp.name) / "company.sqlite3"
        with self.assertRaises(RuntimeError):
            tech.list_tickets(wrong)


if __name__ == "__main__":
    unittest.main()
