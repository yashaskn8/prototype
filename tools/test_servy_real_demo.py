"""Offline safety checks for the separate real-data demo boundary."""
import unittest
import hashlib
import io
import json
import os
from contextlib import redirect_stdout
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from tools import servy_real_demo as demo
from tools import servy_100_readonly_eval as evaluation

from django.conf import settings
from django.core import signing

if not settings.configured:
    settings.configure(SECRET_KEY="synthetic-test-signing-key")


class RealDemoBoundaryTests(unittest.TestCase):
    def test_unapproved_asset_refuses_before_rag_or_database_lookup(self):
        with patch.object(demo, "_asset", side_effect=AssertionError("DB lookup")):
            result = demo._answer(62, "Why is this equipment failing?")
        self.assertEqual(result["response_type"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["citations"], [])

    def test_missing_customer_binding_never_matches_missing_scope(self):
        self.assertNotIn(64, demo.ASSET_SCOPE)
        self.assertNotIn(63, demo.ASSET_SCOPE)
        self.assertEqual(demo.ASSET_SCOPE, {59: 14, 60: 13, 61: 14})

    def test_bearer_identity_requires_loopback_and_local_host(self):
        handler = object.__new__(demo.Handler)
        handler.token = "a-private-synthetic-test-value"
        handler.headers = {"Host": "127.0.0.1:8002", "Authorization": "Demo a-private-synthetic-test-value"}
        handler.client_address = ("127.0.0.1", 12345)
        self.assertTrue(handler._authorized())
        handler.headers["Host"] = "company.example"
        self.assertFalse(handler._authorized())
        handler.headers["Host"] = "127.0.0.1:8002"
        handler.headers["Authorization"] = "Demo wrong"
        self.assertFalse(handler._authorized())
        handler.headers["Authorization"] = "Demo a-private-synthetic-test-value"
        handler.client_address = ("10.0.0.2", 12345)
        self.assertFalse(handler._authorized())

    def test_ticket_preview_uses_signed_question_and_approved_customer_asset(self):
        question = "Synthetic test question?"
        token = signing.dumps({"asset": 60, "customer": 13,
            "question_sha256": hashlib.sha256(question.encode()).hexdigest()},
            salt="servy-real-demo-ticket-preview-v1")
        body = {"ticket_context": token, "question": question,
                "complaint_type": "Synthetic issue", "complaint_text": "Synthetic issue not resolved"}
        asset = SimpleNamespace(id=60, site_id=7, tenant_id=273,
                                site=SimpleNamespace(customer_id=13, tenant_id=273))
        with patch.object(demo, "_asset", return_value=(None, SimpleNamespace(id=13), asset)) as lookup:
            fields = demo._ticket_fields(body)
        lookup.assert_called_once_with(60)
        self.assertEqual(fields["customer_id"], 13)
        self.assertEqual(fields["asset_id"], 60)
        self.assertEqual(fields["complaint_type"], "Synthetic issue")
        self.assertEqual(fields["priority"], "normal")
        wrong_site = SimpleNamespace(id=60, site_id=7, tenant_id=273,
                                     site=SimpleNamespace(customer_id=14, tenant_id=273))
        with patch.object(demo, "_asset", return_value=(None, SimpleNamespace(id=13), wrong_site)):
            with self.assertRaises(ValueError):
                demo._ticket_fields(body)
        with patch.object(demo, "_asset", side_effect=AssertionError("DB lookup")):
            with self.assertRaises(ValueError):
                demo._ticket_fields({**body, "question": "Different question"})
            wrong_asset = signing.dumps({"asset": 62, "customer": 13,
                "question_sha256": hashlib.sha256(question.encode()).hexdigest()},
                salt="servy-real-demo-ticket-preview-v1")
            with self.assertRaises(ValueError):
                demo._ticket_fields({**body, "ticket_context": wrong_asset})
            wrong_customer = signing.dumps({"asset": 60, "customer": 14,
                "question_sha256": hashlib.sha256(question.encode()).hexdigest()},
                salt="servy-real-demo-ticket-preview-v1")
            with self.assertRaises(ValueError):
                demo._ticket_fields({**body, "ticket_context": wrong_customer})

    def test_preview_never_claims_a_created_ticket(self):
        handler = object.__new__(demo.Handler)
        handler.path = "/demo-api/tickets/preview/"
        body = json.dumps({"complaint_type": "Synthetic issue"}).encode()
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = io.BytesIO(body)
        handler._authorized = lambda: True
        handler._send = lambda status, value: (status, value)
        with patch.object(demo, "read_only_database", return_value=nullcontext()), \
                patch.object(demo, "_ticket_fields", return_value={"asset_id": 60}):
            status, response = handler.do_POST()
        self.assertEqual(status, 200)
        self.assertEqual(response["mode"], "PREVIEW_ONLY")
        self.assertNotIn("mock_reference", response)
        self.assertNotIn("servy_id", response)

    def test_real_ticket_write_path_is_not_exposed_by_demo_handler(self):
        handler = object.__new__(demo.Handler)
        handler.path = "/api/customer-support/conversations/fake/ticket/"
        handler._authorized = lambda: True
        handler._send = lambda status, value: (status, value)
        with patch.object(demo, "read_only_database", side_effect=AssertionError("DB access")):
            status, _ = handler.do_POST()
        self.assertEqual(status, 404)
        handler.path = "/demo-api/tickets/confirm/"
        status, _ = handler.do_POST()
        self.assertEqual(status, 404)

    def test_credential_loader_falls_back_to_windows_powershell(self):
        synthetic_file = SimpleNamespace(is_file=lambda: True, read_text=lambda encoding: "synthetic-cipher")
        with patch.object(evaluation, "SECRET", synthetic_file), \
                patch.object(evaluation.shutil, "which", side_effect=[None, "powershell.exe"]), \
                patch.object(evaluation.subprocess, "run", return_value=SimpleNamespace(stdout="synthetic-password")) as run, \
                patch("django.setup"), patch.dict(os.environ, {}, clear=False):
            evaluation.configure_process()
            self.assertEqual(os.environ["MYSQL_USER"], "servy_eval_ro")
        self.assertEqual(run.call_args.args[0][0], "powershell.exe")

    def test_request_log_excludes_source_token_and_response_body(self):
        handler = object.__new__(demo.Handler)
        handler.path = "/demo-api/source/?citation=private-sentinel"
        handler.command = "GET"
        handler.wfile = io.BytesIO()
        handler.send_response = lambda status: None
        handler.send_header = lambda name, value: None
        handler.end_headers = lambda: None
        output = io.StringIO()
        with redirect_stdout(output):
            handler._send(200, {"answer": "private-answer-sentinel"})
        line = output.getvalue()
        self.assertIn("GET /demo-api/source/ 200", line)
        self.assertNotIn("private-sentinel", line)
        self.assertNotIn("private-answer-sentinel", line)


if __name__ == "__main__":
    unittest.main()
