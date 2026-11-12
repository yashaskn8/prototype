"""Offline safety checks for the separate real-data demo boundary."""
import unittest
import hashlib
import io
import json
import os
import base64
import sqlite3
import tempfile
from pathlib import Path
from contextlib import redirect_stdout
from contextlib import nullcontext
from types import SimpleNamespace, ModuleType
from unittest.mock import patch, Mock

# This standalone offline test must also import on Linux CI runners.
os.environ.setdefault("LOCALAPPDATA", "/tmp")
from tools import servy_real_demo as demo
from tools import servy_real_demo_index_review as index_review
from tools import servy_100_readonly_eval as evaluation

from django.conf import settings
from django.core import signing

if not settings.configured:
    settings.configure(SECRET_KEY="synthetic-test-signing-key")


def _fake_demo_answer_modules():
    answer = ModuleType("core.services.customer_answer")
    answer._question_guard = Mock(return_value=None)
    answer.build_customer_answer = Mock()
    answer.exception_presentation = Mock()
    diagnostic = ModuleType("core.services.customer_diagnostic")
    diagnostic.check_for_safety_hazard = Mock(return_value=(False, ""))
    diagnostic.check_for_vague_complaint = Mock(return_value=(False, "", []))
    rag = ModuleType("core.services.rag")
    rag.ask_rag = Mock()
    return {
        "core.services.customer_answer": answer,
        "core.services.customer_diagnostic": diagnostic,
        "core.services.rag": rag,
    }


class RealDemoBoundaryTests(unittest.TestCase):
    def test_unapproved_asset_refuses_before_rag_or_database_lookup(self):
        for question in (
            "Why is this equipment failing?",
            "What details should I give the technician?",
        ):
            with self.subTest(question=question), patch.object(
                demo, "_asset", side_effect=AssertionError("DB lookup"),
            ):
                result = demo._answer(62, question)
            self.assertEqual(result["response_type"], "INSUFFICIENT_EVIDENCE")
            self.assertEqual(result["citations"], [])

    def test_handoff_request_uses_signed_scoped_history_without_rag(self):
        question = "The issue is still happening. What details should I give the technician?"
        first = "The outdoor module starts, but the room does not cool and then the unit stops."
        thread_id = "thread-01234567-89ab-cdef-0123-456789abcdef"
        token = demo._sign_demo_context(
            60, 13, first, first, [first], last_topic="next", thread_id=thread_id,
        )
        asset = SimpleNamespace(
            id=60, name="VRF Outdoor Module", asset_code="SKY-AST-00002",
            model_number="VRF-1",
            product=SimpleNamespace(name="VRF Outdoor Module"),
        )
        tenant, customer = SimpleNamespace(id=273), SimpleNamespace(id=13)
        modules = _fake_demo_answer_modules()
        with patch.dict("sys.modules", modules):
            with patch.object(demo, "_asset", return_value=(tenant, customer, asset)):
                with patch.object(
                    demo, "plan_turn",
                    side_effect=AssertionError("Handoff must not invoke a planner"),
                ):
                    result = demo._answer(
                        60, question, continuation=token, thread_id=thread_id,
                    )

        self.assertEqual(result["response_type"], "TECHNICIAN_HANDOFF")
        self.assertEqual(result["reason_code"], "CUSTOMER_TECHNICIAN_HANDOFF")
        self.assertEqual(result["retrieval_attempts"], 0)
        self.assertEqual(result["retrieval_backend"], "not-required")
        self.assertEqual(result["citations"], [])
        self.assertIsNone(result["presentation"])
        self.assertIn("room does not cool", " ".join(
            item["text"] for item in result["handoff"]["reported_details"]
        ))
        self.assertIsNotNone(result["ticket_context"])
        self.assertEqual(result["dialogue_planner"], "local-handoff-checklist")
        modules["core.services.rag"].ask_rag.assert_not_called()

    def test_safety_escalation_takes_precedence_over_handoff_checklist(self):
        asset = SimpleNamespace(
            id=60, name="VRF Outdoor Module", asset_code="SKY-AST-00002",
            model_number="VRF-1",
            product=SimpleNamespace(name="VRF Outdoor Module"),
        )
        tenant, customer = SimpleNamespace(id=273), SimpleNamespace(id=13)
        modules = _fake_demo_answer_modules()
        modules["core.services.customer_diagnostic"].check_for_safety_hazard.return_value = (
            True, "Stop using the equipment and contact authorized service."
        )
        with patch.dict("sys.modules", modules):
            with patch.object(demo, "_asset", return_value=(tenant, customer, asset)):
                result = demo._answer(
                    60,
                    "There is smoke near the equipment. What details should I give the technician?",
                )

        self.assertEqual(result["response_type"], "ESCALATION_REQUIRED")
        self.assertIsNone(result["handoff"])
        self.assertEqual(result["retrieval_attempts"], 0)
        modules["core.services.rag"].ask_rag.assert_not_called()

    def test_handoff_cannot_cross_customer_scope_for_approved_asset_id(self):
        asset = SimpleNamespace(
            id=60, name="VRF Outdoor Module", asset_code="SKY-AST-00002",
            model_number="VRF-1",
            product=SimpleNamespace(name="VRF Outdoor Module"),
        )
        tenant, wrong_customer = SimpleNamespace(id=273), SimpleNamespace(id=14)
        with patch.dict("sys.modules", _fake_demo_answer_modules()):
            with patch.object(demo, "_asset", return_value=(tenant, wrong_customer, asset)):
                with self.assertRaisesRegex(ValueError, "Customer scope changed"):
                    demo._answer(60, "What details should I give the technician?")

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

    def test_local_ticket_fields_derive_customer_and_equipment_from_signed_scope(self):
        question = "Synthetic local ticket handoff?"
        token = signing.dumps({"asset": 60, "customer": 13,
            "question_sha256": hashlib.sha256(question.encode()).hexdigest()},
            salt="servy-real-demo-ticket-preview-v1")
        body = {"ticket_context": token, "question": question,
            "call_type": "Breakdown Service", "complaint_type": "Cooling / performance",
            "complaint_text": "Customer and Servy synthetic transcript",
            "contact_name": "", "severity": "Medium", "priority": "normal",
            "machine_status": "Intermittent", "customer_id": 999, "asset_id": 999}
        site = SimpleNamespace(id=7, name="Synthetic site", customer_id=13, tenant_id=273)
        asset = SimpleNamespace(id=60, site_id=7, tenant_id=273, site=site,
            name="Synthetic asset", asset_code="SYN-60")
        tenant = SimpleNamespace(id=273)
        customer = SimpleNamespace(id=13, name="Synthetic customer")
        with patch.object(demo, "_asset", return_value=(tenant, customer, asset)):
            fields = demo._local_ticket_fields(body)
        self.assertEqual(fields["tenant_id"], 273)
        self.assertEqual(fields["customer_id"], 13)
        self.assertEqual(fields["asset_id"], 60)
        self.assertEqual(fields["customer_name"], "Synthetic customer")
        self.assertEqual(fields["complaint_text"], "Customer and Servy synthetic transcript")

    def test_local_ticket_is_written_only_to_a_separate_sqlite_workspace(self):
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "technician-tickets.sqlite3"
            fields = {
                "tenant_id": 273, "customer_id": 13, "customer_name": "Synthetic customer",
                "site_id": 7, "site_name": "Synthetic site", "asset_id": 60,
                "asset_name": "Synthetic asset", "asset_code": "SYN-60",
                "call_type": "Breakdown Service", "complaint_type": "Cooling / performance",
                "complaint_text": "Customer and Servy synthetic transcript", "contact_name": "",
                "severity": "Medium", "priority": "normal", "machine_status": "Intermittent",
                "status": "open", "technician_notes": "", "source_conversation_id": "",
                "created_by": "synthetic-test",
            }
            with patch.object(settings, "DATABASES", {
                    "default": {"ENGINE": "django.db.backends.mysql", "NAME": "unchanged_company_db"}},
                    create=True), \
                    patch.object(settings, "SERVY_LOCAL_TICKET_DB_PATH", database, create=True):
                ticket = demo._store_local_ticket(fields, [{
                    "name": "synthetic-note.txt", "content_type": "text/plain", "content": b"synthetic",
                }])
            self.assertTrue(ticket["id"].startswith("LOCAL-"))
            self.assertEqual(ticket["customerId"], "13")
            self.assertEqual(ticket["complaintText"], "Customer and Servy synthetic transcript")
            self.assertEqual(ticket["attachments"][0]["name"], "synthetic-note.txt")
            connection = sqlite3.connect(database)
            try:
                row = connection.execute(
                    "SELECT tenant_id, customer_id, asset_id, complaint_text FROM local_tickets WHERE id = ?",
                    (ticket["id"],),
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(row, (273, 13, 60, "Customer and Servy synthetic transcript"))
            from core.local_ticket_store import list_tickets, open_local_store
            with patch.object(settings, "DATABASES", {
                    "default": {"ENGINE": "django.db.backends.mysql", "NAME": "unchanged_company_db"}}, create=True), \
                    patch.object(settings, "SERVY_LOCAL_TICKET_DB_PATH", database, create=True):
                technician_connection = open_local_store()
                try:
                    tenant_tickets = list_tickets(technician_connection, 273)
                    other_tenant_tickets = list_tickets(technician_connection, 274)
                    other_customer_tickets = list_tickets(technician_connection, 273, 14)
                finally:
                    technician_connection.close()
            self.assertEqual([item["id"] for item in tenant_tickets], [ticket["id"]])
            self.assertEqual(other_tenant_tickets, [])
            self.assertEqual(other_customer_tickets, [])

    def test_local_create_route_writes_after_readonly_company_scope_validation(self):
        handler = object.__new__(demo.Handler)
        handler.path = "/demo-api/tickets/create-local/"
        payload = json.dumps({"synthetic": "form"}).encode()
        handler.headers = {"Content-Length": str(len(payload)), "Content-Type": "application/json"}
        handler.rfile = io.BytesIO(payload)
        handler._authorized = lambda: True
        handler._send = lambda status, value: (status, value)
        state = {"readonly": False}

        class ReadOnlyScope:
            def __enter__(self):
                state["readonly"] = True

            def __exit__(self, *_args):
                state["readonly"] = False

        def store(_fields, _attachments):
            self.assertFalse(state["readonly"])
            return {"id": "LOCAL-SYNTHETIC"}

        with patch.object(demo, "read_only_database", return_value=ReadOnlyScope()), \
                patch.object(demo, "_local_ticket_fields", return_value={"asset_id": 60}), \
                patch.object(demo, "_store_local_ticket", side_effect=store):
            status, response = handler.do_POST()
        self.assertEqual(status, 201)
        self.assertEqual(response["mode"], "LOCAL_SQLITE")
        self.assertEqual(response["ticket"]["id"], "LOCAL-SYNTHETIC")

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
                patch.object(evaluation.subprocess, "CREATE_NO_WINDOW", 0, create=True), \
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
        self.assertIn("HTTP 200 OK | GET /demo-api/source/", line)
        self.assertNotIn("private-sentinel", line)
        self.assertNotIn("private-answer-sentinel", line)


    def test_continuation_is_signed_scoped_and_not_database_persisted(self):
        token = demo._sign_demo_context(
            60, 13, "Why is the VRF not cooling?", "", [], hazard=False
        )
        first, history, hazard = demo._load_demo_context(token, 60, 13)
        self.assertEqual(first, "Why is the VRF not cooling?")
        self.assertEqual(history, ["Why is the VRF not cooling?"])
        self.assertFalse(hazard)
        with self.assertRaises(ValueError):
            demo._load_demo_context(token, 59, 14)
        with self.assertRaises(ValueError):
            demo._load_demo_context(token, 60, 14)
        with self.assertRaises(ValueError):
            demo._load_demo_context(token + "tampered", 60, 13)

    def test_signed_dialogue_state_remembers_previously_asked_questions(self):
        token = demo._sign_demo_context(
            60, 13, "What causes the issue?", "The VRF is not cooling.",
            ["The VRF is not cooling."], last_topic="causes",
            shown_facts=["a" * 64],
            asked_followups=["Is a fault code or warning visible on the display?"],
        )
        topic, facts, asked = demo._load_demo_dialogue_state(token)
        self.assertEqual(topic, "causes")
        self.assertEqual(facts, ["a" * 64])
        self.assertEqual(asked, ["Is a fault code or warning visible on the display?"])

    def test_signed_dialogue_state_rejects_oversized_or_malformed_question_memory(self):
        from django.core import signing
        bad = signing.dumps({
            "last_topic": "causes", "shown_facts": [],
            "asked_followups": ["x" * 241],
        }, salt=demo.DEMO_CONTEXT_SALT)
        with self.assertRaises(ValueError):
            demo._load_demo_dialogue_state(bad)

    def test_signed_continuation_cannot_be_replayed_in_another_chat(self):
        import uuid
        first_chat = "thread-" + str(uuid.uuid4())
        second_chat = "thread-" + str(uuid.uuid4())
        token = demo._sign_demo_context(
            60, 13, "The split AC is not cooling",
            "", [], thread_id=first_chat, last_topic="causes",
        )
        original, history, hazard = demo._load_demo_context(
            token, 60, 13, thread_id=first_chat)
        self.assertEqual(original, "The split AC is not cooling")
        self.assertEqual(history, ["The split AC is not cooling"])
        self.assertFalse(hazard)
        for wrong_thread in (second_chat, None, "thread-" + "0" * 36):
            with self.subTest(wrong_thread=wrong_thread):
                with self.assertRaises(ValueError):
                    demo._load_demo_context(token, 60, 13, thread_id=wrong_thread)
        with self.assertRaises(ValueError):
            demo._load_demo_context(token, 59, 14, thread_id=first_chat)
        # Legacy unbound tokens are not accepted by newly scoped chats.
        legacy = demo._sign_demo_context(
            60, 13, "Old unbound chat", "", [])
        with self.assertRaises(ValueError):
            demo._load_demo_context(legacy, 60, 13, thread_id=second_chat)

    def test_real_demo_query_passes_thread_identity_to_backend(self):
        handler = object.__new__(demo.Handler)
        handler.path = "/demo-api/query/"
        handler._authorized = lambda: True
        handler._send = lambda status, body: (status, body)
        thread_id = "thread-01234567-89ab-cdef-0123-456789abcdef"
        payload = {
            "asset_id": 60, "question": "What about the compressor?",
            "continuation": None, "thread_id": thread_id,
        }
        encoded = json.dumps(payload).encode("utf-8")
        handler.headers = {"Content-Length": str(len(encoded))}
        handler.rfile = io.BytesIO(encoded)
        with patch.object(demo, "read_only_database", return_value=nullcontext()), \
             patch.object(demo, "_answer", return_value={
                 "response_type": "INSUFFICIENT_EVIDENCE"
             }) as mocked:
            status, body = handler.do_POST()
        self.assertEqual(status, 200)
        self.assertEqual(body["response_type"], "INSUFFICIENT_EVIDENCE")
        mocked.assert_called_once_with(
            60, "What about the compressor?",
            continuation=None, thread_id=thread_id,
        )

    def test_follow_up_retrieval_uses_prior_customer_report_without_inventing_completion(self):
        asset = SimpleNamespace(
            id=60, name="VRF Outdoor Module", asset_code="SKY-AST-00002",
            model_number="VRF-1", product=SimpleNamespace(name="VRF Outdoor Module"),
        )
        context, query = demo._demo_query_context(
            asset, "What next?", "The VRF module is not cooling.",
            ["The VRF module is not cooling.", "It has weak airflow."],
        )
        self.assertTrue(context.is_followup)
        self.assertIn("weak airflow", context.reported_symptoms)
        self.assertIn("next troubleshooting steps", query)
        self.assertIn("VRF Outdoor Module", query)
        denied, _ = demo._demo_query_context(
            asset, "What next?", "The VRF module is not cooling.",
            ["I haven't checked the air filter."],
        )
        self.assertEqual(denied.completed_checks, [])
        confirmed, _ = demo._demo_query_context(
            asset, "What next?", "The VRF module is not cooling.",
            ["I checked the air filter."],
        )
        self.assertIn("air filter", confirmed.completed_checks)

    def test_handoff_follow_up_retains_reported_non_cooling_symptom(self):
        asset = SimpleNamespace(
            id=60, name="VRF Outdoor Module", asset_code="SKY-AST-00002",
            model_number="VRF-1", product=SimpleNamespace(name="VRF Outdoor Module"),
        )
        first = "The outdoor module starts, but the room does not cool and then the unit stops."
        context, query = demo._demo_query_context(
            asset,
            "The issue is still happening. What details should I give the technician?",
            first, [first],
        )
        self.assertTrue(context.is_followup)
        self.assertIn("does not cool", context.reported_symptoms)
        self.assertIn("does not cool", query)

    def test_reportable_detail_follow_up_retrieval_keeps_prior_reported_fault(self):
        asset = SimpleNamespace(
            id=60, name="VRF Outdoor Module", asset_code="SKY-AST-00002",
            model_number="VRF-1", product=SimpleNamespace(name="VRF Outdoor Module"),
        )
        first = "The VRF module reports a low-pressure or capacity-fault alarm."
        question = "Should I mention which indoor units are affected by the alarm?"
        context, query = demo._demo_query_context(asset, question, first, [first])
        self.assertTrue(context.is_followup)
        self.assertIn("low-pressure or capacity-fault alarm", query)
        self.assertIn("affected by the alarm", query)

    def test_follow_up_completion_does_not_invent_equipment_or_clause(self):
        asset = SimpleNamespace(
            id=61, name="Split AC Unit 2T", asset_code="SKY-AST-00001",
            model_number="", product=SimpleNamespace(name="Split AC Unit 2T"),
        )
        first = "My Split AC is running, but the room is not cooling."
        question = "The filter was cleaned but cooling is still weak. What else should I report?"
        context, _ = demo._demo_query_context(asset, question, first, [first])
        self.assertEqual(context.completed_checks, ["filter"])
        self.assertNotIn("Split AC", context.completed_checks)
        self.assertNotIn("but cooling", context.completed_checks)

        active, _ = demo._demo_query_context(
            asset, "I cleaned the air filter but cooling is still weak. What next?",
            first, [first],
        )
        self.assertEqual(active.completed_checks, ["air filter"])

    def test_demo_query_route_passes_signed_continuation_in_read_only_context(self):
        handler = object.__new__(demo.Handler)
        handler.path = "/demo-api/query/"
        handler._authorized = lambda: True
        handler._send = lambda status, value: (status, value)
        payload = {"asset_id": 60, "question": "What next?", "continuation": "opaque-state"}
        data = json.dumps(payload).encode("utf-8")
        handler.headers = {"Content-Length": str(len(data))}
        handler.rfile = io.BytesIO(data)
        with patch.object(demo, "read_only_database", return_value=nullcontext()) as readonly, \
                patch.object(demo, "_answer", return_value={"response_type": "CLARIFICATION"}) as answer:
            code, response = handler.do_POST()
        self.assertEqual(code, 200)
        self.assertEqual(response["response_type"], "CLARIFICATION")
        answer.assert_called_once_with(60, "What next?", continuation="opaque-state")
        readonly.assert_called_once_with()

    def test_continuation_is_bounded_and_preserves_original_complaint(self):
        first = "VRF is not cooling"
        previous = []
        for index in range(15):
            token = demo._sign_demo_context(
                60, 13, f"Message {index}", first, previous
            )
            stored_first, previous, hazard = demo._load_demo_context(token, 60, 13)
            self.assertFalse(hazard)
            self.assertEqual(stored_first, first)
            self.assertLessEqual(len(previous), demo.DEMO_CONTEXT_MAX_TURNS)
        self.assertEqual(previous[-1], "Message 14")


    def test_index_drift_limits_and_signed_independent_review(self):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from tools.servy_500_benchmark import write_json
        expected = {"chroma.sqlite3": "a" * 64, "segment/data_level0.bin": "b" * 64,
                    "segment/length.bin": "c" * 64}
        actual = {**expected, "chroma.sqlite3": "d" * 64}
        self.assertEqual(index_review.changed_files(expected, actual), ["chroma.sqlite3"])
        with self.assertRaises(RuntimeError):
            index_review.changed_files(expected, {**actual, "added.bin": "e" * 64})
        with self.assertRaises(RuntimeError):
            index_review.changed_files(expected, {**actual, "segment/header.bin": "f" * 64})
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_json(root / "before.json", {"index": expected})
            write_json(root / "production_index_manifest.json", expected)
            request = index_review.review_request(
                root, expected=expected, production=actual, private_copy=actual)
            with patch.dict(os.environ, {"SERVY_REAL_DEMO_REVIEW_PUBLIC_KEY": ""}):
                with self.assertRaises(RuntimeError):
                    index_review.verify_signed_approval(root, request)
            key = Ed25519PrivateKey.generate()
            public = root / "reviewer.pem"
            public.write_bytes(key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo))
            signed = {
                "request": request, "reviewer": "Independent reviewer",
                "reason": "Verified against separate trusted company collection reference",
                "approved_utc": "2026-11-04T15:00:00Z",
                "attestation": index_review.ATTESTATION,
            }
            write_json(root / "approved_chroma_drift_v2.json", {
                "review": signed,
                "signature": base64.b64encode(
                    key.sign(index_review.canonical_bytes(signed))).decode("ascii"),
            })
            with patch.dict(os.environ, {"SERVY_REAL_DEMO_REVIEW_PUBLIC_KEY": str(public)}):
                self.assertTrue(index_review.verify_signed_approval(root, request))
                with self.assertRaises(RuntimeError):
                    index_review.verify_signed_approval(
                        root, {**request, "current_production_manifest_sha256": "0" * 64})
            self.assertEqual(index_review.sha256_file(root / "before.json"),
                             request["frozen_before_sha256"])
            self.assertEqual(index_review.sha256_file(root / "production_index_manifest.json"),
                             request["frozen_manifest_sha256"])


    def _mock_verified_source_modules(self, candidates, approved_chunk=None):
        """Fake imported service modules without bootstrapping live Django models."""
        retriever = ModuleType("core.services.retriever")
        retriever._knowledge_candidates = Mock(return_value=candidates)
        answer = ModuleType("core.services.customer_answer")
        answer.source_chunk = Mock(return_value=approved_chunk)
        isolation = ModuleType("core.services.customer_isolation")
        isolation.is_verified_real_document = Mock(return_value=True)
        return {
            "core.services.retriever": retriever,
            "core.services.customer_answer": answer,
            "core.services.customer_isolation": isolation,
        }, answer.source_chunk

    def test_verified_source_rebuild_refuses_unapproved_or_missing_passages(self):
        from tools.servy_real_demo_source_rebuild import authoritative_chunks
        tenant = SimpleNamespace(id=273)
        customer = SimpleNamespace(id=13)
        asset = SimpleNamespace(id=60, customer_id=13, tenant_id=273)
        lookup = lambda unused: (tenant, customer, asset)
        modules, _ = self._mock_verified_source_modules([])
        with patch.dict("sys.modules", modules):
            with self.assertRaisesRegex(RuntimeError, "required approved source"):
                authoritative_chunks({60: 13}, lookup)
            bad_lookup = lambda unused: (tenant, SimpleNamespace(id=14), asset)
            with self.assertRaisesRegex(RuntimeError, "scope differs"):
                authoritative_chunks({60: 13}, bad_lookup)

    def test_verified_source_rebuild_uses_only_authorized_documents(self):
        from tools.servy_real_demo_source_rebuild import authoritative_chunks
        tenant = SimpleNamespace(id=273)
        customer = SimpleNamespace(id=13)
        asset = SimpleNamespace(id=60, customer_id=13, tenant_id=273)
        doc = SimpleNamespace(
            disable_sharing=False, is_confidential=False,
            is_rag_enabled=True, index_status="INDEXED",
        )
        chunk = SimpleNamespace(
            id=9, tenant_id=273, document_id=8, document=doc,
            is_quarantined=False, text="Approved original manual observation.",
            heading="Observations",
        )
        lookup = lambda unused: (tenant, customer, asset)
        modules, source = self._mock_verified_source_modules([chunk], chunk)
        with patch.dict("sys.modules", modules):
            passages, scope = authoritative_chunks({60: 13}, lookup)
            self.assertEqual(passages, [chunk])
            self.assertEqual(scope, {60: 1})
            source.assert_called_once_with(
                tenant=tenant, customer=customer, asset=asset,
                document_id=8, chunk_id=9,
            )
            doc.is_confidential = True
            with self.assertRaisesRegex(RuntimeError, "Unverified or unauthorized"):
                authoritative_chunks({60: 13}, lookup)

    def test_verified_source_rebuild_fails_if_chroma_is_already_open(self):
        from tools.servy_real_demo_source_rebuild import rebuild_isolated_bge_index
        from core.services import chroma_store
        with patch.object(chroma_store, "_chroma_client", object()):
            with self.assertRaisesRegex(RuntimeError, "already open"):
                rebuild_isolated_bge_index("/tmp", {}, {60: 13}, lambda unused: None)


    def test_off_topic_query_is_blocked_before_rag_and_keeps_signed_context(self):
        from types import ModuleType
        from unittest.mock import Mock
        from tools.servy_real_demo_dialogue import out_of_scope_response
        token = demo._sign_demo_context(
            60, 13, "My split AC is not cooling", "",
            ["My split AC is not cooling"],
            last_topic="causes", shown_facts=["a" * 64],
        )
        asset = SimpleNamespace(
            id=60, name="Synthetic split AC",
            model_number="SPLIT-1", asset_code="SYNTH-1",
            product=SimpleNamespace(name="Synthetic split AC"),
        )
        customer = SimpleNamespace(id=13)
        tenant = SimpleNamespace(id=273)
        answer_mod = ModuleType("core.services.customer_answer")
        answer_mod._question_guard = Mock(return_value=None)
        answer_mod.build_customer_answer = Mock(
            side_effect=AssertionError("Technical evidence should not be searched")
        )
        answer_mod.exception_presentation = lambda request: {
            "schema_version": "customer-answer-v1",
            "response_type": request["response_type"],
            "equipment": request["asset"],
            "sections": [], "citations": [], "hazard": {
                "severity": "none", "message": "",
            },
            "message": request.get("summary", ""),
        }
        hazard_mod = ModuleType("core.services.customer_diagnostic")
        hazard_mod.check_for_safety_hazard = Mock(return_value=(False, ""))
        hazard_mod.check_for_vague_complaint = Mock(return_value=(False, "", []))
        rag_mod = ModuleType("core.services.rag")
        rag_mod.ask_rag = Mock(
            side_effect=AssertionError("Off-topic query reached document retrieval")
        )
        modules = {
            "core.services.customer_answer": answer_mod,
            "core.services.customer_diagnostic": hazard_mod,
            "core.services.rag": rag_mod,
        }
        for question in ("who is yashas", "what is the gender of yashas kn"):
            with self.subTest(question=question), \
                 patch.dict("sys.modules", modules), \
                 patch.object(demo, "_asset", return_value=(tenant, customer, asset)), \
                 patch.object(demo, "_demo_query_context",
                              return_value=(SimpleNamespace(is_followup=False), question)):
                response = demo._answer(60, question, continuation=token)
            self.assertIsNotNone(out_of_scope_response(question))
            self.assertEqual(response["response_type"], "INSUFFICIENT_EVIDENCE")
            self.assertEqual(response["reason_code"], "OUT_OF_SCOPE_QUERY")
            self.assertEqual(response["engine"], "dialogue-scope-guard")
            self.assertEqual(response["citations"], [])
            self.assertEqual(response["continuation"], token)
            self.assertIsNone(response["ticket_context"])
            first, recent, _ = demo._load_demo_context(token, 60, 13)
            self.assertNotIn(question, recent)
            self.assertEqual(demo._load_demo_dialogue_state(token)[0], "causes")

    def test_new_subject_is_not_rewritten_with_past_cooling_reports(self):
        asset = SimpleNamespace(
            id=60, name="Synthetic split AC", asset_code="TEST-60",
            model_number="AC-60", product=SimpleNamespace(name="Synthetic AC"),
        )
        history = ["The AC is not cooling", "What causes it?"]
        for independent in (
            "Who is Yashas?", "What is the gender of Yashas KN?",
            "The AC is overheating", "What is the capital of France?",
        ):
            context, query = demo._demo_query_context(
                asset, independent, "The AC is not cooling", history
            )
            self.assertFalse(context.is_followup)
            self.assertEqual(query, independent)
        context, query = demo._demo_query_context(
            asset, "What possible causes are documented?",
            "The AC is not cooling", history,
        )
        self.assertTrue(context.is_followup)
        self.assertIn("not cooling", query)


    def test_component_followup_uses_active_signed_issue_and_explicit_component(self):
        from tools.servy_real_demo_dialogue import is_component_followup
        asset = SimpleNamespace(
            id=60, name="Synthetic Split AC", asset_code="TEST-60",
            model_number="AC-60", product=SimpleNamespace(name="Synthetic AC"),
        )
        history = [
            "The AC overheats.",
            "The AC is running but is not cooling.",
        ]
        for short in ("What about the compressor?", "And the fan?",
                      "Could it be the filter?"):
            with self.subTest(short=short):
                self.assertTrue(is_component_followup(short))
                context, query = demo._demo_query_context(
                    asset, short, history[0], history)
                self.assertTrue(context.is_followup)
                self.assertIn("not cooling", query)
                self.assertNotIn("overheats", query)
        for unrelated in ("Who is Yashas?", "What is the capital of France?"):
            context, query = demo._demo_query_context(
                asset, unrelated, history[0], history)
            self.assertFalse(context.is_followup)
            self.assertEqual(query, unrelated)

    def test_generic_vrf_causes_followup_keeps_selected_fault_in_retrieval_query(self):
        from tools.test_servy_real_demo_dialogue import contract

        asset = SimpleNamespace(
            id=60, name="VRF Outdoor Module", model_number="VRF-1",
            asset_code="SKY-AST-00002",
            product=SimpleNamespace(name="VRF Outdoor Module"),
        )
        tenant, customer = SimpleNamespace(id=273), SimpleNamespace(id=13)
        modules = _fake_demo_answer_modules()
        modules["core.services.customer_answer"].build_customer_answer.return_value = contract()
        modules["core.services.rag"].ask_rag.return_value = {
            "engine": "extractive", "references": [],
            "retrieval_backend": "hybrid", "reranker_applied": True,
            "reranker_model": "servy_v5_bge_verified",
        }
        history = [
            "What could be the possible reasons why my VRF outdoor module is not working?",
            "I think it may be a communication or addressing fault. What causes are documented in the approved manual?",
        ]
        thread_id = "thread-vrf-category-0001"
        continuation = demo._sign_demo_context(
            60, 13, "What possible causes are documented?",
            history[0], history, last_topic="causes", thread_id=thread_id,
        )
        with patch.dict("sys.modules", modules), \
             patch.object(demo, "_asset", return_value=(tenant, customer, asset)), \
             patch.object(demo, "plan_turn", return_value=None), \\
             patch.object(demo, "search_all_approved_documents",
                          side_effect=RuntimeError("synthetic scan unavailable")):
            demo._answer(
                60, "What possible causes are documented?",
                continuation=continuation, thread_id=thread_id,
            )

        self.assertGreaterEqual(modules["core.services.rag"].ask_rag.call_count, 1)
        first_query = modules["core.services.rag"].ask_rag.call_args_list[0].kwargs["question"]
        self.assertIn("communication or addressing fault", first_query.casefold())
        self.assertIn("not working", first_query.casefold())

    def test_targeted_retry_reruns_verified_contract_once_for_a_specific_component(self):
        from tools.test_servy_real_demo_dialogue import contract
        import copy

        asset = SimpleNamespace(
            id=60, name="Synthetic Split AC", model_number="AC-60",
            asset_code="SYN-60", product=SimpleNamespace(name="Synthetic AC"),
        )
        tenant, customer = SimpleNamespace(id=273), SimpleNamespace(id=13)
        question = "What causes involve the compressor?"
        first, second = contract(), copy.deepcopy(contract())
        cause = second["presentation"]["supporting_claims"][0]
        cause["source_text"] = "A compressor-related issue may cause poor cooling."
        cause["text"] = "Possible documented cause: " + cause["source_text"]
        approved = [first, second]

        answer_mod = ModuleType("core.services.customer_answer")
        answer_mod._question_guard = Mock(return_value=None)
        answer_mod.build_customer_answer = Mock(side_effect=approved)
        answer_mod.exception_presentation = Mock(
            side_effect=AssertionError("No exception contract expected")
        )
        diagnostic_mod = ModuleType("core.services.customer_diagnostic")
        diagnostic_mod.check_for_safety_hazard = Mock(return_value=(False, ""))
        diagnostic_mod.check_for_vague_complaint = Mock(return_value=(False, "", []))
        rag_mod = ModuleType("core.services.rag")
        rag_mod.ask_rag = Mock(return_value={
            "engine": "extractive", "references": [],
            "retrieval_backend": "hybrid",
            "reranker_applied": True, "reranker_model": "servy_v5_bge_verified",
        })
        fake_modules = {
            "core.services.customer_answer": answer_mod,
            "core.services.customer_diagnostic": diagnostic_mod,
            "core.services.rag": rag_mod,
        }
        with patch.dict("sys.modules", fake_modules), \
             patch.object(demo, "_asset", return_value=(tenant, customer, asset)), \
             patch.object(demo, "plan_turn", return_value=None):
            result = demo._answer(60, question)
        self.assertEqual(result["response_type"], "DIAGNOSTIC_GUIDANCE")
        self.assertEqual([item["chunk_id"] for item in result["citations"]], [9])
        self.assertEqual(result["retrieval_attempts"], 2)
        self.assertEqual(rag_mod.ask_rag.call_count, 2)
        self.assertEqual(rag_mod.ask_rag.call_args.kwargs["top_k"], 8)
        self.assertEqual(answer_mod.build_customer_answer.call_count, 2)
        self.assertIn("compressor", rag_mod.ask_rag.call_args.kwargs["question"])
        self.assertFalse(result.get("customer_safe_actions"))

    def test_explain_each_cited_issue_runs_one_deeper_source_retrieval(self):
        from tools.test_servy_real_demo_dialogue import contract
        import copy

        asset = SimpleNamespace(
            id=60, name="Synthetic VRF Outdoor Module", model_number="VRF-1",
            asset_code="SYN-60", product=SimpleNamespace(name="Synthetic VRF Outdoor Module"),
        )
        tenant, customer = SimpleNamespace(id=273), SimpleNamespace(id=13)
        question = "ok could you explain each issue in detailed"
        history = ["The VRF outdoor module is not working.", "What are its possible causes?"]
        continuation = demo._sign_demo_context(
            60, 13, question, history[0], history, last_topic="causes",
        )
        first, second = contract(), copy.deepcopy(contract())
        new_source = "An operating-condition limit may cause the outdoor module to stop."
        second["presentation"]["citations"].append({
            "id": "s6", "document_id": 3, "chunk_id": 30,
            "title": "Synthetic authorized VRF manual", "heading": "Operating limits",
            "url": "/demo-api/source/?citation=synthetic-evidence-6",
        })
        second["presentation"]["supporting_claims"].append({
            "id": "c6", "text": "Possible documented cause: " + new_source,
            "source_text": new_source, "kind": "possible_cause", "evidence_ids": ["s6"],
        })
        approved = [first, second]

        answer_mod = ModuleType("core.services.customer_answer")
        answer_mod._question_guard = Mock(return_value=None)
        answer_mod.build_customer_answer = Mock(side_effect=approved)
        answer_mod.exception_presentation = Mock(
            side_effect=AssertionError("No exception contract expected")
        )
        diagnostic_mod = ModuleType("core.services.customer_diagnostic")
        diagnostic_mod.check_for_safety_hazard = Mock(return_value=(False, ""))
        diagnostic_mod.check_for_vague_complaint = Mock(return_value=(False, "", []))
        rag_mod = ModuleType("core.services.rag")
        rag_mod.ask_rag = Mock(return_value={
            "engine": "extractive", "references": [],
            "retrieval_backend": "hybrid", "reranker_applied": True,
            "reranker_model": "servy_v5_bge_verified",
        })
        fake_modules = {
            "core.services.customer_answer": answer_mod,
            "core.services.customer_diagnostic": diagnostic_mod,
            "core.services.rag": rag_mod,
        }
        with patch.dict("sys.modules", fake_modules), \
             patch.object(demo, "_asset", return_value=(tenant, customer, asset)), \
             patch.object(demo, "plan_turn", return_value=None):
            result = demo._answer(60, question, continuation=continuation)

        self.assertEqual(result["response_type"], "DIAGNOSTIC_GUIDANCE")
        self.assertIn(
            "An operating-condition limit",
            "\n".join(
                claim.get("text", "")
                for claim in result["presentation"]["supporting_claims"]
            ),
        )
        self.assertIn(30, [item["chunk_id"] for item in result["citations"]])
        self.assertEqual(result["retrieval_attempts"], 2)
        self.assertEqual(rag_mod.ask_rag.call_count, 2)
        self.assertIn(
            "detailed documented causes explanation",
            rag_mod.ask_rag.call_args_list[1].kwargs["question"],
        )
        self.assertEqual(rag_mod.ask_rag.call_args_list[1].kwargs["top_k"], 8)
        self.assertEqual(answer_mod.build_customer_answer.call_count, 2)

    def test_detailed_cause_followup_stays_scoped_for_all_three_assets(self):
        from tools.test_servy_real_demo_dialogue import contract
        import copy

        cases = (
            (59, "Split AC Unit 2T", "SKY-AST-00001", 14, 168),
            (60, "VRF Outdoor Module", "SKY-AST-00002", 13, 171),
            (61, "Ductable AC 5T", "SKY-AST-00003", 14, 174),
        )
        for asset_id, asset_name, asset_code, customer_id, document_id in cases:
            with self.subTest(asset=asset_code):
                asset = SimpleNamespace(
                    id=asset_id, name=asset_name, asset_code=asset_code,
                    model_number=f"SYN-{asset_id}",
                    product=SimpleNamespace(name=asset_name),
                )
                tenant = SimpleNamespace(id=273)
                customer = SimpleNamespace(id=customer_id)
                question = "Could you explain each issue in detail?"
                history = [
                    f"The {asset_name} is not working.",
                    "What are its possible causes?",
                ]
                continuation = demo._sign_demo_context(
                    asset_id, customer_id, question, history[0], history,
                    last_topic="causes",
                )
                first, second = contract(), copy.deepcopy(contract())
                for answer_contract in (first, second):
                    for citation in answer_contract["presentation"]["citations"]:
                        citation["document_id"] = document_id

                new_source = (
                    f"A documented operating-condition limit may affect the {asset_name}."
                )
                new_chunk_id = 6000 + asset_id
                second["presentation"]["citations"].append({
                    "id": "s6", "document_id": document_id,
                    "chunk_id": new_chunk_id,
                    "title": f"Synthetic authorized {asset_name} manual",
                    "heading": "Operating limits",
                    "url": "/demo-api/source/?citation=synthetic-evidence-6",
                })
                second["presentation"]["supporting_claims"].append({
                    "id": "c6",
                    "text": "Possible documented cause: " + new_source,
                    "source_text": new_source,
                    "kind": "possible_cause",
                    "evidence_ids": ["s6"],
                })

                answer_mod = ModuleType("core.services.customer_answer")
                answer_mod._question_guard = Mock(return_value=None)
                answer_mod.build_customer_answer = Mock(side_effect=[first, second])
                answer_mod.exception_presentation = Mock(
                    side_effect=AssertionError("No exception contract expected")
                )
                diagnostic_mod = ModuleType("core.services.customer_diagnostic")
                diagnostic_mod.check_for_safety_hazard = Mock(return_value=(False, ""))
                diagnostic_mod.check_for_vague_complaint = Mock(return_value=(False, "", []))
                rag_mod = ModuleType("core.services.rag")
                rag_mod.ask_rag = Mock(return_value={
                    "engine": "extractive", "references": [],
                    "retrieval_backend": "hybrid",
                    "reranker_applied": True,
                    "reranker_model": "servy_v5_bge_verified",
                })
                fake_modules = {
                    "core.services.customer_answer": answer_mod,
                    "core.services.customer_diagnostic": diagnostic_mod,
                    "core.services.rag": rag_mod,
                }

                with (
                    patch.dict("sys.modules", fake_modules),
                    patch.object(demo, "_asset", return_value=(tenant, customer, asset)),
                    patch.object(demo, "plan_turn", return_value=None),
                ):
                    result = demo._answer(
                        asset_id, question, continuation=continuation,
                    )

                self.assertEqual(result["response_type"], "DIAGNOSTIC_GUIDANCE")
                self.assertEqual(result["retrieval_attempts"], 2)
                self.assertEqual(rag_mod.ask_rag.call_count, 2)
                self.assertEqual(answer_mod.build_customer_answer.call_count, 2)
                self.assertIn(
                    "detailed documented causes explanation",
                    rag_mod.ask_rag.call_args_list[1].kwargs["question"],
                )
                self.assertIn(
                    new_source,
                    "\n".join(
                        claim.get("text", "")
                        for claim in result["presentation"]["supporting_claims"]
                    ),
                )
                self.assertIn(
                    new_chunk_id,
                    [citation["chunk_id"] for citation in result["citations"]],
                )
                for rag_call in rag_mod.ask_rag.call_args_list:
                    self.assertIs(rag_call.kwargs["asset"], asset)
                    self.assertIs(rag_call.kwargs["customer"], customer)
                    self.assertFalse(rag_call.kwargs["include_confidential"])
                    self.assertFalse(rag_call.kwargs["staff_mode"])
                self.assertEqual(
                    {citation["document_id"] for citation in result["citations"]},
                    {document_id},
                )

    def test_targeted_retry_has_no_unbounded_loop_when_both_passes_fail(self):
        from tools.test_servy_real_demo_dialogue import contract
        asset = SimpleNamespace(
            id=60, name="Synthetic Split AC", model_number="AC-60",
            asset_code="SYN-60", product=SimpleNamespace(name="Synthetic AC"),
        )
        tenant, customer = SimpleNamespace(id=273), SimpleNamespace(id=13)
        answer_mod = ModuleType("core.services.customer_answer")
        answer_mod._question_guard = Mock(return_value=None)
        answer_mod.build_customer_answer = Mock(side_effect=[contract(), contract()])
        answer_mod.exception_presentation = Mock(
            side_effect=AssertionError("No exception contract expected")
        )
        diagnostic_mod = ModuleType("core.services.customer_diagnostic")
        diagnostic_mod.check_for_safety_hazard = Mock(return_value=(False, ""))
        diagnostic_mod.check_for_vague_complaint = Mock(return_value=(False, "", []))
        rag_mod = ModuleType("core.services.rag")
        rag_mod.ask_rag = Mock(return_value={
            "engine": "extractive", "references": [],
            "retrieval_backend": "hybrid", "reranker_applied": True,
        })
        mods = {
            "core.services.customer_answer": answer_mod,
            "core.services.customer_diagnostic": diagnostic_mod,
            "core.services.rag": rag_mod,
        }
        with patch.dict("sys.modules", mods), \
             patch.object(demo, "_asset", return_value=(tenant, customer, asset)), \
             patch.object(demo, "plan_turn", return_value=None):
            result = demo._answer(60, "What causes involve the compressor?")
        self.assertEqual(result["response_type"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["retrieval_attempts"], 2)
        self.assertEqual(rag_mod.ask_rag.call_count, 2)
        self.assertEqual(answer_mod.build_customer_answer.call_count, 2)

    def test_latest_new_symptom_replaces_stale_first_symptom_in_retrieval(self):
        asset = SimpleNamespace(
            id=60, name="Synthetic Split AC", asset_code="TEST-60",
            model_number="AC-60", product=SimpleNamespace(name="Synthetic AC"),
        )
        history = [
            "The AC is overheating.",
            "The AC runs but is not cooling.",
        ]
        ctx, query = demo._demo_query_context(
            asset, "What possible causes are documented?",
            history[0], history,
        )
        self.assertTrue(ctx.is_followup)
        self.assertIn("not cooling", query)
        self.assertNotIn("overheating", query)
        self.assertEqual(ctx.original_complaint, history[-1])
        current, standalone = demo._demo_query_context(
            asset, "Now the AC is overheating.", history[0], history,
        )
        self.assertFalse(current.is_followup)
        self.assertEqual(standalone, "Now the AC is overheating.")
        self.assertEqual(current.original_complaint, "Now the AC is overheating.")
        self.assertNotIn("not cooling", current.reported_symptoms)



if __name__ == "__main__":
    unittest.main()
