"""Offline safety checks for the separate real-data demo boundary."""
import unittest
import hashlib
import io
import json
import os
import base64
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
        self.assertIn("GET /demo-api/source/ 200", line)
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
