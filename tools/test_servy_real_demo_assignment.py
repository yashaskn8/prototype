"""Offline tests: internal-only answers, evidence states and issue escalation.

All retrieval, answer-contract and ticket-store operations are synthetic or
mocked. No company database, network or real ticket workspace is touched.
"""
import copy
import io
import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("LOCALAPPDATA", "/tmp")
from tools import servy_real_demo as demo
from tools import servy_real_demo_assignment as assignment
from tools.test_servy_real_demo_dialogue import MIXED_QUESTION, contract

from django.conf import settings

if not settings.configured:
    settings.configure(SECRET_KEY="synthetic-test-signing-key")

THREAD = "thread-synthetic-0001"
COMPLAINT = "My AC has poor cooling since yesterday"
CAUSE_QUESTION = "What causes are documented?"
DETAIL = "Blocked external clearance restricts heat rejection at the outdoor unit."


def _asset(asset_id=60, model="AC-60"):
    return SimpleNamespace(
        id=asset_id, name="Synthetic Split AC", model_number=model,
        asset_code=f"SYN-{asset_id}", product=SimpleNamespace(name="Synthetic AC"),
    )


def _with_claim(base, claim_id, text, kind, index, document_id=3):
    result = copy.deepcopy(base)
    presentation = result["presentation"]
    presentation["supporting_claims"].append({
        "id": claim_id, "source_text": text, "kind": kind,
        "text": ("Possible documented cause: " + text) if kind == "possible_cause" else text,
        "evidence_ids": [f"s{index}"],
    })
    presentation["citations"].append({
        "id": f"s{index}", "document_id": document_id, "chunk_id": 8 + index,
        "title": "Synthetic authorized cooling manual",
        "heading": "3. SYMPTOM: POOR COOLING",
        "url": f"/demo-api/source/?citation=synthetic-evidence-{index}",
    })
    return result


def _run(question, contracts, *, continuation=None, thread_id=None, asset=None,
         customer_id=13):
    """Run one customer turn with synthetic retrieval and answer contracts."""
    asset = asset or _asset()
    answer_mod = ModuleType("core.services.customer_answer")
    answer_mod._question_guard = Mock(return_value=None)
    answer_mod.build_customer_answer = Mock(
        side_effect=[copy.deepcopy(item) for item in contracts])
    answer_mod.exception_presentation = Mock(side_effect=lambda value: dict(value))
    diagnostic_mod = ModuleType("core.services.customer_diagnostic")
    diagnostic_mod.check_for_safety_hazard = Mock(return_value=(False, ""))
    diagnostic_mod.check_for_vague_complaint = Mock(return_value=(False, "", []))
    rag_mod = ModuleType("core.services.rag")
    rag_mod.ask_rag = Mock(return_value={
        "engine": "extractive", "references": [], "retrieval_backend": "hybrid",
        "reranker_applied": True, "reranker_model": "servy_v5_bge_verified",
    })
    modules = {
        "core.services.customer_answer": answer_mod,
        "core.services.customer_diagnostic": diagnostic_mod,
        "core.services.rag": rag_mod,
    }
    tenant, customer = SimpleNamespace(id=273), SimpleNamespace(id=customer_id)
    with patch.dict("sys.modules", modules), \
            patch.object(demo, "_asset", return_value=(tenant, customer, asset)), \
            patch.object(demo, "plan_turn", return_value=None):
        result = demo._answer(asset.id, question, continuation=continuation,
                              thread_id=thread_id)
    return result, rag_mod.ask_rag, answer_mod.build_customer_answer


def _cause_turn(thread_id=None, base=None):
    """First two turns: a complaint, then the documented cause list."""
    history = [COMPLAINT]
    continuation = demo._sign_demo_context(
        60, 13, COMPLAINT, COMPLAINT, [], thread_id=thread_id)
    base = base or contract()
    result, _, _ = _run(CAUSE_QUESTION, [base, base], continuation=continuation,
                        thread_id=thread_id)
    return result, history


def _claims(result):
    return result["presentation"]["supporting_claims"]


class EvidenceStateTests(unittest.TestCase):
    def test_sufficient_internal_evidence_gives_a_cited_answer_without_escalation(self):
        result, _ = _cause_turn()
        self.assertEqual(result["response_type"], "DIAGNOSTIC_GUIDANCE")
        self.assertEqual(result["evidence_state"], "SUFFICIENT")
        self.assertIsNone(result["issue_assignment"])
        self.assertTrue(result["citations"])
        cited = {c["id"] for c in result["presentation"]["citations"]}
        for claim in _claims(result):
            self.assertEqual(len(claim["evidence_ids"]), 1)
            self.assertIn(claim["evidence_ids"][0], cited)

    def test_multiple_documented_causes_stay_separate_and_unconfirmed(self):
        result, _ = _cause_turn()
        causes = [c for c in _claims(result) if c["kind"] == "possible_cause"]
        self.assertEqual([c["source_text"] for c in causes],
                         ["Reduced airflow.", "Blocked external clearance."])
        self.assertEqual(len({c["evidence_ids"][0] for c in causes}), 2)
        self.assertIn("not", result["presentation"]["message"].casefold())
        self.assertNotIn("diagnosis is", result["presentation"]["message"].casefold())

    def test_no_relevant_internal_document_triggers_assignment(self):
        result, ask_rag, _ = _run("What causes involve the compressor?",
                                  [contract(), contract()])
        self.assertEqual(result["response_type"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["evidence_state"], "INSUFFICIENT")
        self.assertEqual(result["citations"], [])
        block = result["issue_assignment"]
        self.assertTrue(block["required"])
        self.assertEqual(block["status"], "AWAITING_CUSTOMER_CONFIRMATION")
        self.assertEqual(block["reason"], "NO_SUFFICIENT_INTERNAL_EVIDENCE")
        self.assertIn("compressor", block["unanswered_question"])
        self.assertIn("Model: AC-60", block["description"])
        self.assertIn("no relevant approved passage", block["description"])
        self.assertLessEqual(ask_rag.call_count, 2)

    def test_partial_evidence_answers_the_supported_part_and_escalates_the_rest(self):
        result, _, _ = _run(MIXED_QUESTION, [contract(), contract()])
        self.assertEqual(result["response_type"], "DIAGNOSTIC_GUIDANCE")
        self.assertEqual(result["evidence_state"], "PARTIAL")
        self.assertEqual([c["source_text"] for c in _claims(result)],
                         ["Blocked external clearance."])
        titles = [s["title"] for s in result["presentation"]["sections"]]
        self.assertIn("What I could not verify", titles)
        block = result["issue_assignment"]
        self.assertEqual(block["status"], "AWAITING_CUSTOMER_CONFIRMATION")
        self.assertEqual(block["reason"], "PARTIAL_INTERNAL_EVIDENCE")
        self.assertIn("Blocked external clearance.", block["description"])
        self.assertIn("cannot establish", block["description"])

    def test_undocumented_fault_code_is_never_explained(self):
        with patch.object(demo, "explain_claims",
                          side_effect=AssertionError("no generation")):
            result, _, _ = _run("What does E-09 mean?", [contract(), contract()])
        self.assertEqual(result["evidence_state"], "INSUFFICIENT")
        self.assertEqual(_claims(result), [])
        self.assertNotIn("E-04", json.dumps(result["presentation"]))
        self.assertTrue(result["issue_assignment"]["required"])

    def test_fault_code_documented_only_for_another_model_is_not_reused(self):
        # The authorized scope of asset 59 returns no E-04 statement.
        other_model = contract()
        other_model["presentation"]["supporting_claims"] = [
            c for c in other_model["presentation"]["supporting_claims"]
            if "E-04" not in c["source_text"]]
        asset = _asset(59, "AC-59")
        result, ask_rag, _ = _run("What does E-04 mean?", [other_model, other_model],
                                  asset=asset, customer_id=14)
        self.assertEqual(result["evidence_state"], "INSUFFICIENT")
        self.assertNotIn("indicates reduced airflow", json.dumps(result))
        for call in ask_rag.call_args_list:
            self.assertIs(call.kwargs["asset"], asset)
        self.assertIn("Model: AC-59", result["issue_assignment"]["description"])

    def test_conflicting_internal_documents_escalate_instead_of_choosing(self):
        conflicting = _with_claim(contract(), "c6", "E-04 indicates a sensor fault.",
                                  "information", 6, document_id=4)
        result, _, _ = _run("What does E-04 mean?", [conflicting, conflicting])
        self.assertEqual(result["response_type"], "DIAGNOSTIC_GUIDANCE")
        self.assertEqual(result["evidence_state"], "PARTIAL")
        self.assertEqual({c["source_text"] for c in _claims(result)},
                         {"E-04 indicates reduced airflow.", "E-04 indicates a sensor fault."})
        self.assertEqual(result["issue_assignment"]["reason"],
                         "CONFLICTING_INTERNAL_DOCUMENTS")
        self.assertIn("differently", json.dumps(result["presentation"]["sections"]))

    def test_same_statement_in_two_documents_is_not_a_conflict(self):
        same = _with_claim(contract(), "c6", "E-04 indicates reduced airflow.",
                           "information", 6, document_id=4)
        self.assertIsNone(assignment.code_conflict(same["presentation"], "What does E-04 mean?"))


class CauseFollowUpTests(unittest.TestCase):
    def test_second_cause_is_resolved_and_additional_internal_passage_is_retrieved(self):
        first, history = _cause_turn()
        detailed = _with_claim(contract(), "c6", DETAIL, "information", 6)
        result, ask_rag, _ = _run(
            "Explain the second cause in more detail", [detailed, detailed],
            continuation=first["continuation"])
        self.assertEqual(result["response_type"], "DIAGNOSTIC_GUIDANCE")
        self.assertEqual([c["source_text"] for c in _claims(result)],
                         ["Blocked external clearance.", DETAIL])
        self.assertNotIn("Reduced airflow.", json.dumps(result["presentation"]["sections"]))
        self.assertEqual(result["evidence_state"], "SUFFICIENT")
        self.assertIsNone(result["issue_assignment"])
        self.assertIn(14, [c["chunk_id"] for c in result["citations"]])
        query = ask_rag.call_args_list[0].kwargs["question"]
        self.assertIn("clearance", query)
        self.assertIn("AC-60", query)
        self.assertIn("cooling", query.casefold())
        self.assertLessEqual(ask_rag.call_count, 2)

    def test_no_additional_explanation_escalates_instead_of_rewording(self):
        first, _ = _cause_turn()
        with patch.object(demo, "explain_claims",
                          side_effect=AssertionError("no rewording")):
            result, _, _ = _run(
                "Explain the second cause in more detail", [contract(), contract()],
                continuation=first["continuation"])
        self.assertEqual([c["source_text"] for c in _claims(result)],
                         ["Blocked external clearance."])
        self.assertEqual(result["evidence_state"], "PARTIAL")
        self.assertIn("Evidence limitation",
                      [s["title"] for s in result["presentation"]["sections"]])
        block = result["issue_assignment"]
        self.assertEqual(block["status"], "AWAITING_CUSTOMER_CONFIRMATION")
        self.assertIn("refers to documented cause: Blocked external clearance.",
                      block["description"])
        self.assertIn("Explain the second cause", block["unanswered_question"])

    def test_explain_each_without_new_passages_keeps_limitation_and_escalates(self):
        first, _ = _cause_turn()
        with patch.object(demo, "explain_claims",
                          side_effect=AssertionError("no rewording")):
            result, _, _ = _run("Explain each of those causes in detail",
                                [contract(), contract()],
                                continuation=first["continuation"])
        self.assertEqual(result["evidence_state"], "PARTIAL")
        self.assertTrue(result["issue_assignment"]["required"])

    def test_reference_beyond_the_listed_causes_asks_which_one(self):
        first, _ = _cause_turn()
        result, ask_rag, _ = _run("Explain the fifth cause", [contract()],
                                  continuation=first["continuation"])
        self.assertEqual(result["response_type"], "CLARIFICATION")
        self.assertIn("2 documented cause", json.dumps(result["presentation"]))
        self.assertEqual(ask_rag.call_count, 0)
        self.assertIsNone(result["issue_assignment"])

    def test_cause_list_is_signed_and_scoped_to_the_same_asset_conversation(self):
        first, _ = _cause_turn()
        self.assertEqual(demo._load_demo_cause_state(first["continuation"]),
                         ["Reduced airflow.", "Blocked external clearance."])
        with self.assertRaises(ValueError):
            _run("Explain the second cause", [contract()],
                 continuation=first["continuation"], asset=_asset(59), customer_id=14)
        self.assertIsNone(assignment.cause_reference("Explain the second cause", []))

    def test_already_supplied_onset_is_not_requested_again(self):
        first, _ = _cause_turn()
        follow_up = first["presentation"]["follow_up_question"] or ""
        self.assertNotIn("first notice", follow_up)
        self.assertNotIn("suddenly or gradually", follow_up)


class SourceBoundaryTests(unittest.TestCase):
    def test_customer_answer_path_never_calls_external_research(self):
        self.assertFalse(hasattr(demo, "search_model_matched_sources"))
        source = Path(demo.__file__).read_text(encoding="utf-8")
        self.assertNotIn("servy_real_demo_web_research", source)
        self.assertNotIn("online_research", source)
        blocked = AssertionError("external research is prohibited")
        with patch("tools.servy_real_demo_web_research.search_model_matched_sources",
                   side_effect=blocked), \
                patch("requests.get", side_effect=blocked), \
                patch("requests.post", side_effect=blocked), \
                patch.dict(os.environ, {"SERVY_BRAVE_SEARCH_API_KEY": "synthetic",
                                        "SERVY_WEB_RESEARCH_ALLOWED_DOMAINS": "example.test"}):
            answered, _ = _cause_turn()
            unanswered, _, _ = _run("What causes involve the compressor?",
                                    [contract(), contract()])
        for result in (answered, unanswered):
            self.assertNotIn("online_research", result["presentation"])

    def test_technician_only_evidence_never_reaches_answer_or_issue_description(self):
        secret = "Technician only: bypass the pressure switch to test the compressor."
        leaking = _with_claim(contract(), "c9", secret, "technician_procedure", 9)
        result, ask_rag, _ = _run(MIXED_QUESTION, [leaking, leaking])
        self.assertNotIn("pressure switch", json.dumps(result))
        for call in ask_rag.call_args_list:
            self.assertFalse(call.kwargs["staff_mode"])
            self.assertFalse(call.kwargs["include_confidential"])


class AssignmentWorkflowTests(unittest.TestCase):
    def _post(self, fields, store):
        handler = object.__new__(demo.Handler)
        handler.path = "/demo-api/tickets/create-local/"
        payload = json.dumps({"synthetic": "form"}).encode()
        handler.headers = {"Content-Length": str(len(payload)),
                           "Content-Type": "application/json"}
        handler.rfile = io.BytesIO(payload)
        handler._authorized = lambda: True
        handler._send = lambda status, value: (status, value)
        with patch.object(demo, "read_only_database", return_value=nullcontext()), \
                patch.object(demo, "_local_ticket_fields", return_value=fields), \
                patch.object(demo, "_store_local_ticket", side_effect=store) as stored:
            status, response = handler.do_POST()
        return status, response, stored

    def test_answer_never_claims_an_assignment_before_the_workflow_confirms(self):
        result, _, _ = _run("What causes involve the compressor?",
                            [contract(), contract()], thread_id=THREAD)
        block = result["issue_assignment"]
        self.assertEqual(block["status"], "AWAITING_CUSTOMER_CONFIRMATION")
        self.assertIsNone(block["existing_ticket"])
        self.assertIn("Nothing has been submitted yet", block["notice"])
        self.assertNotIn("LOCAL-", json.dumps(result))
        self.assertTrue(result["ticket_context"])

    def test_assignment_success_is_reported_only_from_the_store_result(self):
        status, response, stored = self._post(
            {"asset_id": 60}, lambda *_: {"id": "LOCAL-SYNTHETIC", "status": "open"})
        self.assertEqual(status, 201)
        self.assertEqual(response["ticket"]["id"], "LOCAL-SYNTHETIC")
        self.assertNotIn("duplicate", response)
        self.assertEqual(stored.call_count, 1)

    def test_assignment_failure_is_reported_honestly(self):
        def fail(*_):
            raise OSError("synthetic store failure")
        status, response, _ = self._post({"asset_id": 60}, fail)
        self.assertEqual(status, 503)
        self.assertNotIn("ticket", response)
        self.assertIn("could not be saved", response["detail"])

    def test_already_assigned_issue_does_not_create_a_duplicate(self):
        key = assignment.issue_key(THREAD, "What causes involve the compressor?")
        fields = {
            "tenant_id": 273, "customer_id": 13, "customer_name": "Synthetic customer",
            "site_id": 7, "site_name": "Synthetic site", "asset_id": 60,
            "asset_name": "Synthetic asset", "asset_code": "SYN-60",
            "call_type": "Breakdown Service", "complaint_type": "Cooling / performance",
            "complaint_text": "Synthetic escalation", "contact_name": "",
            "severity": "Medium", "priority": "normal", "machine_status": "Intermittent",
            "status": "open", "technician_notes": "", "source_conversation_id": key,
            "created_by": "synthetic-test",
        }
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "technician-tickets.sqlite3"
            with patch.object(settings, "DATABASES", {"default": {
                    "ENGINE": "django.db.backends.mysql", "NAME": "unchanged"}},
                    create=True), \
                    patch.object(settings, "SERVY_LOCAL_TICKET_DB_PATH", database,
                                 create=True):
                self.assertIsNone(demo._existing_local_ticket(fields))
                self.assertFalse(database.exists())  # the lookup never creates a store
                ticket = demo._store_local_ticket(fields)

                status, response, stored = self._post(
                    fields, AssertionError("a duplicate ticket must not be stored"))
                self.assertEqual(status, 200)
                self.assertTrue(response["duplicate"])
                self.assertEqual(response["ticket"]["id"], ticket["id"])
                self.assertEqual(stored.call_count, 0)

                # A further follow-up on the same unresolved issue reports the
                # existing assignment instead of asking for another ticket.
                result, _, _ = _run("What causes involve the compressor?",
                                    [contract(), contract()], thread_id=THREAD)
                block = result["issue_assignment"]
                self.assertEqual(block["status"], "ALREADY_ASSIGNED")
                self.assertEqual(block["existing_ticket"],
                                 {"id": ticket["id"], "status": "open"})
                self.assertIn("No duplicate ticket", block["notice"])

                # Another customer, asset or chat never sees that ticket.
                self.assertIsNone(assignment.find_open_ticket(273, 14, 60, key))
                self.assertIsNone(assignment.find_open_ticket(273, 13, 59, key))
                self.assertIsNone(assignment.find_open_ticket(
                    273, 13, 60, assignment.issue_key("thread-other-0002", "x")))

    def test_ticket_binding_is_signed_and_rejects_a_forged_issue_key(self):
        from django.core import signing
        import hashlib
        question = "Synthetic local ticket handoff?"
        site = SimpleNamespace(id=7, name="Synthetic site", customer_id=13, tenant_id=273)
        asset = SimpleNamespace(id=60, site_id=7, tenant_id=273, site=site,
                                name="Synthetic asset", asset_code="SYN-60")
        scope = (SimpleNamespace(id=273), SimpleNamespace(id=13, name="Synthetic"), asset)

        def body(issue):
            token = signing.dumps({
                "asset": 60, "customer": 13, "issue": issue,
                "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            }, salt="servy-real-demo-ticket-preview-v1")
            return {"ticket_context": token, "question": question,
                    "call_type": "Breakdown Service",
                    "complaint_type": "Cooling / performance",
                    "complaint_text": "Synthetic", "contact_name": "",
                    "severity": "Medium", "priority": "normal",
                    "machine_status": "Intermittent"}

        key = assignment.issue_key(THREAD, COMPLAINT)
        with patch.object(demo, "_asset", return_value=scope):
            self.assertEqual(
                demo._local_ticket_fields(body(key))["source_conversation_id"], key)
            self.assertEqual(
                demo._local_ticket_fields(body(""))["source_conversation_id"], "")
            with self.assertRaises(ValueError):
                demo._local_ticket_fields(body("another-conversation"))


if __name__ == "__main__":
    unittest.main()
