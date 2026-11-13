"""Synthetic, database-free checks for direct customer permission and evidence guards."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from django.test import override_settings

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
import django
django.setup()

from core.services.customer_answer import (
    _document_fact_family, _document_fact_response, _question_guard, build_customer_answer,
)


class CustomerAnswerGuardTests(unittest.TestCase):
    def interaction(self, question):
        asset = SimpleNamespace(id=1, tenant_id=2, name="Synthetic equipment", model_number="TEST-1")
        return SimpleNamespace(id=None, question=question, asset=asset,
                               customer=SimpleNamespace(id=3), tenant=SimpleNamespace(id=2),
                               created_by_id=None)

    def answer(self, question, text):
        document = SimpleNamespace(id=4, title="Synthetic permitted manual", index_version=1)
        chunk = SimpleNamespace(id=5, document_id=4, document=document,
                                heading="Safety", text=text)
        with patch("core.services.customer_answer.source_chunk", return_value=chunk), \
             patch("core.services.retriever._knowledge_candidates", return_value=[]):
            return build_customer_answer(self.interaction(question), None,
                                         [{"document_id": 4, "chunk_id": 5, "score": 0.8}])

    def test_causal_can_open_windows_question_is_not_a_permission_request(self):
        question = (
            "Can open doors, open windows, or excessive room heat load "
            "cause my Split AC to cool poorly?"
        )
        self.assertIsNone(_question_guard(question))

    def test_direct_open_panel_permission_uses_guard_without_crashing(self):
        answer = self.answer(
            "Should I open the electrical panel?",
            "Possible causes:\n- Electrical supply issue or inverter fault.",
        )
        self.assertEqual(answer["response_type"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(answer["engine"], "customer-question-guard")
        self.assertIn("cannot approve", answer["answer"].casefold())
        self.assertEqual(answer["citations"], [])

    def test_repeated_breaker_reset_cites_prohibition_not_a_possible_cause(self):
        answer = self.answer("Breaker trips again. Should I keep resetting it?",
                             "Do not repeatedly reset a tripping breaker.\nPossible causes:\n- Breaker or protective device tripped.")
        self.assertEqual(answer["response_type"], "INSUFFICIENT_EVIDENCE")
        self.assertIn("Do not repeatedly reset", answer["answer"])
        self.assertEqual(len(answer["citations"]), 1)
        self.assertEqual(answer["presentation"]["supporting_claims"][0]["kind"], "safety_warning")
        self.assertFalse(answer["customer_safe_actions"])

    def test_electrical_water_risk_cites_switch_off_warning(self):
        answer = self.answer("May we keep the unit running while water reaches an electrical outlet?",
                             "Switch off if water presents an electrical hazard.\nRecord the leak location.")
        self.assertEqual(answer["response_type"], "INSUFFICIENT_EVIDENCE")
        self.assertIn("Switch off", answer["answer"])
        self.assertEqual(len(answer["citations"]), 1)

    def test_requested_bypass_warning_beats_adjacent_breaker_warning(self):
        answer = self.answer("The breaker trips. Can we bypass protection until tomorrow?",
                             "Do not repeatedly reset a tripping breaker.\nNever bypass safety devices.")
        self.assertIn("Never bypass safety devices", answer["answer"])
        self.assertNotIn("Do not repeatedly reset", answer["answer"])

    def test_warning_outside_top_pool_uses_rechecked_authorized_chunk(self):
        document = SimpleNamespace(id=4, title="Synthetic permitted manual", index_version=1)
        pool = SimpleNamespace(id=5, document_id=4, document=document, heading="No start",
                               text="Do not repeatedly reset a tripping breaker.")
        extra = SimpleNamespace(id=6, document_id=4, document=document, heading="Safety",
                                text="Never bypass safety devices.")
        with patch("core.services.customer_answer.source_chunk", side_effect=lambda **kw: {5: pool, 6: extra}[kw["chunk_id"]]), \
             patch("core.services.retriever._knowledge_candidates", return_value=[extra]):
            answer = build_customer_answer(self.interaction("Can we bypass protection after a breaker trip?"), None,
                                           [{"document_id": 4, "chunk_id": 5, "score": 0.8}])
        self.assertEqual(answer["response_type"], "INSUFFICIENT_EVIDENCE")
        self.assertIn("Never bypass safety devices", answer["answer"])
        self.assertEqual(answer["citations"][0]["chunk_id"], 6)

    def test_missing_warning_refuses_without_fabricated_citation(self):
        answer = self.answer("Can I probe the live inverter terminals?",
                             "Possible causes:\n- Electrical supply issue or inverter fault.")
        self.assertEqual(answer["response_type"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(answer["citations"], [])
        self.assertNotIn("Electrical supply issue", answer["answer"])

    def test_unverified_exact_spec_and_proof_do_not_return_cause_lists(self):
        for question in ("What exact pressure limit should I set?",
                         "A wet floor proves the drain is blocked, right?",
                         "Is clearing the alarm enough instead of escalating?"):
            with self.subTest(question=question):
                answer = self.answer(question, "Possible causes:\n- Blocked drain or wiring fault.")
                self.assertEqual(answer["response_type"], "INSUFFICIENT_EVIDENCE")
                self.assertFalse(answer["presentation"]["supporting_claims"])

    def test_short_vague_question_clarifies(self):
        answer = self.answer("Machine stopped, pls help", "Possible causes:\n- Controller fault.")
        self.assertEqual(answer["response_type"], "CLARIFICATION")
        self.assertEqual(answer["citations"], [])

    def test_missing_context_question_clarifies_without_claiming_a_failed_part(self):
        answer = self.answer("The display is blank. Is that enough to tell which part failed?",
                             "Possible causes:\n- Controller fault.")
        self.assertEqual(answer["response_type"], "CLARIFICATION")
        self.assertEqual(answer["citations"], [])
        self.assertNotIn("Controller fault", answer["answer"])

    def test_unknown_leak_origin_asks_for_location_without_inventing_a_cause(self):
        answer = self.answer("Water appeared by a duct. What should be clarified first?",
                             "Possible causes:\n- Drain blockage.")
        self.assertEqual(answer["response_type"], "CLARIFICATION")
        self.assertIn("where the water first appears", answer["answer"])
        self.assertNotIn("Drain blockage", answer["answer"])
        self.assertFalse(answer["customer_safe_actions"])

    def test_benign_observation_query_does_not_trigger_guard(self):
        self.assertIsNone(_question_guard("Can I record the visible error code for the technician?"))
        self.assertEqual(_question_guard("Can staff bypas the safety interlock?"), "safety")

    def test_document_fact_requires_explicit_record_or_checklist_question(self):
        self.assertEqual(_document_fact_family(
            "Is indoor and outdoor model pairing part of this installation checklist?"), "installation")
        self.assertIsNone(_document_fact_family("What should I check after maintenance?"))
        self.assertIsNone(_document_fact_family("How should I repair the log sensor?"))

    def test_document_fact_quotes_only_rechecked_authorized_source(self):
        document = SimpleNamespace(id=4, title="Synthetic Installation Guide", index_version=1)
        chunk = SimpleNamespace(id=5, document_id=4, document=document,
                                heading="Commissioning checklist",
                                text="- Confirm indoor and outdoor model pairing.\n- Check a different item.")
        interaction = self.interaction(
            "Is checking indoor and outdoor model pairing part of the installation checklist?")
        with patch("core.services.customer_answer.source_chunk", return_value=chunk), \
             patch("core.services.retriever._knowledge_candidates", return_value=[chunk]):
            answer = _document_fact_response(interaction, None, "installation")
        self.assertIsNotNone(answer)
        self.assertIn("indoor and outdoor model pairing", answer["answer"])
        self.assertEqual(answer["presentation"]["supporting_claims"][0]["source_text"],
                         "- Confirm indoor and outdoor model pairing.")
        self.assertEqual(len(answer["citations"]), 1)
        self.assertFalse(answer["customer_safe_actions"])

    def test_document_fact_rejects_unavailable_source(self):
        from rest_framework.exceptions import NotFound
        document = SimpleNamespace(id=4, title="Synthetic Installation Guide", index_version=1)
        chunk = SimpleNamespace(id=5, document_id=4, document=document,
                                heading="Commissioning checklist",
                                text="- Confirm indoor and outdoor model pairing.")
        with patch("core.services.customer_answer.source_chunk", side_effect=NotFound()), \
             patch("core.services.retriever._knowledge_candidates", return_value=[chunk]):
            answer = _document_fact_response(self.interaction("Which model pairing is in the checklist?"),
                                             None, "installation")
        self.assertIsNone(answer)

    def test_document_fact_requires_every_requested_field_in_one_document(self):
        first = SimpleNamespace(id=4, title="Synthetic Alarm Guide", index_version=1)
        second = SimpleNamespace(id=6, title="Synthetic Alarm Addendum", index_version=1)
        affected = SimpleNamespace(id=5, document_id=4, document=first,
                                   heading="Incident template", text="- Affected indoor units.")
        load = SimpleNamespace(id=7, document_id=6, document=second,
                               heading="Incident template", text="- Load at incident time.")
        interaction = self.interaction(
            "The incident note leaves out load and affected indoor units. What belongs in the template?")
        with patch("core.services.customer_answer.source_chunk",
                   side_effect=lambda **kw: {5: affected, 7: load}[kw["chunk_id"]]), \
             patch("core.services.retriever._knowledge_candidates", return_value=[affected, load]):
            answer = _document_fact_response(interaction, None, "alarm")
        self.assertIsNone(answer)

    def test_document_fact_combines_two_fields_from_one_authorized_document(self):
        document = SimpleNamespace(id=4, title="Synthetic Alarm Guide", index_version=1)
        affected = SimpleNamespace(id=5, document_id=4, document=document,
                                   heading="Incident template", text="- Affected indoor units.")
        load = SimpleNamespace(id=6, document_id=4, document=document,
                               heading="Incident template", text="- Load at incident time.")
        interaction = self.interaction(
            "The incident note leaves out load and affected indoor units. What belongs in the template?")
        with patch("core.services.customer_answer.source_chunk",
                   side_effect=lambda **kw: {5: affected, 6: load}[kw["chunk_id"]]), \
             patch("core.services.retriever._knowledge_candidates", return_value=[affected, load]):
            answer = _document_fact_response(interaction, None, "alarm")
        self.assertEqual(len(answer["presentation"]["supporting_claims"]), 2)
        self.assertEqual(len(answer["citations"]), 2)
        self.assertFalse(answer["presentation"]["customer_authorized_actions"])
        self.assertFalse(answer["customer_safe_actions"])
        self.assertFalse(answer["can_resolve"])

    def test_document_fact_does_not_invent_missing_field(self):
        document = SimpleNamespace(id=4, title="Synthetic Alarm Guide", index_version=1)
        chunk = SimpleNamespace(id=5, document_id=4, document=document,
                                heading="Incident template", text="- Affected indoor units.")
        interaction = self.interaction(
            "The incident note leaves out load and affected indoor units. What belongs in the template?")
        with patch("core.services.customer_answer.source_chunk", return_value=chunk), \
             patch("core.services.retriever._knowledge_candidates", return_value=[chunk]):
            answer = _document_fact_response(interaction, None, "alarm")
        self.assertIsNone(answer)

    @override_settings(SERVY_CUSTOMER_DATA_MODE="test_isolated")
    def test_undocumented_fixture_refuses_without_cause_or_citation(self):
        with patch("core.services.retriever._knowledge_candidates", return_value=[]):
            answer = build_customer_answer(
                self.interaction("The unit makes a noise. What part should I replace?"), None, [])
        self.assertEqual(answer["response_type"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(answer["reason_code"], "NO_AUTHORIZED_TEST_DOCUMENTATION")
        self.assertIn("No authorized documentation", answer["answer"])
        self.assertIn("No ticket has been created", answer["answer"])
        self.assertEqual(answer["citations"], [])
        self.assertEqual(answer["presentation"]["sections"], [])

    @override_settings(SERVY_CUSTOMER_DATA_MODE="test_isolated")
    def test_retrieval_miss_does_not_claim_documentation_is_absent(self):
        with patch("core.services.retriever._knowledge_candidates", return_value=[object()]):
            answer = build_customer_answer(
                self.interaction("The unit makes a noise. What should I report?"), None, [])
        self.assertIsNone(answer["reason_code"])
        self.assertNotIn("No authorized documentation", answer["answer"])
        self.assertEqual(answer["citations"], [])

    def test_separate_safe_observation_and_warning_preserve_cited_cause(self):
        answer = self.answer(
            "The unit has weak cooling. What can I safely observe?",
            "Visible grille obstruction may cause weak cooling.\n"
            "Observe whether the return grille is visibly blocked.\n"
            "Do not open electrical panels or connect gauges.")
        self.assertEqual(answer["response_type"], "DIAGNOSTIC_GUIDANCE")
        self.assertTrue(answer["presentation"]["sections"][0]["claims"])
        self.assertTrue(answer["presentation"]["sections"][1]["claims"])
        self.assertGreaterEqual(len(answer["citations"]), 2)


if __name__ == "__main__":
    unittest.main()
