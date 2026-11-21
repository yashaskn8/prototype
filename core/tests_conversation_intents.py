"""Unit tests for conversational intent helpers (no database, no models)."""
from django.test import SimpleTestCase

from core.services.conversation_context import (
    conversation_title, is_default_title, is_explain_all_request, is_greeting, is_supported_answer, listed_causes,
    split_pasted_history)
from core.services.turn_diagnostics import summarize_turn


def _contract(causes, response_type="DIAGNOSTIC_GUIDANCE"):
    claims = [{"kind": "possible_cause", "source_text": c, "text": "Possible documented cause: " + c,
               "source_heading": "5. SYMPTOM: X", "evidence_ids": [f"s{i + 1}"]} for i, c in enumerate(causes)]
    return {"presentation": {
        "response_type": response_type, "supporting_claims": claims,
        "sections": [{"number": 1, "title": "What could be wrong?", "claims": claims}],
        "citations": [{"id": f"s{i + 1}", "document_id": 7, "chunk_id": 70 + i} for i in range(len(causes))]}}


class GreetingTests(SimpleTestCase):
    def test_greetings_and_thanks(self):
        for text in ("Hi.", "hello", "Hey there!", "Good morning", "thanks", "Thank you very much!", "ok thanks"):
            with self.subTest(text=text):
                self.assertTrue(is_greeting(text))

    def test_requests_are_not_greetings(self):
        for text in ("Hi, my unit is not cooling.", "Hello what causes ice on the coil?",
                     "Thanks, now explain each issue.", "", "high pressure alarm"):
            with self.subTest(text=text):
                self.assertFalse(is_greeting(text))


class ExplainAllTests(SimpleTestCase):
    def test_explain_all_phrasings(self):
        for text in ("Explain each issue.", "Explain each issue mentioned above.", "explain them all",
                     "Please explain these issues.", "Can you explain all the causes?",
                     "Explain the causes listed above", "describe each cause", "explain each of them"):
            with self.subTest(text=text):
                self.assertTrue(is_explain_all_request(text))

    def test_single_or_unrelated_requests(self):
        for text in ("Explain the second cause.", "What causes ice on the coil?", "Explain how to reset it",
                     "Each filter should be cleaned monthly."):
            with self.subTest(text=text):
                self.assertFalse(is_explain_all_request(text))


class PastedHistoryTests(SimpleTestCase):
    previous = ["For outdoor module not working on the VRF, the manual lists these possible causes:\n"
                "- Power supply to the outdoor module is interrupted. [1]\n"
                "- High-pressure protection has tripped. [2]\n"
                "These are documented possibilities, not a confirmed diagnosis of your unit."]

    def test_pasted_answer_is_separated_from_the_request(self):
        message = ("For outdoor module not working on the VRF, the manual lists these possible causes:\n"
                   "- Power supply to the outdoor module is interrupted. [1]\n"
                   "- High-pressure protection has tripped. [2]\n\n"
                   "Please explain these issues.")
        request, quoted = split_pasted_history(message, self.previous)
        self.assertEqual(request, "Please explain these issues.")
        self.assertEqual(quoted, ["Power supply to the outdoor module is interrupted.",
                                  "High-pressure protection has tripped."])

    def test_single_line_message_is_unchanged(self):
        self.assertEqual(split_pasted_history("Explain each issue.", self.previous), ("Explain each issue.", []))

    def test_multiline_message_without_quotes_is_unchanged(self):
        message = "The outdoor unit stopped.\nThe display shows nothing."
        self.assertEqual(split_pasted_history(message, self.previous), (message, []))


class SupportedAnswerTests(SimpleTestCase):
    def test_only_cited_guidance_counts(self):
        self.assertTrue(is_supported_answer(_contract(["A cause."])))
        self.assertFalse(is_supported_answer(_contract(["A cause."], "INSUFFICIENT_EVIDENCE")))
        self.assertFalse(is_supported_answer({"presentation": {"response_type": "GREETING"}}))
        self.assertFalse(is_supported_answer({}))

    def test_listed_causes_keep_order_and_source_location(self):
        causes = listed_causes(_contract(["First cause here.", "Second cause here."]))
        self.assertEqual([c["source_text"] for c in causes], ["First cause here.", "Second cause here."])
        self.assertEqual([c["chunk_id"] for c in causes], [70, 71])


class TurnDiagnosticsTests(SimpleTestCase):
    def test_no_retrieval_is_reported_as_not_run(self):
        events = [{"stage": "customer_rag_execution", "status": "skipped", "metrics": {}},
                  {"stage": "customer_qwen_generation", "status": "skipped", "metrics": {}}]
        summary = summarize_turn(events, {"response_type": "GREETING"})
        self.assertEqual(summary["retrieval"]["passes"], 0)
        self.assertIsNone(summary["retrieval"]["dense_candidates"])
        self.assertEqual(summary["qwen"], {"ran": False, "status": "skipped",
                                           "eligible_claims": None, "accepted_claims": None})

    def test_counts_and_cited_evidence_come_from_execution(self):
        events = [{"stage": "rank_fusion_summary", "metrics": {"dense_count": 4, "sparse_count": 3,
                                                                "retrieved_count": 5, "retrieval_backend": "hybrid"}},
                  {"stage": "rank_fusion_summary", "metrics": {"dense_count": 2, "sparse_count": 1,
                                                                "retrieved_count": 2, "retrieval_backend": "hybrid"}},
                  {"stage": "customer_qwen_generation", "status": "completed", "metrics": {}},
                  {"stage": "customer_qwen_outcome", "metrics": {"generation_status": "applied",
                                                                 "candidate_count": 3, "supported_count": 2}}]
        contract = {"presentation": {
            "citations": [{"id": "s1", "title": "Manual", "heading": "6. X"}, {"id": "s2", "title": "M", "heading": ""}],
            "response": {"segments": [{"kind": "list_item", "text": "x", "citation_ids": ["s1"]}],
                         "explanations": [{"status": "insufficient_evidence"}]}}}
        summary = summarize_turn(events, contract)
        self.assertEqual(summary["retrieval"]["passes"], 2)
        self.assertEqual(summary["retrieval"]["dense_candidates"], 6)
        self.assertEqual(summary["retrieval"]["sparse_candidates"], 4)
        self.assertEqual([c["id"] for c in summary["cited_evidence"]], ["s1"])
        self.assertIn("cause_without_explaining_evidence", summary["rejection_reasons"])
        self.assertEqual(summary["qwen"]["accepted_claims"], 2)


class ConversationTitleTests(SimpleTestCase):
    def test_titles_use_the_customers_own_words(self):
        for question, title in (
                ("What are the possible causes of my Split AC not working?", "Split AC not working"),
                ("Why is the VRF showing an error code?", "VRF error code"),
                ("Why is the AC leaking water?", "AC water leakage"),
                ("What causes weak airflow?", "Weak airflow"),
                ("What does a low-pressure fault mean?", "Low-pressure fault"),
                ("The display shows E4 after startup", "Display E4 after startup")):
            with self.subTest(question=question):
                self.assertEqual(conversation_title(question), title)

    def test_contact_details_are_removed_and_length_is_bounded(self):
        title = conversation_title("AC stops, call +1 555 0100 2222 or mail me at a.person@example.com " + "x " * 60)
        self.assertNotIn("555", title)
        self.assertNotIn("@", title)
        self.assertLessEqual(len(title), 60)

    def test_only_default_titles_are_replaced(self):
        self.assertTrue(is_default_title("Inquiry for Split AC Unit 2T"))
        self.assertTrue(is_default_title("Support Inquiry"))
        self.assertFalse(is_default_title("Kitchen unit noise"))
