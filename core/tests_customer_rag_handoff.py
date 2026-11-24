"""Focused RAG handoff and evidence-scope regressions (synthetic only)."""
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase

from core.services.rag import customer_evidence_candidates, NO_AUTHORIZED_ORIGINAL_DOCUMENTATION
from core.services.customer_answer import customer_evidence_question
from core.services.retriever import intact_customer_candidates


class CustomerRagHandoffTests(SimpleTestCase):
    def test_full_ranked_candidates_reach_the_validator(self):
        first, second = {"chunk_id": 1}, {"chunk_id": 2}
        result = {"references": [first], "retrieved_passages": [first, second]}
        self.assertEqual(customer_evidence_candidates(result), [first, second])

    def test_explicit_empty_and_blocked_results_fail_closed(self):
        refs = [{"chunk_id": 1}]
        self.assertEqual(customer_evidence_candidates({"references": refs, "retrieved_passages": []}), [])
        for guard in ({"reason_code": NO_AUTHORIZED_ORIGINAL_DOCUMENTATION}, {"engine": "policy-guard"}):
            self.assertEqual(customer_evidence_candidates({**guard, "references": refs, "retrieved_passages": refs}), [])
        self.assertEqual(customer_evidence_candidates({"references": refs}), refs)

    def test_explicit_question_does_not_inherit_previous_symptom(self):
        ctx = SimpleNamespace(active_issue="Water leakage", answers_pending_question=False)
        self.assertEqual(customer_evidence_question("What causes poor cooling?", ctx), "What causes poor cooling?")

    def test_reply_and_ordinal_have_explicit_evidence_scope(self):
        ctx = SimpleNamespace(active_issue="Noise at startup", answers_pending_question=True)
        self.assertEqual(customer_evidence_question("Only at startup.", ctx), "Only at startup.; Noise at startup")
        reference = {"selected": {"heading": "Cooling", "source_text": "Restricted airflow."}}
        self.assertEqual(customer_evidence_question("The second cause?", ctx, reference), "Cooling: Restricted airflow.")

    def test_invalid_original_cannot_consume_ranking_slots(self):
        tenant = SimpleNamespace(id=1)
        good = SimpleNamespace(id=1)
        bad = SimpleNamespace(id=2)
        chunks = [SimpleNamespace(document=bad), SimpleNamespace(document=good), SimpleNamespace(document=bad)]
        with patch("core.services.customer_isolation.get_customer_data_mode", return_value="verified_real_only"), \
                patch("core.services.customer_isolation.is_real_tenant", return_value=True), \
                patch("core.services.customer_isolation.original_attachment_is_intact", side_effect=lambda doc: doc.id == 1) as intact:
            self.assertEqual(intact_customer_candidates(chunks, tenant=tenant, staff_mode=False), [chunks[1]])
            self.assertEqual(intact.call_count, 2)
            self.assertEqual(intact_customer_candidates(chunks, tenant=tenant, staff_mode=True), chunks)
            self.assertEqual(intact.call_count, 2)


class CanonicalCustomerCitationTests(SimpleTestCase):
    def test_saved_references_match_visible_answer(self):
        from core.services.customer_conversation import customer_answer_citations
        first, second = {"id": "s1"}, {"id": "s2"}
        contract = {"citations": [first, second], "presentation": {
            "citations": [first, second], "response": {
                "segments": [{"kind": "claim", "citation_ids": ["s2"]}]}}}
        self.assertEqual(customer_answer_citations(contract), [second])
        contract["presentation"]["response"]["segments"] = [{"kind": "text", "citation_ids": []}]
        self.assertEqual(customer_answer_citations(contract), [])

    def test_empty_authorized_presentation_does_not_restore_legacy_refs(self):
        from core.services.customer_conversation import customer_answer_citations
        self.assertEqual(customer_answer_citations({"citations": [{"id": "s1"}],
            "presentation": {"response_type": "INSUFFICIENT_EVIDENCE", "citations": []}}), [])
