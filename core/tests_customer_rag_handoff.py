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


class CustomerLanguageTests(SimpleTestCase):
    def test_missing_details_are_expressed_once_without_manual_jargon(self):
        from core.services.cause_explanations import explanation_response
        items = [{"cause": "Example cause " + str(n), "cause_citation_ids": ["s" + str(n)],
                  "status": "insufficient_evidence", "evidence": [], "notes": []}
                 for n in (1, 2, 3)]
        result = explanation_response("Equipment", {"explanations": items,
            "citations": [{"id": "s" + str(n)} for n in (1, 2, 3)]})
        self.assertEqual(result["text"].count("don't have enough information"), 1)
        self.assertNotIn("manual", result["text"].casefold())
        self.assertEqual(len([s for s in result["segments"] if s["kind"] == "list_item"]), 3)
        self.assertIn("[3]", result["text"])


class ProductionServingPlanTests(SimpleTestCase):
    def test_match_plan_does_not_inherit_generic_complaint(self):
        from core.services.conversation_context import ConversationContext
        from core.services.customer_turn_plan import plan_technical_turn
        context = ConversationContext(conversation_uuid="", active_issue="poor cooling",
            original_complaint="My AC is not working properly", is_followup=True,
            active_observations=["The airflow is also very weak", "There is no water leakage"])
        plan = plan_technical_turn("Which of the possible causes matches my problem?", context)
        self.assertEqual(plan.intent, "match_causes")
        self.assertIn("poor cooling", plan.retrieval_question)
        self.assertIn("airflow", plan.retrieval_question)
        self.assertNotIn("water leakage", plan.retrieval_question)
        self.assertNotIn("not working", plan.retrieval_question)

    def test_document_cap_applies_to_ranked_order(self):
        from core.services.retriever import apply_document_diversity
        def row(doc):
            return SimpleNamespace(source_kind="knowledge_document", document_id=doc)
        ranked = [row(1), row(1), row(2)]
        self.assertEqual(apply_document_diversity(ranked, 1, 2), [ranked[0], ranked[2]])

    def test_retrieval_only_preserves_candidates_without_legacy_answer(self):
        from core.services.rag import ask_rag
        from core.services.retriever import RetrievedResultList
        with patch("core.services.rag.retrieve", return_value=RetrievedResultList()), \
             patch("core.services.rag.generate_answer") as legacy, \
             patch("core.services.customer_isolation.is_real_tenant", return_value=False):
            result = ask_rag(tenant=SimpleNamespace(id=1), question="poor cooling", asset=None,
                             retrieval_only=True)
        legacy.assert_not_called()
        self.assertEqual(result["engine"], "retrieval-only")
        self.assertEqual(result["references"], [])

    def test_final_gate_rejects_uncited_and_unrelated_causes(self):
        from core.services.customer_turn_plan import TechnicalTurnPlan, final_response_is_valid
        plan = TechnicalTurnPlan("which cause", "poor cooling", "poor cooling",
                                 "poor cooling", "match_causes")
        claim = {"kind": "possible_cause", "text": "Dirty filter.", "source_text": "Dirty filter.",
                 "source_heading": "3. SYMPTOM: POOR COOLING", "evidence_ids": ["s1"]}
        segment = {"kind": "list_item", "text": "Dirty filter.", "citation_ids": ["s1"]}
        contract = {"response_type": "DIAGNOSTIC_GUIDANCE", "presentation": {
            "citations": [{"id": "s1"}], "supporting_claims": [claim],
            "response": {"segments": [segment]}}}
        self.assertTrue(final_response_is_valid(contract, plan))
        claim["source_heading"] = "4. SYMPTOM: WATER LEAKAGE"
        self.assertFalse(final_response_is_valid(contract, plan))
        claim["source_heading"] = "3. SYMPTOM: POOR COOLING"
        segment["citation_ids"] = []
        self.assertFalse(final_response_is_valid(contract, plan))


class PlainLanguageAndSemanticIntentTests(SimpleTestCase):
    def test_plain_language_keeps_source_meaning_without_new_mechanisms(self):
        from core.services.customer_answer_generation import plain_cause_wording, _validate_paraphrase
        source = "Restricted airflow or blocked outdoor unit clearance."
        wording = plain_cause_wording(source)
        self.assertEqual(wording, "Limited air movement or something blocking the space around the outdoor unit.")
        self.assertTrue(_validate_paraphrase(source, "Possible cause: " + wording, "possible_cause",
                                             plain_language=True))
        self.assertFalse(_validate_paraphrase(source, "Possible cause: " + wording + " The compressor has failed.",
                                              "possible_cause", plain_language=True))
        self.assertFalse(_validate_paraphrase(source, "Possible cause: " + wording, "possible_cause"))

    def test_semantic_model_cannot_create_an_intent_or_query(self):
        from core.services.customer_semantic_intent import validate_route
        self.assertEqual(validate_route({"intent": "next_step"}), "next_step")
        for value in ({"intent": "repair"}, {"intent": "causes", "query": "invented"},
                      {"intent": []}, "next_step"):
            self.assertEqual(validate_route(value), "unknown")

    def test_request_guard_survives_planning(self):
        from core.services.conversation_context import ConversationContext
        from core.services.customer_turn_plan import plan_technical_turn
        context = ConversationContext(conversation_uuid="", active_issue="poor cooling", is_followup=True)
        request = "Can I open the electrical panel to check this?"
        plan = plan_technical_turn(request, context)
        self.assertEqual(plan.evidence_question, request)


class ExplanationContractRegressionTests(SimpleTestCase):
    def test_explanation_contract_returns_cited_answer_in_both_modes(self):
        from core.services.customer_conversation import _explanation_contract
        from django.test import override_settings
        source = "Restricted airflow or blocked outdoor unit clearance."
        for mode in (None, "simpler"):
            with self.subTest(mode=mode), override_settings(SERVY_CUSTOMER_ANSWER_GENERATION="disabled"):
                built = {"sections": [], "citations": [{"id": "s1"}],
                         "supporting_claims": [{"id": "c1", "kind": "possible_cause",
                             "text": source, "source_text": source, "evidence_ids": ["s1"]}],
                         "explanations": [{"cause": source, "cause_citation_ids": ["s1"],
                             "status": "insufficient_evidence", "evidence": [], "notes": []}]}
                contract = _explanation_contract(SimpleNamespace(id=1, question="Explain it"), None,
                    SimpleNamespace(id=1, name="Equipment", model_number=""), built, mode=mode)
                self.assertEqual(contract["response_type"], "DIAGNOSTIC_GUIDANCE")
                self.assertIn("[1]", contract["answer"])
                self.assertIn("don't have enough information", contract["answer"])
                self.assertIn("limited air movement" if mode == "simpler" else source.casefold(),
                              contract["answer"].casefold())
