"""Conversational reply composition: natural structure, nothing added, every claim cited."""
from types import SimpleNamespace

from django.test import SimpleTestCase

from core.services.conversation_context import topic_shift_query
from core.services.customer_response_composer import LIMITATION, compose_response, render_text


def _claim(cid, kind, source, heading="3. SYMPTOM: POOR COOLING", evidence=None):
    text = ("Possible documented cause: " + source) if kind == "possible_cause" else source
    claim = {"id": cid, "kind": kind, "text": text, "source_text": source, "evidence_ids": [evidence or "s" + cid[1:]]}
    if kind == "possible_cause":
        claim["source_heading"] = heading
    return claim


def _contract(causes=(), facts=(), checks=()):
    claims = list(causes) + list(facts) + list(checks)
    citations = [{"id": c["evidence_ids"][0], "title": "Manual", "heading": c.get("source_heading", "")} for c in claims]
    sections = [
        {"number": 1, "title": "What could be wrong?", "claims": list(causes) + list(facts)},
        {"number": 2, "title": "What can I safely check or report?", "claims": list(checks) or
         [{"kind": "uncertainty", "text": "No customer procedure has been approved.", "evidence_ids": []}]},
        {"number": 3, "title": "What is still unknown?", "claims": [{"kind": "uncertainty", "text": "x", "evidence_ids": []}]},
        {"number": 4, "title": "What should I do next?", "claims": [{"kind": "next_step", "text": "y", "evidence_ids": []}]},
    ]
    return {"answer": "1. What could be wrong?\n...", "presentation": {
        "response_type": "DIAGNOSTIC_GUIDANCE", "equipment": {"name": "Split AC Unit 2T"},
        "sections": sections, "citations": citations, "supporting_claims": claims}}


CAUSES = [_claim("c1", "possible_cause", "Dirty filter or coil."),
          _claim("c2", "possible_cause", "Restricted airflow or blocked outdoor unit clearance."),
          _claim("c3", "possible_cause", "Doors/windows open or heat load exceeds capacity.")]


class ComposerTests(SimpleTestCase):
    def assert_well_formed(self, response, contract):
        cited = {c["id"] for c in contract["presentation"]["citations"]}
        for seg in response["segments"]:
            if seg["kind"] in ("claim", "list_item"):
                self.assertEqual(len(seg["citation_ids"]), 1)
                self.assertIn(seg["citation_ids"][0], cited)
            else:
                self.assertEqual(seg["citation_ids"], [])
        for heading in ("What could be wrong?", "What is still unknown?", "What should I do next?"):
            self.assertNotIn(heading, response["text"])

    def test_symptom_reply_lists_cited_causes_naturally(self):
        contract = _contract(CAUSES)
        response = compose_response(contract, "My split AC is not cooling.")
        self.assert_well_formed(response, contract)
        self.assertEqual(response["style"], "symptom_causes")
        self.assertIn("For poor cooling on the Split AC Unit 2T", response["text"])
        self.assertIn("- Dirty filter or coil. [1]", response["text"])
        self.assertIn(LIMITATION, response["text"])
        self.assertTrue(response["text"].rstrip().endswith("?"))

    def test_claims_are_never_reworded_by_the_composer(self):
        contract = _contract(CAUSES)
        response = compose_response(contract, "Not cooling.")
        bodies = [s["text"] for s in response["segments"] if s["kind"] == "list_item"]
        self.assertEqual(bodies, [c["source_text"] for c in CAUSES])

    def test_ordinal_reply_shows_only_the_selected_cause(self):
        extra = _claim("c4", "possible_cause", "Blocked or incorrectly routed condensate drain.", "4. SYMPTOM: WATER LEAKAGE")
        contract = _contract(CAUSES + [extra])
        reference = {"position": 1, "selected": {"source_text": CAUSES[1]["source_text"]}}
        response = compose_response(contract, "Explain the second cause.", None, reference)
        self.assert_well_formed(response, contract)
        self.assertIn("second possible cause", response["text"])
        self.assertIn("Restricted airflow", response["text"])
        for other in ("Dirty filter", "condensate drain", "Doors/windows"):
            self.assertNotIn(other, response["text"])

    def test_subject_followup_keeps_only_claims_about_that_subject(self):
        fact = _claim("c5", "information", "Quarterly: clean filters and inspect coils.")
        contract = _contract(CAUSES, [fact])
        context = SimpleNamespace(is_followup=True, last_contract={})
        response = compose_response(contract, "What about the filters?", context)
        self.assert_well_formed(response, contract)
        self.assertIn("Dirty filter or coil", response["text"])
        self.assertIn("Quarterly: clean filters", response["text"])
        self.assertNotIn("Restricted airflow", response["text"])
        self.assertNotIn("Doors/windows", response["text"])

    def test_followup_does_not_repeat_previous_claims_when_new_ones_exist(self):
        new = _claim("c6", "possible_cause", "Incorrect settings, fan issue, or refrigerant-system fault.")
        contract = _contract(CAUSES + [new])
        previous = {"presentation": {"supporting_claims": CAUSES}}
        context = SimpleNamespace(is_followup=True, last_contract=previous)
        response = compose_response(contract, "Anything else it could be?", context)
        self.assertIn("Incorrect settings", response["text"])
        self.assertNotIn("Dirty filter", response["text"])

    def test_limitation_is_not_repeated_in_consecutive_turns(self):
        contract = _contract(CAUSES)
        first = compose_response(contract, "Not cooling.")
        context = SimpleNamespace(is_followup=True, last_contract={"presentation": {"response": first}})
        second = compose_response(_contract(CAUSES), "It is still not cooling.", context)
        self.assertNotIn(LIMITATION, second["text"])

    def test_refusal_is_left_unchanged(self):
        contract = {"presentation": {"response_type": "INSUFFICIENT_EVIDENCE"}}
        self.assertIsNone(compose_response(contract, "What is the warranty period?"))

    def test_render_numbers_citations_in_presentation_order(self):
        segments = [{"kind": "text", "text": "Lead:", "citation_ids": []},
                    {"kind": "list_item", "text": "B", "citation_ids": ["s2"]}]
        self.assertEqual(render_text(segments, [{"id": "s1"}, {"id": "s2"}]), "Lead:\n- B [2]")


class TopicShiftTests(SimpleTestCase):
    def test_corrections_and_explicit_topic_changes(self):
        self.assertEqual(topic_shift_query("Sorry, I meant ice is forming on the coil, not water."),
                         "ice is forming on the coil")
        self.assertEqual(topic_shift_query("What about a communication fault instead?"),
                         "What about a communication fault?")

    def test_ordinary_followups_and_questions_are_untouched(self):
        for text in ("What about the filters?", "Explain the second possible cause.",
                     "The AC is not cooling.", "Is the cable not damaged?"):
            self.assertIsNone(topic_shift_query(text), text)


class TicketQuestionExtractionTests(SimpleTestCase):
    """The technician handoff lists the composed reply's question, not the whole reply."""

    def _message(self, segments, text):
        return SimpleNamespace(message_text=text, structured_contract={
            "presentation": {"response": {"segments": segments}}})

    def test_question_segment_is_used(self):
        from core.services.conversation_ticket import _questions_from
        segments = [{"kind": "text", "text": "For poor cooling, the manual lists:"},
                    {"kind": "list_item", "text": "Dirty filter or coil.", "citation_ids": ["s1"]},
                    {"kind": "question", "text": "Does the airflow seem weaker than usual?"}]
        text = "For poor cooling, the manual lists:\n- Dirty filter or coil. [1]\nDoes the airflow seem weaker than usual?"
        self.assertEqual(_questions_from(self._message(segments, text)), ["Does the airflow seem weaker than usual?"])

    def test_composed_reply_without_a_question_contributes_none(self):
        from core.services.conversation_ticket import _questions_from
        segments = [{"kind": "claim", "text": "Quarterly: clean filters.", "citation_ids": ["s1"]}]
        self.assertEqual(_questions_from(self._message(segments, "Quarterly: clean filters. [1]")), [])


class RoleLabelTests(SimpleTestCase):
    def test_observations_are_not_presented_as_customer_checks(self):
        observation = _claim("c7", "observation", "Record the leak location and take photographs.")
        procedure = _claim("c8", "customer_procedure", "Clean the washable front filter.")
        contract = _contract(CAUSES[:1], (), [observation, procedure])
        text = compose_response(contract, "Water is leaking.")["text"]
        self.assertIn("Details worth noting for the technician:\n- Record the leak location", text)
        self.assertIn("Checks the manual approves for customers:\n- Clean the washable front filter", text)
        self.assertNotIn("allows you to check safely", text)
