"""Root-cause regressions from the three-asset conversation evaluation.

Each test covers a shared defect with wording that differs from the evaluation
scripts, so the fix is general rather than tied to fixed questions.
"""
from types import SimpleNamespace

from django.test import SimpleTestCase

from core.services.conversation_context import (
    ConversationContext, _track_customer_turn, canonical_issue_question, conversation_intent,
    extract_symptoms_from_text, is_problem_report, names_own_subject, ordinal_beyond_list, topic_shift_query)
from core.services.customer_answer import (
    _FRONT_MATTER_HEADING, _METADATA_LINE, asks_about_other_equipment)
from core.services.customer_conversation import _TECHNICIAN_DECISION
from core.services.term_matching import normalize_domain_terms


def _run(messages):
    ctx = ConversationContext(conversation_uuid="")
    trail = []
    for text in messages:
        before = ctx.active_issue
        _track_customer_turn(ctx, text)
        trail.append("obs" if ctx.answers_pending_question else ("issue" if ctx.active_issue != before else "-"))
    return ctx, trail


class ActiveIssueTrackingTests(SimpleTestCase):
    """A: questions never replace the current problem; statements add detail."""

    def test_help_requests_and_questions_do_not_become_the_problem(self):
        ctx, _ = _run(["Hello", "Could you help me with the unit?", "Why does that happen?",
                       "The unit won't turn on at all.", "What could be wrong with it?"])
        self.assertEqual(ctx.active_issue, "The unit won't turn on at all")
        self.assertFalse(is_problem_report("Why is it not cooling?"))
        self.assertTrue(is_problem_report("It is not cooling the room"))

    def test_details_are_observations_and_new_symptoms_are_new_issues(self):
        ctx, trail = _run(["The unit is making a grinding noise.", "Only in the morning.",
                           "No code on the display.", "The fan does turn.", "Also water is dripping from it."])
        self.assertEqual(trail, ["issue", "obs", "obs", "obs", "issue"])
        self.assertEqual(ctx.issue_history, ["The unit is making a grinding noise"])

    def test_corrections_with_a_preface_sentence(self):
        for text in ("Sorry, I described that wrongly. It runs but the air is not cool.",
                     "Let me correct myself: it runs but the room is not cooling."):
            with self.subTest(text=text):
                shifted = topic_shift_query(text) or ""
                self.assertNotIn("wrong", shifted.casefold())
                self.assertNotIn("correct", shifted.casefold())
        ctx, trail = _run(["My AC is not working.", "Wait, I explained it badly. It starts but isn't cooling."])
        self.assertEqual(trail[-1], "issue")
        self.assertIn("isn't cooling", ctx.active_issue)

    def test_codes_and_doubtful_readings_are_details(self):
        ctx, trail = _run(["The indoor units keep losing connection.", "It shows code F-12 now.",
                           "I might have misread that number."])
        self.assertEqual(trail, ["issue", "obs", "obs"])
        self.assertIn("losing connection", ctx.active_issue)

    def test_negated_symptoms_are_never_new_problems(self):
        ctx, trail = _run(["Rooms on the first floor are hotter than the others.", "No water is leaking."])
        self.assertEqual(trail, ["issue", "obs"])
        self.assertNotIn("water leak", extract_symptoms_from_text("No water is leaking."))


class IssueQuestionTests(SimpleTestCase):
    """C: subject-less cause/check requests use the current issue; named subjects don't."""

    def setUp(self):
        self.ctx = ConversationContext(conversation_uuid="", active_issue="The airflow is weak in the bedroom")

    def test_canonical_forms(self):
        self.assertTrue(canonical_issue_question("So what might be causing it?", self.ctx).startswith(
            "What are the possible causes of this:"))
        self.assertTrue(canonical_issue_question("Are there more possibilities?", self.ctx).startswith(
            "What other possible causes are there for this:"))
        self.assertTrue(canonical_issue_question("How can I tell which one it is?", self.ctx).startswith(
            "What can I safely check or observe for this:"))

    def test_questions_with_their_own_subject_keep_it(self):
        for text in ("What causes a low-pressure alarm on this model?", "What are the reasons for a tripped breaker?"):
            with self.subTest(text=text):
                self.assertTrue(names_own_subject(text))
                self.assertIsNone(canonical_issue_question(text, self.ctx))

    def test_ordinal_past_the_displayed_list(self):
        contract = {"presentation": {"citations": [{"id": "s1", "document_id": 1, "chunk_id": 1}],
                                     "response": {"displayed_claims": [{
                                         "kind": "possible_cause", "source_text": "Duct leakage.",
                                         "source_heading": "3. SYMPTOM: UNEVEN ROOM TEMPERATURES",
                                         "evidence_ids": ["s1"]}]}}}
        ctx = ConversationContext(conversation_uuid="", last_cause_list_contract=contract)
        self.assertEqual(len(ordinal_beyond_list("And the third one?", ctx)), 1)
        self.assertIsNone(ordinal_beyond_list("Tell me about the first one", ctx))


class IntentGapTests(SimpleTestCase):
    """B: conversation-only requests phrased differently from the scripts."""

    def test_intents(self):
        cases = {
            "ticket_help": ["Please open a support ticket for me.", "Could you log a service ticket?"],
            "ticket_contents": ["What details will go in the ticket?", "What exactly will the technician get?"],
            "certainty": ["What has actually been confirmed so far?", "Is this only a possibility?"],
            "assessment": ["What's the most likely problem?", "What do you think the real issue is?"],
            "relation": ["Is the leak connected to the issue I described before?"],
        }
        for intent, texts in cases.items():
            for text in texts:
                with self.subTest(text=text):
                    self.assertEqual(conversation_intent(text), intent)
        self.assertTrue(_TECHNICIAN_DECISION.search("Honestly, do you think we should call an engineer?"))


class EvidenceSelectionTests(SimpleTestCase):
    """E/F/G: front matter never answers; number words match digits; other equipment."""

    def test_front_matter_and_metadata(self):
        for heading in ("1. PURPOSE", "General Overview", "2. Scope", "Introduction"):
            self.assertTrue(_FRONT_MATTER_HEADING.match(heading), heading)
        self.assertFalse(_FRONT_MATTER_HEADING.match("3. SYMPTOM: POOR COOLING"))
        self.assertTrue(_METADATA_LINE.match("Asset: VRF Outdoor Module"))
        self.assertTrue(_METADATA_LINE.match("Document ID: AC2T-TS-001"))
        self.assertFalse(_METADATA_LINE.match("Dirty filter or coil."))

    def test_number_words_match_schedule_digits(self):
        self.assertIn("6 months", normalize_domain_terms("what is due every six months"))
        self.assertIn("12 months", normalize_domain_terms("and every twelve months?"))

    def test_other_equipment_questions(self):
        ductable = SimpleNamespace(name="Ductable AC 5T", model_number="")
        self.assertTrue(asks_about_other_equipment("Does a chiller alarm code mean anything for this ductable?", ductable))
        self.assertFalse(asks_about_other_equipment("Does the ductable unit need a filter change?", ductable))


class FollowUpReferenceTests(SimpleTestCase):
    def _one_cause_contract(self):
        return {"presentation": {"response_type": "DIAGNOSTIC_GUIDANCE",
                                 "citations": [{"id": "s1", "document_id": 1, "chunk_id": 1}],
                                 "supporting_claims": [{"kind": "possible_cause", "source_text": "Duct leakage.",
                                                        "source_heading": "3. SYMPTOM: UNEVEN ROOM TEMPERATURES",
                                                        "evidence_ids": ["s1"]}],
                                 "response": {"displayed_claims": [{
                                     "kind": "possible_cause", "source_text": "Duct leakage.",
                                     "source_heading": "3. SYMPTOM: UNEVEN ROOM TEMPERATURES",
                                     "evidence_ids": ["s1"]}]}}}

    def test_simpler_after_a_conversation_only_reply_uses_the_single_supported_cause(self):
        from core.services.conversation_context import resolve_cause_reference
        ctx = ConversationContext(conversation_uuid="", is_followup=True,
                                  last_contract={"response_type": "CONVERSATION_REPLY"},
                                  last_supported_contract=self._one_cause_contract())
        ref = resolve_cause_reference("Could you put that in plain English?", ctx)
        self.assertEqual((ref["selected"]["source_text"], ref["mode"]), ("Duct leakage.", "simpler"))

    def test_other_causes_request_shows_only_unshown_causes(self):
        from core.services.customer_response_composer import compose_response
        def cause(cid, text):
            return {"id": cid, "kind": "possible_cause", "text": "Possible documented cause: " + text,
                    "source_text": text, "source_heading": "3. SYMPTOM: POOR COOLING", "evidence_ids": ["s" + cid]}
        claims = [cause("1", "Dirty filter."), cause("2", "Blocked outdoor unit.")]
        contract = {"presentation": {"response_type": "DIAGNOSTIC_GUIDANCE", "equipment": {"name": "AC"},
                                     "sections": [{"number": 1, "title": "x", "claims": claims}],
                                     "citations": [{"id": "s1"}, {"id": "s2"}], "supporting_claims": claims}}
        shown = {"presentation": {"supporting_claims": [claims[0]]}}
        ctx = SimpleNamespace(is_followup=True, last_contract=shown, last_cause_list_contract={},
                              original_complaint="", active_topics=[], resolved_answers=[])
        text = compose_response(contract, "What other possible causes are there for this: not cooling?", ctx)["text"]
        self.assertIn("Blocked outdoor unit", text)
        self.assertNotIn("Dirty filter", text)
