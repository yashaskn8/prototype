from django.test import SimpleTestCase

from core.services.conversation_context import ConversationContext, contextualize_customer_query
from core.services.query_decomposition import (
    ASK_CAUSE, ASK_OBSERVATION, NEGATIVE_REPORT, REPORT, decompose,
    negated_text, relevance_text, retrieval_focus,
)

MIXED = ("No warning is showing. The filter is clean, but the outdoor unit is close "
         "to a wall. Could blocked clearance explain poor cooling? What should I "
         "safely observe or report without opening it?")


class QueryDecompositionTests(SimpleTestCase):
    def test_mixed_question_clause_roles(self):
        roles = [c.role for c in decompose(MIXED)]
        self.assertEqual(roles, [NEGATIVE_REPORT, NEGATIVE_REPORT, REPORT, ASK_CAUSE, ASK_OBSERVATION])

    def test_focus_keeps_context_and_drops_incidentals(self):
        focus = retrieval_focus(MIXED).casefold()
        self.assertIn("close to a wall", focus)
        self.assertIn("blocked clearance", focus)
        for dropped in ("filter", "warning", "observe"):
            self.assertNotIn(dropped, focus)

    def test_negatives_are_not_relevance_subjects(self):
        self.assertNotIn("filter", relevance_text(MIXED).casefold())
        self.assertIn("filter", negated_text(MIXED).casefold())

    def test_single_intent_questions_are_untouched(self):
        for q in ("Why is this Split AC not cooling?", "What does E-04 mean?", ""):
            self.assertIsNone(retrieval_focus(q))

    def test_asked_about_component_is_not_treated_as_negative(self):
        self.assertIn("filter", relevance_text("Could a dirty filter cause poor cooling?").casefold())

    def test_followup_rewrite_uses_focus_not_negatives(self):
        ctx = ConversationContext(conversation_uuid="x", asset_name="Split AC", turn_count=1,
                                  is_followup=True, original_complaint="Split AC not cooling")
        query = contextualize_customer_query(MIXED, ctx)
        self.assertIn("close to a wall", query)
        self.assertNotIn("warning is showing", query)

    def test_no_invented_content(self):
        focus = retrieval_focus(MIXED)
        source_words = set(MIXED.casefold().replace("?", " ").replace(".", " ").split())
        self.assertTrue(set(focus.casefold().replace("?", " ").replace(".", " ").split()) <= source_words)

