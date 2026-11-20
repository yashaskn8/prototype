"""Answer feedback (official customer API via the gateway), follow-up suggestions, validator glue."""
from django.test import SimpleTestCase

from core.models import AnswerFeedback, CustomerConversation, CustomerMessage
from core.services import customer_answer_generation as generation
from core.services.answer_feedback import regression_cases
from core.services.followup_suggestions import suggest
from core.tests_demo_gateway import GatewayTestBase


class AnswerFeedbackGatewayTests(GatewayTestBase):
    def answer(self):
        uuid = self.start_conversation("What can cause poor cooling?")
        message = CustomerMessage.objects.filter(conversation__conversation_uuid=uuid, sender_role="assistant").first()
        return uuid, message

    def put(self, uuid, message_id, body):
        return self.as_customer("put", f"/demo-api/customer-support/conversations/{uuid}/messages/{message_id}/feedback/",
                                data=body, content_type="application/json")

    def test_feedback_is_saved_updated_and_shown_after_reload(self):
        uuid, message = self.answer()
        r = self.put(uuid, message.id, {"rating": "helpful"})
        self.assertEqual(r.status_code, 200, r.content[:200])
        r = self.put(uuid, message.id, {"rating": "not_helpful", "reason": "missing", "comment": "No airflow check"})
        self.assertEqual(r.json()["feedback"]["reason"], "missing")
        self.assertEqual(AnswerFeedback.objects.count(), 1)
        item = AnswerFeedback.objects.get()
        self.assertEqual((item.customer_id, item.asset_id, item.conversation.conversation_uuid.hex),
                         (self.customer.id, self.asset.id, uuid.replace("-", "")))
        history = self.as_customer("get", f"/demo-api/customer-support/conversations/{uuid}/messages/").json()["messages"]
        assistant = [m for m in history if m["sender_role"] == "assistant"][0]
        self.assertEqual(assistant["feedback"]["rating"], "not_helpful")

    def test_invalid_values_and_non_answers_are_refused(self):
        uuid, message = self.answer()
        self.assertEqual(self.put(uuid, message.id, {"rating": "great"}).status_code, 400)
        self.assertEqual(self.put(uuid, message.id, {"rating": "not_helpful", "reason": "x"}).status_code, 400)
        self.assertEqual(self.put(uuid, message.id, {"rating": "helpful", "comment": "x" * 501}).status_code, 400)
        customer_msg = CustomerMessage.objects.filter(conversation=message.conversation, sender_role="customer").first()
        self.assertEqual(self.put(uuid, customer_msg.id, {"rating": "helpful"}).status_code, 404)
        self.assertEqual(AnswerFeedback.objects.count(), 0)

    def test_other_customers_answer_cannot_be_rated(self):
        uuid, message = self.answer()
        foreign = CustomerConversation.objects.create(tenant=self.tenant, customer=self.other_customer,
                                                      asset=self.other_asset, title="other")
        foreign_answer = CustomerMessage.objects.create(tenant=self.tenant, conversation=foreign,
                                                        sender_role="assistant", message_text="x")
        # Wrong conversation for this message, and a conversation the customer does not own.
        self.assertEqual(self.put(uuid, foreign_answer.id, {"rating": "helpful"}).status_code, 404)
        self.assertEqual(self.put(str(foreign.conversation_uuid), foreign_answer.id, {"rating": "helpful"}).status_code, 404)
        self.assertEqual(AnswerFeedback.objects.count(), 0)

    def test_only_reviewed_approved_sanitized_cases_are_exported(self):
        uuid, message = self.answer()
        self.put(uuid, message.id, {"rating": "not_helpful", "reason": "incorrect", "comment": "call me on 555 0100"})
        item = AnswerFeedback.objects.get()
        self.assertEqual(regression_cases(AnswerFeedback.objects.all()), [])
        item.review_status, item.export_approved, item.eval_question = "reviewed", True, "What can cause poor cooling?"
        item.save()
        cases = regression_cases(AnswerFeedback.objects.all())
        self.assertEqual([c["question"] for c in cases], ["What can cause poor cooling?"])
        self.assertNotIn("555", str(cases))


def _contract(style="symptom_causes", items=3, sections=("poor cooling", "unit does not start"), hazard="none",
              response_type="DIAGNOSTIC_GUIDANCE"):
    segments = [{"kind": "list_item", "text": f"cause {i}", "citation_ids": ["s1"]} for i in range(items)]
    return {"response_type": response_type, "documented_cause_sections": list(sections),
            "presentation": {"response_type": response_type, "citations": [{"id": "s1"}],
                             "hazard": {"severity": hazard},
                             "sections": [{"claims": [{"source_heading": "3. SYMPTOM: POOR COOLING"}]}],
                             "response": {"style": style, "segments": segments}}}


class FollowUpSuggestionTests(SimpleTestCase):
    def test_cause_list_gets_explain_and_other_documented_symptoms(self):
        self.assertEqual(suggest(_contract(), ["What can cause poor cooling?"]), [
            "Explain each of these causes.", "Tell me more about the first cause.",
            "What are the possible causes of unit does not start?"])

    def test_already_asked_and_shown_sections_are_not_suggested(self):
        out = suggest(_contract(), ["explain each of these causes", "What are the possible causes of unit does not start?"])
        self.assertEqual(out, ["Tell me more about the first cause."])

    def test_no_suggestions_for_refusals_safety_or_uncited_answers(self):
        self.assertEqual(suggest(_contract(response_type="INSUFFICIENT_EVIDENCE"), []), [])
        self.assertEqual(suggest(_contract(hazard="immediate"), []), [])
        contract = _contract()
        contract["presentation"]["citations"] = []
        self.assertEqual(suggest(contract, []), [])

    def test_single_cause_does_not_offer_explain_each(self):
        self.assertNotIn("Explain each of these causes.", suggest(_contract(items=1), []))


class ValidatorUnchangedTests(SimpleTestCase):
    """The rejected-claim analysis kept the validator as is; pin the two failure modes it rejects."""

    def test_question_echo_and_synonym_swaps_are_rejected(self):
        source = "Blocked airflow, dirty coil, fan issue, or operating conditions outside design."
        echo = "Possible cause: the VRF outdoor module shows a high-pressure alarm."
        swap = ("Possible causes are blocked airflow, a dirty coil, a fan problem, or operating conditions "
                "that are not within the design.")
        self.assertFalse(generation._validate_paraphrase(source, echo, "possible_cause"))
        self.assertFalse(generation._validate_paraphrase(source, swap, "possible_cause"))
