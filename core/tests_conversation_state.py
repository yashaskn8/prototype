"""Conversation-state regressions: replies to the assistant's own question, the
active issue after a topic change, negated observations, displayed-list ordinals,
and the Split AC five-turn conversation through the real customer gateway.

All data is synthetic. The Split AC manual below mirrors the section structure
of an approved troubleshooting manual (numbered SYMPTOM headings with "Possible
causes" and observation bullets); it is not the protected document.
"""
import json
from types import SimpleNamespace
from unittest import mock

from django.test import SimpleTestCase, TestCase

from core.models import (
    Customer, CustomerConversation, CustomerMessage, KnowledgeChunk, KnowledgeDocument, SourceProvenanceRecord, Tenant)
from core.services.conversation_context import (
    ConversationContext, asked_question, contextualize_customer_query, extract_conversation_context,
    extract_negated_observations, extract_symptoms_from_text, is_reply_to_question, listed_causes,
    resolve_cause_reference)
from core.services.customer_response_composer import compose_response
from core.services.indexing import index_document
from core.tests_conversational_gateway import VRFConversationGatewayTests, parse_sse  # noqa: F401
from core.tests_demo_gateway import GatewayTestBase

SPLIT_AC_MANUAL = """1. PURPOSE
Synthetic troubleshooting reference for regression tests.

2. SYMPTOM: UNIT DOES NOT START
Possible causes:
- Power supply unavailable or isolator switched off.
- Breaker or protective device tripped.
- Controller batteries depleted or incorrect mode/setpoint.
- Loose connection or internal control fault.
Initial checks:
- Verify the supply externally and inspect the controller settings.

3. SYMPTOM: POOR COOLING
Possible causes:
- Dirty filter or coil.
- Restricted airflow or blocked outdoor unit clearance.
- Incorrect settings, fan issue, or refrigerant-system fault.

4. SYMPTOM: WATER LEAKAGE
Possible causes:
- Blocked or incorrectly routed condensate drain.
Initial checks:
- Record the leak location and take photographs.

6. SYMPTOM: NOISE OR VIBRATION
Possible causes:
- Loose panel or mounting fastener.
- Fan obstruction, worn bearing, or unstable base.
Action:
- Record whether noise occurs at startup, continuously, or only at a particular fan speed.
"""


class ReplyToPendingQuestionTests(SimpleTestCase):
    """Short replies are resolved against the question actually asked."""

    CASES = [
        ("When does the noise happen: at startup, continuously, or at a fan speed?",
         ["ONLY AT STARTUP", "Only when starting.", "At startup.", "Just during startup.",
          "It stops after a few seconds.", "All the time.", "Only when the fan runs.",
          "It happens after switching it on.", "No, there is no noise now."]),
        ("Is the indoor fan running?", ["Yes, the fan is running.", "Yes.", "No, the fan is not running."]),
        ("Is any error code shown on the controller?", ["No error code.", "No.", "Nothing on the display."]),
    ]

    def test_short_answers_are_replies(self):
        for question, replies in self.CASES:
            for reply in replies:
                with self.subTest(question=question, reply=reply):
                    self.assertTrue(is_reply_to_question(reply, question, "There is a noise in the machine"))

    def test_questions_topic_changes_and_new_symptoms_are_not_replies(self):
        pending = "Is the indoor fan running?"
        for text in ("What should I do next?", "Should I contact a technician?",
                     "Actually, I meant a communication fault.", "Explain every cause you mentioned.",
                     "Tell me more about the first cause.", "Water is leaking from the indoor unit.",
                     "What is the warranty period for the compressor in years?", "hi"):
            with self.subTest(text=text):
                self.assertFalse(is_reply_to_question(text, pending))
        self.assertFalse(is_reply_to_question("Yes, the fan is running.", None))

    def test_asked_question_covers_clarifications_closing_questions_and_observations(self):
        self.assertEqual(asked_question({"clarification_question": "Is the fan running?"}), "Is the fan running?")
        composed = {"presentation": {"response": {"segments": [
            {"kind": "text", "text": "Here are the possible causes:"},
            {"kind": "list_item", "text": "Record whether noise occurs at startup, continuously, or only at a "
                                          "particular fan speed.", "citation_ids": ["s1"]},
            {"kind": "question", "text": "Do any of these match what you are seeing?"}]}}}
        question = asked_question(composed)
        self.assertIn("Do any of these match", question)
        self.assertIn("noise occurs at startup", question)
        self.assertIsNone(asked_question({"presentation": {"response": {"segments": [
            {"kind": "text", "text": "Thanks."}]}}}))


class NegatedObservationTests(SimpleTestCase):
    def test_negated_symptoms_are_not_reported_symptoms(self):
        self.assertEqual(extract_symptoms_from_text("No water leak is visible."), [])
        self.assertIn("water leak", extract_negated_observations("No water leak is visible."))
        self.assertEqual(extract_symptoms_from_text("There is no noise now."), [])
        self.assertIn("noise", extract_negated_observations("There is no noise now."))
        self.assertIn("noise", extract_symptoms_from_text("There is a noise in the machine"))
        # Negation is per clause.
        self.assertIn("noise", extract_symptoms_from_text("No water leak, but there is a loud noise."))


def _cause(cid, source, heading):
    return {"id": cid, "kind": "possible_cause", "text": "Possible documented cause: " + source,
            "source_text": source, "source_heading": heading, "evidence_ids": ["s" + cid[1:]]}


def _contract(claims):
    citations = [{"id": c["evidence_ids"][0], "title": "Manual", "heading": c["source_heading"],
                  "document_id": 1, "chunk_id": i + 1} for i, c in enumerate(claims)]
    return {"answer": "", "presentation": {
        "response_type": "DIAGNOSTIC_GUIDANCE", "equipment": {"name": "Split AC Unit 2T"},
        "sections": [{"number": 1, "title": "What could be wrong?", "claims": list(claims)}],
        "citations": citations, "supporting_claims": list(claims)}}


NOISE = "6. SYMPTOM: NOISE OR VIBRATION"
COOLING = "3. SYMPTOM: POOR COOLING"


class ReplyCompositionTests(SimpleTestCase):
    def _context(self, last_contract, answer="ONLY AT STARTUP"):
        return SimpleNamespace(
            is_followup=True, answers_pending_question=True, active_topics=["noise or vibration"],
            resolved_answers=[{"question": "When does the noise occur?", "answer": answer}],
            last_contract=last_contract, original_complaint="What are the possible causes it stopped working")

    def test_reply_does_not_switch_to_an_unrelated_section(self):
        shown = _contract([_cause("c1", "Loose panel or mounting fastener.", NOISE),
                           _cause("c2", "Fan obstruction, worn bearing, or unstable base.", NOISE)])
        shown["presentation"]["response"] = compose_response(shown, "There is a noise in the machine")
        retrieved = _contract([_cause("c1", "Loose panel or mounting fastener.", NOISE),
                               _cause("c3", "Incorrect settings, fan issue, or refrigerant-system fault.", COOLING)])
        response = compose_response(retrieved, "ONLY AT STARTUP", self._context(shown))
        text = response["text"].casefold()
        self.assertNotIn("cooling", text)
        self.assertNotIn("refrigerant", text)
        self.assertIn("only at startup", text)
        self.assertEqual(response["style"], "reply_acknowledged")
        self.assertEqual(response["displayed_claims"], [])
        for segment in response["segments"]:
            self.assertEqual(segment["citation_ids"], [])

    def test_reply_shows_new_on_topic_evidence_with_an_acknowledgement(self):
        shown = _contract([_cause("c1", "Loose panel or mounting fastener.", NOISE)])
        shown["presentation"]["response"] = compose_response(shown, "There is a noise")
        retrieved = _contract([_cause("c2", "Fan obstruction, worn bearing, or unstable base.", NOISE),
                               _cause("c3", "Dirty filter or coil.", COOLING)])
        response = compose_response(retrieved, "Only at startup.", self._context(shown, "Only at startup."))
        self.assertTrue(response["text"].startswith("Thanks, I've noted that: only at startup."))
        self.assertIn("Fan obstruction", response["text"])
        self.assertNotIn("Dirty filter", response["text"])
        self.assertNotIn("Exact cited manual wording", response["text"])


class DisplayedListOrdinalTests(SimpleTestCase):
    def test_ordinals_resolve_against_the_displayed_list_up_to_twelve(self):
        headings = [COOLING] * 6 + [NOISE] * 6
        claims = [_cause(f"c{i}", f"Documented cause number {i}.", headings[i - 1]) for i in range(1, 13)]
        contract = _contract(claims)
        contract["presentation"]["response"] = compose_response(contract, "What are the possible causes?")
        displayed = [c["source_text"] for c in listed_causes(contract)]
        self.assertEqual(len(displayed), 12)
        context = ConversationContext(conversation_uuid="", is_followup=True, last_cause_list_contract=contract)
        for text, index in (("Explain the seventh cause.", 6), ("Tell me more about the first cause.", 0),
                            ("What about the last cause?", 11), ("Explain cause number 12.", 11),
                            ("Explain the twelfth cause", 11)):
            with self.subTest(text=text):
                self.assertEqual(resolve_cause_reference(text, context)["selected"]["source_text"],
                                 displayed[index])
        self.assertIsNone(resolve_cause_reference("Explain the thirteenth cause.", context))

    def test_displayed_order_wins_over_validation_order(self):
        claims = [_cause("c1", "Cooling cause.", COOLING), _cause("c2", "Noise cause.", NOISE),
                  _cause("c3", "Second cooling cause.", COOLING)]
        contract = _contract(claims)
        contract["presentation"]["response"] = compose_response(contract, "What are the possible causes?")
        # Grouped by section: both cooling causes are shown before the noise cause.
        self.assertEqual([c["source_text"] for c in listed_causes(contract)],
                         ["Cooling cause.", "Second cooling cause.", "Noise cause."])


class ActiveIssueQueryTests(TestCase):
    def setUp(self):
        self.tenant = Tenant.objects.create(name="Synthetic", slug="synthetic-active-issue")
        self.customer = Customer.objects.create(tenant=self.tenant, name="Synthetic Customer")
        self.conv = CustomerConversation.objects.create(tenant=self.tenant, customer=self.customer, status="ACTIVE")

    def _say(self, role, text, contract=None):
        CustomerMessage.objects.create(tenant=self.tenant, conversation=self.conv, sender_role=role,
                                       message_text=text, structured_contract=contract or {})

    def test_reply_query_uses_the_current_issue_not_the_first_complaint(self):
        self._say("customer", "WHAT ARE THE POSSIBLE CAUSES THE SPLIT AC UNIT STOPPED WORKING")
        self._say("assistant", "Here are the possible causes...", {"response_type": "DIAGNOSTIC_GUIDANCE"})
        self._say("customer", "THERE IS A NOISE IN THE MACHINE")
        noise = _contract([_cause("c1", "Loose panel or mounting fastener.", NOISE)])
        noise["presentation"]["response"] = {"segments": [
            {"kind": "list_item", "text": "Record whether noise occurs at startup, continuously, or only at a "
                                          "particular fan speed.", "citation_ids": ["s1"]}],
            "displayed_claims": [{"kind": "possible_cause", "source_text": "Loose panel or mounting fastener.",
                                  "source_heading": NOISE, "evidence_ids": ["s1"]}]}
        self._say("assistant", "noise answer", noise)
        ctx = extract_conversation_context(self.conv, "ONLY AT STARTUP")
        self.assertTrue(ctx.is_followup)
        self.assertTrue(ctx.answers_pending_question)
        self.assertEqual(ctx.active_issue, "THERE IS A NOISE IN THE MACHINE")
        self.assertEqual(ctx.active_topics, ["noise or vibration"])
        query = contextualize_customer_query("ONLY AT STARTUP", ctx).casefold()
        self.assertIn("noise", query)
        self.assertIn("only at startup", query)
        self.assertNotIn("stopped working", query)

    def test_negated_observation_is_tracked_and_excluded_from_the_query(self):
        self._say("customer", "My AC is not cooling")
        self._say("assistant", "answer", {"response_type": "DIAGNOSTIC_GUIDANCE",
                                          "clarification_question": "Is water leaking from the unit?"})
        ctx = extract_conversation_context(self.conv, "No water leak is visible.")
        self.assertIn("water leak", ctx.negative_observations)
        self.assertNotIn("water leak", ctx.reported_symptoms)
        self.assertNotIn("water leak", contextualize_customer_query("What should I do next?", ctx).casefold()
                         .split("symptoms:")[-1].split(";")[0])

    def test_topic_change_replaces_the_active_issue(self):
        self._say("customer", "My AC is not cooling")
        self._say("assistant", "answer", {"response_type": "DIAGNOSTIC_GUIDANCE"})
        ctx = extract_conversation_context(self.conv, "Actually, I meant a communication fault.")
        self.assertEqual(ctx.active_issue, "a communication fault")
        self.assertNotIn("cooling", contextualize_customer_query(
            "Actually, I meant a communication fault.", ctx).casefold())


def install_split_ac_fixture(test, manual=None):
    """Inline SSE worker plus the synthetic Split AC manual on the gateway asset."""
    from core.services import rag_event_stream
    from core.tests_conversational_gateway import _InlineThread
    for name, value in (("Thread", _InlineThread), ("close_old_connections", lambda: None),
                        ("connections", mock.Mock())):
        patcher = mock.patch.object(rag_event_stream, name, value)
        patcher.start()
        test.addCleanup(patcher.stop)
    KnowledgeDocument.objects.filter(asset=test.asset).delete()
    doc = KnowledgeDocument.objects.create(
        tenant=test.tenant, product=test.asset.product, asset=test.asset,
        title="Split AC Unit 2T - Troubleshooting and Faults", source_type="text",
        is_rag_enabled=True, content_text=manual or SPLIT_AC_MANUAL)
    index_document(doc)
    test.vrf = test.asset


class SplitACConversationTests(GatewayTestBase):
    """The exact five-turn Split AC conversation through the real customer gateway."""

    def setUp(self):
        super().setUp()
        install_split_ac_fixture(self)

    create = VRFConversationGatewayTests.create
    ask = VRFConversationGatewayTests.ask
    segments = staticmethod(VRFConversationGatewayTests.segments)
    assert_citations_resolve = VRFConversationGatewayTests.assert_citations_resolve

    def test_split_ac_five_turn_conversation(self):
        uuid = self.create()
        hello, logs = self.ask(uuid, "HI")
        self.assertEqual(hello["response_type"], "GREETING")
        self.assertFalse((hello.get("presentation") or {}).get("citations"))

        overview, _ = self.ask(uuid, "WHAT ARE THE POSSIBLE CAUSES THE SPLIT AC UNIT STOPPED WORKING")
        self.assertEqual(overview["response_type"], "DIAGNOSTIC_GUIDANCE", overview["answer"])
        self.assertIn("not a confirmed diagnosis", overview["answer"].casefold())
        self.assert_citations_resolve(overview)
        displayed = overview["presentation"]["response"]["displayed_claims"]
        first = next(c for c in displayed if c["kind"] == "possible_cause")

        detail, _ = self.ask(uuid, "Tell me more about the first cause.")
        self.assertEqual(detail["response_type"], "DIAGNOSTIC_GUIDANCE", detail["answer"])
        self.assertIn(first["source_text"].rstrip(".").casefold(), detail["answer"].casefold())
        self.assert_citations_resolve(detail)

        noise, _ = self.ask(uuid, "THERE IS A NOISE IN THE MACHINE")
        self.assertEqual(noise["response_type"], "DIAGNOSTIC_GUIDANCE", noise["answer"])
        self.assertIn("noise", noise["answer"].casefold())
        self.assertNotIn("refrigerant", noise["answer"].casefold())

        startup, _ = self.ask(uuid, "ONLY AT STARTUP")
        text = startup["answer"].casefold()
        for unrelated in ("poor cooling", "refrigerant", "dirty filter", "condensate"):
            self.assertNotIn(unrelated, text, startup["answer"])
        self.assertIn("only at startup", text)
        for internal in ("exact cited manual wording", "wording step", "ran out of time"):
            self.assertNotIn(internal, text)
        self.assert_citations_resolve(startup)

        # Reopening the conversation keeps the noise context for the next reply.
        detail_view = self.as_customer("get", f"/demo-api/customer-support/conversations/{uuid}/")
        self.assertEqual(detail_view.status_code, 200)
        self.assertEqual(len(detail_view.json()["messages"]), 10)
        conv = CustomerConversation.objects.get(conversation_uuid=uuid)
        reopened = extract_conversation_context(conv, "It stops after a few seconds.", asset=self.asset)
        self.assertEqual(reopened.active_issue, "THERE IS A NOISE IN THE MACHINE")
        self.assertIn("noise", contextualize_customer_query("It stops after a few seconds.", reopened).casefold())


class PostRefusalAuthorizationTests(TestCase):
    """R03: the clarification offered after a refusal uses only authorized sections."""

    def test_clarification_options_skip_unshareable_and_revoked_documents(self):
        from django.test import override_settings
        from core.models import Asset, Brand, Product, ProductCategory, ProductDomain, Site
        from core.services.customer_response_composer import clarification_for_ambiguous
        tenant = Tenant.objects.create(name="Synthetic R03", slug="synthetic-r03")
        customer = Customer.objects.create(tenant=tenant, name="Synthetic R03 Customer")
        domain = ProductDomain.objects.create(tenant=tenant, name="HVAC")
        category = ProductCategory.objects.create(tenant=tenant, domain=domain, name="AC")
        brand = Brand.objects.create(tenant=tenant, name="B")
        product = Product.objects.create(tenant=tenant, category=category, brand=brand, name="AC")
        site = Site.objects.create(tenant=tenant, customer=customer, name="Site")
        asset = Asset.objects.create(tenant=tenant, customer=customer, site=site, product=product,
                                     name="AC", asset_code="R03-1")
        with override_settings(SERVY_CUSTOMER_DATA_MODE="demo_only"):
            def manual(title, text, **flags):
                doc = KnowledgeDocument.objects.create(tenant=tenant, product=product, asset=asset, title=title,
                                                       source_type="text", is_rag_enabled=True,
                                                       content_text=text, **flags)
                index_document(doc)
                return doc
            manual("Shareable", "1. SYMPTOM: POOR COOLING\nDirty filter.\n\n2. SYMPTOM: WATER LEAKAGE\nDrain.")
            manual("Unshareable", "1. SYMPTOM: SECRET LEAK ALPHA\nx.\n\n2. SYMPTOM: SECRET LEAK BETA\ny.",
                   disable_sharing=True)
            revoked = manual("Revoked", "1. SYMPTOM: REVOKED GAMMA\nx.\n\n2. SYMPTOM: REVOKED DELTA\ny.")
            SourceProvenanceRecord.objects.create(
                tenant=tenant, source_system="synthetic", external_object_type="document", external_id="r03",
                local_object_type="KnowledgeDocument", local_object_id=revoked.id, verification_status="REVOKED")
            result = clarification_for_ambiguous("help", asset, tenant, customer=customer)
        options = json.dumps(result).casefold() if result else ""
        self.assertNotIn("secret", options)
        self.assertNotIn("revoked", options)
        if result:
            self.assertEqual(set(result["options"]), {"poor cooling", "water leakage"})


class AuditFindingRegressionTests(SimpleTestCase):
    def test_r13_identical_wording_under_different_headings_stays_distinct(self):
        claims = [_cause("c1", "Dirty air filter.", COOLING), _cause("c2", "Dirty air filter.", "5. SYMPTOM: ICE ON COIL")]
        contract = _contract(claims)
        contract["presentation"]["response"] = compose_response(contract, "What are the possible causes?")
        self.assertEqual(len(listed_causes(contract)), 2)

    def test_r15_next_step_query_keeps_the_customers_words_first(self):
        ctx = ConversationContext(conversation_uuid="", is_followup=True, turn_count=2, asset_name="Split AC",
                                  original_complaint="unit not cooling " * 30,
                                  reported_symptoms=["not cooling"], active_issue="unit not cooling " * 30)
        text = "Now the outdoor fan makes a grinding noise and the breaker trips. What next?"
        query = contextualize_customer_query(text, ctx)
        self.assertTrue(query.startswith("Now the outdoor fan"), query)
        self.assertIn("breaker trips", query)
        self.assertIn("next troubleshooting steps", query)

    def test_r16_request_clause_naming_a_new_component_is_kept(self):
        from core.services.query_decomposition import retrieval_focus
        focus = retrieval_focus("Water is dripping from the indoor unit. How do I check the drain pan for standing water?")
        self.assertTrue(focus is None or "drain pan" in focus, focus)

    def test_r18_capped_cause_list_is_disclosed(self):
        claims = [_cause(f"c{i}", f"Documented cause number {i}.", COOLING if i % 2 else NOISE) for i in range(1, 13)]
        text = compose_response(_contract(claims), "What are the possible causes?")["text"]
        self.assertIn("may describe more", text)


class TechnicianAdviceWordingTests(SimpleTestCase):
    def test_r21_advice_states_truthfully_whether_a_ticket_exists(self):
        from core.services.customer_conversation import _technician_decision_contract
        asset = SimpleNamespace(id=1, name="AC", model_number="M")
        advice = _technician_decision_contract(asset, None)["answer"]
        self.assertTrue(advice.startswith("Yes."))
        self.assertIn("No service request has been created yet", advice)
        ticketed = _technician_decision_contract(asset, None, ticketed=True)["answer"]
        self.assertIn("already open", ticketed)
        self.assertNotIn("No service request", ticketed)


class CitationRefreshTests(TestCase):
    def test_r19_reopened_answers_get_fresh_links_only_for_the_same_user(self):
        from django.contrib.auth.models import User
        from django.core import signing
        from core.services.customer_answer import SALT, refresh_citation_urls
        tenant = Tenant.objects.create(name="Synthetic R19", slug="synthetic-r19")
        customer = Customer.objects.create(tenant=tenant, name="R19 Customer")
        user = User.objects.create(username="r19_user")
        other = User.objects.create(username="r19_other")
        payload = {"tenant": tenant.id, "customer": customer.id, "asset": 1, "document": 1, "chunk": 1,
                   "version": 1, "sha": "x", "start": 0, "end": 1, "user": user.id, "kind": "possible_cause"}
        prefix = "/api/customer-support/source/?citation="
        old = prefix + signing.dumps(payload, salt=SALT, compress=True)
        contract = {"presentation": {"citations": [{"id": "s1", "url": old}]}}
        with mock.patch("django.core.signing.time.time", return_value=__import__("time").time() + 7200):
            refreshed = refresh_citation_urls(json.loads(json.dumps(contract)),
                                              {"user": user, "tenant": tenant, "customer": customer})
            url = refreshed["presentation"]["citations"][0]["url"]
            self.assertNotEqual(url, old)
            self.assertEqual(signing.loads(url[len(prefix):], salt=SALT, max_age=3600)["chunk"], 1)
            untouched = refresh_citation_urls(json.loads(json.dumps(contract)),
                                              {"user": other, "tenant": tenant, "customer": customer})
            self.assertEqual(untouched["presentation"]["citations"][0]["url"], old)


class LongConversationMemoryTests(TestCase):
    """Facts from early turns survive far beyond the 10-message wording window."""

    def setUp(self):
        self.tenant = Tenant.objects.create(name="Synthetic Long", slug="synthetic-long")
        self.customer = Customer.objects.create(tenant=self.tenant, name="Long Customer")
        self.conv = CustomerConversation.objects.create(tenant=self.tenant, customer=self.customer, status="ACTIVE")

    def _say(self, role, text, contract=None):
        CustomerMessage.objects.create(tenant=self.tenant, conversation=self.conv, sender_role=role,
                                       message_text=text, structured_contract=contract or {})

    def test_corrections_answers_and_negations_survive_60_turns(self):
        self._say("customer", "Hi")
        self._say("assistant", "Hello!", {"response_type": "GREETING"})
        self._say("customer", "My AC isn't working.")
        self._say("assistant", "answer", {"response_type": "DIAGNOSTIC_GUIDANCE",
                                          "clarification_question": "Is any error code shown?"})
        self._say("customer", "No error code.")
        self._say("assistant", "answer", {"response_type": "DIAGNOSTIC_GUIDANCE"})
        self._say("customer", "No water leak is visible.")
        self._say("assistant", "answer", {"response_type": "DIAGNOSTIC_GUIDANCE"})
        self._say("customer", "Actually, the AC starts, but it doesn't cool.")
        self._say("assistant", "answer", {"response_type": "DIAGNOSTIC_GUIDANCE"})
        for i in range(26):   # 52 more messages of small talk / follow-ups
            self._say("customer", f"Thanks, noted {i}.")
            self._say("assistant", "ok", {"response_type": "DIAGNOSTIC_GUIDANCE"})
        self.assertGreater(self.conv.messages.count(), 60)
        ctx = extract_conversation_context(self.conv, "What should I do next?")
        self.assertEqual(ctx.original_complaint, "My AC isn't working.")
        self.assertEqual(ctx.active_issue, "the AC starts, but it doesn't cool")
        self.assertIn("water leak", ctx.negative_observations)
        self.assertIn({"question": "Is any error code shown?", "answer": "No error code."}, ctx.resolved_answers)
        self.assertEqual(len(ctx.history_turns), 10)   # wording window stays bounded
        query = contextualize_customer_query("What should I do next?", ctx).casefold()
        self.assertIn("cool", query)
        self.assertNotIn("isn't working", query)

    def test_summary_lists_only_what_the_customer_said(self):
        from core.services.conversation_context import conversation_summary_text, is_summary_request
        self._say("customer", "There is a noise in the machine")
        self._say("assistant", "answer", {"response_type": "DIAGNOSTIC_GUIDANCE",
                                          "clarification_question": "When does the noise happen?"})
        self._say("customer", "Only at startup")
        self._say("assistant", "answer", {"response_type": "DIAGNOSTIC_GUIDANCE"})
        for text in ("Can you summarize everything I've told you?", "Give me a recap", "What have I told you so far?"):
            self.assertTrue(is_summary_request(text), text)
        self.assertFalse(is_summary_request("What causes noise at startup?"))
        summary = conversation_summary_text(extract_conversation_context(self.conv, "summarize"))
        self.assertIn("There is a noise in the machine", summary)
        self.assertIn("Only at startup", summary)
        self.assertIn("No service request has been created.", summary)

    def test_ticket_complaint_skips_greetings_and_keeps_corrections(self):
        from core.services.conversation_ticket import build_conversation_ticket_description
        self._say("customer", "Hi")
        self._say("assistant", "Hello!", {"response_type": "GREETING"})
        self._say("customer", "My AC isn't working.")
        self._say("assistant", "answer", {"response_type": "DIAGNOSTIC_GUIDANCE"})
        self._say("customer", "Actually, the AC starts, but it doesn't cool.")
        built = build_conversation_ticket_description(self.conv)
        self.assertEqual(built["complaint"], "My AC isn't working.")
        first = built["sections"]["1. Customer-reported problem"]
        self.assertIn("Current problem (later customer statement): the AC starts, but it doesn't cool", first)
        self.assertFalse(any(line.endswith("] Hi (customer-reported, not independently verified)")
                             for line in built["sections"]["3. Symptoms and observations reported by the customer"]))


class SummaryThroughGatewayTests(SplitACConversationTests):
    def test_summary_needs_no_retrieval(self):
        uuid = self.create()
        self.ask(uuid, "HI")
        self.ask(uuid, "THERE IS A NOISE IN THE MACHINE")
        with mock.patch("core.services.rag.ask_rag", side_effect=AssertionError("RAG must not run")):
            summary, _ = self.ask(uuid, "Can you summarize everything I've told you?")
        self.assertEqual(summary["response_type"], "CONVERSATION_SUMMARY")
        self.assertIn("NOISE IN THE MACHINE", summary["answer"])
        self.assertFalse((summary.get("presentation") or {}).get("citations"))


class RagMemoryAttributionTests(SimpleTestCase):
    def test_questions_do_not_create_reported_symptoms(self):
        from core.services.conversation_context import _record_customer_evidence
        ctx = ConversationContext(conversation_uuid="")
        _record_customer_evidence(ctx, "Could a loud noise mean a fan fault?", [])
        self.assertEqual(ctx.reported_symptoms, [])
        self.assertEqual(ctx.customer_observations, [])
        self.assertEqual(ctx.original_complaint, "")
        self.assertEqual(ctx.active_issue, "")

    def test_mixed_report_and_hypothetical_question(self):
        from core.services.conversation_context import _record_customer_evidence
        ctx = ConversationContext(conversation_uuid="")
        _record_customer_evidence(ctx, "The AC is not cooling. Could water leakage explain it?", [])
        self.assertIn("not cooling", ctx.reported_symptoms)
        self.assertNotIn("water leak", ctx.reported_symptoms)
        self.assertEqual(ctx.original_complaint, "The AC is not cooling.")
        self.assertFalse(ctx.answers_pending_question)

    def test_question_cannot_change_negative_observations(self):
        from core.services.conversation_context import _record_customer_evidence
        ctx = ConversationContext(conversation_uuid="")
        _record_customer_evidence(ctx, "No water leak is visible.", [])
        before = list(ctx.negative_observations)
        self.assertIn("water leak", before)
        _record_customer_evidence(ctx, "Why could water leak from the unit?", [])
        self.assertEqual(ctx.negative_observations, before)


class SharedAttributionRegressionTests(TestCase):
    def setUp(self):
        self.tenant = Tenant.objects.create(name="Attribution synthetic", slug="attribution-synthetic")
        self.customer = Customer.objects.create(tenant=self.tenant, name="Test customer")
        self.conv = CustomerConversation.objects.create(tenant=self.tenant, customer=self.customer, status="ACTIVE")

    def say(self, role, text, contract=None):
        CustomerMessage.objects.create(tenant=self.tenant, conversation=self.conv,
                                       sender_role=role, message_text=text,
                                       structured_contract=contract or {})

    def test_questions_do_not_become_observations_or_ticket_complaint(self):
        from core.services.conversation_ticket import build_conversation_ticket_description, SECTION_TITLES
        self.say("customer", "Can you help me with my AC?")
        self.say("customer", "Could water be leaking from the unit?")
        self.say("customer", "The AC is not cooling.")
        self.say("assistant", "Is there any noise?", {"clarification_question": "Is there any noise?"})
        self.say("customer", "What causes a loud noise?")
        self.say("customer", "Can you create a call for me?")
        context = extract_conversation_context(self.conv, "")
        self.assertEqual(context.original_complaint, "The AC is not cooling.")
        self.assertNotIn("noise", context.reported_symptoms)
        self.assertNotIn("water leak", context.reported_symptoms)
        self.assertFalse(any("loud noise" in item for item in context.customer_observations))
        built = build_conversation_ticket_description(self.conv)
        self.assertEqual(built["complaint"], "The AC is not cooling.")
        self.assertFalse(any("loud noise" in item for item in built["sections"][SECTION_TITLES[2]]))
        self.assertIn("What causes a loud noise?", built["description"])
        self.assertFalse(any("create a call" in item for item in built["sections"][SECTION_TITLES[4]]))

    def test_current_and_persisted_turns_use_same_attribution(self):
        self.say("customer", "The AC is not cooling.")
        current = "There is no water leak. Could a fan fault explain this?"
        before = extract_conversation_context(self.conv, current)
        self.say("customer", current)
        after = extract_conversation_context(self.conv, "")
        self.assertEqual(before.reported_symptoms, after.reported_symptoms)
        self.assertEqual(before.negative_observations, after.negative_observations)
        self.assertIn("water leak", after.negative_observations)
        self.assertNotIn("fan fault", after.reported_symptoms)


class SplitACLiveTranscriptRegressionTests(GatewayTestBase):
    def setUp(self):
        super().setUp()
        install_split_ac_fixture(self)

    create = VRFConversationGatewayTests.create
    ask = VRFConversationGatewayTests.ask
    segments = staticmethod(VRFConversationGatewayTests.segments)
    assert_citations_resolve = VRFConversationGatewayTests.assert_citations_resolve

    def test_live_transcript_keeps_cause_reference_and_negative_observation(self):
        uuid = self.create()
        for message in ("hi", "My AC is not working properly", "poor cooling",
                        "It turns on but the room is not getting cold"):
            self.ask(uuid, message)
        overview, _ = self.ask(uuid, "What could be the reason?")
        causes = listed_causes(overview)
        self.assertGreaterEqual(len(causes), 2)
        selected = causes[1]["source_text"]
        detail, _ = self.ask(uuid, "Can you explain the second cause?")
        self.assertIn(selected.casefold(), detail["answer"].casefold())
        simpler, _ = self.ask(uuid, "Explain it in simple words")
        self.assertIn(selected.casefold(), simpler["answer"].casefold())
        self.assert_citations_resolve(simpler)
        weak, _ = self.ask(uuid, "The airflow is also very weak")
        self.assertIn("airflow", weak["answer"].casefold())
        absent, _ = self.ask(uuid, "There is no water leakage")
        self.assertEqual(absent["reason_code"], "NEGATIVE_OBSERVATION")
        self.assertNotIn("condensate drain", absent["answer"].casefold())
        conv = CustomerConversation.objects.get(conversation_uuid=uuid)
        state = extract_conversation_context(conv, "")
        self.assertIn("water leak", state.negative_observations)
        self.assertNotIn("water leak", state.reported_symptoms)
        self.assertIn("weak airflow", state.reported_symptoms)
        self.assertIn("cooling", state.active_issue.casefold())
        self.assertNotIn("water leak", state.active_issue.casefold())
        matched, _ = self.ask(uuid, "Which of the possible causes matches my problem?")
        self.assertEqual(matched["response_type"], "DIAGNOSTIC_GUIDANCE", matched["answer"])
        self.assertIn("airflow", matched["answer"].casefold())
        for unrelated in ("condensate drain", "breaker", "mounting fastener"):
            self.assertNotIn(unrelated, matched["answer"].casefold())
        self.assert_citations_resolve(matched)


class PastedAnswerAttributionTests(TestCase):
    def test_pasted_assistant_causes_are_not_customer_reports(self):
        tenant = Tenant.objects.create(name="Pasted history synthetic", slug="pasted-history-synthetic")
        customer = Customer.objects.create(tenant=tenant, name="Test customer")
        conv = CustomerConversation.objects.create(tenant=tenant, customer=customer, status="ACTIVE")
        for role, message in (("customer", "Poor cooling."),
                              ("assistant", "Possible causes:\n- Water leakage. [1]\n- Noise. [2]")):
            CustomerMessage.objects.create(tenant=tenant, conversation=conv,
                                           sender_role=role, message_text=message)
        message = "- Water leakage. [1]\n- Noise. [2]\nExplain these causes."
        before = extract_conversation_context(conv, message)
        self.assertNotIn("water leak", before.reported_symptoms)
        self.assertNotIn("noise", before.reported_symptoms)
        CustomerMessage.objects.create(tenant=tenant, conversation=conv, sender_role="customer",
                                       message_text=message)
        after = extract_conversation_context(conv, "")
        self.assertEqual(before.reported_symptoms, after.reported_symptoms)
        self.assertEqual(before.active_issue, after.active_issue)
