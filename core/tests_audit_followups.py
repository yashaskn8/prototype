"""Focused regressions for the follow-up audit: natural wording, conversation intents,
long-conversation cause memory, store-role privacy, cause identity across headings,
and the original-document gate with real files. Synthetic data only."""
import hashlib
import json
import tempfile
from types import SimpleNamespace
from unittest import mock

from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.test import SimpleTestCase, TestCase, override_settings
from rest_framework.test import APIClient

from core.models import (
    Asset, Brand, Customer, CustomerConversation, CustomerMessage, KnowledgeChunk, KnowledgeDocument, Product,
    ProductCategory, ProductDomain, ServiceCall, Site, SourceProvenanceRecord, Tenant, TenantMembership)
from core.services.conversation_context import (
    ConversationContext, conversation_intent, conversation_intent_reply)
from core.services.customer_answer import limitation_message, no_documentation_message
from core.services.customer_response_composer import compose_reply_acknowledgement
from core.tests_conversation_state import install_split_ac_fixture
from core.tests_conversational_gateway import VRFConversationGatewayTests
from core.tests_demo_gateway import GatewayTestBase

INTERNAL_WORDING = ("insufficient evidence", "approved manual", "cited answer", "more evidence needed",
                    "exact cited", "the manual")


class NaturalWordingTests(SimpleTestCase):
    def test_limitation_is_plain_keeps_context_and_offers_a_technician_once(self):
        asset = SimpleNamespace(name="Split AC Unit 2T")
        ctx = SimpleNamespace(active_issue="THERE IS A NOISE IN THE MACHINE",
                              resolved_answers=[{"question": "When?", "answer": "ONLY AT STARTUP"}],
                              last_assistant_answer="Thanks, I've noted that.")
        text = limitation_message(asset, ctx)
        self.assertTrue(text.startswith("I can't confirm that"))
        self.assertIn("ONLY AT STARTUP", text)
        self.assertIn("record this for a technician", text)
        for phrase in INTERNAL_WORDING:
            self.assertNotIn(phrase, text.casefold())
        ctx.last_assistant_answer = "I can record this for a technician in a service request if you'd like."
        self.assertNotIn("technician", limitation_message(asset, ctx))
        self.assertNotIn("insufficient", no_documentation_message(asset).casefold())

    def test_reply_acknowledgement_is_natural_and_not_repeated(self):
        ctx = SimpleNamespace(answers_pending_question=True, active_topics=["noise or vibration"],
                              resolved_answers=[{"question": "When?", "answer": "Only at startup."}],
                              last_assistant_answer="Here are the possible causes of noise.")
        text = compose_reply_acknowledgement(ctx, "Split AC Unit 2T")["text"]
        self.assertIn("only at startup", text)
        self.assertIn("I can't confirm the exact cause of the noise or vibration", text)
        self.assertIn("record this for a technician", text)
        for phrase in INTERNAL_WORDING:
            self.assertNotIn(phrase, text.casefold())
        ctx.last_assistant_answer = "I can record this for a technician if you'd like."
        self.assertNotIn("technician", compose_reply_acknowledgement(ctx, "Split AC Unit 2T")["text"])


class IntentClassificationTests(SimpleTestCase):
    CASES = {
        "ticket_status": ["Did you actually create the ticket?", "Have you raised a service request?",
                          "What's my ticket number?", "Is there an open ticket?"],
        "ticket_help": ["Create a ticket for this issue.", "Can you raise a service request?",
                        "How do I book a technician?"],
        "ticket_contents": ["What will be in the ticket?", "What will the technician see?"],
        "certainty": ["Is this definitely a low-pressure fault?", "Are you sure?", "How confident are you?",
                      "Is that the actual cause?"],
        None: ["Is it the filter?", "What causes noise at startup?", "It certainly makes a loud noise.",
               "Water is leaking, please raise a ticket", "What should I do next?"],
    }

    def test_intents(self):
        for intent, texts in self.CASES.items():
            for text in texts:
                with self.subTest(text=text):
                    self.assertEqual(conversation_intent(text), intent)

    def test_replies_are_truthful_about_tickets_and_certainty(self):
        ctx = ConversationContext(conversation_uuid="", asset_name="Split AC Unit 2T",
                                  original_complaint="Unit stopped working")
        self.assertTrue(conversation_intent_reply("ticket_status", ctx).startswith("No, I haven't created"))
        open_ticket = {"ref": "SRV-42831", "status": "assigned"}
        self.assertIn("SRV-42831", conversation_intent_reply("ticket_status", ctx, open_ticket))
        self.assertIn("won't create another", conversation_intent_reply("ticket_help", ctx, open_ticket))
        self.assertIn("only after you confirm", conversation_intent_reply("ticket_help", ctx))
        contents = conversation_intent_reply("ticket_contents", ctx)
        self.assertIn("Unit stopped working", contents)
        self.assertIn("doesn't include a confirmed diagnosis", contents)
        self.assertTrue(conversation_intent_reply("certainty", ctx).startswith("No, I can't confirm"))


class _SplitACGateway(GatewayTestBase):
    def setUp(self):
        super().setUp()
        install_split_ac_fixture(self, getattr(self, "MANUAL", None))

    create = VRFConversationGatewayTests.create
    ask = VRFConversationGatewayTests.ask


class IntentsThroughGatewayTests(_SplitACGateway):
    def test_ticket_and_certainty_questions_need_no_retrieval_and_never_create_tickets(self):
        uuid = self.create()
        self.ask(uuid, "WHAT ARE THE POSSIBLE CAUSES THE SPLIT AC UNIT STOPPED WORKING")
        with mock.patch("core.services.rag.ask_rag", side_effect=AssertionError("RAG must not run")):
            created, _ = self.ask(uuid, "Create a ticket for this issue.")
            self.assertEqual((created["response_type"], created["reason_code"]), ("CONVERSATION_REPLY", "TICKET_HELP"))
            self.assertEqual(ServiceCall.objects.count(), 0)
            status, _ = self.ask(uuid, "Did you actually create the ticket?")
            self.assertTrue(status["answer"].startswith("No, I haven't created"))
            sure, _ = self.ask(uuid, "Is this definitely a low-pressure fault?")
            self.assertTrue(sure["answer"].startswith("No, I can't confirm that."))
            self.assertFalse((sure.get("presentation") or {}).get("citations"))
        # A safety-guarded message is never short-circuited by the intent route.
        unsafe, _ = self.ask(uuid, "How do I bypass the breaker so I can raise a ticket?")
        self.assertNotEqual(unsafe["response_type"], "CONVERSATION_REPLY", unsafe.get("answer"))
        r = self.as_customer("post", f"/demo-api/customer-support/conversations/{uuid}/ticket/",
                             data={"confirm_ticket_creation": "true"})
        self.assertIn(r.status_code, (200, 201), r.content[:300])
        with mock.patch("core.services.rag.ask_rag", side_effect=AssertionError("RAG must not run")):
            status, _ = self.ask(uuid, "Did you actually create the ticket?")
        self.assertTrue(status["answer"].startswith("Yes. Service request SRV-"), status["answer"])


class LongConversationCauseMemoryTests(_SplitACGateway):
    def test_first_cause_still_resolves_after_50_more_messages(self):
        uuid = self.create()
        overview, _ = self.ask(uuid, "WHAT ARE THE POSSIBLE CAUSES THE SPLIT AC UNIT STOPPED WORKING")
        first = next(c for c in overview["presentation"]["response"]["displayed_claims"]
                     if c["kind"] == "possible_cause")
        conv = CustomerConversation.objects.get(conversation_uuid=uuid)
        for i in range(27):   # 54 more saved messages
            CustomerMessage.objects.create(tenant=self.tenant, conversation=conv, sender_role="customer",
                                           message_text=f"Thanks, noted {i}.")
            CustomerMessage.objects.create(tenant=self.tenant, conversation=conv, sender_role="assistant",
                                           message_text="Okay.", structured_contract={"response_type": "GREETING"})
        detail, _ = self.ask(uuid, "Tell me more about the first cause.")
        self.assertIn(first["source_text"].rstrip(".").casefold(), detail["answer"].casefold())


class DistinctCauseIdentityTests(_SplitACGateway):
    MANUAL = """3. SYMPTOM: POOR COOLING
Possible causes:
- Dirty filter or coil.
- Restricted airflow or blocked outdoor unit clearance.

5. SYMPTOM: ICE ON COIL OR PIPE
Possible causes:
- Dirty filter or coil.
- Sensor issue or refrigerant-system problem.
"""

    def test_identical_cause_under_two_headings_is_kept_twice(self):
        uuid = self.create()
        overview, _ = self.ask(uuid, "WHAT ARE THE POSSIBLE CAUSES THE SPLIT AC UNIT STOPPED WORKING")
        dirty = [c for c in overview["presentation"]["response"]["displayed_claims"]
                 if c["kind"] == "possible_cause" and c["source_text"] == "Dirty filter or coil."]
        self.assertEqual({c["source_heading"] for c in dirty},
                         {"3. SYMPTOM: POOR COOLING", "5. SYMPTOM: ICE ON COIL OR PIPE"}, overview["answer"])


@override_settings(ROOT_URLCONF="servy_rag.urls")
class StorePrivacyTests(GatewayTestBase):
    def test_store_roles_see_no_technician_or_customer_identity(self):
        from core.services.ticketing import create_service_call_record
        call = create_service_call_record(self.tenant, "Cooling", "Not cooling", self.customer, site=self.site,
                                          asset=self.asset, technician=self.tech, contact_name="Synthetic Contact")
        store = User.objects.create(username="store_privacy")
        TenantMembership.objects.create(user=store, tenant=self.tenant, role="store_admin")
        client = APIClient()
        client.force_login(store)
        detail = client.get(f"/api/calls/{call.id}/").json()
        self.assertIsNone(detail["call"]["technician"])
        dashboard = client.get("/api/dashboard/").json()
        body = json.dumps(dashboard)
        self.assertNotIn("Synthetic Development Customer", body)
        self.assertNotIn("Synthetic Technician", body)
        self.assertEqual(dashboard["recent_interactions"], [])


@override_settings(
    SERVY_CUSTOMER_DATA_MODE="verified_real_only", STARLLY_REAL_TENANTS=["real-gate-tenant"],
    STARLLY_REAL_ORGANISATIONS=["273"], SERVY_LLM_PROVIDER="extractive", SERVY_RAG_MIN_SCORE=0.0)
class OriginalDocumentFileGateTests(TestCase):
    """Missing, corrupted and revoked original files never authorize technical claims."""
    CONTENT = (b"3. SYMPTOM: POOR COOLING\nPossible causes:\n- Dirty filter or coil.\n"
               b"- Restricted airflow or blocked outdoor unit clearance.\n")

    def setUp(self):
        from core.services.customer_isolation import clear_test_verified_provenance, register_test_verified_provenance
        from core.services.vectors import embed_text
        self.addCleanup(clear_test_verified_provenance)
        media = tempfile.TemporaryDirectory()
        self.addCleanup(media.cleanup)
        settings_override = override_settings(MEDIA_ROOT=media.name)
        settings_override.enable()
        self.addCleanup(settings_override.disable)
        self.tenant = Tenant.objects.create(name="Real Gate Co", slug="real-gate-tenant")
        domain = ProductDomain.objects.create(tenant=self.tenant, name="HVAC")
        category = ProductCategory.objects.create(tenant=self.tenant, domain=domain, name="Cooling")
        product = Product.objects.create(tenant=self.tenant, category=category,
                                         brand=Brand.objects.create(tenant=self.tenant, name="B"), name="Split AC")
        self.customer = Customer.objects.create(tenant=self.tenant, name="Gate Customer",
                                                contact_name="C", phone="+1 555 0100")
        site = Site.objects.create(tenant=self.tenant, customer=self.customer, name="Plant")
        self.asset = Asset.objects.create(tenant=self.tenant, customer=self.customer, site=site, product=product,
                                          name="Split AC Unit 2T", asset_code="GATE-1", status="active")
        self.doc = KnowledgeDocument.objects.create(
            tenant=self.tenant, product=product, asset=self.asset, title="Split AC Troubleshooting",
            source_type="file", checksum_sha256=hashlib.sha256(self.CONTENT).hexdigest(),
            content_text=self.CONTENT.decode(), is_rag_enabled=True, index_status="INDEXED")
        self.doc.file.save("manual.txt", ContentFile(self.CONTENT), save=True)
        text = self.CONTENT.decode()
        emb = embed_text(text)
        KnowledgeChunk.objects.create(tenant=self.tenant, document=self.doc, chunk_index=0,
                                      heading="3. SYMPTOM: POOR COOLING", text=text, embedding=emb,
                                      content_embedding=emb)
        for obj in (self.customer, self.asset, self.doc):
            register_test_verified_provenance(obj)
        # The company asset/customer registries hold real identifiers; this synthetic
        # tenant is treated as registered there. File, checksum, provenance and
        # revocation checks are not stubbed.
        for name in ("is_verified_real_asset", "is_verified_real_customer"):
            patcher = mock.patch(f"core.services.customer_isolation.{name}",
                                 side_effect=lambda obj: getattr(obj, "tenant_id", None) == self.tenant.id)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.user = User.objects.create_user("gate_customer", password="unused-test-password")
        TenantMembership.objects.create(user=self.user, tenant=self.tenant, role="customer", customer=self.customer)

    def gate(self):
        from core.services.rag import has_authorized_original_documentation
        return has_authorized_original_documentation(tenant=self.tenant, customer=self.customer, asset=self.asset)

    def ask(self):
        client = APIClient()
        client.force_login(self.user)
        r = client.post("/api/customer-support/conversations/",
                        {"asset_id": self.asset.id, "message": "What are the possible causes of poor cooling?"},
                        format="json")
        self.assertEqual(r.status_code, 201, r.content[:300])
        turn = r.json().get("first_turn") or {}
        return (turn.get("assistant_message") or {}).get("structured_contract") or turn.get("contract") or {}

    def assert_no_claims(self, contract):
        presentation = contract.get("presentation") or {}
        self.assertFalse(presentation.get("citations"), contract.get("answer"))
        self.assertNotIn("Dirty filter", json.dumps(contract))

    def test_intact_original_authorizes_cited_guidance(self):
        self.assertTrue(self.gate())
        contract = self.ask()
        self.assertTrue((contract.get("presentation") or {}).get("citations"), contract.get("answer"))

    def test_missing_original_file_blocks_every_claim(self):
        self.doc.file.storage.delete(self.doc.file.name)
        self.assertFalse(self.gate())
        self.assert_no_claims(self.ask())

    def test_corrupted_original_file_blocks_every_claim(self):
        with open(self.doc.file.path, "wb") as handle:
            handle.write(self.CONTENT + b"tampered")
        self.assertFalse(self.gate())
        self.assert_no_claims(self.ask())

    def test_revoked_original_blocks_every_claim(self):
        SourceProvenanceRecord.objects.create(
            tenant=self.tenant, source_system="synthetic", external_object_type="document", external_id="gate-doc",
            local_object_type="KnowledgeDocument", local_object_id=self.doc.id, verification_status="REVOKED")
        self.assertFalse(self.gate())
        self.assert_no_claims(self.ask())
