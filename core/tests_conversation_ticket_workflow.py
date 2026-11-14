"""Two-stage customer workflow: asset conversation (API 1) -> ServiceCall (API 2).

All data is synthetic and lives only in the throwaway test database. The turn
pipeline is the real customer conversation service in its test-isolated mode.
"""
import threading

from django.contrib.auth.models import User
from django.core.cache import cache
from django.db import connection
from django.test import TestCase, TransactionTestCase, override_settings
from django.urls import reverse

from core.models import (
    Asset, Brand, Customer, CustomerConversation, CustomerMessage, KnowledgeDocument,
    Product, ProductCategory, ProductDomain, ServiceCall, Site, Tenant, TenantMembership,
)
from core.services.conversation_ticket import (
    SECTION_TITLES, build_conversation_ticket_description, missing_contact_fields)
from core.services.indexing import index_document

SETTINGS = dict(SERVY_CUSTOMER_DATA_MODE="test_isolated", SERVY_LLM_PROVIDER="extractive",
                SERVY_RAG_MIN_SCORE=0.0)


def build_world(test):
    cache.clear()
    test.tenant = Tenant.objects.create(name="Synthetic HVAC Tenant", slug="synthetic-hvac", is_active=True)
    test.other_tenant = Tenant.objects.create(name="Synthetic Other Tenant", slug="synthetic-other", is_active=True)
    domain = ProductDomain.objects.create(tenant=test.tenant, name="HVAC")
    category = ProductCategory.objects.create(tenant=test.tenant, domain=domain, name="Air Conditioning")
    brand = Brand.objects.create(tenant=test.tenant, name="Synthetic Brand")
    split = Product.objects.create(tenant=test.tenant, category=category, brand=brand, name="Synthetic Split AC 2T")
    chiller = Product.objects.create(tenant=test.tenant, category=category, brand=brand, name="Synthetic Chiller 10TR")
    test.customer = Customer.objects.create(
        tenant=test.tenant, name="Synthetic Customer One", contact_name="Test Contact",
        phone="+1 555 0100", email="contact@example.invalid")
    test.customer_no_contact = Customer.objects.create(
        tenant=test.tenant, name="Synthetic Customer Two", contact_name="Second Contact")
    test.other_customer = Customer.objects.create(tenant=test.other_tenant, name="Synthetic Other Customer",
                                                  contact_name="Other", phone="+1 555 0199")
    test.site = Site.objects.create(tenant=test.tenant, customer=test.customer, name="Synthetic Site A")
    site2 = Site.objects.create(tenant=test.tenant, customer=test.customer_no_contact, name="Synthetic Site B")
    other_site = Site.objects.create(tenant=test.other_tenant, customer=test.other_customer, name="Other Site")
    test.split = Asset.objects.create(tenant=test.tenant, customer=test.customer, site=test.site, product=split,
                                      name="Synthetic Split AC", asset_code="SYN-AC-01", model_number="SYN-2T")
    test.chiller = Asset.objects.create(tenant=test.tenant, customer=test.customer, site=test.site, product=chiller,
                                        name="Synthetic Chiller", asset_code="SYN-CH-01", model_number="SYN-10TR")
    test.split2 = Asset.objects.create(tenant=test.tenant, customer=test.customer_no_contact, site=site2,
                                       product=split, name="Synthetic Split AC B", asset_code="SYN-AC-02")
    test.other_asset = Asset.objects.create(tenant=test.other_tenant, customer=test.other_customer, site=other_site,
                                            name="Other Tenant AC", asset_code="OTH-AC-01")
    test.user = User.objects.create_user("syn_customer", password="SafePass!123")
    TenantMembership.objects.create(user=test.user, tenant=test.tenant, role="customer", customer=test.customer)
    test.user2 = User.objects.create_user("syn_customer2", password="SafePass!123")
    TenantMembership.objects.create(user=test.user2, tenant=test.tenant, role="customer", customer=test.customer_no_contact)
    test.other_user = User.objects.create_user("syn_other", password="SafePass!123")
    TenantMembership.objects.create(user=test.other_user, tenant=test.other_tenant, role="customer",
                                    customer=test.other_customer)
    test.split_doc = KnowledgeDocument.objects.create(
        tenant=test.tenant, product=split, title="Synthetic Split AC Troubleshooting Guide", source_type="text",
        is_rag_enabled=True, is_confidential=False,
        content_text=("Symptom: poor cooling.\nPossible causes:\n- Dirty air filter restricting airflow.\n"
                      "- Blocked outdoor condenser coil.\nObservations:\n- Observe whether the indoor fan is running.\n"
                      "- Note the room temperature shown on the remote controller."))
    index_document(test.split_doc)
    test.chiller_doc = KnowledgeDocument.objects.create(
        tenant=test.tenant, product=chiller, title="Synthetic Chiller Operating Guide", source_type="text",
        is_rag_enabled=True, is_confidential=False,
        content_text=("Symptom: high leaving water temperature.\nPossible causes:\n- Fouled evaporator tubes.\n"
                      "Observations:\n- Note the chilled water supply temperature on the controller display."))
    index_document(test.chiller_doc)


@override_settings(**SETTINGS)
class ConversationTicketWorkflowTests(TestCase):
    def setUp(self):
        build_world(self)
        self.client.force_login(self.user)

    # -- helpers ---------------------------------------------------------------------
    def start(self, asset, message=None, client=None, **extra):
        body = {"asset_id": asset.id, **extra}
        if message is not None:
            body["message"] = message
        return (client or self.client).post(reverse("api_customer_conversations"), body,
                                            content_type="application/json")

    def say(self, uuid, message, client=None):
        return (client or self.client).post(
            reverse("api_customer_conversation_messages", kwargs={"conversation_uuid": uuid}),
            {"message": message}, content_type="application/json")

    def ticket_url(self, uuid):
        return reverse("api_customer_conversation_ticket", kwargs={"conversation_uuid": uuid})

    # -- API 1 -------------------------------------------------------------------------
    def test_start_with_first_message_answers_and_persists_both_messages(self):
        r = self.start(self.split, "My split AC is running but not cooling.")
        self.assertEqual(r.status_code, 201, r.content[:400])
        data = r.json()
        turn = data["first_turn"]
        self.assertEqual(turn["user_message"]["message_text"], "My split AC is running but not cooling.")
        self.assertTrue(turn["assistant_message"]["message_text"])
        contract = turn["assistant_message"]["structured_contract"]
        titles = {c.get("title") for c in (contract.get("references") or contract.get("citations") or [])}
        self.assertIn("Synthetic Split AC Troubleshooting Guide", titles)
        self.assertNotIn("Synthetic Chiller Operating Guide", titles)
        conv = CustomerConversation.objects.get(conversation_uuid=data["conversation_uuid"])
        self.assertEqual(conv.asset_id, self.split.id)
        self.assertEqual(list(conv.messages.order_by("id").values_list("sender_role", flat=True)),
                         ["customer", "assistant"])

    def test_start_without_message_is_backward_compatible(self):
        r = self.start(self.split)
        self.assertEqual(r.status_code, 201)
        self.assertNotIn("first_turn", r.json())
        conv = CustomerConversation.objects.get(conversation_uuid=r.json()["conversation_uuid"])
        self.assertEqual(conv.messages.count(), 0)
        r = self.start(self.split, complaint="Stored only")
        conv = CustomerConversation.objects.get(conversation_uuid=r.json()["conversation_uuid"])
        self.assertEqual(list(conv.messages.values_list("sender_role", "message_text")), [("customer", "Stored only")])

    def test_first_message_requires_an_owned_asset(self):
        r = self.client.post(reverse("api_customer_conversations"), {"message": "help"},
                             content_type="application/json")
        self.assertEqual(r.status_code, 400)
        for asset in (self.split2, self.other_asset):
            with self.subTest(asset=asset.asset_code):
                r = self.start(asset, "help")
                self.assertEqual(r.status_code, 400)
        self.assertEqual(CustomerConversation.objects.count(), 0)

    def test_continuation_persists_every_turn_and_survives_a_refresh(self):
        uuid = self.start(self.split, "My split AC is not cooling.").json()["conversation_uuid"]
        for text in ("The indoor fan is running.", "The filter looks dusty.", "The room is still warm."):
            self.assertEqual(self.say(uuid, text).status_code, 200)
        # "Refresh": a brand-new client session reloads the conversation from MySQL.
        self.client.logout()
        self.client.force_login(self.user)
        r = self.client.get(reverse("api_customer_conversation_messages", kwargs={"conversation_uuid": uuid}))
        self.assertEqual(r.status_code, 200)
        roles = [m["sender_role"] for m in r.json()["messages"]]
        texts = [m["message_text"] for m in r.json()["messages"] if m["sender_role"] == "customer"]
        self.assertEqual(roles.count("customer"), 4)
        self.assertEqual(roles.count("assistant"), 4)
        self.assertEqual(texts, ["My split AC is not cooling.", "The indoor fan is running.",
                                 "The filter looks dusty.", "The room is still warm."])

    def test_two_conversations_on_the_same_asset_stay_isolated(self):
        a = self.start(self.split, "First incident: not cooling.").json()["conversation_uuid"]
        b = self.start(self.split, "Second incident: water dripping.").json()["conversation_uuid"]
        self.assertNotEqual(a, b)
        self.say(a, "Follow-up only for A.")
        self.say(b, "Follow-up only for B.")
        conv_a = CustomerConversation.objects.get(conversation_uuid=a)
        conv_b = CustomerConversation.objects.get(conversation_uuid=b)
        texts_a = set(conv_a.messages.values_list("message_text", flat=True))
        texts_b = set(conv_b.messages.values_list("message_text", flat=True))
        self.assertIn("Follow-up only for A.", texts_a)
        self.assertNotIn("Follow-up only for B.", texts_a)
        self.assertIn("Follow-up only for B.", texts_b)
        self.assertNotIn("First incident: not cooling.", texts_b)

    def test_conversations_for_different_assets_do_not_leak_sources(self):
        split = self.start(self.split, "What can cause poor cooling?").json()
        chiller = self.start(self.chiller, "What can cause high leaving water temperature?").json()

        def titles(turn):
            contract = turn["assistant_message"]["structured_contract"]
            return {c.get("title") for c in (contract.get("references") or contract.get("citations") or [])}
        self.assertNotIn("Synthetic Chiller Operating Guide", titles(split["first_turn"]))
        self.assertNotIn("Synthetic Split AC Troubleshooting Guide", titles(chiller["first_turn"]))

    def test_other_customers_cannot_read_continue_preview_or_convert(self):
        uuid = self.start(self.split, "My split AC is not cooling.").json()["conversation_uuid"]
        for user in (self.user2, self.other_user):
            with self.subTest(user=user.username):
                self.client.force_login(user)
                self.assertEqual(self.client.get(reverse("api_customer_conversation_messages",
                                                         kwargs={"conversation_uuid": uuid})).status_code, 404)
                self.assertEqual(self.say(uuid, "intrusion").status_code, 404)
                self.assertEqual(self.client.get(self.ticket_url(uuid)).status_code, 404)
                r = self.client.post(self.ticket_url(uuid), {"confirm_ticket_creation": True},
                                     content_type="application/json")
                self.assertEqual(r.status_code, 404)
        self.client.logout()
        self.assertEqual(self.client.get(self.ticket_url(uuid)).status_code, 403)
        self.assertEqual(ServiceCall.objects.count(), 0)
        self.assertFalse(CustomerMessage.objects.filter(message_text="intrusion").exists())

    def test_unsupported_question_is_not_answered_with_invented_content(self):
        r = self.start(self.split, "What is the warranty period for the compressor in years?")
        contract = r.json()["first_turn"]["assistant_message"]["structured_contract"]
        answer = r.json()["first_turn"]["assistant_message"]["message_text"]
        self.assertNotRegex(answer, r"\b\d+\s*(?:years?|yrs?)\b")
        self.assertNotIn(contract.get("response_type"), ("RESOLVED",))

    def test_unsupported_question_after_a_symptom_does_not_repeat_the_previous_answer(self):
        uuid = self.start(self.split, "What can cause poor cooling?").json()["conversation_uuid"]
        first = CustomerConversation.objects.get(conversation_uuid=uuid).messages.filter(
            sender_role="assistant").first().message_text
        r = self.say(uuid, "What is the warranty period for the compressor in years?")
        self.assertEqual(r.status_code, 200)
        answer = r.json()["assistant_message"]["message_text"]
        self.assertNotRegex(answer, r"\b\d+\s*(?:years?|yrs?)\b")
        if "Dirty air filter restricting airflow" in first:
            self.assertNotIn("Dirty air filter restricting airflow", answer)

    # -- API 2 -------------------------------------------------------------------------
    def test_preview_is_read_only_and_matches_the_created_service_call(self):
        uuid = self.start(self.split, "My split AC is running but not cooling.").json()["conversation_uuid"]
        self.say(uuid, "The indoor fan is running but the room stays warm.")
        conv = CustomerConversation.objects.get(conversation_uuid=uuid)
        before = (ServiceCall.objects.count(), conv.messages.count(), conv.status)
        r = self.client.get(self.ticket_url(uuid))
        self.assertEqual(r.status_code, 200)
        preview = r.json()["preview"]
        conv.refresh_from_db()
        self.assertEqual((ServiceCall.objects.count(), conv.messages.count(), conv.status), before)
        self.assertTrue(preview["can_create"])
        self.assertEqual(preview["missing_fields"], [])
        for title in SECTION_TITLES:
            self.assertIn(title, preview["description"])
        self.assertIn("My split AC is running but not cooling.", preview["description"])
        self.assertIn("The indoor fan is running but the room stays warm.", preview["description"])
        self.assertIn(f"Asset ID {self.split.id}", preview["description"])
        self.assertIn("Synthetic Split AC Troubleshooting Guide", preview["description"])
        self.assertIn("No diagnosis has been confirmed", preview["description"])

        r = self.client.post(self.ticket_url(uuid), {"confirm_ticket_creation": True},
                             content_type="application/json")
        self.assertEqual(r.status_code, 200, r.content[:400])
        call = ServiceCall.objects.get()
        self.assertIn(preview["description"], call.technician_notes)
        self.assertEqual(call.complaint_text, "My split AC is running but not cooling.")
        # Relationships: customer, contact (from the customer record), asset, site, conversation.
        conv.refresh_from_db()
        self.assertEqual(conv.linked_service_call_id, call.id)
        self.assertEqual(conv.status, "TICKETED")
        self.assertEqual((call.tenant_id, call.customer_id, call.asset_id, call.site_id),
                         (self.tenant.id, self.customer.id, self.split.id, self.site.id))
        self.assertEqual((call.contact_name, call.contact_phone, call.contact_email),
                         ("Test Contact", "+1 555 0100", "contact@example.invalid"))
        self.assertEqual(r.json()["ticket"]["servy_id"], call.servy_id)

    def test_repeated_ticket_requests_return_the_same_service_call(self):
        uuid = self.start(self.split, "Not cooling.").json()["conversation_uuid"]
        first = self.client.post(self.ticket_url(uuid), {"confirm_ticket_creation": True},
                                 content_type="application/json").json()["ticket"]
        second = self.client.post(self.ticket_url(uuid), {"confirm_ticket_creation": True},
                                  content_type="application/json").json()["ticket"]
        self.assertEqual(first["servy_id"], second["servy_id"])
        self.assertEqual(ServiceCall.objects.count(), 1)
        self.assertEqual(self.client.get(self.ticket_url(uuid)).json()["ticket"]["servy_id"], first["servy_id"])

    def test_confirmation_is_required(self):
        uuid = self.start(self.split, "Not cooling.").json()["conversation_uuid"]
        for body in ({}, {"confirm_ticket_creation": "true"}, {"confirm_ticket_creation": False}):
            with self.subTest(body=body):
                r = self.client.post(self.ticket_url(uuid), body, content_type="application/json")
                self.assertEqual(r.status_code, 400)
        self.assertEqual(ServiceCall.objects.count(), 0)

    def test_missing_contact_details_are_reported_not_invented(self):
        self.client.force_login(self.user2)
        uuid = self.start(self.split2, "My split AC is not cooling.").json()["conversation_uuid"]
        preview = self.client.get(self.ticket_url(uuid)).json()["preview"]
        self.assertEqual(preview["missing_fields"], ["phone_or_email"])
        self.assertFalse(preview["can_create"])
        r = self.client.post(self.ticket_url(uuid), {"confirm_ticket_creation": True},
                             content_type="application/json")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["missing_fields"], ["phone_or_email"])
        self.assertEqual(ServiceCall.objects.count(), 0)
        self.customer_no_contact.refresh_from_db()
        self.assertEqual((self.customer_no_contact.phone, self.customer_no_contact.email), ("", ""))

    def test_contact_rules(self):
        cases = (
            (dict(contact_name="A", phone="+44 20 7946 0000", email=""), []),
            (dict(contact_name="A", phone="", email="a@example.invalid"), []),
            (dict(contact_name="", phone="+1 555 0100", email=""), ["contact_name"]),
            (dict(contact_name="A", phone="abc", email="not-an-email"), ["phone_or_email"]),
            (dict(contact_name=" ", phone="", email=""), ["contact_name", "phone_or_email"]),
        )
        for fields, expected in cases:
            with self.subTest(fields=fields):
                self.assertEqual(missing_contact_fields(Customer(**fields)), expected)

    def test_long_history_is_summarised_completely_in_order(self):
        conv = CustomerConversation.objects.create(tenant=self.tenant, customer=self.customer, site=self.site,
                                                   asset=self.split, created_by=self.user, status="ACTIVE")
        for i in range(1, 41):
            CustomerMessage.objects.create(tenant=self.tenant, conversation=conv, sender_role="customer",
                                           message_text=f"customer statement number {i:02d}")
            CustomerMessage.objects.create(tenant=self.tenant, conversation=conv, sender_role="assistant",
                                           message_text=f"assistant reply number {i:02d}?",
                                           structured_contract={"response_type": "CLARIFICATION_REQUIRED"})
        built = build_conversation_ticket_description(conv)
        self.assertEqual(built["message_count"], 80)
        positions = [built["description"].index(f"customer statement number {i:02d}") for i in range(1, 41)]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("assistant reply number 01?", built["description"])
        self.assertIn("assistant reply number 40?", built["description"])
        self.assertIn("Unanswered question: assistant reply number 40?", built["description"])


@override_settings(**SETTINGS)
class ConcurrentTicketCreationTests(TransactionTestCase):
    """Two simultaneous confirmations for one conversation create one ServiceCall."""

    def setUp(self):
        build_world(self)

    def test_concurrent_confirmations_create_exactly_one_call(self):
        if connection.vendor == "sqlite":
            self.skipTest("row locking needs a server database (MySQL)")
        from django.test import Client
        owner = Client()
        owner.force_login(self.user)
        uuid = owner.post(reverse("api_customer_conversations"),
                          {"asset_id": self.split.id, "message": "Not cooling."},
                          content_type="application/json").json()["conversation_uuid"]
        url = reverse("api_customer_conversation_ticket", kwargs={"conversation_uuid": uuid})
        results, barrier = [], threading.Barrier(4)

        def confirm():
            client = Client()
            client.force_login(self.user)
            barrier.wait()
            body = client.post(url, {"confirm_ticket_creation": True}, content_type="application/json").json()
            results.append((body["ticket"]["servy_id"], body.get("created")))
            connection.close()

        threads = [threading.Thread(target=confirm) for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(len({servy_id for servy_id, _ in results}), 1, results)
        self.assertEqual(ServiceCall.objects.count(), 1)
        # Exactly one request reports the creation; the others report the existing call.
        self.assertEqual(sorted(created for _, created in results), [False, False, False, True], results)
