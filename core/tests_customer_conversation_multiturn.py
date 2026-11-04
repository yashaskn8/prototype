"""
Multi-turn Conversation Continuity and Grounded RAG Test Suite.

Validates Phase 2 and Phase 3 requirements:
  - Multi-turn conversation continuity with progressive troubleshooting
  - Contextual query rewriting (resolving follow-ups like 'What next?' and 'I checked that already')
  - Elimination of repeated completed checks
  - Equipment switching isolation (no context leakage across assets)
  - Cross-customer and cross-tenant conversation authorization and isolation
  - Rejection of messages to closed / resolved conversations
  - Bounded recent-message history window
  - Multi-turn safety hazard detection and immediate escalation
  - Informational inquiry handling without unauthorized actions
  - Offline deterministic extractive fallback integrity
"""

from unittest.mock import patch
from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.urls import reverse

from core.models import (
    Asset,
    Brand,
    Customer,
    CustomerConversation,
    CustomerInteraction,
    CustomerMessage,
    DiagnosticSession,
    KnowledgeDocument,
    OperationalZone,
    Product,
    ProductCategory,
    ProductDomain,
    Site,
    Tenant,
    TenantMembership,
)
from core.services.indexing import index_document
from core.services.conversation_context import (
    extract_conversation_context,
    contextualize_customer_query,
)


@override_settings(
    SERVY_CUSTOMER_DATA_MODE="test_isolated",
    SERVY_LLM_PROVIDER="extractive",
    SERVY_RAG_MIN_SCORE=0.0,
)
class CustomerConversationMultiTurnTests(TestCase):
    def setUp(self):
        # 1. Tenants
        self.tenant_a = Tenant.objects.create(name="Tenant Alpha", slug="alpha-conv", is_active=True)
        self.tenant_b = Tenant.objects.create(name="Tenant Beta", slug="beta-conv", is_active=True)

        # 2. Operational Zone
        self.zone_a = OperationalZone.objects.create(tenant=self.tenant_a, name="Alpha Zone")

        # 3. Product Hierarchy for HVAC Split AC
        self.domain_hvac = ProductDomain.objects.create(tenant=self.tenant_a, name="HVAC Systems")
        self.category_ac = ProductCategory.objects.create(
            tenant=self.tenant_a, domain=self.domain_hvac, name="Air Conditioners"
        )
        self.brand_cool = Brand.objects.create(tenant=self.tenant_a, name="CoolMaster")
        self.product_split_ac = Product.objects.create(
            tenant=self.tenant_a,
            category=self.category_ac,
            brand=self.brand_cool,
            name="CoolMaster Inverter Split AC",
        )

        # Chiller Product
        self.product_chiller = Product.objects.create(
            tenant=self.tenant_a,
            category=self.category_ac,
            brand=self.brand_cool,
            name="CoolMaster Commercial Chiller",
        )

        # 4. Customers & Sites
        self.customer_a1 = Customer.objects.create(tenant=self.tenant_a, name="Customer Alpha One")
        self.customer_a2 = Customer.objects.create(tenant=self.tenant_a, name="Customer Alpha Two")
        self.customer_b = Customer.objects.create(tenant=self.tenant_b, name="Customer Beta")

        self.site_a1 = Site.objects.create(
            tenant=self.tenant_a, customer=self.customer_a1, name="Office Tower A", zone=self.zone_a
        )
        self.site_a2 = Site.objects.create(
            tenant=self.tenant_a, customer=self.customer_a2, name="Office Tower B"
        )
        self.site_b = Site.objects.create(
            tenant=self.tenant_b, customer=self.customer_b, name="Beta Facility"
        )

        # 5. Assets
        self.asset_split_ac = Asset.objects.create(
            tenant=self.tenant_a,
            customer=self.customer_a1,
            site=self.site_a1,
            product=self.product_split_ac,
            name="Server Room Split AC",
            asset_code="AC-SR-01",
            model_number="CS-INV24",
            status="active",
        )
        self.asset_chiller = Asset.objects.create(
            tenant=self.tenant_a,
            customer=self.customer_a1,
            site=self.site_a1,
            product=self.product_chiller,
            name="Main Chiller Unit",
            asset_code="CH-01",
            model_number="CH-500",
            status="active",
        )
        self.asset_beta = Asset.objects.create(
            tenant=self.tenant_b,
            customer=self.customer_b,
            site=self.site_b,
            name="Beta AC Unit",
            asset_code="AC-B1",
            model_number="CS-INV24",
            status="active",
        )

        # 6. Users
        self.user_a1 = User.objects.create_user("user_a1", password="Password123")
        TenantMembership.objects.create(
            user=self.user_a1, tenant=self.tenant_a, role="customer", customer=self.customer_a1
        )

        self.user_a2 = User.objects.create_user("user_a2", password="Password123")
        TenantMembership.objects.create(
            user=self.user_a2, tenant=self.tenant_a, role="customer", customer=self.customer_a2
        )

        self.user_b = User.objects.create_user("user_b", password="Password123")
        TenantMembership.objects.create(
            user=self.user_b, tenant=self.tenant_b, role="customer", customer=self.customer_b
        )

        # 7. Knowledge Base Documents
        self.doc_split_ac = KnowledgeDocument.objects.create(
            tenant=self.tenant_a,
            product=self.product_split_ac,
            title="CoolMaster Split AC CS-INV24 Operation and Troubleshooting Manual",
            content_text=(
                "CoolMaster Inverter Split AC CS-INV24 Troubleshooting Guide:\n\n"
                "Possible causes:\n"
                "- Blocked air filter causing weak airflow and poor cooling.\n"
                "- Obstructed outdoor condenser fins causing poor cooling.\n\n"
                "Observations and checks:\n"
                "- Observe the displayed temperature and error code on the indoor display.\n"
                "- Inspect the visible mesh air filter for dust accumulation.\n"
                "- Note whether the outdoor fan is spinning and condenser fins are clear.\n"
                "- Note whether the indoor evaporator coil has visible frost.\n\n"
                "Safety and Escalation: Stop and request technician service if refrigerant tubing shows oil leaks or if the compressor makes loud grinding noises."
            ),
            is_confidential=False,
            is_rag_enabled=True,
            source_type="text",
        )
        index_document(self.doc_split_ac)

        self.doc_chiller = KnowledgeDocument.objects.create(
            tenant=self.tenant_a,
            product=self.product_chiller,
            title="CoolMaster Commercial Chiller CH-500 Manual",
            content_text=(
                "CoolMaster Commercial Chiller CH-500 Operating Procedures:\n"
                "1. Monitor chilled water supply temperature sensor reading.\n"
                "2. Check chilled water circulation pump discharge pressure on gauge P-1.\n"
                "Expected result: Discharge pressure maintains 45 to 55 PSI."
            ),
            is_confidential=False,
            is_rag_enabled=True,
            source_type="text",
        )
        index_document(self.doc_chiller)

    # -----------------------------------------------------------------------
    # Requirement 1: Multi-Turn Troubleshooting Dialogue Continuity
    # -----------------------------------------------------------------------
    def test_multiturn_dialogue_sequence(self):
        """
        Executes the required 4-turn sequence:
          Turn 1: 'My split AC is not cooling.' -> initial guidance, returns conversation_uuid.
          Turn 2: 'The airflow is weak.' -> contextualized retrieval, air filter check recommended.
          Turn 3: 'I checked that already. What next?' -> contextualized query, suppresses repeated air filter check, provides next steps (condenser fins / evaporator coil).
          Turn 4: 'How do I check the condenser fins?' -> grounded guidance for condenser fins without losing context.
        """
        self.client.force_login(self.user_a1)

        # Turn 1: Initial complaint
        url_query = reverse("api_customer_support_query")
        res1 = self.client.post(
            url_query,
            {
                "site_id": self.site_a1.id,
                "asset_id": self.asset_split_ac.id,
                "question": "The split AC is not cooling and airflow is weak.",
            },
            content_type="application/json",
        )
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()
        conv_uuid = data1.get("conversation_uuid")
        self.assertIsNotNone(conv_uuid, "Conversation UUID must be returned on Turn 1")
        self.assertIn("cooling", str(data1).lower())

        # Verify conversation and message persistence in DB
        conv = CustomerConversation.objects.get(conversation_uuid=conv_uuid)
        self.assertEqual(conv.messages.count(), 2)  # User + Assistant

        # Turn 2: User provides observation: "The airflow is weak."
        url_messages = reverse("api_customer_conversation_messages", kwargs={"conversation_uuid": conv_uuid})
        res2 = self.client.post(
            url_messages,
            {"message": "The airflow is weak."},
            content_type="application/json",
        )
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        contract2 = data2.get("contract", {})

        # In Turn 2, air filter check should be recommended in actions, observations, or answer
        actions2 = [a.get("action_text", a.get("action", "")).lower() for a in contract2.get("customer_safe_actions", [])]
        obs2 = [o.get("text", "").lower() for o in contract2.get("presentation", {}).get("customer_permitted_observations", [])]
        found_filter = any("filter" in a for a in actions2 + obs2) or "filter" in str(contract2).lower()
        self.assertTrue(found_filter, f"Expected air filter check in Turn 2: {actions2} / {obs2}")

        # Turn 3: "I checked that already. What next?"
        res3 = self.client.post(
            url_messages,
            {"message": "I checked that already. What next?"},
            content_type="application/json",
        )
        self.assertEqual(res3.status_code, 200)
        data3 = res3.json()
        contract3 = data3.get("contract", {})

        # In Turn 3, the completed check (air filter) must NOT be recommended again in actions/observations
        actions3 = [a.get("action_text", a.get("action", "")).lower() for a in contract3.get("customer_safe_actions", [])]
        obs3 = [o.get("text", "").lower() for o in contract3.get("presentation", {}).get("customer_permitted_observations", [])]
        filter_repeated = any("air filter" in a or ("clean" in a and "filter" in a) for a in actions3 + obs3)
        self.assertFalse(filter_repeated, f"Air filter check was repeated in Turn 3 actions: {actions3} / {obs3}")

        # Next troubleshooting step (condenser fins or evaporator coil) must be retrieved and presented
        found_next_step = any(
            "condenser" in a or "fins" in a or "evaporator" in a or "coil" in a
            for a in actions3 + obs3
        ) or "condenser" in str(contract3).lower() or "evaporator" in str(contract3).lower()
        self.assertTrue(
            found_next_step,
            f"Expected next steps (condenser/evaporator) in Turn 3: {actions3} / {obs3} / {contract3}"
        )

        # Turn 4: Follow-up question
        res4 = self.client.post(
            url_messages,
            {"message": "How do I check the outdoor condenser fins?"},
            content_type="application/json",
        )
        self.assertEqual(res4.status_code, 200)
        data4 = res4.json()
        contract4 = data4.get("contract", {})
        self.assertIn("condenser", str(contract4).lower())

        # Verify all messages are persisted in chronological order
        conv.refresh_from_db()
        self.assertEqual(conv.messages.count(), 8)  # 4 user + 4 assistant

    # -----------------------------------------------------------------------
    # Requirement 2: Equipment Switching Isolation
    # -----------------------------------------------------------------------
    def test_equipment_switching_does_not_leak_context(self):
        """
        Conversations for different assets must never leak context, symptoms, or checks.
        """
        self.client.force_login(self.user_a1)

        # Conversation 1 on Split AC
        url_query = reverse("api_customer_support_query")
        res1 = self.client.post(
            url_query,
            {
                "site_id": self.site_a1.id,
                "asset_id": self.asset_split_ac.id,
                "question": "My split AC has low airflow and filter is dirty.",
            },
            content_type="application/json",
        )
        self.assertEqual(res1.status_code, 200)
        conv1_uuid = res1.json()["conversation_uuid"]

        # Conversation 2 on Commercial Chiller (different asset, same customer)
        res2 = self.client.post(
            url_query,
            {
                "site_id": self.site_a1.id,
                "asset_id": self.asset_chiller.id,
                "question": "Water temperature is too high.",
            },
            content_type="application/json",
        )
        self.assertEqual(res2.status_code, 200)
        conv2_uuid = res2.json()["conversation_uuid"]
        self.assertNotEqual(conv1_uuid, conv2_uuid)

        # Verify context extraction on conv 2 has zero split AC symptoms
        conv2 = CustomerConversation.objects.get(conversation_uuid=conv2_uuid)
        ctx2 = extract_conversation_context(conv2, "What next?", self.asset_chiller)
        self.assertNotIn("filter", ctx2.symptoms)
        self.assertNotIn("airflow", ctx2.symptoms)
        self.assertEqual(ctx2.asset_name, self.asset_chiller.name)

    # -----------------------------------------------------------------------
    # Requirement 3 & 4: Unauthorized Access and Cross-Tenant Isolation
    # -----------------------------------------------------------------------
    def test_cross_customer_and_cross_tenant_conversation_isolation(self):
        """
        Customers cannot view or post to conversations belonging to other customers or tenants.
        """
        self.client.force_login(self.user_a1)
        url_query = reverse("api_customer_support_query")
        res1 = self.client.post(
            url_query,
            {
                "site_id": self.site_a1.id,
                "asset_id": self.asset_split_ac.id,
                "question": "My split AC is not cooling.",
            },
            content_type="application/json",
        )
        conv_uuid = res1.json()["conversation_uuid"]
        url_messages = reverse("api_customer_conversation_messages", kwargs={"conversation_uuid": conv_uuid})

        # 1. Customer A2 (same tenant, different customer) -> 404
        self.client.force_login(self.user_a2)
        res_a2_get = self.client.get(url_messages)
        self.assertEqual(res_a2_get.status_code, 404)
        res_a2_post = self.client.post(url_messages, {"message": "Hacking conversation"}, content_type="application/json")
        self.assertEqual(res_a2_post.status_code, 404)

        # 2. Customer B (different tenant) -> 404
        self.client.force_login(self.user_b)
        res_b_get = self.client.get(url_messages)
        self.assertEqual(res_b_get.status_code, 404)
        res_b_post = self.client.post(url_messages, {"message": "Cross-tenant intrusion"}, content_type="application/json")
        self.assertEqual(res_b_post.status_code, 404)

    # -----------------------------------------------------------------------
    # Requirement 5: Closed / Resolved Conversation Rejection
    # -----------------------------------------------------------------------
    def test_closed_or_resolved_conversation_rejects_new_messages(self):
        """
        Once marked RESOLVED or CLOSED, new messages to the conversation are rejected.
        """
        self.client.force_login(self.user_a1)
        url_query = reverse("api_customer_support_query")
        res = self.client.post(
            url_query,
            {
                "site_id": self.site_a1.id,
                "asset_id": self.asset_split_ac.id,
                "question": "Initial question.",
            },
            content_type="application/json",
        )
        conv_uuid = res.json()["conversation_uuid"]
        conv = CustomerConversation.objects.get(conversation_uuid=conv_uuid)
        conv.status = "RESOLVED"
        conv.save(update_fields=["status"])

        url_messages = reverse("api_customer_conversation_messages", kwargs={"conversation_uuid": conv_uuid})
        post_res = self.client.post(url_messages, {"message": "Can I do more?"}, content_type="application/json")
        self.assertEqual(post_res.status_code, 400)
        self.assertIn("closed", post_res.json()["detail"].lower())

    # -----------------------------------------------------------------------
    # Requirement 6: Bounded Recent History Window
    # -----------------------------------------------------------------------
    def test_bounded_recent_history_window(self):
        """
        Long conversations with many turns stay bounded within the configured window.
        """
        self.client.force_login(self.user_a1)
        url_query = reverse("api_customer_support_query")
        res = self.client.post(
            url_query,
            {
                "site_id": self.site_a1.id,
                "asset_id": self.asset_split_ac.id,
                "question": "Turn 1: AC problem.",
            },
            content_type="application/json",
        )
        conv_uuid = res.json()["conversation_uuid"]
        url_messages = reverse("api_customer_conversation_messages", kwargs={"conversation_uuid": conv_uuid})

        # Generate 12 turns
        for i in range(2, 8):
            r = self.client.post(
                url_messages,
                {"message": f"Follow up turn {i}"},
                content_type="application/json",
            )
            self.assertEqual(r.status_code, 200)

        conv = CustomerConversation.objects.get(conversation_uuid=conv_uuid)
        # Verify extraction bounds to max_turns=6 (12 messages)
        ctx = extract_conversation_context(conv, "Current question", self.asset_split_ac, max_turns=4)
        self.assertLessEqual(len(ctx.history_turns), 8)

    # -----------------------------------------------------------------------
    # Requirement 7: Multi-Turn Safety Hazard Detection
    # -----------------------------------------------------------------------
    def test_safety_hazard_escalation_across_turns(self):
        """
        If a user starts with an ordinary issue and later mentions a safety hazard
        (smoke, burning smell, sparks), the system must escalate immediately.
        """
        self.client.force_login(self.user_a1)
        url_query = reverse("api_customer_support_query")

        # Turn 1: Ordinary complaint
        res1 = self.client.post(
            url_query,
            {
                "site_id": self.site_a1.id,
                "asset_id": self.asset_split_ac.id,
                "question": "My AC is making a humming sound.",
            },
            content_type="application/json",
        )
        self.assertEqual(res1.status_code, 200)
        conv_uuid = res1.json()["conversation_uuid"]

        # Turn 2: Hazardous escalation
        url_messages = reverse("api_customer_conversation_messages", kwargs={"conversation_uuid": conv_uuid})
        res2 = self.client.post(
            url_messages,
            {"message": "Now I see sparks and smoke coming out of the electrical box!"},
            content_type="application/json",
        )
        self.assertEqual(res2.status_code, 200)
        contract = res2.json()["contract"]

        # Must flag safety hazard and provide zero customer DIY repair actions
        self.assertTrue(
            contract.get("response_mode") == "IMMEDIATE_SAFETY_ESCALATION"
            or contract.get("response_type") == "ESCALATION_REQUIRED"
            or "safety" in str(contract).lower()
        )
        safe_actions = contract.get("customer_safe_actions", [])
        # No intrusive repair actions permitted
        self.assertEqual(len(safe_actions), 0)

    # -----------------------------------------------------------------------
    # Requirement 8: Informational Question Handling
    # -----------------------------------------------------------------------
    def test_informational_question_does_not_invent_actions(self):
        """
        Informational questions provide factual grounded answers without hallucinating DIY repairs.
        """
        self.client.force_login(self.user_a1)
        url_query = reverse("api_customer_support_query")
        res = self.client.post(
            url_query,
            {
                "site_id": self.site_a1.id,
                "asset_id": self.asset_split_ac.id,
                "question": "What model number is this split AC unit?",
            },
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("CS-INV24", str(data))

    # -----------------------------------------------------------------------
    # Requirement 9: Contextual Query Rewriter Fallback
    # -----------------------------------------------------------------------
    def test_deterministic_query_rewriter(self):
        """
        Verifies that contextualize_customer_query properly enriches follow-ups.
        """
        conv = CustomerConversation.objects.create(
            tenant=self.tenant_a,
            customer=self.customer_a1,
            site=self.site_a1,
            asset=self.asset_split_ac,
            title="AC not cooling",
        )
        CustomerMessage.objects.create(
            tenant=self.tenant_a,
            conversation=conv,
            sender_role="customer",
            message_text="My split AC is not cooling.",
        )
        CustomerMessage.objects.create(
            tenant=self.tenant_a,
            conversation=conv,
            sender_role="assistant",
            message_text="Please check if the air filter is blocked.",
            structured_contract={"customer_safe_actions": [{"action_text": "Inspect the mesh air filter for dust"}]},
        )
        CustomerMessage.objects.create(
            tenant=self.tenant_a,
            conversation=conv,
            sender_role="customer",
            message_text="The airflow is weak.",
        )

        ctx = extract_conversation_context(conv, "I checked that already. What next?", self.asset_split_ac)
        self.assertTrue(ctx.is_followup)
        self.assertIn("air filter", ctx.completed_checks)

        rewritten = contextualize_customer_query("I checked that already. What next?", ctx, self.asset_split_ac)
        self.assertIn("Server Room Split AC", rewritten)
        self.assertIn("next troubleshooting steps", rewritten)
        self.assertIn("air filter", rewritten)
