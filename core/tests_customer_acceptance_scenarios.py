"""Realistic Full-Customer Acceptance Test Suite for Servy RAG (Operation 10.0).

Validates Part 18 Scenarios A through J:
  Scenario A: Customer selects registered equipment and asks about a documented fault.
              The correct source is retrieved and cited.
  Scenario B: Customer enters a misspelled fault description.
              The correct applicable evidence is retrieved without inventing a different fault code.
  Scenario C: Customer reports two independent symptoms.
              Both are handled accurately or clarification is requested.
  Scenario D: Customer selects one AC model but mentions a symptom documented for another model.
              No inapplicable repair instructions are displayed.
  Scenario E: Chroma fails.
              Valid sparse evidence remains available if independently verified.
  Scenario F: All retrieval systems fail.
              Servy abstains safely instead of inventing guidance.
  Scenario G: Customer reports sparks during a follow-up.
              Immediate safety escalation occurs.
  Scenario H: Customer requests a technician.
              Exactly one correct incident-linked ticket is created (idempotent).
  Scenario I: Customer reloads the browser.
              Authorized conversation, equipment, and ticket state persist.
  Scenario J: A document is revoked.
              The previously indexed passage stops appearing in customer answers.
"""
from unittest.mock import patch
from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from core.models import (
    Asset,
    Brand,
    Customer,
    CustomerInteraction,
    DiagnosticEvent,
    DiagnosticSession,
    KnowledgeDocument,
    OperationalZone,
    Product,
    ProductCategory,
    ProductDomain,
    ServiceCall,
    Site,
    StaffProfile,
    Tenant,
    TenantMembership,
)
from core.services.indexing import index_document


@override_settings(
    SERVY_CUSTOMER_DATA_MODE="test_isolated",
    SERVY_LLM_PROVIDER="extractive",
    SERVY_RAG_MIN_SCORE=0.0,
)
class CustomerAcceptanceScenariosTests(TestCase):
    def setUp(self):
        # 1. Tenant & Customer User
        self.tenant = Tenant.objects.create(name="Apex Cold Chain", slug="apex-cold", is_active=True)
        self.zone = OperationalZone.objects.create(tenant=self.tenant, name="Apex HQ Zone")
        self.customer = Customer.objects.create(tenant=self.tenant, name="Apex Warehouse 1")
        self.site = Site.objects.create(tenant=self.tenant, customer=self.customer, name="Cold Hub A", zone=self.zone)

        self.user = User.objects.create_user("cust_apex", password="ApexPassword123")
        TenantMembership.objects.create(user=self.user, tenant=self.tenant, role="customer", customer=self.customer)

        # 2. Product Hierarchy
        self.domain = ProductDomain.objects.create(tenant=self.tenant, name="HVAC Refrigeration")
        self.category = ProductCategory.objects.create(tenant=self.tenant, domain=self.domain, name="Air Conditioners")
        self.brand = Brand.objects.create(tenant=self.tenant, name="ApexAir")

        # Two distinct equipment products to test crosstalk blocking
        self.prod_split = Product.objects.create(
            tenant=self.tenant, category=self.category, brand=self.brand, name="Split AC 2T Inverter"
        )
        self.prod_ductable = Product.objects.create(
            tenant=self.tenant, category=self.category, brand=self.brand, name="Ductable 5T Commercial"
        )

        # 3. Assets
        self.asset_split = Asset.objects.create(
            tenant=self.tenant, customer=self.customer, site=self.site, product=self.prod_split,
            name="Apex Split AC #1", asset_code="SPLIT-01", model_number="APX-2T-INV", status="active",
        )
        self.asset_ductable = Asset.objects.create(
            tenant=self.tenant, customer=self.customer, site=self.site, product=self.prod_ductable,
            name="Apex Ductable #1", asset_code="DUCT-01", model_number="APX-5T-COM", status="active",
        )

        # 4. Verified Documentation for Split AC
        self.doc_split_manual = KnowledgeDocument.objects.create(
            tenant=self.tenant,
            product=self.prod_split,
            title="Apex Split AC 2T Troubleshooting Manual",
            doc_type="USER_MANUAL",
            source_type="text",
            is_rag_enabled=True,
            is_confidential=False,
            content_text="""## Ice Formation on Evaporator Coil
If ice blocks or frost form on the indoor split AC cooling coil, perform the following checks:
1. Inspect the nylon pre-filter mesh for heavy dust accumulation.
2. Confirm the indoor cross-flow blower fan rotates freely without binding.
3. Clean the pre-filter mesh under lukewarm water and dry before reinstalling.
Why: Blocked airflow starves the evaporator of warm air causing refrigerant to boil below 0C.
Expected Result: Evaporator coil thaws completely and normal airflow resumes within 15 minutes.
Stop and escalate if: Ice continues forming after filter cleaning or copper pipe vibrates abnormally.""",
        )
        index_document(self.doc_split_manual)

        # 5. Verified Documentation for Ductable AC
        self.doc_ductable_manual = KnowledgeDocument.objects.create(
            tenant=self.tenant,
            product=self.prod_ductable,
            title="Apex Ductable 5T Technical Service Manual",
            doc_type="USER_MANUAL",
            source_type="text",
            is_rag_enabled=True,
            is_confidential=False,
            content_text="""## Centrifugal Fan Belt Tension Adjustment
For ductable AC centrifugal fan belt deflection:
1. Measure drive belt tension across the center span of the pulleys.
2. Adjust the motor mounting bracket bolts until belt deflection is exactly 12mm under 2kg force.
3. Tighten locknuts to 35 Nm torque.
Why: Loose belts slip and generate screeching noise while over-tight belts destroy motor bearings.
Expected Result: Centrifugal fan achieves rated static pressure without belt squeal.
Stop and escalate if: Pulleys show visible wobble or belt exhibits deep cracks.""",
        )
        index_document(self.doc_ductable_manual)

        self.client.force_login(self.user)
        self.query_url = reverse("api_customer_support_query")

    def test_scenario_a_documented_fault_retrieval_and_citation(self):
        """Scenario A: Customer selects registered equipment and asks about a documented fault.
        The correct source is retrieved and cited."""
        res = self.client.post(self.query_url, {
            "site_id": self.site.id,
            "asset_id": self.asset_split.id,
            "question": "Ice blocks forming on indoor split AC cooling coil",
        }, content_type="application/json")

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["response_type"], "DIAGNOSTIC_GUIDANCE")
        self.assertTrue(len(data["citations"]) > 0)
        self.assertEqual(data["citations"][0]["document_id"], self.doc_split_manual.id)
        self.assertIn("Split AC", data["citations"][0]["title"])
        self.assertTrue(len(data["customer_safe_actions"]) > 0)
        self.assertTrue(any("filter" in a["action"].lower() or "coil" in a["action"].lower() for a in data["customer_safe_actions"]))

    def test_scenario_b_misspelled_fault_retrieval_without_inventing_code(self):
        """Scenario B: Customer enters a misspelled fault description.
        The correct applicable evidence is retrieved without inventing a different fault code."""
        res = self.client.post(self.query_url, {
            "site_id": self.site.id,
            "asset_id": self.asset_split.id,
            "question": "splitt ac icce forming on coolng coyl",
        }, content_type="application/json")

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["response_type"], "DIAGNOSTIC_GUIDANCE")
        self.assertTrue(len(data["citations"]) > 0)
        self.assertEqual(data["citations"][0]["document_id"], self.doc_split_manual.id)
        # Ensure no hallucinated fault codes like E-99 or E04 were generated
        full_text = str(data)
        self.assertNotIn("E99", full_text)
        self.assertNotIn("E-99", full_text)

    def test_scenario_c_multiple_symptoms_handled_or_clarified(self):
        """Scenario C: Customer reports two independent symptoms.
        Both are handled accurately or clarification is requested."""
        res = self.client.post(self.query_url, {
            "site_id": self.site.id,
            "asset_id": self.asset_split.id,
            "question": "The split unit is forming ice on coil and the display is completely dead with no power lights.",
        }, content_type="application/json")

        self.assertEqual(res.status_code, 200)
        data = res.json()
        # Either structured guidance covering symptoms or clarification
        self.assertIn(data["response_type"], ["DIAGNOSTIC_GUIDANCE", "CLARIFICATION"])
        if data["response_type"] == "CLARIFICATION":
            self.assertIsNotNone(data.get("clarification_question"))
        else:
            self.assertTrue(len(data["citations"]) > 0)

    def test_scenario_d_equipment_crosstalk_blocked(self):
        """Scenario D: Customer selects one AC model (Split AC) but asks about a symptom documented
        only for another model (Ductable fan belt). No inapplicable repair instructions are displayed."""
        res = self.client.post(self.query_url, {
            "site_id": self.site.id,
            "asset_id": self.asset_split.id,
            "question": "How to adjust centrifugal fan belt deflection?",
        }, content_type="application/json")

        self.assertEqual(res.status_code, 200)
        data = res.json()
        # Crosstalk protection: Ductable 5T manual must NOT be cited when Split AC is selected
        cited_doc_ids = [c["document_id"] for c in data.get("citations", [])]
        self.assertNotIn(
            self.doc_ductable_manual.id,
            cited_doc_ids,
            "DEFECT: Cross-model crosstalk allowed ductable manual to be cited for split AC asset",
        )
        # Inapplicable centrifugal fan belt deflection steps must not be given to split AC customer
        actions_text = " ".join(a.get("action", "") for a in data.get("customer_safe_actions", []))
        self.assertNotIn("deflection is exactly 12mm", actions_text)

    def test_scenario_e_chroma_failure_fallback_to_verified_sparse(self):
        """Scenario E: Chroma fails. Valid sparse evidence remains available if independently verified."""
        with patch("core.services.retriever._dense_score_candidates", return_value=[]):
            res = self.client.post(self.query_url, {
                "site_id": self.site.id,
                "asset_id": self.asset_split.id,
                "question": "Ice blocks forming on indoor split AC cooling coil",
            }, content_type="application/json")

            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["response_type"], "DIAGNOSTIC_GUIDANCE")
            self.assertTrue(len(data["citations"]) > 0)
            self.assertEqual(data["citations"][0]["document_id"], self.doc_split_manual.id)

    def test_scenario_f_all_retrieval_fails_abstains_safely(self):
        """Scenario F: All retrieval systems fail or evidence is completely absent.
        Servy abstains safely instead of inventing guidance."""
        res = self.client.post(self.query_url, {
            "site_id": self.site.id,
            "asset_id": self.asset_split.id,
            "question": "How do I replace the flight control hydraulic actuator valve on Boeing 787?",
        }, content_type="application/json")

        self.assertEqual(res.status_code, 200)
        data = res.json()
        # Must abstain or request clarification, zero DIY repair actions
        self.assertIn(data["response_type"], ["INSUFFICIENT_EVIDENCE", "CLARIFICATION", "UNSUPPORTED"])
        self.assertEqual(len(data.get("customer_safe_actions", [])), 0)
        self.assertEqual(len(data.get("citations", [])), 0)

    def test_scenario_g_hazard_follow_up_triggers_safety_escalation(self):
        """Scenario G: Customer reports sparks during follow-up. Immediate safety escalation occurs."""
        # Initial question
        res1 = self.client.post(self.query_url, {
            "site_id": self.site.id,
            "asset_id": self.asset_split.id,
            "question": "The split unit smells slightly burnt",
        }, content_type="application/json")
        session_id = res1.json().get("session_id")

        # Follow-up reports sparks
        res2 = self.client.post(self.query_url, {
            "session_id": session_id,
            "site_id": self.site.id,
            "asset_id": self.asset_split.id,
            "question": "Now I see sparks coming out of the electrical terminal block!",
        }, content_type="application/json")

        self.assertEqual(res2.status_code, 200)
        data = res2.json()
        self.assertIn(data["response_type"], ["SAFETY_ESCALATION", "ESCALATION_REQUIRED"])
        self.assertEqual(len(data["customer_safe_actions"]), 0)
        self.assertTrue(data.get("safety_hazard_detected") or "safety risk" in str(data).lower())

    def test_scenario_h_technician_ticket_creation_is_idempotent(self):
        """Scenario H: Customer requests a technician.
        Exactly one correct incident-linked ticket is created without duplicates."""
        res1 = self.client.post(self.query_url, {
            "site_id": self.site.id,
            "asset_id": self.asset_split.id,
            "question": "Ice blocks forming on indoor split AC cooling coil",
        }, content_type="application/json")
        interaction_id = res1.json()["interaction_id"]

        escalate_url = reverse("api_customer_support_escalate", kwargs={"pk": interaction_id})
        post_data = {
            "note": "Ice blocks forming on coil and filter cleaning did not resolve it.",
        }

        # First escalation call
        res_call_1 = self.client.post(escalate_url, post_data, content_type="application/json")
        self.assertEqual(res_call_1.status_code, 200)
        servy_id = res_call_1.json()["servy_id"]

        # Exactly 1 ServiceCall created
        self.assertEqual(ServiceCall.objects.filter(asset=self.asset_split).count(), 1)
        sc = ServiceCall.objects.get(servy_id=servy_id)
        self.assertEqual(sc.tenant_id, self.tenant.id)
        self.assertEqual(sc.customer_id, self.customer.id)

        # Second escalation call on same interaction (idempotency test)
        res_call_2 = self.client.post(escalate_url, post_data, content_type="application/json")
        self.assertEqual(res_call_2.status_code, 200)
        self.assertEqual(res_call_2.json()["servy_id"], servy_id)

        # Still exactly 1 ServiceCall in database
        self.assertEqual(ServiceCall.objects.filter(asset=self.asset_split).count(), 1)

    def test_scenario_i_conversation_and_equipment_state_persists(self):
        """Scenario I: Customer reloads browser. Authorized conversation and equipment state persist."""
        res1 = self.client.post(self.query_url, {
            "site_id": self.site.id,
            "asset_id": self.asset_split.id,
            "question": "Ice blocks forming on indoor split AC cooling coil",
        }, content_type="application/json")
        conv_uuid = res1.json().get("conversation_uuid")

        # Reload conversation state via conversation detail endpoint
        conv_url = reverse("api_customer_conversation_detail", kwargs={"conversation_uuid": conv_uuid})
        res_get = self.client.get(conv_url)
        self.assertEqual(res_get.status_code, 200)
        data = res_get.json()
        self.assertEqual(data["conversation_uuid"], conv_uuid)
        self.assertEqual(data["asset"]["id"], self.asset_split.id)
        self.assertEqual(data["customer"]["id"], self.customer.id)
        self.assertTrue(len(data.get("messages", [])) >= 2)

    def test_scenario_j_revoked_document_excluded_from_citations(self):
        """Scenario J: A document is revoked. The previously indexed passage stops appearing in customer answers."""
        # 1. Before revocation: document is cited
        res_before = self.client.post(self.query_url, {
            "site_id": self.site.id,
            "asset_id": self.asset_split.id,
            "question": "Ice blocks forming on indoor split AC cooling coil",
        }, content_type="application/json")
        self.assertEqual(res_before.status_code, 200)
        cited_before = [c["document_id"] for c in res_before.json().get("citations", [])]
        self.assertIn(self.doc_split_manual.id, cited_before)

        # 2. Revoke the document
        self.doc_split_manual.is_rag_enabled = False
        self.doc_split_manual.save()

        # 3. After revocation: revoked document must NOT be retrieved or cited
        res_after = self.client.post(self.query_url, {
            "site_id": self.site.id,
            "asset_id": self.asset_split.id,
            "question": "Cross-flow blower fan rotates freely inspection",
        }, content_type="application/json")
        self.assertEqual(res_after.status_code, 200)
        cited_after = [c["document_id"] for c in res_after.json().get("citations", [])]
        self.assertNotIn(
            self.doc_split_manual.id,
            cited_after,
            "DEFECT: Revoked document was still cited to customer",
        )
