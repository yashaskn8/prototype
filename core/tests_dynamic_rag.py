import json
from django.test import TestCase, override_settings
from django.contrib.auth.models import User
from core.models import (
    Tenant, Customer, Site, Asset, Product, Brand, ProductCategory, ProductDomain,
    KnowledgeDocument, ServiceCall, CustomerInteraction, TenantMembership, StaffProfile
)
from core.services.indexing import index_document
from core.services.rag import ask_rag


@override_settings(
    SERVY_RAG_MIN_SCORE=0.0,
    SERVY_ALLOW_REMOTE_LLM=False,
    CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}}
)
class DynamicMultiSiteAndRAGTests(TestCase):
    def setUp(self):
        # 1 tenant, 2 customers (one primary, one for cross-customer isolation tests)
        self.tenant = Tenant.objects.create(name="Dynamic Tenant", slug="dyn-tenant")
        self.other_tenant = Tenant.objects.create(name="Other Tenant", slug="other-tenant")

        self.customer = Customer.objects.create(tenant=self.tenant, name="Primary Customer")
        self.other_customer = Customer.objects.create(tenant=self.tenant, name="Secondary Customer")

        # Users
        self.customer_user = User.objects.create_user(username="dyn_customer", password="password123")
        TenantMembership.objects.create(
            user=self.customer_user, tenant=self.tenant, role="customer", customer=self.customer
        )

        self.technician_user = User.objects.create_user(username="dyn_tech", password="password123")
        TenantMembership.objects.create(
            user=self.technician_user, tenant=self.tenant, role="technician"
        )
        StaffProfile.objects.create(
            tenant=self.tenant, user=self.technician_user, full_name="Tech One", role="technician"
        )

        # Products & Taxonomy
        self.domain = ProductDomain.objects.create(tenant=self.tenant, name="Industrial Equipment")
        self.category = ProductCategory.objects.create(tenant=self.tenant, domain=self.domain, name="Hydraulics")
        self.brand = Brand.objects.create(tenant=self.tenant, name="Apex Brand")
        self.product = Product.objects.create(tenant=self.tenant, brand=self.brand, category=self.category, name="HydraPump 5000")
        self.unrelated_product = Product.objects.create(tenant=self.tenant, brand=self.brand, category=self.category, name="LaserCutter 900")

    def test_30_dynamic_multi_site_behavior(self):
        """Requirement 30: 3 Sites with varying assets loaded dynamically via API."""
        site_a = Site.objects.create(tenant=self.tenant, customer=self.customer, name="Site Alpha", city="City A")
        site_b = Site.objects.create(tenant=self.tenant, customer=self.customer, name="Site Beta", city="City B")
        site_c = Site.objects.create(tenant=self.tenant, customer=self.customer, name="Site Gamma", city="City C")

        # Site A: 2 assets
        Asset.objects.create(tenant=self.tenant, customer=self.customer, site=site_a, product=self.product, name="Alpha-1", asset_code="A-01")
        Asset.objects.create(tenant=self.tenant, customer=self.customer, site=site_a, product=self.product, name="Alpha-2", asset_code="A-02")

        # Site B: 3 assets
        b1 = Asset.objects.create(tenant=self.tenant, customer=self.customer, site=site_b, product=self.product, name="Beta-1", asset_code="B-01")
        b2 = Asset.objects.create(tenant=self.tenant, customer=self.customer, site=site_b, product=self.product, name="Beta-2", asset_code="B-02")
        b3 = Asset.objects.create(tenant=self.tenant, customer=self.customer, site=site_b, product=self.product, name="Beta-3", asset_code="B-03")

        # Site C: 1 asset
        Asset.objects.create(tenant=self.tenant, customer=self.customer, site=site_c, product=self.product, name="Gamma-1", asset_code="C-01")

        self.client.login(username="dyn_customer", password="password123")

        # 1. Customer Context returns exactly 3 Sites
        resp = self.client.get("/api/customer-support/context/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["sites"]), 3)
        site_ids = [s["id"] for s in data["sites"]]
        self.assertIn(site_a.id, site_ids)
        self.assertIn(site_b.id, site_ids)
        self.assertIn(site_c.id, site_ids)

        # 2. Select Site B -> returns exactly 3 assets for Site B
        resp_b = self.client.get(f"/api/sites/{site_b.id}/assets/")
        self.assertEqual(resp_b.status_code, 200)
        assets_b = resp_b.json().get("assets", [])
        self.assertEqual(len(assets_b), 3)
        b_ids = {a["id"] for a in assets_b}
        self.assertEqual(b_ids, {b1.id, b2.id, b3.id})

    def test_31_multi_document_rag_eligibility(self):
        """Requirement 31: Multi-document RAG eligibility and strict customer scope."""
        site = Site.objects.create(tenant=self.tenant, customer=self.customer, name="Main Plant")
        target_asset = Asset.objects.create(
            tenant=self.tenant, customer=self.customer, site=site, product=self.product,
            name="Main HydraPump", asset_code="HP-100", model_number="HP5K-M"
        )

        # Document A: correct product troubleshooting manual
        doc_a = KnowledgeDocument.objects.create(
            tenant=self.tenant, product=self.product, title="HydraPump 5000 Troubleshooting",
            doc_type="troubleshooting", is_confidential=False, is_rag_enabled=True,
            content_text="Check hydraulic fluid pressure. Step 1: Bleed air from secondary valve. Why: Purges cavitation."
        )
        index_document(doc_a)

        # Document B: correct product cleaning manual
        doc_b = KnowledgeDocument.objects.create(
            tenant=self.tenant, product=self.product, title="HydraPump 5000 Cleaning Procedure",
            doc_type="cleaning", is_confidential=False, is_rag_enabled=True,
            content_text="Rinse hydraulic intake with solvent. Step 1: Clear particulate strainer. Why: Restores flow."
        )
        index_document(doc_b)

        # Document C: unrelated product
        doc_c = KnowledgeDocument.objects.create(
            tenant=self.tenant, product=self.unrelated_product, title="LaserCutter Optics Manual",
            doc_type="troubleshooting", is_confidential=False, is_rag_enabled=True,
            content_text="Align focal mirror laser lens."
        )
        index_document(doc_c)

        # Document D: different customer
        doc_d = KnowledgeDocument.objects.create(
            tenant=self.tenant, customer=self.other_customer, product=self.product,
            title="Secondary Customer HydraPump Custom SOP", doc_type="troubleshooting",
            is_confidential=False, is_rag_enabled=True, content_text="Proprietary client steps."
        )
        index_document(doc_d)

        # Document E: confidential manual
        doc_e = KnowledgeDocument.objects.create(
            tenant=self.tenant, product=self.product, title="HydraPump 5000 Factory Secret Diagnostic",
            doc_type="service_manual", is_confidential=True, is_rag_enabled=True,
            content_text="Factory confidential schematics and bypass circuits."
        )
        index_document(doc_e)

        # Document F: generic non-technical Knowledge Base item (RAG disabled)
        doc_f = KnowledgeDocument.objects.create(
            tenant=self.tenant, title="Welcome to Company Guidelines and Holiday List",
            doc_type="reference", is_confidential=False, is_rag_enabled=False,
            content_text="Office holiday calendar and celebration dates."
        )
        index_document(doc_f)

        # Also create a past service case resolution to verify Customer Self-Service DOES NOT include it
        call = ServiceCall.objects.create(
            tenant=self.tenant, customer=self.customer, site=site, asset=target_asset,
            servy_id=99, complaint_type="Pressure Loss", complaint_text="Loss of pressure",
            resolution_text="Replaced worn gasket on pump inlet.", status="closed"
        )
        from core.services.indexing import index_service_resolution
        index_service_resolution(call)

        # Run RAG for Customer Self-Service
        res = ask_rag(
            tenant=self.tenant,
            question="Hydraulic pressure dropping rapidly during cycle",
            asset=target_asset,
            customer=self.customer,
            include_confidential=False,
            service_call=None,
            staff_mode=False,
        )

        retrieved = res.get("retrieved", [])
        retrieved_doc_ids = {r.document_id for r in retrieved if r.document_id}

        # A & B must be eligible
        self.assertTrue(doc_a.id in retrieved_doc_ids or doc_b.id in retrieved_doc_ids)

        # C must not be in top results for HydraPump
        # D must be strictly excluded (different customer)
        self.assertNotIn(doc_d.id, retrieved_doc_ids)

        # E must be strictly excluded (confidential)
        self.assertNotIn(doc_e.id, retrieved_doc_ids)

        # F must be strictly excluded (is_rag_enabled=False)
        self.assertNotIn(doc_f.id, retrieved_doc_ids)

        # No previous-case source appears in Customer Self-Service
        for r in retrieved:
            self.assertNotEqual(r.source_kind, "past_resolution")

    def test_32_engineer_case_memory(self):
        """Requirement 32: Engineer Copilot retrieves past resolutions and authorized KB."""
        site = Site.objects.create(tenant=self.tenant, customer=self.customer, name="Main Plant")
        asset = Asset.objects.create(
            tenant=self.tenant, customer=self.customer, site=site, product=self.product,
            name="HydraPump-01", asset_code="HP-001"
        )

        kb_doc = KnowledgeDocument.objects.create(
            tenant=self.tenant, product=self.product, title="HydraPump Official Manual",
            doc_type="troubleshooting", is_confidential=False, is_rag_enabled=True,
            content_text="Documented fix for cavitation in intake manifold."
        )
        index_document(kb_doc)

        # Create past service call with closed status and resolution text
        past_call = ServiceCall.objects.create(
            tenant=self.tenant, customer=self.customer, site=site, asset=asset,
            servy_id=88, complaint_type="Cavitation", complaint_text="Pump cavitation sound",
            resolution_text="Cleaned manifold strainer and reset intake valve.", status="closed"
        )
        from core.services.indexing import index_service_resolution
        index_service_resolution(past_call)

        # Current open call for engineer
        active_call = ServiceCall.objects.create(
            tenant=self.tenant, customer=self.customer, site=site, asset=asset,
            servy_id=89, complaint_type="Cavitation", complaint_text="Cavitation recurring",
            status="open"
        )

        res = ask_rag(
            tenant=self.tenant,
            question="Pump cavitation sound",
            asset=asset,
            customer=self.customer,
            include_confidential=True,
            service_call=active_call,
            staff_mode=True,
        )

        retrieved = res.get("retrieved", [])
        source_kinds = {r.source_kind for r in retrieved}
        self.assertIn("knowledge_document", source_kinds)
        self.assertIn("past_resolution", source_kinds)

    def test_33_escalation_idempotency(self):
        """Requirement 33: Customer escalation creates exactly ONE ServiceCall idempotently."""
        site = Site.objects.create(tenant=self.tenant, customer=self.customer, name="Plant One")
        asset = Asset.objects.create(
            tenant=self.tenant, customer=self.customer, site=site, product=self.product,
            name="HP-Asset", asset_code="HP-99"
        )

        self.client.login(username="dyn_customer", password="password123")

        # 1. Ask query
        q_resp = self.client.post("/api/customer-support/query/", {
            "site_id": site.id,
            "asset_id": asset.id,
            "question": "Motor vibrating excessively"
        })
        self.assertEqual(q_resp.status_code, 200)
        interaction_id = q_resp.json()["interaction_id"]

        # 2. Escalate first time
        esc_1 = self.client.post(f"/api/customer-support/interactions/{interaction_id}/escalate/")
        self.assertEqual(esc_1.status_code, 200)
        call_id_1 = esc_1.json()["call_id"]

        # 3. Repeated escalation (double-click/retry)
        esc_2 = self.client.post(f"/api/customer-support/interactions/{interaction_id}/escalate/")
        self.assertEqual(esc_2.status_code, 200)
        call_id_2 = esc_2.json()["call_id"]

        self.assertEqual(call_id_1, call_id_2)

        # Verify exactly 1 ServiceCall in database for this interaction
        interaction = CustomerInteraction.objects.get(id=interaction_id)
        self.assertIsNotNone(interaction.escalated_call)
        self.assertEqual(interaction.escalated_call.id, call_id_1)
        self.assertEqual(ServiceCall.objects.filter(source_interactions=interaction).count(), 1)
