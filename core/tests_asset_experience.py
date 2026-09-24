import io
from django.test import TestCase
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.utils import timezone
from datetime import timedelta

from core.models import (
    Tenant, Customer, Site, Asset, Product, ProductCategory, ProductDomain,
    Brand, KnowledgeDocument, ServiceCall, TenantMembership, StaffProfile,
    CallUpdate
)


class AssetExperienceTests(TestCase):
    def setUp(self):
        # Create Tenants
        self.t1 = Tenant.objects.create(name="Tenant Alpha", slug="alpha")
        self.t2 = Tenant.objects.create(name="Tenant Beta", slug="beta")

        # Create Users
        self.cust_user = User.objects.create_user("cust_alice", "alice@example.com", "pass123")
        self.cust_user_other = User.objects.create_user("cust_bob", "bob@example.com", "pass123")
        self.staff_user = User.objects.create_user("tech_carol", "carol@example.com", "pass123")
        self.tech_user = User.objects.create_user("tech_dave", "dave@example.com", "pass123")

        # Create Customers & Sites
        self.c1 = Customer.objects.create(tenant=self.t1, name="Dairy Co 1")
        self.c2 = Customer.objects.create(tenant=self.t1, name="Dairy Co 2")
        self.c_other_tenant = Customer.objects.create(tenant=self.t2, name="Beta Farm")

        self.site1 = Site.objects.create(tenant=self.t1, customer=self.c1, name="Plant A")
        self.site2 = Site.objects.create(tenant=self.t1, customer=self.c2, name="Plant B")

        # Memberships
        TenantMembership.objects.create(
            tenant=self.t1, user=self.cust_user, role="customer", customer=self.c1, is_active=True
        )
        TenantMembership.objects.create(
            tenant=self.t1, user=self.cust_user_other, role="customer", customer=self.c2, is_active=True
        )
        TenantMembership.objects.create(
            tenant=self.t1, user=self.staff_user, role="admin", is_active=True
        )
        TenantMembership.objects.create(
            tenant=self.t1, user=self.tech_user, role="technician", is_active=True
        )

        # Staff profile for technician
        self.tech_staff = StaffProfile.objects.create(
            tenant=self.t1, user=self.tech_user, full_name="Dave Technician",
            role="technician", is_active=True
        )

        # Hierarchy: Domain -> Category -> Brand -> Product
        self.domain = ProductDomain.objects.create(tenant=self.t1, name="Dairy Equipment")
        self.category = ProductCategory.objects.create(tenant=self.t1, domain=self.domain, name="Analyzers")
        self.brand = Brand.objects.create(tenant=self.t1, name="LactoTech")
        self.product = Product.objects.create(
            tenant=self.t1, brand=self.brand, category=self.category, name="LactoScan Pro"
        )

        # Assets
        self.asset1 = Asset.objects.create(
            tenant=self.t1, customer=self.c1, site=self.site1, product=self.product,
            name="Milk Analyzer 01", asset_code="MA-001", model_number="LSP-300",
            serial_number="SN-1001", warranty_until=timezone.now().date() + timedelta(days=90)
        )
        self.asset_c2 = Asset.objects.create(
            tenant=self.t1, customer=self.c2, site=self.site2, product=self.product,
            name="Milk Analyzer 02", asset_code="MA-002", model_number="LSP-300"
        )

        # Documents
        # 1. Product level manual (applicable to asset1)
        self.doc_prod = KnowledgeDocument.objects.create(
            tenant=self.t1, product=self.product, title="LactoScan Pro Manual",
            doc_type="service_manual", content_text="Standard maintenance procedures for LactoScan Pro.",
            is_rag_enabled=True, index_status="INDEXED"
        )
        # 2. Asset specific document (applicable only to asset1)
        self.doc_asset = KnowledgeDocument.objects.create(
            tenant=self.t1, customer=self.c1, asset=self.asset1, title="Asset 01 Calibration Cert",
            doc_type="reference", content_text="Calibration performed on MA-001.",
            is_rag_enabled=False, index_status="NOT_INDEXED"
        )
        # 3. Confidential document (should not leak to customer)
        self.doc_secret = KnowledgeDocument.objects.create(
            tenant=self.t1, product=self.product, title="Confidential Engineering Schematics",
            doc_type="reference", content_text="Secret PCB schematic.",
            is_confidential=True, is_rag_enabled=True, index_status="INDEXED"
        )

        # Service Calls
        self.call_open = ServiceCall.objects.create(
            tenant=self.t1, customer=self.c1, site=self.site1, asset=self.asset1,
            servy_id=101, complaint_type="Sensor Error", complaint_text="Sensor reading zero.",
            status="open", technician_notes="Internal diagnostic: sensor wire degraded."
        )
        self.call_closed = ServiceCall.objects.create(
            tenant=self.t1, customer=self.c1, site=self.site1, asset=self.asset1,
            servy_id=102, complaint_type="Power Issue", complaint_text="Device would not turn on.",
            status="closed", resolution_text="Replaced fuse.", technician_notes="Customer tried opening chassis."
        )

        # Call Updates (internal technician notes attached to open call)
        CallUpdate.objects.create(
            tenant=self.t1, service_call=self.call_open, author=self.tech_staff,
            status="assigned", note="Internal: assigned to Dave, suspect sensor wiring harness."
        )

    def test_customer_can_list_only_own_assets(self):
        self.client.force_login(self.cust_user)
        session = self.client.session
        session["active_tenant_id"] = self.t1.id
        session.save()

        res = self.client.get("/api/assets/")
        self.assertEqual(res.status_code, 200)
        asset_ids = [a["id"] for a in res.json()["assets"]]
        self.assertIn(self.asset1.id, asset_ids)
        self.assertNotIn(self.asset_c2.id, asset_ids)

    def test_asset_detail_idor_prevention(self):
        self.client.force_login(self.cust_user)
        session = self.client.session
        session["active_tenant_id"] = self.t1.id
        session.save()

        # Accessing own asset: 200
        res = self.client.get(f"/api/assets/{self.asset1.id}/")
        self.assertEqual(res.status_code, 200)
        data = res.json()["asset"]
        self.assertEqual(data["name"], "Milk Analyzer 01")
        self.assertEqual(data["open_calls_count"], 1)
        self.assertFalse(data["is_warranty_expired"])

        # Accessing another customer's asset: 404
        res_other = self.client.get(f"/api/assets/{self.asset_c2.id}/")
        self.assertEqual(res_other.status_code, 404)

    def test_asset_documents_hierarchy_inclusion_and_confidential_filtering(self):
        self.client.force_login(self.cust_user)
        session = self.client.session
        session["active_tenant_id"] = self.t1.id
        session.save()

        res = self.client.get(f"/api/assets/{self.asset1.id}/documents/")
        self.assertEqual(res.status_code, 200)
        doc_titles = [d["title"] for d in res.json()["documents"]]

        # Product manual is inherited
        self.assertIn("LactoScan Pro Manual", doc_titles)
        # Asset calibration cert is included
        self.assertIn("Asset 01 Calibration Cert", doc_titles)
        # Confidential schematic must be filtered out for customers
        self.assertNotIn("Confidential Engineering Schematics", doc_titles)

    def test_customer_upload_sets_rag_disabled_by_default(self):
        self.client.force_login(self.cust_user)
        session = self.client.session
        session["active_tenant_id"] = self.t1.id
        session.save()

        payload = {
            "title": "Customer Maintenance Report",
            "description": "Monthly check by site manager",
            "content_text": "All clean and running normal.",
            "doc_type": "maintenance",
        }
        res = self.client.post(f"/api/assets/{self.asset1.id}/documents/upload/", payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertFalse(data["is_rag_enabled"])
        self.assertEqual(data["index_status"], "NOT_INDEXED")

        # Verify DB object
        doc = KnowledgeDocument.objects.get(id=data["id"])
        self.assertFalse(doc.is_rag_enabled)
        self.assertEqual(doc.customer_id, self.c1.id)
        self.assertEqual(doc.asset_id, self.asset1.id)

    def test_staff_can_approve_customer_document_for_rag(self):
        # Admin approves doc
        self.client.force_login(self.staff_user)
        session = self.client.session
        session["active_tenant_id"] = self.t1.id
        session.save()

        res = self.client.post(f"/api/knowledge/{self.doc_asset.id}/approve-for-rag/")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["is_rag_enabled"])

        doc = KnowledgeDocument.objects.get(id=self.doc_asset.id)
        self.assertTrue(doc.is_rag_enabled)
        self.assertEqual(doc.index_status, "INDEXED")

    def test_customer_cannot_approve_document_for_rag(self):
        self.client.force_login(self.cust_user)
        session = self.client.session
        session["active_tenant_id"] = self.t1.id
        session.save()

        res = self.client.post(f"/api/knowledge/{self.doc_asset.id}/approve-for-rag/")
        self.assertEqual(res.status_code, 403)

    def test_customer_service_history_strips_technician_notes(self):
        self.client.force_login(self.cust_user)
        session = self.client.session
        session["active_tenant_id"] = self.t1.id
        session.save()

        # Asset calls list
        res = self.client.get(f"/api/assets/{self.asset1.id}/calls/?scope=all")
        self.assertEqual(res.status_code, 200)
        calls = res.json()["calls"]
        self.assertEqual(len(calls), 2)
        for c in calls:
            self.assertEqual(c["technician_notes"], "")
            self.assertIsNone(c["technician_name"])

        # Direct CallDetailView
        res_detail = self.client.get(f"/api/calls/{self.call_open.id}/")
        self.assertEqual(res_detail.status_code, 200)
        call_obj = res_detail.json()["call"]
        self.assertEqual(call_obj["technician_notes"], "")
        self.assertIsNone(call_obj["technician"])
        self.assertEqual(res_detail.json()["part_requests"], [])

    def test_service_history_filter_open_vs_history(self):
        self.client.force_login(self.cust_user)
        session = self.client.session
        session["active_tenant_id"] = self.t1.id
        session.save()

        # Open calls
        res_open = self.client.get(f"/api/assets/{self.asset1.id}/calls/?scope=open")
        self.assertEqual(res_open.status_code, 200)
        open_ids = [c["id"] for c in res_open.json()["calls"]]
        self.assertIn(self.call_open.id, open_ids)
        self.assertNotIn(self.call_closed.id, open_ids)

        # History calls
        res_hist = self.client.get(f"/api/assets/{self.asset1.id}/calls/?scope=history")
        self.assertEqual(res_hist.status_code, 200)
        hist_ids = [c["id"] for c in res_hist.json()["calls"]]
        self.assertIn(self.call_closed.id, hist_ids)
        self.assertNotIn(self.call_open.id, hist_ids)

    def test_technician_cannot_approve_rag(self):
        """Red-team: Only admin/manager/superuser may approve RAG; technician is blocked."""
        self.client.force_login(self.tech_user)
        session = self.client.session
        session["active_tenant_id"] = self.t1.id
        session.save()

        res = self.client.post(f"/api/knowledge/{self.doc_asset.id}/approve-for-rag/")
        self.assertEqual(res.status_code, 403)

        res_remove = self.client.post(f"/api/knowledge/{self.doc_prod.id}/remove-from-rag/")
        self.assertEqual(res_remove.status_code, 403)

    def test_customer_call_detail_sanitizes_updates(self):
        """Red-team: Customer must not see internal notes or staff identity in call updates."""
        self.client.force_login(self.cust_user)
        session = self.client.session
        session["active_tenant_id"] = self.t1.id
        session.save()

        res = self.client.get(f"/api/calls/{self.call_open.id}/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        updates = data["updates"]
        self.assertTrue(len(updates) > 0, "There should be at least one call update")
        for u in updates:
            # Internal notes must be empty for customer
            self.assertEqual(u["note"], "")
            # Staff author name must be hidden
            self.assertIsNone(u["author__full_name"])
