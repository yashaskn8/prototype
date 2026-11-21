"""Regression tests for the red-team authorization review (synthetic data only)."""
import tempfile
from unittest import mock

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from core.models import (
    Asset, Customer, CustomerConversation, KnowledgeChunk, KnowledgeDocument, Product, ServiceCall, Site,
    SourceProvenanceRecord, StaffProfile, Tenant, TenantMembership)


class AuthorizationGapTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.t1 = Tenant.objects.create(name="Auth T1", slug="auth-t1")
        cls.t2 = Tenant.objects.create(name="Auth T2", slug="auth-t2")
        cls.product = Product.objects.create(tenant=cls.t1, name="Shared Split AC")
        cls.cust_a = Customer.objects.create(tenant=cls.t1, name="Customer A", phone="+1 555 0001")
        cls.cust_b = Customer.objects.create(tenant=cls.t1, name="Customer B", phone="+1 555 0002")
        cls.cust_c = Customer.objects.create(tenant=cls.t2, name="Customer C")
        site_a = Site.objects.create(tenant=cls.t1, customer=cls.cust_a, name="Site A")
        site_b = Site.objects.create(tenant=cls.t1, customer=cls.cust_b, name="Site B")
        cls.asset_a = Asset.objects.create(tenant=cls.t1, customer=cls.cust_a, site=site_a, product=cls.product,
                                           name="A unit", asset_code="AUTH-A")
        cls.asset_b = Asset.objects.create(tenant=cls.t1, customer=cls.cust_b, site=site_b, product=cls.product,
                                           name="B unit", asset_code="AUTH-B")

        def doc(title, **fields):
            return KnowledgeDocument.objects.create(tenant=cls.t1, title=title, source_type="text",
                                                    content_text=f"{title} body text", **fields)
        cls.doc_a = doc("A asset manual", customer=cls.cust_a, asset=cls.asset_a)
        cls.doc_shared = doc("Shared product manual", product=cls.product)
        cls.doc_b_asset = doc("B asset manual (customer unset)", asset=cls.asset_b)
        cls.doc_b_customer = doc("B customer manual", customer=cls.cust_b)
        cls.doc_confidential = doc("Confidential manual", product=cls.product, is_confidential=True)
        cls.doc_no_sharing = doc("Internal-only manual", product=cls.product, disable_sharing=True)
        KnowledgeChunk.objects.create(tenant=cls.t1, document=cls.doc_b_asset, chunk_index=0,
                                      heading="B", text="B-ASSET-SECRET passage")

        def user(name, tenant, role, customer=None, staff=False):
            u = User.objects.create(username=name)
            TenantMembership.objects.create(user=u, tenant=tenant, role=role, customer=customer)
            profile = StaffProfile.objects.create(tenant=tenant, user=u, full_name=name, role=role) if staff else None
            return u, profile
        cls.u_cust_a, _ = user("auth_cust_a", cls.t1, "customer", cls.cust_a)
        cls.u_cust_c, _ = user("auth_cust_c", cls.t2, "customer", cls.cust_c)
        cls.u_tech1, cls.tech1 = user("auth_tech1", cls.t1, "technician", staff=True)
        cls.u_tech2, cls.tech2 = user("auth_tech2", cls.t1, "technician", staff=True)
        cls.u_store, _ = user("auth_store_op", cls.t1, "store_operator", staff=True)
        cls.u_store_admin, _ = user("auth_store_admin", cls.t1, "store_admin", staff=True)
        cls.u_manager, _ = user("auth_manager", cls.t1, "manager", staff=True)

        cls.call_a = ServiceCall.objects.create(tenant=cls.t1, servy_id=910001, customer=cls.cust_a, asset=cls.asset_a,
                                                complaint_text="A complaint", status="assigned", technician=cls.tech1)
        cls.call_b = ServiceCall.objects.create(tenant=cls.t1, servy_id=910002, customer=cls.cust_b, asset=cls.asset_b,
                                                complaint_text="B complaint", status="assigned", technician=cls.tech2)

    def as_user(self, user):
        client = APIClient()
        client.force_login(user)
        return client

    # -- knowledge: list, detail (chunks), download ----------------------------
    def test_customer_knowledge_list_excludes_other_customers_asset_documents(self):
        titles = {d["title"] for d in self.as_user(self.u_cust_a).get("/api/knowledge/").json()["documents"]}
        self.assertIn(self.doc_a.title, titles)
        self.assertIn(self.doc_shared.title, titles)
        for hidden in (self.doc_b_asset, self.doc_b_customer, self.doc_confidential, self.doc_no_sharing):
            self.assertNotIn(hidden.title, titles)
        staff_titles = {d["title"] for d in self.as_user(self.u_manager).get("/api/knowledge/").json()["documents"]}
        self.assertTrue({self.doc_b_asset.title, self.doc_no_sharing.title} <= staff_titles)

    def test_asset_document_list_hides_sharing_disabled_documents_from_customers(self):
        path = f"/api/assets/{self.asset_a.id}/documents/"
        titles = {d["title"] for d in self.as_user(self.u_cust_a).get(path).json()["documents"]}
        self.assertTrue({self.doc_a.title, self.doc_shared.title} <= titles)
        self.assertFalse({self.doc_no_sharing.title, self.doc_confidential.title} & titles)
        staff_titles = {d["title"] for d in self.as_user(self.u_manager).get(path).json()["documents"]}
        self.assertIn(self.doc_no_sharing.title, staff_titles)

    def test_revoked_document_is_hidden_from_customers_only(self):
        SourceProvenanceRecord.objects.create(
            tenant=self.t1, external_object_type="document", external_id="auth-revoked-1",
            local_object_type="KnowledgeDocument", local_object_id=self.doc_shared.id, verification_status="REVOKED")
        client = self.as_user(self.u_cust_a)
        titles = {d["title"] for d in client.get("/api/knowledge/").json()["documents"]}
        self.assertNotIn(self.doc_shared.title, titles)
        self.assertIn(self.doc_a.title, titles)
        self.assertEqual(client.get(f"/api/knowledge/{self.doc_shared.id}/").status_code, 404)
        self.assertEqual(self.as_user(self.u_manager).get(f"/api/knowledge/{self.doc_shared.id}/").status_code, 200)

    def test_customer_cannot_read_or_download_other_customers_asset_document(self):
        client = self.as_user(self.u_cust_a)
        detail = client.get(f"/api/knowledge/{self.doc_b_asset.id}/")
        self.assertIn(detail.status_code, (403, 404))
        self.assertNotIn(b"B-ASSET-SECRET", detail.content)
        self.assertIn(client.get(f"/api/knowledge/{self.doc_b_asset.id}/download/").status_code, (403, 404))
        self.assertIn(client.get(f"/api/knowledge/{self.doc_no_sharing.id}/").status_code, (403, 404))

    def test_shared_and_own_documents_remain_readable(self):
        client = self.as_user(self.u_cust_a)
        self.assertEqual(client.get(f"/api/knowledge/{self.doc_shared.id}/").status_code, 200)
        self.assertEqual(client.get(f"/api/knowledge/{self.doc_a.id}/").status_code, 200)

    def test_other_tenant_customer_sees_nothing_from_this_tenant(self):
        titles = {d["title"] for d in self.as_user(self.u_cust_c).get("/api/knowledge/").json()["documents"]}
        self.assertFalse(titles & {self.doc_shared.title, self.doc_a.title, self.doc_b_asset.title})

    @override_settings(MEDIA_ROOT=tempfile.mkdtemp())
    def test_customer_cannot_download_other_customers_asset_file(self):
        self.doc_b_asset.file = SimpleUploadedFile("b.txt", b"B-FILE-SECRET")
        self.doc_b_asset.source_type = "file"
        self.doc_b_asset.save()
        response = self.as_user(self.u_cust_a).get(f"/api/knowledge/{self.doc_b_asset.id}/download/")
        self.assertIn(response.status_code, (403, 404))
        self.assertNotIn(b"B-FILE-SECRET", b"".join(getattr(response, "streaming_content", [response.content])))

    # -- service tickets ---------------------------------------------------------
    def test_customer_ticket_detail_is_scoped_for_technicians_and_denied_to_store_roles(self):
        path = f"/api/customer-support/tickets/{self.call_b.id}/"
        self.assertEqual(self.as_user(self.u_tech1).get(path).status_code, 404)   # assigned to tech2
        self.assertEqual(self.as_user(self.u_tech2).get(path).status_code, 200)   # assignee
        self.assertEqual(self.as_user(self.u_store).get(path).status_code, 403)
        self.assertEqual(self.as_user(self.u_store_admin).get(path).status_code, 403)
        self.assertEqual(self.as_user(self.u_manager).get(path).status_code, 200)
        self.assertEqual(self.as_user(self.u_cust_a).get(path).status_code, 404)
        self.assertEqual(self.as_user(self.u_cust_c).get(path).status_code, 404)

    def test_technician_cannot_list_unassigned_customers_tickets(self):
        r = self.as_user(self.u_tech1).get(f"/api/customer-support/tickets/?customer_id={self.cust_b.id}")
        ids = [t.get("id") for t in (r.json().get("tickets") or [])] if r.status_code == 200 else []
        self.assertNotIn(self.call_b.id, ids)
        for user in (self.u_tech2, self.u_manager):  # assignee and tenant manager keep access
            r = self.as_user(user).get(f"/api/customer-support/tickets/?customer_id={self.cust_b.id}")
            self.assertEqual([t["id"] for t in r.json()["tickets"]], [self.call_b.id])
        self.assertEqual(self.as_user(self.u_store).get("/api/customer-support/tickets/").status_code, 403)

    def test_customer_sees_own_tickets(self):
        r = self.as_user(self.u_cust_a).get("/api/customer-support/tickets/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual([t["id"] for t in r.json()["tickets"]], [self.call_a.id])
        self.assertEqual(self.as_user(self.u_cust_a).get(f"/api/customer-support/tickets/{self.call_a.id}/").status_code, 200)

    # -- RAG publishing ----------------------------------------------------------
    def upload(self, user, title):
        return self.as_user(user).post(f"/api/assets/{self.asset_a.id}/documents/upload/",
                                       {"title": title, "content_text": "Filter cleaning interval is quarterly.",
                                        "is_rag_enabled": "true"}, format="multipart")

    def test_only_managers_and_admins_can_publish_rag_documents(self):
        for user, may_publish in ((self.u_tech1, False), (self.u_store, False), (self.u_store_admin, False),
                                  (self.u_cust_a, False), (self.u_manager, True)):
            with self.subTest(user=user.username):
                r = self.upload(user, f"upload by {user.username}")
                if r.status_code in (403, 404):
                    continue
                self.assertEqual(r.status_code, 201, r.content[:200])
                doc = KnowledgeDocument.objects.get(title=f"upload by {user.username}")
                self.assertEqual(doc.is_rag_enabled, may_publish)
                self.assertEqual(doc.chunks.exists(), may_publish)

    # -- error disclosure ----------------------------------------------------------
    @override_settings(SERVY_CUSTOMER_DATA_MODE="demo_only")
    def test_unexpected_turn_error_is_not_shown_to_the_customer(self):
        conv = CustomerConversation.objects.create(tenant=self.t1, customer=self.cust_a, asset=self.asset_a, title="t")
        with mock.patch("core.services.customer_conversation.process_customer_diagnostic_turn",
                        side_effect=RuntimeError(r"C:\private\db.py password=hunter2 table core_x")):
            r = self.as_user(self.u_cust_a).post(
                f"/api/customer-support/conversations/{conv.conversation_uuid}/messages/",
                {"message": "My unit is not cooling"}, format="json")
        self.assertGreaterEqual(r.status_code, 400)
        for leaked in (b"hunter2", b"private", b"core_x", b"RuntimeError"):
            self.assertNotIn(leaked, r.content)
